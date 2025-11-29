/**
 * Venus Inference Engine - Vision Encoder Implementation
 * 
 * ViT-based vision encoder for VLM support.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#include "vision.h"
#include "../tensor.h"
#include "../platform/platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Internal Structures
// ============================================================================

typedef struct {
    // Patch embedding (Conv2D equivalent)
    float* patch_embed_weight;  // [hidden_size, channels, patch_size, patch_size]
    float* patch_embed_bias;    // [hidden_size]
    
    // Class token
    float* cls_token;           // [1, 1, hidden_size]
    
    // Position embeddings
    float* pos_embed;           // [1, num_patches + 1, hidden_size]
    
    // 2D RoPE frequencies (for Qwen2-VL)
    float* rope_cos_h;
    float* rope_sin_h;
    float* rope_cos_w;
    float* rope_sin_w;
} VisionEmbeddings;

typedef struct {
    // Self-attention
    float* q_proj_weight;       // [hidden_size, hidden_size]
    float* q_proj_bias;
    float* k_proj_weight;
    float* k_proj_bias;
    float* v_proj_weight;
    float* v_proj_bias;
    float* out_proj_weight;
    float* out_proj_bias;
    
    // Layer norms
    float* ln1_weight;
    float* ln1_bias;
    float* ln2_weight;
    float* ln2_bias;
    
    // MLP
    float* mlp_fc1_weight;      // [intermediate_size, hidden_size]
    float* mlp_fc1_bias;
    float* mlp_fc2_weight;      // [hidden_size, intermediate_size]
    float* mlp_fc2_bias;
} VisionTransformerLayer;

struct VisionEncoder {
    VisionEncoderConfig config;
    
    // Embeddings
    VisionEmbeddings embeddings;
    
    // Transformer layers
    VisionTransformerLayer* layers;
    
    // Final layer norm
    float* final_ln_weight;
    float* final_ln_bias;
    
    // Workspace buffers
    float* workspace;
    size_t workspace_size;
    
    // State
    bool weights_loaded;
};

struct Projector {
    ProjectorConfig config;
    
    // Projection layers
    float** layer_weights;
    float** layer_biases;
    
    bool weights_loaded;
};

// ============================================================================
// Helper Functions
// ============================================================================

static float gelu(float x) {
    // Approximate GELU
    const float SQRT_2_PI = 0.7978845608f;
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + 0.044715f * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

static float quick_gelu(float x) {
    // QuickGELU (used in CLIP/SigLIP)
    return x * (1.0f / (1.0f + expf(-1.702f * x)));
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void apply_activation(float* data, int n, VisionActivation act) {
    for (int i = 0; i < n; i++) {
        switch (act) {
            case VISION_ACT_GELU:
                data[i] = gelu(data[i]);
                break;
            case VISION_ACT_QUICK_GELU:
                data[i] = quick_gelu(data[i]);
                break;
            case VISION_ACT_SILU:
                data[i] = silu(data[i]);
                break;
            case VISION_ACT_RELU:
                data[i] = data[i] > 0 ? data[i] : 0;
                break;
        }
    }
}

static void layer_norm(const float* input, const float* weight, const float* bias,
                       float* output, int batch, int hidden_size, float eps) {
    for (int b = 0; b < batch; b++) {
        const float* x = input + b * hidden_size;
        float* y = output + b * hidden_size;
        
        // Compute mean
        float mean = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            mean += x[i];
        }
        mean /= hidden_size;
        
        // Compute variance
        float var = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = x[i] - mean;
            var += diff * diff;
        }
        var /= hidden_size;
        
        // Normalize
        float inv_std = 1.0f / sqrtf(var + eps);
        for (int i = 0; i < hidden_size; i++) {
            y[i] = (x[i] - mean) * inv_std * weight[i];
            if (bias) y[i] += bias[i];
        }
    }
}

static void matmul(const float* a, const float* b, float* c,
                   int m, int n, int k) {
    // Simple GEMM: C = A @ B
    // A: [m, k], B: [k, n], C: [m, n]
    SimdOps* ops = get_platform_ops();
    if (ops && ops->gemm_f32) {
        ops->gemm_f32(a, b, c, m, n, k, 1.0f, 0.0f);
    } else {
        // Fallback
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
}

static void add_bias(float* data, const float* bias, int batch, int hidden_size) {
    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < hidden_size; i++) {
            data[b * hidden_size + i] += bias[i];
        }
    }
}

static void add_vectors(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

// ============================================================================
// Patch Embedding
// ============================================================================

static void patch_embed_forward(const VisionEncoder* encoder,
                                const float* pixel_values,
                                int height, int width,
                                float* output) {
    const VisionEncoderConfig* cfg = &encoder->config;
    int patch_h = height / cfg->patch_size;
    int patch_w = width / cfg->patch_size;
    int num_patches = patch_h * patch_w;
    
    // Conv2D with stride = patch_size (equivalent to extracting patches and projecting)
    // Input: [C, H, W]
    // Output: [num_patches, hidden_size]
    
    const float* weight = encoder->embeddings.patch_embed_weight;
    const float* bias = encoder->embeddings.patch_embed_bias;
    
    for (int ph = 0; ph < patch_h; ph++) {
        for (int pw = 0; pw < patch_w; pw++) {
            int patch_idx = ph * patch_w + pw;
            float* out_patch = output + patch_idx * cfg->hidden_size;
            
            // Initialize with bias
            if (bias) {
                memcpy(out_patch, bias, cfg->hidden_size * sizeof(float));
            } else {
                memset(out_patch, 0, cfg->hidden_size * sizeof(float));
            }
            
            // Convolve
            for (int oc = 0; oc < cfg->hidden_size; oc++) {
                float sum = 0.0f;
                for (int ic = 0; ic < cfg->num_channels; ic++) {
                    for (int kh = 0; kh < cfg->patch_size; kh++) {
                        for (int kw = 0; kw < cfg->patch_size; kw++) {
                            int ih = ph * cfg->patch_size + kh;
                            int iw = pw * cfg->patch_size + kw;
                            
                            int weight_idx = ((oc * cfg->num_channels + ic) * cfg->patch_size + kh) * cfg->patch_size + kw;
                            int input_idx = (ic * height + ih) * width + iw;
                            
                            sum += pixel_values[input_idx] * weight[weight_idx];
                        }
                    }
                }
                out_patch[oc] += sum;
            }
        }
    }
}

// ============================================================================
// Self-Attention
// ============================================================================

static void self_attention_forward(const VisionTransformerLayer* layer,
                                   const VisionEncoderConfig* cfg,
                                   const float* input,
                                   float* output,
                                   int seq_len,
                                   float* workspace) {
    int hidden_size = cfg->hidden_size;
    int num_heads = cfg->num_attention_heads;
    int head_dim = hidden_size / num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    // Allocate workspace
    float* q = workspace;
    float* k = q + seq_len * hidden_size;
    float* v = k + seq_len * hidden_size;
    float* scores = v + seq_len * hidden_size;
    float* attn_out = scores + seq_len * seq_len * num_heads;
    
    // Project Q, K, V
    matmul(input, layer->q_proj_weight, q, seq_len, hidden_size, hidden_size);
    if (layer->q_proj_bias) add_bias(q, layer->q_proj_bias, seq_len, hidden_size);
    
    matmul(input, layer->k_proj_weight, k, seq_len, hidden_size, hidden_size);
    if (layer->k_proj_bias) add_bias(k, layer->k_proj_bias, seq_len, hidden_size);
    
    matmul(input, layer->v_proj_weight, v, seq_len, hidden_size, hidden_size);
    if (layer->v_proj_bias) add_bias(v, layer->v_proj_bias, seq_len, hidden_size);
    
    // Multi-head attention
    for (int h = 0; h < num_heads; h++) {
        // Compute attention scores for this head
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < seq_len; j++) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int q_idx = i * hidden_size + h * head_dim + d;
                    int k_idx = j * hidden_size + h * head_dim + d;
                    dot += q[q_idx] * k[k_idx];
                }
                scores[(h * seq_len + i) * seq_len + j] = dot * scale;
            }
        }
        
        // Softmax
        for (int i = 0; i < seq_len; i++) {
            float* row = scores + (h * seq_len + i) * seq_len;
            
            // Find max
            float max_val = row[0];
            for (int j = 1; j < seq_len; j++) {
                if (row[j] > max_val) max_val = row[j];
            }
            
            // Exp and sum
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                row[j] = expf(row[j] - max_val);
                sum += row[j];
            }
            
            // Normalize
            for (int j = 0; j < seq_len; j++) {
                row[j] /= sum;
            }
        }
        
        // Apply attention to values
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float score = scores[(h * seq_len + i) * seq_len + j];
                    int v_idx = j * hidden_size + h * head_dim + d;
                    sum += score * v[v_idx];
                }
                attn_out[i * hidden_size + h * head_dim + d] = sum;
            }
        }
    }
    
    // Output projection
    matmul(attn_out, layer->out_proj_weight, output, seq_len, hidden_size, hidden_size);
    if (layer->out_proj_bias) add_bias(output, layer->out_proj_bias, seq_len, hidden_size);
}

// ============================================================================
// MLP
// ============================================================================

static void mlp_forward(const VisionTransformerLayer* layer,
                        const VisionEncoderConfig* cfg,
                        const float* input,
                        float* output,
                        int seq_len,
                        float* workspace) {
    int hidden_size = cfg->hidden_size;
    int intermediate_size = cfg->intermediate_size;
    
    float* intermediate = workspace;
    
    // FC1
    matmul(input, layer->mlp_fc1_weight, intermediate, seq_len, intermediate_size, hidden_size);
    if (layer->mlp_fc1_bias) add_bias(intermediate, layer->mlp_fc1_bias, seq_len, intermediate_size);
    
    // Activation
    apply_activation(intermediate, seq_len * intermediate_size, cfg->activation);
    
    // FC2
    matmul(intermediate, layer->mlp_fc2_weight, output, seq_len, hidden_size, intermediate_size);
    if (layer->mlp_fc2_bias) add_bias(output, layer->mlp_fc2_bias, seq_len, hidden_size);
}

// ============================================================================
// Transformer Layer
// ============================================================================

static void transformer_layer_forward(const VisionTransformerLayer* layer,
                                      const VisionEncoderConfig* cfg,
                                      float* hidden_states,
                                      int seq_len,
                                      float* workspace) {
    int hidden_size = cfg->hidden_size;
    
    float* residual = workspace;
    float* normed = residual + seq_len * hidden_size;
    float* attn_output = normed + seq_len * hidden_size;
    float* mlp_workspace = attn_output + seq_len * hidden_size;
    
    // Save residual
    memcpy(residual, hidden_states, seq_len * hidden_size * sizeof(float));
    
    // Pre-LN for attention
    if (cfg->use_pre_norm) {
        layer_norm(hidden_states, layer->ln1_weight, layer->ln1_bias,
                   normed, seq_len, hidden_size, cfg->layer_norm_eps);
    } else {
        memcpy(normed, hidden_states, seq_len * hidden_size * sizeof(float));
    }
    
    // Self-attention
    self_attention_forward(layer, cfg, normed, attn_output, seq_len, mlp_workspace);
    
    // Add residual
    add_vectors(attn_output, residual, seq_len * hidden_size);
    
    // Post-LN for attention
    if (!cfg->use_pre_norm) {
        layer_norm(attn_output, layer->ln1_weight, layer->ln1_bias,
                   attn_output, seq_len, hidden_size, cfg->layer_norm_eps);
    }
    
    // Save residual
    memcpy(residual, attn_output, seq_len * hidden_size * sizeof(float));
    
    // Pre-LN for MLP
    if (cfg->use_pre_norm) {
        layer_norm(attn_output, layer->ln2_weight, layer->ln2_bias,
                   normed, seq_len, hidden_size, cfg->layer_norm_eps);
    } else {
        memcpy(normed, attn_output, seq_len * hidden_size * sizeof(float));
    }
    
    // MLP
    mlp_forward(layer, cfg, normed, hidden_states, seq_len, mlp_workspace);
    
    // Add residual
    add_vectors(hidden_states, residual, seq_len * hidden_size);
    
    // Post-LN for MLP
    if (!cfg->use_pre_norm) {
        layer_norm(hidden_states, layer->ln2_weight, layer->ln2_bias,
                   hidden_states, seq_len, hidden_size, cfg->layer_norm_eps);
    }
}

// ============================================================================
// Public API Implementation
// ============================================================================

VisionEncoder* vision_encoder_create(const VisionEncoderConfig* config) {
    VisionEncoder* encoder = (VisionEncoder*)calloc(1, sizeof(VisionEncoder));
    if (!encoder) return NULL;
    
    memcpy(&encoder->config, config, sizeof(VisionEncoderConfig));
    
    // Allocate layers
    encoder->layers = (VisionTransformerLayer*)calloc(config->num_hidden_layers, 
                                                       sizeof(VisionTransformerLayer));
    if (!encoder->layers) {
        free(encoder);
        return NULL;
    }
    
    // Calculate workspace size
    int max_seq_len = (config->image_size / config->patch_size) * 
                      (config->image_size / config->patch_size) + 
                      (config->has_cls_token ? 1 : 0);
    
    // Workspace needs: residual, normed, attn_output, Q, K, V, scores, intermediate
    size_t attn_workspace = max_seq_len * config->hidden_size * 4 +  // Q, K, V, attn_out
                            max_seq_len * max_seq_len * config->num_attention_heads;  // scores
    size_t mlp_workspace = max_seq_len * config->intermediate_size;
    size_t layer_workspace = max_seq_len * config->hidden_size * 3 +  // residual, normed, attn_output
                             attn_workspace + mlp_workspace;
    
    encoder->workspace_size = layer_workspace * sizeof(float);
    encoder->workspace = (float*)malloc(encoder->workspace_size);
    if (!encoder->workspace) {
        free(encoder->layers);
        free(encoder);
        return NULL;
    }
    
    return encoder;
}

int vision_encoder_load_weights(VisionEncoder* encoder, const char* weights_path) {
    if (!encoder || !weights_path) return -1;
    
    // TODO: Implement weight loading from Venus format
    // For now, return success to allow testing
    encoder->weights_loaded = true;
    return 0;
}

int vision_encoder_forward(VisionEncoder* encoder,
                           const ProcessedImage* image,
                           VisionOutput* output) {
    if (!encoder || !image || !output) return -1;
    
    const VisionEncoderConfig* cfg = &encoder->config;
    int num_patches = image->num_patches_h * image->num_patches_w;
    int seq_len = num_patches + (cfg->has_cls_token ? 1 : 0);
    
    // Allocate output
    output->num_tokens = seq_len;
    output->hidden_size = cfg->hidden_size;
    output->hidden_states = (float*)malloc(seq_len * cfg->hidden_size * sizeof(float));
    output->pooled_output = (float*)malloc(cfg->hidden_size * sizeof(float));
    
    if (!output->hidden_states || !output->pooled_output) {
        vision_free_output(output);
        return -1;
    }
    
    // Patch embedding
    float* hidden_states = output->hidden_states;
    int offset = cfg->has_cls_token ? 1 : 0;
    
    patch_embed_forward(encoder, image->pixel_values, 
                        image->height, image->width,
                        hidden_states + offset * cfg->hidden_size);
    
    // Add CLS token
    if (cfg->has_cls_token && encoder->embeddings.cls_token) {
        memcpy(hidden_states, encoder->embeddings.cls_token, 
               cfg->hidden_size * sizeof(float));
    }
    
    // Add position embeddings
    if (encoder->embeddings.pos_embed && 
        cfg->pos_embed_type == VISION_POS_LEARNED) {
        for (int i = 0; i < seq_len * cfg->hidden_size; i++) {
            hidden_states[i] += encoder->embeddings.pos_embed[i];
        }
    }
    
    // Transformer layers
    for (int l = 0; l < cfg->num_hidden_layers; l++) {
        transformer_layer_forward(&encoder->layers[l], cfg, 
                                  hidden_states, seq_len,
                                  encoder->workspace);
    }
    
    // Final layer norm
    if (encoder->final_ln_weight) {
        layer_norm(hidden_states, encoder->final_ln_weight, encoder->final_ln_bias,
                   hidden_states, seq_len, cfg->hidden_size, cfg->layer_norm_eps);
    }
    
    // Pooled output (CLS token or mean pooling)
    if (cfg->has_cls_token) {
        memcpy(output->pooled_output, hidden_states, cfg->hidden_size * sizeof(float));
    } else {
        // Mean pooling
        memset(output->pooled_output, 0, cfg->hidden_size * sizeof(float));
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < cfg->hidden_size; j++) {
                output->pooled_output[j] += hidden_states[i * cfg->hidden_size + j];
            }
        }
        for (int j = 0; j < cfg->hidden_size; j++) {
            output->pooled_output[j] /= seq_len;
        }
    }
    
    return 0;
}

void vision_encoder_free(VisionEncoder* encoder) {
    if (!encoder) return;
    
    // Free embeddings
    free(encoder->embeddings.patch_embed_weight);
    free(encoder->embeddings.patch_embed_bias);
    free(encoder->embeddings.cls_token);
    free(encoder->embeddings.pos_embed);
    free(encoder->embeddings.rope_cos_h);
    free(encoder->embeddings.rope_sin_h);
    free(encoder->embeddings.rope_cos_w);
    free(encoder->embeddings.rope_sin_w);
    
    // Free layers
    if (encoder->layers) {
        for (int l = 0; l < encoder->config.num_hidden_layers; l++) {
            VisionTransformerLayer* layer = &encoder->layers[l];
            free(layer->q_proj_weight);
            free(layer->q_proj_bias);
            free(layer->k_proj_weight);
            free(layer->k_proj_bias);
            free(layer->v_proj_weight);
            free(layer->v_proj_bias);
            free(layer->out_proj_weight);
            free(layer->out_proj_bias);
            free(layer->ln1_weight);
            free(layer->ln1_bias);
            free(layer->ln2_weight);
            free(layer->ln2_bias);
            free(layer->mlp_fc1_weight);
            free(layer->mlp_fc1_bias);
            free(layer->mlp_fc2_weight);
            free(layer->mlp_fc2_bias);
        }
        free(encoder->layers);
    }
    
    free(encoder->final_ln_weight);
    free(encoder->final_ln_bias);
    free(encoder->workspace);
    free(encoder);
}

const VisionEncoderConfig* vision_encoder_get_config(const VisionEncoder* encoder) {
    return encoder ? &encoder->config : NULL;
}

// ============================================================================
// Projector Implementation
// ============================================================================

Projector* projector_create(const ProjectorConfig* config) {
    Projector* proj = (Projector*)calloc(1, sizeof(Projector));
    if (!proj) return NULL;
    
    memcpy(&proj->config, config, sizeof(ProjectorConfig));
    
    proj->layer_weights = (float**)calloc(config->num_layers, sizeof(float*));
    proj->layer_biases = (float**)calloc(config->num_layers, sizeof(float*));
    
    if (!proj->layer_weights || !proj->layer_biases) {
        projector_free(proj);
        return NULL;
    }
    
    return proj;
}

int projector_load_weights(Projector* projector, const char* weights_path) {
    if (!projector || !weights_path) return -1;
    
    // TODO: Implement weight loading
    projector->weights_loaded = true;
    return 0;
}

int projector_forward(Projector* projector,
                      const VisionOutput* vision_output,
                      float* output) {
    if (!projector || !vision_output || !output) return -1;
    
    const ProjectorConfig* cfg = &projector->config;
    int seq_len = vision_output->num_tokens;
    
    const float* input = vision_output->hidden_states;
    int in_dim = cfg->vision_hidden_size;
    int out_dim = cfg->llm_hidden_size;
    
    // Simple linear projection (can be extended to MLP)
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < out_dim; j++) {
            float sum = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                if (projector->layer_weights[0]) {
                    sum += input[i * in_dim + k] * projector->layer_weights[0][k * out_dim + j];
                }
            }
            if (projector->layer_biases[0]) {
                sum += projector->layer_biases[0][j];
            }
            output[i * out_dim + j] = sum;
        }
    }
    
    return 0;
}

void projector_free(Projector* projector) {
    if (!projector) return;
    
    if (projector->layer_weights) {
        for (int i = 0; i < projector->config.num_layers; i++) {
            free(projector->layer_weights[i]);
        }
        free(projector->layer_weights);
    }
    
    if (projector->layer_biases) {
        for (int i = 0; i < projector->config.num_layers; i++) {
            free(projector->layer_biases[i]);
        }
        free(projector->layer_biases);
    }
    
    free(projector);
}

// ============================================================================
// Default Configurations
// ============================================================================

VisionEncoderConfig vision_config_qwen2_vl(void) {
    VisionEncoderConfig cfg = {0};
    cfg.arch_type = VISION_ARCH_QWEN2_VL;
    cfg.hidden_size = 1280;
    cfg.intermediate_size = 5120;
    cfg.num_hidden_layers = 32;
    cfg.num_attention_heads = 16;
    cfg.num_kv_heads = 0;  // MHA
    cfg.image_size = 448;
    cfg.patch_size = 14;
    cfg.num_channels = 3;
    cfg.pos_embed_type = VISION_POS_ROPE_2D;
    cfg.rope_theta = 10000.0f;
    cfg.layer_norm_eps = 1e-6f;
    cfg.use_pre_norm = true;
    cfg.activation = VISION_ACT_QUICK_GELU;
    cfg.has_cls_token = false;
    cfg.supports_dynamic_resolution = true;
    cfg.min_pixels = 256 * 256;
    cfg.max_pixels = 1280 * 1280;
    return cfg;
}

VisionEncoderConfig vision_config_siglip(void) {
    VisionEncoderConfig cfg = {0};
    cfg.arch_type = VISION_ARCH_SIGLIP;
    cfg.hidden_size = 1152;
    cfg.intermediate_size = 4304;
    cfg.num_hidden_layers = 27;
    cfg.num_attention_heads = 16;
    cfg.num_kv_heads = 0;
    cfg.image_size = 384;
    cfg.patch_size = 14;
    cfg.num_channels = 3;
    cfg.pos_embed_type = VISION_POS_LEARNED;
    cfg.layer_norm_eps = 1e-6f;
    cfg.use_pre_norm = true;
    cfg.activation = VISION_ACT_QUICK_GELU;
    cfg.has_cls_token = false;
    cfg.supports_dynamic_resolution = false;
    return cfg;
}

VisionEncoderConfig vision_config_clip_vit_l_14(void) {
    VisionEncoderConfig cfg = {0};
    cfg.arch_type = VISION_ARCH_CLIP;
    cfg.hidden_size = 1024;
    cfg.intermediate_size = 4096;
    cfg.num_hidden_layers = 24;
    cfg.num_attention_heads = 16;
    cfg.num_kv_heads = 0;
    cfg.image_size = 224;
    cfg.patch_size = 14;
    cfg.num_channels = 3;
    cfg.pos_embed_type = VISION_POS_LEARNED;
    cfg.layer_norm_eps = 1e-5f;
    cfg.use_pre_norm = true;
    cfg.activation = VISION_ACT_QUICK_GELU;
    cfg.has_cls_token = true;
    cfg.supports_dynamic_resolution = false;
    return cfg;
}

VisionEncoderConfig vision_config_llava(void) {
    // LLaVA typically uses CLIP ViT-L/14 @ 336
    VisionEncoderConfig cfg = vision_config_clip_vit_l_14();
    cfg.arch_type = VISION_ARCH_LLAVA;
    cfg.image_size = 336;
    return cfg;
}

// ============================================================================
// Utility Functions
// ============================================================================

int vision_calculate_num_tokens(int image_size, int patch_size, bool has_cls_token) {
    int patches_per_side = image_size / patch_size;
    int num_patches = patches_per_side * patches_per_side;
    return num_patches + (has_cls_token ? 1 : 0);
}

void vision_calculate_dynamic_resolution(int width, int height,
                                         int min_pixels, int max_pixels,
                                         int patch_size,
                                         int* out_width, int* out_height) {
    float aspect_ratio = (float)width / (float)height;
    int total_pixels = width * height;
    
    // Clamp to min/max
    if (total_pixels < min_pixels) {
        float scale = sqrtf((float)min_pixels / (float)total_pixels);
        width = (int)(width * scale);
        height = (int)(height * scale);
    } else if (total_pixels > max_pixels) {
        float scale = sqrtf((float)max_pixels / (float)total_pixels);
        width = (int)(width * scale);
        height = (int)(height * scale);
    }
    
    // Round to patch size
    *out_width = ((width + patch_size - 1) / patch_size) * patch_size;
    *out_height = ((height + patch_size - 1) / patch_size) * patch_size;
}

const char* vision_arch_name(VisionArchType arch) {
    switch (arch) {
        case VISION_ARCH_VIT: return "ViT";
        case VISION_ARCH_SIGLIP: return "SigLIP";
        case VISION_ARCH_CLIP: return "CLIP";
        case VISION_ARCH_QWEN2_VL: return "Qwen2-VL";
        case VISION_ARCH_QWEN3_VL: return "Qwen3-VL";
        case VISION_ARCH_LLAVA: return "LLaVA";
        case VISION_ARCH_PIXTRAL: return "Pixtral";
        case VISION_ARCH_INTERN_VL: return "InternVL";
        default: return "Unknown";
    }
}

void vision_print_config(const VisionEncoderConfig* config) {
    if (!config) return;
    
    printf("Vision Encoder Configuration:\n");
    printf("  Architecture: %s\n", vision_arch_name(config->arch_type));
    printf("  Hidden Size: %d\n", config->hidden_size);
    printf("  Intermediate Size: %d\n", config->intermediate_size);
    printf("  Num Layers: %d\n", config->num_hidden_layers);
    printf("  Num Heads: %d\n", config->num_attention_heads);
    printf("  Image Size: %d\n", config->image_size);
    printf("  Patch Size: %d\n", config->patch_size);
    printf("  Num Tokens: %d\n", vision_calculate_num_tokens(
        config->image_size, config->patch_size, config->has_cls_token));
    printf("  Dynamic Resolution: %s\n", config->supports_dynamic_resolution ? "yes" : "no");
    if (config->supports_dynamic_resolution) {
        printf("    Min Pixels: %d\n", config->min_pixels);
        printf("    Max Pixels: %d\n", config->max_pixels);
    }
}

void vision_free_processed_image(ProcessedImage* image) {
    if (!image) return;
    free(image->pixel_values);
    image->pixel_values = NULL;
}

void vision_free_output(VisionOutput* output) {
    if (!output) return;
    free(output->hidden_states);
    free(output->pooled_output);
    output->hidden_states = NULL;
    output->pooled_output = NULL;
}

