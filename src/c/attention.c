#include "attention.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Helper functions
float get_attention_scale(int head_dim) {
    return 1.0f / sqrtf((float)head_dim);
}

Tensor* create_causal_mask(int seq_len) {
    int32_t shape[] = {seq_len, seq_len};
    Tensor* mask = tensor_create(shape, 2, DTYPE_F32);
    if (!mask) return NULL;
    
    float* data = (float*)mask->data;
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            data[i * seq_len + j] = (j <= i) ? 0.0f : -FLT_MAX;
        }
    }
    
    return mask;
}

// Standard attention implementation
void standard_attention_impl(Tensor* q, Tensor* k, Tensor* v, Tensor* out,
                            float scale, Tensor* mask) {
    // Assume shapes: q/k/v [batch, seq_len, n_heads, head_dim]
    // out: [batch, seq_len, n_heads, head_dim]
    
    int batch_size = q->shape[0];
    int seq_len = q->shape[1];
    int n_heads = q->shape[2];
    int head_dim = q->shape[3];
    
    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;
    float* v_data = (float*)v->data;
    float* out_data = (float*)out->data;
    float* mask_data = mask ? (float*)mask->data : NULL;
    
    // Temporary buffer for attention scores
    float* scores = (float*)calloc(seq_len * seq_len, sizeof(float));
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            // Compute attention scores: Q @ K^T
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float score = 0.0f;
                    
                    // Dot product of Q[i] and K[j]
                    for (int d = 0; d < head_dim; d++) {
                        int q_idx = b * seq_len * n_heads * head_dim + 
                                   i * n_heads * head_dim + 
                                   h * head_dim + d;
                        int k_idx = b * seq_len * n_heads * head_dim + 
                                   j * n_heads * head_dim + 
                                   h * head_dim + d;
                        score += q_data[q_idx] * k_data[k_idx];
                    }
                    
                    scores[i * seq_len + j] = score * scale;
                    
                    // Apply mask if provided
                    if (mask_data) {
                        scores[i * seq_len + j] += mask_data[i * seq_len + j];
                    }
                }
            }
            
            // Softmax over scores
            for (int i = 0; i < seq_len; i++) {
                float* row = scores + i * seq_len;
                
                // Find max for numerical stability
                float max_score = row[0];
                for (int j = 1; j < seq_len; j++) {
                    max_score = fmaxf(max_score, row[j]);
                }
                
                // Exp and sum
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    row[j] = expf(row[j] - max_score);
                    sum += row[j];
                }
                
                // Normalize
                for (int j = 0; j < seq_len; j++) {
                    row[j] /= sum;
                }
            }
            
            // Apply attention to values: scores @ V
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    
                    for (int j = 0; j < seq_len; j++) {
                        int v_idx = b * seq_len * n_heads * head_dim + 
                                   j * n_heads * head_dim + 
                                   h * head_dim + d;
                        sum += scores[i * seq_len + j] * v_data[v_idx];
                    }
                    
                    int out_idx = b * seq_len * n_heads * head_dim + 
                                 i * n_heads * head_dim + 
                                 h * head_dim + d;
                    out_data[out_idx] = sum;
                }
            }
        }
    }
    
    free(scores);
}

// Flash attention implementation (CPU optimized with tiling)
void flash_attention_impl(Tensor* q, Tensor* k, Tensor* v, Tensor* out,
                         float scale, Tensor* mask) {
    // Flash attention uses tiling to reduce memory access
    const int TILE_SIZE = 64; // Tune based on cache size
    
    int batch_size = q->shape[0];
    int seq_len = q->shape[1];
    int n_heads = q->shape[2];
    int head_dim = q->shape[3];
    
    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;
    float* v_data = (float*)v->data;
    float* out_data = (float*)out->data;
    
    // Initialize output to zero
    memset(out_data, 0, batch_size * seq_len * n_heads * head_dim * sizeof(float));
    
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < n_heads; h++) {
            // Process in tiles
            for (int i_tile = 0; i_tile < seq_len; i_tile += TILE_SIZE) {
                int i_end = fmin(i_tile + TILE_SIZE, seq_len);
                
                // Allocate tile buffers
                float* tile_max = (float*)calloc(i_end - i_tile, sizeof(float));
                float* tile_sum = (float*)calloc(i_end - i_tile, sizeof(float));
                
                // Initialize with -inf and 0
                for (int i = 0; i < i_end - i_tile; i++) {
                    tile_max[i] = -FLT_MAX;
                    tile_sum[i] = 0.0f;
                }
                
                for (int j_tile = 0; j_tile < seq_len; j_tile += TILE_SIZE) {
                    int j_end = fmin(j_tile + TILE_SIZE, seq_len);
                    
                    // Compute scores for this tile
                    float* scores = (float*)calloc((i_end - i_tile) * (j_end - j_tile), sizeof(float));
                    
                    for (int i = i_tile; i < i_end; i++) {
                        for (int j = j_tile; j < j_end; j++) {
                            float score = 0.0f;
                            
                            // Q @ K^T
                            for (int d = 0; d < head_dim; d++) {
                                int q_idx = b * seq_len * n_heads * head_dim + 
                                           i * n_heads * head_dim + 
                                           h * head_dim + d;
                                int k_idx = b * seq_len * n_heads * head_dim + 
                                           j * n_heads * head_dim + 
                                           h * head_dim + d;
                                score += q_data[q_idx] * k_data[k_idx];
                            }
                            
                            scores[(i - i_tile) * (j_end - j_tile) + (j - j_tile)] = score * scale;
                        }
                    }
                    
                    // Online softmax update
                    for (int i = 0; i < i_end - i_tile; i++) {
                        float row_max = tile_max[i];
                        
                        // Find new max
                        for (int j = 0; j < j_end - j_tile; j++) {
                            float score = scores[i * (j_end - j_tile) + j];
                            row_max = fmaxf(row_max, score);
                        }
                        
                        // Update sum with new max
                        float sum_correction = expf(tile_max[i] - row_max);
                        tile_sum[i] *= sum_correction;
                        
                        // Add new exponentials
                        for (int j = 0; j < j_end - j_tile; j++) {
                            float score = scores[i * (j_end - j_tile) + j];
                            tile_sum[i] += expf(score - row_max);
                        }
                        
                        tile_max[i] = row_max;
                    }
                    
                    // Apply attention to values
                    for (int i = i_tile; i < i_end; i++) {
                        for (int j = j_tile; j < j_end; j++) {
                            float score = scores[(i - i_tile) * (j_end - j_tile) + (j - j_tile)];
                            float attn_weight = expf(score - tile_max[i - i_tile]) / tile_sum[i - i_tile];
                            
                            for (int d = 0; d < head_dim; d++) {
                                int v_idx = b * seq_len * n_heads * head_dim + 
                                           j * n_heads * head_dim + 
                                           h * head_dim + d;
                                int out_idx = b * seq_len * n_heads * head_dim + 
                                             i * n_heads * head_dim + 
                                             h * head_dim + d;
                                out_data[out_idx] += attn_weight * v_data[v_idx];
                            }
                        }
                    }
                    
                    free(scores);
                }
                
                free(tile_max);
                free(tile_sum);
            }
        }
    }
}

// KV Cache management
KVCache* create_kv_cache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim) {
    KVCache* cache = (KVCache*)calloc(1, sizeof(KVCache));
    if (!cache) return NULL;
    
    cache->n_layers = n_layers;
    cache->max_seq_len = max_seq_len;
    cache->n_kv_heads = n_kv_heads;
    cache->head_dim = head_dim;
    cache->current_pos = 0;
    
    // Allocate cache tensors
    cache->k_cache = (Tensor**)calloc(n_layers, sizeof(Tensor*));
    cache->v_cache = (Tensor**)calloc(n_layers, sizeof(Tensor*));
    
    if (!cache->k_cache || !cache->v_cache) {
        free(cache->k_cache);
        free(cache->v_cache);
        free(cache);
        return NULL;
    }
    
    int32_t shape[] = {max_seq_len, n_kv_heads, head_dim};
    for (int i = 0; i < n_layers; i++) {
        cache->k_cache[i] = tensor_create(shape, 3, DTYPE_F32);
        cache->v_cache[i] = tensor_create(shape, 3, DTYPE_F32);
        
        if (!cache->k_cache[i] || !cache->v_cache[i]) {
            // Cleanup on failure
            for (int j = 0; j <= i; j++) {
                tensor_free(cache->k_cache[j]);
                tensor_free(cache->v_cache[j]);
            }
            free(cache->k_cache);
            free(cache->v_cache);
            free(cache);
            return NULL;
        }
    }
    
    return cache;
}

void free_kv_cache(KVCache* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->n_layers; i++) {
        tensor_free(cache->k_cache[i]);
        tensor_free(cache->v_cache[i]);
    }
    
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}

void kv_cache_update(KVCache* cache, Tensor* k, Tensor* v, int layer_idx, int pos) {
    if (!cache || layer_idx >= cache->n_layers || pos >= cache->max_seq_len) return;
    
    float* k_data = (float*)k->data;
    float* v_data = (float*)v->data;
    float* k_cache_data = (float*)cache->k_cache[layer_idx]->data;
    float* v_cache_data = (float*)cache->v_cache[layer_idx]->data;
    
    // Copy new K and V into cache at position pos
    int offset = pos * cache->n_kv_heads * cache->head_dim;
    memcpy(k_cache_data + offset, k_data, cache->n_kv_heads * cache->head_dim * sizeof(float));
    memcpy(v_cache_data + offset, v_data, cache->n_kv_heads * cache->head_dim * sizeof(float));
    
    // Update position
    cache->current_pos = pos + 1;
}

void kv_cache_clear(KVCache* cache) {
    if (!cache) return;
    
    cache->current_pos = 0;
    
    for (int i = 0; i < cache->n_layers; i++) {
        // Clear cache data
        int cache_size = cache->max_seq_len * cache->n_kv_heads * cache->head_dim * sizeof(float);
        memset(cache->k_cache[i]->data, 0, cache_size);
        memset(cache->v_cache[i]->data, 0, cache_size);
    }
}

// RoPE implementation
RoPECache* create_rope_cache(int max_seq_len, int head_dim, float theta) {
    RoPECache* cache = (RoPECache*)calloc(1, sizeof(RoPECache));
    if (!cache) return NULL;
    
    cache->max_seq_len = max_seq_len;
    cache->head_dim = head_dim;
    cache->theta = theta;
    
    // Allocate cos and sin caches
    cache->cos_cached = (float*)malloc(max_seq_len * head_dim * sizeof(float));
    cache->sin_cached = (float*)malloc(max_seq_len * head_dim * sizeof(float));
    
    if (!cache->cos_cached || !cache->sin_cached) {
        free(cache->cos_cached);
        free(cache->sin_cached);
        free(cache);
        return NULL;
    }
    
    // Precompute cos and sin values
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < head_dim / 2; i++) {
            float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
            float angle = pos * freq;
            
            cache->cos_cached[pos * head_dim + 2 * i] = cosf(angle);
            cache->cos_cached[pos * head_dim + 2 * i + 1] = cosf(angle);
            cache->sin_cached[pos * head_dim + 2 * i] = sinf(angle);
            cache->sin_cached[pos * head_dim + 2 * i + 1] = sinf(angle);
        }
    }
    
    return cache;
}

void free_rope_cache(RoPECache* cache) {
    if (!cache) return;
    
    free(cache->cos_cached);
    free(cache->sin_cached);
    free(cache);
}

void apply_rope(Tensor* q, Tensor* k, RoPECache* cache, int pos) {
    if (!cache || !q || !k) return;
    
    float* q_data = (float*)q->data;
    float* k_data = (float*)k->data;
    
    int seq_len = q->shape[1];
    int n_heads = q->shape[2];
    int head_dim = q->shape[3];
    
    #pragma omp parallel for
    for (int i = 0; i < seq_len; i++) {
        int rope_pos = pos + i;
        if (rope_pos >= cache->max_seq_len) continue;
        
        float* cos = cache->cos_cached + rope_pos * head_dim;
        float* sin = cache->sin_cached + rope_pos * head_dim;
        
        for (int h = 0; h < n_heads; h++) {
            // Apply RoPE to Q
            for (int d = 0; d < head_dim; d += 2) {
                int idx = i * n_heads * head_dim + h * head_dim + d;
                float q0 = q_data[idx];
                float q1 = q_data[idx + 1];
                
                q_data[idx] = q0 * cos[d] - q1 * sin[d];
                q_data[idx + 1] = q0 * sin[d] + q1 * cos[d];
            }
            
            // Apply RoPE to K
            for (int d = 0; d < head_dim; d += 2) {
                int idx = i * n_heads * head_dim + h * head_dim + d;
                float k0 = k_data[idx];
                float k1 = k_data[idx + 1];
                
                k_data[idx] = k0 * cos[d] - k1 * sin[d];
                k_data[idx + 1] = k0 * sin[d] + k1 * cos[d];
            }
        }
    }
}

// Create attention operations
AttentionOps* create_attention_ops(ModelConfig* config) {
    AttentionOps* ops = (AttentionOps*)calloc(1, sizeof(AttentionOps));
    if (!ops) return NULL;
    
    ops->standard_attention = standard_attention_impl;
    ops->flash_attention = config->use_flash_attention ? flash_attention_impl : standard_attention_impl;
    ops->paged_attention = NULL; // TODO: Implement paged attention
    ops->grouped_query_attention = NULL; // TODO: Implement GQA
    
    return ops;
}

void free_attention_ops(AttentionOps* ops) {
    free(ops);
}