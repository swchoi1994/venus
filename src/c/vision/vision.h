/**
 * Venus Inference Engine - Vision Encoder API
 * 
 * VLM (Vision Language Model) support for:
 * - ViT (Vision Transformer)
 * - SigLIP
 * - CLIP
 * - Qwen2-VL / Qwen3-VL
 * - LLaVA
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#ifndef VISION_H
#define VISION_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Configuration Structures
// ============================================================================

/**
 * Vision encoder architecture types
 */
typedef enum {
    VISION_ARCH_VIT,        // Standard Vision Transformer
    VISION_ARCH_SIGLIP,     // SigLIP (Sigmoid Loss for Language-Image Pre-training)
    VISION_ARCH_CLIP,       // CLIP ViT
    VISION_ARCH_QWEN2_VL,   // Qwen2-VL (with 2D RoPE)
    VISION_ARCH_QWEN3_VL,   // Qwen3-VL
    VISION_ARCH_LLAVA,      // LLaVA
    VISION_ARCH_PIXTRAL,    // Pixtral (Mistral Vision)
    VISION_ARCH_INTERN_VL   // InternVL
} VisionArchType;

/**
 * Activation function types for vision models
 */
typedef enum {
    VISION_ACT_GELU,        // Standard GELU
    VISION_ACT_QUICK_GELU,  // QuickGELU (CLIP/SigLIP)
    VISION_ACT_SILU,        // SiLU/Swish
    VISION_ACT_RELU         // ReLU
} VisionActivation;

/**
 * Position embedding types
 */
typedef enum {
    VISION_POS_LEARNED,     // Learned absolute position embeddings
    VISION_POS_SINUSOIDAL,  // Sinusoidal position embeddings
    VISION_POS_ROPE_2D,     // 2D Rotary Position Embeddings (Qwen2-VL)
    VISION_POS_NONE         // No position embeddings (relative attention)
} VisionPosEmbedType;

/**
 * Vision encoder configuration
 */
typedef struct {
    VisionArchType arch_type;
    
    // Model dimensions
    int hidden_size;          // Hidden dimension (e.g., 1024)
    int intermediate_size;    // MLP intermediate size (e.g., 4096)
    int num_hidden_layers;    // Number of transformer layers
    int num_attention_heads;  // Number of attention heads
    int num_kv_heads;         // Number of KV heads (for GQA, 0 = same as num_attention_heads)
    
    // Image configuration
    int image_size;           // Default image size (e.g., 224, 336, 448)
    int patch_size;           // Patch size (e.g., 14, 16)
    int num_channels;         // Input channels (usually 3 for RGB)
    
    // Position embeddings
    VisionPosEmbedType pos_embed_type;
    float rope_theta;         // RoPE theta for 2D RoPE (Qwen2-VL uses 10000)
    
    // Normalization
    float layer_norm_eps;     // LayerNorm epsilon
    bool use_pre_norm;        // Pre-LN vs Post-LN
    
    // Activation
    VisionActivation activation;
    
    // Special tokens
    bool has_cls_token;       // Whether model uses [CLS] token
    
    // Dynamic resolution (Qwen2-VL style)
    bool supports_dynamic_resolution;
    int min_pixels;           // Minimum pixels for dynamic resolution
    int max_pixels;           // Maximum pixels for dynamic resolution
    
    // Quantization
    bool is_quantized;
    int quant_bits;           // 4 or 8 for quantized models
} VisionEncoderConfig;

/**
 * Image preprocessing configuration
 */
typedef struct {
    // Resize
    int target_size;          // Target size for resize
    bool do_resize;
    bool keep_aspect_ratio;
    
    // Normalization
    float mean[3];            // Per-channel mean (RGB)
    float std[3];             // Per-channel std (RGB)
    bool do_normalize;
    
    // Rescale
    float rescale_factor;     // Usually 1/255
    bool do_rescale;
    
    // Center crop
    bool do_center_crop;
    int crop_size;
    
    // Padding
    bool do_pad;
    float pad_value;
} ImagePreprocessConfig;

/**
 * Processed image data
 */
typedef struct {
    float* pixel_values;      // Preprocessed pixel data [C, H, W]
    int channels;
    int height;
    int width;
    int num_patches_h;        // Number of patches in height
    int num_patches_w;        // Number of patches in width
    int total_patches;        // Total number of patches
} ProcessedImage;

/**
 * Vision encoder output
 */
typedef struct {
    float* hidden_states;     // Output hidden states [num_patches, hidden_size]
    float* pooled_output;     // Pooled output (CLS token or mean pooling) [hidden_size]
    int num_tokens;           // Number of output tokens
    int hidden_size;          // Hidden dimension
} VisionOutput;

// ============================================================================
// Vision Encoder
// ============================================================================

/**
 * Vision encoder state
 */
typedef struct VisionEncoder VisionEncoder;

/**
 * Create a vision encoder with the given configuration.
 * 
 * @param config Encoder configuration
 * @return Vision encoder or NULL on failure
 */
VisionEncoder* vision_encoder_create(const VisionEncoderConfig* config);

/**
 * Load vision encoder weights from file.
 * 
 * @param encoder Vision encoder
 * @param weights_path Path to weights file
 * @return 0 on success, error code on failure
 */
int vision_encoder_load_weights(VisionEncoder* encoder, const char* weights_path);

/**
 * Run vision encoder forward pass.
 * 
 * @param encoder Vision encoder
 * @param image Preprocessed image
 * @param output Output structure (caller must free hidden_states and pooled_output)
 * @return 0 on success, error code on failure
 */
int vision_encoder_forward(VisionEncoder* encoder, 
                           const ProcessedImage* image,
                           VisionOutput* output);

/**
 * Free vision encoder.
 * 
 * @param encoder Vision encoder to free
 */
void vision_encoder_free(VisionEncoder* encoder);

/**
 * Get vision encoder configuration.
 * 
 * @param encoder Vision encoder
 * @return Configuration (read-only)
 */
const VisionEncoderConfig* vision_encoder_get_config(const VisionEncoder* encoder);

// ============================================================================
// Projector (Vision-to-LLM)
// ============================================================================

/**
 * Projector configuration
 */
typedef struct {
    int vision_hidden_size;   // Input dimension from vision encoder
    int llm_hidden_size;      // Output dimension for LLM
    int num_layers;           // Number of projection layers (usually 1-2)
    VisionActivation activation;
} ProjectorConfig;

/**
 * Projector state
 */
typedef struct Projector Projector;

/**
 * Create a vision-to-LLM projector.
 * 
 * @param config Projector configuration
 * @return Projector or NULL on failure
 */
Projector* projector_create(const ProjectorConfig* config);

/**
 * Load projector weights.
 * 
 * @param projector Projector
 * @param weights_path Path to weights file
 * @return 0 on success, error code on failure
 */
int projector_load_weights(Projector* projector, const char* weights_path);

/**
 * Project vision features to LLM space.
 * 
 * @param projector Projector
 * @param vision_output Vision encoder output
 * @param output Output tensor [num_tokens, llm_hidden_size]
 * @return 0 on success, error code on failure
 */
int projector_forward(Projector* projector,
                      const VisionOutput* vision_output,
                      float* output);

/**
 * Free projector.
 * 
 * @param projector Projector to free
 */
void projector_free(Projector* projector);

// ============================================================================
// Image Preprocessing
// ============================================================================

/**
 * Create default preprocessing config for a given architecture.
 * 
 * @param arch Vision architecture type
 * @return Default preprocessing configuration
 */
ImagePreprocessConfig vision_get_default_preprocess_config(VisionArchType arch);

/**
 * Preprocess a raw image.
 * 
 * @param raw_data Raw image data (RGB, HWC format)
 * @param width Image width
 * @param height Image height
 * @param config Preprocessing configuration
 * @param output Processed image output (caller must free pixel_values)
 * @return 0 on success, error code on failure
 */
int vision_preprocess_image(const uint8_t* raw_data,
                            int width, int height,
                            const ImagePreprocessConfig* config,
                            ProcessedImage* output);

/**
 * Preprocess image with dynamic resolution (Qwen2-VL style).
 * Automatically determines optimal resolution based on aspect ratio.
 * 
 * @param raw_data Raw image data
 * @param width Image width
 * @param height Image height
 * @param encoder_config Vision encoder config (for resolution constraints)
 * @param preprocess_config Preprocessing configuration
 * @param output Processed image output
 * @return 0 on success, error code on failure
 */
int vision_preprocess_dynamic(const uint8_t* raw_data,
                              int width, int height,
                              const VisionEncoderConfig* encoder_config,
                              const ImagePreprocessConfig* preprocess_config,
                              ProcessedImage* output);

/**
 * Free processed image data.
 * 
 * @param image Processed image to free
 */
void vision_free_processed_image(ProcessedImage* image);

/**
 * Free vision output data.
 * 
 * @param output Vision output to free
 */
void vision_free_output(VisionOutput* output);

// ============================================================================
// Default Configurations
// ============================================================================

/**
 * Get default configuration for Qwen2-VL.
 */
VisionEncoderConfig vision_config_qwen2_vl(void);

/**
 * Get default configuration for SigLIP.
 */
VisionEncoderConfig vision_config_siglip(void);

/**
 * Get default configuration for CLIP ViT-L/14.
 */
VisionEncoderConfig vision_config_clip_vit_l_14(void);

/**
 * Get default configuration for LLaVA.
 */
VisionEncoderConfig vision_config_llava(void);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate number of patches for given image size.
 * 
 * @param image_size Image size (assumes square)
 * @param patch_size Patch size
 * @param has_cls_token Whether model uses CLS token
 * @return Number of output tokens
 */
int vision_calculate_num_tokens(int image_size, int patch_size, bool has_cls_token);

/**
 * Calculate optimal resolution for dynamic resolution models.
 * 
 * @param width Original width
 * @param height Original height
 * @param min_pixels Minimum total pixels
 * @param max_pixels Maximum total pixels
 * @param patch_size Patch size (resolution must be divisible by this)
 * @param out_width Output width
 * @param out_height Output height
 */
void vision_calculate_dynamic_resolution(int width, int height,
                                         int min_pixels, int max_pixels,
                                         int patch_size,
                                         int* out_width, int* out_height);

/**
 * Get architecture name as string.
 */
const char* vision_arch_name(VisionArchType arch);

/**
 * Print vision encoder configuration.
 */
void vision_print_config(const VisionEncoderConfig* config);

#ifdef __cplusplus
}
#endif

#endif // VISION_H

