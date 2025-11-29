/**
 * Venus Inference Engine - Image Preprocessing
 * 
 * Image loading, resizing, and preprocessing for vision models.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#include "vision.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
// Default Preprocessing Configurations
// ============================================================================

ImagePreprocessConfig vision_get_default_preprocess_config(VisionArchType arch) {
    ImagePreprocessConfig cfg = {0};
    
    // Common defaults
    cfg.do_resize = true;
    cfg.keep_aspect_ratio = false;
    cfg.do_normalize = true;
    cfg.do_rescale = true;
    cfg.rescale_factor = 1.0f / 255.0f;
    cfg.do_center_crop = false;
    cfg.do_pad = false;
    cfg.pad_value = 0.0f;
    
    switch (arch) {
        case VISION_ARCH_CLIP:
        case VISION_ARCH_LLAVA:
            // CLIP/LLaVA uses ImageNet normalization
            cfg.target_size = 224;
            cfg.mean[0] = 0.48145466f;
            cfg.mean[1] = 0.4578275f;
            cfg.mean[2] = 0.40821073f;
            cfg.std[0] = 0.26862954f;
            cfg.std[1] = 0.26130258f;
            cfg.std[2] = 0.27577711f;
            break;
            
        case VISION_ARCH_SIGLIP:
            // SigLIP uses different normalization
            cfg.target_size = 384;
            cfg.mean[0] = 0.5f;
            cfg.mean[1] = 0.5f;
            cfg.mean[2] = 0.5f;
            cfg.std[0] = 0.5f;
            cfg.std[1] = 0.5f;
            cfg.std[2] = 0.5f;
            break;
            
        case VISION_ARCH_QWEN2_VL:
        case VISION_ARCH_QWEN3_VL:
            // Qwen2-VL uses dynamic resolution
            cfg.target_size = 448;
            cfg.keep_aspect_ratio = true;
            cfg.mean[0] = 0.48145466f;
            cfg.mean[1] = 0.4578275f;
            cfg.mean[2] = 0.40821073f;
            cfg.std[0] = 0.26862954f;
            cfg.std[1] = 0.26130258f;
            cfg.std[2] = 0.27577711f;
            cfg.do_pad = true;
            break;
            
        case VISION_ARCH_PIXTRAL:
            // Pixtral uses different size
            cfg.target_size = 1024;
            cfg.keep_aspect_ratio = true;
            cfg.mean[0] = 0.5f;
            cfg.mean[1] = 0.5f;
            cfg.mean[2] = 0.5f;
            cfg.std[0] = 0.5f;
            cfg.std[1] = 0.5f;
            cfg.std[2] = 0.5f;
            break;
            
        default:
            // Generic ViT defaults
            cfg.target_size = 224;
            cfg.mean[0] = 0.485f;
            cfg.mean[1] = 0.456f;
            cfg.mean[2] = 0.406f;
            cfg.std[0] = 0.229f;
            cfg.std[1] = 0.224f;
            cfg.std[2] = 0.225f;
            break;
    }
    
    return cfg;
}

// ============================================================================
// Image Resizing
// ============================================================================

/**
 * Bilinear interpolation for image resizing.
 */
static void bilinear_resize(const uint8_t* src, int src_w, int src_h,
                            uint8_t* dst, int dst_w, int dst_h,
                            int channels) {
    float x_ratio = (float)(src_w - 1) / (float)(dst_w - 1);
    float y_ratio = (float)(src_h - 1) / (float)(dst_h - 1);
    
    for (int y = 0; y < dst_h; y++) {
        float src_y = y * y_ratio;
        int y0 = (int)src_y;
        int y1 = y0 + 1;
        if (y1 >= src_h) y1 = src_h - 1;
        float y_frac = src_y - y0;
        
        for (int x = 0; x < dst_w; x++) {
            float src_x = x * x_ratio;
            int x0 = (int)src_x;
            int x1 = x0 + 1;
            if (x1 >= src_w) x1 = src_w - 1;
            float x_frac = src_x - x0;
            
            for (int c = 0; c < channels; c++) {
                // Get four corner pixels
                float p00 = src[(y0 * src_w + x0) * channels + c];
                float p01 = src[(y0 * src_w + x1) * channels + c];
                float p10 = src[(y1 * src_w + x0) * channels + c];
                float p11 = src[(y1 * src_w + x1) * channels + c];
                
                // Bilinear interpolation
                float p0 = p00 * (1 - x_frac) + p01 * x_frac;
                float p1 = p10 * (1 - x_frac) + p11 * x_frac;
                float p = p0 * (1 - y_frac) + p1 * y_frac;
                
                dst[(y * dst_w + x) * channels + c] = (uint8_t)(p + 0.5f);
            }
        }
    }
}

/**
 * Resize image with optional aspect ratio preservation.
 */
static int resize_image(const uint8_t* src, int src_w, int src_h,
                        uint8_t** dst, int* dst_w, int* dst_h,
                        int target_size, bool keep_aspect_ratio,
                        int channels) {
    int new_w, new_h;
    
    if (keep_aspect_ratio) {
        // Scale to fit within target_size x target_size
        float scale = (float)target_size / fmaxf((float)src_w, (float)src_h);
        new_w = (int)(src_w * scale);
        new_h = (int)(src_h * scale);
    } else {
        // Resize to exactly target_size x target_size
        new_w = target_size;
        new_h = target_size;
    }
    
    *dst = (uint8_t*)malloc(new_w * new_h * channels);
    if (!*dst) return -1;
    
    bilinear_resize(src, src_w, src_h, *dst, new_w, new_h, channels);
    
    *dst_w = new_w;
    *dst_h = new_h;
    return 0;
}

// ============================================================================
// Center Cropping
// ============================================================================

static int center_crop(const uint8_t* src, int src_w, int src_h,
                       uint8_t** dst, int crop_size, int channels) {
    if (src_w < crop_size || src_h < crop_size) {
        // Image too small, return copy
        *dst = (uint8_t*)malloc(src_w * src_h * channels);
        if (!*dst) return -1;
        memcpy(*dst, src, src_w * src_h * channels);
        return 0;
    }
    
    int x_start = (src_w - crop_size) / 2;
    int y_start = (src_h - crop_size) / 2;
    
    *dst = (uint8_t*)malloc(crop_size * crop_size * channels);
    if (!*dst) return -1;
    
    for (int y = 0; y < crop_size; y++) {
        for (int x = 0; x < crop_size; x++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = ((y_start + y) * src_w + (x_start + x)) * channels + c;
                int dst_idx = (y * crop_size + x) * channels + c;
                (*dst)[dst_idx] = src[src_idx];
            }
        }
    }
    
    return 0;
}

// ============================================================================
// Padding
// ============================================================================

static int pad_image(const uint8_t* src, int src_w, int src_h,
                     uint8_t** dst, int target_w, int target_h,
                     float pad_value, int channels) {
    *dst = (uint8_t*)malloc(target_w * target_h * channels);
    if (!*dst) return -1;
    
    // Fill with pad value
    uint8_t pad_byte = (uint8_t)(pad_value * 255.0f);
    memset(*dst, pad_byte, target_w * target_h * channels);
    
    // Center the image
    int x_offset = (target_w - src_w) / 2;
    int y_offset = (target_h - src_h) / 2;
    
    for (int y = 0; y < src_h; y++) {
        for (int x = 0; x < src_w; x++) {
            for (int c = 0; c < channels; c++) {
                int src_idx = (y * src_w + x) * channels + c;
                int dst_idx = ((y_offset + y) * target_w + (x_offset + x)) * channels + c;
                (*dst)[dst_idx] = src[src_idx];
            }
        }
    }
    
    return 0;
}

// ============================================================================
// Normalization and Conversion
// ============================================================================

/**
 * Convert HWC uint8 to CHW float with rescaling and normalization.
 * Also converts from RGB HWC to CHW format.
 */
static void convert_and_normalize(const uint8_t* src, int width, int height,
                                  float* dst, const ImagePreprocessConfig* config) {
    int channels = 3;
    
    for (int c = 0; c < channels; c++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int src_idx = (y * width + x) * channels + c;
                int dst_idx = (c * height + y) * width + x;
                
                float val = (float)src[src_idx];
                
                // Rescale
                if (config->do_rescale) {
                    val *= config->rescale_factor;
                }
                
                // Normalize
                if (config->do_normalize) {
                    val = (val - config->mean[c]) / config->std[c];
                }
                
                dst[dst_idx] = val;
            }
        }
    }
}

// ============================================================================
// Main Preprocessing Functions
// ============================================================================

int vision_preprocess_image(const uint8_t* raw_data,
                            int width, int height,
                            const ImagePreprocessConfig* config,
                            ProcessedImage* output) {
    if (!raw_data || !config || !output) return -1;
    
    const int channels = 3;  // RGB
    uint8_t* current = NULL;
    uint8_t* temp = NULL;
    int cur_w = width;
    int cur_h = height;
    
    // Make a copy of input
    current = (uint8_t*)malloc(width * height * channels);
    if (!current) return -1;
    memcpy(current, raw_data, width * height * channels);
    
    // Resize
    if (config->do_resize) {
        int new_w, new_h;
        if (resize_image(current, cur_w, cur_h, &temp, &new_w, &new_h,
                         config->target_size, config->keep_aspect_ratio, channels) != 0) {
            free(current);
            return -1;
        }
        free(current);
        current = temp;
        cur_w = new_w;
        cur_h = new_h;
    }
    
    // Center crop
    if (config->do_center_crop && config->crop_size > 0) {
        if (center_crop(current, cur_w, cur_h, &temp, config->crop_size, channels) != 0) {
            free(current);
            return -1;
        }
        free(current);
        current = temp;
        cur_w = config->crop_size;
        cur_h = config->crop_size;
    }
    
    // Padding (for aspect ratio preservation)
    if (config->do_pad && config->keep_aspect_ratio) {
        int target = config->target_size;
        if (cur_w != target || cur_h != target) {
            if (pad_image(current, cur_w, cur_h, &temp, target, target,
                          config->pad_value, channels) != 0) {
                free(current);
                return -1;
            }
            free(current);
            current = temp;
            cur_w = target;
            cur_h = target;
        }
    }
    
    // Allocate output (CHW format)
    output->pixel_values = (float*)malloc(channels * cur_h * cur_w * sizeof(float));
    if (!output->pixel_values) {
        free(current);
        return -1;
    }
    
    // Convert and normalize
    convert_and_normalize(current, cur_w, cur_h, output->pixel_values, config);
    
    // Set output metadata
    output->channels = channels;
    output->width = cur_w;
    output->height = cur_h;
    
    // Calculate patches (assuming square patches)
    // This will be overridden by the caller if patch_size is known
    output->num_patches_h = cur_h / 14;  // Default patch size
    output->num_patches_w = cur_w / 14;
    output->total_patches = output->num_patches_h * output->num_patches_w;
    
    free(current);
    return 0;
}

int vision_preprocess_dynamic(const uint8_t* raw_data,
                              int width, int height,
                              const VisionEncoderConfig* encoder_config,
                              const ImagePreprocessConfig* preprocess_config,
                              ProcessedImage* output) {
    if (!raw_data || !encoder_config || !preprocess_config || !output) return -1;
    
    // Calculate optimal resolution
    int target_w, target_h;
    vision_calculate_dynamic_resolution(
        width, height,
        encoder_config->min_pixels,
        encoder_config->max_pixels,
        encoder_config->patch_size,
        &target_w, &target_h
    );
    
    // Create modified config with dynamic resolution
    ImagePreprocessConfig dynamic_config = *preprocess_config;
    dynamic_config.do_resize = true;
    dynamic_config.keep_aspect_ratio = true;
    
    const int channels = 3;
    uint8_t* resized = NULL;
    int resized_w, resized_h;
    
    // First resize maintaining aspect ratio
    float scale = fminf((float)target_w / (float)width, 
                        (float)target_h / (float)height);
    int scaled_w = (int)(width * scale);
    int scaled_h = (int)(height * scale);
    
    // Round to patch size
    scaled_w = ((scaled_w + encoder_config->patch_size - 1) / encoder_config->patch_size) 
               * encoder_config->patch_size;
    scaled_h = ((scaled_h + encoder_config->patch_size - 1) / encoder_config->patch_size) 
               * encoder_config->patch_size;
    
    // Resize
    if (resize_image(raw_data, width, height, &resized, &resized_w, &resized_h,
                     fmaxf(scaled_w, scaled_h), true, channels) != 0) {
        return -1;
    }
    
    // Pad to target size if needed
    uint8_t* padded = NULL;
    if (resized_w != scaled_w || resized_h != scaled_h) {
        if (pad_image(resized, resized_w, resized_h, &padded, 
                      scaled_w, scaled_h, 0.0f, channels) != 0) {
            free(resized);
            return -1;
        }
        free(resized);
        resized = padded;
        resized_w = scaled_w;
        resized_h = scaled_h;
    }
    
    // Allocate output
    output->pixel_values = (float*)malloc(channels * resized_h * resized_w * sizeof(float));
    if (!output->pixel_values) {
        free(resized);
        return -1;
    }
    
    // Convert and normalize
    convert_and_normalize(resized, resized_w, resized_h, 
                          output->pixel_values, preprocess_config);
    
    // Set output metadata
    output->channels = channels;
    output->width = resized_w;
    output->height = resized_h;
    output->num_patches_h = resized_h / encoder_config->patch_size;
    output->num_patches_w = resized_w / encoder_config->patch_size;
    output->total_patches = output->num_patches_h * output->num_patches_w;
    
    free(resized);
    return 0;
}

