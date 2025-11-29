/**
 * Venus Inference Engine - RoPE OpenCL Kernels
 * 
 * Rotary Position Embeddings for transformer models.
 * Supports 1D (text) and 2D (vision) variants.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

// 1D Rotary Position Embedding (for text models)
__kernel void rope(
    __global const float* x,
    __global const float* cos_freq,
    __global const float* sin_freq,
    __global float* output,
    const int batch,
    const int seq_len,
    const int heads,
    const int head_dim
) {
    const int d2 = get_global_id(0);  // head_dim / 2
    const int pos = get_global_id(1);
    const int bh = get_global_id(2);
    
    if (d2 >= head_dim / 2 || pos >= seq_len) return;
    
    const int b = bh / heads;
    const int h = bh % heads;
    
    int base_idx = ((b * heads + h) * seq_len + pos) * head_dim;
    int d0 = d2 * 2;
    int d1 = d2 * 2 + 1;
    
    float x0 = x[base_idx + d0];
    float x1 = x[base_idx + d1];
    
    float cos_val = cos_freq[pos * (head_dim / 2) + d2];
    float sin_val = sin_freq[pos * (head_dim / 2) + d2];
    
    // Apply rotation
    output[base_idx + d0] = x0 * cos_val - x1 * sin_val;
    output[base_idx + d1] = x0 * sin_val + x1 * cos_val;
}

// 2D Rotary Position Embedding (for vision models, Qwen2-VL style)
// Applies separate rotations for height and width dimensions
__kernel void rope_2d(
    __global const float* x,
    __global const float* cos_h,
    __global const float* sin_h,
    __global const float* cos_w,
    __global const float* sin_w,
    __global float* output,
    const int batch,
    const int height,
    const int width,
    const int heads,
    const int head_dim
) {
    const int w = get_global_id(0);
    const int h = get_global_id(1);
    const int bh = get_global_id(2);
    
    if (w >= width || h >= height) return;
    
    const int b = bh / heads;
    const int head = bh % heads;
    const int seq_pos = h * width + w;
    
    int base_idx = ((b * heads + head) * (height * width) + seq_pos) * head_dim;
    int half_dim = head_dim / 2;
    int quarter_dim = head_dim / 4;
    
    // Apply height RoPE to first quarter of head_dim
    for (int d = 0; d < quarter_dim; d++) {
        int d0 = d * 2;
        int d1 = d * 2 + 1;
        
        float x0 = x[base_idx + d0];
        float x1 = x[base_idx + d1];
        
        float cos_val = cos_h[h * quarter_dim + d];
        float sin_val = sin_h[h * quarter_dim + d];
        
        output[base_idx + d0] = x0 * cos_val - x1 * sin_val;
        output[base_idx + d1] = x0 * sin_val + x1 * cos_val;
    }
    
    // Apply width RoPE to second quarter of head_dim
    for (int d = 0; d < quarter_dim; d++) {
        int d0 = half_dim + d * 2;
        int d1 = half_dim + d * 2 + 1;
        
        float x0 = x[base_idx + d0];
        float x1 = x[base_idx + d1];
        
        float cos_val = cos_w[w * quarter_dim + d];
        float sin_val = sin_w[w * quarter_dim + d];
        
        output[base_idx + d0] = x0 * cos_val - x1 * sin_val;
        output[base_idx + d1] = x0 * sin_val + x1 * cos_val;
    }
}

// Precompute RoPE frequencies
__kernel void rope_precompute_freqs(
    __global float* cos_out,
    __global float* sin_out,
    const int max_seq_len,
    const int head_dim,
    const float theta
) {
    const int pos = get_global_id(0);
    const int d = get_global_id(1);
    
    if (pos >= max_seq_len || d >= head_dim / 2) return;
    
    // Compute frequency: theta^(-2d/head_dim)
    float freq = pow(theta, -2.0f * (float)d / (float)head_dim);
    float angle = (float)pos * freq;
    
    int idx = pos * (head_dim / 2) + d;
    cos_out[idx] = cos(angle);
    sin_out[idx] = sin(angle);
}

// Precompute 2D RoPE frequencies for vision
__kernel void rope_2d_precompute_freqs(
    __global float* cos_h_out,
    __global float* sin_h_out,
    __global float* cos_w_out,
    __global float* sin_w_out,
    const int max_height,
    const int max_width,
    const int head_dim,
    const float theta
) {
    const int pos = get_global_id(0);
    const int d = get_global_id(1);
    const int is_width = get_global_id(2);  // 0 for height, 1 for width
    
    int max_pos = is_width ? max_width : max_height;
    int quarter_dim = head_dim / 4;
    
    if (pos >= max_pos || d >= quarter_dim) return;
    
    // Compute frequency
    float freq = pow(theta, -2.0f * (float)d / (float)(head_dim / 2));
    float angle = (float)pos * freq;
    
    int idx = pos * quarter_dim + d;
    
    if (is_width) {
        cos_w_out[idx] = cos(angle);
        sin_w_out[idx] = sin(angle);
    } else {
        cos_h_out[idx] = cos(angle);
        sin_h_out[idx] = sin(angle);
    }
}

// In-place RoPE application (saves memory)
__kernel void rope_inplace(
    __global float* x,
    __global const float* cos_freq,
    __global const float* sin_freq,
    const int batch,
    const int seq_len,
    const int heads,
    const int head_dim
) {
    const int d2 = get_global_id(0);
    const int pos = get_global_id(1);
    const int bh = get_global_id(2);
    
    if (d2 >= head_dim / 2 || pos >= seq_len) return;
    
    const int b = bh / heads;
    const int h = bh % heads;
    
    int base_idx = ((b * heads + h) * seq_len + pos) * head_dim;
    int d0 = d2 * 2;
    int d1 = d2 * 2 + 1;
    
    float x0 = x[base_idx + d0];
    float x1 = x[base_idx + d1];
    
    float cos_val = cos_freq[pos * (head_dim / 2) + d2];
    float sin_val = sin_freq[pos * (head_dim / 2) + d2];
    
    x[base_idx + d0] = x0 * cos_val - x1 * sin_val;
    x[base_idx + d1] = x0 * sin_val + x1 * cos_val;
}

