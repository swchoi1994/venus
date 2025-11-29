/**
 * Venus Inference Engine - Element-wise OpenCL Kernels
 * 
 * Basic element-wise operations and dequantization.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#ifndef DEQUANT_BLOCK
#define DEQUANT_BLOCK 32
#endif

// Vector addition: c = a + b
__kernel void vec_add(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    c[idx] = a[idx] + b[idx];
}

// Vector subtraction: c = a - b
__kernel void vec_sub(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    c[idx] = a[idx] - b[idx];
}

// Vector multiplication: c = a * b
__kernel void vec_mul(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    c[idx] = a[idx] * b[idx];
}

// Vector division: c = a / b
__kernel void vec_div(
    __global const float* a,
    __global const float* b,
    __global float* c,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    c[idx] = a[idx] / (b[idx] + 1e-9f);
}

// Vector scale: y = x * scale
__kernel void vec_scale(
    __global const float* x,
    __global float* y,
    const float scale,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    y[idx] = x[idx] * scale;
}

// Vector add scalar: y = x + scalar
__kernel void vec_add_scalar(
    __global const float* x,
    __global float* y,
    const float scalar,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    y[idx] = x[idx] + scalar;
}

// Fused multiply-add: d = a * b + c
__kernel void vec_fma(
    __global const float* a,
    __global const float* b,
    __global const float* c,
    __global float* d,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    d[idx] = fma(a[idx], b[idx], c[idx]);
}

// Copy kernel
__kernel void vec_copy(
    __global const float* src,
    __global float* dst,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    dst[idx] = src[idx];
}

// Fill kernel
__kernel void vec_fill(
    __global float* dst,
    const float value,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    dst[idx] = value;
}

// ============================================================================
// Dequantization Kernels
// ============================================================================

// Dequantize Q8_0 to float32
__kernel void dequant_q8(
    __global const char* input,
    __global const float* scales,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    int block_idx = idx / DEQUANT_BLOCK;
    float scale = scales[block_idx];
    output[idx] = (float)input[idx] * scale;
}

// Dequantize Q4_0 to float32 (packed 2 values per byte)
__kernel void dequant_q4(
    __global const uchar* input,
    __global const float* scales,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    int block_idx = idx / DEQUANT_BLOCK;
    float scale = scales[block_idx];
    
    int byte_idx = idx / 2;
    uchar packed = input[byte_idx];
    
    int val;
    if (idx % 2 == 0) {
        val = (int)(packed & 0x0F) - 8;  // Lower nibble
    } else {
        val = (int)(packed >> 4) - 8;    // Upper nibble
    }
    
    output[idx] = (float)val * scale;
}

// Dequantize Q4_K to float32 (k-means quantization)
__kernel void dequant_q4_k(
    __global const uchar* input,
    __global const float* scales,
    __global const float* mins,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    // Q4_K uses 256-element super-blocks with 8 sub-blocks of 32
    int super_block = idx / 256;
    int sub_block = (idx % 256) / 32;
    int in_block = idx % 32;
    
    float scale = scales[super_block * 8 + sub_block];
    float min_val = mins[super_block * 8 + sub_block];
    
    int byte_idx = (super_block * 128) + (sub_block * 16) + (in_block / 2);
    uchar packed = input[byte_idx];
    
    int q;
    if (in_block % 2 == 0) {
        q = (int)(packed & 0x0F);
    } else {
        q = (int)(packed >> 4);
    }
    
    output[idx] = (float)q * scale + min_val;
}

// ============================================================================
// Reduction Kernels
// ============================================================================

// Sum reduction
__kernel void reduce_sum(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int n
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    
    // Load into local memory
    scratch[local_id] = (global_id < n) ? input[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Max reduction
__kernel void reduce_max(
    __global const float* input,
    __global float* output,
    __local float* scratch,
    const int n
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    
    scratch[local_id] = (global_id < n) ? input[global_id] : -INFINITY;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            scratch[local_id] = fmax(scratch[local_id], scratch[local_id + stride]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

// Dot product
__kernel void dot_product(
    __global const float* a,
    __global const float* b,
    __global float* output,
    __local float* scratch,
    const int n
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    const int global_id = get_global_id(0);
    
    scratch[local_id] = (global_id < n) ? a[global_id] * b[global_id] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            scratch[local_id] += scratch[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (local_id == 0) {
        output[group_id] = scratch[0];
    }
}

