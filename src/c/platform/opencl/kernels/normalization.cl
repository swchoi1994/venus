/**
 * Venus Inference Engine - Normalization OpenCL Kernels
 * 
 * RMSNorm and LayerNorm implementations.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#ifndef REDUCTION_BLOCK
#define REDUCTION_BLOCK 256
#endif

// RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight
__kernel void rmsnorm(
    __global const float* input,
    __global const float* weight,
    __global float* output,
    const int batch,
    const int hidden_size,
    const float eps
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    
    if (group_id >= batch) return;
    
    __local float shared[REDUCTION_BLOCK];
    
    // Compute sum of squares (parallel reduction)
    float sum_sq = 0.0f;
    for (int i = local_id; i < hidden_size; i += local_size) {
        float val = input[group_id * hidden_size + i];
        sum_sq += val * val;
    }
    
    shared[local_id] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared[local_id] += shared[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Compute RMS inverse
    float rms_inv = rsqrt(shared[0] / hidden_size + eps);
    
    // Apply normalization and weight
    for (int i = local_id; i < hidden_size; i += local_size) {
        float val = input[group_id * hidden_size + i];
        output[group_id * hidden_size + i] = val * rms_inv * weight[i];
    }
}

// Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
__kernel void layernorm(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int batch,
    const int hidden_size,
    const float eps
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    
    if (group_id >= batch) return;
    
    __local float shared_sum[REDUCTION_BLOCK];
    __local float shared_sq[REDUCTION_BLOCK];
    
    // Compute sum and sum of squares
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int i = local_id; i < hidden_size; i += local_size) {
        float val = input[group_id * hidden_size + i];
        sum += val;
        sum_sq += val * val;
    }
    
    shared_sum[local_id] = sum;
    shared_sq[local_id] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
            shared_sq[local_id] += shared_sq[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Compute mean and inverse standard deviation
    float mean = shared_sum[0] / hidden_size;
    float var = shared_sq[0] / hidden_size - mean * mean;
    float inv_std = rsqrt(var + eps);
    
    // Apply normalization, weight, and bias
    for (int i = local_id; i < hidden_size; i += local_size) {
        float val = input[group_id * hidden_size + i];
        float normalized = (val - mean) * inv_std;
        float result = normalized * weight[i];
        if (bias) result += bias[i];
        output[group_id * hidden_size + i] = result;
    }
}

// Group Normalization (for vision models)
__kernel void groupnorm(
    __global const float* input,
    __global const float* weight,
    __global const float* bias,
    __global float* output,
    const int batch,
    const int channels,
    const int spatial,
    const int num_groups,
    const float eps
) {
    const int local_id = get_local_id(0);
    const int group_id = get_group_id(0);
    const int local_size = get_local_size(0);
    
    const int b = group_id / num_groups;
    const int g = group_id % num_groups;
    const int channels_per_group = channels / num_groups;
    const int group_size = channels_per_group * spatial;
    
    if (b >= batch) return;
    
    __local float shared_sum[REDUCTION_BLOCK];
    __local float shared_sq[REDUCTION_BLOCK];
    
    // Compute sum and sum of squares for this group
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    int base = b * channels * spatial + g * channels_per_group * spatial;
    for (int i = local_id; i < group_size; i += local_size) {
        int c = i / spatial;
        int s = i % spatial;
        int idx = base + c * spatial + s;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    shared_sum[local_id] = sum;
    shared_sq[local_id] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Tree reduction
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
            shared_sq[local_id] += shared_sq[local_id + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Compute mean and inverse standard deviation
    float mean = shared_sum[0] / group_size;
    float var = shared_sq[0] / group_size - mean * mean;
    float inv_std = rsqrt(var + eps);
    
    // Apply normalization, weight, and bias
    for (int i = local_id; i < group_size; i += local_size) {
        int c = i / spatial;
        int s = i % spatial;
        int global_c = g * channels_per_group + c;
        int idx = base + c * spatial + s;
        
        float val = input[idx];
        float normalized = (val - mean) * inv_std;
        float result = normalized * weight[global_c];
        if (bias) result += bias[global_c];
        output[idx] = result;
    }
}

