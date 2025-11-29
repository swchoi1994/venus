/**
 * Venus Inference Engine - Activation OpenCL Kernels
 * 
 * Activation functions for neural networks.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

// SiLU (Swish) activation: x * sigmoid(x)
__kernel void silu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[idx] = x * sigmoid_x;
}

// GELU activation (approximate version)
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__kernel void gelu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    const float SQRT_2_PI = 0.7978845608f;
    float x3 = x * x * x;
    float inner = SQRT_2_PI * (x + 0.044715f * x3);
    output[idx] = 0.5f * x * (1.0f + tanh(inner));
}

// GELU activation (exact version using erf)
__kernel void gelu_exact(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    const float SQRT_2_INV = 0.7071067811865475f;
    output[idx] = 0.5f * x * (1.0f + erf(x * SQRT_2_INV));
}

// QuickGELU (used in CLIP/SigLIP)
// x * sigmoid(1.702 * x)
__kernel void quick_gelu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + exp(-1.702f * x));
    output[idx] = x * sigmoid_x;
}

// ReLU activation: max(0, x)
__kernel void relu(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    output[idx] = fmax(0.0f, input[idx]);
}

// Leaky ReLU: max(alpha * x, x)
__kernel void leaky_relu(
    __global const float* input,
    __global float* output,
    const float alpha,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    float x = input[idx];
    output[idx] = x >= 0.0f ? x : alpha * x;
}

// Sigmoid activation: 1 / (1 + exp(-x))
__kernel void sigmoid(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    output[idx] = 1.0f / (1.0f + exp(-input[idx]));
}

// Tanh activation
__kernel void tanh_act(
    __global const float* input,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    output[idx] = tanh(input[idx]);
}

// Softmax (along last dimension)
__kernel void softmax(
    __global float* data,
    const int batch,
    const int dim
) {
    const int b = get_global_id(0);
    if (b >= batch) return;
    
    int base = b * dim;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < dim; i++) {
        max_val = fmax(max_val, data[base + i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = exp(data[base + i] - max_val);
        data[base + i] = val;
        sum += val;
    }
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int i = 0; i < dim; i++) {
        data[base + i] *= inv_sum;
    }
}

// Fused SiLU-Gate (for LLaMA/Qwen MLP)
// output = silu(gate) * up
__kernel void silu_gate(
    __global const float* gate,
    __global const float* up,
    __global float* output,
    const int n
) {
    const int idx = get_global_id(0);
    if (idx >= n) return;
    
    float g = gate[idx];
    float sigmoid_g = 1.0f / (1.0f + exp(-g));
    output[idx] = g * sigmoid_g * up[idx];
}

