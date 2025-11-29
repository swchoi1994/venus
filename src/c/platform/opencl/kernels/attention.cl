/**
 * Venus Inference Engine - Attention OpenCL Kernels
 * 
 * Multi-head attention operations for transformer models.
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

// Compute attention scores: scores = Q @ K^T * scale
__kernel void attention_scores(
    __global const float* Q,
    __global const float* K,
    __global float* scores,
    const int batch,
    const int heads,
    const int seq_len,
    const int kv_len,
    const int head_dim,
    const float scale
) {
    const int q_pos = get_global_id(0);
    const int k_pos = get_global_id(1);
    const int bh = get_global_id(2);  // batch * heads
    
    if (q_pos >= seq_len || k_pos >= kv_len) return;
    
    const int b = bh / heads;
    const int h = bh % heads;
    
    // Compute dot product Q[q_pos] . K[k_pos]
    float dot = 0.0f;
    int q_offset = ((b * heads + h) * seq_len + q_pos) * head_dim;
    int k_offset = ((b * heads + h) * kv_len + k_pos) * head_dim;
    
    #pragma unroll 8
    for (int d = 0; d < head_dim; d++) {
        dot += Q[q_offset + d] * K[k_offset + d];
    }
    
    // Store scaled score
    int score_idx = ((b * heads + h) * seq_len + q_pos) * kv_len + k_pos;
    scores[score_idx] = dot * scale;
}

// Apply causal mask and softmax to attention scores
__kernel void attention_softmax(
    __global float* scores,
    const int batch,
    const int heads,
    const int seq_len,
    const int kv_len,
    const int causal
) {
    const int q_pos = get_global_id(0);
    const int bh = get_global_id(1);
    
    if (q_pos >= seq_len) return;
    
    int base_idx = (bh * seq_len + q_pos) * kv_len;
    
    // Apply causal mask and find max for numerical stability
    float max_val = -INFINITY;
    for (int k = 0; k < kv_len; k++) {
        if (causal && k > q_pos) {
            scores[base_idx + k] = -INFINITY;
        } else {
            max_val = fmax(max_val, scores[base_idx + k]);
        }
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int k = 0; k < kv_len; k++) {
        if (!causal || k <= q_pos) {
            float val = exp(scores[base_idx + k] - max_val);
            scores[base_idx + k] = val;
            sum += val;
        }
    }
    
    // Normalize
    float inv_sum = 1.0f / (sum + 1e-9f);
    for (int k = 0; k < kv_len; k++) {
        if (causal && k > q_pos) {
            scores[base_idx + k] = 0.0f;
        } else {
            scores[base_idx + k] *= inv_sum;
        }
    }
}

// Compute attention output: output = scores @ V
__kernel void attention_output(
    __global const float* scores,
    __global const float* V,
    __global float* output,
    const int batch,
    const int heads,
    const int seq_len,
    const int kv_len,
    const int head_dim
) {
    const int q_pos = get_global_id(0);
    const int d = get_global_id(1);
    const int bh = get_global_id(2);
    
    if (q_pos >= seq_len || d >= head_dim) return;
    
    const int b = bh / heads;
    const int h = bh % heads;
    
    // Weighted sum of values
    float sum = 0.0f;
    int score_base = (bh * seq_len + q_pos) * kv_len;
    
    for (int k = 0; k < kv_len; k++) {
        float score = scores[score_base + k];
        int v_idx = ((b * heads + h) * kv_len + k) * head_dim + d;
        sum += score * V[v_idx];
    }
    
    int out_idx = ((b * heads + h) * seq_len + q_pos) * head_dim + d;
    output[out_idx] = sum;
}

// Fused attention for small sequences (no intermediate storage)
__kernel void attention_fused(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const int batch,
    const int heads,
    const int seq_len,
    const int kv_len,
    const int head_dim,
    const float scale,
    const int causal
) {
    const int q_pos = get_global_id(0);
    const int d = get_global_id(1);
    const int bh = get_global_id(2);
    
    if (q_pos >= seq_len || d >= head_dim) return;
    
    const int b = bh / heads;
    const int h = bh % heads;
    
    int q_base = ((b * heads + h) * seq_len + q_pos) * head_dim;
    
    // Compute all attention scores for this query position
    float scores[256];  // Assuming max kv_len of 256 for fused version
    float max_score = -INFINITY;
    
    for (int k = 0; k < kv_len && k < 256; k++) {
        if (causal && k > q_pos) {
            scores[k] = -INFINITY;
            continue;
        }
        
        int k_base = ((b * heads + h) * kv_len + k) * head_dim;
        float dot = 0.0f;
        for (int dd = 0; dd < head_dim; dd++) {
            dot += Q[q_base + dd] * K[k_base + dd];
        }
        scores[k] = dot * scale;
        max_score = fmax(max_score, scores[k]);
    }
    
    // Softmax
    float sum = 0.0f;
    for (int k = 0; k < kv_len && k < 256; k++) {
        if (causal && k > q_pos) continue;
        scores[k] = exp(scores[k] - max_score);
        sum += scores[k];
    }
    
    float inv_sum = 1.0f / (sum + 1e-9f);
    
    // Compute weighted output
    float out_val = 0.0f;
    for (int k = 0; k < kv_len && k < 256; k++) {
        if (causal && k > q_pos) continue;
        float weight = scores[k] * inv_sum;
        int v_idx = ((b * heads + h) * kv_len + k) * head_dim + d;
        out_val += weight * V[v_idx];
    }
    
    int out_idx = ((b * heads + h) * seq_len + q_pos) * head_dim + d;
    output[out_idx] = out_val;
}

