#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <stdbool.h>

// Sampling utilities for text generation

// Sample from logits using temperature
int sample_temperature(const float* logits, int vocab_size, float temperature, unsigned int* seed) {
    if (temperature <= 0.0f) {
        // Greedy sampling (argmax)
        int best_idx = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) {
                best_val = logits[i];
                best_idx = i;
            }
        }
        return best_idx;
    }
    
    // Apply temperature scaling
    float* scaled_logits = (float*)malloc(vocab_size * sizeof(float));
    if (!scaled_logits) return 0;
    
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] = logits[i] / temperature;
    }
    
    // Convert to probabilities with softmax
    float max_logit = scaled_logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (scaled_logits[i] > max_logit) {
            max_logit = scaled_logits[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] = expf(scaled_logits[i] - max_logit);
        sum += scaled_logits[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        scaled_logits[i] /= sum;
    }
    
    // Sample from distribution
    float r = (float)rand_r(seed) / (float)RAND_MAX;
    float cumsum = 0.0f;
    int sampled = vocab_size - 1;
    
    for (int i = 0; i < vocab_size; i++) {
        cumsum += scaled_logits[i];
        if (r < cumsum) {
            sampled = i;
            break;
        }
    }
    
    free(scaled_logits);
    return sampled;
}

// Top-k sampling
int sample_top_k(const float* logits, int vocab_size, int k, float temperature, unsigned int* seed) {
    if (k <= 0 || k > vocab_size) {
        return sample_temperature(logits, vocab_size, temperature, seed);
    }
    
    // Create indices array
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    if (!indices) return 0;
    
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
    }
    
    // Partial sort to get top-k
    for (int i = 0; i < k; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[indices[j]] > logits[indices[i]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Create top-k logits
    float* top_k_logits = (float*)calloc(vocab_size, sizeof(float));
    if (!top_k_logits) {
        free(indices);
        return 0;
    }
    
    for (int i = 0; i < k; i++) {
        top_k_logits[indices[i]] = logits[indices[i]];
    }
    
    // Set rest to -inf
    for (int i = 0; i < vocab_size; i++) {
        bool in_top_k = false;
        for (int j = 0; j < k; j++) {
            if (i == indices[j]) {
                in_top_k = true;
                break;
            }
        }
        if (!in_top_k) {
            top_k_logits[i] = -FLT_MAX;
        }
    }
    
    int sampled = sample_temperature(top_k_logits, vocab_size, temperature, seed);
    
    free(indices);
    free(top_k_logits);
    return sampled;
}

// Top-p (nucleus) sampling
int sample_top_p(const float* logits, int vocab_size, float p, float temperature, unsigned int* seed) {
    if (p <= 0.0f || p >= 1.0f) {
        return sample_temperature(logits, vocab_size, temperature, seed);
    }
    
    // Create indices array
    int* indices = (int*)malloc(vocab_size * sizeof(int));
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    if (!indices || !probs) {
        free(indices);
        free(probs);
        return 0;
    }
    
    // Apply temperature and softmax
    for (int i = 0; i < vocab_size; i++) {
        indices[i] = i;
        probs[i] = logits[i] / temperature;
    }
    
    // Softmax
    float max_logit = probs[0];
    for (int i = 1; i < vocab_size; i++) {
        if (probs[i] > max_logit) {
            max_logit = probs[i];
        }
    }
    
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf(probs[i] - max_logit);
        sum += probs[i];
    }
    
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    // Sort by probability
    for (int i = 0; i < vocab_size - 1; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (probs[indices[j]] > probs[indices[i]]) {
                int temp = indices[i];
                indices[i] = indices[j];
                indices[j] = temp;
            }
        }
    }
    
    // Find cutoff for top-p
    float cumsum = 0.0f;
    int cutoff = vocab_size;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += probs[indices[i]];
        if (cumsum > p) {
            cutoff = i + 1;
            break;
        }
    }
    
    // Create nucleus logits
    float* nucleus_logits = (float*)malloc(vocab_size * sizeof(float));
    if (!nucleus_logits) {
        free(indices);
        free(probs);
        return 0;
    }
    
    for (int i = 0; i < vocab_size; i++) {
        nucleus_logits[i] = -FLT_MAX;
    }
    
    for (int i = 0; i < cutoff; i++) {
        nucleus_logits[indices[i]] = logits[indices[i]];
    }
    
    int sampled = sample_temperature(nucleus_logits, vocab_size, temperature, seed);
    
    free(indices);
    free(probs);
    free(nucleus_logits);
    return sampled;
}

// Combined sampling with all parameters
int sample_logits(const float* logits, int vocab_size, float temperature, 
                  float top_p, int top_k, unsigned int* seed) {
    // Apply top-k first if specified
    if (top_k > 0 && top_k < vocab_size) {
        // Create temporary buffer for top-k filtered logits
        float* filtered_logits = (float*)malloc(vocab_size * sizeof(float));
        if (!filtered_logits) return 0;
        
        memcpy(filtered_logits, logits, vocab_size * sizeof(float));
        
        // Apply top-k filtering
        int sampled = sample_top_k(filtered_logits, vocab_size, top_k, temperature, seed);
        free(filtered_logits);
        return sampled;
    }
    
    // Apply top-p if specified
    if (top_p > 0.0f && top_p < 1.0f) {
        return sample_top_p(logits, vocab_size, top_p, temperature, seed);
    }
    
    // Default to temperature sampling
    return sample_temperature(logits, vocab_size, temperature, seed);
}