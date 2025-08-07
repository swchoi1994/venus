#ifndef ATTENTION_H
#define ATTENTION_H

#include "inference_engine.h"
#include "tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// KV Cache structure
typedef struct {
    Tensor** k_cache;  // [n_layers][seq_len, n_kv_heads, head_dim]
    Tensor** v_cache;  // [n_layers][seq_len, n_kv_heads, head_dim]
    int n_layers;
    int max_seq_len;
    int n_kv_heads;
    int head_dim;
    int current_pos;
} KVCache;

// Attention operations
typedef struct {
    // Standard attention
    void (*standard_attention)(Tensor* q, Tensor* k, Tensor* v, Tensor* out, 
                              float scale, Tensor* mask);
    
    // Flash attention
    void (*flash_attention)(Tensor* q, Tensor* k, Tensor* v, Tensor* out,
                           float scale, Tensor* mask);
    
    // Paged attention
    void (*paged_attention)(Tensor* q, KVCache* kv_cache, Tensor* out,
                           int layer_idx, float scale);
    
    // Grouped Query Attention (GQA)
    void (*grouped_query_attention)(Tensor* q, Tensor* k, Tensor* v, Tensor* out,
                                   int n_groups, float scale, Tensor* mask);
} AttentionOps;

// RoPE (Rotary Position Embeddings)
typedef struct {
    float* cos_cached;
    float* sin_cached;
    int max_seq_len;
    int head_dim;
    float theta;
} RoPECache;

// Create attention operations
AttentionOps* create_attention_ops(ModelConfig* config);
void free_attention_ops(AttentionOps* ops);

// KV Cache management
KVCache* create_kv_cache(int n_layers, int max_seq_len, int n_kv_heads, int head_dim);
void free_kv_cache(KVCache* cache);
void kv_cache_update(KVCache* cache, Tensor* k, Tensor* v, int layer_idx, int pos);
void kv_cache_clear(KVCache* cache);

// RoPE operations
RoPECache* create_rope_cache(int max_seq_len, int head_dim, float theta);
void free_rope_cache(RoPECache* cache);
void apply_rope(Tensor* q, Tensor* k, RoPECache* cache, int pos);

// Attention implementations
void standard_attention_impl(Tensor* q, Tensor* k, Tensor* v, Tensor* out,
                            float scale, Tensor* mask);
void flash_attention_impl(Tensor* q, Tensor* k, Tensor* v, Tensor* out,
                         float scale, Tensor* mask);
void paged_attention_impl(Tensor* q, KVCache* kv_cache, Tensor* out,
                         int layer_idx, float scale);

// Helper functions
Tensor* create_causal_mask(int seq_len);
float get_attention_scale(int head_dim);

#ifdef __cplusplus
}
#endif

#endif // ATTENTION_H