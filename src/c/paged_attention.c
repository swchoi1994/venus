#include "attention.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// PagedAttention implementation for efficient KV cache management

typedef struct {
    int block_size;
    int n_blocks;
    int n_free_blocks;
    float** blocks;
    int* block_table;
    int* free_list;
} PagedKVCache;

// Create paged KV cache
PagedKVCache* create_paged_kv_cache(int n_layers, int max_blocks, int block_size, int head_dim) {
    PagedKVCache* cache = (PagedKVCache*)calloc(1, sizeof(PagedKVCache));
    if (!cache) return NULL;
    
    cache->block_size = block_size;
    cache->n_blocks = max_blocks;
    cache->n_free_blocks = max_blocks;
    
    // Allocate blocks
    cache->blocks = (float**)calloc(max_blocks, sizeof(float*));
    cache->block_table = (int*)calloc(max_blocks, sizeof(int));
    cache->free_list = (int*)malloc(max_blocks * sizeof(int));
    
    if (!cache->blocks || !cache->block_table || !cache->free_list) {
        free(cache->blocks);
        free(cache->block_table);
        free(cache->free_list);
        free(cache);
        return NULL;
    }
    
    // Initialize blocks and free list
    for (int i = 0; i < max_blocks; i++) {
        cache->blocks[i] = (float*)calloc(block_size * head_dim, sizeof(float));
        cache->free_list[i] = i;
        cache->block_table[i] = -1;
    }
    
    return cache;
}

// Free paged KV cache
void free_paged_kv_cache(PagedKVCache* cache) {
    if (!cache) return;
    
    for (int i = 0; i < cache->n_blocks; i++) {
        free(cache->blocks[i]);
    }
    
    free(cache->blocks);
    free(cache->block_table);
    free(cache->free_list);
    free(cache);
}

// Allocate block from cache
int allocate_block(PagedKVCache* cache) {
    if (cache->n_free_blocks == 0) {
        return -1; // No free blocks
    }
    
    int block_idx = cache->free_list[--cache->n_free_blocks];
    return block_idx;
}

// Free block back to cache
void free_block(PagedKVCache* cache, int block_idx) {
    if (block_idx < 0 || block_idx >= cache->n_blocks) return;
    
    cache->free_list[cache->n_free_blocks++] = block_idx;
    memset(cache->blocks[block_idx], 0, cache->block_size * sizeof(float));
}

// Paged attention implementation
void paged_attention_impl(Tensor* q, KVCache* kv_cache, Tensor* out,
                         int layer_idx, float scale) {
    // TODO: Implement paged attention with block-based KV cache
    // For now, fall back to standard attention
    standard_attention_impl(q, kv_cache->k_cache[layer_idx], 
                           kv_cache->v_cache[layer_idx], out, scale, NULL);
}