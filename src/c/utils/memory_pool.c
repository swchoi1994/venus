#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct MemoryBlock {
    void* data;
    size_t size;
    int in_use;
    struct MemoryBlock* next;
} MemoryBlock;

typedef struct {
    MemoryBlock* blocks;
    size_t total_allocated;
    size_t total_used;
    int n_blocks;
} MemoryPool;

static MemoryPool* global_pool = NULL;

// Initialize memory pool
void init_memory_pool(void) {
    if (global_pool) return;
    
    global_pool = (MemoryPool*)calloc(1, sizeof(MemoryPool));
    if (!global_pool) {
        fprintf(stderr, "Failed to initialize memory pool\n");
        exit(1);
    }
}

// Allocate from pool
void* pool_alloc(size_t size) {
    if (!global_pool) init_memory_pool();
    
    // First, try to find a free block of sufficient size
    MemoryBlock* block = global_pool->blocks;
    while (block) {
        if (!block->in_use && block->size >= size) {
            block->in_use = 1;
            global_pool->total_used += block->size;
            return block->data;
        }
        block = block->next;
    }
    
    // No suitable block found, allocate new one
    MemoryBlock* new_block = (MemoryBlock*)malloc(sizeof(MemoryBlock));
    if (!new_block) return NULL;
    
    new_block->data = calloc(1, size);
    if (!new_block->data) {
        free(new_block);
        return NULL;
    }
    
    new_block->size = size;
    new_block->in_use = 1;
    new_block->next = global_pool->blocks;
    global_pool->blocks = new_block;
    
    global_pool->total_allocated += size;
    global_pool->total_used += size;
    global_pool->n_blocks++;
    
    return new_block->data;
}

// Free to pool (mark as available, don't actually free)
void pool_free(void* ptr) {
    if (!global_pool || !ptr) return;
    
    MemoryBlock* block = global_pool->blocks;
    while (block) {
        if (block->data == ptr) {
            block->in_use = 0;
            global_pool->total_used -= block->size;
            return;
        }
        block = block->next;
    }
}

// Cleanup memory pool
void cleanup_memory_pool(void) {
    if (!global_pool) return;
    
    MemoryBlock* block = global_pool->blocks;
    while (block) {
        MemoryBlock* next = block->next;
        free(block->data);
        free(block);
        block = next;
    }
    
    free(global_pool);
    global_pool = NULL;
}

// Get pool statistics
void get_pool_stats(size_t* allocated, size_t* used, int* n_blocks) {
    if (!global_pool) {
        *allocated = 0;
        *used = 0;
        *n_blocks = 0;
        return;
    }
    
    *allocated = global_pool->total_allocated;
    *used = global_pool->total_used;
    *n_blocks = global_pool->n_blocks;
}