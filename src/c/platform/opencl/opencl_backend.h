/**
 * Venus Inference Engine - Universal OpenCL Backend
 * 
 * Provides a unified OpenCL interface that works across:
 * - Qualcomm Adreno (Snapdragon 6xx, 7xx, 8xx, 8 Elite)
 * - AMD RDNA/GCN GPUs
 * - Intel Arc/Xe/UHD Graphics
 * - NVIDIA GPUs (legacy OpenCL path)
 * - ARM Mali GPUs
 * - CPU via POCL/Intel OpenCL runtime
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#ifndef OPENCL_BACKEND_H
#define OPENCL_BACKEND_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for OpenCL types (avoid including CL headers in public API)
typedef struct _cl_context* cl_context;
typedef struct _cl_command_queue* cl_command_queue;
typedef struct _cl_program* cl_program;
typedef struct _cl_kernel* cl_kernel;
typedef struct _cl_mem* cl_mem;
typedef struct _cl_device_id* cl_device_id;
typedef struct _cl_platform_id* cl_platform_id;

// ============================================================================
// Device Detection and Vendor Types
// ============================================================================

typedef enum {
    OPENCL_VENDOR_UNKNOWN = 0,
    OPENCL_VENDOR_QUALCOMM,    // Adreno GPUs
    OPENCL_VENDOR_AMD,         // RDNA/GCN GPUs
    OPENCL_VENDOR_INTEL,       // Arc/Xe/UHD
    OPENCL_VENDOR_NVIDIA,      // GeForce/Quadro
    OPENCL_VENDOR_ARM,         // Mali GPUs
    OPENCL_VENDOR_APPLE,       // Apple GPUs (rare, usually Metal)
    OPENCL_VENDOR_CPU          // POCL/Intel CPU runtime
} OpenCLVendor;

typedef enum {
    OPENCL_DEVICE_GPU = 0,
    OPENCL_DEVICE_CPU,
    OPENCL_DEVICE_ACCELERATOR
} OpenCLDeviceType;

// Adreno generation detection
typedef enum {
    ADRENO_UNKNOWN = 0,
    ADRENO_6XX,      // Snapdragon 8xx series (older)
    ADRENO_7XX,      // Snapdragon 8 Gen 1/2
    ADRENO_8XX       // Snapdragon 8 Elite
} AdrenoGeneration;

// ============================================================================
// Device Information
// ============================================================================

typedef struct {
    char name[256];
    char vendor_string[256];
    char driver_version[64];
    char opencl_version[64];
    
    OpenCLVendor vendor;
    OpenCLDeviceType device_type;
    AdrenoGeneration adreno_gen;  // Only valid for Qualcomm
    
    // Compute capabilities
    uint32_t compute_units;
    uint32_t max_work_group_size;
    uint32_t max_work_item_dims;
    size_t max_work_item_sizes[3];
    
    // Memory
    uint64_t global_mem_size;
    uint64_t local_mem_size;
    uint64_t max_alloc_size;
    uint32_t mem_base_addr_align;
    
    // Features
    bool has_fp16;
    bool has_fp64;
    bool has_int8;
    bool has_subgroups;
    uint32_t preferred_vector_width_float;
    
    // Cache
    uint64_t global_cache_size;
    uint32_t global_cache_line_size;
} OpenCLDeviceInfo;

// ============================================================================
// Tuning Parameters (auto-configured per vendor)
// ============================================================================

typedef struct {
    // GEMM tile sizes
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    
    // Work group sizes
    uint32_t wg_size_x;
    uint32_t wg_size_y;
    uint32_t wg_size_z;
    
    // Vectorization
    uint32_t vector_width;
    
    // Memory
    bool use_local_memory;
    uint32_t local_mem_padding;
    
    // Kernel-specific
    uint32_t attention_block_size;
    uint32_t reduction_block_size;
    
    // Quantization
    uint32_t dequant_block_size;
} OpenCLTuningParams;

// ============================================================================
// Backend Context
// ============================================================================

typedef struct {
    // OpenCL handles
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    
    // Device info
    OpenCLDeviceInfo info;
    OpenCLTuningParams tuning;
    
    // Kernels (lazily compiled)
    cl_kernel kernel_gemm;
    cl_kernel kernel_gemm_q8;
    cl_kernel kernel_gemm_q4;
    cl_kernel kernel_attention_scores;
    cl_kernel kernel_attention_softmax;
    cl_kernel kernel_attention_output;
    cl_kernel kernel_rmsnorm;
    cl_kernel kernel_layernorm;
    cl_kernel kernel_silu;
    cl_kernel kernel_gelu;
    cl_kernel kernel_rope;
    cl_kernel kernel_rope_2d;
    cl_kernel kernel_add;
    cl_kernel kernel_mul;
    cl_kernel kernel_scale;
    cl_kernel kernel_dequant_q8;
    cl_kernel kernel_dequant_q4;
    
    // Memory tracking
    size_t allocated_bytes;
    size_t peak_allocated_bytes;
    uint32_t allocation_count;
    
    // State
    bool initialized;
    bool kernels_compiled;
} OpenCLBackend;

// ============================================================================
// Initialization and Cleanup
// ============================================================================

/**
 * Initialize OpenCL backend with automatic device selection.
 * Prefers GPU over CPU, and selects best available GPU.
 * 
 * @return Initialized backend or NULL on failure
 */
OpenCLBackend* opencl_init(void);

/**
 * Initialize OpenCL backend with specific vendor preference.
 * Falls back to any available device if preferred vendor not found.
 * 
 * @param preferred_vendor Preferred vendor (OPENCL_VENDOR_QUALCOMM, etc.)
 * @return Initialized backend or NULL on failure
 */
OpenCLBackend* opencl_init_with_vendor(OpenCLVendor preferred_vendor);

/**
 * Initialize OpenCL backend from environment variable.
 * Reads VENUS_OPENCL_DEVICE to determine device selection.
 * Values: "adreno", "amd", "intel", "nvidia", "cpu", "auto"
 * 
 * @return Initialized backend or NULL on failure
 */
OpenCLBackend* opencl_init_from_env(void);

/**
 * Cleanup and free OpenCL backend resources.
 * 
 * @param backend Backend to cleanup
 */
void opencl_cleanup(OpenCLBackend* backend);

/**
 * Get tuning parameters for a specific vendor.
 * 
 * @param vendor Vendor type
 * @param info Device info for fine-tuning
 * @return Optimized tuning parameters
 */
OpenCLTuningParams opencl_get_tuning_params(OpenCLVendor vendor, const OpenCLDeviceInfo* info);

// ============================================================================
// Memory Management
// ============================================================================

/**
 * Allocate GPU memory buffer.
 * 
 * @param backend OpenCL backend
 * @param size Size in bytes
 * @return GPU buffer handle or NULL on failure
 */
cl_mem opencl_alloc(OpenCLBackend* backend, size_t size);

/**
 * Allocate GPU memory and copy data from host.
 * 
 * @param backend OpenCL backend
 * @param data Host data pointer
 * @param size Size in bytes
 * @return GPU buffer handle or NULL on failure
 */
cl_mem opencl_alloc_copy(OpenCLBackend* backend, const void* data, size_t size);

/**
 * Free GPU memory buffer.
 * 
 * @param backend OpenCL backend
 * @param buffer Buffer to free
 */
void opencl_free(OpenCLBackend* backend, cl_mem buffer);

/**
 * Copy data from host to GPU.
 * 
 * @param backend OpenCL backend
 * @param dst GPU buffer
 * @param src Host data
 * @param size Size in bytes
 * @return 0 on success, error code on failure
 */
int opencl_copy_to_device(OpenCLBackend* backend, cl_mem dst, const void* src, size_t size);

/**
 * Copy data from GPU to host.
 * 
 * @param backend OpenCL backend
 * @param dst Host buffer
 * @param src GPU buffer
 * @param size Size in bytes
 * @return 0 on success, error code on failure
 */
int opencl_copy_to_host(OpenCLBackend* backend, void* dst, cl_mem src, size_t size);

/**
 * Get memory usage statistics.
 * 
 * @param backend OpenCL backend
 * @param current_bytes Output: current allocated bytes
 * @param peak_bytes Output: peak allocated bytes
 * @param allocation_count Output: number of allocations
 */
void opencl_get_memory_stats(OpenCLBackend* backend, 
                             size_t* current_bytes,
                             size_t* peak_bytes,
                             uint32_t* allocation_count);

// ============================================================================
// Matrix Operations
// ============================================================================

/**
 * General matrix multiplication: C = alpha * A @ B + beta * C
 * 
 * @param backend OpenCL backend
 * @param a Matrix A (M x K)
 * @param b Matrix B (K x N)
 * @param c Matrix C (M x N) - output
 * @param m Rows of A and C
 * @param n Columns of B and C
 * @param k Columns of A, rows of B
 * @param alpha Scaling factor for A @ B
 * @param beta Scaling factor for C
 * @return 0 on success
 */
int opencl_gemm_f32(OpenCLBackend* backend,
                    cl_mem a, cl_mem b, cl_mem c,
                    int m, int n, int k,
                    float alpha, float beta);

/**
 * Quantized GEMM with Q8_0 weights.
 * Dequantizes on-the-fly during computation.
 */
int opencl_gemm_q8(OpenCLBackend* backend,
                   cl_mem a_f32, cl_mem b_q8, cl_mem scales, cl_mem c,
                   int m, int n, int k);

/**
 * Quantized GEMM with Q4_0 weights.
 * Dequantizes on-the-fly during computation.
 */
int opencl_gemm_q4(OpenCLBackend* backend,
                   cl_mem a_f32, cl_mem b_q4, cl_mem scales, cl_mem c,
                   int m, int n, int k);

// ============================================================================
// Attention Operations
// ============================================================================

/**
 * Compute attention scores: scores = Q @ K^T / sqrt(d_k)
 * 
 * @param backend OpenCL backend
 * @param q Query tensor (batch, heads, seq_len, head_dim)
 * @param k Key tensor (batch, heads, kv_len, head_dim)
 * @param scores Output scores (batch, heads, seq_len, kv_len)
 * @param batch Batch size
 * @param heads Number of attention heads
 * @param seq_len Query sequence length
 * @param kv_len Key/Value sequence length
 * @param head_dim Head dimension
 * @param scale Attention scale (usually 1/sqrt(head_dim))
 * @return 0 on success
 */
int opencl_attention_scores(OpenCLBackend* backend,
                            cl_mem q, cl_mem k, cl_mem scores,
                            int batch, int heads, int seq_len, int kv_len, int head_dim,
                            float scale);

/**
 * Apply causal mask and softmax to attention scores.
 * 
 * @param backend OpenCL backend
 * @param scores Attention scores (in-place modification)
 * @param batch Batch size
 * @param heads Number of heads
 * @param seq_len Sequence length
 * @param kv_len Key/Value length
 * @param causal Apply causal mask
 * @return 0 on success
 */
int opencl_attention_softmax(OpenCLBackend* backend,
                             cl_mem scores,
                             int batch, int heads, int seq_len, int kv_len,
                             bool causal);

/**
 * Compute attention output: output = softmax(scores) @ V
 * 
 * @param backend OpenCL backend
 * @param scores Softmax attention scores
 * @param v Value tensor
 * @param output Output tensor
 * @param batch Batch size
 * @param heads Number of heads
 * @param seq_len Sequence length
 * @param kv_len Key/Value length
 * @param head_dim Head dimension
 * @return 0 on success
 */
int opencl_attention_output(OpenCLBackend* backend,
                            cl_mem scores, cl_mem v, cl_mem output,
                            int batch, int heads, int seq_len, int kv_len, int head_dim);

// ============================================================================
// Normalization
// ============================================================================

/**
 * RMS Normalization: y = x * rsqrt(mean(x^2) + eps) * weight
 * 
 * @param backend OpenCL backend
 * @param input Input tensor
 * @param weight Weight tensor
 * @param output Output tensor
 * @param batch Batch size
 * @param hidden_size Hidden dimension
 * @param eps Epsilon for numerical stability
 * @return 0 on success
 */
int opencl_rmsnorm(OpenCLBackend* backend,
                   cl_mem input, cl_mem weight, cl_mem output,
                   int batch, int hidden_size, float eps);

/**
 * Layer Normalization: y = (x - mean) / sqrt(var + eps) * weight + bias
 * 
 * @param backend OpenCL backend
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor (can be NULL)
 * @param output Output tensor
 * @param batch Batch size
 * @param hidden_size Hidden dimension
 * @param eps Epsilon for numerical stability
 * @return 0 on success
 */
int opencl_layernorm(OpenCLBackend* backend,
                     cl_mem input, cl_mem weight, cl_mem bias, cl_mem output,
                     int batch, int hidden_size, float eps);

// ============================================================================
// Activation Functions
// ============================================================================

/**
 * SiLU activation: y = x * sigmoid(x)
 */
int opencl_silu(OpenCLBackend* backend, cl_mem input, cl_mem output, int n);

/**
 * GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 */
int opencl_gelu(OpenCLBackend* backend, cl_mem input, cl_mem output, int n);

/**
 * ReLU activation: y = max(0, x)
 */
int opencl_relu(OpenCLBackend* backend, cl_mem input, cl_mem output, int n);

// ============================================================================
// Position Embeddings
// ============================================================================

/**
 * Apply Rotary Position Embedding (RoPE).
 * 
 * @param backend OpenCL backend
 * @param x Input tensor (batch, seq_len, heads, head_dim)
 * @param cos Cosine frequencies
 * @param sin Sine frequencies
 * @param output Output tensor
 * @param batch Batch size
 * @param seq_len Sequence length
 * @param heads Number of heads
 * @param head_dim Head dimension (must be even)
 * @return 0 on success
 */
int opencl_rope(OpenCLBackend* backend,
                cl_mem x, cl_mem cos, cl_mem sin, cl_mem output,
                int batch, int seq_len, int heads, int head_dim);

/**
 * Apply 2D Rotary Position Embedding for vision (Qwen2-VL style).
 * 
 * @param backend OpenCL backend
 * @param x Input tensor
 * @param cos_h Height cosine frequencies
 * @param sin_h Height sine frequencies
 * @param cos_w Width cosine frequencies
 * @param sin_w Width sine frequencies
 * @param output Output tensor
 * @param batch Batch size
 * @param height Image height (in patches)
 * @param width Image width (in patches)
 * @param heads Number of heads
 * @param head_dim Head dimension
 * @return 0 on success
 */
int opencl_rope_2d(OpenCLBackend* backend,
                   cl_mem x, 
                   cl_mem cos_h, cl_mem sin_h,
                   cl_mem cos_w, cl_mem sin_w,
                   cl_mem output,
                   int batch, int height, int width, int heads, int head_dim);

// ============================================================================
// Element-wise Operations
// ============================================================================

/**
 * Element-wise addition: c = a + b
 */
int opencl_add(OpenCLBackend* backend, cl_mem a, cl_mem b, cl_mem c, int n);

/**
 * Element-wise multiplication: c = a * b
 */
int opencl_mul(OpenCLBackend* backend, cl_mem a, cl_mem b, cl_mem c, int n);

/**
 * Scale: y = x * scale
 */
int opencl_scale(OpenCLBackend* backend, cl_mem x, cl_mem y, float scale, int n);

/**
 * Fused add and scale: y = (a + b) * scale
 */
int opencl_add_scale(OpenCLBackend* backend, cl_mem a, cl_mem b, cl_mem y, float scale, int n);

// ============================================================================
// Quantization/Dequantization
// ============================================================================

/**
 * Dequantize Q8_0 to float32.
 * 
 * @param backend OpenCL backend
 * @param input Quantized Q8_0 data
 * @param scales Scale factors (one per block of 32)
 * @param output Float32 output
 * @param n Number of elements
 * @return 0 on success
 */
int opencl_dequant_q8(OpenCLBackend* backend,
                      cl_mem input, cl_mem scales, cl_mem output, int n);

/**
 * Dequantize Q4_0 to float32.
 * 
 * @param backend OpenCL backend
 * @param input Quantized Q4_0 data (packed, 2 values per byte)
 * @param scales Scale factors (one per block of 32)
 * @param output Float32 output
 * @param n Number of elements
 * @return 0 on success
 */
int opencl_dequant_q4(OpenCLBackend* backend,
                      cl_mem input, cl_mem scales, cl_mem output, int n);

// ============================================================================
// Synchronization
// ============================================================================

/**
 * Wait for all queued operations to complete.
 */
int opencl_synchronize(OpenCLBackend* backend);

/**
 * Flush command queue (non-blocking).
 */
int opencl_flush(OpenCLBackend* backend);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get human-readable vendor name.
 */
const char* opencl_vendor_name(OpenCLVendor vendor);

/**
 * Get human-readable Adreno generation name.
 */
const char* opencl_adreno_gen_name(AdrenoGeneration gen);

/**
 * Print device info to stdout.
 */
void opencl_print_device_info(const OpenCLDeviceInfo* info);

/**
 * Check if OpenCL is available on this system.
 */
bool opencl_is_available(void);

/**
 * Get last error message.
 */
const char* opencl_get_error(void);

#ifdef __cplusplus
}
#endif

#endif // OPENCL_BACKEND_H

