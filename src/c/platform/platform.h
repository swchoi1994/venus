#ifndef PLATFORM_H
#define PLATFORM_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Platform detection
typedef enum {
    PLATFORM_GENERIC,
    PLATFORM_APPLE_SILICON_ENUM,
    PLATFORM_X86_64,
    PLATFORM_ARM64,
    PLATFORM_RISCV64,
    PLATFORM_MIPS,            // MIPS processors
    PLATFORM_MIPS64,          // 64-bit MIPS
    PLATFORM_POWERPC,         // PowerPC (32-bit)
    PLATFORM_POWERPC64,       // PowerPC64
    PLATFORM_SPARC,           // SPARC processors  
    PLATFORM_SPARC64,         // SPARC64/UltraSPARC
    PLATFORM_LOONGARCH64,     // LoongArch processors
    PLATFORM_S390X,           // IBM z/Architecture
    PLATFORM_WASM32,          // WebAssembly 32-bit
    PLATFORM_WASM64           // WebAssembly 64-bit
} Platform;

// SIMD operations interface
typedef struct {
    // Vector operations
    void (*vec_add_f32)(const float* a, const float* b, float* c, size_t n);
    void (*vec_mul_f32)(const float* a, const float* b, float* c, size_t n);
    void (*vec_scale_f32)(const float* a, float scale, float* c, size_t n);
    float (*vec_dot_f32)(const float* a, const float* b, size_t n);
    
    // Matrix operations
    void (*gemm_f32)(const float* a, const float* b, float* c,
                     int m, int n, int k, float alpha, float beta);
    void (*gemv_f32)(const float* a, const float* x, float* y,
                     int m, int n, float alpha, float beta);
    
    // Activation functions
    void (*gelu_f32)(const float* x, float* y, size_t n);
    void (*silu_f32)(const float* x, float* y, size_t n);
    void (*relu_f32)(const float* x, float* y, size_t n);
    
    // Reduction operations
    float (*sum_f32)(const float* x, size_t n);
    float (*max_f32)(const float* x, size_t n);
    
    // Memory operations
    void (*memcpy_aligned)(void* dst, const void* src, size_t n);
    void (*memset_f32)(float* dst, float value, size_t n);
} SimdOps;

// Platform information
typedef struct {
    Platform platform;
    const char* name;
    int num_cores;
    size_t cache_line_size;
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t l3_cache_size;
    bool has_avx;
    bool has_avx2;
    bool has_avx512;
    bool has_neon;
    bool has_sve;
    bool has_amx;
} PlatformInfo;

// Platform detection and initialization
Platform detect_platform(void);
PlatformInfo* get_platform_info(void);
SimdOps* get_platform_ops(void);

// Platform-specific initialization
void init_platform(void);
void cleanup_platform(void);

// Utility functions
const char* get_platform_name(void);
const char* get_simd_features(void);
int get_cpu_count(void);
size_t get_total_memory(void);
size_t get_available_memory(void);

// Threading utilities
void set_thread_affinity(int thread_id, int core_id);
int get_optimal_thread_count(void);

#ifdef __cplusplus
}
#endif

#endif // PLATFORM_H