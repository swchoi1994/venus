#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <mach/mach.h>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

// Apple Silicon specific SIMD operations

static void apple_vec_add_f32(const float* a, const float* b, float* c, size_t n) {
    vDSP_vadd(a, 1, b, 1, c, 1, n);
}

static void apple_vec_mul_f32(const float* a, const float* b, float* c, size_t n) {
    vDSP_vmul(a, 1, b, 1, c, 1, n);
}

static void apple_vec_scale_f32(const float* a, float scale, float* c, size_t n) {
    vDSP_vsmul(a, 1, &scale, c, 1, n);
}

static float apple_vec_dot_f32(const float* a, const float* b, size_t n) {
    float result;
    vDSP_dotpr(a, 1, b, 1, &result, n);
    return result;
}

static void apple_gemm_f32(const float* a, const float* b, float* c,
                          int m, int n, int k, float alpha, float beta) {
    // Use Accelerate's BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, a, k, b, n, beta, c, n);
}

static void apple_gemv_f32(const float* a, const float* x, float* y,
                          int m, int n, float alpha, float beta) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, a, n, x, 1, beta, y, 1);
}

static void apple_gelu_f32(const float* x, float* y, size_t n) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    
    for (size_t i = 0; i < n; i++) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float arg = sqrt_2_over_pi * (xi + 0.044715f * x_cubed);
        y[i] = 0.5f * xi * (1.0f + tanhf(arg));
    }
}

static void apple_silu_f32(const float* x, float* y, size_t n) {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void apple_relu_f32(const float* x, float* y, size_t n) {
    float zero = 0.0f;
    vDSP_vmax(x, 1, &zero, 0, y, 1, n);
}

static float apple_sum_f32(const float* x, size_t n) {
    float result;
    vDSP_sve(x, 1, &result, n);
    return result;
}

static float apple_max_f32(const float* x, size_t n) {
    float result;
    vDSP_maxv(x, 1, &result, n);
    return result;
}

static void apple_memcpy_aligned(void* dst, const void* src, size_t n) {
    memcpy(dst, src, n);
}

static void apple_memset_f32(float* dst, float value, size_t n) {
    vDSP_vfill(&value, dst, 1, n);
}

// Platform information functions

static int get_apple_cpu_count(void) {
    int count;
    size_t size = sizeof(count);
    if (sysctlbyname("hw.ncpu", &count, &size, NULL, 0) == 0) {
        return count;
    }
    return 1;
}

static size_t get_apple_memory(void) {
    int64_t memsize;
    size_t size = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &size, NULL, 0) == 0) {
        return (size_t)memsize;
    }
    return 0;
}

static size_t get_apple_available_memory(void) {
    vm_size_t page_size;
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
    
    if (host_page_size(mach_host_self(), &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
        return (size_t)(vm_stat.free_count * page_size);
    }
    return 0;
}

// Initialize Apple Silicon platform
static SimdOps apple_simd_ops = {
    .vec_add_f32 = apple_vec_add_f32,
    .vec_mul_f32 = apple_vec_mul_f32,
    .vec_scale_f32 = apple_vec_scale_f32,
    .vec_dot_f32 = apple_vec_dot_f32,
    .gemm_f32 = apple_gemm_f32,
    .gemv_f32 = apple_gemv_f32,
    .gelu_f32 = apple_gelu_f32,
    .silu_f32 = apple_silu_f32,
    .relu_f32 = apple_relu_f32,
    .sum_f32 = apple_sum_f32,
    .max_f32 = apple_max_f32,
    .memcpy_aligned = apple_memcpy_aligned,
    .memset_f32 = apple_memset_f32
};

static PlatformInfo apple_platform_info = {
    .platform = PLATFORM_APPLE_SILICON_ENUM,
    .name = "Apple Silicon",
    .cache_line_size = 128,
    .l1_cache_size = 128 * 1024,    // 128KB per performance core
    .l2_cache_size = 4 * 1024 * 1024, // 4MB shared
    .l3_cache_size = 0,              // No L3 on M1
    .has_avx = 0,
    .has_avx2 = 0,
    .has_avx512 = 0,
    .has_neon = 1,
    .has_sve = 0,
    .has_amx = 1  // Apple Neural Engine
};

// Public interface
SimdOps* get_apple_silicon_ops(void) {
    return &apple_simd_ops;
}

PlatformInfo* get_apple_silicon_info(void) {
    apple_platform_info.num_cores = get_apple_cpu_count();
    return &apple_platform_info;
}

void init_apple_silicon(void) {
    printf("Initialized Apple Silicon optimizations\n");
    printf("Using Accelerate framework for BLAS operations\n");
}

void cleanup_apple_silicon(void) {
    // Nothing to cleanup for Apple Silicon
}