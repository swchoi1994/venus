#include "platform.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// Generic C implementations (no SIMD)

static void generic_vec_add_f32(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

static void generic_vec_mul_f32(const float* a, const float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

static void generic_vec_scale_f32(const float* a, float scale, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * scale;
    }
}

static float generic_vec_dot_f32(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static void generic_gemm_f32(const float* a, const float* b, float* c,
                            int m, int n, int k, float alpha, float beta) {
    // C = alpha * A * B + beta * C
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

static void generic_gemv_f32(const float* a, const float* x, float* y,
                            int m, int n, float alpha, float beta) {
    // y = alpha * A * x + beta * y
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            sum += a[i * n + j] * x[j];
        }
        y[i] = alpha * sum + beta * y[i];
    }
}

static void generic_gelu_f32(const float* x, float* y, size_t n) {
    const float sqrt_2_over_pi = 0.7978845608f;
    
    for (size_t i = 0; i < n; i++) {
        float xi = x[i];
        float x_cubed = xi * xi * xi;
        float arg = sqrt_2_over_pi * (xi + 0.044715f * x_cubed);
        y[i] = 0.5f * xi * (1.0f + tanhf(arg));
    }
}

static void generic_silu_f32(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

static void generic_relu_f32(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = fmaxf(0.0f, x[i]);
    }
}

static float generic_sum_f32(const float* x, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i];
    }
    return sum;
}

static float generic_max_f32(const float* x, size_t n) {
    if (n == 0) return 0.0f;
    
    float max_val = x[0];
    for (size_t i = 1; i < n; i++) {
        max_val = fmaxf(max_val, x[i]);
    }
    return max_val;
}

static void generic_memcpy_aligned(void* dst, const void* src, size_t n) {
    memcpy(dst, src, n);
}

static void generic_memset_f32(float* dst, float value, size_t n) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = value;
    }
}

// Platform information functions

static int get_generic_cpu_count(void) {
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
#else
    return (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
}

static size_t get_generic_memory(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullTotalPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (size_t)pages * (size_t)page_size;
#endif
}

static size_t get_generic_available_memory(void) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullAvailPhys;
#else
    // Fallback: return 25% of total memory as available
    return get_generic_memory() / 4;
#endif
}

// Initialize generic platform
static SimdOps generic_simd_ops = {
    .vec_add_f32 = generic_vec_add_f32,
    .vec_mul_f32 = generic_vec_mul_f32,
    .vec_scale_f32 = generic_vec_scale_f32,
    .vec_dot_f32 = generic_vec_dot_f32,
    .gemm_f32 = generic_gemm_f32,
    .gemv_f32 = generic_gemv_f32,
    .gelu_f32 = generic_gelu_f32,
    .silu_f32 = generic_silu_f32,
    .relu_f32 = generic_relu_f32,
    .sum_f32 = generic_sum_f32,
    .max_f32 = generic_max_f32,
    .memcpy_aligned = generic_memcpy_aligned,
    .memset_f32 = generic_memset_f32
};

static PlatformInfo generic_platform_info = {
    .platform = PLATFORM_GENERIC,
    .name = "Generic",
    .cache_line_size = 64,
    .l1_cache_size = 32 * 1024,
    .l2_cache_size = 256 * 1024,
    .l3_cache_size = 0,
    .has_avx = 0,
    .has_avx2 = 0,
    .has_avx512 = 0,
    .has_neon = 0,
    .has_sve = 0,
    .has_amx = 0
};

// Public interface
SimdOps* get_generic_ops(void) {
    return &generic_simd_ops;
}

PlatformInfo* get_generic_info(void) {
    generic_platform_info.num_cores = get_generic_cpu_count();
    return &generic_platform_info;
}

void init_generic(void) {
    printf("Initialized generic platform (no SIMD optimizations)\n");
}

void cleanup_generic(void) {
    // Nothing to cleanup for generic platform
}