#include "platform.h"
#include <stdio.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

// External platform implementations
extern SimdOps* get_apple_silicon_ops(void);
extern PlatformInfo* get_apple_silicon_info(void);
extern void init_apple_silicon(void);
extern void cleanup_apple_silicon(void);

extern SimdOps* get_generic_ops(void);
extern PlatformInfo* get_generic_info(void);
extern void init_generic(void);
extern void cleanup_generic(void);

// Global platform state
static Platform current_platform = PLATFORM_GENERIC;
static SimdOps* current_ops = NULL;
static PlatformInfo* current_info = NULL;

// Platform detection
Platform detect_platform(void) {
#if defined(__APPLE__) && defined(__arm64__)
    return PLATFORM_APPLE_SILICON_ENUM;
#elif defined(__x86_64__) || defined(_M_X64)
    return PLATFORM_X86_64;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return PLATFORM_ARM64;
#elif defined(__riscv) && __riscv_xlen == 64
    return PLATFORM_RISCV64;
#elif defined(__mips__) || defined(__MIPS__)
    #if defined(__LP64__) || defined(_LP64)
        return PLATFORM_MIPS64;
    #else
        return PLATFORM_MIPS;
    #endif
#elif defined(__powerpc__) || defined(__PPC__)
    #if defined(__powerpc64__) || defined(__PPC64__)
        return PLATFORM_POWERPC64;
    #else
        return PLATFORM_POWERPC;
    #endif
#elif defined(__sparc__) || defined(__sparc)
    #if defined(__arch64__) || defined(__sparcv9)
        return PLATFORM_SPARC64;
    #else
        return PLATFORM_SPARC;
    #endif
#elif defined(__loongarch64)
    return PLATFORM_LOONGARCH64;
#elif defined(__s390x__)
    return PLATFORM_S390X;
#elif defined(__EMSCRIPTEN__)
    #if defined(__wasm64__)
        return PLATFORM_WASM64;
    #else
        return PLATFORM_WASM32;
    #endif
#else
    return PLATFORM_GENERIC;
#endif
}

// Initialize platform
void init_platform(void) {
    current_platform = detect_platform();
    
    switch (current_platform) {
        case PLATFORM_APPLE_SILICON_ENUM:
#if defined(__APPLE__) && defined(__arm64__)
            current_ops = get_apple_silicon_ops();
            current_info = get_apple_silicon_info();
            init_apple_silicon();
#else
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            init_generic();
#endif
            break;
            
        case PLATFORM_X86_64:
            // TODO: Implement x86_64 specific optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            init_generic();
            break;
            
        case PLATFORM_ARM64:
            // TODO: Implement ARM64 specific optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            init_generic();
            break;
            
        case PLATFORM_RISCV64:
            // TODO: Implement RISC-V specific optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            init_generic();
            break;
            
        case PLATFORM_MIPS:
        case PLATFORM_MIPS64:
            // MIPS uses generic ops with potential MSA/MSA2 optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            if (current_info) current_info->name = "MIPS";
            init_generic();
            break;
            
        case PLATFORM_POWERPC:
        case PLATFORM_POWERPC64:
            // PowerPC uses generic ops with potential AltiVec/VSX optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            if (current_info) current_info->name = "PowerPC";
            init_generic();
            break;
            
        case PLATFORM_SPARC:
        case PLATFORM_SPARC64:
            // SPARC uses generic ops with potential VIS optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            if (current_info) current_info->name = "SPARC";
            init_generic();
            break;
            
        case PLATFORM_LOONGARCH64:
            // LoongArch uses generic ops with potential LSX/LASX optimizations
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            if (current_info) current_info->name = "LoongArch64";
            init_generic();
            break;
            
        case PLATFORM_S390X:
            // IBM z/Architecture with vector facility
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            if (current_info) current_info->name = "s390x";
            init_generic();
            break;
            
        case PLATFORM_WASM32:
        case PLATFORM_WASM64:
            // WebAssembly with SIMD128 support
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            if (current_info) current_info->name = "WebAssembly";
            init_generic();
            break;
            
        default:
            current_ops = get_generic_ops();
            current_info = get_generic_info();
            init_generic();
            break;
    }
}

// Cleanup platform
void cleanup_platform(void) {
    switch (current_platform) {
        case PLATFORM_APPLE_SILICON_ENUM:
#if defined(__APPLE__) && defined(__arm64__)
            cleanup_apple_silicon();
#else
            cleanup_generic();
#endif
            break;
            
        default:
            cleanup_generic();
            break;
    }
}

// Get platform operations
SimdOps* get_platform_ops(void) {
    if (!current_ops) {
        init_platform();
    }
    return current_ops;
}

// Get platform info
PlatformInfo* get_platform_info(void) {
    if (!current_info) {
        init_platform();
    }
    return current_info;
}

// Get platform name
const char* get_platform_name(void) {
    PlatformInfo* info = get_platform_info();
    return info ? info->name : "Unknown";
}

// Get SIMD features
const char* get_simd_features(void) {
    static char features[256];
    PlatformInfo* info = get_platform_info();
    
    if (!info) return "None";
    
    features[0] = '\0';
    
    if (info->has_avx) strcat(features, "AVX ");
    if (info->has_avx2) strcat(features, "AVX2 ");
    if (info->has_avx512) strcat(features, "AVX512 ");
    if (info->has_neon) strcat(features, "NEON ");
    if (info->has_sve) strcat(features, "SVE ");
    if (info->has_amx) strcat(features, "AMX ");
    
    if (features[0] == '\0') {
        strcpy(features, "None");
    }
    
    return features;
}

// Get CPU count
int get_cpu_count(void) {
    PlatformInfo* info = get_platform_info();
    return info ? info->num_cores : 1;
}

// Get total memory
size_t get_total_memory(void) {
#if defined(__APPLE__)
    int64_t memsize;
    size_t size = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &size, NULL, 0) == 0) {
        return (size_t)memsize;
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullTotalPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (size_t)pages * (size_t)page_size;
#endif
    return 0;
}

// Get available memory
size_t get_available_memory(void) {
#if defined(__APPLE__)
    vm_size_t page_size;
    vm_statistics64_data_t vm_stat;
    mach_msg_type_number_t host_size = sizeof(vm_stat) / sizeof(natural_t);
    
    if (host_page_size(mach_host_self(), &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) == KERN_SUCCESS) {
        return (size_t)(vm_stat.free_count * page_size);
    }
#elif defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (size_t)status.ullAvailPhys;
#else
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    return (size_t)pages * (size_t)page_size;
#endif
    return 0;
}

// Get optimal thread count
int get_optimal_thread_count(void) {
    int cpu_count = get_cpu_count();
    // Use 75% of available cores by default
    return (cpu_count * 3) / 4;
}

// Set thread affinity (platform-specific)
void set_thread_affinity(int thread_id, int core_id) {
    // TODO: Implement platform-specific thread affinity
    (void)thread_id;
    (void)core_id;
}