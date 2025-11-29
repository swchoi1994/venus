/**
 * Venus Inference Engine - Universal OpenCL Backend Implementation
 * 
 * Copyright (c) 2024 EdgeFlow AI
 * Licensed under Apache 2.0
 */

#include "opencl_backend.h"
#include "opencl_kernels.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// ============================================================================
// Error Handling
// ============================================================================

static char g_error_buffer[512] = {0};

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_error_buffer, sizeof(g_error_buffer), fmt, args);
    va_end(args);
}

const char* opencl_get_error(void) {
    return g_error_buffer;
}

static const char* cl_error_string(cl_int error) {
    switch (error) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
        case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
        default: return "Unknown OpenCL error";
    }
}

// ============================================================================
// Vendor Detection
// ============================================================================

static OpenCLVendor detect_vendor(const char* vendor_str, const char* device_name) {
    // Convert to lowercase for comparison
    char vendor_lower[256];
    char device_lower[256];
    
    size_t i;
    for (i = 0; i < sizeof(vendor_lower) - 1 && vendor_str[i]; i++) {
        vendor_lower[i] = tolower(vendor_str[i]);
    }
    vendor_lower[i] = '\0';
    
    for (i = 0; i < sizeof(device_lower) - 1 && device_name[i]; i++) {
        device_lower[i] = tolower(device_name[i]);
    }
    device_lower[i] = '\0';
    
    // Check vendor string
    if (strstr(vendor_lower, "qualcomm") || strstr(device_lower, "adreno")) {
        return OPENCL_VENDOR_QUALCOMM;
    }
    if (strstr(vendor_lower, "amd") || strstr(vendor_lower, "advanced micro")) {
        return OPENCL_VENDOR_AMD;
    }
    if (strstr(vendor_lower, "intel")) {
        return OPENCL_VENDOR_INTEL;
    }
    if (strstr(vendor_lower, "nvidia")) {
        return OPENCL_VENDOR_NVIDIA;
    }
    if (strstr(vendor_lower, "arm") || strstr(device_lower, "mali")) {
        return OPENCL_VENDOR_ARM;
    }
    if (strstr(vendor_lower, "apple")) {
        return OPENCL_VENDOR_APPLE;
    }
    if (strstr(device_lower, "cpu") || strstr(device_lower, "pthread") || 
        strstr(vendor_lower, "pocl") || strstr(vendor_lower, "portable")) {
        return OPENCL_VENDOR_CPU;
    }
    
    return OPENCL_VENDOR_UNKNOWN;
}

static AdrenoGeneration detect_adreno_gen(const char* device_name) {
    // Parse Adreno model number from device name
    // Examples: "Adreno (TM) 730", "Adreno (TM) 650", "Adreno (TM) 830"
    
    const char* adreno = strstr(device_name, "Adreno");
    if (!adreno) return ADRENO_UNKNOWN;
    
    // Find the model number
    const char* p = adreno;
    while (*p && !isdigit(*p)) p++;
    
    if (!*p) return ADRENO_UNKNOWN;
    
    int model = atoi(p);
    
    if (model >= 800) return ADRENO_8XX;
    if (model >= 700) return ADRENO_7XX;
    if (model >= 600) return ADRENO_6XX;
    
    return ADRENO_UNKNOWN;
}

// ============================================================================
// Device Query
// ============================================================================

static int query_device_info(cl_device_id device, OpenCLDeviceInfo* info) {
    cl_int err;
    
    memset(info, 0, sizeof(*info));
    
    // Basic info
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(info->name), info->name, NULL);
    if (err != CL_SUCCESS) return -1;
    
    err = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(info->vendor_string), info->vendor_string, NULL);
    if (err != CL_SUCCESS) return -1;
    
    err = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(info->driver_version), info->driver_version, NULL);
    if (err != CL_SUCCESS) return -1;
    
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(info->opencl_version), info->opencl_version, NULL);
    if (err != CL_SUCCESS) return -1;
    
    // Vendor detection
    info->vendor = detect_vendor(info->vendor_string, info->name);
    if (info->vendor == OPENCL_VENDOR_QUALCOMM) {
        info->adreno_gen = detect_adreno_gen(info->name);
    }
    
    // Device type
    cl_device_type type;
    clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    if (type & CL_DEVICE_TYPE_GPU) {
        info->device_type = OPENCL_DEVICE_GPU;
    } else if (type & CL_DEVICE_TYPE_CPU) {
        info->device_type = OPENCL_DEVICE_CPU;
    } else {
        info->device_type = OPENCL_DEVICE_ACCELERATOR;
    }
    
    // Compute capabilities
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info->compute_units), &info->compute_units, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info->max_work_group_size), &info->max_work_group_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(info->max_work_item_dims), &info->max_work_item_dims, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(info->max_work_item_sizes), info->max_work_item_sizes, NULL);
    
    // Memory
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info->global_mem_size), &info->global_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(info->local_mem_size), &info->local_mem_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(info->max_alloc_size), &info->max_alloc_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(info->mem_base_addr_align), &info->mem_base_addr_align, NULL);
    
    // Features
    char extensions[4096];
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL);
    info->has_fp16 = strstr(extensions, "cl_khr_fp16") != NULL;
    info->has_fp64 = strstr(extensions, "cl_khr_fp64") != NULL;
    info->has_int8 = strstr(extensions, "cl_khr_int8") != NULL || 
                     strstr(extensions, "cl_intel_subgroups_char") != NULL;
    info->has_subgroups = strstr(extensions, "cl_khr_subgroups") != NULL ||
                          strstr(extensions, "cl_intel_subgroups") != NULL;
    
    clGetDeviceInfo(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(info->preferred_vector_width_float), 
                    &info->preferred_vector_width_float, NULL);
    
    // Cache
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(info->global_cache_size), &info->global_cache_size, NULL);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(info->global_cache_line_size), &info->global_cache_line_size, NULL);
    
    return 0;
}

// ============================================================================
// Tuning Parameters
// ============================================================================

OpenCLTuningParams opencl_get_tuning_params(OpenCLVendor vendor, const OpenCLDeviceInfo* info) {
    OpenCLTuningParams params = {0};
    
    switch (vendor) {
        case OPENCL_VENDOR_QUALCOMM:
            // Adreno GPUs: tile-based architecture, prefer smaller tiles
            params.tile_m = 8;
            params.tile_n = 8;
            params.tile_k = 8;
            params.wg_size_x = 8;
            params.wg_size_y = 8;
            params.wg_size_z = 1;
            params.vector_width = 4;
            params.use_local_memory = true;
            params.local_mem_padding = 1;  // Avoid bank conflicts
            params.attention_block_size = 64;
            params.reduction_block_size = 256;
            params.dequant_block_size = 32;
            
            // Adjust for generation
            if (info && info->adreno_gen == ADRENO_8XX) {
                // Snapdragon 8 Elite has more compute units
                params.tile_m = 16;
                params.tile_n = 16;
                params.wg_size_x = 16;
                params.wg_size_y = 16;
            }
            break;
            
        case OPENCL_VENDOR_AMD:
            // AMD GPUs: wavefront 64, larger tiles
            params.tile_m = 16;
            params.tile_n = 16;
            params.tile_k = 16;
            params.wg_size_x = 16;
            params.wg_size_y = 16;
            params.wg_size_z = 1;
            params.vector_width = 4;
            params.use_local_memory = true;
            params.local_mem_padding = 1;
            params.attention_block_size = 128;
            params.reduction_block_size = 256;
            params.dequant_block_size = 32;
            break;
            
        case OPENCL_VENDOR_INTEL:
            // Intel GPUs: EU-optimized
            params.tile_m = 16;
            params.tile_n = 16;
            params.tile_k = 16;
            params.wg_size_x = 16;
            params.wg_size_y = 16;
            params.wg_size_z = 1;
            params.vector_width = 8;  // Intel likes wider vectors
            params.use_local_memory = true;
            params.local_mem_padding = 0;
            params.attention_block_size = 128;
            params.reduction_block_size = 256;
            params.dequant_block_size = 32;
            break;
            
        case OPENCL_VENDOR_NVIDIA:
            // NVIDIA: warp 32, good local memory
            params.tile_m = 16;
            params.tile_n = 16;
            params.tile_k = 16;
            params.wg_size_x = 16;
            params.wg_size_y = 16;
            params.wg_size_z = 1;
            params.vector_width = 4;
            params.use_local_memory = true;
            params.local_mem_padding = 1;
            params.attention_block_size = 128;
            params.reduction_block_size = 256;
            params.dequant_block_size = 32;
            break;
            
        case OPENCL_VENDOR_ARM:
            // Mali: very limited local memory
            params.tile_m = 4;
            params.tile_n = 4;
            params.tile_k = 4;
            params.wg_size_x = 4;
            params.wg_size_y = 4;
            params.wg_size_z = 1;
            params.vector_width = 4;
            params.use_local_memory = false;  // Mali has limited local mem
            params.local_mem_padding = 0;
            params.attention_block_size = 32;
            params.reduction_block_size = 128;
            params.dequant_block_size = 32;
            break;
            
        case OPENCL_VENDOR_CPU:
            // CPU: maximize cache usage with large tiles
            params.tile_m = 64;
            params.tile_n = 64;
            params.tile_k = 64;
            params.wg_size_x = 1;  // Single work-item per group for CPU
            params.wg_size_y = 1;
            params.wg_size_z = 1;
            params.vector_width = info ? info->preferred_vector_width_float : 8;
            params.use_local_memory = false;
            params.local_mem_padding = 0;
            params.attention_block_size = 256;
            params.reduction_block_size = 512;
            params.dequant_block_size = 32;
            break;
            
        default:
            // Conservative defaults
            params.tile_m = 8;
            params.tile_n = 8;
            params.tile_k = 8;
            params.wg_size_x = 8;
            params.wg_size_y = 8;
            params.wg_size_z = 1;
            params.vector_width = 4;
            params.use_local_memory = true;
            params.local_mem_padding = 0;
            params.attention_block_size = 64;
            params.reduction_block_size = 256;
            params.dequant_block_size = 32;
            break;
    }
    
    return params;
}

// ============================================================================
// Kernel Compilation
// ============================================================================

static int compile_kernels(OpenCLBackend* backend) {
    if (backend->kernels_compiled) return 0;
    
    cl_int err;
    
    // Build options based on tuning params
    char build_opts[512];
    snprintf(build_opts, sizeof(build_opts),
             "-DTILE_M=%u -DTILE_N=%u -DTILE_K=%u "
             "-DWG_SIZE_X=%u -DWG_SIZE_Y=%u "
             "-DVEC_WIDTH=%u "
             "-DUSE_LOCAL_MEM=%d "
             "-DLOCAL_MEM_PAD=%u "
             "-DATTENTION_BLOCK=%u "
             "-DREDUCTION_BLOCK=%u "
             "-DDEQUANT_BLOCK=%u",
             backend->tuning.tile_m, backend->tuning.tile_n, backend->tuning.tile_k,
             backend->tuning.wg_size_x, backend->tuning.wg_size_y,
             backend->tuning.vector_width,
             backend->tuning.use_local_memory ? 1 : 0,
             backend->tuning.local_mem_padding,
             backend->tuning.attention_block_size,
             backend->tuning.reduction_block_size,
             backend->tuning.dequant_block_size);
    
    // Add FP16 support if available
    if (backend->info.has_fp16) {
        strcat(build_opts, " -DHAS_FP16=1");
    }
    
    // Create program from embedded kernel source
    const char* kernel_source = OPENCL_KERNELS_SOURCE;
    size_t source_len = strlen(kernel_source);
    
    backend->program = clCreateProgramWithSource(backend->context, 1, &kernel_source, &source_len, &err);
    if (err != CL_SUCCESS) {
        set_error("Failed to create program: %s", cl_error_string(err));
        return -1;
    }
    
    // Build program
    err = clBuildProgram(backend->program, 1, &backend->device, build_opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Get build log
        size_t log_size;
        clGetProgramBuildInfo(backend->program, backend->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* log = (char*)malloc(log_size);
        clGetProgramBuildInfo(backend->program, backend->device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        set_error("Kernel build failed: %s\n%s", cl_error_string(err), log);
        free(log);
        return -1;
    }
    
    // Create kernels
    #define CREATE_KERNEL(name, var) \
        backend->var = clCreateKernel(backend->program, name, &err); \
        if (err != CL_SUCCESS) { \
            set_error("Failed to create kernel %s: %s", name, cl_error_string(err)); \
            return -1; \
        }
    
    CREATE_KERNEL("gemm_f32", kernel_gemm);
    CREATE_KERNEL("gemm_q8", kernel_gemm_q8);
    CREATE_KERNEL("gemm_q4", kernel_gemm_q4);
    CREATE_KERNEL("attention_scores", kernel_attention_scores);
    CREATE_KERNEL("attention_softmax", kernel_attention_softmax);
    CREATE_KERNEL("attention_output", kernel_attention_output);
    CREATE_KERNEL("rmsnorm", kernel_rmsnorm);
    CREATE_KERNEL("layernorm", kernel_layernorm);
    CREATE_KERNEL("silu", kernel_silu);
    CREATE_KERNEL("gelu", kernel_gelu);
    CREATE_KERNEL("rope", kernel_rope);
    CREATE_KERNEL("rope_2d", kernel_rope_2d);
    CREATE_KERNEL("vec_add", kernel_add);
    CREATE_KERNEL("vec_mul", kernel_mul);
    CREATE_KERNEL("vec_scale", kernel_scale);
    CREATE_KERNEL("dequant_q8", kernel_dequant_q8);
    CREATE_KERNEL("dequant_q4", kernel_dequant_q4);
    
    #undef CREATE_KERNEL
    
    backend->kernels_compiled = true;
    return 0;
}

// ============================================================================
// Initialization
// ============================================================================

static OpenCLBackend* init_with_device(cl_platform_id platform, cl_device_id device) {
    cl_int err;
    
    OpenCLBackend* backend = (OpenCLBackend*)calloc(1, sizeof(OpenCLBackend));
    if (!backend) {
        set_error("Failed to allocate backend");
        return NULL;
    }
    
    backend->platform = platform;
    backend->device = device;
    
    // Query device info
    if (query_device_info(device, &backend->info) != 0) {
        set_error("Failed to query device info");
        free(backend);
        return NULL;
    }
    
    // Get tuning parameters
    backend->tuning = opencl_get_tuning_params(backend->info.vendor, &backend->info);
    
    // Create context
    backend->context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        set_error("Failed to create context: %s", cl_error_string(err));
        free(backend);
        return NULL;
    }
    
    // Create command queue
#ifdef CL_VERSION_2_0
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    backend->queue = clCreateCommandQueueWithProperties(backend->context, device, props, &err);
#else
    backend->queue = clCreateCommandQueue(backend->context, device, CL_QUEUE_PROFILING_ENABLE, &err);
#endif
    if (err != CL_SUCCESS) {
        set_error("Failed to create command queue: %s", cl_error_string(err));
        clReleaseContext(backend->context);
        free(backend);
        return NULL;
    }
    
    // Compile kernels
    if (compile_kernels(backend) != 0) {
        clReleaseCommandQueue(backend->queue);
        clReleaseContext(backend->context);
        free(backend);
        return NULL;
    }
    
    backend->initialized = true;
    return backend;
}

OpenCLBackend* opencl_init(void) {
    return opencl_init_with_vendor(OPENCL_VENDOR_UNKNOWN);  // Auto-select
}

OpenCLBackend* opencl_init_with_vendor(OpenCLVendor preferred_vendor) {
    cl_int err;
    cl_uint num_platforms;
    
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        set_error("No OpenCL platforms found");
        return NULL;
    }
    
    cl_platform_id* platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    clGetPlatformIDs(num_platforms, platforms, NULL);
    
    cl_device_id best_device = NULL;
    cl_platform_id best_platform = NULL;
    OpenCLVendor best_vendor = OPENCL_VENDOR_UNKNOWN;
    int best_score = -1;
    
    for (cl_uint i = 0; i < num_platforms; i++) {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0) continue;
        
        cl_device_id* devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        
        for (cl_uint j = 0; j < num_devices; j++) {
            OpenCLDeviceInfo info;
            if (query_device_info(devices[j], &info) != 0) continue;
            
            int score = 0;
            
            // Prefer GPU over CPU
            if (info.device_type == OPENCL_DEVICE_GPU) score += 100;
            
            // Prefer matching vendor
            if (preferred_vendor != OPENCL_VENDOR_UNKNOWN && info.vendor == preferred_vendor) {
                score += 50;
            }
            
            // Prefer more compute units
            score += info.compute_units;
            
            // Prefer more memory
            score += (int)(info.global_mem_size / (1024 * 1024 * 1024));  // GB
            
            if (score > best_score) {
                best_score = score;
                best_device = devices[j];
                best_platform = platforms[i];
                best_vendor = info.vendor;
            }
        }
        
        free(devices);
    }
    
    free(platforms);
    
    if (!best_device) {
        set_error("No suitable OpenCL device found");
        return NULL;
    }
    
    return init_with_device(best_platform, best_device);
}

OpenCLBackend* opencl_init_from_env(void) {
    const char* device_env = getenv("VENUS_OPENCL_DEVICE");
    
    if (!device_env || strcmp(device_env, "auto") == 0) {
        return opencl_init();
    }
    
    OpenCLVendor vendor = OPENCL_VENDOR_UNKNOWN;
    
    if (strcmp(device_env, "adreno") == 0 || strcmp(device_env, "qualcomm") == 0) {
        vendor = OPENCL_VENDOR_QUALCOMM;
    } else if (strcmp(device_env, "amd") == 0) {
        vendor = OPENCL_VENDOR_AMD;
    } else if (strcmp(device_env, "intel") == 0) {
        vendor = OPENCL_VENDOR_INTEL;
    } else if (strcmp(device_env, "nvidia") == 0) {
        vendor = OPENCL_VENDOR_NVIDIA;
    } else if (strcmp(device_env, "mali") == 0 || strcmp(device_env, "arm") == 0) {
        vendor = OPENCL_VENDOR_ARM;
    } else if (strcmp(device_env, "cpu") == 0) {
        vendor = OPENCL_VENDOR_CPU;
    }
    
    return opencl_init_with_vendor(vendor);
}

void opencl_cleanup(OpenCLBackend* backend) {
    if (!backend) return;
    
    // Release kernels
    if (backend->kernel_gemm) clReleaseKernel(backend->kernel_gemm);
    if (backend->kernel_gemm_q8) clReleaseKernel(backend->kernel_gemm_q8);
    if (backend->kernel_gemm_q4) clReleaseKernel(backend->kernel_gemm_q4);
    if (backend->kernel_attention_scores) clReleaseKernel(backend->kernel_attention_scores);
    if (backend->kernel_attention_softmax) clReleaseKernel(backend->kernel_attention_softmax);
    if (backend->kernel_attention_output) clReleaseKernel(backend->kernel_attention_output);
    if (backend->kernel_rmsnorm) clReleaseKernel(backend->kernel_rmsnorm);
    if (backend->kernel_layernorm) clReleaseKernel(backend->kernel_layernorm);
    if (backend->kernel_silu) clReleaseKernel(backend->kernel_silu);
    if (backend->kernel_gelu) clReleaseKernel(backend->kernel_gelu);
    if (backend->kernel_rope) clReleaseKernel(backend->kernel_rope);
    if (backend->kernel_rope_2d) clReleaseKernel(backend->kernel_rope_2d);
    if (backend->kernel_add) clReleaseKernel(backend->kernel_add);
    if (backend->kernel_mul) clReleaseKernel(backend->kernel_mul);
    if (backend->kernel_scale) clReleaseKernel(backend->kernel_scale);
    if (backend->kernel_dequant_q8) clReleaseKernel(backend->kernel_dequant_q8);
    if (backend->kernel_dequant_q4) clReleaseKernel(backend->kernel_dequant_q4);
    
    // Release program
    if (backend->program) clReleaseProgram(backend->program);
    
    // Release queue and context
    if (backend->queue) clReleaseCommandQueue(backend->queue);
    if (backend->context) clReleaseContext(backend->context);
    
    free(backend);
}

// ============================================================================
// Memory Management
// ============================================================================

cl_mem opencl_alloc(OpenCLBackend* backend, size_t size) {
    if (!backend || !backend->initialized) return NULL;
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(backend->context, CL_MEM_READ_WRITE, size, NULL, &err);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to allocate %zu bytes: %s", size, cl_error_string(err));
        return NULL;
    }
    
    backend->allocated_bytes += size;
    backend->allocation_count++;
    if (backend->allocated_bytes > backend->peak_allocated_bytes) {
        backend->peak_allocated_bytes = backend->allocated_bytes;
    }
    
    return buffer;
}

cl_mem opencl_alloc_copy(OpenCLBackend* backend, const void* data, size_t size) {
    if (!backend || !backend->initialized) return NULL;
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(backend->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
                                   size, (void*)data, &err);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to allocate and copy %zu bytes: %s", size, cl_error_string(err));
        return NULL;
    }
    
    backend->allocated_bytes += size;
    backend->allocation_count++;
    if (backend->allocated_bytes > backend->peak_allocated_bytes) {
        backend->peak_allocated_bytes = backend->allocated_bytes;
    }
    
    return buffer;
}

void opencl_free(OpenCLBackend* backend, cl_mem buffer) {
    if (!backend || !buffer) return;
    
    size_t size;
    clGetMemObjectInfo(buffer, CL_MEM_SIZE, sizeof(size), &size, NULL);
    
    clReleaseMemObject(buffer);
    
    backend->allocated_bytes -= size;
}

int opencl_copy_to_device(OpenCLBackend* backend, cl_mem dst, const void* src, size_t size) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err = clEnqueueWriteBuffer(backend->queue, dst, CL_TRUE, 0, size, src, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to copy to device: %s", cl_error_string(err));
        return -1;
    }
    return 0;
}

int opencl_copy_to_host(OpenCLBackend* backend, void* dst, cl_mem src, size_t size) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err = clEnqueueReadBuffer(backend->queue, src, CL_TRUE, 0, size, dst, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to copy to host: %s", cl_error_string(err));
        return -1;
    }
    return 0;
}

void opencl_get_memory_stats(OpenCLBackend* backend, 
                             size_t* current_bytes,
                             size_t* peak_bytes,
                             uint32_t* allocation_count) {
    if (!backend) return;
    
    if (current_bytes) *current_bytes = backend->allocated_bytes;
    if (peak_bytes) *peak_bytes = backend->peak_allocated_bytes;
    if (allocation_count) *allocation_count = backend->allocation_count;
}

// ============================================================================
// Matrix Operations
// ============================================================================

int opencl_gemm_f32(OpenCLBackend* backend,
                    cl_mem a, cl_mem b, cl_mem c,
                    int m, int n, int k,
                    float alpha, float beta) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_gemm, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(backend->kernel_gemm, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(backend->kernel_gemm, 2, sizeof(cl_mem), &c);
    err |= clSetKernelArg(backend->kernel_gemm, 3, sizeof(int), &m);
    err |= clSetKernelArg(backend->kernel_gemm, 4, sizeof(int), &n);
    err |= clSetKernelArg(backend->kernel_gemm, 5, sizeof(int), &k);
    err |= clSetKernelArg(backend->kernel_gemm, 6, sizeof(float), &alpha);
    err |= clSetKernelArg(backend->kernel_gemm, 7, sizeof(float), &beta);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set GEMM kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[2] = {
        ((m + backend->tuning.tile_m - 1) / backend->tuning.tile_m) * backend->tuning.wg_size_x,
        ((n + backend->tuning.tile_n - 1) / backend->tuning.tile_n) * backend->tuning.wg_size_y
    };
    size_t local_work_size[2] = {backend->tuning.wg_size_x, backend->tuning.wg_size_y};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_gemm, 2, NULL, 
                                  global_work_size, local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue GEMM kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_gemm_q8(OpenCLBackend* backend,
                   cl_mem a_f32, cl_mem b_q8, cl_mem scales, cl_mem c,
                   int m, int n, int k) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_gemm_q8, 0, sizeof(cl_mem), &a_f32);
    err |= clSetKernelArg(backend->kernel_gemm_q8, 1, sizeof(cl_mem), &b_q8);
    err |= clSetKernelArg(backend->kernel_gemm_q8, 2, sizeof(cl_mem), &scales);
    err |= clSetKernelArg(backend->kernel_gemm_q8, 3, sizeof(cl_mem), &c);
    err |= clSetKernelArg(backend->kernel_gemm_q8, 4, sizeof(int), &m);
    err |= clSetKernelArg(backend->kernel_gemm_q8, 5, sizeof(int), &n);
    err |= clSetKernelArg(backend->kernel_gemm_q8, 6, sizeof(int), &k);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set GEMM Q8 kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[2] = {
        ((m + backend->tuning.tile_m - 1) / backend->tuning.tile_m) * backend->tuning.wg_size_x,
        ((n + backend->tuning.tile_n - 1) / backend->tuning.tile_n) * backend->tuning.wg_size_y
    };
    size_t local_work_size[2] = {backend->tuning.wg_size_x, backend->tuning.wg_size_y};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_gemm_q8, 2, NULL,
                                  global_work_size, local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue GEMM Q8 kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_gemm_q4(OpenCLBackend* backend,
                   cl_mem a_f32, cl_mem b_q4, cl_mem scales, cl_mem c,
                   int m, int n, int k) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_gemm_q4, 0, sizeof(cl_mem), &a_f32);
    err |= clSetKernelArg(backend->kernel_gemm_q4, 1, sizeof(cl_mem), &b_q4);
    err |= clSetKernelArg(backend->kernel_gemm_q4, 2, sizeof(cl_mem), &scales);
    err |= clSetKernelArg(backend->kernel_gemm_q4, 3, sizeof(cl_mem), &c);
    err |= clSetKernelArg(backend->kernel_gemm_q4, 4, sizeof(int), &m);
    err |= clSetKernelArg(backend->kernel_gemm_q4, 5, sizeof(int), &n);
    err |= clSetKernelArg(backend->kernel_gemm_q4, 6, sizeof(int), &k);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set GEMM Q4 kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[2] = {
        ((m + backend->tuning.tile_m - 1) / backend->tuning.tile_m) * backend->tuning.wg_size_x,
        ((n + backend->tuning.tile_n - 1) / backend->tuning.tile_n) * backend->tuning.wg_size_y
    };
    size_t local_work_size[2] = {backend->tuning.wg_size_x, backend->tuning.wg_size_y};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_gemm_q4, 2, NULL,
                                  global_work_size, local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue GEMM Q4 kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

// ============================================================================
// Attention Operations
// ============================================================================

int opencl_attention_scores(OpenCLBackend* backend,
                            cl_mem q, cl_mem k, cl_mem scores,
                            int batch, int heads, int seq_len, int kv_len, int head_dim,
                            float scale) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_attention_scores, 0, sizeof(cl_mem), &q);
    err |= clSetKernelArg(backend->kernel_attention_scores, 1, sizeof(cl_mem), &k);
    err |= clSetKernelArg(backend->kernel_attention_scores, 2, sizeof(cl_mem), &scores);
    err |= clSetKernelArg(backend->kernel_attention_scores, 3, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_attention_scores, 4, sizeof(int), &heads);
    err |= clSetKernelArg(backend->kernel_attention_scores, 5, sizeof(int), &seq_len);
    err |= clSetKernelArg(backend->kernel_attention_scores, 6, sizeof(int), &kv_len);
    err |= clSetKernelArg(backend->kernel_attention_scores, 7, sizeof(int), &head_dim);
    err |= clSetKernelArg(backend->kernel_attention_scores, 8, sizeof(float), &scale);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set attention scores kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[3] = {(size_t)seq_len, (size_t)kv_len, (size_t)(batch * heads)};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_attention_scores, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue attention scores kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_attention_softmax(OpenCLBackend* backend,
                             cl_mem scores,
                             int batch, int heads, int seq_len, int kv_len,
                             bool causal) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    int causal_int = causal ? 1 : 0;
    
    err = clSetKernelArg(backend->kernel_attention_softmax, 0, sizeof(cl_mem), &scores);
    err |= clSetKernelArg(backend->kernel_attention_softmax, 1, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_attention_softmax, 2, sizeof(int), &heads);
    err |= clSetKernelArg(backend->kernel_attention_softmax, 3, sizeof(int), &seq_len);
    err |= clSetKernelArg(backend->kernel_attention_softmax, 4, sizeof(int), &kv_len);
    err |= clSetKernelArg(backend->kernel_attention_softmax, 5, sizeof(int), &causal_int);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set attention softmax kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[2] = {(size_t)seq_len, (size_t)(batch * heads)};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_attention_softmax, 2, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue attention softmax kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_attention_output(OpenCLBackend* backend,
                            cl_mem scores, cl_mem v, cl_mem output,
                            int batch, int heads, int seq_len, int kv_len, int head_dim) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_attention_output, 0, sizeof(cl_mem), &scores);
    err |= clSetKernelArg(backend->kernel_attention_output, 1, sizeof(cl_mem), &v);
    err |= clSetKernelArg(backend->kernel_attention_output, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_attention_output, 3, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_attention_output, 4, sizeof(int), &heads);
    err |= clSetKernelArg(backend->kernel_attention_output, 5, sizeof(int), &seq_len);
    err |= clSetKernelArg(backend->kernel_attention_output, 6, sizeof(int), &kv_len);
    err |= clSetKernelArg(backend->kernel_attention_output, 7, sizeof(int), &head_dim);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set attention output kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[3] = {(size_t)seq_len, (size_t)head_dim, (size_t)(batch * heads)};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_attention_output, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue attention output kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

// ============================================================================
// Normalization
// ============================================================================

int opencl_rmsnorm(OpenCLBackend* backend,
                   cl_mem input, cl_mem weight, cl_mem output,
                   int batch, int hidden_size, float eps) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_rmsnorm, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(backend->kernel_rmsnorm, 1, sizeof(cl_mem), &weight);
    err |= clSetKernelArg(backend->kernel_rmsnorm, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_rmsnorm, 3, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_rmsnorm, 4, sizeof(int), &hidden_size);
    err |= clSetKernelArg(backend->kernel_rmsnorm, 5, sizeof(float), &eps);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set RMSNorm kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = (size_t)batch * backend->tuning.reduction_block_size;
    size_t local_work_size = backend->tuning.reduction_block_size;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_rmsnorm, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue RMSNorm kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_layernorm(OpenCLBackend* backend,
                     cl_mem input, cl_mem weight, cl_mem bias, cl_mem output,
                     int batch, int hidden_size, float eps) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_layernorm, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(backend->kernel_layernorm, 1, sizeof(cl_mem), &weight);
    err |= clSetKernelArg(backend->kernel_layernorm, 2, sizeof(cl_mem), &bias);
    err |= clSetKernelArg(backend->kernel_layernorm, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_layernorm, 4, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_layernorm, 5, sizeof(int), &hidden_size);
    err |= clSetKernelArg(backend->kernel_layernorm, 6, sizeof(float), &eps);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set LayerNorm kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = (size_t)batch * backend->tuning.reduction_block_size;
    size_t local_work_size = backend->tuning.reduction_block_size;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_layernorm, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue LayerNorm kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

// ============================================================================
// Activation Functions
// ============================================================================

int opencl_silu(OpenCLBackend* backend, cl_mem input, cl_mem output, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_silu, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(backend->kernel_silu, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_silu, 2, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set SiLU kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_silu, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue SiLU kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_gelu(OpenCLBackend* backend, cl_mem input, cl_mem output, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_gelu, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(backend->kernel_gelu, 1, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_gelu, 2, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set GELU kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_gelu, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue GELU kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_relu(OpenCLBackend* backend, cl_mem input, cl_mem output, int n) {
    // ReLU can be implemented using the scale kernel with max(0, x)
    // For now, we'll use a simple element-wise approach
    if (!backend || !backend->initialized) return -1;
    
    // TODO: Add dedicated ReLU kernel
    return opencl_silu(backend, input, output, n);  // Placeholder
}

// ============================================================================
// Position Embeddings
// ============================================================================

int opencl_rope(OpenCLBackend* backend,
                cl_mem x, cl_mem cos, cl_mem sin, cl_mem output,
                int batch, int seq_len, int heads, int head_dim) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_rope, 0, sizeof(cl_mem), &x);
    err |= clSetKernelArg(backend->kernel_rope, 1, sizeof(cl_mem), &cos);
    err |= clSetKernelArg(backend->kernel_rope, 2, sizeof(cl_mem), &sin);
    err |= clSetKernelArg(backend->kernel_rope, 3, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_rope, 4, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_rope, 5, sizeof(int), &seq_len);
    err |= clSetKernelArg(backend->kernel_rope, 6, sizeof(int), &heads);
    err |= clSetKernelArg(backend->kernel_rope, 7, sizeof(int), &head_dim);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set RoPE kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[3] = {(size_t)(head_dim / 2), (size_t)seq_len, (size_t)(batch * heads)};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_rope, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue RoPE kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_rope_2d(OpenCLBackend* backend,
                   cl_mem x, 
                   cl_mem cos_h, cl_mem sin_h,
                   cl_mem cos_w, cl_mem sin_w,
                   cl_mem output,
                   int batch, int height, int width, int heads, int head_dim) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_rope_2d, 0, sizeof(cl_mem), &x);
    err |= clSetKernelArg(backend->kernel_rope_2d, 1, sizeof(cl_mem), &cos_h);
    err |= clSetKernelArg(backend->kernel_rope_2d, 2, sizeof(cl_mem), &sin_h);
    err |= clSetKernelArg(backend->kernel_rope_2d, 3, sizeof(cl_mem), &cos_w);
    err |= clSetKernelArg(backend->kernel_rope_2d, 4, sizeof(cl_mem), &sin_w);
    err |= clSetKernelArg(backend->kernel_rope_2d, 5, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_rope_2d, 6, sizeof(int), &batch);
    err |= clSetKernelArg(backend->kernel_rope_2d, 7, sizeof(int), &height);
    err |= clSetKernelArg(backend->kernel_rope_2d, 8, sizeof(int), &width);
    err |= clSetKernelArg(backend->kernel_rope_2d, 9, sizeof(int), &heads);
    err |= clSetKernelArg(backend->kernel_rope_2d, 10, sizeof(int), &head_dim);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set 2D RoPE kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size[3] = {(size_t)width, (size_t)height, (size_t)(batch * heads)};
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_rope_2d, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue 2D RoPE kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

// ============================================================================
// Element-wise Operations
// ============================================================================

int opencl_add(OpenCLBackend* backend, cl_mem a, cl_mem b, cl_mem c, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_add, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(backend->kernel_add, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(backend->kernel_add, 2, sizeof(cl_mem), &c);
    err |= clSetKernelArg(backend->kernel_add, 3, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set add kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_add, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue add kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_mul(OpenCLBackend* backend, cl_mem a, cl_mem b, cl_mem c, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_mul, 0, sizeof(cl_mem), &a);
    err |= clSetKernelArg(backend->kernel_mul, 1, sizeof(cl_mem), &b);
    err |= clSetKernelArg(backend->kernel_mul, 2, sizeof(cl_mem), &c);
    err |= clSetKernelArg(backend->kernel_mul, 3, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set mul kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_mul, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue mul kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_scale(OpenCLBackend* backend, cl_mem x, cl_mem y, float scale, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_scale, 0, sizeof(cl_mem), &x);
    err |= clSetKernelArg(backend->kernel_scale, 1, sizeof(cl_mem), &y);
    err |= clSetKernelArg(backend->kernel_scale, 2, sizeof(float), &scale);
    err |= clSetKernelArg(backend->kernel_scale, 3, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set scale kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_scale, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue scale kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_add_scale(OpenCLBackend* backend, cl_mem a, cl_mem b, cl_mem y, float scale, int n) {
    // First add, then scale
    int err = opencl_add(backend, a, b, y, n);
    if (err != 0) return err;
    return opencl_scale(backend, y, y, scale, n);
}

// ============================================================================
// Quantization/Dequantization
// ============================================================================

int opencl_dequant_q8(OpenCLBackend* backend,
                      cl_mem input, cl_mem scales, cl_mem output, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_dequant_q8, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(backend->kernel_dequant_q8, 1, sizeof(cl_mem), &scales);
    err |= clSetKernelArg(backend->kernel_dequant_q8, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_dequant_q8, 3, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set dequant Q8 kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_dequant_q8, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue dequant Q8 kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

int opencl_dequant_q4(OpenCLBackend* backend,
                      cl_mem input, cl_mem scales, cl_mem output, int n) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err;
    
    err = clSetKernelArg(backend->kernel_dequant_q4, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(backend->kernel_dequant_q4, 1, sizeof(cl_mem), &scales);
    err |= clSetKernelArg(backend->kernel_dequant_q4, 2, sizeof(cl_mem), &output);
    err |= clSetKernelArg(backend->kernel_dequant_q4, 3, sizeof(int), &n);
    
    if (err != CL_SUCCESS) {
        set_error("Failed to set dequant Q4 kernel args: %s", cl_error_string(err));
        return -1;
    }
    
    size_t global_work_size = ((n + 255) / 256) * 256;
    size_t local_work_size = 256;
    
    err = clEnqueueNDRangeKernel(backend->queue, backend->kernel_dequant_q4, 1, NULL,
                                  &global_work_size, &local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        set_error("Failed to enqueue dequant Q4 kernel: %s", cl_error_string(err));
        return -1;
    }
    
    return 0;
}

// ============================================================================
// Synchronization
// ============================================================================

int opencl_synchronize(OpenCLBackend* backend) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err = clFinish(backend->queue);
    if (err != CL_SUCCESS) {
        set_error("Failed to synchronize: %s", cl_error_string(err));
        return -1;
    }
    return 0;
}

int opencl_flush(OpenCLBackend* backend) {
    if (!backend || !backend->initialized) return -1;
    
    cl_int err = clFlush(backend->queue);
    if (err != CL_SUCCESS) {
        set_error("Failed to flush: %s", cl_error_string(err));
        return -1;
    }
    return 0;
}

// ============================================================================
// Utility Functions
// ============================================================================

const char* opencl_vendor_name(OpenCLVendor vendor) {
    switch (vendor) {
        case OPENCL_VENDOR_QUALCOMM: return "Qualcomm (Adreno)";
        case OPENCL_VENDOR_AMD: return "AMD";
        case OPENCL_VENDOR_INTEL: return "Intel";
        case OPENCL_VENDOR_NVIDIA: return "NVIDIA";
        case OPENCL_VENDOR_ARM: return "ARM (Mali)";
        case OPENCL_VENDOR_APPLE: return "Apple";
        case OPENCL_VENDOR_CPU: return "CPU";
        default: return "Unknown";
    }
}

const char* opencl_adreno_gen_name(AdrenoGeneration gen) {
    switch (gen) {
        case ADRENO_6XX: return "Adreno 6xx";
        case ADRENO_7XX: return "Adreno 7xx";
        case ADRENO_8XX: return "Adreno 8xx (Snapdragon 8 Elite)";
        default: return "Unknown Adreno";
    }
}

void opencl_print_device_info(const OpenCLDeviceInfo* info) {
    if (!info) return;
    
    printf("OpenCL Device Information:\n");
    printf("  Name: %s\n", info->name);
    printf("  Vendor: %s (%s)\n", info->vendor_string, opencl_vendor_name(info->vendor));
    printf("  Driver: %s\n", info->driver_version);
    printf("  OpenCL: %s\n", info->opencl_version);
    
    if (info->vendor == OPENCL_VENDOR_QUALCOMM && info->adreno_gen != ADRENO_UNKNOWN) {
        printf("  Generation: %s\n", opencl_adreno_gen_name(info->adreno_gen));
    }
    
    printf("  Compute Units: %u\n", info->compute_units);
    printf("  Max Work Group Size: %u\n", info->max_work_group_size);
    printf("  Global Memory: %.2f GB\n", info->global_mem_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Local Memory: %.2f KB\n", info->local_mem_size / 1024.0);
    printf("  Max Alloc: %.2f GB\n", info->max_alloc_size / (1024.0 * 1024.0 * 1024.0));
    printf("  Features: FP16=%s FP64=%s INT8=%s Subgroups=%s\n",
           info->has_fp16 ? "yes" : "no",
           info->has_fp64 ? "yes" : "no",
           info->has_int8 ? "yes" : "no",
           info->has_subgroups ? "yes" : "no");
}

bool opencl_is_available(void) {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, NULL, &num_platforms);
    return (err == CL_SUCCESS && num_platforms > 0);
}

