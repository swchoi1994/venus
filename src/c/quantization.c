#include "quantization.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#ifdef USE_OPENMP
#include <omp.h>
#endif

// Helper functions
static float find_absmax(const float* data, int n) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) {
        max_val = fmaxf(max_val, fabsf(data[i]));
    }
    return max_val;
}

// Q8_0 quantization: 8-bit with group scaling
void quantize_q8_0(const float* src, void* dst, int n) {
    const int group_size = 32;
    int n_groups = (n + group_size - 1) / group_size;
    BlockQ8_0* blocks = (BlockQ8_0*)dst;
    
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = fminf(start + group_size, n);
        int group_elements = end - start;
        
        // Find scale for this group
        float absmax = find_absmax(src + start, group_elements);
        float scale = absmax / 127.0f;
        float inv_scale = (scale > 0) ? 1.0f / scale : 0.0f;
        
        blocks[g].scale = scale;
        
        // Quantize values
        for (int i = 0; i < group_elements; i++) {
            int quantized = roundf(src[start + i] * inv_scale);
            blocks[g].qs[i] = fmaxf(-128, fminf(127, quantized));
        }
        
        // Pad remaining values with zeros
        for (int i = group_elements; i < group_size; i++) {
            blocks[g].qs[i] = 0;
        }
    }
}

// Q8_0 dequantization
void dequantize_q8_0(const void* src, float* dst, int n) {
    const int group_size = 32;
    int n_groups = (n + group_size - 1) / group_size;
    const BlockQ8_0* blocks = (const BlockQ8_0*)src;
    
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = fminf(start + group_size, n);
        float scale = blocks[g].scale;
        
        for (int i = start; i < end; i++) {
            dst[i] = blocks[g].qs[i - start] * scale;
        }
    }
}

// Q4_0 quantization: 4-bit with group scaling
void quantize_q4_0(const float* src, void* dst, int n) {
    const int group_size = 32;
    int n_groups = (n + group_size - 1) / group_size;
    BlockQ4_0* blocks = (BlockQ4_0*)dst;
    
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = fminf(start + group_size, n);
        int group_elements = end - start;
        
        // Find scale for this group
        float absmax = find_absmax(src + start, group_elements);
        float scale = absmax / 7.0f;
        float inv_scale = (scale > 0) ? 1.0f / scale : 0.0f;
        
        blocks[g].scale = scale;
        
        // Quantize values (pack 2 4-bit values per byte)
        for (int i = 0; i < group_elements; i += 2) {
            int q0 = roundf(src[start + i] * inv_scale);
            q0 = fmaxf(-8, fminf(7, q0));
            
            int q1 = 0;
            if (i + 1 < group_elements) {
                q1 = roundf(src[start + i + 1] * inv_scale);
                q1 = fmaxf(-8, fminf(7, q1));
            }
            
            // Pack into byte (q0 in lower 4 bits, q1 in upper 4 bits)
            blocks[g].qs[i / 2] = ((q1 + 8) << 4) | (q0 + 8);
        }
    }
}

// Q4_0 dequantization
void dequantize_q4_0(const void* src, float* dst, int n) {
    const int group_size = 32;
    int n_groups = (n + group_size - 1) / group_size;
    const BlockQ4_0* blocks = (const BlockQ4_0*)src;
    
    #pragma omp parallel for
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = fminf(start + group_size, n);
        float scale = blocks[g].scale;
        
        for (int i = start; i < end; i += 2) {
            uint8_t packed = blocks[g].qs[(i - start) / 2];
            
            // Unpack lower 4 bits
            int q0 = (packed & 0x0F) - 8;
            dst[i] = q0 * scale;
            
            // Unpack upper 4 bits
            if (i + 1 < end) {
                int q1 = ((packed >> 4) & 0x0F) - 8;
                dst[i + 1] = q1 * scale;
            }
        }
    }
}

// Dot product for Q8_0 quantized vectors
void vec_dot_q8_0(const void* x, const void* y, float* result, int n) {
    const int group_size = 32;
    int n_groups = (n + group_size - 1) / group_size;
    const BlockQ8_0* x_blocks = (const BlockQ8_0*)x;
    const BlockQ8_0* y_blocks = (const BlockQ8_0*)y;
    
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int g = 0; g < n_groups; g++) {
        float scale = x_blocks[g].scale * y_blocks[g].scale;
        int32_t dot = 0;
        
        // Compute integer dot product
        for (int i = 0; i < group_size; i++) {
            dot += (int32_t)x_blocks[g].qs[i] * (int32_t)y_blocks[g].qs[i];
        }
        
        sum += scale * dot;
    }
    
    *result = sum;
}

// Mixed precision dot product (float32 * Q8_0)
void vec_dot_f32_q8_0(const float* x, const void* y, float* result, int n) {
    const int group_size = 32;
    int n_groups = (n + group_size - 1) / group_size;
    const BlockQ8_0* y_blocks = (const BlockQ8_0*)y;
    
    float sum = 0.0f;
    
    #pragma omp parallel for reduction(+:sum)
    for (int g = 0; g < n_groups; g++) {
        int start = g * group_size;
        int end = fminf(start + group_size, n);
        float scale = y_blocks[g].scale;
        float group_sum = 0.0f;
        
        for (int i = 0; i < end - start; i++) {
            group_sum += x[start + i] * y_blocks[g].qs[i];
        }
        
        sum += scale * group_sum;
    }
    
    *result = sum;
}

// Create quantized tensor
QuantizedTensor* quantize_tensor(Tensor* tensor, QuantType type) {
    if (!tensor || tensor->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: quantize_tensor requires F32 tensor\n");
        return NULL;
    }
    
    QuantizedTensor* qtensor = (QuantizedTensor*)calloc(1, sizeof(QuantizedTensor));
    if (!qtensor) return NULL;
    
    qtensor->type = type;
    qtensor->n_elements = tensor->n_elements;
    qtensor->n_dims = tensor->n_dims;
    qtensor->shape = (int*)malloc(tensor->n_dims * sizeof(int));
    memcpy(qtensor->shape, tensor->shape, tensor->n_dims * sizeof(int));
    
    float* src = (float*)tensor->data;
    
    switch (type) {
        case QUANT_Q8_0:
            qtensor->block_size = 32;
            qtensor->n_blocks = (qtensor->n_elements + qtensor->block_size - 1) / qtensor->block_size;
            qtensor->data = calloc(qtensor->n_blocks, sizeof(BlockQ8_0));
            if (qtensor->data) {
                quantize_q8_0(src, qtensor->data, qtensor->n_elements);
            }
            break;
            
        case QUANT_Q4_0:
            qtensor->block_size = 32;
            qtensor->n_blocks = (qtensor->n_elements + qtensor->block_size - 1) / qtensor->block_size;
            qtensor->data = calloc(qtensor->n_blocks, sizeof(BlockQ4_0));
            if (qtensor->data) {
                quantize_q4_0(src, qtensor->data, qtensor->n_elements);
            }
            break;
            
        default:
            fprintf(stderr, "Error: Unsupported quantization type\n");
            free(qtensor->shape);
            free(qtensor);
            return NULL;
    }
    
    if (!qtensor->data) {
        free(qtensor->shape);
        free(qtensor);
        return NULL;
    }
    
    return qtensor;
}

// Dequantize tensor
Tensor* dequantize_tensor(QuantizedTensor* qtensor) {
    if (!qtensor) return NULL;
    
    Tensor* tensor = tensor_create(qtensor->shape, qtensor->n_dims, DTYPE_F32);
    if (!tensor) return NULL;
    
    float* dst = (float*)tensor->data;
    
    switch (qtensor->type) {
        case QUANT_Q8_0:
            dequantize_q8_0(qtensor->data, dst, qtensor->n_elements);
            break;
            
        case QUANT_Q4_0:
            dequantize_q4_0(qtensor->data, dst, qtensor->n_elements);
            break;
            
        default:
            fprintf(stderr, "Error: Unsupported quantization type for dequantization\n");
            tensor_free(tensor);
            return NULL;
    }
    
    return tensor;
}

// Free quantized tensor
void free_quantized_tensor(QuantizedTensor* qtensor) {
    if (!qtensor) return;
    
    free(qtensor->data);
    free(qtensor->shape);
    free(qtensor);
}

// Get quantized tensor size
size_t quantized_tensor_size(QuantType type, size_t n_elements) {
    const int group_size = 32;
    size_t n_groups = (n_elements + group_size - 1) / group_size;
    
    switch (type) {
        case QUANT_Q8_0:
            return n_groups * sizeof(BlockQ8_0);
        case QUANT_Q4_0:
            return n_groups * sizeof(BlockQ4_0);
        case QUANT_Q4_K:
            return n_groups * sizeof(BlockQ4_K);
        default:
            return n_elements * sizeof(float);
    }
}

// Get quantization type name
const char* quant_type_name(QuantType type) {
    switch (type) {
        case QUANT_NONE: return "none";
        case QUANT_Q8_0: return "q8_0";
        case QUANT_Q4_0: return "q4_0";
        case QUANT_Q4_K: return "q4_k";
        case QUANT_Q5_K: return "q5_k";
        case QUANT_Q6_K: return "q6_k";
        default: return "unknown";
    }
}

// Calculate quantization error
float quantization_error(Tensor* original, QuantizedTensor* quantized) {
    if (!original || !quantized || original->n_elements != quantized->n_elements) {
        return FLT_MAX;
    }
    
    // Dequantize and compute error
    Tensor* reconstructed = dequantize_tensor(quantized);
    if (!reconstructed) return FLT_MAX;
    
    float* orig_data = (float*)original->data;
    float* recon_data = (float*)reconstructed->data;
    
    float mse = 0.0f;
    float max_error = 0.0f;
    
    #pragma omp parallel for reduction(+:mse) reduction(max:max_error)
    for (size_t i = 0; i < original->n_elements; i++) {
        float error = orig_data[i] - recon_data[i];
        mse += error * error;
        max_error = fmaxf(max_error, fabsf(error));
    }
    
    mse /= original->n_elements;
    
    tensor_free(reconstructed);
    
    printf("Quantization %s - MSE: %.6f, Max Error: %.6f\n", 
           quant_type_name(quantized->type), mse, max_error);
    
    return mse;
}