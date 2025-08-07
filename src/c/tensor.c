#include "tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Helper function to calculate strides
static void calculate_strides(int32_t* shape, int32_t n_dims, int32_t* strides) {
    strides[n_dims - 1] = 1;
    for (int i = n_dims - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

// Get size in bytes for data type
size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DTYPE_F32: return sizeof(float);
        case DTYPE_F16: return sizeof(uint16_t);
        case DTYPE_BF16: return sizeof(uint16_t);
        case DTYPE_INT8: return sizeof(int8_t);
        case DTYPE_INT4: return sizeof(int8_t) / 2;
        case DTYPE_INT32: return sizeof(int32_t);
        default: return 0;
    }
}

// Create a new tensor
Tensor* tensor_create(int32_t* shape, int32_t n_dims, DType dtype) {
    Tensor* tensor = (Tensor*)calloc(1, sizeof(Tensor));
    if (!tensor) return NULL;
    
    tensor->n_dims = n_dims;
    tensor->dtype = dtype;
    
    // Allocate shape and strides
    tensor->shape = (int32_t*)malloc(n_dims * sizeof(int32_t));
    tensor->strides = (int32_t*)malloc(n_dims * sizeof(int32_t));
    
    if (!tensor->shape || !tensor->strides) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    // Copy shape and calculate strides
    memcpy(tensor->shape, shape, n_dims * sizeof(int32_t));
    calculate_strides(tensor->shape, tensor->n_dims, tensor->strides);
    
    // Calculate total elements
    tensor->n_elements = 1;
    for (int i = 0; i < n_dims; i++) {
        tensor->n_elements *= shape[i];
    }
    
    // Calculate size in bytes
    tensor->size_bytes = tensor->n_elements * dtype_size(dtype);
    
    // Allocate data
    tensor->data = calloc(1, tensor->size_bytes);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }
    
    tensor->is_contiguous = true;
    
    return tensor;
}

// Free tensor
void tensor_free(Tensor* tensor) {
    if (!tensor) return;
    
    free(tensor->data);
    free(tensor->shape);
    free(tensor->strides);
    free(tensor);
}

// Create zeros tensor
Tensor* tensor_zeros(int32_t* shape, int32_t n_dims, DType dtype) {
    return tensor_create(shape, n_dims, dtype);
}

// Create ones tensor
Tensor* tensor_ones(int32_t* shape, int32_t n_dims, DType dtype) {
    Tensor* tensor = tensor_create(shape, n_dims, dtype);
    if (!tensor) return NULL;
    
    if (dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        for (size_t i = 0; i < tensor->n_elements; i++) {
            data[i] = 1.0f;
        }
    }
    // TODO: Handle other data types
    
    return tensor;
}

// Copy tensor
Tensor* tensor_copy(const Tensor* src) {
    if (!src) return NULL;
    
    Tensor* dst = tensor_create(src->shape, src->n_dims, src->dtype);
    if (!dst) return NULL;
    
    memcpy(dst->data, src->data, src->size_bytes);
    
    return dst;
}

// Get pointer to element
void* tensor_get_ptr(Tensor* tensor, int32_t* indices) {
    size_t offset = 0;
    for (int i = 0; i < tensor->n_dims; i++) {
        offset += indices[i] * tensor->strides[i];
    }
    
    size_t byte_offset = offset * dtype_size(tensor->dtype);
    return (char*)tensor->data + byte_offset;
}

// Get float value
float tensor_get_f32(Tensor* tensor, int32_t* indices) {
    if (tensor->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: tensor_get_f32 called on non-F32 tensor\n");
        return 0.0f;
    }
    
    float* ptr = (float*)tensor_get_ptr(tensor, indices);
    return *ptr;
}

// Set float value
void tensor_set_f32(Tensor* tensor, int32_t* indices, float value) {
    if (tensor->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: tensor_set_f32 called on non-F32 tensor\n");
        return;
    }
    
    float* ptr = (float*)tensor_get_ptr(tensor, indices);
    *ptr = value;
}

// Element-wise addition
void tensor_add(Tensor* a, Tensor* b, Tensor* out) {
    if (!tensor_equal_shape(a, b) || !tensor_equal_shape(a, out)) {
        fprintf(stderr, "Error: tensor_add shape mismatch\n");
        return;
    }
    
    if (a->dtype == DTYPE_F32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* out_data = (float*)out->data;
        
        #pragma omp parallel for
        for (size_t i = 0; i < a->n_elements; i++) {
            out_data[i] = a_data[i] + b_data[i];
        }
    }
    // TODO: Handle other data types
}

// Scale tensor
void tensor_scale(Tensor* tensor, float scale) {
    if (tensor->dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        
        #pragma omp parallel for
        for (size_t i = 0; i < tensor->n_elements; i++) {
            data[i] *= scale;
        }
    }
    // TODO: Handle other data types
}

// Matrix multiplication
void tensor_matmul(Tensor* a, Tensor* b, Tensor* out) {
    // Simple implementation for 2D tensors
    if (a->n_dims != 2 || b->n_dims != 2 || out->n_dims != 2) {
        fprintf(stderr, "Error: tensor_matmul only supports 2D tensors\n");
        return;
    }
    
    int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
    
    if (b->shape[0] != k || out->shape[0] != m || out->shape[1] != n) {
        fprintf(stderr, "Error: tensor_matmul shape mismatch\n");
        return;
    }
    
    if (a->dtype == DTYPE_F32) {
        float* a_data = (float*)a->data;
        float* b_data = (float*)b->data;
        float* out_data = (float*)out->data;
        
        // Zero output
        memset(out_data, 0, out->size_bytes);
        
        // Simple matrix multiplication
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                out_data[i * n + j] = sum;
            }
        }
    }
}

// ReLU activation
void tensor_relu(Tensor* tensor) {
    if (tensor->dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        
        #pragma omp parallel for
        for (size_t i = 0; i < tensor->n_elements; i++) {
            data[i] = fmaxf(0.0f, data[i]);
        }
    }
}

// GELU activation
void tensor_gelu(Tensor* tensor) {
    if (tensor->dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        const float sqrt_2_over_pi = 0.7978845608f;
        
        #pragma omp parallel for
        for (size_t i = 0; i < tensor->n_elements; i++) {
            float x = data[i];
            float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
            data[i] = x * cdf;
        }
    }
}

// SiLU activation
void tensor_silu(Tensor* tensor) {
    if (tensor->dtype == DTYPE_F32) {
        float* data = (float*)tensor->data;
        
        #pragma omp parallel for
        for (size_t i = 0; i < tensor->n_elements; i++) {
            float x = data[i];
            data[i] = x / (1.0f + expf(-x));
        }
    }
}

// Softmax
void tensor_softmax(Tensor* tensor, int32_t dim) {
    if (tensor->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: softmax only supports F32 tensors\n");
        return;
    }
    
    // For simplicity, only support last dimension
    if (dim != tensor->n_dims - 1) {
        fprintf(stderr, "Error: softmax only supports last dimension\n");
        return;
    }
    
    float* data = (float*)tensor->data;
    int inner_size = tensor->shape[dim];
    int outer_size = tensor->n_elements / inner_size;
    
    #pragma omp parallel for
    for (int i = 0; i < outer_size; i++) {
        float* row = data + i * inner_size;
        
        // Find max for numerical stability
        float max_val = row[0];
        for (int j = 1; j < inner_size; j++) {
            max_val = fmaxf(max_val, row[j]);
        }
        
        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < inner_size; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        
        // Normalize
        float inv_sum = 1.0f / sum;
        for (int j = 0; j < inner_size; j++) {
            row[j] *= inv_sum;
        }
    }
}

// RMS normalization
void tensor_rms_norm(Tensor* tensor, Tensor* weight, float eps) {
    if (tensor->dtype != DTYPE_F32 || weight->dtype != DTYPE_F32) {
        fprintf(stderr, "Error: rms_norm only supports F32 tensors\n");
        return;
    }
    
    float* data = (float*)tensor->data;
    float* w = (float*)weight->data;
    
    int hidden_dim = tensor->shape[tensor->n_dims - 1];
    int batch_size = tensor->n_elements / hidden_dim;
    
    #pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        float* row = data + i * hidden_dim;
        
        // Compute RMS
        float sum_sq = 0.0f;
        for (int j = 0; j < hidden_dim; j++) {
            sum_sq += row[j] * row[j];
        }
        float rms = sqrtf(sum_sq / hidden_dim + eps);
        float inv_rms = 1.0f / rms;
        
        // Normalize and apply weight
        for (int j = 0; j < hidden_dim; j++) {
            row[j] = row[j] * inv_rms * w[j];
        }
    }
}

// Print tensor shape
void tensor_print_shape(const Tensor* tensor) {
    printf("Shape: (");
    for (int i = 0; i < tensor->n_dims; i++) {
        printf("%d", tensor->shape[i]);
        if (i < tensor->n_dims - 1) printf(", ");
    }
    printf("), dtype: ");
    
    switch (tensor->dtype) {
        case DTYPE_F32: printf("float32"); break;
        case DTYPE_F16: printf("float16"); break;
        case DTYPE_BF16: printf("bfloat16"); break;
        case DTYPE_INT8: printf("int8"); break;
        case DTYPE_INT4: printf("int4"); break;
        case DTYPE_INT32: printf("int32"); break;
    }
    
    printf(", elements: %zu\n", tensor->n_elements);
}

// Check if shapes are equal
bool tensor_equal_shape(const Tensor* a, const Tensor* b) {
    if (a->n_dims != b->n_dims) return false;
    
    for (int i = 0; i < a->n_dims; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    
    return true;
}