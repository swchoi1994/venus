#ifndef TENSOR_H
#define TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Data types
typedef enum {
    DTYPE_F32,
    DTYPE_F16,
    DTYPE_BF16,
    DTYPE_INT8,
    DTYPE_INT4,
    DTYPE_INT32
} DType;

// Forward declaration
typedef struct Tensor Tensor;

// Tensor structure
struct Tensor {
    void* data;
    int32_t* shape;
    int32_t* strides;
    int32_t n_dims;
    size_t n_elements;
    size_t size_bytes;
    DType dtype;
    bool is_contiguous;
};

// Tensor operations
Tensor* tensor_create(int32_t* shape, int32_t n_dims, DType dtype);
void tensor_free(Tensor* tensor);
Tensor* tensor_zeros(int32_t* shape, int32_t n_dims, DType dtype);
Tensor* tensor_ones(int32_t* shape, int32_t n_dims, DType dtype);
Tensor* tensor_copy(const Tensor* src);
Tensor* tensor_view(Tensor* tensor, int32_t* new_shape, int32_t new_n_dims);
Tensor* tensor_transpose(Tensor* tensor, int32_t dim0, int32_t dim1);
Tensor* tensor_reshape(Tensor* tensor, int32_t* new_shape, int32_t new_n_dims);

// Element access
void* tensor_get_ptr(Tensor* tensor, int32_t* indices);
float tensor_get_f32(Tensor* tensor, int32_t* indices);
void tensor_set_f32(Tensor* tensor, int32_t* indices, float value);

// Basic operations
void tensor_add(Tensor* a, Tensor* b, Tensor* out);
void tensor_sub(Tensor* a, Tensor* b, Tensor* out);
void tensor_mul(Tensor* a, Tensor* b, Tensor* out);
void tensor_div(Tensor* a, Tensor* b, Tensor* out);
void tensor_scale(Tensor* tensor, float scale);

// Matrix operations
void tensor_matmul(Tensor* a, Tensor* b, Tensor* out);
void tensor_matmul_transposed(Tensor* a, Tensor* b, Tensor* out, bool trans_a, bool trans_b);

// Activation functions
void tensor_relu(Tensor* tensor);
void tensor_gelu(Tensor* tensor);
void tensor_swiglu(Tensor* tensor);
void tensor_silu(Tensor* tensor);
void tensor_softmax(Tensor* tensor, int32_t dim);

// Normalization
void tensor_layer_norm(Tensor* tensor, Tensor* weight, Tensor* bias, float eps);
void tensor_rms_norm(Tensor* tensor, Tensor* weight, float eps);

// Utility functions
void tensor_print_shape(const Tensor* tensor);
void tensor_print(const Tensor* tensor);
bool tensor_equal_shape(const Tensor* a, const Tensor* b);
size_t dtype_size(DType dtype);

#ifdef __cplusplus
}
#endif

#endif // TENSOR_H