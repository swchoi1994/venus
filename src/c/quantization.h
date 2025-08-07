#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include "tensor.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Quantization types
typedef enum {
    QUANT_NONE,    // No quantization (FP32)
    QUANT_Q8_0,    // 8-bit quantization with group scaling
    QUANT_Q4_0,    // 4-bit quantization with group scaling
    QUANT_Q4_K,    // 4-bit quantization with k-means
    QUANT_Q5_K,    // 5-bit quantization with k-means
    QUANT_Q6_K     // 6-bit quantization with k-means
} QuantType;

// Quantized block structure for Q8_0
typedef struct {
    float scale;           // Scale factor
    int8_t qs[32];        // Quantized values (group size 32)
} BlockQ8_0;

// Quantized block structure for Q4_0
typedef struct {
    float scale;           // Scale factor
    uint8_t qs[16];       // Quantized values (group size 32, packed 2 per byte)
} BlockQ4_0;

// Quantized block structure for Q4_K
typedef struct {
    float d;              // Delta
    float dmin;           // Minimum delta
    uint8_t scales[12];   // Scales (6-bit each)
    uint8_t qs[128];      // Quantized values
} BlockQ4_K;

// Forward declaration
typedef struct QuantizedTensor QuantizedTensor;

// Quantized tensor structure
struct QuantizedTensor {
    QuantType type;
    void* data;           // Pointer to quantized blocks
    size_t n_blocks;      // Number of blocks
    int block_size;       // Elements per block
    int* shape;           // Original tensor shape
    int n_dims;           // Number of dimensions
    size_t n_elements;    // Total elements
};

// Quantization functions
QuantizedTensor* quantize_tensor(Tensor* tensor, QuantType type);
Tensor* dequantize_tensor(QuantizedTensor* qtensor);
void free_quantized_tensor(QuantizedTensor* qtensor);

// Quantization operations
void quantize_q8_0(const float* src, void* dst, int n);
void quantize_q4_0(const float* src, void* dst, int n);
void quantize_q4_k(const float* src, void* dst, int n);

// Dequantization operations
void dequantize_q8_0(const void* src, float* dst, int n);
void dequantize_q4_0(const void* src, float* dst, int n);
void dequantize_q4_k(const void* src, float* dst, int n);

// Quantized operations
void vec_dot_q8_0(const void* x, const void* y, float* result, int n);
void vec_dot_q4_0(const void* x, const void* y, float* result, int n);
void vec_dot_q4_k(const void* x, const void* y, float* result, int n);

// Mixed precision operations
void vec_dot_f32_q8_0(const float* x, const void* y, float* result, int n);
void vec_dot_f32_q4_0(const float* x, const void* y, float* result, int n);

// Utility functions
size_t quantized_tensor_size(QuantType type, size_t n_elements);
const char* quant_type_name(QuantType type);
float quantization_error(Tensor* original, QuantizedTensor* quantized);

#ifdef __cplusplus
}
#endif

#endif // QUANTIZATION_H