#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "inference_engine.h"
#include "tensor.h"
#include <stdint.h>
#include <stdio.h>

typedef enum {
    FORMAT_VENUS,     // Our custom format
    FORMAT_GGUF,      // GGML Universal Format
    FORMAT_SAFETENSORS,
    FORMAT_UNKNOWN
} ModelFormat;

typedef struct {
    char* name;
    uint32_t shape[4];
    uint32_t n_dims;
    size_t offset;
    size_t size;
    DType dtype;     // Use existing DType from tensor.h
    float scale;     // For quantized types
} TensorInfo;

typedef struct {
    ModelFormat format;
    ModelConfig config;
    TensorInfo* tensors;
    size_t n_tensors;
    void* data;
    size_t data_size;
    char* metadata_json;
} ModelData;

// Main loading functions
ModelData* load_model(const char* path);
void free_model_data(ModelData* data);

// Format-specific loaders
ModelData* load_venus_model(const char* path);
ModelData* load_gguf_model(const char* path);
ModelData* load_safetensors_model(const char* path);

// Utility functions
ModelFormat detect_format(const char* path);
void* get_tensor_data(ModelData* model, const char* name);
TensorInfo* find_tensor(ModelData* model, const char* name);

#endif // MODEL_LOADER_H