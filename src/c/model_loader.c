#include "model_loader.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// GGUF format constants
#define GGUF_MAGIC 0x46554747  // "GGUF"
#define GGUF_VERSION 3

// Safetensors format constants
#define SAFETENSORS_MAGIC "{\"__metadata__\":"

// Venus format constants
#define VENUS_MAGIC "VNUS"

ModelFormat detect_format(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return FORMAT_UNKNOWN;
    
    char magic[16];
    size_t read = fread(magic, 1, 16, f);
    fclose(f);
    
    if (read < 4) return FORMAT_UNKNOWN;
    
    // Check Venus format
    if (memcmp(magic, VENUS_MAGIC, 4) == 0) {
        return FORMAT_VENUS;
    }
    
    // Check GGUF format
    if (*(uint32_t*)magic == GGUF_MAGIC) {
        return FORMAT_GGUF;
    }
    
    // Check safetensors format (JSON header)
    if (memcmp(magic, SAFETENSORS_MAGIC, strlen(SAFETENSORS_MAGIC)) == 0) {
        return FORMAT_SAFETENSORS;
    }
    
    return FORMAT_UNKNOWN;
}

ModelData* load_model(const char* path) {
    ModelFormat format = detect_format(path);
    
    switch (format) {
        case FORMAT_VENUS:
            return load_venus_model(path);
        case FORMAT_GGUF:
            return load_gguf_model(path);
        case FORMAT_SAFETENSORS:
            return load_safetensors_model(path);
        default:
            printf("Unknown model format: %s\n", path);
            return NULL;
    }
}

ModelData* load_venus_model(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open file: %s\n", path);
        return NULL;
    }
    
    ModelData* model = calloc(1, sizeof(ModelData));
    if (!model) {
        fclose(f);
        return NULL;
    }
    
    model->format = FORMAT_VENUS;
    
    // Read magic
    char magic[4];
    fread(magic, 1, 4, f);
    
    // Read header size
    uint32_t header_size;
    fread(&header_size, sizeof(uint32_t), 1, f);
    
    // Read header JSON
    model->metadata_json = malloc(header_size + 1);
    fread(model->metadata_json, 1, header_size, f);
    model->metadata_json[header_size] = '\0';
    
    // Parse header to fill ModelConfig
    // TODO: Use a proper JSON parser
    // For now, we'll set some defaults
    model->config.vocab_size = 32000;
    model->config.hidden_dim = 4096;
    model->config.n_layers = 32;
    model->config.n_heads = 32;
    model->config.n_kv_heads = 32;
    model->config.seq_len = 2048;
    model->config.intermediate_size = 11008;
    model->config.rope_theta = 10000.0f;
    model->config.layer_norm_eps = 1e-6f;
    model->config.architecture = ARCH_LLAMA;
    model->config.use_gqa = false;
    model->config.use_flash_attention = true;
    model->config.use_rope = true;
    model->config.use_alibi = false;
    model->config.is_encoder_decoder = false;
    
    // Read number of tensors
    uint32_t n_tensors;
    fread(&n_tensors, sizeof(uint32_t), 1, f);
    model->n_tensors = n_tensors;
    
    model->tensors = calloc(n_tensors, sizeof(TensorInfo));
    
    // Calculate total data size
    size_t total_data_size = 0;
    
    // Read tensor info
    for (size_t i = 0; i < n_tensors; i++) {
        TensorInfo* tensor = &model->tensors[i];
        
        // Read name
        uint32_t name_len;
        fread(&name_len, sizeof(uint32_t), 1, f);
        tensor->name = malloc(name_len + 1);
        fread(tensor->name, 1, name_len, f);
        tensor->name[name_len] = '\0';
        
        // Read quantization type
        uint32_t quant_len;
        fread(&quant_len, sizeof(uint32_t), 1, f);
        char* quant_type = malloc(quant_len + 1);
        fread(quant_type, 1, quant_len, f);
        quant_type[quant_len] = '\0';
        
        // Set dtype based on quantization
        if (strcmp(quant_type, "none") == 0) {
            tensor->dtype = DTYPE_F32;
        } else if (strcmp(quant_type, "q8_0") == 0) {
            tensor->dtype = DTYPE_INT8;
        } else if (strcmp(quant_type, "q4_0") == 0) {
            tensor->dtype = DTYPE_INT4;
        }
        free(quant_type);
        
        // Read shape
        fread(&tensor->n_dims, sizeof(uint32_t), 1, f);
        for (uint32_t j = 0; j < tensor->n_dims; j++) {
            fread(&tensor->shape[j], sizeof(uint32_t), 1, f);
        }
        
        // Read scale
        fread(&tensor->scale, sizeof(float), 1, f);
        
        // Read data size
        uint64_t data_size;
        fread(&data_size, sizeof(uint64_t), 1, f);
        tensor->size = data_size;
        
        // Set offset (will be updated when we allocate data)
        tensor->offset = total_data_size;
        total_data_size += data_size;
        
        // Skip the actual data for now
        fseek(f, data_size, SEEK_CUR);
    }
    
    // Allocate memory for all tensor data
    model->data_size = total_data_size;
    model->data = malloc(total_data_size);
    if (!model->data) {
        free_model_data(model);
        fclose(f);
        return NULL;
    }
    
    // Re-read file to load actual tensor data
    fseek(f, 4 + sizeof(uint32_t) + header_size + sizeof(uint32_t), SEEK_SET);
    
    // Skip tensor headers and load data
    for (size_t i = 0; i < n_tensors; i++) {
        TensorInfo* tensor = &model->tensors[i];
        
        // Skip header info we already read
        uint32_t name_len;
        fread(&name_len, sizeof(uint32_t), 1, f);
        fseek(f, name_len, SEEK_CUR);
        
        uint32_t quant_len;
        fread(&quant_len, sizeof(uint32_t), 1, f);
        fseek(f, quant_len, SEEK_CUR);
        
        uint32_t n_dims;
        fread(&n_dims, sizeof(uint32_t), 1, f);
        fseek(f, n_dims * sizeof(uint32_t), SEEK_CUR);
        
        float scale;
        fread(&scale, sizeof(float), 1, f);
        
        uint64_t data_size;
        fread(&data_size, sizeof(uint64_t), 1, f);
        
        // Read actual data
        fread((char*)model->data + tensor->offset, 1, tensor->size, f);
    }
    
    fclose(f);
    
    printf("Loaded Venus model: %zu tensors, %.2f MB\n", 
           model->n_tensors, model->data_size / 1024.0 / 1024.0);
    
    return model;
}

ModelData* load_gguf_model(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open GGUF file: %s\n", path);
        return NULL;
    }
    
    ModelData* model = calloc(1, sizeof(ModelData));
    if (!model) {
        fclose(f);
        return NULL;
    }
    
    model->format = FORMAT_GGUF;
    
    // Read GGUF header
    struct {
        uint32_t magic;
        uint32_t version;
        uint64_t n_tensors;
        uint64_t n_kv;
    } header;
    
    fread(&header, sizeof(header), 1, f);
    
    if (header.magic != GGUF_MAGIC) {
        printf("Invalid GGUF magic\n");
        free(model);
        fclose(f);
        return NULL;
    }
    
    printf("GGUF version: %u\n", header.version);
    printf("Tensors: %llu\n", (unsigned long long)header.n_tensors);
    printf("KV pairs: %llu\n", (unsigned long long)header.n_kv);
    
    // TODO: Implement full GGUF loading
    // This is a simplified version
    model->n_tensors = header.n_tensors;
    model->tensors = calloc(header.n_tensors, sizeof(TensorInfo));
    
    // Set default config for now
    model->config.vocab_size = 32000;
    model->config.hidden_dim = 4096;
    model->config.n_layers = 32;
    model->config.n_heads = 32;
    model->config.architecture = ARCH_LLAMA;
    
    fclose(f);
    return model;
}

ModelData* load_safetensors_model(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open safetensors file: %s\n", path);
        return NULL;
    }
    
    ModelData* model = calloc(1, sizeof(ModelData));
    if (!model) {
        fclose(f);
        return NULL;
    }
    
    model->format = FORMAT_SAFETENSORS;
    
    // Read header size (first 8 bytes)
    uint64_t header_size;
    fread(&header_size, sizeof(uint64_t), 1, f);
    
    // Read JSON header
    char* header_json = malloc(header_size + 1);
    fread(header_json, 1, header_size, f);
    header_json[header_size] = '\0';
    
    model->metadata_json = header_json;
    
    printf("Safetensors header: %s\n", header_json);
    
    // TODO: Parse JSON header and load tensors
    // This requires a JSON parser
    
    // Set default config for now
    model->config.vocab_size = 32000;
    model->config.hidden_dim = 4096;
    model->config.n_layers = 32;
    model->config.n_heads = 32;
    model->config.architecture = ARCH_LLAMA;
    
    fclose(f);
    return model;
}

void free_model_data(ModelData* data) {
    if (!data) return;
    
    if (data->tensors) {
        for (size_t i = 0; i < data->n_tensors; i++) {
            free(data->tensors[i].name);
        }
        free(data->tensors);
    }
    
    free(data->data);
    free(data->metadata_json);
    free(data);
}

void* get_tensor_data(ModelData* model, const char* name) {
    TensorInfo* info = find_tensor(model, name);
    if (!info) return NULL;
    
    return (char*)model->data + info->offset;
}

TensorInfo* find_tensor(ModelData* model, const char* name) {
    for (size_t i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensors[i].name, name) == 0) {
            return &model->tensors[i];
        }
    }
    return NULL;
}