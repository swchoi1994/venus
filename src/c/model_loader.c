#include "model_loader.h"
#include "utils/json_helper.h"
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// GGUF format constants
#define GGUF_MAGIC 0x46554747  // "GGUF"
#define GGUF_VERSION 3

// Safetensors format constants
#define SAFETENSORS_MAGIC "{\"__metadata__\":"

// Venus format constants
#define VENUS_MAGIC "VNUS"

ModelFormat detect_format(const char* path) {
    // Prefer detection by file extension when available
    const char* dot = strrchr(path, '.');
    if (dot != NULL) {
        if (strcmp(dot, ".safetensors") == 0) {
            return FORMAT_SAFETENSORS;
        }
        if (strcmp(dot, ".gguf") == 0) {
            return FORMAT_GGUF;
        }
        if (strcmp(dot, ".venus") == 0) {
            return FORMAT_VENUS;
        }
    }

    FILE* f = fopen(path, "rb");
    if (!f) return FORMAT_UNKNOWN;

    char header[16] = {0};
    size_t read = fread(header, 1, sizeof(header), f);
    fclose(f);
    if (read < 4) return FORMAT_UNKNOWN;

    // Check Venus format by magic
    if (memcmp(header, VENUS_MAGIC, 4) == 0) {
        return FORMAT_VENUS;
    }

    // Check GGUF format by magic
    if (*(uint32_t*)header == GGUF_MAGIC) {
        return FORMAT_GGUF;
    }

    // Heuristic: safetensors starts with 8-byte little-endian JSON header length
    // If extension missing, treat as safetensors when size looks plausible
    if (read >= 8) {
        uint64_t json_len = 0;
        memcpy(&json_len, header, sizeof(uint64_t));
        if (json_len > 0 && json_len < (1ULL << 32)) {
            return FORMAT_SAFETENSORS;
        }
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
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        printf("Failed to open file: %s\n", path);
        return NULL;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        printf("Failed to stat file: %s\n", path);
        return NULL;
    }

    size_t file_size = sb.st_size;
    void* mapped_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mapped_data == MAP_FAILED) {
        printf("Failed to mmap file: %s\n", path);
        return NULL;
    }

    ModelData* model = calloc(1, sizeof(ModelData));
    if (!model) {
        munmap(mapped_data, file_size);
        return NULL;
    }
    
    model->format = FORMAT_VENUS;
    model->mapped_addr = mapped_data; // Store original mapped address
    model->mapped_size = file_size;
    model->data = mapped_data;
    model->data_size = file_size;

    char* current_ptr = (char*)mapped_data;

    // Check magic number
    if (memcmp(current_ptr, VENUS_MAGIC, 4) != 0) {
        printf("Invalid Venus magic number\n");
        free_model_data(model);
        return NULL;
    }
    current_ptr += 4;
    
    // Read header size
    uint32_t header_size = *(uint32_t*)current_ptr;
    current_ptr += sizeof(uint32_t);
    
    // Read header JSON
    model->metadata_json = malloc(header_size + 1);
    memcpy(model->metadata_json, current_ptr, header_size);
    model->metadata_json[header_size] = '\0';
    current_ptr += header_size;
    
    // Parse header to fill ModelConfig
    const char* json = model->metadata_json;
    model->config.vocab_size = json_get_int(json, "vocab_size", 32000);
    model->config.hidden_dim = json_get_int(json, "hidden_size", 4096);
    model->config.n_layers = json_get_int(json, "num_layers", 32);
    model->config.n_heads = json_get_int(json, "num_heads", 32);
    model->config.n_kv_heads = json_get_int(json, "num_kv_heads", 32);
    model->config.seq_len = json_get_int(json, "max_position_embeddings", 2048);
    model->config.intermediate_size = json_get_int(json, "intermediate_size", 11008);
    model->config.rope_theta = json_get_float(json, "rope_theta", 10000.0f);
    model->config.layer_norm_eps = json_get_float(json, "layer_norm_eps", 1e-6f);
    model->config.use_gqa = json_get_bool(json, "use_gqa", false);
    model->config.use_rope = json_get_bool(json, "use_rope", true);
    
    // VLM config parsing
    model->config.is_vision_model = json_get_bool(json, "is_vision_model", false);
    if (model->config.is_vision_model) {
        // A real implementation would parse the nested "vision_config" object.
        // This is a simplified stand-in.
        model->config.vision_config.hidden_size = json_get_int(json, "hidden_size", 1024);
        model->config.vision_config.image_size = json_get_int(json, "image_size", 336);
        model->config.vision_config.patch_size = json_get_int(json, "patch_size", 14);
        model->config.vision_config.num_hidden_layers = json_get_int(json, "num_hidden_layers", 24);
        model->config.vision_config.num_attention_heads = json_get_int(json, "num_attention_heads", 16);
        model->config.vision_config.intermediate_size = json_get_int(json, "intermediate_size", 4096);
    }

    // TODO: Parse architecture string and map to enum
    model->config.architecture = ARCH_LLAMA;
    model->config.use_flash_attention = true;
    model->config.use_alibi = false;
    model->config.is_encoder_decoder = false;
    
    // Read number of tensors
    uint32_t n_tensors = *(uint32_t*)current_ptr;
    current_ptr += sizeof(uint32_t);
    model->n_tensors = n_tensors;
    
    model->tensors = calloc(n_tensors, sizeof(TensorInfo));
    
    // The start of the tensor weight data, right after all tensor headers
    char* weights_ptr_start = current_ptr;
    // First, we need to calculate the total size of all tensor headers to find where the weights data begins.
    for (size_t i = 0; i < n_tensors; i++) {
        uint32_t name_len = *(uint32_t*)weights_ptr_start;
        weights_ptr_start += sizeof(uint32_t) + name_len;

        uint32_t quant_len = *(uint32_t*)weights_ptr_start;
        weights_ptr_start += sizeof(uint32_t) + quant_len;

        uint32_t n_dims = *(uint32_t*)weights_ptr_start;
        weights_ptr_start += sizeof(uint32_t) + (n_dims * sizeof(uint32_t));

        weights_ptr_start += sizeof(float); // scale
        weights_ptr_start += sizeof(uint64_t); // data_size
    }


    // Read tensor info, offsets are relative to the start of the weights data
    size_t current_offset = 0;
    for (size_t i = 0; i < n_tensors; i++) {
        TensorInfo* tensor = &model->tensors[i];
        
        uint32_t name_len = *(uint32_t*)current_ptr;
        current_ptr += sizeof(uint32_t);
        tensor->name = malloc(name_len + 1);
        memcpy(tensor->name, current_ptr, name_len);
        tensor->name[name_len] = '\0';
        current_ptr += name_len;
        
        uint32_t quant_len = *(uint32_t*)current_ptr;
        current_ptr += sizeof(uint32_t);
        char* quant_type = malloc(quant_len + 1);
        memcpy(quant_type, current_ptr, quant_len);
        quant_type[quant_len] = '\0';
        current_ptr += quant_len;
        
        if (strcmp(quant_type, "none") == 0) tensor->dtype = DTYPE_F32;
        else if (strcmp(quant_type, "q8_0") == 0) tensor->dtype = DTYPE_INT8;
        else if (strcmp(quant_type, "q4_0") == 0) tensor->dtype = DTYPE_INT4;
        free(quant_type);
        
        tensor->n_dims = *(uint32_t*)current_ptr;
        current_ptr += sizeof(uint32_t);
        memcpy(tensor->shape, current_ptr, tensor->n_dims * sizeof(uint32_t));
        current_ptr += tensor->n_dims * sizeof(uint32_t);
        
        tensor->scale = *(float*)current_ptr;
        current_ptr += sizeof(float);
        
        uint64_t data_size = *(uint64_t*)current_ptr;
        current_ptr += sizeof(uint64_t);
        tensor->size = data_size;
        
        tensor->offset = current_offset;
        current_offset += data_size;
    }

    // With mmap, the tensor data is already "loaded". We just need to point to it.
    // The `data` pointer in ModelData now points to the beginning of the mmap'd file.
    // We adjust it to point to the beginning of the *weights* section.
    model->data = weights_ptr_start;
    model->data_size = file_size - (weights_ptr_start - (char*)mapped_data); // Size of the weights section
    
    printf("Loaded Venus model (mmap): %zu tensors, %.2f MB\n", 
           model->n_tensors, file_size / 1024.0 / 1024.0);
    
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
    
    if (data->format == FORMAT_VENUS && data->mapped_addr) {
        munmap(data->mapped_addr, data->mapped_size);
    } else {
        free(data->data);
    }
    
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