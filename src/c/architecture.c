#include "architecture.h"
#include <string.h>
#include <stdlib.h>

int detect_architecture_from_metadata(const char* model_type, const char* architectures) {
    // Convert to lowercase for comparison
    char type_lower[256];
    strncpy(type_lower, model_type, 255);
    type_lower[255] = '\0';
    
    // Convert to lowercase
    for (int i = 0; type_lower[i]; i++) {
        if (type_lower[i] >= 'A' && type_lower[i] <= 'Z') {
            type_lower[i] = type_lower[i] + 32;
        }
    }
    
    // Check for each architecture
    if (strstr(type_lower, "llama")) {
        if (strstr(architectures, "LlamaForConditionalGeneration")) {
            return ARCH_LLAMA_VISION;
        }
        return ARCH_LLAMA;
    }
    else if (strstr(type_lower, "qwen2")) {
        if (strstr(type_lower, "vl")) {
            return ARCH_QWEN2_VL;
        }
        return ARCH_QWEN;
    }
    else if (strstr(type_lower, "qwen")) {
        return ARCH_QWEN;
    }
    else if (strstr(type_lower, "mistral")) {
        return ARCH_MISTRAL;
    }
    else if (strstr(type_lower, "mixtral")) {
        return ARCH_MIXTRAL;
    }
    else if (strstr(type_lower, "falcon")) {
        return ARCH_FALCON;
    }
    else if (strstr(type_lower, "bloom")) {
        return ARCH_BLOOM;
    }
    else if (strstr(type_lower, "opt")) {
        return ARCH_OPT;
    }
    else if (strstr(type_lower, "codegen")) {
        return ARCH_CODEGEN;
    }
    else if (strstr(type_lower, "gpt_bigcode") || strstr(type_lower, "starcoder")) {
        return ARCH_GPTBIGCODE;
    }
    else if (strstr(type_lower, "baichuan")) {
        return ARCH_BAICHUAN;
    }
    else if (strstr(type_lower, "chatglm")) {
        return ARCH_CHATGLM;
    }
    else if (strstr(type_lower, "persimmon")) {
        return ARCH_PERSIMMON;
    }
    else if (strstr(type_lower, "mamba")) {
        return ARCH_MAMBA;
    }
    else if (strstr(type_lower, "deepseek")) {
        return ARCH_DEEPSEEK;
    }
    else if (strstr(type_lower, "phi3")) {
        return ARCH_PHI3;
    }
    else if (strstr(type_lower, "phi")) {
        return ARCH_PHI;
    }
    else if (strstr(type_lower, "gemma2")) {
        return ARCH_GEMMA2;
    }
    else if (strstr(type_lower, "gemma")) {
        return ARCH_GEMMA;
    }
    else if (strstr(type_lower, "stablelm")) {
        return ARCH_STABLELM;
    }
    else if (strstr(type_lower, "starcoder2")) {
        return ARCH_STARCODER2;
    }
    else if (strstr(type_lower, "exaone")) {
        return ARCH_EXAONE;
    }
    else if (strstr(type_lower, "minicpm3")) {
        return ARCH_MINICPM3;
    }
    else if (strstr(type_lower, "minicpm")) {
        return ARCH_MINICPM;
    }
    else if (strstr(type_lower, "dbrx")) {
        return ARCH_DBRX;
    }
    else if (strstr(type_lower, "olmo")) {
        return ARCH_OLMO;
    }
    else if (strstr(type_lower, "arctic")) {
        return ARCH_ARCTIC;
    }
    else if (strstr(type_lower, "xverse")) {
        return ARCH_XVERSE;
    }
    else if (strstr(type_lower, "command-r") || strstr(type_lower, "cohere")) {
        return ARCH_COMMAND_R;
    }
    else if (strstr(type_lower, "deci")) {
        return ARCH_DECI;
    }
    else if (strstr(type_lower, "bamba")) {
        return ARCH_BAMBA;
    }
    else if (strstr(type_lower, "nemotron")) {
        return ARCH_NEMOTRON;
    }
    else if (strstr(type_lower, "plm0")) {
        return ARCH_PLM0;
    }
    else if (strstr(type_lower, "solar")) {
        return ARCH_SOLAR;
    }
    else if (strstr(type_lower, "granite")) {
        if (strstr(type_lower, "moe")) {
            return ARCH_GRANITE_MOE;
        }
        return ARCH_GRANITE;
    }
    else if (strstr(type_lower, "voxtral")) {
        return ARCH_VOXTRAL;
    }
    else if (strstr(type_lower, "gpt") || strstr(type_lower, "gpt2") || 
             strstr(type_lower, "gptj") || strstr(type_lower, "gpt-j") ||
             strstr(type_lower, "gpt_neox") || strstr(type_lower, "gpt-neox")) {
        return ARCH_GPT;
    }
    else if (strstr(type_lower, "bert")) {
        return ARCH_BERT;
    }
    else if (strstr(type_lower, "t5")) {
        return ARCH_T5;
    }
    
    return ARCH_UNKNOWN;
}

void configure_architecture(ModelConfig* config, int architecture) {
    // Set architecture-specific defaults
    switch (architecture) {
        case ARCH_LLAMA:
        case ARCH_LLAMA_VISION:
            config->use_rope = true;
            config->use_gqa = (config->n_kv_heads < config->n_heads);
            config->rope_theta = 10000.0f;
            break;
            
        case ARCH_QWEN:
        case ARCH_QWEN2_VL:
            config->use_rope = true;
            config->use_gqa = true;
            config->rope_theta = 1000000.0f;  // Qwen uses higher rope_theta
            break;
            
        case ARCH_MISTRAL:
            config->use_rope = true;
            config->use_gqa = true;
            config->sliding_window = 4096;
            break;
            
        case ARCH_MIXTRAL:
            config->use_rope = true;
            config->use_gqa = true;
            config->num_experts = 8;
            config->num_experts_per_tok = 2;
            break;
            
        case ARCH_FALCON:
            config->use_alibi = true;
            config->use_rope = false;
            break;
            
        case ARCH_MAMBA:
            // Mamba uses state-space models, not attention
            config->use_flash_attention = false;
            config->use_rope = false;
            config->use_alibi = false;
            break;
            
        case ARCH_T5:
            config->is_encoder_decoder = true;
            config->use_rope = false;
            break;
            
        default:
            // Use sensible defaults
            config->use_rope = true;
            config->use_gqa = false;
            break;
    }
}

size_t calculate_memory_requirements(ModelConfig* config, bool quantized) {
    // Calculate parameter count
    size_t params = 0;
    
    // Embedding layers
    params += (size_t)config->vocab_size * config->hidden_dim;
    
    // Attention layers
    size_t attn_params_per_layer = 0;
    attn_params_per_layer += config->hidden_dim * config->hidden_dim * 4; // Q, K, V, O projections
    
    // MLP layers
    size_t mlp_params_per_layer = 0;
    mlp_params_per_layer += config->hidden_dim * config->intermediate_size * 2; // Up and down
    
    // Layer norms
    size_t norm_params_per_layer = config->hidden_dim * 2;
    
    // Total for all layers
    params += config->n_layers * (attn_params_per_layer + mlp_params_per_layer + norm_params_per_layer);
    
    // Output layer
    if (!config->tie_word_embeddings) {
        params += (size_t)config->vocab_size * config->hidden_dim;
    }
    
    // MoE models have more parameters
    if (config->num_experts > 0) {
        params *= config->num_experts / config->num_experts_per_tok;
    }
    
    // Calculate memory in bytes
    size_t bytes_per_param = quantized ? 1 : 4;  // INT8 vs FP32
    size_t param_memory = params * bytes_per_param;
    
    // Add KV cache memory (assume 10% of model size)
    size_t kv_cache_memory = param_memory / 10;
    
    // Add activation memory (assume 20% of model size)
    size_t activation_memory = param_memory / 5;
    
    // Total memory requirement
    return param_memory + kv_cache_memory + activation_memory;
}

size_t get_max_parameters_for_memory(size_t available_memory, bool quantized) {
    // Reserve 20% for system overhead
    size_t usable_memory = (available_memory * 80) / 100;
    
    // Account for KV cache and activations (roughly 1.3x model size)
    size_t model_memory = usable_memory / 1.3;
    
    // Calculate parameters based on quantization
    size_t bytes_per_param = quantized ? 1 : 4;
    return model_memory / bytes_per_param;
}