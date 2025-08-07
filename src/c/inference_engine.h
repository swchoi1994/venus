#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version information
#define VENUS_VERSION_MAJOR 0
#define VENUS_VERSION_MINOR 1
#define VENUS_VERSION_PATCH 0

// Model configuration
typedef struct {
    int vocab_size;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int seq_len;
    int intermediate_size;
    float rope_theta;
    float layer_norm_eps;
    
    // Additional parameters for large models
    int num_experts;          // For MoE models
    int num_experts_per_tok;  // For MoE models
    int sliding_window;       // For models with sliding window attention
    int max_position_embeddings;
    int tie_word_embeddings;  // Whether to tie input/output embeddings
    int num_hidden_layers;    // Total hidden layers
    int num_attention_heads;  // Total attention heads
    int num_key_value_heads;  // For GQA
    float rms_norm_eps;       // For models using RMSNorm
    float rope_scaling;       // RoPE scaling factor
    int bos_token_id;
    int eos_token_id;
    int pad_token_id;
    
    // Architecture type - supports all vLLM architectures
    enum {
        ARCH_LLAMA,           // LlamaForCausalLM
        ARCH_QWEN,            // Qwen2ForCausalLM
        ARCH_MISTRAL,         // MistralForCausalLM
        ARCH_GPT,             // GPT2, GPT-J, GPT-NeoX
        ARCH_BERT,            // BERT models
        ARCH_T5,              // T5ForConditionalGeneration
        ARCH_FALCON,          // FalconForCausalLM
        ARCH_BLOOM,           // BloomForCausalLM
        ARCH_OPT,             // OPTForCausalLM
        ARCH_CODEGEN,         // CodeGenForCausalLM
        ARCH_GPTBIGCODE,      // GPTBigCodeForCausalLM
        ARCH_BAICHUAN,        // BaichuanForCausalLM
        ARCH_CHATGLM,         // ChatGLMForCausalLM
        ARCH_PERSIMMON,       // PersimmonForCausalLM
        ARCH_MAMBA,           // MambaForCausalLM
        ARCH_MIXTRAL,         // MixtralForCausalLM (MoE)
        ARCH_DEEPSEEK,        // DeepseekForCausalLM
        ARCH_QWEN2_VL,        // Qwen2VLForConditionalGeneration
        ARCH_LLAMA_VISION,    // LlamaForConditionalGeneration (Vision)
        ARCH_PHI,             // PhiForCausalLM
        ARCH_PHI3,            // Phi3ForCausalLM
        ARCH_GEMMA,           // GemmaForCausalLM
        ARCH_GEMMA2,          // Gemma2ForCausalLM
        ARCH_STABLELM,        // StableLMForCausalLM
        ARCH_STARCODER2,      // Starcoder2ForCausalLM
        ARCH_EXAONE,          // Exaone4ForCausalLM
        ARCH_MINICPM,         // MiniCPMForCausalLM
        ARCH_DBRX,            // DbrxForCausalLM
        ARCH_OLMO,            // OlmoForCausalLM
        ARCH_ARCTIC,          // ArcticForCausalLM
        ARCH_XVERSE,          // XverseForCausalLM
        ARCH_COMMAND_R,       // CohereForCausalLM
        ARCH_DECI,            // DeciLMForCausalLM
        ARCH_BAMBA,           // BambaForCausalLM
        ARCH_NEMOTRON,        // NemotronForCausalLM
        ARCH_MINICPM3,        // MiniCPM3ForCausalLM
        ARCH_PLM0,            // PLM0ForCausalLM
        ARCH_SOLAR,           // SolarForCausalLM
        ARCH_GRANITE,         // GraniteForCausalLM
        ARCH_GRANITE_MOE,     // GraniteMoeForCausalLM
        ARCH_VOXTRAL,         // VoxtralForConditionalGeneration
        ARCH_UNKNOWN
    } architecture;
    
    // Feature flags
    bool use_gqa;
    bool use_flash_attention;
    bool use_rope;
    bool use_alibi;
    bool is_encoder_decoder;
} ModelConfig;

// Generation configuration
typedef struct {
    float temperature;
    float top_p;
    int top_k;
    int max_tokens;
    int seed;
    float repetition_penalty;
    float presence_penalty;
    float frequency_penalty;
} GenerationConfig;

// Forward declarations (defined in tensor.h and quantization.h)
typedef struct Tensor Tensor;
typedef struct QuantizedTensor QuantizedTensor;

// Forward declarations
typedef struct InferenceEngine InferenceEngine;
typedef struct Tokenizer Tokenizer;

// Engine management
InferenceEngine* create_engine(const char* model_path);
void free_engine(InferenceEngine* engine);

// Tokenization
Tokenizer* create_tokenizer(const char* tokenizer_path);
void free_tokenizer(Tokenizer* tokenizer);
int* tokenize(Tokenizer* tokenizer, const char* text, int* n_tokens);
char* decode(Tokenizer* tokenizer, const int* tokens, int n_tokens);

// Generation
char* generate(InferenceEngine* engine, const char* prompt, GenerationConfig* config);
int generate_next_token(InferenceEngine* engine, const int* tokens, int n_tokens, GenerationConfig* config);

// Utility functions
void print_platform_info(void);
void set_num_threads(int n_threads);
size_t get_memory_usage(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_ENGINE_H