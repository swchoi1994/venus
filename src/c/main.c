#include "inference_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -m, --model <path>     Path to model file\n");
    printf("  -p, --prompt <text>    Text prompt for generation\n");
    printf("  -t, --threads <n>      Number of threads (default: auto)\n");
    printf("  -n, --tokens <n>       Max tokens to generate (default: 128)\n");
    printf("  --temperature <f>      Sampling temperature (default: 0.7)\n");
    printf("  --top-p <f>           Top-p sampling (default: 0.9)\n");
    printf("  --top-k <n>           Top-k sampling (default: 40)\n");
    printf("  -h, --help            Show this help message\n");
    printf("  -v, --version         Show version information\n");
}

int main(int argc, char* argv[]) {
    // Default parameters
    const char* model_path = NULL;
    const char* prompt = "Once upon a time";
    int n_threads = 0;
    GenerationConfig gen_config = {
        .temperature = 0.7f,
        .top_p = 0.9f,
        .top_k = 40,
        .max_tokens = 128,
        .seed = -1,
        .repetition_penalty = 1.1f,
        .presence_penalty = 0.0f,
        .frequency_penalty = 0.0f
    };
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (i + 1 < argc) {
                model_path = argv[++i];
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 < argc) {
                prompt = argv[++i];
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                n_threads = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--tokens") == 0) {
            if (i + 1 < argc) {
                gen_config.max_tokens = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 < argc) {
                gen_config.temperature = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 < argc) {
                gen_config.top_p = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (i + 1 < argc) {
                gen_config.top_k = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0) {
            printf("Venus Inference Engine v%d.%d.%d\n", 
                   VENUS_VERSION_MAJOR, VENUS_VERSION_MINOR, VENUS_VERSION_PATCH);
            return 0;
        }
    }
    
    // Print platform information
    print_platform_info();
    printf("\n");
    
    // Check if model path is provided
    if (!model_path) {
        printf("No model specified. Running in demo mode.\n\n");
        model_path = "demo_model.bin";
    }
    
    // Set number of threads
    if (n_threads > 0) {
        set_num_threads(n_threads);
    }
    
    // Create inference engine
    printf("Loading model from: %s\n", model_path);
    InferenceEngine* engine = create_engine(model_path);
    if (!engine) {
        fprintf(stderr, "Failed to create inference engine\n");
        return 1;
    }
    
    // Generate text
    printf("\nPrompt: %s\n", prompt);
    printf("Generating...\n\n");
    
    char* generated = generate(engine, prompt, &gen_config);
    if (generated) {
        printf("%s\n", generated);
        free(generated);
    } else {
        fprintf(stderr, "Generation failed\n");
    }
    
    // Cleanup
    printf("\nMemory usage: %.2f MB\n", get_memory_usage() / (1024.0 * 1024.0));
    free_engine(engine);
    
    return 0;
}