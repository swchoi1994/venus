#ifndef ARCHITECTURE_H
#define ARCHITECTURE_H

#include "inference_engine.h"

// Architecture detection from model metadata
int detect_architecture_from_metadata(const char* model_type, const char* architectures);

// Architecture-specific configurations
void configure_architecture(ModelConfig* config, int architecture);

// Memory requirements calculation
size_t calculate_memory_requirements(ModelConfig* config, bool quantized);

// Maximum supported parameters for different memory sizes
size_t get_max_parameters_for_memory(size_t available_memory, bool quantized);

#endif // ARCHITECTURE_H