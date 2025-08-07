# Venus Inference Engine Makefile

CC = cc
CXX = c++
CFLAGS = -O3 -Wall -Wextra -std=c11 -fPIC
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -fPIC
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform-specific flags
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon
        CFLAGS += -DPLATFORM_APPLE_SILICON -framework Accelerate
        LDFLAGS += -framework Accelerate -framework Metal
        PLATFORM = apple_silicon
    endif
endif

ifeq ($(UNAME_M),x86_64)
    CFLAGS += -DPLATFORM_X86_64 -mavx2 -mfma
    SIMD_FLAGS = -mavx2 -mfma
    PLATFORM = x86_64
    
    # Check for AVX-512
    AVX512_SUPPORTED := $(shell echo | $(CC) -mavx512f -E - > /dev/null 2>&1 && echo 1 || echo 0)
    ifeq ($(AVX512_SUPPORTED),1)
        CFLAGS += -DHAS_AVX512 -mavx512f
        SIMD_FLAGS += -mavx512f
    endif
endif

ifeq ($(UNAME_M),aarch64)
    CFLAGS += -DPLATFORM_ARM64
    PLATFORM = arm64
endif

# Default to generic if no specific platform detected
PLATFORM ?= generic

# OpenMP support
ifdef USE_OPENMP
    CFLAGS += -fopenmp -DUSE_OPENMP
    LDFLAGS += -fopenmp
endif

# Source files
C_SOURCES = src/c/inference_engine.c \
            src/c/tensor.c \
            src/c/attention.c \
            src/c/flash_attention.c \
            src/c/paged_attention.c \
            src/c/quantization.c \
            src/c/model_loader.c \
            src/c/utils/memory_pool.c \
            src/c/utils/tokenizer.c \
            src/c/utils/sampler.c \
            src/c/platform/platform.c \
            src/c/platform/generic.c

PLATFORM_SOURCE = src/c/platform/$(PLATFORM).c

OBJECTS = $(C_SOURCES:.c=.o) $(PLATFORM_SOURCE:.c=.o)

# Targets
all: libvenus.so venus

libvenus.so: $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

venus: src/c/main.c libvenus.so
	$(CC) $(CFLAGS) -o $@ $< -L. -lvenus $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) libvenus.so venus

test: all
	$(CC) $(CFLAGS) -o test_runner tests/c/*.c -L. -lvenus $(LDFLAGS)
	./test_runner

.PHONY: all clean test