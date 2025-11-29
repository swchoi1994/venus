# Venus Inference Engine Makefile
# Universal build with OpenCL and Vision support

CC = cc
CXX = c++
OBJC = cc
CFLAGS = -O3 -Wall -Wextra -std=c11 -fPIC
CXXFLAGS = -O3 -Wall -Wextra -std=c++17 -fPIC
OBJCFLAGS = -O3 -fPIC
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Platform-specific flags
ifeq ($(UNAME_S),Darwin)
    ifeq ($(UNAME_M),arm64)
        # Apple Silicon
        CFLAGS += -DPLATFORM_APPLE_SILICON
        LDFLAGS += -framework Accelerate -framework Foundation -framework Metal -framework MetalPerformanceShaders
        PLATFORM = apple_silicon
        # OpenCL on macOS
        OPENCL_LDFLAGS = -framework OpenCL
    else
        PLATFORM = x86_64
        OPENCL_LDFLAGS = -framework OpenCL
    endif
endif

ifeq ($(UNAME_S),Linux)
    OPENCL_LDFLAGS = -lOpenCL
endif

ifeq ($(UNAME_M),x86_64)
    ifndef PLATFORM
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
endif

ifeq ($(UNAME_M),aarch64)
    CFLAGS += -DPLATFORM_ARM64
    PLATFORM = arm64
endif

# Default to generic if no specific platform detected
PLATFORM ?= generic

# OpenMP support
ifdef USE_OPENMP
    FOPENMP_SUPPORTED := $(shell echo | $(CC) -fopenmp -E - > /dev/null 2>&1 && echo 1 || echo 0)
    ifeq ($(FOPENMP_SUPPORTED),1)
        CFLAGS += -fopenmp -DUSE_OPENMP
        LDFLAGS += -fopenmp
    else
        $(warning OpenMP requested but not supported by $(CC); building without OpenMP)
    endif
endif

# OpenCL support
ifdef USE_OPENCL
    CFLAGS += -DUSE_OPENCL
    LDFLAGS += $(OPENCL_LDFLAGS)
    OPENCL_SOURCES = src/c/platform/opencl/opencl_backend.c
else
    OPENCL_SOURCES =
endif

# Vision support
ifdef USE_VISION
    CFLAGS += -DUSE_VISION
    VISION_SOURCES = src/c/vision/vision.c src/c/vision/image_preprocess.c
else
    VISION_SOURCES =
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
            src/c/utils/json_helper.c \
            src/c/platform/platform.c \
            src/c/platform/generic.c \
            $(OPENCL_SOURCES) \
            $(VISION_SOURCES)

OBJC_SOURCES = 
ifeq ($(PLATFORM),apple_silicon)
    OBJC_SOURCES += src/c/platform/metal.m
endif

PLATFORM_SOURCE = src/c/platform/$(PLATFORM).c

OBJECTS = $(C_SOURCES:.c=.o) $(PLATFORM_SOURCE:.c=.o) $(OBJC_SOURCES:.m=.o)

# Output
LIB_NAME = libvenus
ifeq ($(UNAME_S),Darwin)
    LIB_EXT = .dylib
else
    LIB_EXT = .so
endif

# Targets
all: $(LIB_NAME)$(LIB_EXT) venus

$(LIB_NAME)$(LIB_EXT): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

venus: src/c/main.c $(LIB_NAME)$(LIB_EXT)
	$(CC) $(CFLAGS) -o $@ $< -L. -lvenus $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.m
	$(OBJC) $(OBJCFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJECTS) $(LIB_NAME)$(LIB_EXT) $(LIB_NAME).so $(LIB_NAME).dylib venus
	find src/c -name "*.o" -delete

test: all
	$(CC) $(CFLAGS) -o test_runner tests/c/*.c -L. -lvenus $(LDFLAGS)
	./test_runner

# Build with all features
full: 
	$(MAKE) USE_OPENCL=1 USE_VISION=1 USE_OPENMP=1 all

# Print configuration
info:
	@echo "Platform: $(PLATFORM)"
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "OpenCL: $(if $(USE_OPENCL),enabled,disabled)"
	@echo "Vision: $(if $(USE_VISION),enabled,disabled)"
	@echo "OpenMP: $(if $(USE_OPENMP),enabled,disabled)"

.PHONY: all clean test full info
