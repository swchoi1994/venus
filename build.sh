#!/bin/bash

# Venus Inference Engine Build Script
# Universal build with OpenCL and Vision support

set -e

echo "====================================="
echo "Venus Inference Engine Build"
echo "====================================="

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

echo "Platform: $PLATFORM"
echo "Architecture: $ARCH"
echo ""

# Parse arguments
USE_OPENCL=0
USE_VISION=0
USE_OPENMP=0
CLEAN=0

for arg in "$@"; do
    case $arg in
        --opencl)
            USE_OPENCL=1
            ;;
        --vision)
            USE_VISION=1
            ;;
        --openmp)
            USE_OPENMP=1
            ;;
        --full)
            USE_OPENCL=1
            USE_VISION=1
            USE_OPENMP=1
            ;;
        --clean)
            CLEAN=1
            ;;
        --help)
            echo "Usage: ./build.sh [options]"
            echo ""
            echo "Options:"
            echo "  --opencl    Enable OpenCL backend (Adreno, AMD, Intel, NVIDIA, CPU)"
            echo "  --vision    Enable Vision encoder support (VLM)"
            echo "  --openmp    Enable OpenMP parallelization"
            echo "  --full      Enable all features (opencl + vision + openmp)"
            echo "  --clean     Clean build artifacts before building"
            echo "  --help      Show this help message"
            exit 0
            ;;
    esac
done

# Set platform-specific flags
case "$PLATFORM-$ARCH" in
    Darwin-arm64)
        echo "🍎 Building for Apple Silicon..."
        export CFLAGS="-O3 -march=native"
        export PLATFORM_NAME="apple-silicon"
        ;;
    Linux-x86_64)
        echo "🖥️ Building for x86_64 Linux..."
        export CFLAGS="-O3 -march=native -mavx2 -mfma"
        export PLATFORM_NAME="x86-64"
        ;;
    Linux-aarch64)
        echo "🔧 Building for ARM64 Linux..."
        export CFLAGS="-O3 -march=native"
        export PLATFORM_NAME="arm64"
        ;;
    *)
        echo "⚙️ Building for generic platform..."
        export CFLAGS="-O3"
        export PLATFORM_NAME="generic"
        ;;
esac

# Check dependencies
echo "Checking dependencies..."

if ! command -v cc &> /dev/null; then
    echo "❌ C compiler not found. Please install gcc or clang."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "⚠️ Python 3 not found. Python bindings will be skipped."
    SKIP_PYTHON=1
fi

# Check for OpenCL
if [ "$USE_OPENCL" = "1" ]; then
    if [ "$PLATFORM" = "Darwin" ]; then
        echo "✅ OpenCL available via macOS framework"
    elif [ -f /usr/lib/libOpenCL.so ] || [ -f /usr/lib64/libOpenCL.so ]; then
        echo "✅ OpenCL library found"
    else
        echo "⚠️ OpenCL library not found. Install OpenCL ICD loader."
        echo "   On Ubuntu: sudo apt install ocl-icd-opencl-dev"
        echo "   On Fedora: sudo dnf install ocl-icd-devel"
    fi
fi

# Clean if requested
if [ "$CLEAN" = "1" ]; then
    echo ""
    echo "Cleaning build artifacts..."
    make clean 2>/dev/null || true
fi

# Build C library
echo ""
echo "Building C inference engine..."

MAKE_ARGS=""
[ "$USE_OPENCL" = "1" ] && MAKE_ARGS="$MAKE_ARGS USE_OPENCL=1"
[ "$USE_VISION" = "1" ] && MAKE_ARGS="$MAKE_ARGS USE_VISION=1"
[ "$USE_OPENMP" = "1" ] && MAKE_ARGS="$MAKE_ARGS USE_OPENMP=1"

make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) $MAKE_ARGS

# Build Python bindings
if [ -z "$SKIP_PYTHON" ]; then
    echo ""
    echo "Building Python bindings..."
    if [ -f requirements.txt ]; then
        pip3 install -r requirements.txt 2>/dev/null || echo "⚠️ Some Python dependencies may be missing"
    fi
    if [ -f setup.py ]; then
        python3 setup.py build_ext --inplace 2>/dev/null || echo "⚠️ Python extension build skipped"
    fi
fi

echo ""
echo "====================================="
echo "✅ Build complete!"
echo "====================================="
echo ""
echo "Features enabled:"
[ "$USE_OPENCL" = "1" ] && echo "  ✓ OpenCL backend"
[ "$USE_VISION" = "1" ] && echo "  ✓ Vision encoder"
[ "$USE_OPENMP" = "1" ] && echo "  ✓ OpenMP parallelization"
echo ""
echo "To start the API server:"
echo "  python3 src/python/api_server.py --model-dir ./models"
echo ""
echo "To download a model:"
echo "  python3 scripts/download_model.py --model Qwen/Qwen2.5-0.5B"
echo ""
echo "To convert a model:"
echo "  python3 scripts/convert_hf_model.py --input models/Qwen2.5-0.5B --output models/qwen2.5-0.5b.venus"
echo ""
echo "Environment variables:"
echo "  VENUS_OPENCL_DEVICE=adreno|amd|intel|nvidia|cpu|auto"
echo "  VENUS_USE_METAL=1      (Apple Silicon Metal acceleration)"
echo "  VENUS_FORCE_CPU=1      (Disable GPU entirely)"
