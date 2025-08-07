#!/bin/bash

# Venus Inference Engine Build Script

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

# Set platform-specific flags
case "$PLATFORM-$ARCH" in
    Darwin-arm64)
        echo "🍎 Building for Apple Silicon..."
        export CFLAGS="-O3 -march=native -framework Accelerate"
        export RUSTFLAGS="-C target-cpu=native"
        export PLATFORM_NAME="apple-silicon"
        ;;
    Linux-x86_64)
        echo "🖥️ Building for x86_64 Linux..."
        export CFLAGS="-O3 -march=native -mavx2 -mfma"
        export RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma"
        export PLATFORM_NAME="x86-64"
        ;;
    Linux-aarch64)
        echo "🔧 Building for ARM64 Linux..."
        export CFLAGS="-O3 -march=native"
        export RUSTFLAGS="-C target-cpu=native"
        export PLATFORM_NAME="arm64"
        ;;
    *)
        echo "⚙️ Building for generic platform..."
        export CFLAGS="-O3"
        export RUSTFLAGS="-C opt-level=3"
        export PLATFORM_NAME="generic"
        ;;
esac

# Check dependencies
echo "Checking dependencies..."

if ! command -v cc &> /dev/null; then
    echo "❌ C compiler not found. Please install gcc or clang."
    exit 1
fi

if ! command -v cargo &> /dev/null; then
    echo "❌ Rust not found. Please install from https://rustup.rs"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "⚠️ Python 3 not found. Python bindings will be skipped."
    SKIP_PYTHON=1
fi

# Build C library
echo ""
echo "Building C inference engine..."
make clean
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) USE_OPENMP=1

# Build Rust wrapper and API server
echo ""
echo "Building Rust API server..."
cargo build --release

# Build Python bindings
if [ -z "$SKIP_PYTHON" ]; then
    echo ""
    echo "Building Python bindings..."
    pip3 install -r requirements.txt
    python3 setup.py build_ext --inplace
fi

echo ""
echo "====================================="
echo "✅ Build complete!"
echo "====================================="
echo ""
echo "To start the server:"
echo "  ./target/release/venus --model-dir ./models"
echo ""
echo "To download a model:"
echo "  python3 scripts/download_model.py --model Qwen/Qwen2.5-0.5B"
echo ""
echo "To convert a model:"
echo "  python3 scripts/convert_hf_model.py --input models/Qwen2.5-0.5B --output models/qwen2.5-0.5b.bin"