#!/bin/bash

# Venus Benchmark Runner
# Run comprehensive benchmarks on various models

set -e

echo "🚀 Venus Benchmark Suite"
echo "======================="
echo ""

# Check if Venus is installed
if ! python3 -c "import venus" 2>/dev/null; then
    echo "⚠️  Venus not installed. Installing..."
    pip install -e .
fi

# Create benchmarks directory
mkdir -p benchmarks
mkdir -p models

# Function to download and benchmark a model
benchmark_model() {
    local model_name=$1
    local model_size=$2
    
    echo "📊 Benchmarking $model_name ($model_size)..."
    
    # Download model if needed
    if [ ! -d "models/${model_name//\//_}" ]; then
        echo "   📥 Downloading model..."
        python3 scripts/download_model.py --model "$model_name" || true
    fi
    
    # Convert to Venus format
    if [ ! -f "models/${model_name//\//_}.venus" ]; then
        echo "   🔄 Converting to Venus format..."
        python3 scripts/convert_hf_model.py \
            --input "models/${model_name//\//_}" \
            --output "models/${model_name//\//_}.venus" \
            --quantization q8_0 || true
    fi
    
    echo ""
}

# Quick benchmark (small models only)
quick_benchmark() {
    echo "🏃 Running quick benchmark (small models)..."
    echo ""
    
    python3 scripts/compare_models.py
}

# Full benchmark suite
full_benchmark() {
    echo "🏃 Running full benchmark suite..."
    echo ""
    
    # Test each model family
    for family in qwen2.5 llama3 mistral phi gemma; do
        echo "Testing $family family..."
        python3 scripts/benchmark_models.py --family $family --quantization int8
        sleep 5  # Pause between families
    done
}

# Qwen2.5-Omni specific test
test_qwen_omni() {
    echo "🔬 Testing Qwen2.5-Omni-7B specifically..."
    
    # Download if needed
    benchmark_model "Qwen/Qwen2.5-7B-Instruct" "7B"
    
    # Run specific test
    python3 scripts/test_qwen_omni.py
}

# Parse command line arguments
case "${1:-quick}" in
    quick)
        quick_benchmark
        ;;
    full)
        full_benchmark
        ;;
    qwen)
        test_qwen_omni
        ;;
    *)
        echo "Usage: $0 [quick|full|qwen]"
        echo "  quick - Run quick benchmark with small models"
        echo "  full  - Run full benchmark suite" 
        echo "  qwen  - Test Qwen2.5-Omni-7B specifically"
        exit 1
        ;;
esac

echo ""
echo "✅ Benchmarks complete!"
echo "📄 Results saved in benchmarks/"
echo ""

# Generate summary
if [ -f "benchmarks/PERFORMANCE_COMPARISON.md" ]; then
    echo "📊 Performance Summary:"
    echo "====================="
    grep -A 5 "## Top Performers" benchmarks/PERFORMANCE_COMPARISON.md || true
fi