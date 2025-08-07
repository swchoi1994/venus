# Venus - Universal Inference Engine

A cross-platform, high-performance inference engine for Large Language Models (LLMs) with OpenAI-compatible API.

## Features

- 🚀 **Universal Platform Support**: Runs on Apple Silicon, x86_64, ARM, RISC-V, and more
- 🔥 **High Performance**: Optimized with platform-specific SIMD instructions
- 📦 **Memory Efficient**: Advanced quantization (Q8_0, Q4_0) and PagedAttention
- 🌐 **OpenAI Compatible**: Drop-in replacement for OpenAI API
- 🛠️ **Production Ready**: Docker, Kubernetes, and auto-scaling support

## Quick Start

```bash
# Clone the repository
git clone https://github.com/swchoi1994/venus.git
cd venus

# Build the engine
./build.sh

# Download and convert a model
python scripts/download_model.py --model Qwen/Qwen2.5-0.5B
python scripts/convert_hf_model.py --input models/Qwen2.5-0.5B --output models/qwen2.5-0.5b.bin

# Start the API server
./target/release/venus --model-dir ./models --port 8000
```

## Minimum Requirements

- **CPU**: 8+ cores (Apple M1 equivalent)
- **RAM**: 32GB minimum
- **Storage**: 100GB for models

## Documentation

See [docs/](docs/) for detailed documentation.

## License

MIT License - see LICENSE file for details.