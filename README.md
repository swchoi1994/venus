# Venus - Universal AI Inference Engine

A cross-platform, high-performance inference engine for Large Language Models (LLMs) and Vision Language Models (VLMs). Venus uses OpenCL as a universal compute backend, enabling AI inference on virtually any hardware.

## Features

- **Universal Hardware Support**: Single codebase runs on Qualcomm Adreno, AMD, Intel, NVIDIA, ARM Mali, and CPU
- **VLM Native**: Built-in support for vision-language models (Qwen2-VL, LLaVA, CLIP, SigLIP)
- **High Performance**: Auto-tuned kernels optimized per device family
- **Memory Efficient**: Q8_0 and Q4_0 quantization support
- **OpenAI Compatible**: Drop-in replacement API server
- **Open Source**: Apache 2.0 license for commercial use

## Supported Hardware

| Hardware | Backend | Status |
|----------|---------|--------|
| Apple Silicon (M1/M2/M3) | Metal + Accelerate | ✅ Optimized |
| Qualcomm Adreno | OpenCL | ✅ Optimized |
| AMD RDNA/GCN | OpenCL | ✅ Supported |
| Intel Arc/Xe/UHD | OpenCL | ✅ Supported |
| NVIDIA GPUs | OpenCL | ✅ Supported |
| ARM Mali | OpenCL | ⚠️ Basic |
| x86_64 CPU | AVX2/AVX-512 | ✅ Optimized |
| ARM64 CPU | NEON | ✅ Supported |

## Quick Start

### Build

```bash
# Clone the repository
git clone https://github.com/edgeflow-ai/venus.git
cd venus

# Build with all features
./build.sh --full

# Or build with specific features
./build.sh --opencl --vision
```

### Build Options

```bash
./build.sh [options]

Options:
  --opencl    Enable OpenCL backend (Adreno, AMD, Intel, NVIDIA, CPU)
  --vision    Enable Vision encoder support (VLM)
  --openmp    Enable OpenMP parallelization
  --full      Enable all features
  --clean     Clean before building
```

### Download and Convert a Model

```bash
# Download a model from HuggingFace
python scripts/download_model.py --model Qwen/Qwen2.5-0.5B

# Convert to Venus format
python scripts/convert_hf_model.py \
    --input models/Qwen_Qwen2.5-0.5B \
    --output models/qwen2.5-0.5b.venus \
    --quantization q8_0
```

### Start the API Server

```bash
# Start OpenAI-compatible API server
python src/python/api_server.py --model-dir ./models --port 8000

# Test with curl
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## VLM Setup (Vision Language Models)

For VLM models like Qwen2-VL or Qwen3-VL, you need to set up a `deployment.json` in your models directory:

### 1. Download VLM Model

```bash
# Download Qwen3-VL-8B (or any VLM)
python scripts/download_model.py --model Qwen/Qwen2-VL-7B-Instruct --output-dir ./models
```

### 2. Create deployment.json

Create `models/deployment.json`:

```json
{
  "default_model": "qwen2-vl-7b",
  "models": {
    "qwen2-vl-7b": {
      "model_kind": "vlm",
      "hf_model_dir": "Qwen_Qwen2-VL-7B-Instruct"
    }
  }
}
```

### 3. Start Server with VLM

```bash
# For optimal VLM performance (~4 seconds for 8B model):
VLM_MAX_IMAGE_SIDE=640 python src/python/api_server.py --model-dir ./models --port 8000
```

### 4. Test VLM with Gradio UI

```bash
python scripts/gradio_vlm_chat.py --api http://localhost:8000 --share
```

### VLM Performance Tips

For ~4 second inference on Qwen3-VL-8B:
- Set `VLM_MAX_IMAGE_SIDE=640` (default) to resize large images
- Use GPU with Flash Attention 2 when available
- Enable `torch.compile` on CUDA: `VENUS_TORCH_COMPILE=1`
- Use FP16 precision (automatic on GPU/MPS)

## Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `VENUS_OPENCL_DEVICE` | `adreno`, `amd`, `intel`, `nvidia`, `cpu`, `auto` | Select OpenCL device |
| `VENUS_USE_OPENCL` | `1` | Enable OpenCL backend |
| `VENUS_USE_METAL` | `1` | Enable Metal backend (Apple Silicon) |
| `VENUS_FORCE_CPU` | `1` | Disable all GPU acceleration |

## Project Structure

```
venus/
├── src/c/                      # C inference engine
│   ├── inference_engine.c/h    # Main engine
│   ├── tensor.c/h              # Tensor operations
│   ├── quantization.c/h        # Quantization support
│   ├── attention.c/h           # Attention mechanisms
│   ├── platform/               # Platform backends
│   │   ├── opencl/             # Universal OpenCL backend
│   │   ├── apple_silicon.c     # Apple Accelerate
│   │   └── metal.m             # Metal GPU
│   └── vision/                 # Vision encoder
│       ├── vision.c/h          # ViT implementation
│       └── image_preprocess.c  # Image processing
├── src/python/                 # Python API server
├── scripts/                    # Tools and utilities
│   ├── convert_hf_model.py     # HuggingFace converter
│   ├── download_model.py       # Model downloader
│   └── benchmark_*.py          # Benchmarking tools
├── models/                     # Model storage
├── build.sh                    # Build script
├── Makefile                    # Build system
└── ROADMAP.md                  # Development roadmap
```

## Minimum Requirements

- **CPU**: 4+ cores
- **RAM**: 8GB minimum (16GB+ recommended for larger models)
- **Storage**: Varies by model size
- **OS**: macOS, Linux, Windows (WSL2)

### Build Dependencies

- C compiler (gcc/clang)
- Python 3.8+
- OpenCL SDK (optional, for GPU acceleration)

**macOS:**
```bash
xcode-select --install
```

**Ubuntu/Debian:**
```bash
sudo apt install build-essential python3-dev ocl-icd-opencl-dev
```

**Fedora:**
```bash
sudo dnf install gcc python3-devel ocl-icd-devel
```

## Supported Models

### Text Models
- Qwen 2.5 (0.5B - 72B)
- Llama 3.x (1B - 70B)
- Mistral/Mixtral
- Phi-2/Phi-3
- Gemma 2

### Vision-Language Models
- Qwen2-VL / Qwen3-VL
- LLaVA
- Pixtral

## API Compatibility

Venus provides an OpenAI-compatible API:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="qwen2.5-0.5b",
    messages=[{"role": "user", "content": "Explain quantum computing"}],
    max_tokens=512
)
print(response.choices[0].message.content)
```

## Benchmarks

Performance varies by hardware. See [benchmarks/](benchmarks/) for detailed comparisons.

| Model | Hardware | Tokens/sec |
|-------|----------|------------|
| Qwen2.5-0.5B | M2 Pro | ~120 |
| Qwen2.5-0.5B | Snapdragon 8 Gen 2 | ~45 |
| Llama-3.2-3B | M2 Pro | ~35 |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Priority areas:
1. Kernel optimization for specific hardware
2. Model converter implementations  
3. Benchmarking on different devices
4. Documentation and examples

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the development roadmap.

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by llama.cpp, vLLM, and ONNX Runtime
- Built by [EdgeFlow AI](https://edgeflow.ai)

---

*Venus - AI inference for everyone, everywhere.*
