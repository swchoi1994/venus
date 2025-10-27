# EdgeFlow Qwen3-VL-4B Launch Checklist

- [x] Prepare deployment manifest entry for Qwen3-VL-4B (quantized)
- [x] Confirm ModelRun API handles multimodal attachments and model fallback
- [x] Document runbook for FastAPI-based benchmarking
- [x] Provide benchmarking harness to compare EdgeFlow versus baseline
- [ ] Execute benchmark with real checkpoints and capture results _(requires model download & hardware run)_

## Runbook Overview

1. **Prepare artifacts**  
   - Convert `Qwen/Qwen2-VL-4B-Instruct` (or newer) to EdgeFlow format using Modulus CLI or `scripts/convert_hf_model.py --quantization q4_0`.  
   - Place the converted `.bin` and tokenizer assets under `models/qwen3-vl-4b/` and update `deployment.json` (see `configs/deployment.example.json`).
2. **Start EdgeFlow stack**  
   ```bash
   cargo run --release --bin venus -- --model-dir ./models --port 8000
   ```  
   or use the Python FastAPI server:  
   ```bash
   python -m src.python.api_server --model-dir ./models --port 8000
   ```
3. **Run benchmark script**  
   ```bash
   python scripts/benchmark_vlm.py \
     --edgeflow http://localhost:8000 \
     --baseline qwen/Qwen2-VL-4B-Instruct \
     --prompt "Describe the image in detail." \
     --image ./sample.jpg
   ```
4. **Review output**  
   - The script reports latency and approximate tokens/sec for EdgeFlow and the baseline Hugging Face pipeline.
   - Append findings to this file and tick the final checkbox once results are captured.

## Notes

- Baseline inference uses Hugging Face `transformers` on the local machine; install extras with `pip install -r requirements.txt transformers optimum`.  
- For Apple Silicon, ensure dependencies are built with Metal support when available.
- If the model weights are gated, authenticate with `huggingface-cli login` before running the benchmark script.
