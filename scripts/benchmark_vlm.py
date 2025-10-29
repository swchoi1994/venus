#!/usr/bin/env python3
"""
Benchmark EdgeFlow VLM inference versus a Hugging Face baseline.

Example:
    python scripts/benchmark_vlm.py \
        --edgeflow http://localhost:8000 \
        --baseline qwen/Qwen2-VL-4B-Instruct \
        --prompt "Describe the image in detail." \
        --image ./sample.jpg
"""

from __future__ import annotations

import argparse
import base64
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


try:
    from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
    import torch  # type: ignore

    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    TRANSFORMERS_AVAILABLE = False


def encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def edgeflow_payload(
    prompt: str,
    image_data: str,
    *,
    recursive: bool = False,
    max_depth: int = 3,
    beam_width: int = 1,
    max_tokens: int = 128,
    temperature: float = 0.7,
    vlm_max_side: int | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": "qwen3-vl-4b",
        "messages": [
            {
                "role": "user",
                "content": prompt,
                "attachments": [
                    {
                        "kind": "image",
                        "data": image_data,
                        "mime_type": "image/jpeg",
                    }
                ],
            }
        ],
        "stream": False,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    venus_opts: Dict[str, Any] = {}
    if vlm_max_side:
        venus_opts["max_image_side"] = int(vlm_max_side)
    if recursive:
        venus_opts["recursive_reasoning"] = {
            "enabled": True,
            "max_depth": int(max_depth),
            "beam_width": int(beam_width),
        }
    if venus_opts:
        payload["venus_options"] = venus_opts
    return payload


def run_edgeflow(
    url: str,
    prompt: str,
    image: Path,
    warmup: int,
    runs: int,
    *,
    recursive: bool = False,
    max_depth: int = 3,
    beam_width: int = 1,
    max_tokens: int = 128,
    temperature: float = 0.7,
    vlm_max_side: int | None = None,
) -> Dict[str, Any]:
    image_data = encode_image(image)
    payload = edgeflow_payload(
        prompt,
        image_data,
        recursive=recursive,
        max_depth=max_depth,
        beam_width=beam_width,
        max_tokens=max_tokens,
        temperature=temperature,
        vlm_max_side=vlm_max_side,
    )

    latencies: List[float] = []
    tokens: List[int] = []
    for idx in range(warmup + runs):
        start = time.perf_counter()
        response = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=600)
        duration = time.perf_counter() - start
        if response.status_code != 200:
            raise RuntimeError(f"EdgeFlow request failed: {response.status_code} {response.text}")

        data = response.json()
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens", 0))

        if idx >= warmup:
            latencies.append(duration)
            tokens.append(completion_tokens)

    return {
        "latency_ms": statistics.mean(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else max(latencies) * 1000,
        "tokens_per_second": (sum(tokens) / len(latencies)) / (statistics.mean(latencies) or 1.0) if tokens else 0.0,
        "raw": {"latencies": latencies, "tokens": tokens},
    }


def run_baseline(model_id: str, prompt: str, image: Path, warmup: int, runs: int, *, max_tokens: int = 128, temperature: float = 0.7) -> Optional[Dict[str, Any]]:
    if not TRANSFORMERS_AVAILABLE:
        print("transformers/torch not available; skipping baseline run.", file=sys.stderr)
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForVision2Seq.from_pretrained(model_id, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32, low_cpu_mem_usage=True).to(device)
    model.eval()

    from PIL import Image  # type: ignore

    img = Image.open(image).convert("RGB")
    latencies: List[float] = []
    tokens: List[int] = []

    for idx in range(warmup + runs):
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device, torch.float32)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=bool(temperature and temperature > 0),
                temperature=float(temperature),
                top_p=0.9,
            )
        duration = time.perf_counter() - start

        generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
        completion_tokens = len(generated_text.split())

        if idx >= warmup:
            latencies.append(duration)
            tokens.append(completion_tokens)

    return {
        "latency_ms": statistics.mean(latencies) * 1000,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[18] * 1000 if len(latencies) >= 20 else max(latencies) * 1000,
        "tokens_per_second": (sum(tokens) / len(latencies)) / (statistics.mean(latencies) or 1.0) if tokens else 0.0,
        "raw": {"latencies": latencies, "tokens": tokens},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark EdgeFlow versus baseline VLM inference.")
    parser.add_argument("--edgeflow", required=True, help="EdgeFlow server base URL (e.g., http://localhost:8000)")
    parser.add_argument("--baseline", help="Hugging Face model id for baseline comparison")
    parser.add_argument("--prompt", required=True, help="Prompt text to evaluate")
    parser.add_argument("--image", required=True, type=Path, help="Path to an input image")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup iterations per target")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed iterations per target")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max new tokens for EdgeFlow and baseline")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for both runs")
    parser.add_argument("--recursive", action="store_true", help="Enable recursive controller on EdgeFlow run")
    parser.add_argument("--rec-depth", type=int, default=3, help="Recursive max depth")
    parser.add_argument("--rec-beam", type=int, default=1, help="Recursive beam width")
    parser.add_argument("--vlm-max-side", type=int, default=None, help="Cap VLM max image side (pixels) per request")
    args = parser.parse_args()

    if not args.image.exists():
        parser.error(f"image path {args.image} does not exist")

    print("Running EdgeFlow benchmark..." + (" (recursive)" if args.recursive else ""))
    edgeflow_stats = run_edgeflow(
        args.edgeflow.rstrip("/"),
        args.prompt,
        args.image,
        args.warmup,
        args.runs,
        recursive=args.recursive,
        max_depth=args.rec_depth,
        beam_width=args.rec_beam,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        vlm_max_side=args.vlm_max_side,
    )
    print(f"EdgeFlow{'(rec)' if args.recursive else ''} latency: {edgeflow_stats['latency_ms']:.2f} ms (p95 {edgeflow_stats['latency_p95_ms']:.2f} ms)")
    print(f"EdgeFlow{'(rec)' if args.recursive else ''} tokens/sec: {edgeflow_stats['tokens_per_second']:.2f}")

    if args.baseline:
        print("\nRunning baseline benchmark...")
        baseline_stats = run_baseline(
            args.baseline,
            args.prompt,
            args.image,
            args.warmup,
            args.runs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        if baseline_stats is not None:
            print(f"Baseline  latency: {baseline_stats['latency_ms']:.2f} ms (p95 {baseline_stats['latency_p95_ms']:.2f} ms)")
            print(f"Baseline  tokens/sec: {baseline_stats['tokens_per_second']:.2f}")

            speedup = baseline_stats["latency_ms"] / edgeflow_stats["latency_ms"]
            print(f"\nEstimated latency speedup: {speedup:.2f}x")
        else:
            print("Baseline benchmark skipped.")


if __name__ == "__main__":
    main()
