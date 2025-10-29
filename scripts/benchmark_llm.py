#!/usr/bin/env python3
"""
Benchmark text-only LLM throughput via the Venus API (C engine path).

Example:
  python scripts/benchmark_llm.py \
    --url http://localhost:8000 \
    --model venus-test \
    --prompt "Explain the theory of relativity in two sentences." \
    --warmup 1 --runs 5 --max-tokens 128
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any, Dict

import requests


def run(url: str, model: str, prompt: str, warmup: int, runs: int, max_tokens: int) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": 0.0,
        "stream": False,
    }

    latencies = []
    tps = []
    for i in range(warmup + runs):
        start = time.perf_counter()
        resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=600)
        dur = time.perf_counter() - start
        if resp.status_code != 200:
            raise RuntimeError(f"API error: {resp.status_code} {resp.text}")
        data = resp.json()
        usage = data.get("usage", {})
        comp = int(usage.get("completion_tokens", 0))
        if i >= warmup:
            latencies.append(dur)
            if dur > 0:
                tps.append(comp / dur)
    return {
        "lat_ms": statistics.mean(latencies) * 1000,
        "p95_ms": (statistics.quantiles(latencies, n=20)[18] * 1000) if len(latencies) >= 20 else (max(latencies) * 1000),
        "tps": statistics.mean(tps) if tps else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark text LLM via Venus API")
    ap.add_argument("--url", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--model", default="venus-test", help="Model id to use")
    ap.add_argument("--prompt", required=True, help="Prompt to evaluate")
    ap.add_argument("--max-tokens", type=int, default=128, help="Max new tokens")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    stats = run(args.url.rstrip('/'), args.model, args.prompt, args.warmup, args.runs, args.max_tokens)
    print(f"Latency: {stats['lat_ms']:.2f} ms (p95 {stats['p95_ms']:.2f} ms)")
    print(f"Tokens/sec: {stats['tps']:.2f}")


if __name__ == "__main__":
    main()


