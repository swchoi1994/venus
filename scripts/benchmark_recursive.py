#!/usr/bin/env python3
"""
Benchmark recursive controller versus baseline one-shot on Venus API.

Example:
    python scripts/benchmark_recursive.py \
        --url http://localhost:8000 \
        --model venus-test \
        --prompt "Solve: 12 + 35 = ?" \
        --runs 5 --max-depth 3 --beam 1
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List

import requests


def call_api(url: str, model: str, prompt: str, recursive: bool, max_depth: int, beam: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that shows your reasoning succinctly."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    if recursive:
        payload["venus_options"] = {
            "recursive_reasoning": {
                "enabled": True,
                "max_depth": int(max_depth),
                "beam_width": int(beam),
            }
        }
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=600)
    if resp.status_code != 200:
        raise RuntimeError(f"API error: {resp.status_code} {resp.text}")
    return resp.json()


def run_bench(url: str, model: str, prompt: str, runs: int, max_depth: int, beam: int) -> None:
    # Baseline
    base_lat: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        data = call_api(url, model, prompt, recursive=False, max_depth=max_depth, beam=beam)
        base_lat.append(time.perf_counter() - start)
    # Recursive
    rec_lat: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        data = call_api(url, model, prompt, recursive=True, max_depth=max_depth, beam=beam)
        rec_lat.append(time.perf_counter() - start)

    def stats(xs: List[float]) -> str:
        mean = statistics.mean(xs)
        p95 = statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs)
        return f"{mean*1000:.2f} ms (p95 {p95*1000:.2f} ms)"

    print("Baseline (one-shot):", stats(base_lat))
    print("Recursive:", stats(rec_lat))
    print(f"Speed ratio (baseline/recursive): {statistics.mean(base_lat)/statistics.mean(rec_lat):.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark recursive controller vs baseline.")
    ap.add_argument("--url", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--model", default="venus-test", help="Model id to use")
    ap.add_argument("--prompt", required=True, help="Prompt/question to test")
    ap.add_argument("--runs", type=int, default=3, help="Number of timed runs for each mode")
    ap.add_argument("--max-depth", type=int, default=3, help="Recursion depth")
    ap.add_argument("--beam", type=int, default=1, help="Beam width")
    args = ap.parse_args()

    run_bench(args.url.rstrip('/'), args.model, args.prompt, args.runs, args.max_depth, args.beam)


if __name__ == "__main__":
    main()


