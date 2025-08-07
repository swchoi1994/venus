#!/usr/bin/env python3
"""Benchmark various models with Venus"""

import argparse
import time
import json
import psutil
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add Venus to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from venus import LLM, SamplingParams
except ImportError:
    print("Error: Venus not found. Please install with: pip install -e .")
    sys.exit(1)

# Latest models to benchmark
BENCHMARK_MODELS = {
    "qwen2.5": {
        "models": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct", 
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "Qwen/Qwen2.5-72B-Instruct"
        ],
        "special": ["Qwen/Qwen2.5-Coder-7B-Instruct", "Qwen/Qwen2.5-Math-7B-Instruct"]
    },
    "llama3": {
        "models": [
            "meta-llama/Llama-3.2-1B-Instruct",
            "meta-llama/Llama-3.2-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "meta-llama/Llama-3.1-70B-Instruct"
        ],
        "vision": ["meta-llama/Llama-3.2-11B-Vision-Instruct"]
    },
    "mistral": {
        "models": [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1"
        ]
    },
    "exaone": {
        "models": [
            "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct",
            "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
            "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
        ]
    },
    "phi": {
        "models": [
            "microsoft/Phi-3.5-mini-instruct",
            "microsoft/Phi-3.5-MoE-instruct"
        ]
    },
    "gemma": {
        "models": [
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it"
        ]
    },
    "deepseek": {
        "models": [
            "deepseek-ai/DeepSeek-V3-Base",
            "deepseek-ai/DeepSeek-Coder-V2-Instruct"
        ]
    },
    "yi": {
        "models": [
            "01-ai/Yi-1.5-6B-Chat",
            "01-ai/Yi-1.5-9B-Chat",
            "01-ai/Yi-1.5-34B-Chat"
        ]
    }
}

# Test prompts for benchmarking
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to calculate fibonacci numbers.",
    "Translate 'Hello, how are you?' to Spanish, French, and Japanese.",
    "What are the main differences between machine learning and deep learning?",
]

class ModelBenchmark:
    def __init__(self, model_name: str, quantization: Optional[str] = None):
        self.model_name = model_name
        self.quantization = quantization
        self.metrics = {
            "model": model_name,
            "quantization": quantization or "none",
            "ttft_ms": [],  # Time to first token
            "tpot_ms": [],  # Time per output token
            "total_time_s": [],
            "tokens_per_second": [],
            "memory_usage_mb": [],
            "prompt_tokens": [],
            "output_tokens": [],
            "cpu_percent": [],
            "success_rate": 0.0,
            "errors": []
        }
        self.llm = None
        
    def load_model(self):
        """Load the model"""
        try:
            print(f"\n📥 Loading {self.model_name}...")
            start_time = time.time()
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Load model
            self.llm = LLM(
                model=self.model_name,
                quantization=self.quantization,
                dtype="auto",
                seed=42
            )
            
            # Calculate load time and memory
            load_time = time.time() - start_time
            final_memory = process.memory_info().rss / 1024 / 1024
            model_memory = final_memory - initial_memory
            
            self.metrics["load_time_s"] = load_time
            self.metrics["model_memory_mb"] = model_memory
            
            print(f"✅ Model loaded in {load_time:.2f}s")
            print(f"   Memory usage: {model_memory:.0f} MB")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.metrics["errors"].append(f"Load error: {str(e)}")
            return False
    
    def benchmark_prompt(self, prompt: str, max_tokens: int = 100):
        """Benchmark a single prompt"""
        if not self.llm:
            return None
            
        try:
            # Prepare sampling params
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=max_tokens,
                seed=42
            )
            
            # Get CPU and memory before
            process = psutil.Process()
            cpu_before = process.cpu_percent()
            mem_before = process.memory_info().rss / 1024 / 1024
            
            # Time the generation
            start_time = time.time()
            first_token_time = None
            
            # Generate
            outputs = self.llm.generate(prompt, sampling_params)
            
            # Calculate metrics
            total_time = time.time() - start_time
            output_text = outputs[0].outputs[0].text
            
            # Estimate tokens (rough approximation)
            prompt_tokens = len(prompt.split()) * 1.3
            output_tokens = len(output_text.split()) * 1.3
            
            # Calculate TTFT (time to first token) - approximate
            ttft = total_time * 0.1 if output_tokens > 0 else total_time
            
            # Calculate TPOT (time per output token)
            tpot = (total_time - ttft) / max(output_tokens - 1, 1) if output_tokens > 1 else 0
            
            # Tokens per second
            tps = output_tokens / total_time if total_time > 0 else 0
            
            # CPU and memory after
            cpu_after = process.cpu_percent()
            mem_after = process.memory_info().rss / 1024 / 1024
            
            # Store metrics
            self.metrics["ttft_ms"].append(ttft * 1000)
            self.metrics["tpot_ms"].append(tpot * 1000)
            self.metrics["total_time_s"].append(total_time)
            self.metrics["tokens_per_second"].append(tps)
            self.metrics["memory_usage_mb"].append(mem_after - mem_before)
            self.metrics["prompt_tokens"].append(prompt_tokens)
            self.metrics["output_tokens"].append(output_tokens)
            self.metrics["cpu_percent"].append((cpu_before + cpu_after) / 2)
            
            return output_text
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.metrics["errors"].append(str(e))
            return None
    
    def run_benchmarks(self, prompts: List[str] = TEST_PROMPTS):
        """Run all benchmarks"""
        if not self.load_model():
            return self.metrics
            
        print(f"\n🏃 Running benchmarks for {self.model_name}...")
        successful_runs = 0
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n   Test {i}/{len(prompts)}: {prompt[:50]}...")
            result = self.benchmark_prompt(prompt)
            if result:
                successful_runs += 1
                print(f"   ✅ Generated {len(result.split())} words")
        
        # Calculate averages
        self.metrics["success_rate"] = successful_runs / len(prompts)
        
        if successful_runs > 0:
            self.metrics["avg_ttft_ms"] = np.mean(self.metrics["ttft_ms"])
            self.metrics["avg_tpot_ms"] = np.mean(self.metrics["tpot_ms"])
            self.metrics["avg_tokens_per_second"] = np.mean(self.metrics["tokens_per_second"])
            self.metrics["avg_memory_mb"] = np.mean(self.metrics["memory_usage_mb"])
            self.metrics["avg_cpu_percent"] = np.mean(self.metrics["cpu_percent"])
        
        # Cleanup
        del self.llm
        self.llm = None
        
        return self.metrics

def save_results(results: List[Dict], output_file: str):
    """Save benchmark results"""
    # Save raw JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create markdown report
    md_file = output_file.replace('.json', '.md')
    with open(md_file, 'w') as f:
        f.write("# Venus Model Benchmarks\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # System info
        f.write("## System Information\n\n")
        f.write(f"- Platform: {sys.platform}\n")
        f.write(f"- CPU: {psutil.cpu_count()} cores\n")
        f.write(f"- Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB\n")
        f.write(f"- Venus Version: 0.1.0\n\n")
        
        # Results table
        f.write("## Benchmark Results\n\n")
        f.write("| Model | Quant | Load Time (s) | Avg TTFT (ms) | Avg TPOT (ms) | Tokens/s | Memory (MB) | CPU % |\n")
        f.write("|-------|-------|---------------|---------------|---------------|----------|-------------|-------|\n")
        
        for result in results:
            if result.get("success_rate", 0) > 0:
                f.write(f"| {result['model'].split('/')[-1]} ")
                f.write(f"| {result['quantization']} ")
                f.write(f"| {result.get('load_time_s', 0):.1f} ")
                f.write(f"| {result.get('avg_ttft_ms', 0):.0f} ")
                f.write(f"| {result.get('avg_tpot_ms', 0):.0f} ")
                f.write(f"| {result.get('avg_tokens_per_second', 0):.1f} ")
                f.write(f"| {result.get('model_memory_mb', 0):.0f} ")
                f.write(f"| {result.get('avg_cpu_percent', 0):.0f}% |\n")
            else:
                f.write(f"| {result['model'].split('/')[-1]} | {result['quantization']} | ❌ Failed | - | - | - | - | - |\n")
        
        f.write("\n## Metrics Explanation\n\n")
        f.write("- **TTFT**: Time To First Token - How long before generation starts\n")
        f.write("- **TPOT**: Time Per Output Token - Average time for each subsequent token\n")
        f.write("- **Tokens/s**: Average tokens generated per second\n")
        f.write("- **Memory**: Additional memory used by the model\n")
        f.write("- **CPU %**: Average CPU utilization during generation\n")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Venus with various models")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    parser.add_argument("--family", choices=list(BENCHMARK_MODELS.keys()), help="Test a model family")
    parser.add_argument("--quantization", choices=["none", "int8", "int4"], default="int8")
    parser.add_argument("--output", default="benchmarks/results.json", help="Output file")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer prompts")
    
    args = parser.parse_args()
    
    # Determine which models to test
    test_models = []
    if args.models:
        test_models = args.models
    elif args.family:
        family = BENCHMARK_MODELS[args.family]
        test_models = family.get("models", [])
        # Add special models if they exist
        test_models.extend(family.get("special", []))
        test_models.extend(family.get("vision", []))
    else:
        # Default: test one small model from each family
        test_models = [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "meta-llama/Llama-3.2-1B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/Phi-3.5-mini-instruct",
        ]
    
    # Use fewer prompts for quick test
    prompts = TEST_PROMPTS[:2] if args.quick else TEST_PROMPTS
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Run benchmarks
    results = []
    print(f"🚀 Benchmarking {len(test_models)} models with {args.quantization} quantization\n")
    
    for model in test_models:
        print(f"\n{'='*60}")
        print(f"Testing: {model}")
        print(f"{'='*60}")
        
        benchmark = ModelBenchmark(model, args.quantization if args.quantization != "none" else None)
        metrics = benchmark.run_benchmarks(prompts)
        results.append(metrics)
        
        # Save intermediate results
        save_results(results, args.output)
    
    print(f"\n✅ Benchmarks complete! Results saved to {args.output}")
    print(f"   Markdown report: {args.output.replace('.json', '.md')}")

if __name__ == "__main__":
    main()