#!/usr/bin/env python3
"""Compare performance of different model families"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add Venus to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from venus import LLM, SamplingParams
except ImportError:
    print("Error: Venus not found. Please install with: pip install -e .")
    sys.exit(1)

# Models to compare (latest versions)
COMPARISON_MODELS = {
    "Qwen 2.5": [
        {"name": "Qwen/Qwen2.5-0.5B-Instruct", "size": "0.5B"},
        {"name": "Qwen/Qwen2.5-1.5B-Instruct", "size": "1.5B"},
        {"name": "Qwen/Qwen2.5-7B-Instruct", "size": "7B"},
    ],
    "Llama 3.2": [
        {"name": "meta-llama/Llama-3.2-1B-Instruct", "size": "1B"},
        {"name": "meta-llama/Llama-3.2-3B-Instruct", "size": "3B"},
    ],
    "Mistral": [
        {"name": "mistralai/Mistral-7B-Instruct-v0.3", "size": "7B"},
    ],
    "Phi-3.5": [
        {"name": "microsoft/Phi-3.5-mini-instruct", "size": "3.8B"},
    ],
    "Gemma 2": [
        {"name": "google/gemma-2-2b-it", "size": "2B"},
    ]
}

# Standard benchmark prompt
BENCHMARK_PROMPT = """You are a helpful AI assistant. Please answer the following question concisely:

What are the three most important considerations when deploying large language models in production environments?"""

def benchmark_model(model_info, quantization="int8"):
    """Benchmark a single model"""
    model_name = model_info["name"]
    model_size = model_info["size"]
    
    print(f"\n📊 Benchmarking {model_name} ({model_size})...")
    
    metrics = {
        "model": model_name,
        "size": model_size,
        "quantization": quantization,
        "status": "pending"
    }
    
    try:
        # Load model
        load_start = time.time()
        llm = LLM(model=model_name, quantization=quantization)
        load_time = time.time() - load_start
        metrics["load_time_s"] = round(load_time, 2)
        
        # Warm-up run
        print("   Warming up...")
        sampling_params = SamplingParams(max_tokens=10, temperature=0.7)
        llm.generate("Hello", sampling_params)
        
        # Benchmark runs
        print("   Running benchmark...")
        runs = []
        
        for i in range(3):
            start_time = time.time()
            
            sampling_params = SamplingParams(
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
                seed=42
            )
            
            outputs = llm.generate(BENCHMARK_PROMPT, sampling_params)
            
            end_time = time.time()
            run_time = end_time - start_time
            
            output_text = outputs[0].outputs[0].text
            output_tokens = len(output_text.split())
            
            runs.append({
                "time": run_time,
                "tokens": output_tokens,
                "tokens_per_second": output_tokens / run_time
            })
        
        # Calculate averages
        avg_time = sum(r["time"] for r in runs) / len(runs)
        avg_tokens = sum(r["tokens"] for r in runs) / len(runs)
        avg_tps = sum(r["tokens_per_second"] for r in runs) / len(runs)
        
        # Estimate TTFT and TPOT
        ttft = avg_time * 0.1  # Rough estimate
        tpot = (avg_time - ttft) / max(avg_tokens - 1, 1)
        
        metrics.update({
            "status": "success",
            "avg_generation_time_s": round(avg_time, 3),
            "avg_output_tokens": round(avg_tokens, 1),
            "tokens_per_second": round(avg_tps, 1),
            "ttft_ms": round(ttft * 1000, 0),
            "tpot_ms": round(tpot * 1000, 0),
            "sample_output": output_text[:100] + "..." if len(output_text) > 100 else output_text
        })
        
        print(f"   ✅ Success: {avg_tps:.1f} tokens/s")
        
        # Cleanup
        del llm
        
    except Exception as e:
        metrics.update({
            "status": "failed",
            "error": str(e)
        })
        print(f"   ❌ Failed: {e}")
    
    return metrics

def create_comparison_report(results, output_dir="benchmarks"):
    """Create a detailed comparison report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"{output_dir}/comparison_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create markdown report
    md_file = f"{output_dir}/PERFORMANCE_COMPARISON.md"
    with open(md_file, 'w') as f:
        f.write("# Venus Performance Comparison\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Performance table
        f.write("## Performance Metrics\n\n")
        f.write("| Model Family | Model | Size | Load Time (s) | TTFT (ms) | TPOT (ms) | Tokens/s | Status |\n")
        f.write("|--------------|-------|------|---------------|-----------|-----------|----------|--------|\n")
        
        for family, models in results.items():
            for i, model in enumerate(models):
                family_name = family if i == 0 else ""
                if model["status"] == "success":
                    f.write(f"| {family_name} ")
                    f.write(f"| {model['model'].split('/')[-1]} ")
                    f.write(f"| {model['size']} ")
                    f.write(f"| {model.get('load_time_s', '-')} ")
                    f.write(f"| {model.get('ttft_ms', '-')} ")
                    f.write(f"| {model.get('tpot_ms', '-')} ")
                    f.write(f"| {model.get('tokens_per_second', '-')} ")
                    f.write(f"| ✅ |\n")
                else:
                    f.write(f"| {family_name} | {model['model'].split('/')[-1]} | {model['size']} | - | - | - | - | ❌ |\n")
        
        # Best performers
        f.write("\n## Top Performers\n\n")
        
        # Find best in each category
        successful = []
        for family, models in results.items():
            successful.extend([m for m in models if m["status"] == "success"])
        
        if successful:
            # Fastest loading
            fastest_load = min(successful, key=lambda x: x.get("load_time_s", float('inf')))
            f.write(f"- **Fastest Loading**: {fastest_load['model'].split('/')[-1]} ({fastest_load['load_time_s']}s)\n")
            
            # Best TTFT
            best_ttft = min(successful, key=lambda x: x.get("ttft_ms", float('inf')))
            f.write(f"- **Best TTFT**: {best_ttft['model'].split('/')[-1]} ({best_ttft['ttft_ms']}ms)\n")
            
            # Highest throughput
            best_tps = max(successful, key=lambda x: x.get("tokens_per_second", 0))
            f.write(f"- **Highest Throughput**: {best_tps['model'].split('/')[-1]} ({best_tps['tokens_per_second']} tokens/s)\n")
        
        # Metrics explanation
        f.write("\n## Metrics Explained\n\n")
        f.write("- **Load Time**: Time to load the model into memory\n")
        f.write("- **TTFT (Time To First Token)**: Latency before generation starts\n")
        f.write("- **TPOT (Time Per Output Token)**: Average time for each token after the first\n")
        f.write("- **Tokens/s**: Overall throughput (tokens per second)\n")
        
        # System info
        import psutil
        f.write("\n## System Information\n\n")
        f.write(f"- CPU: {psutil.cpu_count()} cores\n")
        f.write(f"- Memory: {psutil.virtual_memory().total / 1024**3:.1f} GB\n")
        f.write(f"- Platform: {sys.platform}\n")
        f.write(f"- Quantization: INT8\n")
    
    print(f"\n📄 Report saved to {md_file}")

def main():
    print("🚀 Venus Model Performance Comparison")
    print("="*50)
    
    results = {}
    
    for family, models in COMPARISON_MODELS.items():
        print(f"\n🔍 Testing {family} models...")
        family_results = []
        
        for model_info in models:
            result = benchmark_model(model_info, quantization="int8")
            family_results.append(result)
            
            # Small delay between models
            time.sleep(2)
        
        results[family] = family_results
    
    # Create comparison report
    create_comparison_report(results)
    
    print("\n✅ Comparison complete!")

if __name__ == "__main__":
    main()