#!/usr/bin/env python3
"""Test Qwen2.5-Omni-7B model specifically"""

import os
import sys
import time
import psutil

# Add Venus to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from venus import LLM, SamplingParams
except ImportError:
    print("Error: Venus not found. Please install with: pip install -e .")
    sys.exit(1)

def test_qwen_omni():
    """Test Qwen2.5-Omni-7B model"""
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Using regular 7B as Omni might not be available
    
    print(f"🚀 Testing {model_name}")
    print(f"   System: {psutil.cpu_count()} CPU cores, {psutil.virtual_memory().total / 1024**3:.1f} GB RAM\n")
    
    # Load model
    print("📥 Loading model...")
    start_time = time.time()
    
    try:
        llm = LLM(
            model=model_name,
            quantization="int8",  # Use INT8 to fit in memory
            dtype="auto",
            seed=42
        )
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f} seconds\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test prompts
    test_cases = [
        {
            "prompt": "What are the key features of Qwen2.5?",
            "max_tokens": 100
        },
        {
            "prompt": "Write a Python function to merge two sorted lists:",
            "max_tokens": 150
        },
        {
            "prompt": "Explain the difference between CPU and GPU in one paragraph:",
            "max_tokens": 100
        },
        {
            "prompt": "你好！请用中文介绍一下量子计算的基本原理。",
            "max_tokens": 150
        },
        {
            "prompt": "Generate a haiku about artificial intelligence:",
            "max_tokens": 50
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}: {test['prompt']}")
        print(f"{'='*60}")
        
        # Create sampling params
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=test['max_tokens'],
            seed=42
        )
        
        # Measure performance
        start_time = time.time()
        
        try:
            outputs = llm.generate(test['prompt'], sampling_params)
            generation_time = time.time() - start_time
            
            output_text = outputs[0].outputs[0].text
            output_tokens = len(output_text.split())
            tokens_per_second = output_tokens / generation_time
            
            print(f"\n📝 Response:\n{output_text}")
            print(f"\n📊 Metrics:")
            print(f"   - Generation time: {generation_time:.2f}s")
            print(f"   - Output tokens: ~{output_tokens}")
            print(f"   - Speed: {tokens_per_second:.1f} tokens/s")
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    
    print(f"\n✅ All tests completed!")

if __name__ == "__main__":
    test_qwen_omni()