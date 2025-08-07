#!/usr/bin/env python3
"""Test script for Venus API"""

import requests
import json
import sys

def test_api(base_url="http://localhost:8000"):
    """Test the Venus API endpoints"""
    
    print(f"Testing Venus API at {base_url}\n")
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test models endpoint
    print("\n2. Testing models endpoint...")
    try:
        response = requests.get(f"{base_url}/v1/models")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test chat completion
    print("\n3. Testing chat completion...")
    try:
        payload = {
            "model": "demo-model",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello! Can you tell me a short joke?"}
            ],
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Model: {result['model']}")
            print(f"   Response: {result['choices'][0]['message']['content']}")
            print(f"   Usage: {result['usage']}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test streaming
    print("\n4. Testing streaming chat completion...")
    try:
        payload["stream"] = True
        
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   Streaming response:")
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        data = line_text[6:]
                        if data != "[DONE]":
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    print(delta["content"], end="", flush=True)
            print("\n   ✅ Streaming completed")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ All tests completed!")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Venus API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    success = test_api(args.url)
    sys.exit(0 if success else 1)