#!/usr/bin/env python3
"""Download models from Hugging Face Hub"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

def download_model(model_id: str, output_dir: str = "./models", revision: str = "main"):
    """Download a model from Hugging Face Hub"""
    
    output_path = Path(output_dir) / model_id.replace("/", "_")
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {model_id} to {output_path}")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=output_path,
            local_dir_use_symlinks=False,
            resume_download=True,
            revision=revision
        )
        
        print(f"✅ Successfully downloaded {model_id}")
        return output_path
        
    except Exception as e:
        print(f"❌ Failed to download {model_id}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Download models from Hugging Face")
    parser.add_argument("--model", type=str, required=True, help="Model ID (e.g., Qwen/Qwen2.5-0.5B)")
    parser.add_argument("--output-dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--revision", type=str, default="main", help="Model revision/branch")
    
    args = parser.parse_args()
    
    download_model(args.model, args.output_dir, args.revision)

if __name__ == "__main__":
    main()