#!/usr/bin/env python3
"""Convert Hugging Face models to Venus format"""

import argparse
import struct
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import save_file, load_file
import sys

def quantize_tensor(tensor, quantization="q8_0"):
    """Quantize a tensor using specified method"""
    
    if quantization == "q8_0":
        # INT8 quantization with scale
        scale = tensor.abs().max() / 127.0
        if scale == 0:
            scale = 1.0
        quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        return quantized, scale
    
    elif quantization == "q4_0":
        # INT4 quantization
        scale = tensor.abs().max() / 7.0
        if scale == 0:
            scale = 1.0
        quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
        return quantized, scale
    
    else:
        # No quantization
        return tensor, 1.0

def detect_architecture(config):
    """Detect model architecture from config"""
    model_type = config.model_type.lower()
    architectures = getattr(config, 'architectures', [])
    arch_str = ','.join(architectures) if architectures else ''
    
    # Check for vision models first
    if 'LlamaForConditionalGeneration' in arch_str and 'vision' in model_type:
        return "llama_vision"
    elif 'Qwen2VLForConditionalGeneration' in arch_str:
        return "qwen2_vl"
    elif 'VoxtralForConditionalGeneration' in arch_str:
        return "voxtral"
    
    # Map model types to architectures
    arch_map = {
        "llama": "llama",
        "qwen2": "qwen",
        "qwen": "qwen",
        "mistral": "mistral",
        "mixtral": "mixtral",
        "falcon": "falcon",
        "bloom": "bloom",
        "opt": "opt",
        "codegen": "codegen",
        "gpt_bigcode": "gptbigcode",
        "starcoder": "gptbigcode",
        "baichuan": "baichuan",
        "chatglm": "chatglm",
        "persimmon": "persimmon",
        "mamba": "mamba",
        "deepseek": "deepseek",
        "phi3": "phi3",
        "phi": "phi",
        "gemma2": "gemma2",
        "gemma": "gemma",
        "stablelm": "stablelm",
        "starcoder2": "starcoder2",
        "exaone": "exaone",
        "minicpm3": "minicpm3",
        "minicpm": "minicpm",
        "dbrx": "dbrx",
        "olmo": "olmo",
        "arctic": "arctic",
        "xverse": "xverse",
        "command-r": "command_r",
        "cohere": "command_r",
        "deci": "deci",
        "bamba": "bamba",
        "nemotron": "nemotron",
        "plm0": "plm0",
        "solar": "solar",
        "granite": "granite",
        "gpt2": "gpt",
        "gptj": "gpt",
        "gpt-j": "gpt",
        "gpt_neox": "gpt",
        "gpt-neox": "gpt",
        "bert": "bert",
        "t5": "t5",
    }
    
    # Check for granite MoE
    if "granite" in model_type and "moe" in model_type:
        return "granite_moe"
    
    # Check each known architecture
    for key, arch in arch_map.items():
        if key in model_type:
            return arch
    
    # Check architecture class names
    if architectures:
        arch_class = architectures[0].lower()
        for key, arch in arch_map.items():
            if key in arch_class:
                return arch
    
    return "unknown"

def save_venus_model(model, tokenizer, output_path, quantization="q8_0"):
    """Save model in Venus format"""
    
    config = model.config
    
    # Create header with model metadata
    header = {
        "version": 1,
        "format": "venus",
        "architecture": detect_architecture(config),
        "model_type": config.model_type,
        "vocab_size": config.vocab_size,
        "hidden_size": getattr(config, "hidden_size", config.d_model if hasattr(config, "d_model") else 0),
        "num_layers": getattr(config, "num_hidden_layers", config.n_layer if hasattr(config, "n_layer") else 0),
        "num_heads": getattr(config, "num_attention_heads", config.n_head if hasattr(config, "n_head") else 0),
        "num_kv_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "max_position_embeddings": getattr(config, "max_position_embeddings", 2048),
        "intermediate_size": getattr(config, "intermediate_size", 0),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "layer_norm_eps": getattr(config, "layer_norm_epsilon", 1e-6),
        "quantization": quantization,
        "use_gqa": hasattr(config, "num_key_value_heads") and config.num_key_value_heads != config.num_attention_heads,
        "use_rope": True,  # Most modern models use RoPE
        "bos_token_id": getattr(config, "bos_token_id", 1),
        "eos_token_id": getattr(config, "eos_token_id", 2),
        "pad_token_id": getattr(config, "pad_token_id", 0),
    }
    
    print(f"Model configuration:")
    for key, value in header.items():
        print(f"  {key}: {value}")
    
    with open(output_path, "wb") as f:
        # Write magic number
        f.write(b"VNUS")  # Venus format identifier
        
        # Write header
        header_json = json.dumps(header).encode('utf-8')
        f.write(struct.pack('I', len(header_json)))
        f.write(header_json)
        
        # Write model weights
        state_dict = model.state_dict()
        
        # Write number of tensors
        f.write(struct.pack('I', len(state_dict)))
        
        total_params = 0
        quantized_params = 0
        
        for name, param in state_dict.items():
            print(f"Processing {name}: {list(param.shape)}")
            
            # Convert to float32 if needed
            param_data = param.data.float()
            
            # Skip certain tensors from quantization
            skip_quantization = any(skip in name for skip in ["embeddings", "norm", "ln", "bias"])
            
            if skip_quantization or quantization == "none":
                # Save as float32
                quantized = param_data
                scale = 1.0
                actual_quantization = "none"
            else:
                # Quantize tensor
                quantized, scale = quantize_tensor(param_data, quantization)
                actual_quantization = quantization
            
            # Write tensor info
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write quantization type
            quant_bytes = actual_quantization.encode('utf-8')
            f.write(struct.pack('I', len(quant_bytes)))
            f.write(quant_bytes)
            
            # Write shape
            shape = list(param.shape)
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))
            
            # Write scale
            f.write(struct.pack('f', scale))
            
            # Write data
            if actual_quantization == "none":
                data_bytes = quantized.cpu().numpy().astype(np.float32).tobytes()
            else:
                data_bytes = quantized.cpu().numpy().astype(np.int8).tobytes()
            
            f.write(struct.pack('Q', len(data_bytes)))
            f.write(data_bytes)
            
            # Update statistics
            total_params += param.numel()
            if actual_quantization != "none":
                quantized_params += param.numel()
    
    # Save tokenizer
    tokenizer_path = Path(output_path).parent / f"{Path(output_path).stem}_tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    
    # Print statistics
    file_size = Path(output_path).stat().st_size
    print(f"\n✅ Model saved to {output_path}")
    print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.1f}%)")
    print(f"   Tokenizer saved to: {tokenizer_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert HF models to Venus format")
    parser.add_argument("--input", type=str, required=True, help="Input model path or HF model ID")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    parser.add_argument("--quantization", type=str, default="q8_0", 
                       choices=["none", "q8_0", "q4_0"], help="Quantization method")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for loading")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.input}")
    
    try:
        # Load model and tokenizer
        if args.device == "cuda" and torch.cuda.is_available():
            model = AutoModel.from_pretrained(args.input, torch_dtype=torch.float16, device_map="auto")
        else:
            model = AutoModel.from_pretrained(args.input, torch_dtype=torch.float32)
            
        tokenizer = AutoTokenizer.from_pretrained(args.input)
        
        print(f"Model loaded successfully")
        print(f"Model type: {model.config.model_type}")
        
        # Convert and save
        print(f"\nConverting with {args.quantization} quantization")
        save_venus_model(model, tokenizer, args.output, args.quantization)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()