#!/usr/bin/env python3
"""Create a demo model that can generate text"""

import struct
import json
import numpy as np
import os

def create_demo_model(output_path="models/demo_model.venus"):
    """Create a minimal demo model that can generate simple text"""
    
    # Minimal GPT-like configuration
    config = {
        "version": 1,
        "format": "venus",
        "architecture": "gpt",
        "model_type": "demo",
        "vocab_size": 256,  # ASCII characters
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 4,
        "max_position_embeddings": 128,
        "intermediate_size": 256,
        "rope_theta": 10000.0,
        "layer_norm_eps": 1e-6,
        "quantization": "none",
        "use_gqa": False,
        "use_rope": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    
    with open(output_path, "wb") as f:
        # Write magic number
        f.write(b"VNUS")
        
        # Write header
        header_json = json.dumps(config).encode('utf-8')
        f.write(struct.pack('I', len(header_json)))
        f.write(header_json)
        
        # Create minimal tensors for a 2-layer model
        tensors = []
        
        # Token embeddings
        tensors.append(("token_embeddings.weight", (256, 64)))
        
        # Position embeddings
        tensors.append(("position_embeddings.weight", (128, 64)))
        
        # Layers
        for layer in range(2):
            prefix = f"layer_{layer}"
            # Attention
            tensors.append((f"{prefix}.attention.q_proj.weight", (64, 64)))
            tensors.append((f"{prefix}.attention.k_proj.weight", (64, 64)))
            tensors.append((f"{prefix}.attention.v_proj.weight", (64, 64)))
            tensors.append((f"{prefix}.attention.o_proj.weight", (64, 64)))
            
            # FFN
            tensors.append((f"{prefix}.ffn.gate_proj.weight", (256, 64)))
            tensors.append((f"{prefix}.ffn.up_proj.weight", (256, 64)))
            tensors.append((f"{prefix}.ffn.down_proj.weight", (64, 256)))
            
            # Layer norms
            tensors.append((f"{prefix}.attention_norm.weight", (64,)))
            tensors.append((f"{prefix}.ffn_norm.weight", (64,)))
        
        # Output
        tensors.append(("output_norm.weight", (64,)))
        tensors.append(("lm_head.weight", (256, 64)))
        
        # Write number of tensors
        f.write(struct.pack('I', len(tensors)))
        
        # Initialize with small random weights
        np.random.seed(42)
        
        # Write each tensor
        for name, shape in tensors:
            # Write name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write quantization type
            quant_type = b"none"
            f.write(struct.pack('I', len(quant_type)))
            f.write(quant_type)
            
            # Write shape
            f.write(struct.pack('I', len(shape)))
            for dim in shape:
                f.write(struct.pack('I', dim))
            
            # Write scale
            f.write(struct.pack('f', 1.0))
            
            # Generate weights
            if "norm" in name:
                # Layer norm weights initialized to 1
                data = np.ones(shape, dtype=np.float32)
            elif name == "token_embeddings.weight":
                # Initialize embeddings with small random values
                data = np.random.randn(*shape).astype(np.float32) * 0.02
            else:
                # Xavier initialization for other weights
                fan_in = shape[-1] if len(shape) > 1 else 1
                fan_out = shape[0] if len(shape) > 1 else 1
                std = np.sqrt(2.0 / (fan_in + fan_out))
                data = np.random.randn(*shape).astype(np.float32) * std
            
            data_bytes = data.tobytes()
            
            # Write data size and data
            f.write(struct.pack('Q', len(data_bytes)))
            f.write(data_bytes)
    
    print(f"Created demo model: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"Model specs:")
    print(f"  - Vocab size: {config['vocab_size']} (ASCII)")
    print(f"  - Hidden size: {config['hidden_size']}")
    print(f"  - Layers: {config['num_layers']}")
    print(f"  - Total parameters: ~{sum(np.prod(shape) for _, shape in tensors):,}")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    create_demo_model()