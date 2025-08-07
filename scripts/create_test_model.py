#!/usr/bin/env python3
"""Create a small test model in Venus format"""

import struct
import json
import numpy as np

def create_test_model(output_path="models/test_model.venus"):
    """Create a minimal test model"""
    
    # Model configuration
    config = {
        "version": 1,
        "format": "venus",
        "architecture": "gpt",
        "model_type": "test",
        "vocab_size": 1000,
        "hidden_size": 128,
        "num_layers": 2,
        "num_heads": 4,
        "num_kv_heads": 4,
        "max_position_embeddings": 512,
        "intermediate_size": 512,
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
        
        # Create minimal tensors
        tensors = [
            ("embeddings.weight", (1000, 128)),
            ("layer_0.attention.q_proj.weight", (128, 128)),
            ("layer_0.attention.k_proj.weight", (128, 128)),
            ("layer_0.attention.v_proj.weight", (128, 128)),
            ("layer_0.attention.o_proj.weight", (128, 128)),
            ("layer_0.mlp.gate_proj.weight", (512, 128)),
            ("layer_0.mlp.up_proj.weight", (512, 128)),
            ("layer_0.mlp.down_proj.weight", (128, 512)),
            ("layer_0.input_layernorm.weight", (128,)),
            ("layer_0.post_attention_layernorm.weight", (128,)),
            ("lm_head.weight", (1000, 128)),
        ]
        
        # Write number of tensors
        f.write(struct.pack('I', len(tensors)))
        
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
            
            # Generate random data
            data = np.random.randn(*shape).astype(np.float32) * 0.02
            data_bytes = data.tobytes()
            
            # Write data size and data
            f.write(struct.pack('Q', len(data_bytes)))
            f.write(data_bytes)
    
    print(f"Created test model: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    create_test_model()