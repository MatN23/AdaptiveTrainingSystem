#!/usr/bin/env python3
"""
export_model.py - Export trained PyTorch MoE+MoD model to frozen binary format

Usage:
    python export/export_model.py --model path/to/checkpoint.pt --output model.bin
"""

import torch
import struct
import argparse
import sys
import os

# Add parent directory to import your model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Also add the actual location where we're running from
sys.path.insert(0, os.getcwd())

try:
    from core.model import DeepSeekTransformer, DeepSeekConfig
    from core.tokenizer import ConversationTokenizer
    print("✓ Successfully imported model classes")
except ImportError as e:
    print(f"⚠️  Could not import model.py: {e}")
    DeepSeekTransformer = None
    DeepSeekConfig = None

# Create dummy classes to allow unpickling
try:
    from config.config_manager import Config
except ImportError:
    print("⚠️  Could not import config module - creating dummy")
    import types
    
    # Create dummy config module structure
    config = types.ModuleType('config')
    config_manager = types.ModuleType('config.config_manager')
    
    # Create dummy Config class
    class Config:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)
    
    config_manager.Config = Config
    config.config_manager = config_manager
    
    # Register in sys.modules
    sys.modules['config'] = config
    sys.modules['config.config_manager'] = config_manager


def export_tensor(f, tensor):
    """Write tensor data as float32."""
    data = tensor.detach().cpu().float().numpy().flatten()
    f.write(data.tobytes())


def precompute_rope_cache(max_seq_len, head_dim, theta=10000.0):
    """Precompute RoPE cos/sin cache."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    
    cos_cache = emb.cos()
    sin_cache = emb.sin()
    
    return cos_cache, sin_cache


def export_model(model, config, output_path):
    """Export model to binary format."""
    
    print(f"🔄 Exporting model to {output_path}...")
    
    with open(output_path, 'wb') as f:
        # Header
        magic = 0x4D4F4549  # "MOEI"
        version = 1
        
        f.write(struct.pack('I', magic))
        f.write(struct.pack('I', version))
        
        # Config
        f.write(struct.pack('i', config.vocab_size))
        f.write(struct.pack('i', config.hidden_size))
        f.write(struct.pack('i', config.num_layers))
        f.write(struct.pack('i', config.num_heads))
        f.write(struct.pack('i', config.num_kv_heads))
        f.write(struct.pack('i', config.intermediate_size))
        f.write(struct.pack('i', config.seq_length))
        
        # MoE config
        use_moe = getattr(config, 'use_moe', False)
        f.write(struct.pack('?', use_moe))
        if use_moe:
            f.write(struct.pack('i', config.num_experts))
            f.write(struct.pack('i', config.moe_top_k))
        else:
            f.write(struct.pack('i', 0))
            f.write(struct.pack('i', 0))

        # Write moe_layers vector SIZE first (as int32_t), then the bools
        f.write(struct.pack('i', config.num_layers))  # Add this line - SIZE FIRST
        for i in range(config.num_layers):
            is_moe = False
            if use_moe and hasattr(model.layers[i], 'use_moe'):
                is_moe = model.layers[i].use_moe
            f.write(struct.pack('?', is_moe))

        # MoD config
        use_mod = getattr(config, 'use_mod', False)
        f.write(struct.pack('?', use_mod))
        if use_mod:
            f.write(struct.pack('f', getattr(config, 'mod_capacity_factor', 0.5)))
        else:
            f.write(struct.pack('f', 0.0))

        # Write mod_layers vector SIZE first (as int32_t), then the bools
        f.write(struct.pack('i', config.num_layers))  # Add this line - SIZE FIRST
        for i in range(config.num_layers):
            is_mod = False
            if use_mod and hasattr(model.layers[i].ffn, 'use_mod'):
                is_mod = model.layers[i].ffn.use_mod
            f.write(struct.pack('?', is_mod))
        
        # RoPE theta
        f.write(struct.pack('f', getattr(config, 'rope_theta', 10000.0)))
        
        print("✓ Config written")
        
        # Embeddings
        export_tensor(f, model.embed_tokens.weight)
        print(f"✓ Embeddings: {model.embed_tokens.weight.shape}")
        
        # Layers
        for i, layer in enumerate(model.layers):
            print(f"  Layer {i}...")
            
            # Attention
            export_tensor(f, layer.self_attn.q_proj.weight)
            export_tensor(f, layer.self_attn.k_proj.weight)
            export_tensor(f, layer.self_attn.v_proj.weight)
            export_tensor(f, layer.self_attn.o_proj.weight)
            export_tensor(f, layer.input_norm.weight)
            
            # FFN
            if hasattr(layer, 'use_moe') and layer.use_moe:
                # MoE layer
                export_tensor(f, layer.ffn.gate.weight)
                
                for expert in layer.ffn.experts:
                    # Extract gate, up, down from expert
                    if hasattr(expert, 'gate_up_proj'):
                        # Fused projection - split it
                        gate_up = expert.gate_up_proj.weight
                        mid = gate_up.shape[0] // 2
                        export_tensor(f, gate_up[:mid])  # gate
                        export_tensor(f, gate_up[mid:])  # up
                    else:
                        export_tensor(f, expert.gate_proj.weight)
                        export_tensor(f, expert.up_proj.weight)
                    export_tensor(f, expert.down_proj.weight)
                
                print(f"    MoE: {config.num_experts} experts")
            else:
                # Dense FFN
                if hasattr(layer.ffn, 'gate_up_proj'):
                    gate_up = layer.ffn.gate_up_proj.weight
                    mid = gate_up.shape[0] // 2
                    export_tensor(f, gate_up[:mid])
                    export_tensor(f, gate_up[mid:])
                else:
                    export_tensor(f, layer.ffn.gate_proj.weight)
                    export_tensor(f, layer.ffn.up_proj.weight)
                export_tensor(f, layer.ffn.down_proj.weight)
                
                # MoD router if present
                if hasattr(layer.ffn, 'router'):
                    export_tensor(f, layer.ffn.router.router.weight)
                    threshold = getattr(layer.ffn.router, 'skip_threshold', 0.0)
                    f.write(struct.pack('f', threshold))
                    print(f"    Dense + MoD (threshold={threshold:.3f})")
                else:
                    print(f"    Dense")
            
            export_tensor(f, layer.post_attn_norm.weight)
        
        # Final norm
        export_tensor(f, model.norm.weight)
        
        # LM head
        export_tensor(f, model.lm_head.weight)
        print(f"✓ LM head: {model.lm_head.weight.shape}")
        
        # RoPE cache
        head_dim = config.hidden_size // config.num_heads
        cos_cache, sin_cache = precompute_rope_cache(
            config.seq_length, 
            head_dim, 
            getattr(config, 'rope_theta', 10000.0)
        )
        export_tensor(f, cos_cache)
        export_tensor(f, sin_cache)
        print(f"✓ RoPE cache: {cos_cache.shape}")
    
    file_size = os.path.getsize(output_path)
    print(f"\n✅ Export complete: {output_path} ({file_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='Export MoE+MoD model to binary')
    parser.add_argument('--model', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='model.bin', help='Output binary file')
    parser.add_argument('--config', type=str, help='Optional config JSON')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1
    
    # Load checkpoint
    print(f"📦 Loading checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    # Extract model and config
    if isinstance(checkpoint, dict):
        # Check for nested structure first (your checkpoint uses this!)
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            config = checkpoint.get('config', None)
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
            config = checkpoint.get('config', None)
        elif 'state_dict' in checkpoint:
            model_state = checkpoint['state_dict']
            config = checkpoint.get('config', None)
        else:
            model_state = checkpoint
            config = None
    else:
        model_state = checkpoint.state_dict()
        config = None
    
    # Load config from file if provided (this overrides checkpoint config)
    if args.config:
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        
        if DeepSeekConfig is not None:
            config = DeepSeekConfig(**config_dict)
        else:
            # Create a simple config object
            class SimpleConfig:
                def __init__(self, **kwargs):
                    self.__dict__.update(kwargs)
            config = SimpleConfig(**config_dict)
    elif config is not None:
        # Use config from checkpoint if no external config provided
        print("✓ Using config from checkpoint")
    
    # Create model
    if config is None:
        print("❌ Config not found in checkpoint. Please provide --config")
        return 1
    
    if DeepSeekTransformer:
        model = DeepSeekTransformer(config)
        model.load_state_dict(model_state)
    else:
        print("❌ Could not import DeepSeekTransformer")
        return 1
    
    model.eval()
    
    # Export
    export_model(model, config, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())