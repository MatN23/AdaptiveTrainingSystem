import torch
import sys
import os
import types

# Create dummy config module
config = types.ModuleType('config')
config_manager = types.ModuleType('config.config_manager')

class Config:
    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)

config_manager.Config = Config
config.config_manager = config_manager

sys.modules['config'] = config
sys.modules['config.config_manager'] = config_manager

# Now load checkpoint
checkpoint = torch.load('checkpoints/final.pt', map_location='cpu', weights_only=False)
config = checkpoint['config']

print("Actual config from checkpoint:")
print(f"vocab_size: {config.vocab_size}")
print(f"hidden_size: {config.hidden_size}")
print(f"num_layers: {config.num_layers}")
print(f"num_heads: {config.num_heads}")
print(f"num_kv_heads: {config.num_kv_heads}")
print(f"intermediate_size: {config.intermediate_size}")
print(f"seq_length: {config.seq_length}")
print(f"num_experts: {config.num_experts}")
print(f"moe_top_k: {config.moe_top_k}")