
import sys
import os
import torch
from pathlib import Path

# Setup Path
current_file = Path(__file__)
src_root = current_file.parent.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

print("="*60)
print("CUDA/KERNEL AVAILABILITY CHECK")
print("="*60)

print(f"PyTorch Version: {torch.__version__}")
device_name = "CPU"
if torch.cuda.is_available():
    device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
elif torch.backends.mps.is_available():
    device_name = "MPS (Apple Silicon)"
print(f"Active Device: {device_name}")

try:
    from core.model import (
        HAS_TRANSFORMER_CUDA, 
        HAS_TRITON, 
        USE_UNIFIED_BACKEND
    )
    # Access backend name if possible (might be inside unified_ops)
    from core.unified_ops import BACKEND, HAS_CUDA_OPS, HAS_METAL_OPS
    
    print("-" * 30)
    print("BACKEND CONFIGURATION")
    print("-" * 30)
    print(f"Unified Backend System: {'✅ Enabled' if USE_UNIFIED_BACKEND else '❌ Disabled'}")
    print(f"Selected Backend:       {BACKEND.upper()}")
    
    print("-" * 30)
    print("KERNEL STATUS")
    print("-" * 30)
    print(f"Triton Support:         {'✅ Available' if HAS_TRITON else '❌ Missing'}")
    print(f"CUDA Fused Ops:         {'✅ Active' if HAS_TRANSFORMER_CUDA else '❌ Inactive'}")
    print(f"  - C++ Extension:      {'Loaded' if HAS_CUDA_OPS else 'Not Loaded'}")
    print(f"Metal Fused Ops:        {'✅ Active' if HAS_METAL_OPS else '❌ Inactive'}")
    
    if HAS_TRANSFORMER_CUDA or HAS_METAL_OPS:
        print("\n✅ CONCLUSION: Optimized Kernels are ENABLED.")
    else:
        print("\n⚠️  CONCLUSION: Using Standard PyTorch Kernels (Reference Implementation).")
        
except ImportError as e:
    print(f"\n❌ Error importing core modules: {e}")
    print("Make sure you are running from 'Src/Main_Scripts' directory.")

print("="*60)
