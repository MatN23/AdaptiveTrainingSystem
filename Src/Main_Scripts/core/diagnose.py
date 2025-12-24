#!/usr/bin/env python3
"""
CUDA Setup Diagnostic Tool
Checks all aspects of CUDA acceleration setup
"""

import sys
import os
from pathlib import Path
import torch

print("="*80)
print("CUDA SETUP DIAGNOSTICS")
print("="*80)

# 1. Check PyTorch CUDA
print("\n1. PyTorch CUDA Status:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Device name: {torch.cuda.get_device_name(0)}")
    print(f"   Compute capability: {torch.cuda.get_device_capability(0)}")

# 2. Check file structure
print("\n2. File Structure:")
current_dir = Path(__file__).parent
print(f"   Current directory: {current_dir}")

# Check for compiled libraries
cuda_files = {
    'transformer_ops.so': current_dir / 'cuda' / 'transformer_ops.so',
    'moe_cuda_ops (compiled)': current_dir / 'moe_cuda_ops.cpython-*.so',
}

for name, path in cuda_files.items():
    if '*' in str(path):
        # Glob pattern
        matches = list(current_dir.glob(str(path.name)))
        if matches:
            print(f"   âœ… {name}: {matches[0]}")
        else:
            print(f"   âŒ {name}: NOT FOUND")
    else:
        if path.exists():
            print(f"   âœ… {name}: {path}")
        else:
            print(f"   âŒ {name}: NOT FOUND at {path}")

# Check Python files
python_files = [
    'cuda_opt_wrapper.py',
    'moe_cuda_wrapper.py', 
    'unified_ops.py',
    'model.py'
]

print("\n   Python wrappers:")
for fname in python_files:
    fpath = current_dir / fname
    if fpath.exists():
        print(f"   âœ… {fname}")
    else:
        print(f"   âŒ {fname}: NOT FOUND")

# 3. Test imports
print("\n3. Import Tests:")

# Test 3a: Direct transformer ops
try:
    sys.path.insert(0, str(current_dir))
    from cuda_opt_wrapper import (
        TRANSFORMER_OPS_AVAILABLE,
        FusedRMSNorm,
        FusedRoPE,
        FusedSwiGLU
    )
    print(f"   âœ… cuda_opt_wrapper: TRANSFORMER_OPS_AVAILABLE={TRANSFORMER_OPS_AVAILABLE}")
    if TRANSFORMER_OPS_AVAILABLE:
        print(f"      - FusedRMSNorm: available")
        print(f"      - FusedRoPE: available")
        print(f"      - FusedSwiGLU: available")
except Exception as e:
    print(f"   âŒ cuda_opt_wrapper: {e}")

# Test 3b: MoE ops
try:
    from moe_cuda_wrapper import CUDA_OPS_AVAILABLE as MOE_AVAILABLE
    print(f"   âœ… moe_cuda_wrapper: CUDA_OPS_AVAILABLE={MOE_AVAILABLE}")
except Exception as e:
    print(f"   âŒ moe_cuda_wrapper: {e}")

# Test 3c: Unified ops
try:
    from unified_ops import (
        BACKEND,
        HAS_CUDA_OPS,
        HAS_METAL_OPS,
        get_backend_info
    )
    print(f"   âœ… unified_ops:")
    print(f"      Backend: {BACKEND}")
    print(f"      HAS_CUDA_OPS: {HAS_CUDA_OPS}")
    print(f"      HAS_METAL_OPS: {HAS_METAL_OPS}")
except Exception as e:
    print(f"   âŒ unified_ops: {e}")

# 4. Test actual functionality
print("\n4. Functionality Tests:")

if torch.cuda.is_available():
    try:
        from cuda_opt_wrapper import FusedRMSNorm, TRANSFORMER_OPS_AVAILABLE
        
        if TRANSFORMER_OPS_AVAILABLE:
            print("   Testing FusedRMSNorm on GPU...")
            norm = FusedRMSNorm(768).cuda()
            x = torch.randn(4, 128, 768).cuda()
            y = norm(x)
            print(f"   âœ… FusedRMSNorm: Input {x.shape} -> Output {y.shape}")
        else:
            print("   âš ï¸  Transformer ops not available, skipping tests")
            
    except Exception as e:
        print(f"   âŒ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("   âš ï¸  CUDA not available, skipping GPU tests")

# 5. Environment variables
print("\n5. Environment Variables:")
env_vars = ['CUDA_HOME', 'CUDA_PATH', 'PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH']
for var in env_vars:
    value = os.environ.get(var, 'NOT SET')
    if var in ['PATH', 'LD_LIBRARY_PATH', 'PYTHONPATH'] and value != 'NOT SET':
        # Show first few paths only
        paths = value.split(':')[:3]
        print(f"   {var}: {':'.join(paths)}... ({len(value.split(':'))} total)")
    else:
        print(f"   {var}: {value}")

# 6. Recommendations
print("\n" + "="*80)
print("RECOMMENDATIONS:")
print("="*80)

issues_found = []

# Check if CUDA is available but ops aren't loaded
if torch.cuda.is_available():
    try:
        from cuda_opt_wrapper import TRANSFORMER_OPS_AVAILABLE
        from moe_cuda_wrapper import CUDA_OPS_AVAILABLE as MOE_AVAILABLE
        
        if not TRANSFORMER_OPS_AVAILABLE:
            issues_found.append("Transformer CUDA ops not loaded")
        if not MOE_AVAILABLE:
            issues_found.append("MoE CUDA ops not loaded")
    except ImportError as e:
        issues_found.append(f"Import error: {e}")

if issues_found:
    print("\nâš ï¸  Issues Found:")
    for i, issue in enumerate(issues_found, 1):
        print(f"   {i}. {issue}")
    
    print("\nðŸ"§ Suggested Fixes:")
    print("   1. Make sure you're in the correct directory:")
    print("      cd /content/LuminaAI/Src/Main_Scripts/core")
    print()
    print("   2. Recompile CUDA kernels:")
    print("      bash compile_transformer_ops.sh")
    print("      bash compile_cuda_moe.sh")
    print()
    print("   3. Test imports directly:")
    print("      python -c 'from cuda_opt_wrapper import TRANSFORMER_OPS_AVAILABLE; print(TRANSFORMER_OPS_AVAILABLE)'")
    print()
    print("   4. If still failing, check library dependencies:")
    print("      ldd cuda/transformer_ops.so")
else:
    print("\nâœ… All checks passed! CUDA acceleration is properly configured.")

print("="*80)