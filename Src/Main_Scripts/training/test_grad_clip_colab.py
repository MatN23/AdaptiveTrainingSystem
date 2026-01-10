#!/usr/bin/env python3
"""
Enhanced Gradient Clipping Test for Google Colab
Run this after recompiling the CUDA kernel to verify the fix.

Usage:
    cd /content/AdaptiveTrainingSystem/Src/Main_Scripts
    python training/test_grad_clip_colab.py
"""

import torch
import sys

def test_gradient_clip():
    """Test the fused gradient clipping kernel against PyTorch baseline."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot test")
        return False
    
    print("=" * 80)
    print("FUSED GRADIENT CLIPPING - VERIFICATION TEST")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()
    
    # Import after CUDA check
    try:
        from training.cuda_kernels import FusedGradClip
    except ImportError as e:
        print(f"❌ Failed to import: {e}")
        print("   Make sure you're in the Main_Scripts directory")
        return False
    
    fused_clip = FusedGradClip()
    print(f"CUDA enabled: {fused_clip.cuda_enabled}")
    
    if not fused_clip.cuda_enabled:
        print("❌ CUDA kernel not loaded - check compilation")
        return False
    
    all_passed = True
    
    # ==========================================================================
    # TEST 1: Basic Norm Computation (Known Values)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("TEST 1: Basic Norm Computation (Known Values)")
    print("-" * 60)
    
    # Create tensor with known gradient: all 1s
    size = 10000
    param = torch.nn.Parameter(torch.zeros(size, device='cuda'))
    param.grad = torch.ones(size, device='cuda')
    
    # Expected: sqrt(10000 * 1^2) = 100.0
    expected_norm = 100.0
    
    # Get CUDA result
    fused_clip.set_implementation("cuda")
    cuda_norm = fused_clip([param], max_norm=1000.0)  # High max_norm to avoid clipping
    
    # Get PyTorch result
    param.grad = torch.ones(size, device='cuda')  # Reset grad
    fused_clip.set_implementation("pytorch")
    pytorch_norm = fused_clip([param], max_norm=1000.0)
    
    print(f"  Expected norm:  {expected_norm:.6f}")
    print(f"  CUDA norm:      {cuda_norm:.6f}")
    print(f"  PyTorch norm:   {pytorch_norm:.6f}")
    print(f"  CUDA/Expected:  {cuda_norm / expected_norm:.6f}x")
    print(f"  CUDA/PyTorch:   {cuda_norm / pytorch_norm:.6f}x")
    
    if abs(cuda_norm - expected_norm) < 0.01:
        print("  ✅ TEST 1 PASSED")
    else:
        print("  ❌ TEST 1 FAILED - Norm mismatch!")
        all_passed = False
    
    # ==========================================================================
    # TEST 2: Large Scale (Many Small Values)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("TEST 2: Large Scale Accumulation (10M elements)")
    print("-" * 60)
    
    # 10 million elements with value 0.001
    # Expected norm: sqrt(10^7 * 0.001^2) = sqrt(10) ≈ 3.162
    size = 10_000_000
    param = torch.nn.Parameter(torch.zeros(size, device='cuda'))
    param.grad = torch.full((size,), 0.001, device='cuda')
    
    expected_norm = torch.norm(param.grad).item()
    
    fused_clip.set_implementation("cuda")
    cuda_norm = fused_clip([param], max_norm=100.0)
    
    param.grad = torch.full((size,), 0.001, device='cuda')
    fused_clip.set_implementation("pytorch")
    pytorch_norm = fused_clip([param], max_norm=100.0)
    
    print(f"  Expected norm:  {expected_norm:.6f}")
    print(f"  CUDA norm:      {cuda_norm:.6f}")
    print(f"  PyTorch norm:   {pytorch_norm:.6f}")
    print(f"  Relative error: {abs(cuda_norm - expected_norm) / expected_norm * 100:.4f}%")
    
    if abs(cuda_norm - expected_norm) / expected_norm < 0.001:  # < 0.1% error
        print("  ✅ TEST 2 PASSED")
    else:
        print("  ❌ TEST 2 FAILED - Accumulation error!")
        all_passed = False
    
    # ==========================================================================
    # TEST 3: Multiple Tensors (Simulating Real Model)
    # ==========================================================================
    print("\n" + "-" * 60)
    print("TEST 3: Multiple Tensors (Simulating Real Model)")
    print("-" * 60)
    
    # Create 50 tensors of varying sizes
    params = []
    total_elements = 0
    for i in range(50):
        size = 1000 * (i + 1)  # Sizes: 1k, 2k, ..., 50k
        p = torch.nn.Parameter(torch.zeros(size, device='cuda'))
        p.grad = torch.ones(size, device='cuda')
        params.append(p)
        total_elements += size
    
    # Expected: sqrt(total_elements)
    expected_norm = (total_elements ** 0.5)
    
    fused_clip.set_implementation("cuda")
    cuda_norm = fused_clip(params, max_norm=10000.0)
    
    # Reset grads
    for p in params:
        p.grad = torch.ones_like(p.grad)
    fused_clip.set_implementation("pytorch")
    pytorch_norm = fused_clip(params, max_norm=10000.0)
    
    print(f"  Total elements: {total_elements:,}")
    print(f"  Expected norm:  {expected_norm:.6f}")
    print(f"  CUDA norm:      {cuda_norm:.6f}")
    print(f"  PyTorch norm:   {pytorch_norm:.6f}")
    
    if abs(cuda_norm - expected_norm) / expected_norm < 0.001:
        print("  ✅ TEST 3 PASSED")
    else:
        print("  ❌ TEST 3 FAILED - Multi-tensor issue!")
        all_passed = False
    
    # ==========================================================================
    # TEST 4: Clipping Actually Works
    # ==========================================================================
    print("\n" + "-" * 60)
    print("TEST 4: Clipping Application")
    print("-" * 60)
    
    size = 10000
    param = torch.nn.Parameter(torch.zeros(size, device='cuda'))
    param.grad = torch.ones(size, device='cuda')
    
    # Norm is 100, max_norm is 1.0, so clip_coef = 1/100 = 0.01
    max_norm = 1.0
    expected_clip_coef = max_norm / (100.0 + 1e-6)
    
    fused_clip.set_implementation("cuda")
    fused_clip([param], max_norm=max_norm)
    
    # After clipping, each element should be ~0.01
    actual_value = param.grad[0].item()
    expected_value = 1.0 * expected_clip_coef
    
    print(f"  Max norm:       {max_norm}")
    print(f"  Expected clip:  {expected_clip_coef:.8f}")
    print(f"  Expected value: {expected_value:.8f}")
    print(f"  Actual value:   {actual_value:.8f}")
    
    if abs(actual_value - expected_value) < 1e-6:
        print("  ✅ TEST 4 PASSED")
    else:
        print("  ❌ TEST 4 FAILED - Clipping not applied correctly!")
        all_passed = False
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Gradient clipping is working correctly.")
        print("   You can now run training with confidence.")
    else:
        print("❌ SOME TESTS FAILED! Check the debug printf output above.")
        print("   Look for 'DEBUG GRAD_CLIP:' lines in the output.")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = test_gradient_clip()
    sys.exit(0 if success else 1)
