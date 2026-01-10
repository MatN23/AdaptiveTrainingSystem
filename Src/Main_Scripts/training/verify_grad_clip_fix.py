import torch
import numpy as np
import time
from training.cuda_kernels import FusedGradClip

def test_norm_and_clip():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("❌ Skip: CUDA not available")
        return

    print("="*80)
    print("VERIFYING FUSED GRAD CLIP FIX")
    print("="*80)

    fused_clip = FusedGradClip()
    fused_clip.set_implementation("cuda")
    
    # Test 1: Precision with many small elements
    print("\n[Test 1] Precision with 10^7 elements...")
    size = 10_000_000
    param = torch.nn.Parameter(torch.randn(size, device=device))
    # Set grads to constant small value to test accumulation
    param.grad = torch.full((size,), 0.001, device=device)
    
    # Expected norm: sqrt(size * 0.001^2) = sqrt(10^7 * 10^-6) = sqrt(10) \approx 3.162277
    expected_norm = torch.norm(param.grad).item()
    
    actual_norm = fused_clip([param], max_norm=10.0)
    
    diff = abs(actual_norm - expected_norm)
    print(f"   Expected Norm: {expected_norm:.8f}")
    print(f"   Actual Norm:   {actual_norm:.8f}")
    print(f"   Difference:    {diff:.8e}")
    
    if diff < 1e-4:
        print("   ✅ Test 1 Passed!")
    else:
        print("   ❌ Test 1 Failed (Precision issue)")

    # Test 2: Alignment Handling
    print("\n[Test 2] Alignment handling (Offset pointers)...")
    # Base pointer is 256-byte aligned. Offset by 4 bytes makes it unaligned for float4.
    full_grad = torch.randn(1004, device=device)
    unaligned_grad = full_grad[1:1001] # 1000 elements, unaligned pointer
    
    class DummyParam:
        def __init__(self, grad):
            self.grad = grad
    
    p_unaligned = DummyParam(unaligned_grad)
    expected_norm_unaligned = torch.norm(unaligned_grad).item()
    
    try:
        actual_norm_unaligned = fused_clip([p_unaligned], max_norm=1.0)
        print(f"   Expected Norm: {expected_norm_unaligned:.6f}")
        print(f"   Actual Norm:   {actual_norm_unaligned:.6f}")
        if abs(actual_norm_unaligned - expected_norm_unaligned) < 1e-5:
            print("   ✅ Test 2 Passed!")
        else:
            print("   ❌ Test 2 Failed (Wrong value for unaligned)")
    except Exception as e:
        print(f"   ❌ Test 2 Failed (Crashed on unaligned): {e}")

    # Test 3: Clipping Verification
    print("\n[Test 3] Clipping effect...")
    param.grad = torch.full((size,), 1.0, device=device)
    # Norm is sqrt(10^7) \approx 3162.27
    max_norm = 1.0
    expected_coef = max_norm / (3162.27766 + 1e-6)
    
    orig_grad_sum = param.grad.sum().item()
    fused_clip([param], max_norm=max_norm)
    new_grad_sum = param.grad.sum().item()
    
    actual_coef = new_grad_sum / orig_grad_sum
    print(f"   Target Coef: {expected_coef:.8f}")
    print(f"   Actual Coef: {actual_coef:.8f}")
    
    if abs(actual_coef - expected_coef) < 1e-5:
        print("   ✅ Test 3 Passed!")
    else:
        print("   ❌ Test 3 Failed (Wrong clipping)")

    print("\n" + "="*80)
    print("ALL VERIFICATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_norm_and_clip()
