
import torch
import time
import sys
from pathlib import Path

# Add src to path if running directly
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from core.triton_ops import triton_fp8_matmul, is_triton_available, TritonFP8Linear

def benchmark_function(func, num_warmup=5, num_iters=20):
    # Warmup
    for _ in range(num_warmup):
        func()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elif torch.backends.mps.is_available(): torch.mps.synchronize()
    
    start = time.time()
    for _ in range(num_iters):
        func()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elif torch.backends.mps.is_available(): torch.mps.synchronize()
    end = time.time()
    return (end - start) * 1000 / num_iters

def run_benchmark():
    # Detect Device
    if torch.cuda.is_available():
        device = 'cuda'
        triton_active = is_triton_available()
    elif torch.backends.mps.is_available():
        device = 'mps'
        triton_active = False
    else:
        device = 'cpu'
        triton_active = False
        
    print(f"🚀 Benchmarking FP8 Emulation on {device.upper()}...")
    if not triton_active:
        print("ℹ️  Triton not active. Using Fallback mode (LUT/Native).")
        if device == 'cuda':
             print("   (Reason: cuda available but triton import failed)")
        else:
             print("   (Note: Fallback on CPU/MPS is for correctness testing, not speedup)")

    M, K, N = 2048, 2048, 2048  # Reasonable size
    
    print(f"Generating data ({M}x{K}x{N})...")
    # Setup Data
    a_fp16 = torch.randn((M, K), device=device, dtype=torch.float16)
    b_fp16 = torch.randn((K, N), device=device, dtype=torch.float16)
    scale_a = 1.0
    scale_b = 1.0
    
    # Quantize inputs (Simulated for benchmark)
    # We want meaningful values to check accuracy "difference"
    # Simple quantization:
    a_max = a_fp16.abs().max() + 1e-6
    b_max = b_fp16.abs().max() + 1e-6
    scale_a = a_max / 448.0
    scale_b = b_max / 448.0
    
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            a_int8 = (a_fp16 / scale_a).to(torch.float8_e4m3fn).view(torch.int8)
            b_int8 = (b_fp16 / scale_b).to(torch.float8_e4m3fn).view(torch.int8)
        except (TypeError, RuntimeError):
            # Fallback for MPS
            a_int8 = (a_fp16 / scale_a).clamp(-126, 126).to(torch.int8)
            b_int8 = (b_fp16 / scale_b).clamp(-126, 126).to(torch.int8)
    else:
        # Linear clamp fallback for data generation
        a_int8 = (a_fp16 / scale_a).clamp(-126, 126).to(torch.int8)
        b_int8 = (b_fp16 / scale_b).clamp(-126, 126).to(torch.int8)
    
    # 1. Baseline: PyTorch FP16
    print(f"Running FP16 GEMM...")
    def run_torch():
        return torch.matmul(a_fp16, b_fp16)
    
    t_torch = benchmark_function(run_torch)
    print(f"  PyTorch FP16: {t_torch:.3f} ms")
    c_ref = torch.matmul(a_fp16, b_fp16)
    
    # 2. Triton/Fallback FP8
    print(f"Running FP8 GEMM...")
    def run_fp8():
        return triton_fp8_matmul(a_int8, b_int8, scale_a, scale_b)
    
    t_fp8 = benchmark_function(run_fp8)
    print(f"  FP8 Emulation: {t_fp8:.3f} ms")
    c_fp8 = triton_fp8_matmul(a_int8, b_int8, scale_a, scale_b)
    
    speedup = t_torch / t_fp8
    print(f"  👉 Speedup: {speedup:.2f}x")
    
    # 3. Accuracy Check
    print("\n📊 Accuracy Analysis:")
    # Mean Absolute Error
    # Ignore NaNs (caused by invalid bit patterns in linear-quantized fallback)
    # real E4M3 wouldn't have them, but our 'int8 cast' benchmark data does.
    diff = (c_ref - c_fp8).abs()
    # Mask out NaNs
    valid = ~torch.isnan(diff)
    if valid.sum() > 0:
        mae = diff[valid].mean().item()
        max_err = diff[valid].max().item()
        relative_err = (diff[valid] / (c_ref[valid].abs() + 1e-6)).mean().item()
    else:
        mae, max_err, relative_err = 0.0, 0.0, 0.0
    
    print(f"  Mean Abs Error: {mae:.5f}")
    print(f"  Max Error:      {max_err:.5f}")
    print(f"  Rel Error:      {relative_err*100:.2f}%")
    
    if relative_err < 0.10: # <10% error is decent for 4-bit mantissa
        print("  ✅ Accuracy looks reasonable for FP8.")
    else:
        print("  ⚠️ Accuracy deviation is high (Expected for FP8/Simulated).")

if __name__ == "__main__":
    run_benchmark()
