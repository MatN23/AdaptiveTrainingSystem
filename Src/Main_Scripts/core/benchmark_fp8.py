# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

import torch
import time
from core.triton_ops import triton_fp8_matmul, is_triton_available, TritonFP8Linear

def benchmark_function(func, num_warmup=10, num_iters=100):
   # Warmup
   for _ in range(num_warmup):
       func()
   torch.cuda.synchronize()
   
   start = time.time()
   for _ in range(num_iters):
       func()
   torch.cuda.synchronize()
   end = time.time()
   return (end - start) * 1000 / num_iters

def run_benchmark():
   if not is_triton_available():
       print("❌ Triton not available. Skipping FP8 benchmark.")
       return

   print("🚀 Benchmarking Triton FP8 Emulation...")
   
   device = 'cuda'
   M, K, N = 4096, 4096, 4096 # Large GEMM
   
   # Setup Data
   a_fp16 = torch.randn((M, K), device=device, dtype=torch.float16)
   b_fp16 = torch.randn((K, N), device=device, dtype=torch.float16)
   
   # Setup FP8 (Simulated)
   # We use random int8s to simulate pre-quantized data
   a_int8 = torch.randint(-128, 127, (M, K), device=device, dtype=torch.int8)
   b_int8 = torch.randint(-128, 127, (K, N), device=device, dtype=torch.int8)
   
   # 1. Baseline: PyTorch FP16
   print(f"Running FP16 GEMM ({M}x{K}x{N})...")
   def run_torch():
       return torch.matmul(a_fp16, b_fp16)
   
   t_torch = benchmark_function(run_torch)
   print(f"  PyTorch FP16: {t_torch:.3f} ms")
   
   # 2. Triton FP8
   print(f"Running Triton FP8 GEMM ({M}x{K}x{N})...")
   def run_triton():
       return triton_fp8_matmul(a_int8, b_int8)
   
   t_triton = benchmark_function(run_triton)
   print(f"  Triton FP8:   {t_triton:.3f} ms")
   
   speedup = t_torch / t_triton
   print(f"  👉 Speedup: {speedup:.2f}x")
   
   # 3. Check Correctness (Basic)
   # Since inputs are random garbage int8 vs random fp16, we can't compare values.
   # We verify it runs and output shape is correct.
   c = triton_fp8_matmul(a_int8, b_int8)
   print(f"  Output shape: {c.shape} (Expected: {M}x{N})")
   
   if c.shape == (M, N) and not torch.isnan(c).any():
       print("  ✅ Kernel runs successfully!")
   else:
       print("  ❌ Kernel overflow/error!")

if __name__ == "__main__":
   run_benchmark()
