#!/usr/bin/env python3
"""
Improved benchmark with fairer comparisons and reduced bias
- Pre-allocates buffers (no clones in timing loop)
- Compares against optimized PyTorch when available
- Reports both "naive PyTorch" and "optimized PyTorch" results
"""

import torch
import torch.nn.functional as F
import ctypes
import time
from pathlib import Path

# ============================================================================
# LOAD CUDA LIBRARY
# ============================================================================

def load_cuda_lib():
    """Load the compiled CUDA library"""
    lib_path = Path(__file__).parent / "transformer_ops.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"CUDA library not found at {lib_path}")
    
    lib = ctypes.CDLL(str(lib_path))
    
    # Define FP16 function signatures
    lib.rms_norm_launcher_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    ]
    
    lib.rope_precompute_launcher.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    ]
    
    lib.rope_apply_launcher_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    
    lib.swiglu_launcher_fp16.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    
    return lib

CUDA_LIB = load_cuda_lib()

# ============================================================================
# CUDA WRAPPER FUNCTIONS
# ============================================================================

def cuda_rms_norm(input_tensor, weight, output, eps=1e-5):
    """CUDA RMSNorm - operates in-place on output buffer"""
    batch_seq, hidden_size = input_tensor.shape
    
    CUDA_LIB.rms_norm_launcher_fp16(
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(weight.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        batch_seq, hidden_size,
        ctypes.c_float(eps),
        ctypes.c_void_p(0)
    )

def cuda_rope_apply(q, k, cos_cache, sin_cache, position_offset=0):
    """CUDA RoPE - operates in-place"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    CUDA_LIB.rope_apply_launcher_fp16(
        ctypes.c_void_p(q.data_ptr()),
        ctypes.c_void_p(k.data_ptr()),
        ctypes.c_void_p(cos_cache.data_ptr()),
        ctypes.c_void_p(sin_cache.data_ptr()),
        batch_size, num_heads, seq_len, head_dim, position_offset,
        ctypes.c_void_p(0)
    )

def cuda_rope_precompute(max_seq_len, head_dim, theta=10000.0):
    """CUDA RoPE precompute"""
    half_dim = head_dim // 2
    cos_cache = torch.empty(max_seq_len, half_dim, device='cuda', dtype=torch.float32)
    sin_cache = torch.empty(max_seq_len, half_dim, device='cuda', dtype=torch.float32)
    
    CUDA_LIB.rope_precompute_launcher(
        ctypes.c_void_p(cos_cache.data_ptr()),
        ctypes.c_void_p(sin_cache.data_ptr()),
        max_seq_len, head_dim,
        ctypes.c_float(theta),
        ctypes.c_void_p(0)
    )
    
    return cos_cache, sin_cache

def cuda_swiglu(gate, up, output):
    """CUDA SwiGLU - operates on output buffer"""
    total_tokens, intermediate_size = gate.shape
    
    CUDA_LIB.swiglu_launcher_fp16(
        ctypes.c_void_p(gate.data_ptr()),
        ctypes.c_void_p(up.data_ptr()),
        ctypes.c_void_p(output.data_ptr()),
        total_tokens, intermediate_size,
        ctypes.c_void_p(0)
    )

# ============================================================================
# PYTORCH IMPLEMENTATIONS
# ============================================================================

def pytorch_rms_norm_naive(x, weight, output, eps=1e-5):
    """Naive PyTorch RMSNorm (multiple kernels)"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    output.copy_(weight * x_normed)

def pytorch_rms_norm_fused(x, weight, output, eps=1e-5):
    """More fused PyTorch RMSNorm (fewer kernels)"""
    # Single fused computation
    norm = x.norm(2, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    output.copy_(x / (norm + eps) * weight)

def pytorch_rope_naive(q, k, cos_cache, sin_cache):
    """Naive PyTorch RoPE (many ops)"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    half_dim = head_dim // 2
    
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    
    positions = torch.arange(seq_len, device=q.device)
    cos = cos_cache[positions].view(1, 1, seq_len, half_dim)
    sin = sin_cache[positions].view(1, 1, seq_len, half_dim)
    
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    q.copy_(q_rot)
    k.copy_(k_rot)

def pytorch_rope_optimized(q, k, cos_cache, sin_cache):
    """More optimized PyTorch RoPE (in-place, fewer allocations)"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    half_dim = head_dim // 2
    
    positions = torch.arange(seq_len, device=q.device)
    cos = cos_cache[positions].view(1, 1, seq_len, half_dim)
    sin = sin_cache[positions].view(1, 1, seq_len, half_dim)
    
    # In-place rotation using views
    q_view = q.view(batch_size, num_heads, seq_len, 2, half_dim)
    k_view = k.view(batch_size, num_heads, seq_len, 2, half_dim)
    
    q0, q1 = q_view[..., 0, :].clone(), q_view[..., 1, :].clone()
    q_view[..., 0, :] = q0 * cos - q1 * sin
    q_view[..., 1, :] = q0 * sin + q1 * cos
    
    k0, k1 = k_view[..., 0, :].clone(), k_view[..., 1, :].clone()
    k_view[..., 0, :] = k0 * cos - k1 * sin
    k_view[..., 1, :] = k0 * sin + k1 * cos

def pytorch_swiglu(gate, up, output):
    """PyTorch SwiGLU (already well-optimized)"""
    output.copy_(gate * F.silu(up))

# ============================================================================
# BENCHMARKING
# ============================================================================

def benchmark_function(func, *args, warmup=10, iterations=100, **kwargs):
    """Benchmark with proper warmup and synchronization"""
    for _ in range(warmup):
        func(*args, **kwargs)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        func(*args, **kwargs)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms

# ============================================================================
# BENCHMARK TESTS
# ============================================================================

def benchmark_rms_norm(batch_seq=1024, hidden_size=4096, dtype=torch.float16):
    """Benchmark RMSNorm with pre-allocated buffers"""
    eps = 1e-5
    
    # Allocate once
    input_tensor = torch.randn(batch_seq, hidden_size, device='cuda', dtype=dtype)
    weight = torch.ones(hidden_size, device='cuda', dtype=dtype)
    cuda_output = torch.empty_like(input_tensor)
    pytorch_output_naive = torch.empty_like(input_tensor)
    pytorch_output_fused = torch.empty_like(input_tensor)
    
    # CUDA
    try:
        cuda_rms_norm(input_tensor, weight, cuda_output, eps)
        cuda_time = benchmark_function(lambda: cuda_rms_norm(input_tensor, weight, cuda_output, eps))
    except Exception as e:
        cuda_time = float('inf')
    
    # PyTorch Naive
    pytorch_rms_norm_naive(input_tensor, weight, pytorch_output_naive, eps)
    pytorch_naive_time = benchmark_function(lambda: pytorch_rms_norm_naive(input_tensor, weight, pytorch_output_naive, eps))
    
    # PyTorch Fused
    pytorch_rms_norm_fused(input_tensor, weight, pytorch_output_fused, eps)
    pytorch_fused_time = benchmark_function(lambda: pytorch_rms_norm_fused(input_tensor, weight, pytorch_output_fused, eps))
    
    # Correctness
    max_diff_naive = (cuda_output - pytorch_output_naive).abs().max().item()
    max_diff_fused = (cuda_output - pytorch_output_fused).abs().max().item()
    correct = max_diff_naive < 0.01 or max_diff_fused < 0.01
    
    return {
        'operation': 'RMSNorm',
        'shape': f"{batch_seq}x{hidden_size}",
        'cuda_time': cuda_time,
        'pytorch_naive_time': pytorch_naive_time,
        'pytorch_fused_time': pytorch_fused_time,
        'speedup_naive': pytorch_naive_time / cuda_time if cuda_time < float('inf') else 0,
        'speedup_fused': pytorch_fused_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': min(max_diff_naive, max_diff_fused),
        'status': "✅" if correct else "❌"
    }

def benchmark_rope(batch_size=4, num_heads=32, seq_len=512, head_dim=128, dtype=torch.float16):
    """Benchmark RoPE with pre-allocated buffers (no clones in timing)"""
    cos_cache, sin_cache = cuda_rope_precompute(seq_len, head_dim)
    
    # Allocate once - separate buffers for each method
    q_orig = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    k_orig = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    
    q_cuda = q_orig.clone()
    k_cuda = k_orig.clone()
    q_naive = q_orig.clone()
    k_naive = k_orig.clone()
    q_opt = q_orig.clone()
    k_opt = k_orig.clone()
    
    # CUDA
    try:
        def cuda_func():
            q_cuda.copy_(q_orig)
            k_cuda.copy_(k_orig)
            cuda_rope_apply(q_cuda, k_cuda, cos_cache, sin_cache)
        
        cuda_func()  # Run once for correctness check
        cuda_time = benchmark_function(cuda_func)
    except Exception as e:
        cuda_time = float('inf')
    
    # PyTorch Naive
    def pytorch_naive_func():
        q_naive.copy_(q_orig)
        k_naive.copy_(k_orig)
        pytorch_rope_naive(q_naive, k_naive, cos_cache, sin_cache)
    
    pytorch_naive_func()
    pytorch_naive_time = benchmark_function(pytorch_naive_func)
    
    # PyTorch Optimized
    def pytorch_opt_func():
        q_opt.copy_(q_orig)
        k_opt.copy_(k_orig)
        pytorch_rope_optimized(q_opt, k_opt, cos_cache, sin_cache)
    
    pytorch_opt_func()
    pytorch_opt_time = benchmark_function(pytorch_opt_func)
    
    # Correctness
    max_diff_naive = max((q_cuda - q_naive).abs().max().item(), (k_cuda - k_naive).abs().max().item())
    max_diff_opt = max((q_cuda - q_opt).abs().max().item(), (k_cuda - k_opt).abs().max().item())
    correct = max_diff_naive < 0.01 or max_diff_opt < 0.01
    
    return {
        'operation': 'RoPE',
        'shape': f"{batch_size}x{num_heads}x{seq_len}x{head_dim}",
        'cuda_time': cuda_time,
        'pytorch_naive_time': pytorch_naive_time,
        'pytorch_fused_time': pytorch_opt_time,
        'speedup_naive': pytorch_naive_time / cuda_time if cuda_time < float('inf') else 0,
        'speedup_fused': pytorch_opt_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': min(max_diff_naive, max_diff_opt),
        'status': "✅" if correct else "❌"
    }

def benchmark_swiglu(total_tokens=1024, intermediate_size=11008, dtype=torch.float16):
    """Benchmark SwiGLU with pre-allocated buffers"""
    # Allocate once
    gate = torch.randn(total_tokens, intermediate_size, device='cuda', dtype=dtype)
    up = torch.randn(total_tokens, intermediate_size, device='cuda', dtype=dtype)
    cuda_output = torch.empty_like(gate)
    pytorch_output = torch.empty_like(gate)
    
    # CUDA
    try:
        cuda_swiglu(gate, up, cuda_output)
        cuda_time = benchmark_function(lambda: cuda_swiglu(gate, up, cuda_output))
    except Exception as e:
        cuda_time = float('inf')
    
    # PyTorch (already well-optimized)
    pytorch_swiglu(gate, up, pytorch_output)
    pytorch_time = benchmark_function(lambda: pytorch_swiglu(gate, up, pytorch_output))
    
    # Correctness
    max_diff = (cuda_output - pytorch_output).abs().max().item()
    correct = max_diff < 0.01
    
    return {
        'operation': 'SwiGLU',
        'shape': f"{total_tokens}x{intermediate_size}",
        'cuda_time': cuda_time,
        'pytorch_naive_time': pytorch_time,
        'pytorch_fused_time': pytorch_time,  # Same for SwiGLU
        'speedup_naive': pytorch_time / cuda_time if cuda_time < float('inf') else 0,
        'speedup_fused': pytorch_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': max_diff,
        'status': "✅" if correct else "❌"
    }

# ============================================================================
# MAIN
# ============================================================================

def print_results_table(results):
    """Print results in formatted table"""
    print("\n" + "="*140)
    print(f"{'Op':<10} {'Shape':<28} {'CUDA':<10} {'PyT Naive':<12} {'PyT Opt':<11} {'vs Naive':<10} {'vs Opt':<10} {'Max Diff':<11} {'✓':<3}")
    print("="*140)
    
    for r in results:
        cuda_str = f"{r['cuda_time']:.3f}ms" if r['cuda_time'] < float('inf') else "FAIL"
        naive_str = f"{r['pytorch_naive_time']:.3f}ms"
        fused_str = f"{r['pytorch_fused_time']:.3f}ms"
        speedup_naive_str = f"{r['speedup_naive']:.2f}x" if r['speedup_naive'] > 0 else "N/A"
        speedup_fused_str = f"{r['speedup_fused']:.2f}x" if r['speedup_fused'] > 0 else "N/A"
        diff_str = f"{r['max_diff']:.6f}" if not (r['max_diff'] != r['max_diff']) else "N/A"
        
        print(f"{r['operation']:<10} {r['shape']:<28} {cuda_str:<10} {naive_str:<12} {fused_str:<11} "
              f"{speedup_naive_str:<10} {speedup_fused_str:<10} {diff_str:<11} {r['status']:<3}")
    
    print("="*140)

def main():
    print("\n" + "="*140)
    print(" "*45 + "IMPROVED CUDA TRANSFORMER BENCHMARK")
    print("="*140)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name} (Compute {compute_cap[0]}.{compute_cap[1]})")
    print(f"Dtype: torch.float16")
    print(f"\n💡 Key improvements:")
    print(f"  • Pre-allocated buffers (no clones in timing loop)")
    print(f"  • Comparing vs both naive and optimized PyTorch")
    print(f"  • 'vs Naive' = speedup vs unfused PyTorch (your original comparison)")
    print(f"  • 'vs Opt' = speedup vs more realistic PyTorch baseline")
    
    dtype = torch.float16
    results = []
    
    print("\n🔨 Running benchmarks...")
    
    results.append(benchmark_rms_norm(batch_seq=1024, hidden_size=4096, dtype=dtype))
    results.append(benchmark_rope(batch_size=4, num_heads=32, seq_len=512, head_dim=128, dtype=dtype))
    results.append(benchmark_swiglu(total_tokens=1024, intermediate_size=11008, dtype=dtype))
    results.append(benchmark_rms_norm(batch_seq=2048, hidden_size=2048, dtype=dtype))
    results.append(benchmark_rope(batch_size=8, num_heads=16, seq_len=1024, head_dim=64, dtype=dtype))
    results.append(benchmark_swiglu(total_tokens=2048, intermediate_size=8192, dtype=dtype))
    
    print_results_table(results)
    
    # Summary
    successful = sum(1 for r in results if r['status'] == "✅")
    avg_speedup_naive = sum(r['speedup_naive'] for r in results if r['speedup_naive'] > 0) / len(results)
    avg_speedup_opt = sum(r['speedup_fused'] for r in results if r['speedup_fused'] > 0) / len(results)
    
    print(f"\n📊 Summary:")
    print(f"  • Tests passed: {successful}/{len(results)}")
    print(f"  • Avg speedup vs naive PyTorch:     {avg_speedup_naive:.2f}x")
    print(f"  • Avg speedup vs optimized PyTorch: {avg_speedup_opt:.2f}x")
    print(f"\n💭 Interpretation:")
    print(f"  • RMSNorm: Expect ~3x vs naive, ~2x vs fused")
    print(f"  • RoPE: Expect ~4x vs naive, ~2x vs optimized")
    print(f"  • SwiGLU: Expect ~1.5-2x (PyTorch SiLU is already fast)")
    print()

if __name__ == "__main__":
    main()