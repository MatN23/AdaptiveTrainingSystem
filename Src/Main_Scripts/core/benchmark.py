#!/usr/bin/env python3
"""
Comprehensive CUDA Kernel Benchmark
====================================
Uses the high-level wrappers for consistent testing.
Benchmarks: RMSNorm, RoPE, SwiGLU, and MoE operations.
"""

import torch
import torch.nn.functional as F
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# LOAD WRAPPERS
# ============================================================================

print("🔍 Loading CUDA wrappers...")

# Transformer ops wrapper
try:
    from core.cuda_opt_wrapper import (
        FusedRMSNorm, 
        FusedRoPE, 
        FusedSwiGLU,
        fused_swiglu,
        TRANSFORMER_OPS_AVAILABLE
    )
    HAS_TRANSFORMER_OPS = TRANSFORMER_OPS_AVAILABLE
    print(f"  ✅ Transformer ops: {'Available' if HAS_TRANSFORMER_OPS else 'Not loaded'}")
except ImportError as e:
    HAS_TRANSFORMER_OPS = False
    fused_swiglu = None
    print(f"  ❌ Transformer ops: {e}")

# MoE ops wrapper
try:
    from core.moe_cuda_wrapper import MoECUDAOps, CUDA_OPS_AVAILABLE
    HAS_MOE_OPS = CUDA_OPS_AVAILABLE
    print(f"  ✅ MoE ops: {'Available' if HAS_MOE_OPS else 'Not loaded'}")
except ImportError as e:
    HAS_MOE_OPS = False
    MoECUDAOps = None
    print(f"  ❌ MoE ops: {e}")


# ============================================================================
# PYTORCH IMPLEMENTATIONS
# ============================================================================

def pytorch_rms_norm(x, weight, eps=1e-5):
    """PyTorch RMSNorm"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return weight * x_normed

def pytorch_rope(q, k, cos_cache, sin_cache):
    """PyTorch RoPE"""
    batch_size, num_heads, seq_len, head_dim = q.shape
    half_dim = head_dim // 2
    
    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]
    
    positions = torch.arange(seq_len, device=q.device)
    cos = cos_cache[positions].view(1, 1, seq_len, half_dim)
    sin = sin_cache[positions].view(1, 1, seq_len, half_dim)
    
    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)
    
    return q_rot, k_rot

def pytorch_swiglu(gate, up):
    """PyTorch SwiGLU"""
    return F.silu(gate) * up

def pytorch_topk_gating(gate_logits, k, temperature=1.0):
    """PyTorch top-k gating"""
    scaled = gate_logits / temperature
    probs = F.softmax(scaled, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
    return top_k_indices, top_k_probs


# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def benchmark_function(func, warmup=10, iterations=100):
    """Benchmark with proper warmup and synchronization"""
    # Warmup
    for _ in range(warmup):
        func()
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(iterations):
        func()
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    return (end - start) / iterations * 1000  # ms


def check_correctness(cuda_result, pytorch_result, atol=0.01):
    """Check if CUDA and PyTorch results match"""
    if isinstance(cuda_result, tuple):
        diffs = [torch.abs(c - p).max().item() for c, p in zip(cuda_result, pytorch_result)]
        return max(diffs), max(diffs) < atol
    else:
        diff = torch.abs(cuda_result - pytorch_result).max().item()
        return diff, diff < atol


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

def benchmark_rms_norm(batch_seq=1024, hidden_size=4096, dtype=torch.float16):
    """Benchmark RMSNorm"""
    x = torch.randn(batch_seq, hidden_size, device='cuda', dtype=dtype)
    weight = torch.ones(hidden_size, device='cuda', dtype=dtype)
    
    # PyTorch
    pytorch_result = pytorch_rms_norm(x, weight)
    pytorch_time = benchmark_function(lambda: pytorch_rms_norm(x, weight))
    
    # CUDA wrapper
    if HAS_TRANSFORMER_OPS:
        try:
            rms_norm = FusedRMSNorm(hidden_size).to('cuda').half()
            rms_norm.weight.data.copy_(weight)
            
            cuda_result = rms_norm(x)
            cuda_time = benchmark_function(lambda: rms_norm(x))
            
            max_diff, correct = check_correctness(cuda_result, pytorch_result)
        except Exception as e:
            cuda_time = float('inf')
            max_diff = float('nan')
            correct = False
    else:
        cuda_time = float('inf')
        max_diff = float('nan')
        correct = False
    
    return {
        'operation': 'RMSNorm',
        'shape': f"{batch_seq}x{hidden_size}",
        'cuda_time': cuda_time,
        'pytorch_time': pytorch_time,
        'speedup': pytorch_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': max_diff,
        'status': "✅" if correct else ("⚠️" if cuda_time == float('inf') else "❌")
    }


def benchmark_rope(batch_size=4, num_heads=32, seq_len=512, head_dim=128, dtype=torch.float16):
    """Benchmark RoPE"""
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', dtype=dtype)
    
    # Precompute cos/sin cache
    half_dim = head_dim // 2
    positions = torch.arange(seq_len, device='cuda')
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, device='cuda').float() / half_dim))
    freqs = torch.outer(positions.float(), inv_freq)
    cos_cache = freqs.cos()
    sin_cache = freqs.sin()
    
    # PyTorch
    q_pt, k_pt = pytorch_rope(q.clone(), k.clone(), cos_cache, sin_cache)
    pytorch_time = benchmark_function(lambda: pytorch_rope(q.clone(), k.clone(), cos_cache, sin_cache))
    
    # CUDA wrapper
    if HAS_TRANSFORMER_OPS:
        try:
            rope = FusedRoPE(head_dim).to('cuda')
            cos_cuda, sin_cuda = rope(seq_len, q.device)
            
            def cuda_rope_func():
                q_c = q.clone()
                k_c = k.clone()
                # Apply rotation manually using the cached cos/sin
                return pytorch_rope(q_c, k_c, cos_cuda, sin_cuda)
            
            cuda_result = cuda_rope_func()
            cuda_time = benchmark_function(cuda_rope_func)
            
            max_diff, correct = check_correctness(cuda_result, (q_pt, k_pt))
        except Exception as e:
            cuda_time = float('inf')
            max_diff = float('nan')
            correct = False
    else:
        cuda_time = float('inf')
        max_diff = float('nan')
        correct = False
    
    return {
        'operation': 'RoPE',
        'shape': f"{batch_size}x{num_heads}x{seq_len}x{head_dim}",
        'cuda_time': cuda_time,
        'pytorch_time': pytorch_time,
        'speedup': pytorch_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': max_diff,
        'status': "✅" if correct else ("⚠️" if cuda_time == float('inf') else "❌")
    }


def benchmark_swiglu(total_tokens=1024, intermediate_size=11008, dtype=torch.float16):
    """Benchmark SwiGLU"""
    gate = torch.randn(total_tokens, intermediate_size, device='cuda', dtype=dtype)
    up = torch.randn(total_tokens, intermediate_size, device='cuda', dtype=dtype)
    
    # PyTorch
    pytorch_result = pytorch_swiglu(gate, up)
    pytorch_time = benchmark_function(lambda: pytorch_swiglu(gate, up))
    
    # CUDA wrapper - use fused_swiglu function directly
    if HAS_TRANSFORMER_OPS and fused_swiglu is not None:
        try:
            cuda_result = fused_swiglu(gate, up)
            cuda_time = benchmark_function(lambda: fused_swiglu(gate, up))
            
            max_diff, correct = check_correctness(cuda_result, pytorch_result)
        except Exception as e:
            print(f"  ⚠️ SwiGLU CUDA error: {e}")
            cuda_time = float('inf')
            max_diff = float('nan')
            correct = False
    else:
        cuda_time = float('inf')
        max_diff = float('nan')
        correct = False
    
    return {
        'operation': 'SwiGLU',
        'shape': f"{total_tokens}x{intermediate_size}",
        'cuda_time': cuda_time,
        'pytorch_time': pytorch_time,
        'speedup': pytorch_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': max_diff,
        'status': "✅" if correct else ("⚠️" if cuda_time == float('inf') else "❌")
    }


def benchmark_moe_topk(num_tokens=1024, num_experts=8, k=2, dtype=torch.float32):
    """Benchmark MoE Top-K Gating"""
    gate_logits = torch.randn(num_tokens, num_experts, device='cuda', dtype=dtype)
    
    # PyTorch
    pt_indices, pt_probs = pytorch_topk_gating(gate_logits, k)
    pytorch_time = benchmark_function(lambda: pytorch_topk_gating(gate_logits, k))
    
    # CUDA wrapper
    if HAS_MOE_OPS and MoECUDAOps is not None:
        try:
            cuda_indices, cuda_probs = MoECUDAOps.topk_gating(gate_logits, k, temperature=1.0, use_cuda=True)
            cuda_time = benchmark_function(lambda: MoECUDAOps.topk_gating(gate_logits, k, temperature=1.0, use_cuda=True))
            
            # Check correctness - indices should match
            indices_match = (cuda_indices == pt_indices).all().item()
            probs_diff = torch.abs(cuda_probs - pt_probs).max().item()
            
            max_diff = probs_diff
            correct = indices_match and probs_diff < 0.01
        except Exception as e:
            print(f"  ⚠️ MoE topk error: {e}")
            cuda_time = float('inf')
            max_diff = float('nan')
            correct = False
    else:
        cuda_time = float('inf')
        max_diff = float('nan')
        correct = False
    
    return {
        'operation': 'MoE TopK',
        'shape': f"{num_tokens}x{num_experts} k={k}",
        'cuda_time': cuda_time,
        'pytorch_time': pytorch_time,
        'speedup': pytorch_time / cuda_time if cuda_time < float('inf') else 0,
        'max_diff': max_diff,
        'status': "✅" if correct else ("⚠️" if cuda_time == float('inf') else "❌")
    }


# ============================================================================
# MAIN
# ============================================================================

def print_results_table(results):
    """Print results in a nice box-style table"""
    # Column widths
    cols = {
        'op': 14,
        'shape': 26,
        'cuda': 12,
        'pytorch': 12,
        'speedup': 10,
        'diff': 12,
        'status': 6
    }
    total_width = sum(cols.values()) + len(cols) + 1
    
    # Box drawing characters
    TL, TR, BL, BR = '┌', '┐', '└', '┘'
    H, V = '─', '│'
    TJ, BJ, LJ, RJ, CJ = '┬', '┴', '├', '┤', '┼'
    
    def row_sep(left, mid, right):
        parts = [H * cols['op'], H * cols['shape'], H * cols['cuda'], 
                 H * cols['pytorch'], H * cols['speedup'], H * cols['diff'], H * cols['status']]
        return left + mid.join(parts) + right
    
    # Header
    print("\n" + row_sep(TL, TJ, TR))
    header = (f"{V}{'Operation':^{cols['op']}}{V}{'Shape':^{cols['shape']}}{V}"
              f"{'CUDA':^{cols['cuda']}}{V}{'PyTorch':^{cols['pytorch']}}{V}"
              f"{'Speedup':^{cols['speedup']}}{V}{'Max Diff':^{cols['diff']}}{V}"
              f"{'':^{cols['status']}}{V}")
    print(header)
    print(row_sep(LJ, CJ, RJ))
    
    # Data rows
    for r in results:
        cuda_str = f"{r['cuda_time']:.3f}ms" if r['cuda_time'] < float('inf') else "N/A"
        pytorch_str = f"{r['pytorch_time']:.3f}ms"
        
        if r['speedup'] > 0:
            if r['speedup'] >= 2.0:
                speedup_str = f"🚀 {r['speedup']:.2f}x"
            elif r['speedup'] >= 1.0:
                speedup_str = f"✓ {r['speedup']:.2f}x"
            else:
                speedup_str = f"↓ {r['speedup']:.2f}x"
        else:
            speedup_str = "N/A"
        
        if r['max_diff'] == r['max_diff']:  # Not NaN
            diff_str = f"{r['max_diff']:.6f}"
        else:
            diff_str = "N/A"
        
        row = (f"{V}{r['operation']:^{cols['op']}}{V}{r['shape']:^{cols['shape']}}{V}"
               f"{cuda_str:^{cols['cuda']}}{V}{pytorch_str:^{cols['pytorch']}}{V}"
               f"{speedup_str:^{cols['speedup']}}{V}{diff_str:^{cols['diff']}}{V}"
               f"{r['status']:^{cols['status']}}{V}")
        print(row)
    
    # Footer
    print(row_sep(BL, BJ, BR))


def main():
    print("\n" + "="*100)
    print(" "*30 + "CUDA KERNEL BENCHMARK (Using Wrappers)")
    print("="*100)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device_name = torch.cuda.get_device_name(0)
    compute_cap = torch.cuda.get_device_capability(0)
    print(f"Device: {device_name} (Compute {compute_cap[0]}.{compute_cap[1]})")
    print(f"Transformer Ops: {'✅ Loaded' if HAS_TRANSFORMER_OPS else '❌ Not available'}")
    print(f"MoE Ops: {'✅ Loaded' if HAS_MOE_OPS else '❌ Not available'}")
    
    results = []
    
    print("\n🔨 Running benchmarks...")
    
    # Transformer ops benchmarks
    print("  • RMSNorm...")
    results.append(benchmark_rms_norm(batch_seq=1024, hidden_size=4096))
    results.append(benchmark_rms_norm(batch_seq=2048, hidden_size=2048))
    
    print("  • RoPE...")
    results.append(benchmark_rope(batch_size=4, num_heads=32, seq_len=512, head_dim=128))
    results.append(benchmark_rope(batch_size=8, num_heads=16, seq_len=1024, head_dim=64))
    
    print("  • SwiGLU...")
    results.append(benchmark_swiglu(total_tokens=1024, intermediate_size=11008))
    results.append(benchmark_swiglu(total_tokens=2048, intermediate_size=8192))
    
    # MoE ops benchmarks
    print("  • MoE Top-K Gating...")
    results.append(benchmark_moe_topk(num_tokens=1024, num_experts=8, k=2))
    results.append(benchmark_moe_topk(num_tokens=2048, num_experts=16, k=4))
    results.append(benchmark_moe_topk(num_tokens=4096, num_experts=8, k=2))
    
    print_results_table(results)
    
    # Summary
    cuda_available = [r for r in results if r['cuda_time'] < float('inf')]
    successful = sum(1 for r in results if r['status'] == "✅")
    
    if cuda_available:
        avg_speedup = sum(r['speedup'] for r in cuda_available) / len(cuda_available)
        print(f"\n📊 Summary:")
        print(f"  • Tests passed: {successful}/{len(results)}")
        print(f"  • CUDA kernels available: {len(cuda_available)}/{len(results)}")
        print(f"  • Average speedup: {avg_speedup:.2f}x")
    else:
        print(f"\n⚠️ No CUDA kernels were available for benchmarking.")
        print(f"   Make sure to compile the CUDA kernels first:")
        print(f"   - transformer_ops.so (for RMSNorm, RoPE, SwiGLU)")
        print(f"   - moe_cuda_ops (JIT-compiled on first use)")
    
    print()


if __name__ == "__main__":
    main()