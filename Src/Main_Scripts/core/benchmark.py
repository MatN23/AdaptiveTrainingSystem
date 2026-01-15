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
                # FIXED: Call the fused kernel wrapper
                return rope.apply_rotary_pos_emb(q_c, k_c, position_offset=0)
            
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
# FULL TRAINING STEP BENCHMARK
# ============================================================================

def benchmark_full_training_step(batch_size=8, seq_len=512, hidden_size=512, 
                                  num_layers=4, vocab_size=32000, 
                                  iterations=50, warmup=10):
    """
    Benchmark FULL training step with transformer ops: forward → loss → backward → optimizer.
    
    This tests the realistic impact of CUDA kernels vs PyTorch in actual training.
    """
    print(f"\n{'='*80}")
    print("FULL TRAINING STEP BENCHMARK (Realistic)")
    print(f"{'='*80}")
    print(f"  Batch: {batch_size} | Seq: {seq_len} | Hidden: {hidden_size} | Layers: {num_layers}")
    print(f"  Tokens per step: {batch_size * seq_len:,}")
    print(f"  Iterations: {iterations} | Warmup: {warmup}")
    
    device = torch.device('cuda')
    
    # Build model with/without CUDA ops
    class TransformerBlock(torch.nn.Module):
        def __init__(self, hidden_size, use_cuda_ops=False):
            super().__init__()
            self.use_cuda_ops = use_cuda_ops
            
            if use_cuda_ops and HAS_TRANSFORMER_OPS:
                self.norm1 = FusedRMSNorm(hidden_size).to(device)
                self.norm2 = FusedRMSNorm(hidden_size).to(device)
            else:
                self.norm1 = torch.nn.LayerNorm(hidden_size)
                self.norm2 = torch.nn.LayerNorm(hidden_size)
            
            self.attn = torch.nn.MultiheadAttention(hidden_size, 8, batch_first=True)
            self.ffn_gate = torch.nn.Linear(hidden_size, hidden_size * 4, bias=False)
            self.ffn_up = torch.nn.Linear(hidden_size, hidden_size * 4, bias=False)
            self.ffn_down = torch.nn.Linear(hidden_size * 4, hidden_size, bias=False)
        
        def forward(self, x):
            # Attention
            h = self.norm1(x)
            h, _ = self.attn(h, h, h, need_weights=False)
            x = x + h
            
            # FFN with SwiGLU
            h = self.norm2(x)
            gate = self.ffn_gate(h)
            up = self.ffn_up(h)
            
            if self.use_cuda_ops and HAS_TRANSFORMER_OPS and fused_swiglu is not None:
                h = fused_swiglu(gate, up)
            else:
                h = F.silu(gate) * up
            
            h = self.ffn_down(h)
            return x + h
    
    class SimpleTransformer(torch.nn.Module):
        def __init__(self, use_cuda_ops=False):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.layers = torch.nn.ModuleList([
                TransformerBlock(hidden_size, use_cuda_ops) for _ in range(num_layers)
            ])
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        
        def forward(self, x):
            h = self.embed(x)
            for layer in self.layers:
                h = layer(h)
            return self.lm_head(h)
    
    # Test data
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    results = []
    
    for mode, use_cuda in [("PyTorch", False), ("CUDA Ops", True)]:
        if use_cuda and not HAS_TRANSFORMER_OPS:
            print(f"\n  {mode}: Skipped (ops not available)")
            continue
        
        print(f"\n  {mode}: Running...")
        
        model = SimpleTransformer(use_cuda_ops=use_cuda).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Warmup
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        torch.cuda.synchronize()
        
        # Measure
        times = []
        for i in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)
            
            if (i + 1) % 10 == 0:
                throughput = (batch_size * seq_len) / (elapsed_ms / 1000)
                print(f"    Iter {i+1}/{iterations}: {elapsed_ms:.2f}ms, {throughput:.0f} tok/s")
        
        import numpy as np
        mean_time = np.mean(times)
        throughput = (batch_size * seq_len) / (mean_time / 1000)
        
        results.append({
            'mode': mode,
            'mean_ms': mean_time,
            'throughput': throughput
        })
        
        del model, optimizer
        torch.cuda.empty_cache()
    
    # Compare
    if len(results) == 2:
        pytorch, cuda = results
        speedup = pytorch['mean_ms'] / cuda['mean_ms']
        diff = cuda['throughput'] - pytorch['throughput']
        
        print(f"\n{'='*80}")
        print(f"RESULTS: Full Training Step")
        print(f"{'='*80}")
        print(f"  PyTorch:   {pytorch['mean_ms']:.2f}ms, {pytorch['throughput']:.0f} tok/s")
        print(f"  CUDA Ops:  {cuda['mean_ms']:.2f}ms, {cuda['throughput']:.0f} tok/s")
        print(f"\n  Speedup: {speedup:.2f}x ({'+' if diff > 0 else ''}{diff:.0f} tok/s)")
        
        if speedup > 1.0:
            print(f"  🏆 CUDA WINS by {speedup:.2f}x")
        else:
            print(f"  🏆 PyTorch WINS by {1/speedup:.2f}x")
        print(f"{'='*80}")
    
    return results


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
    import argparse
    parser = argparse.ArgumentParser(description='Core CUDA Kernel Benchmark')
    parser.add_argument('--training-only', action='store_true', help='Only run full training benchmark')
    parser.add_argument('--skip-training', action='store_true', help='Skip full training benchmark')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training benchmark')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length')
    parser.add_argument('--hidden-size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--iterations', type=int, default=50, help='Number of iterations')
    args = parser.parse_args()
    
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
    
    # Full training benchmark FIRST (most important)
    if not args.skip_training or args.training_only:
        benchmark_full_training_step(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            iterations=args.iterations
        )
    
    if args.training_only:
        return
    
    results = []
    
    print("\n🔨 Running isolated kernel benchmarks...")
    
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