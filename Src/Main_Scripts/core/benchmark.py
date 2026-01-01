#!/usr/bin/env python3
# Copyright (c) 2025 MatN23. All rights reserved.

"""
Comprehensive Benchmark: PyTorch vs torch.compile vs Custom CUDA
================================================================
Benchmarks both Transformer ops (RMSNorm, RoPE, SwiGLU) and MoE ops
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import your wrappers
from cuda_opt_wrapper import (
    FusedRMSNorm as CUDARMSNorm,
    FusedRoPE as CUDARoPE,
    FusedSwiGLU as CUDASwiGLU,
    TRANSFORMER_OPS_AVAILABLE
)
from moe_cuda_wrapper import MoECUDAOps, CUDA_OPS_AVAILABLE as MOE_CUDA_AVAILABLE

# ============================================================================
# PYTORCH BASELINE IMPLEMENTATIONS
# ============================================================================

class PyTorchRMSNorm(nn.Module):
    """PyTorch RMSNorm baseline"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class PyTorchRoPE(nn.Module):
    """PyTorch RoPE baseline"""
    def __init__(self, head_dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)


class PyTorchSwiGLU(nn.Module):
    """PyTorch SwiGLU baseline"""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class PyTorchMoEGating:
    """PyTorch MoE gating baseline"""
    @staticmethod
    def topk_gating(gate_logits, k, temperature=1.0):
        scaled = gate_logits / temperature
        values, indices = torch.topk(scaled, k, dim=-1)
        weights = F.softmax(values, dim=-1)
        return indices, weights


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    name: str
    mean_ms: float
    std_ms: float
    throughput: float  # items/sec
    speedup: float = 1.0


def benchmark_op(func, *args, warmup=10, runs=100, name="operation") -> BenchmarkResult:
    """
    Benchmark a single operation.
    
    Args:
        func: Function to benchmark
        *args: Arguments to pass to func
        warmup: Number of warmup iterations
        runs: Number of benchmark iterations
        name: Name for display
        
    Returns:
        BenchmarkResult with timing statistics
    """
    device = args[0].device if torch.is_tensor(args[0]) else torch.device('cuda')
    
    # Warmup
    for _ in range(warmup):
        _ = func(*args)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = func(*args)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    mean_ms = sum(times) / len(times)
    std_ms = (sum((t - mean_ms) ** 2 for t in times) / len(times)) ** 0.5
    
    # Calculate throughput (operations per second)
    throughput = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    
    return BenchmarkResult(name, mean_ms, std_ms, throughput)


# ============================================================================
# TRANSFORMER OPS BENCHMARKS
# ============================================================================

def benchmark_rmsnorm(batch_size: int, seq_len: int, hidden_size: int, 
                      runs: int = 100, dtype=torch.float32) -> Dict[str, BenchmarkResult]:
    """Benchmark RMSNorm implementations"""
    print(f"\n{'='*70}")
    print(f"RMSNorm: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, dtype={dtype}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    results = {}
    
    # 1. PyTorch baseline
    print("Testing PyTorch baseline...")
    norm_pytorch = PyTorchRMSNorm(hidden_size).to(device).to(dtype)
    results['pytorch'] = benchmark_op(norm_pytorch, x, runs=runs, name="PyTorch")
    
    # 2. torch.compile
    if hasattr(torch, 'compile'):
        print("Testing torch.compile...")
        norm_compiled = torch.compile(norm_pytorch, mode='max-autotune')
        # Extra warmup for compile
        for _ in range(5):
            _ = norm_compiled(x)
        torch.cuda.synchronize()
        results['compiled'] = benchmark_op(norm_compiled, x, runs=runs, name="torch.compile")
    
    # 3. Custom CUDA
    if TRANSFORMER_OPS_AVAILABLE and device.type == 'cuda':
        print("Testing Custom CUDA...")
        norm_cuda = CUDARMSNorm(hidden_size).to(device)
        results['cuda'] = benchmark_op(norm_cuda, x, runs=runs, name="Custom CUDA")
    
    # Calculate speedups
    baseline = results['pytorch'].mean_ms
    for name, result in results.items():
        result.speedup = baseline / result.mean_ms
    
    return results


def benchmark_rope(batch_size: int, num_heads: int, seq_len: int, 
                   head_dim: int, runs: int = 100, dtype=torch.float32) -> Dict[str, BenchmarkResult]:
    """Benchmark RoPE implementations"""
    print(f"\n{'='*70}")
    print(f"RoPE: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}, dtype={dtype}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    # 1. PyTorch baseline
    print("Testing PyTorch baseline...")
    rope_pytorch = PyTorchRoPE(head_dim).to(device)
    def pytorch_forward():
        return rope_pytorch(seq_len, device)
    results['pytorch'] = benchmark_op(pytorch_forward, runs=runs, name="PyTorch")
    
    # 2. torch.compile
    if hasattr(torch, 'compile'):
        print("Testing torch.compile...")
        rope_compiled = torch.compile(rope_pytorch, mode='max-autotune')
        def compiled_forward():
            return rope_compiled(seq_len, device)
        # Extra warmup
        for _ in range(5):
            _ = compiled_forward()
        torch.cuda.synchronize()
        results['compiled'] = benchmark_op(compiled_forward, runs=runs, name="torch.compile")
    
    # 3. Custom CUDA
    if TRANSFORMER_OPS_AVAILABLE and device.type == 'cuda':
        print("Testing Custom CUDA...")
        rope_cuda = CUDARoPE(head_dim).to(device)
        def cuda_forward():
            return rope_cuda(seq_len, device)
        results['cuda'] = benchmark_op(cuda_forward, runs=runs, name="Custom CUDA")
    
    # Calculate speedups
    baseline = results['pytorch'].mean_ms
    for name, result in results.items():
        result.speedup = baseline / result.mean_ms
    
    return results


def benchmark_swiglu(batch_size: int, seq_len: int, hidden_size: int, 
                     intermediate_size: int, runs: int = 100, dtype=torch.float32) -> Dict[str, BenchmarkResult]:
    """Benchmark SwiGLU implementations"""
    print(f"\n{'='*70}")
    print(f"SwiGLU: batch={batch_size}, seq_len={seq_len}, hidden={hidden_size}, intermediate={intermediate_size}, dtype={dtype}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    results = {}
    
    # 1. PyTorch baseline
    print("Testing PyTorch baseline...")
    swiglu_pytorch = PyTorchSwiGLU(hidden_size, intermediate_size).to(device).to(dtype)
    results['pytorch'] = benchmark_op(swiglu_pytorch, x, runs=runs, name="PyTorch")
    
    # 2. torch.compile
    if hasattr(torch, 'compile'):
        print("Testing torch.compile...")
        swiglu_compiled = torch.compile(swiglu_pytorch, mode='max-autotune')
        # Extra warmup
        for _ in range(5):
            _ = swiglu_compiled(x)
        torch.cuda.synchronize()
        results['compiled'] = benchmark_op(swiglu_compiled, x, runs=runs, name="torch.compile")
    
    # 3. Custom CUDA
    if TRANSFORMER_OPS_AVAILABLE and device.type == 'cuda':
        print("Testing Custom CUDA...")
        swiglu_cuda = CUDASwiGLU(hidden_size, intermediate_size).to(device)
        results['cuda'] = benchmark_op(swiglu_cuda, x, runs=runs, name="Custom CUDA")
    
    # Calculate speedups
    baseline = results['pytorch'].mean_ms
    for name, result in results.items():
        result.speedup = baseline / result.mean_ms
    
    return results


# ============================================================================
# MOE OPS BENCHMARKS
# ============================================================================

def benchmark_moe_gating(num_tokens: int, num_experts: int, k: int, 
                         runs: int = 100, dtype=torch.float32) -> Dict[str, BenchmarkResult]:
    """Benchmark MoE top-k gating"""
    print(f"\n{'='*70}")
    print(f"MoE Gating: tokens={num_tokens}, experts={num_experts}, k={k}, dtype={dtype}")
    print(f"{'='*70}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gate_logits = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
    
    results = {}
    
    # 1. PyTorch baseline
    print("Testing PyTorch baseline...")
    def pytorch_gating():
        return PyTorchMoEGating.topk_gating(gate_logits, k)
    results['pytorch'] = benchmark_op(pytorch_gating, runs=runs, name="PyTorch")
    
    # 2. torch.compile
    if hasattr(torch, 'compile'):
        print("Testing torch.compile...")
        compiled_gating = torch.compile(PyTorchMoEGating.topk_gating, mode='max-autotune')
        def compiled_forward():
            return compiled_gating(gate_logits, k)
        # Extra warmup
        for _ in range(5):
            _ = compiled_forward()
        torch.cuda.synchronize()
        results['compiled'] = benchmark_op(compiled_forward, runs=runs, name="torch.compile")
    
    # 3. Custom CUDA
    if MOE_CUDA_AVAILABLE and device.type == 'cuda':
        print("Testing Custom CUDA...")
        def cuda_gating():
            return MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
        results['cuda'] = benchmark_op(cuda_gating, runs=runs, name="Custom CUDA")
    
    # Calculate speedups
    baseline = results['pytorch'].mean_ms
    for name, result in results.items():
        result.speedup = baseline / result.mean_ms
    
    return results


# ============================================================================
# RESULTS DISPLAY
# ============================================================================

def print_results(title: str, results: Dict[str, BenchmarkResult]):
    """Pretty print benchmark results"""
    print(f"\n{title}")
    print("-" * 70)
    print(f"{'Implementation':<20} {'Time (ms)':<15} {'Std (ms)':<15} {'Speedup':<10}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{result.name:<20} {result.mean_ms:>10.4f}     {result.std_ms:>10.4f}     {result.speedup:>6.2f}x")
    
    print("-" * 70)
    
    # Highlight winner
    fastest = min(results.values(), key=lambda r: r.mean_ms)
    print(f"🏆 Fastest: {fastest.name} ({fastest.mean_ms:.4f}ms)")
    print()


def print_summary(all_results: Dict[str, Dict[str, BenchmarkResult]]):
    """Print summary of all benchmarks"""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    
    for op_name, results in all_results.items():
        print(f"\n{op_name}:")
        for impl_name, result in results.items():
            print(f"  {result.name:<20} {result.mean_ms:>8.4f}ms  ({result.speedup:>5.2f}x)")
    
    print("\n" + "="*70)
    print("OVERALL WINNERS")
    print("="*70)
    
    for op_name, results in all_results.items():
        fastest = min(results.values(), key=lambda r: r.mean_ms)
        print(f"{op_name:<30} 🏆 {fastest.name:<15} ({fastest.speedup:.2f}x speedup)")
    
    print("="*70 + "\n")


# ============================================================================
# MAIN BENCHMARK SUITE
# ============================================================================

def run_full_benchmark(batch_size=4, seq_len=512, hidden_size=768, num_heads=12,
                      num_experts=8, k=2, runs=100, dtype=torch.float32):
    """Run complete benchmark suite"""
    
    print("\n" + "="*70)
    print("COMPREHENSIVE CUDA OPERATIONS BENCHMARK")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")
    print(f"Transformer CUDA ops: {'✅ Available' if TRANSFORMER_OPS_AVAILABLE else '❌ Not available'}")
    print(f"MoE CUDA ops: {'✅ Available' if MOE_CUDA_AVAILABLE else '❌ Not available'}")
    print(f"Data type: {dtype}")
    print("="*70)
    
    all_results = {}
    
    # Transformer ops
    head_dim = hidden_size // num_heads
    intermediate_size = hidden_size * 4
    
    all_results['RMSNorm'] = benchmark_rmsnorm(batch_size, seq_len, hidden_size, runs, dtype)
    print_results("RMSNorm Results", all_results['RMSNorm'])
    
    all_results['RoPE'] = benchmark_rope(batch_size, num_heads, seq_len, head_dim, runs, dtype)
    print_results("RoPE Results", all_results['RoPE'])
    
    all_results['SwiGLU'] = benchmark_swiglu(batch_size, seq_len, hidden_size, intermediate_size, runs, dtype)
    print_results("SwiGLU Results", all_results['SwiGLU'])
    
    # MoE ops
    num_tokens = batch_size * seq_len
    all_results['MoE Gating'] = benchmark_moe_gating(num_tokens, num_experts, k, runs, dtype)
    print_results("MoE Gating Results", all_results['MoE Gating'])
    
    # Print summary
    print_summary(all_results)
    
    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    # ============================================================================
    # HARDCODED BENCHMARK CONFIGURATIONS
    # ============================================================================
    
    print("\n" + "="*70)
    print("🚀 RUNNING ALL BENCHMARK CONFIGURATIONS")
    print("="*70 + "\n")
    
    # Configuration 1: Small (fast, for testing)
    print("\n" + "🔹"*35)
    print("CONFIGURATION 1: SMALL (Fast Testing)")
    print("🔹"*35)
    run_full_benchmark(
        batch_size=2,
        seq_len=128,
        hidden_size=512,
        num_heads=8,
        num_experts=4,
        k=2,
        runs=50,
        dtype=torch.float32
    )
    
    # Configuration 2: Medium (typical training)
    print("\n" + "🔸"*35)
    print("CONFIGURATION 2: MEDIUM (Typical Training)")
    print("🔸"*35)
    run_full_benchmark(
        batch_size=4,
        seq_len=512,
        hidden_size=768,
        num_heads=12,
        num_experts=8,
        k=2,
        runs=100,
        dtype=torch.float32
    )
    
    # Configuration 3: Large (production scale)
    print("\n" + "🔶"*35)
    print("CONFIGURATION 3: LARGE (Production Scale)")
    print("🔶"*35)
    run_full_benchmark(
        batch_size=8,
        seq_len=1024,
        hidden_size=1024,
        num_heads=16,
        num_experts=16,
        k=2,
        runs=50,
        dtype=torch.float32
    )
    
    # Configuration 4: FP16 (mixed precision)
    print("\n" + "💎"*35)
    print("CONFIGURATION 4: FP16 (Mixed Precision)")
    print("💎"*35)
    run_full_benchmark(
        batch_size=4,
        seq_len=512,
        hidden_size=768,
        num_heads=12,
        num_experts=8,
        k=2,
        runs=100,
        dtype=torch.float16
    )
    
    # Configuration 5: BF16 (brain float)
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        print("\n" + "🧠"*35)
        print("CONFIGURATION 5: BF16 (Brain Float)")
        print("🧠"*35)
        run_full_benchmark(
            batch_size=4,
            seq_len=512,
            hidden_size=768,
            num_heads=12,
            num_experts=8,
            k=2,
            runs=100,
            dtype=torch.bfloat16
        )
    
    print("\n" + "="*70)
    print("✅ ALL BENCHMARK CONFIGURATIONS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()