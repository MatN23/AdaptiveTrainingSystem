# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

"""
CUDA Kernel Performance Benchmark with torch.compile
Tests speed differences between PyTorch, torch.compile, and custom CUDA.

Usage:
    python benchmark_cuda_kernels.py
    python benchmark_cuda_kernels.py --iterations 1000
    python benchmark_cuda_kernels.py --vocab-size 50000 --batch-size 32
"""

import torch
import torch.nn.functional as F
import time
import argparse
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Try to import custom kernels
try:
    from cuda_kernels import FusedLoss, FusedGradClip, CUSTOM_KERNELS_AVAILABLE
    KERNELS_AVAILABLE = CUSTOM_KERNELS_AVAILABLE
except ImportError:
    print("⚠️  Could not import cuda_kernels module")
    KERNELS_AVAILABLE = False


class BenchmarkResults:
    """Store and analyze benchmark results."""
    
    def __init__(self, name: str):
        self.name = name
        self.times = []
        
    def add_time(self, elapsed: float):
        """Add a timing measurement in milliseconds."""
        self.times.append(elapsed)
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary."""
        if not self.times:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}
        
        times = np.array(self.times)
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'median': float(np.median(times))
        }
    
    def print_summary(self):
        """Print formatted summary."""
        stats = self.get_stats()
        print(f"\n{self.name}:")
        print(f"  Mean:   {stats['mean']:8.3f} ms")
        print(f"  Median: {stats['median']:8.3f} ms")
        print(f"  Std:    {stats['std']:8.3f} ms")
        print(f"  Min:    {stats['min']:8.3f} ms")
        print(f"  Max:    {stats['max']:8.3f} ms")


def pytorch_cross_entropy_accuracy(logits: torch.Tensor, 
                                   labels: torch.Tensor,
                                   pad_token_id: int = -100) -> Dict[str, torch.Tensor]:
    """Reference PyTorch implementation."""
    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
    
    mask = (labels != pad_token_id).float()
    valid_token_count = mask.sum()
    
    if valid_token_count == 0:
        return {
            'loss': torch.tensor(0.0, device=logits.device),
            'accuracy': torch.tensor(0.0, device=logits.device),
            'valid_tokens': torch.tensor(0, device=logits.device)
        }
    
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = (predictions == labels).float()
        masked_correct = correct_predictions * mask
        accuracy = masked_correct.sum() / valid_token_count
    
    loss_per_token = F.cross_entropy(logits, labels, reduction='none')
    masked_loss_sum = (loss_per_token * mask).sum()
    loss = masked_loss_sum / valid_token_count
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'valid_tokens': valid_token_count
    }


# Compile the PyTorch version
@torch.compile(mode='max-autotune', fullgraph=True)
def compiled_cross_entropy_accuracy(logits: torch.Tensor,
                                    labels: torch.Tensor,
                                    pad_token_id: int = -100) -> Dict[str, torch.Tensor]:
    """torch.compile optimized version."""
    if logits.dim() == 3:
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)
    
    mask = (labels != pad_token_id).float()
    valid_token_count = mask.sum()
    
    if valid_token_count == 0:
        return {
            'loss': torch.tensor(0.0, device=logits.device),
            'accuracy': torch.tensor(0.0, device=logits.device),
            'valid_tokens': torch.tensor(0, device=logits.device)
        }
    
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=-1)
        correct_predictions = (predictions == labels).float()
        masked_correct = correct_predictions * mask
        accuracy = masked_correct.sum() / valid_token_count
    
    loss_per_token = F.cross_entropy(logits, labels, reduction='none')
    masked_loss_sum = (loss_per_token * mask).sum()
    loss = masked_loss_sum / valid_token_count
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'valid_tokens': valid_token_count
    }


def pytorch_grad_clip(parameters, max_norm: float) -> float:
    """Reference PyTorch gradient clipping."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()


@torch.compile(mode='max-autotune')
def compiled_grad_clip(parameters, max_norm: float) -> float:
    """torch.compile optimized gradient clipping."""
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()


def benchmark_loss_computation(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_iterations: int,
    warmup: int = 10
) -> Tuple[BenchmarkResults, BenchmarkResults, BenchmarkResults]:
    """
    Benchmark cross-entropy + accuracy computation.
    
    Returns:
        (pytorch_results, compiled_results, cuda_results)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: Cross-Entropy + Accuracy")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size:,}")
    print(f"  Total tokens: {batch_size * seq_len:,}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup: {warmup}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pytorch_results = BenchmarkResults("PyTorch Cross-Entropy")
    compiled_results = BenchmarkResults("torch.compile Cross-Entropy")
    cuda_results = BenchmarkResults("CUDA Fused Loss")
    
    # Create test data
    logits = torch.randn(batch_size, seq_len, vocab_size, device=device, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels[labels < vocab_size // 10] = -100
    
    # Initialize implementations
    if KERNELS_AVAILABLE:
        fused_loss = FusedLoss()
        if not fused_loss.enabled:
            print("\n⚠️  CUDA kernel not enabled, skipping CUDA benchmark")
            cuda_results = None
    else:
        print("\n⚠️  CUDA kernels not available, skipping CUDA benchmark")
        cuda_results = None
    
    # Benchmark PyTorch
    print("\n[1/3] Running PyTorch benchmark...")
    for _ in range(warmup):
        _ = pytorch_cross_entropy_accuracy(logits, labels)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for i in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = pytorch_cross_entropy_accuracy(logits, labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000
        pytorch_results.add_time(elapsed)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Benchmark torch.compile
    print("\n[2/3] Running torch.compile benchmark...")
    print("  (First run will be slower due to compilation...)")
    
    for _ in range(warmup + 5):  # Extra warmup for compilation
        _ = compiled_cross_entropy_accuracy(logits, labels)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for i in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        result = compiled_cross_entropy_accuracy(logits, labels)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000
        compiled_results.add_time(elapsed)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Benchmark CUDA
    if cuda_results is not None:
        print("\n[3/3] Running CUDA benchmark...")
        
        for _ in range(warmup):
            _ = fused_loss(logits, labels)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        for i in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            result = fused_loss(logits, labels)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000
            cuda_results.add_time(elapsed)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
    
    return pytorch_results, compiled_results, cuda_results


def benchmark_gradient_clipping(
    num_params: int,
    param_size: int,
    num_iterations: int,
    max_norm: float = 1.0,
    warmup: int = 10
) -> Tuple[BenchmarkResults, BenchmarkResults, BenchmarkResults]:
    """
    Benchmark gradient clipping.
    
    Returns:
        (pytorch_results, compiled_results, cuda_results)
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARKING: Gradient Clipping")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  Number of parameters: {num_params}")
    print(f"  Parameter size: {param_size:,}")
    print(f"  Total elements: {num_params * param_size:,}")
    print(f"  Max norm: {max_norm}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Warmup: {warmup}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pytorch_results = BenchmarkResults("PyTorch Grad Clip")
    compiled_results = BenchmarkResults("torch.compile Grad Clip")
    cuda_results = BenchmarkResults("CUDA Fused Grad Clip")
    
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.params = torch.nn.ParameterList([
                torch.nn.Parameter(torch.randn(param_size, device=device))
                for _ in range(num_params)
            ])
    
    model = DummyModel()
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    
    if KERNELS_AVAILABLE:
        fused_clip = FusedGradClip()
        if not fused_clip.cuda_enabled:
            print("\n⚠️  CUDA kernel not enabled, skipping CUDA benchmark")
            cuda_results = None
    else:
        print("\n⚠️  CUDA kernels not available, skipping CUDA benchmark")
        cuda_results = None
    
    # Benchmark PyTorch
    print("\n[1/3] Running PyTorch benchmark...")
    for _ in range(warmup):
        _ = pytorch_grad_clip(model.parameters(), max_norm)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for i in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        norm = pytorch_grad_clip(model.parameters(), max_norm)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000
        pytorch_results.add_time(elapsed)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Benchmark torch.compile
    print("\n[2/3] Running torch.compile benchmark...")
    print("  Note: torch.compile may not optimize grad clipping significantly")
    
    for _ in range(warmup + 5):
        _ = compiled_grad_clip(model.parameters(), max_norm)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    for i in range(num_iterations):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        norm = compiled_grad_clip(model.parameters(), max_norm)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.perf_counter() - start) * 1000
        compiled_results.add_time(elapsed)
        
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{num_iterations}")
    
    # Benchmark CUDA
    if cuda_results is not None:
        print("\n[3/3] Running CUDA benchmark...")
        
        for _ in range(warmup):
            _ = fused_clip(model.parameters(), max_norm)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        for i in range(num_iterations):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            norm = fused_clip(model.parameters(), max_norm)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000
            cuda_results.add_time(elapsed)
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{num_iterations}")
    
    return pytorch_results, compiled_results, cuda_results


def print_comparison(pytorch_results: BenchmarkResults,
                    compiled_results: BenchmarkResults,
                    cuda_results: BenchmarkResults):
    """Print detailed comparison."""
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*80}")
    
    pytorch_results.print_summary()
    compiled_results.print_summary()
    
    if cuda_results is not None:
        cuda_results.print_summary()
    
    pytorch_stats = pytorch_results.get_stats()
    compiled_stats = compiled_results.get_stats()
    
    print(f"\n{'='*80}")
    print(f"SPEEDUP ANALYSIS")
    print(f"{'='*80}")
    
    compile_speedup = pytorch_stats['mean'] / compiled_stats['mean']
    print(f"\ntorch.compile vs PyTorch:")
    print(f"  Speedup:    {compile_speedup:.2f}x faster")
    print(f"  Time saved: {pytorch_stats['mean'] - compiled_stats['mean']:.3f} ms per call")
    
    if cuda_results is not None:
        cuda_stats = cuda_results.get_stats()
        
        cuda_vs_pytorch = pytorch_stats['mean'] / cuda_stats['mean']
        cuda_vs_compiled = compiled_stats['mean'] / cuda_stats['mean']
        
        print(f"\nCUDA vs PyTorch:")
        print(f"  Speedup:    {cuda_vs_pytorch:.2f}x faster")
        print(f"  Time saved: {pytorch_stats['mean'] - cuda_stats['mean']:.3f} ms per call")
        
        print(f"\nCUDA vs torch.compile:")
        print(f"  Speedup:    {cuda_vs_compiled:.2f}x faster")
        print(f"  Time saved: {compiled_stats['mean'] - cuda_stats['mean']:.3f} ms per call")
        
        print(f"\n{'='*80}")
        print(f"WINNER: ", end="")
        if cuda_stats['mean'] < compiled_stats['mean'] * 0.9:
            print(f"✅ CUDA ({cuda_stats['mean']:.3f} ms)")
        elif compiled_stats['mean'] < cuda_stats['mean'] * 0.9:
            print(f"✅ torch.compile ({compiled_stats['mean']:.3f} ms)")
        else:
            print("🤝 TIE (within 10%)")


def main():
    parser = argparse.ArgumentParser(description='Benchmark CUDA kernels vs PyTorch vs torch.compile')
    parser.add_argument('--iterations', type=int, default=100,
                       help='Number of iterations per benchmark (default: 100)')
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for loss benchmark (default: 16)')
    parser.add_argument('--seq-len', type=int, default=512,
                       help='Sequence length for loss benchmark (default: 512)')
    parser.add_argument('--vocab-size', type=int, default=32000,
                       help='Vocabulary size for loss benchmark (default: 32000)')
    parser.add_argument('--num-params', type=int, default=100,
                       help='Number of parameters for grad clip benchmark (default: 100)')
    parser.add_argument('--param-size', type=int, default=10000,
                       help='Size of each parameter for grad clip benchmark (default: 10000)')
    parser.add_argument('--skip-loss', action='store_true',
                       help='Skip loss computation benchmark')
    parser.add_argument('--skip-grad', action='store_true',
                       help='Skip gradient clipping benchmark')
    
    args = parser.parse_args()
    
    print("="*80)
    print("CUDA KERNEL PERFORMANCE BENCHMARK (with torch.compile)")
    print("="*80)
    print(f"\nDevice: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Custom kernels available: {KERNELS_AVAILABLE}")
    print(f"PyTorch version: {torch.__version__}")
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - benchmarks will run on CPU")
        return
    
    # Benchmark 1: Loss Computation
    if not args.skip_loss:
        pytorch_loss, compiled_loss, cuda_loss = benchmark_loss_computation(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            num_iterations=args.iterations,
            warmup=args.warmup
        )
        print_comparison(pytorch_loss, compiled_loss, cuda_loss)
    
    # Benchmark 2: Gradient Clipping
    if not args.skip_grad:
        pytorch_clip, compiled_clip, cuda_clip = benchmark_gradient_clipping(
            num_params=args.num_params,
            param_size=args.param_size,
            num_iterations=args.iterations,
            warmup=args.warmup
        )
        print_comparison(pytorch_clip, compiled_clip, cuda_clip)
    
    print(f"\n{'='*80}")
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()