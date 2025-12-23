# Copyright (c) 2025 MatN23. All rights reserved.

"""
CUDA MoE Wrapper - FIXED
=========================
Proper error handling and compilation flags.
"""

import torch
import torch.nn.functional as F
from typing import Tuple
import time
import os

# ============================================================================
# LOAD CUDA OPS - FIXED
# ============================================================================

CUDA_OPS_AVAILABLE = False
moe_cuda_ops = None

try:
    from torch.utils.cpp_extension import load
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(current_dir, 'moe_cuda_ops.cu')
    
    if not os.path.exists(cuda_src):
        raise FileNotFoundError(f"CUDA source not found: {cuda_src}")
    
    print(f"🔨 Compiling CUDA MoE ops from: {cuda_src}")
    print(f"   This takes ~60s on first run...")
    
    # FIXED: Detect GPU arch correctly
    extra_cuda_cflags = ['-O3', '--use_fast_math']
    
    if torch.cuda.is_available():
        # Get actual GPU capability
        capability = torch.cuda.get_device_capability(0)
        arch = f"compute_{capability[0]}{capability[1]}"
        code = f"sm_{capability[0]}{capability[1]}"
        extra_cuda_cflags.append(f'-gencode=arch={arch},code={code}')
        print(f"   Target: {code}")
    else:
        print(f"   ⚠️  CUDA not available, compiling anyway...")
    
    # FIXED: Enable verbose to catch errors
    moe_cuda_ops = load(
        name='moe_cuda_ops',
        sources=[cuda_src],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=['-O3'],
        verbose=True,  # CRITICAL: Shows compilation errors
        with_cuda=True
    )
    
    CUDA_OPS_AVAILABLE = True
    print(f"✅ CUDA MoE ops loaded successfully")
    
    # FIXED: Verify functions are actually exported
    required_funcs = ['topk_gating', 'dispatch_tokens', 'combine_expert_outputs']
    available_funcs = [f for f in dir(moe_cuda_ops) if not f.startswith('_')]
    
    for func in required_funcs:
        if func not in available_funcs:
            raise RuntimeError(f"Function '{func}' not found in compiled module!")
    
    print(f"   Functions: {', '.join(available_funcs)}")

except Exception as e:
    print(f"⚠️  CUDA compilation failed: {e}")
    print(f"   Falling back to PyTorch implementation")
    CUDA_OPS_AVAILABLE = False

# ============================================================================
# WRAPPER CLASS
# ============================================================================

class MoECUDAOps:
    """MoE operations with automatic CUDA/PyTorch fallback."""
    
    @staticmethod
    def topk_gating(gate_logits, k, temperature=1.0, use_cuda=True):
        """Top-k gating with softmax normalization."""
        
        # FIXED: Explicit CUDA availability check
        if use_cuda and CUDA_OPS_AVAILABLE and gate_logits.is_cuda:
            try:
                indices, weights = moe_cuda_ops.topk_gating(gate_logits, k, temperature)
                # FIXED: Verify output dtype
                assert indices.dtype == torch.int64, f"Expected int64, got {indices.dtype}"
                return indices, weights
            except Exception as e:
                print(f"⚠️  CUDA topk_gating failed: {e}, using PyTorch")
                # Fall through to PyTorch
        
        # PyTorch fallback
        scaled = gate_logits / temperature
        values, indices = torch.topk(scaled, k, dim=-1)
        weights = F.softmax(values, dim=-1)
        return indices, weights
    
    @staticmethod
    def dispatch_tokens(tokens, indices, num_experts, capacity, use_cuda=True):
        """Dispatch tokens to experts."""
        
        if use_cuda and CUDA_OPS_AVAILABLE and tokens.is_cuda:
            try:
                # FIXED: Verify input dtype
                if indices.dtype != torch.int64:
                    indices = indices.to(torch.int64)
                
                expert_inputs, token_map = moe_cuda_ops.dispatch_tokens(
                    tokens, indices, num_experts, capacity
                )
                return expert_inputs, token_map
            except Exception as e:
                print(f"⚠️  CUDA dispatch_tokens failed: {e}, using PyTorch")
        
        # PyTorch fallback
        num_tokens, hidden_dim = tokens.shape
        k = indices.size(1)
        
        expert_inputs = torch.zeros(num_experts, capacity, hidden_dim, 
                                    dtype=tokens.dtype, device=tokens.device)
        token_map = torch.full((num_experts, capacity), -1, 
                              dtype=torch.int64, device=tokens.device)
        positions = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)
        
        for i in range(num_tokens):
            for j in range(k):
                expert_id = indices[i, j].item()
                pos = positions[expert_id].item()
                if pos < capacity and expert_id >= 0:
                    expert_inputs[expert_id, pos] = tokens[i]
                    token_map[expert_id, pos] = i * k + j
                    positions[expert_id] += 1
        
        return expert_inputs, token_map
    
    @staticmethod
    def combine_expert_outputs(expert_outputs, token_map, weights, num_tokens, k, use_cuda=True):
        """Combine expert outputs with weights."""
        
        if use_cuda and CUDA_OPS_AVAILABLE and expert_outputs.is_cuda:
            try:
                combined = moe_cuda_ops.combine_expert_outputs(
                    expert_outputs, token_map, weights, num_tokens, k
                )
                return combined
            except Exception as e:
                print(f"⚠️  CUDA combine_expert_outputs failed: {e}, using PyTorch")
        
        # PyTorch fallback
        num_experts, capacity, hidden_dim = expert_outputs.shape
        combined = torch.zeros(num_tokens, hidden_dim, 
                              dtype=expert_outputs.dtype, device=expert_outputs.device)
        
        for expert_id in range(num_experts):
            for pos in range(capacity):
                idx = token_map[expert_id, pos].item()
                if idx >= 0:
                    token_idx = idx // k
                    if token_idx < num_tokens:
                        weight = weights.flatten()[idx]
                        combined[token_idx] += weight * expert_outputs[expert_id, pos]
        
        return combined


# ============================================================================
# BENCHMARK
# ============================================================================

def benchmark_moe_ops(num_tokens=1024, hidden_dim=768, num_experts=8, k=2, runs=100):
    """Benchmark CUDA vs PyTorch."""
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    print(f"\n{'='*70}")
    print(f"🚀 BENCHMARK: {num_tokens} tokens, {num_experts} experts, k={k}")
    print(f"{'='*70}\n")
    
    gate_logits = torch.randn(num_tokens, num_experts, device='cuda')
    
    # Warmup
    for _ in range(10):
        if CUDA_OPS_AVAILABLE:
            _ = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
        _ = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
    
    torch.cuda.synchronize()
    
    # CUDA benchmark
    if CUDA_OPS_AVAILABLE:
        print("Testing CUDA...")
        start = time.perf_counter()
        for _ in range(runs):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
        torch.cuda.synchronize()
        cuda_time = (time.perf_counter() - start) * 1000
        
        cuda_per_call = cuda_time / runs
        cuda_throughput = num_tokens / (cuda_per_call / 1000)
        
        print(f"✓ CUDA:    {cuda_time:.2f}ms total | {cuda_per_call:.4f}ms per call")
        print(f"           {cuda_throughput:,.0f} tokens/sec")
    
    # PyTorch benchmark
    print("Testing PyTorch...")
    start = time.perf_counter()
    for _ in range(runs):
        indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) * 1000
    
    pytorch_per_call = pytorch_time / runs
    pytorch_throughput = num_tokens / (pytorch_per_call / 1000)
    
    print(f"✓ PyTorch: {pytorch_time:.2f}ms total | {pytorch_per_call:.4f}ms per call")
    print(f"           {pytorch_throughput:,.0f} tokens/sec")
    
    # Results
    print(f"\n{'='*70}")
    if CUDA_OPS_AVAILABLE and cuda_time > 0:
        speedup = pytorch_time / cuda_time
        saved = pytorch_time - cuda_time
        throughput_gain = cuda_throughput - pytorch_throughput
        
        print(f"🚀 SPEEDUP: {speedup:.2f}x faster")
        print(f"   Time saved: {saved:.2f}ms ({saved/pytorch_time*100:.1f}%)")
        print(f"   Throughput gain: +{throughput_gain:,.0f} tokens/sec")
        
        if speedup > 3:
            print(f"   ✅ EXCELLENT!")
        elif speedup > 2:
            print(f"   ✅ GREAT!")
        elif speedup > 1.5:
            print(f"   ✓ GOOD")
        else:
            print(f"   ⚠️  Marginal")
    print(f"{'='*70}\n")


# Export with the name model.py expects
HAS_CUDA_OPS = CUDA_OPS_AVAILABLE

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MoE CUDA Operations - FIXED")
    print("="*70)
    print(f"Status: {'Available ✅' if CUDA_OPS_AVAILABLE else 'PyTorch only ⚠️'}")
    
    if CUDA_OPS_AVAILABLE:
        funcs = [f for f in dir(moe_cuda_ops) if not f.startswith('_')]
        print(f"Functions: {', '.join(funcs)}")
    print("="*70)
    
    if torch.cuda.is_available():
        print("\n📊 RUNNING BENCHMARKS")
        
        # Test 1: Small
        print("\n" + "="*70)
        print("TEST 1: Small (~300M params)")
        benchmark_moe_ops(num_tokens=1024, hidden_dim=768, num_experts=8, k=2, runs=100)
        
        # Test 2: Medium
        print("\n" + "="*70)
        print("TEST 2: Medium (~500M params)")
        benchmark_moe_ops(num_tokens=1024, hidden_dim=1024, num_experts=8, k=2, runs=100)
        
        # Test 3: Large
        print("\n" + "="*70)
        print("TEST 3: Large (~1B params)")
        benchmark_moe_ops(num_tokens=2048, hidden_dim=1536, num_experts=16, k=2, runs=100)
    else:
        print("\n⚠️  CUDA not available - skipping benchmarks")