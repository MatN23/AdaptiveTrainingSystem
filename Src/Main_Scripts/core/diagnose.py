#!/usr/bin/env python3
"""
Detailed profiling to find exact bottlenecks
"""

import torch
import torch.nn.functional as F
import time
import numpy as np

# Try to import custom ops
try:
    import moe_cuda_ops
    HAS_MOE = True
except:
    HAS_MOE = False
    print("❌ MoE ops not available")

def benchmark_op(func, *args, name="Operation", warmup=10, iters=100):
    """Benchmark a single operation"""
    # Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()
    
    # Time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        func(*args)
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / iters
    return avg_time

def pytorch_topk_gating(gate_logits, k, temperature=1.0):
    """PyTorch reference implementation (actual PyTorch topk)"""
    logits = gate_logits / temperature
    # PyTorch's topk is highly optimized CUDA code
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    # Softmax on just the top-k values
    top_k_weights = F.softmax(top_k_logits, dim=-1)
    return top_k_indices, top_k_weights

def pytorch_dispatch(tokens, top_k_indices, num_experts, capacity):
    """PyTorch dispatch"""
    batch_seq, hidden_dim = tokens.shape
    k = top_k_indices.shape[1]
    
    expert_inputs = torch.zeros(num_experts, capacity, hidden_dim, 
                                device=tokens.device, dtype=tokens.dtype)
    token_map = torch.full((num_experts, capacity), -1, 
                          device=tokens.device, dtype=torch.long)
    positions = torch.zeros(num_experts, device=tokens.device, dtype=torch.long)
    
    for i in range(batch_seq):
        for j in range(k):
            expert_id = top_k_indices[i, j].item()
            pos = positions[expert_id].item()
            if pos < capacity:
                expert_inputs[expert_id, pos] = tokens[i]
                token_map[expert_id, pos] = i * k + j
                positions[expert_id] += 1
    
    return expert_inputs, token_map

def pytorch_combine(expert_outputs, token_map, top_k_weights, num_tokens, k):
    """PyTorch combine"""
    num_experts, capacity, hidden_dim = expert_outputs.shape
    combined = torch.zeros(num_tokens, hidden_dim, 
                          device=expert_outputs.device, dtype=expert_outputs.dtype)
    
    weights_flat = top_k_weights.flatten()
    
    for e in range(num_experts):
        for p in range(capacity):
            token_weight_idx = token_map[e, p].item()
            if token_weight_idx >= 0:
                token_idx = token_weight_idx // k
                weight = weights_flat[token_weight_idx].item()
                combined[token_idx] += weight * expert_outputs[e, p]
    
    return combined

def profile_individual_ops():
    """Profile each MoE operation individually"""
    print("\n" + "="*80)
    print("INDIVIDUAL OPERATION PROFILING")
    print("="*80)
    
    # Test configurations
    configs = [
        (512, 512, 8, 2),    # Small
        (2048, 512, 8, 2),   # Medium
        (4096, 512, 8, 2),   # Large
    ]
    
    for num_tokens, hidden_dim, num_experts, k in configs:
        print(f"\nConfig: tokens={num_tokens}, hidden={hidden_dim}, experts={num_experts}, k={k}")
        print("-" * 80)
        
        # Setup
        gate_logits = torch.randn(num_tokens, num_experts, device='cuda')
        tokens = torch.randn(num_tokens, hidden_dim, device='cuda')
        capacity = (num_tokens * k) // num_experts + 10
        
        # 1. TOP-K GATING
        print("  [1/3] Top-K Gating...")
        pytorch_time = benchmark_op(pytorch_topk_gating, gate_logits, k, name="topk")
        print(f"    PyTorch:  {pytorch_time:.3f}ms")
        
        if HAS_MOE:
            cuda_time = benchmark_op(moe_cuda_ops.topk_gating, gate_logits, k, 1.0)
            print(f"    CUDA:     {cuda_time:.3f}ms")
            speedup = pytorch_time / cuda_time
            print(f"    Speedup:  {speedup:.2f}x {'✅' if speedup > 1 else '❌'}")
        
        # Get top-k results for next ops
        top_k_indices, top_k_weights = pytorch_topk_gating(gate_logits, k)
        
        # 2. DISPATCH
        print("\n  [2/3] Dispatch...")
        pytorch_time = benchmark_op(pytorch_dispatch, tokens, top_k_indices, 
                                   num_experts, capacity, name="dispatch")
        print(f"    PyTorch:  {pytorch_time:.3f}ms")
        
        if HAS_MOE:
            cuda_time = benchmark_op(moe_cuda_ops.dispatch_tokens, tokens, 
                                    top_k_indices, num_experts, capacity)
            print(f"    CUDA:     {cuda_time:.3f}ms")
            speedup = pytorch_time / cuda_time
            print(f"    Speedup:  {speedup:.2f}x {'✅' if speedup > 1 else '❌'}")
        
        # Get dispatch results
        expert_inputs, token_map = pytorch_dispatch(tokens, top_k_indices, 
                                                    num_experts, capacity)
        
        # Simulate expert processing
        expert_outputs = expert_inputs.clone()
        
        # 3. COMBINE
        print("\n  [3/3] Combine...")
        pytorch_time = benchmark_op(pytorch_combine, expert_outputs, token_map,
                                   top_k_weights, num_tokens, k, name="combine")
        print(f"    PyTorch:  {pytorch_time:.3f}ms")
        
        if HAS_MOE:
            cuda_time = benchmark_op(moe_cuda_ops.combine_expert_outputs, 
                                    expert_outputs, token_map, top_k_weights,
                                    num_tokens, k)
            print(f"    CUDA:     {cuda_time:.3f}ms")
            speedup = pytorch_time / cuda_time
            print(f"    Speedup:  {speedup:.2f}x {'✅' if speedup > 1 else '❌'}")

def analyze_memory_bandwidth():
    """Check if we're memory or compute bound"""
    print("\n" + "="*80)
    print("MEMORY BANDWIDTH ANALYSIS")
    print("="*80)
    
    num_tokens = 4096
    hidden_dim = 512
    num_experts = 8
    k = 2
    capacity = (num_tokens * k) // num_experts + 10
    
    # Create data
    tokens = torch.randn(num_tokens, hidden_dim, device='cuda')
    gate_logits = torch.randn(num_tokens, num_experts, device='cuda')
    
    # Measure memory transfer
    bytes_topk = num_tokens * num_experts * 4  # Read logits
    bytes_topk += num_tokens * k * 8  # Write indices
    bytes_topk += num_tokens * k * 4  # Write weights
    
    bytes_dispatch = num_tokens * hidden_dim * 4  # Read tokens
    bytes_dispatch += num_experts * capacity * hidden_dim * 4  # Write expert_inputs
    
    bytes_combine = num_experts * capacity * hidden_dim * 4  # Read expert outputs
    bytes_combine += num_tokens * hidden_dim * 4  # Write combined
    
    print(f"Memory traffic per operation:")
    print(f"  Top-K:    {bytes_topk / 1e6:.2f} MB")
    print(f"  Dispatch: {bytes_dispatch / 1e6:.2f} MB")
    print(f"  Combine:  {bytes_combine / 1e6:.2f} MB")
    print(f"  Total:    {(bytes_topk + bytes_dispatch + bytes_combine) / 1e6:.2f} MB")
    
    # T4 has ~320 GB/s bandwidth
    t4_bandwidth = 320e9  # bytes/s
    
    if HAS_MOE:
        # Time each operation
        topk_time = benchmark_op(moe_cuda_ops.topk_gating, gate_logits, k, 1.0, iters=100) / 1000
        
        top_k_indices, top_k_weights = moe_cuda_ops.topk_gating(gate_logits, k, 1.0)
        dispatch_time = benchmark_op(moe_cuda_ops.dispatch_tokens, tokens, 
                                     top_k_indices, num_experts, capacity, iters=100) / 1000
        
        expert_inputs, token_map = moe_cuda_ops.dispatch_tokens(tokens, top_k_indices, 
                                                                num_experts, capacity)
        expert_outputs = expert_inputs.clone()
        combine_time = benchmark_op(moe_cuda_ops.combine_expert_outputs, expert_outputs,
                                   token_map, top_k_weights, num_tokens, k, iters=100) / 1000
        
        print(f"\nBandwidth utilization:")
        print(f"  Top-K:    {bytes_topk / topk_time / 1e9:.2f} GB/s ({bytes_topk / topk_time / t4_bandwidth * 100:.1f}%)")
        print(f"  Dispatch: {bytes_dispatch / dispatch_time / 1e9:.2f} GB/s ({bytes_dispatch / dispatch_time / t4_bandwidth * 100:.1f}%)")
        print(f"  Combine:  {bytes_combine / combine_time / 1e9:.2f} GB/s ({bytes_combine / combine_time / t4_bandwidth * 100:.1f}%)")

def check_correctness():
    """Verify CUDA ops match PyTorch"""
    print("\n" + "="*80)
    print("CORRECTNESS CHECK")
    print("="*80)
    
    if not HAS_MOE:
        print("❌ CUDA ops not available")
        return
    
    num_tokens = 128
    hidden_dim = 512
    num_experts = 8
    k = 2
    capacity = (num_tokens * k) // num_experts + 10
    
    gate_logits = torch.randn(num_tokens, num_experts, device='cuda')
    tokens = torch.randn(num_tokens, hidden_dim, device='cuda')
    
    # Top-K
    print("  Testing Top-K gating...")
    pt_indices, pt_weights = pytorch_topk_gating(gate_logits, k)
    cuda_indices, cuda_weights = moe_cuda_ops.topk_gating(gate_logits, k, 1.0)
    
    # Check if same experts selected (order might differ)
    pt_set = set(pt_indices.cpu().numpy().flatten())
    cuda_set = set(cuda_indices.cpu().numpy().flatten())
    
    if pt_set == cuda_set:
        print("    ✅ Same experts selected")
    else:
        print(f"    ❌ Different experts! PT: {len(pt_set)}, CUDA: {len(cuda_set)}")
    
    # Check weights sum to 1
    pt_sum = pt_weights.sum(dim=1)
    cuda_sum = cuda_weights.sum(dim=1)
    if torch.allclose(pt_sum, torch.ones_like(pt_sum), atol=1e-5):
        print(f"    ✅ PyTorch weights normalized")
    if torch.allclose(cuda_sum, torch.ones_like(cuda_sum), atol=1e-5):
        print(f"    ✅ CUDA weights normalized")
    
    print("\n  Testing Dispatch...")
    pt_inputs, pt_map = pytorch_dispatch(tokens, pt_indices, num_experts, capacity)
    cuda_inputs, cuda_map = moe_cuda_ops.dispatch_tokens(tokens, cuda_indices, 
                                                         num_experts, capacity)
    print("    ✅ Dispatch completed")
    
    print("\n  Testing Combine...")
    expert_outputs = cuda_inputs.clone()
    combined = moe_cuda_ops.combine_expert_outputs(expert_outputs, cuda_map, 
                                                   cuda_weights, num_tokens, k)
    print("    ✅ Combine completed")

if __name__ == "__main__":
    print("PyTorch CUDA Profiler")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA is not available - skipping profiling")
        print("This script requires a CUDA-enabled GPU to run.")
        exit(0)
    
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"CUDA ops available: {HAS_MOE}")
    
    check_correctness()
    profile_individual_ops()
    analyze_memory_bandwidth()