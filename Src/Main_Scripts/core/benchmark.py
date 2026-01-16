#!/usr/bin/env python3
"""
Detailed MoE Operation Profiler
================================
Identifies exact bottlenecks in MoE CUDA kernels vs PyTorch
"""

import torch
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.moe_cuda_wrapper import MoECUDAOps, CUDA_OPS_AVAILABLE
    HAS_MOE = CUDA_OPS_AVAILABLE
except:
    HAS_MOE = False

# ============================================================================
# PYTORCH REFERENCE IMPLEMENTATIONS
# ============================================================================

class PyTorchMoEOps:
    """Pure PyTorch MoE operations for comparison"""
    
    @staticmethod
    def topk_gating(gate_logits, k, temperature=1.0):
        """Top-k gating with softmax"""
        scaled = gate_logits / temperature
        top_k_weights, top_k_indices = torch.topk(scaled, k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        return top_k_indices, top_k_weights
    
    @staticmethod
    def dispatch_tokens(tokens, top_k_indices, num_experts, capacity):
        """Dispatch tokens to experts"""
        batch_seq, hidden_dim = tokens.shape
        k = top_k_indices.shape[1]
        
        expert_inputs = torch.zeros(
            num_experts, capacity, hidden_dim,
            device=tokens.device, dtype=tokens.dtype
        )
        token_map = torch.full(
            (num_experts, capacity), -1,
            device=tokens.device, dtype=torch.long
        )
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
    
    @staticmethod
    def combine_expert_outputs(expert_outputs, token_map, top_k_weights, num_tokens, k):
        """Combine expert outputs with weights"""
        num_experts, capacity, hidden_dim = expert_outputs.shape
        combined = torch.zeros(
            num_tokens, hidden_dim,
            device=expert_outputs.device, dtype=expert_outputs.dtype
        )
        
        weights_flat = top_k_weights.flatten()
        
        for e in range(num_experts):
            for p in range(capacity):
                token_weight_idx = token_map[e, p].item()
                if token_weight_idx >= 0:
                    token_idx = token_weight_idx // k
                    weight = weights_flat[token_weight_idx].item()
                    combined[token_idx] += weight * expert_outputs[e, p]
        
        return combined


# ============================================================================
# OPTIMIZED PYTORCH IMPLEMENTATIONS (What PyTorch actually uses internally)
# ============================================================================

class OptimizedPyTorchMoE:
    """Optimized PyTorch using scatter/gather primitives"""
    
    @staticmethod
    def dispatch_tokens_optimized(tokens, top_k_indices, num_experts, capacity):
        """Vectorized dispatch using scatter"""
        batch_seq, hidden_dim = tokens.shape
        k = top_k_indices.shape[1]
        
        # Flatten for vectorized operations
        flat_indices = top_k_indices.flatten()  # [batch_seq * k]
        flat_tokens = tokens.unsqueeze(1).expand(-1, k, -1).reshape(-1, hidden_dim)
        
        # Count experts
        expert_counts = torch.zeros(num_experts, device=tokens.device, dtype=torch.long)
        expert_counts.scatter_add_(0, flat_indices, torch.ones_like(flat_indices))
        
        # Compute positions with cumsum
        positions = torch.zeros_like(flat_indices)
        for i in range(batch_seq * k):
            expert_id = flat_indices[i].item()
            positions[i] = (flat_indices[:i] == expert_id).sum().item()
        
        # Create expert inputs
        expert_inputs = torch.zeros(
            num_experts, capacity, hidden_dim,
            device=tokens.device, dtype=tokens.dtype
        )
        token_map = torch.full(
            (num_experts, capacity), -1,
            device=tokens.device, dtype=torch.long
        )
        
        # Scatter tokens
        valid_mask = positions < capacity
        for i in range(batch_seq * k):
            if valid_mask[i]:
                expert_id = flat_indices[i].item()
                pos = positions[i].item()
                expert_inputs[expert_id, pos] = flat_tokens[i]
                token_map[expert_id, pos] = i
        
        return expert_inputs, token_map
    
    @staticmethod
    def combine_expert_outputs_optimized(expert_outputs, token_map, top_k_weights, num_tokens, k):
        """Vectorized combine using scatter_add"""
        num_experts, capacity, hidden_dim = expert_outputs.shape
        combined = torch.zeros(
            num_tokens, hidden_dim,
            device=expert_outputs.device, dtype=expert_outputs.dtype
        )
        
        weights_flat = top_k_weights.flatten()
        
        # Vectorized version
        valid_mask = token_map >= 0
        valid_expert_outputs = expert_outputs[valid_mask]  # [N, hidden_dim]
        valid_token_indices = (token_map[valid_mask] // k).long()
        valid_weights = weights_flat[token_map[valid_mask]]
        
        weighted_outputs = valid_expert_outputs * valid_weights.unsqueeze(-1)
        combined.index_add_(0, valid_token_indices, weighted_outputs)
        
        return combined


# ============================================================================
# BENCHMARK UTILITIES
# ============================================================================

def benchmark_op(func, *args, warmup=10, iters=100):
    """Benchmark with CUDA events for accurate timing"""
    # Warmup
    for _ in range(warmup):
        result = func(*args)
    torch.cuda.synchronize()
    
    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        result = func(*args)
    end.record()
    torch.cuda.synchronize()
    
    return start.elapsed_time(end) / iters, result


def measure_memory_bandwidth(operation_name, bytes_transferred, time_ms):
    """Calculate memory bandwidth utilization"""
    bandwidth_gbps = (bytes_transferred / 1e9) / (time_ms / 1000)
    
    # T4 peak bandwidth: 320 GB/s
    t4_peak = 320.0
    utilization = (bandwidth_gbps / t4_peak) * 100
    
    return bandwidth_gbps, utilization


# ============================================================================
# DETAILED PROFILING
# ============================================================================

def profile_moe_operations():
    """Profile each MoE operation in detail"""
    
    print("\n" + "="*100)
    print("DETAILED MOE OPERATION PROFILING")
    print("="*100)
    
    configs = [
        # (num_tokens, hidden_dim, num_experts, k)
        (1024, 512, 8, 2, "Small"),
        (2048, 512, 8, 2, "Medium"),
        (4096, 512, 8, 2, "Large"),
        (4096, 1024, 8, 2, "Large+Wide"),
    ]
    
    for num_tokens, hidden_dim, num_experts, k, label in configs:
        print(f"\n{'='*100}")
        print(f"Config: {label} - tokens={num_tokens}, hidden={hidden_dim}, experts={num_experts}, k={k}")
        print(f"{'='*100}")
        
        capacity = (num_tokens * k) // num_experts + 20
        
        # Generate test data
        gate_logits = torch.randn(num_tokens, num_experts, device='cuda', dtype=torch.float32)
        tokens = torch.randn(num_tokens, hidden_dim, device='cuda', dtype=torch.float32)
        
        # ====================================================================
        # 1. TOP-K GATING
        # ====================================================================
        print(f"\n{'─'*100}")
        print("1. TOP-K GATING")
        print(f"{'─'*100}")
        
        # PyTorch naive
        pt_time, (pt_indices, pt_weights) = benchmark_op(
            PyTorchMoEOps.topk_gating, gate_logits, k
        )
        
        # Memory analysis
        bytes_read = num_tokens * num_experts * 4  # gate_logits
        bytes_write = num_tokens * k * (8 + 4)  # indices + weights
        total_bytes = bytes_read + bytes_write
        
        bw, util = measure_memory_bandwidth("TopK", total_bytes, pt_time)
        
        print(f"  PyTorch naive:     {pt_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
        
        if HAS_MOE:
            cuda_time, (cuda_indices, cuda_weights) = benchmark_op(
                MoECUDAOps.topk_gating, gate_logits, k, 1.0, True
            )
            bw, util = measure_memory_bandwidth("TopK", total_bytes, cuda_time)
            speedup = pt_time / cuda_time
            
            print(f"  CUDA kernel:       {cuda_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
            print(f"  Speedup:           {speedup:.2f}x {'✅' if speedup > 1.0 else '❌'}")
            
            # Correctness check
            indices_match = torch.allclose(cuda_indices.float(), pt_indices.float(), atol=1)
            weights_match = torch.allclose(cuda_weights, pt_weights, rtol=0.01)
            print(f"  Correctness:       Indices: {'✅' if indices_match else '❌'}, Weights: {'✅' if weights_match else '❌'}")
        
        # ====================================================================
        # 2. DISPATCH TOKENS
        # ====================================================================
        print(f"\n{'─'*100}")
        print("2. DISPATCH TOKENS")
        print(f"{'─'*100}")
        
        # Use PyTorch results for consistency
        if HAS_MOE:
            top_k_indices = cuda_indices
            top_k_weights = cuda_weights
        else:
            top_k_indices = pt_indices
            top_k_weights = pt_weights
        
        # PyTorch naive
        pt_time, (pt_expert_inputs, pt_token_map) = benchmark_op(
            PyTorchMoEOps.dispatch_tokens, tokens, top_k_indices, num_experts, capacity
        )
        
        bytes_read = num_tokens * hidden_dim * 4  # tokens
        bytes_write = num_experts * capacity * hidden_dim * 4  # expert_inputs
        bytes_write += num_experts * capacity * 8  # token_map
        total_bytes = bytes_read + bytes_write
        
        bw, util = measure_memory_bandwidth("Dispatch", total_bytes, pt_time)
        print(f"  PyTorch naive:     {pt_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
        
        # PyTorch optimized
        pt_opt_time, (pt_opt_inputs, pt_opt_map) = benchmark_op(
            OptimizedPyTorchMoE.dispatch_tokens_optimized, tokens, top_k_indices, 
            num_experts, capacity
        )
        bw, util = measure_memory_bandwidth("Dispatch", total_bytes, pt_opt_time)
        print(f"  PyTorch optimized: {pt_opt_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
        
        if HAS_MOE:
            cuda_time, (cuda_expert_inputs, cuda_token_map) = benchmark_op(
                MoECUDAOps.dispatch_tokens, tokens, top_k_indices, num_experts, capacity, True
            )
            bw, util = measure_memory_bandwidth("Dispatch", total_bytes, cuda_time)
            speedup = pt_opt_time / cuda_time
            
            print(f"  CUDA kernel:       {cuda_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
            print(f"  Speedup:           {speedup:.2f}x vs optimized {'✅' if speedup > 1.0 else '❌'}")
        
        # ====================================================================
        # 3. COMBINE EXPERT OUTPUTS
        # ====================================================================
        print(f"\n{'─'*100}")
        print("3. COMBINE EXPERT OUTPUTS")
        print(f"{'─'*100}")
        
        # Simulate expert processing
        if HAS_MOE:
            expert_outputs = cuda_expert_inputs.clone()
            token_map = cuda_token_map
        else:
            expert_outputs = pt_expert_inputs.clone()
            token_map = pt_token_map
        
        # PyTorch naive
        pt_time, pt_combined = benchmark_op(
            PyTorchMoEOps.combine_expert_outputs, expert_outputs, token_map,
            top_k_weights, num_tokens, k
        )
        
        bytes_read = num_experts * capacity * hidden_dim * 4  # expert_outputs
        bytes_read += num_experts * capacity * 8  # token_map
        bytes_write = num_tokens * hidden_dim * 4  # combined
        total_bytes = bytes_read + bytes_write
        
        bw, util = measure_memory_bandwidth("Combine", total_bytes, pt_time)
        print(f"  PyTorch naive:     {pt_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
        
        # PyTorch optimized
        pt_opt_time, pt_opt_combined = benchmark_op(
            OptimizedPyTorchMoE.combine_expert_outputs_optimized, expert_outputs,
            token_map, top_k_weights, num_tokens, k
        )
        bw, util = measure_memory_bandwidth("Combine", total_bytes, pt_opt_time)
        print(f"  PyTorch optimized: {pt_opt_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
        
        if HAS_MOE:
            cuda_time, cuda_combined = benchmark_op(
                MoECUDAOps.combine_expert_outputs, expert_outputs, token_map,
                top_k_weights, num_tokens, k, True
            )
            bw, util = measure_memory_bandwidth("Combine", total_bytes, cuda_time)
            speedup = pt_opt_time / cuda_time
            
            print(f"  CUDA kernel:       {cuda_time:.3f} ms  ({bw:.1f} GB/s, {util:.1f}% util)")
            print(f"  Speedup:           {speedup:.2f}x vs optimized {'✅' if speedup > 1.0 else '❌'}")
            
            # Correctness
            match = torch.allclose(cuda_combined, pt_opt_combined, rtol=0.01, atol=0.01)
            max_diff = (cuda_combined - pt_opt_combined).abs().max().item()
            print(f"  Correctness:       {'✅' if match else '❌'} (max diff: {max_diff:.6f})")
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        print(f"\n{'─'*100}")
        print("OPERATION SUMMARY")
        print(f"{'─'*100}")
        
        if HAS_MOE:
            total_pt = pt_time + pt_opt_time + pt_opt_time
            total_cuda = cuda_time + cuda_time + cuda_time
            
            print(f"  Total time (PyTorch optimized): {total_pt:.3f} ms")
            print(f"  Total time (CUDA):              {total_cuda:.3f} ms")
            print(f"  Overall speedup:                {total_pt / total_cuda:.2f}x")


def profile_nsys():
    """Instructions for profiling with Nsight Systems"""
    print("\n" + "="*100)
    print("PROFILING WITH NSIGHT SYSTEMS")
    print("="*100)
    
    print("""
To get detailed GPU profiling information, run:

    nsys profile -o moe_profile \\
        --trace=cuda,nvtx \\
        --cuda-memory-usage=true \\
        python benchmark_core.py --training-only --iterations=10

Then analyze with:
    
    nsys-ui moe_profile.qdrep

Key metrics to look for:
  1. Kernel execution time
  2. Memory throughput (should be >200 GB/s on T4)
  3. SM occupancy (should be >50%)
  4. Atomic operation contention (look for serialization)
  5. Kernel launch overhead

Common bottlenecks:
  ❌ Low occupancy (<30%) - increase blocks/threads
  ❌ Low bandwidth (<150 GB/s) - memory access pattern issues
  ❌ High atomic contention - use different reduction strategy
  ❌ Frequent kernel launches - fuse operations
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\nDetailed MoE Operation Profiler")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}")
    print(f"MoE CUDA ops: {'✅ Available' if HAS_MOE else '❌ Not available'}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    if not HAS_MOE:
        print("\n⚠️  MoE CUDA ops not available. Install with:")
        print("    cd cuda && ./compile.sh")
        print("\nRunning PyTorch-only benchmarks...")
    
    profile_moe_operations()
    profile_nsys()


if __name__ == "__main__":
    main()