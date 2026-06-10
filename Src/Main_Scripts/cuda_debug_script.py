# Copyright (c) 2025 MatN23. All rights reserved.
"""
CUDA MoE Debugging Script - FULLY WORKING VERSION
==================================================

This will tell us EXACTLY what's happening:
1. Is CUDA actually being called?
2. How much time is CUDA vs PyTorch?
3. Where is the bottleneck?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import CUDA ops
try:
    import moe_cuda_ops
    CUDA_AVAILABLE = True
    print(" CUDA ops imported successfully")
except ImportError as e:
    CUDA_AVAILABLE = False
    print(f" CUDA ops import failed: {e}")

print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print(f"CUDA ops module available: {CUDA_AVAILABLE}")

print("\n" + "="*70)
print("DETAILED MoE LAYER PROFILING")
print("="*70 + "\n")


class SimplifiedMoELayer(nn.Module):
    """
    Simplified MoE layer for testing - avoids complex imports.
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        self.hidden_size = config.hidden_size
        self.capacity_factor = config.capacity_factor
        self.routing_temperature = config.routing_temperature
        self.load_balancing_weight = config.load_balancing_weight
        self.use_cuda_ops = config.use_cuda_moe and CUDA_AVAILABLE
        
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        
        # Simple experts (just linear for testing)
        self.experts = nn.ModuleList([
            nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            for _ in range(config.num_experts)
        ])
        
        # Timing buckets
        self.timings = {
            'gate_compute': 0.0,
            'cuda_routing': 0.0,
            'pytorch_routing': 0.0,
            'dispatch': 0.0,
            'expert_compute': 0.0,
            'combine': 0.0,
            'aux_loss': 0.0,
            'total': 0.0,
        }
        
        self.call_counts = {
            'cuda_routing': 0,
            'pytorch_routing': 0,
            'total_forwards': 0,
        }
        
        print(f" MoE Layer: {config.num_experts} experts, top-{config.moe_top_k}")
        print(f"   CUDA ops enabled: {self.use_cuda_ops}")
    
    def _pytorch_dispatch(self, tokens, top_k_indices, num_experts, capacity):
        """PyTorch fallback for dispatch."""
        num_tokens, hidden_size = tokens.shape
        
        expert_inputs = torch.zeros(
            num_experts, capacity, hidden_size,
            dtype=tokens.dtype, device=tokens.device
        )
        token_map = torch.full(
            (num_experts, capacity), -1,
            dtype=torch.int32, device=tokens.device
        )
        expert_positions = torch.zeros(num_experts, dtype=torch.int32, device=tokens.device)
        
        for token_idx in range(num_tokens):
            for k_idx in range(self.top_k):
                expert_id = top_k_indices[token_idx, k_idx].item()
                pos = expert_positions[expert_id].item()
                
                if pos < capacity:
                    expert_inputs[expert_id, pos] = tokens[token_idx]
                    token_map[expert_id, pos] = token_idx * self.top_k + k_idx
                    expert_positions[expert_id] += 1
        
        return expert_inputs, token_map
    
    def _pytorch_combine(self, expert_outputs, token_map, top_k_probs, num_tokens):
        """PyTorch fallback for combine."""
        num_experts, capacity, hidden_size = expert_outputs.shape
        output = torch.zeros(
            num_tokens, hidden_size,
            dtype=expert_outputs.dtype, device=expert_outputs.device
        )
        
        for expert_id in range(num_experts):
            for pos in range(capacity):
                token_weight_idx = token_map[expert_id, pos].item()
                if token_weight_idx < 0:
                    continue
                
                token_idx = token_weight_idx // self.top_k
                weight_idx = token_weight_idx % self.top_k
                
                if token_idx >= num_tokens:
                    continue
                
                weight = top_k_probs[token_idx, weight_idx]
                expert_out = expert_outputs[expert_id, pos]
                output[token_idx] += weight * expert_out
        
        return output
    
    def forward(self, x):
        """Instrumented forward pass with detailed timing."""
        start_total = time.perf_counter()
        
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        total_tokens = x_flat.shape[0]
        
        # === GATE COMPUTATION ===
        start = time.perf_counter()
        gate_logits = self.gate(x_flat)
        torch.cuda.synchronize()
        self.timings['gate_compute'] += (time.perf_counter() - start) * 1000
        
        # === ROUTING ===
        start = time.perf_counter()
        
        if self.use_cuda_ops and CUDA_AVAILABLE and x.is_cuda:
            try:
                from Main_Scripts.core.wrappers.moe_cuda_wrapper import MoECUDAOps
                should_use = MoECUDAOps.should_use_cuda(
                    total_tokens, 
                    self.num_experts, 
                    hidden_size,
                    True, 
                    True
                )
            except ImportError:
                should_use = True  # Assume yes if wrapper not available
            
            if should_use:
                try:
                    # CUDA ROUTING
                    top_k_indices, top_k_probs = moe_cuda_ops.topk_gating(
                        gate_logits, self.top_k, self.routing_temperature
                    )
                    torch.cuda.synchronize()
                    self.timings['cuda_routing'] += (time.perf_counter() - start) * 1000
                    self.call_counts['cuda_routing'] += 1
                    routing_method = "CUDA"
                except Exception as e:
                    print(f"  CUDA routing failed: {e}, falling back to PyTorch")
                    # Fallback to PyTorch
                    top_k_values, top_k_indices = torch.topk(
                        gate_logits / self.routing_temperature, 
                        self.top_k, 
                        dim=-1
                    )
                    top_k_probs = F.softmax(top_k_values, dim=-1)
                    torch.cuda.synchronize()
                    self.timings['pytorch_routing'] += (time.perf_counter() - start) * 1000
                    self.call_counts['pytorch_routing'] += 1
                    routing_method = "PyTorch (fallback)"
            else:
                # PyTorch routing (size below threshold)
                top_k_values, top_k_indices = torch.topk(
                    gate_logits / self.routing_temperature, 
                    self.top_k, 
                    dim=-1
                )
                top_k_probs = F.softmax(top_k_values, dim=-1)
                torch.cuda.synchronize()
                self.timings['pytorch_routing'] += (time.perf_counter() - start) * 1000
                self.call_counts['pytorch_routing'] += 1
                routing_method = "PyTorch (below threshold)"
        else:
            # PyTorch routing (CUDA not available)
            top_k_values, top_k_indices = torch.topk(
                gate_logits / self.routing_temperature, 
                self.top_k, 
                dim=-1
            )
            top_k_probs = F.softmax(top_k_values, dim=-1)
            torch.cuda.synchronize()
            self.timings['pytorch_routing'] += (time.perf_counter() - start) * 1000
            self.call_counts['pytorch_routing'] += 1
            routing_method = "PyTorch (CUDA disabled)"
        
        # === DISPATCH ===
        start = time.perf_counter()
        capacity = int((total_tokens * self.top_k / self.num_experts) * self.capacity_factor)
        
        # Try CUDA dispatch
        if self.use_cuda_ops and CUDA_AVAILABLE:
            try:
                expert_inputs, token_map = moe_cuda_ops.dispatch_tokens(
                    x_flat, top_k_indices, self.num_experts, capacity
                )
                torch.cuda.synchronize()
                self.timings['dispatch'] += (time.perf_counter() - start) * 1000
                dispatch_method = "CUDA"
            except Exception as e:
                print(f"  CUDA dispatch failed: {e}, falling back to PyTorch")
                # PyTorch fallback
                expert_inputs, token_map = self._pytorch_dispatch(
                    x_flat, top_k_indices, self.num_experts, capacity
                )
                torch.cuda.synchronize()
                self.timings['dispatch'] += (time.perf_counter() - start) * 1000
                dispatch_method = "PyTorch (fallback)"
        else:
            expert_inputs, token_map = self._pytorch_dispatch(
                x_flat, top_k_indices, self.num_experts, capacity
            )
            torch.cuda.synchronize()
            self.timings['dispatch'] += (time.perf_counter() - start) * 1000
            dispatch_method = "PyTorch"
        
        # === EXPERT COMPUTATION ===
        start = time.perf_counter()
        expert_outputs = torch.zeros_like(expert_inputs)
        for expert_id in range(self.num_experts):
            mask = (token_map[expert_id] >= 0)
            if mask.any():
                valid_inputs = expert_inputs[expert_id][mask]
                expert_outputs[expert_id][mask] = self.experts[expert_id](valid_inputs)
        
        torch.cuda.synchronize()
        self.timings['expert_compute'] += (time.perf_counter() - start) * 1000
        
        # === COMBINE ===
        start = time.perf_counter()
        
        # Try CUDA combine
        if self.use_cuda_ops and CUDA_AVAILABLE:
            try:
                output = moe_cuda_ops.combine_expert_outputs(
                    expert_outputs, token_map, top_k_probs, total_tokens, self.top_k
                )
                torch.cuda.synchronize()
                self.timings['combine'] += (time.perf_counter() - start) * 1000
                combine_method = "CUDA"
            except Exception as e:
                print(f"  CUDA combine failed: {e}, falling back to PyTorch")
                # PyTorch fallback
                output = self._pytorch_combine(
                    expert_outputs, token_map, top_k_probs, total_tokens
                )
                torch.cuda.synchronize()
                self.timings['combine'] += (time.perf_counter() - start) * 1000
                combine_method = "PyTorch (fallback)"
        else:
            output = self._pytorch_combine(
                expert_outputs, token_map, top_k_probs, total_tokens
            )
            torch.cuda.synchronize()
            self.timings['combine'] += (time.perf_counter() - start) * 1000
            combine_method = "PyTorch"
        
        # === AUX LOSS (FIXED) ===
        start = time.perf_counter()
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_usage = torch.zeros(self.num_experts, device=gate_probs.device)
        
        for k in range(self.top_k):
            # Ensure indices are valid integers
            indices_k = top_k_indices[:, k].long()
            # Clamp to valid range
            indices_k = torch.clamp(indices_k, 0, self.num_experts - 1)
            # Move to CPU for bincount (it doesn't work well on GPU)
            indices_cpu = indices_k.cpu()
            expert_counts = torch.bincount(
                indices_cpu, 
                minlength=self.num_experts
            )
            expert_usage += expert_counts.float().to(gate_probs.device)
        
        expert_usage = expert_usage / (total_tokens * self.top_k + 1e-9)
        gate_importance = gate_probs.mean(dim=0)
        aux_loss = torch.sum(expert_usage * gate_importance) * self.num_experts
        aux_loss = torch.clamp(aux_loss * self.load_balancing_weight, max=1.0)
        
        torch.cuda.synchronize()
        self.timings['aux_loss'] += (time.perf_counter() - start) * 1000
        
        # === TOTAL ===
        torch.cuda.synchronize()
        self.timings['total'] += (time.perf_counter() - start_total) * 1000
        self.call_counts['total_forwards'] += 1
        
        return output.view(batch_size, seq_len, hidden_size), aux_loss
    
    def print_profile(self):
        """Print detailed profiling results."""
        n = self.call_counts['total_forwards']
        if n == 0:
            print("No forwards recorded!")
            return
        
        print("\n" + "="*70)
        print("MoE LAYER PROFILING RESULTS")
        print("="*70)
        print(f"Total forward passes: {n}")
        print(f"CUDA routing calls: {self.call_counts['cuda_routing']}")
        print(f"PyTorch routing calls: {self.call_counts['pytorch_routing']}")
        
        if self.call_counts['cuda_routing'] == 0:
            print("\n WARNING: CUDA ROUTING NEVER USED!")
            print("   Your CUDA kernels are NOT being called!")
            print("   Possible reasons:")
            print("   - Problem size below thresholds")
            print("   - CUDA ops not properly compiled")
            print("   - Should_use_cuda() returning False")
        else:
            cuda_pct = self.call_counts['cuda_routing'] / n * 100
            print(f"\n CUDA routing used {self.call_counts['cuda_routing']}/{n} times ({cuda_pct:.1f}%)")
        
        print(f"\n{'Operation':<20} {'Total (ms)':<12} {'Avg (ms)':<12} {'% of Total':<12}")
        print("-"*70)
        
        total_time = self.timings['total']
        
        for key in ['gate_compute', 'cuda_routing', 'pytorch_routing', 
                    'dispatch', 'expert_compute', 'combine', 'aux_loss']:
            time_ms = self.timings[key]
            avg_ms = time_ms / n if n > 0 else 0
            pct = (time_ms / total_time * 100) if total_time > 0 else 0
            print(f"{key:<20} {time_ms:>10.2f}  {avg_ms:>10.4f}  {pct:>10.1f}%")
        
        print("-"*70)
        print(f"{'TOTAL':<20} {total_time:>10.2f}  {total_time/n:>10.4f}  {100.0:>10.1f}%")
        print("="*70)
        
        # Identify bottleneck
        operation_times = {k: v for k, v in self.timings.items() if k != 'total'}
        max_time = max(operation_times.values())
        bottleneck = [k for k, v in operation_times.items() if v == max_time][0]
        print(f"\n BOTTLENECK: {bottleneck} ({max_time/n:.4f} ms per call)")
        
        if bottleneck == 'expert_compute':
            print("    This is EXPECTED - expert computation should dominate")
            print("   The experts are doing real work (matrix multiplications)")
        elif bottleneck == 'dispatch' or bottleneck == 'combine':
            print("     Dispatch/combine is slow - CUDA kernels should help here!")
            if self.call_counts['cuda_routing'] == 0:
                print("   But CUDA isn't being used - check thresholds")
        elif bottleneck == 'pytorch_routing':
            print("    Using PyTorch routing instead of CUDA!")
            print("   This means CUDA acceleration is NOT working")
        elif bottleneck == 'cuda_routing':
            print("     CUDA routing is the bottleneck")
            print("   This is unusual - may indicate kernel inefficiency")
        
        # Speedup analysis
        if self.timings['cuda_routing'] > 0 and self.timings['pytorch_routing'] > 0:
            cuda_avg = self.timings['cuda_routing'] / max(self.call_counts['cuda_routing'], 1)
            pytorch_avg = self.timings['pytorch_routing'] / max(self.call_counts['pytorch_routing'], 1)
            if cuda_avg > 0:
                speedup = pytorch_avg / cuda_avg
                print(f"\n ROUTING SPEEDUP: {speedup:.2f}x")
                if speedup > 2.0:
                    print(f"    EXCELLENT - CUDA is {speedup:.1f}x faster than PyTorch!")
                elif speedup > 1.2:
                    print(f"    GOOD - CUDA is {speedup:.1f}x faster")
                elif speedup > 0.8:
                    print(f"     Marginal speedup - only {speedup:.1f}x")
                else:
                    print(f"    CUDA is SLOWER - only {speedup:.1f}x (problem too small?)")


def debug_moe_performance(config):
    """Run comprehensive debugging."""
    print("\n" + "="*70)
    print("CREATING TEST MODEL")
    print("="*70)
    
    moe_layer = SimplifiedMoELayer(config).cuda()
    
    print(f"\nConfiguration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num experts: {config.num_experts}")
    print(f"  Top-K: {config.moe_top_k}")
    print(f"  CUDA enabled: {moe_layer.use_cuda_ops}")
    
    batch_size = 2
    seq_len = 256
    x = torch.randn(batch_size, seq_len, config.hidden_size, device='cuda')
    
    print(f"\nTest input: {batch_size}x{seq_len}x{config.hidden_size} = {batch_size*seq_len} tokens")
    
    # Warmup
    print("\nWarming up (10 iterations)...")
    for _ in range(10):
        _ = moe_layer(x)
    
    # Reset stats
    moe_layer.timings = {k: 0.0 for k in moe_layer.timings}
    moe_layer.call_counts = {k: 0 for k in moe_layer.call_counts}
    
    # Profile
    print("Profiling (100 iterations)...")
    for _ in range(100):
        output, aux_loss = moe_layer(x)
    
    # Print results
    moe_layer.print_profile()
    
    # Additional checks
    print("\n" + "="*70)
    print("DIAGNOSTIC CHECKS")
    print("="*70)
    
    try:
        from Main_Scripts.core.wrappers.moe_cuda_wrapper import MoECUDAOps
        total_tokens = batch_size * seq_len
        should_use = MoECUDAOps.should_use_cuda(
            total_tokens,
            config.num_experts,
            config.hidden_size,
            True,
            True
        )
        print(f"MoECUDAOps.should_use_cuda() returns: {should_use}")
        print(f"  Tokens: {total_tokens} (threshold: {MoECUDAOps.CUDA_THRESHOLD_TOKENS})")
        print(f"  Experts: {config.num_experts} (threshold: {MoECUDAOps.CUDA_THRESHOLD_EXPERTS})")
        print(f"  Hidden: {config.hidden_size} (threshold: {MoECUDAOps.CUDA_THRESHOLD_HIDDEN})")
        
        if not should_use:
            print("\n  Problem size is BELOW CUDA thresholds!")
            print("   Try increasing batch_size or seq_len for better speedup")
    except ImportError:
        print("  Could not import MoECUDAOps for threshold checks")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    from dataclasses import dataclass
    
    @dataclass
    class SimpleConfig:
        # Architecture
        hidden_size: int = 128
        intermediate_size: int = 512
        num_experts: int = 8
        moe_top_k: int = 2
        num_layers: int = 2
        num_heads: int = 2
        num_kv_heads: int = 2
        vocab_size: int = 50257
        seq_length: int = 2048
        
        # MoE settings
        use_moe: bool = True
        use_cuda_moe: bool = True
        capacity_factor: float = 1.25
        routing_temperature: float = 1.0
        routing_noise_std: float = 0.1
        load_balancing_weight: float = 0.01
        
        # Training settings
        dropout: float = 0.0
        init_std: float = 0.02
        rms_norm_eps: float = 1e-6
        gradient_checkpointing: bool = False
    
    config = SimpleConfig()
    
    debug_moe_performance(config)