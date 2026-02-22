#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for ALL Custom Kernels
=====================================================
Benchmarks:
1. MoE Operations: Top-K Gating, Dispatch, Combine
2. Transformer Operations: RMSNorm, RoPE, SwiGLU
3. Triton Operations: FP8 Linear
4. Simulated Training: Full Transformer Step
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from pathlib import Path

# Add parent directory to path to import core
sys.path.insert(0, str(Path(__file__).parent.parent))

# --- IMPORTS ---
print("Loading kernels...")

# 1. MoE
try:
    from core.moe_cuda_wrapper import MoECUDAOps, CUDA_OPS_AVAILABLE as MOE_AVAILABLE
except ImportError:
    MOE_AVAILABLE = False
    print("⚠️  MoE wrapper not found")

# 2. Transformer Ops
try:
    from core.cuda_opt_wrapper import (
        FusedRMSNorm, FusedRoPE, FusedSwiGLU, 
        TRANSFORMER_OPS_AVAILABLE as TRANSFORMER_AVAILABLE,
        _load_transformer_ops
    )
    if not _load_transformer_ops():
        TRANSFORMER_AVAILABLE = False
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("⚠️  Transformer wrapper not found")




# ============================================================================
# UTILITIES
# ============================================================================

RESULTS = []

def benchmark_func(func, name="Op", warmup=10, iters=50):
    """Benchmark a function using CUDA events."""
    if not torch.cuda.is_available():
        return 0.0
    
    # Warmup
    try:
        # Warmup loop (Crucial for JIT-compiled kernels like Triton)
        for _ in range(max(2, warmup)):
            func()
        torch.cuda.synchronize()
        
        # Timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            func()
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) / iters
    except Exception as e:
        print(f"Error benchmarking {name}: {e}")
        return 0.0

def run_compare(name, shape_str, func_pt, func_cuda, available=True):
    print(f"\nRunning {name}...")
    
    # PyTorch
    pt_time = benchmark_func(func_pt, f"PyTorch {name}")
    print(f"  PyTorch: {pt_time:.4f} ms")
    
    # Custom
    cuda_time = 0.0
    speedup = 0.0
    
    if available:
        cuda_time = benchmark_func(func_cuda, f"CUDA {name}")
        print(f"  CUDA:    {cuda_time:.4f} ms")
        
        if cuda_time > 0:
            speedup = pt_time / cuda_time
            print(f"  Speedup: {speedup:.2f}x")
    else:
        print(f"  CUDA:    N/A (Not available)")

    RESULTS.append({
        "Operation": name,
        "Input": shape_str,
        "PyTorch (ms)": pt_time,
        "CUDA (ms)": cuda_time,
        "Speedup": speedup
    })

# ============================================================================
# 1. MOE OPERATIONS
# ============================================================================

def benchmark_moe_gating(batch_size, num_experts, k):
    logits = torch.randn(batch_size, num_experts, device='cuda', requires_grad=True)
    
    def run_pt():
        MoECUDAOps.topk_gating(logits, k, use_cuda=False)
        
    def run_cuda():
        MoECUDAOps.topk_gating(logits, k, use_cuda=True)
        
    run_compare("MoE Gating", f"{batch_size}x{num_experts}, k={k}", 
               run_pt, run_cuda, MOE_AVAILABLE)

def benchmark_moe_dispatch(batch_size, hidden_dim, num_experts, k):
    capacity = (batch_size * k + num_experts - 1) // num_experts
    capacity = int(capacity * 1.25) # slack
    
    tokens = torch.randn(batch_size, hidden_dim, device='cuda', requires_grad=True)
    logits = torch.randn(batch_size, num_experts, device='cuda')
    indices, _ = MoECUDAOps.topk_gating(logits, k, use_cuda=MOE_AVAILABLE)
    
    # Pre-allocate for fairness to pure kernel check
    expert_inputs = torch.zeros(num_experts, capacity, hidden_dim, device='cuda')
    token_map = torch.zeros(num_experts, capacity, dtype=torch.int64, device='cuda')
    
    def run_pt():
        MoECUDAOps.dispatch_tokens(tokens, indices, num_experts, capacity, 
                                   use_cuda=False)
        
    def run_cuda():
        MoECUDAOps.dispatch_tokens(tokens, indices, num_experts, capacity, 
                                   use_cuda=True, 
                                   out_expert_inputs=expert_inputs, 
                                   out_token_map=token_map)
        
    run_compare("MoE Dispatch", f"{batch_size}x{hidden_dim}->{num_experts}", 
               run_pt, run_cuda, MOE_AVAILABLE)

def benchmark_moe_combine(batch_size, hidden_dim, num_experts, k):
    capacity = (batch_size * k + num_experts - 1) // num_experts
    capacity = int(capacity * 1.25)
    
    # Setup inputs
    tokens = torch.randn(batch_size, hidden_dim, device='cuda')
    logits = torch.randn(batch_size, num_experts, device='cuda')
    indices, weights = MoECUDAOps.topk_gating(logits, k, use_cuda=MOE_AVAILABLE)
    expert_inputs, token_map = MoECUDAOps.dispatch_tokens(tokens, indices, num_experts, capacity, use_cuda=MOE_AVAILABLE)
    
    # Fake specific output
    expert_outputs = expert_inputs.clone() 
    
    combined = torch.zeros(batch_size, hidden_dim, device='cuda')

    def run_pt():
        MoECUDAOps.combine_expert_outputs(expert_outputs, token_map, weights, batch_size, k, 
                                          use_cuda=False)
        
    def run_cuda():
        MoECUDAOps.combine_expert_outputs(expert_outputs, token_map, weights, batch_size, k, 
                                          use_cuda=True, out_combined=combined)
        
    run_compare("MoE Combine", f"{batch_size}x{hidden_dim}", 
               run_pt, run_cuda, MOE_AVAILABLE)


# ============================================================================
# 2. TRANSFORMER OPS
# ============================================================================

def benchmark_rms_norm(batch_size, seq_len, hidden_size):
    x = torch.randn(batch_size, seq_len, hidden_size, device='cuda', requires_grad=True)
    
    # PT
    model_pt = FusedRMSNorm(hidden_size).cuda()
    model_pt.cuda_enabled = False
    
    # CUDA
    model_cuda = FusedRMSNorm(hidden_size).cuda()
    model_cuda.cuda_enabled = True
    
    def run_pt():
        out = model_pt(x)
        out.sum().backward()
        
    def run_cuda():
        out = model_cuda(x)
        out.sum().backward()
        
    run_compare("RMSNorm", f"{batch_size}x{seq_len}x{hidden_size}", 
               run_pt, run_cuda, TRANSFORMER_AVAILABLE)

def benchmark_rope(batch_size, seq_len, num_heads, head_dim):
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda', requires_grad=True)
    
    model_pt = FusedRoPE(head_dim).cuda()
    model_pt.cuda_enabled = False
    
    model_cuda = FusedRoPE(head_dim).cuda()
    model_cuda.cuda_enabled = True
    
    def run_pt():
        qr, kr = model_pt.apply_rotary_pos_emb(q, k, 0)
        (qr.sum() + kr.sum()).backward()
        
    def run_cuda():
        qr, kr = model_cuda.apply_rotary_pos_emb(q, k, 0)
        (qr.sum() + kr.sum()).backward()
        
    run_compare("RoPE", f"{batch_size}x{seq_len}x{num_heads}", 
               run_pt, run_cuda, TRANSFORMER_AVAILABLE)

def benchmark_swiglu(batch_size, seq_len, hidden_size):
    inter_size = hidden_size * 4
    # Create random inputs representing (gate, up) projections
    # total tokens = batch * seq
    total_tokens = batch_size * seq_len
    
    gate = torch.randn(total_tokens, inter_size, device='cuda', requires_grad=True)
    up = torch.randn(total_tokens, inter_size, device='cuda', requires_grad=True)
    
    # PT: x * Silu(y)
    def run_pt():
        out = gate * F.silu(up)
        out.sum().backward()
        
    # CUDA: Fused kernel
    # Need to check if available
    from core.cuda_opt_wrapper import fused_swiglu
    
    def run_cuda():
        out = fused_swiglu(gate, up)
        out.sum().backward()
        
    run_compare("SwiGLU (Act Only)", f"{total_tokens}x{inter_size}", 
               run_pt, run_cuda, TRANSFORMER_AVAILABLE)





# ============================================================================
# 4. TRAINING SIMULATION
# ============================================================================

class MockTransformerLayer(nn.Module):
    def __init__(self, hidden, heads, ops_available=True):
        super().__init__()
        self.norm = FusedRMSNorm(hidden)
        self.rope = FusedRoPE(hidden//heads)
        self.swiglu = FusedSwiGLU(hidden, hidden*4)
        
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        
        self.set_cuda(ops_available)
        
    def set_cuda(self, enabled):
        self.norm.cuda_enabled = enabled
        self.rope.cuda_enabled = enabled
        self.swiglu.cuda_enabled = enabled
        
    def forward(self, x):
        batch, seq, hidden = x.shape
        heads = 32
        head_dim = hidden // heads
        
        # Norm
        normed = self.norm(x)
        
        # QKV
        q = self.q(normed).view(batch, seq, heads, head_dim)
        k = self.k(normed).view(batch, seq, heads, head_dim)
        v = self.v(normed).view(batch, seq, heads, head_dim)
        
        # RoPE
        q, k = self.rope.apply_rotary_pos_emb(q, k)
        
        # Attn (Mock flash attn via SDPA)
        attn = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
        attn = attn.transpose(1,2).reshape(batch, seq, hidden)
        
        # Out
        h = self.o(attn) + x
        
        # MLP
        normed2 = self.norm(h)
        return self.swiglu(normed2) + h

def benchmark_dense_full_step(batch, seq, hidden):
    model_pt = MockTransformerLayer(hidden, 32).cuda()
    model_pt.set_cuda(False)
    opt_pt = torch.optim.AdamW(model_pt.parameters())
    
    model_cuda = MockTransformerLayer(hidden, 32).cuda()
    model_cuda.set_cuda(True)
    opt_cuda = torch.optim.AdamW(model_cuda.parameters())
    
    x = torch.randn(batch, seq, hidden, device='cuda')
    y = torch.randn(batch, seq, hidden, device='cuda')
    
    def run_pt():
        opt_pt.zero_grad(set_to_none=True)
        out = model_pt(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt_pt.step()
        
    def run_cuda():
        opt_cuda.zero_grad(set_to_none=True)
        out = model_cuda(x)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt_cuda.step()
        
    run_compare("Dense Full Train Step", f"{batch}x{seq}x{hidden}",
               run_pt, run_cuda, TRANSFORMER_AVAILABLE)


class MockMoELayer(nn.Module):
    """
    MoE-focused training benchmark layer.
    Includes:
      - gating
      - top-k routing
      - dispatch/combine
      - lightweight per-expert compute
      - output projection
    """

    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.expert_scale = nn.Parameter(torch.ones(num_experts, 1, hidden_size))
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _capacity(self, num_tokens: int) -> int:
        base = (num_tokens * self.top_k + self.num_experts - 1) // self.num_experts
        return max(1, int(base * 1.25))

    def forward(self, x: torch.Tensor, use_cuda_ops: bool = True) -> torch.Tensor:
        batch, seq, hidden = x.shape
        x_flat = x.reshape(batch * seq, hidden)
        num_tokens = x_flat.size(0)

        gate_logits = self.gate(x_flat)
        topk_indices, topk_weights = MoECUDAOps.topk_gating(
            gate_logits, self.top_k, use_cuda=use_cuda_ops
        )

        capacity = self._capacity(num_tokens)
        expert_inputs, token_map = MoECUDAOps.dispatch_tokens(
            x_flat, topk_indices, self.num_experts, capacity, use_cuda=use_cuda_ops
        )

        # Lightweight expert compute to preserve a realistic autograd training path.
        expert_outputs = F.silu(expert_inputs) * self.expert_scale

        combined = MoECUDAOps.combine_expert_outputs(
            expert_outputs, token_map, topk_weights, num_tokens, self.top_k, use_cuda=use_cuda_ops
        )
        out = self.out_proj(combined.view(batch, seq, hidden))
        return out + x


def benchmark_moe_full_step(batch, seq, hidden, num_experts=8, top_k=2):
    model_pt = MockMoELayer(hidden, num_experts=num_experts, top_k=top_k).cuda()
    model_cuda = MockMoELayer(hidden, num_experts=num_experts, top_k=top_k).cuda()

    # Keep weights aligned for fair comparison.
    model_cuda.load_state_dict(model_pt.state_dict())

    opt_pt = torch.optim.AdamW(model_pt.parameters(), lr=1e-4)
    opt_cuda = torch.optim.AdamW(model_cuda.parameters(), lr=1e-4)

    x = torch.randn(batch, seq, hidden, device='cuda')
    y = torch.randn(batch, seq, hidden, device='cuda')

    def run_pt():
        opt_pt.zero_grad(set_to_none=True)
        out = model_pt(x, use_cuda_ops=False)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt_pt.step()

    def run_cuda():
        opt_cuda.zero_grad(set_to_none=True)
        out = model_cuda(x, use_cuda_ops=True)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt_cuda.step()

    run_compare(
        "MoE Full Train Step",
        f"{batch}x{seq}x{hidden}",
        run_pt,
        run_cuda,
        MOE_AVAILABLE
    )


def print_summary():
    print("\n" + "="*95)
    print(f"{'OPERATION':<20} | {'INPUT':<25} | {'PYTORCH (ms)':<12} | {'CUDA (ms)':<10} | {'SPEEDUP':<8}")
    print("-" * 95)
    for r in RESULTS:
        pt = f"{r['PyTorch (ms)']:.3f}"
        cuda = f"{r['CUDA (ms)']:.3f}" if r['CUDA (ms)'] > 0 else "N/A"
        speedup = f"{r['Speedup']:.2f}x" if r['Speedup'] > 0 else "-"
        print(f"{r['Operation']:<20} | {r['Input']:<25} | {pt:<12} | {cuda:<10} | {speedup:<8}")
    print("="*95 + "\n")

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        sys.exit(0)
        
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    # 1. MoE
    # Large scale to see benefits
    B, hidden, experts, k = 4, 4096, 8, 2
    total_tokens = B * 2048
    benchmark_moe_gating(total_tokens, experts, k)
    benchmark_moe_dispatch(total_tokens, hidden, experts, k)
    benchmark_moe_combine(total_tokens, hidden, experts, k)
    
    # 2. Transformer
    B, L, H, Heads = 4, 2048, 4096, 32
    head_dim = H // Heads
    benchmark_rms_norm(B, L, H)
    benchmark_rope(B, L, Heads, head_dim)
    benchmark_swiglu(B, L, H)
    

    
    # 4. Full Steps
    benchmark_dense_full_step(B, L, H)
    try:
        benchmark_moe_full_step(B, L, H, experts, k)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("⚠️  MoE full-step benchmark OOM at full shape, retrying smaller shape...")
            torch.cuda.empty_cache()
            benchmark_moe_full_step(max(1, B // 2), max(512, L // 2), max(1024, H // 2), experts, k)
        else:
            raise
    
    print_summary()
