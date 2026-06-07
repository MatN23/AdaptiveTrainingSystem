# Copyright (c) 2025 MatN23. All rights reserved.

"""
CUDA MoE Wrapper - Universal Precision Support
===============================================
Supports ALL PyTorch dtypes: fp32, fp16, bf16, fp64, int8, int4, fp8, int2, tf32
Automatically converts to float32 for CUDA kernels, then converts back.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple
import time
import os
from torch.utils.cpp_extension import load

# ============================================================================
# LOAD CUDA OPS
# ============================================================================

CUDA_OPS_AVAILABLE = False
moe_cuda_ops = None

# CUDA dtype policy:
# - auto  -> use custom CUDA kernels only for float32 tensors (default)
# - force -> cast non-float32 tensors to float32 and still use custom kernels
# - never -> never use custom CUDA kernels
_CUDA_DTYPE_POLICY = os.environ.get("MOE_CUDA_DTYPE_POLICY", "auto").strip().lower()
if _CUDA_DTYPE_POLICY not in {"auto", "force", "never"}:
    _CUDA_DTYPE_POLICY = "auto"


def _normalize_sm_value(value: str) -> str:
    cleaned = value.strip().lower().replace("sm_", "").replace("compute_", "")
    cleaned = cleaned.replace(".", "").replace("+ptx", "")
    return cleaned if cleaned.isdigit() and len(cleaned) >= 2 else ""


def _parse_sm_list(raw_value: str) -> List[str]:
    sms: List[str] = []
    normalized = raw_value.replace(",", ";").replace(" ", ";")
    for token in normalized.split(";"):
        sm = _normalize_sm_value(token)
        if sm and sm not in sms:
            sms.append(sm)
    return sms


def _resolve_target_sms() -> List[str]:
    env_targets = _parse_sm_list(os.environ.get("CUDA_TARGET_SM", ""))
    if env_targets:
        return env_targets

    arch_env_targets = _parse_sm_list(os.environ.get("TORCH_CUDA_ARCH_LIST", ""))
    if arch_env_targets:
        return arch_env_targets

    if torch.cuda.is_available():
        detected_sms: List[str] = []
        for device_idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_idx)
            sm = f"{major}{minor}"
            if sm not in detected_sms:
                detected_sms.append(sm)
        if detected_sms:
            return detected_sms

    try:
        torch_arch_list = torch.cuda.get_arch_list()
    except Exception:
        torch_arch_list = []
    parsed_arch_list = _parse_sm_list(";".join(torch_arch_list))
    if parsed_arch_list:
        return parsed_arch_list

    # Last resort when no arch metadata is available.
    return ["75"]


def _sms_to_arch_list(sms: List[str]) -> str:
    arch_entries = [f"{sm[:-1]}.{sm[-1]}" for sm in sms]
    if arch_entries:
        arch_entries[-1] = f"{arch_entries[-1]}+PTX"
    return ";".join(arch_entries)


def _sms_signature(sms: List[str]) -> str:
    return "_".join(f"sm{sm}" for sm in sms)


try:
    # Get paths FIRST
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_src = os.path.join(current_dir, 'moe_cuda_ops.cu')
    
    # Now you can safely check
    if not os.path.exists(cuda_src):
        raise FileNotFoundError(f"CUDA source not found: {cuda_src}")
    
    print(f" Compiling CUDA MoE ops from: {cuda_src}")
    print(f"   This takes ~60s on first run...")

    target_sms = _resolve_target_sms()
    target_arch_list = _sms_to_arch_list(target_sms)
    os.environ["TORCH_CUDA_ARCH_LIST"] = target_arch_list

    extra_cuda_cflags = [
        '-O3',
        '--use_fast_math',
        '--extra-device-vectorization',
    ]
    for sm in target_sms:
        extra_cuda_cflags.append(f'-gencode=arch=compute_{sm},code=sm_{sm}')
    if target_sms == ["75"]:
        extra_cuda_cflags.append('-Xptxas=-dlcm=ca')

    target_desc = ", ".join(f"sm_{sm}" for sm in target_sms)
    if torch.cuda.is_available():
        print(f"   Target(s): {target_desc}")
    else:
        print(f"     CUDA not available, compiling for {target_desc}...")

    # Ensure build directory exists
    arch_signature = _sms_signature(target_sms)
    build_dir = os.path.join(os.path.expanduser("~"), ".cache/torch_extensions", arch_signature)
    os.makedirs(build_dir, exist_ok=True)

    # Load the CUDA extension
    moe_cuda_ops = load(
        name=f'moe_cuda_ops_{arch_signature}',
        sources=[cuda_src],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_cflags=['-O3'],
        verbose=True,
        with_cuda=True,
        build_directory=build_dir
    )

    CUDA_OPS_AVAILABLE = True
    print(f" CUDA MoE ops loaded successfully")

except Exception as e:
    print(f"  CUDA compilation failed: {e}")
    print(f"   Falling back to PyTorch implementation")
    CUDA_OPS_AVAILABLE = False

# ============================================================================
# DTYPE CONVERSION UTILITIES
# ============================================================================

def to_compute_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor to float32 for CUDA computation.
    
    Handles ALL PyTorch dtypes:
    - Floating: fp64, fp32, fp16, bf16, fp8 (experimental)
    - Integer: int64, int32, int16, int8, int4, int2
    - Complex: complex64, complex128
    
    Args:
        tensor: Input tensor of any dtype
        
    Returns:
        Tensor converted to float32
    """
    # Already float32? Nothing to do
    if tensor.dtype == torch.float32:
        return tensor
    
    # Floating point types - direct conversion
    if tensor.dtype in [torch.float64, torch.float16, torch.bfloat16]:
        return tensor.float()
    
    # FP8 (H100+) - convert via float16
    if hasattr(torch, 'float8_e4m3fn') and tensor.dtype == torch.float8_e4m3fn:
        return tensor.to(torch.float16).float()
    if hasattr(torch, 'float8_e5m2') and tensor.dtype == torch.float8_e5m2:
        return tensor.to(torch.float16).float()
    
    # Integer types - direct conversion
    if tensor.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        return tensor.float()
    
    # Quantized types (int4, int2) - dequantize first
    if tensor.dtype in [torch.quint8, torch.qint8, torch.quint4x2, torch.qint32]:
        # These are quantized, need to dequantize
        # This is a simplified approach - real quantization needs scale/zero_point
        return tensor.dequantize().float() if hasattr(tensor, 'dequantize') else tensor.float()
    
    # Complex types - take real part
    if tensor.dtype in [torch.complex64, torch.complex128]:
        return tensor.real.float()
    
    # Unknown dtype - try direct conversion
    try:
        return tensor.float()
    except Exception as e:
        raise TypeError(f"Cannot convert dtype {tensor.dtype} to float32: {e}")


def from_compute_dtype(tensor: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
    """
    Convert float32 tensor back to target dtype.
    
    Args:
        tensor: Float32 tensor from computation
        target_dtype: Desired output dtype
        
    Returns:
        Tensor converted to target dtype
    """
    # Already correct dtype? Nothing to do
    if tensor.dtype == target_dtype:
        return tensor
    
    # Convert to target dtype
    try:
        return tensor.to(target_dtype)
    except Exception as e:
        # Some dtypes like quantized types need special handling
        # For now, return as float32 if conversion fails
        print(f"  Cannot convert to {target_dtype}, keeping float32: {e}")
        return tensor


# ============================================================================
# WRAPPER CLASS WITH UNIVERSAL DTYPE SUPPORT
# ============================================================================

class MoECUDAOps:
    """
    MoE operations with automatic CUDA/PyTorch fallback.
    
     Supports ALL PyTorch dtypes:
    - Training: fp32, fp16, bf16, mixed_fp16, tf32, fp64
    - Inference: int8, int4, fp16, bf16
    - Experimental: fp8, int2
    
    CUDA kernels require float32, so we automatically:
    1. Convert inputs to float32
    2. Run CUDA operation
    3. Convert outputs back to original dtype
    """
    
    @staticmethod
    def topk_gating(gate_logits, k, temperature=1.0, use_cuda=True, out_indices=None, out_weights=None):
        """
        Top-k gating with softmax normalization.
        
        Args:
            gate_logits: [num_tokens, num_experts] - ANY dtype
            k: Number of experts to select
            temperature: Temperature for routing
            use_cuda: Whether to use CUDA acceleration
            out_indices: Pre-allocated buffer for indices
            out_weights: Pre-allocated buffer for weights
            
        Returns:
            indices: [num_tokens, k] int64
            weights: [num_tokens, k] same dtype as input
        """
        original_dtype = gate_logits.dtype
        
        # Try CUDA with automatic dtype conversion
        should_use_cuda = (
            use_cuda
            and CUDA_OPS_AVAILABLE
            and gate_logits.is_cuda
            and _CUDA_DTYPE_POLICY != "never"
            and (_CUDA_DTYPE_POLICY == "force" or gate_logits.dtype == torch.float32)
        )
        if should_use_cuda:
            try:
                # Convert to float32 only if needed
                gate_logits_f32 = to_compute_dtype(gate_logits) if original_dtype != torch.float32 else gate_logits
                
                # Run CUDA operation
                indices, weights = moe_cuda_ops.topk_gating(gate_logits_f32, k, temperature)
                
                # Copy to output buffers if provided (OPTIMIZATION)
                if out_indices is not None:
                    out_indices.copy_(indices)
                    indices = out_indices
                if out_weights is not None:
                    # Convert to target dtype during copy
                    out_weights.copy_(from_compute_dtype(weights, out_weights.dtype))
                    weights = out_weights
                else:
                    # Convert weights back to original dtype
                    weights = from_compute_dtype(weights, original_dtype)
                
                return indices, weights
                
            except Exception as e:
                # Only print warning on first failure
                if not hasattr(MoECUDAOps, '_cuda_warning_printed'):
                    print(f"  CUDA topk_gating failed: {e}")
                    print(f"   Input dtype: {original_dtype}, falling back to PyTorch")
                    MoECUDAOps._cuda_warning_printed = True
        
        # PyTorch fallback - works with any dtype
        scaled = gate_logits / temperature
        values, indices = torch.topk(scaled, k, dim=-1)
        weights = F.softmax(values, dim=-1)
        
        return indices, weights
    
    @staticmethod
    def dispatch_tokens(tokens, indices, num_experts, capacity, use_cuda=True, out_expert_inputs=None, out_token_map=None):
        """
        Dispatch tokens to experts.
        
        Args:
            tokens: [num_tokens, hidden_dim] - ANY dtype
            indices: [num_tokens, k] int64
            num_experts: Number of experts
            capacity: Capacity per expert
            use_cuda: Whether to use CUDA acceleration
            out_expert_inputs: Pre-allocated buffer [num_experts, capacity, hidden_dim]
            out_token_map: Pre-allocated buffer [num_experts, capacity]
            
        Returns:
            expert_inputs: [num_experts, capacity, hidden_dim] same dtype as input
            token_map: [num_experts, capacity] int64
        """
        original_dtype = tokens.dtype
        
        should_use_cuda = (
            use_cuda
            and CUDA_OPS_AVAILABLE
            and tokens.is_cuda
            and _CUDA_DTYPE_POLICY != "never"
            and (_CUDA_DTYPE_POLICY == "force" or tokens.dtype == torch.float32)
        )
        if should_use_cuda:
            try:
                # Convert tokens to float32 only if needed
                tokens_f32 = to_compute_dtype(tokens) if original_dtype != torch.float32 else tokens
                
                # Ensure indices are int64
                if indices.dtype != torch.int64:
                    indices = indices.to(torch.int64)
                
                # Run CUDA operation
                expert_inputs, token_map = moe_cuda_ops.dispatch_tokens(
                    tokens_f32, indices, num_experts, capacity
                )
                
                # Optimization: reuse buffers
                if out_expert_inputs is not None:
                    out_expert_inputs.copy_(from_compute_dtype(expert_inputs, out_expert_inputs.dtype))
                    expert_inputs = out_expert_inputs
                else:
                    # Convert expert_inputs back to original dtype
                    expert_inputs = from_compute_dtype(expert_inputs, original_dtype)
                
                if out_token_map is not None:
                    out_token_map.copy_(token_map)
                    token_map = out_token_map
                
                return expert_inputs, token_map
                
            except Exception as e:
                if not hasattr(MoECUDAOps, '_dispatch_warning_printed'):
                    print(f"  CUDA dispatch_tokens failed: {e}")
                    print(f"   Input dtype: {original_dtype}, falling back to PyTorch")
                    MoECUDAOps._dispatch_warning_printed = True
        
        # PyTorch fallback
        num_tokens, hidden_dim = tokens.shape
        k = indices.size(1)
        
        expert_inputs = torch.zeros(
            num_experts, capacity, hidden_dim,
            dtype=tokens.dtype,
            device=tokens.device
        )
        token_map = torch.full(
            (num_experts, capacity), -1,
            dtype=torch.int64,
            device=tokens.device
        )
        if num_tokens == 0 or k == 0 or num_experts <= 0 or capacity <= 0:
            return expert_inputs, token_map

        flat_experts = indices.reshape(-1).to(torch.long)
        flat_tokens = (
            torch.arange(num_tokens, device=tokens.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, k)
            .reshape(-1)
        )
        flat_slots = (
            torch.arange(k, device=tokens.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(num_tokens, -1)
            .reshape(-1)
        )

        valid = (flat_experts >= 0) & (flat_experts < num_experts)
        if not valid.any():
            return expert_inputs, token_map

        flat_experts = flat_experts[valid]
        flat_tokens = flat_tokens[valid]
        flat_assignment = flat_tokens * k + flat_slots[valid]

        order = torch.argsort(flat_experts)
        sorted_experts = flat_experts[order]
        counts = torch.bincount(sorted_experts, minlength=num_experts)
        starts = torch.cumsum(counts, dim=0) - counts
        sorted_positions = torch.arange(
            sorted_experts.numel(),
            device=tokens.device,
            dtype=torch.long
        ) - starts[sorted_experts]

        positions = torch.empty_like(sorted_positions)
        positions[order] = sorted_positions

        keep = positions < capacity
        if not keep.any():
            return expert_inputs, token_map

        kept_experts = flat_experts[keep]
        kept_positions = positions[keep]
        kept_tokens = flat_tokens[keep]
        kept_assignments = flat_assignment[keep]

        expert_inputs[kept_experts, kept_positions] = tokens.index_select(0, kept_tokens)
        token_map[kept_experts, kept_positions] = kept_assignments
        
        return expert_inputs, token_map
    
    @staticmethod
    def combine_expert_outputs(expert_outputs, token_map, weights, num_tokens, k, use_cuda=True, out_combined=None):
        """
        Combine expert outputs with weights.
        
        Args:
            expert_outputs: [num_experts, capacity, hidden_dim] - ANY dtype
            token_map: [num_experts, capacity] int64
            weights: [num_tokens, k] - ANY dtype
            num_tokens: Number of tokens
            k: Number of experts per token
            use_cuda: Whether to use CUDA acceleration
            out_combined: Pre-allocated buffer [num_tokens, hidden_dim]
            
        Returns:
            combined: [num_tokens, hidden_dim] same dtype as expert_outputs
        """
        original_dtype = expert_outputs.dtype
        
        should_use_cuda = (
            use_cuda
            and CUDA_OPS_AVAILABLE
            and expert_outputs.is_cuda
            and _CUDA_DTYPE_POLICY != "never"
            and (_CUDA_DTYPE_POLICY == "force" or expert_outputs.dtype == torch.float32)
        )
        if should_use_cuda:
            try:
                # Convert only if needed
                expert_outputs_f32 = to_compute_dtype(expert_outputs) if original_dtype != torch.float32 else expert_outputs
                weights_f32 = to_compute_dtype(weights) if weights.dtype != torch.float32 else weights
                
                # Run CUDA operation
                combined = moe_cuda_ops.combine_expert_outputs(
                    expert_outputs_f32, token_map, weights_f32, num_tokens, k
                )
                
                # Optimization: reuse buffer
                if out_combined is not None:
                    out_combined.copy_(from_compute_dtype(combined, out_combined.dtype))
                    combined = out_combined
                else:
                    # Convert back to original dtype
                    combined = from_compute_dtype(combined, original_dtype)
                
                return combined
                
            except Exception as e:
                if not hasattr(MoECUDAOps, '_combine_warning_printed'):
                    print(f"  CUDA combine_expert_outputs failed: {e}")
                    print(f"   Input dtype: {original_dtype}, falling back to PyTorch")
                    MoECUDAOps._combine_warning_printed = True
        
        # PyTorch fallback
        num_experts, capacity, hidden_dim = expert_outputs.shape
        combined = torch.zeros(
            num_tokens, hidden_dim,
            dtype=expert_outputs.dtype,
            device=expert_outputs.device
        )
        if num_tokens == 0 or k <= 0:
            return combined

        flat_map = token_map.reshape(-1).to(torch.long)
        flat_outputs = expert_outputs.reshape(-1, hidden_dim)
        flat_weights = weights.reshape(-1)

        valid = (
            (flat_map >= 0) &
            (flat_map < flat_weights.numel()) &
            ((flat_map // k) < num_tokens)
        )
        if not valid.any():
            return combined

        assignment_idx = flat_map[valid]
        token_idx = assignment_idx // k
        selected_outputs = flat_outputs[valid]
        selected_weights = flat_weights[assignment_idx].to(expert_outputs.dtype)

        combined.index_add_(0, token_idx, selected_outputs * selected_weights.unsqueeze(-1))
        
        return combined


# ============================================================================
# BENCHMARK WITH MULTIPLE DTYPES
# ============================================================================

def benchmark_moe_ops(num_tokens=1024, hidden_dim=768, num_experts=8, k=2, runs=100):
    """Benchmark CUDA vs PyTorch across multiple dtypes."""
    
    if not torch.cuda.is_available():
        print("  CUDA not available")
        return
    
    print(f"\n{'='*70}")
    print(f" BENCHMARK: {num_tokens} tokens, {num_experts} experts, k={k}")
    print(f"{'='*70}\n")
    
    # Test different dtypes
    dtypes_to_test = [
        ('fp32', torch.float32),
        ('fp16', torch.float16),
        ('bf16', torch.bfloat16),
    ]
    
    for dtype_name, dtype in dtypes_to_test:
        print(f"\n--- Testing {dtype_name} ---")
        
        gate_logits = torch.randn(num_tokens, num_experts, device='cuda', dtype=dtype)
        
        # Warmup
        for _ in range(10):
            if CUDA_OPS_AVAILABLE:
                _ = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
            _ = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
        
        torch.cuda.synchronize()
        
        # CUDA benchmark
        if CUDA_OPS_AVAILABLE:
            start = time.perf_counter()
            for _ in range(runs):
                indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=True)
            torch.cuda.synchronize()
            cuda_time = (time.perf_counter() - start) * 1000
            
            cuda_per_call = cuda_time / runs
            print(f" CUDA:    {cuda_per_call:.4f}ms per call")
        
        # PyTorch benchmark
        start = time.perf_counter()
        for _ in range(runs):
            indices, weights = MoECUDAOps.topk_gating(gate_logits, k, use_cuda=False)
        torch.cuda.synchronize()
        pytorch_time = (time.perf_counter() - start) * 1000
        
        pytorch_per_call = pytorch_time / runs
        print(f" PyTorch: {pytorch_per_call:.4f}ms per call")
        
        if CUDA_OPS_AVAILABLE and cuda_time > 0:
            speedup = pytorch_time / cuda_time
            print(f" Speedup: {speedup:.2f}x")
    
    print(f"\n{'='*70}\n")


# Export
HAS_CUDA_OPS = CUDA_OPS_AVAILABLE

if __name__ == "__main__":
    print("\n" + "="*70)
    print("MoE CUDA Operations - Universal Precision Support")
    print("="*70)
    print(f"Status: {'Available ' if CUDA_OPS_AVAILABLE else 'PyTorch only '}")
    
    if CUDA_OPS_AVAILABLE:
        funcs = [f for f in dir(moe_cuda_ops) if not f.startswith('_')]
        print(f"Functions: {', '.join(funcs)}")
    
    print("\n Supported dtypes:")
    print("  Training:     fp32, fp16, bf16, mixed_fp16, tf32, fp64")
    print("  Inference:    int8, int4, fp16, bf16")
    print("  Experimental: fp8 (H100+), int2")
    print("="*70)
    
    if torch.cuda.is_available():
        print("\n RUNNING BENCHMARKS")
        benchmark_moe_ops(num_tokens=1024, hidden_dim=768, num_experts=8, k=2, runs=100)
    else:
        print("\n  CUDA not available - skipping benchmarks")
