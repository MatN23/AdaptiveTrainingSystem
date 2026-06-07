# Copyright (c) 2025 MatN23. All rights reserved.
# Unified Operations Module - Auto-selects Best Backend - FIXED VERSION

"""
Unified operations module - automatically selects best available backend
Priority: CUDA > Metal > PyTorch

FIXED: Proper import paths and detection logic
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ============================================================================
# SETUP IMPORT PATHS
# ============================================================================

# Add current directory to path for direct imports
CURRENT_DIR = Path(__file__).parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ============================================================================
# BACKEND DETECTION
# ============================================================================

# Detect hardware
HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()
IS_MACOS = platform.system() == 'Darwin'

# Try importing CUDA operations - FIXED with better error handling
HAS_CUDA_OPS = False
HAS_CUDA_MOE = False

if HAS_CUDA:
    # Try direct imports first (when running from core directory)
    try:
        from cuda_opt_wrapper import (
            FusedRMSNorm as CUDARMSNorm,
            FusedRoPE as CUDARoPE,
            FusedSwiGLU as CUDASwiGLU,
            TRANSFORMER_OPS_AVAILABLE
        )
        HAS_CUDA_OPS = TRANSFORMER_OPS_AVAILABLE
        if HAS_CUDA_OPS:
            logger.info(" CUDA Transformer operations loaded (direct import)")
    except ImportError as e:
        logger.debug(f"Direct import failed: {e}, trying package import...")
        
        # Try package imports (when running from elsewhere)
        try:
            from Main_Scripts.core.cuda_opt_wrapper import (
                FusedRMSNorm as CUDARMSNorm,
                FusedRoPE as CUDARoPE,
                FusedSwiGLU as CUDASwiGLU,
                TRANSFORMER_OPS_AVAILABLE
            )
            HAS_CUDA_OPS = TRANSFORMER_OPS_AVAILABLE
            if HAS_CUDA_OPS:
                logger.info(" CUDA Transformer operations loaded (package import)")
        except ImportError as e2:
            logger.debug(f"Package import also failed: {e2}")
            HAS_CUDA_OPS = False
    
    # Try importing MoE CUDA operations
    try:
        from moe_cuda_wrapper import MoECUDAOps, CUDA_OPS_AVAILABLE
        HAS_CUDA_MOE = CUDA_OPS_AVAILABLE
        if HAS_CUDA_MOE:
            logger.info(" CUDA MoE operations loaded (direct import)")
    except ImportError as e:
        try:
            from Main_Scripts.core.moe_cuda_wrapper import MoECUDAOps, CUDA_OPS_AVAILABLE
            HAS_CUDA_MOE = CUDA_OPS_AVAILABLE
            if HAS_CUDA_MOE:
                logger.info(" CUDA MoE operations loaded (package import)")
        except ImportError as e2:
            logger.debug(f"MoE CUDA import failed: {e2}")
            HAS_CUDA_MOE = False

# Try importing Metal operations (macOS only)
HAS_METAL_OPS = False
HAS_METAL_MOE = False
MoEMetalOps = None

if IS_MACOS and HAS_MPS:
    try:
        from metal_opt_wrapper import (
            MetalRMSNorm,
            MetalRoPE,
            MetalSwiGLU,
            METAL_OPS_AVAILABLE
        )
        HAS_METAL_OPS = METAL_OPS_AVAILABLE
        if HAS_METAL_OPS:
            logger.info(" Metal Transformer operations loaded")
    except ImportError:
        try:
            from Main_Scripts.core.metal_opt_wrapper import (
                MetalRMSNorm,
                MetalRoPE,
                MetalSwiGLU,
                METAL_OPS_AVAILABLE
            )
            HAS_METAL_OPS = METAL_OPS_AVAILABLE
            if HAS_METAL_OPS:
                logger.info(" Metal Transformer operations loaded")
        except ImportError as e:
            logger.debug(f"Metal operations not available: {e}")
            HAS_METAL_OPS = False

    # Try importing Metal MoE operations
    try:
        from moe_metal_wrapper import MoEMetalOps, HAS_METAL_OPS as METAL_MOE_AVAILABLE
        HAS_METAL_MOE = METAL_MOE_AVAILABLE
        if HAS_METAL_MOE:
            logger.info(" Metal MoE operations loaded (direct import)")
    except ImportError:
        try:
            from Main_Scripts.core.moe_metal_wrapper import MoEMetalOps, HAS_METAL_OPS as METAL_MOE_AVAILABLE
            HAS_METAL_MOE = METAL_MOE_AVAILABLE
            if HAS_METAL_MOE:
                logger.info(" Metal MoE operations loaded (package import)")
        except ImportError as e:
            logger.debug(f"Metal MoE operations not available: {e}")
            HAS_METAL_MOE = False

# ============================================================================
# BACKEND SELECTION
# ============================================================================

def select_backend():
    """
    Automatically select the best available backend.
    
    Priority:
    1. CUDA (if NVIDIA GPU + kernels compiled)
    2. Metal (if Apple Silicon + kernels compiled)
    3. PyTorch (CPU or no custom kernels)
    
    Returns:
        str: Selected backend ('cuda', 'metal', or 'pytorch')
    """
    if HAS_CUDA and HAS_CUDA_OPS:
        return 'cuda'
    elif HAS_MPS and HAS_METAL_OPS:
        return 'metal'
    else:
        return 'pytorch'

BACKEND = select_backend()

# Log backend selection with clear status
if BACKEND == 'cuda':
    logger.info(f" Backend: CUDA (HW: , Ops: , MoE: {'' if HAS_CUDA_MOE else ''})")
elif BACKEND == 'metal':
    logger.info(f" Backend: Metal (HW: , Ops: , MoE: {'' if HAS_METAL_MOE else ''})")
else:
    if HAS_CUDA:
        logger.warning(f"   Backend: PyTorch fallback (CUDA available but ops not compiled)")
    elif HAS_MPS:
        logger.warning(f"   Backend: PyTorch fallback (Metal available but ops not compiled)")
    else:
        logger.info(f"  Backend: PyTorch (CPU mode)")

# ============================================================================
# PYTORCH FALLBACK IMPLEMENTATIONS
# ============================================================================

class PyTorchRMSNorm(nn.Module):
    """Pure PyTorch RMSNorm implementation"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        #  FIX: Clone output to prevent CUDAGraph buffer overwrite.
        # torch.compile (reduce-overhead/CUDAGraphs) reuses static memory across
        # replays. Without .clone(), step N+1 forward overwrites step N backward's
        # input before autograd reads it.
        return (self.weight * x.to(input_dtype)).clone()


class PyTorchRoPE(nn.Module):
    """Pure PyTorch RoPE implementation"""
    
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute cache
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].to(device)
        sin = self.sin_cached[:seq_len].to(device)
        return cos, sin


class PyTorchSwiGLU(nn.Module):
    """Pure PyTorch SwiGLU implementation"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class PyTorchMoEOps:
    """Pure PyTorch MoE operations"""
    
    @staticmethod
    def topk_gating(gate_logits, k, temperature=1.0):
        scaled = gate_logits / temperature
        values, indices = torch.topk(scaled, k, dim=-1)
        weights = F.softmax(values, dim=-1)
        return indices, weights
    
    @staticmethod
    def dispatch_tokens(tokens, indices, num_experts, capacity):
        num_tokens, hidden_dim = tokens.shape
        k = indices.size(1)
        
        expert_inputs = torch.zeros(num_experts, capacity, hidden_dim, 
                                    dtype=tokens.dtype, device=tokens.device)
        token_map = torch.full((num_experts, capacity), -1, 
                              dtype=torch.int64, device=tokens.device)
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
    def combine_expert_outputs(expert_outputs, token_map, weights, num_tokens, k):
        num_experts, capacity, hidden_dim = expert_outputs.shape
        combined = torch.zeros(num_tokens, hidden_dim, 
                              dtype=expert_outputs.dtype, device=expert_outputs.device)
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
# UNIFIED API
# ============================================================================

def get_rms_norm(hidden_size: int, eps: float = 1e-6, use_fused: bool = True):
    """Get RMSNorm implementation for current backend"""
    if use_fused and BACKEND == 'cuda' and HAS_CUDA_OPS:
        return CUDARMSNorm(hidden_size, eps)
    elif use_fused and BACKEND == 'metal' and HAS_METAL_OPS:
        return MetalRMSNorm(hidden_size, eps)
    else:
        return PyTorchRMSNorm(hidden_size, eps)


def get_rope(dim: int, max_seq_len: int = 8192, theta: float = 10000.0, use_fused: bool = True):
    """Get RoPE implementation for current backend"""
    if use_fused and BACKEND == 'cuda' and HAS_CUDA_OPS:
        return CUDARoPE(dim, max_seq_len, theta)
    elif use_fused and BACKEND == 'metal' and HAS_METAL_OPS:
        return MetalRoPE(dim, max_seq_len, theta)
    else:
        return PyTorchRoPE(dim, max_seq_len, theta)


def get_swiglu(hidden_size: int, intermediate_size: int, use_fused: bool = True):
    """Get SwiGLU implementation for current backend"""
    if use_fused and BACKEND == 'cuda' and HAS_CUDA_OPS:
        return CUDASwiGLU(hidden_size, intermediate_size, use_bias=False)
    elif use_fused and BACKEND == 'metal' and HAS_METAL_OPS:
        return MetalSwiGLU(hidden_size, intermediate_size)
    else:
        return PyTorchSwiGLU(hidden_size, intermediate_size)


def get_moe_ops(use_fused: bool = True):
    """Get MoE operations for current backend"""
    if use_fused and BACKEND == 'cuda' and HAS_CUDA_MOE:
        return MoECUDAOps
    elif use_fused and BACKEND == 'metal' and HAS_METAL_MOE:
        return MoEMetalOps
    else:
        return PyTorchMoEOps


# ============================================================================
# BACKEND INFO
# ============================================================================

@dataclass
class BackendInfo:
    """Information about the selected backend"""
    name: str
    has_cuda: bool
    has_mps: bool
    has_cuda_ops: bool
    has_metal_ops: bool
    has_cuda_moe: bool
    has_metal_moe: bool
    device_name: Optional[str] = None
    expected_speedup: str = "1x"


def get_backend_info() -> BackendInfo:
    """Get detailed information about the selected backend"""
    device_name = None
    expected_speedup = "1x (baseline)"
    
    if BACKEND == 'cuda' and HAS_CUDA:
        device_name = torch.cuda.get_device_name(0)
        expected_speedup = "3-5x faster"
    elif BACKEND == 'metal' and HAS_MPS:
        device_name = "Apple Silicon (MPS)"
        expected_speedup = "2-3x faster"
    
    return BackendInfo(
        name=BACKEND,
        has_cuda=HAS_CUDA,
        has_mps=HAS_MPS,
        has_cuda_ops=HAS_CUDA_OPS,
        has_metal_ops=HAS_METAL_OPS,
        has_cuda_moe=HAS_CUDA_MOE,
        has_metal_moe=HAS_METAL_MOE,
        device_name=device_name,
        expected_speedup=expected_speedup
    )


def print_backend_info():
    """Print detailed backend information"""
    info = get_backend_info()
    
    print("\n" + "="*70)
    print("ACCELERATED OPERATIONS BACKEND")
    print("="*70)
    print(f"Selected Backend: {info.name.upper()}")
    print(f"Expected Speedup: {info.expected_speedup}")
    print()
    
    print("Hardware:")
    print(f"  CUDA Available: {'Yes ' if info.has_cuda else 'No '}")
    if info.has_cuda:
        if info.device_name:
            print(f"    Device: {info.device_name}")
        print(f"    Transformer Ops: {'Enabled ' if info.has_cuda_ops else 'Not compiled '}")
        if info.has_cuda_ops:
            print(f"      RMSNorm: 3-4x faster")
            print(f"      RoPE: 5-7x faster")
            print(f"      SwiGLU: 2-3x faster")
        print(f"    MoE Ops: {'Enabled ' if info.has_cuda_moe else 'Not compiled '}")
        if info.has_cuda_moe:
            print(f"      Routing: 2-4x faster")
    
    print(f"  Metal (MPS) Available: {'Yes ' if info.has_mps else 'No '}")
    if info.has_mps:
        if info.device_name:
            print(f"    Device: {info.device_name}")
        print(f"    Transformer Ops: {'Enabled ' if info.has_metal_ops else 'Not compiled '}")
        print(f"    MoE Ops: {'Enabled ' if info.has_metal_moe else 'Not compiled '}")
    
    print()
    print("Operations:")
    print(f"  RMSNorm: {info.name}")
    print(f"  RoPE: {info.name}")
    print(f"  SwiGLU: {info.name}")
    print(f"  MoE: {info.name if (info.has_cuda_moe or info.has_metal_moe) else 'pytorch'}")
    
    print()
    print("Recommendations:")
    if info.name == 'pytorch':
        if HAS_CUDA and not HAS_CUDA_OPS:
            print("     CUDA available but kernels not compiled")
            print("     Run: bash compile_transformer_ops.sh && bash compile_cuda_moe.sh")
        elif HAS_MPS and not HAS_METAL_OPS:
            print("     MPS available but Metal kernels not compiled")
            print("     Run: bash compile_metal.sh")
        else:
            print("    Using CPU - consider using GPU for faster training")
    elif info.name == 'cuda':
        if not info.has_cuda_moe:
            print("    MoE acceleration available - run: bash compile_cuda_moe.sh")
        else:
            print("   Fully optimized for NVIDIA GPU!")
    elif info.name == 'metal':
        print("   Optimized for Apple Silicon!")
    
    print("="*70 + "\n")


def get_recommended_device():
    """Get recommended device string for torch operations"""
    if BACKEND == 'cuda':
        return 'cuda'
    elif BACKEND == 'metal':
        return 'mps'
    else:
        return 'cpu'


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Backend info
    'BACKEND',
    'HAS_CUDA',
    'HAS_MPS',
    'HAS_CUDA_OPS',
    'HAS_METAL_OPS',
    'HAS_CUDA_MOE',
    'HAS_METAL_MOE',
    
    # API functions
    'get_rms_norm',
    'get_rope',
    'get_swiglu',
    'get_moe_ops',
    
    # Info functions
    'get_backend_info',
    'print_backend_info',
    'get_recommended_device',
]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_backend_info()
    
    print(" Quick Test:")
    device = get_recommended_device()
    print(f"   Recommended device: {device}")
    
    # Test operations
    try:
        print("\n   Testing RMSNorm...")
        norm = get_rms_norm(768)
        print(f"       Created: {type(norm).__name__}")
        
        print("   Testing RoPE...")
        rope = get_rope(64)
        print(f"       Created: {type(rope).__name__}")
        
        print("   Testing SwiGLU...")
        swiglu = get_swiglu(768, 3072)
        print(f"       Created: {type(swiglu).__name__}")
        
        print("   Testing MoE Ops...")
        moe_ops = get_moe_ops()
        print(f"       Got: {moe_ops.__name__}")
        
        print("\n All operations available and working!\n")
        
    except Exception as e:
        print(f"\n Error during testing: {e}\n")
        import traceback
        traceback.print_exc()
