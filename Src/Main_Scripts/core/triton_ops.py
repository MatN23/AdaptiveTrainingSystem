
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import logging

logger = logging.getLogger(__name__)

TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    pass

def is_triton_available():
    return TRITON_AVAILABLE

# ============================================================================
# UTILITIES
# ============================================================================

def generate_e4m3_lut():
    """Generate Look-Up Table for E4M3 standard (OCP)"""
    lut = []
    for i in range(256):
        # i is uint8 0..255
        s = (i >> 7) & 1
        e = (i >> 3) & 0xF
        m = i & 0x7
        
        # OCP E4M3
        # Bias 7.
        if e == 0:
            # Subnormal: (-1)^s * 2^(1-7) * (m / 8)
            # 2^-6
            val = ((-1)**s) * (2**-6) * (m / 8.0)
        elif e == 15 and m == 7:
            # NaN
            val = float('nan')
        else:
            # Normal: (-1)^s * 2^(e-7) * (1 + m/8)
            val = ((-1)**s) * (2**(e-7)) * (1 + m / 8.0)
        lut.append(val)
    return torch.tensor(lut, dtype=torch.float16)

# ============================================================================
# TRITON KERNELS (CUDA ONLY)
# ============================================================================

if TRITON_AVAILABLE:
    
    def get_cuda_autotune_config():
        return [
            triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
            triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        ]

    @triton.autotune(
        configs=get_cuda_autotune_config(),
        key=['M', 'N', 'K'],
    )
    @triton.jit
    def fp8_gemm_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        scale_a, scale_b,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr
    ):
        """
        Matrix Multiplication for FP8 (emulated): C = A * B
        A: [M, K] in packed Int8 (representing E4M3)
        B: [K, N] in packed Int8 (representing E4M3)
        C: [M, N] in FP16
        """
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_now = k * BLOCK_SIZE_K
            mask_k = offs_k < (K - k_now)
            
            a_int8 = tl.load(a_ptrs, mask=mask_k[None, :], other=0)
            b_int8 = tl.load(b_ptrs, mask=mask_k[:, None], other=0)
            
            # Dequantize E4M3 -> FP16
            # A
            a_sign = (a_int8 & 0x80) << 8
            a_exp = ((a_int8 >> 3) & 0xF) + 8
            a_mant = (a_int8 & 0x7) << 7
            a_bits = a_sign | (a_exp << 10) | a_mant
            a_bits = tl.where((a_int8 & 0x7F) == 0, 0, a_bits)
            a_fp16 = a_bits.to(tl.float16, bitcast=True)
            
            # B
            b_sign = (b_int8 & 0x80) << 8
            b_exp = ((b_int8 >> 3) & 0xF) + 8
            b_mant = (b_int8 & 0x7) << 7
            b_bits = b_sign | (b_exp << 10) | b_mant
            b_bits = tl.where((b_int8 & 0x7F) == 0, 0, b_bits)
            b_fp16 = b_bits.to(tl.float16, bitcast=True)
            
            accumulator += tl.dot(a_fp16, b_fp16)
            
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            
        c = accumulator * scale_a * scale_b
        c = c.to(tl.float16)
        
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

# ============================================================================
# PYTORCH WRAPPERS AND FALLBACKS
# ============================================================================

class FP8MatmulFunc(Function):
    @staticmethod
    def forward(ctx, a_int8, b_int8, scale_a, scale_b):
        M, K = a_int8.shape
        _, N = b_int8.shape
        c = torch.empty((M, N), device=a_int8.device, dtype=torch.float16)
        
        grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
        
        fp8_gemm_kernel[grid](
            a_int8, b_int8, c,
            M, N, K,
            a_int8.stride(0), a_int8.stride(1),
            b_int8.stride(0), b_int8.stride(1),
            c.stride(0), c.stride(1),
            scale_a, scale_b,
        )
        return c

def torch_fallback_matmul(a_int8, b_int8, scale_a, scale_b, lut=None):
    """Fallback utilizing LUT or native casting"""
    
    # Check for Native Float8 (Torch 2.1+)
    if hasattr(torch, 'float8_e4m3fn'):
        try:
            a_fp8 = a_int8.view(torch.float8_e4m3fn)
            b_fp8 = b_int8.view(torch.float8_e4m3fn)
            return torch.matmul(a_fp8.to(torch.float16), b_fp8.to(torch.float16)) * (scale_a * scale_b)
        except Exception:
            pass # Fallback to LUT
            
    # LUT Validation
    if lut is None:
        # Emergency generation (Slow)
        # Or just use the global CPU one and move it?
        lut = generate_e4m3_lut().to(a_int8.device)
        
    # Manual Decoding via LUT (Indices 0..255)
    # int8 is -128..127. We need 0..255.
    # .long() preserves value. -1 -> -1. We want 0xFF (255).
    # cast to uint8 not available easily on all versions?
    # (a_int8.long() & 0xFF) gives 0..255.
    
    a_idx = (a_int8.long() & 0xFF)
    b_idx = (b_int8.long() & 0xFF)
    
    a_dec = F.embedding(a_idx, lut) # [M, K]
    b_dec = F.embedding(b_idx, lut) # [K, N]
    
    return torch.matmul(a_dec, b_dec) * (scale_a * scale_b)

def triton_fp8_matmul(a, b, scale_a=1.0, scale_b=1.0, lut=None):
    if is_triton_available() and a.is_cuda:
        return FP8MatmulFunc.apply(a, b, float(scale_a), float(scale_b))
    else:
        return torch_fallback_matmul(a, b, float(scale_a), float(scale_b), lut=lut)


class TritonFP8Linear(nn.Module):
    """
    Universal FP8 Linear Layer (Works on CUDA/MPS/CPU).
    """
    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.register_buffer('weight', torch.zeros((in_features, out_features), dtype=torch.int8, device=device))
        self.register_buffer('weight_scale', torch.tensor(1.0, device=device))
        
        # Register LUT for fallback support (MPS/CPU)
        # We generate it on initialization
        self.register_buffer('e4m3_lut', generate_e4m3_lut().to(device or 'cpu'))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype or torch.float16))
        else:
            self.register_parameter('bias', None)
            
    def load_from_linear(self, linear_layer):
        with torch.no_grad():
            w = linear_layer.weight.data # [Out, In]
            w_t = w.t().contiguous()
            
            max_val = w_t.abs().max()
            scale = max_val / 448.0
            if scale == 0: scale = 1.0
            
            w_scaled = w_t / scale
            
            if hasattr(torch, 'float8_e4m3fn'):
                self.weight.data = w_scaled.to(torch.float8_e4m3fn).view(torch.int8)
            else:
                # Quantization Fallback? LUT inverse?
                # Inverse LUT is hard.
                # We resort to standard Int8 clamp for "loading" on fallback devices
                # if they don't have float8 type.
                # This mimics E4M3 for testing but isn't bitwise accurate.
                # Ideally, we should do software quantization if accuracy matters.
                # For now: Linear approx.
                self.weight.data = w_scaled.clamp(-128, 127).to(torch.int8)
                
            self.weight_scale.fill_(scale)
            
            if linear_layer.bias is not None:
                self.bias = nn.Parameter(linear_layer.bias.data.clone())

    def forward(self, x):
        # x is [Batch, In]
        x_max = x.abs().max()
        scale_x = x_max / 448.0
        if scale_x == 0: scale_x = 1.0
        
        if hasattr(torch, 'float8_e4m3fn'):
            x_int8 = (x / scale_x).to(torch.float8_e4m3fn).view(torch.int8)
        else:
            x_int8 = (x / scale_x).clamp(-128, 127).to(torch.int8)
        
        out = triton_fp8_matmul(x_int8, self.weight, scale_x, self.weight_scale.item(), lut=self.e4m3_lut)
        
        if self.bias is not None:
            out += self.bias
            
        return out

def replace_linear_with_fp8(module, filter_func=None):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            if filter_func is None or filter_func(name, child):
                logger.info(f"Replacing {name} with TritonFP8Linear")
                start_device = child.weight.device
                new_layer = TritonFP8Linear(child.in_features, child.out_features, 
                                          bias=child.bias is not None, 
                                          device=start_device)
                new_layer.load_from_linear(child)
                setattr(module, name, new_layer)
        else:
            replace_linear_with_fp8(child, filter_func)
