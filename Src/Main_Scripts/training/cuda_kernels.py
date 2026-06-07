# cuda_kernels.py
import ctypes
import logging
import os
import subprocess
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Paths
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_CU_FILE   = os.path.join(_THIS_DIR, "fused_grad_clip.cu")
_SO_FILE   = os.path.join(_THIS_DIR, "fused_grad_clip.so")

# Compile if needed
def _compile_so() -> bool:
    if not os.path.exists(_CU_FILE):
        logger.error("  Source not found: %s", _CU_FILE)
        return False

    # Re-compile if .cu is newer than .so
    if (os.path.exists(_SO_FILE) and
            os.path.getmtime(_SO_FILE) >= os.path.getmtime(_CU_FILE)):
        return True

    prop = torch.cuda.get_device_properties(0)
    sm   = f"sm_{prop.major}{prop.minor}"
    arch = f"compute_{prop.major}{prop.minor}"

    cmd = [
        "nvcc", "-O2", "-shared", "-fPIC",
        "--expt-relaxed-constexpr",
        f"--generate-code=arch={arch},code={sm}",
        "-o", _SO_FILE, _CU_FILE,
    ]
    logger.info("  Compiling %s  %s  [%s]", _CU_FILE, _SO_FILE, sm)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            logger.error("  nvcc failed:\n%s\n%s", res.stdout, res.stderr)
            return False
        logger.info("  Compiled successfully")
        return True
    except Exception as exc:
        logger.error("  Compilation exception: %s", exc)
        return False


_lib        = None
_NORM_BYTES = 32          # 8-byte aligned, covers double + 2 floats + padding


def _load_lib() -> bool:
    global _lib
    if _lib is not None:
        return True
    if not _compile_so():
        return False
    try:
        _lib = ctypes.CDLL(_SO_FILE)
    except OSError as exc:
        logger.error("  Failed to dlopen %s: %s", _SO_FILE, exc)
        return False

    #                          float max_norm, void* buf, int fp16, void* stream)
    _lib.fused_grad_clip_launcher.restype  = ctypes.c_int
    _lib.fused_grad_clip_launcher.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int),       # grad_sizes_device
        ctypes.c_int,                       # num_tensors
        ctypes.c_float,                     # max_norm
        ctypes.c_void_p,                    # norm_buffer
        ctypes.c_int,                       # use_fp16
        ctypes.c_void_p,
    ]

    _lib.fused_grad_clip_get_results.restype  = None
    _lib.fused_grad_clip_get_results.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    logger.info("  fused_grad_clip CUDA kernel loaded")
    return True


# Per-session buffer cache  (one buffer per CUDA device)
_norm_buffers: dict[int, torch.Tensor] = {}

def _get_norm_buffer(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else 0
    if idx not in _norm_buffers:
        # float32 tensor with 8 elements = 32 bytes, guaranteed 8-byte aligned
        _norm_buffers[idx] = torch.zeros(8, dtype=torch.float32, device=device)
    return _norm_buffers[idx]


# Build device pointer arrays
def _make_ptr_array(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Returns a 1-D int64 tensor on the same device as tensors[0]
       whose values are the data_ptrs of each tensor."""
    device = tensors[0].device
    ptrs   = [t.data_ptr() for t in tensors]
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def _make_size_array(tensors: list[torch.Tensor]) -> torch.Tensor:
    device = tensors[0].device
    sizes  = [t.numel() for t in tensors]
    return torch.tensor(sizes, dtype=torch.int32, device=device)


# Public API
_CUDA_KERNELS_AVAILABLE = _load_lib()
CUSTOM_KERNELS_AVAILABLE = _CUDA_KERNELS_AVAILABLE and torch.cuda.is_available()


def clip_grad_norm_cuda(
    parameters,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Fused CUDA gradient norm + clip.
    Falls back to torch.nn.utils.clip_grad_norm_ if kernel unavailable.

    Returns the (pre-clip) gradient norm as a Python float.
    """
    params = list(parameters)
    if norm_type != 2.0:
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type).item())

    # Collect tensors with gradients
    grads = [
        p.grad
        for p in params
        if p.grad is not None and p.grad.is_cuda and p.grad.is_floating_point()
    ]
    if not grads:
        return 0.0

    # Fallback path
    if not _CUDA_KERNELS_AVAILABLE:
        import torch.nn.utils as nn_utils
        total_norm = torch.norm(
            torch.stack([torch.norm(g.float(), 2.0) for g in grads]), 2.0
        ).item()
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for g in grads:
                g.mul_(clip_coef)
        return float(total_norm)

    device = grads[0].device
    dtype = grads[0].dtype

    # CUDA kernel supports homogeneous fp16/fp32 tensors.
    if dtype not in (torch.float16, torch.float32):
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type).item())
    if any(g.device != device or g.dtype != dtype or not g.is_contiguous() for g in grads):
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type).item())

    use_fp16 = int(dtype == torch.float16)

    # Build device-side pointer + size arrays
    ptr_array  = _make_ptr_array(grads)
    size_array = _make_size_array(grads)
    norm_buf   = _get_norm_buffer(device)

    num_tensors = ctypes.c_int(len(grads))
    c_max_norm  = ctypes.c_float(max_norm)
    c_fp16      = ctypes.c_int(use_fp16)

    stream = torch.cuda.current_stream(device)
    ret = _lib.fused_grad_clip_launcher(
        ctypes.c_void_p(ptr_array.data_ptr()),
        ctypes.cast(size_array.data_ptr(), ctypes.POINTER(ctypes.c_int)),
        num_tensors,
        c_max_norm,
        ctypes.c_void_p(norm_buf.data_ptr()),
        c_fp16,
        ctypes.c_void_p(stream.cuda_stream),
    )

    if ret != 0:
        logger.warning("fused_grad_clip_launcher returned error %d  "
                       "falling back to PyTorch", ret)
        total_norm = torch.norm(
            torch.stack([torch.norm(g.float(), 2.0) for g in grads]), 2.0
        ).item()
        return float(total_norm)

    # Buffer layout: [double norm_sq][float coef][float norm]
    return float(norm_buf[3].item())


def _compute_loss_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_weights: Optional[torch.Tensor] = None,
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    """Compute loss/accuracy metrics with robust masking and zero-valid-token handling."""
    if logits.dim() == 3:
        vocab_size = logits.size(-1)
        flat_logits = logits.reshape(-1, vocab_size)
        flat_labels = labels.reshape(-1)
    else:
        flat_logits = logits
        flat_labels = labels.reshape(-1)
        vocab_size = flat_logits.size(-1)

    if loss_weights is not None:
        flat_weights = loss_weights.reshape(-1).to(flat_logits.dtype)
    else:
        flat_weights = None

    valid_mask = (
        (flat_labels != pad_token_id)
        & (flat_labels >= 0)
        & (flat_labels < vocab_size)
    )
    if flat_weights is not None:
        valid_mask = valid_mask & (flat_weights > 0)

    valid_token_count = valid_mask.sum()
    has_valid = valid_mask.any()
    denom = valid_token_count.to(flat_logits.dtype).clamp(min=1.0)

    safe_labels = flat_labels.clone()
    safe_labels[~valid_mask] = 0

    loss_per_token = F.cross_entropy(flat_logits, safe_labels, reduction="none")
    valid_mask_f = valid_mask.to(flat_logits.dtype)
    masked_loss_sum = (loss_per_token * valid_mask_f).sum()
    raw_loss = masked_loss_sum / denom

    if flat_weights is not None:
        weighted_loss = loss_per_token * valid_mask_f * flat_weights
        total_weight = (flat_weights * valid_mask_f).sum().clamp(min=1e-8)
        loss = weighted_loss.sum() / total_weight
    else:
        loss = raw_loss

    with torch.no_grad():
        predictions = torch.argmax(flat_logits, dim=-1)
        correct = (predictions == safe_labels).to(flat_logits.dtype) * valid_mask_f
        accuracy = correct.sum() / denom

    loss = torch.where(has_valid, loss, torch.zeros_like(loss, requires_grad=True))
    raw_loss = torch.where(has_valid, raw_loss, torch.zeros_like(raw_loss))
    accuracy = torch.where(has_valid, accuracy, torch.zeros_like(accuracy))
    perplexity = torch.where(
        has_valid,
        torch.exp(torch.clamp(raw_loss.detach(), min=0.0, max=15.0)),
        torch.ones((), device=flat_logits.device, dtype=flat_logits.dtype),
    )

    return {
        "loss": loss,
        "raw_loss": raw_loss.detach(),
        "perplexity": perplexity,
        "valid_tokens": valid_token_count,
        "accuracy": accuracy,
    }


class FusedLoss:
    """
    Backward-compatible loss wrapper used by trainer and benchmark scripts.
    The original CUDA fused loss kernel is optional; this wrapper preserves API.
    """

    def __init__(self):
        self.enabled = torch.cuda.is_available()

    def _cuda_implementation(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Fast CUDA path using fused cross-entropy kernels from PyTorch.
        if logits.dim() == 3:
            vocab_size = logits.size(-1)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)
        else:
            flat_logits = logits
            flat_labels = labels.reshape(-1)
            vocab_size = flat_logits.size(-1)

        valid_mask = (
            (flat_labels != pad_token_id)
            & (flat_labels >= 0)
            & (flat_labels < vocab_size)
        )
        valid_tokens = valid_mask.sum()
        denom = valid_tokens.to(flat_logits.dtype).clamp(min=1.0)

        safe_labels = flat_labels.clone()
        safe_labels[~valid_mask] = 0

        loss_per_token = F.cross_entropy(flat_logits, safe_labels, reduction="none")
        valid_mask_f = valid_mask.to(flat_logits.dtype)
        raw_loss = (loss_per_token * valid_mask_f).sum() / denom
        loss = torch.where(valid_mask.any(), raw_loss, torch.zeros_like(raw_loss, requires_grad=True))

        with torch.no_grad():
            predictions = torch.argmax(flat_logits, dim=-1)
            correct = (predictions == safe_labels).to(flat_logits.dtype) * valid_mask_f
            accuracy = correct.sum() / denom
            accuracy = torch.where(valid_mask.any(), accuracy, torch.zeros_like(accuracy))

        perplexity = torch.where(
            valid_mask.any(),
            torch.exp(torch.clamp(raw_loss.detach(), min=0.0, max=15.0)),
            torch.ones((), device=flat_logits.device, dtype=flat_logits.dtype),
        )

        return {
            "loss": loss,
            "raw_loss": raw_loss.detach(),
            "perplexity": perplexity,
            "valid_tokens": valid_tokens,
            "accuracy": accuracy,
        }

    def _pytorch_fallback(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        return _compute_loss_metrics(logits, labels, loss_weights, pad_token_id)

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: Optional[torch.Tensor] = None,
        pad_token_id: int = 0,
    ) -> Dict[str, torch.Tensor]:
        # Preserve previous behavior: when explicit weights are provided,
        # route to fallback implementation.
        if self.enabled and logits.is_cuda and loss_weights is None:
            return self._cuda_implementation(logits, labels, loss_weights, pad_token_id)
        return self._pytorch_fallback(logits, labels, loss_weights, pad_token_id)


class FusedGradClip:
    """Backward-compatible wrapper around fused gradient clipping CUDA kernel."""

    def __init__(self):
        self.cuda_enabled = bool(CUSTOM_KERNELS_AVAILABLE)
        self.enabled = self.cuda_enabled
        self._implementation = "cuda" if self.cuda_enabled else "pytorch"

    def set_implementation(self, implementation: str):
        impl = str(implementation).strip().lower()
        if impl not in {"auto", "cuda", "pytorch"}:
            raise ValueError(f"Unknown implementation '{implementation}'")
        if impl == "auto":
            self._implementation = "cuda" if self.cuda_enabled else "pytorch"
        elif impl == "cuda" and not self.cuda_enabled:
            self._implementation = "pytorch"
        else:
            self._implementation = impl

    def __call__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        max_norm: float = 1.0,
        norm_type: float = 2.0,
    ) -> float:
        params = list(parameters)
        if self._implementation == "cuda" and self.cuda_enabled:
            return clip_grad_norm_cuda(params, max_norm=max_norm, norm_type=norm_type)
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type=norm_type).item())


__all__ = [
    "CUSTOM_KERNELS_AVAILABLE",
    "FusedLoss",
    "FusedGradClip",
    "clip_grad_norm_cuda",
]