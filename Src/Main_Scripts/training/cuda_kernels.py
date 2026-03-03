# cuda_kernels.py
import ctypes
import logging
import os
import subprocess
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_CU_FILE   = os.path.join(_THIS_DIR, "fused_grad_clip.cu")
_SO_FILE   = os.path.join(_THIS_DIR, "fused_grad_clip.so")

# ---------------------------------------------------------------------------
# Compile if needed
# ---------------------------------------------------------------------------
def _compile_so() -> bool:
    if not os.path.exists(_CU_FILE):
        logger.error("❌  Source not found: %s", _CU_FILE)
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
    logger.info("🔨  Compiling %s → %s  [%s]", _CU_FILE, _SO_FILE, sm)
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            logger.error("❌  nvcc failed:\n%s\n%s", res.stdout, res.stderr)
            return False
        logger.info("✅  Compiled successfully")
        return True
    except Exception as exc:
        logger.error("❌  Compilation exception: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Load & bind
# ---------------------------------------------------------------------------
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
        logger.error("❌  Failed to dlopen %s: %s", _SO_FILE, exc)
        return False

    # fused_grad_clip_launcher(void** ptrs, int* sizes, int n,
    #                          float max_norm, void* buf, int fp16, void* stream)
    _lib.fused_grad_clip_launcher.restype  = ctypes.c_int
    _lib.fused_grad_clip_launcher.argtypes = [
        ctypes.c_void_p,                   # grad_ptrs_device  (device ptr to ptr array)
        ctypes.POINTER(ctypes.c_int),       # grad_sizes_device
        ctypes.c_int,                       # num_tensors
        ctypes.c_float,                     # max_norm
        ctypes.c_void_p,                    # norm_buffer
        ctypes.c_int,                       # use_fp16
        ctypes.c_void_p,                    # stream (cudaStream_t)
    ]

    _lib.fused_grad_clip_get_results.restype  = None
    _lib.fused_grad_clip_get_results.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]

    logger.info("✅  fused_grad_clip CUDA kernel loaded")
    return True


# ---------------------------------------------------------------------------
# Per-session buffer cache  (one buffer per CUDA device)
# ---------------------------------------------------------------------------
_norm_buffers: dict[int, torch.Tensor] = {}

def _get_norm_buffer(device: torch.device) -> torch.Tensor:
    idx = device.index if device.index is not None else 0
    if idx not in _norm_buffers:
        # float32 tensor with 8 elements = 32 bytes, guaranteed 8-byte aligned
        _norm_buffers[idx] = torch.zeros(8, dtype=torch.float32, device=device)
    return _norm_buffers[idx]


# ---------------------------------------------------------------------------
# Build device pointer arrays
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_CUDA_KERNELS_AVAILABLE = _load_lib()


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
    # Collect tensors with gradients
    grads = [
        p.grad.detach().contiguous()
        for p in parameters
        if p.grad is not None
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

    device   = grads[0].device
    use_fp16 = int(grads[0].dtype == torch.float16)

    # Ensure all grads match expected dtype
    expected_dtype = torch.float16 if use_fp16 else torch.float32
    grads = [g.to(expected_dtype) for g in grads]

    # Build device-side pointer + size arrays
    ptr_array  = _make_ptr_array(grads)
    size_array = _make_size_array(grads)
    norm_buf   = _get_norm_buffer(device)

    num_tensors = ctypes.c_int(len(grads))
    c_max_norm  = ctypes.c_float(max_norm)
    c_fp16      = ctypes.c_int(use_fp16)

    ret = _lib.fused_grad_clip_launcher(
        ctypes.c_void_p(ptr_array.data_ptr()),
        ctypes.cast(size_array.data_ptr(), ctypes.POINTER(ctypes.c_int)),
        num_tensors,
        c_max_norm,
        ctypes.c_void_p(norm_buf.data_ptr()),
        c_fp16,
        ctypes.c_void_p(0),   # default stream
    )

    if ret != 0:
        logger.warning("fused_grad_clip_launcher returned error %d — "
                       "falling back to PyTorch", ret)
        total_norm = torch.norm(
            torch.stack([torch.norm(g.float(), 2.0) for g in grads]), 2.0
        ).item()
        return float(total_norm)

    # Sync and read results back
    torch.cuda.synchronize(device)

    out_norm     = ctypes.c_float(0.0)
    out_coef     = ctypes.c_float(0.0)
    _lib.fused_grad_clip_get_results(
        ctypes.c_void_p(norm_buf.data_ptr()),
        ctypes.byref(out_norm),
        ctypes.byref(out_coef),
    )

    return float(out_norm.value)