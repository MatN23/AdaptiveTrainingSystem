# Copyright (c) 2025 MatN23. All rights reserved.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import torch


def _normalize_sm(token: str) -> str:
    cleaned = token.strip().lower().replace("sm_", "").replace("compute_", "")
    cleaned = cleaned.replace(".", "").replace("+ptx", "")
    return cleaned if cleaned.isdigit() and len(cleaned) >= 2 else ""


def _parse_sm_list(raw_value: str):
    sms = []
    normalized = raw_value.replace(",", ";").replace(" ", ";")
    for token in normalized.split(";"):
        sm = _normalize_sm(token)
        if sm and sm not in sms:
            sms.append(sm)
    return sms


def _resolve_target_sms():
    env_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    parsed_env_sms = _parse_sm_list(env_arch_list)
    if parsed_env_sms:
        return parsed_env_sms

    if torch.cuda.is_available():
        detected_sms = []
        for device_idx in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(device_idx)
            sm = f"{major}{minor}"
            if sm not in detected_sms:
                detected_sms.append(sm)
        if detected_sms:
            return detected_sms

    parsed_torch_sms = _parse_sm_list(";".join(torch.cuda.get_arch_list()))
    if parsed_torch_sms:
        return parsed_torch_sms

    return ["75"]


target_sms = _resolve_target_sms()
nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--extra-device-vectorization",
]
for sm in target_sms:
    nvcc_flags.extend(["-gencode", f"arch=compute_{sm},code=sm_{sm}"])
if target_sms == ["75"]:
    nvcc_flags.append("-Xptxas=-dlcm=ca")

setup(
    name='moe_cuda_ops',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda_ops',
            sources=[
                'moe_cuda_ops.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': nvcc_flags
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)