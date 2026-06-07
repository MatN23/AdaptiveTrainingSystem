#!/bin/bash
# Copyright (c) 2025 MatN23. All rights reserved.
# Licensed under the Custom License below.

set -e

echo "=================================================="
echo "Compiling Custom CUDA Kernels"
echo "=================================================="

normalize_arch_list() {
    local raw="$1"
    local output=""
    for token in $(echo "$raw" | tr ',;' ' '); do
        local cleaned
        cleaned=$(echo "$token" | tr '[:upper:]' '[:lower:]' | sed -E 's/^sm_//; s/^compute_//; s/\+ptx$//; s/\.//g')
        if [[ "$cleaned" =~ ^[0-9]{2,}$ ]]; then
            if [[ " $output " != *" $cleaned "* ]]; then
                output="${output} ${cleaned}"
            fi
        fi
    done
    echo "$output" | xargs
}

format_arch_list() {
    local raw="$1"
    local formatted=""
    for arch in $raw; do
        formatted="${formatted}sm_${arch} "
    done
    echo "$formatted" | xargs
}

# Resolve target architectures
if [ -n "${CUDA_TARGET_SM}" ]; then
    GPU_ARCHS=$(normalize_arch_list "${CUDA_TARGET_SM}")
    if [ -z "${GPU_ARCHS}" ]; then
        echo "  Invalid CUDA_TARGET_SM='${CUDA_TARGET_SM}', defaulting to sm_75"
        GPU_ARCHS="75"
    else
        echo " Using forced CUDA_TARGET_SM: $(format_arch_list "${GPU_ARCHS}")"
    fi
elif command -v nvidia-smi &> /dev/null; then
    GPU_ARCHS=$(normalize_arch_list "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr '\n' ' ')")
    if [ -z "${GPU_ARCHS}" ]; then
        echo "  Could not parse GPU compute capability, defaulting to sm_75"
        GPU_ARCHS="75"
    else
        echo " Detected GPU compute capabilities: $(format_arch_list "${GPU_ARCHS}")"
    fi
else
    echo "  nvidia-smi not found, defaulting to sm_75"
    GPU_ARCHS="75"
fi

ARCH_FLAGS=""
for arch in ${GPU_ARCHS}; do
    ARCH_FLAGS="${ARCH_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}"
done

# Optimization flags
NVCC_FLAGS="-O3 ${ARCH_FLAGS} --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v --extra-device-vectorization"
if [ "${GPU_ARCHS}" = "75" ]; then
    NVCC_FLAGS="${NVCC_FLAGS} -Xptxas=-dlcm=ca"
fi

echo ""
echo "Compilation flags: ${NVCC_FLAGS}"
echo ""

# Compile fused loss kernel
echo "1  Compiling fused_loss.cu..."
nvcc ${NVCC_FLAGS} -shared fused_loss.cu -o fused_loss.so 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f fused_loss.so ]; then
    echo "    fused_loss.so compiled successfully"
    ls -lh fused_loss.so
else
    echo "    fused_loss.cu compilation failed"
    exit 1
fi

echo ""

# Compile fused gradient clipping kernel
echo "2  Compiling fused_grad_clip.cu..."
nvcc ${NVCC_FLAGS} -shared fused_grad_clip.cu -o fused_grad_clip.so 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f fused_grad_clip.so ]; then
    echo "    fused_grad_clip.so compiled successfully"
    ls -lh fused_grad_clip.so
else
    echo "    fused_grad_clip.cu compilation failed"
    exit 1
fi

echo ""
echo "=================================================="
echo " All kernels compiled successfully!"
echo "=================================================="
echo ""
echo "Generated files:"
ls -lh *.so
echo ""
echo "Next steps:"
echo "  1. Verify kernels work: python cuda_kernels.py"
echo "  2. Update trainer.py with the new functions"
echo "  3. Run training - kernels will be used automatically"
echo ""
