#!/bin/bash
set -e

CUDA_DIR="./"
CU_FILE="${CUDA_DIR}/transformer_ops.cu"
OUT_FILE="${CUDA_DIR}/transformer_ops.so"

echo "=================================================="
echo "Compiling Transformer CUDA Kernels"
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

NVCC_FLAGS="-O3 ${ARCH_FLAGS} --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v --extra-device-vectorization -lcublasLt"
if [ "${GPU_ARCHS}" = "75" ]; then
    NVCC_FLAGS="${NVCC_FLAGS} -Xptxas=-dlcm=ca"
fi

echo ""
echo "Compilation flags: ${NVCC_FLAGS}"
echo ""

# Compile
echo " Compiling ${CU_FILE}..."
nvcc ${NVCC_FLAGS} -shared ${CU_FILE} -o ${OUT_FILE} 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f ${OUT_FILE} ]; then
    echo " transformer_ops.so compiled successfully in ${CUDA_DIR}"
    ls -lh ${OUT_FILE}
else
    echo " Compilation failed"
    exit 1
fi
