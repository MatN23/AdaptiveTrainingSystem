#!/bin/bash
set -e

CUDA_DIR="./"
CU_FILE="${CUDA_DIR}/transformer_ops.cu"
OUT_FILE="${CUDA_DIR}/transformer_ops.so"

echo "=================================================="
echo "Compiling Transformer CUDA Kernels"
echo "=================================================="

# Detect GPU architecture
if command -v nvidia-smi &> /dev/null; then
    GPU_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d '.')
    echo "✅ Detected GPU compute capability: ${GPU_ARCH}"
else
    echo "⚠️  nvidia-smi not found, defaulting to sm_75 (T4)"
    GPU_ARCH=75
fi

ARCH_FLAG="-arch=sm_${GPU_ARCH}"
NVCC_FLAGS="-O3 ${ARCH_FLAG} --compiler-options '-fPIC' --use_fast_math --ptxas-options=-v"

echo ""
echo "Compilation flags: ${NVCC_FLAGS}"
echo ""

# Compile
echo "🔨 Compiling ${CU_FILE}..."
nvcc ${NVCC_FLAGS} -shared ${CU_FILE} -o ${OUT_FILE} 2>&1 | grep -E "ptxas|error|warning" || true

if [ -f ${OUT_FILE} ]; then
    echo "✅ transformer_ops.so compiled successfully in ${CUDA_DIR}"
    ls -lh ${OUT_FILE}
else
    echo "❌ Compilation failed"
    exit 1
fi
