#!/bin/bash
# compile.sh - Build MoE inference runtime
# Usage: ./compile.sh [--cpu|--mps|--cuda|--auto]

set -e

# Auto-detect best backend if not specified
if [[ -z "$1" || "$1" == "--auto" || "$1" == "auto" ]]; then
    # Check for CUDA first
    if command -v nvcc &> /dev/null; then
        TARGET="cuda"
        echo "🔍 Auto-detected: NVIDIA CUDA"
    # Check for macOS (MPS)
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        TARGET="mps"
        echo "🔍 Auto-detected: Apple Metal (MPS)"
    # Fallback to CPU
    else
        TARGET="cpu"
        echo "🔍 Auto-detected: CPU (no GPU acceleration found)"
    fi
else
    TARGET="$1"
fi

echo "🔨 Building MoE Inference Runtime for: $TARGET"

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

CXX=${CXX:-g++}
CXXFLAGS="-std=c++17 -O3 -march=native -pthread -Icore -Ibackends"
LDFLAGS=""

case $TARGET in
    --cpu|cpu)
        echo "✓ Target: CPU (SIMD + threading)"
        SOURCES="core/runtime.cpp backends/cpu_kernels.cpp"
        OUTPUT="moe_inference_cpu"
        
        # Add SIMD flags
        if [[ "$ARCH" == "arm64" || "$ARCH" == "aarch64" ]]; then
            CXXFLAGS="$CXXFLAGS -DUSE_NEON"
        elif [[ "$ARCH" == "x86_64" ]]; then
            CXXFLAGS="$CXXFLAGS -mavx2 -mfma -DUSE_AVX2"
        fi
        ;;
        
    --mps|mps)
        echo "✓ Target: Apple Metal (MPS)"
        if [[ "$OS" != "Darwin" ]]; then
            echo "❌ MPS only available on macOS"
            exit 1
        fi
        
        CXX="clang++"
        SOURCES="core/runtime.cpp backends/mps_kernels.mm"
        OUTPUT="moe_inference_mps"
        CXXFLAGS="$CXXFLAGS -DUSE_MPS"
        LDFLAGS="-framework Metal -framework MetalPerformanceShaders -framework Foundation"
        ;;
        
    --cuda|cuda)
        echo "✓ Target: NVIDIA CUDA"
        if ! command -v nvcc &> /dev/null; then
            echo "❌ nvcc not found - falling back to CPU"
            TARGET="cpu"
            exec $0 --cpu
        fi
        
        # Detect CUDA paths
        CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}
        if [[ ! -d "$CUDA_PATH" ]]; then
            echo "❌ CUDA not found at $CUDA_PATH"
            echo "   Set CUDA_PATH environment variable"
            exit 1
        fi
        
        echo "   CUDA Path: $CUDA_PATH"
        
        # CUDA compilation needs special handling
        echo "Compiling CUDA kernels..."
        nvcc -c backends/cuda_kernels.cu -o cuda_kernels.o \
            -O3 -DUSE_CUDA --compiler-options "-fPIC" \
            -I$CUDA_PATH/include -Icore -Ibackends
        
        echo "Compiling runtime..."
        g++ -c core/runtime.cpp -o runtime.o \
            -std=c++17 -O3 -DUSE_CUDA \
            -I$CUDA_PATH/include -Icore -Ibackends
        
        echo "Linking..."
        g++ runtime.o cuda_kernels.o -o moe_inference_cuda \
            -lcudart -lcublas \
            -L$CUDA_PATH/lib64 -Wl,-rpath,$CUDA_PATH/lib64
        
        rm -f runtime.o cuda_kernels.o
        OUTPUT="moe_inference_cuda"
        
        echo "✅ Build complete: $OUTPUT"
        echo "   Run with: ./run_inference.sh $OUTPUT model.bin \"prompt\""
        exit 0
        ;;
        
    *)
        echo "❌ Unknown target: $TARGET"
        echo "Usage: ./compile.sh [--cpu|--mps|--cuda]"
        exit 1
        ;;
esac

# Build
echo "Compiling..."
$CXX $CXXFLAGS $SOURCES -o $OUTPUT $LDFLAGS

echo "✅ Build complete: $OUTPUT"
echo ""
echo "Run with: ./run_inference.sh $OUTPUT model.bin input.txt"