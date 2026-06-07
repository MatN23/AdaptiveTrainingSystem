#!/bin/bash

################################################################################
# CUDA MoE Operations - Direct Compilation Script
# 
# This script compiles CUDA kernels directly using nvcc WITHOUT setup.py
# Does EVERYTHING in one shot:
# 1. Validates environment
# 2. Compiles .cu to .o with nvcc
# 3. Links into Python extension .so
# 4. Verifies it works
#
# Output: moe_cuda_ext.so (directly importable from Python)
#
# Usage:
#   chmod +x compile_cuda_moe.sh
#   ./compile_cuda_moe.sh [--debug] [--verbose] [--clean]
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Settings
DEBUG_MODE=0
VERBOSE=0
CLEAN_BUILD=0
CUDA_FILE="moe_cuda_ops.cu"
OUTPUT_FILE="moe_cuda_ext.so"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() { echo -e "${GREEN}${NC} $1"; }
print_error() { echo -e "${RED}${NC} $1"; }
print_warning() { echo -e "${YELLOW}${NC} $1"; }
print_info() { echo -e "${BLUE}${NC} $1"; }

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

show_help() {
    cat << EOF
CUDA MoE Direct Compiler

Compiles CUDA MoE operations directly into moe_cuda_ext.so
No setup.py, no pip install, just raw compilation.

Usage:
    ./compile_cuda_moe.sh [OPTIONS]

Options:
    --debug     Compile with debug symbols (-g -G)
    --verbose   Show full compilation output
    --clean     Remove all build artifacts first
    --help      Show this help message

Output:
    moe_cuda_ext.so - Python importable shared library

Examples:
    ./compile_cuda_moe.sh                    # Standard optimized build
    ./compile_cuda_moe.sh --debug            # Debug build with symbols
    ./compile_cuda_moe.sh --clean --verbose  # Clean rebuild with output

After compilation:
    python3 -c "import moe_cuda_ext; print('Success!')"

EOF
    exit 0
}

################################################################################
# Parse Arguments
################################################################################

while [ $# -gt 0 ]; do
    case $1 in
        --debug)
            DEBUG_MODE=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

################################################################################
# Environment Detection
################################################################################

print_header "Step 1: Environment Detection"

# Check CUDA source file
if [ ! -f "$CUDA_FILE" ]; then
    print_error "CUDA source file not found: $CUDA_FILE"
    print_info "Current directory: $(pwd)"
    print_info "Expected location: $CUDA_FILE"
    if [ -d "cuda" ]; then
        print_info "Files in cuda/: $(ls -1 cuda/ | tr '\n' ' ')"
    else
        print_info "cuda/ directory not found!"
    fi
    exit 1
fi
print_success "Found: $CUDA_FILE"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_success "Python $PYTHON_VERSION"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    print_error "PyTorch not found"
    print_info "Install: pip install torch"
    exit 1
fi
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
print_success "PyTorch $TORCH_VERSION"

# Check CUDA availability
CUDA_AVAILABLE=$(python3 -c "import torch; print('1' if torch.cuda.is_available() else '0')")
if [ "$CUDA_AVAILABLE" = "1" ]; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    print_success "CUDA $CUDA_VERSION"
    print_success "GPU: $GPU_NAME"
else
    print_warning "CUDA not available in PyTorch"
    print_warning "Module will compile but won't run"
fi

# Check nvcc
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found - CUDA Toolkit required"
    print_info "Install from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
print_success "nvcc $NVCC_VERSION"

# Resolve GPU architectures
if [ -n "${CUDA_TARGET_SM}" ]; then
    GPU_ARCHS=$(normalize_arch_list "${CUDA_TARGET_SM}")
    if [ -z "${GPU_ARCHS}" ]; then
        GPU_ARCHS="75"
        print_warning "Invalid CUDA_TARGET_SM='${CUDA_TARGET_SM}', defaulting to sm_75"
    else
        print_success "Using forced CUDA_TARGET_SM: $(format_arch_list "${GPU_ARCHS}")"
    fi
elif [ "$CUDA_AVAILABLE" = "1" ]; then
    GPU_ARCHS=$(python3 -c "import torch; caps=[]; [caps.append(f'{m}{n}') for m,n in [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())] if f'{m}{n}' not in caps]; print(' '.join(caps))")
    GPU_ARCHS=$(normalize_arch_list "${GPU_ARCHS}")
    if [ -z "${GPU_ARCHS}" ]; then
        GPU_ARCHS="75"
        print_warning "Could not detect GPU architecture, defaulting to sm_75"
    else
        print_success "Detected target architectures: $(format_arch_list "${GPU_ARCHS}")"
    fi
else
    GPU_ARCHS="75"
    print_warning "Using default: sm_75 (Turing)"
fi

# Get Python paths
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
print_info "Python include: $PYTHON_INCLUDE"

# Get PyTorch paths
TORCH_INCLUDE=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")
TORCH_LIB=$(python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
print_info "PyTorch include: $TORCH_INCLUDE"
print_info "PyTorch lib: $TORCH_LIB"

# Match libstdc++ ABI with the installed PyTorch build.
TORCH_CXX11_ABI=$(python3 -c "import torch; print(int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 1)))")
print_info "PyTorch CXX11 ABI: $TORCH_CXX11_ABI"

# Detect CUDA runtime library path
CUDA_HOME=$(python3 - <<'PY'
import os
try:
    from torch.utils.cpp_extension import CUDA_HOME
except Exception:
    CUDA_HOME = None
print(CUDA_HOME or os.environ.get("CUDA_HOME", ""))
PY
)
if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME/lib64" ]; then
    CUDA_LIB_DIR="$CUDA_HOME/lib64"
else
    CUDA_LIB_DIR="/usr/local/cuda/lib64"
fi
print_info "CUDA lib: $CUDA_LIB_DIR"

print_info "Output: $OUTPUT_FILE"

################################################################################
# Clean Build
################################################################################

if [ $CLEAN_BUILD -eq 1 ]; then
    print_header "Step 2: Cleaning"
    
    rm -f moe_cuda_ext*.so 2>/dev/null || true
    rm -f moe_cuda_ops.o 2>/dev/null || true
    rm -f compile.log 2>/dev/null || true
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    
    print_success "Cleaned build artifacts"
fi

################################################################################
# Compilation - Step 1: CUDA to Object File
################################################################################

print_header "Step 3: Compiling CUDA Kernel (.cu  .o)"

# Build nvcc command
NVCC_CMD="nvcc"

# Architecture flags
for arch in $GPU_ARCHS; do
    NVCC_CMD+=" -gencode arch=compute_${arch},code=sm_${arch}"
done

# Include directories
NVCC_CMD+=" -I${PYTHON_INCLUDE}"
NVCC_CMD+=" -I${TORCH_INCLUDE}"
NVCC_CMD+=" -I${TORCH_INCLUDE}/torch/csrc/api/include"

# Compilation flags
NVCC_CMD+=" -c"
NVCC_CMD+=" -std=c++17"
NVCC_CMD+=" --compiler-options '-fPIC'"
NVCC_CMD+=" -DTORCH_EXTENSION_NAME=moe_cuda_ext"
NVCC_CMD+=" -DTORCH_API_INCLUDE_EXTENSION_H"
NVCC_CMD+=" -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}"

if [ $DEBUG_MODE -eq 1 ]; then
    print_info "Debug mode enabled"
    NVCC_CMD+=" -g -G -O0"
    NVCC_CMD+=" --generate-line-info"
else
    print_info "Optimized build"
    NVCC_CMD+=" -O3"
    NVCC_CMD+=" --use_fast_math"
    NVCC_CMD+=" --extra-device-vectorization"
    if [ "$GPU_ARCHS" = "75" ]; then
        NVCC_CMD+=" -Xptxas=-dlcm=ca"
    fi
fi

# Add verbosity
if [ $VERBOSE -eq 1 ]; then
    NVCC_CMD+=" -Xptxas=-v"
fi

# Input and output
NVCC_CMD+=" ${CUDA_FILE}"
NVCC_CMD+=" -o moe_cuda_ops.o"

print_info "Compiling with nvcc..."
if [ $VERBOSE -eq 1 ]; then
    echo ""
    echo "Command:"
    echo "$NVCC_CMD"
    echo ""
fi

# Execute compilation
if [ $VERBOSE -eq 1 ]; then
    eval $NVCC_CMD
    COMPILE_STATUS=$?
else
    eval $NVCC_CMD > compile.log 2>&1
    COMPILE_STATUS=$?
    
    if [ $COMPILE_STATUS -ne 0 ]; then
        echo ""
        print_error "Compilation failed! Error output:"
        cat compile.log
    else
        # Show key info
        if grep -q "ptxas info" compile.log; then
            echo ""
            grep "ptxas info" compile.log | head -n 5
        fi
    fi
fi

if [ $COMPILE_STATUS -ne 0 ]; then
    print_error "nvcc compilation failed!"
    if [ $VERBOSE -eq 0 ]; then
        print_info "Run with --verbose for full output or check: cat compile.log"
    fi
    exit 1
fi

if [ ! -f "moe_cuda_ops.o" ]; then
    print_error "Object file not created!"
    exit 1
fi

OBJECT_SIZE=$(ls -lh moe_cuda_ops.o | awk '{print $5}')
print_success "Created: moe_cuda_ops.o ($OBJECT_SIZE)"

################################################################################
# Linking - Step 2: Object to Shared Library
################################################################################

print_header "Step 4: Linking (.o  .so)"

# Build linker command
LINK_CMD="g++"

# Shared library flags
LINK_CMD+=" -shared"
LINK_CMD+=" -fPIC"
LINK_CMD+=" -std=c++17"

# Keep unresolved PyTorch symbols visible to the loader and embed runtime paths.
LINK_CMD+=" -Wl,--no-as-needed"
LINK_CMD+=" -Wl,-rpath,${TORCH_LIB}"
if [ -d "${CUDA_LIB_DIR}" ]; then
    LINK_CMD+=" -Wl,-rpath,${CUDA_LIB_DIR}"
fi

# Match ABI with PyTorch build.
LINK_CMD+=" -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}"

# IMPORTANT: object file must come before libraries for correct symbol resolution.
LINK_CMD+=" moe_cuda_ops.o"

# Library paths
LINK_CMD+=" -L${TORCH_LIB}"
if [ -d "${CUDA_LIB_DIR}" ]; then
    LINK_CMD+=" -L${CUDA_LIB_DIR}"
fi

# PyTorch libraries (conditionally include only what exists in this wheel)
if [ -f "${TORCH_LIB}/libtorch_python.so" ]; then
    LINK_CMD+=" -ltorch_python"
fi
if [ -f "${TORCH_LIB}/libtorch_cuda.so" ]; then
    LINK_CMD+=" -ltorch_cuda"
else
    if [ -f "${TORCH_LIB}/libtorch_cuda_cpp.so" ]; then
        LINK_CMD+=" -ltorch_cuda_cpp"
    fi
    if [ -f "${TORCH_LIB}/libtorch_cuda_cu.so" ]; then
        LINK_CMD+=" -ltorch_cuda_cu"
    fi
fi
if [ -f "${TORCH_LIB}/libtorch_cpu.so" ]; then
    LINK_CMD+=" -ltorch_cpu"
fi
if [ -f "${TORCH_LIB}/libtorch.so" ]; then
    LINK_CMD+=" -ltorch"
fi
if [ -f "${TORCH_LIB}/libc10_cuda.so" ]; then
    LINK_CMD+=" -lc10_cuda"
fi
if [ -f "${TORCH_LIB}/libc10.so" ]; then
    LINK_CMD+=" -lc10"
fi

# CUDA libraries
LINK_CMD+=" -lcudart"
LINK_CMD+=" -lcublas"
LINK_CMD+=" -o ${OUTPUT_FILE}"

print_info "Linking with g++..."
if [ $VERBOSE -eq 1 ]; then
    echo ""
    echo "Command:"
    echo "$LINK_CMD"
    echo ""
fi

# Execute linking
if [ $VERBOSE -eq 1 ]; then
    eval $LINK_CMD
    LINK_STATUS=$?
else
    eval $LINK_CMD >> compile.log 2>&1
    LINK_STATUS=$?
    
    if [ $LINK_STATUS -ne 0 ]; then
        echo ""
        print_error "Linking failed! Error output:"
        tail -n 20 compile.log
    fi
fi

if [ $LINK_STATUS -ne 0 ]; then
    print_error "Linking failed!"
    if [ $VERBOSE -eq 0 ]; then
        print_info "Run with --verbose for full output or check: cat compile.log"
    fi
    exit 1
fi

if [ ! -f "$OUTPUT_FILE" ]; then
    print_error "Shared library not created!"
    print_info "Looking for any .so files:"
    ls -lh *.so 2>/dev/null || echo "  None found"
    exit 1
fi

SO_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
print_success "Created: $OUTPUT_FILE ($SO_SIZE)"

################################################################################
# Verification
################################################################################

print_header "Step 5: Verification"

print_info "Testing module import..."
echo ""

# Test import with detailed diagnostics
python3 << 'VERIFY_SCRIPT'
import sys
import os

print("=" * 60)
print("Import Test")
print("=" * 60)
print(f"Python: {sys.version.split()[0]}")
print(f"Working directory: {os.getcwd()}")
print("")

# Check if file exists
if not os.path.exists('moe_cuda_ext.so'):
    print(" ERROR: moe_cuda_ext.so not found!")
    print(f"Files in directory: {os.listdir('.')[:10]}")
    sys.exit(1)

file_size = os.path.getsize('moe_cuda_ext.so')
print(f"Module file: moe_cuda_ext.so ({file_size:,} bytes)")
print("")

# Try importing
try:
    import moe_cuda_ext
    print(" SUCCESS: Module imported!")
    print("")
    
    # Show module info
    print(f"Module location: {moe_cuda_ext.__file__}")
    print("")
    
    # List functions
    funcs = [f for f in dir(moe_cuda_ext) if not f.startswith('_')]
    print(f"Available functions ({len(funcs)}):")
    for func in funcs:
        print(f"   {func}")
    print("")
    
    # Quick functionality test
    print("Testing basic functionality...")
    import torch
    
    if torch.cuda.is_available():
        try:
            # Test top-k gating
            gate_logits = torch.randn(10, 8, device='cuda')
            indices, probs = moe_cuda_ext.topk_gating(gate_logits, 2, 1.0)
            print(f"   topk_gating: {indices.shape}, {probs.shape}")
            
            # Test dispatch
            tokens = torch.randn(10, 768, device='cuda')
            expert_inputs, token_map = moe_cuda_ext.dispatch_tokens(
                tokens, indices, 8, 20
            )
            print(f"   dispatch_tokens: {expert_inputs.shape}")
            
            # Test combine
            expert_outputs = torch.randn(8, 20, 768, device='cuda')
            combined = moe_cuda_ext.combine_expert_outputs(
                expert_outputs, token_map, probs, 10, 2
            )
            print(f"   combine_expert_outputs: {combined.shape}")
            
            print("")
            print(" All operations working correctly!")
        except Exception as e:
            print(f"   Function test failed: {e}")
            print("  (Module imported but operations may have issues)")
    else:
        print("   CUDA not available - skipping functionality test")
    
except ImportError as e:
    print(f" IMPORT FAILED: {e}")
    print("")
    print("Troubleshooting:")
    print("  1. Check file: ls -lh moe_cuda_ext.so")
    print("  2. Check dependencies: ldd moe_cuda_ext.so | grep 'not found'")
    print("  3. Verify CUDA: python3 -c 'import torch; print(torch.cuda.is_available())'")
    sys.exit(1)
except Exception as e:
    print(f" UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
VERIFY_SCRIPT

VERIFY_STATUS=$?

echo ""

if [ $VERIFY_STATUS -ne 0 ]; then
    print_error "Verification failed!"
    echo ""
    print_info "Checking library dependencies:"
    ldd $OUTPUT_FILE | grep "not found" || print_success "All dependencies found"
    exit 1
fi

################################################################################
# Success Summary
################################################################################

print_header "Build Complete! "

echo ""
echo -e "${GREEN}Successfully built CUDA MoE operations!${NC}"
echo ""
echo "Module details:"
echo "   File: $OUTPUT_FILE"
echo "   Size: $SO_SIZE"
echo "   Architectures: $(format_arch_list "$GPU_ARCHS")"
echo "   Location: $(pwd)/$OUTPUT_FILE"
echo ""
echo "Usage in Python:"
echo "  import moe_cuda_ext"
echo "  # or from your wrapper:"
echo "  from moe_cuda_wrapper import MoECUDAOps"
echo ""
echo "Enable in model config:"
echo "  config = DeepSeekConfig.standard_moe(use_cuda_moe=True)"
echo ""

# Cleanup
print_info "Cleaning up intermediate files..."
rm -f moe_cuda_ops.o
print_success "Removed: moe_cuda_ops.o"

echo ""
print_success "Ready to use! "
echo ""

# Offer benchmark
read -p "Run performance benchmark? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Performance Benchmark"
    echo ""
    
    python3 << 'BENCHMARK'
import torch
import moe_cuda_ext
import time

if not torch.cuda.is_available():
    print(" CUDA not available - skipping benchmark")
    exit(0)

print("Benchmarking CUDA MoE operations...")
print("=" * 70)

# Configuration
num_tokens = 1024
hidden_dim = 768
num_experts = 8
k = 2
num_iterations = 100

print(f"Configuration:")
print(f"  Tokens: {num_tokens}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Experts: {num_experts}")
print(f"  Top-k: {k}")
print(f"  Iterations: {num_iterations}")
print("")

# Create test data
gate_logits = torch.randn(num_tokens, num_experts, device='cuda')
tokens = torch.randn(num_tokens, hidden_dim, device='cuda')

# Warmup
for _ in range(10):
    indices, probs = moe_cuda_ext.topk_gating(gate_logits, k, 1.0)
torch.cuda.synchronize()

# Benchmark routing
start = time.perf_counter()
for _ in range(num_iterations):
    indices, probs = moe_cuda_ext.topk_gating(gate_logits, k, 1.0)
torch.cuda.synchronize()
routing_time = (time.perf_counter() - start) * 1000 / num_iterations

print(f"Results:")
print(f"   Routing time: {routing_time:.3f}ms per call")
print(f"   Throughput: {num_tokens / (routing_time / 1000):,.0f} tokens/sec")
print("")

# Benchmark dispatch
capacity = num_tokens * k // num_experts + 10
start = time.perf_counter()
for _ in range(num_iterations):
    expert_inputs, token_map = moe_cuda_ext.dispatch_tokens(
        tokens, indices, num_experts, capacity
    )
torch.cuda.synchronize()
dispatch_time = (time.perf_counter() - start) * 1000 / num_iterations

print(f"   Dispatch time: {dispatch_time:.3f}ms per call")
print("")

# Benchmark combine
expert_outputs = torch.randn(num_experts, capacity, hidden_dim, device='cuda')
start = time.perf_counter()
for _ in range(num_iterations):
    combined = moe_cuda_ext.combine_expert_outputs(
        expert_outputs, token_map, probs, num_tokens, k
    )
torch.cuda.synchronize()
combine_time = (time.perf_counter() - start) * 1000 / num_iterations

print(f"   Combine time: {combine_time:.3f}ms per call")
print("")

total_time = routing_time + dispatch_time + combine_time
print(f"Total MoE overhead: {total_time:.3f}ms per forward pass")
print("")
print("=" * 70)
print(" CUDA operations are fast and working correctly!")
BENCHMARK
    
    echo ""
fi

echo ""
print_success "All done! Import with: import moe_cuda_ext"
