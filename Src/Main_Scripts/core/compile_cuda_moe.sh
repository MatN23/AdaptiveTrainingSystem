#!/bin/bash

################################################################################
# CUDA MoE Operations Compilation Script
# 
# This script compiles the optimized CUDA kernels for Mixture-of-Experts
# operations with automatic environment detection and error handling.
#
# Usage:
#   chmod +x compile_cuda_moe.sh
#   ./compile_cuda_moe.sh [--debug] [--verbose] [--clean]
#
# Options:
#   --debug     Compile with debug symbols and no optimization
#   --verbose   Show detailed compilation output
#   --clean     Clean build artifacts before compiling
#   --help      Show this help message
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
DEBUG_MODE=0
VERBOSE=0
CLEAN_BUILD=0
BUILD_DIR="build"
CUDA_FILE="moe_cuda_ops.cu"
MODULE_NAME="moe_cuda_ext"

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

show_help() {
    head -n 20 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

################################################################################
# Parse Command Line Arguments
################################################################################

while [[ $# -gt 0 ]]; do
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

print_header "Environment Detection"

# Check for required files
if [ ! -f "$CUDA_FILE" ]; then
    print_error "CUDA source file not found: $CUDA_FILE"
    print_info "Make sure you're in the correct directory"
    exit 1
fi
print_success "Found CUDA source file: $CUDA_FILE"

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
print_success "Python 3: $(python3 --version)"

# Check for PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    print_error "PyTorch not found"
    print_info "Install with: pip install torch"
    exit 1
fi
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
print_success "PyTorch: $TORCH_VERSION"

# Check for CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    print_warning "CUDA not available in PyTorch"
    print_info "The module will compile but may not run on GPU"
else
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    print_success "CUDA: $CUDA_VERSION"
fi

# Check for nvcc
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found"
    print_info "Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
print_success "NVCC: $NVCC_VERSION"

# Detect GPU compute capability
print_info "Detecting GPU compute capability..."
GPU_ARCH=$(python3 << 'EOF'
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"{cap[0]}{cap[1]}")
else:
    print("75")  # Default to Turing (RTX 20xx, T4, etc.)
EOF
)
print_success "Target GPU architecture: sm_$GPU_ARCH"

################################################################################
# Clean Build (if requested)
################################################################################

if [ $CLEAN_BUILD -eq 1 ]; then
    print_header "Cleaning Build Artifacts"
    
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_success "Removed build directory"
    fi
    
    # Clean Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.so" -delete 2>/dev/null || true
    print_success "Removed Python cache files"
fi

################################################################################
# Setup Build Directory
################################################################################

print_header "Setting Up Build Environment"

mkdir -p "$BUILD_DIR"
print_success "Created build directory: $BUILD_DIR"

################################################################################
# Compilation
################################################################################

print_header "Compiling CUDA MoE Operations"

# Build compilation command
COMPILE_CMD="python3 -m torch.utils.cpp_extension.build_extension"

# Set compiler flags based on mode
if [ $DEBUG_MODE -eq 1 ]; then
    print_info "Debug mode enabled - compiling with debug symbols"
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    export CFLAGS="-g -O0"
    export CXXFLAGS="-g -O0"
else
    print_info "Release mode - compiling with optimizations"
    export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
    export CFLAGS="-O3 -DNDEBUG"
    export CXXFLAGS="-O3 -DNDEBUG"
fi

# Create setup script
cat > "$BUILD_DIR/setup.py" << 'SETUP_EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get source file path
cuda_file = os.path.join(os.path.dirname(__file__), '..', 'moe_cuda_ops.cu')

# Compilation flags
extra_compile_args = {
    'cxx': ['-O3'] if os.getenv('CFLAGS', '').find('-g') == -1 else ['-g', '-O0'],
    'nvcc': [
        '-O3',
        '--use_fast_math',
        '-lineinfo',
        '--extra-device-vectorization',
        '-Xptxas=-v',  # Verbose register usage
    ]
}

# Add debug flags if needed
if os.getenv('CFLAGS', '').find('-g') != -1:
    extra_compile_args['nvcc'] = ['-g', '-G', '-lineinfo']

setup(
    name='moe_cuda_ext',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda_ext',
            sources=[cuda_file],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
SETUP_EOF

print_success "Created setup.py"

# Run compilation
print_info "Starting compilation (this may take 1-3 minutes)..."
echo ""

if [ $VERBOSE -eq 1 ]; then
    cd "$BUILD_DIR" && python3 setup.py build_ext --inplace
else
    cd "$BUILD_DIR" && python3 setup.py build_ext --inplace 2>&1 | \
        grep -E "(error|warning|Building|Compiling|Linking)" || true
fi

BUILD_STATUS=$?
cd ..

if [ $BUILD_STATUS -eq 0 ]; then
    print_success "Compilation successful!"
    echo ""
    
    # Find the compiled .so file
    SO_FILE=$(find "$BUILD_DIR" -name "*.so" | head -n 1)
    
    if [ -n "$SO_FILE" ]; then
        # Copy to project root for easy import
        cp "$SO_FILE" "./moe_cuda_ext$(python3-config --extension-suffix)"
        print_success "Module installed: ./moe_cuda_ext$(python3-config --extension-suffix)"
        
        # Show file info
        FILE_SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
        print_info "Module size: $FILE_SIZE"
    fi
else
    print_error "Compilation failed!"
    print_info "Try running with --verbose to see detailed errors"
    exit 1
fi

################################################################################
# Verification
################################################################################

print_header "Verification"

# Test import
print_info "Testing module import..."
if python3 -c "import moe_cuda_ext; print('Functions:', dir(moe_cuda_ext))" 2>/dev/null; then
    print_success "Module imports successfully!"
    echo ""
    
    # Show available functions
    print_info "Available functions:"
    python3 << 'EOF'
import moe_cuda_ext
funcs = [f for f in dir(moe_cuda_ext) if not f.startswith('_')]
for func in funcs:
    print(f"  • {func}")
EOF
else
    print_error "Module import failed"
    print_info "Check error messages above"
    exit 1
fi

################################################################################
# Summary
################################################################################

print_header "Build Summary"

echo -e "${GREEN}Compilation completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Import the module: from core.moe_cuda_wrapper import MoECUDAOps"
echo "  2. Enable in config: config.use_cuda_moe = True"
echo "  3. Run training/inference"
echo ""
echo "Troubleshooting:"
echo "  • If import fails, check: echo \$LD_LIBRARY_PATH"
echo "  • For debug mode: ./compile_cuda_moe.sh --debug"
echo "  • For verbose output: ./compile_cuda_moe.sh --verbose"
echo ""

# Optional: Run quick benchmark
read -p "Run quick benchmark? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Running benchmark..."
    python3 << 'EOF'
try:
    import torch
    import moe_cuda_ext
    
    print("\n🚀 Quick Benchmark")
    print("=" * 50)
    
    # Test routing
    gate_logits = torch.randn(1024, 8, device='cuda')
    
    import time
    start = time.perf_counter()
    for _ in range(100):
        indices, probs = moe_cuda_ext.topk_gating(gate_logits, 2, 1.0)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000 / 100
    
    print(f"Routing (1024 tokens, 8 experts, k=2): {elapsed:.3f}ms")
    print(f"Throughput: {1024 / (elapsed / 1000):.0f} tokens/sec")
    print("\n✓ CUDA operations working correctly!")
    
except Exception as e:
    print(f"Benchmark failed: {e}")
EOF
fi

print_success "All done! 🎉"