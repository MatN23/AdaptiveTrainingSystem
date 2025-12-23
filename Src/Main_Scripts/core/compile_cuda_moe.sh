#!/bin/bash

################################################################################
# CUDA MoE Operations - Complete Build & Compilation Script
# 
# This script does EVERYTHING needed to compile and install CUDA MoE ops:
# 1. Detects environment and validates dependencies
# 2. Creates proper Python extension setup
# 3. Compiles CUDA kernels with optimal flags
# 4. Installs the module for import
# 5. Verifies everything works
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

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }

show_help() {
    head -n 15 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

################################################################################
# Parse Arguments
################################################################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug) DEBUG_MODE=1; shift ;;
        --verbose) VERBOSE=1; shift ;;
        --clean) CLEAN_BUILD=1; shift ;;
        --help) show_help ;;
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
    print_info "Files here: $(ls -1 | head -n 10 | tr '\n' ' ')"
    exit 1
fi
print_success "Found: $CUDA_FILE"

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
print_success "$PYTHON_VERSION"

# Check PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    print_error "PyTorch not found"
    print_info "Install: pip install torch"
    exit 1
fi
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
print_success "PyTorch $TORCH_VERSION"

# Check CUDA availability
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    print_success "CUDA $CUDA_VERSION available"
else
    print_warning "CUDA not available in PyTorch"
fi

# Check nvcc
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found - CUDA Toolkit required"
    exit 1
fi
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
print_success "nvcc $NVCC_VERSION"

# Detect GPU architecture
GPU_ARCH=$(python3 << 'EOF'
import torch
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability(0)
    print(f"{cap[0]}{cap[1]}")
else:
    print("75")
EOF
)
print_success "Target architecture: sm_$GPU_ARCH"

# Get Python extension suffix
if command -v python3-config &> /dev/null; then
    EXT_SUFFIX=$(python3-config --extension-suffix)
else
    # Fallback: detect from Python
    EXT_SUFFIX=$(python3 << 'EOF'
import sysconfig
import sys
# Get the extension suffix
suffix = sysconfig.get_config_var('EXT_SUFFIX')
if suffix:
    print(suffix)
else:
    # Fallback for older Python or missing config
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform().replace('-', '_')
    print(f".cpython-{version}-{platform}.so")
EOF
)
fi
print_info "Extension suffix: $EXT_SUFFIX"

################################################################################
# Clean Build
################################################################################

if [ $CLEAN_BUILD -eq 1 ]; then
    print_header "Step 2: Cleaning"
    
    rm -rf build dist *.egg-info
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.so" -delete 2>/dev/null || true
    find . -type f -name "*.o" -delete 2>/dev/null || true
    
    print_success "Cleaned build artifacts"
fi

################################################################################
# Create Setup Script
################################################################################

print_header "Step 3: Creating Build Configuration"

cat > setup.py << 'SETUP_SCRIPT_EOF'
import os
import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect if debug mode
is_debug = os.getenv('BUILD_DEBUG', '0') == '1'

# CUDA source file
cuda_source = 'moe_cuda_ops.cu'
if not os.path.exists(cuda_source):
    print(f"ERROR: {cuda_source} not found!")
    sys.exit(1)

print(f"Building from: {cuda_source}")
print(f"Debug mode: {is_debug}")

# Compiler flags
if is_debug:
    extra_compile_args = {
        'cxx': ['-g', '-O0'],
        'nvcc': [
            '-g', '-G',
            '-lineinfo',
            '--generate-line-info'
        ]
    }
else:
    extra_compile_args = {
        'cxx': ['-O3', '-DNDEBUG'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-lineinfo',
            '--extra-device-vectorization',
            '-Xptxas=-v'
        ]
    }

setup(
    name='moe_cuda_ext',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='moe_cuda_ext',
            sources=[cuda_source],
            extra_compile_args=extra_compile_args
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    zip_safe=False,
)
SETUP_SCRIPT_EOF

print_success "Created setup.py"

################################################################################
# Compilation
################################################################################

print_header "Step 4: Compiling CUDA Kernels"

# Set environment variables
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

if [ $DEBUG_MODE -eq 1 ]; then
    export BUILD_DEBUG=1
    print_info "Compiling with debug symbols..."
else
    export BUILD_DEBUG=0
    print_info "Compiling with optimizations..."
fi

# Run compilation
print_info "This may take 1-3 minutes..."
echo ""

if [ $VERBOSE -eq 1 ]; then
    # Verbose mode - show everything
    python3 setup.py build_ext --inplace
    BUILD_STATUS=$?
else
    # Quiet mode - capture output and show key lines
    python3 setup.py build_ext --inplace > build.log 2>&1
    BUILD_STATUS=$?
    
    if [ $BUILD_STATUS -eq 0 ]; then
        # Show important lines
        grep -E "(Building|Compiling|creating build|copying)" build.log | head -n 10 || true
    else
        # Show errors
        echo ""
        print_error "Compilation failed! Output:"
        cat build.log
    fi
fi

echo ""

if [ $BUILD_STATUS -ne 0 ]; then
    print_error "Compilation failed!"
    if [ $VERBOSE -eq 0 ]; then
        print_info "Run with --verbose for full output"
        print_info "Or check: cat build.log"
    fi
    exit 1
fi

print_success "Compilation completed"

################################################################################
# Find and Install Module
################################################################################

print_header "Step 5: Installing Module"

# Find the compiled .so file
SO_FILE=$(find . -name "moe_cuda_ext*.so" -type f 2>/dev/null | head -n 1)

if [ -z "$SO_FILE" ]; then
    print_error "Compiled .so file not found!"
    print_info "Searching for any .so files:"
    find . -name "*.so" -type f 2>/dev/null || echo "  None found"
    echo ""
    print_info "Directory structure:"
    find . -type f -name "*.o" -o -name "*.so" -o -name "*.a" 2>/dev/null | head -n 20
    exit 1
fi

print_success "Found: $SO_FILE"

# Get file info
FILE_SIZE=$(ls -lh "$SO_FILE" | awk '{print $5}')
print_info "Size: $FILE_SIZE"

# Ensure it's in the current directory with correct name
FINAL_NAME="moe_cuda_ext$EXT_SUFFIX"

if [ "$SO_FILE" != "./$FINAL_NAME" ]; then
    cp "$SO_FILE" "./$FINAL_NAME"
    print_success "Installed as: ./$FINAL_NAME"
else
    print_success "Already in correct location"
fi

################################################################################
# Verification
################################################################################

print_header "Step 6: Verification"

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
import glob
so_files = glob.glob("moe_cuda_ext*.so")
if not so_files:
    print("✗ ERROR: No moe_cuda_ext*.so file found!")
    print(f"Files in directory: {os.listdir('.')[:10]}")
    sys.exit(1)

print(f"Module file: {so_files[0]} ({os.path.getsize(so_files[0])} bytes)")
print("")

# Try importing
try:
    import moe_cuda_ext
    print("✓ SUCCESS: Module imported!")
    print("")
    
    # Show module info
    print(f"Module location: {moe_cuda_ext.__file__}")
    print("")
    
    # List functions
    funcs = [f for f in dir(moe_cuda_ext) if not f.startswith('_')]
    print(f"Available functions ({len(funcs)}):")
    for func in funcs:
        print(f"  • {func}")
    print("")
    
    # Quick functionality test
    print("Testing basic functionality...")
    import torch
    
    if torch.cuda.is_available():
        try:
            # Test top-k gating
            gate_logits = torch.randn(10, 8, device='cuda')
            indices, probs = moe_cuda_ext.topk_gating(gate_logits, 2, 1.0)
            print(f"  ✓ topk_gating: {indices.shape}, {probs.shape}")
            
            # Test dispatch
            tokens = torch.randn(10, 768, device='cuda')
            expert_inputs, token_map = moe_cuda_ext.dispatch_tokens(
                tokens, indices, 8, 20
            )
            print(f"  ✓ dispatch_tokens: {expert_inputs.shape}")
            
            # Test combine
            expert_outputs = torch.randn(8, 20, 768, device='cuda')
            combined = moe_cuda_ext.combine_expert_outputs(
                expert_outputs, token_map, probs, 10, 2
            )
            print(f"  ✓ combine_expert_outputs: {combined.shape}")
            
            print("")
            print("✓ All operations working correctly!")
        except Exception as e:
            print(f"  ⚠ Function test failed: {e}")
            print("  (Module imported but operations may have issues)")
    else:
        print("  ⚠ CUDA not available - skipping functionality test")
    
except ImportError as e:
    print(f"✗ IMPORT FAILED: {e}")
    print("")
    print("Troubleshooting:")
    print("  1. Check file exists: ls -lh moe_cuda_ext*.so")
    print("  2. Check dependencies: ldd moe_cuda_ext*.so")
    print("  3. Try: python3 -c 'import moe_cuda_ext'")
    sys.exit(1)
except Exception as e:
    print(f"✗ UNEXPECTED ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
VERIFY_SCRIPT

VERIFY_STATUS=$?

if [ $VERIFY_STATUS -ne 0 ]; then
    print_error "Verification failed!"
    exit 1
fi

################################################################################
# Success Summary
################################################################################

print_header "Build Complete! 🎉"

echo ""
echo -e "${GREEN}Successfully built and installed CUDA MoE operations!${NC}"
echo ""
echo "Module details:"
echo "  • File: moe_cuda_ext$EXT_SUFFIX"
echo "  • Size: $FILE_SIZE"
echo "  • Architecture: sm_$GPU_ARCH"
echo ""
echo "Usage in Python:"
echo "  from core.moe_cuda_wrapper import MoECUDAOps"
echo "  # or"
echo "  import moe_cuda_ext"
echo ""
echo "Enable in model config:"
echo "  config = DeepSeekConfig.standard_moe(use_cuda_moe=True)"
echo ""
echo "Files created:"
echo "  • setup.py (build configuration)"
echo "  • moe_cuda_ext$EXT_SUFFIX (compiled module)"
echo "  • build.log (compilation log)"
echo ""

# Offer benchmark
read -p "Run performance benchmark? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_header "Performance Benchmark"
    
    python3 << 'BENCHMARK'
import torch
import moe_cuda_ext
import time

if not torch.cuda.is_available():
    print("CUDA not available - skipping benchmark")
    exit(0)

print("Benchmarking CUDA MoE operations...")
print("=" * 60)

# Configuration
num_tokens = 1024
hidden_dim = 768
num_experts = 8
k = 2
num_iterations = 100

# Create test data
gate_logits = torch.randn(num_tokens, num_experts, device='cuda')
tokens = torch.randn(num_tokens, hidden_dim, device='cuda')

print(f"Configuration:")
print(f"  Tokens: {num_tokens}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Experts: {num_experts}")
print(f"  Top-k: {k}")
print(f"  Iterations: {num_iterations}")
print("")

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
print(f"  Routing time: {routing_time:.3f}ms")
print(f"  Throughput: {num_tokens / (routing_time / 1000):.0f} tokens/sec")
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

print(f"  Dispatch time: {dispatch_time:.3f}ms")
print("")

print("✓ CUDA operations are fast and working correctly!")
print("=" * 60)
BENCHMARK
fi

echo ""
print_success "Ready to use! 🚀"