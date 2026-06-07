#!/bin/bash
# get_native_properties.sh - System Properties Detection Script
# Detects hardware capabilities for optimal LuminaAI training configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print header
echo -e "${CYAN}${NC}"
echo -e "${CYAN}           LuminaAI Native System Properties Detection                 ${NC}"
echo -e "${CYAN}${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${MAGENTA} $1 ${NC}"
}

# Function to print key-value pairs
print_kv() {
    printf "  ${GREEN}%-30s${NC}: %s\n" "$1" "$2"
}

# Function to print warnings
print_warn() {
    echo -e "  ${YELLOW} WARNING${NC}: $1"
}

# Function to print errors
print_error() {
    echo -e "  ${RED} ERROR${NC}: $1"
}

# Function to print success
print_success() {
    echo -e "  ${GREEN}${NC} $1"
}

# ============================================================================
# OPERATING SYSTEM DETECTION
# ============================================================================
print_section "Operating System"

OS_TYPE=$(uname -s)
OS_ARCH=$(uname -m)
OS_VERSION=$(uname -r)

print_kv "OS Type" "$OS_TYPE"
print_kv "Architecture" "$OS_ARCH"
print_kv "Kernel Version" "$OS_VERSION"

# Detect specific OS
if [[ "$OS_TYPE" == "Linux" ]]; then
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        print_kv "Distribution" "$NAME $VERSION"
    fi
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    MACOS_VERSION=$(sw_vers -productVersion)
    print_kv "macOS Version" "$MACOS_VERSION"
fi

# ============================================================================
# CPU INFORMATION
# ============================================================================
print_section "CPU Information"

if [[ "$OS_TYPE" == "Linux" ]]; then
    CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
    CPU_CORES=$(nproc)
    CPU_THREADS=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    CPU_SOCKETS=$(lscpu | grep "Socket(s):" | awk '{print $2}')
    CPU_MHZ=$(lscpu | grep "CPU MHz" | awk '{print $3}')
    
    print_kv "Model" "$CPU_MODEL"
    print_kv "Physical Cores" "$CPU_CORES"
    print_kv "Threads" "$CPU_THREADS"
    print_kv "Sockets" "$CPU_SOCKETS"
    print_kv "Current Frequency" "${CPU_MHZ} MHz"
    
    # CPU flags
    if grep -q "avx2" /proc/cpuinfo; then
        print_success "AVX2 supported"
    fi
    if grep -q "avx512" /proc/cpuinfo; then
        print_success "AVX512 supported"
    fi
    
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    CPU_MODEL=$(sysctl -n machdep.cpu.brand_string)
    CPU_CORES=$(sysctl -n hw.physicalcpu)
    CPU_THREADS=$(sysctl -n hw.logicalcpu)
    
    print_kv "Model" "$CPU_MODEL"
    print_kv "Physical Cores" "$CPU_CORES"
    print_kv "Threads" "$CPU_THREADS"
    
    # Check for Apple Silicon
    if [[ "$OS_ARCH" == "arm64" ]]; then
        print_success "Apple Silicon detected (M1/M2/M3/M4)"
        print_kv "MPS Support" "Available (Metal Performance Shaders)"
    fi
fi

# ============================================================================
# MEMORY INFORMATION
# ============================================================================
print_section "Memory Information"

if [[ "$OS_TYPE" == "Linux" ]]; then
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAILABLE_MEM=$(free -h | awk '/^Mem:/ {print $7}')
    USED_MEM=$(free -h | awk '/^Mem:/ {print $3}')
    
    print_kv "Total Memory" "$TOTAL_MEM"
    print_kv "Available Memory" "$AVAILABLE_MEM"
    print_kv "Used Memory" "$USED_MEM"
    
    # Check if enough memory for training
    TOTAL_MEM_GB=$(free -g | awk '/^Mem:/ {print $2}')
    if [ "$TOTAL_MEM_GB" -lt 8 ]; then
        print_warn "Less than 8GB RAM detected. Consider upgrading for better performance."
    elif [ "$TOTAL_MEM_GB" -lt 16 ]; then
        print_warn "Less than 16GB RAM. Recommended: 16GB+ for medium models."
    else
        print_success "Sufficient RAM for training (${TOTAL_MEM_GB}GB)"
    fi
    
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024 " GB"}')
    
    print_kv "Total Memory" "$TOTAL_MEM"
    
    if [[ "$OS_ARCH" == "arm64" ]]; then
        print_kv "Memory Type" "Unified Memory (shared with GPU)"
    fi
fi

# ============================================================================
# GPU DETECTION (NVIDIA)
# ============================================================================
print_section "GPU Information (NVIDIA)"

if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    
    print_success "NVIDIA GPU detected"
    print_kv "GPU Count" "$GPU_COUNT"
    
    for i in $(seq 0 $((GPU_COUNT-1))); do
        echo -e "\n  ${BLUE}GPU $i:${NC}"
        
        GPU_NAME=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader)
        GPU_MEM=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader)
        GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
        GPU_CUDA=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1)
        GPU_COMPUTE=$(nvidia-smi --id=$i --query-gpu=compute_cap --format=csv,noheader)
        GPU_POWER=$(nvidia-smi --id=$i --query-gpu=power.limit --format=csv,noheader)
        
        print_kv "Model" "$GPU_NAME"
        print_kv "Memory" "$GPU_MEM"
        print_kv "Compute Capability" "$GPU_COMPUTE"
        print_kv "Power Limit" "$GPU_POWER"
        
        # Check compute capability for precision support
        COMPUTE_MAJOR=$(echo $GPU_COMPUTE | cut -d. -f1)
        if [ "$COMPUTE_MAJOR" -ge 8 ]; then
            print_success "Ampere or newer - BF16 & TF32 supported"
            print_kv "Recommended Precision" "mixed_bf16 or tf32"
        elif [ "$COMPUTE_MAJOR" -ge 7 ]; then
            print_success "Volta/Turing - FP16 supported"
            print_kv "Recommended Precision" "mixed_fp16"
        else
            print_warn "Older architecture - FP32 only recommended"
            print_kv "Recommended Precision" "fp32"
        fi
    done
    
    print_kv "Driver Version" "$GPU_DRIVER"
    print_kv "CUDA Version" "$GPU_CUDA"
    
    # DeepSpeed compatibility
    print_success "DeepSpeed supported"
    print_success "Flash Attention supported"
    
else
    print_error "NVIDIA GPU not detected or nvidia-smi not available"
    echo -e "  Install CUDA toolkit and drivers from: https://developer.nvidia.com/cuda-downloads"
fi

# ============================================================================
# APPLE SILICON (MPS) DETECTION
# ============================================================================
if [[ "$OS_TYPE" == "Darwin" ]] && [[ "$OS_ARCH" == "arm64" ]]; then
    print_section "Apple Silicon (MPS) Capabilities"
    
    # Detect chip type
    CHIP_TYPE=$(sysctl -n machdep.cpu.brand_string)
    print_kv "Chip" "$CHIP_TYPE"
    
    # GPU cores (approximate)
    if [[ "$CHIP_TYPE" == *"M1"* ]]; then
        print_kv "GPU Cores" "7-8 (M1)"
        print_kv "Memory Bandwidth" "~68 GB/s"
    elif [[ "$CHIP_TYPE" == *"M2"* ]]; then
        print_kv "GPU Cores" "8-10 (M2)"
        print_kv "Memory Bandwidth" "~100 GB/s"
    elif [[ "$CHIP_TYPE" == *"M3"* ]]; then
        print_kv "GPU Cores" "10-16 (M3)"
        print_kv "Memory Bandwidth" "~100-150 GB/s"
    fi
    
    print_success "MPS (Metal Performance Shaders) available"
    print_kv "Recommended Precision" "fp16"
    
    # MPS limitations
    echo -e "\n  ${YELLOW}MPS Limitations:${NC}"
    echo "     DeepSpeed: Not supported"
    echo "     Flash Attention: Not supported (CUDA only)"
    echo "     BF16: Limited support (use FP16 instead)"
    echo "     Model Compilation: Can be unstable"
    
    # MPS recommendations
    echo -e "\n  ${GREEN}MPS Recommendations:${NC}"
    echo "     Start with small batch sizes (2-4)"
    echo "     Use FP16 precision"
    echo "     Disable DeepSpeed and Flash Attention"
    echo "     Monitor unified memory usage"
fi

# ============================================================================
# PYTHON ENVIRONMENT
# ============================================================================
print_section "Python Environment"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_kv "Python Version" "$PYTHON_VERSION"
    
    # Check Python version
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        print_success "Python version is compatible (3.8+)"
    else
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
    fi
else
    print_error "Python 3 not found"
fi

# ============================================================================
# PYTORCH DETECTION
# ============================================================================
print_section "PyTorch Installation"

if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    print_kv "PyTorch Version" "$TORCH_VERSION"
    
    # CUDA availability
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" == "True" ]; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
        print_success "CUDA available in PyTorch"
        print_kv "PyTorch CUDA Version" "$CUDA_VERSION"
    else
        print_warn "CUDA not available in PyTorch"
    fi
    
    # MPS availability
    MPS_AVAILABLE=$(python3 -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())" 2>/dev/null || echo "False")
    if [ "$MPS_AVAILABLE" == "True" ]; then
        print_success "MPS available in PyTorch"
    fi
    
else
    print_error "PyTorch not installed"
    echo -e "  Install with: pip install torch torchvision torchaudio"
fi

# ============================================================================
# DEEPSPEED DETECTION
# ============================================================================
print_section "DeepSpeed Availability"

if python3 -c "import deepspeed" 2>/dev/null; then
    DEEPSPEED_VERSION=$(python3 -c "import deepspeed; print(deepspeed.__version__)")
    print_success "DeepSpeed installed"
    print_kv "Version" "$DEEPSPEED_VERSION"
else
    print_warn "DeepSpeed not installed (optional for distributed training)"
    echo -e "  Install with: pip install deepspeed"
fi

# ============================================================================
# DISK SPACE
# ============================================================================
print_section "Disk Space"

DISK_TOTAL=$(df -h . | awk 'NR==2 {print $2}')
DISK_USED=$(df -h . | awk 'NR==2 {print $3}')
DISK_AVAIL=$(df -h . | awk 'NR==2 {print $4}')
DISK_PERCENT=$(df -h . | awk 'NR==2 {print $5}')

print_kv "Total" "$DISK_TOTAL"
print_kv "Used" "$DISK_USED ($DISK_PERCENT)"
print_kv "Available" "$DISK_AVAIL"

# Check disk space
DISK_AVAIL_GB=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_AVAIL_GB" -lt 10 ]; then
    print_error "Less than 10GB free space. LuminaAI needs 10GB+ for checkpoints."
elif [ "$DISK_AVAIL_GB" -lt 50 ]; then
    print_warn "Less than 50GB free space. Recommended: 50GB+ for large models."
else
    print_success "Sufficient disk space available"
fi

# ============================================================================
# RECOMMENDED CONFIGURATION
# ============================================================================
print_section "Recommended LuminaAI Configuration"

echo -e "\n  ${CYAN}Based on detected hardware:${NC}\n"

# Generate recommendations
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    GPU_MEM_GB=$((GPU_MEM_MB / 1024))
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    COMPUTE_MAJOR=$(echo $COMPUTE_CAP | cut -d. -f1)
    
    echo "  # NVIDIA GPU Configuration"
    echo "  training_params = {"
    
    # Precision
    if [ "$COMPUTE_MAJOR" -ge 8 ]; then
        echo "      'precision': 'mixed_bf16',  # Ampere+ GPU"
    elif [ "$COMPUTE_MAJOR" -ge 7 ]; then
        echo "      'precision': 'mixed_fp16',  # Volta/Turing GPU"
    else
        echo "      'precision': 'fp32',  # Older GPU"
    fi
    
    # Batch size
    if [ "$GPU_MEM_GB" -ge 40 ]; then
        echo "      'batch_size': 8,"
        echo "      'gradient_accumulation_steps': 4,"
    elif [ "$GPU_MEM_GB" -ge 16 ]; then
        echo "      'batch_size': 4,"
        echo "      'gradient_accumulation_steps': 8,"
    else
        echo "      'batch_size': 2,"
        echo "      'gradient_accumulation_steps': 16,"
    fi
    
    echo "      'use_flash_attention': True,"
    echo "      'use_deepspeed': True,"
    echo "      'compile': True,"
    echo "      'gradient_checkpointing': True,"
    echo "  }"
    
elif [[ "$OS_TYPE" == "Darwin" ]] && [[ "$OS_ARCH" == "arm64" ]]; then
    echo "  # Apple Silicon (MPS) Configuration"
    echo "  training_params = {"
    echo "      'precision': 'fp16',  # Best for MPS"
    echo "      'batch_size': 2,      # Start small"
    echo "      'gradient_accumulation_steps': 16,"
    echo "      'use_flash_attention': False,  # Not supported on MPS"
    echo "      'use_deepspeed': False,        # Not supported on MPS"
    echo "      'compile': False,              # Can be unstable on MPS"
    echo "      'gradient_checkpointing': True,"
    echo "      'num_workers': 0,              # MPS prefers main thread"
    echo "  }"
    
else
    echo "  # CPU-Only Configuration"
    echo "  training_params = {"
    echo "      'precision': 'fp32',"
    echo "      'batch_size': 1,"
    echo "      'gradient_accumulation_steps': 32,"
    echo "      'use_flash_attention': False,"
    echo "      'use_deepspeed': False,"
    echo "      'compile': False,"
    echo "  }"
    echo ""
    print_warn "CPU-only training is slow. Consider using cloud GPU instances."
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
print_section "Summary"

echo -e "\n  ${GREEN}System Status:${NC}"

# Count capabilities
CAPABILITY_COUNT=0

if command -v nvidia-smi &> /dev/null; then
    echo "     NVIDIA GPU detected"
    ((CAPABILITY_COUNT++))
fi

if [[ "$OS_TYPE" == "Darwin" ]] && [[ "$OS_ARCH" == "arm64" ]]; then
    echo "     Apple Silicon (MPS) detected"
    ((CAPABILITY_COUNT++))
fi

if python3 -c "import torch" 2>/dev/null; then
    echo "     PyTorch installed"
    ((CAPABILITY_COUNT++))
fi

if [ "$CAPABILITY_COUNT" -eq 0 ]; then
    echo -e "\n  ${RED} No GPU acceleration available${NC}"
    echo "    Training will be slow on CPU only."
    echo "    Consider cloud GPU providers: AWS, GCP, Lambda Labs, RunPod"
else
    echo -e "\n  ${GREEN} System ready for LuminaAI training!${NC}"
fi

echo ""
echo -e "${CYAN}${NC}"
echo ""

# Save to file
OUTPUT_FILE="system_properties_$(date +%Y%m%d_%H%M%S).txt"
exec > >(tee -a "$OUTPUT_FILE")
echo "System properties saved to: $OUTPUT_FILE"