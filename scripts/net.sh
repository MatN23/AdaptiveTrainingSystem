#!/bin/bash
# net.sh - Network Diagnostics for Distributed Training
# Tests network connectivity and bandwidth for multi-node training

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print header
echo -e "${CYAN}${NC}"
echo -e "${CYAN}           LuminaAI Network Diagnostics for Distributed Training       ${NC}"
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
# BASIC NETWORK INFORMATION
# ============================================================================
print_section "Network Configuration"

HOSTNAME=$(hostname)
print_kv "Hostname" "$HOSTNAME"

# Get primary network interface
if command -v ip &> /dev/null; then
    PRIMARY_IF=$(ip route | grep default | awk '{print $5}' | head -1)
    PRIMARY_IP=$(ip addr show $PRIMARY_IF | grep "inet " | awk '{print $2}' | cut -d/ -f1)
    
    print_kv "Primary Interface" "$PRIMARY_IF"
    print_kv "Primary IP" "$PRIMARY_IP"
    
    # Get MAC address
    MAC_ADDR=$(ip addr show $PRIMARY_IF | grep "link/ether" | awk '{print $2}')
    print_kv "MAC Address" "$MAC_ADDR"
    
    # Get MTU
    MTU=$(ip addr show $PRIMARY_IF | grep mtu | awk '{print $5}')
    print_kv "MTU" "$MTU"
    
    if [ "$MTU" -lt 1500 ]; then
        print_warn "MTU is less than 1500. May impact performance."
    elif [ "$MTU" -ge 9000 ]; then
        print_success "Jumbo frames enabled (MTU: $MTU)"
    fi
    
elif [[ "$(uname -s)" == "Darwin" ]]; then
    PRIMARY_IF=$(route get default | grep interface | awk '{print $2}')
    PRIMARY_IP=$(ifconfig $PRIMARY_IF | grep "inet " | awk '{print $2}')
    
    print_kv "Primary Interface" "$PRIMARY_IF"
    print_kv "Primary IP" "$PRIMARY_IP"
    
    MAC_ADDR=$(ifconfig $PRIMARY_IF | grep ether | awk '{print $2}')
    print_kv "MAC Address" "$MAC_ADDR"
fi

# ============================================================================
# NETWORK INTERFACES
# ============================================================================
print_section "Network Interfaces"

if command -v ip &> /dev/null; then
    echo ""
    ip -br addr show | while read -r line; do
        IFACE=$(echo $line | awk '{print $1}')
        STATE=$(echo $line | awk '{print $2}')
        IP=$(echo $line | awk '{print $3}')
        
        if [ "$STATE" == "UP" ]; then
            echo -e "  ${GREEN}${NC} $IFACE ($STATE) - $IP"
        else
            echo -e "  ${YELLOW}${NC} $IFACE ($STATE) - $IP"
        fi
    done
elif [[ "$(uname -s)" == "Darwin" ]]; then
    echo ""
    ifconfig | grep "^[a-z]" | while read -r line; do
        IFACE=$(echo $line | awk -F: '{print $1}')
        echo -e "  ${GREEN}${NC} $IFACE"
    done
fi

# ============================================================================
# INTERNET CONNECTIVITY
# ============================================================================
print_section "Internet Connectivity"

echo ""
echo "  Testing connectivity to common services..."

# Test DNS resolution
if host google.com > /dev/null 2>&1; then
    print_success "DNS resolution working"
else
    print_error "DNS resolution failed"
fi

# Test ping to common hosts
HOSTS=("8.8.8.8" "1.1.1.1" "google.com")
for host in "${HOSTS[@]}"; do
    if ping -c 1 -W 2 $host > /dev/null 2>&1; then
        LATENCY=$(ping -c 3 -W 2 $host 2>/dev/null | tail -1 | awk -F '/' '{print $5}')
        if [ -n "$LATENCY" ]; then
            print_success "Ping to $host: ${LATENCY}ms"
        else
            print_success "Ping to $host: OK"
        fi
    else
        print_error "Cannot reach $host"
    fi
done

# ============================================================================
# BANDWIDTH TEST
# ============================================================================
print_section "Bandwidth Test"

echo ""
echo "  Testing download speed..."

# Simple bandwidth test using curl
TEST_URL="http://speedtest.tele2.net/1MB.zip"
START_TIME=$(date +%s.%N)

if curl -s -o /dev/null -w "%{speed_download}" $TEST_URL > /tmp/speed_test.txt 2>/dev/null; then
    END_TIME=$(date +%s.%N)
    SPEED=$(cat /tmp/speed_test.txt)
    SPEED_MBPS=$(echo "scale=2; $SPEED / 1024 / 1024 * 8" | bc)
    
    print_kv "Download Speed" "${SPEED_MBPS} Mbps"
    rm -f /tmp/speed_test.txt
    
    # Interpret results
    if (( $(echo "$SPEED_MBPS > 100" | bc -l) )); then
        print_success "Excellent bandwidth (100+ Mbps)"
    elif (( $(echo "$SPEED_MBPS > 10" | bc -l) )); then
        print_success "Good bandwidth (10+ Mbps)"
    else
        print_warn "Low bandwidth (<10 Mbps). May affect distributed training."
    fi
else
    print_warn "Could not measure download speed"
fi

# ============================================================================
# PORT AVAILABILITY (Common distributed training ports)
# ============================================================================
print_section "Port Availability"

echo ""
echo "  Checking common distributed training ports..."

PORTS=(
    "22:SSH"
    "6379:Redis"
    "29500:PyTorch DDP (default)"
    "29501:PyTorch DDP (alt)"
    "8888:Jupyter"
    "6006:TensorBoard"
)

for port_info in "${PORTS[@]}"; do
    PORT=$(echo $port_info | cut -d: -f1)
    DESC=$(echo $port_info | cut -d: -f2)
    
    if command -v netstat &> /dev/null; then
        if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
            echo -e "  ${YELLOW}${NC} Port $PORT ($DESC) - IN USE"
        else
            echo -e "  ${GREEN}${NC} Port $PORT ($DESC) - Available"
        fi
    elif command -v ss &> /dev/null; then
        if ss -tuln 2>/dev/null | grep -q ":$PORT "; then
            echo -e "  ${YELLOW}${NC} Port $PORT ($DESC) - IN USE"
        else
            echo -e "  ${GREEN}${NC} Port $PORT ($DESC) - Available"
        fi
    else
        echo -e "  ${BLUE}?${NC} Port $PORT ($DESC) - Cannot check (netstat/ss not available)"
    fi
done

# ============================================================================
# FIREWALL STATUS
# ============================================================================
print_section "Firewall Status"

if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status 2>/dev/null | head -1 || echo "Cannot check (requires sudo)")
    print_kv "UFW Status" "$UFW_STATUS"
elif command -v firewall-cmd &> /dev/null; then
    FW_STATUS=$(sudo firewall-cmd --state 2>/dev/null || echo "Cannot check (requires sudo)")
    print_kv "Firewalld Status" "$FW_STATUS"
elif [[ "$(uname -s)" == "Darwin" ]]; then
    FW_STATUS=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null || echo "Unknown")
    print_kv "macOS Firewall" "$FW_STATUS"
else
    print_kv "Firewall" "Unknown"
fi

# ============================================================================
# DISTRIBUTED TRAINING READINESS
# ============================================================================
print_section "Distributed Training Readiness"

echo ""

# Check for NCCL (NVIDIA Collective Communications Library)
if python3 -c "import torch; torch.cuda.nccl.version()" 2>/dev/null; then
    NCCL_VERSION=$(python3 -c "import torch; print('.'.join(map(str, torch.cuda.nccl.version())))")
    print_success "NCCL available (version $NCCL_VERSION)"
else
    print_warn "NCCL not available (needed for multi-GPU training)"
fi

# Check for MPI (Message Passing Interface)
if command -v mpirun &> /dev/null; then
    MPI_VERSION=$(mpirun --version 2>/dev/null | head -1 || echo "Unknown")
    print_success "MPI available: $MPI_VERSION"
else
    print_warn "MPI not available (optional for distributed training)"
fi

# Check environment variables
echo ""
echo "  Distributed Training Environment Variables:"
if [ -n "$MASTER_ADDR" ]; then
    print_kv "MASTER_ADDR" "$MASTER_ADDR"
else
    echo -e "  ${BLUE}${NC} MASTER_ADDR not set"
fi

if [ -n "$MASTER_PORT" ]; then
    print_kv "MASTER_PORT" "$MASTER_PORT"
else
    echo -e "  ${BLUE}${NC} MASTER_PORT not set"
fi

if [ -n "$WORLD_SIZE" ]; then
    print_kv "WORLD_SIZE" "$WORLD_SIZE"
else
    echo -e "  ${BLUE}${NC} WORLD_SIZE not set"
fi

if [ -n "$RANK" ]; then
    print_kv "RANK" "$RANK"
else
    echo -e "  ${BLUE}${NC} RANK not set"
fi

# ============================================================================
# NETWORK PERFORMANCE TEST (if multiple nodes specified)
# ============================================================================
if [ $# -gt 0 ]; then
    print_section "Multi-Node Network Test"
    
    echo ""
    echo "  Testing connectivity to specified nodes..."
    
    for node in "$@"; do
        echo ""
        echo -e "  ${CYAN}Testing: $node${NC}"
        
        # Ping test
        if ping -c 3 -W 2 $node > /dev/null 2>&1; then
            LATENCY=$(ping -c 3 -W 2 $node 2>/dev/null | tail -1 | awk -F '/' '{print $5}')
            print_success "Ping: ${LATENCY}ms"
        else
            print_error "Cannot reach $node"
            continue
        fi
        
        # SSH test (port 22)
        if nc -z -w 2 $node 22 2>/dev/null; then
            print_success "SSH port (22) is open"
        else
            print_warn "SSH port (22) is not accessible"
        fi
        
        # PyTorch DDP port test (29500)
        if nc -z -w 2 $node 29500 2>/dev/null; then
            print_success "PyTorch DDP port (29500) is open"
        else
            echo -e "  ${BLUE}${NC} PyTorch DDP port (29500) not open (may be fine if not running)"
        fi
    done
fi

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print_section "Recommendations"

echo ""

# Single node vs multi-node
if [ $# -eq 0 ]; then
    echo "  ${CYAN}Single Node Training:${NC}"
    echo "     Use NCCL for multi-GPU communication"
    echo "     Set MASTER_PORT to an available port (e.g., 29500)"
    echo "     Example: export MASTER_PORT=29500"
    echo ""
    echo "  ${CYAN}To test multi-node connectivity:${NC}"
    echo "     Run: ./net.sh node1 node2 node3"
else
    echo "  ${CYAN}Multi-Node Training:${NC}"
    echo "     Ensure all nodes can reach each other"
    echo "     Use consistent network interface on all nodes"
    echo "     Configure firewall to allow ports 22 and 29500"
    echo "     Set up SSH key-based authentication"
    echo ""
    echo "  ${CYAN}Environment setup:${NC}"
    echo "    export MASTER_ADDR=$1"
    echo "    export MASTER_PORT=29500"
    echo "    export WORLD_SIZE=$(($# + 1))"
    echo "    export RANK=0  # Set to 0, 1, 2, ... on each node"
fi

# Bandwidth recommendations
echo ""
echo "  ${CYAN}Bandwidth Requirements:${NC}"
echo "     Local training: N/A"
echo "     Multi-GPU (single node): PCIe/NVLink bandwidth important"
echo "     Multi-node training: 10+ Gbps recommended"
echo "     Large model training: 25+ Gbps or InfiniBand preferred"

# ============================================================================
# DEEPSPEED SPECIFIC CHECKS
# ============================================================================
if python3 -c "import deepspeed" 2>/dev/null; then
    print_section "DeepSpeed Configuration"
    
    echo ""
    
    # Check hostfile
    if [ -f "hostfile" ]; then
        print_success "Found hostfile"
        echo ""
        echo "  Hostfile contents:"
        cat hostfile | while read -r line; do
            echo "    $line"
        done
    else
        echo -e "  ${BLUE}${NC} No hostfile found (create one for multi-node training)"
        echo ""
        echo "  Example hostfile:"
        echo "    node1 slots=4"
        echo "    node2 slots=4"
        echo "    node3 slots=4"
    fi
    
    # SSH configuration
    echo ""
    echo "  ${CYAN}DeepSpeed SSH Requirements:${NC}"
    echo "     Password-less SSH between all nodes"
    echo "     Consistent Python environment on all nodes"
    echo "     Same working directory on all nodes"
    echo ""
    echo "  Test SSH access:"
    if [ $# -gt 0 ]; then
        for node in "$@"; do
            if ssh -o BatchMode=yes -o ConnectTimeout=5 $node "echo OK" 2>/dev/null | grep -q "OK"; then
                echo -e "    ${GREEN}${NC} SSH to $node: Working"
            else
                echo -e "    ${RED}${NC} SSH to $node: Failed (configure key-based auth)"
            fi
        done
    else
        echo "    Run: ./net.sh node1 node2 node3 to test"
    fi
fi

# ============================================================================
# SUMMARY
# ============================================================================
print_section "Summary"

echo ""
echo "  ${GREEN}Network Status:${NC}"

READY=true

# Check basic connectivity
if ping -c 1 -W 2 8.8.8.8 > /dev/null 2>&1; then
    echo "     Internet connectivity working"
else
    echo "     No internet connectivity"
    READY=false
fi

# Check for distributed training capabilities
if python3 -c "import torch; torch.cuda.nccl.version()" 2>/dev/null; then
    echo "     NCCL available for multi-GPU"
else
    echo "     NCCL not available (needed for multi-GPU)"
fi

if [ $# -gt 0 ]; then
    ALL_REACHABLE=true
    for node in "$@"; do
        if ! ping -c 1 -W 2 $node > /dev/null 2>&1; then
            ALL_REACHABLE=false
            break
        fi
    done
    
    if $ALL_REACHABLE; then
        echo "     All specified nodes reachable"
    else
        echo "     Some nodes unreachable"
        READY=false
    fi
fi

echo ""
if $READY; then
    echo -e "  ${GREEN} Network ready for distributed training!${NC}"
else
    echo -e "  ${YELLOW} Some network issues detected. Review recommendations above.${NC}"
fi

echo ""
echo -e "${CYAN}${NC}"
echo ""

# Save to file
OUTPUT_FILE="network_diagnostics_$(date +%Y%m%d_%H%M%S).txt"
echo "Network diagnostics saved to: $OUTPUT_FILE"