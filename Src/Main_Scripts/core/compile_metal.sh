#!/bin/bash
# Compile Metal shaders for Apple Silicon
# Copyright (c) 2025 MatN23. All rights reserved.

set -e  # Exit on error

export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"

echo " Compiling Metal Shaders for Apple Silicon"
echo "=============================================="
echo ""

# Check if on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo " Error: Metal compilation only available on macOS"
    echo "   Current OS: $OSTYPE"
    exit 1
fi

# Check if xcrun is available
if ! command -v xcrun &> /dev/null; then
    echo " Error: xcrun not found"
    echo "   Install Xcode Command Line Tools:"
    echo "   xcode-select --install"
    exit 1
fi

# Check if metal compiler is available
if ! xcrun -sdk macosx metal --version &> /dev/null; then
    echo " Error: Metal compiler not found"
    echo "   Install Xcode Command Line Tools:"
    echo "   xcode-select --install"
    exit 1
fi

echo " Metal compiler found"
xcrun -sdk macosx metal --version
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo " Working directory: $SCRIPT_DIR"
echo ""

# Compile transformer_ops.metal
echo "1  Compiling transformer_ops.metal..."
if [ ! -f "transformer_ops.metal" ]; then
    echo "    transformer_ops.metal not found!"
    exit 1
fi

echo "   Building AIR..."
xcrun -sdk macosx metal -c transformer_ops.metal -o transformer_ops.air
if [ $? -ne 0 ]; then
    echo "    Failed to compile transformer_ops.metal"
    exit 1
fi

echo "   Linking metallib..."
xcrun -sdk macosx metallib transformer_ops.air -o transformer_ops.metallib
if [ $? -ne 0 ]; then
    echo "    Failed to create transformer_ops.metallib"
    rm -f transformer_ops.air
    exit 1
fi

rm -f transformer_ops.air
echo "    transformer_ops.metallib created"
echo ""

# Compile moe_ops.metal
echo "2  Compiling moe_ops.metal..."
if [ ! -f "moe_ops.metal" ]; then
    echo "    moe_ops.metal not found!"
    exit 1
fi

echo "   Building AIR..."
xcrun -sdk macosx metal -c moe_ops.metal -o moe_ops.air
if [ $? -ne 0 ]; then
    echo "    Failed to compile moe_ops.metal"
    exit 1
fi

echo "   Linking metallib..."
xcrun -sdk macosx metallib moe_ops.air -o moe_ops.metallib
if [ $? -ne 0 ]; then
    echo "    Failed to create moe_ops.metallib"
    rm -f moe_ops.air
    exit 1
fi

rm -f moe_ops.air
echo "    moe_ops.metallib created"
echo ""

# Verify outputs
echo " Verification:"
if [ -f "transformer_ops.metallib" ]; then
    SIZE=$(du -h transformer_ops.metallib | cut -f1)
    echo "    transformer_ops.metallib ($SIZE)"
else
    echo "    transformer_ops.metallib missing"
    exit 1
fi

if [ -f "moe_ops.metallib" ]; then
    SIZE=$(du -h moe_ops.metallib | cut -f1)
    echo "    moe_ops.metallib ($SIZE)"
else
    echo "    moe_ops.metallib missing"
    exit 1
fi

echo ""
echo "=============================================="
echo " Metal shaders compiled