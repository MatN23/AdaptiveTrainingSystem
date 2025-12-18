#!/bin/bash
# run_inference.sh - Run MoE inference
# Usage: ./run_inference.sh <binary> <model.bin> <prompt>

set -e

BINARY=${1:-moe_inference_cpu}
MODEL=${2:-model.bin}
PROMPT=${3:-"Once upon a time"}

if [[ ! -f "$BINARY" ]]; then
    echo "❌ Binary not found: $BINARY"
    echo "Run ./compile.sh first"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "❌ Model not found: $MODEL"
    echo "Run: python export/export_model.py"
    exit 1
fi

echo "🚀 Running inference..."
echo "   Binary: $BINARY"
echo "   Model:  $MODEL"
echo "   Prompt: $PROMPT"
echo ""

# Run inference
$BINARY "$MODEL" "$PROMPT"