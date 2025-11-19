#!/usr/bin/env bash
set -e

echo "Checking for NVIDIA GPU..."

# Detect GPU using nvidia-smi (works in Docker + bare metal)
#if command -v nvidia-smi &> /dev/null; then
#    echo "GPU detected! Installing CUDA-enabled PyTorch..."
#    uv add --index-url https://download.pytorch.org/whl/cu121 \
#        torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
#    echo "Installed PyTorch GPU build (CUDA 12.1)"
#else
#    echo "No GPU detected. Installing CPU-only PyTorch..."
#    uv add torch torchvision torchaudio
#    echo "Installed CPU-only PyTorch"
#fi


uv add --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1