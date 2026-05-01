#!/bin/bash

################################################################################
# KuiperAI Setup Script
# Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.
################################################################################

echo "=========================================="
echo "KuiperAI Setup v3.0.0"
echo "Copyright © 2024-2026 Moude AI LLC"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install transformers datasets accelerate tensorboard wandb
pip3 install sentencepiece protobuf

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies."
    exit 1
fi

echo ""
echo "✓ Dependencies installed successfully!"

# Check for GPU
echo ""
echo "Checking for GPU..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Initialize pretrained model
echo ""
echo "=========================================="
echo "Initializing Pretrained Model"
echo "=========================================="
echo ""

read -p "Initialize pretrained model now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    python3 initialize_pretrained.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Pretrained model initialized successfully!"
        echo ""
        read -p "Remove initialization files? (recommended) (y/n) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ./cleanup_init.sh
        else
            echo "You can run ./cleanup_init.sh later to remove initialization files."
        fi
    fi
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Review datasets in knowledge/official/"
echo "  2. Choose training mode:"
echo "     - ./train_knowledge.sh (knowledge only)"
echo "     - ./train_response.sh (reasoning only)"
echo "     - ./train_combined.sh (both - recommended)"
echo "  3. Monitor training with TensorBoard"
echo "  4. Trained models will be exported to multiple formats automatically"
echo ""
echo "Export formats:"
echo "  - PyTorch (.bin, .pt) - Standard format"
echo "  - SafeTensors (.safetensors) - Safe format"
echo "  - ONNX (.onnx) - Deployment format"
echo "  - ZPM (.zpm) - KuiperAI proprietary format"
echo ""
echo "For help: See knowledge/official/help/NEED_HELP.md"
echo ""
