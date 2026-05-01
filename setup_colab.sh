#!/bin/bash
# KuiperAI Colab Setup Script
# This script sets up the complete training environment in Google Colab

echo "=========================================="
echo "KuiperAI Colab Setup"
echo "=========================================="
echo ""

# Install gdown if not available
echo "Installing dependencies..."
pip install -q gdown

# Download the pretrained model
echo ""
echo "Downloading pretrained model (4.6GB)..."
gdown --id 1h4L7RQiGQceW0w7htDa-RwwFvfqCJw1i -O kuiperai_model.zip

# Extract the model
echo ""
echo "Extracting model..."
unzip -q kuiperai_model.zip

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Model location: $(pwd)/train/pre/"
echo ""
echo "To start training, run:"
echo "python3 knowledge/official/scripts/train_knowledge.py \\"
echo "  --train_file knowledge/official/knowledge_training.txt \\"
echo "  --output_dir knowledge/official/train/out \\"
echo "  --model_name_or_path knowledge/official/train/pre \\"
echo "  --do_train"
echo ""
