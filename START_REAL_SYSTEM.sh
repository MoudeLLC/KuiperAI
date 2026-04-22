#!/bin/bash

echo "======================================================================"
echo "KUIPERAI - REAL AUTOREGRESSIVE LANGUAGE MODEL"
echo "======================================================================"
echo ""
echo "This is a REAL language model that:"
echo "  ✓ Learns to predict next tokens (like GPT)"
echo "  ✓ Generates new text autoregressively"
echo "  ✓ Understands language patterns"
echo "  ✓ Not just retrieval from dataset"
echo ""
echo "======================================================================"
echo ""

# Check if model exists
if [ -f "checkpoints/autoregressive_best.pt" ]; then
    echo "✓ Trained model found!"
    echo ""
    echo "Starting chat system..."
    echo ""
    python3 chat_autoregressive.py
else
    echo "⚠ No trained model found!"
    echo ""
    echo "You need to train the model first:"
    echo "  python3 train_autoregressive.py"
    echo ""
    echo "Training takes ~5-10 minutes and will:"
    echo "  • Train on 1,311 examples"
    echo "  • Learn next-token prediction"
    echo "  • Save best model automatically"
    echo ""
    read -p "Start training now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 train_autoregressive.py
        echo ""
        echo "======================================================================"
        echo "Training complete! Starting chat..."
        echo "======================================================================"
        echo ""
        python3 chat_autoregressive.py
    fi
fi
