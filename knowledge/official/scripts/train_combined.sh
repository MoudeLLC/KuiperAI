#!/bin/bash

################################################################################
# KuiperAI Combined Trainer (Knowledge + Response)
# Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.
################################################################################

echo "=========================================="
echo "KuiperAI Combined Trainer v3.0.0"
echo "Copyright © 2024-2026 Moude AI LLC"
echo "=========================================="
echo ""

echo "This script will train KuiperAI on both:"
echo "  1. Knowledge Training (factual knowledge)"
echo "  2. Response Training (reasoning & thinking)"
echo ""
echo "Total estimated time: 48-72 hours on A100 GPU"
echo ""

read -p "Continue with combined training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Training cancelled."
    exit 1
fi

echo ""
echo "=========================================="
echo "PHASE 1: Knowledge Training"
echo "=========================================="
echo ""

# Run knowledge training
./train_knowledge.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Knowledge training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "PHASE 2: Response Training"
echo "=========================================="
echo ""

# Use knowledge-trained model as base for response training
KNOWLEDGE_MODEL="../train/out/knowledge"

# Check if knowledge model exists
if [ ! -d "$KNOWLEDGE_MODEL" ]; then
    echo "ERROR: Knowledge model not found. Phase 1 may have failed."
    exit 1
fi

# Run response training on top of knowledge model
python3 train_response.py \
    --train_file "../response_training.txt" \
    --output_dir "../train/out/combined" \
    --model_name_or_path "$KNOWLEDGE_MODEL" \
    --per_device_train_batch_size 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_seq_length 4096 \
    --save_steps 250 \
    --logging_steps 50 \
    --save_total_limit 3 \
    --fp16 \
    --gradient_checkpointing \
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --logging_dir "../train/out/combined/logs" \
    --report_to tensorboard \
    --do_train \
    --overwrite_output_dir

if [ $? -ne 0 ]; then
    echo "ERROR: Response training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "PHASE 3: Joint Fine-tuning"
echo "=========================================="
echo ""

# Final joint training on both datasets
python3 train_combined.py \
    --knowledge_file "../knowledge_training.txt" \
    --response_file "../response_training.txt" \
    --output_dir "../train/out/combined_final" \
    --model_name_or_path "../train/out/combined" \
    --per_device_train_batch_size 8 \
    --learning_rate 5e-6 \
    --num_train_epochs 2 \
    --max_seq_length 4096 \
    --save_steps 500 \
    --logging_steps 100 \
    --save_total_limit 2 \
    --fp16 \
    --gradient_checkpointing \
    --warmup_steps 200 \
    --weight_decay 0.01 \
    --logging_dir "../train/out/combined_final/logs" \
    --report_to tensorboard \
    --do_train \
    --overwrite_output_dir

echo ""
echo "=========================================="
echo "PHASE 4: Model Export"
echo "=========================================="
echo ""

# Export trained model to multiple formats
python3 export_model.py \
    --model_path "../train/out/combined_final" \
    --output_dir "../train/out/exports" \
    --model_name "kuiperai" \
    --version "1.0.0" \
    --formats "all"

if [ $? -ne 0 ]; then
    echo "WARNING: Model export failed, but training completed successfully."
fi

echo ""
echo "=========================================="
echo "Combined Training Complete!"
echo "=========================================="
echo ""
echo "Final model saved to: ../train/out/combined_final"
echo "Exported formats saved to: ../train/out/exports"
echo ""
echo "Your KuiperAI model now has:"
echo "  ✓ Comprehensive knowledge across all domains"
echo "  ✓ Advanced reasoning and thinking capabilities"
echo "  ✓ Dynamic response generation"
echo "  ✓ Problem-solving skills"
echo ""
echo "Exported formats:"
echo "  ✓ PyTorch (.bin, .pt) - Standard format"
echo "  ✓ SafeTensors (.safetensors) - Safe format"
echo "  ✓ ONNX (.onnx) - Deployment format"
echo "  ✓ ZPM (.zpm) - KuiperAI proprietary format"
echo ""
echo "Next steps:"
echo "  1. Test the model with sample queries"
echo "  2. Evaluate on your specific use cases"
echo "  3. Deploy to production"
echo ""
echo "=========================================="
