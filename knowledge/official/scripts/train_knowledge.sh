#!/bin/bash

################################################################################
# KuiperAI Knowledge Trainer
# Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.
################################################################################

echo "=========================================="
echo "KuiperAI Knowledge Trainer v3.0.0"
echo "Copyright © 2024-2026 Moude AI LLC"
echo "=========================================="
echo ""

# Configuration
KNOWLEDGE_DATA="../knowledge_training.txt"
OUTPUT_DIR="../train/out/knowledge"
PRETRAINED_MODEL="../train/pre"
BATCH_SIZE=32
LEARNING_RATE=2e-5
EPOCHS=3
MAX_LENGTH=2048
SAVE_STEPS=500
LOGGING_STEPS=100

# Check if dataset exists
if [ ! -f "$KNOWLEDGE_DATA" ]; then
    echo "ERROR: Knowledge training dataset not found at $KNOWLEDGE_DATA"
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $KNOWLEDGE_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Max Length: $MAX_LENGTH"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting Knowledge Training..."
echo "This will train the AI on comprehensive factual knowledge."
echo "Estimated time: 12-24 hours on A100 GPU"
echo ""

# Run training
python3 train_knowledge.py \
    --train_file "$KNOWLEDGE_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$PRETRAINED_MODEL" \
    --per_device_train_batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 3 \
    --fp16 \
    --gradient_checkpointing \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --logging_dir "$OUTPUT_DIR/logs" \
    --report_to tensorboard \
    --do_train \
    --overwrite_output_dir

echo ""
echo "=========================================="
echo "Knowledge Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
