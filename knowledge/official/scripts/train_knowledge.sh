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
BATCH_SIZE=16  # Reduced for dual T4 with BF16
LEARNING_RATE=5e-5  # Higher for pretraining from random weights
EPOCHS=3
MAX_LENGTH=1024  # Reduced to fit in memory
SAVE_STEPS=1000
LOGGING_STEPS=100

# Check if dataset exists
if [ ! -f "$KNOWLEDGE_DATA" ]; then
    echo "ERROR: Knowledge training dataset not found at $KNOWLEDGE_DATA"
    exit 1
fi

# Check if model exists
if [ ! -f "$PRETRAINED_MODEL/model.safetensors" ]; then
    echo "ERROR: Model not found at $PRETRAINED_MODEL"
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $KNOWLEDGE_DATA"
echo "  Model: $PRETRAINED_MODEL (random weights - needs pretraining)"
echo "  Output: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE per GPU"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Max Length: $MAX_LENGTH tokens"
echo "  Mixed Precision: BF16 (for dual GPU)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

echo "Starting Knowledge Training..."
echo "This will train the AI on comprehensive factual knowledge."
echo "Estimated time: 24-48 hours on dual T4 GPUs"
echo ""

# Run training with BF16 for dual GPU setup
python3 train_knowledge.py \
    --train_file "$KNOWLEDGE_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --model_name_or_path "$PRETRAINED_MODEL" \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 2 \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $EPOCHS \
    --max_seq_length $MAX_LENGTH \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 3 \
    --bf16 \
    --gradient_checkpointing \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --logging_dir "$OUTPUT_DIR/logs" \
    --report_to tensorboard \
    --do_train \
    --overwrite_output_dir \
    --dataloader_num_workers 4 \
    --dataloader_pin_memory \
    --lr_scheduler_type cosine \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0

echo ""
echo "=========================================="
echo "Knowledge Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Evaluate the pretrained model"
echo "  2. Fine-tune on instruction data"
echo "  3. Test with chat interface"
echo ""
