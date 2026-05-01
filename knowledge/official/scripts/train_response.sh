#!/bin/bash

################################################################################
# KuiperAI Response Trainer (Reasoning & Thinking)
# Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.
################################################################################

echo "=========================================="
echo "KuiperAI Response Trainer v3.0.0"
echo "Copyright © 2024-2026 Moude AI LLC"
echo "=========================================="
echo ""

# Configuration
RESPONSE_DATA="../response_training.txt"
OUTPUT_DIR="../train/out/response"
PRETRAINED_MODEL="../train/pre"
BATCH_SIZE=16
LEARNING_RATE=1e-5
EPOCHS=5
MAX_LENGTH=4096
SAVE_STEPS=250
LOGGING_STEPS=50

# Check if dataset exists
if [ ! -f "$RESPONSE_DATA" ]; then
    echo "ERROR: Response training dataset not found at $RESPONSE_DATA"
    exit 1
fi

echo "Configuration:"
echo "  Dataset: $RESPONSE_DATA"
echo "  Output: $OUTPUT_DIR"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $EPOCHS"
echo "  Max Length: $MAX_LENGTH"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting Response Training..."
echo "This will train the AI on reasoning, thinking, and response generation."
echo "Estimated time: 24-48 hours on A100 GPU"
echo ""

# Run training
python3 train_response.py \
    --train_file "$RESPONSE_DATA" \
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
    --warmup_steps 1000 \
    --weight_decay 0.01 \
    --logging_dir "$OUTPUT_DIR/logs" \
    --report_to tensorboard \
    --do_train \
    --overwrite_output_dir

echo ""
echo "=========================================="
echo "Response Training Complete!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
