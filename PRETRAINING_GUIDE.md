# KuiperAI K1 Pretraining Guide

## Overview
Your K1-4k model currently has **random weights** and needs pretraining to learn language understanding before it can be fine-tuned for instruction following.

## What We Have
- **Model**: K1-4k (1.3B parameters, GPT-2 architecture)
- **Training Data**: 18.3M lines of knowledge in `knowledge/official/knowledge_training.txt`
- **Hardware**: Dual T4 GPUs (30GB total VRAM)
- **Enhanced Script**: `knowledge/official/scripts/train_knowledge.py` with robust error handling

## Enhanced Features Added

### 1. Better GPU Detection
- Shows GPU memory, compute capability
- Automatically uses `device_map="auto"` for multi-GPU
- Falls back gracefully if multi-GPU fails

### 2. Improved Error Handling
- Try-catch blocks for dataset loading
- Graceful handling of tokenization errors
- Saves model on interruption or failure
- Detailed error messages

### 3. Memory Optimization
- Proper BF16 support (not just FP16)
- GPU memory monitoring before/after training
- Gradient checkpointing with error handling
- Optimized data loading with prefetch

### 4. Better Logging
- Training progress with detailed stats
- Saves `training_stats.json` with final metrics
- Shows effective batch size calculation
- Memory usage tracking

### 5. Pretraining-Specific Settings
- Higher learning rate (5e-5 vs 2e-5) for random weights
- Cosine LR scheduler for better convergence
- More warmup steps (2000 vs 500)
- Higher weight decay (0.1 vs 0.01)
- Adam beta2 = 0.95 (better for pretraining)

## How to Run

### Option 1: Using the Shell Script (Recommended)
```bash
cd knowledge/official/scripts
./train_knowledge.sh
```

### Option 2: Direct Python Command
```bash
cd knowledge/official/scripts

python3 train_knowledge.py \
    --train_file "../knowledge_training.txt" \
    --output_dir "../train/out/knowledge" \
    --model_name_or_path "../train/pre" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --max_seq_length 1024 \
    --save_steps 1000 \
    --logging_steps 100 \
    --save_total_limit 3 \
    --bf16 \
    --gradient_checkpointing \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --logging_dir "../train/out/knowledge/logs" \
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
```

## Key Configuration Explained

| Parameter | Value | Why |
|-----------|-------|-----|
| `--bf16` | enabled | Uses both GPUs without OOM, better than FP16 |
| `--per_device_train_batch_size` | 16 | Safe for T4 memory |
| `--gradient_accumulation_steps` | 2 | Effective batch size = 32 |
| `--max_seq_length` | 1024 | Reduced from 2048 to fit in memory |
| `--learning_rate` | 5e-5 | Higher for pretraining from scratch |
| `--warmup_steps` | 2000 | Gradual warmup for stability |
| `--weight_decay` | 0.1 | Prevents overfitting during pretraining |
| `--lr_scheduler_type` | cosine | Better convergence for pretraining |
| `--gradient_checkpointing` | enabled | Saves memory by recomputing activations |
| `--adam_beta2` | 0.95 | Better for pretraining (vs 0.999 for fine-tuning) |

## Expected Timeline

- **Total Time**: 24-48 hours on dual T4 GPUs
- **Checkpoints**: Saved every 1000 steps
- **Logs**: Available in `train/out/knowledge/logs/`

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir knowledge/official/train/out/knowledge/logs
```

### Watch GPU Usage
```bash
watch -n 1 nvidia-smi
```

### Check Progress
```bash
tail -f knowledge/official/train/out/knowledge/logs/events.out.tfevents.*
```

## What Happens During Pretraining

1. **Model loads** with random weights from `train/pre/`
2. **Dataset loads** 18.3M lines of text
3. **Tokenization** converts text to tokens (may take 10-20 minutes)
4. **Training begins** - model learns language patterns
5. **Checkpoints saved** every 1000 steps
6. **Final model saved** to `train/out/knowledge/`

## After Pretraining

Once pretraining completes, you'll have a model that understands language. Next steps:

1. **Evaluate**: Test the model's language understanding
2. **Fine-tune**: Train on OpenHermes for instruction following
3. **Deploy**: Use the model for chat/inference

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--per_device_train_batch_size` to 8
- Reduce `--max_seq_length` to 512
- Ensure `--gradient_checkpointing` is enabled

### Training Too Slow
- Increase `--dataloader_num_workers` to 8
- Reduce `--logging_steps` to 500
- Use `--max_steps` to limit training

### Model Not Learning
- Check loss is decreasing (should go from ~10 to ~3-4)
- Increase `--learning_rate` to 1e-4
- Increase `--warmup_steps` to 5000

### Interrupted Training
- Resume with `--resume_from_checkpoint train/out/knowledge/checkpoint-XXXX`
- Model auto-saves on interruption to `train/out/knowledge/interrupted/`

## Files Generated

```
train/out/knowledge/
├── config.json                 # Model configuration
├── model.safetensors          # Trained weights
├── tokenizer.json             # Tokenizer
├── training_stats.json        # Training metrics
├── checkpoint-1000/           # Checkpoint at step 1000
├── checkpoint-2000/           # Checkpoint at step 2000
└── logs/                      # TensorBoard logs
```

## Next Steps After Pretraining

1. **Test the pretrained model**:
   ```bash
   python3 chat_with_ai.sh
   ```

2. **Fine-tune on instructions** (OpenHermes):
   ```bash
   cd knowledge/official/scripts
   ./train_response.sh
   ```

3. **Evaluate performance**:
   - Test on common sense questions
   - Check perplexity on validation set
   - Compare to baseline GPT-2

---

**Status**: Ready to pretrain K1-4k from random weights
**Last Updated**: May 3, 2026
