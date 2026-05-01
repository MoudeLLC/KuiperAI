# KuiperAI Training System - Help & Support Guide

**Version:** 3.0.0  
**Last Updated:** April 29, 2026  
**Support:** support@kuiperai.com

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Training Options](#training-options)
3. [Hardware Requirements](#hardware-requirements)
4. [Troubleshooting](#troubleshooting)
5. [Advanced Configuration](#advanced-configuration)
6. [Support Resources](#support-resources)

---

## Quick Start

### Step 1: Verify Your Setup

Check that you have all required files:

```bash
cd knowledge/official
ls -la
```

You should see:
- `knowledge_training.txt` - Knowledge base dataset
- `response_training.txt` - Reasoning and response generation dataset
- `scripts/` - Training scripts directory
- `train/pre/` - Pre-trained model directory
- `train/out/` - Output directory for trained models

### Step 2: Choose Your Training Mode

KuiperAI offers three training modes:

**Mode 1: Knowledge Trainer Only**
- Trains on factual knowledge across multiple domains
- Best for: Building comprehensive knowledge base
- Training time: 12-24 hours on A100 GPU
- Command: `./scripts/train_knowledge.sh`

**Mode 2: Response Trainer Only**
- Trains on reasoning, thinking, and response generation
- Best for: Improving reasoning and problem-solving
- Training time: 24-48 hours on A100 GPU
- Command: `./scripts/train_response.sh`

**Mode 3: Combined Training (Recommended)**
- Trains on both knowledge and reasoning
- Best for: Complete AI system with knowledge + thinking
- Training time: 48-72 hours on A100 GPU
- Command: `./scripts/train_combined.sh`

### Step 3: Run Training

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Run your chosen training mode
./scripts/train_combined.sh
```

---

## Training Options

### Knowledge Trainer

**Purpose:** Trains the AI on comprehensive factual knowledge

**Dataset:** `knowledge_training.txt`

**What it learns:**
- Computer science and programming
- Mathematics and statistics
- Natural sciences (physics, chemistry, biology)
- Engineering and technology
- Medicine and health sciences
- Social sciences and humanities
- Business and economics

**Configuration:**
```bash
# Edit scripts/train_knowledge.sh
BATCH_SIZE=32          # Adjust based on GPU memory
LEARNING_RATE=2e-5     # Learning rate
EPOCHS=3               # Number of training epochs
MAX_LENGTH=2048        # Maximum sequence length
```

**Expected Results:**
- Comprehensive factual knowledge
- Ability to explain complex concepts
- Cross-domain knowledge integration
- Accurate information retrieval

### Response Trainer

**Purpose:** Trains the AI on reasoning, thinking, and response generation

**Dataset:** `response_training.txt`

**What it learns:**
- Step-by-step problem decomposition
- Multi-perspective analysis
- Causal and diagnostic reasoning
- Creative problem solving
- Hypothesis testing and verification
- Ethical reasoning and trade-off analysis
- Adaptive response generation
- Self-correction and refinement

**Configuration:**
```bash
# Edit scripts/train_response.sh
BATCH_SIZE=16          # Smaller batch for longer sequences
LEARNING_RATE=1e-5     # Lower learning rate for fine-tuning
EPOCHS=5               # More epochs for reasoning patterns
MAX_LENGTH=4096        # Longer sequences for reasoning chains
```

**Expected Results:**
- Explicit reasoning processes
- Novel response generation
- Context-aware adaptation
- Problem-solving capabilities
- Explanation of thinking process

### Combined Training

**Purpose:** Complete AI system with both knowledge and reasoning

**Process:**
1. Phase 1: Train on knowledge dataset (3 epochs)
2. Phase 2: Fine-tune on response dataset (5 epochs)
3. Phase 3: Joint training on both datasets (2 epochs)

**Configuration:**
```bash
# Edit scripts/train_combined.sh
# Automatically uses optimal settings for each phase
```

**Expected Results:**
- Comprehensive knowledge base
- Advanced reasoning capabilities
- Contextual response generation
- Problem-solving with domain knowledge
- Adaptive thinking strategies

---

## Hardware Requirements

### Minimum Requirements

**For Knowledge Training:**
- GPU: NVIDIA RTX 3090 (24GB) or equivalent
- RAM: 32GB system memory
- Storage: 100GB free space
- Training time: 48-72 hours

**For Response Training:**
- GPU: NVIDIA A100 (40GB) or equivalent
- RAM: 64GB system memory
- Storage: 200GB free space
- Training time: 72-96 hours

**For Combined Training:**
- GPU: NVIDIA A100 (80GB) or equivalent
- RAM: 128GB system memory
- Storage: 500GB free space
- Training time: 120-168 hours

### Recommended Cloud Platforms

**RunPod (Recommended)**
- Cost: $0.79-$2.89/hour
- GPUs: RTX 4090, A100 (40GB/80GB)
- Easy setup and management
- Website: https://runpod.io

**Vast.ai (Budget Option)**
- Cost: $0.20-$2.50/hour
- GPUs: Various options
- Marketplace model
- Website: https://vast.ai

**Lambda Labs (Enterprise)**
- Cost: $1.10-$2.49/hour
- GPUs: A100, H100
- Reliable and fast
- Website: https://lambdalabs.com

### Cost Estimates

**Knowledge Training:**
- RTX 4090: $20-40 total
- A100 (40GB): $30-60 total
- A100 (80GB): $40-80 total

**Response Training:**
- A100 (40GB): $60-120 total
- A100 (80GB): $80-160 total

**Combined Training:**
- A100 (80GB): $150-300 total

---

## Troubleshooting

### Issue: Out of Memory Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce batch size in training script
2. Reduce max sequence length
3. Enable gradient checkpointing
4. Use a GPU with more memory

**Example fix:**
```bash
# Edit training script
BATCH_SIZE=16  # Reduce from 32
MAX_LENGTH=1024  # Reduce from 2048
```

### Issue: Training Loss Not Decreasing

**Symptoms:**
- Loss stays constant or increases
- No improvement in validation metrics

**Solutions:**
1. Check learning rate (may be too high or too low)
2. Verify dataset format is correct
3. Ensure data is properly shuffled
4. Check for data quality issues

**Example fix:**
```bash
# Try different learning rates
LEARNING_RATE=5e-6  # Lower if loss explodes
LEARNING_RATE=5e-5  # Higher if loss plateaus
```

### Issue: Training Too Slow

**Symptoms:**
- Very slow iterations per second
- Training will take weeks to complete

**Solutions:**
1. Enable mixed precision training (FP16/BF16)
2. Increase batch size if memory allows
3. Use gradient accumulation
4. Check GPU utilization

**Example fix:**
```bash
# Enable mixed precision
--fp16  # or --bf16 for newer GPUs

# Gradient accumulation
GRADIENT_ACCUMULATION_STEPS=4
```

### Issue: Model Not Learning Reasoning

**Symptoms:**
- Model gives short, generic answers
- No step-by-step reasoning shown
- Responses don't adapt to context

**Solutions:**
1. Train longer on response dataset
2. Increase max sequence length
3. Adjust sampling temperature
4. Verify response dataset quality

**Example fix:**
```bash
# Train more epochs on response data
RESPONSE_EPOCHS=10  # Increase from 5

# Longer sequences for reasoning
MAX_LENGTH=4096  # Increase from 2048
```

### Issue: Checkpoint Corruption

**Symptoms:**
```
Error loading checkpoint
Unexpected EOF
```

**Solutions:**
1. Check available disk space
2. Verify checkpoint wasn't interrupted
3. Use previous checkpoint
4. Restart training from last good checkpoint

**Example fix:**
```bash
# Resume from specific checkpoint
--resume_from_checkpoint train/out/checkpoint-5000
```

---

## Advanced Configuration

### Custom Dataset Paths

Edit training scripts to use custom datasets:

```bash
# In train_knowledge.sh
KNOWLEDGE_DATA="path/to/your/knowledge_data.txt"

# In train_response.sh
RESPONSE_DATA="path/to/your/response_data.txt"
```

### Hyperparameter Tuning

**Learning Rate Schedule:**
```bash
--learning_rate 2e-5 \
--lr_scheduler_type cosine \
--warmup_steps 500
```

**Optimization:**
```bash
--optimizer adamw_torch \
--weight_decay 0.01 \
--adam_beta1 0.9 \
--adam_beta2 0.999
```

**Regularization:**
```bash
--dropout 0.1 \
--attention_dropout 0.1
```

### Multi-GPU Training

For multiple GPUs:

```bash
# Use torchrun for distributed training
torchrun --nproc_per_node=4 scripts/train_combined.py
```

### Monitoring Training

**TensorBoard:**
```bash
# Start TensorBoard
tensorboard --logdir train/out/logs

# View at http://localhost:6006
```

**Weights & Biases:**
```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
```

---

## Support Resources

### Documentation

- **Main Documentation:** `doc/README.md`
- **Dataset Documentation:** See dataset headers
- **Script Documentation:** Comments in each script

### Contact Support

**Email Support:**
- General: support@kuiperai.com
- Technical: tech@kuiperai.com
- Datasets: datasets@kuiperai.com

**Response Time:**
- Critical issues: 4-8 hours
- General inquiries: 24-48 hours

### Community Resources

**GitHub Issues:**
- Report bugs and issues
- Request features
- Share training results

**Discord Community:**
- Real-time support
- Share experiences
- Collaborate with other users

### Training Tips

1. **Start Small:** Test with small subset before full training
2. **Monitor Closely:** Watch first few hours for issues
3. **Save Checkpoints:** Enable frequent checkpointing
4. **Validate Often:** Check validation metrics regularly
5. **Document Changes:** Keep notes on configuration changes

### Best Practices

1. **Data Quality:** Verify dataset integrity before training
2. **Reproducibility:** Set random seeds for reproducible results
3. **Version Control:** Track model versions and configurations
4. **Backup Models:** Save important checkpoints externally
5. **Test Thoroughly:** Evaluate on diverse test cases

---

## Frequently Asked Questions

**Q: How long does training take?**
A: Knowledge training: 12-24 hours. Response training: 24-48 hours. Combined: 48-72 hours on A100 GPU.

**Q: Can I pause and resume training?**
A: Yes, training automatically saves checkpoints. Resume with `--resume_from_checkpoint`.

**Q: What GPU do I need?**
A: Minimum RTX 3090 (24GB). Recommended A100 (40GB or 80GB).

**Q: Can I train on CPU?**
A: Not recommended. Training would take weeks or months.

**Q: How much does cloud training cost?**
A: $150-300 for complete training on A100 (80GB).

**Q: Can I use my own data?**
A: Yes, format your data similarly to provided datasets.

**Q: How do I know if training is working?**
A: Monitor loss decreasing and validation metrics improving.

**Q: What if I run out of GPU memory?**
A: Reduce batch size, sequence length, or use gradient checkpointing.

---

**Document Version:** 1.0.0  
**Last Updated:** April 29, 2026  
**Maintained By:** KuiperAI Support Team, Moude AI LLC
