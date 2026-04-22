# Getting Started with KuiperAI

This guide will help you get KuiperAI up and running, from installation to training your first model.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- 8GB+ RAM recommended
- GPU optional (CPU training supported)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/KuiperAI.git
cd KuiperAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

```bash
python scripts/prepare_data.py --all
```

This will:
- Create necessary directory structure
- Validate the knowledge base
- Show statistics about available data

## Quick Start: Training Your First Model

### Step 1: Verify Installation

```bash
python -c "import sys; sys.path.append('src'); from core.tensor import Tensor; print('✓ Installation successful!')"
```

### Step 2: Run Tests

```bash
cd tests
pytest test_core.py -v
```

### Step 3: Train a Small Model

```bash
python scripts/train.py --config configs/transformer_base.yaml
```

This will:
- Load training data from the knowledge base
- Initialize a Transformer model
- Train for the specified number of epochs
- Save checkpoints to `checkpoints/`
- Log training metrics to `logs/`

### Step 4: Run Inference

```bash
python scripts/inference.py \
    --model checkpoints/best_model.json \
    --vocab knowledge/vocab.json \
    --input "Artificial intelligence is"
```

## Understanding the Configuration

The `configs/transformer_base.yaml` file controls all training parameters:

```yaml
model:
  type: transformer        # Model architecture
  d_model: 512            # Model dimension
  num_heads: 8            # Number of attention heads
  num_layers: 6           # Number of transformer blocks
  
training:
  batch_size: 32          # Samples per batch
  epochs: 50              # Training epochs
  lr: 0.0001             # Learning rate
```

### Adjusting for Your Hardware

**Low Memory (< 8GB RAM):**
```yaml
model:
  d_model: 256
  num_layers: 4
training:
  batch_size: 16
```

**High Memory (16GB+ RAM):**
```yaml
model:
  d_model: 768
  num_layers: 12
training:
  batch_size: 64
```

## Training on Your Own Data

### 1. Prepare Your Dataset

Create a text file in the appropriate domain:

```bash
echo "Your training text here..." > knowledge/datasets/custom/my_data.txt
```

### 2. Update Configuration

Edit `configs/transformer_base.yaml`:

```yaml
data:
  domains:
    - custom  # Add your domain
```

### 3. Train

```bash
python scripts/train.py --config configs/transformer_base.yaml
```

## Monitoring Training

### View Training Progress

Training metrics are printed to console:

```
Epoch 1/50
--------------------------------------------------
Batch 0/100, Loss: 4.2341
Batch 10/100, Loss: 3.8765
...
Train Loss: 3.5432
Val Loss: 3.6789
✓ New best model saved!
```

### Check Logs

Training history is saved to `logs/`:

```bash
cat logs/history_*.json
```

### TensorBoard (Optional)

```bash
tensorboard --logdir logs/
```

## Using the API Server

### Start the Server

```bash
python src/deployment/api_server.py
```

The server will start on `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Example API Calls

**Generate Text:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Artificial intelligence is",
    "max_length": 50,
    "temperature": 0.8
  }'
```

**Classify Text:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a great product!"
  }'
```

## Common Issues and Solutions

### Issue: Out of Memory

**Solution:** Reduce batch size or model size in config:
```yaml
training:
  batch_size: 8  # Reduce from 32
model:
  d_model: 256   # Reduce from 512
```

### Issue: Training is Slow

**Solutions:**
1. Reduce sequence length: `max_seq_len: 256`
2. Use fewer layers: `num_layers: 4`
3. Enable GPU acceleration (if available)

### Issue: Model Not Learning

**Solutions:**
1. Check learning rate (try 0.001 or 0.0001)
2. Verify data quality
3. Increase model capacity
4. Train for more epochs

### Issue: Vocabulary Not Found

**Solution:** Build vocabulary first:
```python
from src.data.dataset import TextDataset
dataset = TextDataset(texts)
dataset.save_vocab('knowledge/vocab.json')
```

## Next Steps

### Experiment with Hyperparameters

Try different configurations:
- Learning rates: 0.0001, 0.001, 0.01
- Model sizes: small (256), medium (512), large (768)
- Dropout rates: 0.0, 0.1, 0.2

### Add More Training Data

The more diverse, high-quality data you have, the better your model will perform.

### Fine-tune on Specific Tasks

1. Train a base model on general data
2. Fine-tune on task-specific data
3. Evaluate on test set

### Deploy to Production

1. Optimize model (quantization, pruning)
2. Set up monitoring and logging
3. Implement A/B testing
4. Scale with load balancing

## Learning Resources

### Understanding the Code

- Read `docs/ARCHITECTURE.md` for system design
- Explore `src/core/` for fundamental building blocks
- Study `src/models/transformer.py` for architecture details

### Improving Your Model

- Experiment with different architectures
- Try various optimization techniques
- Implement custom loss functions
- Add regularization methods

### Contributing

See `CONTRIBUTING.md` for guidelines on contributing to KuiperAI.

## Getting Help

- Check documentation in `docs/`
- Review example notebooks in `notebooks/`
- Open an issue on GitHub
- Join our community discussions

## Summary

You now have:
- ✓ Installed KuiperAI
- ✓ Prepared training data
- ✓ Trained your first model
- ✓ Run inference
- ✓ Started the API server

Happy training! 🚀
