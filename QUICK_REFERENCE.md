# KuiperAI Quick Reference

## Installation & Setup

```bash
# Clone and install
git clone https://github.com/yourusername/KuiperAI.git
cd KuiperAI
pip install -r requirements.txt

# Prepare data
python scripts/prepare_data.py --all
```

## Training

```bash
# Train with default config
python scripts/train.py --config configs/transformer_base.yaml

# Resume from checkpoint
python scripts/train.py --config configs/transformer_base.yaml --resume checkpoints/checkpoint_epoch_10.json
```

## Inference

```bash
# Interactive mode
python scripts/inference.py --model checkpoints/best_model.json --vocab knowledge/vocab.json

# Single inference
python scripts/inference.py \
    --model checkpoints/best_model.json \
    --vocab knowledge/vocab.json \
    --input "Your prompt here" \
    --max-length 100 \
    --temperature 0.8
```

## API Server

```bash
# Start server
python src/deployment/api_server.py

# API docs at: http://localhost:8000/docs
```

## API Examples

### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Artificial intelligence is",
    "max_length": 50,
    "temperature": 0.8
  }'
```

### Classify Text
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "This is amazing!"}'
```

### Get Embedding
```bash
curl -X POST "http://localhost:8000/embed?text=Hello%20world"
```

## Python API

### Basic Usage
```python
from src.models.transformer import Transformer
from src.core.optimizers import AdamW
from src.core.losses import CrossEntropyLoss
from src.training.trainer import Trainer

# Create model
model = Transformer(vocab_size=10000, d_model=512, num_heads=8, num_layers=6)

# Setup training
optimizer = AdamW(model.parameters(), lr=0.0001)
loss_fn = CrossEntropyLoss()
trainer = Trainer(model, optimizer, loss_fn)

# Train
history = trainer.fit(train_loader, val_loader, epochs=50)
```

### Custom Tensor Operations
```python
from src.core.tensor import Tensor

# Create tensors
a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)

# Operations
c = a + b
d = a * b
e = a @ b.reshape(3, 1)

# Backpropagation
e.backward()
print(a.grad)  # Gradients
```

### Data Loading
```python
from src.data.dataset import TextDataset, DataLoader

# Create dataset
texts = ["Sample text 1", "Sample text 2"]
dataset = TextDataset(texts)

# Create loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate
for batch_data, batch_labels in loader:
    # Training code here
    pass
```

## Configuration

### Model Config (configs/transformer_base.yaml)
```yaml
model:
  type: transformer
  d_model: 512        # Model dimension
  num_heads: 8        # Attention heads
  num_layers: 6       # Transformer blocks
  d_ff: 2048         # FFN dimension
  max_seq_len: 512   # Max sequence length
  dropout: 0.1       # Dropout rate

training:
  batch_size: 32
  epochs: 50
  optimizer:
    type: adamw
    lr: 0.0001
    weight_decay: 0.01
```

## Testing

```bash
# Run all tests
cd tests
pytest test_core.py -v

# Run specific test
pytest test_core.py::TestTensor::test_backward -v

# With coverage
pytest test_core.py --cov=src --cov-report=html
```

## Common Tasks

### Add New Training Data
```bash
# Create text file
echo "Your training data" > knowledge/datasets/custom/data.txt

# Update config
# Add 'custom' to domains list in configs/transformer_base.yaml
```

### Adjust Model Size
```yaml
# Small model (low memory)
model:
  d_model: 256
  num_layers: 4
  
# Large model (high memory)
model:
  d_model: 768
  num_layers: 12
```

### Change Learning Rate
```yaml
training:
  optimizer:
    lr: 0.001  # Higher for faster learning
    # or
    lr: 0.00001  # Lower for fine-tuning
```

## Troubleshooting

### Out of Memory
```yaml
# Reduce batch size
training:
  batch_size: 8

# Reduce model size
model:
  d_model: 256
  num_layers: 4
```

### Model Not Learning
```yaml
# Increase learning rate
training:
  optimizer:
    lr: 0.001

# Train longer
training:
  epochs: 100
```

### Slow Training
```yaml
# Reduce sequence length
model:
  max_seq_len: 256

# Increase batch size (if memory allows)
training:
  batch_size: 64
```

## File Locations

| Component | Location |
|-----------|----------|
| Models | `src/models/` |
| Core primitives | `src/core/` |
| Training code | `src/training/` |
| Data pipeline | `src/data/` |
| API server | `src/deployment/` |
| Configs | `configs/` |
| Scripts | `scripts/` |
| Tests | `tests/` |
| Docs | `docs/` |
| Knowledge base | `knowledge/` |
| Checkpoints | `checkpoints/` (generated) |
| Logs | `logs/` (generated) |

## Key Classes

| Class | Purpose | Location |
|-------|---------|----------|
| `Tensor` | Autograd tensor | `src/core/tensor.py` |
| `Linear` | Fully connected layer | `src/core/layers.py` |
| `Transformer` | Transformer model | `src/models/transformer.py` |
| `Trainer` | Training loop | `src/training/trainer.py` |
| `TextDataset` | Text data handling | `src/data/dataset.py` |
| `AdamW` | Optimizer | `src/core/optimizers.py` |

## Performance Tips

1. **Use appropriate batch size**: Balance memory and speed
2. **Enable gradient accumulation**: For large effective batch sizes
3. **Use mixed precision**: If GPU available (future enhancement)
4. **Profile your code**: Identify bottlenecks
5. **Vectorize operations**: Avoid Python loops
6. **Cache preprocessed data**: Speed up data loading

## Best Practices

1. **Always validate data**: Check quality before training
2. **Monitor metrics**: Track loss and accuracy
3. **Save checkpoints**: Regular backups during training
4. **Use version control**: Track experiments
5. **Document changes**: Keep notes on modifications
6. **Test incrementally**: Verify each component
7. **Start small**: Test with small model/data first

## Resources

- **Documentation**: `docs/`
- **Examples**: `notebooks/` (create as needed)
- **Tests**: `tests/`
- **API Docs**: http://localhost:8000/docs (when server running)

## Quick Commands Cheat Sheet

```bash
# Setup
pip install -r requirements.txt
python scripts/prepare_data.py --all

# Train
python scripts/train.py --config configs/transformer_base.yaml

# Inference
python scripts/inference.py --model checkpoints/best_model.json --vocab knowledge/vocab.json

# API
python src/deployment/api_server.py

# Test
cd tests && pytest test_core.py -v

# Stats
python scripts/prepare_data.py --stats
```

## Support

- Read documentation in `docs/`
- Check `PROJECT_SUMMARY.md` for overview
- Review `ARCHITECTURE.md` for technical details
- See `GETTING_STARTED.md` for detailed setup

---

**Quick Tip**: Start with the sample data and small model to verify everything works, then scale up!
