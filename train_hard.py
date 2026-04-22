#!/usr/bin/env python3
"""
HARD Training System - Trains with massive dataset, prevents overfitting
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import os
import json
from pathlib import Path

from models.transformer import Transformer
from core.optimizers import AdamW
from core.losses import CrossEntropyLoss
from training.trainer import Trainer
from training.scheduler import WarmupLR
from data.dataset import TextDataset, DataLoader, Dataset

print("=" * 70)
print("KUIPERAI HARD TRAINING")
print("Massive Dataset + Strong Regularization")
print("=" * 70)

print("\n[1/6] Loading HARD dataset...")

# Load hard training dataset
dataset_file = 'knowledge/hard_training_dataset.txt'

if not Path(dataset_file).exists():
    print("❌ Hard dataset not found!")
    print("Run: python3 generate_hard_dataset.py")
    sys.exit(1)

with open(dataset_file, 'r') as f:
    training_data = [line.strip() for line in f if line.strip()]

print(f"  ✓ Loaded {len(training_data)} training examples")

print("\n[2/6] Creating dataset...")
dataset = TextDataset(training_data, max_length=64)
print(f"  Vocabulary: {len(dataset.vocab)} words")
print(f"  Samples: {len(dataset)}")

# Save vocabulary
os.makedirs('checkpoints', exist_ok=True)
dataset.save_vocab('checkpoints/vocab_hard.json')

# Split with MORE validation data to prevent overfitting
split = int(0.80 * len(dataset))  # 80/20 split

train_dataset = Dataset()
train_dataset.data = [dataset.data[i] for i in range(split)]
train_dataset.labels = [dataset.labels[i] for i in range(split)]

val_dataset = Dataset()
val_dataset.data = [dataset.data[i] for i in range(split, len(dataset))]
val_dataset.labels = [dataset.labels[i] for i in range(split, len(dataset))]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Larger batch
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")

print("\n[3/6] Creating HARD model...")
# Larger model for more data, but with strong regularization
model = Transformer(
    vocab_size=len(dataset.vocab),
    d_model=256,      # Larger for more data
    num_heads=8,      # More attention
    num_layers=4,     # More depth
    d_ff=1024,        # More capacity
    max_seq_len=64,
    dropout=0.4       # VERY HIGH dropout to prevent overfitting
)

total_params = sum(np.prod(p.shape) for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

print("\n[4/6] Setting up HARD training...")
optimizer = AdamW(
    model.parameters(), 
    lr=0.0001,          # Lower learning rate
    weight_decay=0.15   # VERY HIGH weight decay
)
loss_fn = CrossEntropyLoss()
scheduler = WarmupLR(optimizer, warmup_epochs=5, target_lr=0.0001)

trainer = Trainer(
    model, optimizer, loss_fn,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    gradient_clip_norm=0.3  # Strong clipping
)

print("\n[5/6] HARD training configuration:")
print("  • Large dataset: 1,311 examples")
print("  • Large model: 3.5M parameters")
print("  • VERY HIGH dropout: 0.4")
print("  • VERY HIGH weight decay: 0.15")
print("  • Low learning rate: 0.0001")
print("  • Strong gradient clipping: 0.3")
print("  • Early stopping: patience=3")
print("  • 80/20 train/val split")

print("\n[6/6] Training HARD...")
print("=" * 70)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    early_stopping_patience=3,  # Stop early if overfitting
    scheduler=scheduler
)

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"  Best val loss: {trainer.best_val_loss:.4f}")
print(f"  Epochs trained: {len(history['train_loss'])}")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

# Analyze results
train_val_gap = abs(history['train_loss'][-1] - history['val_loss'][-1])

print("\n📊 Analysis:")
if trainer.best_val_loss < 0.1:
    print("  ⚠️  Very low loss - might be overfitting")
elif trainer.best_val_loss > 2.0:
    print("  ⚠️  High loss - needs more training")
else:
    print("  ✅ Good loss range!")

if train_val_gap > 0.5:
    print("  ⚠️  Large train/val gap - overfitting detected")
else:
    print("  ✅ Small train/val gap - good generalization!")

# Save training info
training_info = {
    'dataset_size': len(training_data),
    'vocab_size': len(dataset.vocab),
    'model_params': int(total_params),
    'epochs': len(history['train_loss']),
    'best_val_loss': float(trainer.best_val_loss),
    'final_train_loss': float(history['train_loss'][-1]),
    'final_val_loss': float(history['val_loss'][-1]),
    'train_val_gap': float(train_val_gap),
    'training_type': 'hard_massive_dataset'
}

with open('checkpoints/training_info_hard.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("\n" + "=" * 70)
print("✅ HARD TRAINING COMPLETE")
print("=" * 70)
print("\nModel trained with:")
print(f"  ✓ {len(training_data)} diverse examples")
print(f"  ✓ Strong regularization")
print(f"  ✓ Early stopping")
print(f"  ✓ Quality validation")
print("\nTest with: python3 chat_real.py")
