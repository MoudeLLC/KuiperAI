#!/usr/bin/env python3
"""
Train with improved dataset and better configuration
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import os
from pathlib import Path

from models.transformer import Transformer
from core.optimizers import AdamW
from core.losses import CrossEntropyLoss
from training.trainer import Trainer
from training.scheduler import WarmupLR
from data.dataset import TextDataset, DataLoader
from safety.content_filter import ContentFilter

print("=" * 70)
print("TRAINING WITH IMPROVED DATASET")
print("=" * 70)

# Load improved dataset
dataset_file = 'knowledge/improved_dataset.txt'

if not Path(dataset_file).exists():
    print(f"\n❌ Run: python3 improve_dataset.py first")
    sys.exit(1)

print("\n[1/5] Loading improved dataset...")
with open(dataset_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(lines)} lines")

# Create dataset with better parameters
print("\n[2/5] Creating dataset...")
dataset = TextDataset(lines, max_length=64)
print(f"Vocabulary: {len(dataset.vocab)} words")
print(f"Samples: {len(dataset)}")

# Save vocab
os.makedirs('checkpoints', exist_ok=True)
dataset.save_vocab('checkpoints/vocab_improved.json')

# Split
split = int(0.9 * len(dataset))

from data.dataset import Dataset
train_dataset = Dataset()
train_dataset.data = [dataset.data[i] for i in range(split)]
train_dataset.labels = [dataset.labels[i] for i in range(split)]

val_dataset = Dataset()
val_dataset.data = [dataset.data[i] for i in range(split, len(dataset))]
val_dataset.labels = [dataset.labels[i] for i in range(split, len(dataset))]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")

# Create model
print("\n[3/5] Creating model...")
model = Transformer(
    vocab_size=len(dataset.vocab),
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024,
    max_seq_len=128,
    dropout=0.1
)

total_params = sum(np.prod(p.shape) for p in model.parameters())
print(f"Parameters: {total_params:,}")

# Setup training
print("\n[4/5] Setting up training...")
optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
loss_fn = CrossEntropyLoss()
scheduler = WarmupLR(optimizer, warmup_epochs=3, target_lr=0.0003)

trainer = Trainer(
    model, optimizer, loss_fn,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    gradient_clip_norm=1.0
)

# Train
print("\n[5/5] Training...")
print("=" * 70)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=25,
    early_stopping_patience=5,
    scheduler=scheduler
)

print("\n" + "=" * 70)
print("✅ TRAINING COMPLETE")
print("=" * 70)
print(f"Best val loss: {trainer.best_val_loss:.4f}")
print(f"Model saved to: checkpoints/best_model.json")

# Save info
import json
training_info = {
    'dataset': dataset_file,
    'samples': len(lines),
    'vocab_size': len(dataset.vocab),
    'model_params': int(total_params),
    'epochs': len(history['train_loss']),
    'best_val_loss': float(trainer.best_val_loss),
}

with open('checkpoints/training_info_improved.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("\nTest the model with:")
print("  python3 chat_improved.py")
