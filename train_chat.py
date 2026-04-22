#!/usr/bin/env python3
"""
Quick training script for chat model
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import os

from models.transformer import Transformer
from core.optimizers import AdamW
from core.losses import CrossEntropyLoss
from training.trainer import Trainer
from training.scheduler import WarmupLR
from data.dataset import TextDataset, DataLoader, Dataset

print("=" * 70)
print("TRAINING KUIPERAI CHAT MODEL")
print("=" * 70)

# Load chat data
print("\n[1/6] Loading chat data...")
with open('knowledge/datasets/chat/conversations.txt', 'r') as f:
    text = f.read()

# Simple preprocessing
lines = [line.strip() for line in text.split('\n') if line.strip()]
print(f"Loaded {len(lines)} lines")

# Create dataset
print("\n[2/6] Creating dataset...")
dataset = TextDataset(lines, max_length=32)  # Smaller for faster training
print(f"Vocabulary size: {len(dataset.vocab)}")
print(f"Dataset size: {len(dataset)}")

# Save vocabulary
os.makedirs('checkpoints', exist_ok=True)
dataset.save_vocab('checkpoints/vocab.json')
print("Saved vocabulary to checkpoints/vocab.json")

# Split train/val - use indices to avoid re-tokenization
split = int(0.9 * len(dataset))
indices = list(range(len(dataset)))

# Create train dataset
train_dataset = Dataset()
train_dataset.data = [dataset.data[i] for i in indices[:split]]
train_dataset.labels = [dataset.labels[i] for i in indices[:split]]

# Create val dataset
val_dataset = Dataset()
val_dataset.data = [dataset.data[i] for i in indices[split:]]
val_dataset.labels = [dataset.labels[i] for i in indices[split:]]

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

print(f"Train: {len(train_dataset)} samples")
print(f"Val: {len(val_dataset)} samples")

# Create model
print("\n[3/6] Creating model...")
model = Transformer(
    vocab_size=len(dataset.vocab),
    d_model=128,
    num_heads=4,
    num_layers=3,
    d_ff=512,
    max_seq_len=64,  # Larger than dataset max_length
    dropout=0.1
)

total_params = sum(np.prod(p.shape) for p in model.parameters())
print(f"Model parameters: {total_params:,}")

# Create optimizer and loss
print("\n[4/6] Setting up training...")
optimizer = AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
loss_fn = CrossEntropyLoss()
scheduler = WarmupLR(optimizer, warmup_epochs=2, target_lr=0.0003)

# Create trainer
trainer = Trainer(
    model, optimizer, loss_fn,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    gradient_clip_norm=1.0
)

# Train
print("\n[5/6] Training...")
print("=" * 70)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    early_stopping_patience=5,
    scheduler=scheduler
)

print("\n[6/6] Training complete!")
print("=" * 70)
print(f"Best validation loss: {trainer.best_val_loss:.4f}")
print(f"Model saved to: checkpoints/best_model.json")
print("\nYou can now chat with the model using:")
print("  python chat.py")
