#!/usr/bin/env python3
"""
Fixed Training System - Prevents overfitting, better generalization
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
print("KUIPERAI FIXED TRAINING")
print("Better Generalization & Less Overfitting")
print("=" * 70)

# Load all knowledge
print("\n[1/6] Loading knowledge...")
all_knowledge = []

sources = [
    'knowledge/improved_dataset.txt',
    'knowledge/ecosystem_knowledge.txt',
    'knowledge/comprehensive_dataset.txt'
]

for source in sources:
    if Path(source).exists():
        with open(source, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 20]
            all_knowledge.extend(lines)
            print(f"  ✓ {len(lines)} from {source}")

# Load definitions and create training sentences
if Path('knowledge/ecosystem_vocab.json').exists():
    with open('knowledge/ecosystem_vocab.json', 'r') as f:
        vocab_data = json.load(f)
        definitions = vocab_data.get('definitions', {})
        
        # Add definitions as training data
        for word, defs in definitions.items():
            for definition in defs:
                if len(definition) > 20:
                    all_knowledge.append(f"{word.capitalize()}: {definition}")

print(f"Total knowledge: {len(all_knowledge)}")

# Remove duplicates
all_knowledge = list(set(all_knowledge))
print(f"After deduplication: {len(all_knowledge)}")

print("\n[2/6] Creating dataset...")
dataset = TextDataset(all_knowledge, max_length=64)
print(f"Vocabulary: {len(dataset.vocab)} words")
print(f"Samples: {len(dataset)}")

# Save vocabulary
os.makedirs('checkpoints', exist_ok=True)
dataset.save_vocab('checkpoints/vocab_fixed.json')

# Split with more validation data
split = int(0.85 * len(dataset))  # More validation data

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

print("\n[3/6] Creating model...")
# Smaller model to prevent overfitting
model = Transformer(
    vocab_size=len(dataset.vocab),
    d_model=128,      # Smaller
    num_heads=4,      # Fewer heads
    num_layers=2,     # Fewer layers
    d_ff=512,         # Smaller feed-forward
    max_seq_len=64,
    dropout=0.3       # Higher dropout
)

total_params = sum(np.prod(p.shape) for p in model.parameters())
print(f"Parameters: {total_params:,}")

print("\n[4/6] Setting up training...")
optimizer = AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)  # Lower LR, higher weight decay
loss_fn = CrossEntropyLoss()
scheduler = WarmupLR(optimizer, warmup_epochs=2, target_lr=0.0001)

trainer = Trainer(
    model, optimizer, loss_fn,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    gradient_clip_norm=0.5  # Stronger clipping
)

print("\n[5/6] Training...")
print("=" * 70)
print("Configuration:")
print("  • Smaller model (prevents overfitting)")
print("  • Higher dropout (0.3)")
print("  • Lower learning rate (0.0001)")
print("  • Higher weight decay (0.1)")
print("  • Early stopping (patience=3)")
print("=" * 70)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=15,  # Fewer epochs
    early_stopping_patience=3,  # Stop early
    scheduler=scheduler
)

print("\n[6/6] Training complete!")
print("=" * 70)
print(f"Best val loss: {trainer.best_val_loss:.4f}")
print(f"Epochs trained: {len(history['train_loss'])}")

# Save training info
training_info = {
    'dataset_size': len(all_knowledge),
    'vocab_size': len(dataset.vocab),
    'model_params': int(total_params),
    'epochs': len(history['train_loss']),
    'best_val_loss': float(trainer.best_val_loss),
    'training_type': 'fixed_no_overfit'
}

with open('checkpoints/training_info_fixed.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("\n" + "=" * 70)
print("✅ FIXED TRAINING COMPLETE")
print("=" * 70)
print("\nImprovements:")
print("  ✓ Smaller model (less memorization)")
print("  ✓ Higher dropout (better generalization)")
print("  ✓ Early stopping (prevents overfitting)")
print("  ✓ More validation data")
print("\nTest with: python3 chat_hybrid.py")
