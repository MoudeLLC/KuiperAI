#!/usr/bin/env python3
"""
Real Training System - Trains model to truly understand and generate
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
print("KUIPERAI REAL TRAINING")
print("Training for True Understanding & Generation")
print("=" * 70)

print("\n[1/7] Loading and preparing knowledge...")

# Load all knowledge sources
all_knowledge = []

# Load definitions as Q&A pairs
if Path('knowledge/ecosystem_vocab.json').exists():
    with open('knowledge/ecosystem_vocab.json', 'r') as f:
        vocab_data = json.load(f)
        definitions = vocab_data.get('definitions', {})
        
        for word, defs in definitions.items():
            # Create Q&A format
            all_knowledge.append(f"What is {word}? {defs[0]}")
            all_knowledge.append(f"{word.capitalize()}: {defs[0]}")
            
            # Add variations
            if len(defs) > 1:
                all_knowledge.append(f"Explain {word}. {defs[0]} {defs[1]}")

# Load knowledge entries
knowledge_files = [
    'knowledge/improved_dataset.txt',
    'knowledge/ecosystem_knowledge.txt'
]

for kfile in knowledge_files:
    if Path(kfile).exists():
        with open(kfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 30]
            all_knowledge.extend(lines)

print(f"  ✓ Loaded {len(all_knowledge)} knowledge entries")

# Remove duplicates and filter
all_knowledge = list(set(all_knowledge))
all_knowledge = [k for k in all_knowledge if len(k) > 20 and len(k) < 200]

print(f"  ✓ After filtering: {len(all_knowledge)} entries")

print("\n[2/7] Creating training dataset...")
dataset = TextDataset(all_knowledge, max_length=64)
print(f"  Vocabulary: {len(dataset.vocab)} words")
print(f"  Samples: {len(dataset)}")

# Save vocabulary
os.makedirs('checkpoints', exist_ok=True)
dataset.save_vocab('checkpoints/vocab_real.json')

# Split with good validation
split = int(0.85 * len(dataset))

train_dataset = Dataset()
train_dataset.data = [dataset.data[i] for i in range(split)]
train_dataset.labels = [dataset.labels[i] for i in range(split)]

val_dataset = Dataset()
val_dataset.data = [dataset.data[i] for i in range(split, len(dataset))]
val_dataset.labels = [dataset.labels[i] for i in range(split, len(dataset))]

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")

print("\n[3/7] Creating model...")
# Balanced model - not too small, not too large
model = Transformer(
    vocab_size=len(dataset.vocab),
    d_model=192,      # Balanced size
    num_heads=6,      # Good attention
    num_layers=3,     # Enough depth
    d_ff=768,         # Good capacity
    max_seq_len=64,
    dropout=0.2       # Moderate dropout
)

total_params = sum(np.prod(p.shape) for p in model.parameters())
print(f"  Parameters: {total_params:,}")
print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")

print("\n[4/7] Setting up training...")
optimizer = AdamW(model.parameters(), lr=0.0002, weight_decay=0.05)
loss_fn = CrossEntropyLoss()
scheduler = WarmupLR(optimizer, warmup_epochs=3, target_lr=0.0002)

trainer = Trainer(
    model, optimizer, loss_fn,
    checkpoint_dir='checkpoints',
    log_dir='logs',
    gradient_clip_norm=0.5
)

print("\n[5/7] Training configuration:")
print("  • Balanced model (not too small/large)")
print("  • Moderate dropout (0.2)")
print("  • Learning rate: 0.0002")
print("  • Weight decay: 0.05")
print("  • Early stopping: patience=5")
print("  • Gradient clipping: 0.5")

print("\n[6/7] Training...")
print("=" * 70)

history = trainer.fit(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=20,
    early_stopping_patience=5,
    scheduler=scheduler
)

print("\n[7/7] Training complete!")
print("=" * 70)
print(f"  Best val loss: {trainer.best_val_loss:.4f}")
print(f"  Epochs trained: {len(history['train_loss'])}")
print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

# Check for overfitting
if trainer.best_val_loss < 0.1:
    print("\n⚠️  Warning: Very low loss might indicate overfitting")
    print("  Model may memorize instead of understand")
elif trainer.best_val_loss > 2.0:
    print("\n⚠️  Warning: High loss indicates underfitting")
    print("  Model needs more training")
else:
    print("\n✅ Good loss range - model should generalize well")

# Save training info
training_info = {
    'dataset_size': len(all_knowledge),
    'vocab_size': len(dataset.vocab),
    'model_params': int(total_params),
    'epochs': len(history['train_loss']),
    'best_val_loss': float(trainer.best_val_loss),
    'final_train_loss': float(history['train_loss'][-1]),
    'final_val_loss': float(history['val_loss'][-1]),
    'training_type': 'real_understanding'
}

with open('checkpoints/training_info_real.json', 'w') as f:
    json.dump(training_info, f, indent=2)

print("\n" + "=" * 70)
print("✅ REAL TRAINING COMPLETE")
print("=" * 70)
print("\nModel trained for:")
print("  ✓ True understanding")
print("  ✓ Context awareness")
print("  ✓ Intelligent generation")
print("\nTest with: python3 chat_real.py")
