#!/usr/bin/env python3
"""
ADVANCED Training Strategy
- Larger model (8M+ parameters)
- Different dataset each epoch (prevents overfitting)
- Curriculum learning (easy → hard)
- Adaptive learning rate (reduces when stuck)
- Saves every epoch + best model
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from models.transformer import Transformer
from core.optimizers import Adam
from core.losses import CrossEntropyLoss
import json
import random

print("=" * 70)
print("ADVANCED TRAINING - SMART STRATEGIES")
print("Larger model + Varied data + Adaptive learning")
print("=" * 70)

# Load ALL available datasets
print("\n[1/7] Loading ALL training datasets...")
datasets = []

dataset_files = [
    'knowledge/hard_training_dataset.txt',
    'knowledge/harder_training_dataset.txt', 
    'knowledge/expert_training_dataset.txt',
    'knowledge/ultimate_training_dataset.txt'
]

all_texts = []
for file in dataset_files:
    if Path(file).exists():
        with open(file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
            all_texts.extend(texts)
            print(f"  ✓ Loaded {len(texts)} from {file}")

# Remove duplicates
all_texts = list(set(all_texts))
print(f"\n  ✓ Total unique examples: {len(all_texts)}")

# Build vocabulary from ALL data
print("\n[2/7] Building comprehensive vocabulary...")
vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

for text in all_texts:
    for token in text.split():
        if token not in vocab:
            vocab[token] = len(vocab)

idx_to_token = {v: k for k, v in vocab.items()}
print(f"  ✓ Vocabulary size: {len(vocab)} tokens")

# Tokenize
print("\n[3/7] Tokenizing all texts...")
max_length = 64

def tokenize(text):
    tokens = text.split()[:max_length - 2]
    token_ids = [vocab['<SOS>']]
    
    for token in tokens:
        token_ids.append(vocab.get(token, vocab['<UNK>']))
    
    token_ids.append(vocab['<EOS>'])
    
    while len(token_ids) < max_length:
        token_ids.append(vocab['<PAD>'])
    
    return np.array(token_ids, dtype=np.int32)

tokenized_texts = [tokenize(text) for text in all_texts]
print(f"  ✓ Tokenized {len(tokenized_texts)} sequences")

# Create curriculum: easy (short) → hard (long)
print("\n[4/7] Creating curriculum (easy → hard)...")
text_lengths = [len(text.split()) for text in all_texts]
sorted_indices = np.argsort(text_lengths)

# Split into 4 difficulty levels
n = len(sorted_indices)
curriculum = {
    'easy': [tokenized_texts[i] for i in sorted_indices[:n//4]],
    'medium': [tokenized_texts[i] for i in sorted_indices[n//4:n//2]],
    'hard': [tokenized_texts[i] for i in sorted_indices[n//2:3*n//4]],
    'expert': [tokenized_texts[i] for i in sorted_indices[3*n//4:]]
}

print(f"  ✓ Easy: {len(curriculum['easy'])} examples")
print(f"  ✓ Medium: {len(curriculum['medium'])} examples")
print(f"  ✓ Hard: {len(curriculum['hard'])} examples")
print(f"  ✓ Expert: {len(curriculum['expert'])} examples")

# Initialize LARGER model
print("\n[5/7] Initializing LARGER model...")
model = Transformer(
    vocab_size=len(vocab),
    d_model=384,      # Increased from 256
    num_heads=12,     # Increased from 8
    num_layers=8,     # Increased from 6
    d_ff=1536,        # Increased from 1024
    max_seq_len=max_length,
    dropout=0.4       # Increased dropout to prevent overfitting
)

total_params = sum(p.data.size for p in model.parameters())
print(f"  ✓ Model parameters: {total_params:,}")
print(f"  ✓ That's {total_params/1e6:.1f}M parameters!")

# Adaptive optimizer
print("\n[6/7] Setting up adaptive training...")
initial_lr = 0.00003  # Lower starting LR
optimizer = Adam(model.parameters(), lr=initial_lr, weight_decay=0.1)
loss_fn = CrossEntropyLoss()

# Checkpoint directory
import os
if os.path.exists('/kaggle/working'):
    checkpoint_dir = '/kaggle/working/checkpoints'
else:
    checkpoint_dir = 'checkpoints'

Path(checkpoint_dir).mkdir(exist_ok=True)

# Training with curriculum and adaptation
print("\n[7/7] Training with SMART strategies...")
print("=" * 70)

epochs = 50
batch_size = 16
best_loss = float('inf')
patience = 5
no_improve_count = 0
current_lr = initial_lr

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 50)
    
    # Curriculum: gradually introduce harder examples
    if epoch < 10:
        level = 'easy'
    elif epoch < 20:
        level = 'medium'
    elif epoch < 35:
        level = 'hard'
    else:
        level = 'expert'
    
    # Mix in some from all levels for variety
    epoch_data = curriculum[level].copy()
    for other_level in curriculum:
        if other_level != level:
            epoch_data.extend(random.sample(curriculum[other_level], 
                                          min(50, len(curriculum[other_level]))))
    
    random.shuffle(epoch_data)
    
    print(f"  Training level: {level.upper()}")
    print(f"  Examples this epoch: {len(epoch_data)}")
    
    # Create batches
    num_batches = len(epoch_data) // batch_size
    batches = []
    for i in range(num_batches):
        batch_texts = epoch_data[i * batch_size:(i + 1) * batch_size]
        batch = np.array(batch_texts)
        batches.append(batch)
    
    epoch_losses = []
    
    for batch_idx, batch in enumerate(batches):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        outputs = model.forward(inputs)
        
        batch_size_actual, seq_len, vocab_size_actual = outputs.shape
        outputs_flat = outputs.reshape(batch_size_actual * seq_len, vocab_size_actual)
        targets_flat = targets.reshape(-1)
        
        loss = loss_fn(outputs_flat, targets_flat)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += np.sum(param.grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        if total_norm > 1.0:
            clip_coef = 1.0 / (total_norm + 1e-6)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= clip_coef
        
        optimizer.step()
        
        loss_value = float(loss.data) if hasattr(loss.data, 'item') else float(loss.data)
        epoch_losses.append(loss_value)
        
        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}/{len(batches)}, Loss: {loss_value:.4f}, LR: {current_lr:.6f}")
    
    avg_loss = np.mean(epoch_losses)
    print(f"\n  Epoch {epoch + 1} Loss: {avg_loss:.4f}")
    
    # ALWAYS save checkpoint (not just best)
    model.save(f'{checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pt')
    print(f"  ✓ Saved checkpoint for epoch {epoch + 1}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save(f'{checkpoint_dir}/advanced_best.pt')
        print(f"  ✓✓ NEW BEST MODEL! Loss: {best_loss:.4f}")
        no_improve_count = 0
    else:
        no_improve_count += 1
        print(f"  No improvement for {no_improve_count} epochs")
    
    # Adaptive learning rate: reduce if stuck
    if no_improve_count >= patience:
        current_lr *= 0.5
        optimizer.learning_rate = current_lr
        print(f"  ⚡ Reduced learning rate to {current_lr:.6f}")
        no_improve_count = 0
        
        # If LR too small, stop
        if current_lr < 1e-6:
            print("\n  Learning rate too small, stopping training")
            break

print("\n" + "=" * 70)
print("ADVANCED TRAINING COMPLETE!")
print("=" * 70)
print(f"Best Loss: {best_loss:.4f}")
print(f"Model saved to: {checkpoint_dir}/advanced_best.pt")

# Save vocabulary
with open(f'{checkpoint_dir}/vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

print(f"Vocabulary saved to: {checkpoint_dir}/vocab.json")
print("\n✓ Larger model + Smart training = SMARTER AI!")
