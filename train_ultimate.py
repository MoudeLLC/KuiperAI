#!/usr/bin/env python3
"""
ULTIMATE Training - 3,816 examples with optimized hyperparameters
Lower learning rate + more data = stable improvement
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from models.transformer import Transformer
from core.optimizers import Adam
from core.losses import CrossEntropyLoss

print("=" * 70)
print("ULTIMATE AUTOREGRESSIVE TRAINING")
print("3,816 examples + Optimized hyperparameters")
print("=" * 70)

# Load ULTIMATE dataset
print("\n[1/6] Loading ULTIMATE training data...")
with open('knowledge/ultimate_training_dataset.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"  ✓ Loaded {len(texts)} training examples")
print(f"  ✓ That's 3x more than the original 1,311!")

# Build vocabulary
print("\n[2/6] Building vocabulary...")
vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

for text in texts:
    for token in text.split():
        if token not in vocab:
            vocab[token] = len(vocab)

idx_to_token = {v: k for k, v in vocab.items()}
print(f"  ✓ Vocabulary size: {len(vocab)} tokens")

# Tokenize
print("\n[3/6] Tokenizing texts...")
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

tokenized_texts = [tokenize(text) for text in texts]
print(f"  ✓ Tokenized {len(tokenized_texts)} sequences")

# Create batches
print("\n[4/6] Creating training batches...")
batch_size = 16
num_batches = len(tokenized_texts) // batch_size

batches = []
for i in range(num_batches):
    batch_texts = tokenized_texts[i * batch_size:(i + 1) * batch_size]
    batch = np.array(batch_texts)
    batches.append(batch)

print(f"  ✓ Created {len(batches)} batches")

# Initialize model
print("\n[5/6] Initializing model...")
model = Transformer(
    vocab_size=len(vocab),
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_len=max_length,
    dropout=0.3
)

total_params = sum(p.data.size for p in model.parameters())
print(f"  ✓ Model parameters: {total_params:,}")

# LOWER learning rate to prevent loss from increasing!
optimizer = Adam(model.parameters(), lr=0.00005, weight_decay=0.1)  # Was 0.0001, now 0.00005
loss_fn = CrossEntropyLoss()

print("  ✓ Using LOWER learning rate (0.00005) for stable training")

# Training
print("\n[6/6] Training ULTIMATE...")
print("=" * 70)

epochs = 40  # More epochs since we have more data
best_loss = float('inf')

import os
if os.path.exists('/kaggle/working'):
    checkpoint_dir = '/kaggle/working/checkpoints'
else:
    checkpoint_dir = 'checkpoints'

Path(checkpoint_dir).mkdir(exist_ok=True)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 50)
    
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
            print(f"  Batch {batch_idx}/{len(batches)}, Loss: {loss_value:.4f}")
    
    avg_loss = np.mean(epoch_losses)
    print(f"\n  Train Loss: {avg_loss:.4f}")
    
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save(f'{checkpoint_dir}/ultimate_model_best.pt')
        print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

print("\n" + "=" * 70)
print("ULTIMATE TRAINING COMPLETE!")
print("=" * 70)
print(f"Best Loss: {best_loss:.4f}")
print(f"Model saved to: {checkpoint_dir}/ultimate_model_best.pt")

# Save vocabulary
import json
with open(f'{checkpoint_dir}/vocab_ultimate.json', 'w') as f:
    json.dump(vocab, f, indent=2)

print("\n✓ 3x more data + optimized training = BEST AI!")
