#!/usr/bin/env python3
"""
Real Autoregressive Language Model Training
Predicts next token at each position - the RIGHT way
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from models.transformer import Transformer
from core.optimizers import Adam
from core.losses import CrossEntropyLoss

print("=" * 70)
print("AUTOREGRESSIVE LANGUAGE MODEL TRAINING")
print("Training model to predict next token - the real way")
print("=" * 70)

# Load training data
print("\n[1/6] Loading training data...")
with open('knowledge/hard_training_dataset.txt', 'r') as f:
    texts = [line.strip() for line in f if line.strip()]

print(f"  ✓ Loaded {len(texts)} training examples")

# Build vocabulary
print("\n[2/6] Building vocabulary...")
vocab = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}

for text in texts:
    for token in text.split():
        if token not in vocab:
            vocab[token] = len(vocab)

idx_to_token = {v: k for k, v in vocab.items()}
print(f"  ✓ Vocabulary size: {len(vocab)} tokens")

# Tokenize all texts
print("\n[3/6] Tokenizing texts...")
max_length = 64

def tokenize(text):
    """Convert text to token IDs"""
    tokens = text.split()[:max_length - 2]
    token_ids = [vocab['<SOS>']]
    
    for token in tokens:
        token_ids.append(vocab.get(token, vocab['<UNK>']))
    
    token_ids.append(vocab['<EOS>'])
    
    # Pad to max_length
    while len(token_ids) < max_length:
        token_ids.append(vocab['<PAD>'])
    
    return np.array(token_ids, dtype=np.int32)

tokenized_texts = [tokenize(text) for text in texts]
print(f"  ✓ Tokenized {len(tokenized_texts)} sequences")

# Create training batches
print("\n[4/6] Creating training batches...")
batch_size = 16
num_batches = len(tokenized_texts) // batch_size

batches = []
for i in range(num_batches):
    batch_texts = tokenized_texts[i * batch_size:(i + 1) * batch_size]
    batch = np.array(batch_texts)
    batches.append(batch)

print(f"  ✓ Created {len(batches)} batches of size {batch_size}")

# Initialize model
print("\n[5/6] Initializing model...")
model = Transformer(
    vocab_size=len(vocab),
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_length=max_length,
    dropout=0.3  # Moderate dropout
)

print(f"  ✓ Model parameters: {model.count_parameters():,}")

# Initialize optimizer and loss
optimizer = Adam(learning_rate=0.0001, weight_decay=0.1)
loss_fn = CrossEntropyLoss()

# Training loop
print("\n[6/6] Training HARD...")
print("=" * 70)

epochs = 30
best_loss = float('inf')

Path('checkpoints').mkdir(exist_ok=True)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 50)
    
    epoch_losses = []
    
    for batch_idx, batch in enumerate(batches):
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        inputs = batch[:, :-1]  # Shape: (batch_size, seq_len-1)
        targets = batch[:, 1:]  # Shape: (batch_size, seq_len-1)
        
        # Forward pass
        outputs = model.forward(inputs)  # Shape: (batch_size, seq_len-1, vocab_size)
        
        # Reshape for loss calculation
        # outputs: (batch_size * seq_len, vocab_size)
        # targets: (batch_size * seq_len,)
        batch_size_actual, seq_len, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)
        
        # Calculate loss (only on non-padding tokens)
        loss = loss_fn(outputs_flat, targets_flat)
        
        # Backward pass
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
        
        # Update weights
        optimizer.step(model.parameters())
        
        # Track loss
        loss_value = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        epoch_losses.append(loss_value)
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(batches)}, Loss: {loss_value:.4f}")
    
    # Epoch summary
    avg_loss = np.mean(epoch_losses)
    print(f"\n  Train Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save('checkpoints/autoregressive_best.pt')
        print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best Loss: {best_loss:.4f}")
print(f"Model saved to: checkpoints/autoregressive_best.pt")
print(f"Vocabulary saved to: checkpoints/vocab.json")

# Save vocabulary
import json
with open('checkpoints/vocab.json', 'w') as f:
    json.dump(vocab, f, indent=2)

print("\n✓ Ready for generation with chat_autoregressive.py")
