#!/usr/bin/env python3
"""
Train Autoregressive Language Model with BPE Tokenizer
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import json
from pathlib import Path
from models.transformer import Transformer
from core.optimizers import Adam
from core.losses import CrossEntropyLoss

try:
    from tokenizers import Tokenizer
except ImportError:
    print("ERROR: tokenizers library not installed")
    print("Install with: pip install tokenizers")
    sys.exit(1)

print("=" * 70)
print("AUTOREGRESSIVE TRAINING WITH BPE TOKENIZER")
print("=" * 70)

# Load BPE tokenizer
print("\n[1/7] Loading BPE tokenizer...")
tokenizer_path = Path('checkpoints/tokenizer_bpe_80k.json')
vocab_path = Path('checkpoints/vocab_bpe_80k.json')

if not tokenizer_path.exists():
    print(f"ERROR: Tokenizer not found: {tokenizer_path}")
    print("Run: python scripts/train_vocabulary.py")
    sys.exit(1)

tokenizer = Tokenizer.from_file(str(tokenizer_path))

# Load vocab metadata
with open(vocab_path, 'r') as f:
    vocab_data = json.load(f)

vocab_size = vocab_data['metadata']['vocab_size']
print(f"  ✓ Loaded BPE tokenizer")
print(f"  ✓ Vocabulary size: {vocab_size:,} tokens")

# Load training data
print("\n[2/7] Loading training data...")
data_files = [
    'knowledge/hard_training_dataset.txt',
    'knowledge/expert_training_dataset.txt',
    'knowledge/comprehensive_dataset.txt',
]

texts = []
for file_path in data_files:
    if Path(file_path).exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            texts.extend([line.strip() for line in f if line.strip()])

print(f"  ✓ Loaded {len(texts):,} training examples")

# Tokenize texts
print("\n[3/7] Tokenizing texts...")
max_length = 128

def tokenize_text(text):
    """Tokenize text using BPE"""
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids[:max_length]
    
    # Pad to max_length
    while len(token_ids) < max_length:
        token_ids.append(0)  # PAD token
    
    return np.array(token_ids, dtype=np.int32)

tokenized_texts = [tokenize_text(text) for text in texts[:5000]]  # Limit for faster training
print(f"  ✓ Tokenized {len(tokenized_texts):,} sequences")

# Create batches
print("\n[4/7] Creating training batches...")
batch_size = 8
num_batches = len(tokenized_texts) // batch_size

batches = []
for i in range(num_batches):
    batch_texts = tokenized_texts[i * batch_size:(i + 1) * batch_size]
    batch = np.array(batch_texts)
    batches.append(batch)

print(f"  ✓ Created {len(batches)} batches of size {batch_size}")

# Initialize model
print("\n[5/7] Initializing model...")
model = Transformer(
    vocab_size=vocab_size,
    d_model=256,
    num_heads=8,
    num_layers=4,
    d_ff=1024,
    max_seq_len=max_length,
    dropout=0.1
)

total_params = sum(p.data.size for p in model.parameters())
print(f"  ✓ Model parameters: {total_params:,}")

# Initialize optimizer
optimizer = Adam(model.parameters(), lr=0.0003, weight_decay=0.01)
loss_fn = CrossEntropyLoss()

# Training
print("\n[6/7] Training...")
print("=" * 70)

epochs = 20
best_loss = float('inf')
checkpoint_dir = Path('checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    print("-" * 50)
    
    epoch_losses = []
    
    for batch_idx, batch in enumerate(batches):
        # Autoregressive: predict next token
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        
        # Forward
        outputs = model.forward(inputs)
        
        # Reshape for loss
        batch_size_actual, seq_len, vocab_size_actual = outputs.shape
        outputs_flat = outputs.reshape(batch_size_actual * seq_len, vocab_size_actual)
        targets_flat = targets.reshape(-1)
        
        # Loss
        loss = loss_fn(outputs_flat, targets_flat)
        
        # Backward
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
        
        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(batches)}, Loss: {loss_value:.4f}")
    
    avg_loss = np.mean(epoch_losses)
    print(f"\n  Epoch Loss: {avg_loss:.4f}")
    
    # Save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        model.save('checkpoints/model_bpe_best.pt')
        print(f"  ✓ Saved best model (loss: {best_loss:.4f})")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best Loss: {best_loss:.4f}")
print(f"Model: checkpoints/model_bpe_best.pt")
print(f"Tokenizer: checkpoints/tokenizer_bpe_80k.json")

print("\n✓ Ready to chat! Run: python chat_with_bpe.py")
