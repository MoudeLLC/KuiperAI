#!/usr/bin/env python3
"""
Real Autoregressive Chat System
Generates text one token at a time - the RIGHT way
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import json
from pathlib import Path
from models.transformer import Transformer

print("=" * 70)
print("KUIPERAI - AUTOREGRESSIVE GENERATION")
print("Real next-token prediction")
print("=" * 70)

# Load vocabulary
print("\n🔤 Loading vocabulary...")
vocab_path = Path('checkpoints/vocab.json')
if not vocab_path.exists():
    print("  ⚠ No vocabulary found!")
    print("  Run train_autoregressive.py first")
    sys.exit(1)

with open(vocab_path, 'r') as f:
    vocab = json.load(f)

idx_to_token = {v: k for k, v in vocab.items()}
print(f"  ✓ Vocabulary: {len(vocab)} tokens")

# Load model
print("\n🧠 Loading trained model...")
model = Transformer(
    vocab_size=len(vocab),
    d_model=256,
    num_heads=8,
    num_layers=6,
    d_ff=1024,
    max_seq_len=64,
    dropout=0.3
)

# Try advanced model first, fallback to autoregressive
checkpoint_path = Path('checkpoints/advanced_best.pt.npz')
if not checkpoint_path.exists():
    checkpoint_path = Path('checkpoints/autoregressive_best.pt.npz')

if checkpoint_path.exists():
    model_name = 'advanced_best.pt' if 'advanced' in str(checkpoint_path) else 'autoregressive_best.pt'
    model.load(f'checkpoints/{model_name}')
    total_params = sum(p.data.size for p in model.parameters())
    print(f"  ✓ Model loaded!")
    print(f"  ✓ Parameters: {total_params:,}")
else:
    print(f"  ⚠ No checkpoint found!")
    print("  Run train_autoregressive.py first")
    sys.exit(1)

def tokenize(text):
    """Convert text to token IDs"""
    tokens = text.lower().split()
    token_ids = [vocab['<SOS>']]
    
    for token in tokens:
        token_ids.append(vocab.get(token, vocab['<UNK>']))
    
    return np.array(token_ids, dtype=np.int32)

def generate_text(prompt, max_new_tokens=50, temperature=0.8, top_k=40):
    """
    Generate text autoregressively
    
    Args:
        prompt: Input text
        max_new_tokens: Maximum tokens to generate
        temperature: Controls randomness (higher = more random)
        top_k: Only sample from top K most likely tokens
    """
    # Tokenize prompt
    tokens = tokenize(prompt).tolist()
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Prepare input (last 63 tokens max, leave room for generation)
        input_tokens = tokens[-63:]
        input_array = np.array([input_tokens], dtype=np.int32)
        
        # Get model predictions
        output = model.forward(input_array)  # (1, seq_len, vocab_size)
        
        # Get logits for next token (last position)
        next_token_logits = output.data[0, -1, :]  # (vocab_size,)
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Convert to probabilities
        exp_logits = np.exp(next_token_logits - np.max(next_token_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Top-k sampling
        if top_k > 0:
            # Get top k indices
            top_k_indices = np.argsort(probs)[-top_k:]
            top_k_probs = probs[top_k_indices]
            
            # Renormalize
            top_k_probs = top_k_probs / np.sum(top_k_probs)
            
            # Sample from top k
            next_token = np.random.choice(top_k_indices, p=top_k_probs)
        else:
            # Sample from full distribution
            next_token = np.random.choice(len(vocab), p=probs)
        
        # Stop if EOS
        if next_token == vocab['<EOS>']:
            break
        
        # Skip special tokens
        if next_token in [vocab['<PAD>'], vocab['<SOS>']]:
            continue
        
        # Add to sequence
        tokens.append(int(next_token))
    
    # Decode tokens (skip <SOS>)
    generated_tokens = tokens[1:]  # Skip <SOS>
    text = ' '.join([idx_to_token[t] for t in generated_tokens 
                     if t not in [vocab['<PAD>'], vocab['<SOS>'], vocab['<EOS>']]])
    
    return text

print("\n" + "=" * 70)
print("✅ System ready!")
print("=" * 70)
print("Commands:")
print("  quit - Exit")
print("  temp <value> - Set temperature (0.1-2.0, default 0.8)")
print("  topk <value> - Set top-k (1-100, default 40)")
print("=" * 70)

# Generation settings
temperature = 0.8
top_k = 40

# Chat loop
conversation_count = 0

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        # Handle commands
        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                temperature = max(0.1, min(2.0, temperature))
                print(f"  ✓ Temperature set to {temperature}")
                continue
            except:
                print("  ⚠ Usage: temp <value>")
                continue
        
        if user_input.lower().startswith('topk '):
            try:
                top_k = int(user_input.split()[1])
                top_k = max(1, min(100, top_k))
                print(f"  ✓ Top-k set to {top_k}")
                continue
            except:
                print("  ⚠ Usage: topk <value>")
                continue
        
        if not user_input:
            continue
        
        # Generate response
        print("\n🤖 Generating", end='', flush=True)
        for _ in range(3):
            import time
            time.sleep(0.15)
            print(".", end='', flush=True)
        print("\n")
        
        response = generate_text(
            user_input, 
            max_new_tokens=50,
            temperature=temperature,
            top_k=top_k
        )
        
        print(f"KuiperAI: {response}")
        
        conversation_count += 1
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\nConversations: {conversation_count}")
print("Thank you for using KuiperAI!")
