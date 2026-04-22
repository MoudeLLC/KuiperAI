#!/usr/bin/env python3
"""
Test chat model on training examples
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import json

from models.transformer import Transformer

print("=" * 70)
print("TESTING KUIPERAI CHAT MODEL")
print("=" * 70)

# Load vocabulary
with open('checkpoints/vocab.json', 'r') as f:
    vocab = json.load(f)

idx_to_token = {v: k for k, v in vocab.items()}

# Load model
with open('checkpoints/best_model.json', 'r') as f:
    checkpoint = json.load(f)

model = Transformer(
    vocab_size=len(vocab),
    d_model=128,
    num_heads=4,
    num_layers=3,
    d_ff=512,
    max_seq_len=64
)

# Load weights
params = model.parameters()
for i, param in enumerate(params):
    if f'param_{i}' in checkpoint['model_state']:
        param.data = np.array(checkpoint['model_state'][f'param_{i}'], dtype=np.float32)

print("✓ Model loaded!")

# Test on some training examples
test_prompts = [
    "What is machine learning",
    "How does AI work",
    "Tell me about neural networks",
]

print("\n" + "=" * 70)
print("TESTING ON PROMPTS")
print("=" * 70)

for prompt in test_prompts:
    print(f"\nPrompt: {prompt}")
    
    # Tokenize
    tokens = [vocab['<SOS>']]
    for word in prompt.lower().split():
        tokens.append(vocab.get(word, vocab['<UNK>']))
    
    tokens = np.array([tokens], dtype=np.int32)
    
    # Generate
    generated = model.generate(tokens, max_length=40, temperature=0.5)
    
    # Decode
    response = []
    for token_id in generated[0][len(tokens[0]):]:  # Skip input tokens
        if token_id == vocab['<EOS>']:
            break
        if token_id not in [vocab['<PAD>'], vocab['<SOS>']]:
            token = idx_to_token.get(int(token_id), '<UNK>')
            if token != '<UNK>':
                response.append(token)
    
    print(f"Response: {' '.join(response)}")

print("\n" + "=" * 70)
print("✓ Model can generate text!")
print("=" * 70)
