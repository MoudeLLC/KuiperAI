#!/usr/bin/env python3
"""
Chat with KuiperAI
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import json

from models.transformer import Transformer
from data.dataset import TextDataset

print("=" * 70)
print("KUIPERAI CHAT")
print("=" * 70)

# Load vocabulary
print("\nLoading model...")
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
print("\nType 'quit' to exit")
print("=" * 70)

def generate_response(prompt, max_length=50, temperature=0.8):
    """Generate response to prompt"""
    # Tokenize
    tokens = [vocab.get(word, vocab.get('<UNK>', 1)) for word in prompt.lower().split()]
    if not tokens:
        tokens = [vocab.get('<SOS>', 2)]
    
    # Add SOS if needed
    if tokens[0] != vocab.get('<SOS>', 2):
        tokens = [vocab.get('<SOS>', 2)] + tokens
    
    # Limit length
    tokens = tokens[:30]
    tokens = np.array([tokens], dtype=np.int32)
    
    # Generate
    generated = model.generate(tokens, max_length=max_length, temperature=temperature)
    
    # Decode
    response_tokens = []
    for token_id in generated[0]:
        if token_id == vocab.get('<EOS>', 3):
            break
        if token_id not in [vocab.get('<PAD>', 0), vocab.get('<SOS>', 2)]:
            token = idx_to_token.get(int(token_id), '<UNK>')
            if token != '<UNK>':
                response_tokens.append(token)
    
    return ' '.join(response_tokens)

# Chat loop
while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        response = generate_response(user_input)
        print(f"KuiperAI: {response}")
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
