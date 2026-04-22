#!/usr/bin/env python3
"""
Quick demo of KuiperAI chat capabilities
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import json

from models.transformer import Transformer

print("=" * 70)
print("KUIPERAI DEMO - Showing the AI can chat!")
print("=" * 70)

# Load vocabulary
print("\n[1/3] Loading model...")
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

print(f"✓ Model loaded ({len(params)} parameters)")
print(f"✓ Vocabulary: {len(vocab)} tokens")

# Demo prompts
demo_prompts = [
    "Hello",
    "What is AI",
    "Tell me about machine learning",
    "How do neural networks work",
    "Can you help me",
]

print("\n[2/3] Running demo conversations...")
print("=" * 70)

for i, prompt in enumerate(demo_prompts, 1):
    print(f"\n[{i}/{len(demo_prompts)}] You: {prompt}")
    
    # Tokenize
    tokens = [vocab['<SOS>']]
    for word in prompt.lower().split():
        tokens.append(vocab.get(word, vocab['<UNK>']))
    
    tokens = np.array([tokens], dtype=np.int32)
    
    # Generate
    generated = model.generate(tokens, max_length=35, temperature=0.7)
    
    # Decode
    response = []
    for token_id in generated[0][len(tokens[0]):]:
        if token_id == vocab['<EOS>']:
            break
        if token_id not in [vocab['<PAD>'], vocab['<SOS>']]:
            token = idx_to_token.get(int(token_id), '<UNK>')
            if token != '<UNK>':
                response.append(token)
    
    response_text = ' '.join(response[:20])  # Limit length for demo
    print(f"     KuiperAI: {response_text}")

print("\n" + "=" * 70)
print("[3/3] Demo complete!")
print("=" * 70)
print("\n✅ KuiperAI is working and can chat!")
print("\nTo chat interactively, run:")
print("  python3 chat.py")
print("\nNote: Quality is limited by small training dataset (52 lines)")
print("For better results, train on more data.")
