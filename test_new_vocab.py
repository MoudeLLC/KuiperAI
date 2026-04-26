#!/usr/bin/env python3
"""
Test the new BPE vocabulary
"""
import json
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    print("ERROR: tokenizers not installed")
    print("Run: pip install tokenizers")
    exit(1)

print("=" * 70)
print("TESTING NEW 80K BPE VOCABULARY")
print("=" * 70)

# Load vocabulary
vocab_path = Path('checkpoints/vocab_bpe_80k.json')
tokenizer_path = Path('checkpoints/tokenizer_bpe_80k.json')

if not vocab_path.exists():
    print(f"ERROR: Vocabulary not found: {vocab_path}")
    exit(1)

if not tokenizer_path.exists():
    print(f"ERROR: Tokenizer not found: {tokenizer_path}")
    exit(1)

# Load vocab data
with open(vocab_path, 'r') as f:
    vocab_data = json.load(f)

vocab = vocab_data['vocab']
print(f"\n✓ Vocabulary size: {len(vocab):,} tokens")

# Load tokenizer
tokenizer = Tokenizer.from_file(str(tokenizer_path))
print(f"✓ Tokenizer loaded")

# Test sentences
test_sentences = [
    "Hello, how are you today?",
    "Machine learning is revolutionizing artificial intelligence.",
    "The transformer architecture uses self-attention mechanisms.",
    "KuiperAI is an advanced language model trained on diverse data.",
    "What is the meaning of life, the universe, and everything?",
    "Deep neural networks can learn complex patterns from data.",
    "Natural language processing enables computers to understand human language.",
]

print("\n" + "=" * 70)
print("TOKENIZATION TESTS")
print("=" * 70)

total_tokens = 0
total_chars = 0
unk_count = 0

for i, sentence in enumerate(test_sentences, 1):
    encoding = tokenizer.encode(sentence)
    tokens = encoding.tokens
    
    # Count unknowns
    unk_in_sentence = tokens.count("<UNK>")
    unk_count += unk_in_sentence
    
    total_tokens += len(tokens)
    total_chars += len(sentence)
    
    print(f"\n{i}. {sentence}")
    print(f"   Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
    if unk_in_sentence > 0:
        print(f"   ⚠️  Unknown tokens: {unk_in_sentence}")
    
    # Decode
    decoded = tokenizer.decode(encoding.ids)
    if decoded.strip() != sentence.strip():
        print(f"   ⚠️  Decode mismatch!")
        print(f"   Original: {sentence}")
        print(f"   Decoded:  {decoded}")

print("\n" + "=" * 70)
print("STATISTICS")
print("=" * 70)

coverage = (total_tokens - unk_count) / total_tokens * 100 if total_tokens > 0 else 0
compression = total_chars / total_tokens if total_tokens > 0 else 0

print(f"Total tokens: {total_tokens}")
print(f"Total characters: {total_chars}")
print(f"Unknown tokens: {unk_count}")
print(f"Coverage: {coverage:.2f}%")
print(f"Compression ratio: {compression:.2f}x")
print(f"Avg chars per token: {compression:.2f}")

print("\n" + "=" * 70)
if unk_count == 0:
    print("✅ PERFECT! No unknown tokens!")
elif coverage >= 95:
    print("✅ EXCELLENT! Coverage > 95%")
elif coverage >= 90:
    print("⚠️  GOOD! Coverage > 90%")
else:
    print("❌ POOR! Coverage < 90%")
print("=" * 70)

print("\n" + "=" * 70)
print("COMPARISON WITH OLD VOCABULARY")
print("=" * 70)

old_vocab_path = Path('checkpoints/vocab.json')
if old_vocab_path.exists():
    with open(old_vocab_path, 'r') as f:
        old_vocab = json.load(f)
    
    print(f"Old vocabulary: {len(old_vocab):,} tokens")
    print(f"New vocabulary: {len(vocab):,} tokens")
    print(f"Improvement: {len(vocab) / len(old_vocab):.1f}x larger")
else:
    print("Old vocabulary not found for comparison")

print("\n✅ Vocabulary test complete!")
