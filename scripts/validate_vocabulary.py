#!/usr/bin/env python3
"""
Validate trained vocabulary
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

try:
    from tokenizers import Tokenizer
except ImportError:
    print("ERROR: tokenizers library not installed")
    sys.exit(1)


def load_vocabulary(vocab_path: Path) -> Dict:
    """Load vocabulary from JSON"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """Load tokenizer"""
    return Tokenizer.from_file(str(tokenizer_path))


def test_coverage(tokenizer: Tokenizer, test_samples: List[str]) -> Dict:
    """Test vocabulary coverage on samples"""
    total_tokens = 0
    unk_tokens = 0
    
    for sample in test_samples:
        encoding = tokenizer.encode(sample)
        total_tokens += len(encoding.tokens)
        unk_tokens += encoding.tokens.count("<UNK>")
    
    coverage = (total_tokens - unk_tokens) / total_tokens * 100 if total_tokens > 0 else 0
    
    return {
        "total_tokens": total_tokens,
        "unknown_tokens": unk_tokens,
        "coverage_percent": round(coverage, 2)
    }


def main():
    parser = argparse.ArgumentParser(description='Validate vocabulary')
    parser.add_argument('--vocab-path', type=str, required=True,
                       help='Path to vocabulary JSON')
    parser.add_argument('--test-samples', type=int, default=100,
                       help='Number of test samples')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VOCABULARY VALIDATION")
    print("=" * 70)
    
    vocab_path = Path(args.vocab_path)
    if not vocab_path.exists():
        print(f"❌ Error: Vocabulary not found: {vocab_path}")
        sys.exit(1)
    
    # Load vocabulary
    print(f"\n📖 Loading vocabulary from {vocab_path}")
    vocab_data = load_vocabulary(vocab_path)
    
    vocab_size = vocab_data.get('metadata', {}).get('vocab_size', len(vocab_data.get('vocab', {})))
    print(f"  ✓ Vocabulary size: {vocab_size:,}")
    
    # Load tokenizer
    tokenizer_path = vocab_path.parent / vocab_path.name.replace('vocab_', 'tokenizer_')
    if tokenizer_path.exists():
        print(f"\n🔧 Loading tokenizer from {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
        
        # Test samples
        test_sentences = [
            "Hello, how are you today?",
            "Machine learning is revolutionizing artificial intelligence.",
            "The transformer architecture uses self-attention mechanisms.",
            "Natural language processing enables computers to understand human language.",
            "Deep neural networks can learn complex patterns from data.",
            "KuiperAI is an advanced language model trained on diverse data.",
            "Gradient descent optimizes model parameters during training.",
            "Tokenization is the first step in text processing pipelines.",
        ]
        
        print(f"\n✅ Testing coverage on {len(test_sentences)} samples")
        stats = test_coverage(tokenizer, test_sentences)
        
        print(f"  Total tokens: {stats['total_tokens']}")
        print(f"  Unknown tokens: {stats['unknown_tokens']}")
        print(f"  Coverage: {stats['coverage_percent']}%")
        
        if stats['coverage_percent'] >= 95:
            print("\n✅ VALIDATION PASSED: Excellent coverage")
        elif stats['coverage_percent'] >= 90:
            print("\n⚠ VALIDATION WARNING: Good coverage but could be improved")
        else:
            print("\n❌ VALIDATION FAILED: Poor coverage")
            sys.exit(1)
    else:
        print(f"\n⚠ Tokenizer not found: {tokenizer_path}")
        print("  Skipping coverage test")
    
    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
