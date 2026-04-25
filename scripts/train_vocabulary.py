#!/usr/bin/env python3
"""
Advanced BPE Vocabulary Training Script
Trains 64K-96K vocabulary with proper tokenization
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import time

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
    from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
except ImportError:
    print("ERROR: tokenizers library not installed")
    print("Install with: pip install tokenizers")
    sys.exit(1)


def collect_training_files(data_dir: Path) -> List[str]:
    """Collect all text files for training"""
    print(f"\n📂 Collecting training files from {data_dir}")
    
    text_files = []
    patterns = ['*.txt', '*.md']
    
    for pattern in patterns:
        files = list(data_dir.rglob(pattern))
        text_files.extend([str(f) for f in files])
    
    # Filter out very small files
    valid_files = []
    total_size = 0
    
    for file_path in text_files:
        size = Path(file_path).stat().st_size
        if size > 100:  # At least 100 bytes
            valid_files.append(file_path)
            total_size += size
    
    print(f"  ✓ Found {len(valid_files)} files")
    print(f"  ✓ Total size: {total_size / 1024 / 1024:.2f} MB")
    
    return valid_files


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int,
    min_frequency: int = 2,
    special_tokens: List[str] = None
) -> Tokenizer:
    """Train BPE tokenizer"""
    
    if special_tokens is None:
        special_tokens = ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<MASK>"]
    
    print(f"\n🔧 Training BPE tokenizer")
    print(f"  Target vocabulary size: {vocab_size:,}")
    print(f"  Minimum frequency: {min_frequency}")
    print(f"  Special tokens: {len(special_tokens)}")
    
    # Initialize BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    # Add normalizers (lowercase, strip accents)
    tokenizer.normalizer = Sequence([
        NFD(),
        Lowercase(),
        StripAccents()
    ])
    
    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # Decoder
    tokenizer.decoder = decoders.ByteLevel()
    
    # Post-processor for special tokens
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # Train
    print(f"\n⏳ Training on {len(files)} files...")
    start_time = time.time()
    
    tokenizer.train(files, trainer)
    
    elapsed = time.time() - start_time
    print(f"  ✓ Training completed in {elapsed:.2f}s")
    
    return tokenizer


def validate_tokenizer(tokenizer: Tokenizer, test_sentences: List[str]):
    """Validate tokenizer with test sentences"""
    print(f"\n✅ Validating tokenizer")
    
    for sentence in test_sentences:
        encoding = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoding.ids)
        
        print(f"\n  Original: {sentence}")
        print(f"  Tokens: {encoding.tokens[:10]}{'...' if len(encoding.tokens) > 10 else ''}")
        print(f"  Token count: {len(encoding.tokens)}")
        print(f"  Decoded: {decoded}")
        
        # Check for unknown tokens
        unk_count = encoding.tokens.count("<UNK>")
        if unk_count > 0:
            print(f"  ⚠ Warning: {unk_count} unknown tokens")


def export_vocabulary(tokenizer: Tokenizer, output_path: Path) -> Dict:
    """Export vocabulary to JSON format"""
    print(f"\n💾 Exporting vocabulary to {output_path}")
    
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Create metadata
    metadata = {
        "vocab_size": vocab_size,
        "algorithm": "BPE",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "special_tokens": ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<MASK>"]
    }
    
    # Save vocabulary
    vocab_data = {
        "metadata": metadata,
        "vocab": vocab
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    
    file_size = output_path.stat().st_size
    print(f"  ✓ Vocabulary size: {vocab_size:,} tokens")
    print(f"  ✓ File size: {file_size / 1024:.2f} KB")
    
    return metadata


def generate_statistics(tokenizer: Tokenizer, files: List[str]) -> Dict:
    """Generate vocabulary statistics"""
    print(f"\n📊 Generating statistics")
    
    vocab = tokenizer.get_vocab()
    
    # Sample some files for statistics
    sample_files = files[:min(10, len(files))]
    total_tokens = 0
    total_chars = 0
    
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                encoding = tokenizer.encode(text)
                total_tokens += len(encoding.tokens)
                total_chars += len(text)
        except Exception as e:
            print(f"  ⚠ Error processing {file_path}: {e}")
    
    avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    
    stats = {
        "vocabulary_size": len(vocab),
        "sample_files": len(sample_files),
        "total_tokens_sampled": total_tokens,
        "total_chars_sampled": total_chars,
        "avg_chars_per_token": round(avg_chars_per_token, 2),
        "compression_ratio": round(total_chars / total_tokens, 2) if total_tokens > 0 else 0
    }
    
    print(f"  ✓ Vocabulary size: {stats['vocabulary_size']:,}")
    print(f"  ✓ Avg chars per token: {stats['avg_chars_per_token']}")
    print(f"  ✓ Compression ratio: {stats['compression_ratio']}x")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Train BPE vocabulary')
    parser.add_argument('--vocab-size', type=int, default=80000,
                       help='Target vocabulary size (default: 80000)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory for vocabulary')
    parser.add_argument('--training-data', type=str, default='knowledge',
                       help='Directory containing training data')
    parser.add_argument('--algorithm', type=str, default='bpe',
                       choices=['bpe'], help='Tokenization algorithm')
    parser.add_argument('--min-frequency', type=int, default=2,
                       help='Minimum token frequency')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BPE VOCABULARY TRAINING")
    print("=" * 70)
    
    # Setup paths
    data_dir = Path(args.training_data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not data_dir.exists():
        print(f"❌ Error: Training data directory not found: {data_dir}")
        sys.exit(1)
    
    # Collect training files
    training_files = collect_training_files(data_dir)
    
    if len(training_files) == 0:
        print("❌ Error: No training files found")
        sys.exit(1)
    
    # Train tokenizer
    tokenizer = train_bpe_tokenizer(
        training_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )
    
    # Test sentences
    test_sentences = [
        "Hello, how are you today?",
        "Machine learning is a subset of artificial intelligence.",
        "The quick brown fox jumps over the lazy dog.",
        "KuiperAI is an advanced language model.",
        "What is the meaning of life, the universe, and everything?"
    ]
    
    validate_tokenizer(tokenizer, test_sentences)
    
    # Generate statistics
    stats = generate_statistics(tokenizer, training_files)
    
    # Export vocabulary
    vocab_size_k = args.vocab_size // 1000
    vocab_path = output_dir / f"vocab_bpe_{vocab_size_k}k.json"
    metadata = export_vocabulary(tokenizer, vocab_path)
    
    # Save tokenizer (for easy loading)
    tokenizer_path = output_dir / f"tokenizer_bpe_{vocab_size_k}k.json"
    tokenizer.save(str(tokenizer_path))
    print(f"  ✓ Tokenizer saved to {tokenizer_path}")
    
    # Save statistics
    stats_path = output_dir / f"vocab_stats_{vocab_size_k}k.json"
    with open(stats_path, 'w') as f:
        json.dump({**metadata, **stats}, f, indent=2)
    print(f"  ✓ Statistics saved to {stats_path}")
    
    print("\n" + "=" * 70)
    print("✅ VOCABULARY TRAINING COMPLETE")
    print("=" * 70)
    print(f"Vocabulary: {vocab_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Statistics: {stats_path}")


if __name__ == "__main__":
    main()
