#!/usr/bin/env python3
"""
Simple OpenWebText downloader with size options
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("OpenWebText Dataset Downloader")
print("=" * 70)
print()

# Check if datasets is available
try:
    from datasets import load_dataset
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required libraries not installed")
    print()
    print("Please run:")
    print("  source .venv/bin/activate")
    print("  pip install datasets tqdm")
    print()
    sys.exit(1)

# Size options
print("Choose dataset size:")
print()
print("1. Tiny (100K docs)    - ~500MB   - 30 minutes download  - Good for testing")
print("2. Small (500K docs)   - ~2.5GB   - 2 hours download     - Quick pretraining")
print("3. Medium (1M docs)    - ~5GB     - 4 hours download     - Decent results")
print("4. Large (2M docs)     - ~10GB    - 8 hours download     - Good results")
print("5. Full (8M docs)      - ~40GB    - 24+ hours download   - Best results")
print()

choice = input("Enter choice (1-5): ").strip()

size_map = {
    "1": (100000, "tiny", "500MB"),
    "2": (500000, "small", "2.5GB"),
    "3": (1000000, "medium", "5GB"),
    "4": (2000000, "large", "10GB"),
    "5": (None, "full", "40GB")
}

if choice not in size_map:
    print("Invalid choice")
    sys.exit(1)

num_docs, size_name, size_str = size_map[choice]

print()
print(f"Downloading: {size_name.upper()} ({size_str})")
print()

# Setup directories
OUTPUT_DIR = "train/pre/data"
CACHE_DIR = ".cache/huggingface"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Check disk space
import shutil
total, used, free = shutil.disk_usage("/")
free_gb = free // (2**30)
print(f"Available disk space: {free_gb} GB")

required_gb = {"1": 1, "2": 5, "3": 10, "4": 20, "5": 80}[choice]
if free_gb < required_gb:
    print(f"⚠ WARNING: You need at least {required_gb}GB free space")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

print()
print("Downloading from HuggingFace...")
print("This may take a while depending on your internet speed")
print()

try:
    # Load dataset
    if num_docs:
        print(f"Loading {num_docs:,} documents...")
        dataset = load_dataset(
            "openwebtext",
            split=f"train[:{num_docs}]",
            cache_dir=CACHE_DIR
        )
    else:
        print("Loading full dataset (8M+ documents)...")
        dataset = load_dataset(
            "openwebtext",
            split="train",
            cache_dir=CACHE_DIR
        )
    
    print(f"✓ Loaded {len(dataset):,} documents")
    print()
    
    # Show sample
    print("Sample document:")
    print("-" * 70)
    print(dataset[0]['text'][:400])
    print("...")
    print("-" * 70)
    print()
    
    # Save to file
    output_file = f"{OUTPUT_DIR}/webtext_{size_name}.txt"
    print(f"Saving to: {output_file}")
    print("This will take a few minutes...")
    print()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Writing"):
            text = example['text'].strip()
            if text and len(text) > 50:  # Skip very short docs
                f.write(text + '\n\n')
    
    # Stats
    file_size = os.path.getsize(output_file) / (1024**3)
    
    with open(output_file, 'r') as f:
        line_count = sum(1 for _ in f)
    
    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print()
    print(f"File: {output_file}")
    print(f"Size: {file_size:.2f} GB")
    print(f"Documents: {len(dataset):,}")
    print(f"Lines: {line_count:,}")
    print()
    print("Next steps:")
    print()
    print("1. Update training script to use this file:")
    print(f"   cd knowledge/official/scripts")
    print(f"   # Edit train_knowledge.sh")
    print(f"   # Change KNOWLEDGE_DATA to: ../../../{output_file}")
    print()
    print("2. Start pretraining:")
    print("   ./train_knowledge.sh")
    print()
    print("3. Expected loss progression:")
    print("   Start: ~9.0 (random)")
    print("   After 1 epoch: ~4.5")
    print("   After 3 epochs: ~3.2")
    print()
    
except KeyboardInterrupt:
    print("\n\nDownload interrupted")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print()
    print("Troubleshooting:")
    print("  - Check internet connection")
    print("  - Try a smaller size")
    print("  - Make sure you have enough disk space")
    sys.exit(1)
