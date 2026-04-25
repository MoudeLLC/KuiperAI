#!/usr/bin/env python3
"""
Download high-quality training data from public sources
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import time

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Required libraries not installed")
    print("Install with: pip install requests tqdm")
    sys.exit(1)


# Public domain / open datasets
DATASETS = {
    "wikipedia_sample": {
        "url": "https://dumps.wikimedia.org/other/cirrussearch/current/enwiki-20240101-cirrussearch-content.json.gz",
        "description": "Wikipedia articles (sample)",
        "size_mb": 100,
        "enabled": False  # Too large for automated download
    },
    "gutenberg_sample": {
        "urls": [
            "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",  # Pride and Prejudice
            "https://www.gutenberg.org/cache/epub/84/pg84.txt",      # Frankenstein
            "https://www.gutenberg.org/cache/epub/1661/pg1661.txt",  # Sherlock Holmes
            "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",  # Moby Dick
            "https://www.gutenberg.org/cache/epub/11/pg11.txt",      # Alice in Wonderland
        ],
        "description": "Classic literature from Project Gutenberg",
        "enabled": True
    },
    "common_crawl_news": {
        "description": "Common Crawl News dataset",
        "enabled": False  # Requires special handling
    }
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress bar"""
    try:
        print(f"\n📥 Downloading: {description or url}")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        file_size = output_path.stat().st_size
        print(f"  ✓ Downloaded: {file_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"  ❌ Error downloading {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def clean_gutenberg_text(text: str) -> str:
    """Clean Project Gutenberg text (remove headers/footers)"""
    lines = text.split('\n')
    
    # Find start of actual content
    start_idx = 0
    for i, line in enumerate(lines):
        if '*** START OF' in line or '***START OF' in line:
            start_idx = i + 1
            break
    
    # Find end of actual content
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if '*** END OF' in line or '***END OF' in line:
            end_idx = i
            break
    
    # Extract content
    content = '\n'.join(lines[start_idx:end_idx])
    
    # Basic cleaning
    content = content.strip()
    
    return content


def download_gutenberg_books(output_dir: Path) -> int:
    """Download books from Project Gutenberg"""
    print("\n" + "=" * 70)
    print("DOWNLOADING PROJECT GUTENBERG BOOKS")
    print("=" * 70)
    
    urls = DATASETS["gutenberg_sample"]["urls"]
    success_count = 0
    
    for i, url in enumerate(urls, 1):
        book_name = url.split('/')[-1].replace('.txt', '')
        output_path = output_dir / f"gutenberg_{book_name}.txt"
        
        if output_path.exists():
            print(f"\n✓ Already exists: {output_path.name}")
            success_count += 1
            continue
        
        # Download
        temp_path = output_dir / f"temp_{book_name}.txt"
        if download_file(url, temp_path, f"Book {i}/{len(urls)}"):
            # Clean text
            try:
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                cleaned_text = clean_gutenberg_text(text)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                temp_path.unlink()
                success_count += 1
                print(f"  ✓ Cleaned and saved: {output_path.name}")
                
            except Exception as e:
                print(f"  ❌ Error cleaning text: {e}")
                if temp_path.exists():
                    temp_path.unlink()
        
        # Be nice to the server
        time.sleep(2)
    
    return success_count


def generate_synthetic_data(output_dir: Path, count: int = 1000) -> bool:
    """Generate synthetic training data"""
    print("\n" + "=" * 70)
    print("GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 70)
    
    output_path = output_dir / "synthetic_training_data.txt"
    
    # Templates for diverse sentences
    templates = [
        "The {adj} {noun} {verb} {adverb} in the {place}.",
        "Machine learning {verb} {noun} using {adj} algorithms.",
        "Artificial intelligence can {verb} {noun} and {verb2} {noun2}.",
        "{noun} is an important concept in {field}.",
        "Understanding {concept} requires {skill} and {skill2}.",
        "The {adj} approach to {noun} involves {verb} and {verb2}.",
        "Modern {noun} systems use {adj} {noun2} for {purpose}.",
        "Deep learning models can {verb} {noun} with {adj} accuracy.",
        "{field} combines {concept} with {concept2} to {verb} {noun}.",
        "The future of {field} depends on {adj} {noun} and {noun2}.",
    ]
    
    # Word banks
    words = {
        "adj": ["advanced", "complex", "efficient", "powerful", "sophisticated", "robust", 
                "innovative", "intelligent", "adaptive", "scalable"],
        "noun": ["algorithm", "model", "system", "network", "data", "pattern", "feature",
                 "parameter", "function", "structure"],
        "noun2": ["architecture", "framework", "methodology", "technique", "approach",
                  "strategy", "mechanism", "process", "pipeline", "workflow"],
        "verb": ["analyze", "process", "optimize", "transform", "generate", "predict",
                 "classify", "detect", "extract", "learn"],
        "verb2": ["improve", "enhance", "refine", "develop", "implement", "evaluate",
                  "train", "test", "validate", "deploy"],
        "adverb": ["efficiently", "effectively", "accurately", "rapidly", "automatically",
                   "intelligently", "systematically", "dynamically", "adaptively"],
        "place": ["system", "network", "model", "framework", "environment", "domain",
                  "context", "space", "field", "application"],
        "field": ["machine learning", "artificial intelligence", "data science", 
                  "computer vision", "natural language processing", "deep learning"],
        "concept": ["neural networks", "gradient descent", "backpropagation", "attention",
                    "embeddings", "transformers", "optimization", "regularization"],
        "concept2": ["supervised learning", "unsupervised learning", "reinforcement learning",
                     "transfer learning", "meta-learning", "few-shot learning"],
        "skill": ["practice", "study", "experience", "training", "research", "analysis"],
        "skill2": ["dedication", "patience", "creativity", "persistence", "curiosity"],
        "purpose": ["classification", "prediction", "generation", "optimization",
                    "detection", "recognition", "understanding", "reasoning"]
    }
    
    import random
    sentences = []
    
    print(f"  Generating {count} sentences...")
    
    for _ in range(count):
        template = random.choice(templates)
        
        # Fill template
        sentence = template
        for key, values in words.items():
            if f"{{{key}}}" in sentence:
                sentence = sentence.replace(f"{{{key}}}", random.choice(values))
        
        sentences.append(sentence)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sentences))
    
    file_size = output_path.stat().st_size
    print(f"  ✓ Generated {count} sentences")
    print(f"  ✓ File size: {file_size / 1024:.2f} KB")
    print(f"  ✓ Saved to: {output_path}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Download training data')
    parser.add_argument('--output-dir', type=str, default='knowledge/downloaded_corpus',
                       help='Output directory')
    parser.add_argument('--min-size', type=int, default=10000000,
                       help='Minimum total size in bytes (default: 10MB)')
    parser.add_argument('--synthetic-count', type=int, default=5000,
                       help='Number of synthetic sentences to generate')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("TRAINING DATA DOWNLOADER")
    print("=" * 70)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download Gutenberg books
    gutenberg_count = download_gutenberg_books(output_dir)
    
    # Generate synthetic data
    generate_synthetic_data(output_dir, args.synthetic_count)
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in output_dir.glob('*.txt'))
    
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"  Books downloaded: {gutenberg_count}")
    print(f"  Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"  Output directory: {output_dir}")
    
    if total_size < args.min_size:
        print(f"\n⚠ Warning: Total size ({total_size / 1024 / 1024:.2f} MB) is below minimum ({args.min_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"\n✅ Success: Downloaded sufficient training data")


if __name__ == "__main__":
    main()
