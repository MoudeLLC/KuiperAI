"""
Data preparation script for KuiperAI
Downloads and prepares training datasets
"""
import os
import sys
import argparse
from pathlib import Path

sys.path.append('..')


def create_directory_structure():
    """Create necessary directories for knowledge base"""
    directories = [
        'knowledge/datasets/nlp',
        'knowledge/datasets/vision',
        'knowledge/datasets/math',
        'knowledge/datasets/general',
        'knowledge/datasets/code',
        'knowledge/datasets/domains',
        'knowledge/benchmarks',
        'checkpoints',
        'logs',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")


def download_public_datasets():
    """
    Download public datasets for training
    
    Note: This is a placeholder. In production, you would:
    1. Download from sources like Hugging Face, Kaggle, etc.
    2. Process and clean the data
    3. Convert to appropriate format
    4. Store in knowledge base
    """
    print("\nDownloading public datasets...")
    print("Note: This is a placeholder. Implement actual downloads as needed.")
    
    # Example datasets to consider:
    datasets = [
        "Wikipedia dumps",
        "Common Crawl",
        "BookCorpus",
        "OpenWebText",
        "GitHub code repositories",
        "Stack Overflow Q&A",
        "ImageNet (for vision)",
        "COCO (for vision)",
        "Mathematical problem datasets",
    ]
    
    print("\nRecommended datasets:")
    for dataset in datasets:
        print(f"  - {dataset}")
    
    print("\nPlease download and place datasets in knowledge/datasets/")


def validate_knowledge_base():
    """Validate that knowledge base has required structure"""
    print("\nValidating knowledge base structure...")
    
    required_dirs = [
        'knowledge/datasets/nlp',
        'knowledge/datasets/general',
        'knowledge/datasets/code',
    ]
    
    all_valid = True
    for directory in required_dirs:
        if not os.path.exists(directory):
            print(f"✗ Missing directory: {directory}")
            all_valid = False
        else:
            # Check if directory has any files
            files = [f for f in os.listdir(directory) if f.endswith('.txt')]
            if files:
                print(f"✓ {directory}: {len(files)} file(s)")
            else:
                print(f"⚠ {directory}: No .txt files found")
    
    return all_valid


def create_sample_datasets():
    """Create sample datasets for testing"""
    print("\nCreating sample datasets...")
    
    # Sample datasets are already created in knowledge/datasets/
    # This function can be extended to create more samples
    
    print("✓ Sample datasets available in knowledge/datasets/")


def generate_statistics():
    """Generate statistics about the knowledge base"""
    print("\nKnowledge Base Statistics:")
    print("-" * 50)
    
    total_files = 0
    total_size = 0
    
    for root, dirs, files in os.walk('knowledge/datasets'):
        for file in files:
            if file.endswith('.txt'):
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                total_files += 1
                total_size += size
    
    print(f"Total text files: {total_files}")
    print(f"Total size: {total_size / 1024:.2f} KB")
    
    # Count by domain
    domains = ['nlp', 'vision', 'math', 'general', 'code', 'domains']
    for domain in domains:
        domain_path = f'knowledge/datasets/{domain}'
        if os.path.exists(domain_path):
            files = [f for f in os.listdir(domain_path) if f.endswith('.txt')]
            print(f"  {domain}: {len(files)} file(s)")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for KuiperAI training')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create directory structure')
    parser.add_argument('--download', action='store_true',
                       help='Download public datasets')
    parser.add_argument('--validate', action='store_true',
                       help='Validate knowledge base')
    parser.add_argument('--stats', action='store_true',
                       help='Show knowledge base statistics')
    parser.add_argument('--all', action='store_true',
                       help='Run all preparation steps')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    print("=" * 70)
    print("KuiperAI Data Preparation")
    print("=" * 70)
    
    if args.all or args.create_dirs:
        create_directory_structure()
    
    if args.all or args.download:
        download_public_datasets()
    
    if args.all:
        create_sample_datasets()
    
    if args.all or args.validate:
        validate_knowledge_base()
    
    if args.all or args.stats:
        generate_statistics()
    
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
