#!/usr/bin/env python3
"""
Clean and improve the training dataset for better responses
"""
import re
from pathlib import Path

print("=" * 70)
print("IMPROVING TRAINING DATASET")
print("=" * 70)

# Load current dataset
dataset_file = 'knowledge/comprehensive_dataset.txt'
with open(dataset_file, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"\nOriginal lines: {len(lines)}")

# Clean and filter
cleaned_lines = []
seen = set()

for line in lines:
    # Skip metadata lines
    if any(x in line for x in ['Source:', 'Topic:', 'Learned:', '===', 'test_topic']):
        continue
    
    # Skip very short lines
    if len(line.split()) < 5:
        continue
    
    # Skip questions without context (they confuse the model)
    if line.endswith('?') and len(line.split()) < 8:
        continue
    
    # Skip duplicates
    if line.lower() in seen:
        continue
    
    seen.add(line.lower())
    cleaned_lines.append(line)

print(f"Cleaned lines: {len(cleaned_lines)}")

# Add some simple conversational patterns
conversational_data = [
    "Hello! I'm KuiperAI, an AI assistant focused on learning and education.",
    "I can help explain concepts in machine learning, programming, and science.",
    "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
    "Neural networks are computational models inspired by the human brain.",
    "Machine learning allows computers to learn from data without explicit programming.",
    "Python is a popular programming language for data science and machine learning.",
    "Artificial intelligence aims to create systems that can perform tasks requiring human intelligence.",
    "Natural language processing helps computers understand and generate human language.",
    "Training a model involves adjusting its parameters to minimize prediction errors.",
    "Algorithms are step-by-step procedures for solving problems or performing computations.",
]

# Combine
final_lines = cleaned_lines + conversational_data

# Save improved dataset
output_file = 'knowledge/improved_dataset.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    for line in final_lines:
        f.write(line + '\n')

print(f"Final lines: {len(final_lines)}")
print(f"\nSaved to: {output_file}")
print("\n" + "=" * 70)
print("✅ DATASET IMPROVED")
print("=" * 70)
print("\nNext steps:")
print("  1. Review: knowledge/improved_dataset.txt")
print("  2. Retrain with: python3 train_improved.py")
