#!/usr/bin/env python3
"""
Generate HARD training dataset with massive variety
Creates thousands of training examples with real understanding
"""
import json
import random
from pathlib import Path

print("=" * 70)
print("GENERATING HARD TRAINING DATASET")
print("Creating thousands of diverse examples")
print("=" * 70)

# Load definitions
definitions = {}
if Path('knowledge/ecosystem_vocab.json').exists():
    with open('knowledge/ecosystem_vocab.json', 'r') as f:
        vocab_data = json.load(f)
        definitions = vocab_data.get('definitions', {})

print(f"\n✓ Loaded {len(definitions)} definitions")

# Templates for generating diverse training data
question_templates = [
    "What is {word}?",
    "Define {word}.",
    "Explain {word}.",
    "Tell me about {word}.",
    "What does {word} mean?",
    "Can you explain {word}?",
    "I want to understand {word}.",
    "Help me learn about {word}.",
    "Describe {word}.",
    "What is the meaning of {word}?",
]

answer_templates = [
    "{word}: {definition}",
    "{word} is {definition}",
    "{word} means {definition}",
    "The term {word} refers to {definition}",
    "In simple terms, {word} is {definition}",
    "{word} can be defined as {definition}",
    "Understanding {word}: {definition}",
    "Let me explain {word}: {definition}",
]

# Generate training examples
training_data = []

print("\nGenerating training examples...")

# 1. Q&A pairs from definitions
for word, defs in definitions.items():
    for definition in defs[:3]:  # Use up to 3 definitions per word
        # Generate multiple question variations
        for q_template in question_templates:
            question = q_template.format(word=word)
            
            # Generate multiple answer variations
            for a_template in answer_templates[:3]:  # Use 3 answer templates
                answer = a_template.format(word=word, definition=definition)
                training_data.append(f"{question} {answer}")

print(f"  ✓ Generated {len(training_data)} Q&A pairs")

# 2. Statement variations
statement_data = []
for word, defs in definitions.items():
    for definition in defs[:2]:
        # Direct statements
        statement_data.append(f"{word.capitalize()}: {definition}")
        statement_data.append(f"{word.capitalize()} is {definition}")
        statement_data.append(f"The concept of {word}: {definition}")

print(f"  ✓ Generated {len(statement_data)} statements")
training_data.extend(statement_data)

# 3. Conversational examples
conversation_data = []
for word, defs in definitions.items():
    definition = defs[0]
    
    # Conversational patterns
    conversation_data.append(f"User asks about {word}. {word.capitalize()}: {definition}")
    conversation_data.append(f"To understand {word}, know that {definition}")
    conversation_data.append(f"When discussing {word}, remember: {definition}")

print(f"  ✓ Generated {len(conversation_data)} conversational examples")
training_data.extend(conversation_data)

# 4. Load existing knowledge
existing_knowledge = []
knowledge_files = [
    'knowledge/improved_dataset.txt',
    'knowledge/ecosystem_knowledge.txt'
]

for kfile in knowledge_files:
    if Path(kfile).exists():
        with open(kfile, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and len(line.strip()) > 30]
            existing_knowledge.extend(lines)

print(f"  ✓ Loaded {len(existing_knowledge)} existing knowledge entries")
training_data.extend(existing_knowledge)

# 5. Generate combination examples
combination_data = []
words_list = list(definitions.keys())

for i in range(min(100, len(words_list))):
    if i + 1 < len(words_list):
        word1 = words_list[i]
        word2 = words_list[i + 1]
        
        if word1 in definitions and word2 in definitions:
            def1 = definitions[word1][0]
            def2 = definitions[word2][0]
            
            combination_data.append(
                f"{word1.capitalize()} and {word2} are related concepts. "
                f"{word1.capitalize()}: {def1} {word2.capitalize()}: {def2}"
            )

print(f"  ✓ Generated {len(combination_data)} combination examples")
training_data.extend(combination_data)

# Remove duplicates
print("\nCleaning dataset...")
original_count = len(training_data)
training_data = list(set(training_data))
print(f"  ✓ Removed {original_count - len(training_data)} duplicates")

# Filter by length
training_data = [d for d in training_data if 20 < len(d) < 300]
print(f"  ✓ Filtered to {len(training_data)} quality examples")

# Shuffle
random.shuffle(training_data)

# Save
output_file = 'knowledge/hard_training_dataset.txt'
with open(output_file, 'w') as f:
    for line in training_data:
        f.write(line + '\n')

print("\n" + "=" * 70)
print("✅ HARD DATASET GENERATED")
print("=" * 70)
print(f"Total examples: {len(training_data)}")
print(f"Saved to: {output_file}")
print("\nDataset includes:")
print(f"  • Q&A pairs: ~{len(training_data) // 3}")
print(f"  • Statements: ~{len(statement_data)}")
print(f"  • Conversations: ~{len(conversation_data)}")
print(f"  • Combinations: ~{len(combination_data)}")
print(f"  • Existing knowledge: {len(existing_knowledge)}")
print("\nNext: python3 train_hard.py")
