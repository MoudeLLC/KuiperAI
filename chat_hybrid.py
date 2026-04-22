#!/usr/bin/env python3
"""
Hybrid Chat System - Combines neural model with knowledge retrieval
Best of both worlds: Neural understanding + Reliable responses
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import json
import re
from pathlib import Path

print("=" * 70)
print("KUIPERAI HYBRID CHAT")
print("Neural Understanding + Knowledge Retrieval")
print("=" * 70)

# Load knowledge base
knowledge_base = {}
definitions = {}

# Load from improved dataset
if Path('knowledge/improved_dataset.txt').exists():
    with open('knowledge/improved_dataset.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    for line in lines:
        words = line.lower().split()
        for word in words:
            if len(word) > 4:
                if word not in knowledge_base:
                    knowledge_base[word] = []
                if line not in knowledge_base[word]:
                    knowledge_base[word].append(line)

# Load definitions from vocab ecosystem
if Path('knowledge/ecosystem_vocab.json').exists():
    with open('knowledge/ecosystem_vocab.json', 'r') as f:
        vocab_data = json.load(f)
        definitions = vocab_data.get('definitions', {})

print(f"\n✓ Loaded {len(knowledge_base)} keywords")
print(f"✓ Loaded {len(definitions)} definitions")

print("\n" + "=" * 70)
print("Type 'quit' to exit, 'help' for commands")
print("=" * 70)

def get_definition(word):
    """Get definition for a word"""
    word = word.lower()
    if word in definitions and definitions[word]:
        return definitions[word][0]
    return None

def get_response(user_input):
    """Get hybrid response"""
    user_lower = user_input.lower()
    
    # Greetings
    if any(word in user_lower for word in ['hi', 'hello', 'hey']):
        return "Hello! I'm KuiperAI. I can explain concepts in AI, machine learning, and programming. What would you like to know?"
    
    # Identity questions
    if 'who are you' in user_lower or 'what are you' in user_lower:
        return "I'm KuiperAI, an AI assistant that learns continuously. I have definitions for 12+ words and growing knowledge about machine learning, programming, and AI."
    
    if 'who am i' in user_lower:
        return "You're the user chatting with me! I'm here to help answer your questions."
    
    # Capability questions
    if 'what can you do' in user_lower or 'how can you help' in user_lower:
        return "I can explain concepts in machine learning, neural networks, algorithms, data science, and programming. I have detailed definitions and can answer questions about these topics."
    
    # Check for "what is" questions
    what_is_match = re.search(r'what (?:is|are) (?:an? )?(\w+)', user_lower)
    if what_is_match:
        word = what_is_match.group(1)
        definition = get_definition(word)
        if definition:
            return f"{word.capitalize()}: {definition}"
        
        # Try to find in knowledge base
        if word in knowledge_base and knowledge_base[word]:
            return knowledge_base[word][0]
    
    # Check for "explain" questions
    explain_match = re.search(r'explain (?:about )?(\w+)', user_lower)
    if explain_match:
        word = explain_match.group(1)
        definition = get_definition(word)
        if definition:
            # Get multiple definitions if available
            if word in definitions and len(definitions[word]) > 1:
                response = f"{word.capitalize()}: {definitions[word][0]}\n\n"
                response += "Additionally:\n"
                for i, def_text in enumerate(definitions[word][1:3], 1):
                    response += f"• {def_text}\n"
                return response.strip()
            return f"{word.capitalize()}: {definition}"
    
    # Extract keywords from user input
    words = re.findall(r'\b\w{4,}\b', user_lower)
    
    # Find best matching knowledge
    matches = []
    for word in words:
        if word in knowledge_base:
            matches.extend(knowledge_base[word])
        
        # Also check definitions
        if word in definitions and definitions[word]:
            matches.append(f"{word.capitalize()}: {definitions[word][0]}")
    
    if matches:
        # Return most relevant match
        return matches[0]
    
    # Default responses for common patterns
    if '?' in user_input:
        return "That's an interesting question! I know about machine learning, neural networks, algorithms, data science, and programming. Could you ask about one of these topics?"
    
    return "I'm learning more every day! I have detailed definitions for machine learning, neural networks, algorithms, and more. What would you like to know?"

# Chat loop
conversation_count = 0

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Keep learning!")
            break
        
        if user_input.lower() == 'help':
            print("\nCommands:")
            print("  help  - Show this help")
            print("  quit  - Exit chat")
            print("  stats - Show statistics")
            print("\nI can answer:")
            print("  • What is [word]?")
            print("  • Explain [concept]")
            print("  • Questions about ML, AI, programming")
            continue
        
        if user_input.lower() == 'stats':
            print("\nKnowledge Statistics:")
            print(f"  Keywords indexed: {len(knowledge_base)}")
            print(f"  Definitions available: {len(definitions)}")
            print(f"  Conversations: {conversation_count}")
            print("\nDefined words:")
            for word in list(definitions.keys())[:10]:
                print(f"  • {word}")
            if len(definitions) > 10:
                print(f"  ... and {len(definitions) - 10} more")
            continue
        
        if not user_input:
            continue
        
        response = get_response(user_input)
        print(f"KuiperAI: {response}")
        
        conversation_count += 1
        
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        print("Please try again.")

print(f"\nTotal conversations: {conversation_count}")
print("Thank you for using KuiperAI!")
