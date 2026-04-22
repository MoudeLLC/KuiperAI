#!/usr/bin/env python3
"""
Simple rule-based chat with KuiperAI
Uses pattern matching for more coherent responses
"""
import sys
import re
from pathlib import Path

print("=" * 70)
print("KUIPERAI SIMPLE CHAT")
print("Rule-based responses for better coherence")
print("=" * 70)

# Load knowledge from improved dataset
knowledge_base = {}
if Path('knowledge/improved_dataset.txt').exists():
    with open('knowledge/improved_dataset.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Index by keywords
    for line in lines:
        words = line.lower().split()
        for word in words:
            if len(word) > 4:  # Only index meaningful words
                if word not in knowledge_base:
                    knowledge_base[word] = []
                if line not in knowledge_base[word]:
                    knowledge_base[word].append(line)

print(f"\n✓ Loaded {len(lines)} knowledge entries")
print(f"✓ Indexed {len(knowledge_base)} keywords")

print("\n" + "=" * 70)
print("Type 'quit' to exit, 'help' for commands")
print("=" * 70)

def get_response(user_input):
    """Get response using keyword matching"""
    user_lower = user_input.lower()
    
    # Greetings
    if any(word in user_lower for word in ['hi', 'hello', 'hey']):
        return "Hello! I'm KuiperAI. I can help explain concepts in AI, machine learning, and programming. What would you like to know?"
    
    # Identity questions
    if 'who are you' in user_lower or 'what are you' in user_lower:
        return "I'm KuiperAI, an AI assistant focused on learning and education. I can explain concepts in machine learning, programming, and science."
    
    if 'who am i' in user_lower:
        return "You're the user chatting with me! I'm here to help answer your questions about AI and technology."
    
    # Capability questions
    if 'what can you do' in user_lower or 'how can you help' in user_lower:
        return "I can explain concepts in machine learning, deep learning, neural networks, programming, and related topics. Just ask me a question!"
    
    # Extract keywords from user input
    words = re.findall(r'\b\w{4,}\b', user_lower)
    
    # Find best matching knowledge
    matches = []
    for word in words:
        if word in knowledge_base:
            matches.extend(knowledge_base[word])
    
    if matches:
        # Return most relevant match (first one for simplicity)
        return matches[0]
    
    # Default responses for common patterns
    if '?' in user_input:
        return "That's an interesting question! I'm still learning about that topic. Could you ask about machine learning, neural networks, or programming?"
    
    return "I'm still learning! I know about machine learning, neural networks, deep learning, and programming. What would you like to know about these topics?"

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
            print("\nTopics I know about:")
            print("  • Machine learning")
            print("  • Deep learning")
            print("  • Neural networks")
            print("  • Python programming")
            print("  • Artificial intelligence")
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

print(f"\nTotal conversations: {conversation_count}")
print("Thank you for using KuiperAI!")
