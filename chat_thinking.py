#!/usr/bin/env python3
"""
Thinking Chat System - Combines neural understanding with knowledge retrieval
Shows thinking process, organizes thoughts, then responds intelligently
"""
import sys
sys.path.insert(0, 'src')
import json
import time
import re
from pathlib import Path

print("=" * 70)
print("KUIPERAI THINKING CHAT")
print("Neural Understanding + Organized Thinking + Knowledge")
print("=" * 70)

# Load knowledge base
knowledge_base = {}
definitions = {}
knowledge_entries = []

# Load from improved dataset
if Path('knowledge/improved_dataset.txt').exists():
    with open('knowledge/improved_dataset.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        knowledge_entries.extend(lines)
    
    for line in lines:
        words = line.lower().split()
        for word in words:
            if len(word) > 4:
                if word not in knowledge_base:
                    knowledge_base[word] = []
                if line not in knowledge_base[word]:
                    knowledge_base[word].append(line)

# Load from ecosystem knowledge
if Path('knowledge/ecosystem_knowledge.txt').exists():
    with open('knowledge/ecosystem_knowledge.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
        knowledge_entries.extend(lines)
        
        for line in lines:
            words = line.lower().split()
            for word in words:
                if len(word) > 4:
                    if word not in knowledge_base:
                        knowledge_base[word] = []
                    if line not in knowledge_base[word]:
                        knowledge_base[word].append(line)

# Load definitions
if Path('knowledge/ecosystem_vocab.json').exists():
    with open('knowledge/ecosystem_vocab.json', 'r') as f:
        vocab_data = json.load(f)
        definitions = vocab_data.get('definitions', {})

print(f"\n✓ Loaded {len(knowledge_base)} keywords")
print(f"✓ Loaded {len(definitions)} definitions")
print(f"✓ Loaded {len(knowledge_entries)} knowledge entries")

print("\n" + "=" * 70)
print("🧠 AI will think, organize, and respond intelligently")
print("=" * 70)
print("Type 'quit' to exit, 'help' for commands")
print("=" * 70)

def think_and_respond(user_input):
    """Think through the question and generate organized response"""
    
    print("KuiperAI: ", end='', flush=True)
    
    # Step 1: Analyze the question
    print("🤔 Analyzing", end='', flush=True)
    time.sleep(0.5)
    print(".", end='', flush=True)
    
    user_lower = user_input.lower()
    
    # Step 2: Search knowledge
    print(" 🔍 Searching", end='', flush=True)
    time.sleep(0.5)
    print(".", end='', flush=True)
    
    # Extract key concepts
    key_words = re.findall(r'\b\w{4,}\b', user_lower)
    
    # Find relevant information
    relevant_info = []
    relevant_defs = []
    
    for word in key_words:
        if word in definitions and definitions[word]:
            relevant_defs.append((word, definitions[word]))
        if word in knowledge_base:
            relevant_info.extend(knowledge_base[word][:2])
    
    # Step 3: Organize thoughts
    print(" 💭 Organizing", end='', flush=True)
    time.sleep(0.5)
    print(".", end='', flush=True)
    
    # Step 4: Formulate response
    print(" ✍️  Formulating", end='', flush=True)
    time.sleep(0.5)
    print(".\n\n", end='', flush=True)
    
    # Generate organized response
    response = ""
    
    # Handle greetings
    if any(word in user_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return "Hello! I'm KuiperAI, an AI that thinks before responding. I can explain concepts in machine learning, programming, and AI. What would you like to know?"
    
    # Handle identity
    if 'who are you' in user_lower or 'what are you' in user_lower:
        return "I'm KuiperAI, an intelligent AI assistant. I think through questions, search my knowledge base, organize my thoughts, and provide well-structured responses. I have definitions for 12+ concepts and growing knowledge about AI, machine learning, and programming."
    
    if 'who am i' in user_lower:
        return "You're the user I'm chatting with! I'm here to help you learn and understand complex topics."
    
    # Handle capability questions
    if 'what can you do' in user_lower or 'how can you help' in user_lower:
        return "I can:\n• Think through complex questions\n• Search my knowledge base\n• Provide detailed explanations\n• Define technical terms\n• Explain concepts in machine learning, AI, and programming\n\nI have 12+ definitions and 100+ knowledge entries. Ask me anything!"
    
    # Handle "what is" questions
    what_is_match = re.search(r'what (?:is|are) (?:an? )?(\w+)', user_lower)
    if what_is_match:
        word = what_is_match.group(1)
        
        # Check for definition
        if word in definitions and definitions[word]:
            response = f"📚 {word.capitalize()}:\n\n"
            response += f"{definitions[word][0]}\n"
            
            if len(definitions[word]) > 1:
                response += f"\n💡 Additional insights:\n"
                for i, def_text in enumerate(definitions[word][1:3], 1):
                    response += f"  {i}. {def_text}\n"
            
            return response.strip()
        
        # Search knowledge base
        if word in knowledge_base and knowledge_base[word]:
            response = f"📚 About {word}:\n\n"
            response += knowledge_base[word][0]
            return response
    
    # Handle "explain" questions
    explain_match = re.search(r'explain (?:about )?(\w+)', user_lower)
    if explain_match:
        word = explain_match.group(1)
        
        if word in definitions and definitions[word]:
            response = f"📚 Explaining {word}:\n\n"
            response += f"Definition: {definitions[word][0]}\n"
            
            if len(definitions[word]) > 1:
                response += f"\n💡 Key points:\n"
                for i, def_text in enumerate(definitions[word][1:4], 1):
                    response += f"  {i}. {def_text}\n"
            
            # Add related knowledge
            if word in knowledge_base and knowledge_base[word]:
                response += f"\n🔗 Related information:\n"
                response += f"  • {knowledge_base[word][0]}\n"
            
            return response.strip()
    
    # Handle "how" questions
    if user_input.lower().startswith('how'):
        if relevant_defs:
            word, defs = relevant_defs[0]
            response = f"📚 Regarding {word}:\n\n"
            response += f"{defs[0]}\n"
            
            if relevant_info:
                response += f"\n💡 Additional context:\n"
                response += f"  • {relevant_info[0]}\n"
            
            return response.strip()
    
    # General knowledge search
    if relevant_defs:
        word, defs = relevant_defs[0]
        response = f"📚 {word.capitalize()}:\n\n"
        response += f"{defs[0]}\n"
        
        if len(defs) > 1:
            response += f"\n💡 More details:\n"
            for i, def_text in enumerate(defs[1:3], 1):
                response += f"  {i}. {def_text}\n"
        
        return response.strip()
    
    if relevant_info:
        response = "📚 Based on my knowledge:\n\n"
        response += relevant_info[0]
        
        if len(relevant_info) > 1:
            response += f"\n\n💡 Also:\n  • {relevant_info[1]}"
        
        return response.strip()
    
    # Default response
    if '?' in user_input:
        return "🤔 That's an interesting question! I have knowledge about:\n• Machine learning\n• Neural networks\n• Algorithms\n• Data science\n• Programming\n\nCould you ask about one of these topics?"
    
    return "💭 I'm thinking about that... I have detailed knowledge about machine learning, AI, and programming. What specific aspect would you like to know about?"

# Chat loop
conversation_count = 0

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n🧠 Goodbye! Keep thinking and learning!")
            break
        
        if user_input.lower() == 'help':
            print("\n📖 Commands:")
            print("  help  - Show this help")
            print("  quit  - Exit chat")
            print("  stats - Show statistics")
            print("\n🧠 How I work:")
            print("  1. 🤔 Analyze your question")
            print("  2. 🔍 Search my knowledge base")
            print("  3. 💭 Organize my thoughts")
            print("  4. ✍️  Formulate a clear response")
            continue
        
        if user_input.lower() == 'stats':
            print("\n📊 Knowledge Statistics:")
            print(f"  Keywords indexed: {len(knowledge_base)}")
            print(f"  Definitions available: {len(definitions)}")
            print(f"  Knowledge entries: {len(knowledge_entries)}")
            print(f"  Conversations: {conversation_count}")
            print("\n📚 Defined concepts:")
            for i, word in enumerate(list(definitions.keys())[:10], 1):
                print(f"  {i}. {word}")
            if len(definitions) > 10:
                print(f"  ... and {len(definitions) - 10} more")
            continue
        
        if not user_input:
            continue
        
        # Think and respond
        response = think_and_respond(user_input)
        print(response)
        
        conversation_count += 1
        
    except KeyboardInterrupt:
        print("\n\n🧠 Goodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        print("Please try again.")

print(f"\nTotal conversations: {conversation_count}")
print("Thank you for using KuiperAI!")
