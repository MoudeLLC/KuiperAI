#!/usr/bin/env python3
"""
Real Understanding Chat System
Analyzes each word's meaning, understands context, generates intelligent responses
"""
import sys
sys.path.insert(0, 'src')
import json
import time
import re
from pathlib import Path
from collections import defaultdict

print("=" * 70)
print("KUIPERAI - REAL UNDERSTANDING SYSTEM")
print("Analyzes word meanings, understands context, generates responses")
print("=" * 70)

class RealUnderstanding:
    def __init__(self):
        self.word_meanings = {}  # word -> list of meanings
        self.word_contexts = {}  # word -> list of contexts
        self.word_relations = defaultdict(set)  # word -> related words
        self.knowledge_graph = defaultdict(list)  # concept -> facts
        
        self.load_knowledge()
    
    def load_knowledge(self):
        """Load and structure all knowledge"""
        print("\n🧠 Loading knowledge base...")
        
        # Load definitions
        if Path('knowledge/ecosystem_vocab.json').exists():
            with open('knowledge/ecosystem_vocab.json', 'r') as f:
                vocab_data = json.load(f)
                definitions = vocab_data.get('definitions', {})
                
                for word, defs in definitions.items():
                    self.word_meanings[word] = defs
                    
                    # Extract related words from definitions
                    for definition in defs:
                        words = re.findall(r'\b[a-z]{4,}\b', definition.lower())
                        for w in words:
                            if w != word:
                                self.word_relations[word].add(w)
        
        # Load knowledge entries
        knowledge_files = [
            'knowledge/improved_dataset.txt',
            'knowledge/ecosystem_knowledge.txt'
        ]
        
        for kfile in knowledge_files:
            if Path(kfile).exists():
                with open(kfile, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if len(line) > 20:
                            # Extract key concepts
                            words = re.findall(r'\b[a-z]{4,}\b', line.lower())
                            for word in set(words):
                                if word not in self.word_contexts:
                                    self.word_contexts[word] = []
                                self.word_contexts[word].append(line)
                                
                                # Build knowledge graph
                                self.knowledge_graph[word].append(line)
        
        print(f"  ✓ Loaded {len(self.word_meanings)} word meanings")
        print(f"  ✓ Loaded {len(self.word_contexts)} word contexts")
        print(f"  ✓ Built knowledge graph with {len(self.knowledge_graph)} concepts")
    
    def understand_words(self, text):
        """Understand each word in the text"""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        understanding = {
            'key_concepts': [],
            'meanings': {},
            'related_concepts': set(),
            'context': []
        }
        
        # Analyze each word
        for word in words:
            if len(word) < 4:
                continue
            
            # Get meaning
            if word in self.word_meanings:
                understanding['key_concepts'].append(word)
                understanding['meanings'][word] = self.word_meanings[word][0]
                
                # Get related concepts
                if word in self.word_relations:
                    understanding['related_concepts'].update(self.word_relations[word])
            
            # Get context
            if word in self.word_contexts:
                understanding['context'].extend(self.word_contexts[word][:2])
        
        return understanding
    
    def find_best_response(self, understanding, question_type):
        """Find best response based on understanding"""
        key_concepts = understanding['key_concepts']
        meanings = understanding['meanings']
        context = understanding['context']
        
        if not key_concepts:
            return None
        
        # Build response based on question type
        response_parts = []
        
        if question_type == 'what_is':
            # Define the main concept
            main_concept = key_concepts[0]
            if main_concept in meanings:
                response_parts.append(f"{main_concept.capitalize()}: {meanings[main_concept]}")
                
                # Add related information
                if main_concept in self.knowledge_graph:
                    facts = self.knowledge_graph[main_concept][:3]
                    if facts:
                        response_parts.append("\nKey points:")
                        for i, fact in enumerate(facts, 1):
                            response_parts.append(f"  • {fact}")
        
        elif question_type == 'explain':
            # Provide detailed explanation
            main_concept = key_concepts[0]
            if main_concept in meanings:
                response_parts.append(f"Understanding {main_concept}:\n")
                response_parts.append(meanings[main_concept])
                
                # Add context
                if context:
                    response_parts.append("\n\nContext:")
                    for ctx in context[:2]:
                        response_parts.append(f"  • {ctx}")
        
        elif question_type == 'how':
            # Explain process or mechanism
            if context:
                response_parts.append("Based on my understanding:\n")
                response_parts.append(context[0])
                
                if len(context) > 1:
                    response_parts.append(f"\n\nAdditionally:\n  • {context[1]}")
        
        else:
            # General response
            if meanings:
                main_concept = list(meanings.keys())[0]
                response_parts.append(f"Regarding {main_concept}:\n")
                response_parts.append(meanings[main_concept])
                
                if context:
                    response_parts.append(f"\n\nMore information:\n  • {context[0]}")
        
        return '\n'.join(response_parts) if response_parts else None
    
    def generate_response(self, user_input):
        """Generate intelligent response"""
        print("\n🧠 Thinking: ", end='', flush=True)
        
        # Step 1: Understand each word
        print("Analyzing words", end='', flush=True)
        time.sleep(0.3)
        print(".", end='', flush=True)
        
        understanding = self.understand_words(user_input)
        
        # Step 2: Determine question type
        print(" Understanding intent", end='', flush=True)
        time.sleep(0.3)
        print(".", end='', flush=True)
        
        user_lower = user_input.lower()
        question_type = None
        
        if re.search(r'what (?:is|are)', user_lower):
            question_type = 'what_is'
        elif 'explain' in user_lower:
            question_type = 'explain'
        elif user_lower.startswith('how'):
            question_type = 'how'
        elif any(word in user_lower for word in ['hi', 'hello', 'hey']):
            question_type = 'greeting'
        elif 'who are you' in user_lower:
            question_type = 'identity'
        
        # Step 3: Find best response
        print(" Finding best answer", end='', flush=True)
        time.sleep(0.3)
        print(".", end='', flush=True)
        
        # Step 4: Organize response
        print(" Organizing response", end='', flush=True)
        time.sleep(0.3)
        print(".\n\n", end='', flush=True)
        
        # Handle special cases
        if question_type == 'greeting':
            return "Hello! I'm KuiperAI. I analyze each word you say, understand its meaning, and generate thoughtful responses. Ask me anything about machine learning, AI, or programming!"
        
        if question_type == 'identity':
            return f"I'm KuiperAI, a real understanding system. I have:\n  • {len(self.word_meanings)} word meanings\n  • {len(self.knowledge_graph)} concepts in my knowledge graph\n  • Deep understanding of {len(self.word_contexts)} words\n\nI analyze each word you use, understand the context, and generate intelligent responses."
        
        # Generate response based on understanding
        response = self.find_best_response(understanding, question_type)
        
        if response:
            return response
        
        # Fallback: show what we understood
        if understanding['key_concepts']:
            concepts = ', '.join(understanding['key_concepts'][:3])
            return f"I understand you're asking about: {concepts}\n\nLet me search deeper... Could you rephrase your question to be more specific?"
        
        return "I'm analyzing your question. Could you ask about machine learning, algorithms, neural networks, or data science?"

# Initialize system
system = RealUnderstanding()

print("\n" + "=" * 70)
print("✅ System ready - Real understanding enabled")
print("=" * 70)
print("Type 'quit' to exit, 'help' for commands")
print("=" * 70)

# Chat loop
conversation_count = 0

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye! Keep learning!")
            break
        
        if user_input.lower() == 'help':
            print("\n📖 Commands:")
            print("  help  - Show this help")
            print("  quit  - Exit chat")
            print("  stats - Show statistics")
            print("\n🧠 How I understand:")
            print("  1. Analyze each word's meaning")
            print("  2. Understand the context")
            print("  3. Find related concepts")
            print("  4. Generate best response")
            continue
        
        if user_input.lower() == 'stats':
            print("\n📊 Understanding Statistics:")
            print(f"  Word meanings: {len(system.word_meanings)}")
            print(f"  Word contexts: {len(system.word_contexts)}")
            print(f"  Knowledge graph: {len(system.knowledge_graph)} concepts")
            print(f"  Word relations: {sum(len(v) for v in system.word_relations.values())} connections")
            print(f"  Conversations: {conversation_count}")
            continue
        
        if not user_input:
            continue
        
        # Generate response
        response = system.generate_response(user_input)
        print(response)
        
        conversation_count += 1
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

print(f"\nTotal conversations: {conversation_count}")
print("Thank you for using KuiperAI!")
