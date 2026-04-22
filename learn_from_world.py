#!/usr/bin/env python3
"""
Learn Everything from the World
Comprehensive learning system that aggregates knowledge from multiple sources
"""
import sys
sys.path.insert(0, 'src')
import json
from pathlib import Path
from typing import List, Dict
import time

from network.web_learner import WebLearner, KnowledgeAggregator
from safety.content_filter import ContentFilter
from data.dataset import TextDataset


class WorldKnowledgeLearner:
    """Learn from all available knowledge sources"""
    
    def __init__(self):
        self.web_learner = WebLearner()
        self.aggregator = KnowledgeAggregator()
        self.content_filter = ContentFilter()
        
        # Topics to learn about
        self.world_topics = [
            # Science & Technology
            "Artificial Intelligence",
            "Machine Learning",
            "Deep Learning",
            "Neural Networks",
            "Computer Science",
            "Physics",
            "Chemistry",
            "Biology",
            "Mathematics",
            "Astronomy",
            
            # Programming
            "Python Programming",
            "JavaScript",
            "Data Structures",
            "Algorithms",
            "Software Engineering",
            "Web Development",
            "Database Systems",
            
            # General Knowledge
            "History",
            "Geography",
            "Literature",
            "Philosophy",
            "Psychology",
            "Economics",
            "Politics",
            
            # Arts & Culture
            "Music",
            "Art",
            "Cinema",
            "Theater",
            
            # Practical Skills
            "Cooking",
            "Health",
            "Fitness",
            "Communication",
            "Problem Solving",
        ]
        
        self.learning_plan = []
        self.learned_knowledge = {}
    
    def create_learning_plan(self) -> List[Dict]:
        """Create a comprehensive learning plan"""
        print("=" * 70)
        print("CREATING WORLD KNOWLEDGE LEARNING PLAN")
        print("=" * 70)
        
        plan = []
        
        for topic in self.world_topics:
            plan.append({
                'topic': topic,
                'category': self._categorize_topic(topic),
                'priority': self._get_priority(topic),
                'sources': self._get_sources_for_topic(topic),
                'status': 'pending'
            })
        
        # Sort by priority
        plan.sort(key=lambda x: x['priority'], reverse=True)
        
        self.learning_plan = plan
        
        print(f"\n✓ Created learning plan with {len(plan)} topics")
        print(f"\nTop 10 Priority Topics:")
        for i, item in enumerate(plan[:10], 1):
            print(f"  {i}. {item['topic']} ({item['category']})")
        
        return plan
    
    def _categorize_topic(self, topic: str) -> str:
        """Categorize a topic"""
        topic_lower = topic.lower()
        
        if any(kw in topic_lower for kw in ['ai', 'machine', 'neural', 'computer', 'programming']):
            return 'Technology'
        elif any(kw in topic_lower for kw in ['physics', 'chemistry', 'biology', 'math']):
            return 'Science'
        elif any(kw in topic_lower for kw in ['history', 'geography', 'literature']):
            return 'Humanities'
        elif any(kw in topic_lower for kw in ['music', 'art', 'cinema']):
            return 'Arts'
        else:
            return 'General'
    
    def _get_priority(self, topic: str) -> int:
        """Get priority for a topic (1-10)"""
        topic_lower = topic.lower()
        
        # High priority for AI/ML topics
        if any(kw in topic_lower for kw in ['artificial intelligence', 'machine learning', 'neural']):
            return 10
        
        # Medium-high for programming
        if any(kw in topic_lower for kw in ['programming', 'python', 'algorithm']):
            return 8
        
        # Medium for science
        if any(kw in topic_lower for kw in ['physics', 'math', 'science']):
            return 7
        
        # Default
        return 5
    
    def _get_sources_for_topic(self, topic: str) -> List[str]:
        """Get potential sources for a topic"""
        sources = []
        
        # Add existing knowledge base files
        knowledge_base = Path('knowledge/datasets')
        if knowledge_base.exists():
            for subdir in knowledge_base.iterdir():
                if subdir.is_dir():
                    for file in subdir.glob('*.txt'):
                        sources.append(str(file))
        
        return sources
    
    def learn_from_existing_knowledge(self):
        """Learn from existing knowledge base"""
        print("\n" + "=" * 70)
        print("LEARNING FROM EXISTING KNOWLEDGE BASE")
        print("=" * 70)
        
        knowledge_base = Path('knowledge/datasets')
        
        if not knowledge_base.exists():
            print("No existing knowledge base found")
            return
        
        learned_count = 0
        filtered_count = 0
        
        for subdir in knowledge_base.iterdir():
            if subdir.is_dir():
                topic = subdir.name
                print(f"\nLearning from: {topic}")
                
                for file in subdir.glob('*.txt'):
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            text = f.read()
                        
                        # Filter content
                        allow, reason = self.content_filter.should_allow(text)
                        
                        if allow:
                            # Store in learned knowledge
                            if topic not in self.learned_knowledge:
                                self.learned_knowledge[topic] = []
                            
                            self.learned_knowledge[topic].append({
                                'source': str(file),
                                'text': text,
                                'learned_at': time.strftime('%Y-%m-%d %H:%M:%S')
                            })
                            
                            learned_count += 1
                            print(f"  ✓ Learned: {file.name}")
                        else:
                            filtered_count += 1
                            print(f"  ✗ Filtered: {file.name} - {reason}")
                    
                    except Exception as e:
                        print(f"  Error: {e}")
        
        print(f"\n✓ Learned from {learned_count} sources")
        print(f"✗ Filtered {filtered_count} sources")
    
    def create_comprehensive_dataset(self, output_file: str = 'knowledge/comprehensive_dataset.txt'):
        """Create comprehensive training dataset"""
        print("\n" + "=" * 70)
        print("CREATING COMPREHENSIVE TRAINING DATASET")
        print("=" * 70)
        
        all_texts = []
        
        # Aggregate from learned knowledge
        for topic, items in self.learned_knowledge.items():
            for item in items:
                text = item['text']
                
                # Split into sentences/paragraphs
                paragraphs = [p.strip() for p in text.split('\n') 
                            if p.strip() and len(p.strip()) > 20]
                
                all_texts.extend(paragraphs)
        
        # Also aggregate from knowledge aggregator
        aggregated = self.aggregator.aggregate_all_knowledge()
        all_texts.extend(aggregated)
        
        # Remove duplicates
        all_texts = list(set(all_texts))
        
        # Filter again for safety
        filtered_texts = []
        for text in all_texts:
            allow, _ = self.content_filter.should_allow(text)
            if allow:
                filtered_texts.append(text)
        
        # Save to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in filtered_texts:
                f.write(text + '\n')
        
        print(f"\n✓ Created comprehensive dataset")
        print(f"  Total texts: {len(all_texts):,}")
        print(f"  After filtering: {len(filtered_texts):,}")
        print(f"  Saved to: {output_file}")
        
        return filtered_texts
    
    def generate_learning_report(self, output_file: str = 'knowledge/learning_report.json'):
        """Generate comprehensive learning report"""
        print("\n" + "=" * 70)
        print("GENERATING LEARNING REPORT")
        print("=" * 70)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_topics': len(self.world_topics),
            'topics_learned': len(self.learned_knowledge),
            'learning_plan': self.learning_plan,
            'learned_knowledge_summary': {},
            'statistics': {
                'total_sources': sum(len(items) for items in self.learned_knowledge.values()),
                'categories': {},
            }
        }
        
        # Summarize learned knowledge
        for topic, items in self.learned_knowledge.items():
            report['learned_knowledge_summary'][topic] = {
                'sources': len(items),
                'total_text_length': sum(len(item['text']) for item in items)
            }
        
        # Category statistics
        for item in self.learning_plan:
            category = item['category']
            if category not in report['statistics']['categories']:
                report['statistics']['categories'][category] = 0
            report['statistics']['categories'][category] += 1
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Learning report saved to: {output_file}")
        
        # Print summary
        print(f"\nLearning Summary:")
        print(f"  Topics in plan: {report['total_topics']}")
        print(f"  Topics learned: {report['topics_learned']}")
        print(f"  Total sources: {report['statistics']['total_sources']}")
        print(f"\nBy Category:")
        for category, count in report['statistics']['categories'].items():
            print(f"  {category}: {count} topics")
        
        return report
    
    def execute_learning_plan(self):
        """Execute the complete learning plan"""
        print("\n" + "=" * 70)
        print("EXECUTING WORLD KNOWLEDGE LEARNING PLAN")
        print("=" * 70)
        
        # Step 1: Create plan
        self.create_learning_plan()
        
        # Step 2: Learn from existing knowledge
        self.learn_from_existing_knowledge()
        
        # Step 3: Create comprehensive dataset
        dataset_texts = self.create_comprehensive_dataset()
        
        # Step 4: Generate report
        report = self.generate_learning_report()
        
        # Step 5: Summary
        print("\n" + "=" * 70)
        print("LEARNING COMPLETE!")
        print("=" * 70)
        print(f"\n✅ Learned from {len(self.learned_knowledge)} topics")
        print(f"✅ Created dataset with {len(dataset_texts):,} texts")
        print(f"✅ Generated comprehensive report")
        
        print("\nNext Steps:")
        print("  1. Review: knowledge/learning_report.json")
        print("  2. Train on: knowledge/comprehensive_dataset.txt")
        print("  3. Run: python3 train_comprehensive.py")
        
        return report


def main():
    """Main execution"""
    print("=" * 70)
    print("KUIPERAI - LEARN EVERYTHING FROM THE WORLD")
    print("=" * 70)
    print("\nThis system will:")
    print("  • Create a comprehensive learning plan")
    print("  • Learn from existing knowledge sources")
    print("  • Filter content for safety")
    print("  • Create training datasets")
    print("  • Generate learning reports")
    
    input("\nPress Enter to start learning...")
    
    learner = WorldKnowledgeLearner()
    learner.execute_learning_plan()
    
    print("\n" + "=" * 70)
    print("✅ WORLD KNOWLEDGE LEARNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
