"""
Web learning system - Learn from internet sources
Downloads and processes knowledge from the web
"""
import urllib.request
import urllib.error
import json
import re
import time
from typing import List, Dict, Optional
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from safety.content_filter import ContentFilter, ContentModerator


class WebLearner:
    """Learn from web sources with safety filtering"""
    
    def __init__(self, knowledge_dir: str = 'knowledge/web_learned'):
        self.knowledge_dir = Path(knowledge_dir)
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        self.content_filter = ContentFilter()
        self.moderator = ContentModerator()
        
        # Educational sources (Wikipedia, educational sites)
        self.trusted_sources = [
            'wikipedia.org',
            'britannica.com',
            'khanacademy.org',
            'coursera.org',
            'mit.edu',
            'stanford.edu',
        ]
        
        self.learned_topics = []
        self.stats = {
            'total_fetched': 0,
            'filtered_out': 0,
            'learned': 0,
            'errors': 0
        }
    
    def is_trusted_source(self, url: str) -> bool:
        """Check if URL is from trusted source"""
        return any(source in url.lower() for source in self.trusted_sources)
    
    def fetch_text_from_url(self, url: str, timeout: int = 10) -> Optional[str]:
        """
        Fetch text content from URL
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Text content or None if failed
        """
        try:
            headers = {
                'User-Agent': 'KuiperAI-Educational-Bot/1.0'
            }
            
            request = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status == 200:
                    content = response.read().decode('utf-8', errors='ignore')
                    self.stats['total_fetched'] += 1
                    return content
                    
        except urllib.error.URLError as e:
            print(f"Error fetching {url}: {e}")
            self.stats['errors'] += 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            self.stats['errors'] += 1
        
        return None
    
    def extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML"""
        # Remove script and style tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def learn_from_url(self, url: str, topic: str) -> bool:
        """
        Learn from a specific URL
        
        Args:
            url: URL to learn from
            topic: Topic name for organization
            
        Returns:
            True if successfully learned
        """
        print(f"\nLearning from: {url}")
        
        # Check if trusted source
        if not self.is_trusted_source(url):
            print(f"⚠️  Warning: Not a trusted source")
        
        # Fetch content
        html = self.fetch_text_from_url(url)
        if not html:
            return False
        
        # Extract text
        text = self.extract_text_from_html(html)
        
        # Limit text length
        text = text[:10000]  # First 10k chars
        
        # Filter content
        allow, reason = self.content_filter.should_allow(text)
        
        if not allow:
            print(f"❌ Content filtered: {reason}")
            self.stats['filtered_out'] += 1
            return False
        
        print(f"✓ Content approved: {reason}")
        
        # Save learned content
        topic_dir = self.knowledge_dir / topic
        topic_dir.mkdir(exist_ok=True)
        
        # Create filename from URL
        filename = re.sub(r'[^\w\-]', '_', url.split('//')[-1][:50]) + '.txt'
        filepath = topic_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Source: {url}\n")
            f.write(f"Topic: {topic}\n")
            f.write(f"Learned: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(text)
        
        self.learned_topics.append(topic)
        self.stats['learned'] += 1
        
        print(f"✓ Saved to: {filepath}")
        return True
    
    def learn_from_text_file(self, filepath: str, topic: str) -> bool:
        """
        Learn from a local text file
        
        Args:
            filepath: Path to text file
            topic: Topic name
            
        Returns:
            True if successfully learned
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Filter content
            allow, reason = self.content_filter.should_allow(text)
            
            if not allow:
                print(f"❌ Content filtered: {reason}")
                self.stats['filtered_out'] += 1
                return False
            
            print(f"✓ Content approved: {reason}")
            
            # Save to knowledge base
            topic_dir = self.knowledge_dir / topic
            topic_dir.mkdir(exist_ok=True)
            
            filename = Path(filepath).name
            dest = topic_dir / filename
            
            with open(dest, 'w', encoding='utf-8') as f:
                f.write(f"Source: {filepath}\n")
                f.write(f"Topic: {topic}\n")
                f.write(f"Learned: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 70 + "\n\n")
                f.write(text)
            
            self.learned_topics.append(topic)
            self.stats['learned'] += 1
            
            print(f"✓ Learned from: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error reading file: {e}")
            self.stats['errors'] += 1
            return False
    
    def learn_from_wikipedia(self, topic: str) -> bool:
        """
        Learn from Wikipedia article
        
        Args:
            topic: Wikipedia article title
            
        Returns:
            True if successfully learned
        """
        # Wikipedia API endpoint
        topic_encoded = topic.replace(' ', '_')
        url = f"https://en.wikipedia.org/wiki/{topic_encoded}"
        
        return self.learn_from_url(url, topic)
    
    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        return self.stats.copy()
    
    def save_statistics(self, filepath: str = 'knowledge/learning_stats.json'):
        """Save statistics to file"""
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"✓ Statistics saved to: {filepath}")


class KnowledgeAggregator:
    """Aggregate learned knowledge for training"""
    
    def __init__(self, knowledge_dir: str = 'knowledge'):
        self.knowledge_dir = Path(knowledge_dir)
    
    def aggregate_all_knowledge(self) -> List[str]:
        """
        Aggregate all knowledge from all sources
        
        Returns:
            List of text samples
        """
        all_texts = []
        
        # Scan all subdirectories
        for subdir in self.knowledge_dir.rglob('*'):
            if subdir.is_dir():
                # Read all .txt files
                for filepath in subdir.glob('*.txt'):
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            text = f.read()
                            
                        # Split into paragraphs
                        paragraphs = [p.strip() for p in text.split('\n\n') 
                                    if p.strip() and len(p.strip()) > 50]
                        
                        all_texts.extend(paragraphs)
                        
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
        
        return all_texts
    
    def create_training_dataset(self, output_file: str, max_samples: int = 10000):
        """
        Create training dataset from all knowledge
        
        Args:
            output_file: Output file path
            max_samples: Maximum number of samples
        """
        print(f"\nAggregating knowledge for training...")
        
        texts = self.aggregate_all_knowledge()
        
        # Limit samples
        if len(texts) > max_samples:
            import random
            texts = random.sample(texts, max_samples)
        
        # Save to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        print(f"✓ Created training dataset: {output_file}")
        print(f"  Total samples: {len(texts)}")


def demo_web_learning():
    """Demo the web learning system"""
    print("=" * 70)
    print("KUIPERAI WEB LEARNING DEMO")
    print("=" * 70)
    
    learner = WebLearner()
    
    # Test content filtering
    print("\n[1/3] Testing content filter...")
    test_texts = [
        "Machine learning is a subset of artificial intelligence",
        "How to hack into systems illegally",
        "Python programming tutorial for beginners",
    ]
    
    for text in test_texts:
        allow, reason = learner.content_filter.should_allow(text)
        status = "✓ ALLOWED" if allow else "❌ BLOCKED"
        print(f"{status}: {text[:50]}...")
        print(f"  Reason: {reason}")
    
    # Learn from existing files
    print("\n[2/3] Learning from existing knowledge base...")
    
    existing_files = [
        'knowledge/datasets/general/sample_knowledge.txt',
        'knowledge/datasets/nlp/sample_text.txt',
        'knowledge/datasets/code/sample_code.txt',
    ]
    
    for filepath in existing_files:
        if Path(filepath).exists():
            learner.learn_from_text_file(filepath, Path(filepath).parent.name)
    
    # Show statistics
    print("\n[3/3] Learning statistics:")
    stats = learner.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    learner.save_statistics()
    
    print("\n" + "=" * 70)
    print("✓ Web learning demo complete")
    print("=" * 70)


if __name__ == "__main__":
    demo_web_learning()
