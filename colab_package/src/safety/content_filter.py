"""
Content filtering and safety system
Filters harmful, banned, or inappropriate content
"""
import re
from typing import Dict, List, Tuple
from enum import Enum


class ContentCategory(Enum):
    """Content categories for filtering"""
    SAFE = "safe"
    EDUCATIONAL = "educational"
    QUESTIONABLE = "questionable"
    HARMFUL = "harmful"
    BANNED = "banned"


class ContentFilter:
    """Filter and classify content for safety"""
    
    def __init__(self):
        # Banned content patterns
        self.banned_patterns = [
            r'\b(hack|crack|exploit|malware|virus)\s+(tutorial|guide|how\s+to)\b',
            r'\b(illegal|pirate|steal|fraud)\b',
            r'\b(weapon|bomb|explosive)\s+(make|build|create)\b',
        ]
        
        # Harmful content indicators
        self.harmful_keywords = [
            'violence', 'harm', 'attack', 'kill', 'destroy',
            'hate', 'discriminate', 'abuse', 'threat'
        ]
        
        # Educational content indicators
        self.educational_keywords = [
            'learn', 'study', 'understand', 'explain', 'teach',
            'science', 'math', 'history', 'programming', 'research'
        ]
        
        # Safe topics
        self.safe_topics = [
            'technology', 'science', 'education', 'art', 'music',
            'sports', 'cooking', 'travel', 'nature', 'health'
        ]
    
    def classify_content(self, text: str) -> Tuple[ContentCategory, float, str]:
        """
        Classify content and return category, confidence, and reason
        
        Returns:
            (category, confidence_score, reason)
        """
        text_lower = text.lower()
        
        # Check for banned content
        for pattern in self.banned_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return (ContentCategory.BANNED, 0.95, 
                       "Contains banned content pattern")
        
        # Count harmful keywords
        harmful_count = sum(1 for kw in self.harmful_keywords 
                          if kw in text_lower)
        
        # Count educational keywords
        educational_count = sum(1 for kw in self.educational_keywords 
                               if kw in text_lower)
        
        # Count safe topics
        safe_count = sum(1 for topic in self.safe_topics 
                        if topic in text_lower)
        
        # Classify based on counts
        if harmful_count >= 3:
            return (ContentCategory.HARMFUL, 0.8,
                   f"Contains {harmful_count} harmful indicators")
        
        if harmful_count >= 1 and educational_count == 0:
            return (ContentCategory.QUESTIONABLE, 0.6,
                   "Contains potentially harmful content")
        
        if educational_count >= 2 or safe_count >= 2:
            return (ContentCategory.EDUCATIONAL, 0.85,
                   "Educational or safe content")
        
        return (ContentCategory.SAFE, 0.7, "General safe content")
    
    def should_allow(self, text: str) -> Tuple[bool, str]:
        """
        Determine if content should be allowed
        
        Returns:
            (allow, reason)
        """
        category, confidence, reason = self.classify_content(text)
        
        if category == ContentCategory.BANNED:
            return (False, f"BLOCKED: {reason}")
        
        if category == ContentCategory.HARMFUL and confidence > 0.7:
            return (False, f"BLOCKED: {reason}")
        
        if category == ContentCategory.QUESTIONABLE:
            return (True, f"WARNING: {reason}")
        
        return (True, f"ALLOWED: {reason}")
    
    def filter_dataset(self, texts: List[str]) -> Tuple[List[str], Dict]:
        """
        Filter a dataset and return safe content with statistics
        
        Returns:
            (filtered_texts, statistics)
        """
        filtered = []
        stats = {
            'total': len(texts),
            'allowed': 0,
            'blocked': 0,
            'warnings': 0,
            'categories': {cat.value: 0 for cat in ContentCategory}
        }
        
        for text in texts:
            category, confidence, reason = self.classify_content(text)
            stats['categories'][category.value] += 1
            
            allow, msg = self.should_allow(text)
            
            if allow:
                filtered.append(text)
                stats['allowed'] += 1
                if 'WARNING' in msg:
                    stats['warnings'] += 1
            else:
                stats['blocked'] += 1
        
        return filtered, stats


class ContentModerator:
    """Moderate AI responses in real-time"""
    
    def __init__(self):
        self.filter = ContentFilter()
        self.response_rules = [
            "Be helpful and informative",
            "Avoid harmful or dangerous content",
            "Respect all people and cultures",
            "Provide educational value",
            "Decline illegal requests politely"
        ]
    
    def moderate_response(self, prompt: str, response: str) -> Tuple[str, bool]:
        """
        Moderate AI response before showing to user
        
        Returns:
            (moderated_response, is_safe)
        """
        # Check response safety
        allow, reason = self.filter.should_allow(response)
        
        if not allow:
            safe_response = (
                "I can't provide that information as it may be harmful. "
                "I'm here to help with educational and constructive topics. "
                "How else can I assist you?"
            )
            return safe_response, False
        
        # Check if response is appropriate for prompt
        prompt_category, _, _ = self.filter.classify_content(prompt)
        response_category, _, _ = self.filter.classify_content(response)
        
        if (prompt_category == ContentCategory.QUESTIONABLE and 
            response_category != ContentCategory.EDUCATIONAL):
            # Redirect questionable prompts to educational responses
            safe_response = (
                "I understand your question. Let me provide helpful, "
                "educational information instead."
            )
            return safe_response, False
        
        return response, True
    
    def get_safety_guidelines(self) -> List[str]:
        """Return safety guidelines for the AI"""
        return self.response_rules.copy()


def test_content_filter():
    """Test the content filter"""
    print("=" * 70)
    print("TESTING CONTENT FILTER")
    print("=" * 70)
    
    filter = ContentFilter()
    
    test_cases = [
        "How do I learn Python programming?",
        "Explain machine learning algorithms",
        "How to hack into a computer tutorial",
        "Best practices for web development",
        "How to make a bomb",
        "What is artificial intelligence?",
        "Violence and destruction guide",
        "Cooking recipes for beginners",
    ]
    
    for text in test_cases:
        category, confidence, reason = filter.classify_content(text)
        allow, msg = filter.should_allow(text)
        
        print(f"\nText: {text}")
        print(f"Category: {category.value}")
        print(f"Confidence: {confidence:.2f}")
        print(f"Decision: {msg}")
    
    print("\n" + "=" * 70)
    print("✓ Content filter test complete")


if __name__ == "__main__":
    test_content_filter()
