#!/usr/bin/env python3
"""
Content moderation filter for KuiperAI
Filters training data and provides inference-time moderation
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

class ContentModerator:
    """Handles content moderation for training and inference"""
    
    def __init__(self, config_path: str = "train/pre/content_moderation.json"):
        """Load moderation configuration"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.enabled = self.config['moderation_config']['enabled']
        
        # Build word lists
        self.mild_words = set(self.config['profanity_filter']['mild'])
        self.moderate_words = set(self.config['profanity_filter']['moderate'])
        self.severe_words = set(self.config['profanity_filter']['severe'])
        
        # Harmful content patterns
        self.harmful_patterns = []
        for category in self.config['harmful_content'].values():
            self.harmful_patterns.extend(category)
        
        # Response templates
        self.responses = self.config['response_templates']
    
    def check_text(self, text: str) -> Dict:
        """
        Check text for problematic content
        
        Returns:
            {
                'is_clean': bool,
                'severity': str,
                'issues': list,
                'action': str
            }
        """
        if not self.enabled:
            return {'is_clean': True, 'severity': None, 'issues': [], 'action': 'allow'}
        
        text_lower = text.lower()
        issues = []
        max_severity = 0
        
        # Check profanity
        for word in self.severe_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                issues.append(('severe_profanity', word))
                max_severity = max(max_severity, 3)
        
        for word in self.moderate_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                issues.append(('moderate_profanity', word))
                max_severity = max(max_severity, 2)
        
        for word in self.mild_words:
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                issues.append(('mild_profanity', word))
                max_severity = max(max_severity, 1)
        
        # Check harmful content
        for pattern in self.harmful_patterns:
            if pattern.lower() in text_lower:
                issues.append(('harmful_content', pattern))
                max_severity = max(max_severity, 4)
        
        # Determine action
        if max_severity == 0:
            action = 'allow'
        elif max_severity == 1:
            action = 'allow_with_warning'
        elif max_severity == 2:
            action = 'filter_from_training'
        elif max_severity == 3:
            action = 'block_and_refuse'
        else:
            action = 'block_and_log'
        
        return {
            'is_clean': max_severity == 0,
            'severity': max_severity,
            'issues': issues,
            'action': action
        }
    
    def should_filter_for_training(self, text: str) -> Tuple[bool, str]:
        """
        Determine if text should be filtered from training data
        
        Returns:
            (should_filter: bool, reason: str)
        """
        result = self.check_text(text)
        
        # Count profanity
        profanity_count = len([i for i in result['issues'] if 'profanity' in i[0]])
        
        # Filter if too much profanity
        min_count = self.config['training_filters']['remove_documents_containing']['min_profanity_count']
        if profanity_count >= min_count:
            return True, f"Contains {profanity_count} profane words (threshold: {min_count})"
        
        # Filter harmful content
        if result['severity'] >= 4:
            return True, "Contains harmful content"
        
        # Filter severe profanity
        if result['severity'] >= 3:
            return True, "Contains severe profanity"
        
        return False, "Clean"
    
    def get_polite_response(self, user_message: str) -> str:
        """
        Generate polite response if user uses profanity
        
        Returns:
            Response string or None if no intervention needed
        """
        result = self.check_text(user_message)
        
        if not self.config['inference_behavior']['detect_user_profanity']['enabled']:
            return None
        
        if result['severity'] >= 3:
            return self.responses['profanity_detected']
        elif result['severity'] == 4:
            return self.responses['harmful_content']
        
        return None
    
    def clean_text_for_training(self, text: str) -> str:
        """
        Clean text by replacing profanity if configured
        
        Returns:
            Cleaned text
        """
        if not self.config['training_filters']['replace_profanity']['enabled']:
            return text
        
        replacement = self.config['training_filters']['replace_profanity']['replacement']
        
        # Replace profanity
        cleaned = text
        for word in self.severe_words | self.moderate_words:
            pattern = r'\b' + re.escape(word) + r'\b'
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        return cleaned


def filter_training_file(input_file: str, output_file: str, moderator: ContentModerator):
    """
    Filter a training data file, removing problematic content
    
    Args:
        input_file: Path to input text file
        output_file: Path to output filtered file
        moderator: ContentModerator instance
    """
    print(f"Filtering: {input_file}")
    print(f"Output: {output_file}")
    print()
    
    total_docs = 0
    filtered_docs = 0
    clean_docs = 0
    
    with open(input_file, 'r', encoding='utf-8') as inf, \
         open(output_file, 'w', encoding='utf-8') as outf:
        
        current_doc = []
        
        for line in inf:
            line = line.strip()
            
            if not line:  # Empty line = document separator
                if current_doc:
                    doc_text = '\n'.join(current_doc)
                    total_docs += 1
                    
                    should_filter, reason = moderator.should_filter_for_training(doc_text)
                    
                    if should_filter:
                        filtered_docs += 1
                        if total_docs % 1000 == 0:
                            print(f"Filtered doc {total_docs}: {reason}")
                    else:
                        # Write clean document
                        outf.write(doc_text + '\n\n')
                        clean_docs += 1
                    
                    current_doc = []
                    
                    if total_docs % 10000 == 0:
                        print(f"Processed {total_docs:,} docs, filtered {filtered_docs:,} ({filtered_docs/total_docs*100:.1f}%)")
            else:
                current_doc.append(line)
        
        # Handle last document
        if current_doc:
            doc_text = '\n'.join(current_doc)
            total_docs += 1
            should_filter, _ = moderator.should_filter_for_training(doc_text)
            if not should_filter:
                outf.write(doc_text + '\n\n')
                clean_docs += 1
            else:
                filtered_docs += 1
    
    print()
    print("=" * 70)
    print("FILTERING COMPLETE")
    print("=" * 70)
    print(f"Total documents: {total_docs:,}")
    print(f"Clean documents: {clean_docs:,} ({clean_docs/total_docs*100:.1f}%)")
    print(f"Filtered documents: {filtered_docs:,} ({filtered_docs/total_docs*100:.1f}%)")
    print(f"Output file: {output_file}")
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 content_filter.py <input_file> <output_file>")
        print()
        print("Example:")
        print("  python3 train/pre/content_filter.py train/pre/data/webtext_medium.txt train/pre/data/webtext_medium_filtered.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    moderator = ContentModerator()
    filter_training_file(input_file, output_file, moderator)
