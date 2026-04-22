#!/usr/bin/env python3
"""
KuiperAI Autonomous Learning Daemon
Runs in background, continuously learning and training
"""
import sys
sys.path.insert(0, 'src')
import json
import time
import yaml
import os
from pathlib import Path
from datetime import datetime
import subprocess

print("=" * 70)
print("KUIPERAI AUTONOMOUS LEARNING DAEMON")
print("Background Learning & Training System")
print("=" * 70)

class AutonomousLearner:
    def __init__(self, config_path='configs/autonomous_learning.yaml'):
        self.config = self.load_config(config_path)
        self.stats = {
            'cycles_completed': 0,
            'total_vocab': 0,
            'total_knowledge': 0,
            'total_definitions': 0,
            'trainings_completed': 0,
            'start_time': datetime.now().isoformat(),
            'last_notification': None,
            'notifications_sent': 0
        }
        self.log_file = self.config['autonomous_learning']['output']['log_file']
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
    
    def load_config(self, path):
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def send_notification(self, message):
        """Send notification"""
        self.log(f"📢 NOTIFICATION: {message}")
        self.stats['notifications_sent'] += 1
        self.stats['last_notification'] = datetime.now().isoformat()
        
        # In production, integrate with email/SMS/webhook
        notification_file = 'notifications.txt'
        with open(notification_file, 'a') as f:
            f.write(f"[{datetime.now()}] {message}\n")
    
    def run_research_cycle(self):
        """Run one research cycle"""
        self.log("Starting research cycle...")
        
        try:
            # Import vocab ecosystem
            from vocab_ecosystem import VocabEcosystem
            
            ecosystem = VocabEcosystem()
            
            # Research configured number of words
            words_per_cycle = self.config['autonomous_learning']['research']['words_per_cycle']
            
            researched_count = 0
            unresearched = [w for w in ecosystem.vocab['words'] if w not in ecosystem.researched]
            
            if not unresearched and not ecosystem.vocab['words']:
                # Initialize with seed words
                seed_words = [
                    'machine', 'learning', 'neural', 'network', 'python',
                    'language', 'grammar', 'syntax', 'artificial', 'intelligence',
                    'algorithm', 'data', 'science', 'computer', 'programming'
                ]
                ecosystem.vocab['words'].extend(seed_words)
                unresearched = seed_words
            
            for word in unresearched[:words_per_cycle]:
                new_words = ecosystem.research_word(word)
                researched_count += 1
                self.log(f"  ✓ Researched '{word}', found {len(new_words)} new words")
            
            # Save progress
            ecosystem.save_all()
            
            # Update stats
            self.stats['cycles_completed'] += 1
            self.stats['total_vocab'] = len(ecosystem.vocab['words'])
            self.stats['total_knowledge'] = ecosystem.stats['total_knowledge']
            self.stats['total_definitions'] = len(ecosystem.vocab['definitions'])
            
            self.log(f"Cycle complete. Vocab: {self.stats['total_vocab']}, Definitions: {self.stats['total_definitions']}")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Error in research cycle: {e}")
            return False
    
    def should_train(self):
        """Check if we should trigger training"""
        train_every = self.config['autonomous_learning']['training']['train_every']
        return self.stats['cycles_completed'] % train_every == 0
    
    def run_training(self):
        """Run model training"""
        self.log("Starting model training...")
        
        try:
            # Run training script
            result = subprocess.run(
                ['python3', 'train_improved.py'],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                self.stats['trainings_completed'] += 1
                self.log("✓ Training completed successfully")
                return True
            else:
                self.log(f"❌ Training failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error in training: {e}")
            return False
    
    def should_notify(self):
        """Check if we should send notification"""
        notify_every = self.config['autonomous_learning']['notifications']['notify_every']
        return self.stats['cycles_completed'] % notify_every == 0
    
    def run(self):
        """Main daemon loop"""
        config = self.config['autonomous_learning']
        
        if not config['enabled']:
            self.log("Autonomous learning is disabled in config")
            return
        
        max_runs = config['schedule']['max_runs']
        interval_minutes = config['schedule']['interval_minutes']
        
        self.log("=" * 70)
        self.log("AUTONOMOUS LEARNING STARTED")
        self.log("=" * 70)
        self.log(f"Max runs: {max_runs}")
        self.log(f"Interval: {interval_minutes} minutes")
        self.log(f"Auto-train: {config['training']['auto_train']}")
        
        try:
            while self.stats['cycles_completed'] < max_runs:
                cycle_start = time.time()
                
                self.log(f"\n{'='*70}")
                self.log(f"CYCLE {self.stats['cycles_completed'] + 1}/{max_runs}")
                self.log(f"{'='*70}")
                
                # Run research cycle
                success = self.run_research_cycle()
                
                if success:
                    # Check if we should train
                    if config['training']['auto_train'] and self.should_train():
                        self.run_training()
                    
                    # Check if we should notify
                    if config['notifications']['enabled'] and self.should_notify():
                        self.send_notification(
                            f"Completed {self.stats['cycles_completed']} cycles. "
                            f"Vocab: {self.stats['total_vocab']}, "
                            f"Definitions: {self.stats['total_definitions']}, "
                            f"Knowledge: {self.stats['total_knowledge']}"
                        )
                
                # Save stats
                self.save_stats()
                
                # Calculate sleep time
                cycle_duration = time.time() - cycle_start
                sleep_time = max(0, (interval_minutes * 60) - cycle_duration)
                
                if self.stats['cycles_completed'] < max_runs:
                    self.log(f"Sleeping for {sleep_time/60:.1f} minutes...")
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            self.log("\n⏸️  Autonomous learning stopped by user")
        
        # Final notification
        if config['notifications']['notify_at_completion']:
            self.send_notification(
                f"Autonomous learning completed! "
                f"Total cycles: {self.stats['cycles_completed']}, "
                f"Vocab: {self.stats['total_vocab']}, "
                f"Definitions: {self.stats['total_definitions']}, "
                f"Trainings: {self.stats['trainings_completed']}"
            )
        
        self.log("=" * 70)
        self.log("AUTONOMOUS LEARNING COMPLETE")
        self.log("=" * 70)
        self.log(f"Total cycles: {self.stats['cycles_completed']}")
        self.log(f"Total vocabulary: {self.stats['total_vocab']}")
        self.log(f"Total definitions: {self.stats['total_definitions']}")
        self.log(f"Total trainings: {self.stats['trainings_completed']}")
        self.log(f"Notifications sent: {self.stats['notifications_sent']}")
    
    def save_stats(self):
        """Save statistics"""
        stats_file = 'knowledge/autonomous_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

def main():
    """Main entry point"""
    print("\nKuiperAI Autonomous Learning Daemon")
    print("This will run continuously in the background")
    print("\nPress Ctrl+C to stop\n")
    
    learner = AutonomousLearner()
    learner.run()

if __name__ == '__main__':
    main()
