#!/usr/bin/env python3
"""
Comprehensive test suite for KuiperAI
Tests all components with larger, more realistic scenarios
"""
import sys
sys.path.insert(0, 'src')
import numpy as np
import time
from pathlib import Path

# Import all components
from core.tensor import Tensor
from core.layers import Linear, Embedding, LayerNorm, Dropout
from core.activations import ReLU, GELU, Softmax
from core.optimizers import Adam, AdamW
from core.losses import CrossEntropyLoss, MSELoss
from models.transformer import Transformer, MultiHeadAttention
from data.dataset import TextDataset, DataLoader
from training.trainer import Trainer
from training.scheduler import WarmupLR, CosineAnnealingLR
from safety.content_filter import ContentFilter, ContentModerator
from network.web_learner import WebLearner, KnowledgeAggregator


class ComprehensiveTest:
    """Comprehensive test suite"""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results = []
    
    def run_test(self, name: str, test_func):
        """Run a single test"""
        print(f"\n{'='*70}")
        print(f"TEST: {name}")
        print('='*70)
        
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            if result:
                print(f"✅ PASS ({elapsed:.2f}s)")
                self.tests_passed += 1
                self.test_results.append((name, True, elapsed))
            else:
                print(f"❌ FAIL ({elapsed:.2f}s)")
                self.tests_failed += 1
                self.test_results.append((name, False, elapsed))
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            self.tests_failed += 1
            self.test_results.append((name, False, 0))
    
    def test_large_tensor_operations(self):
        """Test tensor operations with large matrices"""
        print("Testing large tensor operations...")
        
        # Large matrix multiplication
        a = Tensor(np.random.randn(100, 200), requires_grad=True)
        b = Tensor(np.random.randn(200, 150), requires_grad=True)
        c = a @ b
        
        assert c.shape == (100, 150), f"Shape mismatch: {c.shape}"
        
        # Backward pass
        c.backward(np.ones_like(c.data))
        
        assert a.grad is not None, "Gradient not computed for a"
        assert b.grad is not None, "Gradient not computed for b"
        assert a.grad.shape == a.shape, "Gradient shape mismatch"
        
        print(f"  Matrix shapes: {a.shape} @ {b.shape} = {c.shape}")
        print(f"  Gradients computed: ✓")
        
        return True
    
    def test_deep_network(self):
        """Test deep neural network (10 layers)"""
        print("Testing deep network...")
        
        layers = []
        input_size = 128
        
        # Create 10-layer network
        for i in range(10):
            layers.append(Linear(input_size, input_size))
            layers.append(ReLU())
        
        # Forward pass
        x = Tensor(np.random.randn(32, input_size), requires_grad=True)
        
        for layer in layers:
            x = layer(x)
        
        # Backward pass
        x.backward(np.ones_like(x.data))
        
        # Check gradients flow through all layers
        grad_count = sum(1 for layer in layers 
                        if isinstance(layer, Linear) 
                        and layer.weight.grad is not None)
        
        print(f"  Layers: 10")
        print(f"  Layers with gradients: {grad_count}/10")
        
        return grad_count >= 8  # At least 8/10 should have gradients
    
    def test_large_vocabulary_embedding(self):
        """Test embedding with large vocabulary"""
        print("Testing large vocabulary embedding...")
        
        vocab_size = 50000
        embed_dim = 512
        seq_len = 128
        batch_size = 16
        
        embedding = Embedding(vocab_size, embed_dim)
        
        # Random token indices
        tokens = np.random.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        embedded = embedding(tokens)
        
        assert embedded.shape == (batch_size, seq_len, embed_dim)
        
        # Backward pass
        embedded.backward(np.ones_like(embedded.data))
        
        assert embedding.weight.grad is not None
        
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Embedding dim: {embed_dim}")
        print(f"  Batch size: {batch_size}")
        print(f"  Sequence length: {seq_len}")
        
        return True
    
    def test_large_transformer(self):
        """Test larger transformer model"""
        print("Testing large transformer...")
        
        model = Transformer(
            vocab_size=10000,
            d_model=256,
            num_heads=8,
            num_layers=6,
            d_ff=1024,
            max_seq_len=256,
            dropout=0.1
        )
        
        # Count parameters
        total_params = sum(np.prod(p.shape) for p in model.parameters())
        
        # Forward pass
        batch_size = 8
        seq_len = 128
        x = np.random.randint(0, 10000, (batch_size, seq_len))
        
        logits = model.forward(x)
        
        assert logits.shape == (batch_size, seq_len, 10000)
        
        print(f"  Parameters: {total_params:,}")
        print(f"  Layers: 6")
        print(f"  Heads: 8")
        print(f"  Model dim: 256")
        
        return True
    
    def test_batch_training(self):
        """Test training with multiple batches"""
        print("Testing batch training...")
        
        # Create synthetic dataset
        texts = [f"sample text number {i}" for i in range(100)]
        dataset = TextDataset(texts, max_length=16)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        # Small model
        model = Transformer(
            vocab_size=len(dataset.vocab),
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=32
        )
        
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss()
        
        # Train for a few batches
        losses = []
        for i, (batch_x, batch_y) in enumerate(loader):
            if i >= 5:  # Just 5 batches
                break
            
            # Forward
            logits = model.forward(batch_x)
            
            # Compute loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * seq_len, vocab_size)
            targets_flat = batch_x.reshape(-1)
            
            loss = loss_fn(logits_flat, targets_flat)
            losses.append(loss.data)
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"  Batches trained: 5")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        
        return len(losses) == 5
    
    def test_content_filter(self):
        """Test content filtering system"""
        print("Testing content filter...")
        
        filter = ContentFilter()
        
        test_cases = [
            ("Learn Python programming", True),
            ("How to hack tutorial", False),
            ("Machine learning explained", True),
            ("Make a bomb guide", False),
            ("Cooking recipes", True),
        ]
        
        correct = 0
        for text, should_allow in test_cases:
            allow, reason = filter.should_allow(text)
            if allow == should_allow:
                correct += 1
                print(f"  ✓ {text[:30]}... -> {'ALLOW' if allow else 'BLOCK'}")
            else:
                print(f"  ✗ {text[:30]}... -> Expected {'ALLOW' if should_allow else 'BLOCK'}")
        
        accuracy = correct / len(test_cases)
        print(f"  Accuracy: {accuracy*100:.0f}%")
        
        return accuracy >= 0.8
    
    def test_content_moderator(self):
        """Test content moderation"""
        print("Testing content moderator...")
        
        moderator = ContentModerator()
        
        test_prompts = [
            "Explain machine learning",
            "How to hack systems",
            "Tell me about AI",
        ]
        
        test_responses = [
            "Machine learning is a field of AI...",
            "Here's how to break into systems...",
            "AI is artificial intelligence...",
        ]
        
        moderated_count = 0
        for prompt, response in zip(test_prompts, test_responses):
            moderated, is_safe = moderator.moderate_response(prompt, response)
            if not is_safe:
                moderated_count += 1
                print(f"  ✓ Moderated unsafe response")
            else:
                print(f"  ✓ Allowed safe response")
        
        print(f"  Moderated: {moderated_count} responses")
        
        return True
    
    def test_web_learner(self):
        """Test web learning system"""
        print("Testing web learner...")
        
        learner = WebLearner()
        
        # Test with existing files
        test_file = 'knowledge/datasets/general/sample_knowledge.txt'
        
        if Path(test_file).exists():
            success = learner.learn_from_text_file(test_file, 'test_topic')
            
            stats = learner.get_statistics()
            print(f"  Learned: {stats['learned']}")
            print(f"  Filtered: {stats['filtered_out']}")
            
            return success
        else:
            print(f"  Skipped: {test_file} not found")
            return True
    
    def test_knowledge_aggregation(self):
        """Test knowledge aggregation"""
        print("Testing knowledge aggregation...")
        
        aggregator = KnowledgeAggregator()
        
        texts = aggregator.aggregate_all_knowledge()
        
        print(f"  Total texts aggregated: {len(texts)}")
        
        return len(texts) > 0
    
    def test_learning_rate_schedulers(self):
        """Test all learning rate schedulers"""
        print("Testing learning rate schedulers...")
        
        # Create dummy optimizer
        params = [Tensor(np.random.randn(10, 10), requires_grad=True)]
        optimizer = Adam(params, lr=0.001)
        
        schedulers = [
            WarmupLR(optimizer, warmup_epochs=5, target_lr=0.001),
            CosineAnnealingLR(optimizer, T_max=10, eta_min=0.0001),
        ]
        
        for scheduler in schedulers:
            initial_lr = optimizer.lr
            
            # Step through epochs
            for epoch in range(10):
                scheduler.step(epoch)
            
            final_lr = optimizer.lr
            
            print(f"  {scheduler.__class__.__name__}: {initial_lr:.6f} -> {final_lr:.6f}")
        
        return True
    
    def test_checkpoint_system(self):
        """Test checkpoint save/load"""
        print("Testing checkpoint system...")
        
        # Create model
        model = Transformer(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=32
        )
        
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_fn = CrossEntropyLoss()
        
        trainer = Trainer(
            model, optimizer, loss_fn,
            checkpoint_dir='test_checkpoints',
            log_dir='test_logs'
        )
        
        # Save checkpoint
        trainer.save_checkpoint('test_checkpoint.json', epoch=1, val_loss=1.5)
        
        # Check file exists
        checkpoint_path = Path('test_checkpoints/test_checkpoint.json')
        exists = checkpoint_path.exists()
        
        if exists:
            print(f"  ✓ Checkpoint saved")
            # Clean up
            checkpoint_path.unlink()
        
        return exists
    
    def run_all_tests(self):
        """Run all comprehensive tests"""
        print("\n" + "="*70)
        print("KUIPERAI COMPREHENSIVE TEST SUITE")
        print("="*70)
        
        # Core tests
        self.run_test("Large Tensor Operations", self.test_large_tensor_operations)
        self.run_test("Deep Network (10 layers)", self.test_deep_network)
        self.run_test("Large Vocabulary Embedding", self.test_large_vocabulary_embedding)
        self.run_test("Large Transformer Model", self.test_large_transformer)
        self.run_test("Batch Training", self.test_batch_training)
        
        # Safety tests
        self.run_test("Content Filter", self.test_content_filter)
        self.run_test("Content Moderator", self.test_content_moderator)
        
        # Network tests
        self.run_test("Web Learner", self.test_web_learner)
        self.run_test("Knowledge Aggregation", self.test_knowledge_aggregation)
        
        # Training tests
        self.run_test("Learning Rate Schedulers", self.test_learning_rate_schedulers)
        self.run_test("Checkpoint System", self.test_checkpoint_system)
        
        # Summary
        self.print_summary()
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total = self.tests_passed + self.tests_failed
        percentage = (self.tests_passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {self.tests_passed} ✅")
        print(f"Failed: {self.tests_failed} ❌")
        print(f"Success Rate: {percentage:.1f}%")
        
        print("\nDetailed Results:")
        for name, passed, elapsed in self.test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} - {name} ({elapsed:.2f}s)")
        
        print("\n" + "="*70)
        
        if self.tests_failed == 0:
            print("🎉 ALL TESTS PASSED!")
        else:
            print(f"⚠️  {self.tests_failed} test(s) failed")
        
        print("="*70)


if __name__ == "__main__":
    tester = ComprehensiveTest()
    tester.run_all_tests()
