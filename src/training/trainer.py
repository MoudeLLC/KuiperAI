"""
Training infrastructure for KuiperAI models
"""
import numpy as np
import json
import os
from typing import Dict, List, Optional, Callable
from datetime import datetime
import sys
sys.path.append('..')
from core.tensor import Tensor
from core.losses import Loss
from core.optimizers import Optimizer


class Trainer:
    """Handles model training with checkpointing and logging"""
    
    def __init__(self, model, optimizer: Optimizer, loss_fn: Loss,
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs',
                 gradient_clip_value: Optional[float] = None,
                 gradient_clip_norm: Optional[float] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        self.gradient_clip_value = gradient_clip_value
        self.gradient_clip_norm = gradient_clip_norm
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.epoch = 0
    
    def clip_gradients(self):
        """Clip gradients to prevent exploding gradients"""
        if self.gradient_clip_value is not None:
            # Clip by value
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = np.clip(param.grad, -self.gradient_clip_value, 
                                        self.gradient_clip_value)
        
        if self.gradient_clip_norm is not None:
            # Clip by global norm
            total_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    total_norm += np.sum(param.grad ** 2)
            total_norm = np.sqrt(total_norm)
            
            if total_norm > self.gradient_clip_norm:
                clip_coef = self.gradient_clip_norm / (total_norm + 1e-6)
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad *= clip_coef
    
    def train_epoch(self, train_loader, val_loader=None, scheduler=None) -> Dict[str, float]:
        """Train for one epoch"""
        train_losses = []
        train_accs = []
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs = self.model.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            self.clip_gradients()
            
            # Optimizer step
            self.optimizer.step()
            
            # Track metrics
            train_losses.append(loss.data.item() if hasattr(loss.data, 'item') else float(loss.data))
            
            # Calculate accuracy - skip for language modeling
            # (targets are dummy labels, not actual next tokens)
            # Accuracy will be measured by loss reduction instead
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {train_losses[-1]:.4f}")
        
        metrics = {
            'train_loss': np.mean(train_losses),
            'train_acc': np.mean(train_accs) if train_accs else 0.0,
            'learning_rate': self.optimizer.lr
        }
        
        # Learning rate scheduling
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if isinstance(scheduler, type(scheduler)) and hasattr(scheduler, 'patience'):
                    # ReduceLROnPlateau needs validation loss
                    pass  # Will be called after validation
                else:
                    scheduler.step()
        
        # Validation
        if val_loader is not None:
            val_metrics = self.validate(val_loader)
            metrics.update(val_metrics)
            
            # Update scheduler with validation loss if needed
            if scheduler is not None and hasattr(scheduler, 'patience'):
                scheduler.step(val_metrics['val_loss'])
        
        return metrics
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model"""
        val_losses = []
        val_accs = []
        
        for inputs, targets in val_loader:
            outputs = self.model.forward(inputs)
            loss = self.loss_fn(outputs, targets)
            
            val_losses.append(loss.data.item() if hasattr(loss.data, 'item') else loss.data)
            
            # Skip accuracy for language modeling
            # (targets are dummy labels, not actual next tokens)
        
        return {
            'val_loss': np.mean(val_losses),
            'val_acc': np.mean(val_accs) if val_accs else 0.0
        }
    
    def fit(self, train_loader, val_loader=None, epochs: int = 10,
            early_stopping_patience: int = 5, scheduler=None):
        """
        Train the model for multiple epochs
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            early_stopping_patience: Stop if no improvement for N epochs
            scheduler: Learning rate scheduler
        """
        patience_counter = 0
        
        for epoch in range(epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 50)
            
            # Train
            metrics = self.train_epoch(train_loader, val_loader, scheduler)
            
            # Update history
            for key, value in metrics.items():
                if key in self.history:
                    self.history[key].append(value)
            
            # Print metrics
            print(f"\nTrain Loss: {metrics['train_loss']:.4f}")
            if 'train_acc' in metrics:
                print(f"Train Acc: {metrics['train_acc']:.4f}")
            print(f"Learning Rate: {metrics['learning_rate']:.6f}")
            
            if val_loader is not None:
                print(f"Val Loss: {metrics['val_loss']:.4f}")
                if 'val_acc' in metrics:
                    print(f"Val Acc: {metrics['val_acc']:.4f}")
                
                # Save best model
                if metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = metrics['val_loss']
                    self.save_checkpoint('best_model.json')
                    patience_counter = 0
                    print("✓ New best model saved!")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.json')
        
        # Save final model
        self.save_checkpoint('final_model.json')
        self.save_history()
        
        print("\nTraining complete!")
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state': self._get_model_state(),
            'optimizer_state': self._get_optimizer_state(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, default=self._numpy_to_list)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
        
        print(f"Loading checkpoint from {filepath}")
        
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        self._set_model_state(checkpoint['model_state'])
        self._set_optimizer_state(checkpoint['optimizer_state'])
        
        print(f"✓ Checkpoint loaded: epoch {self.epoch}, best_val_loss {self.best_val_loss:.4f}")
    
    def save_history(self):
        """Save training history"""
        filepath = os.path.join(self.log_dir, f'history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def _get_model_state(self) -> Dict:
        """Get model parameters as dictionary"""
        state = {}
        for i, param in enumerate(self.model.parameters()):
            state[f'param_{i}'] = param.data.tolist()
        return state
    
    def _set_model_state(self, state: Dict):
        """Set model parameters from dictionary"""
        params = self.model.parameters()
        
        if len(params) != len([k for k in state.keys() if k.startswith('param_')]):
            raise ValueError(f"Checkpoint has {len([k for k in state.keys() if k.startswith('param_')])} parameters, "
                           f"but model has {len(params)} parameters")
        
        for i, param in enumerate(params):
            if f'param_{i}' not in state:
                raise KeyError(f"Parameter param_{i} not found in checkpoint")
            
            loaded_data = np.array(state[f'param_{i}'], dtype=np.float32)
            
            if loaded_data.shape != param.data.shape:
                raise ValueError(f"Shape mismatch for param_{i}: "
                               f"checkpoint has {loaded_data.shape}, model has {param.data.shape}")
            
            param.data = loaded_data
            
        print(f"✓ Loaded {len(params)} parameters")
    
    def _get_optimizer_state(self) -> Dict:
        """Get optimizer state"""
        state = {
            'lr': self.optimizer.lr,
            'type': self.optimizer.__class__.__name__
        }
        
        # Save optimizer-specific state
        if hasattr(self.optimizer, 'velocities'):
            state['velocities'] = [v.tolist() for v in self.optimizer.velocities]
        
        if hasattr(self.optimizer, 'm'):
            state['m'] = [m.tolist() for m in self.optimizer.m]
        
        if hasattr(self.optimizer, 'v'):
            state['v'] = [v.tolist() for v in self.optimizer.v]
        
        if hasattr(self.optimizer, 't'):
            state['t'] = self.optimizer.t
        
        return state
    
    def _set_optimizer_state(self, state: Dict):
        """Set optimizer state"""
        self.optimizer.lr = state['lr']
        
        # Restore optimizer-specific state (momentum, Adam moments, etc.)
        if 'velocities' in state and hasattr(self.optimizer, 'velocities'):
            self.optimizer.velocities = [np.array(v, dtype=np.float32) for v in state['velocities']]
        
        if 'm' in state and hasattr(self.optimizer, 'm'):
            self.optimizer.m = [np.array(m, dtype=np.float32) for m in state['m']]
        
        if 'v' in state and hasattr(self.optimizer, 'v'):
            self.optimizer.v = [np.array(v, dtype=np.float32) for v in state['v']]
        
        if 't' in state and hasattr(self.optimizer, 't'):
            self.optimizer.t = state['t']
    
    @staticmethod
    def _numpy_to_list(obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
