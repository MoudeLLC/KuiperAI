"""
Learning rate schedulers for training
"""
import numpy as np
from typing import Optional


class LRScheduler:
    """Base class for learning rate schedulers"""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.lr
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate"""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        new_lr = self.get_lr()
        self.optimizer.lr = new_lr
        return new_lr
    
    def get_lr(self):
        """Calculate new learning rate"""
        raise NotImplementedError


class StepLR(LRScheduler):
    """Decays learning rate by gamma every step_size epochs"""
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """Decays learning rate exponentially"""
    
    def __init__(self, optimizer, gamma: float, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate schedule"""
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * \
               (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2


class WarmupLR(LRScheduler):
    """Linear warmup followed by constant or decay"""
    
    def __init__(self, optimizer, warmup_epochs: int, target_lr: Optional[float] = None,
                 last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.target_lr = target_lr if target_lr is not None else optimizer.lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return self.target_lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            return self.target_lr


class ReduceLROnPlateau:
    """Reduce learning rate when metric plateaus"""
    
    def __init__(self, optimizer, mode: str = 'min', factor: float = 0.1,
                 patience: int = 10, threshold: float = 1e-4,
                 min_lr: float = 0, verbose: bool = True):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.verbose = verbose
        
        self.best = None
        self.num_bad_epochs = 0
        self.last_epoch = 0
        
        if mode == 'min':
            self.mode_worse = float('inf')
            self.is_better = lambda a, b: a < b - threshold
        else:
            self.mode_worse = -float('inf')
            self.is_better = lambda a, b: a > b + threshold
    
    def step(self, metric: float):
        """Update learning rate based on metric"""
        self.last_epoch += 1
        
        if self.best is None:
            self.best = metric
        elif self.is_better(metric, self.best):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
    
    def _reduce_lr(self):
        """Reduce learning rate"""
        old_lr = self.optimizer.lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        
        if new_lr < old_lr:
            self.optimizer.lr = new_lr
            if self.verbose:
                print(f"Reducing learning rate: {old_lr:.6f} -> {new_lr:.6f}")


class OneCycleLR(LRScheduler):
    """One cycle learning rate policy"""
    
    def __init__(self, optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, div_factor: float = 25.0,
                 final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.final_lr = self.initial_lr / final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.step_size_up:
            # Warmup phase
            return self.initial_lr + (self.max_lr - self.initial_lr) * \
                   self.last_epoch / self.step_size_up
        else:
            # Annealing phase
            progress = (self.last_epoch - self.step_size_up) / self.step_size_down
            return self.max_lr - (self.max_lr - self.final_lr) * progress
