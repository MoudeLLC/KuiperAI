"""
Optimization algorithms for training neural networks
"""
import numpy as np
from typing import List
from .tensor import Tensor


class Optimizer:
    """Base class for optimizers"""
    
    def __init__(self, parameters: List[Tensor], lr: float):
        self.parameters = parameters
        self.lr = lr
    
    def zero_grad(self):
        """Reset all gradients to zero"""
        for param in self.parameters:
            param.zero_grad()
    
    def step(self):
        """Update parameters"""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with momentum"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01, 
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Momentum
            if self.momentum > 0:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * grad
                param.data += self.velocities[i]
            else:
                param.data -= self.lr * grad


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # First and second moment estimates
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """AdamW optimizer (Adam with decoupled weight decay)"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.001,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.01):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters with decoupled weight decay
            param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + 
                                    self.weight_decay * param.data)


class RMSprop(Optimizer):
    """RMSprop optimizer"""
    
    def __init__(self, parameters: List[Tensor], lr: float = 0.01,
                 alpha: float = 0.99, eps: float = 1e-8,
                 weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.v = [np.zeros_like(p.data) for p in parameters]
    
    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param.data
            
            # Update moving average of squared gradients
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            
            # Update parameters
            param.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)
