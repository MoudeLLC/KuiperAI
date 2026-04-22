"""
Loss functions for training neural networks
"""
import numpy as np
from .tensor import Tensor


class Loss:
    """Base class for loss functions"""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)


class MSELoss(Loss):
    """Mean Squared Error Loss"""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions - targets
        return (diff * diff).mean()


class CrossEntropyLoss(Loss):
    """Cross Entropy Loss for classification"""
    
    def forward(self, logits: Tensor, targets: np.ndarray) -> Tensor:
        """
        Args:
            logits: Raw predictions, shape (batch_size, num_classes)
            targets: Target class indices, shape (batch_size,)
        """
        batch_size = logits.shape[0]
        
        # Numerical stability: subtract max
        logits_shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Negative log likelihood
        log_probs = np.log(probs[np.arange(batch_size), targets] + 1e-10)
        loss_data = -np.mean(log_probs)
        
        out = Tensor(loss_data, requires_grad=logits.requires_grad,
                    _children=(logits,), _op='CrossEntropy')
        
        def _backward():
            if logits.requires_grad:
                grad = probs.copy()
                grad[np.arange(batch_size), targets] -= 1
                grad /= batch_size
                logits.grad = logits.grad + grad if logits.grad is not None else grad
        
        out._backward = _backward
        return out


class BCELoss(Loss):
    """Binary Cross Entropy Loss"""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            predictions: Predicted probabilities (after sigmoid), shape (batch_size,)
            targets: Binary targets (0 or 1), shape (batch_size,)
        """
        # Clip for numerical stability
        preds_clipped = np.clip(predictions.data, 1e-10, 1 - 1e-10)
        
        loss_data = -np.mean(
            targets.data * np.log(preds_clipped) + 
            (1 - targets.data) * np.log(1 - preds_clipped)
        )
        
        out = Tensor(loss_data, requires_grad=predictions.requires_grad,
                    _children=(predictions, targets), _op='BCE')
        
        def _backward():
            if predictions.requires_grad:
                grad = -(targets.data / preds_clipped - 
                        (1 - targets.data) / (1 - preds_clipped))
                grad /= predictions.shape[0]
                predictions.grad = predictions.grad + grad if predictions.grad is not None else grad
        
        out._backward = _backward
        return out


class HuberLoss(Loss):
    """Huber Loss (smooth L1 loss)"""
    
    def __init__(self, delta: float = 1.0):
        self.delta = delta
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions - targets
        abs_diff = Tensor(np.abs(diff.data))
        
        # Quadratic for small errors, linear for large errors
        quadratic = Tensor(0.5 * diff.data ** 2)
        linear = Tensor(self.delta * (abs_diff.data - 0.5 * self.delta))
        
        loss_data = np.where(abs_diff.data <= self.delta, 
                            quadratic.data, linear.data)
        
        out = Tensor(np.mean(loss_data), requires_grad=predictions.requires_grad,
                    _children=(predictions, targets), _op='Huber')
        
        def _backward():
            if predictions.requires_grad:
                grad = np.where(abs_diff.data <= self.delta,
                               diff.data,
                               self.delta * np.sign(diff.data))
                grad /= predictions.shape[0]
                predictions.grad = predictions.grad + grad if predictions.grad is not None else grad
        
        out._backward = _backward
        return out
