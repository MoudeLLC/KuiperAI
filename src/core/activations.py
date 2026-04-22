"""
Activation functions for neural networks
All implemented from scratch with gradients
"""
import numpy as np
from .tensor import Tensor


class Activation:
    """Base class for activation functions"""
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class ReLU(Activation):
    """Rectified Linear Unit: f(x) = max(0, x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.maximum(0, x.data),
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='ReLU')
        
        def _backward():
            if x.requires_grad:
                grad = (x.data > 0).astype(np.float32) * out.grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out


class Sigmoid(Activation):
    """Sigmoid: f(x) = 1 / (1 + exp(-x))"""
    
    def forward(self, x: Tensor) -> Tensor:
        sigmoid_data = 1 / (1 + np.exp(-np.clip(x.data, -500, 500)))
        out = Tensor(sigmoid_data,
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='Sigmoid')
        
        def _backward():
            if x.requires_grad:
                grad = sigmoid_data * (1 - sigmoid_data) * out.grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out


class Tanh(Activation):
    """Hyperbolic tangent: f(x) = tanh(x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        tanh_data = np.tanh(x.data)
        out = Tensor(tanh_data,
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='Tanh')
        
        def _backward():
            if x.requires_grad:
                grad = (1 - tanh_data ** 2) * out.grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out


class Softmax(Activation):
    """Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))"""
    
    def forward(self, x: Tensor, axis: int = -1) -> Tensor:
        # Numerical stability: subtract max
        x_shifted = x.data - np.max(x.data, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        softmax_data = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        out = Tensor(softmax_data,
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='Softmax')
        
        def _backward():
            if x.requires_grad:
                # Jacobian of softmax
                s = softmax_data
                grad = s * out.grad
                sum_grad = np.sum(grad, axis=axis, keepdims=True)
                grad = grad - s * sum_grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out


class LeakyReLU(Activation):
    """Leaky ReLU: f(x) = x if x > 0 else alpha * x"""
    
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
    
    def forward(self, x: Tensor) -> Tensor:
        out = Tensor(np.where(x.data > 0, x.data, self.alpha * x.data),
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='LeakyReLU')
        
        def _backward():
            if x.requires_grad:
                grad = np.where(x.data > 0, 1.0, self.alpha) * out.grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out


class GELU(Activation):
    """
    Gaussian Error Linear Unit
    Approximation: f(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    
    def forward(self, x: Tensor) -> Tensor:
        # GELU approximation
        c = np.sqrt(2 / np.pi)
        inner = c * (x.data + 0.044715 * x.data ** 3)
        gelu_data = 0.5 * x.data * (1 + np.tanh(inner))
        
        out = Tensor(gelu_data,
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='GELU')
        
        def _backward():
            if x.requires_grad:
                tanh_inner = np.tanh(inner)
                sech2 = 1 - tanh_inner ** 2
                grad_inner = c * (1 + 3 * 0.044715 * x.data ** 2)
                grad = 0.5 * (1 + tanh_inner) + 0.5 * x.data * sech2 * grad_inner
                grad = grad * out.grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out


class Swish(Activation):
    """Swish (SiLU): f(x) = x * sigmoid(x)"""
    
    def forward(self, x: Tensor) -> Tensor:
        sigmoid_data = 1 / (1 + np.exp(-np.clip(x.data, -500, 500)))
        swish_data = x.data * sigmoid_data
        
        out = Tensor(swish_data,
                    requires_grad=x.requires_grad,
                    _children=(x,), _op='Swish')
        
        def _backward():
            if x.requires_grad:
                grad = sigmoid_data + swish_data * (1 - sigmoid_data)
                grad = grad * out.grad
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out
