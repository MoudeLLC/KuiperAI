"""
Neural network layers built from scratch
"""
import numpy as np
from typing import Optional, Tuple
from .tensor import Tensor
from .activations import Activation


class Layer:
    """Base class for neural network layers"""
    
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
    
    def parameters(self):
        """Return list of trainable parameters"""
        return []
    
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Layer):
    """Fully connected linear layer: y = xW + b"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            # Broadcast bias correctly
            bias_broadcasted = Tensor(
                np.broadcast_to(self.bias.data, out.shape),
                requires_grad=self.bias.requires_grad,
                _children=(self.bias,), _op='broadcast_bias'
            )
            
            def _backward_bias():
                if self.bias.requires_grad:
                    # Sum over all dimensions except last
                    grad = np.sum(bias_broadcasted.grad, axis=tuple(range(len(bias_broadcasted.shape) - 1)))
                    self.bias.grad = self.bias.grad + grad if self.bias.grad is not None else grad
            
            bias_broadcasted._backward = _backward_bias
            out = out + bias_broadcasted
        return out
    
    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params


class Embedding(Layer):
    """Embedding layer for discrete tokens"""
    
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        self.weight = Tensor(
            np.random.randn(num_embeddings, embedding_dim) * 0.01,
            requires_grad=True
        )
    
    def forward(self, indices) -> Tensor:
        """
        Args:
            indices: Array of token indices, shape (batch_size, seq_len)
                    Can be numpy array or will be converted
        Returns:
            Embedded vectors, shape (batch_size, seq_len, embedding_dim)
        """
        # Convert to numpy if needed
        if isinstance(indices, Tensor):
            indices = indices.data
        if not isinstance(indices, np.ndarray):
            indices = np.array(indices, dtype=np.int32)
        
        embedded = self.weight.data[indices]
        out = Tensor(embedded, requires_grad=True, _children=(self.weight,), _op='Embedding')
        
        def _backward():
            if self.weight.requires_grad:
                grad = np.zeros_like(self.weight.data)
                np.add.at(grad, indices, out.grad)
                self.weight.grad = self.weight.grad + grad if self.weight.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def parameters(self):
        return [self.weight]


class LayerNorm(Layer):
    """Layer Normalization"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.gamma = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.beta = Tensor(np.zeros(normalized_shape), requires_grad=True)
    
    def forward(self, x: Tensor) -> Tensor:
        mean = x.data.mean(axis=-1, keepdims=True)
        var = x.data.var(axis=-1, keepdims=True)
        
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.gamma.data * x_norm + self.beta.data
        
        out = Tensor(out_data, requires_grad=x.requires_grad,
                    _children=(x, self.gamma, self.beta), _op='LayerNorm')
        
        def _backward():
            N = self.normalized_shape
            
            if x.requires_grad:
                dx_norm = out.grad * self.gamma.data
                dvar = np.sum(dx_norm * (x.data - mean) * -0.5 * (var + self.eps) ** -1.5, 
                             axis=-1, keepdims=True)
                dmean = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=-1, keepdims=True) + \
                       dvar * np.sum(-2 * (x.data - mean), axis=-1, keepdims=True) / N
                
                dx = dx_norm / np.sqrt(var + self.eps) + \
                     dvar * 2 * (x.data - mean) / N + \
                     dmean / N
                
                x.grad = x.grad + dx if x.grad is not None else dx
            
            if self.gamma.requires_grad:
                dgamma = np.sum(out.grad * x_norm, axis=tuple(range(len(out.grad.shape) - 1)))
                self.gamma.grad = self.gamma.grad + dgamma if self.gamma.grad is not None else dgamma
            
            if self.beta.requires_grad:
                dbeta = np.sum(out.grad, axis=tuple(range(len(out.grad.shape) - 1)))
                self.beta.grad = self.beta.grad + dbeta if self.beta.grad is not None else dbeta
        
        out._backward = _backward
        return out
    
    def parameters(self):
        return [self.gamma, self.beta]


class Dropout(Layer):
    """Dropout regularization"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.training = True
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
        
        mask = (np.random.rand(*x.shape) > self.p).astype(np.float32)
        out_data = x.data * mask / (1 - self.p)
        
        out = Tensor(out_data, requires_grad=x.requires_grad,
                    _children=(x,), _op='Dropout')
        
        def _backward():
            if x.requires_grad:
                grad = out.grad * mask / (1 - self.p)
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
