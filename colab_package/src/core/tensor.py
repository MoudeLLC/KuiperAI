"""
Custom Tensor implementation for KuiperAI
Provides automatic differentiation and gradient computation
"""
import numpy as np
from typing import Optional, Tuple, List, Union


class Tensor:
    """
    Core tensor class with automatic differentiation support.
    Similar to PyTorch tensors but built from scratch.
    """
    
    def __init__(self, data: Union[np.ndarray, list, float], 
                 requires_grad: bool = False,
                 _children: Tuple['Tensor', ...] = (),
                 _op: str = ''):
        """
        Initialize a tensor.
        
        Args:
            data: The actual data (numpy array or convertible)
            requires_grad: Whether to track gradients
            _children: Parent tensors for backprop
            _op: Operation that created this tensor
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        
        self.data = data.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        
        # For autograd graph
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    def zero_grad(self):
        """Reset gradients to zero"""
        self.grad = np.zeros_like(self.data)
    
    def backward(self, gradient: Optional[np.ndarray] = None):
        """
        Compute gradients via backpropagation.
        
        Args:
            gradient: Gradient from upstream (defaults to ones)
        """
        # Build topological order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Initialize gradient
        if gradient is None:
            self.grad = np.ones_like(self.data)
        else:
            self.grad = gradient
        
        # Backpropagate
        for node in reversed(topo):
            node._backward()
    
    def __add__(self, other: 'Tensor') -> 'Tensor':
        """Element-wise addition"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                self.grad = self.grad + out.grad if self.grad is not None else out.grad
            if other.requires_grad:
                other.grad = other.grad + out.grad if other.grad is not None else out.grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        """Element-wise multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                grad = other.data * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
            if other.requires_grad:
                grad = self.data * out.grad
                other.grad = other.grad + grad if other.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """Matrix multiplication"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='@')
        
        def _backward():
            if self.requires_grad:
                # Handle batched matmul: if self is (..., n, k) and other is (k, m)
                # then out is (..., n, m) and grad is (..., n, m)
                # We need: grad @ other.T -> (..., n, k)
                if other.data.ndim == 2 and out.grad.ndim > 2:
                    # Batched case: out.grad is (..., n, m), other is (k, m)
                    grad = out.grad @ other.data.T
                else:
                    # Standard case
                    grad = out.grad @ other.data.T
                self.grad = self.grad + grad if self.grad is not None else grad
            
            if other.requires_grad:
                # Handle batched matmul: if self is (..., n, k) and other is (k, m)
                # We need: self.T @ grad -> (k, m)
                if self.data.ndim > 2 and other.data.ndim == 2:
                    # Batched case: need to sum over batch dimensions
                    # self is (..., n, k), out.grad is (..., n, m)
                    # We want (k, m) so we do: sum over batch of (self.T @ grad)
                    batch_shape = self.data.shape[:-2]
                    n, k = self.data.shape[-2:]
                    m = out.grad.shape[-1]
                    
                    # Reshape to (batch_size, n, k)
                    self_reshaped = self.data.reshape(-1, n, k)
                    grad_reshaped = out.grad.reshape(-1, n, m)
                    
                    # Compute gradient for each batch and sum
                    grad = np.zeros((k, m), dtype=np.float32)
                    for i in range(self_reshaped.shape[0]):
                        grad += self_reshaped[i].T @ grad_reshaped[i]
                else:
                    # Standard case
                    grad = self.data.T @ out.grad
                
                other.grad = other.grad + grad if other.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, power: float) -> 'Tensor':
        """Element-wise power"""
        out = Tensor(self.data ** power,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op=f'**{power}')
        
        def _backward():
            if self.requires_grad:
                grad = power * (self.data ** (power - 1)) * out.grad
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Sum along axis"""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    grad = np.expand_dims(grad, axis)
                grad = np.broadcast_to(grad, self.shape)
                self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        """Mean along axis"""
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)
    
    def reshape(self, *shape) -> 'Tensor':
        """Reshape tensor"""
        # Handle both reshape(2, 3) and reshape((2, 3))
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        
        out = Tensor(self.data.reshape(shape),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='reshape')
        
        original_shape = self.shape
        
        def _backward():
            if self.requires_grad:
                if out.grad is not None:
                    grad = out.grad.reshape(original_shape)
                    self.grad = self.grad + grad if self.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})\n{self.data}"
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self * (other ** -1)
