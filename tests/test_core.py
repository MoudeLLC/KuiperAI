"""
Unit tests for core components
"""
import sys
import numpy as np
import pytest

sys.path.append('..')
from src.core.tensor import Tensor
from src.core.activations import ReLU, Sigmoid, Softmax
from src.core.layers import Linear, LayerNorm
from src.core.losses import MSELoss, CrossEntropyLoss
from src.core.optimizers import SGD, Adam


class TestTensor:
    """Test Tensor class and autograd"""
    
    def test_tensor_creation(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.shape == (3,)
        assert t.requires_grad == True
    
    def test_addition(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        
        assert np.allclose(c.data, [5, 7, 9])
    
    def test_multiplication(self):
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([2, 3, 4], requires_grad=True)
        c = a * b
        
        assert np.allclose(c.data, [2, 6, 12])
    
    def test_matmul(self):
        a = Tensor([[1, 2], [3, 4]], requires_grad=True)
        b = Tensor([[5, 6], [7, 8]], requires_grad=True)
        c = a @ b
        
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(c.data, expected)
    
    def test_backward(self):
        a = Tensor([2.0], requires_grad=True)
        b = Tensor([3.0], requires_grad=True)
        c = a * b
        c.backward()
        
        assert np.allclose(a.grad, [3.0])
        assert np.allclose(b.grad, [2.0])


class TestActivations:
    """Test activation functions"""
    
    def test_relu(self):
        relu = ReLU()
        x = Tensor([-1, 0, 1, 2], requires_grad=True)
        y = relu(x)
        
        assert np.allclose(y.data, [0, 0, 1, 2])
    
    def test_sigmoid(self):
        sigmoid = Sigmoid()
        x = Tensor([0], requires_grad=True)
        y = sigmoid(x)
        
        assert np.allclose(y.data, [0.5], atol=1e-5)
    
    def test_softmax(self):
        softmax = Softmax()
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = softmax(x)
        
        # Softmax should sum to 1
        assert np.allclose(np.sum(y.data), 1.0)


class TestLayers:
    """Test neural network layers"""
    
    def test_linear_forward(self):
        layer = Linear(3, 2)
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = layer(x)
        
        assert y.shape == (1, 2)
    
    def test_layer_norm(self):
        layer = LayerNorm(3)
        x = Tensor([[1, 2, 3]], requires_grad=True)
        y = layer(x)
        
        # Output should have mean ~0 and std ~1
        assert np.allclose(np.mean(y.data), 0, atol=1e-5)
        assert np.allclose(np.std(y.data), 1, atol=1e-5)


class TestLosses:
    """Test loss functions"""
    
    def test_mse_loss(self):
        loss_fn = MSELoss()
        pred = Tensor([1, 2, 3], requires_grad=True)
        target = Tensor([1, 2, 3])
        
        loss = loss_fn(pred, target)
        assert np.allclose(loss.data, 0)
    
    def test_cross_entropy_loss(self):
        loss_fn = CrossEntropyLoss()
        logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        targets = np.array([0])
        
        loss = loss_fn(logits, targets)
        assert loss.data > 0


class TestOptimizers:
    """Test optimization algorithms"""
    
    def test_sgd_step(self):
        param = Tensor([1.0], requires_grad=True)
        param.grad = np.array([0.1])
        
        optimizer = SGD([param], lr=0.1)
        old_value = param.data.copy()
        optimizer.step()
        
        # Parameter should decrease
        assert param.data[0] < old_value[0]
    
    def test_adam_step(self):
        param = Tensor([1.0], requires_grad=True)
        param.grad = np.array([0.1])
        
        optimizer = Adam([param], lr=0.001)
        old_value = param.data.copy()
        optimizer.step()
        
        # Parameter should change
        assert param.data[0] != old_value[0]


def test_simple_training():
    """Test a simple training loop"""
    # Create simple linear model: y = 2x + 1
    layer = Linear(1, 1, bias=True)
    optimizer = SGD(layer.parameters(), lr=0.01)
    loss_fn = MSELoss()
    
    # Training data
    X = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y = np.array([[3], [5], [7], [9]], dtype=np.float32)
    
    # Train for a few steps
    for _ in range(10):
        for i in range(len(X)):
            x_tensor = Tensor(X[i:i+1], requires_grad=True)
            y_tensor = Tensor(y[i:i+1])
            
            # Forward
            pred = layer(x_tensor)
            loss = loss_fn(pred, y_tensor)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Loss should decrease
    final_pred = layer(Tensor(X, requires_grad=True))
    final_loss = loss_fn(final_pred, Tensor(y))
    assert final_loss.data < 10  # Should be reasonably small


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
