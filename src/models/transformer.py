"""
Transformer architecture implementation from scratch
Based on "Attention is All You Need" (Vaswani et al., 2017)
"""
import numpy as np
from typing import Optional
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from core.tensor import Tensor
from core.layers import Layer, Linear, LayerNorm, Dropout, Embedding
from core.activations import GELU, Softmax


class MultiHeadAttention(Layer):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value projections
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        
        # Output projection
        self.W_o = Linear(d_model, d_model)
        
        self.dropout = Dropout(dropout)
        self.softmax = Softmax()
    
    def __call__(self, query: Tensor, key: Tensor, value: Tensor,
                 mask: Optional[np.ndarray] = None) -> Tensor:
        """Override to accept multiple arguments"""
        return self.forward(query, key, value, mask)
    
    def split_heads(self, x: Tensor, batch_size: int) -> Tensor:
        """
        Split the last dimension into (num_heads, d_k)
        Reshape from (batch_size, seq_len, d_model) to 
        (batch_size, num_heads, seq_len, d_k)
        """
        seq_len = x.shape[1]
        # Reshape: (batch, seq, d_model) -> (batch, seq, heads, d_k)
        x_reshaped = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        # Transpose: (batch, seq, heads, d_k) -> (batch, heads, seq, d_k)
        data_transposed = x_reshaped.data.transpose(0, 2, 1, 3)
        
        out = Tensor(data_transposed, requires_grad=x.requires_grad,
                    _children=(x,), _op='split_heads')
        
        def _backward():
            if x.requires_grad:
                # Reverse transpose and reshape
                grad = out.grad.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def merge_heads(self, x: Tensor, batch_size: int, seq_len: int) -> Tensor:
        """
        Merge heads back to original shape
        (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        """
        # Transpose: (batch, heads, seq, d_k) -> (batch, seq, heads, d_k)
        data_transposed = x.data.transpose(0, 2, 1, 3)
        # Reshape: (batch, seq, heads, d_k) -> (batch, seq, d_model)
        data_merged = data_transposed.reshape(batch_size, seq_len, self.d_model)
        
        out = Tensor(data_merged, requires_grad=x.requires_grad,
                    _children=(x,), _op='merge_heads')
        
        def _backward():
            if x.requires_grad:
                # Reverse reshape and transpose
                grad = out.grad.reshape(batch_size, seq_len, self.num_heads, self.d_k)
                grad = grad.transpose(0, 2, 1, 3)
                x.grad = x.grad + grad if x.grad is not None else grad
        
        out._backward = _backward
        return out
    
    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor,
                                     mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Compute scaled dot-product attention with proper gradient flow
        Q, K, V: (batch_size, num_heads, seq_len, d_k)
        """
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        K_transposed = Tensor(K.data.transpose(0, 1, 3, 2), requires_grad=K.requires_grad,
                             _children=(K,), _op='transpose')
        
        def _backward_transpose():
            if K.requires_grad:
                grad = K_transposed.grad.transpose(0, 1, 3, 2)
                K.grad = K.grad + grad if K.grad is not None else grad
        K_transposed._backward = _backward_transpose
        
        # Q @ K^T
        scores_data = np.matmul(Q.data, K_transposed.data)
        scores = Tensor(scores_data, requires_grad=Q.requires_grad or K.requires_grad,
                       _children=(Q, K_transposed), _op='matmul_attention')
        
        def _backward_scores():
            if Q.requires_grad:
                grad = np.matmul(scores.grad, K_transposed.data.transpose(0, 1, 3, 2))
                Q.grad = Q.grad + grad if Q.grad is not None else grad
            if K_transposed.requires_grad:
                grad = np.matmul(Q.data.transpose(0, 1, 3, 2), scores.grad)
                K_transposed.grad = K_transposed.grad + grad if K_transposed.grad is not None else grad
        scores._backward = _backward_scores
        
        # Scale by sqrt(d_k)
        scale = 1.0 / np.sqrt(self.d_k)
        scores = scores * scale
        
        # Apply mask if provided
        if mask is not None:
            mask_value = -1e9
            scores_data = scores.data + (mask * mask_value)
            scores = Tensor(scores_data, requires_grad=scores.requires_grad,
                          _children=(scores,), _op='mask')
            
            def _backward_mask():
                if scores.requires_grad:
                    # Gradient flows through unmasked positions
                    grad = scores.grad.copy()
                    scores._prev.pop().grad = grad if scores._prev.pop().grad is None else scores._prev.pop().grad + grad
            scores._backward = _backward_mask
        
        # Softmax over last dimension
        scores_max = np.max(scores.data, axis=-1, keepdims=True)
        exp_scores = np.exp(scores.data - scores_max)
        sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)
        attention_weights_data = exp_scores / sum_exp
        
        attention_weights = Tensor(attention_weights_data, requires_grad=scores.requires_grad,
                                  _children=(scores,), _op='softmax_attention')
        
        def _backward_softmax():
            if scores.requires_grad:
                # Softmax gradient
                s = attention_weights_data
                grad_out = attention_weights.grad
                grad = s * grad_out
                sum_grad = np.sum(grad, axis=-1, keepdims=True)
                grad = grad - s * sum_grad
                scores.grad = scores.grad + grad if scores.grad is not None else grad
        attention_weights._backward = _backward_softmax
        
        # Apply attention to values: attention_weights @ V
        context_data = np.matmul(attention_weights.data, V.data)
        context = Tensor(context_data, requires_grad=attention_weights.requires_grad or V.requires_grad,
                        _children=(attention_weights, V), _op='matmul_context')
        
        def _backward_context():
            if attention_weights.requires_grad:
                grad = np.matmul(context.grad, V.data.transpose(0, 1, 3, 2))
                attention_weights.grad = attention_weights.grad + grad if attention_weights.grad is not None else grad
            if V.requires_grad:
                grad = np.matmul(attention_weights.data.transpose(0, 1, 3, 2), context.grad)
                V.grad = V.grad + grad if V.grad is not None else grad
        context._backward = _backward_context
        
        return context
    
    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Args:
            query, key, value: Input tensors, shape (batch_size, seq_len, d_model)
            mask: Optional attention mask, shape (batch_size, seq_len, seq_len)
        """
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into multiple heads (maintains gradient flow)
        Q_split = self.split_heads(Q, batch_size)
        K_split = self.split_heads(K, batch_size)
        V_split = self.split_heads(V, batch_size)
        
        # Scaled dot-product attention (with proper gradients)
        context = self.scaled_dot_product_attention(Q_split, K_split, V_split, mask)
        
        # Merge heads back (maintains gradient flow)
        context = self.merge_heads(context, batch_size, seq_len)
        
        # Final linear projection
        output = self.W_o(context)
        output = self.dropout(output)
        
        return output
    
    def parameters(self):
        return (self.W_q.parameters() + self.W_k.parameters() + 
                self.W_v.parameters() + self.W_o.parameters())


class FeedForward(Layer):
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.activation = GELU()
        self.dropout = Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x
    
    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()


class TransformerBlock(Layer):
    """Single transformer encoder/decoder block"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 dropout: float = 0.1):
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
    
    def __call__(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        """Override to accept mask parameter"""
        return self.forward(x, mask)
    
    def forward(self, x: Tensor, mask: Optional[np.ndarray] = None) -> Tensor:
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x
    
    def parameters(self):
        return (self.attention.parameters() + self.feed_forward.parameters() +
                self.norm1.parameters() + self.norm2.parameters())


class Transformer(Layer):
    """Complete Transformer model"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = Embedding(vocab_size, d_model)
        self.position_embedding = Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        
        self.dropout = Dropout(dropout)
        self.norm = LayerNorm(d_model)
        
        # Output projection
        self.output_projection = Linear(d_model, vocab_size)
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tensor:
        """
        Args:
            x: Input token indices, shape (batch_size, seq_len)
            mask: Optional attention mask
        """
        batch_size, seq_len = x.shape
        
        # Create position indices
        positions = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        
        # Debug: check bounds
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")
        
        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        
        # Combine embeddings (maintain gradient flow!)
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits
    
    def parameters(self):
        params = (self.token_embedding.parameters() + 
                 self.position_embedding.parameters() +
                 self.norm.parameters() +
                 self.output_projection.parameters())
        
        for block in self.blocks:
            params.extend(block.parameters())
        
        return params
    
    def generate(self, start_tokens: np.ndarray, max_length: int = 100,
                temperature: float = 1.0) -> np.ndarray:
        """
        Generate text autoregressively
        
        Args:
            start_tokens: Initial tokens, shape (1, start_len)
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
        """
        generated = start_tokens.copy()
        
        for _ in range(max_length - start_tokens.shape[1]):
            # Create causal mask
            seq_len = generated.shape[1]
            mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
            mask = mask[None, None, :, :]
            
            # Forward pass
            logits = self.forward(generated, mask)
            
            # Get logits for last position
            next_token_logits = logits.data[0, -1, :] / temperature
            
            # Sample from distribution
            probs = np.exp(next_token_logits) / np.sum(np.exp(next_token_logits))
            next_token = np.random.choice(len(probs), p=probs)
            
            # Append to sequence
            generated = np.concatenate([generated, [[next_token]]], axis=1)
        
        return generated
    
    def save(self, path: str):
        """Save model parameters to file"""
        params_dict = {}
        for i, param in enumerate(self.parameters()):
            params_dict[f'param_{i}'] = param.data
        np.savez(path, **params_dict)
    
    def load(self, path: str):
        """Load model parameters from file"""
        data = np.load(path)
        params = self.parameters()
        for i, param in enumerate(params):
            param.data = data[f'param_{i}']
