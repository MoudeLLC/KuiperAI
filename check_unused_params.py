#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import numpy as np

from models.transformer import Transformer
from core.losses import CrossEntropyLoss

model = Transformer(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
tokens = np.array([[1, 2, 3, 4, 5]])
targets = np.array([2, 3, 4, 5, 6])

logits = model.forward(tokens)
loss_fn = CrossEntropyLoss()
loss = loss_fn(logits.reshape(-1, 100), targets)

for p in model.parameters():
    p.grad = None

loss.backward()

print("Parameters without gradients:")
params = model.parameters()
for i, p in enumerate(params):
    if p.grad is None:
        print(f"  param_{i}: shape {p.shape}")
