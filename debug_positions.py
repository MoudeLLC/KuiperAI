#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
import numpy as np

from data.dataset import TextDataset

lines = ["Hello world", "How are you"]
dataset = TextDataset(lines, max_length=32)

print(f"Max length: {dataset.max_length}")
print(f"Sample shape: {dataset.data[0].shape}")
print(f"Sample: {dataset.data[0]}")
print(f"Max value: {np.max(dataset.data[0])}")
print(f"Seq length: {len(dataset.data[0])}")

# Check positions
batch_size = 1
seq_len = len(dataset.data[0])
positions = np.arange(seq_len)[None, :].repeat(batch_size, axis=0)
print(f"Positions shape: {positions.shape}")
print(f"Positions: {positions}")
print(f"Max position: {np.max(positions)}")
