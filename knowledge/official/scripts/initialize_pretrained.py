#!/usr/bin/env python3
"""
KuiperAI Pretrained Model Initializer
Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.

This script initializes the KuiperAI pretrained base model.
"""

import os
import sys
import json
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForCausalLM
)

def create_kuiperai_config(model_size="medium"):
    """Create KuiperAI model configuration."""
    
    configs = {
        "small": {
            "vocab_size": 50257,
            "n_positions": 2048,
            "n_embd": 1024,
            "n_layer": 12,
            "n_head": 8,
            "n_inner": 4096,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "summary_activation": None,
            "summary_proj_to_labels": True,
            "summary_first_dropout": 0.1,
            "scale_attn_weights": True,
            "use_cache": True,
            "bos_token_id": 50256,
            "eos_token_id": 50256,
        },
        "medium": {
            "vocab_size": 50257,
            "n_positions": 4096,
            "n_embd": 2048,
            "n_layer": 24,
            "n_head": 16,
            "n_inner": 8192,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "summary_activation": None,
            "summary_proj_to_labels": True,
            "summary_first_dropout": 0.1,
            "scale_attn_weights": True,
            "use_cache": True,
            "bos_token_id": 50256,
            "eos_token_id": 50256,
        },
        "large": {
            "vocab_size": 50257,
            "n_positions": 4096,
            "n_embd": 4096,
            "n_layer": 32,
            "n_head": 32,
            "n_inner": 16384,
            "activation_function": "gelu_new",
            "resid_pdrop": 0.1,
            "embd_pdrop": 0.1,
            "attn_pdrop": 0.1,
            "layer_norm_epsilon": 1e-5,
            "initializer_range": 0.02,
            "summary_type": "cls_index",
            "summary_use_proj": True,
            "summary_activation": None,
            "summary_proj_to_labels": True,
            "summary_first_dropout": 0.1,
            "scale_attn_weights": True,
            "use_cache": True,
            "bos_token_id": 50256,
            "eos_token_id": 50256,
        }
    }
    
    return GPT2Config(**configs[model_size])

def initialize_from_scratch(output_dir, model_size="medium"):
    """Initialize KuiperAI model from scratch."""
    
    print(f"Initializing KuiperAI {model_size} model from scratch...")
    
    # Create configuration
    config = create_kuiperai_config(model_size)
    
    # Initialize model with random weights
    print("Creating model architecture...")
    model = GPT2LMHeadModel(config)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Calculate parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Model Size: {total_params * 4 / 1e9:.2f} GB (FP32)")
    print(f"  Hidden Size: {config.n_embd}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Attention Heads: {config.n_head}")
    print(f"  Context Length: {config.n_positions}")
    
    # Save model and tokenizer
    print(f"\nSaving to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "model_name": "KuiperAI",
        "version": "1.0.0",
        "model_size": model_size,
        "total_parameters": total_params,
        "architecture": "GPT-2 based",
        "copyright": "2024-2026 Moude AI LLC",
        "license": "Proprietary - KuiperAI systems only"
    }
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✓ Model initialized successfully!")
    print(f"✓ Files saved to: {output_dir}")
    
    return model, tokenizer

def initialize_from_checkpoint(checkpoint_name, output_dir):
    """Initialize from existing checkpoint (e.g., GPT-2)."""
    
    print(f"Initializing from checkpoint: {checkpoint_name}...")
    
    try:
        # Load model and tokenizer
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(checkpoint_name)
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Calculate parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\nModel Statistics:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Model Size: {total_params * 4 / 1e9:.2f} GB (FP32)")
        
        # Save to output directory
        print(f"\nSaving to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save metadata
        metadata = {
            "model_name": "KuiperAI",
            "version": "1.0.0",
            "base_model": checkpoint_name,
            "total_parameters": total_params,
            "copyright": "2024-2026 Moude AI LLC",
            "license": "Proprietary - KuiperAI systems only"
        }
        
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print("\n✓ Model initialized successfully!")
        print(f"✓ Files saved to: {output_dir}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
        print("Falling back to initialization from scratch...")
        return initialize_from_scratch(output_dir)

def main():
    print("=" * 60)
    print("KuiperAI Pretrained Model Initializer")
    print("Copyright © 2024-2026 Moude AI LLC")
    print("=" * 60)
    print()
    
    # Configuration
    output_dir = "../train/pre"
    
    print("Choose initialization method:")
    print("1. Initialize from scratch (recommended)")
    print("2. Initialize from GPT-2 checkpoint")
    print("3. Initialize from custom checkpoint")
    print()
    
    choice = input("Enter choice (1-3) [1]: ").strip() or "1"
    
    if choice == "1":
        print("\nChoose model size:")
        print("1. Small (350M parameters) - For testing")
        print("2. Medium (1.3B parameters) - Recommended")
        print("3. Large (6.7B parameters) - Maximum performance")
        print()
        
        size_choice = input("Enter choice (1-3) [2]: ").strip() or "2"
        size_map = {"1": "small", "2": "medium", "3": "large"}
        model_size = size_map.get(size_choice, "medium")
        
        model, tokenizer = initialize_from_scratch(output_dir, model_size)
        
    elif choice == "2":
        checkpoint = "gpt2"  # or gpt2-medium, gpt2-large
        model, tokenizer = initialize_from_checkpoint(checkpoint, output_dir)
        
    elif choice == "3":
        checkpoint = input("Enter checkpoint name or path: ").strip()
        if not checkpoint:
            print("No checkpoint specified. Exiting.")
            sys.exit(1)
        model, tokenizer = initialize_from_checkpoint(checkpoint, output_dir)
        
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Initialization Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Review the model in: " + output_dir)
    print("2. Run training: ./scripts/train_combined.sh")
    print("3. Monitor training progress")
    print()

if __name__ == "__main__":
    main()
