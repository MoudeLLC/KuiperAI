"""
Inference script for KuiperAI models
"""
import sys
import argparse
import numpy as np
import json
import os

sys.path.append('..')
from src.models.transformer import Transformer
from src.data.dataset import TextDataset


def load_model(checkpoint_path: str, vocab_path: str):
    """Load trained model from checkpoint"""
    print(f"Loading vocabulary from {vocab_path}")
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    # Load vocabulary
    vocab = TextDataset.load_vocab(vocab_path)
    print(f"✓ Vocabulary loaded: {len(vocab)} tokens")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    
    # Get model configuration from checkpoint or use defaults
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Default configuration
        model_config = {
            'vocab_size': len(vocab),
            'd_model': 512,
            'num_heads': 8,
            'num_layers': 6,
            'd_ff': 2048,
            'max_seq_len': 512
        }
        print("⚠ No model config in checkpoint, using defaults")
    
    # Ensure vocab size matches
    model_config['vocab_size'] = len(vocab)
    
    print(f"Creating model with config: {model_config}")
    
    # Create model
    model = Transformer(**model_config)
    
    # Load model weights
    if 'model_state' in checkpoint:
        params = model.parameters()
        model_state = checkpoint['model_state']
        
        loaded_params = 0
        for i, param in enumerate(params):
            param_key = f'param_{i}'
            if param_key in model_state:
                param.data = np.array(model_state[param_key], dtype=np.float32)
                loaded_params += 1
            else:
                print(f"⚠ Parameter {param_key} not found in checkpoint")
        
        print(f"✓ Loaded {loaded_params}/{len(params)} parameters")
        
        if loaded_params < len(params):
            print("⚠ Some parameters were not loaded from checkpoint")
    else:
        raise ValueError("No model_state found in checkpoint")
    
    print("✓ Model loaded successfully!")
    
    return model, vocab


def generate_text(model, vocab, prompt: str, max_length: int = 100,
                 temperature: float = 1.0, top_k: int = 50):
    """
    Generate text from a prompt
    
    Args:
        model: Trained model
        vocab: Vocabulary dictionary
        prompt: Input prompt text
        max_length: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Sample from top-k tokens only
    """
    # Tokenize prompt
    tokens = [vocab.get(word, vocab['<UNK>']) for word in prompt.split()]
    tokens = np.array([tokens], dtype=np.int32)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating (max_length={max_length}, temperature={temperature}, top_k={top_k})...")
    print("-" * 70)
    
    # Generate
    generated = model.generate(tokens, max_length, temperature)
    
    # Decode
    idx_to_token = {v: k for k, v in vocab.items()}
    generated_text = []
    
    for token_id in generated[0]:
        if token_id == vocab['<EOS>']:
            break
        if token_id not in [vocab['<PAD>'], vocab['<SOS>']]:
            generated_text.append(idx_to_token.get(token_id, '<UNK>'))
    
    result = ' '.join(generated_text)
    print(result)
    print("-" * 70)
    
    return result


def interactive_mode(model, vocab):
    """Interactive text generation mode"""
    print("\n" + "=" * 70)
    print("KuiperAI Interactive Mode")
    print("=" * 70)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print()
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            generate_text(model, vocab, prompt)
            print()
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run inference with KuiperAI')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--vocab', type=str, required=True,
                       help='Path to vocabulary file')
    parser.add_argument('--input', type=str, default=None,
                       help='Input prompt (if not provided, enters interactive mode)')
    parser.add_argument('--max-length', type=int, default=100,
                       help='Maximum length to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                       help='Sample from top-k tokens')
    
    args = parser.parse_args()
    
    # Load model
    model, vocab = load_model(args.model, args.vocab)
    
    if args.input:
        # Single inference
        generate_text(model, vocab, args.input, args.max_length,
                     args.temperature, args.top_k)
    else:
        # Interactive mode
        interactive_mode(model, vocab)


if __name__ == '__main__':
    main()
