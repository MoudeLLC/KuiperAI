"""
Training script for KuiperAI models
"""
import sys
import argparse
import yaml
import numpy as np

sys.path.append('..')
from src.models.transformer import Transformer
from src.core.optimizers import AdamW
from src.core.losses import CrossEntropyLoss
from src.training.trainer import Trainer
from src.data.dataset import TextDataset, DataLoader, KnowledgeBase


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(config: dict):
    """Prepare training and validation datasets"""
    print("Loading knowledge base...")
    kb = KnowledgeBase(config['data']['knowledge_base_path'])
    
    # Load domains
    domains = config['data']['domains']
    print(f"Loading domains: {domains}")
    
    dataset = kb.create_mixed_dataset(domains)
    
    # Split into train/val
    split_idx = int(len(dataset) * config['data']['train_split'])
    
    train_data = dataset.data[:split_idx]
    train_labels = dataset.labels[:split_idx]
    
    val_data = dataset.data[split_idx:]
    val_labels = dataset.labels[split_idx:]
    
    # Create datasets
    train_dataset = TextDataset(
        [dataset.decode(d) for d in train_data],
        train_labels,
        vocab=dataset.vocab
    )
    
    val_dataset = TextDataset(
        [dataset.decode(d) for d in val_data],
        val_labels,
        vocab=dataset.vocab
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    return train_loader, val_loader, dataset.vocab


def create_model(config: dict, vocab_size: int):
    """Create model from configuration"""
    model_config = config['model']
    
    print(f"Creating {model_config['type']} model...")
    
    if model_config['type'] == 'transformer':
        model = Transformer(
            vocab_size=vocab_size,
            d_model=model_config['d_model'],
            num_heads=model_config['num_heads'],
            num_layers=model_config['num_layers'],
            d_ff=model_config['d_ff'],
            max_seq_len=model_config['max_seq_len'],
            dropout=model_config['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
    
    # Count parameters
    total_params = sum(np.prod(p.shape) for p in model.parameters())
    print(f"Model created with {total_params:,} parameters")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train KuiperAI model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Set random seed for reproducibility
    np.random.seed(config['training']['seed'])
    
    # Prepare data
    train_loader, val_loader, vocab = prepare_data(config)
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config, len(vocab))
    
    # Create optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adamw':
        optimizer = AdamW(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
    
    # Create loss function
    loss_fn = CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=config['training']['checkpoint_dir'],
        log_dir=config['training']['log_dir']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    print("=" * 70)
    
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    
    if history['val_loss']:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")


if __name__ == '__main__':
    main()
