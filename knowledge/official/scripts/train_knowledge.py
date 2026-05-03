#!/usr/bin/env python3
"""
KuiperAI Knowledge Trainer
Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.
"""

import os
import sys
import argparse
from datasets import load_dataset

# Auto-detect hardware and import appropriate libraries
def detect_hardware():
    """Detect available hardware: TPU, GPU, or CPU"""
    hardware_type = 'cpu'
    
    # Check for TPU
    try:
        import jax
        import jax.numpy as jnp
        devices = jax.devices()
        tpu_devices = [d for d in devices if d.platform == 'tpu']
        if tpu_devices:
            print(f"✓ TPU detected: {len(tpu_devices)} TPU device(s) - Using JAX/Flax")
            print(f"  TPU devices: {tpu_devices}")
            hardware_type = 'tpu'
            return hardware_type
    except ImportError:
        print("  JAX not installed, skipping TPU detection")
    except Exception as e:
        print(f"  TPU check failed: {e}")
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ GPU detected: {gpu_count} GPU(s) ({gpu_name}) - Using PyTorch")
            hardware_type = 'gpu'
            return hardware_type
    except ImportError:
        print("  PyTorch not installed, skipping GPU detection")
    except Exception as e:
        print(f"  GPU check failed: {e}")
    
    print("✓ CPU only - Using PyTorch")
    return hardware_type

print("Detecting hardware...")
HARDWARE = detect_hardware()

# Import based on detected hardware
if HARDWARE == 'tpu':
    try:
        import jax
        import jax.numpy as jnp
        from flax import linen as nn
        from flax.training import train_state
        import optax
        from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
        print("✓ JAX/Flax libraries loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import JAX/Flax libraries: {e}")
        print("  Falling back to PyTorch")
        HARDWARE = 'cpu'
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
else:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )
    print("✓ PyTorch libraries loaded successfully")

def main():
    parser = argparse.ArgumentParser(description="KuiperAI Knowledge Trainer")
    
    # Required arguments
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for model checkpoints")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier")
    
    # Basic training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=32, help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Checkpoint and logging
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Log every X steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--logging_dir", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Reporting integration (tensorboard, wandb, none)")
    
    # Mixed precision training
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision training")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision training")
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="FP16 optimization level (O0, O1, O2, O3)")
    parser.add_argument("--fp16_backend", type=str, default="auto", help="FP16 backend (auto, apex, cpu_amp)")
    parser.add_argument("--half_precision_backend", type=str, default="auto", help="Half precision backend")
    
    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses for data loading")
    parser.add_argument("--dataloader_pin_memory", action="store_true", help="Pin memory in data loaders")
    parser.add_argument("--dataloader_drop_last", action="store_true", help="Drop the last incomplete batch")
    
    # Optimizer arguments
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.0, help="Ratio of warmup steps to total steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (overrides num_train_epochs)")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", help="Learning rate scheduler type")
    
    # Advanced training options
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--ddp_backend", type=str, default=None, help="Distributed backend (nccl, gloo, mpi)")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Find unused parameters in DDP")
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2, help="Number of batches to prefetch")
    
    # Evaluation and validation
    parser.add_argument("--eval_steps", type=int, default=None, help="Run evaluation every X steps")
    parser.add_argument("--eval_file", type=str, default=None, help="Path to evaluation data file")
    parser.add_argument("--do_eval", action="store_true", help="Run evaluation during training")
    parser.add_argument("--evaluation_strategy", type=str, default="no", help="Evaluation strategy (no, steps, epoch)")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation")
    
    # Model loading options
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--ignore_data_skip", action="store_true", help="Skip data that was already processed in resumed training")
    
    # Special options
    parser.add_argument("--do_train", action="store_true", help="Run training")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite output directory if it exists")
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bars")
    parser.add_argument("--load_best_model_at_end", action="store_true", help="Load best model at end of training")
    parser.add_argument("--metric_for_best_model", type=str, default=None, help="Metric to use for best model selection")
    parser.add_argument("--greater_is_better", action="store_true", help="Whether better metric values are higher")
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Model ID for Hugging Face Hub")
    parser.add_argument("--hub_token", type=str, default=None, help="Token for Hugging Face Hub")
    
    # Debug options
    parser.add_argument("--debug_mode", action="store_true", help="Enable debug mode with extra logging")
    parser.add_argument("--log_level", type=str, default="info", help="Logging level (debug, info, warning, error)")
    parser.add_argument("--sharded_ddp", type=str, default="", help="Sharded DDP options")
    parser.add_argument("--fsdp", type=str, default="", help="Fully Sharded Data Parallel options")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")
    
    args = parser.parse_args()
    
    if HARDWARE == 'tpu':
        train_with_jax(args)
    else:
        train_with_pytorch(args)

def train_with_pytorch(args):
    """Training with PyTorch (for GPU/CPU)"""
    print("Loading tokenizer and model (PyTorch)...")
    
    # Force GPU usage if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
    dataset = load_dataset("text", data_files={"train": args.train_file})
    
    # Load evaluation dataset if provided
    eval_dataset = None
    if args.eval_file and args.do_eval:
        print("Loading evaluation dataset...")
        eval_data = load_dataset("text", data_files={"validation": args.eval_file})
        eval_dataset = eval_data["validation"]
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_dir=args.logging_dir,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        report_to=args.report_to,
        seed=args.seed,
        disable_tqdm=args.disable_tqdm,
        optim="adamw_torch",
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
    )
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )
    
    print("Starting training (PyTorch)...")
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
    
    print("Saving final model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")

def train_with_jax(args):
    """Training with JAX/Flax (for TPU)"""
    print("Loading tokenizer and model (JAX/Flax)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = FlaxAutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading dataset...")
    dataset = load_dataset("text", data_files={"train": args.train_file})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Create optimizer
    num_train_steps = len(tokenized_dataset["train"]) // args.per_device_train_batch_size * args.num_train_epochs
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=args.learning_rate,
        transition_steps=args.warmup_steps
    )
    decay_fn = optax.linear_schedule(
        init_value=args.learning_rate,
        end_value=0.0,
        transition_steps=num_train_steps - args.warmup_steps
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[args.warmup_steps]
    )
    
    optimizer = optax.adamw(
        learning_rate=schedule_fn,
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=args.weight_decay
    )
    
    # Create train state
    state = train_state.TrainState.create(
        apply_fn=model.__call__,
        params=model.params,
        tx=optimizer
    )
    
    # Training loop
    print("Starting training (JAX/Flax on TPU)...")
    import numpy as np
    from tqdm import tqdm
    
    def train_step(state, batch):
        def loss_fn(params):
            outputs = state.apply_fn(**batch, params=params, train=True)
            loss = outputs.loss
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    
    # Prepare data
    train_data = tokenized_dataset["train"]
    batch_size = args.per_device_train_batch_size
    
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for i in tqdm(range(0, len(train_data), batch_size)):
            batch = train_data[i:i+batch_size]
            batch = {k: jnp.array(v) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            
            if 'labels' not in batch:
                batch['labels'] = batch['input_ids']
            
            state, loss = train_step(state, batch)
            epoch_loss += loss
            num_batches += 1
            
            if (i // batch_size) % args.logging_steps == 0:
                print(f"Step {i // batch_size}, Loss: {loss:.4f}")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    
    print("Saving final model...")
    # Convert back to PyTorch format for compatibility
    model.params = state.params
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
