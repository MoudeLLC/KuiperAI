#!/usr/bin/env python3
"""
KuiperAI Model Export Utility
Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.

Exports trained models to multiple formats:
- PyTorch (.bin, .pt)
- SafeTensors (.safetensors)
- ONNX (.onnx)
- ZPM (.zpm) - Zero Model Package Manager (KuiperAI proprietary)
"""

import os
import sys
import json
import torch
import zipfile
import hashlib
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

def calculate_checksum(file_path):
    """Calculate SHA256 checksum of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def export_pytorch(model, tokenizer, output_dir):
    """Export to standard PyTorch format."""
    print("Exporting to PyTorch format...")
    
    pytorch_dir = os.path.join(output_dir, "pytorch")
    os.makedirs(pytorch_dir, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(pytorch_dir)
    tokenizer.save_pretrained(pytorch_dir)
    
    print(f"✓ PyTorch model saved to: {pytorch_dir}")
    return pytorch_dir

def export_safetensors(model, tokenizer, output_dir):
    """Export to SafeTensors format."""
    print("Exporting to SafeTensors format...")
    
    try:
        from safetensors.torch import save_file
        
        safetensors_dir = os.path.join(output_dir, "safetensors")
        os.makedirs(safetensors_dir, exist_ok=True)
        
        # Save model weights as safetensors
        state_dict = model.state_dict()
        save_file(state_dict, os.path.join(safetensors_dir, "model.safetensors"))
        
        # Save config and tokenizer
        model.config.save_pretrained(safetensors_dir)
        tokenizer.save_pretrained(safetensors_dir)
        
        print(f"✓ SafeTensors model saved to: {safetensors_dir}")
        return safetensors_dir
    except ImportError:
        print("⚠ SafeTensors not installed. Skipping...")
        return None

def export_onnx(model, tokenizer, output_dir):
    """Export to ONNX format."""
    print("Exporting to ONNX format...")
    
    try:
        import torch.onnx
        
        onnx_dir = os.path.join(output_dir, "onnx")
        os.makedirs(onnx_dir, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 128))
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(onnx_dir, "model.onnx"),
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'logits': {0: 'batch', 1: 'sequence'}
            },
            opset_version=14
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(onnx_dir)
        
        print(f"✓ ONNX model saved to: {onnx_dir}")
        return onnx_dir
    except Exception as e:
        print(f"⚠ ONNX export failed: {e}")
        return None

def export_zpm(model, tokenizer, output_dir, model_name="kuiperai", version="1.0.0"):
    """
    Export to ZPM (Zero Model Package Manager) format.
    
    ZPM Format Structure:
    model.zpm (ZIP archive)
    ├── manifest.json          # Model metadata
    ├── model/
    │   ├── pytorch_model.bin  # Model weights
    │   └── config.json        # Model config
    ├── tokenizer/
    │   ├── tokenizer.json
    │   ├── vocab.json
    │   └── merges.txt
    ├── checksums.json         # File checksums
    └── LICENSE.txt            # License information
    """
    print("Exporting to ZPM (Zero Model Package Manager) format...")
    
    zpm_dir = os.path.join(output_dir, "zpm")
    os.makedirs(zpm_dir, exist_ok=True)
    
    # Create temporary directory for ZPM contents
    temp_dir = os.path.join(zpm_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save model
    model_dir = os.path.join(temp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    model.config.save_pretrained(model_dir)
    
    # Save tokenizer
    tokenizer_dir = os.path.join(temp_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    
    # Calculate checksums
    checksums = {}
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, temp_dir)
            checksums[rel_path] = calculate_checksum(file_path)
    
    # Save checksums
    with open(os.path.join(temp_dir, "checksums.json"), "w") as f:
        json.dump(checksums, f, indent=2)
    
    # Create manifest
    manifest = {
        "format": "zpm",
        "format_version": "1.0.0",
        "model_name": model_name,
        "model_version": version,
        "architecture": model.config.model_type,
        "parameters": sum(p.numel() for p in model.parameters()),
        "created_at": datetime.now().isoformat(),
        "created_by": "KuiperAI Training System",
        "copyright": "2024-2026 Moude AI LLC",
        "license": "Proprietary - KuiperAI systems only",
        "files": {
            "model": "model/pytorch_model.bin",
            "config": "model/config.json",
            "tokenizer": "tokenizer/",
            "checksums": "checksums.json"
        },
        "requirements": {
            "python": ">=3.8",
            "torch": ">=2.0.0",
            "transformers": ">=4.30.0"
        },
        "metadata": {
            "training_datasets": ["knowledge_training.txt", "response_training.txt"],
            "training_method": "Combined Knowledge + Response Training",
            "context_length": model.config.n_positions if hasattr(model.config, 'n_positions') else 2048,
            "vocab_size": model.config.vocab_size
        }
    }
    
    with open(os.path.join(temp_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Create LICENSE
    license_text = """KuiperAI Model License
Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL

This model is the exclusive property of Moude AI LLC and is protected by
copyright laws and international treaty provisions.

Licensed for use with KuiperAI systems only.

Unauthorized reproduction, distribution, or use of this model, in whole or
in part, is strictly prohibited and may result in severe civil and criminal
penalties.

For licensing inquiries: licensing@moudeai.com
"""
    
    with open(os.path.join(temp_dir, "LICENSE.txt"), "w") as f:
        f.write(license_text)
    
    # Create ZPM archive
    zpm_file = os.path.join(zpm_dir, f"{model_name}-{version}.zpm")
    with zipfile.ZipFile(zpm_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, temp_dir)
                zipf.write(file_path, arcname)
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir)
    
    # Calculate ZPM checksum
    zpm_checksum = calculate_checksum(zpm_file)
    
    # Create ZPM info file
    zpm_info = {
        "file": os.path.basename(zpm_file),
        "size_bytes": os.path.getsize(zpm_file),
        "size_mb": round(os.path.getsize(zpm_file) / 1024 / 1024, 2),
        "checksum": zpm_checksum,
        "manifest": manifest
    }
    
    with open(os.path.join(zpm_dir, f"{model_name}-{version}.zpm.json"), "w") as f:
        json.dump(zpm_info, f, indent=2)
    
    print(f"✓ ZPM package saved to: {zpm_file}")
    print(f"  Size: {zpm_info['size_mb']} MB")
    print(f"  Checksum: {zpm_checksum[:16]}...")
    
    return zpm_file

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Export KuiperAI model to multiple formats")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_name", type=str, default="kuiperai", help="Model name for ZPM")
    parser.add_argument("--version", type=str, default="1.0.0", help="Model version")
    parser.add_argument("--formats", type=str, default="all", help="Formats to export (all, pytorch, safetensors, onnx, zpm)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("KuiperAI Model Export Utility")
    print("Copyright © 2024-2026 Moude AI LLC")
    print("=" * 60)
    print()
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export to requested formats
    formats = args.formats.lower().split(',')
    
    if 'all' in formats or 'pytorch' in formats:
        export_pytorch(model, tokenizer, args.output_dir)
        print()
    
    if 'all' in formats or 'safetensors' in formats:
        export_safetensors(model, tokenizer, args.output_dir)
        print()
    
    if 'all' in formats or 'onnx' in formats:
        export_onnx(model, tokenizer, args.output_dir)
        print()
    
    if 'all' in formats or 'zpm' in formats:
        export_zpm(model, tokenizer, args.output_dir, args.model_name, args.version)
        print()
    
    print("=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"\nAll formats saved to: {args.output_dir}")
    print()

if __name__ == "__main__":
    main()
