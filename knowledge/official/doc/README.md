# KuiperAI Official Training System

**Version:** 3.0.0  
**Release Date:** April 29, 2026  
**Copyright:** © 2024-2026 Moude AI LLC. All Rights Reserved.

---

## 🚀 Quick Start

```bash
cd knowledge/official/scripts
./setup.sh                    # Setup & initialize
./train_combined.sh           # Train model (recommended)
```

**Training Time:** 48-72 hours on A100 GPU  
**Output:** Models exported to `train/out/exports/` in multiple formats

---

## 📁 Directory Structure

```
knowledge/official/
├── README.md                       # This file - complete guide
├── knowledge_training.txt          # 232 KB - Factual knowledge
├── response_training.txt           # 96 KB - Reasoning patterns
│
├── scripts/                        # All training scripts
│   ├── setup.sh                   # Initial setup & dependencies
│   ├── cleanup_init.sh            # Cleanup after initialization
│   ├── initialize_pretrained.py   # Create base model
│   ├── export_model.py            # Export to all formats
│   ├── train_knowledge.sh         # Knowledge-only training
│   ├── train_response.sh          # Response-only training
│   ├── train_combined.sh          # Combined training (recommended)
│   └── train_knowledge.py         # Python training implementation
│
├── train/                          # Training directories
│   ├── pre/                       # Pretrained base models
│   └── out/                       # Output trained models
│       ├── combined_final/        # Final trained model
│       └── exports/               # Exported formats
│           ├── pytorch/           # PyTorch format
│           ├── safetensors/       # SafeTensors format
│           ├── onnx/              # ONNX format
│           └── zpm/               # ZPM format (KuiperAI proprietary)
│
└── help/                           # Support & troubleshooting
    └── NEED_HELP.md               # Comprehensive troubleshooting guide
```

---

## 📊 Training Datasets

### knowledge_training.txt (232 KB, 1,232 lines)
**Purpose:** Comprehensive factual knowledge across all domains

**Content Includes:**
- Computer Science & Programming (algorithms, data structures, languages)
- Mathematics & Statistics (calculus, linear algebra, probability)
- Physics & Natural Sciences (mechanics, thermodynamics, quantum)
- Chemistry & Materials Science (organic, inorganic, biochemistry)
- Biology & Life Sciences (molecular, cellular, ecology)
- Engineering & Technology (electrical, mechanical, software)
- Medicine & Health Sciences (anatomy, pharmacology, diagnostics)
- Economics & Business (microeconomics, finance, management)
- Philosophy & Logic (epistemology, ethics, reasoning)
- And much more...

### response_training.txt (96 KB, 3,209 lines)
**Purpose:** Deep reasoning, thinking, and problem-solving patterns

**Reasoning Domains (9 comprehensive patterns):**
1. **Algorithmic Thinking** - Sorting algorithms, complexity analysis, optimization
2. **Machine Learning** - Diagnostics, overfitting, model evaluation
3. **Physics** - Causal reasoning, multi-step analysis, light scattering
4. **Biology** - Systems thinking, immune response, interconnections
5. **Chemistry** - Molecular reasoning, solubility, hydrogen bonding
6. **Mathematics** - Proof techniques, problem solving, √2 irrationality
7. **Computer Networks** - Protocol analysis, DNS, TCP, HTTP optimization
8. **Database Systems** - Query optimization, indexing, B-trees, joins
9. **Cybersecurity** - Threat modeling, SQL injection, XSS, CSRF, authentication

**Each pattern demonstrates:**
- Step-by-step reasoning process
- Hypothesis formation and testing
- Multi-level analysis (principles → applications)
- Trade-off evaluation and decision-making
- Systematic problem-solving approaches
- Self-correction and verification
- Complete synthesis and integration

---

## 🎯 Training Modes

### Mode 1: Knowledge Training
```bash
./scripts/train_knowledge.sh
```
- Trains on factual knowledge only
- Best for: Knowledge base, Q&A systems
- Output: `train/out/knowledge/`
- Time: 12-24 hours

### Mode 2: Response Training
```bash
./scripts/train_response.sh
```
- Trains on reasoning patterns only
- Best for: Problem-solving, analytical thinking
- Output: `train/out/response/`
- Time: 24-48 hours

### Mode 3: Combined Training ⭐ (Recommended)
```bash
./scripts/train_combined.sh
```
- Trains on both knowledge AND reasoning
- Best for: Complete AI system, production use
- Output: `train/out/combined_final/`
- Time: 48-72 hours
- **Auto-exports to all formats after training**

---

## 📦 Export Formats

After training, models are automatically exported to 4 formats:

### 1. PyTorch Format (.bin, .pt)
- **Use:** Standard Python/PyTorch applications
- **Compatibility:** Hugging Face Transformers
- **Location:** `train/out/exports/pytorch/`

### 2. SafeTensors Format (.safetensors)
- **Use:** Security-conscious deployments
- **Advantage:** Prevents arbitrary code execution
- **Location:** `train/out/exports/safetensors/`

### 3. ONNX Format (.onnx)
- **Use:** Cross-platform deployment, production inference
- **Advantage:** Framework-agnostic, optimized
- **Location:** `train/out/exports/onnx/`

### 4. ZPM Format (.zpm) - KuiperAI Proprietary
- **Use:** KuiperAI ecosystem, complete model distribution
- **Structure:** ZIP archive with manifest, checksums, license
- **Advantage:** Self-contained, verified, documented
- **Location:** `train/out/exports/zpm/`
- **Files:** `kuiperai-1.0.0.zpm` + `kuiperai-1.0.0.zpm.json`

**ZPM Package Contents:**
```
kuiperai-1.0.0.zpm (ZIP archive)
├── manifest.json          # Model metadata
├── model/                 # Model weights & config
├── tokenizer/             # Tokenizer files
├── checksums.json         # SHA256 verification
└── LICENSE.txt            # License information
```

---

## 💻 Hardware Requirements

### Minimum Configuration
- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **RAM:** 32 GB
- **Storage:** 100 GB free
- **Training Time:** 72-96 hours

### Recommended Configuration
- **GPU:** NVIDIA A100 (40GB or 80GB VRAM)
- **RAM:** 64-128 GB
- **Storage:** 500 GB free
- **Training Time:** 48-72 hours

### Cloud GPU Services
| Provider | GPU | Price/Hour | Recommended |
|----------|-----|------------|-------------|
| RunPod | A100 80GB | $2.89 | ⭐ Best |
| Lambda Labs | A100 40GB | $1.10 | Good |
| Vast.ai | A100 40GB | $0.80-2.50 | Budget |
| Google Colab Pro+ | A100 40GB | $50/month | Casual |

**Estimated Total Cost:** $150-300 for complete training

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- CUDA 11.8 or higher (for GPU training)
- 100GB+ free disk space
- Git (for cloning)

### Setup Process

**Step 1: Navigate to scripts directory**
```bash
cd knowledge/official/scripts
```

**Step 2: Run setup script**
```bash
./setup.sh
```

This automatically:
- ✅ Installs Python dependencies (PyTorch, Transformers, etc.)
- ✅ Checks GPU availability and CUDA version
- ✅ Initializes pretrained base model
- ✅ Verifies training datasets
- ✅ Optionally runs cleanup (removes initialization files)

**Step 3: Verify installation**
```bash
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

## 🎓 Training Process

### Starting Training

```bash
cd knowledge/official/scripts
./train_combined.sh
```

### Training Pipeline (4 Phases)

**Phase 1: Knowledge Training**
- Trains on `knowledge_training.txt`
- Learns factual knowledge across all domains
- Time: ~20-30 hours

**Phase 2: Response Training**
- Trains on `response_training.txt`
- Learns reasoning and thinking patterns
- Time: ~20-30 hours

**Phase 3: Joint Fine-tuning**
- Combined training on both datasets
- Integrates knowledge with reasoning
- Time: ~8-12 hours

**Phase 4: Model Export**
- Automatically exports to all 4 formats
- Generates checksums and metadata
- Time: ~10-30 minutes

### Monitoring Training

**TensorBoard (Real-time monitoring):**
```bash
tensorboard --logdir train/out/combined_final/logs
# Open browser: http://localhost:6006
```

**Weights & Biases (Cloud monitoring):**
```bash
wandb login
# Training automatically logs to W&B dashboard
```

**Check Progress:**
```bash
# View latest checkpoint
ls -lh train/out/combined_final/

# Check training logs
tail -f train/out/combined_final/logs/training.log
```

---

## 🔧 Configuration & Customization

### Adjusting Training Parameters

Edit training scripts to customize:

```bash
# In train_combined.sh or train_knowledge.sh

# Batch size (reduce if out of memory)
--per_device_train_batch_size 16    # Default: 16, try 8 or 4

# Learning rate (adjust for convergence)
--learning_rate 1e-5                # Default: 1e-5, try 5e-6 or 5e-5

# Training epochs (more = better learning)
--num_train_epochs 5                # Default: 5, try 3 or 10

# Sequence length (reduce if out of memory)
--max_seq_length 4096               # Default: 4096, try 2048

# Save frequency
--save_steps 250                    # Save checkpoint every 250 steps
```

### Model Sizes

Choose model size during initialization:

```bash
./initialize_pretrained.py
# Choose:
# 1. Small (350M params) - Testing, 8GB VRAM
# 2. Medium (1.3B params) - Recommended, 16GB VRAM
# 3. Large (6.7B params) - Maximum performance, 40GB VRAM
```

---

## 🚨 Troubleshooting

### Out of Memory (OOM)
```bash
# Solution 1: Reduce batch size
--per_device_train_batch_size 8

# Solution 2: Reduce sequence length
--max_seq_length 2048

# Solution 3: Enable gradient checkpointing (already enabled)
--gradient_checkpointing

# Solution 4: Use smaller model
# Re-run initialize_pretrained.py and choose "Small"
```

### Training Too Slow
```bash
# Check GPU utilization
nvidia-smi

# Enable mixed precision (already enabled)
--fp16

# Increase batch size if memory allows
--per_device_train_batch_size 32
```

### Model Not Learning (Loss not decreasing)
```bash
# Adjust learning rate
--learning_rate 5e-6    # Lower (more stable)
--learning_rate 5e-5    # Higher (faster learning)

# Train longer
--num_train_epochs 10

# Check data quality
head -100 knowledge_training.txt
```

### CUDA Out of Memory
```bash
# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Kill other GPU processes
nvidia-smi
kill -9 <PID>
```

**For comprehensive troubleshooting:** See `help/NEED_HELP.md`

---

## 📤 Using Trained Models

### Loading Models

**PyTorch/Transformers:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("train/out/combined_final")
tokenizer = AutoTokenizer.from_pretrained("train/out/combined_final")

# Generate response
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=500, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

**Loading ZPM Package:**
```python
import zipfile
import json
import tempfile
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_zpm(zpm_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zpm_path, 'r') as zpm:
            zpm.extractall(temp_dir)
        
        # Load manifest
        with open(f"{temp_dir}/manifest.json") as f:
            manifest = json.load(f)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(f"{temp_dir}/model")
        tokenizer = AutoTokenizer.from_pretrained(f"{temp_dir}/tokenizer")
        
        return model, tokenizer, manifest

# Usage
model, tokenizer, info = load_zpm("train/out/exports/zpm/kuiperai-1.0.0.zpm")
print(f"Loaded: {info['model_name']} v{info['model_version']}")
```

---

## 📈 Performance Metrics

### Expected Training Results

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Training Loss | < 2.0 | < 1.5 | < 1.0 |
| Validation Loss | < 2.5 | < 2.0 | < 1.5 |
| Perplexity | < 10 | < 7 | < 5 |

### Model Capabilities After Training

**Knowledge:**
- Factual question answering across all domains
- Technical explanations
- Domain expertise
- Information synthesis

**Reasoning:**
- Step-by-step problem solving
- Multi-perspective analysis
- Causal reasoning
- Hypothesis testing
- Trade-off evaluation
- Systematic debugging

---

## 📞 Support & Contact

### Documentation
- **Main Guide:** This README.md
- **Troubleshooting:** `help/NEED_HELP.md`
- **Export Formats:** `train/out/examples/README.md`

### Contact Information
- **General Support:** support@moudeai.com
- **Technical Issues:** tech@moudeai.com
- **Dataset Questions:** datasets@moudeai.com
- **Licensing:** licensing@moudeai.com

### Response Times
- **Critical Issues:** 4-8 hours
- **General Questions:** 24-48 hours
- **Feature Requests:** 3-5 business days

---

## 📜 License & Copyright

**Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.**

This training system, including all datasets, scripts, and documentation, is proprietary and confidential. Unauthorized reproduction, distribution, or use is strictly prohibited.

**Licensed for use with KuiperAI systems only.**

For licensing inquiries: licensing@moudeai.com

---

## 🔄 Version History

### Version 3.0.0 (April 29, 2026) - Current
- ✅ Complete training pipeline
- ✅ Knowledge dataset: 232 KB (1,232 lines)
- ✅ Response dataset: 96 KB (3,209 lines) - 9 reasoning domains
- ✅ Multi-format export system (PyTorch, SafeTensors, ONNX, ZPM)
- ✅ Automatic cleanup system
- ✅ Comprehensive documentation
- ✅ Production-ready

---

## 🎯 Quick Reference

### Essential Commands
```bash
# Setup
cd knowledge/official/scripts && ./setup.sh

# Train (recommended)
./train_combined.sh

# Monitor
tensorboard --logdir train/out/combined_final/logs

# Check GPU
nvidia-smi

# Test model
python3 -c "from transformers import pipeline; \
  pipe = pipeline('text-generation', model='train/out/combined_final'); \
  print(pipe('Explain AI:'))"
```

### File Sizes
- knowledge_training.txt: 232 KB
- response_training.txt: 96 KB
- Total training data: 328 KB
- Trained model (Medium): ~5 GB
- All exports: ~20 GB

### Training Time Estimates (A100 GPU)
- Knowledge training: 20-30 hours
- Response training: 20-30 hours
- Joint fine-tuning: 8-12 hours
- **Total: 48-72 hours**

---

**Maintained By:** KuiperAI Research Division, Moude AI LLC  
**Last Updated:** April 29, 2026  
**Status:** ✅ Production Ready

For the latest updates and documentation, visit: https://kuiperai.com/docs
