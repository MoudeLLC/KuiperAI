# Automated Vocabulary Training System

## Overview

This system automatically trains 64K-96K vocabulary using BPE (Byte-Pair Encoding) tokenization and runs on GitHub Actions daily, saving your local compute resources.

## Features

✅ **64K-96K Vocabulary Range** - Industry-standard vocabulary size
✅ **BPE Tokenization** - Handles rare words without `<UNK>` tokens
✅ **Automated Daily Training** - Runs every day at 2 AM UTC
✅ **GitHub Actions** - No local compute needed
✅ **Auto-commit Results** - Vocabulary automatically pushed to repo
✅ **Quality Validation** - Tests coverage and generates reports
✅ **Public Domain Data** - Downloads from Project Gutenberg
✅ **Synthetic Data Generation** - Creates diverse training examples

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflow                   │
│                  (Runs Daily at 2 AM UTC)                    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Download Training Data                             │
│  - Project Gutenberg books (5 classics)                     │
│  - Synthetic data generation (5000+ sentences)              │
│  - Existing knowledge base                                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Train BPE Tokenizer                                │
│  - Target: 80K vocabulary (configurable 64K-96K)            │
│  - Algorithm: Byte-Pair Encoding                            │
│  - Normalization: Lowercase, strip accents                  │
│  - Special tokens: <PAD>, <SOS>, <EOS>, <UNK>, <MASK>      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Validate Vocabulary                                │
│  - Test coverage on sample sentences                        │
│  - Check for unknown tokens                                 │
│  - Generate statistics                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Export & Commit                                    │
│  - Save vocabulary JSON (~2-5 MB)                           │
│  - Save tokenizer JSON                                      │
│  - Generate markdown report                                 │
│  - Auto-commit to repository                                │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
.github/workflows/
  └── train_vocabulary.yml          # GitHub Actions workflow

scripts/
  ├── train_vocabulary.py            # Main training script
  ├── download_training_data.py     # Data downloader
  ├── validate_vocabulary.py        # Validation script
  └── generate_vocab_report.py      # Report generator

checkpoints/
  ├── vocab_bpe_80k.json            # Vocabulary (auto-generated)
  ├── tokenizer_bpe_80k.json        # Tokenizer (auto-generated)
  ├── vocab_stats_80k.json          # Statistics (auto-generated)
  └── vocab_report.md               # Report (auto-generated)

knowledge/
  ├── downloaded_corpus/            # Downloaded training data
  │   ├── gutenberg_*.txt           # Classic books
  │   └── synthetic_training_data.txt
  └── [existing knowledge files]
```

## GitHub Storage Limits

✅ **Individual Files:** Up to 100 MB (vocabulary ~2-5 MB)
✅ **Repository Size:** No hard limit
✅ **Actions Cache:** 2 GB (auto-evicts old)
✅ **Artifacts:** 500 MB free, 2 GB Pro

**Conclusion:** 64K-96K vocabulary JSON files are ~2-5 MB, well within GitHub limits!

## Setup Instructions

### 1. Enable GitHub Actions

The workflow is already configured in `.github/workflows/train_vocabulary.yml`

It will automatically run:
- **Daily** at 2 AM UTC
- **On push** to main branch (when knowledge files change)
- **Manually** via GitHub Actions UI

### 2. Manual Trigger

Go to GitHub → Actions → "Automated Vocabulary Training" → Run workflow

Options:
- **vocab_size**: 64000-96000 (default: 80000)
- **force_retrain**: true/false (creates release)

### 3. Local Testing (Optional)

```bash
# Install dependencies
pip install tokenizers numpy requests beautifulsoup4 tqdm

# Download training data
python scripts/download_training_data.py \
  --output-dir knowledge/downloaded_corpus \
  --synthetic-count 5000

# Train vocabulary
python scripts/train_vocabulary.py \
  --vocab-size 80000 \
  --output-dir checkpoints \
  --training-data knowledge

# Validate
python scripts/validate_vocabulary.py \
  --vocab-path checkpoints/vocab_bpe_80k.json

# Generate report
python scripts/generate_vocab_report.py \
  --vocab-path checkpoints/vocab_bpe_80k.json \
  --output checkpoints/vocab_report.md
```

## Vocabulary Sizes

| Size | Use Case | File Size | Training Time |
|------|----------|-----------|---------------|
| 64K  | Small models, fast training | ~2 MB | ~5 min |
| 80K  | **Recommended** balanced | ~3 MB | ~8 min |
| 96K  | Large models, best coverage | ~4 MB | ~12 min |

## Training Data Sources

### Automatic Downloads
1. **Project Gutenberg** (Public Domain)
   - Pride and Prejudice
   - Frankenstein
   - Sherlock Holmes
   - Moby Dick
   - Alice in Wonderland

2. **Synthetic Generation**
   - 5000+ diverse sentences
   - AI/ML domain vocabulary
   - Varied sentence structures

### Existing Data
- All files in `knowledge/` directory
- Custom datasets you add

## Output Files

### vocab_bpe_80k.json
```json
{
  "metadata": {
    "vocab_size": 80000,
    "algorithm": "BPE",
    "created_at": "2026-04-25 12:00:00",
    "special_tokens": ["<PAD>", "<SOS>", "<EOS>", "<UNK>", "<MASK>"]
  },
  "vocab": {
    "token1": 0,
    "token2": 1,
    ...
  }
}
```

### tokenizer_bpe_80k.json
Complete tokenizer configuration for easy loading with `Tokenizer.from_file()`

### vocab_report.md
Markdown report with:
- Vocabulary statistics
- Coverage metrics
- Usage examples
- Integration code

## Integration with Your Model

### Update Transformer Model

```python
from tokenizers import Tokenizer
import json

# Load new vocabulary
with open('checkpoints/vocab_bpe_80k.json', 'r') as f:
    vocab_data = json.load(f)
    vocab = vocab_data['vocab']

# Load tokenizer
tokenizer = Tokenizer.from_file('checkpoints/tokenizer_bpe_80k.json')

# Create model with new vocabulary size
model = Transformer(
    vocab_size=len(vocab),  # 80,000 instead of 896!
    d_model=512,            # Increased from 256
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_len=128,        # Increased from 64
    dropout=0.3
)
```

### Update Chat Script

```python
# Replace simple word tokenization with BPE
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file('checkpoints/tokenizer_bpe_80k.json')

def tokenize(text):
    """Convert text to token IDs using BPE"""
    encoding = tokenizer.encode(text)
    return encoding.ids

def detokenize(token_ids):
    """Convert token IDs back to text"""
    return tokenizer.decode(token_ids)
```

## Monitoring

### Check Workflow Status
1. Go to GitHub → Actions
2. View "Automated Vocabulary Training" runs
3. Check logs for any errors

### View Generated Files
After workflow completes:
1. Check `checkpoints/` directory
2. View `vocab_report.md` for statistics
3. Download artifacts from Actions page

## Troubleshooting

### Workflow Fails
- Check Actions logs for error messages
- Verify training data is available
- Ensure sufficient GitHub Actions minutes

### Vocabulary Too Small
- Increase `--vocab-size` parameter
- Add more training data
- Lower `--min-frequency` threshold

### Too Many Unknown Tokens
- Increase vocabulary size
- Add domain-specific training data
- Check tokenizer normalization settings

## Advanced Configuration

### Custom Vocabulary Size

Edit `.github/workflows/train_vocabulary.yml`:
```yaml
- name: Train BPE vocabulary
  run: |
    python scripts/train_vocabulary.py \
      --vocab-size 96000 \  # Change this
      --output-dir checkpoints \
      --training-data knowledge
```

### Add Custom Training Data

1. Add text files to `knowledge/` directory
2. Commit and push
3. Workflow automatically includes them

### Change Schedule

Edit `.github/workflows/train_vocabulary.yml`:
```yaml
on:
  schedule:
    - cron: '0 */12 * * *'  # Every 12 hours instead of daily
```

## Benefits Over Local Training

✅ **No Local Compute** - Runs on GitHub servers
✅ **Automated** - Set and forget
✅ **Version Control** - All vocabularies tracked in git
✅ **Reproducible** - Same environment every time
✅ **Free** - GitHub Actions free tier sufficient
✅ **Scalable** - Easy to increase vocabulary size

## Expected Results

After implementing this system:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Vocabulary Size | 896 | 80,000 | 89x larger |
| Unknown Tokens | 30-40% | <2% | 15-20x better |
| File Size | ~50 KB | ~3 MB | Manageable |
| Coverage | Poor | Excellent | 95%+ |
| Training | Manual | Automated | Daily updates |

## Next Steps

1. ✅ System is configured and ready
2. 🔄 Push to GitHub to trigger first run
3. 🔄 Wait for workflow to complete (~15 min)
4. 🔄 Check `checkpoints/vocab_report.md`
5. 🔄 Update model to use new vocabulary
6. 🔄 Retrain model with 80K vocabulary
7. 🔄 Test chat quality improvements

## Support

For issues or questions:
1. Check workflow logs in GitHub Actions
2. Review `vocab_report.md` for statistics
3. Test locally with provided commands
4. Adjust vocabulary size if needed

---

**Status:** ✅ Ready to deploy
**Estimated Setup Time:** 5 minutes
**First Run Time:** ~15 minutes
**Subsequent Runs:** Automated daily

🚀 Push to GitHub to start automated vocabulary training!
