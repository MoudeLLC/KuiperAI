#!/bin/bash
# Quick Start: Test Vocabulary Training Locally

set -e

echo "========================================================================"
echo "KUIPERAI - VOCABULARY TRAINING QUICK START"
echo "========================================================================"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 not found"
    exit 1
fi

echo "✓ Python 3 found"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -q tokenizers numpy requests tqdm 2>/dev/null || pip3 install -q tokenizers numpy requests tqdm

echo "✓ Dependencies installed"

# Create directories
mkdir -p knowledge/downloaded_corpus
mkdir -p checkpoints

# Download training data
echo ""
echo "========================================================================"
echo "STEP 1: DOWNLOADING TRAINING DATA"
echo "========================================================================"
python3 scripts/download_training_data.py \
    --output-dir knowledge/downloaded_corpus \
    --synthetic-count 5000

# Train vocabulary
echo ""
echo "========================================================================"
echo "STEP 2: TRAINING BPE VOCABULARY (80K)"
echo "========================================================================"
python3 scripts/train_vocabulary.py \
    --vocab-size 80000 \
    --output-dir checkpoints \
    --training-data knowledge \
    --min-frequency 2

# Validate
echo ""
echo "========================================================================"
echo "STEP 3: VALIDATING VOCABULARY"
echo "========================================================================"
python3 scripts/validate_vocabulary.py \
    --vocab-path checkpoints/vocab_bpe_80k.json

# Generate report
echo ""
echo "========================================================================"
echo "STEP 4: GENERATING REPORT"
echo "========================================================================"
python3 scripts/generate_vocab_report.py \
    --vocab-path checkpoints/vocab_bpe_80k.json \
    --output checkpoints/vocab_report.md

echo ""
echo "========================================================================"
echo "✅ VOCABULARY TRAINING COMPLETE!"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  📄 checkpoints/vocab_bpe_80k.json      - Vocabulary (80,000 tokens)"
echo "  📄 checkpoints/tokenizer_bpe_80k.json  - Tokenizer"
echo "  📄 checkpoints/vocab_stats_80k.json    - Statistics"
echo "  📄 checkpoints/vocab_report.md         - Report"
echo ""
echo "Next steps:"
echo "  1. Read checkpoints/vocab_report.md"
echo "  2. Update your model to use 80K vocabulary"
echo "  3. Retrain model with new vocabulary"
echo "  4. Test improved chat quality"
echo ""
echo "To run on GitHub Actions:"
echo "  1. Commit and push these changes"
echo "  2. Go to GitHub → Actions"
echo "  3. Workflow runs automatically daily at 2 AM UTC"
echo ""
