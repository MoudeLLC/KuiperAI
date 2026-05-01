#!/bin/bash

################################################################################
# KuiperAI Initialization Cleanup Script
# Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.
#
# This script removes initialization files after the pretrained model has been
# created. Run this after successfully running initialize_pretrained.py
################################################################################

echo "=========================================="
echo "KuiperAI Initialization Cleanup"
echo "Copyright © 2024-2026 Moude AI LLC"
echo "=========================================="
echo ""

# Check if pretrained model exists
if [ ! -d "../train/pre" ] || [ ! -f "../train/pre/config.json" ]; then
    echo "ERROR: Pretrained model not found in ../train/pre"
    echo "Please run initialize_pretrained.py first before cleanup."
    echo ""
    exit 1
fi

echo "Pretrained model found. Ready to cleanup initialization files."
echo ""
echo "This will remove:"
echo "  - initialize_pretrained.py"
echo "  - This cleanup script (cleanup_init.sh)"
echo ""
echo "These files are only needed for initial setup and can be safely removed."
echo ""

read -p "Continue with cleanup? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo ""
echo "Removing initialization files..."

# Remove initialize_pretrained.py
if [ -f "initialize_pretrained.py" ]; then
    rm initialize_pretrained.py
    echo "✓ Removed initialize_pretrained.py"
else
    echo "⚠ initialize_pretrained.py not found (may already be removed)"
fi

# Create a marker file to indicate cleanup was done
echo "Initialization cleanup completed on $(date)" > ../train/pre/.initialized
echo "✓ Created initialization marker"

echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Your KuiperAI system is now ready for training."
echo ""
echo "Next steps:"
echo "  1. Run: ./train_combined.sh"
echo "  2. Monitor training progress"
echo "  3. Test your trained model"
echo ""
echo "Note: This cleanup script will self-destruct now."
echo ""

# Self-destruct this script
rm -- "$0"

echo "✓ Cleanup script removed"
echo ""
