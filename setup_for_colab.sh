#!/bin/bash
# Package KuiperAI for Google Colab

echo "📦 Packaging KuiperAI for Google Colab..."

# Create package directory
mkdir -p colab_package

# Copy essential files
echo "Copying essential files..."

# Core training files
cp train_hard.py colab_package/
cp generate_hard_dataset.py colab_package/
cp chat_real.py colab_package/

# Source code
cp -r src colab_package/

# Knowledge base
mkdir -p colab_package/knowledge
cp knowledge/ecosystem_vocab.json colab_package/knowledge/ 2>/dev/null || echo "No vocab file yet"
cp knowledge/improved_dataset.txt colab_package/knowledge/ 2>/dev/null || echo "No improved dataset yet"
cp knowledge/ecosystem_knowledge.txt colab_package/knowledge/ 2>/dev/null || echo "No ecosystem knowledge yet"

# Create directories
mkdir -p colab_package/checkpoints
mkdir -p colab_package/logs

# Create README
cat > colab_package/README.txt << 'EOF'
🚀 KuiperAI - Ready for Google Colab

UPLOAD THIS ENTIRE FOLDER TO COLAB

Then run:
1. !python3 generate_hard_dataset.py
2. !python3 train_hard.py
3. Download the trained model files

Full training:
- 1,311 examples
- 3.6M parameters
- 20-30 minutes on Colab
EOF

# Create zip file
cd colab_package
zip -r ../KuiperAI_Colab.zip .
cd ..

echo ""
echo "✅ Package created!"
echo ""
echo "📦 File: KuiperAI_Colab.zip"
echo ""
echo "Next steps:"
echo "1. Upload KuiperAI_Colab.zip to Google Colab"
echo "2. Unzip it: !unzip KuiperAI_Colab.zip"
echo "3. Run: !python3 generate_hard_dataset.py"
echo "4. Run: !python3 train_hard.py"
echo ""
echo "🚀 Ready to train!"
