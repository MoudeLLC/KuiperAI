#!/bin/bash
# Demo the autonomous learning system

echo "=========================================="
echo "KUIPERAI AUTONOMOUS LEARNING DEMO"
echo "=========================================="
echo ""

echo "This demo will show you:"
echo "  1. Interactive vocabulary research"
echo "  2. Advanced training with vocab combinations"
echo "  3. Improved chat responses"
echo ""

read -p "Press Enter to start demo..."
echo ""

# Step 1: Research
echo "=========================================="
echo "STEP 1: VOCABULARY RESEARCH"
echo "=========================================="
echo ""
echo "Researching 3 words interactively..."
echo ""

echo -e "1\n3\nC\nC\nQ" | python3 vocab_ecosystem.py

echo ""
read -p "Press Enter to continue to training..."
echo ""

# Step 2: Train
echo "=========================================="
echo "STEP 2: ADVANCED TRAINING"
echo "=========================================="
echo ""
echo "Training model with vocabulary combinations..."
echo "(This may take a few minutes)"
echo ""

timeout 180 python3 train_advanced.py || echo "Training started (stopped for demo)"

echo ""
read -p "Press Enter to continue to chat demo..."
echo ""

# Step 3: Chat
echo "=========================================="
echo "STEP 3: CHAT DEMONSTRATION"
echo "=========================================="
echo ""
echo "Testing chat with: 'hi' and 'what is machine learning'"
echo ""

echo -e "hi\nwhat is machine learning\nquit" | python3 chat_simple.py

echo ""
echo "=========================================="
echo "DEMO COMPLETE"
echo "=========================================="
echo ""
echo "What you saw:"
echo "  ✓ Vocabulary research with 'Continue? Press C'"
echo "  ✓ Advanced training with vocab combinations"
echo "  ✓ Improved chat responses"
echo ""
echo "Next steps:"
echo "  1. Read: QUICK_START_AUTONOMOUS.md"
echo "  2. Start autonomous: ./start_autonomous.sh"
echo "  3. Monitor: tail -f logs/autonomous_learning.log"
echo ""
echo "Your AI will learn and improve 24/7!"
