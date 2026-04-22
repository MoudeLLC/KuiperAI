#!/bin/bash
# Complete system test

echo "=========================================="
echo "KUIPERAI COMPLETE SYSTEM TEST"
echo "=========================================="
echo ""

# Test 1: Vocabulary System
echo "TEST 1: Vocabulary Ecosystem"
echo "----------------------------------------"
echo -e "2\n5" | python3 vocab_ecosystem.py
echo ""

# Test 2: View Definitions
echo "TEST 2: View Definitions"
echo "----------------------------------------"
echo -e "3\n5" | python3 vocab_ecosystem.py
echo ""

# Test 3: Simple Chat
echo "TEST 3: Simple Chat (Rule-Based)"
echo "----------------------------------------"
echo -e "hi\nwhat is machine learning\nwhat is an algorithm\nquit" | python3 chat_simple.py
echo ""

# Test 4: Check if improved model exists
echo "TEST 4: Check Trained Models"
echo "----------------------------------------"
if [ -f "checkpoints/vocab_improved.json" ]; then
    echo "✓ Improved model found"
    echo "  Testing improved chat..."
    echo -e "hi\nwhat is learning\nquit" | python3 chat_improved.py
else
    echo "✗ Improved model not found (still training?)"
fi
echo ""

# Test 5: Check if advanced model exists
if [ -f "checkpoints/vocab_advanced.json" ]; then
    echo "✓ Advanced model found"
    echo "  Testing advanced chat..."
    echo -e "hi\nwhat is machine learning\nstats\nquit" | python3 chat_advanced.py
else
    echo "✗ Advanced model not found (still training?)"
fi
echo ""

# Test 6: Knowledge Report
echo "TEST 5: Knowledge Report"
echo "----------------------------------------"
if [ -f "knowledge/knowledge_report.txt" ]; then
    echo "✓ Knowledge report exists"
    echo "  First 20 lines:"
    head -20 knowledge/knowledge_report.txt
else
    echo "✗ Knowledge report not found"
fi
echo ""

# Summary
echo "=========================================="
echo "TEST SUMMARY"
echo "=========================================="
echo "Vocabulary: $(cat knowledge/ecosystem_vocab.json 2>/dev/null | grep -o '"words"' | wc -l) words"
echo "Definitions: $(ls -la knowledge/knowledge_report.txt 2>/dev/null | awk '{print $5}') bytes"
echo "Knowledge entries: $(wc -l < knowledge/ecosystem_knowledge.txt 2>/dev/null || echo 0)"
echo ""
echo "Models:"
ls -lh checkpoints/*.json 2>/dev/null | tail -5
echo ""
echo "✅ System test complete!"
