#!/bin/bash
# Compare different chat versions

echo "=========================================="
echo "KUIPERAI CHAT COMPARISON"
echo "=========================================="
echo ""

echo "1. Testing SIMPLE (Rule-Based) - RECOMMENDED"
echo "------------------------------------------"
echo -e "hi\nwhat is machine learning\nquit" | python3 chat_simple.py
echo ""
echo ""

echo "2. Testing IMPROVED (Neural with better generation)"
echo "------------------------------------------"
echo -e "hi\nwhat is machine learning\nquit" | python3 chat_improved.py
echo ""
echo ""

echo "=========================================="
echo "COMPARISON COMPLETE"
echo "=========================================="
echo ""
echo "Recommendation: Use chat_simple.py for coherent responses"
echo "See CHAT_IMPROVEMENTS.md for details"
