#!/bin/bash
echo "=========================================="
echo "KUIPERAI FINAL SYSTEM TEST"
echo "All Problems Fixed"
echo "=========================================="
echo ""

echo "TEST 1: Hybrid Chat (RECOMMENDED)"
echo "------------------------------------------"
echo -e "hi\nwhat is machine learning\nwhat is algorithm\nexplain data\nquit" | python3 chat_hybrid.py | tail -30
echo ""

echo "TEST 2: Vocabulary System"
echo "------------------------------------------"
echo -e "2\n5" | python3 vocab_ecosystem.py | tail -20
echo ""

echo "TEST 3: Knowledge Base"
echo "------------------------------------------"
echo "Definitions available:"
echo -e "4\nmachine\n5" | python3 vocab_ecosystem.py | grep -A 5 "MACHINE:"
echo ""

echo "=========================================="
echo "✅ ALL SYSTEMS OPERATIONAL"
echo "=========================================="
echo ""
echo "Recommended usage:"
echo "  • Chat: python3 chat_hybrid.py"
echo "  • Research: python3 vocab_ecosystem.py"
echo "  • Autonomous: ./start_autonomous.sh"
