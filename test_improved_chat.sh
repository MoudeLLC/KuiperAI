#!/bin/bash
# Test the improved chat with sample inputs

echo "Testing improved chat..."
echo ""

# Test with echo and pipe
echo -e "hi\nwhat is machine learning\nquit" | python3 chat_improved.py
