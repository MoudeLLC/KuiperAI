#!/bin/bash
# Start KuiperAI Autonomous Learning System

echo "=========================================="
echo "KUIPERAI AUTONOMOUS LEARNING"
echo "=========================================="
echo ""

# Check if config exists
if [ ! -f "configs/autonomous_learning.yaml" ]; then
    echo "❌ Config file not found!"
    echo "Please ensure configs/autonomous_learning.yaml exists"
    exit 1
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import yaml" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing PyYAML..."
    pip3 install pyyaml
fi

echo "✓ Dependencies OK"
echo ""

# Show configuration
echo "Configuration:"
echo "  • Interval: 30 minutes"
echo "  • Max runs: 1000"
echo "  • Auto-train: Every 50 cycles"
echo "  • Notifications: Every 100 cycles"
echo ""

# Ask for confirmation
read -p "Start autonomous learning? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting autonomous learner..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Run in background with nohup
    nohup python3 autonomous_learner.py > autonomous_learning.log 2>&1 &
    
    PID=$!
    echo "✓ Started with PID: $PID"
    echo "  Log file: autonomous_learning.log"
    echo "  Stats: knowledge/autonomous_stats.json"
    echo "  Notifications: notifications.txt"
    echo ""
    echo "To stop: kill $PID"
    echo "To monitor: tail -f autonomous_learning.log"
else
    echo "Cancelled"
fi
