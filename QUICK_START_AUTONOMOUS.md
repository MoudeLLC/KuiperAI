# Quick Start: Autonomous Learning System

## 🚀 Get Started in 3 Steps

### Step 1: Interactive Research (Test the System)
```bash
python3 vocab_ecosystem.py
```
- Choose option `1` (Run research cycle)
- Enter `5` for 5 iterations
- Press `C` to continue after each word
- Press `Q` to quit anytime

**What happens:**
- Researches vocabulary words
- Finds new words and knowledge
- Builds knowledge base
- Shows progress in real-time

### Step 2: Train Advanced Model
```bash
python3 train_advanced.py
```
- Generates 500+ training samples from vocabulary
- Trains transformer model (takes 5-10 minutes)
- Creates advanced model with better responses

**What you get:**
- Model that combines vocabulary intelligently
- Better generalization
- More coherent responses

### Step 3: Chat with Advanced Model
```bash
python3 chat_advanced.py
```
Try these:
- "hi"
- "what is machine learning"
- "explain neural networks"
- "help"
- "stats"

## 🤖 Background Autonomous Learning

### Start Background Learning
```bash
chmod +x start_autonomous.sh
./start_autonomous.sh
```

Answer `y` when prompted.

**The system will:**
- ✅ Run every 30 minutes
- ✅ Research 5 words per cycle
- ✅ Auto-train every 50 cycles
- ✅ Send notifications every 100 cycles
- ✅ Run 1000 total cycles
- ✅ Get smarter every day!

### Monitor Progress
```bash
# Watch live log
tail -f logs/autonomous_learning.log

# Check statistics
cat knowledge/autonomous_stats.json

# View notifications
cat notifications.txt
```

### Stop Background Learning
```bash
# Find process
ps aux | grep autonomous_learner

# Stop it
kill <PID>
```

## 📊 What Gets Better Over Time

### Day 1
- 48 research cycles
- ~240 new words
- 1 training session
- Basic responses

### Week 1
- 336 research cycles
- ~1,680 new words
- 6 training sessions
- Improved responses

### Month 1
- 1,000 cycles complete
- ~5,000 words
- 20 training sessions
- Smart, coherent responses

## 🎯 Comparison: Before vs After

### Before (Original Model)
```
You: hi
KuiperAI: progress makes penalties. evolved programmed navigation...
```

### After Simple (Rule-Based)
```
You: hi
KuiperAI: Hello! I'm KuiperAI. I can help explain concepts in AI...
```

### After Advanced (Neural + Vocab)
```
You: hi
KuiperAI: Hello! Understanding machine learning requires knowledge of algorithms.
```

## 🔧 Configuration

Edit `configs/autonomous_learning.yaml`:

```yaml
schedule:
  interval_minutes: 30  # Change to 60 for hourly
  max_runs: 1000        # Change to 2000 for longer

research:
  words_per_cycle: 5    # Change to 10 for faster growth

training:
  train_every: 50       # Change to 25 for more frequent training
```

## 📁 Files Created

```
knowledge/
  ├── ecosystem_vocab.json       # Growing vocabulary
  ├── ecosystem_knowledge.txt    # Accumulated knowledge
  ├── ecosystem_stats.json       # Research stats
  └── autonomous_stats.json      # Daemon stats

checkpoints/
  ├── vocab_advanced.json        # Advanced model vocab
  └── best_model.json           # Latest trained model

logs/
  └── autonomous_learning.log    # Activity log

notifications.txt                 # Progress notifications
```

## 🎓 Learning Path

1. **Test Interactive** → `vocab_ecosystem.py`
2. **Train Once** → `train_advanced.py`
3. **Chat & Test** → `chat_advanced.py`
4. **Start Autonomous** → `start_autonomous.sh`
5. **Monitor Progress** → `tail -f logs/autonomous_learning.log`
6. **Check Results** → `chat_advanced.py` (gets better over time!)

## 💡 Tips

- Start with 10 research cycles to test
- Monitor the first few cycles
- Check notifications.txt for milestones
- Retrain manually if needed: `python3 train_advanced.py`
- Use `chat_simple.py` for reliable demos
- Use `chat_advanced.py` to see learning progress

## 🐛 Troubleshooting

### "Model not found"
```bash
python3 train_advanced.py
```

### "No vocabulary"
```bash
python3 vocab_ecosystem.py
# Choose option 1, do a few cycles
```

### "Training too slow"
Reduce epochs in `train_advanced.py`:
```python
epochs=15  # Instead of 30
```

### "Want faster learning"
Edit `configs/autonomous_learning.yaml`:
```yaml
interval_minutes: 15  # Run every 15 minutes
words_per_cycle: 10   # Research 10 words per cycle
```

## 🎉 Success Indicators

You'll know it's working when:
- ✅ Vocabulary grows (check `ecosystem_vocab.json`)
- ✅ Knowledge accumulates (check `ecosystem_knowledge.txt`)
- ✅ Notifications appear (check `notifications.txt`)
- ✅ Model improves (chat responses get better)
- ✅ Stats increase (check `autonomous_stats.json`)

## 📚 Full Documentation

See `AUTONOMOUS_LEARNING.md` for complete details.

---

**Ready to make KuiperAI learn autonomously? Start with Step 1!** 🚀
