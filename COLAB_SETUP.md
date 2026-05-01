# Run KuiperAI on Google Colab (FREE GPU)

## Quick Start (5 minutes)

### Step 1: Open Google Colab
Go to: **https://colab.research.google.com**

### Step 2: Upload the Notebook
1. Click **File** → **Upload notebook**
2. Go to **GitHub** tab
3. Enter: `MoudeLLC/KuiperAI`
4. Select: `run_on_colab.ipynb`

**OR** use this direct link:
```
https://colab.research.google.com/github/MoudeLLC/KuiperAI/blob/main/run_on_colab.ipynb
```

### Step 3: Enable GPU
1. Click **Runtime** → **Change runtime type**
2. Select **T4 GPU** from the dropdown
3. Click **Save**

### Step 4: Run All Cells
1. Click **Runtime** → **Run all**
2. Wait ~10 minutes for initialization
3. Download the model when complete

## What This Does

- Initializes a 1.3B parameter KuiperAI model
- Uses choice 1 (from scratch) and choice 2 (medium size)
- Creates the model in `../train/pre/` directory
- Packages and downloads the initialized model

## Free GPU Resources

- **Google Colab Free**: 15-30 GPU hours/week
- **GPU Type**: NVIDIA T4 (16GB VRAM)
- **Perfect for**: Model initialization and testing

## Troubleshooting

**Out of Memory?**
- The medium model (1.3B params) needs ~6GB VRAM
- T4 GPU has 16GB, so it should work fine
- If it fails, try restarting runtime

**Session Timeout?**
- Free Colab sessions timeout after 12 hours
- Just restart and run again

**Need More Power?**
- Upgrade to Colab Pro ($9.99/month) for better GPUs
- Or use Google Cloud $300 free credits

## Next Steps

After initialization:
1. Download the model zip file
2. Upload to your training environment
3. Run training with `./scripts/train_combined.sh`

## Alternative: Google Cloud Platform

If you need more power, get $300 free credits:
1. Sign up: https://console.cloud.google.com
2. Create VM with T4 or A100 GPU
3. Clone repo and run initialization
4. 100+ hours of free GPU time

---

**Repository**: https://github.com/MoudeLLC/KuiperAI
**Questions?** Open an issue on GitHub
