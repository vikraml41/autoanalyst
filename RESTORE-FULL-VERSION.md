# How to Restore Full Version After Upgrading to Paid Plan

## Current Status
You're running a **MINIMAL version** to deploy on Render's free tier (512MB RAM).

The full version with all ML capabilities is saved in `requirements-FULL.txt`.

## When You Upgrade to Paid Plan

Follow these steps to restore full functionality:

### Step 1: Replace requirements.txt
```bash
cp requirements-FULL.txt requirements.txt
```

### Step 2: Commit and push
```bash
git add requirements.txt
git commit -m "Restore full ML requirements after paid plan upgrade"
git push origin main
```

### Step 3: Redeploy on Render
Render will automatically redeploy with the full version.

## What You Get Back

With the full version:
- ✅ PyTorch LSTM-Transformer models
- ✅ HuggingFace Transformers (FinBERT sentiment)
- ✅ XGBoost models
- ✅ Advanced visualizations (matplotlib, seaborn, plotly)
- ✅ Time series forecasting (statsmodels)
- ✅ All technical indicators

## Current Minimal Version Includes

- ✅ FastAPI server (all endpoints work)
- ✅ Basic ML with scikit-learn
- ✅ Stock data with yfinance
- ✅ Sentiment with VADER (lightweight)
- ✅ All core functionality

The app will work fine with reduced ML capabilities until you upgrade!
