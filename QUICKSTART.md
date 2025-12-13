# Quick Start Guide - Stock Analyzer

## 🚀 Deploy to Render.com in 5 Minutes

### Prerequisites
- GitHub account with this code pushed
- Render.com account (free)

---

## Step 1: Deploy Backend (2 minutes)

1. Go to https://dashboard.render.com/
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `autoanalyst-backend` (or any name you like)
   - **Root Directory**: `backend`
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api_server.py`
   - **Instance Type**: Free
5. Click **"Create Web Service"**
6. **COPY THE URL** (e.g., `https://autoanalyst-backend.onrender.com`)

---

## Step 2: Update Frontend with Backend URL (1 minute)

1. Open `frontend/.env.production`
2. Replace the URL with YOUR backend URL:
   ```
   REACT_APP_API_URL=https://YOUR-BACKEND-NAME.onrender.com
   ```
3. Save and commit:
   ```bash
   git add frontend/.env.production
   git commit -m "Update backend URL"
   git push
   ```

---

## Step 3: Deploy Frontend (2 minutes)

1. Go to https://dashboard.render.com/
2. Click **"New +"** → **"Static Site"**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `autoanalyst-frontend` (or any name you like)
   - **Root Directory**: `frontend`
   - **Build Command**: `npm install && npm run build`
   - **Publish Directory**: `build`
5. Click **"Create Static Site"**

---

## Step 4: Test! ✅

1. Visit your frontend URL (e.g., `https://autoanalyst-frontend.onrender.com`)
2. Enter a stock ticker: `AAPL`
3. Click **"ANALYZE STOCK"**
4. Wait ~30 seconds for the first analysis
5. See the complete dashboard! 🎉

---

## ⚠️ Important Notes

### First Request is Slow
The free tier backend spins down after 15 minutes of inactivity.
- **First request**: 30-60 seconds (backend waking up)
- **Subsequent requests**: 10-30 seconds

### Keep It Awake (Optional)
Use [UptimeRobot](https://uptimerobot.com/) to ping your backend every 10 minutes:
- URL to ping: `https://YOUR-BACKEND-NAME.onrender.com/api/health`
- Interval: Every 10 minutes

---

## 🎯 What You Get

Your deployed app includes:

✅ **DCF Valuation** - 6-step discounted cash flow analysis
✅ **Revenue Forecasting** - 5-year projections with charts
✅ **Comparable Companies** - Peer group valuation multiples
✅ **ML Synthesis** - AI-powered final recommendation
✅ **Beautiful Dashboard** - Holographic UI with charts

---

## 🔧 Local Development

### Backend:
```bash
cd backend
python api_server.py
```
Runs on http://localhost:8000

### Frontend:
```bash
cd frontend
npm start
```
Runs on http://localhost:3000

---

## 📊 Example Analysis

Try these tickers:
- `AAPL` - Apple Inc.
- `TSLA` - Tesla
- `MSFT` - Microsoft
- `GOOGL` - Google
- `NVDA` - NVIDIA

---

## 🆘 Troubleshooting

**Backend won't start?**
- Check Render logs for errors
- Verify `requirements.txt` is in `backend/` folder

**Frontend shows errors?**
- Make sure backend URL is correct in `.env.production`
- Check browser console for error messages

**Analysis fails?**
- Wait 60 seconds if backend just woke up
- Try a different ticker (some have limited data)

---

## 💰 Costs

**Current Setup (Free):**
- Backend: $0/month (with 15min spin-down)
- Frontend: $0/month (always active)

**Upgrade to 24/7 (Optional):**
- Backend: $7/month (no spin-down)
- Frontend: $0/month

---

**You're all set! Happy analyzing! 📈**
