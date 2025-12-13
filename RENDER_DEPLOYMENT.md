# Render.com Deployment Guide

## Complete Setup for Stock Analyzer on Render.com

This guide will help you deploy both the backend (FastAPI) and frontend (React) to Render.com.

---

## Prerequisites

1. GitHub repository with your code pushed
2. Render.com account (free tier works!)
3. Both services will be deployed separately

---

## Option 1: Automatic Deployment (Using render.yaml)

### Step 1: Update Backend URL in Frontend

Before deploying, you need to update the backend URL in your frontend:

1. Open `frontend/src/App.js`
2. Find this line (around line 10):
   ```javascript
   return process.env.REACT_APP_API_URL || 'https://autoanalyst-backend.onrender.com';
   ```
3. Replace `autoanalyst-backend` with YOUR backend service name (you'll create this in Step 2)

### Step 2: Deploy Using render.yaml

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml` and create both services
5. Wait for deployment to complete (~5-10 minutes)

**Note:** The `render.yaml` file is already configured in the root directory!

---

## Option 2: Manual Deployment (Step-by-Step)

### Part A: Deploy Backend API

1. **Create New Web Service**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Backend Service**
   ```
   Name: autoanalyst-backend
   Region: Oregon (or closest to you)
   Branch: main
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python api_server.py
   Instance Type: Free
   ```

3. **Environment Variables** (if needed)
   ```
   PORT=8000
   PYTHON_VERSION=3.11.0
   ```

4. **Health Check Path**
   ```
   /api/health
   ```

5. **Click "Create Web Service"**

6. **Copy Backend URL**
   - Once deployed, copy the URL (e.g., `https://autoanalyst-backend.onrender.com`)
   - You'll need this for the frontend!

### Part B: Deploy Frontend

1. **Update API URL in Code**
   - Open `frontend/src/App.js`
   - Update line 10 with YOUR backend URL:
     ```javascript
     return process.env.REACT_APP_API_URL || 'https://YOUR-BACKEND-NAME.onrender.com';
     ```
   - Commit and push this change!

2. **Create New Static Site**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Static Site"
   - Connect your GitHub repository

3. **Configure Frontend Service**
   ```
   Name: autoanalyst-frontend
   Region: Oregon (or closest to you)
   Branch: main
   Root Directory: frontend
   Build Command: npm install && npm run build
   Publish Directory: build
   ```

4. **Auto-Deploy**
   - Enable "Auto-Deploy" for automatic updates on git push

5. **Click "Create Static Site"**

---

## Verification

### Test Backend
1. Visit `https://YOUR-BACKEND-NAME.onrender.com/api/health`
2. Should see: `{"status":"healthy","service":"stock-analyzer"}`

### Test Frontend
1. Visit `https://YOUR-FRONTEND-NAME.onrender.com`
2. Enter a stock ticker (e.g., AAPL)
3. Click "ANALYZE STOCK"
4. Should see complete analysis with charts!

---

## Important Notes

### Free Tier Limitations
- **Backend**: Spins down after 15 minutes of inactivity
  - First request after spin-down takes ~30-60 seconds
  - Subsequent requests are fast
- **Frontend**: Always active (static site)

### Cold Start Warning
When the backend spins down, the first stock analysis will take longer:
- Normal: 10-30 seconds
- After spin down: 45-90 seconds

**Solution**: Consider adding a loading message in the frontend!

### Performance Tips
1. **Keep Backend Warm**: Use a service like [UptimeRobot](https://uptimerobot.com/) to ping your backend every 10 minutes
2. **Upgrade Plan**: For $7/month, get 24/7 uptime with no spin-down

---

## Environment Variables (Optional)

If you need to add API keys in the future:

### Backend (.env)
```bash
PORT=8000
PYTHON_VERSION=3.11.0
# Add your API keys here
```

### Frontend (.env)
```bash
REACT_APP_API_URL=https://your-backend-url.onrender.com
```

---

## Troubleshooting

### Backend Issues

**Problem**: Backend fails to start
- **Check Logs**: Render Dashboard → Your Service → Logs
- **Common Fix**: Make sure `requirements.txt` includes all dependencies
  ```bash
  pip freeze > requirements.txt
  ```

**Problem**: Port binding error
- **Fix**: `api_server.py` already uses `PORT` environment variable
- Render sets this automatically

### Frontend Issues

**Problem**: API calls fail (CORS errors)
- **Check**: Backend URL in `App.js` is correct
- **Verify**: Backend has CORS enabled (already configured in `api_server.py`)

**Problem**: Build fails
- **Check Logs**: Build logs in Render dashboard
- **Common Fix**: Make sure all dependencies are in `package.json`
  ```bash
  cd frontend
  npm install
  ```

### Stock Analysis Issues

**Problem**: Analysis takes too long / times out
- **Cause**: Backend cold start after spin-down
- **Solution**: Wait 60 seconds and try again, or upgrade to paid tier

**Problem**: "Failed to analyze stock"
- **Check**: Stock ticker is valid (e.g., AAPL, not Apple)
- **Verify**: yfinance can access the data (some tickers might not have complete data)

---

## Monitoring

### Backend Health
```bash
curl https://YOUR-BACKEND-NAME.onrender.com/api/health
```

### Frontend Accessibility
```bash
curl https://YOUR-FRONTEND-NAME.onrender.com
```

### Check Logs
- Render Dashboard → Your Service → Logs
- Real-time logs show all API requests and errors

---

## Updating Your App

### Automatic Deployment (Recommended)
1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update stock analyzer"
   git push origin main
   ```
3. Render automatically redeploys both services!

### Manual Deployment
1. Go to Render Dashboard
2. Select your service
3. Click "Manual Deploy" → "Deploy latest commit"

---

## Cost Breakdown

### Free Tier (Current Setup)
- **Backend**: Free (with spin-down)
- **Frontend**: Free (always active)
- **Total**: $0/month ✅

### Paid Tier (Optional)
- **Backend**: $7/month (24/7 uptime, no spin-down)
- **Frontend**: Free
- **Total**: $7/month

---

## Quick Reference

### File Structure for Deployment
```
/autoanalyst
├── backend/
│   ├── api_server.py        ✅ Production-ready
│   ├── backend.py           ✅ Stock analyzer
│   ├── requirements.txt     ✅ All dependencies
│   ├── start.sh            ✅ Start script
│   └── .env.example        ✅ Environment template
│
├── frontend/
│   ├── src/
│   │   └── App.js          ✅ Updated with API URL
│   ├── public/
│   ├── package.json        ✅ React dependencies
│   └── build/              (Created during deployment)
│
└── render.yaml             ✅ Automatic deployment config
```

### URLs After Deployment
- **Backend API**: `https://YOUR-BACKEND-NAME.onrender.com`
- **Frontend**: `https://YOUR-FRONTEND-NAME.onrender.com`
- **Health Check**: `https://YOUR-BACKEND-NAME.onrender.com/api/health`

---

## Support

If you encounter issues:
1. Check Render Dashboard logs
2. Verify all configuration matches this guide
3. Test backend health endpoint
4. Check browser console for frontend errors

---

## Success Checklist

- [ ] Backend deployed successfully
- [ ] Backend health endpoint returns "healthy"
- [ ] Frontend deployed successfully
- [ ] Frontend loads without errors
- [ ] Can enter stock ticker and click analyze
- [ ] Analysis completes and shows dashboard
- [ ] All 4 cards display (DCF, Revenue, Comps, ML)
- [ ] Charts render correctly

**You're all set! 🚀**
