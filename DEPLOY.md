# Deploy to Render.com

## Files Ready for Deployment

All code is ready to deploy to Render.com. Here's what you need:

---

## Backend Deployment

**Service Type:** Web Service

**Settings:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api_server:app --host 0.0.0.0 --port $PORT`
- **Root Directory:** `backend`

**Alternative Start Command:** `python api_server.py`

The backend already:
- ✅ Uses `PORT` environment variable (Render sets this automatically)
- ✅ Has CORS configured for all origins
- ✅ Has health check endpoint at `/api/health`
- ✅ All dependencies in `requirements.txt`

**After deployment, copy your backend URL** (e.g., `https://your-app-backend.onrender.com`)

---

## Frontend Deployment

**Service Type:** Static Site

**Settings:**
- **Build Command:** `npm install && npm run build`
- **Publish Directory:** `build`
- **Root Directory:** `frontend`

**IMPORTANT:** Before deploying frontend, update the backend URL:

1. Edit `frontend/src/App.js` line 10
2. Change this line:
   ```javascript
   return process.env.REACT_APP_API_URL || 'https://autoanalyst-backend.onrender.com';
   ```
3. Replace `autoanalyst-backend` with YOUR actual backend service name

---

## That's It!

Both services will work together. The frontend will call the backend API which runs the stock analysis.

### Test URLs:
- **Backend Health:** `https://YOUR-BACKEND.onrender.com/api/health`
- **Frontend:** `https://YOUR-FRONTEND.onrender.com`

### Note on Free Tier:
Backend spins down after 15 minutes of inactivity. First request after spin-down takes ~30-60 seconds.
