# Backend Connection Troubleshooting Guide

## Issue: Frontend Shows "Backend Offline"

Your frontend is now configured to connect to `http://4.206.202.59:8000`, but showing "Offline" status.

## Quick Diagnostic Steps

### 1. Check Backend Console Logs

Open your browser console (press **F12**) and look for these messages:
```
Backend API URL configured: http://4.206.202.59:8000
Current hostname: <your-hostname>
Checking backend health at: http://4.206.202.59:8000/health
```

If you see errors, they will tell you exactly what's wrong.

### 2. Verify Backend is Running

On the server where your backend should be running (`4.206.202.59`), check if FastAPI is running:

```bash
# Check if the process is running
ps aux | grep uvicorn

# Check if port 8000 is listening
netstat -tlnp | grep 8000
# or
lsof -i :8000
```

### 3. Test Backend Health Endpoint

From your **local machine** (where the frontend runs), try:

```bash
curl http://4.206.202.59:8000/health
```

**Expected response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**If connection fails:** This means there's a network/firewall issue preventing access to the backend.

### 4. Start the Backend (if not running)

On the server `4.206.202.59`, navigate to your backend directory and run:

```bash
# Navigate to your project
cd /path/to/exait-ai

# Install dependencies (if needed)
pip install -r requirements.txt

# Start the FastAPI backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Important:** Use `--host 0.0.0.0` to allow external connections, not just localhost.

## Common Issues and Solutions

### Issue 1: "Connection Refused" or "Connection Timeout"

**Cause:** Backend is not running or firewall is blocking port 8000.

**Solutions:**
1. Start the backend with `uvicorn api:app --host 0.0.0.0 --port 8000`
2. Check firewall rules:
   ```bash
   # On Ubuntu/Debian
   sudo ufw status
   sudo ufw allow 8000
   
   # On RHEL/CentOS
   sudo firewall-cmd --list-all
   sudo firewall-cmd --permanent --add-port=8000/tcp
   sudo firewall-cmd --reload
   ```
3. If using cloud provider (AWS, Azure, GCP), check security groups/network rules

### Issue 2: "CORS Error" in Browser Console

**Cause:** Backend CORS is not configured properly.

**Solution:** Verify `api.py` has this CORS configuration:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 3: Backend Returns `model_loaded: false`

**Cause:** ML model is not loaded.

**Solution:** Train and save the model first:
```bash
cd analysis/models
python train_and_save_model.py
```

Then restart the backend.

### Issue 4: Frontend Uses Wrong URL

**Cause:** Browser is running on localhost and defaults to `http://localhost:8000`.

**Solution:** The URL detection logic is:
- If hostname is `localhost` or `127.0.0.1` → uses `http://localhost:8000`
- If hostname is `streamlit` → uses `http://backend:8000`
- **Otherwise** → uses `http://4.206.202.59:8000`

Deploy your frontend to a remote server (not localhost) for it to use `4.206.202.59:8000`.

## Testing Checklist

- [ ] Backend is running on `4.206.202.59:8000`
- [ ] `curl http://4.206.202.59:8000/health` returns `{"status": "ok", "model_loaded": true}`
- [ ] Port 8000 is open in firewall/security group
- [ ] Frontend is NOT running on localhost (or backend should run on localhost too)
- [ ] Browser console shows correct backend URL
- [ ] No CORS errors in browser console

## Architecture Overview

```
Frontend (Streamlit)          Backend (FastAPI)
Location: Anywhere            Location: 4.206.202.59:8000
┌─────────────────┐          ┌──────────────────┐
│ Browser runs    │          │ FastAPI Server   │
│ new_dashboard   │  HTTP    │ - /health        │
│ .html           │─────────>│ - /readiness     │
│                 │          │ - /upload/signal │
└─────────────────┘          └──────────────────┘
```

## Need More Help?

1. Share the output of:
   - Browser console logs (F12)
   - `curl http://4.206.202.59:8000/health` from your machine
   - Backend server logs (if backend is running)

2. Confirm:
   - Where is your frontend running? (localhost, remote server, cloud)
   - Where is your backend supposed to run? (same machine, different machine)
   - Can you access port 8000 from your browser's machine?
