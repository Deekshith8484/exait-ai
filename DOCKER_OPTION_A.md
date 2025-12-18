# 🐳 EXRT AI - Docker Setup (Option A: Streamlit + FastAPI Backend)

## ✅ What's Running Now

```
Docker Container Network:
┌────────────────────────────────────────────────────┐
│                  exrt-network                      │
├────────────────────────────────────────────────────┤
│                                                    │
│  Service 1: Streamlit              Service 2: FastAPI
│  ├─ Port: 8501                     ├─ Port: 8000
│  ├─ Container: exrt-ai-streamlit   ├─ Container: exrt-ai-backend
│  └─ app.py                         └─ api_backup.py
│                                                    │
│  Both services can communicate via service names: │
│  Streamlit → Backend: http://backend:8000         │
│  Frontend → Backend: http://backend:8000 (Docker) │
│                                     ↓             │
│                              http://localhost:8000│
│                                  (Local access)   │
│                                                    │
└────────────────────────────────────────────────────┘
```

---

## 🚀 How to Run (Option A)

### **Step 1: Ensure Docker is Running**
- Windows: Open Docker Desktop
- Linux: `sudo systemctl start docker`
- Mac: Click Docker icon in menu

### **Step 2: Start Both Services**
```bash
cd g:\exait ai
docker-compose up -d
```

**What happens:**
- Builds image (first time: 3-5 min)
- Starts Streamlit service on port 8501
- Starts FastAPI backend on port 8000
- Both services connect on same network

### **Step 3: Access the App**
```
Streamlit UI:  http://localhost:8501
Backend API:   http://localhost:8000
Health check:  http://localhost:8000/health
```

### **Step 4: Upload & Analyze**
1. Open http://localhost:8501
2. Upload ECG file via sidebar
3. Backend processes file
4. Results display in modal

### **Step 5: Stop Services**
```bash
docker-compose down
```

---

## 📊 What's Different from Option B (Streamlit-only)

| Feature | Option A (Current) | Option B (Streamlit-only) |
|---------|-------------------|--------------------------|
| **Streamlit** | ✅ http://8501 | ✅ http://8501 |
| **FastAPI Backend** | ✅ http://8000 | ❌ No backend |
| **Architecture** | Microservices | Monolith |
| **Scalability** | Scale services separately | All in one |
| **Complexity** | Medium | Low |
| **File Processing** | Backend (separate) | Streamlit (integrated) |

---

## 🔧 Configuration

### **.env File**
```
Gemini_API_KEY=AIzaSyDoYWh1ar4DEyx7-q-S3au8u10fNdraJUk
```

This is loaded by both services automatically.

### **Backend URL Detection** (Automatic!)
The HTML dashboard automatically detects the environment:

```javascript
// Local: http://localhost:8000
// Docker: http://backend:8000 (service name)
// Remote: http://server-ip:8000
BACKEND_API_URL: (automatically determined)
```

---

## 📋 Key Files Updated

✅ **docker-compose.yml**
- Now has TWO services: `streamlit` and `backend`
- Both on same network (`exrt-network`)
- Each has its own healthcheck
- Both load `.env` environment variables

✅ **Dockerfile**
- Supports both Streamlit and FastAPI
- Exposes ports 8501 (Streamlit) and 8000 (Backend)
- Can run either service via `command` override in docker-compose

✅ **new_dashboard.html**
- Smart backend URL detection
- Works with localhost, Docker service names, or remote IP
- Automatically routes to correct backend

---

## 🔄 Service Communication

### **From Browser to Streamlit**
```
Browser → http://localhost:8501
          (Port 8501)
```

### **From Streamlit to Backend**
```
HTML (in Streamlit) → http://backend:8000
                      (Uses service name in Docker)
```

### **From Backend to Database** (if needed)
```
Backend → (Ready for database connection)
```

---

## 📈 Workflow

```
1. User uploads ECG file in Streamlit UI (port 8501)
   ↓
2. Frontend sends to backend (http://backend:8000/upload/signal)
   ↓
3. Backend processes file (FastAPI service)
   ├─ Parse ECG data
   ├─ Run ML model
   └─ Return results
   ↓
4. Results display in modal on frontend
```

---

## 📊 View Services

### **Check running services:**
```bash
docker-compose ps
```

**Output:**
```
NAME                  COMMAND                  SERVICE     STATUS
exrt-ai-streamlit     "streamlit run app.py"   streamlit   running
exrt-ai-backend       "python api_backup.py"   backend     running
```

### **View logs:**
```bash
# All services
docker-compose logs -f

# Just Streamlit
docker-compose logs -f streamlit

# Just Backend
docker-compose logs -f backend
```

### **Connect to a service:**
```bash
# Access Streamlit container
docker exec -it exrt-ai-streamlit bash

# Access Backend container
docker exec -it exrt-ai-backend bash
```

---

## 🧪 Testing

### **Test Streamlit (from browser)**
```
http://localhost:8501
```
- Upload a file
- Check results

### **Test Backend API**
```bash
curl http://localhost:8000/health
```

**Output:**
```json
{"status": "healthy"}
```

### **Upload via curl**
```bash
curl -X POST http://localhost:8000/upload/signal \
  -F "file=@test_signal.pkl"
```

---

## 🌐 Remote Deployment

When deploying to AWS/GCP/Azure/DigitalOcean:

### **On cloud server:**
```bash
git clone https://github.com/Deekshith8484/exait-ai.git
cd exait-ai
echo "Gemini_API_KEY=AIzaSy..." > .env
docker-compose up -d
```

### **Access remotely:**
```
Streamlit UI:  http://your-server-ip:8501
Backend API:   http://your-server-ip:8000
```

---

## ✨ Benefits of Option A

✅ **Modular Architecture**
- Separate concerns (frontend vs backend)
- Easier to maintain and debug

✅ **Scalability**
- Can run multiple Streamlit instances
- Can run multiple backend instances
- Can add load balancer later

✅ **Production-Ready**
- Standard microservices pattern
- Used by major companies
- Easy to monitor and log

✅ **Development**
- Can develop frontend and backend independently
- Hot-reload both services
- Easy to test APIs

---

## 🚨 Troubleshooting

### "Connection refused" on upload
**Check:** Backend is running
```bash
docker-compose ps  # Both services should show "running"
docker-compose logs -f backend  # Check for errors
```

### "Cannot reach http://backend:8000"
**Check:** Services are on same network
```bash
docker network ls
docker network inspect exrt-network  # Should show both containers
```

### "Port 8000 already in use"
**Solution:** Change port in docker-compose.yml
```yaml
backend:
  ports:
    - "8001:8000"  # Changed from 8000 to 8001
```

### Backend service won't start
**Check:** api_backup.py syntax
```bash
docker-compose logs -f backend
```

---

## 📚 Next Steps

1. **Test Locally**
   ```bash
   docker-compose up -d
   # Visit http://localhost:8501
   # Upload test file
   # Check results
   ```

2. **Deploy to Cloud** (see DOCKER_GUIDE.md)
   - AWS EC2
   - Google Cloud Run
   - Azure
   - DigitalOcean

3. **Monitor in Production**
   - Watch logs: `docker-compose logs -f`
   - Check health: `curl http://server:8000/health`

---

## 📖 Related Documentation

- [DOCKER_GUIDE.md](DOCKER_GUIDE.md) - Full Docker guide
- [DOCKER_COMMANDS.md](DOCKER_COMMANDS.md) - Command reference
- [DOCKER_START_HERE.md](DOCKER_START_HERE.md) - Quick start
- [README_DEPLOYMENT.md](README_DEPLOYMENT.md) - Deployment overview

---

**Status:** ✅ Option A Setup Complete

Run `docker-compose up -d` to start both services!
