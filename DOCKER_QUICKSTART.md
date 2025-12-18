# EXRT AI - Docker Quick Start

## 🚀 Quick Commands

### Build Docker Image
```bash
cd g:\exait ai
docker build -t exrt-ai .
```

### Run Container (Single)
```bash
docker run -p 8501:8501 -e Gemini_API_KEY="AIzaSyDoYWh1ar4DEyx7-q-S3au8u10fNdraJUk" exrt-ai
```

### Run with Docker Compose (Recommended)
```bash
docker-compose up -d
```

### Access Application
```
http://localhost:8501
```

### Stop Container
```bash
docker-compose down
```

### View Logs
```bash
docker-compose logs -f
```

---

## 📦 What's Inside Docker Image

- **Base:** Python 3.10-slim
- **Size:** ~800MB (compressed)
- **Port:** 8501 (Streamlit)
- **Startup:** ~10-15 seconds

---

## 🌐 Remote Access

After building the Docker image, you can:

### Option 1: Push to Docker Hub
```bash
docker tag exrt-ai docker.io/your-username/exrt-ai
docker push docker.io/your-username/exrt-ai
```

### Option 2: Deploy to Cloud
- **AWS EC2** - Run docker-compose on virtual server
- **Google Cloud Run** - Push image and deploy
- **Azure Container Instances** - Deploy from registry
- **DigitalOcean** - App Platform or Droplet + Docker

### Option 3: Deploy to Kubernetes (Advanced)
```bash
kubectl run exrt-ai --image=exrt-ai:latest -p 8501:8501
```

---

## ✅ Verification

Check Docker installation:
```bash
docker --version        # Should show version info
docker ps              # Should show running containers
```

Verify build:
```bash
docker images | findstr exrt-ai
```

---

See **DOCKER_GUIDE.md** for complete documentation.
