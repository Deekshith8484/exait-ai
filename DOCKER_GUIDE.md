# EXRT AI - Docker Deployment Guide

## 🐳 What is Docker?

Docker containerizes your app so it runs identically on **any machine** (your PC, cloud servers, etc.) without dependency issues.

---

## ✅ Prerequisites

### Local Machine
- [Docker Desktop](https://www.docker.com/products/docker-desktop) (Windows/Mac)
- [Docker Engine](https://docs.docker.com/engine/install/) (Linux)
- [Docker Compose](https://docs.docker.com/compose/install/)

### Cloud Options (for remote access)
- **AWS EC2** - Virtual server in AWS cloud
- **DigitalOcean** - Simple droplets for Docker apps
- **Google Cloud Run** - Serverless container hosting
- **Azure Container Instances** - Microsoft cloud containers
- **Heroku** - Easy push-to-deploy (with Docker support)

---

## 📁 New Docker Files

```
g:\exait ai\
├── Dockerfile              ← Container blueprint
├── docker-compose.yml      ← Multi-container orchestration
├── .dockerignore           ← Exclude files from container
├── docker-run.bat          ← Windows helper script
├── docker-run.sh           ← Linux/Mac helper script
└── .env.example            ← Template for environment variables
```

---

## 🚀 Local Setup (Docker)

### Step 1: Create .env File

```bash
# Copy the example
cp .env.example .env

# Edit .env and add your Gemini API key
# Gemini_API_KEY = "AIzaSy..."
```

### Step 2: Build Docker Image

```bash
# Option A: Using docker-compose
docker-compose build

# Option B: Using Docker directly
docker build -t exrt-ai .
```

**What happens:**
- Pulls Python 3.10 base image
- Installs all dependencies from `requirements.txt`
- Copies app files into container
- Exposes port 8501

### Step 3: Run Container

```bash
# Option A: docker-compose (recommended)
docker-compose up -d

# Option B: Docker directly
docker run -p 8501:8501 -e Gemini_API_KEY="AIzaSy..." exrt-ai

# Option C: Windows batch script
.\docker-run.bat up

# Option D: Linux/Mac shell script
./docker-run.sh up
```

### Step 4: Access App

```
Open browser: http://localhost:8501
```

### Step 5: Stop Container

```bash
# docker-compose
docker-compose down

# Or Windows script
.\docker-run.bat down
```

---

## 🌐 Remote Access (Cloud Deployment)

### Option 1: AWS EC2 (Most Common)

**Cost:** ~$5-10/month for t2.micro

**Steps:**

1. **Create EC2 Instance**
   - Go to AWS Console → EC2 → Launch Instance
   - AMI: Ubuntu 22.04 LTS (free tier)
   - Instance type: t2.micro (free tier)
   - Security group: Allow ports 22 (SSH), 8501 (Streamlit)

2. **Connect via SSH**
   ```bash
   ssh -i "key.pem" ubuntu@your-instance-ip
   ```

3. **Install Docker on Server**
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker ubuntu
   ```

4. **Clone Your Repo**
   ```bash
   git clone https://github.com/Deekshith8484/exait-ai.git
   cd exait-ai
   ```

5. **Create .env on Server**
   ```bash
   nano .env
   # Paste: Gemini_API_KEY=AIzaSy...
   # Save: Ctrl+O, Enter, Ctrl+X
   ```

6. **Run Docker Container**
   ```bash
   docker-compose up -d
   ```

7. **Access Remotely**
   ```
   http://your-instance-ip:8501
   ```

### Option 2: DigitalOcean (Easiest)

**Cost:** $4-5/month for basic droplet

**Steps:**

1. Create droplet with Docker pre-installed
2. SSH into droplet
3. Clone repo and run `docker-compose up -d`
4. Access at `http://droplet-ip:8501`

### Option 3: Google Cloud Run (Serverless)

**Cost:** Free tier available, pay-as-you-go

**Steps:**

1. Push Docker image to Google Container Registry
   ```bash
   gcloud auth configure-docker
   docker tag exrt-ai gcr.io/PROJECT_ID/exrt-ai
   docker push gcr.io/PROJECT_ID/exrt-ai
   ```

2. Deploy to Cloud Run
   ```bash
   gcloud run deploy exrt-ai \
     --image gcr.io/PROJECT_ID/exrt-ai \
     --platform managed \
     --port 8501 \
     --allow-unauthenticated \
     --set-env-vars Gemini_API_KEY="AIzaSy..."
   ```

3. Access via generated URL (e.g., `https://exrt-ai-xxx.a.run.app`)

### Option 4: Azure Container Instances

**Cost:** ~$10-20/month

**Steps:**

1. Push to Azure Container Registry
   ```bash
   az acr build --registry exrtai --image exrt-ai:latest .
   ```

2. Deploy
   ```bash
   az container create \
     --resource-group exrt-ai \
     --name exrt-ai-app \
     --image exrtai.azurecr.io/exrt-ai:latest \
     --ports 8501 \
     --environment-variables Gemini_API_KEY="AIzaSy..."
   ```

---

## 📊 Docker Architecture

```
┌─────────────────────────────────────┐
│   Your Cloud Server (AWS/GCP/etc)   │
├─────────────────────────────────────┤
│  Docker Engine                      │
│  ┌──────────────────────────────┐   │
│  │ EXRT AI Container            │   │
│  ├──────────────────────────────┤   │
│  │ Python 3.10                  │   │
│  │ - Streamlit 1.28             │   │
│  │ - ML Model                   │   │
│  │ - Dependencies               │   │
│  ├──────────────────────────────┤   │
│  │ Port 8501 (exposed)          │   │
│  └──────────────────────────────┘   │
│                                      │
│  Volume Mounts (optional):           │
│  - ./app.py → /app/app.py            │
│  - ./analysis → /app/analysis        │
└─────────────────────────────────────┘
         ↑
    Internet Access (port 8501)
         ↓
   ┌──────────────┐
   │ Your Browser │
   └──────────────┘
```

---

## 🔧 Docker Commands Reference

```bash
# Build image
docker build -t exrt-ai .

# Run container
docker run -p 8501:8501 -e Gemini_API_KEY="..." exrt-ai

# Start services (docker-compose)
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# List running containers
docker ps

# Stop specific container
docker stop container_id

# Remove image
docker rmi exrt-ai

# Execute command in running container
docker exec -it container_id bash
```

---

## 🔐 Security Best Practices

### 1. Never Commit .env
```bash
# .env is already in .gitignore
git status  # Verify .env is NOT listed
```

### 2. Use Environment Variables
```bash
# Good: Pass as env var
docker run -e Gemini_API_KEY="..." exrt-ai

# Bad: Hardcode in Dockerfile or app
# DON'T do this!
```

### 3. Use Secrets Management (Production)
- **AWS:** Secrets Manager or Parameter Store
- **Google Cloud:** Secret Manager
- **Azure:** Key Vault

### 4. Restrict Port Access
```bash
# Only allow port 8501 from your IP
# In cloud security group/firewall settings
```

---

## 📈 Performance & Scaling

### Single Container (Current)
- Suitable for: Individual, small teams
- Concurrent users: 10-50
- RAM: 512MB - 2GB

### Multiple Containers (docker-compose)
- Scale with load balancer (Nginx, HAProxy)
- Separate ML model service
- Separate chat/API service

### Kubernetes (Advanced)
- Auto-scaling
- High availability
- Production-grade

---

## 🐛 Troubleshooting

### Issue: "Cannot connect to Docker daemon"
```bash
# Solution: Start Docker Desktop or Docker service
docker --version  # Test connection
```

### Issue: "Port 8501 already in use"
```bash
# Solution: Change port mapping
docker run -p 8502:8501 exrt-ai
# Access at http://localhost:8502
```

### Issue: "Gemini API key not found"
```bash
# Solution: Check .env file exists and has correct format
cat .env
docker-compose config  # Verify env vars loaded
```

### Issue: "Container exits immediately"
```bash
# View error logs
docker-compose logs
docker logs container_id
```

---

## 📝 Docker Compose File Breakdown

```yaml
version: '3.8'                    # Docker Compose version

services:
  exrt-ai:                        # Service name
    build:
      context: .                  # Build from current directory
      dockerfile: Dockerfile      # Use Dockerfile
    ports:
      - "8501:8501"              # Map container:host port
    environment:
      - Gemini_API_KEY=${Gemini_API_KEY}  # From .env file
    volumes:
      - ./app.py:/app/app.py      # Mount local file for live editing
    restart: unless-stopped       # Auto-restart on failure
    healthcheck:                  # Check container health
      test: ["CMD", "curl", "-f", "http://localhost:8501/..."]
```

---

## ✨ Next Steps

1. ✅ Build locally with `docker-compose build`
2. ✅ Test with `docker-compose up -d`
3. ✅ Access at `http://localhost:8501`
4. ✅ Choose cloud platform (AWS/GCP/Azure/DigitalOcean)
5. ✅ Deploy to cloud
6. ✅ Share remote URL with team

---

## 📞 Useful Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [Streamlit + Docker](https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker)
- [AWS EC2 Guide](https://docs.aws.amazon.com/ec2/)
- [DigitalOcean Tutorials](https://www.digitalocean.com/community/tutorials)

---

**Status:** ✅ Docker setup complete and ready to deploy!
