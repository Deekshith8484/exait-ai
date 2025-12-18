# 📋 EXRT AI - All Essential Commands

## 🚀 Quick Start (Pick One)

### Option 1: Run Locally (Fastest)
```bash
cd g:\exait ai
pip install -r requirements.txt
streamlit run app.py
```
📍 Access: http://localhost:8501

---

### Option 2: Run with Docker (Production-like)
```bash
cd g:\exait ai
docker-compose up -d
```
📍 Access: http://localhost:8501

**Stop it:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

---

### Option 3: Deploy to Cloud (Shareable)
See [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for step-by-step instructions

---

## 🐳 Docker Commands

### Build Docker Image
```bash
cd g:\exait ai
docker build -t exrt-ai .
```

### Run Container (Single)
```bash
docker run -p 8501:8501 \
  -e Gemini_API_KEY="AIzaSyDoYWh1ar4DEyx7-q-S3au8u10fNdraJUk" \
  exrt-ai
```

### Using docker-compose (Recommended)
```bash
# Start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Clean up
docker-compose down -v
```

### Helper Scripts
```bash
# Windows
.\docker-run.bat build       # Build image
.\docker-run.bat up          # Start container
.\docker-run.bat down        # Stop container
.\docker-run.bat logs        # View logs

# Linux/Mac
./docker-run.sh build        # Build image
./docker-run.sh up           # Start container
./docker-run.sh down         # Stop container
./docker-run.sh logs         # View logs
```

---

## 🔍 Troubleshooting Commands

### Check Docker Status
```bash
docker --version              # Verify Docker installed
docker ps                      # List running containers
docker images | findstr exrt   # Check if image built
```

### View Logs
```bash
docker-compose logs           # View all logs
docker-compose logs -f        # Follow logs in real-time
docker logs container_id      # View specific container
```

### Debug Container
```bash
docker exec -it container_id bash   # SSH into container
docker inspect container_id         # View container details
docker stats                        # View resource usage
```

### Clean Up
```bash
docker stop container_id           # Stop container
docker rm container_id             # Remove container
docker rmi exrt-ai                 # Remove image
docker system prune                # Clean unused resources
```

---

## 📝 Configuration Commands

### View Configuration
```bash
cat .env                           # View environment variables
docker-compose config              # View final docker-compose config
```

### Edit Configuration
```bash
# Update Gemini API key in .env
# Then restart: docker-compose restart
```

---

## 🌐 Cloud Deployment Commands

### AWS EC2
```bash
# Connect to instance
ssh -i "key.pem" ubuntu@instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Clone repo and run
git clone https://github.com/Deekshith8484/exait-ai.git
cd exait-ai
docker-compose up -d
```

### Google Cloud Run
```bash
# Push to container registry
gcloud auth configure-docker
docker tag exrt-ai gcr.io/PROJECT_ID/exrt-ai
docker push gcr.io/PROJECT_ID/exrt-ai

# Deploy
gcloud run deploy exrt-ai \
  --image gcr.io/PROJECT_ID/exrt-ai \
  --platform managed \
  --region us-central1 \
  --port 8501 \
  --allow-unauthenticated
```

### Docker Hub (for sharing)
```bash
docker login
docker tag exrt-ai:latest your-username/exrt-ai:latest
docker push your-username/exrt-ai:latest
```

---

## 📊 Monitoring Commands

### Check Application Health
```bash
# Local
curl http://localhost:8501

# Remote
curl http://your-server-ip:8501
```

### Monitor Resources
```bash
docker stats                    # Real-time resource usage
docker ps -a                   # Show all containers
docker system df              # Disk usage
```

---

## 🔄 Common Workflows

### Development Workflow
```bash
# 1. Make code changes
# 2. (Auto-reload for Streamlit, restart for Docker)
# 3. Test
# 4. Commit and push
git add .
git commit -m "Feature description"
git push origin main
```

### Deployment Workflow
```bash
# 1. Build image
docker build -t exrt-ai .

# 2. Test locally
docker-compose up -d

# 3. Push to registry (if deploying)
docker tag exrt-ai:latest docker.io/username/exrt-ai
docker push docker.io/username/exrt-ai

# 4. Deploy to cloud (cloud-specific)
# See DOCKER_GUIDE.md
```

### Debugging Workflow
```bash
# 1. Check if container is running
docker ps

# 2. View logs
docker-compose logs -f

# 3. Connect to container
docker exec -it container_id bash

# 4. Check configuration
docker-compose config

# 5. Restart if needed
docker-compose restart
```

---

## 🛠️ Useful One-Liners

```bash
# Restart container
docker-compose restart

# View last 50 log lines
docker-compose logs --tail=50

# Remove stopped containers
docker container prune

# Show all port mappings
docker port container_id

# Environment variables in container
docker exec container_id env

# Check container IP
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_id

# Export container as image
docker save exrt-ai:latest -o exrt-ai.tar

# Import container from tar
docker load -i exrt-ai.tar
```

---

## 📱 Access URLs

| Deployment | URL |
|-----------|-----|
| Local Streamlit | `http://localhost:8501` |
| Docker Local | `http://localhost:8501` |
| AWS EC2 | `http://ec2-instance-ip:8501` |
| Google Cloud Run | `https://project-name-xxx.a.run.app` |
| DigitalOcean | `http://droplet-ip:8501` |
| Custom Domain | `http://your-domain.com` |

---

## 🚨 Emergency Commands

### Force stop and remove everything
```bash
docker-compose down -v
docker rmi exrt-ai
docker system prune -a
```

### Restore from scratch
```bash
docker-compose down -v
docker build -t exrt-ai .
docker-compose up -d
```

### Free up disk space
```bash
docker system prune
docker system prune -a  # More aggressive
```

---

## 📚 Files to Know

```
Essential Files:
├─ app.py                 ← Main application
├─ requirements.txt       ← Dependencies
├─ .env                  ← Secrets (in .gitignore)
├─ Dockerfile            ← Container blueprint
└─ docker-compose.yml    ← Docker orchestration

Helper Scripts:
├─ docker-run.bat        ← Windows (Windows only)
└─ docker-run.sh         ← Linux/Mac (Linux/Mac only)

Documentation:
├─ DOCKER_START_HERE.md  ← Read this first!
├─ DOCKER_GUIDE.md       ← Complete guide
└─ README_DEPLOYMENT.md  ← Deployment overview
```

---

## 🎯 Command Cheatsheet

```bash
# Setup
streamlit run app.py                    # Local
docker-compose up -d                    # Docker
docker-compose logs -f                  # View logs

# Clean up
docker-compose down                     # Stop
docker system prune                     # Clean

# Debugging
docker ps                               # List containers
docker-compose config                   # Check config
curl http://localhost:8501              # Test endpoint

# Deployment
git push origin main                    # Push to Git
# Then deploy via cloud platform UI
```

---

## ✨ Pro Tips

✅ **Use docker-compose for local testing** - It's how production will run

✅ **Always commit code, never commit .env** - Security best practice

✅ **Check logs before asking for help** - They usually show the issue

✅ **Use health checks** - Already enabled in docker-compose.yml

✅ **Pin dependency versions** - Prevents "it works on my machine"

✅ **Tag your Docker images** - Makes it easy to track versions

---

**Ready to use these commands?** Start with Option 1, 2, or 3 above!

See [DOCKER_START_HERE.md](DOCKER_START_HERE.md) for complete setup guide.
