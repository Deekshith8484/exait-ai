# EXRT AI - Cloud Deployment Quick Reference

## Single Command Deployment

```bash
# 1. SSH into your cloud server
ssh -i your-key.pem ubuntu@your-server-ip

# 2. Prepare environment
mkdir -p ~/exrt-ai && cd ~/exrt-ai

# 3. Upload files (from your local machine)
scp -r -i your-key.pem ./* ubuntu@your-server-ip:~/exrt-ai/

# 4. Setup
ssh -i your-key.pem ubuntu@your-server-ip << 'EOF'
  cd ~/exrt-ai
  
  # Install Docker (if not already installed)
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker $USER
  newgrp docker
  
  # Create .env with your API key
  echo "Gemini_API_KEY=YOUR_API_KEY_HERE" > .env
  
  # Build and start
  docker-compose build
  docker-compose up -d
  
  # Verify
  docker-compose ps
EOF

# 5. Access application
echo "Application ready at: http://your-server-ip"
```

---

## Architecture Summary

| Component | Port | Purpose | Notes |
|-----------|------|---------|-------|
| **Nginx** | 80, 443 | Reverse Proxy | Entry point for all traffic |
| **Streamlit** | 8501 | UI/Frontend | Not directly exposed |
| **FastAPI** | 8000 | Backend API | Not directly exposed |
| **Docker Network** | bridge | Inter-service comms | Automatic DNS resolution |

---

## URL Routing (Handled by Nginx)

```
User Browser (http://your-ip or http://your-domain)
         ↓
      Nginx (Port 80)
         ├─ /               → Streamlit:8501 (UI)
         ├─ /_stcore/*      → Streamlit:8501 (WebSocket)
         ├─ /api/*          → FastAPI:8000 (Backend)
         └─ /upload/*       → FastAPI:8000 (File Upload)
```

---

## File Structure Required

```
exrt-ai/
├── app.py                 # Streamlit app (MUST EXIST)
├── api_backup.py          # FastAPI backend (MUST EXIST)
├── new_dashboard.html     # Dashboard UI (MUST EXIST)
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Service orchestration
├── nginx.conf            # Reverse proxy config
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (CREATE THIS)
├── analysis/             # ML models directory
├── simulator/            # ECG simulator
└── ... other files ...
```

---

## Essential Environment Variables

```bash
# .env file
Gemini_API_KEY=AIzaSyDoYWh1ar4DEyx7-q-S3au8u10fNdraJUk  # Your actual key
```

---

## Post-Deployment Commands

### Check Status
```bash
docker-compose ps                    # All services status
docker-compose logs -f --tail=50    # Last 50 lines of logs
docker stats                         # Resource usage
```

### Restart Services
```bash
docker-compose restart               # Restart all
docker-compose restart backend       # Restart specific service
docker-compose down && docker-compose up -d  # Full restart
```

### View Logs
```bash
docker-compose logs -f streamlit     # Streamlit logs
docker-compose logs -f backend       # Backend logs
docker-compose logs -f nginx         # Nginx logs
```

### Update Configuration
```bash
nano .env                    # Edit environment file
docker-compose restart       # Apply changes
```

---

## Testing Endpoints

### Health Checks
```bash
# Nginx health
curl http://your-ip/nginx-health

# Backend API health
curl http://your-ip/api/health

# Streamlit is running if you can access http://your-ip in browser
```

### API Endpoints
```bash
# Upload a file (multipart form)
curl -X POST \
  -F "file=@test.csv" \
  http://your-ip/api/upload/signal

# Get Gemini API key
curl http://your-ip/api/gemini-key

# Check readiness
curl http://your-ip/api/readiness
```

---

## Enable HTTPS (Let's Encrypt)

```bash
# 1. Install certbot
sudo apt-get install certbot python3-certbot-nginx -y

# 2. Get certificate
sudo certbot certonly --standalone -d your-domain.com

# 3. Update nginx.conf with SSL certificates
# (See CLOUD_DEPLOYMENT_GUIDE.md for full config)

# 4. Restart Nginx
docker-compose restart nginx

# 5. Auto-renewal
sudo systemctl enable certbot.timer
```

---

## Firewall Setup

```bash
# Configure UFW firewall
sudo ufw default deny incoming
sudo ufw allow 22/tcp                # SSH
sudo ufw allow 80/tcp                # HTTP
sudo ufw allow 443/tcp               # HTTPS
sudo ufw enable

# Check status
sudo ufw status
```

---

## Common Issues & Fixes

### Services won't start
```bash
docker-compose logs          # Check error messages
docker-compose down -v       # Remove volumes
docker-compose build --no-cache  # Rebuild
docker-compose up -d         # Start fresh
```

### Can't access application
```bash
curl http://localhost/nginx-health    # Test from server
docker-compose ps                     # Verify all running
sudo ufw status                       # Check firewall
```

### High memory usage
```bash
docker stats                # Identify culprit
docker-compose restart      # Restart services
```

### Upload fails
```bash
docker-compose logs backend | grep -i error   # Check backend
df -h                                         # Check disk space
docker exec exrt-ai-backend ls -la /app/     # Check permissions
```

---

## Performance Tuning

### Increase Resource Limits (docker-compose.yml)
```yaml
services:
  streamlit:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Server Recommendations
- **Small deployment** (< 100 users): 2 CPU, 4GB RAM
- **Medium** (100-1000 users): 4 CPU, 8GB RAM
- **Large** (1000+ users): 8+ CPU, 16GB+ RAM, Load balancing

---

## Backup & Recovery

### Backup
```bash
cd ~/exrt-ai
tar -czf backup-$(date +%Y%m%d).tar.gz \
  app.py api_backup.py new_dashboard.html \
  nginx.conf docker-compose.yml .env \
  requirements.txt analysis/ simulator/
```

### Restore
```bash
docker-compose down -v
tar -xzf backup-YYYYMMDD.tar.gz
docker-compose build
docker-compose up -d
```

---

## Cloud Provider Quick Links

### AWS EC2
1. Launch Ubuntu 20.04 instance (t2.medium minimum)
2. Security group: Allow 80, 443, 22 inbound
3. Elastic IP for static IP
4. SSH and follow deployment steps above

### Google Cloud
1. Create Compute Engine VM (e2-medium minimum)
2. Ubuntu 20.04 LTS image
3. Firewall: Allow 80, 443, 22
4. SSH and follow deployment steps above

### Azure
1. Create Virtual Machine (B2s minimum)
2. Ubuntu 20.04 LTS
3. NSG: Allow 80, 443, 22 inbound
4. SSH and follow deployment steps above

### DigitalOcean
1. Create Droplet (2GB RAM, 2 CPU minimum)
2. Ubuntu 20.04
3. SSH and follow deployment steps above

---

## Monitoring (Optional)

### Simple Health Monitor (runs on server)
```bash
# Check every 5 minutes (add to crontab)
*/5 * * * * curl -f http://localhost/nginx-health || docker-compose restart
```

### View Real-time Metrics
```bash
# CPU and Memory
docker stats --no-stream

# Continuous monitoring
watch -n 2 'docker stats --no-stream'
```

---

## Security Checklist

- [ ] SSH key-based authentication (no passwords)
- [ ] Firewall configured (UFW or Security Groups)
- [ ] HTTPS/SSL enabled for domain
- [ ] .env not committed to Git
- [ ] Regular Docker image updates
- [ ] Rate limiting in Nginx
- [ ] Regular backups scheduled
- [ ] Monitor logs for suspicious activity

---

## Support Files

- **Full Guide**: `CLOUD_DEPLOYMENT_GUIDE.md`
- **Docker Setup**: `DOCKER_COMPLETE.md` or `OPTION_A_COMPLETE.md`
- **Architecture**: `ARCHITECTURE.md`

---

**Status**: ✅ Ready for cloud deployment!

For detailed instructions, see `CLOUD_DEPLOYMENT_GUIDE.md`
