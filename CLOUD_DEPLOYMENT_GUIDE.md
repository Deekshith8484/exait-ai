# EXRT AI - Cloud Deployment Guide with Nginx Reverse Proxy

## Overview

This guide explains how to deploy EXRT AI to the cloud using Docker, Docker Compose, and Nginx as a reverse proxy. This is **Option A-2**: Single URL with Nginx reverse proxy - the professional, production-ready setup for cloud deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CLOUD SERVER (e.g., AWS EC2)          │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │   NGINX Reverse Proxy (Port 80/443)              │  │
│  │   ├─ /              → Streamlit (8501)           │  │
│  │   ├─ /api/*         → FastAPI Backend (8000)     │  │
│  │   ├─ /upload/*      → FastAPI Backend (8000)     │  │
│  │   └─ /_stcore/*     → WebSocket to Streamlit     │  │
│  └──────────────────────────────────────────────────┘  │
│         ↓                          ↓                     │
│  ┌─────────────────┐       ┌──────────────────┐        │
│  │   Streamlit     │       │  FastAPI Backend │        │
│  │  :8501          │       │     :8000        │        │
│  │  (UI)           │       │  (ML Analysis)   │        │
│  └─────────────────┘       └──────────────────┘        │
│                                                          │
│  Connected via Docker Network: exrt-network (bridge)   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Benefits of This Setup

- **Single Entry Point**: Users access `http://your-domain.com` or `http://your-ip`
- **Professional**: Uses industry-standard reverse proxy (Nginx)
- **Scalable**: Can add load balancing, caching, and SSL/TLS termination
- **Easy SSL/HTTPS**: Configure once at Nginx level for all services
- **Secure**: Internal services not directly exposed to the internet
- **Cloud-Ready**: Works on AWS, Google Cloud, Azure, DigitalOcean, etc.

---

## Prerequisites

1. **Cloud Server**: Any Linux-based server (Ubuntu 20.04+ recommended)
   - AWS EC2
   - Google Cloud VM
   - Azure VM
   - DigitalOcean Droplet
   - Any VPS with SSH access

2. **Installed Software**:
   - Docker (20.10+)
   - Docker Compose (1.29+)
   - Git

3. **Network**:
   - Port 80 (HTTP) accessible from internet
   - Port 443 (HTTPS) accessible from internet (for production)
   - SSH access (port 22)

4. **Environment**:
   - Gemini API Key
   - Domain name (optional but recommended)

---

## Step 1: Prepare Cloud Server

### 1.1 Connect to Your Server

```bash
# SSH into your cloud server
ssh -i your-key.pem ubuntu@your-server-ip
```

### 1.2 Update System

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 1.3 Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker-compose --version
```

### 1.4 Create Application Directory

```bash
mkdir -p ~/exrt-ai
cd ~/exrt-ai
```

---

## Step 2: Deploy Application

### 2.1 Clone or Copy Repository

**Option A: Clone from Git**
```bash
git clone <your-repo-url> .
cd exrt-ai
```

**Option B: Upload Files**
```bash
# On your local machine
scp -r -i your-key.pem ./* ubuntu@your-server-ip:~/exrt-ai/

# Or use SCP for individual files if needed
```

### 2.2 Create .env File

```bash
# On the server, create .env with your Gemini API key
cat > .env << 'EOF'
Gemini_API_KEY=YOUR_ACTUAL_GEMINI_API_KEY_HERE
EOF
```

Replace `YOUR_ACTUAL_GEMINI_API_KEY_HERE` with your actual key.

### 2.3 Verify Files Are Present

```bash
# Check required files exist
ls -la

# Should see:
# - app.py
# - api_backup.py
# - new_dashboard.html
# - Dockerfile
# - docker-compose.yml
# - nginx.conf
# - requirements.txt
# - analysis/
# - simulator/
```

### 2.4 Build and Start Services

```bash
# Build Docker images (first time only, takes 2-5 minutes)
docker-compose build

# Start all services in background
docker-compose up -d

# Verify services are running
docker-compose ps

# Output should show:
# NAME              STATUS          PORTS
# exrt-ai-nginx     Up              0.0.0.0:80->80/tcp
# exrt-ai-streamlit Up              8501/tcp
# exrt-ai-backend   Up              8000/tcp
```

---

## Step 3: Verify Deployment

### 3.1 Check Logs

```bash
# View logs from all services
docker-compose logs -f

# View specific service logs
docker-compose logs -f streamlit
docker-compose logs -f backend
docker-compose logs -f nginx

# Press Ctrl+C to exit logs
```

### 3.2 Check Health Endpoints

```bash
# From your local machine or inside the server
curl http://your-server-ip/nginx-health
curl http://your-server-ip/api/health

# Should return 200 status
```

### 3.3 Access the Application

Open browser and navigate to:
```
http://your-server-ip
```

or if you have a domain:
```
http://your-domain.com
```

---

## Step 4: Production Configuration

### 4.1 Configure Domain Name (Optional)

1. Point your domain DNS to your server IP
2. Update your DNS records:
   ```
   example.com A 12.34.56.78
   ```

### 4.2 Enable HTTPS with Let's Encrypt (Recommended)

#### Install Certbot

```bash
sudo apt-get install certbot python3-certbot-nginx -y
```

#### Get SSL Certificate

```bash
# Obtain certificate (replace with your domain)
sudo certbot certonly --standalone -d your-domain.com -d www.your-domain.com

# This creates certificates in /etc/letsencrypt/live/your-domain.com/
```

#### Update nginx.conf for HTTPS

Replace the `listen 80;` section in `nginx.conf` with:

```nginx
# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS server block
server {
    listen 443 ssl http2;
    server_name your-domain.com www.your-domain.com;

    # SSL certificates from Let's Encrypt
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Rest of your nginx configuration...
    client_max_body_size 3G;
    
    location / {
        proxy_pass http://streamlit;
        # ... proxy settings ...
    }
    
    # ... rest of locations ...
}
```

#### Update docker-compose.yml

Mount the certificates:

```yaml
nginx:
  # ... other settings ...
  volumes:
    - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    - /etc/letsencrypt:/etc/letsencrypt:ro  # Add this line
```

#### Restart Services

```bash
docker-compose down
docker-compose up -d
```

#### Auto-Renewal

```bash
# Set up automatic renewal (runs daily)
sudo systemctl enable certbot.timer
sudo systemctl start certbot.timer

# Check status
sudo systemctl status certbot.timer
```

---

## Step 5: Monitoring and Maintenance

### 5.1 Monitor Services

```bash
# Watch service status in real-time
watch -n 2 'docker-compose ps'

# Check container resource usage
docker stats

# View service logs continuously
docker-compose logs -f --tail=50
```

### 5.2 Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart streamlit
docker-compose restart backend

# Full restart (stop and start)
docker-compose down
docker-compose up -d
```

### 5.3 Update Environment Variables

If you need to update your Gemini API key:

```bash
# Edit .env file
nano .env

# Update Gemini_API_KEY
# Save and exit (Ctrl+X, Y, Enter)

# Restart services to apply changes
docker-compose restart streamlit backend
```

### 5.4 View Application Metrics

Monitor uptime and performance:

```bash
# Tail Nginx logs
docker exec exrt-ai-nginx tail -f /var/log/nginx/access.log
docker exec exrt-ai-nginx tail -f /var/log/nginx/error.log

# Backend API logs
docker-compose logs -f backend | grep -i "error\|upload\|readiness"
```

---

## Step 6: Backup and Recovery

### 6.1 Backup Configuration

```bash
# Backup everything important
tar -czf exrt-ai-backup-$(date +%Y%m%d).tar.gz \
    app.py \
    api_backup.py \
    new_dashboard.html \
    nginx.conf \
    docker-compose.yml \
    Dockerfile \
    .env \
    requirements.txt \
    analysis/ \
    simulator/

# Download backup to local machine
scp -r -i your-key.pem ubuntu@your-server-ip:~/exrt-ai/exrt-ai-backup-*.tar.gz .
```

### 6.2 Disaster Recovery

```bash
# Stop services
docker-compose down -v

# Remove volumes and images
docker system prune -a --volumes

# Restore from backup
tar -xzf exrt-ai-backup-YYYYMMDD.tar.gz

# Rebuild and restart
docker-compose build
docker-compose up -d
```

---

## Step 7: Scaling and Optimization

### 7.1 Increase Resource Limits

Edit `docker-compose.yml`:

```yaml
services:
  streamlit:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
  
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 7.2 Add Caching Headers

Update `nginx.conf` for better performance:

```nginx
# Add after proxy_pass to Streamlit
proxy_cache_valid 200 1m;
proxy_cache_key "$scheme$request_method$host$request_uri";
add_header X-Cache-Status $upstream_cache_status;
```

---

## Troubleshooting

### Issue: Services Won't Start

```bash
# Check Docker daemon is running
docker ps

# View detailed error logs
docker-compose logs

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Issue: Can't Access Application

```bash
# Check if Nginx is running
curl http://localhost/nginx-health

# Check all services
docker-compose ps

# Check firewall rules
sudo ufw status
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
```

### Issue: Upload Fails

```bash
# Check backend logs
docker-compose logs backend | grep -i "error\|upload"

# Verify file permissions
docker exec exrt-ai-backend ls -la /app/

# Check disk space
docker exec exrt-ai-backend df -h
```

### Issue: High Memory Usage

```bash
# Monitor resource usage
docker stats

# Check what's consuming memory
docker-compose logs streamlit | tail -100

# Restart services to free memory
docker-compose restart
```

---

## Security Best Practices

### 1. Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### 2. Secure Environment Variables

```bash
# Don't commit .env to Git
echo ".env" >> .gitignore

# Use proper file permissions
chmod 600 .env

# Consider using secrets management:
# - AWS Secrets Manager
# - Google Cloud Secret Manager
# - HashiCorp Vault
```

### 3. Regular Updates

```bash
# Update base images regularly
docker-compose down
docker pull nginx:alpine
docker pull python:3.10-slim
docker-compose build --no-cache
docker-compose up -d
```

### 4. Rate Limiting

Add to `nginx.conf`:

```nginx
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;

location /api/ {
    limit_req zone=api burst=20 nodelay;
    proxy_pass http://backend/;
}
```

---

## Next Steps

1. **Monitor**: Set up monitoring with tools like Prometheus/Grafana
2. **Logging**: Centralize logs with ELK Stack or CloudWatch
3. **Scaling**: Use Kubernetes (K8s) for multiple instances
4. **CI/CD**: Set up GitHub Actions for automatic deployments
5. **Analytics**: Track user activity and application metrics

---

## Support Commands Reference

```bash
# Quick status check
docker-compose ps

# Detailed logs
docker-compose logs -f --tail=100

# Restart everything
docker-compose restart

# Full restart
docker-compose down && docker-compose up -d

# Check resource usage
docker stats

# Cleanup unused resources
docker system prune -a

# Access Nginx container
docker exec -it exrt-ai-nginx sh

# Access Backend container
docker exec -it exrt-ai-backend sh

# Access Streamlit container
docker exec -it exrt-ai-streamlit sh
```

---

## FAQ

**Q: How do I update the application code?**
A: Update files on server, then run `docker-compose restart` to reload changes (volumes are mounted).

**Q: Can I use HTTPS without a domain?**
A: Yes, but with a self-signed certificate. Not recommended for production.

**Q: How much server resources do I need?**
A: Minimum: 2 CPU cores, 4GB RAM. Recommended: 4 CPU cores, 8GB RAM.

**Q: How do I handle file uploads larger than 3GB?**
A: Edit `nginx.conf`: Change `client_max_body_size 3G;` to your desired limit.

**Q: Can I run this on Windows/Mac?**
A: Docker Desktop works on both. Use Docker for Mac or Docker for Windows with Linux containers.

---

## Quick Start Checklist

- [ ] Server provisioned and SSH accessible
- [ ] Docker and Docker Compose installed
- [ ] Repository cloned or files uploaded
- [ ] `.env` file created with Gemini API key
- [ ] `docker-compose build` completed successfully
- [ ] `docker-compose up -d` running
- [ ] Nginx health check returns 200
- [ ] Application loads at `http://your-ip`
- [ ] File upload works
- [ ] AI chat widget responds
- [ ] SSL/HTTPS configured (optional)
- [ ] Domain name pointed to server (optional)

---

**Deployment Complete!** Your EXRT AI application is now running on the cloud! 🚀
