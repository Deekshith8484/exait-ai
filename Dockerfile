# EXRT AI - Multi-service Docker Container (Streamlit + FastAPI Backend)
# Build: docker build -t exrt-ai .
# Run: docker-compose up -d

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY api_backup.py .
COPY new_dashboard.html .
COPY analysis/ ./analysis/
COPY simulator/ ./simulator/
COPY .streamlit/ ./.streamlit/

# Expose both ports
EXPOSE 8501 8000

# Health check (can be overridden per service)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
