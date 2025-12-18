#!/bin/bash
# EXRT AI - Docker Quick Commands

echo ""
echo "================== EXRT AI - Docker Commands =================="
echo ""
echo "Build Docker image:"
echo "  docker build -t exrt-ai ."
echo ""
echo "Run container (single):"
echo "  docker run -p 8501:8501 -e Gemini_API_KEY='YOUR_KEY' exrt-ai"
echo ""
echo "Run with docker-compose:"
echo "  docker-compose up -d"
echo ""
echo "Stop container:"
echo "  docker-compose down"
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Access app:"
echo "  http://localhost:8501"
echo ""
echo "=============================================================="
echo ""

# Check for Docker installation
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    echo "Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

docker --version

# Parse command
case "$1" in
    build)
        echo "Building Docker image..."
        docker build -t exrt-ai .
        echo "Build complete! Run with: docker run -p 8501:8501 -e Gemini_API_KEY='YOUR_KEY' exrt-ai"
        ;;
    up)
        echo "Starting container with docker-compose..."
        if [ ! -f .env ]; then
            echo "ERROR: .env file not found"
            echo "Create .env with: Gemini_API_KEY=YOUR_KEY"
            exit 1
        fi
        docker-compose up -d
        echo "Container started. Access at: http://localhost:8501"
        docker-compose logs -f
        ;;
    down)
        echo "Stopping container..."
        docker-compose down
        echo "Container stopped."
        ;;
    logs)
        echo "Showing logs..."
        docker-compose logs -f
        ;;
    *)
        echo "Usage:"
        echo "  ./docker-run.sh build    - Build Docker image"
        echo "  ./docker-run.sh up       - Start container"
        echo "  ./docker-run.sh down     - Stop container"
        echo "  ./docker-run.sh logs     - View logs"
        ;;
esac
