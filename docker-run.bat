@echo off
REM EXRT AI - Docker Quick Commands

echo.
echo ================== EXRT AI - Docker Commands ==================
echo.
echo Build Docker image:
echo   docker build -t exrt-ai .
echo.
echo Run container (single):
echo   docker run -p 8501:8501 -e Gemini_API_KEY="YOUR_KEY" exrt-ai
echo.
echo Run with docker-compose:
echo   docker-compose up -d
echo.
echo Stop container:
echo   docker-compose down
echo.
echo View logs:
echo   docker-compose logs -f
echo.
echo Access app:
echo   http://localhost:8501
echo.
echo ================================================================
echo.

REM Check for Docker installation
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

echo Docker is installed: %DOCKER_VERSION%

REM Optional: Build and run
if "%1"=="build" (
    echo Building Docker image...
    docker build -t exrt-ai .
    echo Build complete! Run with: docker run -p 8501:8501 -e Gemini_API_KEY="YOUR_KEY" exrt-ai
)

if "%1"=="up" (
    echo Starting container with docker-compose...
    if not exist .env (
        echo ERROR: .env file not found
        echo Create .env with: Gemini_API_KEY=YOUR_KEY
        pause
        exit /b 1
    )
    docker-compose up -d
    echo Container started. Access at: http://localhost:8501
    docker-compose logs -f
)

if "%1"=="down" (
    echo Stopping container...
    docker-compose down
    echo Container stopped.
)

if "%1"=="logs" (
    echo Showing logs...
    docker-compose logs -f
)

if "%1"=="" (
    echo Usage:
    echo   docker-run.bat build    - Build Docker image
    echo   docker-run.bat up       - Start container
    echo   docker-run.bat down     - Stop container
    echo   docker-run.bat logs     - View logs
)

pause
