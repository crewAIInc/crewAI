@echo off
REM Start Ollama service
REM This keeps Ollama running in the foreground so you can see logs

echo ========================================
echo Starting Ollama Service
echo ========================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not installed or not in PATH
    echo Please install Ollama from https://ollama.ai/download
    pause
    exit /b 1
)

echo [INFO] Ollama found
ollama --version
echo.

echo [INFO] Starting Ollama server...
echo This window will show Ollama logs
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

ollama serve
