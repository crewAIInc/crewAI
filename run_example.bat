@echo off
REM Run the Windows Local LLM Example
REM This demonstrates CrewAI with local Ollama models

echo ========================================
echo CrewAI Local LLM Example
echo ========================================
echo.

REM Check if virtual environment exists
if not exist venv (
    echo [ERROR] Virtual environment not found!
    echo Please run setup_windows.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if .env exists
if not exist .env (
    echo [WARNING] .env file not found, using defaults
    echo Creating .env from .env.example...
    copy .env.example .env
    echo [OK] Created .env file
    echo.
)

REM Check if Ollama is running
echo [INFO] Checking Ollama connection...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama service is not responding
    echo.
    echo Starting Ollama service...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    echo [OK] Ollama service started
    echo.
)

REM Run the example
echo [INFO] Running CrewAI example with local models...
echo.
echo ========================================
echo.

cd windows_local_example
python main.py %*

echo.
echo ========================================
echo.

REM Check exit code
if errorlevel 1 (
    echo [ERROR] Example failed to run
    echo.
    echo Common issues:
    echo   1. Models not installed - run install_models.bat
    echo   2. Ollama not running - it should auto-start
    echo   3. Out of memory - try smaller models
    echo.
) else (
    echo [SUCCESS] Example completed!
    echo Check windows_local_example\output\ for results
    echo.
)

pause
