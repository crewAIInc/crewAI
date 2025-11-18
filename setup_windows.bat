@echo off
REM CrewAI Windows Setup Script for Local LLM with Ollama
REM Designed for Windows 11 with GTX 5080 + 64GB RAM
REM No Docker required - native Windows setup

echo ========================================
echo CrewAI Windows Setup for Local LLMs
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.10, 3.11, 3.12, or 3.13 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [OK] Found Python %PYTHON_VERSION%

REM Check Python version is compatible (3.10-3.13)
python -c "import sys; assert sys.version_info >= (3, 10) and sys.version_info < (3, 14), 'Python 3.10-3.13 required'" 2>nul
if errorlevel 1 (
    echo [ERROR] CrewAI requires Python 3.10, 3.11, 3.12, or 3.13
    echo Your version: %PYTHON_VERSION%
    pause
    exit /b 1
)

echo [OK] Python version is compatible
echo.

REM Create virtual environment
echo [STEP 1/5] Creating virtual environment...
if exist venv (
    echo Virtual environment already exists, skipping creation
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)
echo.

REM Activate virtual environment and install dependencies
echo [STEP 2/5] Installing CrewAI and dependencies...
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip --quiet

REM Install crewAI from local source
pip install -e . --quiet
if errorlevel 1 (
    echo [ERROR] Failed to install CrewAI
    pause
    exit /b 1
)

echo [OK] CrewAI installed successfully
echo.

REM Install optional but useful tools
echo [STEP 3/5] Installing optional tools (crewai-tools)...
pip install crewai-tools --quiet
if errorlevel 1 (
    echo [WARNING] Failed to install crewai-tools (optional)
) else (
    echo [OK] CrewAI tools installed
)
echo.

REM Check if Ollama is installed
echo [STEP 4/5] Checking Ollama installation...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama is not installed or not in PATH
    echo.
    echo To install Ollama:
    echo 1. Visit https://ollama.ai/download
    echo 2. Download and install Ollama for Windows
    echo 3. Run this setup script again
    echo.
    echo You can continue without Ollama, but you'll need it to run local models.
    echo.
    set OLLAMA_INSTALLED=0
) else (
    echo [OK] Ollama is installed
    ollama --version
    set OLLAMA_INSTALLED=1

    REM Check if Ollama service is running
    echo [INFO] Checking if Ollama service is running...
    curl -s http://localhost:11434/api/version >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Ollama service is not running
        echo Starting Ollama service...
        start /B ollama serve
        timeout /t 3 /nobreak >nul
        echo [OK] Ollama service started
    ) else (
        echo [OK] Ollama service is running
    )
)
echo.

REM Create .env file if it doesn't exist
echo [STEP 5/5] Setting up configuration...
if not exist .env (
    echo Creating .env file for Ollama configuration...
    (
        echo # CrewAI Configuration for Local Ollama LLMs
        echo # No API keys needed for local models!
        echo.
        echo # Default model - will be used if not specified in agent config
        echo # Recommended: qwen2.5:32b, deepseek-r1:14b, or phi4:14b
        echo MODEL=ollama/qwen2.5:32b
        echo.
        echo # Ollama API base URL
        echo OLLAMA_API_BASE=http://localhost:11434
        echo.
        echo # Optional: OpenAI API key if you want to use cloud models as fallback
        echo # OPENAI_API_KEY=your-key-here
        echo.
        echo # Optional: Set to 1 to enable detailed logging
        echo # CREWAI_DEBUG=0
    ) > .env
    echo [OK] Created .env file with Ollama defaults
) else (
    echo [OK] .env file already exists
)
echo.

REM Summary
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment: venv\
echo Configuration file: .env
echo.
if %OLLAMA_INSTALLED%==1 (
    echo [NEXT STEPS]
    echo 1. Run install_models.bat to download recommended MoE models
    echo 2. Run run_example.bat to test your setup
    echo 3. Check WINDOWS_SETUP.md for detailed documentation
) else (
    echo [NEXT STEPS]
    echo 1. Install Ollama from https://ollama.ai/download
    echo 2. Run install_models.bat to download recommended MoE models
    echo 3. Run run_example.bat to test your setup
    echo 4. Check WINDOWS_SETUP.md for detailed documentation
)
echo.

pause
