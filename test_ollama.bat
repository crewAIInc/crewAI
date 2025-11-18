@echo off
REM Test Ollama installation and models

echo ========================================
echo Ollama Installation Test
echo ========================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Ollama is not installed or not in PATH
    echo Please install Ollama from https://ollama.ai/download
    pause
    exit /b 1
)

echo [PASS] Ollama is installed
ollama --version
echo.

REM Check if Ollama service is running
echo [TEST] Checking Ollama service...
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [WARN] Ollama service is not running
    echo Attempting to start...
    start /B ollama serve
    timeout /t 3 /nobreak >nul

    curl -s http://localhost:11434/api/version >nul 2>&1
    if errorlevel 1 (
        echo [FAIL] Could not start Ollama service
        echo Please run 'ollama serve' manually
        pause
        exit /b 1
    ) else (
        echo [PASS] Ollama service started successfully
    )
) else (
    echo [PASS] Ollama service is running
)
echo.

REM Get version info
echo [INFO] Ollama version info:
curl -s http://localhost:11434/api/version
echo.
echo.

REM List installed models
echo [INFO] Installed models:
ollama list
echo.

REM Check for recommended models
echo [TEST] Checking for recommended models...

set MODELS_FOUND=0

ollama list | findstr /C:"qwen2.5:32b" >nul 2>&1
if not errorlevel 1 (
    echo [PASS] qwen2.5:32b found
    set /a MODELS_FOUND+=1
) else (
    echo [MISS] qwen2.5:32b not installed
)

ollama list | findstr /C:"deepseek-r1:14b" >nul 2>&1
if not errorlevel 1 (
    echo [PASS] deepseek-r1:14b found
    set /a MODELS_FOUND+=1
) else (
    echo [MISS] deepseek-r1:14b not installed
)

ollama list | findstr /C:"phi4:14b" >nul 2>&1
if not errorlevel 1 (
    echo [PASS] phi4:14b found
    set /a MODELS_FOUND+=1
) else (
    echo [MISS] phi4:14b not installed
)

ollama list | findstr /C:"llama3.2:3b" >nul 2>&1
if not errorlevel 1 (
    echo [PASS] llama3.2:3b found
    set /a MODELS_FOUND+=1
) else (
    echo [MISS] llama3.2:3b not installed
)

echo.

if %MODELS_FOUND% EQU 0 (
    echo [WARN] No recommended models installed
    echo Run install_models.bat to download them
) else if %MODELS_FOUND% LSS 4 (
    echo [INFO] %MODELS_FOUND%/4 recommended models installed
    echo Run install_models.bat to get the rest
) else (
    echo [PASS] All recommended models installed!
)

echo.

REM Test a simple prompt (if a model is available)
ollama list | findstr /C:"qwen2.5:32b" >nul 2>&1
if not errorlevel 1 (
    echo [TEST] Testing qwen2.5:32b with a simple prompt...
    echo.
    echo Prompt: "Say 'Hello from Ollama' in one sentence"
    echo.
    ollama run qwen2.5:32b "Say 'Hello from Ollama' in one sentence" --verbose
    echo.
    echo [PASS] Model response received
) else (
    echo [SKIP] No models available to test
    echo Install models with: ollama pull qwen2.5:32b
)

echo.
echo ========================================
echo Test Complete!
echo ========================================
echo.

pause
