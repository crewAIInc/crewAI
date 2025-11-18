@echo off
REM Ollama Model Installation Script
REM Optimized for GTX 5080 with 64GB RAM
REM Downloads recommended Mixture-of-Experts (MoE) models

echo ========================================
echo Ollama Model Installer for CrewAI
echo Optimized for GTX 5080 + 64GB RAM
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

echo [OK] Ollama found
ollama --version
echo.

REM Check if Ollama service is running
curl -s http://localhost:11434/api/version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama service is not running
    echo Starting Ollama service...
    start /B ollama serve
    timeout /t 5 /nobreak >nul
    echo [OK] Ollama service started
)
echo.

echo ========================================
echo RECOMMENDED MODELS FOR YOUR SETUP
echo ========================================
echo.
echo Your system: GTX 5080 with 64GB RAM
echo.
echo MoE (Mixture of Experts) Models:
echo   1. Qwen2.5:32b       - Excellent reasoning, 32B params (RECOMMENDED)
echo   2. Qwen2.5:14b       - Lighter Qwen variant, 14B params
echo   3. DeepSeek-R1:14b   - DeepSeek reasoning model, 14B params
echo   4. DeepSeek-R1:32b   - Larger DeepSeek, 32B params
echo   5. Phi-4:14b         - Microsoft Phi-4, 14B params
echo.
echo Standard Models (for comparison):
echo   6. Llama3.3:70b      - Meta's latest, 70B params (requires quantization)
echo   7. Llama3.2:3b       - Fast, small model for quick tasks
echo.
echo All models - Download everything (recommended)
echo.
echo ========================================
echo.

set /p CHOICE="Enter your choice (1-7 or 'all'): "

if /i "%CHOICE%"=="all" goto INSTALL_ALL
if "%CHOICE%"=="1" goto INSTALL_QWEN32
if "%CHOICE%"=="2" goto INSTALL_QWEN14
if "%CHOICE%"=="3" goto INSTALL_DEEPSEEK14
if "%CHOICE%"=="4" goto INSTALL_DEEPSEEK32
if "%CHOICE%"=="5" goto INSTALL_PHI4
if "%CHOICE%"=="6" goto INSTALL_LLAMA70
if "%CHOICE%"=="7" goto INSTALL_LLAMA3

echo [ERROR] Invalid choice
pause
exit /b 1

:INSTALL_ALL
echo.
echo ========================================
echo Installing ALL recommended models...
echo This will take 30-60 minutes depending on your internet speed
echo Total download size: ~100GB
echo ========================================
echo.

call :PULL_MODEL qwen2.5:32b
call :PULL_MODEL qwen2.5:14b
call :PULL_MODEL deepseek-r1:14b
call :PULL_MODEL deepseek-r1:32b
call :PULL_MODEL phi4:14b
call :PULL_MODEL llama3.3:70b
call :PULL_MODEL llama3.2:3b

goto DONE

:INSTALL_QWEN32
call :PULL_MODEL qwen2.5:32b
goto DONE

:INSTALL_QWEN14
call :PULL_MODEL qwen2.5:14b
goto DONE

:INSTALL_DEEPSEEK14
call :PULL_MODEL deepseek-r1:14b
goto DONE

:INSTALL_DEEPSEEK32
call :PULL_MODEL deepseek-r1:32b
goto DONE

:INSTALL_PHI4
call :PULL_MODEL phi4:14b
goto DONE

:INSTALL_LLAMA70
echo.
echo [NOTE] Llama3.3:70b is large - will use quantization on your GPU
call :PULL_MODEL llama3.3:70b
goto DONE

:INSTALL_LLAMA3
call :PULL_MODEL llama3.2:3b
goto DONE

:PULL_MODEL
echo.
echo ----------------------------------------
echo Downloading: %~1
echo ----------------------------------------
ollama pull %~1
if errorlevel 1 (
    echo [ERROR] Failed to download %~1
) else (
    echo [OK] %~1 downloaded successfully
)
echo.
goto :eof

:DONE
echo.
echo ========================================
echo Model Installation Complete!
echo ========================================
echo.
echo To see all installed models:
echo   ollama list
echo.
echo To test a model:
echo   ollama run qwen2.5:32b
echo.
echo To use in CrewAI, update your .env file:
echo   MODEL=ollama/qwen2.5:32b
echo.
echo Or specify in agents.yaml:
echo   llm: ollama/qwen2.5:32b
echo.

REM List installed models
echo [INSTALLED MODELS]
ollama list
echo.

pause
