@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)" >nul 2>&1
if errorlevel 1 (
    start /b "" ollama serve >nul 2>&1
    for /l %%i in (1,1,10) do (
        timeout /t 1 /nobreak >nul 2>&1
        python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1)" >nul 2>&1
        if not errorlevel 1 goto :ollama_ready
    )
    echo [WARNING] Ollama is not running. The app will run in fallback mode.
    goto :start_server
)

:ollama_ready
echo [OK] Ollama is running.

:start_server
echo Open http://localhost:8000
python -m uvicorn backend.main:app --port 8000
