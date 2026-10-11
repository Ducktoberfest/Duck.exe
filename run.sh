#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if ! python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)" >/dev/null 2>&1; then
    ollama serve >/dev/null 2>&1 &
    for i in $(seq 1 10); do
        sleep 1
        if python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=1)" >/dev/null 2>&1; then
            break
        fi
    done
fi

if python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)" >/dev/null 2>&1; then
    echo "[OK] Ollama is running."
else
    echo "[WARNING] Ollama is not running. The app will run in fallback mode."
fi

echo "Open http://localhost:8000"
exec python -m uvicorn backend.main:app --port 8000