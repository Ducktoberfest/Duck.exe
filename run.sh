#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

if python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/api/tags', timeout=2)" >/dev/null 2>&1; then
    echo "[OK] Ollama is running."
else
    echo "[WARNING] Ollama is not running. The app will use the rule-based fallback plan."
    echo "          Start it with: ollama serve"
fi

python -m uvicorn backend.main:app --port 8000 &
BACKEND_PID=$!

trap 'kill $BACKEND_PID 2>/dev/null' EXIT INT TERM

sleep 2

python -m streamlit run frontend/app.py