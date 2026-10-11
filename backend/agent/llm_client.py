import json
import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("CLEANVISION_MODEL", "llama3.2:3b")


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def _num_ctx():
    try:
        return int(os.environ.get("CLEANVISION_NUM_CTX", "4096"))
    except Exception:
        return 4096


def ask_llm(system, user, json_mode=True, temperature=0.1, timeout=None):
    try:
        payload = {
            "model": os.environ.get("CLEANVISION_MODEL", MODEL),
            "messages": [
                {"role": "system", "content": str(system)},
                {"role": "user", "content": str(user)},
            ],
            "stream": False,
            "options": {"temperature": float(temperature), "num_ctx": _num_ctx()},
        }
        if json_mode:
            payload["format"] = "json"
        response = requests.post(
            OLLAMA_URL + "/api/chat",
            json=payload,
            timeout=float(timeout) if timeout else _env_float("CLEANVISION_TIMEOUT", 120),
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content", "")
        return content if isinstance(content, str) else ""
    except Exception:
        return ""


def llm_available():
    try:
        response = requests.get(OLLAMA_URL + "/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def warm_up():
    try:
        requests.post(
            OLLAMA_URL + "/api/chat",
            json={
                "model": os.environ.get("CLEANVISION_MODEL", MODEL),
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": _num_ctx()},
            },
            timeout=60,
        )
    except Exception:
        pass


def extract_json(text):
    try:
        if not isinstance(text, str):
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        result = json.loads(text[start:end + 1])
        return result if isinstance(result, dict) else None
    except Exception:
        return None