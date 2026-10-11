import json
import os
import requests

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
CLEANVISION_MODEL = os.environ.get("CLEANVISION_MODEL", "llama3.2:3b")


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


def ask_llm(system, user, json_mode=True, temperature=0.0, timeout=None):
    try:
        host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
        model = os.environ.get("CLEANVISION_MODEL", CLEANVISION_MODEL)
        payload = {
            "model": model,
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
            f"{host}/api/chat",
            json=payload,
            timeout=float(timeout) if timeout else _env_float("CLEANVISION_TIMEOUT", 60),
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
        host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
        target_model = os.environ.get("CLEANVISION_MODEL", CLEANVISION_MODEL).strip().lower()
        response = requests.get(f"{host}/api/tags", timeout=2)
        if response.status_code != 200:
            return False
        data = response.json()
        models = data.get("models")
        if not isinstance(models, list):
            return False
        for m in models:
            name = str(m.get("name", "")).strip().lower()
            if name == target_model or name.startswith(f"{target_model}:") or target_model in name:
                return True
        return False
    except Exception:
        return False


def warm_up():
    try:
        host = os.environ.get("OLLAMA_HOST", OLLAMA_HOST).rstrip("/")
        model = os.environ.get("CLEANVISION_MODEL", CLEANVISION_MODEL)
        requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "ping"}],
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1, "temperature": 0, "num_ctx": _num_ctx()},
            },
            timeout=10,
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
        result = json.loads(text[start : end + 1])
        return result if isinstance(result, dict) else None
    except Exception:
        return None