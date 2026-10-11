import sys
from pathlib import Path
import requests

BASE_URL = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "sample_data" / "dirty_sales.csv"

CONTRACT_FIELDS = {
    "job_id",
    "filename",
    "profile_before",
    "plan",
    "log",
    "profile_after",
    "charts",
    "insights",
    "used_fallback",
    "timings",
    "warnings",
    "download_url",
    "preview_before",
    "preview_after",
}


def main():
    issues = []

    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        if r.status_code != 200:
            issues.append(f"GET / returned status {r.status_code}")
        elif "CleanVision" not in r.text:
            issues.append("GET / did not return CleanVision frontend HTML")
    except Exception as exc:
        issues.append(f"Could not connect to {BASE_URL}/: {exc}")

    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        if r.status_code != 200:
            issues.append(f"GET /health returned status {r.status_code}")
        else:
            data = r.json()
            if data.get("status") != "ok" or "llm" not in data:
                issues.append(f"GET /health invalid response: {data}")
    except Exception as exc:
        issues.append(f"GET /health failed: {exc}")

    try:
        r = requests.post(f"{BASE_URL}/analyze?mock=true", timeout=10)
        if r.status_code != 200:
            issues.append(f"POST /analyze?mock=true returned status {r.status_code}")
        else:
            data = r.json()
            missing = CONTRACT_FIELDS - set(data.keys())
            if missing:
                issues.append(f"POST /analyze?mock=true missing fields: {missing}")
    except Exception as exc:
        issues.append(f"POST /analyze?mock=true failed: {exc}")

    if SAMPLE_CSV.is_file():
        try:
            with open(SAMPLE_CSV, "rb") as handle:
                r = requests.post(
                    f"{BASE_URL}/analyze?lang=en",
                    files={"file": ("dirty_sales.csv", handle, "text/csv")},
                    timeout=30,
                )
            if r.status_code != 200:
                issues.append(f"POST /analyze real file returned status {r.status_code}")
            else:
                data = r.json()
                missing = CONTRACT_FIELDS - set(data.keys())
                if missing:
                    issues.append(f"POST /analyze real file missing fields: {missing}")
                if not data.get("download_url"):
                    issues.append("POST /analyze real file missing download_url")
                else:
                    dl_url = f"{BASE_URL}{data['download_url']}"
                    dl = requests.get(dl_url, timeout=10)
                    if dl.status_code != 200:
                        issues.append(f"GET {dl_url} returned status {dl.status_code}")
        except Exception as exc:
            issues.append(f"POST /analyze real file failed: {exc}")
    else:
        issues.append(f"Sample data file not found: {SAMPLE_CSV}")

    if issues:
        print("ISSUES DETECTED:")
        for issue in issues:
            print(f"- {issue}")
        sys.exit(1)
    else:
        print("DEMO IS READY")
        sys.exit(0)


if __name__ == "__main__":
    main()
