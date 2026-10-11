import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app

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


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data["llm"], bool)


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "CleanVision" in response.text


def test_analyze_mock_mode(client):
    response = client.post("/analyze?mock=true")
    assert response.status_code == 200
    data = response.json()
    assert CONTRACT_FIELDS <= set(data.keys())


def test_analyze_real_csv_and_download(client):
    assert SAMPLE_CSV.is_file()
    content = SAMPLE_CSV.read_bytes()
    response = client.post(
        "/analyze?lang=en",
        files={"file": ("dirty_sales.csv", io.BytesIO(content), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert CONTRACT_FIELDS <= set(data.keys())
    assert isinstance(data["plan"]["steps"], list) and len(data["plan"]["steps"]) > 0
    assert isinstance(data["log"], list) and len(data["log"]) > 0
    assert isinstance(data["charts"], list) and len(data["charts"]) > 0
    assert isinstance(data["insights"], str) and len(data["insights"].strip()) > 0
    assert isinstance(data["preview_before"], list) and len(data["preview_before"]) > 0
    assert isinstance(data["preview_after"], list) and len(data["preview_after"]) > 0
    assert data["download_url"].startswith("/download/")

    dl_response = client.get(data["download_url"])
    assert dl_response.status_code == 200
    assert len(dl_response.content) > 0


def test_analyze_wrong_extension(client):
    response = client.post(
        "/analyze",
        files={"file": ("data.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "File format not supported."


def test_analyze_file_too_large(client):
    huge_bytes = b"0" * (21 * 1024 * 1024)
    response = client.post(
        "/analyze",
        files={"file": ("huge.csv", io.BytesIO(huge_bytes), "text/csv")},
    )
    assert response.status_code == 413
    assert response.json()["detail"] == "File is too large (20 MB limit)."


def test_analyze_header_only_csv(client):
    header_only = b"id,name,age\n"
    response = client.post(
        "/analyze",
        files={"file": ("empty.csv", io.BytesIO(header_only), "text/csv")},
    )
    assert response.status_code == 500
    assert "no data rows" in response.json()["detail"].lower()


def test_analyze_llm_failure_fallback(client, monkeypatch):
    import backend.agent.planner as planner_mod
    import backend.agent.llm_client as client_mod

    monkeypatch.setattr(planner_mod, "llm_available", lambda: False)
    monkeypatch.setattr(client_mod, "llm_available", lambda: False)

    content = SAMPLE_CSV.read_bytes()
    response = client.post(
        "/analyze?lang=en",
        files={"file": ("dirty_sales.csv", io.BytesIO(content), "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["used_fallback"] is True
    assert isinstance(data["insights"], str) and len(data["insights"].strip()) > 0
