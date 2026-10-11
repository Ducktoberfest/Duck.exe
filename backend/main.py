import json
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from backend import pipeline
from backend.schemas import AnalyzeResponse

try:
    from backend.agent.llm_client import llm_available, warm_up
except Exception:

    def llm_available():
        return False

    def warm_up():
        pass


ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "jobs"
FIXTURE_PATH = ROOT / "fixtures" / "sample_response.json"
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_BYTES = 20 * 1024 * 1024
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{8}$")


def _warmup_background():
    try:
        if llm_available():
            warm_up()
    except Exception:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warmup_background, daemon=True).start()
    yield


app = FastAPI(title="CleanVision API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def builtin_mock(filename):
    return {
        "job_id": "mock-job",
        "filename": filename,
        "profile_before": {
            "rows": 120,
            "columns": 4,
            "duplicate_rows": 6,
            "total_null_cells": 18,
            "columns_info": [],
        },
        "plan": {
            "summary": "Remove duplicates and fill missing values.",
            "steps": [
                {"action": "drop_duplicates"},
                {"action": "fill_nulls", "column": "price", "method": "median"},
            ],
        },
        "log": [
            {
                "step": 1,
                "action": "drop_duplicates",
                "affected": 6,
                "status": "ok",
                "rows_before": 120,
                "rows_after": 114,
            },
            {
                "step": 2,
                "action": "fill_nulls",
                "column": "price",
                "affected": 18,
                "status": "ok",
                "fill_value": 42.5,
            },
        ],
        "profile_after": {
            "rows": 114,
            "columns": 4,
            "duplicate_rows": 0,
            "total_null_cells": 0,
            "columns_info": [],
        },
        "charts": [],
        "insights": "### What changed\n- 6 duplicate rows were removed.\n- 18 empty cells were filled.",
        "used_fallback": False,
        "timings": {"total": 0.0},
        "warnings": ["Built-in mock response is used."],
        "download_url": "",
    }


def load_mock(filename):
    try:
        with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return AnalyzeResponse.model_validate(data)
    except Exception:
        return AnalyzeResponse.model_validate(builtin_mock(filename or "sample.csv"))


def read_meta_name(job_dir):
    try:
        with open(job_dir / "meta.json", "r", encoding="utf-8") as handle:
            name = json.load(handle).get("filename")
        if isinstance(name, str) and name:
            return name
    except Exception:
        pass
    return None


@app.get("/health")
def health():
    try:
        available = bool(llm_available())
    except Exception:
        available = False
    return {"status": "ok", "llm": available}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    file: Optional[UploadFile] = File(None),
    mock: bool = False,
    lang: str = "az",
):
    if mock:
        response = load_mock(file.filename if file is not None else None)
        if file is not None and file.filename:
            response.filename = Path(file.filename).name
        return response

    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="File format not supported.")

    filename = Path(file.filename).name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="File format not supported.",
        )

    try:
        file_bytes = await file.read(MAX_BYTES + 1)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the uploaded file: {exc}")

    if len(file_bytes) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (20 MB limit).")
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    try:
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / f"raw{extension}").write_bytes(file_bytes)
        with open(job_dir / "meta.json", "w", encoding="utf-8") as handle:
            json.dump({"filename": filename}, handle, ensure_ascii=False)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save the uploaded file: {exc}")

    try:
        result = await run_in_threadpool(
            pipeline.run, file_bytes, filename, lang, job_dir, job_id
        )
        response = AnalyzeResponse.model_validate(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")

    if not response.download_url and (job_dir / "cleaned.csv").exists():
        response.download_url = f"/download/{job_id}"
    return response


@app.get("/download/{job_id}")
def download(job_id: str):
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=404, detail="Cleaned file not found.")
    job_dir = JOBS_DIR / job_id
    cleaned_path = job_dir / "cleaned.csv"
    if not cleaned_path.is_file():
        raise HTTPException(status_code=404, detail="Cleaned file not found.")
    original = read_meta_name(job_dir) or job_id
    return FileResponse(
        cleaned_path,
        media_type="text/csv",
        filename=f"cleaned_{Path(original).stem}.csv",
    )


@app.get("/", include_in_schema=False)
def serve_index():
    index_file = ROOT / "frontend" / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")