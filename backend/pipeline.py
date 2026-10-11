import json
import math
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.agent.insights import write_insights
from backend.agent.planner import build_plan
from backend.executor import execute_plan
from backend.loader import load_dataframe
from backend.profiler import profile_dataframe
from backend.visualizer import generate_charts

PREVIEW_ROWS = 50


def clean_json(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def elapsed(start):
    return round(time.perf_counter() - start, 3)


def make_preview(df, warnings, label):
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        text = df.head(PREVIEW_ROWS).to_json(
            orient="records", date_format="iso", default_handler=str
        )
        return clean_json(json.loads(text))
    except Exception as exc:
        warnings.append(f"Preview ({label}) could not be created: {exc}")
        return []


def normalize_log(log):
    items = []
    for index, item in enumerate(log, start=1):
        if not isinstance(item, dict):
            continue
        item = clean_json(item)
        item.setdefault("step", index)
        item.setdefault("action", "")
        item.setdefault("affected", 0)
        if item.get("status") not in ("ok", "failed"):
            item["status"] = "failed"
        items.append(item)
    return items


def normalize_charts(charts):
    items = []
    for index, chart in enumerate(charts, start=1):
        if not isinstance(chart, dict) or chart.get("figure_json") is None:
            continue
        if isinstance(chart.get("figure_json"), dict):
            chart = clean_json(chart)
        else:
            base = clean_json({k: v for k, v in chart.items() if k != "figure_json"})
            base["figure_json"] = chart["figure_json"]
            chart = base
        chart.setdefault("title", f"Chart {index}")
        chart.setdefault("kind", "unknown")
        items.append(chart)
    return items


def run(file_bytes, filename, lang, job_dir, job_id):
    total_start = time.perf_counter()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    warnings = []

    start = time.perf_counter()
    df = load_dataframe(file_bytes, filename)
    timings["load"] = elapsed(start)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("load_dataframe did not return a DataFrame")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The uploaded file contains no data rows.")

    start = time.perf_counter()
    profile_before = profile_dataframe(df)
    timings["profile_before"] = elapsed(start)
    if not isinstance(profile_before, dict):
        raise TypeError("profile_dataframe did not return a dict")
    profile_before = clean_json(profile_before)

    preview_before = make_preview(df, warnings, "before")

    start = time.perf_counter()
    used_fallback = False
    try:
        result = build_plan(profile_before)
        plan, used_fallback = result
        if not isinstance(plan, dict) or not isinstance(plan.get("steps"), list):
            raise ValueError("plan must be a dict with a 'steps' list")
        plan = clean_json(plan)
        plan["summary"] = str(plan.get("summary", ""))
        used_fallback = bool(used_fallback)
    except Exception as exc:
        plan = {"summary": "", "steps": []}
        used_fallback = True
        warnings.append(f"Plan could not be built: {exc}")
    timings["plan"] = elapsed(start)

    start = time.perf_counter()
    try:
        df_clean, log = execute_plan(df.copy(), plan)
        if not isinstance(df_clean, pd.DataFrame):
            raise TypeError("execute_plan did not return a DataFrame")
        log = normalize_log(log if isinstance(log, list) else [])
    except Exception as exc:
        df_clean = df.copy()
        log = []
        warnings.append(f"Plan could not be executed, original data is kept: {exc}")
    timings["execute"] = elapsed(start)

    start = time.perf_counter()
    download_url = f"/download/{job_id}"
    try:
        df_clean.to_csv(job_dir / "cleaned.csv", index=False, encoding="utf-8-sig")
    except Exception as exc:
        download_url = ""
        warnings.append(f"Cleaned file could not be saved: {exc}")
    timings["save"] = elapsed(start)

    start = time.perf_counter()
    try:
        profile_after = profile_dataframe(df_clean)
        if not isinstance(profile_after, dict):
            raise TypeError("profile_dataframe did not return a dict")
        profile_after = clean_json(profile_after)
    except Exception as exc:
        profile_after = {}
        warnings.append(f"Profile after cleaning failed: {exc}")
    timings["profile_after"] = elapsed(start)

    start = time.perf_counter()
    try:
        charts = generate_charts(df_clean, max_charts=3)
        charts = normalize_charts(charts if isinstance(charts, list) else [])
    except Exception as exc:
        charts = []
        warnings.append(f"Charts could not be generated: {exc}")
    timings["charts"] = elapsed(start)

    start = time.perf_counter()
    try:
        insights = write_insights(
            profile_before, profile_after, log, df_clean=df_clean, lang=lang
        )
        insights = insights if isinstance(insights, str) else str(insights)
    except Exception as exc:
        insights = ""
        warnings.append(f"Insights could not be written: {exc}")
    timings["insights"] = elapsed(start)

    preview_after = make_preview(df_clean, warnings, "after")
    timings["total"] = elapsed(total_start)

    return {
        "job_id": job_id,
        "filename": filename,
        "profile_before": profile_before,
        "plan": plan,
        "log": log,
        "profile_after": profile_after,
        "charts": charts,
        "insights": insights,
        "used_fallback": used_fallback,
        "timings": timings,
        "warnings": warnings,
        "download_url": download_url,
        "preview_before": preview_before,
        "preview_after": preview_after,
    }