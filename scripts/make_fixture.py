import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.executor import execute_plan
from backend.loader import load_dataframe
from backend.profiler import profile_dataframe, to_jsonable
from backend.visualizer import generate_charts

PLAN = {
    "summary": "Trim whitespace, normalize city names, remove duplicates, fix column types, handle outliers, fill missing values and drop the empty Notes column.",
    "steps": [
        {"action": "strip_whitespace", "columns": ["City", "Notes"]},
        {"action": "standardize_text", "columns": ["City"], "case": "title"},
        {"action": "drop_duplicates"},
        {"action": "convert_type", "column": "Date", "to": "datetime"},
        {"action": "convert_type", "column": "Price", "to": "numeric"},
        {"action": "convert_type", "column": "Quantity", "to": "numeric"},
        {"action": "remove_outliers", "column": "Price", "method": "clip_iqr"},
        {"action": "fill_nulls", "column": "Price", "method": "median"},
        {"action": "fill_nulls", "column": "Quantity", "method": "median"},
        {"action": "fill_nulls", "column": "City", "method": "mode"},
        {"action": "drop_column", "column": "Notes", "reason": "Column is almost entirely empty"},
    ],
}


def build_insights(before, after, log):
    ok_steps = sum(1 for entry in log if entry.get("status") == "ok")
    failed_steps = len(log) - ok_steps
    removed_rows = before["rows"] - after["rows"]
    lines = [
        "## Nəticə",
        f"- Sətir sayı: {before['rows']} → {after['rows']} ({removed_rows} sətir çıxarıldı).",
        f"- Sütun sayı: {before['columns']} → {after['columns']}.",
        f"- Təkrarlanan sətirlər: {before['duplicate_rows']} → {after['duplicate_rows']}.",
        f"- Boş xanalar: {before['total_null_cells']} → {after['total_null_cells']}.",
        f"- İcra olunan addımlar: {ok_steps} uğurlu, {failed_steps} uğursuz.",
    ]
    return "\n".join(lines)


def main():
    started = time.perf_counter()
    path = ROOT / "sample_data" / "dirty_sales.csv"
    t0 = time.perf_counter()
    df = load_dataframe(path.read_bytes(), path.name)
    load_ms = round((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    profile_before = profile_dataframe(df)
    profile_ms = round((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    clean, log = execute_plan(df, PLAN)
    execute_ms = round((time.perf_counter() - t0) * 1000)

    profile_after = profile_dataframe(clean)

    t0 = time.perf_counter()
    charts = generate_charts(clean)
    charts_ms = round((time.perf_counter() - t0) * 1000)

    insights = build_insights(profile_before, profile_after, log)

    response = {
        "job_id": "demo",
        "filename": path.name,
        "profile_before": profile_before,
        "plan": PLAN,
        "log": log,
        "profile_after": profile_after,
        "charts": charts,
        "insights": insights,
        "used_fallback": False,
        "timings": {
            "load_ms": load_ms,
            "profile_ms": profile_ms,
            "plan_ms": 1800,
            "execute_ms": execute_ms,
            "charts_ms": charts_ms,
            "insights_ms": 950,
            "total_ms": round((time.perf_counter() - started) * 1000) + 2750,
        },
        "warnings": [],
        "download_url": "/download/demo",
    }

    out_dir = ROOT / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sample_response.json"
    out_path.write_text(
        json.dumps(to_jsonable(response), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    reread = json.loads(out_path.read_text(encoding="utf-8"))
    expected = {
        "job_id", "filename", "profile_before", "plan", "log", "profile_after",
        "charts", "insights", "used_fallback", "timings", "warnings", "download_url",
    }
    print("keys ok:", set(reread.keys()) == expected)
    print("plan actions:", len({step["action"] for step in PLAN["steps"]}), "distinct")
    print("log statuses:", [entry["status"] for entry in reread["log"]])
    print("charts:", len(reread["charts"]))
    print("rows:", reread["profile_before"]["rows"], "->", reread["profile_after"]["rows"])
    print("nulls:", reread["profile_before"]["total_null_cells"], "->", reread["profile_after"]["total_null_cells"])
    print("saved:", out_path)


main()