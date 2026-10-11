import json
import sys
import time
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.executor import execute_plan
from backend.loader import load_dataframe
from backend.profiler import profile_dataframe, to_jsonable
from backend.visualizer import generate_charts, null_comparison_chart

DATA_DIR = ROOT / "sample_data"
OUT_ROOT = ROOT / "outputs"
SUPPORTED = {".csv", ".xlsx", ".xls"}


def build_rule_plan(profile):
    steps = [{"action": "strip_whitespace"}]
    categorical = [c["name"] for c in profile["columns_info"] if c["inferred_type"] == "categorical"]
    if categorical:
        steps.append({"action": "standardize_text", "columns": categorical, "case": "title"})
    steps.append({"action": "drop_duplicates"})
    for info in profile["columns_info"]:
        name = info["name"]
        kind = info["inferred_type"]
        if info["null_pct"] > 50:
            steps.append({"action": "drop_column", "column": name, "reason": "More than 50 percent empty"})
        elif kind == "numeric":
            steps.append({"action": "convert_type", "column": name, "to": "numeric"})
            if info["outlier_count"] > 0:
                steps.append({"action": "remove_outliers", "column": name, "method": "clip_iqr"})
            if info["null_count"] > 0:
                steps.append({"action": "fill_nulls", "column": name, "method": "median"})
        elif kind == "datetime":
            steps.append({"action": "convert_type", "column": name, "to": "datetime"})
        elif kind == "categorical" and info["null_count"] > 0:
            steps.append({"action": "fill_nulls", "column": name, "method": "mode"})
    return {"summary": "Rule based cleaning plan", "steps": steps}


def save_json(path, data):
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def process(path):
    out_dir = OUT_ROOT / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    df = load_dataframe(path.read_bytes(), path.name)
    before = profile_dataframe(df)
    plan = build_rule_plan(before)
    clean, log = execute_plan(df, plan)
    after = profile_dataframe(clean)
    charts = generate_charts(clean)
    extra = null_comparison_chart(before, after)
    all_charts = charts + ([extra] if extra else [])

    for index, chart in enumerate(all_charts, start=1):
        figure = go.Figure(json.loads(chart["figure_json"]))
        stem = f"chart_{index}_{chart['kind']}"
        (out_dir / f"{stem}.json").write_text(chart["figure_json"], encoding="utf-8")
        figure.write_html(str(out_dir / f"{stem}.html"), include_plotlyjs="directory")

    save_json(out_dir / "plan.json", plan)
    save_json(out_dir / "log.json", log)
    save_json(out_dir / "profile_before.json", before)
    save_json(out_dir / "profile_after.json", after)
    save_json(out_dir / "charts_response.json", all_charts)
    clean.to_csv(out_dir / f"{path.stem}_clean.csv", index=False, encoding="utf-8-sig")

    failed = [entry for entry in log if entry["status"] == "failed"]
    elapsed = time.perf_counter() - started
    print(f"== {path.name}")
    print(f"   shape: {df.shape} -> {clean.shape}")
    print(f"   nulls: {before['total_null_cells']} -> {after['total_null_cells']}")
    print(f"   duplicates: {before['duplicate_rows']} -> {after['duplicate_rows']}")
    print(f"   steps: {len(log)} total, {len(failed)} failed")
    for entry in failed:
        print("   failed:", entry.get("action"), entry.get("column"), "|", entry.get("error"))
    print(f"   charts: {len(charts)} (+{1 if extra else 0} null comparison)")
    for chart in all_charts:
        print("     -", chart["kind"], "|", chart["title"])
    print(f"   time: {elapsed:.2f}s")
    print(f"   saved: {out_dir}")


def main():
    files = sorted(p for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        print("No datasets found in", DATA_DIR)
        return
    for path in files:
        try:
            process(path)
        except Exception as exc:
            print(f"== {path.name} CRASH: {type(exc).__name__}: {exc}")


main()