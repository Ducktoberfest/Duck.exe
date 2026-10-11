import json
import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.executor import execute_plan
from backend.loader import load_dataframe
from backend.profiler import profile_dataframe
from backend.visualizer import generate_charts, null_comparison_chart

OUT_DIR = ROOT / "outputs" / "stage5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

path = ROOT / "sample_data" / "dirty_sales.csv"
df = load_dataframe(path.read_bytes(), path.name)
before = profile_dataframe(df)
plan = {
    "summary": "check",
    "steps": [
        {"action": "strip_whitespace"},
        {"action": "standardize_text", "columns": ["City"], "case": "title"},
        {"action": "drop_duplicates"},
        {"action": "convert_type", "column": "Date", "to": "datetime"},
        {"action": "convert_type", "column": "Price", "to": "numeric"},
        {"action": "convert_type", "column": "Quantity", "to": "numeric"},
        {"action": "fill_nulls", "column": "Price", "method": "median"},
        {"action": "fill_nulls", "column": "City", "method": "mode"},
    ],
}
clean, log = execute_plan(df, plan)
after = profile_dataframe(clean)
print("log statuses:", [e["status"] for e in log])

charts = generate_charts(clean)
extra = null_comparison_chart(before, after)
all_charts = charts + ([extra] if extra else [])
print("charts:", len(charts))

for index, chart in enumerate(all_charts, start=1):
    figure = go.Figure(json.loads(chart["figure_json"]))
    stem = f"chart_{index}_{chart['kind']}"
    (OUT_DIR / f"{stem}.json").write_text(chart["figure_json"], encoding="utf-8")
    figure.write_html(str(OUT_DIR / f"{stem}.html"), include_plotlyjs=True)
    print(chart["kind"], "|", chart["title"], "|", len(figure.data), "trace(s)")

(OUT_DIR / "charts_response.json").write_text(
    json.dumps(all_charts, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT_DIR / "profile_before.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT_DIR / "profile_after.json").write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT_DIR / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
clean.to_csv(OUT_DIR / "dirty_sales_clean.csv", index=False, encoding="utf-8-sig")

print("empty df:", generate_charts(df.iloc[0:0]))
print("max_charts=1:", len(generate_charts(clean, 1)))
print("saved to:", OUT_DIR)
for file in sorted(OUT_DIR.iterdir()):
    print(" -", file.name)