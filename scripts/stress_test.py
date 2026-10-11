import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.executor import execute_plan
from backend.profiler import profile_dataframe
from backend.visualizer import generate_charts


def make_single_row():
    return pd.DataFrame({"Name": ["Alice"], "Amount": [10.5], "Date": ["2025-01-01"]})


def make_single_column():
    return pd.DataFrame({"Amount": [1.0, 2.0, None, 4.0, 1000.0, 3.0, 2.5]})


def make_all_null():
    return pd.DataFrame(
        {
            "A": [None] * 10,
            "B": [np.nan] * 10,
            "C": ["", "N/A", "null", " ", "-", "none", "?", "unknown", "NA", ""],
        }
    )


def make_all_text():
    return pd.DataFrame(
        {
            "Name": ["  Alice ", "BOB", "carol", "Dave  ", "eve", "Frank", "  GRACE", "heidi"],
            "City": ["Baku", "baku", " Ganja", "SUMQAYIT", "Baku", "ganja", "Baku", "Sumqayit"],
            "Comment": ["good", "bad", "ok", "ok", "good", "bad", "ok", "good"],
        }
    )


def make_large():
    rng = np.random.default_rng(42)
    size = 60000
    return pd.DataFrame(
        {
            "OrderID": np.arange(size),
            "Date": pd.date_range("2020-01-01", periods=size, freq="h").astype(str),
            "Category": rng.choice(["A", "B", "C", "D", "E"], size),
            "Region": rng.choice([" North", "south ", "EAST", "West"], size),
            "Revenue": np.where(rng.random(size) < 0.05, np.nan, rng.lognormal(3, 1, size)),
            "Quantity": rng.integers(1, 20, size),
            "Discount": rng.random(size),
            "Cost": rng.normal(50, 10, size),
        }
    )


def build_plan(df, profile):
    steps = [
        {"action": "strip_whitespace"},
        {"action": "standardize_text", "case": "lower"},
        {"action": "drop_duplicates"},
    ]
    for info in profile["columns_info"]:
        name = info["name"]
        if info["inferred_type"] == "numeric":
            steps.append({"action": "convert_type", "column": name, "to": "numeric"})
            steps.append({"action": "remove_outliers", "column": name, "method": "iqr"})
            steps.append({"action": "fill_nulls", "column": name, "method": "median"})
        elif info["inferred_type"] == "datetime":
            steps.append({"action": "convert_type", "column": name, "to": "datetime"})
        elif info["null_pct"] >= 100:
            steps.append({"action": "drop_column", "column": name})
        elif info["null_count"] > 0:
            steps.append({"action": "fill_nulls", "column": name, "method": "mode"})
    steps.append({"action": "fill_nulls", "column": "does_not_exist", "method": "mean"})
    return {"summary": "stress", "steps": steps}


CASES = [
    ("single_row", make_single_row),
    ("single_column", make_single_column),
    ("all_null_columns", make_all_null),
    ("all_text_columns", make_all_text),
    ("large_60k_rows", make_large),
]


def timed(label, function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def run_case(name, factory):
    df = factory()
    profile_before, t_profile = timed("profile", lambda: profile_dataframe(df))
    json.dumps(profile_before, allow_nan=False)
    plan = build_plan(df, profile_before)
    (clean, log), t_execute = timed("execute", lambda: execute_plan(df, plan))
    json.dumps(log, allow_nan=False)
    profile_after, t_profile_after = timed("profile_after", lambda: profile_dataframe(clean))
    json.dumps(profile_after, allow_nan=False)
    charts, t_charts = timed("charts", lambda: generate_charts(clean))
    json.dumps(charts, allow_nan=False)
    failed = [entry for entry in log if entry["status"] == "failed"]
    print(
        f"{name:18s} OK | shape {df.shape} -> {clean.shape} | charts {len(charts)} | "
        f"failed steps {len(failed)} | profile {t_profile:.2f}s execute {t_execute:.2f}s "
        f"profile_after {t_profile_after:.2f}s charts {t_charts:.2f}s"
    )
    for entry in failed:
        print("    failed:", entry.get("action"), entry.get("column"), "|", entry.get("error"))
    return True


def main():
    crashed = []
    for name, factory in CASES:
        try:
            run_case(name, factory)
        except Exception:
            crashed.append(name)
            print(f"{name:18s} CRASH")
            traceback.print_exc()
    print("crashed cases:", crashed if crashed else "none")


main()