import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.executor import ACTIONS, execute_plan
from backend.loader import load_dataframe
from backend.profiler import profile_dataframe

EXPECTED_ACTIONS = {
    "strip_whitespace",
    "standardize_text",
    "drop_duplicates",
    "convert_type",
    "remove_outliers",
    "fill_nulls",
    "drop_column",
    "drop_rows_where_null",
}


def run(df, step):
    clean, log = execute_plan(df, {"summary": "t", "steps": [step]})
    json.dumps(log, allow_nan=False)
    return clean, log[0]


def test_all_eight_actions_registered():
    assert EXPECTED_ACTIONS <= set(ACTIONS)


def test_strip_whitespace():
    df = pd.DataFrame({"a": ["  x ", "y   z", None]})
    clean, entry = run(df, {"action": "strip_whitespace", "columns": ["a"]})
    assert entry["status"] == "ok" and entry["affected"] == 2
    assert clean["a"].iloc[0] == "x" and clean["a"].iloc[1] == "y z"
    assert pd.isna(clean["a"].iloc[2])


def test_standardize_text():
    df = pd.DataFrame({"a": ["AbC", "dEf gh", None]})
    lower, entry = run(df, {"action": "standardize_text", "columns": ["a"], "case": "lower"})
    assert entry["status"] == "ok"
    assert lower["a"].iloc[0] == "abc" and pd.isna(lower["a"].iloc[2])
    upper, _ = run(df, {"action": "standardize_text", "columns": ["a"], "case": "upper"})
    assert upper["a"].iloc[1] == "DEF GH"
    title, _ = run(df, {"action": "standardize_text", "columns": ["a"], "case": "title"})
    assert title["a"].iloc[1] == "Def Gh"
    assert "nan" not in set(title["a"].dropna().astype(str).str.lower())


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "z"]})
    clean, entry = run(df, {"action": "drop_duplicates"})
    assert entry["affected"] == 1 and entry["rows_before"] == 4 and entry["rows_after"] == 3
    assert list(clean.index) == [0, 1, 2]


def test_convert_type_numeric():
    df = pd.DataFrame({"a": ["1", " 2.5 ", "x", None, "n/a"]})
    clean, entry = run(df, {"action": "convert_type", "column": "a", "to": "numeric"})
    assert entry["status"] == "ok" and entry["affected"] == 1
    assert pd.api.types.is_numeric_dtype(clean["a"])
    assert clean["a"].iloc[:2].tolist() == [1.0, 2.5]
    assert int(clean["a"].isna().sum()) == 3


def test_convert_type_datetime():
    df = pd.DataFrame({"d": ["2025-01-02", "03/01/2025", "25/12/2025", "bad", None]})
    clean, entry = run(df, {"action": "convert_type", "column": "d", "to": "datetime"})
    assert entry["status"] == "ok" and entry["affected"] == 1
    assert pd.api.types.is_datetime64_any_dtype(clean["d"])
    assert clean["d"].iloc[0] == pd.Timestamp("2025-01-02")
    assert clean["d"].iloc[1] == pd.Timestamp("2025-01-03")
    assert clean["d"].iloc[2] == pd.Timestamp("2025-12-25")


def test_convert_type_category_and_string():
    df = pd.DataFrame({"a": ["x", "y", "x"], "n": [1.0, 2.0, np.nan]})
    clean, entry = run(df, {"action": "convert_type", "column": "a", "to": "category"})
    assert entry["affected"] == 0 and str(clean["a"].dtype) == "category"
    clean, entry = run(df, {"action": "convert_type", "column": "n", "to": "string"})
    assert entry["status"] == "ok" and entry["affected"] == 0
    assert clean["n"].iloc[0] == "1" and pd.isna(clean["n"].iloc[2])


def test_convert_type_bad_target():
    _, entry = run(pd.DataFrame({"a": [1]}), {"action": "convert_type", "column": "a", "to": "xyz"})
    assert entry["status"] == "failed" and entry["error"]


OUTLIER_DF = pd.DataFrame({"v": [10.0, 11, 12, 13, 14, 15, 12, 11, 13, 500]})


def test_remove_outliers_iqr():
    clean, entry = run(OUTLIER_DF, {"action": "remove_outliers", "column": "v", "method": "iqr"})
    assert entry["status"] == "ok" and entry["affected"] == 1
    assert entry["rows_before"] == 10 and entry["rows_after"] == 9
    assert clean["v"].max() == 15


def test_remove_outliers_zscore():
    values = [10.0] * 30 + [11.0] * 30 + [1000.0]
    clean, entry = run(pd.DataFrame({"v": values}), {"action": "remove_outliers", "column": "v", "method": "zscore"})
    assert entry["status"] == "ok" and entry["affected"] == 1
    assert entry["rows_after"] == 60 and clean["v"].max() == 11.0


def test_remove_outliers_clip_iqr():
    clean, entry = run(OUTLIER_DF, {"action": "remove_outliers", "column": "v", "method": "clip_iqr"})
    assert entry["status"] == "ok" and entry["affected"] == 1
    assert "rows_before" not in entry
    assert len(clean) == 10 and clean["v"].max() < 500


def test_remove_outliers_non_numeric_fails():
    df = pd.DataFrame({"s": ["a", "b", "c", "d", "e"]})
    clean, entry = run(df, {"action": "remove_outliers", "column": "s", "method": "iqr"})
    assert entry["status"] == "failed" and entry["error"]
    assert clean.equals(df)


def test_fill_nulls_methods():
    df = pd.DataFrame({"n": [1.0, np.nan, 3.0, "?"], "c": ["a", "a", None, "b"]})
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    clean, entry = run(df, {"action": "fill_nulls", "column": "n", "method": "mean"})
    assert entry["fill_value"] == 2.0 and entry["affected"] == 2 and clean["n"].isna().sum() == 0
    clean, entry = run(df, {"action": "fill_nulls", "column": "n", "method": "median"})
    assert entry["fill_value"] == 2.0
    clean, entry = run(df, {"action": "fill_nulls", "column": "c", "method": "mode"})
    assert entry["fill_value"] == "a" and clean["c"].iloc[2] == "a"
    clean, entry = run(df, {"action": "fill_nulls", "column": "c", "method": "constant", "value": "none"})
    assert entry["fill_value"] == "none" and clean["c"].iloc[2] == "none"
    clean, entry = run(df, {"action": "fill_nulls", "column": "n", "method": "ffill"})
    assert entry["affected"] == 2 and clean["n"].iloc[1] == 1.0 and clean["n"].iloc[3] == 3.0
    clean, entry = run(df, {"action": "fill_nulls", "column": "n", "method": "drop_rows"})
    assert entry["rows_before"] == 4 and entry["rows_after"] == 2 and entry["affected"] == 2


def test_fill_nulls_pseudo_null_before_fill():
    df = pd.DataFrame({"c": ["a", "N/A", "-", "a", "b"]})
    clean, entry = run(df, {"action": "fill_nulls", "column": "c", "method": "mode"})
    assert entry["affected"] == 2 and entry["fill_value"] == "a"
    assert clean["c"].tolist() == ["a", "a", "a", "a", "b"]


def test_drop_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    clean, entry = run(df, {"action": "drop_column", "column": "B", "reason": "test"})
    assert entry["status"] == "ok" and list(clean.columns) == ["a"]


def test_drop_rows_where_null():
    df = pd.DataFrame({"a": [1, None, 3, 4], "b": ["x", "y", "unknown", "z"]})
    clean, entry = run(df, {"action": "drop_rows_where_null", "column": "b"})
    assert entry["affected"] == 1 and entry["rows_before"] == 4 and entry["rows_after"] == 3
    clean, entry = run(df, {"action": "drop_rows_where_null", "column": "a"})
    assert entry["affected"] == 1 and len(clean) == 3


def test_unknown_column_and_action_do_not_crash():
    df = pd.DataFrame({"a": [1, 2]})
    plan = {"steps": [
        {"action": "drop_column", "column": "zzz"},
        {"action": "fly"},
        {"action": "fill_nulls", "column": "a", "method": "median"},
    ]}
    clean, log = execute_plan(df, plan)
    assert [e["status"] for e in log] == ["failed", "failed", "ok"]
    assert clean.equals(df)


@pytest.mark.parametrize("plan", [
    None,
    {},
    {"steps": None},
    {"steps": "abc"},
    {"steps": [None, 5, "x", [], {}]},
    {"steps": [{"action": None}, {"action": 5}, {"action": "convert_type"}, {"action": "fill_nulls"}]},
    {"steps": [{"action": "remove_outliers", "column": ["a"], "method": []}]},
])
def test_bad_plans_never_crash(plan):
    df = pd.DataFrame({"a": [1, 2, 3]})
    clean, log = execute_plan(df, plan)
    json.dumps(log, allow_nan=False)
    assert clean.equals(df)
    assert all(e["status"] == "failed" for e in log)


def test_original_dataframe_not_modified():
    df = pd.DataFrame({"a": [" X ", " X ", None], "n": [1.0, 100.0, np.nan]})
    snapshot = df.copy(deep=True)
    plan = {"steps": [
        {"action": "strip_whitespace", "columns": ["a"]},
        {"action": "drop_duplicates"},
        {"action": "fill_nulls", "column": "n", "method": "mean"},
        {"action": "drop_column", "column": "a"},
    ]}
    execute_plan(df, plan)
    assert df.equals(snapshot)


ONE_ROW_STEPS = [
    {"action": "strip_whitespace", "columns": ["t"]},
    {"action": "standardize_text", "columns": ["t"], "case": "upper"},
    {"action": "drop_duplicates"},
    {"action": "convert_type", "column": "t", "to": "datetime"},
    {"action": "convert_type", "column": "n", "to": "numeric"},
    {"action": "remove_outliers", "column": "n", "method": "iqr"},
    {"action": "remove_outliers", "column": "n", "method": "zscore"},
    {"action": "remove_outliers", "column": "n", "method": "clip_iqr"},
    {"action": "fill_nulls", "column": "n", "method": "mean"},
    {"action": "fill_nulls", "column": "t", "method": "mode"},
    {"action": "drop_rows_where_null", "column": "t"},
    {"action": "drop_column", "column": "t"},
]


def test_single_row_dataframe():
    df = pd.DataFrame({"t": [" 2025-01-01 "], "n": [5.0]})
    clean, log = execute_plan(df, {"steps": ONE_ROW_STEPS})
    json.dumps(log, allow_nan=False)
    assert [e["status"] for e in log] == ["ok"] * len(ONE_ROW_STEPS)
    assert len(clean) == 1 and list(clean.columns) == ["n"]
    assert log[5]["affected"] == 0 and log[6]["affected"] == 0 and log[7]["affected"] == 0


def test_single_row_numeric_only_edge():
    df = pd.DataFrame({"n": [7.0]})
    for method in ["iqr", "zscore", "clip_iqr"]:
        clean, entry = run(df, {"action": "remove_outliers", "column": "n", "method": method})
        assert entry["status"] == "ok" and entry["affected"] == 0 and len(clean) == 1


def test_fully_empty_columns():
    df = pd.DataFrame({
        "obj": [None, "", "n/a", " ? "],
        "num": [np.nan, np.nan, np.nan, np.nan],
        "dt": [None, None, None, None],
        "ok": [1, 2, 3, 4],
    })
    steps = [
        {"action": "strip_whitespace", "columns": ["obj", "num"]},
        {"action": "standardize_text", "columns": ["obj"], "case": "lower"},
        {"action": "convert_type", "column": "obj", "to": "numeric"},
        {"action": "convert_type", "column": "dt", "to": "datetime"},
        {"action": "remove_outliers", "column": "num", "method": "iqr"},
        {"action": "remove_outliers", "column": "num", "method": "zscore"},
        {"action": "remove_outliers", "column": "num", "method": "clip_iqr"},
        {"action": "fill_nulls", "column": "num", "method": "ffill"},
    ]
    clean, log = execute_plan(df, {"steps": steps})
    json.dumps(log, allow_nan=False)
    assert all(e["status"] == "ok" for e in log), log
    assert len(clean) == 4

    for method in ["mean", "median", "mode"]:
        frame = pd.DataFrame({"num": [np.nan, np.nan]})
        _, entry = run(frame, {"action": "fill_nulls", "column": "num", "method": method})
        assert entry["status"] == "failed" and entry["affected"] == 0

    clean, entry = run(df, {"action": "drop_rows_where_null", "column": "obj"})
    assert entry["status"] == "ok" and len(clean) == 0 and entry["affected"] == 4


def test_empty_dataframe():
    clean, log = execute_plan(pd.DataFrame(), {"steps": [{"action": "drop_duplicates"}, {"action": "drop_column", "column": "a"}]})
    assert [e["status"] for e in log] == ["ok", "failed"]
    assert clean.empty


SALES = ROOT / "sample_data" / "dirty_sales.csv"


@pytest.mark.skipif(not SALES.exists(), reason="sample_data/dirty_sales.csv tapılmadı")
def test_full_plan_on_dirty_sales():
    df = load_dataframe(SALES.read_bytes(), SALES.name)
    snapshot = df.copy(deep=True)
    plan = {"summary": "full", "steps": [
        {"action": "strip_whitespace", "columns": ["City", "Notes"]},
        {"action": "standardize_text", "columns": ["City"], "case": "title"},
        {"action": "drop_duplicates"},
        {"action": "convert_type", "column": "Date", "to": "datetime"},
        {"action": "convert_type", "column": "Quantity", "to": "numeric"},
        {"action": "remove_outliers", "column": "Price", "method": "iqr"},
        {"action": "fill_nulls", "column": "Quantity", "method": "median"},
        {"action": "drop_column", "column": "Notes", "reason": "mostly empty"},
        {"action": "drop_rows_where_null", "column": "City"},
    ]}
    clean, log = execute_plan(df, plan)
    json.dumps(log, allow_nan=False)
    for entry in log:
        print(entry)

    assert df.equals(snapshot)
    assert [e["status"] for e in log] == ["ok"] * len(plan["steps"])
    assert {e["action"] for e in log} == EXPECTED_ACTIONS

    assert log[2]["rows_before"] == 1230 and log[2]["rows_after"] == 1200
    assert pd.api.types.is_datetime64_any_dtype(clean["Date"])
    assert clean["Date"].isna().sum() == 0
    assert clean["Date"].min() == pd.Timestamp("2025-01-01")
    assert clean["Date"].max() <= pd.Timestamp("2025-12-31")
    assert log[5]["affected"] > 0 and log[5]["rows_after"] == log[5]["rows_before"] - log[5]["affected"]
    assert clean["Quantity"].isna().sum() == 0
    assert "Notes" not in clean.columns
    assert clean["City"].isna().sum() == 0
    cities = set(clean["City"].unique())
    assert len(cities) == 3 and {"Baku", "Ganja"} <= cities
    assert len(clean) == log[-1]["rows_after"]

    profile = profile_dataframe(clean)
    json.dumps(profile, allow_nan=False)
    assert profile["rows"] == len(clean) and profile["total_null_cells"] == 0