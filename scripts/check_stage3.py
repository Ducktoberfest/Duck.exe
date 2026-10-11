import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.executor import (
    execute_plan,
    normalize_pseudo_nulls,
    resolve_column,
)
from backend.loader import load_dataframe


def normalize_city_name(value):
    """Normalize Unicode representation and surrounding whitespace."""
    value = unicodedata.normalize("NFC", str(value))
    return " ".join(value.split())


def check_sales():
    data_path = ROOT / "sample_data" / "dirty_sales.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Sales dataset not found: {data_path}"
        )

    data_frame = load_dataframe(
        data_path.read_bytes(),
        data_path.name,
    )

    data_snapshot = data_frame.copy(deep=True)

    execution_plan = {
        "summary": "Stage 3 sales data cleaning test",
        "steps": [
            {
                "action": "strip_whitespace",
                "columns": ["City", "Notes"],
            },
            {
                "action": "standardize_text",
                "columns": ["City"],
                "case": "title",
            },
            {
                "action": "drop_duplicates",
            },
            {
                "action": "fill_nulls",
                "column": "Quantity",
                "method": "median",
            },
        ],
    }

    cleaned_df, execution_log = execute_plan(
        data_frame,
        execution_plan,
    )

    json.dumps(execution_log, allow_nan=False)

    for log_entry in execution_log:
        print(log_entry)

    assert data_frame.equals(data_snapshot), (
        "Original DataFrame was modified"
    )

    statuses = [
        entry["status"]
        for entry in execution_log
    ]

    assert statuses == ["ok"] * 4, (
        f"One or more cleaning steps failed: {execution_log}"
    )

    assert execution_log[2]["rows_before"] == 1230
    assert execution_log[2]["rows_after"] == len(cleaned_df)
    assert execution_log[2]["affected"] == 30

    assert execution_log[3]["affected"] == 173
    assert cleaned_df["Quantity"].isna().sum() == 0

    actual_values = cleaned_df["City"].dropna().unique()

    print(
        "\nActual city names:",
        [repr(value) for value in actual_values],
    )

    unique_cities = {
        normalize_city_name(value)
        for value in actual_values
    }

    # dirty_sales.csv uses Sumqayit, not Sumgayit.
    expected_cities = {
        "Baku",
        "Ganja",
        "Sumqayit",
    }

    print("\nExpected cities:", expected_cities)
    print("Actual cities:", unique_cities)

    assert unique_cities == expected_cities, (
        "City names do not match the expected values. "
        f"Expected: {expected_cities}; actual: {unique_cities}"
    )

    city_strings = (
        cleaned_df["City"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    assert "nan" not in set(city_strings)

    assert cleaned_df["City"].isna().sum() > 0

    print(
        "\nsales OK:",
        len(data_frame),
        "->",
        len(cleaned_df),
    )


def check_failures_and_edges():
    test_df = pd.DataFrame({
        "Name": [" Ali ", "VELI", None, "n/a", "Ali"],
        "Age": ["25", "unknown", "30", None, "25"],
        "Score": [1.0, np.nan, 3.0, 4.0, np.nan],
    })

    test_snapshot = test_df.copy(deep=True)

    execution_plan = {
        "steps": [
            {"action": "explode"},
            {
                "action": "strip_whitespace",
                "columns": ["Nope"],
            },
            {
                "action": "standardize_text",
                "columns": ["NAME"],
                "case": "upper",
            },
            {
                "action": "standardize_text",
                "columns": ["Name"],
                "case": "weird",
            },
            {
                "action": "fill_nulls",
                "column": "age",
                "method": "mean",
            },
            {
                "action": "fill_nulls",
                "column": "Name",
                "method": "mode",
            },
            {
                "action": "fill_nulls",
                "column": "Score",
                "method": "constant",
                "value": "7",
            },
            {
                "action": "fill_nulls",
                "column": "Score",
                "method": "constant",
            },
            {
                "action": "fill_nulls",
                "column": "Name",
                "method": "drop_rows",
            },
            {
                "action": "fill_nulls",
                "column": "Score",
                "method": "ffill",
            },
            "bad step",
        ],
    }

    cleaned_df, execution_log = execute_plan(
        test_df,
        execution_plan,
    )

    json.dumps(execution_log, allow_nan=False)

    for log_entry in execution_log:
        print(log_entry)

    assert test_df.equals(test_snapshot), (
        "Original edge-case DataFrame was modified"
    )

    step_statuses = [
        entry["status"]
        for entry in execution_log
    ]

    expected_statuses = [
        "failed",
        "failed",
        "ok",
        "failed",
        "ok",
        "ok",
        "ok",
        "failed",
        "ok",
        "ok",
        "failed",
    ]

    assert step_statuses == expected_statuses, (
        f"Unexpected step statuses: {step_statuses}"
    )

    assert execution_log[4]["fill_value"] == 26.6667
    assert execution_log[4]["affected"] == 2

    assert execution_log[5]["fill_value"] == " ALI "
    assert execution_log[5]["affected"] == 2

    names = (
        cleaned_df["Name"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    assert "nan" not in set(names)

    assert len(cleaned_df) == execution_log[8]["rows_after"]

    assert (
        execution_log[8]["rows_before"]
        - execution_log[8]["rows_after"]
        == execution_log[8]["affected"]
    )

    assert execute_plan(test_df, None)[1] == []

    empty_df, empty_log = execute_plan(
        None,
        {"steps": []},
    )

    assert empty_df.empty
    assert empty_log == []

    assert resolve_column(test_df, " score ") == "Score"

    try:
        resolve_column(test_df, "missing")
    except KeyError:
        pass
    else:
        raise AssertionError(
            "Expected KeyError for a missing column"
        )

    series_sample = normalize_pseudo_nulls(
        pd.Series(["a", " N/A ", "?", "-", "", "b"])
    )

    assert int(series_sample.isna().sum()) == 4

    print("edges OK")


if __name__ == "__main__":
    check_sales()
    check_failures_and_edges()
    print("\nALL CHECKS PASSED")
