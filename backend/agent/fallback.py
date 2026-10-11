import json

from backend.agent.validator import validate_plan

SUMMARY = "Qaydalı ehtiyat plan"
NULL_DROP_PCT = 60.0
TEXT_DTYPES = {"object", "str", "string"}


def _num(value, default=0.0):
    try:
        number = float(value)
        if number != number:
            return default
        return number
    except Exception:
        return default


def _null_pct(info, rows):
    if info.get("null_pct") is not None:
        return _num(info.get("null_pct"))
    if rows > 0:
        return _num(info.get("null_count")) / rows * 100
    return 0.0


def build_rule_based_plan(profile):
    empty = {"summary": SUMMARY, "steps": []}
    try:
        if not isinstance(profile, dict):
            return empty
        rows = _num(profile.get("rows"))
        infos = [
            i for i in (profile.get("columns_info") or [])
            if isinstance(i, dict) and i.get("name") is not None
        ]
        steps = []
        if _num(profile.get("duplicate_rows")) > 0:
            steps.append({"action": "drop_duplicates"})
        strip_cols = []
        title_cols = []
        for info in infos:
            name = str(info["name"])
            kind = info.get("inferred_type")
            if kind == "id":
                continue
            pct = _null_pct(info, rows)
            if pct > NULL_DROP_PCT:
                steps.append({
                    "action": "drop_column",
                    "column": name,
                    "reason": str(round(pct)) + "% boş",
                })
                continue
            if info.get("whitespace_issues"):
                strip_cols.append(name)
            if kind == "categorical":
                title_cols.append(name)
            text_dtype = str(info.get("dtype", "")).strip().lower() in TEXT_DTYPES
            if kind == "numeric" and text_dtype:
                steps.append({"action": "convert_type", "column": name, "to": "numeric"})
            elif kind == "datetime" and text_dtype:
                steps.append({"action": "convert_type", "column": name, "to": "datetime"})
            if kind == "numeric" and _num(info.get("outlier_count")) > 0:
                steps.append({"action": "remove_outliers", "column": name, "method": "clip_iqr"})
            if pct > 0 or _num(info.get("null_count")) > 0:
                if kind == "numeric":
                    method = "median"
                elif kind == "datetime":
                    method = "ffill"
                else:
                    method = "mode"
                steps.append({"action": "fill_nulls", "column": name, "method": method})
        if strip_cols:
            steps.append({"action": "strip_whitespace", "columns": strip_cols})
        if title_cols:
            steps.append({"action": "standardize_text", "columns": title_cols, "case": "title"})
        plan, _ = validate_plan({"summary": SUMMARY, "steps": steps}, profile)
        plan["summary"] = SUMMARY
        return plan
    except Exception:
        return empty


if __name__ == "__main__":
    test_profile = {
        "rows": 200,
        "columns": 6,
        "duplicate_rows": 3,
        "total_null_cells": 156,
        "columns_info": [
            {"name": "order_id", "dtype": "int64", "inferred_type": "id", "null_count": 0, "null_pct": 0.0, "unique": 200, "sample_values": ["1001", "1002"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
            {"name": "product", "dtype": "object", "inferred_type": "categorical", "null_count": 0, "null_pct": 0.0, "unique": 8, "sample_values": [" Laptop", "PHONE"], "stats": None, "outlier_count": 0, "whitespace_issues": True},
            {"name": "price", "dtype": "object", "inferred_type": "numeric", "null_count": 12, "null_pct": 6.0, "unique": 150, "sample_values": ["450", "99999"], "stats": None, "outlier_count": 4, "whitespace_issues": False},
            {"name": "quantity", "dtype": "int64", "inferred_type": "numeric", "null_count": 0, "null_pct": 0.0, "unique": 9, "sample_values": ["1", "2"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
            {"name": "notes", "dtype": "object", "inferred_type": "text", "null_count": 144, "null_pct": 72.0, "unique": 20, "sample_values": ["gift", "nan"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
            {"name": "order_date", "dtype": "object", "inferred_type": "datetime", "null_count": 0, "null_pct": 0.0, "unique": 90, "sample_values": ["2024-01-05", "2024-01-06"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
        ],
    }
    print(json.dumps(build_rule_based_plan(test_profile), indent=2, ensure_ascii=False))