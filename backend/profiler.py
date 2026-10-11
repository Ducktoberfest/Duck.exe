import math
import re
import warnings
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

PSEUDO_NULLS = {"", "n/a", "na", "null", "none", "unknown", "-", "?"}
BOOL_TOKENS = {
    "true", "false", "yes", "no", "y", "n", "t", "f",
    "0", "1", "0.0", "1.0", "bəli", "xeyr", "hə", "yox", "da", "net",
}
WHITESPACE_PATTERN = re.compile(r"^\s|\s$|\s{2,}")
SEED = 42


def to_jsonable(obj):
    if obj is None or obj is pd.NaT or obj is pd.NA:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating, Decimal)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, (bytes, bytearray)):
        return bytes(obj).decode("utf-8", errors="replace")
    if isinstance(obj, (pd.Timestamp, datetime, date, np.datetime64)):
        try:
            stamp = pd.Timestamp(obj)
            return None if pd.isna(stamp) else stamp.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (pd.Timedelta, np.timedelta64)):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in obj]
    if isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
        return to_jsonable(obj.tolist())
    if isinstance(obj, pd.DataFrame):
        return to_jsonable(obj.to_dict("records"))
    return str(obj)


def _is_plain_dtype(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    )


def _null_mask(series):
    mask = series.isna().to_numpy(dtype=bool)
    if _is_plain_dtype(series):
        return mask
    pseudo = np.fromiter(
        (isinstance(v, str) and v.strip().lower() in PSEUDO_NULLS for v in series),
        dtype=bool,
        count=len(series),
    )
    return mask | pseudo


def _nunique(valid):
    try:
        return int(valid.nunique())
    except Exception:
        return int(valid.astype(str).nunique())


def _sample_values(valid):
    if len(valid) == 0:
        return []
    try:
        unique_values = valid.drop_duplicates()
        picked = unique_values.sample(min(5, len(unique_values)), random_state=SEED)
        return [str(v) for v in picked.tolist()]
    except Exception:
        return [str(v) for v in valid.head(5).tolist()]


def _whitespace_issues(series):
    if _is_plain_dtype(series):
        return False
    return any(isinstance(v, str) and WHITESPACE_PATTERN.search(v) for v in series)


def _numeric_values(valid):
    if pd.api.types.is_numeric_dtype(valid) and not pd.api.types.is_bool_dtype(valid):
        converted = pd.to_numeric(valid, errors="coerce")
    else:
        converted = pd.to_numeric(valid.astype(str).str.strip(), errors="coerce")
    array = converted.to_numpy(dtype="float64", na_value=np.nan)
    return array[np.isfinite(array)]


def _parse_dates(values):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(values, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(values, errors="coerce")


def _looks_like_datetime(valid):
    sample = valid.sample(min(len(valid), 500), random_state=SEED).astype(str).str.strip()
    candidates = sample[(sample.str.len() >= 6) & sample.str.contains(r"\d", regex=True)]
    if len(candidates) / len(sample) < 0.7:
        return False
    parsed = _parse_dates(candidates)
    return parsed.notna().sum() / len(sample) >= 0.7


def _id_like_name(name):
    name = str(name).strip()
    return bool(
        re.search(r"(?i)(^|[_\s\-.])id$", name)
        or re.search(r"[a-z0-9]ID$", name)
        or re.search(r"[a-z]Id$", name)
    )


def _boolean_like(series, valid, unique):
    if pd.api.types.is_bool_dtype(series):
        return True
    if unique > 6:
        return False
    normalized = set(valid.astype(str).str.strip().str.lower().unique())
    return len(normalized) == 2 and normalized <= BOOL_TOKENS


def _infer_type(series, valid, name, rows, unique):
    if len(valid) == 0:
        return "text", None
    if _boolean_like(series, valid, unique):
        return "boolean", None
    if _id_like_name(name) and unique >= 0.9 * rows:
        return "id", None
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", _numeric_values(valid)
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime", None
    numbers = _numeric_values(valid)
    if len(numbers) / len(valid) > 0.8:
        return "numeric", numbers
    if _looks_like_datetime(valid):
        return "datetime", None
    if unique < 20 or unique < 0.05 * rows:
        return "categorical", None
    return "text", None


def _outlier_count(numbers):
    if numbers is None or len(numbers) < 4:
        return 0
    q1, q3 = np.percentile(numbers, [25, 75])
    spread = q3 - q1
    if spread <= 0:
        return 0
    return int(((numbers < q1 - 1.5 * spread) | (numbers > q3 + 1.5 * spread)).sum())


def _numeric_stats(numbers):
    if numbers is None or len(numbers) == 0:
        return None
    return {
        "min": round(float(numbers.min()), 4),
        "max": round(float(numbers.max()), 4),
        "mean": round(float(numbers.mean()), 4),
        "median": round(float(np.median(numbers)), 4),
    }


def _minimal_column(name, dtype, rows, null_count):
    return {
        "name": str(name),
        "dtype": str(dtype),
        "inferred_type": "text",
        "null_count": int(null_count),
        "null_pct": round(null_count / rows * 100, 2) if rows else 0.0,
        "unique": 0,
        "sample_values": [],
        "stats": None,
        "outlier_count": 0,
        "whitespace_issues": False,
    }


def _profile_column(series, name, rows):
    null_mask = _null_mask(series)
    null_count = int(null_mask.sum())
    valid = series[~null_mask]
    unique = _nunique(valid)
    inferred, numbers = _infer_type(series, valid, name, rows, unique)
    return {
        "name": str(name),
        "dtype": str(series.dtype),
        "inferred_type": inferred,
        "null_count": null_count,
        "null_pct": round(null_count / rows * 100, 2) if rows else 0.0,
        "unique": unique,
        "sample_values": _sample_values(valid),
        "stats": _numeric_stats(numbers) if inferred == "numeric" else None,
        "outlier_count": _outlier_count(numbers) if inferred == "numeric" else 0,
        "whitespace_issues": _whitespace_issues(series),
    }


def _duplicate_rows(df):
    try:
        return int(df.duplicated().sum())
    except Exception:
        try:
            return int(df.astype(str).duplicated().sum())
        except Exception:
            return 0


def _build_profile(df):
    rows = int(len(df))
    columns_info = []
    for position in range(df.shape[1]):
        name = df.columns[position]
        series = df.iloc[:, position]
        try:
            columns_info.append(_profile_column(series, name, rows))
        except Exception:
            try:
                nulls = int(series.isna().sum())
            except Exception:
                nulls = 0
            columns_info.append(_minimal_column(name, getattr(series, "dtype", "object"), rows, nulls))
    return {
        "rows": rows,
        "columns": int(df.shape[1]),
        "duplicate_rows": _duplicate_rows(df),
        "total_null_cells": int(sum(c["null_count"] for c in columns_info)),
        "columns_info": columns_info,
    }


def _empty_profile(df):
    try:
        rows, columns = int(df.shape[0]), int(df.shape[1])
    except Exception:
        rows, columns = 0, 0
    return {
        "rows": rows,
        "columns": columns,
        "duplicate_rows": 0,
        "total_null_cells": 0,
        "columns_info": [],
    }


def profile_dataframe(df):
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()
    try:
        return to_jsonable(_build_profile(df))
    except Exception:
        return to_jsonable(_empty_profile(df))