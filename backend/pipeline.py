import io
import json
import math
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PREVIEW_ROWS = 50
STUBS = {}


def stub_load_dataframe(file_bytes, filename):
    extension = Path(filename).suffix.lower()
    if extension == ".csv":
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
            try:
                return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
            except Exception as exc:
                last_error = exc
        raise last_error
    return pd.read_excel(io.BytesIO(file_bytes))


def stub_infer_type(series, name, unique):
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "id" if str(name).lower().endswith("id") and unique == len(series.dropna()) and unique > 0 else "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna()
    if non_null.empty:
        return "text"
    text = non_null.astype(str).str.strip()
    if pd.to_numeric(text, errors="coerce").notna().mean() >= 0.8:
        return "numeric"
    if len(text) > 0 and text.nunique() / len(text) > 0.5:
        return "text"
    return "categorical"


def stub_profile_dataframe(df):
    rows = int(len(df))
    columns_info = []
    for position, name in enumerate(df.columns):
        series = df.iloc[:, position]
        null_count = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        inferred = stub_infer_type(series, name, unique)
        stats = None
        outliers = 0
        if inferred in ("numeric", "id"):
            nums = pd.to_numeric(series, errors="coerce").dropna()
            if not nums.empty:
                stats = {
                    "min": float(nums.min()),
                    "max": float(nums.max()),
                    "mean": float(nums.mean()),
                    "median": float(nums.median()),
                }
                q1, q3 = nums.quantile(0.25), nums.quantile(0.75)
                spread = q3 - q1
                outliers = int(((nums < q1 - 1.5 * spread) | (nums > q3 + 1.5 * spread)).sum())
        whitespace = False
        if series.dtype == object:
            text = series.dropna().astype(str)
            whitespace = bool((text != text.str.strip()).any())
        columns_info.append(
            {
                "name": str(name),
                "dtype": str(series.dtype),
                "inferred_type": inferred,
                "null_count": null_count,
                "null_pct": round(null_count / rows * 100, 2) if rows else 0.0,
                "unique": unique,
                "sample_values": [str(v) for v in series.dropna().unique()[:5]],
                "stats": stats,
                "outlier_count": outliers,
                "whitespace_issues": whitespace,
            }
        )
    return {
        "rows": rows,
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "total_null_cells": int(df.isna().sum().sum()),
        "columns_info": columns_info,
    }


def stub_build_plan(profile):
    info = profile.get("columns_info", [])
    steps = []
    whitespace_columns = [c["name"] for c in info if c.get("whitespace_issues")]
    if whitespace_columns:
        steps.append({"action": "strip_whitespace", "columns": whitespace_columns})
    if profile.get("duplicate_rows", 0) > 0:
        steps.append({"action": "drop_duplicates"})
    for c in info:
        if c.get("inferred_type") == "numeric" and c.get("dtype") == "object":
            steps.append({"action": "convert_type", "column": c["name"], "to": "numeric"})
    dropped = {c["name"] for c in info if c.get("null_pct", 0) > 60}
    for c in info:
        if c["name"] in dropped or c.get("null_count", 0) == 0:
            continue
        method = "median" if c.get("inferred_type") == "numeric" else "mode"
        steps.append({"action": "fill_nulls", "column": c["name"], "method": method})
    for name in sorted(dropped):
        steps.append({"action": "drop_column", "column": name, "reason": "mostly empty"})
    summary = f"Stub plan with {len(steps)} rule-based steps."
    return {"summary": summary, "steps": steps}, True


def stub_need_column(df, column):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")


def stub_apply_step(df, step):
    action = step.get("action")
    affected = 0
    fill_value = None

    if action == "strip_whitespace":
        for column in step.get("columns", []):
            stub_need_column(df, column)
            text = df[column].astype("string")
            stripped = text.str.strip()
            affected += int((text != stripped).fillna(False).sum())
            df[column] = df[column].where(df[column].isna(), stripped.astype(object))
    elif action == "standardize_text":
        case = step.get("case", "lower")
        for column in step.get("columns", []):
            stub_need_column(df, column)
            series = df[column]
            mask = series.notna()
            text = series.astype(str)
            if case == "upper":
                new = text.str.upper()
            elif case == "title":
                new = text.str.title()
            else:
                new = text.str.lower()
            affected += int(((text != new) & mask).sum())
            df[column] = new.where(mask, series)
    elif action == "drop_duplicates":
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        affected = before - len(df)
    elif action == "convert_type":
        column = step.get("column")
        stub_need_column(df, column)
        target = step.get("to")
        series = df[column]
        affected = int(series.notna().sum())
        if target == "numeric":
            df[column] = pd.to_numeric(series, errors="coerce")
        elif target == "datetime":
            df[column] = pd.to_datetime(series, errors="coerce")
        elif target == "category":
            df[column] = series.astype("category")
        elif target == "string":
            df[column] = series.astype("string")
        else:
            raise ValueError(f"Unknown target type '{target}'")
    elif action == "remove_outliers":
        column = step.get("column")
        stub_need_column(df, column)
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not numeric")
        nums = df[column]
        method = step.get("method", "iqr")
        if method == "zscore":
            std = nums.std()
            mask = (((nums - nums.mean()) / std).abs() > 3) if std and not math.isnan(std) else pd.Series(False, index=df.index)
            df = df[~mask.fillna(False)].reset_index(drop=True)
            affected = int(mask.fillna(False).sum())
        else:
            q1, q3 = nums.quantile(0.25), nums.quantile(0.75)
            spread = q3 - q1
            low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
            outside = ((nums < low) | (nums > high)).fillna(False)
            affected = int(outside.sum())
            if method == "clip_iqr":
                df[column] = nums.clip(low, high)
            elif method == "iqr":
                df = df[~outside].reset_index(drop=True)
            else:
                raise ValueError(f"Unknown method '{method}'")
    elif action == "fill_nulls":
        column = step.get("column")
        stub_need_column(df, column)
        method = step.get("method")
        series = df[column]
        affected = int(series.isna().sum())
        if method in ("mean", "median"):
            if not pd.api.types.is_numeric_dtype(series):
                raise ValueError(f"Column '{column}' is not numeric")
            fill_value = float(series.mean() if method == "mean" else series.median())
            df[column] = series.fillna(fill_value)
        elif method == "mode":
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError(f"Column '{column}' has no mode")
            fill_value = modes.iloc[0]
            df[column] = series.fillna(fill_value)
        elif method == "constant":
            fill_value = step.get("value")
            if fill_value is None:
                raise ValueError("Constant method needs a value")
            df[column] = series.fillna(fill_value)
        elif method == "ffill":
            df[column] = series.ffill()
        elif method == "drop_rows":
            df = df[series.notna()].reset_index(drop=True)
        else:
            raise ValueError(f"Unknown method '{method}'")
    elif action == "drop_column":
        column = step.get("column")
        stub_need_column(df, column)
        affected = int(df[column].notna().sum())
        df = df.drop(columns=[column])
    elif action == "drop_rows_where_null":
        column = step.get("column")
        stub_need_column(df, column)
        before = len(df)
        df = df[df[column].notna()].reset_index(drop=True)
        affected = before - len(df)
    else:
        raise ValueError(f"Unknown action '{action}'")
    return df, affected, fill_value


def stub_execute_plan(df, plan):
    working = df.copy()
    log = []
    steps = plan.get("steps") if isinstance(plan, dict) else []
    for index, step in enumerate(steps or [], start=1):
        step = step if isinstance(step, dict) else {}
        item = {"step": index, "action": str(step.get("action", ""))}
        column = step.get("column")
        if column is None and isinstance(step.get("columns"), list):
            column = ", ".join(str(c) for c in step["columns"])
        if column is not None:
            item["column"] = str(column)
        rows_before = len(working)
        try:
            working, affected, fill_value = stub_apply_step(working, step)
            item.update(affected=int(affected), status="ok", rows_before=rows_before, rows_after=len(working))
            if fill_value is not None:
                item["fill_value"] = fill_value
        except Exception as exc:
            item.update(affected=0, status="failed", rows_before=rows_before, rows_after=len(working), error=str(exc))
        log.append(item)
    return working, log


def stub_make_chart(title, kind, figure):
    figure.update_layout(title=title)
    return {"title": title, "kind": kind, "figure_json": json.loads(figure.to_json())}


def stub_generate_charts(df, max_charts=3):
    charts = []
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        figure = go.Figure(go.Bar(x=[str(c) for c in nulls.index], y=[int(v) for v in nulls.values]))
        charts.append(stub_make_chart("Null cells by column", "bar", figure))
    numeric_columns = list(df.select_dtypes(include="number").columns)
    if numeric_columns and len(charts) < max_charts:
        column = numeric_columns[0]
        figure = go.Figure(go.Histogram(x=df[column].dropna().tolist()))
        charts.append(stub_make_chart(f"Distribution of {column}", "histogram", figure))
    if len(charts) < max_charts:
        for column in df.select_dtypes(include=["object", "category"]).columns:
            if 2 <= df[column].nunique() <= 30:
                counts = df[column].astype(str).value_counts().head(10)
                figure = go.Figure(go.Bar(x=list(counts.index), y=[int(v) for v in counts.values]))
                charts.append(stub_make_chart(f"Top values of {column}", "bar", figure))
                break
    return charts[:max_charts]


def stub_write_insights(profile_before, profile_after, log, df_clean=None, lang="en"):
    rows_before = profile_before.get("rows", 0)
    rows_after = profile_after.get("rows", rows_before)
    nulls_before = profile_before.get("total_null_cells", 0)
    nulls_after = profile_after.get("total_null_cells", nulls_before)
    ok = sum(1 for item in log if item.get("status") == "ok")
    lines = [
        "### Stub summary",
        f"- Rows: {rows_before} → {rows_after}",
        f"- Null cells: {nulls_before} → {nulls_after}",
        f"- Successful steps: {ok}/{len(log)}",
    ]
    for item in log:
        if item.get("status") == "failed":
            lines.append(f"- Step {item.get('step')} ({item.get('action')}) failed: {item.get('error')}")
    return "\n".join(lines)


try:
    from backend.loader import load_dataframe
except Exception as exc:
    STUBS["loader.load_dataframe"] = str(exc)
    load_dataframe = stub_load_dataframe

try:
    from backend.profiler import profile_dataframe
except Exception as exc:
    STUBS["profiler.profile_dataframe"] = str(exc)
    profile_dataframe = stub_profile_dataframe

try:
    from backend.agent.planner import build_plan
except Exception as exc:
    STUBS["agent.planner.build_plan"] = str(exc)
    build_plan = stub_build_plan

try:
    from backend.executor import execute_plan
except Exception as exc:
    STUBS["executor.execute_plan"] = str(exc)
    execute_plan = stub_execute_plan

try:
    from backend.visualizer import generate_charts
except Exception as exc:
    STUBS["visualizer.generate_charts"] = str(exc)
    generate_charts = stub_generate_charts

try:
    from backend.agent.insights import write_insights
except Exception as exc:
    STUBS["agent.insights.write_insights"] = str(exc)
    write_insights = stub_write_insights


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
        text = df.head(PREVIEW_ROWS).to_json(orient="records", date_format="iso", default_handler=str)
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
        chart = clean_json(chart) if isinstance(chart.get("figure_json"), dict) else {**clean_json({k: v for k, v in chart.items() if k != "figure_json"}), "figure_json": chart["figure_json"]}
        chart.setdefault("title", f"Chart {index}")
        chart.setdefault("kind", "unknown")
        items.append(chart)
    return items


def run(file_bytes, filename, lang, job_dir, job_id):
    total_start = time.perf_counter()
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    warnings = [f"Temporary stub is used for {name}: {reason}" for name, reason in STUBS.items()]

    start = time.perf_counter()
    df = load_dataframe(file_bytes, filename)
    timings["load"] = elapsed(start)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("load_dataframe did not return a DataFrame")
    if df.shape[0] == 0 or df.shape[1] == 0:
        raise ValueError("The file contains no data.")

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
        used_fallback = False
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
        insights = write_insights(profile_before, profile_after, log, df_clean=df_clean, lang=lang)
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
