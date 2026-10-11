import unicodedata
import numpy as np
import pandas as pd
import re
import warnings
from datetime import date, datetime
from backend.profiler import PSEUDO_NULLS, to_jsonable


CASES = {"lower", "upper", "title"}

FILL_METHODS = {
    "mean",
    "median",
    "mode",
    "constant",
    "ffill",
    "drop_rows",
}


def _is_plain(series):
    return (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
        or pd.api.types.is_timedelta64_dtype(series)
    )


def _restore_dtype(series, dtype):
    try:
        return series.astype(dtype)
    except (ValueError, TypeError):
        return series


def normalize_pseudo_nulls(series):
    if _is_plain(series):
        return series

    pseudo_nulls = {
        str(value).strip().casefold()
        for value in PSEUDO_NULLS
    }

    mask = np.fromiter(
        (
            isinstance(value, str)
            and value.strip().casefold() in pseudo_nulls
            for value in series
        ),
        dtype=bool,
        count=len(series),
    )

    if not mask.any():
        return series

    result = series.astype(object).where(~mask, np.nan)
    return _restore_dtype(result, series.dtype)


def resolve_column(df, name):
    if not isinstance(name, str):
        raise KeyError(f"Column not found: {name}")

    if name in df.columns:
        return name

    target = name.strip().casefold()

    for column in df.columns:
        if str(column).strip().casefold() == target:
            return column

    raise KeyError(f"Column not found: {name}")


def _target_columns(df, step):
    raw = step.get("columns", step.get("column"))

    if raw is None:
        names = [
            column
            for column in df.columns
            if not _is_plain(df[column])
        ]
    elif isinstance(raw, str):
        names = [raw]
    elif isinstance(raw, (list, tuple, set)):
        names = list(raw)
    else:
        raise ValueError(
            "'columns' must be a string or a list of column names"
        )

    resolved = [
        resolve_column(df, name)
        for name in names
    ]

    return list(dict.fromkeys(resolved))


def _map_strings(series, transform):
    output = []
    changed = 0

    for value in series:
        if isinstance(value, str):
            normalized = unicodedata.normalize("NFC", value)
            new_value = transform(normalized)

            if new_value != value:
                changed += 1

            output.append(new_value)
        else:
            output.append(value)

    result = pd.Series(
        output,
        index=series.index,
        name=series.name,
        dtype=object,
    )

    return _restore_dtype(result, series.dtype), changed


def _apply_text(df, step, transform):
    columns = _target_columns(df, step)
    affected = 0

    for column in columns:
        if _is_plain(df[column]):
            continue

        df[column], changed = _map_strings(
            df[column],
            transform,
        )
        affected += changed

    return df, {
        "column": ", ".join(str(c) for c in columns),
        "affected": affected,
    }


def _title_case(value):
    return " ".join(
        word[:1].upper() + word[1:].lower()
        for word in value.split()
    )


def _strip_whitespace(df, step):
    return _apply_text(
        df,
        step,
        lambda value: " ".join(value.split()),
    )


def _standardize_text(df, step):
    case = str(step.get("case", "lower")).strip().lower()

    if case not in CASES:
        raise ValueError(f"Invalid case value: {case}")

    transforms = {
        "lower": str.lower,
        "upper": str.upper,
        "title": _title_case,
    }

    df, info = _apply_text(
        df,
        step,
        transforms[case],
    )

    mapping = step.get("mapping", {})

    if mapping:
        if not isinstance(mapping, dict):
            raise ValueError("'mapping' must be a dictionary")

        columns = _target_columns(df, step)
        mapping_normalized = {
            unicodedata.normalize("NFC", str(k)).strip().casefold():
                v
            for k, v in mapping.items()
        }

        for column in columns:
            if _is_plain(df[column]):
                continue

            def replace_value(value):
                if not isinstance(value, str):
                    return value

                key = unicodedata.normalize(
                    "NFC", value
                ).strip().casefold()

                return mapping_normalized.get(key, value)

            df[column], changed = _map_strings(
                df[column],
                replace_value,
            )
            info["affected"] += changed

    return df, info


def _drop_duplicates(df, step):
    rows_before = len(df)

    try:
        result = df.drop_duplicates()
    except TypeError:
        result = df.loc[
            ~df.astype(str).duplicated().to_numpy()
        ]

    result = result.reset_index(drop=True)

    return result, {
        "affected": rows_before - len(result),
        "rows_before": rows_before,
        "rows_after": len(result),
    }


def _fill_value_for(method, series, null_mask, step):
    numeric = (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    )

    if method in {"mean", "median"}:
        if not numeric:
            converted = pd.to_numeric(
                series,
                errors="coerce",
            )

            invalid = converted.isna() & ~null_mask

            if invalid.any():
                raise ValueError(
                    f"{method} is only possible for numeric columns"
                )

            series = converted

        if series.notna().sum() == 0:
            raise ValueError("No values found to fill")

        value = (
            series.mean()
            if method == "mean"
            else series.median()
        )

        return series, round(float(value), 4)

    if method == "mode":
        modes = series.dropna().mode()

        if modes.empty:
            raise ValueError("Mode not found")

        return series, modes.iloc[0]

    if "value" not in step or step["value"] is None:
        raise ValueError(
            "Value is required for constant method"
        )

    value = step["value"]

    if numeric:
        try:
            value = pd.to_numeric(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Value must be numeric for numeric columns"
            ) from exc

    return series, value


def _fill_nulls(df, step):
    method = str(step.get("method", "")).strip().lower()

    if method not in FILL_METHODS:
        raise ValueError(f"Unknown method: {method}")

    column = resolve_column(df, step.get("column"))
    series = normalize_pseudo_nulls(df[column])

    if method in {"mean", "median"}:
        series = pd.to_numeric(series, errors="coerce")

    null_mask = series.isna().to_numpy(dtype=bool)
    nulls_before = int(null_mask.sum())

    info = {"column": str(column)}

    if method == "drop_rows":
        rows_before = len(df)
        df[column] = series

        result = df.loc[~null_mask].reset_index(drop=True)

        info.update(
            affected=rows_before - len(result),
            rows_before=rows_before,
            rows_after=len(result),
        )

        return result, info

    if method == "ffill":
        filled = series.ffill()
        df[column] = filled

        info["affected"] = (
            nulls_before - int(filled.isna().sum())
        )

        return df, info

    series, value = _fill_value_for(
        method,
        series,
        null_mask,
        step,
    )

    if (
        pd.api.types.is_numeric_dtype(series)
        and not pd.api.types.is_bool_dtype(series)
    ):
        filled = series.fillna(value)
    else:
        filled = series.astype(object).where(
            ~null_mask,
            value,
        )
        filled = _restore_dtype(filled, series.dtype)

    df[column] = filled

    info["affected"] = (
        nulls_before - int(filled.isna().sum())
    )
    info["fill_value"] = value

    return df, info


ACTIONS = {
    "strip_whitespace": _strip_whitespace,
    "standardize_text": _standardize_text,
    "drop_duplicates": _drop_duplicates,
    "fill_nulls": _fill_nulls,
}


def _error_text(exc):
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])

    return str(exc) or type(exc).__name__


def execute_plan(df, plan):
    if isinstance(df, pd.DataFrame):
        work = df.copy(deep=True)
    else:
        work = pd.DataFrame()

    steps = plan.get("steps") if isinstance(plan, dict) else None

    if not isinstance(steps, list):
        return work, []

    execution_log = []

    for number, step in enumerate(steps, start=1):
        entry = {
            "step": number,
            "action": "unknown",
            "affected": 0,
            "status": "failed",
        }

        try:
            if not isinstance(step, dict):
                raise ValueError(
                    "Step must be in dictionary format"
                )

            action = str(step.get("action", "")).strip()
            entry["action"] = action or "unknown"

            function = ACTIONS.get(action)

            if function is None:
                raise ValueError(
                    f"Unknown action: {entry['action']}"
                )

            updated_df, info = function(
                work.copy(deep=True),
                step,
            )

            entry.update(info)
            entry["status"] = "ok"
            work = updated_df

        except Exception as exc:
            entry["status"] = "failed"
            entry["affected"] = 0
            entry["error"] = _error_text(exc)

            if isinstance(step, dict):
                column = step.get("column")
                if isinstance(column, str):
                    entry["column"] = column

        execution_log.append(entry)

    return work, to_jsonable(execution_log)

ISO_DATE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")
CONVERT_TARGETS = {"numeric", "datetime", "category", "string"}
OUTLIER_METHODS = {"iqr", "zscore", "clip_iqr"}


def _is_missing(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, (float, np.floating)) and bool(np.isnan(value))


def _clean_strings(series):
    return series.astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)


def _datetime_source(series):
    def prepare(value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
            return value
        return np.nan

    return series.astype(object).map(prepare)


def _to_datetime(source, dayfirst):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(source, errors="coerce", format="mixed", dayfirst=dayfirst)
        except (TypeError, ValueError):
            return pd.to_datetime(source, errors="coerce", dayfirst=dayfirst)


def _parse_datetime(series, dayfirst):
    source = _datetime_source(series)
    iso = np.fromiter(
        (isinstance(v, str) and ISO_DATE.match(v) is not None for v in source),
        dtype=bool,
        count=len(source),
    )
    parsed = _to_datetime(source, dayfirst)
    if not iso.any():
        return parsed
    return parsed.where(~iso, _to_datetime(source, False))


def _best_datetime(series):
    day_first = _parse_datetime(series, True)
    month_first = _parse_datetime(series, False)
    if int(month_first.isna().sum()) < int(day_first.isna().sum()):
        return month_first
    return day_first


def _to_text(value):
    if _is_missing(value):
        return np.nan
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def _is_real_numeric(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _convert_type(df, step):
    target = str(step.get("to", "")).strip().lower()
    if target not in CONVERT_TARGETS:
        raise ValueError(f"Yanlış tip: {step.get('to')}")
    column = resolve_column(df, step.get("column"))
    series = normalize_pseudo_nulls(df[column])
    nulls_before = int(series.isna().sum())
    if target == "numeric":
        if _is_real_numeric(series):
            converted = series
        elif pd.api.types.is_bool_dtype(series):
            converted = series.astype("int64")
        else:
            converted = pd.to_numeric(_clean_strings(series), errors="coerce")
    elif target == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series):
            converted = series
        else:
            converted = _best_datetime(series)
    elif target == "category":
        converted = series.astype("category")
    else:
        converted = series.astype(object).map(_to_text)
    df[column] = converted
    affected = max(0, int(converted.isna().sum()) - nulls_before)
    return df, {"column": str(column), "affected": affected}


def _numeric_values(df, name):
    column = resolve_column(df, name)
    series = df[column]
    if not _is_real_numeric(series):
        raise ValueError(f"Sütun rəqəmli deyil: {column}")
    values = pd.Series(series.to_numpy(dtype="float64", na_value=np.nan), index=series.index)
    return column, values.replace([np.inf, -np.inf], np.nan)


def _iqr_bounds(values):
    present = values.dropna()
    if len(present) < 4:
        return None
    q1, q3 = present.quantile(0.25), present.quantile(0.75)
    spread = q3 - q1
    if not spread > 0:
        return None
    return q1 - 1.5 * spread, q3 + 1.5 * spread


def _remove_outliers(df, step):
    method = str(step.get("method", "")).strip().lower()
    if method not in OUTLIER_METHODS:
        raise ValueError(f"Bilinməyən method: {step.get('method')}")
    column, values = _numeric_values(df, step.get("column"))
    info = {"column": str(column)}

    if method == "zscore":
        present = values.dropna()
        std = float(present.std(ddof=0)) if len(present) > 1 else 0.0
        if std > 0:
            outside = ((values - float(present.mean())).abs() / std > 3).to_numpy(dtype=bool)
        else:
            outside = np.zeros(len(values), dtype=bool)
    else:
        bounds = _iqr_bounds(values)
        if bounds is None:
            outside = np.zeros(len(values), dtype=bool)
        else:
            outside = ((values < bounds[0]) | (values > bounds[1])).to_numpy(dtype=bool)

    if method == "clip_iqr":
        bounds = _iqr_bounds(values)
        count = int(outside.sum())
        if bounds is not None and count > 0:
            df[column] = values.clip(bounds[0], bounds[1])
        info["affected"] = count
        return df, info

    before = len(df)
    result = df.loc[~outside].reset_index(drop=True)
    info.update(affected=before - len(result), rows_before=before, rows_after=len(result))
    return result, info


def _drop_column(df, step):
    column = resolve_column(df, step.get("column"))
    return df.drop(columns=[column]), {"column": str(column), "affected": 1}


def _drop_rows_where_null(df, step):
    column = resolve_column(df, step.get("column"))
    series = normalize_pseudo_nulls(df[column])
    keep = ~series.isna().to_numpy(dtype=bool)
    before = len(df)
    result = df.loc[keep].reset_index(drop=True)
    return result, {
        "column": str(column),
        "affected": before - len(result),
        "rows_before": before,
        "rows_after": len(result),
    }


ACTIONS.update({
    "convert_type": _convert_type,
    "remove_outliers": _remove_outliers,
    "drop_column": _drop_column,
    "drop_rows_where_null": _drop_rows_where_null,
})

