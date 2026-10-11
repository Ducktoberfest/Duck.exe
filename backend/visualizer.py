import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from backend.profiler import _parse_dates, profile_dataframe

PALETTE = ["#2563EB", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#06B6D4", "#F97316", "#84CC16"]
PREFERRED_NAMES = ["revenue", "sales", "amount", "total"]
MAX_CATEGORIES = 8
MAX_HIST_POINTS = 20000
MAX_MONTHS = 120
MAX_HEATMAP_COLUMNS = 10


def _column(df, name):
    data = df.loc[:, df.columns == name]
    return data.iloc[:, 0]


def _numeric(df, name):
    series = _column(df, name)
    if pd.api.types.is_bool_dtype(series):
        series = series.astype("int64")
    return pd.to_numeric(series, errors="coerce")


def _datetime(df, name):
    series = _column(df, name)
    if not pd.api.types.is_datetime64_any_dtype(series):
        series = _parse_dates(series.astype(str).str.strip())
    series = pd.Series(pd.to_datetime(series, errors="coerce"), index=series.index)
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series


def _names_of(profile, kind, df):
    present = set(map(str, df.columns))
    return [
        info["name"]
        for info in profile.get("columns_info", [])
        if info.get("inferred_type") == kind and info.get("name") in present
    ]


def _preferred_numeric(numeric_names):
    for keyword in PREFERRED_NAMES:
        for name in numeric_names:
            if keyword in str(name).lower():
                return name
    return numeric_names[0] if numeric_names else None


def _style(fig, title, x_title=None, y_title=None):
    fig.update_layout(
        template="plotly_white",
        height=400,
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        colorway=PALETTE,
        margin=dict(l=60, r=30, t=60, b=60),
    )
    return fig


def _pack(title, kind, fig):
    return {"title": title, "kind": kind, "figure_json": fig.to_json()}


def _trend_chart(df, profile):
    numeric_names = _names_of(profile, "numeric", df)
    datetime_names = _names_of(profile, "datetime", df)
    value_name = _preferred_numeric(numeric_names)
    if value_name is None or not datetime_names:
        return None
    frame = pd.DataFrame(
        {"date": _datetime(df, datetime_names[0]), "value": _numeric(df, value_name)}
    ).dropna()
    if frame.empty:
        return None
    monthly = frame.groupby(frame["date"].dt.to_period("M"))["value"].sum().sort_index()
    monthly = monthly.tail(MAX_MONTHS)
    if len(monthly) < 2:
        return None
    x_values = [period.to_timestamp().strftime("%Y-%m-%d") for period in monthly.index]
    y_values = [float(v) for v in monthly.to_numpy()]
    fig = go.Figure(
        go.Scatter(x=x_values, y=y_values, mode="lines+markers", line=dict(color=PALETTE[0], width=3), name=str(value_name))
    )
    title = f"Aylıq trend: {value_name}"
    _style(fig, title, "Ay", f"{value_name} (cəm)")
    return _pack(title, "line", fig)


def _category_chart(df, profile):
    numeric_names = _names_of(profile, "numeric", df)
    unique_by_name = {c["name"]: c["unique"] for c in profile["columns_info"]}
    category_names = sorted(
        (name for name in _names_of(profile, "categorical", df) if unique_by_name.get(name, 0) >= 2),
        key=lambda name: unique_by_name[name],
    )
    value_name = _preferred_numeric(numeric_names)
    if value_name is None or not category_names:
        return None
    category_name = category_names[0]
    frame = pd.DataFrame(
        {"category": _column(df, category_name).astype(str), "value": _numeric(df, value_name)}
    )
    frame = frame[_column(df, category_name).notna()].dropna()
    if frame.empty:
        return None
    totals = frame.groupby("category")["value"].sum().sort_values(ascending=False).head(MAX_CATEGORIES)
    fig = go.Figure(
        go.Bar(x=[str(v) for v in totals.index], y=[float(v) for v in totals.to_numpy()], marker_color=PALETTE[: len(totals)])
    )
    title = f"{category_name} üzrə müqayisə: {value_name}"
    _style(fig, title, str(category_name), f"{value_name} (cəm)")
    return _pack(title, "bar", fig)


def _variation(values):
    mean = float(values.mean())
    std = float(values.std())
    if not np.isfinite(std) or std <= 0:
        return 0.0
    return std / abs(mean) if mean != 0 else std


def _distribution_chart(df, profile):
    best_name, best_score, best_values = None, 0.0, None
    for name in _names_of(profile, "numeric", df):
        values = _numeric(df, name).dropna()
        if len(values) < 5:
            continue
        score = _variation(values)
        if score > best_score:
            best_name, best_score, best_values = name, score, values
    if best_name is None:
        return None
    if len(best_values) > MAX_HIST_POINTS:
        best_values = best_values.sample(MAX_HIST_POINTS, random_state=42)
    fig = go.Figure(go.Histogram(x=[float(v) for v in best_values.to_numpy()], nbinsx=30, marker_color=PALETTE[2]))
    title = f"Paylanma: {best_name}"
    _style(fig, title, str(best_name), "Say")
    return _pack(title, "histogram", fig)


def _correlation_chart(df, profile):
    numeric_names = _names_of(profile, "numeric", df)
    if len(numeric_names) <= 3:
        return None
    numeric_names = numeric_names[:MAX_HEATMAP_COLUMNS]
    frame = pd.DataFrame({name: _numeric(df, name).to_numpy() for name in numeric_names})
    corr = frame.corr().round(2)
    z_values = [[None if pd.isna(v) else float(v) for v in row] for row in corr.to_numpy()]
    labels = [str(name) for name in numeric_names]
    fig = go.Figure(go.Heatmap(z=z_values, x=labels, y=labels, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True))
    title = "Korrelyasiya xəritəsi"
    _style(fig, title)
    return _pack(title, "heatmap", fig)


def generate_charts(df, max_charts=3):
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        limit = int(max_charts)
        if limit <= 0:
            return []
        profile = profile_dataframe(df)
    except Exception:
        return []
    charts = []
    for builder in (_trend_chart, _category_chart, _distribution_chart, _correlation_chart):
        if len(charts) >= limit:
            break
        try:
            chart = builder(df, profile)
        except Exception:
            chart = None
        if chart is not None:
            charts.append(chart)
    return charts


def null_comparison_chart(profile_before, profile_after):
    try:
        before = {c["name"]: float(c["null_pct"]) for c in profile_before.get("columns_info", [])}
        after = {c["name"]: float(c["null_pct"]) for c in profile_after.get("columns_info", [])}
        names = list(before.keys())
        if not names:
            return None
        fig = go.Figure(
            [
                go.Bar(name="Əvvəl", x=names, y=[before[n] for n in names], marker_color=PALETTE[3]),
                go.Bar(name="Sonra", x=names, y=[after.get(n, 0.0) for n in names], marker_color=PALETTE[2]),
            ]
        )
        fig.update_layout(barmode="group")
        title = "Boşluq faizi: təmizləmədən əvvəl və sonra"
        _style(fig, title, "Sütun", "Boşluq (%)")
        return _pack(title, "grouped_bar", fig)
    except Exception:
        return None