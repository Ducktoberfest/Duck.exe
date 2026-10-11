import json
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="CleanVision", page_icon="🦆", layout="wide")

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "fixtures" / "sample_response.json"
SAMPLE_DIR = ROOT / "sample_data"

TEXTS = {
    "en": {
        "lang_label": "Language / Dil",
        "demo_label": "Select demo dataset",
        "demo_none": "— none —",
        "mock_label": "Mock mode (no backend)",
        "backend_label": "Backend address",
        "subtitle": "Upload a messy CSV or Excel file and CleanVision will find the problems, clean the data and explain the results in plain language.",
        "upload_label": "Upload CSV or Excel file",
        "clean_button": "Clean Data",
        "no_input": "Upload a file or select a demo dataset to begin.",
        "stages": [
            "Reading data",
            "Searching for issues",
            "AI building plan",
            "Executing cleaning",
            "Generating charts",
        ],
        "sending": "Sending file to backend",
        "processing": "Processing...",
        "done": "Done",
        "failed": "Failed",
        "read_error": "Could not read the selected file",
        "backend_error": "Backend request failed: ",
        "fixture_missing": "Fixture file not found or invalid, built-in sample is used.",
        "download_failed": "Cleaned file could not be downloaded from the backend.",
        "fallback": "Fallback mode activated: the AI plan was unavailable, so a rule-based plan was used.",
        "download": "Download cleaned CSV",
        "warnings": "Warnings",
        "tab_summary": "Summary",
        "tab_plan": "Plan and log",
        "tab_charts": "Charts",
        "tab_data": "Data",
        "m_rows": "Rows",
        "m_deleted": "Deleted rows",
        "m_nulls": "Null cells",
        "m_steps": "Successful steps",
        "no_insights": "No insights available.",
        "plan_summary": "Plan summary",
        "plan_steps": "Steps",
        "no_steps": "No steps in the plan.",
        "log_title": "Execution log",
        "no_log": "No log entries.",
        "col_step": "Step",
        "col_action": "Action",
        "col_column": "Column",
        "col_affected": "Affected",
        "col_status": "Status",
        "col_rows": "Rows",
        "col_fill": "Fill value",
        "col_error": "Error",
        "no_charts": "No charts available.",
        "chart_error": "Chart could not be displayed: ",
        "data_title": "Profile overview",
        "data_note": "Cleaned data preview will be added in real mode.",
        "d_metric": "Metric",
        "d_before": "Before",
        "d_after": "After",
        "d_rows": "Rows",
        "d_columns": "Columns",
        "d_duplicates": "Duplicate rows",
        "d_nulls": "Null cells",
        "job": "Job",
        "file": "File",
    },
    "az": {
        "lang_label": "Language / Dil",
        "demo_label": "Demo verilənlər dəstini seçin",
        "demo_none": "— yoxdur —",
        "mock_label": "Mock rejimi (backend olmadan)",
        "backend_label": "Backend ünvanı",
        "subtitle": "Səliqəsiz CSV və ya Excel faylını yükləyin, CleanVision problemləri tapsın, məlumatı təmizləsin və nəticəni sadə dillə izah etsin.",
        "upload_label": "CSV və ya Excel faylı yükləyin",
        "clean_button": "Məlumatı təmizlə",
        "no_input": "Başlamaq üçün fayl yükləyin və ya demo dəstini seçin.",
        "stages": [
            "Məlumat oxunur",
            "Problemlər axtarılır",
            "Süni intellekt plan qurur",
            "Təmizləmə icra olunur",
            "Qrafiklər yaradılır",
        ],
        "sending": "Fayl backend-ə göndərilir",
        "processing": "İşlənir...",
        "done": "Hazırdır",
        "failed": "Uğursuz oldu",
        "read_error": "Seçilmiş fayl oxunmadı",
        "backend_error": "Backend sorğusu uğursuz oldu: ",
        "fixture_missing": "Fixture faylı tapılmadı və ya yararsızdır, daxili nümunə istifadə olunur.",
        "download_failed": "Təmizlənmiş fayl backend-dən yüklənmədi.",
        "fallback": "Ehtiyat rejimi aktivdir: süni intellekt planı mövcud olmadı, qaydalara əsaslanan plan istifadə edildi.",
        "download": "Təmizlənmiş CSV-ni yüklə",
        "warnings": "Xəbərdarlıqlar",
        "tab_summary": "Xülasə",
        "tab_plan": "Plan və jurnal",
        "tab_charts": "Qrafiklər",
        "tab_data": "Məlumat",
        "m_rows": "Sətir sayı",
        "m_deleted": "Silinən sətirlər",
        "m_nulls": "Boş xanalar",
        "m_steps": "Uğurlu addımlar",
        "no_insights": "Şərh mövcud deyil.",
        "plan_summary": "Planın xülasəsi",
        "plan_steps": "Addımlar",
        "no_steps": "Planda addım yoxdur.",
        "log_title": "İcra jurnalı",
        "no_log": "Jurnalda qeyd yoxdur.",
        "col_step": "Addım",
        "col_action": "Əməliyyat",
        "col_column": "Sütun",
        "col_affected": "Təsirlənən",
        "col_status": "Status",
        "col_rows": "Sətirlər",
        "col_fill": "Doldurma dəyəri",
        "col_error": "Xəta",
        "no_charts": "Qrafik yoxdur.",
        "chart_error": "Qrafik göstərilə bilmədi: ",
        "data_title": "Profil icmalı",
        "data_note": "Təmizlənmiş məlumatın önizləməsi real rejimdə əlavə olunacaq.",
        "d_metric": "Göstərici",
        "d_before": "Əvvəl",
        "d_after": "Sonra",
        "d_rows": "Sətirlər",
        "d_columns": "Sütunlar",
        "d_duplicates": "Təkrar sətirlər",
        "d_nulls": "Boş xanalar",
        "job": "İş",
        "file": "Fayl",
    },
     EXTRA_TEXTS = {
    "en": {
        "err_connection": "Server is not running. Start the backend.",
        "err_400": "File format not supported.",
        "err_413": "File is too large (20 MB limit).",
        "err_500": "An error occurred during cleaning",
        "err_timeout": "The server took too long to respond.",
        "err_other": "Unexpected error: ",
        "err_details": "Error details",
        "total_time": "Total time: {seconds} sec",
        "health_ok": "Backend is running",
        "health_down": "Backend is stopped",
        "llm_on": "LLM available",
        "llm_off": "LLM unavailable",
        "refresh": "Refresh status",
        "steps_title": "How it works",
        "step_1": "Upload a CSV or Excel file, or pick a demo dataset in the sidebar.",
        "step_2": "Click **Clean Data** and wait for the stages to finish.",
        "step_3": "Review the tabs and download the cleaned CSV.",
        "data_preview": "Data preview (first 50 rows)",
        "data_before": "Before",
        "data_after": "After",
        "no_preview": "Previews are available in real mode.",
    },
    "az": {
        "err_connection": "Server işləmir. Backend-i işə salın.",
        "err_400": "Fayl formatı dəstəklənmir.",
        "err_413": "Fayl çox böyükdür (limit 20 MB).",
        "err_500": "Təmizləmə zamanı xəta baş verdi",
        "err_timeout": "Server cavab verməkdə çox gecikdi.",
        "err_other": "Gözlənilməz xəta: ",
        "err_details": "Xəta təfərrüatları",
        "total_time": "Ümumi vaxt: {seconds} san",
        "health_ok": "Backend işləyir",
        "health_down": "Backend dayanıb",
        "llm_on": "LLM mövcuddur",
        "llm_off": "LLM mövcud deyil",
        "refresh": "Statusu yenilə",
        "steps_title": "Necə işləyir",
        "step_1": "CSV və ya Excel faylı yükləyin, ya da yan paneldən demo dəstini seçin.",
        "step_2": "**Məlumatı təmizlə** düyməsini basın və mərhələlərin bitməsini gözləyin.",
        "step_3": "Tabları nəzərdən keçirin və təmizlənmiş CSV-ni yükləyin.",
        "data_preview": "Məlumat önizləməsi (ilk 50 sətir)",
        "data_before": "Əvvəl",
        "data_after": "Sonra",
        "no_preview": "Önizləmələr real rejimdə mövcuddur.",
    },
}

for _code in TEXTS:
    TEXTS[_code].update(EXTRA_TEXTS[_code])
}


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def show_dataframe(df):
    try:
        st.dataframe(df, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_plotly(fig, key):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def list_demo_files():
    try:
        return sorted(
            p.name
            for p in SAMPLE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in (".csv", ".xlsx")
        )
    except OSError:
        return []


def build_internal_sample():
    columns = ["name", "city", "price"]
    nulls_before = [0, 4, 14]
    nulls_after = [0, 0, 0]
    fig = go.Figure(
        [
            go.Bar(name="Before", x=columns, y=nulls_before),
            go.Bar(name="After", x=columns, y=nulls_after),
        ]
    )
    fig.update_layout(title="Null cells by column", barmode="group")
    return {
        "job_id": "mock-job",
        "filename": "sample.csv",
        "profile_before": {
            "rows": 120,
            "columns": 4,
            "duplicate_rows": 6,
            "total_null_cells": 18,
            "columns_info": [
                {
                    "name": "price",
                    "dtype": "object",
                    "inferred_type": "numeric",
                    "null_count": 14,
                    "null_pct": 11.67,
                    "unique": 80,
                    "sample_values": ["10.5", "42", "n/a"],
                    "stats": None,
                    "outlier_count": 2,
                    "whitespace_issues": False,
                },
                {
                    "name": "city",
                    "dtype": "object",
                    "inferred_type": "categorical",
                    "null_count": 4,
                    "null_pct": 3.33,
                    "unique": 6,
                    "sample_values": [" Baku", "baku ", "Ganja"],
                    "stats": None,
                    "outlier_count": 0,
                    "whitespace_issues": True,
                },
            ],
        },
        "plan": {
            "summary": "Trim whitespace, remove duplicates, fix the price column and fill missing values.",
            "steps": [
                {"action": "strip_whitespace", "columns": ["name", "city"]},
                {"action": "drop_duplicates"},
                {"action": "convert_type", "column": "price", "to": "numeric"},
                {"action": "remove_outliers", "column": "price", "method": "iqr"},
                {"action": "fill_nulls", "column": "price", "method": "median"},
                {"action": "fill_nulls", "column": "city", "method": "mode"},
                {"action": "drop_column", "column": "discount", "reason": "mostly empty"},
            ],
        },
        "log": [
            {"step": 1, "action": "strip_whitespace", "column": "name, city", "affected": 14, "status": "ok"},
            {"step": 2, "action": "drop_duplicates", "affected": 6, "status": "ok", "rows_before": 120, "rows_after": 114},
            {"step": 3, "action": "convert_type", "column": "price", "affected": 114, "status": "ok"},
            {"step": 4, "action": "remove_outliers", "column": "price", "affected": 2, "status": "ok", "rows_before": 114, "rows_after": 112},
            {"step": 5, "action": "fill_nulls", "column": "price", "affected": 14, "status": "ok", "fill_value": 42.5},
            {"step": 6, "action": "fill_nulls", "column": "city", "affected": 4, "status": "ok", "fill_value": "Baku"},
            {"step": 7, "action": "drop_column", "column": "discount", "affected": 0, "status": "failed", "error": "Column 'discount' not found"},
        ],
        "profile_after": {
            "rows": 112,
            "columns": 4,
            "duplicate_rows": 0,
            "total_null_cells": 0,
            "columns_info": [],
        },
        "charts": [
            {"title": "Null cells by column", "kind": "bar", "figure_json": fig.to_json()}
        ],
        "insights": (
            "### What changed\n"
            "- **6 duplicate rows** were removed and **2 outliers** were dropped.\n"
            "- **18 empty cells** were filled, so the file has no missing values now.\n"
            "- One step failed: the `discount` column does not exist.\n\n"
            "The data is ready for analysis."
        ),
        "used_fallback": False,
        "timings": {"load": 0.01, "profile": 0.05, "plan": 1.2, "execute": 0.1, "charts": 0.2, "insights": 0.8},
        "warnings": [],
        "download_url": "/download/mock-job",
    }


def normalize_response(data, filename):
    base = {
        "job_id": "mock-job",
        "filename": filename,
        "profile_before": {},
        "plan": {"summary": "", "steps": []},
        "log": [],
        "profile_after": {},
        "charts": [],
        "insights": "",
        "used_fallback": False,
        "timings": {},
        "warnings": [],
        "download_url": "",
        "preview_before": [],
        "preview_after": [],
    }
    expected_types = {
        "job_id": str,
        "filename": str,
        "profile_before": dict,
        "plan": dict,
        "log": list,
        "profile_after": dict,
        "charts": list,
        "insights": str,
        "used_fallback": bool,
        "timings": dict,
        "warnings": list,
        "download_url": str,
        "preview_before": list,
        "preview_after": list,
    }
    if isinstance(data, dict):
        for key, expected in expected_types.items():
            value = data.get(key)
            if isinstance(value, expected):
                base[key] = value
    if not isinstance(base["plan"].get("steps"), list):
        base["plan"] = {**base["plan"], "steps": []}
    base["warnings"] = [str(w) for w in base["warnings"]]
    return base

def load_mock_response(t):
    try:
        with open(FIXTURE_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("fixture is not an object")
        return data, None
    except (OSError, ValueError):
        return build_internal_sample(), t["fixture_missing"]


def sample_csv_bytes():
    df = pd.DataFrame(
        {
            "name": ["Aysel", "Murad", "Leyla", "Orxan"],
            "city": ["Baku", "Baku", "Ganja", "Baku"],
            "price": [10.5, 42.5, 38.0, 42.5],
        }
    )
    return df.to_csv(index=False).encode("utf-8")


def read_input_bytes(uploaded, file_path):
    try:
        if uploaded is not None:
            return uploaded.getvalue()
        if file_path is not None:
            return Path(file_path).read_bytes()
    except OSError:
        return None
    return None


def analyze_real(backend, name, content, lang, t):
    url = backend.rstrip("/") + "/analyze"
    try:
        response = requests.post(
            url,
            files={"file": (name, content)},
            params={"lang": lang},
            timeout=180,
        )
    except requests.ConnectionError:
        return None, t["err_connection"], None
    except requests.Timeout:
        return None, t["err_timeout"], None
    except requests.RequestException as exc:
        return None, t["err_other"] + str(exc), None
    code = response.status_code
    if code == 200:
        try:
            data = response.json()
        except ValueError:
            return None, t["err_other"] + "invalid JSON", None
        if not isinstance(data, dict):
            return None, t["err_other"] + "invalid response format", None
        return data, None, None
    details = error_detail(response)
    if code == 400:
        return None, t["err_400"], details
    if code == 413:
        return None, t["err_413"], details
    if code == 500:
        return None, t["err_500"], details
    return None, t["err_other"] + f"HTTP {code}", details

def fetch_download(backend, download_url):
    if download_url.lower().startswith("http"):
        url = download_url
    else:
        url = backend.rstrip("/") + "/" + download_url.lstrip("/")
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None


def run_analysis(t, mock, backend, file_name, uploaded, file_path, lang):
    st.session_state["error"] = None
    st.session_state["error_details"] = None
    st.session_state["result"] = None
    st.session_state["download_bytes"] = None
    st.session_state["result_mock"] = mock
    with st.status(t["processing"], expanded=True) as status:
        try:
            raw = None
            error = None
            details = None
            extra_warning = None
            if mock:
                for stage in t["stages"]:
                    st.write(f"⏳ {stage}")
                    time.sleep(0.6)
                raw, extra_warning = load_mock_response(t)
            else:
                st.write(f"⏳ {t['sending']}")
                content = read_input_bytes(uploaded, file_path)
                if content is None:
                    error = t["read_error"]
                else:
                    raw, error, details = analyze_real(backend, file_name, content, lang, t)
            if error or raw is None:
                st.session_state["error"] = error or t["failed"]
                st.session_state["error_details"] = details
                status.update(label=t["failed"], state="error", expanded=True)
                return
            data = normalize_response(raw, file_name)
            if mock:
                data["filename"] = file_name
            if extra_warning:
                data["warnings"] = [extra_warning] + data["warnings"]
            download_bytes = None
            if data["download_url"]:
                if mock:
                    download_bytes = sample_csv_bytes()
                else:
                    download_bytes = fetch_download(backend, data["download_url"])
                    if download_bytes is None:
                        data["warnings"].append(t["download_failed"])
            st.session_state["result"] = data
            st.session_state["download_bytes"] = download_bytes
            status.update(label=t["done"], state="complete", expanded=False)
        except Exception as exc:
            st.session_state["error"] = t["err_other"] + str(exc)
            status.update(label=t["failed"], state="error", expanded=True)


def describe_step(index, step):
    if not isinstance(step, dict):
        return f"{index}. {step}"
    action = step.get("action", "?")
    details = ", ".join(f"{k}: {v}" for k, v in step.items() if k != "action")
    if details:
        return f"{index}. **{action}** — `{details}`"
    return f"{index}. **{action}**"


def build_log_table(log, t):
    rows = []
    for item in log:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "ok"))
        icon = "✅" if status == "ok" else "⚠️"
        rows_before = item.get("rows_before")
        rows_after = item.get("rows_after")
        rows_text = ""
        if rows_before is not None and rows_after is not None:
            rows_text = f"{rows_before} → {rows_after}"
        fill_value = item.get("fill_value")
        rows.append(
            {
                t["col_step"]: str(item.get("step", "")),
                t["col_action"]: str(item.get("action", "")),
                t["col_column"]: str(item.get("column") or ""),
                t["col_affected"]: str(item.get("affected", "")),
                t["col_status"]: f"{icon} {status}",
                t["col_rows"]: rows_text,
                t["col_fill"]: "" if fill_value is None else str(fill_value),
                t["col_error"]: str(item.get("error") or ""),
            }
        )
    return pd.DataFrame(rows)


def render_summary(t, data):
    before = data["profile_before"]
    after = data["profile_after"]
    rows_before = to_int(before.get("rows"))
    rows_after = to_int(after.get("rows", before.get("rows", 0)))
    nulls_before = to_int(before.get("total_null_cells"))
    nulls_after = to_int(after.get("total_null_cells", before.get("total_null_cells", 0)))
    log = [item for item in data["log"] if isinstance(item, dict)]
    ok_count = sum(1 for item in log if item.get("status") == "ok")
    columns = st.columns(4)
    columns[0].metric(t["m_rows"], f"{rows_after:,}")
    columns[1].metric(t["m_deleted"], f"{max(rows_before - rows_after, 0):,}")
    columns[2].metric(
        t["m_nulls"],
        f"{nulls_after:,}",
        delta=nulls_after - nulls_before,
        delta_color="inverse",
        help=f"{nulls_before} → {nulls_after}",
    )
    columns[3].metric(t["m_steps"], f"{ok_count}/{len(log)}")
    st.divider()
    if data["insights"].strip():
        st.markdown(data["insights"])
    else:
        st.info(t["no_insights"])


def render_plan_and_log(t, data):
    plan = data["plan"]
    st.subheader(t["plan_summary"])
    summary = str(plan.get("summary", "")).strip()
    st.write(summary if summary else "—")
    st.subheader(t["plan_steps"])
    steps = plan.get("steps", [])
    if steps:
        for index, step in enumerate(steps, start=1):
            st.markdown(describe_step(index, step))
    else:
        st.info(t["no_steps"])
    st.subheader(t["log_title"])
    table = build_log_table(data["log"], t)
    if table.empty:
        st.info(t["no_log"])
    else:
        show_dataframe(table)


def render_charts(t, data):
    charts = data["charts"]
    if not charts:
        st.info(t["no_charts"])
        return
    for index, chart in enumerate(charts):
        if not isinstance(chart, dict):
            continue
        title = str(chart.get("title", f"Chart {index + 1}"))
        st.subheader(title)
        try:
            figure_json = chart.get("figure_json")
            if isinstance(figure_json, str):
                figure_json = json.loads(figure_json)
            show_plotly(go.Figure(figure_json), f"chart_{index}")
        except Exception as exc:
            st.warning(t["chart_error"] + str(exc))


def render_data(t, data):
    before = data["profile_before"]
    after = data["profile_after"]
    st.subheader(t["data_title"])
    table = pd.DataFrame(
        {
            t["d_metric"]: [t["d_rows"], t["d_columns"], t["d_duplicates"], t["d_nulls"]],
            t["d_before"]: [
                to_int(before.get("rows")),
                to_int(before.get("columns")),
                to_int(before.get("duplicate_rows")),
                to_int(before.get("total_null_cells")),
            ],
            t["d_after"]: [
                to_int(after.get("rows")),
                to_int(after.get("columns")),
                to_int(after.get("duplicate_rows")),
                to_int(after.get("total_null_cells")),
            ],
        }
    )
    show_dataframe(table)
    st.subheader(t["data_preview"])
    preview_before = data["preview_before"]
    preview_after = data["preview_after"]
    if not preview_before and not preview_after:
        st.info(t["no_preview"])
        return
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{t['data_before']}** ({len(preview_before)})")
        show_dataframe(preview_frame(preview_before))
    with right:
        st.markdown(f"**{t['data_after']}** ({len(preview_after)})")
        show_dataframe(preview_frame(preview_after))

def render_result(t, data):

def render_results(t):
    data = st.session_state.get("result")
    if data is None:
        return
    render_result(t, data)

def fetch_health(backend):
    try:
        response = requests.get(backend.rstrip("/") + "/health", timeout=2)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        return False, False
    if not isinstance(data, dict):
        return True, False
    return True, bool(data.get("llm"))


def total_seconds(timings):
    if not isinstance(timings, dict) or not timings:
        return None
    try:
        if timings.get("total") is not None:
            return float(timings["total"])
        return float(sum(float(v) for v in timings.values()))
    except (TypeError, ValueError):
        return None


def preview_frame(rows):
    try:
        frame = pd.DataFrame(rows)
        return frame.astype(str).replace({"None": "", "nan": "", "NaT": ""})
    except Exception:
        return pd.DataFrame()


def error_detail(response):
    try:
        body = response.json()
        if isinstance(body, dict) and body.get("detail") is not None:
            return str(body["detail"])
    except ValueError:
        pass
    return response.text[:2000] if response.text else ""


def render_error(t):
    message = st.session_state.get("error")
    if not message:
        return
    st.error(message)
    details = st.session_state.get("error_details")
    if details:
        with st.expander(t["err_details"]):
            st.code(details)


def render_instructions(t):
    st.subheader(t["steps_title"])
    st.markdown(f"1. {t['step_1']}\n2. {t['step_2']}\n3. {t['step_3']}")


def main():
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("error", None)
    st.session_state.setdefault("error_details", None)
    st.session_state.setdefault("download_bytes", None)
    st.session_state.setdefault("result_mock", True)

    lang_choice = st.sidebar.selectbox("Language / Dil", ["English", "Azərbaycanca"])
    lang = "az" if lang_choice == "Azərbaycanca" else "en"
    t = TEXTS[lang]

    demo_options = [""] + list_demo_files()
    demo_choice = st.sidebar.selectbox(
        t["demo_label"],
        demo_options,
        format_func=lambda x: t["demo_none"] if x == "" else x,
        key="demo_choice",
    )
    mock = st.sidebar.checkbox(t["mock_label"], value=True, key="mock_mode")
    backend = st.sidebar.text_input(
        t["backend_label"], value="http://localhost:8000", key="backend_url"
    )

    backend_up, llm_up = fetch_health(backend)
    if backend_up:
        st.sidebar.markdown(f"🟢 {t['health_ok']}")
        st.sidebar.markdown(f"{'✅' if llm_up else '❌'} {t['llm_on'] if llm_up else t['llm_off']}")
    else:
        st.sidebar.markdown(f"🔴 {t['health_down']}")
    st.sidebar.button(t["refresh"])

    st.title("🦆 CleanVision")
    st.write(t["subtitle"])

    uploaded = st.file_uploader(t["upload_label"], type=["csv", "xlsx"])

    file_name = None
    file_path = None
    if uploaded is not None:
        file_name = uploaded.name
    elif demo_choice:
        file_name = demo_choice
        file_path = SAMPLE_DIR / demo_choice

    if file_name is None:
        st.info(t["no_input"])
    else:
        if st.button(t["clean_button"], type="primary"):
            run_analysis(t, mock, backend, file_name, uploaded, file_path, lang)

    render_error(t)

    if st.session_state.get("result") is None:
        render_instructions(t)

    render_results(t)


main()