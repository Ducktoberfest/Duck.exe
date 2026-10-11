import os
import re
import warnings

import pandas as pd

from backend.agent.llm_client import ask_llm, extract_json

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")
MAX_PROMPT_STEPS = 12
MAX_BULLETS = 8
MAX_WORDS = 150

KEY_ALIASES = {
    "cleaned": ("cleaned", "clean", "cleaning"),
    "found": ("found", "findings", "finding"),
    "next": ("next", "next_step", "nextstep", "next step"),
}

STEP_PHRASES = {
    "drop_duplicates": "removed %d duplicate rows",
    "drop_rows_where_null": "removed %d rows with empty values",
    "fill_nulls": "handled %d empty cells",
    "strip_whitespace": "trimmed extra spaces in %d values",
    "standardize_text": "changed the letter case in %d values",
    "convert_type": "converted the type, %d values could not be read and became empty",
    "remove_outliers": "handled %d extreme values",
}

HEADINGS = {
    "az": ("Nə təmizlədim", "Nə tapdım", "Növbəti addım"),
    "en": ("What I cleaned", "What I found", "Next step"),
}

LANGUAGE_NAMES = {"az": "Azerbaijani", "en": "English"}

EXAMPLE_OUTPUT = {
    "az": '{"cleaned": "Təkrar olan 5 sətir silindi, sətirlər 200-dən 195-ə endi. Boş xanalar 156-dan 6-ya azaldı. Uğursuz addım yoxdur.", "found": "\'price\' ilə \'revenue\' arasında güclü müsbət əlaqə var (r = 0.87).", "next": "Bu əlaqəni qrafiklə yoxlayın."}',
    "en": '{"cleaned": "5 duplicate rows were removed, so rows went from 200 to 195. Empty cells dropped from 156 to 6. No step failed.", "found": "\'price\' and \'revenue\' have a strong positive link (r = 0.87).", "next": "Check this link with a chart."}',
}

INSIGHTS_SYSTEM_PROMPT = """You write a short plain-language summary of a data cleaning job for a non-technical reader.

RULES
- Use ONLY the numbered FACTS you are given. No other information.
- Never invent numbers. Copy every number exactly as it appears in the facts. Do not calculate anything yourself.
- Write in __LANGUAGE__. Use simple words and short sentences.
- Under 100 words in total.
- Return ONE JSON object with exactly three keys: "cleaned", "found", "next". Each value is plain text, no lists, no markdown.
  "cleaned": what was cleaned (rows, empty cells, main steps, failed steps if any).
  "found": what was found (monthly change, correlation). If the facts contain nothing like that, say that nothing notable was found.
  "next": one short suggestion for the next step.
- The example below only shows the format. Never reuse its numbers or column names.

EXAMPLE
FACTS:
1. Rows: 200 before cleaning, 195 after cleaning.
2. Empty cells: 156 before cleaning, 6 after cleaning.
3. Step 1: drop_duplicates, removed 5 duplicate rows.
4. Failed steps: 0.
5. Strongest correlation: 'price' and 'revenue', r = 0.87 (positive).
OUTPUT:
__EXAMPLE__
"""

TEXTS = {
    "az": {
        "rows": "Sətirlər: {before} → {after}",
        "nulls": "Boş xanalar: {before} → {after}",
        "steps": "Addımlar: {ok} uğurlu, {failed} uğursuz",
        "bullet": "{label} ({column}): {affected} təsirlənib",
        "bullet_nocol": "{label}: {affected} təsirlənib",
        "fill": ", doldurma dəyəri {value}",
        "no_steps": "Təmizləmə addımı qeyd olunmayıb",
        "more": "və daha {count} addım",
        "growth": "Aylıq cəm ('{column}'): {previous} üçün {previous_total}, {last} üçün {last_total} ({change})",
        "corr": "'{a}' ilə '{b}' arasında güclü {kind} korrelyasiya (r = {r})",
        "positive": "müsbət",
        "negative": "mənfi",
        "none": "Əlavə tendensiya və ya güclü korrelyasiya aşkar edilmədi",
        "next_failed": "Uğursuz addımlara baxın və planı yenidən işə salın",
        "next_nulls": "Qalan {count} boş xananı yoxlayın",
        "next_corr": "'{a}' və '{b}' arasındakı əlaqəni qrafiklə araşdırın",
        "next_ready": "Data analiz və qrafiklər üçün hazırdır",
        "actions": {
            "strip_whitespace": "Boşluqlar təmizləndi",
            "standardize_text": "Mətn standartlaşdırıldı",
            "drop_duplicates": "Təkrar sətirlər silindi",
            "convert_type": "Tip çevrildi",
            "remove_outliers": "Kənar qiymətlər işləndi",
            "fill_nulls": "Boşluqlar dolduruldu",
            "drop_column": "Sütun silindi",
            "drop_rows_where_null": "Boş sətirlər silindi",
        },
    },
    "en": {
        "rows": "Rows: {before} → {after}",
        "nulls": "Empty cells: {before} → {after}",
        "steps": "Steps: {ok} succeeded, {failed} failed",
        "bullet": "{label} ({column}): {affected} affected",
        "bullet_nocol": "{label}: {affected} affected",
        "fill": ", fill value {value}",
        "no_steps": "No cleaning steps were recorded",
        "more": "and {count} more steps",
        "growth": "Monthly total ('{column}'): {previous_total} for {previous}, {last_total} for {last} ({change})",
        "corr": "Strong {kind} correlation between '{a}' and '{b}' (r = {r})",
        "positive": "positive",
        "negative": "negative",
        "none": "No notable trend or strong correlation was found",
        "next_failed": "Review the failed steps and run the plan again",
        "next_nulls": "Check the remaining {count} empty cells",
        "next_corr": "Explore the relationship between '{a}' and '{b}' with a chart",
        "next_ready": "The data is ready for analysis and charts",
        "actions": {
            "strip_whitespace": "Whitespace trimmed",
            "standardize_text": "Text standardized",
            "drop_duplicates": "Duplicate rows removed",
            "convert_type": "Type converted",
            "remove_outliers": "Outliers handled",
            "fill_nulls": "Gaps filled",
            "drop_column": "Column dropped",
            "drop_rows_where_null": "Rows with gaps removed",
        },
    },
}


def _log(message):
    try:
        print(message)
    except Exception:
        pass


def _norm_lang(lang):
    if isinstance(lang, str) and lang.strip().lower().startswith("en"):
        return "en"
    return "az"


def _to_float(value):
    try:
        if isinstance(value, bool):
            return None
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    except Exception:
        return None


def _to_int(value):
    number = _to_float(value)
    return None if number is None else int(round(number))


def _fmt(value):
    try:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (int, float)):
            number = float(value)
            if number != number:
                return "nan"
            if number.is_integer():
                return str(int(number))
            return ("%.2f" % number).rstrip("0").rstrip(".")
        return str(value)
    except Exception:
        return str(value)


def _names_by_type(profile, kind):
    names = set()
    infos = profile.get("columns_info") if isinstance(profile, dict) else None
    for info in infos or []:
        if isinstance(info, dict) and info.get("inferred_type") == kind and info.get("name") is not None:
            names.add(str(info["name"]))
    return names


def _as_dates(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_localize(None)


def _pick_date_column(df, hints):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    for col in df.columns:
        if str(col) in hints and not pd.api.types.is_numeric_dtype(df[col]):
            parsed = _as_dates(df[col])
            if len(parsed) and parsed.notna().mean() >= 0.8:
                return col
    return None


def _pick_value_column(df, date_col, skip):
    for col in df.columns:
        if col == date_col or str(col) in skip:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            continue
        if series.nunique(dropna=True) > 1:
            return col
    return None


def monthly_growth(df, skip=(), date_hints=()):
    try:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        date_col = _pick_date_column(df, date_hints)
        if date_col is None:
            return None
        value_col = _pick_value_column(df, date_col, skip)
        if value_col is None:
            return None
        frame = pd.DataFrame({
            "period": _as_dates(df[date_col]).dt.to_period("M"),
            "value": pd.to_numeric(df[value_col], errors="coerce"),
        }).dropna()
        if frame.empty:
            return None
        monthly = frame.groupby("period")["value"].sum().sort_index()
        if len(monthly) < 2:
            return None
        last_total = float(monthly.iloc[-1])
        previous_total = float(monthly.iloc[-2])
        if previous_total == 0:
            return None
        return {
            "date_column": str(date_col),
            "value_column": str(value_col),
            "last_period": str(monthly.index[-1]),
            "previous_period": str(monthly.index[-2]),
            "last_total": last_total,
            "previous_total": previous_total,
            "pct_change": (last_total - previous_total) / abs(previous_total) * 100,
        }
    except Exception:
        return None


def strongest_correlation(df, skip=()):
    try:
        if not isinstance(df, pd.DataFrame) or len(df) < 3:
            return None
        columns = []
        for col in df.columns:
            series = df[col]
            if str(col) in skip or pd.api.types.is_bool_dtype(series):
                continue
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 1:
                columns.append(col)
        columns = columns[:30]
        if len(columns) < 2:
            return None
        matrix = df[columns].corr()
        best = None
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                r = float(matrix.iat[i, j])
                if r != r or abs(r) <= 0.5:
                    continue
                if best is None or abs(r) > abs(best[2]):
                    best = (columns[i], columns[j], r)
        if best is None:
            return None
        return {"a": str(best[0]), "b": str(best[1]), "r": round(best[2], 2)}
    except Exception:
        return None


def _empty_data():
    return {
        "rows_before": None,
        "rows_after": None,
        "nulls_before": None,
        "nulls_after": None,
        "steps": [],
        "failed": 0,
        "growth": None,
        "corr": None,
    }


def _gather(profile_before, profile_after, log, df_clean):
    before = profile_before if isinstance(profile_before, dict) else {}
    after = profile_after if isinstance(profile_after, dict) else {}
    data = _empty_data()
    data["rows_before"] = _to_int(before.get("rows"))
    data["rows_after"] = _to_int(after.get("rows"))
    data["nulls_before"] = _to_int(before.get("total_null_cells"))
    data["nulls_after"] = _to_int(after.get("total_null_cells"))
    for index, item in enumerate(log if isinstance(log, list) else []):
        if not isinstance(item, dict):
            continue
        status = item.get("status")
        if status == "failed":
            data["failed"] += 1
            continue
        if status != "ok":
            continue
        entry = {
            "step": _to_int(item.get("step")) or index + 1,
            "action": str(item.get("action", "")),
            "column": str(item["column"]) if item.get("column") is not None else None,
            "affected": _to_int(item.get("affected")) or 0,
        }
        if item.get("fill_value") is not None:
            entry["fill_value"] = item["fill_value"]
        data["steps"].append(entry)
    if isinstance(df_clean, pd.DataFrame):
        skip = _names_by_type(after, "id") | _names_by_type(before, "id")
        data["growth"] = monthly_growth(df_clean, skip, _names_by_type(after, "datetime"))
        data["corr"] = strongest_correlation(df_clean, skip)
    return data


def _facts_from(data, max_steps=None):
    facts = []
    if data["rows_before"] is not None and data["rows_after"] is not None:
        facts.append("Rows: %d before cleaning, %d after cleaning." % (data["rows_before"], data["rows_after"]))
    if data["nulls_before"] is not None and data["nulls_after"] is not None:
        facts.append("Empty cells: %d before cleaning, %d after cleaning." % (data["nulls_before"], data["nulls_after"]))
    steps = data["steps"]
    if max_steps is not None and len(steps) > max_steps:
        ranked = sorted(range(len(steps)), key=lambda i: -steps[i]["affected"])[:max_steps]
        steps = [steps[i] for i in sorted(ranked)]
    for step in steps:
        text = "Step %d: %s" % (step["step"], step["action"])
        if step["column"]:
            text += " on column '%s'" % step["column"]
        if step["action"] == "drop_column":
            text += ", dropped the column"
        elif step["action"] in STEP_PHRASES:
            text += ", " + STEP_PHRASES[step["action"]] % step["affected"]
        else:
            text += ", affected %d" % step["affected"]
        if "fill_value" in step:
            text += ", fill value %s" % _fmt(step["fill_value"])
        facts.append(text + ".")
    facts.append("Failed steps: %d." % data["failed"])
    growth = data["growth"]
    if growth:
        pct = round(growth["pct_change"], 1)
        if pct > 0:
            change = "an increase of %s%%" % _fmt(pct)
        elif pct < 0:
            change = "a decrease of %s%%" % _fmt(abs(pct))
        else:
            change = "no change (0%)"
        facts.append(
            "Monthly total of '%s' for %s was %s versus %s in %s, %s."
            % (
                growth["value_column"],
                growth["last_period"],
                _fmt(growth["last_total"]),
                _fmt(growth["previous_total"]),
                growth["previous_period"],
                change,
            )
        )
    corr = data["corr"]
    if corr:
        kind = "positive" if corr["r"] > 0 else "negative"
        facts.append("Strongest correlation: '%s' and '%s', r = %s (%s)." % (corr["a"], corr["b"], _fmt(corr["r"]), kind))
    return facts


def collect_facts(profile_before, profile_after, log, df_clean=None):
    try:
        return _facts_from(_gather(profile_before, profile_after, log, df_clean))
    except Exception:
        return []


def _candidates(token):
    result = []

    def add(text):
        try:
            value = float(text)
        except ValueError:
            return
        decimals = len(text.split(".")[1]) if "." in text else 0
        result.append((value, decimals))

    if "," not in token and token.count(".") <= 1:
        add(token)
        if "." in token:
            head, tail = token.split(".")
            if len(tail) == 3:
                add(head + tail)
        return result
    last_comma = token.rfind(",")
    last_dot = token.rfind(".")
    if last_comma != -1 and last_dot != -1:
        if last_dot > last_comma:
            add(token.replace(",", ""))
        else:
            add(token.replace(".", "").replace(",", "."))
    elif last_comma != -1:
        if token.count(",") == 1:
            add(token.replace(",", "."))
        if all(len(part) == 3 for part in token.split(",")[1:]):
            add(token.replace(",", ""))
    else:
        if all(len(part) == 3 for part in token.split(".")[1:]):
            add(token.replace(".", ""))
    return result


def _numbers_ok(text, facts):
    try:
        known = []
        for fact in facts:
            for token in NUM_RE.findall(fact):
                for value, _ in _candidates(token):
                    known.append(value)
        for token in NUM_RE.findall(MARKER_RE.sub("", text)):
            options = _candidates(token)
            if not options:
                return False
            matched = False
            for value, decimals in options:
                tolerance = 0.5 * 10 ** (-decimals) + 1e-9
                if any(abs(value - fact_value) <= tolerance for fact_value in known):
                    matched = True
                    break
            if not matched:
                return False
        return True
    except Exception:
        return False


def _system_prompt(lang):
    return (
        INSIGHTS_SYSTEM_PROMPT
        .replace("__LANGUAGE__", LANGUAGE_NAMES[lang])
        .replace("__EXAMPLE__", EXAMPLE_OUTPUT[lang])
    )


def _as_text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(item.strip() for item in value if isinstance(item, str)).strip()
    return ""


def _debug(raw, reason):
    if os.environ.get("CLEANVISION_DEBUG"):
        _log("[insights] RAW: " + str(raw)[:700].replace("\n", " ") + " | " + str(reason))


def _pick(parsed, name):
    lowered = {str(key).strip().lower(): value for key, value in parsed.items()}
    for key in KEY_ALIASES[name]:
        if key in lowered:
            text = _as_text(lowered[key])
            if text:
                return text
    return ""


def _llm_sections(prompt_facts, all_facts, lang):
    numbered = "\n".join("%d. %s" % (index + 1, fact) for index, fact in enumerate(prompt_facts))
    raw = ask_llm(_system_prompt(lang), "FACTS:\n" + numbered + "\nOUTPUT:", True, 0.1)
    if not isinstance(raw, str) or not raw.strip():
        return None, "bos cavab"
    parsed = extract_json(raw)
    if parsed is None:
        _debug(raw, "json alinmadi")
        return None, "json alinmadi"
    parts = [_pick(parsed, name) for name in ("cleaned", "found", "next")]
    if len(re.findall(r"\w+", " ".join(parts))) > MAX_WORDS:
        _debug(raw, "metn cox uzundur")
        return None, "metn cox uzundur"
    reasons = []
    for index, name in enumerate(("cleaned", "found", "next")):
        if not parts[index]:
            parts[index] = None
            reasons.append(name + ": bos")
        elif not _numbers_ok(parts[index], all_facts):
            parts[index] = None
            reasons.append(name + ": faktlarda olmayan reqem")
    reason = "; ".join(reasons)
    if all(part is None for part in parts):
        _debug(raw, reason)
        return None, reason
    if reason:
        _debug(raw, reason)
    return parts, reason


def _assemble(lang, cleaned, found, next_step):
    first, second, third = HEADINGS[lang]
    return "### %s\n%s\n\n### %s\n%s\n\n### %s\n%s" % (first, cleaned, second, found, third, next_step)


def _template_parts(data, lang):
    t = TEXTS[lang]
    lines = []
    if data["rows_before"] is not None and data["rows_after"] is not None:
        lines.append("- " + t["rows"].format(before=data["rows_before"], after=data["rows_after"]) + ".")
    if data["nulls_before"] is not None and data["nulls_after"] is not None:
        lines.append("- " + t["nulls"].format(before=data["nulls_before"], after=data["nulls_after"]) + ".")
    ok_count = len(data["steps"])
    if ok_count or data["failed"]:
        lines.append("- " + t["steps"].format(ok=ok_count, failed=data["failed"]) + ".")
    shown = [step for step in data["steps"] if step["affected"] > 0]
    for step in shown[:MAX_BULLETS]:
        label = t["actions"].get(step["action"], step["action"])
        if step["column"]:
            line = t["bullet"].format(label=label, column=step["column"], affected=step["affected"])
        else:
            line = t["bullet_nocol"].format(label=label, affected=step["affected"])
        if "fill_value" in step:
            line += t["fill"].format(value=_fmt(step["fill_value"]))
        lines.append("- " + line + ".")
    if len(shown) > MAX_BULLETS:
        lines.append("- " + t["more"].format(count=len(shown) - MAX_BULLETS) + ".")
    if not lines:
        lines.append("- " + t["no_steps"] + ".")
    cleaned = "\n".join(lines)

    found_lines = []
    growth = data["growth"]
    if growth:
        pct = round(growth["pct_change"], 1)
        change = ("+" if pct > 0 else "") + _fmt(pct) + "%"
        found_lines.append("- " + t["growth"].format(
            column=growth["value_column"],
            previous=growth["previous_period"],
            previous_total=_fmt(growth["previous_total"]),
            last=growth["last_period"],
            last_total=_fmt(growth["last_total"]),
            change=change,
        ) + ".")
    corr = data["corr"]
    if corr:
        kind = t["positive"] if corr["r"] > 0 else t["negative"]
        found_lines.append("- " + t["corr"].format(a=corr["a"], b=corr["b"], kind=kind, r=_fmt(corr["r"])) + ".")
    if not found_lines:
        found_lines.append("- " + t["none"] + ".")
    found = "\n".join(found_lines)

    if data["failed"] > 0:
        next_step = t["next_failed"]
    elif data["nulls_after"] is not None and data["nulls_after"] > 0:
        next_step = t["next_nulls"].format(count=data["nulls_after"])
    elif corr:
        next_step = t["next_corr"].format(a=corr["a"], b=corr["b"])
    else:
        next_step = t["next_ready"]
    return cleaned, found, next_step + "."


def _template(data, lang):
    cleaned, found, next_step = _template_parts(data, lang)
    return _assemble(lang, cleaned, found, next_step)


def write_insights(profile_before, profile_after, log, df_clean=None, lang="az"):
    lang = _norm_lang(lang)
    try:
        data = _gather(profile_before, profile_after, log, df_clean)
    except Exception:
        data = _empty_data()
    try:
        facts = _facts_from(data)
        prompt_facts = _facts_from(data, MAX_PROMPT_STEPS)
        parts, reason = _llm_sections(prompt_facts, facts, lang)
        if parts is not None:
            if any(part is None for part in parts):
                backup = _template_parts(data, lang)
                parts = [part if part is not None else backup[index] for index, part in enumerate(parts)]
                _log("[insights] LLM metni qismen qebul edildi (" + reason + ")")
            else:
                _log("[insights] LLM metni qebul edildi")
            return _assemble(lang, parts[0], parts[1], parts[2])
        _log("[insights] LLM metni atildi (" + reason + "), sablon istifade olunur")
    except Exception:
        _log("[insights] LLM xetasi, sablon istifade olunur")
    try:
        return _template(data, lang)
    except Exception:
        return ""


if __name__ == "__main__":
    test_before = {"rows": 8, "columns": 5, "duplicate_rows": 2, "total_null_cells": 5, "columns_info": [
        {"name": "order_id", "inferred_type": "id"},
        {"name": "price", "inferred_type": "numeric"},
        {"name": "quantity", "inferred_type": "numeric"},
        {"name": "order_date", "inferred_type": "datetime"},
        {"name": "notes", "inferred_type": "text"},
    ]}
    test_after = {"rows": 6, "columns": 4, "duplicate_rows": 0, "total_null_cells": 0, "columns_info": [
        {"name": "order_id", "inferred_type": "id"},
        {"name": "price", "inferred_type": "numeric"},
        {"name": "quantity", "inferred_type": "numeric"},
        {"name": "order_date", "inferred_type": "datetime"},
    ]}
    test_log = [
        {"step": 1, "action": "drop_duplicates", "affected": 2, "status": "ok", "rows_before": 8, "rows_after": 6},
        {"step": 2, "action": "convert_type", "column": "price", "affected": 0, "status": "ok"},
        {"step": 3, "action": "fill_nulls", "column": "price", "affected": 3, "status": "ok", "fill_value": 120.5},
        {"step": 4, "action": "fill_nulls", "column": "quantity", "affected": 2, "status": "ok", "fill_value": 3},
        {"step": 5, "action": "drop_column", "column": "notes", "affected": 1, "status": "ok"},
        {"step": 6, "action": "remove_outliers", "column": "price", "affected": 0, "status": "failed", "error": "test"},
    ]
    test_df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6],
        "price": [100, 120, 110, 130, 150, 170],
        "quantity": [1, 2, 2, 3, 4, 5],
        "order_date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-03", "2024-02-18", "2024-03-07", "2024-03-25"]),
    })
    for number, fact in enumerate(collect_facts(test_before, test_after, test_log, test_df), 1):
        print("%d. %s" % (number, fact))
    for language in ("az", "en"):
        print("\n--- " + language)
        print(write_insights(test_before, test_after, test_log, test_df, language))