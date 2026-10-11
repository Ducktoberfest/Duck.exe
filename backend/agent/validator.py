import json

ORDER = [
    "strip_whitespace",
    "standardize_text",
    "drop_duplicates",
    "convert_type",
    "remove_outliers",
    "fill_nulls",
    "drop_rows_where_null",
    "drop_column",
]
RANK = {name: index for index, name in enumerate(ORDER)}
MULTI_COLUMN = {"strip_whitespace", "standardize_text"}
NO_COLUMN = {"drop_duplicates"}
CASES = {"lower", "upper", "title"}
TYPES = {"numeric", "datetime", "category", "string"}
OUTLIER_METHODS = {"iqr", "zscore", "clip_iqr"}
FILL_METHODS = {"mean", "median", "mode", "constant", "ffill", "drop_rows"}
NUMERIC_FILL = {"mean", "median"}
NON_TEXT_TYPES = {"numeric", "datetime", "boolean"}


def _norm(value):
    return value.strip().lower() if isinstance(value, str) else None


def _index(profile):
    names = set()
    lookup = {}
    id_names = set()
    types = {}
    infos = profile.get("columns_info") if isinstance(profile, dict) else None
    for info in infos or []:
        if not isinstance(info, dict) or info.get("name") is None:
            continue
        name = str(info["name"])
        names.add(name)
        lookup.setdefault(name.strip().lower(), name)
        kind = info.get("inferred_type")
        if isinstance(kind, str):
            types[name] = kind
        if kind == "id":
            id_names.add(name)
    return names, lookup, id_names, types


def _resolve(value, names, lookup):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    if value in names:
        return value
    return lookup.get(value.strip().lower())


def _as_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value.strip().replace(",", ".")) if isinstance(value, str) else float(value)
    except Exception:
        return None
    return number if number == number else None


def _check_params(action, raw):
    if action == "standardize_text":
        case = _norm(raw.get("case"))
        if case not in CASES:
            return None, "case dəyəri yanlışdır"
        return {"case": case}, None
    if action == "convert_type":
        target = _norm(raw.get("to"))
        if target not in TYPES:
            return None, "to dəyəri yanlışdır"
        return {"to": target}, None
    if action == "remove_outliers":
        method = _norm(raw.get("method"))
        if method not in OUTLIER_METHODS:
            return None, "method dəyəri yanlışdır"
        return {"method": method}, None
    if action == "fill_nulls":
        method = _norm(raw.get("method"))
        if method not in FILL_METHODS:
            return None, "method dəyəri yanlışdır"
        if method == "constant":
            value = raw.get("value")
            if value is None or isinstance(value, (list, dict)):
                return None, "constant üçün value yoxdur"
            return {"method": method, "value": value}, None
        return {"method": method}, None
    if action == "drop_column":
        reason = raw.get("reason")
        if isinstance(reason, str) and reason.strip():
            return {"reason": reason.strip()}, None
    return {}, None


def _check_step(raw, names, lookup, id_names, types, seen, dropped):
    if not isinstance(raw, dict):
        dropped.append({"step": raw, "reason": "addım dict deyil"})
        return None
    action = _norm(raw.get("action"))
    if action not in RANK:
        dropped.append({"step": raw, "reason": "icazəli olmayan action"})
        return None
    params, reason = _check_params(action, raw)
    if reason:
        dropped.append({"step": raw, "reason": reason})
        return None
    step = {"action": action}
    if action in NO_COLUMN:
        if (action, None) in seen:
            dropped.append({"step": raw, "reason": "təkrar addım"})
            return None
        seen.add((action, None))
        return step
    if action in MULTI_COLUMN:
        cols = raw.get("columns", raw.get("column"))
        if isinstance(cols, str):
            cols = [cols]
        if not isinstance(cols, list) or not cols:
            dropped.append({"step": raw, "reason": "columns sahəsi yoxdur"})
            return None
        kept = []
        missing = []
        ids = []
        repeats = []
        wrong_type = []
        for col in cols:
            name = _resolve(col, names, lookup)
            if name is None:
                missing.append(col)
            elif name in id_names:
                ids.append(name)
            elif action == "standardize_text" and types.get(name) in NON_TEXT_TYPES:
                wrong_type.append(name)
            elif name in kept or (action, name) in seen:
                repeats.append(name)
            else:
                kept.append(name)
        for items, why in (
            (missing, "sütun profildə yoxdur"),
            (ids, "id tipli sütuna toxunmaq olmaz"),
            (wrong_type, "standardize_text rəqəm, tarix və boolean sütuna aid deyil"),
            (repeats, "təkrar addım"),
        ):
            if items:
                dropped.append({"step": {"action": action, "columns": items}, "reason": why})
        if not kept:
            return None
        for name in kept:
            seen.add((action, name))
        step["columns"] = kept
    else:
        name = _resolve(raw.get("column"), names, lookup)
        if name is None:
            dropped.append({"step": raw, "reason": "sütun profildə yoxdur"})
            return None
        if name in id_names:
            dropped.append({"step": raw, "reason": "id tipli sütuna toxunmaq olmaz"})
            return None
        kind = types.get(name)
        if kind is not None:
            if action == "remove_outliers" and kind != "numeric":
                dropped.append({"step": raw, "reason": "remove_outliers yalnız rəqəmli sütuna aiddir"})
                return None
            if action == "fill_nulls" and params.get("method") in NUMERIC_FILL and kind != "numeric":
                dropped.append({"step": raw, "reason": "mean və median yalnız rəqəmli sütuna aiddir"})
                return None
            if action == "fill_nulls" and params.get("method") == "constant" and kind == "numeric":
                number = _as_number(params.get("value"))
                if number is None:
                    dropped.append({"step": raw, "reason": "rəqəmli sütun üçün constant rəqəm olmalıdır"})
                    return None
                params = {"method": "constant", "value": number}
        if (action, name) in seen:
            dropped.append({"step": raw, "reason": "təkrar addım"})
            return None
        seen.add((action, name))
        step["column"] = name
    step.update(params)
    return step


def validate_plan(plan, profile):
    empty = {"summary": "", "steps": []}
    try:
        if not isinstance(plan, dict):
            return empty, [{"step": plan, "reason": "plan dict deyil"}]
        summary = plan.get("summary")
        summary = summary if isinstance(summary, str) else ""
        raw_steps = plan.get("steps")
        if not isinstance(raw_steps, list):
            return {"summary": summary, "steps": []}, [{"step": raw_steps, "reason": "steps siyahı deyil"}]
        names, lookup, id_names, types = _index(profile)
        seen = set()
        dropped = []
        steps = []
        for raw in raw_steps:
            step = _check_step(raw, names, lookup, id_names, types, seen, dropped)
            if step is not None:
                steps.append(step)
        steps.sort(key=lambda s: RANK[s["action"]])
        return {"summary": summary, "steps": steps}, dropped
    except Exception as error:
        return empty, [{"step": None, "reason": "validator xətası: " + str(error)}]


if __name__ == "__main__":
    test_profile = {
        "rows": 200,
        "columns": 4,
        "duplicate_rows": 3,
        "total_null_cells": 156,
        "columns_info": [
            {"name": "order_id", "inferred_type": "id"},
            {"name": "Product", "inferred_type": "categorical"},
            {"name": "price", "inferred_type": "numeric"},
            {"name": "notes", "inferred_type": "text"},
        ],
    }
    bad_plan = {
        "summary": "test",
        "steps": [
            {"action": "fill_nulls", "column": "Price", "method": "Median"},
            {"action": "fill_nulls", "column": "price", "method": "mean"},
            {"action": "strip_whitespace", "columns": ["product", "order_id", "salary"]},
            {"action": "standardize_text", "columns": ["Product"], "case": "capitalize"},
            {"action": "delete_rows", "column": "price"},
            {"action": "convert_type", "column": "price", "to": "float"},
            {"action": "remove_outliers", "column": "prize", "method": "iqr"},
            {"action": "fill_nulls", "column": "notes", "method": "constant"},
            {"action": "fill_nulls", "column": "order_id", "method": "median"},
            {"action": "remove_outliers", "column": "Product", "method": "iqr"},
            {"action": "fill_nulls", "column": "notes", "method": "mean"},
            {"action": "drop_duplicates"},
            {"action": "drop_duplicates"},
            {"action": "convert_type", "column": "price", "to": "numeric"},
            {"action": "drop_column", "column": "notes", "reason": "72% empty"},
            "salam",
        ],
    }
    clean_plan, dropped_steps = validate_plan(bad_plan, test_profile)
    print(json.dumps(clean_plan, indent=2, ensure_ascii=False))
    for item in dropped_steps:
        print(json.dumps(item, ensure_ascii=False))
    print("dropped:", len(dropped_steps))