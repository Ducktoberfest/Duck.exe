import json
import os
import time

from backend.agent.fallback import build_rule_based_plan
from backend.agent.llm_client import ask_llm, extract_json, llm_available
from backend.agent.prompts import PLAN_SYSTEM_PROMPT
from backend.agent.validator import validate_plan

MAX_COLUMNS = 30
MIN_ATTEMPT_SECONDS = 15.0


def _plan_budget():
    try:
        return float(os.environ.get("CLEANVISION_PLAN_TIMEOUT", "100"))
    except Exception:
        return 100.0


def _num(value, default=0.0):
    try:
        number = float(value)
        if number != number:
            return default
        return number
    except Exception:
        return default


def _score(info):
    try:
        score = _num(info.get("null_pct"))
        if _num(info.get("outlier_count")) > 0:
            score += 20
        if info.get("whitespace_issues"):
            score += 10
        return score
    except Exception:
        return 0.0


def shorten_profile(profile):
    try:
        infos = [
            i for i in (profile.get("columns_info") or [])
            if isinstance(i, dict) and i.get("inferred_type") != "id"
        ]
        if len(infos) > MAX_COLUMNS:
            ranked = sorted(range(len(infos)), key=lambda i: -_score(infos[i]))
            keep = sorted(ranked[:MAX_COLUMNS])
            infos = [infos[i] for i in keep]
        columns_info = []
        for info in infos:
            samples = info.get("sample_values") or []
            item = {
                "name": str(info.get("name", "")),
                "dtype": str(info.get("dtype", "")),
                "inferred_type": str(info.get("inferred_type", "")),
                "null_pct": round(_num(info.get("null_pct")), 1),
                "unique": int(_num(info.get("unique"))),
                "sample_values": [str(v)[:30] for v in list(samples)[:4]],
                "outlier_count": int(_num(info.get("outlier_count"))),
            }
            if _num(info.get("null_pct")) > 60:
                item["too_empty"] = True
            columns_info.append(item)
        return {
            "rows": int(_num(profile.get("rows"))),
            "columns": int(_num(profile.get("columns"))),
            "duplicate_rows": int(_num(profile.get("duplicate_rows"))),
            "columns_info": columns_info,
        }
    except Exception:
        return {"rows": 0, "columns": 0, "duplicate_rows": 0, "columns_info": []}


def _log(message):
    try:
        print(message)
    except Exception:
        pass


def _log_dropped(dropped):
    try:
        for item in dropped:
            print("[validator] atildi: " + json.dumps(item, ensure_ascii=False, default=str))
    except Exception:
        pass


def _prune_dropped(steps):
    dropped = {s.get("column") for s in steps if s.get("action") == "drop_column" and s.get("column")}
    if not dropped:
        return steps
    result = []
    for step in steps:
        if step.get("action") == "drop_column":
            result.append(step)
        elif "columns" in step:
            cols = [c for c in step["columns"] if c not in dropped]
            if cols:
                kept = dict(step)
                kept["columns"] = cols
                result.append(kept)
        elif step.get("column") not in dropped:
            result.append(step)
    return result


SINGLE_ACTIONS = ("drop_duplicates", "drop_column", "convert_type", "fill_nulls", "remove_outliers")
MULTI_ACTIONS = ("strip_whitespace", "standardize_text")


def _covered(steps, action):
    cols = set()
    for step in steps:
        if step.get("action") == action:
            cols.update(step.get("columns") or [])
    return cols


def _missing_steps(plan, rule_plan):
    have = {
        (s.get("action"), s.get("column"))
        for s in plan["steps"]
        if s.get("action") in SINGLE_ACTIONS
    }
    extra = []
    for step in rule_plan["steps"]:
        action = step["action"]
        if action in SINGLE_ACTIONS:
            if (action, step.get("column")) not in have:
                extra.append(step)
        elif action in MULTI_ACTIONS:
            covered = _covered(plan["steps"], action)
            cols = [c for c in step.get("columns", []) if c not in covered]
            if cols:
                item = dict(step)
                item["columns"] = cols
                extra.append(item)
    return extra


def _ensure_required(plan, profile):
    try:
        rule_plan = build_rule_based_plan(profile)
        extra = _missing_steps(plan, rule_plan)
        steps = _prune_dropped(plan["steps"] + extra)
        merged, _ = validate_plan({"summary": plan.get("summary", ""), "steps": steps}, profile)
        if not merged["steps"]:
            return plan
        if extra:
            _log("[planner] LLM planina %d mecburi addim elave olundu" % len(extra))
        return merged
    except Exception:
        return plan


def _reject_bad_drops(plan, profile):
    try:
        pct = {}
        for info in profile.get("columns_info") or []:
            if isinstance(info, dict) and info.get("name") is not None:
                pct[str(info["name"])] = _num(info.get("null_pct"))
        steps = []
        for step in plan["steps"]:
            if step.get("action") == "drop_column" and pct.get(step.get("column"), 100.0) <= 60.0:
                _log("[planner] drop_column(%s) atildi: sutun 60%%-den az bosdur" % step.get("column"))
                continue
            steps.append(step)
        return {"summary": plan.get("summary", ""), "steps": steps}
    except Exception:
        return plan


def _fix_reasons(plan, profile):
    try:
        pct = {}
        for info in profile.get("columns_info") or []:
            if isinstance(info, dict) and info.get("name") is not None:
                pct[str(info["name"])] = _num(info.get("null_pct"))
        steps = []
        for step in plan["steps"]:
            if step.get("action") == "drop_column":
                item = dict(step)
                value = pct.get(item.get("column"), 0.0)
                if value > 0:
                    item["reason"] = str(round(value)) + "% boş"
                else:
                    item.pop("reason", None)
                steps.append(item)
            else:
                steps.append(step)
        return {"summary": plan.get("summary", ""), "steps": steps}
    except Exception:
        return plan


def build_plan(profile):
    try:
        if llm_available():
            short = shorten_profile(profile)
            user = "INPUT PROFILE:\n" + json.dumps(short, ensure_ascii=False) + "\nOUTPUT:"
            started = time.monotonic()
            budget = _plan_budget()
            for temperature in (0.1, 0.3):
                remaining = budget - (time.monotonic() - started)
                if remaining < MIN_ATTEMPT_SECONDS:
                    break
                raw = ask_llm(PLAN_SYSTEM_PROMPT, user, True, temperature, remaining)
                parsed = extract_json(raw)
                if parsed is None:
                    continue
                plan, dropped = validate_plan(parsed, profile)
                _log_dropped(dropped)
                if plan["steps"]:
                    return _fix_reasons(_ensure_required(_reject_bad_drops(plan, profile), profile), profile), False
    except Exception:
        pass
    _log("[planner] LLM plani alinmadi, qaydali ehtiyat plan istifade olunur")
    return build_rule_based_plan(profile), True


if __name__ == "__main__":
    test_profile = {
        "rows": 200,
        "columns": 6,
        "duplicate_rows": 3,
        "total_null_cells": 150,
        "columns_info": [
            {"name": "order_id", "dtype": "int64", "inferred_type": "id", "null_count": 0, "null_pct": 0.0, "unique": 200, "sample_values": ["1001", "1002", "1003", "1004"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
            {"name": "product", "dtype": "object", "inferred_type": "categorical", "null_count": 0, "null_pct": 0.0, "unique": 8, "sample_values": [" Laptop", "laptop ", "PHONE", "Phone"], "stats": None, "outlier_count": 0, "whitespace_issues": True},
            {"name": "price", "dtype": "object", "inferred_type": "numeric", "null_count": 12, "null_pct": 6.0, "unique": 150, "sample_values": ["450", "1200", "399", "99999"], "stats": None, "outlier_count": 4, "whitespace_issues": False},
            {"name": "quantity", "dtype": "int64", "inferred_type": "numeric", "null_count": 0, "null_pct": 0.0, "unique": 9, "sample_values": ["1", "2", "3", "5"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
            {"name": "notes", "dtype": "object", "inferred_type": "text", "null_count": 144, "null_pct": 72.0, "unique": 20, "sample_values": ["gift", "urgent", "nan", "call me"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
            {"name": "order_date", "dtype": "object", "inferred_type": "datetime", "null_count": 0, "null_pct": 0.0, "unique": 90, "sample_values": ["2024-01-05", "2024-01-06", "2024-02-11", "2024-03-01"], "stats": None, "outlier_count": 0, "whitespace_issues": False},
        ],
    }
    result_plan, result_fallback = build_plan(test_profile)
    print("used_fallback:", result_fallback)
    print(json.dumps(result_plan, indent=2, ensure_ascii=False))