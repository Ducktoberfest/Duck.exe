import json
import os
import time

from backend.agent.fallback import build_rule_based_plan
from backend.agent.llm_client import ask_llm, extract_json, llm_available
from backend.agent.prompts import PLAN_SYSTEM_PROMPT
from backend.agent.validator import validate_plan

MAX_COLUMNS = 30


def _plan_budget():
    try:
        return float(os.environ.get("CLEANVISION_PLAN_TIMEOUT", "60"))
    except Exception:
        return 60.0


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
            i
            for i in (profile.get("columns_info") or [])
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


def build_plan(profile):
    try:
        if llm_available():
            short = shorten_profile(profile)
            user = "INPUT PROFILE:\n" + json.dumps(short, ensure_ascii=False) + "\nOUTPUT:"
            timeout = _plan_budget()
            for _ in range(2):
                raw = ask_llm(
                    PLAN_SYSTEM_PROMPT,
                    user,
                    json_mode=True,
                    temperature=0.0,
                    timeout=timeout,
                )
                if not raw:
                    continue
                parsed = extract_json(raw)
                if not isinstance(parsed, dict) or not isinstance(parsed.get("steps"), list):
                    continue
                plan, dropped = validate_plan(parsed, profile)
                if dropped:
                    continue
                if plan and isinstance(plan.get("steps"), list) and len(plan["steps"]) > 0:
                    return plan, False
    except Exception:
        pass
    return build_rule_based_plan(profile), True