import io
import os
import re
import sys
import time
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.agent.fallback import build_rule_based_plan
from backend.agent.llm_client import llm_available
from backend.agent.planner import build_plan
try:
    from backend.executor import execute_plan
    from backend.loader import load_dataframe
    from backend.profiler import profile_dataframe
except Exception:
    from mini_pipeline import execute_plan, load_dataframe, profile_dataframe
    print("DIQQET: backend.loader/profiler tapilmadi, scripts/mini_pipeline.py istifade olunur\n")

EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")
DROPPED_RE = re.compile(r"^\[validator\] atildi:", re.M)
ADDED_RE = re.compile(r"LLM planina (\d+) mecburi")


def _keys(plan):
    keys = set()
    try:
        for step in (plan or {}).get("steps", []):
            action = step.get("action")
            if "columns" in step:
                for column in step["columns"]:
                    keys.add((action, column))
            else:
                keys.add((action, step.get("column")))
    except Exception:
        pass
    return keys


def evaluate_file(path):
    name = os.path.basename(path)
    result = {"file": name, "ok": False, "error": ""}
    try:
        with open(path, "rb") as handle:
            data = handle.read()
        df = load_dataframe(data, name)
        profile = profile_dataframe(df)
    except Exception as error:
        result["error"] = "yuklenme/profil xetasi: " + str(error)
        if name.lower().endswith((".xlsx", ".xls", ".xlsm")):
            result["error"] += " (Excel ucun: pip install openpyxl)"
        return result
    buffer = io.StringIO()
    start = time.perf_counter()
    try:
        with redirect_stdout(buffer):
            plan, used_fallback = build_plan(profile)
    except Exception as error:
        result["error"] = "build_plan xetasi: " + str(error)
        return result
    seconds = time.perf_counter() - start
    text = buffer.getvalue()
    dropped = len(DROPPED_RE.findall(text))
    added = sum(int(n) for n in ADDED_RE.findall(text))
    steps = len(plan.get("steps", [])) if isinstance(plan, dict) else 0
    llm_steps = 0 if used_fallback else max(steps - added, 0)
    rule_keys = _keys(build_rule_based_plan(profile))
    plan_keys = _keys(plan)
    coverage = len(rule_keys & plan_keys) / len(rule_keys) * 100 if rule_keys else 100.0
    order = lambda k: (str(k[0]), str(k[1]))
    missing = sorted(rule_keys - plan_keys, key=order)
    extra = sorted(plan_keys - rule_keys, key=order)
    failed = []
    try:
        with redirect_stdout(io.StringIO()):
            _, exec_log = execute_plan(df.copy(), plan)
        for entry in exec_log:
            if entry.get("status") != "ok":
                failed.append((entry.get("action"), entry.get("column"), entry.get("error")))
    except Exception as error:
        failed.append(("execute_plan", None, str(error)))
    result.update({
        "ok": True,
        "steps": steps,
        "dropped": dropped,
        "added": added,
        "llm_steps": llm_steps,
        "used_fallback": bool(used_fallback),
        "seconds": seconds,
        "coverage": coverage,
        "missing": missing,
        "extra": extra,
        "failed": failed,
        "rows": profile.get("rows"),
        "columns": profile.get("columns"),
    })
    return result


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "sample_data")
    if not os.path.isdir(folder):
        print("Qovluq tapilmadi: " + folder)
        return 1
    files = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith(EXTENSIONS) and not f.startswith("~$")
    )
    if not files:
        print("Qovluqda fayl yoxdur: " + folder)
        return 1
    print("Ollama: " + ("aciqdir" if llm_available() else "SONDURULUB (hamisi fallback olacaq)"))
    print("Fayl sayi: %d\n" % len(files))
    results = []
    for path in files:
        item = evaluate_file(path)
        results.append(item)
        if not item["ok"]:
            print("%-28s XETA: %s" % (item["file"][:28], item["error"]))
            continue
        print(
            "%-28s %3dx%-3d addim=%-3d atilan=%-3d fallback=%-5s vaxt=%6.2fs tutum=%5.1f%%"
            % (
                item["file"][:28],
                item["rows"] or 0,
                item["columns"] or 0,
                item["steps"],
                item["dropped"],
                item["used_fallback"],
                item["seconds"],
                item["coverage"],
            )
        )
        if item["added"]:
            print("    mecburi elave olunan addim: %d" % item["added"])
        if item["missing"]:
            print("    buraxilan: " + ", ".join("%s(%s)" % (a, c) for a, c in item["missing"]))
        if item["extra"]:
            print("    artiq (LLM-in oz addimi): " + ", ".join("%s(%s)" % (a, c) for a, c in item["extra"]))
        for action, column, error in item["failed"]:
            print("    ICRA XETASI: %s(%s): %s" % (action, column, error))
    good = [r for r in results if r["ok"]]
    print("\n" + "=" * 60)
    print("UMUMI STATISTIKA")
    print("=" * 60)
    print("Islenen fayl: %d / %d" % (len(good), len(results)))
    if not good:
        return 1
    passed = sum(r["llm_steps"] for r in good)
    produced = passed + sum(r["dropped"] for r in good)
    rate = passed / produced * 100 if produced else 0.0
    print("Validasiyadan kecme: %.1f%% (%d / %d LLM addimi)" % (rate, passed, produced))
    print("Orta vaxt: %.2f san" % (sum(r["seconds"] for r in good) / len(good)))
    print("Max vaxt: %.2f san" % max(r["seconds"] for r in good))
    print("Fallback sayi: %d / %d" % (sum(1 for r in good if r["used_fallback"]), len(good)))
    print("Orta addim sayi: %.1f" % (sum(r["steps"] for r in good) / len(good)))
    print("Orta tutum (qaydali plana gore): %.1f%%" % (sum(r["coverage"] for r in good) / len(good)))
    print("Mecburi elave olunan addim: %d" % sum(r["added"] for r in good))
    print("Icra xetasi olan addim: %d" % sum(len(r["failed"]) for r in good))
    return 0


if __name__ == "__main__":
    sys.exit(main())