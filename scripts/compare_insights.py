import argparse
import io
import json
import os
import re
import sys
import time
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.agent.insights import _numbers_ok, collect_facts, write_insights
from backend.agent.llm_client import llm_available

from backend.agent.planner import build_plan

try:
    from backend.executor import execute_plan
    from backend.loader import load_dataframe
    from backend.profiler import profile_dataframe
except Exception:
    from mini_pipeline import execute_plan, load_dataframe, profile_dataframe
    print("DIQQET: backend.loader/profiler/executor tapilmadi, scripts/mini_pipeline.py istifade olunur\n")
PIPELINE = True

EXTENSIONS = (".csv", ".tsv", ".txt", ".xlsx", ".xls")


def cases_from_fixture(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [(
        os.path.basename(path),
        data.get("profile_before") or {},
        data.get("profile_after") or {},
        data.get("log") or [],
        None,
    )]


def cases_from_samples(folder, limit):
    cases = []
    names = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(EXTENSIONS) and not f.startswith("~$")
    )
    for name in names[:limit]:
        try:
            with open(os.path.join(folder, name), "rb") as handle:
                data = handle.read()
            df = load_dataframe(data, name)
            before = profile_dataframe(df)
            with redirect_stdout(io.StringIO()):
                plan, _ = build_plan(before)
            df_clean, log = execute_plan(df, plan)
            after = profile_dataframe(df_clean)
            cases.append((name, before, after, log, df_clean))
        except Exception as error:
            print("%s: pipeline xetasi: %s" % (name, error))
    return cases


def run_lang(case, lang):
    _, before, after, log, df_clean = case
    buffer = io.StringIO()
    start = time.perf_counter()
    with redirect_stdout(buffer):
        text = write_insights(before, after, log, df_clean, lang)
    seconds = time.perf_counter() - start
    facts = collect_facts(before, after, log, df_clean)
    notes = [l.strip() for l in buffer.getvalue().splitlines() if l.startswith("[insights]")]
    return text, {
        "note": notes[-1] if notes else "",
        "source": "LLM" if "LLM metni qebul edildi" in buffer.getvalue() else ("qismen" if "qismen qebul" in buffer.getvalue() else "sablon"),
        "seconds": seconds,
        "words": len(re.findall(r"\w+", text)),
        "numbers_ok": _numbers_ok(text, facts),
        "headings_ok": text.count("### ") == 3,
        "empty": not text.strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=os.path.join(ROOT, "sample_data"))
    parser.add_argument("--fixture", default=None)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cases = []
    if args.fixture:
        cases = cases_from_fixture(args.fixture)
    elif PIPELINE and os.path.isdir(args.folder):
        cases = cases_from_samples(args.folder, args.limit)
    if not cases:
        default_fixture = os.path.join(ROOT, "fixtures", "sample_response.json")
        if os.path.isfile(default_fixture):
            cases = cases_from_fixture(default_fixture)
    if not cases:
        print("Test ucun data tapilmadi (sample_data ve ya fixtures/sample_response.json).")
        return 1

    print("Ollama: " + ("aciqdir" if llm_available() else "SONDURULUB (hamisi sablon olacaq)"))
    totals = {"az": [], "en": []}
    for case in cases:
        print("\n" + "=" * 60)
        print("FAYL: " + case[0])
        print("=" * 60)
        for lang in ("az", "en"):
            text, stats = run_lang(case, lang)
            totals[lang].append(stats)
            print(
                "\n[%s] menbe=%s vaxt=%.2fs soz=%d reqemler_dogru=%s basliqlar=%s"
                % (lang, stats["source"], stats["seconds"], stats["words"], stats["numbers_ok"], stats["headings_ok"])
            )
            if stats["note"]:
                print("    " + stats["note"])
            if not args.quiet:
                print(text)

    print("\n" + "=" * 60)
    print("UMUMI MUQAYISE")
    print("=" * 60)
    for lang in ("az", "en"):
        items = totals[lang]
        if not items:
            continue
        llm_count = sum(1 for s in items if s["source"] == "LLM")
        part_count = sum(1 for s in items if s["source"] == "qismen")
        print(
            "%s: LLM tam qebul %d/%d, qismen %d, orta vaxt %.2fs, orta soz %.0f, bosh cavab %d"
            % (
                lang,
                llm_count,
                len(items),
                part_count,
                sum(s["seconds"] for s in items) / len(items),
                sum(s["words"] for s in items) / len(items),
                sum(1 for s in items if s["empty"]),
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())