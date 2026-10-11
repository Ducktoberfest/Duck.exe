import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.loader import load_dataframe
from backend.profiler import profile_dataframe, to_jsonable


def show(filename, profile):
    print(f"\n{filename}: rows={profile['rows']} columns={profile['columns']} "
          f"duplicate_rows={profile['duplicate_rows']} total_null_cells={profile['total_null_cells']}")
    for col in profile["columns_info"]:
        print(f"  {col['name']:<15} {col['inferred_type']:<12} nulls={col['null_count']:<4} "
              f"unique={col['unique']:<5} outliers={col['outlier_count']:<4} ws={col['whitespace_issues']}")


def check_samples():
    folder = ROOT / "sample_data"
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".csv", ".xlsx", ".xls"})
    assert files, "sample_data is empty"
    for path in files:
        df = load_dataframe(path.read_bytes(), path.name)
        profile = profile_dataframe(df)
        json.dumps(profile, allow_nan=False)
        assert profile["rows"] == len(df)
        assert len(profile["columns_info"]) == df.shape[1]
        show(path.name, profile)


def check_loader_edges():
    semicolon = "ad;yaş;şəhər\nAnar;25;Bakı\nAysel;30;Gəncə\n".encode("utf-8")
    df = load_dataframe(semicolon, "a.csv")
    assert list(df.columns) == ["ad", "yaş", "şəhər"] and len(df) == 2

    bom = "\ufeff id , name \n1,x\n2,y\n".encode("utf-8")
    df = load_dataframe(bom, "b.csv")
    assert list(df.columns) == ["id", "name"]

    cp = "ad,şəhər\nАнар,Баку\n".encode("cp1251", errors="replace")
    assert len(load_dataframe(cp, "c.csv")) == 1

    tab = "a\tb\n1\t2\n".encode("utf-8")
    assert list(load_dataframe(tab, "d.tsv").columns) == ["a", "b"]

    dup = "a,a,b\n1,2,3\n".encode("utf-8")
    dup_cols = list(load_dataframe(dup, "e.csv").columns)
    assert len(dup_cols) == 3 and len(set(dup_cols)) == 3

    for bad_name, payload in [("f.json", b"{}"), ("g.csv", b"")]:
        try:
            load_dataframe(payload, bad_name)
        except ValueError as exc:
            print("ValueError:", exc)
        else:
            raise AssertionError(bad_name)


def check_profiler_edges():
    empty = profile_dataframe(pd.DataFrame())
    assert empty["rows"] == 0 and empty["columns_info"] == []

    allnull = profile_dataframe(pd.DataFrame({"x": [None, "", "n/a", " ? "]}))
    assert allnull["columns_info"][0]["null_count"] == 4
    assert allnull["total_null_cells"] == 4

    df = pd.DataFrame({
        "flag": [True, False, True, True],
        "mixed": [1, "a", None, [1, 2]],
        "when": pd.to_datetime(["2025-01-01", None, "2025-02-01", "2025-03-01"]),
        "num": [1.0, 2.0, 3.0, 400.0],
    })
    profile = profile_dataframe(df)
    kinds = {c["name"]: c["inferred_type"] for c in profile["columns_info"]}
    assert kinds["flag"] == "boolean" and kinds["when"] == "datetime" and kinds["num"] == "numeric"
    json.dumps(profile, allow_nan=False)

    payload = {"a": np.int64(3), "b": np.float64("nan"), "c": np.array([1, 2]), "d": pd.Timestamp("2025-01-01"), "e": pd.NaT}
    json.dumps(to_jsonable(payload), allow_nan=False)
    assert to_jsonable(payload)["b"] is None

    assert profile_dataframe(None)["rows"] == 0


if __name__ == "__main__":
    check_samples()
    check_loader_edges()
    check_profiler_edges()
    print("\nALL CHECKS PASSED")