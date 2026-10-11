import csv
import io
import os

import pandas as pd

ENCODINGS = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]
DELIMITERS = [",", ";", "\t", "|"]
CSV_EXTENSIONS = {".csv", ".tsv"}
EXCEL_EXTENSIONS = {".xlsx", ".xlsm", ".xls"}


def _decode(file_bytes):
    for encoding in ENCODINGS:
        try:
            text = file_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        return text.lstrip("\ufeff")
    return file_bytes.decode("utf-8", errors="replace").lstrip("\ufeff")


def _detect_delimiter(text):
    lines = [line for line in text.splitlines()[:50] if line.strip()]
    if not lines:
        return ","
    header = lines[0]
    counts = {d: header.count(d) for d in DELIMITERS}
    fallback = max(counts, key=counts.get)
    fallback = fallback if counts[fallback] > 0 else ","
    try:
        found = csv.Sniffer().sniff("\n".join(lines), delimiters="".join(DELIMITERS)).delimiter
    except csv.Error:
        return fallback
    if found in DELIMITERS and header.count(found) > 0:
        return found
    return fallback


def _read_csv(file_bytes):
    text = _decode(file_bytes)
    sep = _detect_delimiter(text)
    try:
        return pd.read_csv(io.StringIO(text), sep=sep)
    except (pd.errors.ParserError, UnicodeError):
        return pd.read_csv(io.StringIO(text), sep=sep, engine="python", on_bad_lines="skip")


def _read_excel(file_bytes, extension):
    engine = "openpyxl" if extension in {".xlsx", ".xlsm"} else None
    return pd.read_excel(io.BytesIO(file_bytes), engine=engine)


def _clean_columns(df):
    names = []
    seen = {}
    for position, column in enumerate(df.columns):
        name = str(column).replace("\ufeff", "").strip()
        if not name:
            name = f"column_{position + 1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    df.columns = names
    return df


def load_dataframe(file_bytes, filename):
    extension = os.path.splitext(str(filename or "").strip().lower())[1]
    if extension not in CSV_EXTENSIONS and extension not in EXCEL_EXTENSIONS:
        raise ValueError("Unsupported format")
    data = bytes(file_bytes or b"")
    try:
        if extension in CSV_EXTENSIONS:
            df = _read_csv(data)
        else:
            df = _read_excel(data, extension)
    except Exception as exc:
        raise ValueError("File is empty or could not be read") from exc
    return _clean_columns(df)