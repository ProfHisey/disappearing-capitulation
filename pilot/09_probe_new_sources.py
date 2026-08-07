"""Probe the newly-landed library folders (russell, crsp_sp500, djia, mflinks,
re-pulled loads): columns, date formats, sample spans. Metadata only - no data
rows leave the machine. Send output/new_sources_report.txt to Claude.
"""
import re

import pandas as pd

import pilot_lib as P

lines = ["NEW SOURCES PROBE", "=" * 60]

FOLDERS = ["russell", "crsp_sp500", "djia", "mflinks", "nsar",
           "pastor-stambaugh", "thomson_s12"]
EXTRA_FILES = [P.CRSP_DIR / "Front Loads.csv", P.CRSP_DIR / "Rear Loads.csv"]

def classify(v):
    v = str(v).strip()
    for pat, name in [(r"\d{4}-\d{2}-\d{2}", "ISO"), (r"\d{8}", "YYYYMMDD"),
                      (r"\d{4}m\d{1,2}", "statamonth"), (r"\d{6}", "YYYYMM"),
                      (r"\d{1,2}/\d{1,2}/\d{4}", "M/D/YYYY"),
                      (r"-?\d{1,5}", "small int")]:
        if re.fullmatch(pat, v):
            return name
    return "other"

def probe(path):
    lines.append(f"\n--- {path.name}  ({path.stat().st_size/1e6:.1f} MB) ---")
    try:
        df = P.norm_cols(pd.read_csv(path, nrows=5000, low_memory=False,
                                     encoding="latin-1"))
    except Exception as e:  # noqa: BLE001
        lines.append(f"  READ ERROR: {e}")
        return
    lines.append(f"  columns ({len(df.columns)}): {', '.join(map(str, df.columns))}")
    for c in df.columns:
        cl = str(c).lower()
        if ("dt" in cl or "date" in cl or cl in ("ym", "month", "caldt")):
            v = df[c].dropna()
            if len(v):
                lines.append(f"  date col {c!r}: looks like {classify(v.iloc[0])} "
                             f"(e.g. {v.iloc[0]})")
    # numeric ranges for return-like columns (aggregate stats only)
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("ret", "return", "level", "indx", "weight",
                                 "sales", "redem", "repurch", "reinv")):
            v = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(v):
                lines.append(f"  numeric col {c!r}: sample min {v.min():.4g}, "
                             f"max {v.max():.4g} (percent-vs-decimal check)")

for folder in FOLDERS:
    d = P.SOURCES / folder
    lines.append(f"\n=== {folder}/ ===")
    if not d.exists():
        lines.append("  (folder not found)")
        continue
    files = sorted(list(d.glob("*.csv")) + list(d.glob("*.csv.gz"))
                   + list(d.glob("*.zip")))
    if not files:
        lines.append("  (no csv/zip files found)")
    for f in files:
        if f.suffix == ".zip":
            lines.append(f"\n--- {f.name} ({f.stat().st_size/1e6:.1f} MB) --- "
                         "zip: unzip it and re-run this probe")
            continue
        probe(f)

lines.append("\n=== re-pulled loads (crsp_mf/) ===")
for f in EXTRA_FILES:
    if f.exists():
        probe(f)
    else:
        lines.append(f"  missing: {f}")

lines.append("\nPROBE DONE - metadata only; send this file to Claude.")
P.write_report("new_sources_report.txt", lines)
print("\n".join(lines))
