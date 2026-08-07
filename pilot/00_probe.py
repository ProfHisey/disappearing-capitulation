"""Schema probe: verifies every input file exists and reports columns, dtypes,
and date formats WITHOUT exporting any fund-level data.
Run this FIRST and send output/schema_report.txt to Claude before running 01-03.
"""
import re

import pandas as pd

import pilot_lib as P

lines = ["SCHEMA PROBE", "=" * 60]

FILES = {
    "CRSP Monthly Returns": P.F_MONTHLY,
    "CRSP Fund Summary": P.F_SUMMARY,
    "CRSP Fund-Portfolio Map": P.F_MAP,
    "ND Active Share (TR 1979-2019)": P.F_ND_TR,
    "ND Active Share (CRSP 2020-2023)": P.F_ND_CRSP,
    "Petajisto activeshare": P.F_PET,
    "CPZ factors monthly": P.F_CPZ_M,
}

def classify(v: str) -> str:
    v = str(v).strip()
    for pat, name in [(r"\d{4}-\d{2}-\d{2}", "ISO date"), (r"\d{8}", "YYYYMMDD"),
                      (r"\d{4}m\d{1,2}", "statamonth 'YYYYmM'"),
                      (r"\d{6}", "YYYYMM"), (r"-?\d{1,4}", "small int (Stata elapsed?)")]:
        if re.fullmatch(pat, v):
            return name
    return "other"

for label, path in FILES.items():
    lines.append(f"\n--- {label} ---")
    if path is None or not path.exists():
        lines.append(f"  MISSING: {path}")
        continue
    lines.append(f"  path: {path.name}  ({path.stat().st_size/1e6:.1f} MB)")
    try:
        df = P.norm_cols(pd.read_csv(path, nrows=5000, low_memory=False))
    except Exception as e:  # noqa: BLE001
        lines.append(f"  READ ERROR: {e}")
        continue
    lines.append(f"  columns ({len(df.columns)}): {', '.join(df.columns)}")
    # date-format classification for key columns (date VALUES only, no fund data)
    for c in df.columns:
        if c in ("ym", "rdate", "caldt", "begdt", "enddt", "date", "month", "fdate") \
                or c.endswith("_dt"):
            v = df[c].dropna()
            if len(v):
                lines.append(f"  date col {c!r}: format looks like "
                             f"{classify(v.iloc[0])} (e.g. {v.iloc[0]})")
    ascols = [c for c in df.columns if c.startswith("as_")]
    if ascols:
        mx = pd.to_numeric(df[ascols[0]], errors="coerce").max()
        lines.append(f"  as_* columns ({len(ascols)}): {', '.join(ascols)}")
        lines.append(f"  units check ({ascols[0]}): max in sample = {mx} "
                     f"-> {'percent 0-100' if mx and mx > 1.5 else 'fraction 0-1'}")
    for c in ("activeshare", "wficn", "fundno", "crsp_fundno", "crsp_portno"):
        if c in df.columns:
            lines.append(f"  key col present: {c} (dtype {df[c].dtype})")

lines.append("\nPROBE DONE - send this file to Claude before running 01-03.")
P.write_report("schema_report.txt", lines)
