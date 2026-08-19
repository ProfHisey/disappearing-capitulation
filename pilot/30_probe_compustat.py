"""Stage 30: PROBE THE COMPUSTAT PULL (Round-2 data acquisition, Pull 1).

Identifies each file in E:\\Finance\\data\\sources\\compustat by CONTENT (not
filename), so a funda/fundq mix-up is caught here. Verifies schema, screening
filters (consol C, datafmt STD, indfmt INDL+FS, curcd USD+CAD), datadate span,
row and gvkey counts. Prints aggregates only; nothing licensed leaves disk.

Output: output/probe_30_compustat.txt
"""
from pathlib import Path

import pandas as pd

SRC = Path(r"E:\Finance\data\sources\compustat")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["COMPUSTAT PROBE (stage 30)", "=" * 70]


def count_lines(path):
    """Fast line count by streaming raw bytes (multi-GB safe)."""
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):  # 16 MB chunks
            n += chunk.count(b"\n")
    return n - 1  # minus header


def probe(path):
    log.append(f"\n{'-' * 70}\nFILE: {path.name}  "
               f"({path.stat().st_size / 1e9:.2f} GB)")
    head = pd.read_csv(path, nrows=5000, low_memory=False)
    head.columns = [c.lower() for c in head.columns]
    cols = set(head.columns)

    # --- identify by content ---
    if {"linktype", "linkprim"} & cols or {"lpermno", "linkdt"} & cols:
        kind = "CCM LINKING TABLE"
    elif {"fqtr", "datacqtr", "atq"} & cols:
        kind = "FUNDAMENTALS QUARTERLY (fundq)"
    elif "fyear" in cols and "fqtr" not in cols:
        kind = "FUNDAMENTALS ANNUAL (funda)"
    else:
        kind = "UNRECOGNIZED - check manually"
    log.append(f"  identified as: {kind}")
    log.append(f"  columns: {len(cols)}")

    if "LINKING" in kind:
        full = pd.read_csv(path, low_memory=False)
        full.columns = [c.lower() for c in full.columns]
        log.append(f"  rows: {len(full):,}   "
                   f"unique gvkey: {full['gvkey'].nunique():,}")
        for c in ("linktype", "linkprim"):
            if c in full.columns:
                log.append(f"  {c}: {full[c].value_counts().to_dict()}")
        pn = next((c for c in ("lpermno", "permno") if c in full.columns),
                  None)
        if pn:
            log.append(f"  unique {pn}: {full[pn].nunique():,}")
        return

    # --- fundamentals files: stream for counts and spans ---
    n_rows = count_lines(path)
    log.append(f"  rows (streamed count): {n_rows:,}")

    want = [c for c in ("gvkey", "datadate", "indfmt", "datafmt", "consol",
                        "curcd", "curcdq", "fyear", "fyearq", "fqtr")
            if c in cols]
    gv, dmin, dmax = set(), None, None
    screen = {c: {} for c in ("indfmt", "datafmt", "consol", "curcd",
                              "curcdq")}
    for ch in pd.read_csv(path, usecols=lambda c: c.lower() in want,
                          chunksize=2_000_000, low_memory=False):
        ch.columns = [c.lower() for c in ch.columns]
        gv.update(ch["gvkey"].unique())
        d = pd.to_datetime(ch["datadate"], errors="coerce")
        lo, hi = d.min(), d.max()
        dmin = lo if dmin is None or lo < dmin else dmin
        dmax = hi if dmax is None or hi > dmax else dmax
        for c in screen:
            if c in ch.columns:
                for k, v in ch[c].value_counts().items():
                    screen[c][k] = screen[c].get(k, 0) + int(v)
    log.append(f"  unique gvkey: {len(gv):,}")
    log.append(f"  datadate span: {dmin.date()} to {dmax.date()}")
    for c, d in screen.items():
        if d:
            log.append(f"  {c}: "
                       f"{dict(sorted(d.items(), key=lambda x: -x[1]))}")


files = sorted(SRC.glob("*.csv")) + sorted(SRC.glob("*.csv.gz"))
if not files:
    log.append(f"No csv files found in {SRC}")
for p in files:
    probe(p)

log.append("\nPROBE DONE. If a file's identity doesn't match its name, "
           "rename it, rerun, and record the final output as the manifest.")
(OUT / "probe_30_compustat.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
