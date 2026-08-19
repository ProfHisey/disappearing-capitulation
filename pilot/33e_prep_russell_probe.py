"""Stage 33e_prep: PROBE RUSSELL WEIGHTS x N-PORT CUSIP MATCH (pre-33e).

Before writing the AS-extension compute (33e), fingerprint the two sides
of the join: (a) Russell idx_holdings_us.csv schema (columns, index codes,
date span, cusip format); (b) one N-PORT holdings quarter's CUSIPs against
Russell constituents at the matching date - 8 vs 9 char match rates.
Also prints the bench_min code vocabulary from the AS cache so 33e's
benchmark set aligns with the paper's.

Aggregates only; report: output/nport_33e_prep.txt
Light, panel-free - safe alongside 40b.
"""
from pathlib import Path

import pandas as pd

import pilot_lib as P

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["33e PREP: RUSSELL x N-PORT JOIN PROBE", "=" * 60]

# ---- (a) Russell schema -------------------------------------------------
rus_path = SRC / "Russell" / "idx_holdings_us.csv"
head = pd.read_csv(rus_path, nrows=200_000, low_memory=False)
head.columns = [c.lower() for c in head.columns]
log.append(f"Russell holdings columns ({len(head.columns)}): "
           + ", ".join(head.columns))
dcol = next((c for c in head.columns if "date" in c or c.endswith("dt")),
            None)
icol = next((c for c in head.columns
             if "index" in c or "idx" in c or c in ("r_code", "code")),
            None)
ccol = next((c for c in head.columns if "cusip" in c), None)
log.append(f"guessed columns: date={dcol}, index={icol}, cusip={ccol}")
if icol:
    log.append(f"index codes in sample: "
               f"{head[icol].value_counts().head(15).to_dict()}")
if ccol:
    ln = head[ccol].astype(str).str.len().value_counts().head(5)
    log.append(f"cusip length distribution (sample): {ln.to_dict()}")
# full-file date span from a thin read
thin = pd.read_csv(rus_path, usecols=[c for c in (dcol,) if c],
                   low_memory=False)
thin.columns = [c.lower() for c in thin.columns]
d = pd.to_datetime(thin[dcol], errors="coerce")
log.append(f"date span (full file): {d.min().date()} to {d.max().date()}"
           f", {thin[dcol].nunique():,} distinct dates")

# ---- bench code vocabulary from the AS cache ----------------------------
bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet",
                     columns=["bench_min"])
log.append(f"bench_min vocabulary (AS cache): "
           f"{bp['bench_min'].value_counts().head(15).to_dict()}")

# ---- (b) match test: one N-PORT quarter vs Russell at same date ---------
part = P.CACHE / "nport_holdings_parts" / "2024q2.parquet"
q = pd.read_parquet(part, columns=["cusip_filled", "CURRENCY_VALUE"])
q = q[q["cusip_filled"].notna()]
c8 = set(q["cusip_filled"].astype(str).str[:8])
log.append(f"\nN-PORT 2024q2: {len(q):,} cusip-filled equity rows, "
           f"{len(c8):,} distinct 8-char cusips")
if dcol and ccol:
    full = pd.read_csv(rus_path, usecols=[dcol, ccol, icol]
                       if icol else [dcol, ccol], low_memory=False)
    full.columns = [c.lower() for c in full.columns]
    fd = pd.to_datetime(full[dcol], errors="coerce")
    target = fd[fd <= "2024-06-30"].max()
    rus_day = full[fd == target]
    r8 = set(rus_day[ccol].astype(str).str[:8])
    inter = len(c8 & r8)
    log.append(f"Russell constituents on {target.date()}: "
               f"{len(rus_day):,} rows, {len(r8):,} distinct 8-char "
               f"cusips")
    log.append(f"  N-PORT∩Russell 8-char overlap: {inter:,} "
               f"({inter / len(c8):.1%} of N-PORT names)")
    # value-weighted match rate (what share of DOLLARS matches)
    q["c8"] = q["cusip_filled"].astype(str).str[:8]
    q["hit"] = q["c8"].isin(r8)
    vw = (q.loc[q["hit"], "CURRENCY_VALUE"].sum()
          / q["CURRENCY_VALUE"].sum())
    log.append(f"  value-weighted match: {vw:.1%} of N-PORT equity "
               f"dollars are in the Russell universe")

log.append("\nreading: >85% value-weighted match = the 33e join works on "
           "8-char cusips; below that, check 9-char/check-digit handling "
           "and the Russell universe date alignment before building 33e.")
log.append("STAGE 33e_prep DONE - aggregates only.")
P.write_report("nport_33e_prep.txt", log)
print("\n".join(log))
