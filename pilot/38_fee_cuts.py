"""Stage 38: CAPITULATION ON PRICE - fee cuts as a surrender margin (R4).

If managers no longer surrender with the PORTFOLIO (Paper 1), maybe they
surrender on PRICE. From CRSP Fund Summary expense ratios:

 (a) fee trajectory in event time around capitulation crossings (-8q..+8q);
 (b) among STRESSED fund-years (rel4q<0 for 2+ consecutive quarters),
     P(expense ratio cut >= 10bps within the next 8 quarters), by era -
     did price-surrender RISE as portfolio-surrender died?
 (c) the interaction: do funds that resist portfolio capitulation cut
     fees more (substitutes) or less (bundled conviction)?

Aggregates only; report: output/referee_38_fee_cuts.txt
Streams Fund Summary (1.5GB) then builds the panel - run alone.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["FEE CUTS AS SURRENDER (stage 38)", "=" * 60]

# ---- expense ratios: share class -> wficn-quarter -----------------------
fs_path = SRC / "crsp_mf" / "Fund Summary.csv"
want = ("crsp_fundno", "caldt", "exp_ratio", "mgmt_fee")
parts = []
for ch in pd.read_csv(fs_path, chunksize=2_000_000, low_memory=False,
                      encoding="latin-1"):
    ch.columns = [c.lower() for c in ch.columns]
    cols = [c for c in want if c in ch.columns]
    if "exp_ratio" not in cols:
        raise SystemExit(f"exp_ratio not in Fund Summary columns; "
                         f"saw {list(ch.columns)[:20]}")
    ch = ch[cols]
    ch = ch[ch["exp_ratio"].notna() & (ch["exp_ratio"] > 0)]
    parts.append(ch)
fees = pd.concat(parts, ignore_index=True)
fees["quarter"] = pd.to_datetime(fees["caldt"]).dt.to_period("Q")
m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                 encoding="latin-1")
m1.columns = [c.lower() for c in m1.columns]
fees = fees.merge(m1[["crsp_fundno", "wficn"]].drop_duplicates(),
                  on="crsp_fundno", how="inner")
fw = (fees.groupby(["wficn", "quarter"])["exp_ratio"]
      .mean().rename("er").reset_index())   # equal-wt across classes v1
fw["wficn"] = fw["wficn"].astype("int64")
log.append(f"expense panel: {len(fw):,} wficn-quarters, "
           f"{fw['wficn'].nunique():,} funds, "
           f"{fw['quarter'].min()} to {fw['quarter'].max()}")
FW = {w: g.set_index("quarter")["er"] for w, g in fw.groupby("wficn")}

# ---- panel + spells -----------------------------------------------------
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

# ---- (a) event-time fee path around crossings ---------------------------
caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
prof = {}
for _, s in caps.iterrows():
    f = FW.get(int(s["wficn"]))
    if f is None:
        continue
    base = f.get(s["cq"] - 8, np.nan)
    if pd.isna(base) or base <= 0:
        continue
    for k in range(-8, 9):
        v = f.get(s["cq"] + k, np.nan)
        if pd.notna(v):
            prof.setdefault(k, []).append(float(v - base) * 1e4)  # bps
log.append("median expense-ratio change vs t-8, around capitulation "
           "(bps):")
line = []
for k in range(-8, 9, 2):
    vals = prof.get(k, [])
    line.append(f"t{k:+d}: {np.median(vals):+.0f}(n{len(vals)})"
                if vals else f"t{k:+d}: -")
log.append("  " + "  ".join(line))

# ---- (b) stressed fund-quarters: P(cut >= 10bps within 8q), by era ------
pan = panel.sort_values(["wficn", "quarter"]).copy()
pan["stress2"] = ((pan["rel4q"] < 0)
                  & (pan.groupby("wficn")["rel4q"].shift() < 0))
stress = pan[pan["stress2"] & pan["rel4q"].notna()]
# one obs per fund-year to cut serial correlation
stress = stress[stress["quarter"].dt.quarter == 4]
rows = []
for _, r0 in stress.iterrows():
    f = FW.get(int(r0["wficn"]))
    if f is None:
        continue
    now = f.get(r0["quarter"], np.nan)
    if pd.isna(now):
        continue
    fut = [f.get(r0["quarter"] + k, np.nan) for k in range(1, 9)]
    fut = [x for x in fut if pd.notna(x)]
    if not fut:
        continue
    rows.append((r0["quarter"].year, (min(fut) - now) * 1e4))
cut = pd.DataFrame(rows, columns=["year", "dbps"])
cut["era3"] = pd.cut(cut["year"], [0, 1994, 2009, 9999],
                     labels=["1980-94", "1995-2009", "2010-23"])
log.append("stressed fund-years: P(fee cut >= 10bps within 8q):")
for era in ["1980-94", "1995-2009", "2010-23"]:
    d = cut[cut["era3"] == era]
    if len(d):
        log.append(f"    {era}: {(d['dbps'] <= -10).mean():6.1%} "
                   f"(n {len(d):,}; median change {d['dbps'].median():+.0f}"
                   f"bps)")
log.append("  reading: rising by era = price-surrender replacing "
           "portfolio-surrender (M1/M2 prediction). NOTE fees fell "
           "industry-wide - compare against the same statistic for "
           "UNSTRESSED fund-years before interpreting:")
un = pan[(pan["rel4q"] >= 0) & (pan["quarter"].dt.quarter == 4)]
rows = []
for _, r0 in un.iterrows():
    f = FW.get(int(r0["wficn"]))
    if f is None:
        continue
    now = f.get(r0["quarter"], np.nan)
    if pd.isna(now):
        continue
    fut = [f.get(r0["quarter"] + k, np.nan) for k in range(1, 9)]
    fut = [x for x in fut if pd.notna(x)]
    if not fut:
        continue
    rows.append((r0["quarter"].year, (min(fut) - now) * 1e4))
cun = pd.DataFrame(rows, columns=["year", "dbps"])
cun["era3"] = pd.cut(cun["year"], [0, 1994, 2009, 9999],
                     labels=["1980-94", "1995-2009", "2010-23"])
for era in ["1980-94", "1995-2009", "2010-23"]:
    d = cun[cun["era3"] == era]
    if len(d):
        log.append(f"    unstressed {era}: {(d['dbps'] <= -10).mean():6.1%}"
                   f" (n {len(d):,})")

# ---- (c) resisters vs capitulators --------------------------------------
res = sp[(sp["capitulated"] == False) & (sp["end_dur"] >= 12)]
def cut_within(w, q0):
    f = FW.get(int(w))
    if f is None:
        return np.nan
    now = f.get(q0, np.nan)
    fut = [f.get(q0 + k, np.nan) for k in range(1, 9)]
    fut = [x for x in fut if pd.notna(x)]
    if pd.isna(now) or not fut:
        return np.nan
    return float((min(fut) - now) * 1e4 <= -10)
cc = pd.Series([cut_within(w, q) for w, q in
                zip(caps["wficn"], caps["cq"])]).dropna()
rr_ = pd.Series([cut_within(w, s + 8) for w, s in
                 zip(res["wficn"], res["start_p"])]).dropna()
log.append(f"P(fee cut >=10bps within 8q): at capitulation "
           f"{cc.mean():.1%} (n {len(cc):,}) vs long-resisting spells at "
           f"t+8 {rr_.mean():.1%} (n {len(rr_):,})")

log.append("\nSTAGE 38 DONE - aggregates only. Caveats logged: equal-wt "
           "share-class fees v1 (TNA-wt next); industry-wide fee decline "
           "means the stressed-vs-unstressed DIFFERENCE is the statistic, "
           "not the level.")
P.write_report("referee_38_fee_cuts.txt", log)
print("\n".join(log))
