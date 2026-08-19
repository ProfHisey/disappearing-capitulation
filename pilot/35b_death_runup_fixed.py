"""Stage 35b: DEATH RUN-UP, RECENTERED ON THE ACTUAL DEATH DATE (audit C3).

Audit round 4 C3: stage 35 centered the pre-death redemption profile on
end_p (spell end, capped at 2023Q3 by the AS panel) instead of the death
date - every 2024-26 death was mis-centered by up to 3 years and
recovered/as_missing spell-ends polluted the sample. This stage redoes it:
 - event clock = death_q from the death table;
 - one row per FUND (not per spell);
 - deaths restricted to those observable in the flows span (>=2020Q3 so a
   12-month pre-window exists; <=2026Q1);
 - merger vs liquidation split if the death table carries a flag.

Aggregates only; report: output/referee_35b_death_runup.txt
Builds the panel - run alone.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

DRV = Path(r"E:\Finance\data\sources\nport\derived")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["DEATH RUN-UP v2, DEATH-DATE CLOCK (stage 35b)", "=" * 60]

# ---- flows -> wficn-month (as stage 35 v2) ------------------------------
fl = pd.read_csv(DRV / "monthly_gross_flows.csv", low_memory=False)
link = pd.read_csv(DRV / "series_crsp_link_v2.csv", low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
fl = fl.merge(lw, on="series_id", how="inner")
fl["month"] = pd.PeriodIndex(fl["month"], freq="M")
g = (fl.groupby(["wficn", "month"])
       [["sales", "reinvestments", "redemptions", "net_assets"]]
       .sum(min_count=1).reset_index())
g = g[g["net_assets"] > 0]
g["rr"] = (g["redemptions"] / g["net_assets"]).clip(-1, 2)
g["wficn"] = g["wficn"].astype("int64")
gzi = g.set_index(["wficn", "month"])["rr"]
log.append(f"flow months: {len(g):,} ({g['wficn'].nunique():,} funds)")

# ---- death table --------------------------------------------------------
death = PL.get_death(log)
log.append(f"death table columns: {list(death.columns)}")
dq_col = next((c for c in ("death_q", "dq", "end_q")
               if c in death.columns), None)
assert dq_col, "no death-quarter column found - inspect columns above"
dd = death.copy()
if "died" in dd.columns:
    dd = dd[dd["died"] == 1]
dd["wficn"] = dd["wficn"].astype("int64")
dd["dq"] = pd.PeriodIndex(dd[dq_col], freq="Q")
dd = dd.drop_duplicates("wficn")
dd = dd[(dd["dq"] >= pd.Period("2020Q3", freq="Q"))
        & (dd["dq"] <= pd.Period("2026Q1", freq="Q"))]
dd = dd[dd["wficn"].isin(set(g["wficn"]))]
log.append(f"deaths in flows-observable window with flow data: "
           f"{len(dd):,} funds "
           f"(vs stage 35's mis-centered 1,647 SPELL rows)")
type_col = next((c for c in ("merged", "merger", "delist_cd", "dtype")
                 if c in dd.columns), None)

def profile(sub, lab):
    prof = {}
    for w, q in zip(sub["wficn"], sub["dq"]):
        dm = q.asfreq("M", how="end")
        for k in range(-12, 1):
            v = gzi.get((w, dm + k), np.nan)
            if pd.notna(v):
                prof.setdefault(k, []).append(float(v))
    line = [f"{lab}:"]
    for k in range(-12, 1, 2):
        vals = prof.get(k, [])
        line.append(f"t{k:+d} {np.median(vals):.2%}(n{len(vals)})"
                    if vals else f"t{k:+d} -")
    log.append("  " + "  ".join(line))

log.append("median monthly gross redemption rate, months to DEATH:")
profile(dd, "all deaths")
if type_col:
    log.append(f"split by {type_col}:")
    for val, sub in dd.groupby(type_col):
        if len(sub) >= 30:
            profile(sub, f"{type_col}={val} (n {len(sub)})")
pop_med = g["rr"].median()
log.append(f"population median rr for comparison: {pop_med:.2%}")
log.append("\nreading: this is the C3-corrected profile. A rising path "
           "ENDING ABOVE the population median = clients genuinely drain "
           "dying funds; a flat path near population levels = deaths are "
           "board decisions and stage 35's original 'doubling' was the "
           "spell-end artifact. Liquidations should show the drain more "
           "than mergers (merger dates reflect acquirer timing).")
log.append("\nSTAGE 35b DONE - aggregates only.")
P.write_report("referee_35b_death_runup.txt", log)
print("\n".join(log))
