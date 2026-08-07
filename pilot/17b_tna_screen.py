"""Stage 17b: TNA-SCREEN REFINEMENT — follow-up to battery I, section (f).

Battery I showed the TNA<$1M month-drop removes 44-48% of fund-months before
1990 but ~1% after 2000. That is almost certainly not "half of early funds
were micro funds": early CRSP reports TNA quarterly or annually, so most
monthly TNA values are simply MISSING, and the v2 hygiene rule treats
missing as zero and drops the month. The screen's bite is era-varying for a
mechanical reason — exactly the referee's critique 23.

Refined rule tested here: drop a month only when TNA is OBSERVED and under
$1M (missing TNA passes through; the |ret|>200% rule is unchanged). Rebuild
the fund-month series and the panel under that rule, re-extract spells, and
compare the headline diagnostics to V0:

  - if the era decline, the depth sign flip, and the duration gradient are
    unchanged, the early-era bite is a reporting artifact with no bearing on
    the findings, and the paper reports the refined rule as primary with the
    old rule as robustness;
  - if the 1980-94 row moves materially, the early era was being estimated
    on a large-fund subsample and the refined rule becomes the primary spec.

First run rebuilds two caches (a few minutes); reruns are fast.
Output: output/referee_17b_tna.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["TNA-SCREEN REFINEMENT (battery I follow-up)", "=" * 60]

death = PL.get_death(log)

# ------------------------- fund-month series under the refined rule ----
pq = P.CACHE / "fund_month_v3_tnafix.parquet"
if pq.exists():
    fm = pd.read_parquet(pq)
else:
    m1 = PL.get_mflink1()
    ret = P.load_monthly_returns(log).merge(m1, on="crsp_fundno", how="inner")
    ret = ret.sort_values(["crsp_fundno", "caldt"])
    ret["w"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
    ret["w"] = ret["w"].fillna(ret["mtna"]).clip(lower=0)
    ret = ret.dropna(subset=["mret"])
    ret["wr"] = ret["w"] * ret["mret"]
    fm = (ret.groupby(["wficn", "caldt"])
             .agg(wr=("wr", "sum"), w=("w", "sum"), tna=("mtna", "sum"),
                  n_tna=("mtna", "count")).reset_index())
    fm["fret"] = np.where(fm["w"] > 0, fm["wr"] / fm["w"], np.nan)
    fm = fm.dropna(subset=["fret"])
    n0 = len(fm)
    bad_ret = fm["fret"].abs() > 2.0
    bad_tna = (fm["n_tna"] > 0) & (fm["tna"] < 1.0)   # observed AND tiny
    fm = fm[~bad_ret & ~bad_tna]
    log.append(f"  refined hygiene: dropped {n0 - len(fm):,} fund-months "
               f"(v2 rule dropped far more early-sample months by treating "
               f"missing TNA as zero)")
    fm = fm[["wficn", "caldt", "fret", "tna"]]
    fm.to_parquet(pq, index=False)

# ----------------------------------- panel rebuild under refined rule ----
fm["quarter"] = fm["caldt"].dt.to_period("Q")
fq = (fm.assign(g=lambda d: 1 + d["fret"]).groupby(["wficn", "quarter"])
        .agg(qret=("g", lambda x: x.prod() - 1),
             nm=("g", "size")).reset_index())
fq = fq[fq["nm"] == 3].drop(columns="nm")

asp = pd.read_parquet(P.CACHE / "as_panel.parquet").dropna(subset=["wficn"])
asp["wficn"] = asp["wficn"].astype("int64")
asp["quarter"] = pd.to_datetime(asp["month"]).dt.to_period("Q")
asp = (asp.sort_values(["wficn", "quarter", "total_assets"])
          .drop_duplicates(["wficn", "quarter"], keep="last"))
fl = pd.read_parquet(P.CACHE / "flags.parquet")
asp = asp.merge(fl, on="wficn", how="left")
asp = asp[asp["passive"] != True]  # noqa: E712

bq = PL.get_real_bench_q(log)
flows = PL.get_retail_flows(log)
flows["quarter"] = pd.PeriodIndex(flows["quarter"], freq="Q")
asp["bcode"] = (asp["bench_min"].astype(str).str.upper()
                .replace(PL.BENCH_APPROX))
pan = (asp.merge(fq, on=["wficn", "quarter"], how="inner")
          .merge(bq, on=["quarter", "bcode"], how="left")
          .merge(flows, on=["wficn", "quarter"], how="left"))
pan["bench_qret"] = pan["bret"]
pan = pan.dropna(subset=["as_min", "qret", "bench_qret"])
pan = R.retrail(pan[["wficn", "quarter", "as_min", "qret",
                     "bench_qret", "flowq"]])
log.append(f"refined panel: {len(pan):,} fund-quarters, "
           f"{pan['wficn'].nunique():,} funds "
           f"(V0 for comparison: 18,094 spells from the v2 panel)")

# ----------------------------------------------- headline diagnostics ----
sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
R.summarize(sp, log, "REFINED TNA RULE (missing TNA passes)")
pf = {w: g.set_index("quarter") for w, g in pan.groupby("wficn")}
dt = R.build_dt(sp, pf)
R.slim_fit(dt, R.SLIM, "event", log, "capitulation")
R.slim_fit(dt, R.SLIM, "event_die", log, "death")

log.append("""
V0 comparison values (battery I):
  overall: 18,094 spells | cap 3.83% | died 13.46%
  1980-94 cap 6.61% | 1995-2009 cap 5.04% | 2010-23 cap 2.46%
  capitulation: dur_5p HR 2.60, depth HR 117.3, era_1023 HR 0.37
  death:        dur_5p HR 1.03, depth HR 0.42,  era_1023 HR 0.75
Reading: the row that matters most is 1980-94 — that is where the old rule
was silently dropping ~half the fund-months.""")

log.append("\n17b DONE - aggregates only.")
P.write_report("referee_17b_tna.txt", log)
print("\n".join(log))
