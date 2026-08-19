"""Stage 35: WHO BREAKS FIRST, MODERN ERA, TRUE GROSS FLOWS (ranked R1).

The capitulation paper's client arm ran on IMPUTED NET flows. The N-PORT
panel (33b) gives actual monthly gross sales / reinvestments / redemptions
per SEC series, 2019-07 to 2026-05. This stage joins it to our fund panel
(link v2, unambiguous wficn only, series flows summed per wficn) and runs
the first descriptive pass:

 (a) coverage: matched fund-months vs the capitulation universe;
 (b) redemption-rate levels: stressed (rel4q<0) vs unstressed months;
 (c) event study: gross redemption rate in the 12 months before fund
     DEATH (post-2019 deaths - the modern exit that replaced surrender);
 (d) the few post-2019 capitulations: same profile (report n honestly);
 (e) gross churn (sales+redemptions)/NA - the client-conviction measure
     (backlog N2) - vs fund Active Share level.

Aggregates only; report: output/referee_35_gross_flows.txt
Heavier than 34 (builds the full panel) - run in its own window.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

DRV = Path(r"E:\Finance\data\sources\nport\derived")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["GROSS-FLOW CLIENT TEST, MODERN ERA (stage 35)", "=" * 60]

# ---- flows -> wficn-month ----------------------------------------------
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
g["rr"] = g["redemptions"] / g["net_assets"]          # gross redemption rate
g["churn"] = (g["sales"] + g["redemptions"]) / g["net_assets"]
g["netfl"] = (g["sales"] + g["reinvestments"]
              - g["redemptions"]) / g["net_assets"]
for c in ("rr", "churn", "netfl"):                    # winsorize tails
    g[c] = g[c].clip(lower=-1, upper=2)
log.append(f"wficn-months with gross flows: {len(g):,} "
           f"({g['wficn'].nunique():,} funds), "
           f"{g['month'].min()} to {g['month'].max()}")

# ---- panel + spells -----------------------------------------------------
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
univ = set(panel["wficn"].unique())
gz = g[g["wficn"].isin(univ)]
log.append(f"matched to capitulation universe: {gz['wficn'].nunique():,} "
           f"funds, {len(gz):,} fund-months "
           f"(universe {len(univ):,} funds; the rest of the universe is "
           f"pre-2019 history or unlinked)")

# quarterly stress state from the panel (v2 fix: proper period conversion
# + merge; v1's elementwise lookup produced no matching keys -> all NaN)
gz = gz.copy()
gz["wficn"] = gz["wficn"].astype("int64")
gz["quarter"] = pd.PeriodIndex(gz["month"], freq="M").asfreq("Q")
pcols = (panel[["wficn", "quarter", "rel4q", "as_min"]]
         .drop_duplicates(["wficn", "quarter"]).copy())
pcols["wficn"] = pcols["wficn"].astype("int64")
gz = gz.merge(pcols, on=["wficn", "quarter"], how="left")

# ---- (b) stressed vs unstressed ----------------------------------------
s = gz[gz["rel4q"].notna()]
for lab, m in [("stressed (rel4q<0)", s["rel4q"] < 0),
               ("unstressed        ", s["rel4q"] >= 0)]:
    d = s[m]
    log.append(f"  {lab}: median monthly gross redemption rate "
               f"{d['rr'].median():.2%}, mean {d['rr'].mean():.2%}, "
               f"median net flow {d['netfl'].median():+.2%} "
               f"(n {len(d):,})")

# ---- (c) run-up to death ------------------------------------------------
deaths = sp[(sp["died"] == 1)]
deaths = deaths[deaths["end_p"] >= pd.Period("2019Q3", freq="Q")]
log.append(f"post-2019 deaths in spells: {len(deaths):,}")
prof = {k: [] for k in range(-12, 1)}
gzi = gz.set_index(["wficn", "month"])["rr"]
for _, srow in deaths.iterrows():
    dq = srow["end_p"].asfreq("M", how="end")
    for k in range(-12, 1):
        v = gzi.get((srow["wficn"], dq + k), np.nan)
        if pd.notna(v):
            prof[k].append(float(v))
line = ["month-to-death: median gross redemption rate"]
for k in range(-12, 1, 2):
    vals = prof[k]
    line.append(f"t{k:+d}: {np.median(vals):.2%}(n{len(vals)})"
                if vals else f"t{k:+d}: -")
log.append("  " + "  ".join(line))
log.append("  reading: a rising profile = clients drain the fund before "
           "the board pulls the plug; flat-then-cliff = deaths are board "
           "decisions, not client verdicts.")

# ---- (d) post-2019 capitulations ---------------------------------------
caps = sp[(sp["capitulated"] == True)].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps = caps[caps["cq"] >= pd.Period("2019Q3", freq="Q")]
log.append(f"post-2019 capitulations: {len(caps):,} (small by "
           f"construction - the paper's whole point)")
if len(caps):
    pre, post = [], []
    for _, srow in caps.iterrows():
        cm = srow["cq"].asfreq("M", how="end")
        pre += [float(v) for k in range(-6, 0)
                if pd.notna(v := gzi.get((srow["wficn"], cm + k), np.nan))]
        post += [float(v) for k in range(1, 7)
                 if pd.notna(v := gzi.get((srow["wficn"], cm + k), np.nan))]
    if pre and post:
        log.append(f"  median rr 6m before crossing {np.median(pre):.2%} "
                   f"vs 6m after {np.median(post):.2%} "
                   f"(n {len(pre)}/{len(post)} fund-months)")

# ---- (e) churn vs conviction -------------------------------------------
c = gz[gz["as_min"].notna()]
c_band = pd.cut(c["as_min"], [0, 0.6, 0.7, 0.8, 0.9, 1.01],
                labels=["<60", "60-70", "70-80", "80-90", "90+"])
tab = c.groupby(c_band)["churn"].agg(["median", "count"])
log.append("median monthly gross churn (sales+redemptions)/NA by Active "
           "Share band:")
for i, r in tab.iterrows():
    log.append(f"    AS {i}: {r['median']:.2%}  (n {int(r['count']):,})")
log.append("  reading: if high-AS funds carry higher churn, conviction "
           "funds live with flightier clients - the N2 hypothesis.")

log.append("\nSTAGE 35 DONE - aggregates only. Next iteration: monthly "
           "hazard with rr as covariate; N-SAR extends all of this to "
           "1994-2018 when Kellogg delivers.")
P.write_report("referee_35_gross_flows.txt", log)
print("\n".join(log))
