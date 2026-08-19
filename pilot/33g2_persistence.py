"""Stage 33g2: DO THE 127 STAY DOWN? (persistence check on the surge)

One-page addendum to 33g: for each extension-era crosser, where is AS at
the LAST observation - still below 0.70 (persistent fold), or back above
(bounce)? High bounce share = instrument-volatility worry returns.

Aggregates only; report: output/nport_33g2_persistence.txt
Builds the panel - run alone.
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

log = ["SURGE PERSISTENCE CHECK (stage 33g2)", "=" * 60]

ext = pd.read_parquet(P.CACHE / "nport_as_extension.parquet")
link = pd.read_csv(SRC / "nport" / "derived" / "series_crsp_link_v2.csv",
                   low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
ext = ext.merge(lw, on="series_id", how="inner")
ext["wficn"] = ext["wficn"].astype("int64")
ext["q"] = pd.PeriodIndex(ext["period"], freq="M").asfreq("Q")
ew = ext.groupby(["wficn", "q"])["as_min_ru"].mean().reset_index()
EW = {w: g.set_index("q")["as_min_ru"].sort_index()
      for w, g in ew.groupby("wficn")}

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
B = pd.Period("2023Q3", freq="Q")
open_sp = sp[(sp["capitulated"] == False) & (sp["died"] == 0)
             & (sp["end_p"] >= B - 2)].copy()
open_sp["wficn"] = open_sp["wficn"].astype("int64")
open_sp = open_sp[open_sp["wficn"].isin(EW)]

rows = []
for w in open_sp["wficn"]:
    ser = EW[w]
    run, cq = 0, None
    for qq, v in ser.items():
        run = run + 1 if v < 0.70 else 0
        if run == 2 and cq is None:
            cq = qq
    if cq is None:
        continue
    after = ser[ser.index >= cq]
    lastv = float(ser.iloc[-1])
    n_after = len(after)
    below_share = float((after < 0.70).mean())
    rows.append((w, cq, lastv, n_after, below_share))
d = pd.DataFrame(rows, columns=["wficn", "cq", "last_as", "n_after",
                                "below_share"])
log.append(f"crossers: {len(d):,}")
log.append(f"AS at LAST observation: median {d['last_as'].median():.3f}; "
           f"still below 0.70: {(d['last_as'] < 0.70).mean():.1%}; "
           f"back above 0.75 (bounced): "
           f"{(d['last_as'] >= 0.75).mean():.1%}")
log.append(f"share of post-crossing quarters spent below 0.70: median "
           f"{d['below_share'].median():.0%}")
sub = d[d["n_after"] >= 4]
log.append(f"crossers with >=4q of follow-up (n {len(sub):,}): still "
           f"below at last obs {(sub['last_as'] < 0.70).mean():.1%}")
log.append("\nreading: >70% still below = persistent folds, the surge "
           "stands pending 33e2 + integration; <50% = an unstable "
           "instrument at the threshold, downgrade to 'elevated churn' "
           "until the S&P benchmarks settle it.")
log.append("\nSTAGE 33g2 DONE - aggregates only.")
P.write_report("nport_33g2_persistence.txt", log)
print("\n".join(log))
