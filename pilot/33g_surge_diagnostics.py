"""Stage 33g: IS THE EXTENSION-ERA CROSSING SURGE REAL? (33f diagnostics)

33f found 3.12%/fund-year crossings 2023Q4-2026Q2 vs ~0.4-1% in the
paper's modern era - a potential 'surrender returns' headline OR an
artifact. Three signatures tested:
 (a) TIMING: crossing dates by quarter - front-loading at the splice
     (2023Q4-2024Q2) = measurement-transition artifact;
 (b) SPLICE STEP: for crossers vs non-crossers, the jump between the
     paper panel's last AS (2023-09) and our first extension AS -
     a big negative step for crossers = artifact;
 (c) DEPTH: minimum AS reached by crossers - clustered at 0.65-0.70 =
     boundary shimmer; deep crossings = real folds;
 (d) DURATION-MATCHED BASELINE: crossing hazard among wave/modern-era
     spells alive at comparable durations, from the paper's own spells -
     the honest comparator for these long-duration survivors.

Aggregates only; report: output/nport_33g_surge_diagnostics.txt
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

log = ["EXTENSION-ERA SURGE DIAGNOSTICS (stage 33g)", "=" * 60]

# ---- rebuild 33f objects ------------------------------------------------
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

# last paper-panel AS per fund (the splice left side)
bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
bp["month"] = pd.to_datetime(bp["month"])
last = bp[bp["month"] == bp["month"].max()].copy()
ascols = [c for c in last.columns if c.startswith("as_")]
last["as_last"] = last[ascols].min(axis=1)
AS_LAST = last.set_index("wficn")["as_last"]

# ---- classify each followed fund ----------------------------------------
rows = []
for _, s in open_sp.iterrows():
    w = s["wficn"]
    ser = EW[w]
    run, cq, minv = 0, None, float(ser.min())
    for qq, v in ser.items():
        run = run + 1 if v < 0.70 else 0
        if run == 2 and cq is None:
            cq = qq
    first_ext = float(ser.iloc[0])
    a_last = float(AS_LAST.get(w, np.nan))
    rows.append((w, cq is not None, cq, minv, first_ext, a_last,
                 int(s["end_dur"])))
d = pd.DataFrame(rows, columns=["wficn", "crossed", "cq", "min_ext",
                                "first_ext", "as_last", "dur_b"])
log.append(f"followed funds: {len(d):,}; crossers: "
           f"{int(d['crossed'].sum()):,}")

# ---- (a) timing ---------------------------------------------------------
tim = d.loc[d["crossed"], "cq"].value_counts().sort_index()
log.append("crossing dates (2q rule met at):")
for qq, n in tim.items():
    log.append(f"    {qq}: {n}")
early = d.loc[d["crossed"], "cq"] <= pd.Period("2024Q2", freq="Q")
log.append(f"  share of crossings by 2024Q2: {early.mean():.1%} "
           f"(uniform over 10 possible crossing quarters would be ~30%)")

# ---- (b) splice step ----------------------------------------------------
ok = d[d["as_last"].notna()]
step = ok["first_ext"] - ok["as_last"]
for lab, m in (("crossers    ", ok["crossed"]),
               ("non-crossers", ~ok["crossed"])):
    st = step[m]
    log.append(f"  splice step (first ext AS minus last panel AS), "
               f"{lab}: median {st.median():+.3f}, p25 "
               f"{st.quantile(.25):+.3f}, share < -0.05: "
               f"{(st < -0.05).mean():.1%} (n {int(m.sum()):,})")
log.append("  reading: crossers with a big negative step CROSSED AT THE "
           "INSTRUMENT CHANGE, not in the world. Compare the two rows - "
           "a large gap = artifact-driven surge.")

# ---- (c) depth ----------------------------------------------------------
cr = d[d["crossed"]]
log.append(f"  crossers' minimum extension AS: median "
           f"{cr['min_ext'].median():.3f}, p25 "
           f"{cr['min_ext'].quantile(.25):.3f}; share staying >0.65 "
           f"(shallow): {(cr['min_ext'] > 0.65).mean():.1%}")

# ---- (d) duration-matched baseline --------------------------------------
dur_med = int(d["dur_b"].median())
log.append(f"  followed funds' spell duration at boundary: median "
           f"{dur_med}q")
base_rows = []
for era_lab, y0, y1 in (("1995-2009", 1995, 2009),
                        ("2010-23", 2010, 2023)):
    at_risk = events = 0
    for _, s in sp.iterrows():
        start_y = s["start_p"].year
        if not (y0 <= start_y + dur_med // 4 <= y1):
            continue
        if s["end_dur"] <= dur_med:
            continue
        horizon = min(int(s["end_dur"]) - dur_med, 11)
        at_risk += horizon
        if (s["capitulated"] and pd.notna(s["m_dur"])
                and dur_med < s["m_dur"] <= dur_med + horizon):
            events += 1
    if at_risk:
        base_rows.append((era_lab, events, at_risk,
                          events / (at_risk / 4)))
for era_lab, ev, ar, rt in base_rows:
    log.append(f"  duration-matched baseline, {era_lab}: {ev} events / "
               f"{ar:,} fund-quarters beyond {dur_med}q = "
               f"{rt:.2%}/fund-year")
log.append("  reading: THIS is the comparator for 33f's 3.12%/yr - "
           "long-duration survivors always had elevated hazard (the "
           "duration gradient). If the matched wave-era rate is ~3%, the "
           "'surge' is composition, not regime change.")

log.append("\nVERDICT LOGIC: artifact if (a) front-loaded AND (b) "
           "crossers show big negative splice steps; shimmer if (c) "
           "shallow; composition if (d) matched baseline ~3%; REAL "
           "REGIME CHANGE only if all four point the other way - then "
           "the S&P benchmark completion (33e2) becomes urgent before "
           "any claim.")
log.append("\nSTAGE 33g DONE - aggregates only.")
P.write_report("nport_33g_surge_diagnostics.txt", log)
print("\n".join(log))
