"""Stage 37c: RECOVERY FORENSICS - real re-conviction or benchmark swap?

37b found real-quality recoveries (peak AS 0.82-0.89). Last threat: an
apparent recovery can be manufactured if the MIN-AS BENCHMARK changes -
AS vs a worse-fitting index jumps with no portfolio change (the mirror of
the 18b crossing forensics). For each durable (2q) recovery at bar 0.75:

 (a) share where bench_min at recovery differs from bench_min at crossing;
 (b) for benchmark-stable recoveries: confirm the era pattern holds;
 (c) for benchmark-switch recoveries: AS vs the ORIGINAL benchmark at the
     recovery quarter - if also >= 0.75, the switch is cosmetic and the
     recovery still real.

Aggregates only; report: output/referee_37c_recovery_forensics.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["RECOVERY FORENSICS (stage 37c)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
bp["quarter"] = pd.to_datetime(bp["month"]).dt.to_period("Q")
bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
        .drop_duplicates(["wficn", "quarter"], keep="last")
        .set_index(["wficn", "quarter"]))

caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps["era2"] = np.where(caps["cq"].dt.year <= 2009, "1995-2009",
                        "2010-23")

BAR = 0.75
rows = []
for _, s in caps.iterrows():
    g = PF.get(s["wficn"])
    if g is None:
        continue
    post = g.loc[g.index > s["cq"], "as_min"].dropna()
    run, rq = 0, None
    for q, v in post.items():
        run = run + 1 if v >= BAR else 0
        if run == 2:
            rq = q
            break
    if rq is None:
        continue
    w = s["wficn"]
    b0 = (bp.at[(w, s["cq"]), "bench_min"]
          if (w, s["cq"]) in bp.index else None)
    b1 = (bp.at[(w, rq), "bench_min"] if (w, rq) in bp.index else None)
    as_orig = np.nan
    if b0 is not None and (w, rq) in bp.index:
        col = "as_" + str(b0).lower()
        if col in bp.columns:
            as_orig = bp.at[(w, rq), col]
    rows.append((s["era2"], b0, b1,
                 (b0 is not None and b1 is not None and b0 != b1),
                 float(as_orig) if pd.notna(as_orig) else np.nan))

f = pd.DataFrame(rows, columns=["era2", "b0", "b1", "switched", "as_orig"])
known = f[f["b0"].notna() & f["b1"].notna()]
log.append(f"durable recoveries at bar {BAR}: {len(f):,} "
           f"({len(known):,} with benchmark data at both dates)")
log.append(f"  benchmark SWITCHED between crossing and recovery: "
           f"{known['switched'].mean():.1%}")
for era in ("1995-2009", "2010-23"):
    d = known[known["era2"] == era]
    if len(d):
        log.append(f"    {era}: switched {d['switched'].mean():.1%} "
                   f"(n {len(d):,})")

sw = known[known["switched"]]
if len(sw):
    ok = (sw["as_orig"] >= BAR)
    log.append(f"  among switchers, AS vs ORIGINAL benchmark also >= "
               f"{BAR} at recovery: {ok.mean():.1%} of "
               f"{ok.notna().sum():,} measurable "
               f"(median AS-vs-original {sw['as_orig'].median():.3f})")
stable = known[~known["switched"]]
log.append(f"  benchmark-STABLE recoveries (the clean set): "
           f"{len(stable):,}")
for era in ("1995-2009", "2010-23"):
    d = stable[stable["era2"] == era]
    n_era = len(caps[caps["era2"] == era])
    if n_era:
        log.append(f"    {era}: {len(d):,} clean recoveries of "
                   f"{n_era:,} era capitulations ({len(d) / n_era:.1%} "
                   f"raw, uncensored share)")

log.append("\nreading: if switching is rare (<15%) or switchers still "
           "clear the bar vs their ORIGINAL benchmark, 37b's re-conviction "
           "finding survives its last threat and F9 (all-or-nothing "
           "recovery) enters the facts list. If switching dominates, "
           "recovery is partly a benchmark-assignment artifact and gets "
           "reported WITH that decomposition.")
log.append("\nSTAGE 37c DONE - aggregates only.")
P.write_report("referee_37c_recovery_forensics.txt", log)
print("\n".join(log))
