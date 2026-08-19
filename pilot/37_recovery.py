"""Stage 37: UN-CAPITULATION - does anyone ever come back? (ranked R3).

Stage 22 treated recovery as a competing state; this stage documents the
phenomenon itself, which the literature has not: after a durable crossing
below 0.70, does the fund's min-benchmark Active Share ever return above
0.70 - and durably? For each capitulated spell:

 (a) recovery rates under three durability rules (1q touch / 2q / 4q+);
 (b) time from crossing to recovery;
 (c) recovery rates by era (did re-conviction die along with surrender?);
 (d) what happens after recovery: survive active / re-fold / die.

Aggregates only; report: output/referee_37_recovery.txt
Builds the panel - run alone or after 35/39 finishes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["UN-CAPITULATION / RECOVERY (stage 37)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps["era3"] = pd.cut(caps["cq"].dt.year, [0, 1994, 2009, 9999],
                      labels=["1980-94", "1995-2009", "2010-23"])
log.append(f"capitulation events: {len(caps):,}")

rows = []
for _, s in caps.iterrows():
    g = PF.get(s["wficn"])
    if g is None:
        continue
    post = g.loc[g.index > s["cq"], "as_min"].dropna()
    if not len(post):
        rows.append((s["wficn"], s["cq"], s["era3"], 0, np.nan, np.nan,
                     len(post)))
        continue
    above = post >= 0.70
    # longest run lengths & first-recovery timing per durability rule
    first_k = {}
    run, start_q = 0, None
    for q, a in above.items():
        run = run + 1 if a else 0
        if a and run == 1:
            start_q = q
        for k in (1, 2, 4):
            if run == k and k not in first_k:
                first_k[k] = start_q
    gap = ((first_k[2] - s["cq"]).n if 2 in first_k else np.nan)
    rows.append((s["wficn"], s["cq"], s["era3"],
                 max((1 if 1 in first_k else 0),
                     2 * (2 in first_k), 4 * (4 in first_k)),
                 (first_k.get(1) - s["cq"]).n if 1 in first_k else np.nan,
                 gap, len(post)))
rec = pd.DataFrame(rows, columns=["wficn", "cq", "era3", "best_rule",
                                  "gap1", "gap2", "n_post"])
log.append(f"events with any post-crossing AS data: "
           f"{(rec['n_post'] > 0).sum():,} of {len(rec):,} "
           f"(median post-crossing quarters observed "
           f"{rec['n_post'].median():.0f})")

# ---- (a) recovery rates -------------------------------------------------
for k, lab in [(1, "touch >=0.70 once      "),
               (2, "2 consecutive quarters "),
               (4, "4 consecutive quarters ")]:
    r = (rec["best_rule"] >= k).mean()
    n = int((rec["best_rule"] >= k).sum())
    log.append(f"  recovery rate, {lab}: {r:.1%}  ({n:,} funds)")

# ---- (b) timing ---------------------------------------------------------
g2 = rec.loc[rec["best_rule"] >= 2, "gap2"].dropna()
if len(g2):
    log.append(f"  time to durable (2q) recovery: median {g2.median():.0f}q,"
               f" p25 {g2.quantile(.25):.0f}q, p75 {g2.quantile(.75):.0f}q")

# ---- (c) by era ---------------------------------------------------------
log.append("  durable (2q) recovery rate by capitulation era:")
for era in ["1980-94", "1995-2009", "2010-23"]:
    d = rec[rec["era3"] == era]
    if len(d):
        log.append(f"    {era}: {(d['best_rule'] >= 2).mean():6.1%} "
                   f"(events {len(d):,})")

# ---- (d) after recovery -------------------------------------------------
recd = rec[rec["best_rule"] >= 2]
outc = {"refold": 0, "stay": 0, "die": 0}
dd = death.set_index("wficn") if hasattr(death, "set_index") else None
died_w = set(sp.loc[sp["died"] == 1, "wficn"])
for _, s in recd.iterrows():
    g = PF.get(s["wficn"])
    rq = s["cq"] + int(s["gap2"])
    post = g.loc[g.index > rq, "as_min"].dropna()
    # re-fold: 2+ consecutive quarters back under 0.70 after recovery
    run, refolded = 0, False
    for a in (post < 0.70):
        run = run + 1 if a else 0
        if run >= 2:
            refolded = True
            break
    if refolded:
        outc["refold"] += 1
    elif s["wficn"] in died_w:
        outc["die"] += 1
    else:
        outc["stay"] += 1
tot = sum(outc.values())
if tot:
    log.append(f"  after durable recovery (n {tot}): re-fold "
               f"{outc['refold'] / tot:.0%}, stay active "
               f"{outc['stay'] / tot:.0%}, die (fund ends) "
               f"{outc['die'] / tot:.0%}")
log.append("  caveat: 'die' here is fund death any time after recovery, "
           "not death while active - refine before quoting.")

log.append("\nSTAGE 37 DONE - aggregates only. If durable recovery is "
           "rare (<10%), 'capitulation is an absorbing state' becomes a "
           "quotable line and validates treating it as terminal in the "
           "hazard models.")
P.write_report("referee_37_recovery.txt", log)
print("\n".join(log))
