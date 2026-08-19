"""Stage 39: DO MANAGERS SELL CONVICTION AT THE BOTTOM? (ranked R5).

The behavior-gap literature shows RETAIL investors capitulate at lows.
Mirror question for managers: are capitulation events followed by
above-average benchmark returns (folding right before the rebound)?

For each capitulation event, cumulate the fund's OWN benchmark return
(panel bench_qret) over the next 4 and 8 quarters; compare against the
all-fund-quarter baseline distribution, overall and by era. Inference
caveat printed loudly: events cluster in calendar time, so the effective
sample is distinct event-quarters, not events - v1 reports both counts.

Aggregates only; report: output/referee_39_market_timing.txt
Builds the panel - run alone or after 35/37 finishes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["MANAGER MARKET TIMING AT CAPITULATION (stage 39)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

def fwd_bench(w, q0, k):
    """Cumulative benchmark return over the k quarters AFTER q0, from the
    fund's own panel rows; NaN unless all k quarters observed."""
    g = PF.get(w)
    if g is None:
        return np.nan
    r = [g["bench_qret"].get(q0 + j, np.nan) for j in range(1, k + 1)]
    if any(pd.isna(x) for x in r):
        return np.nan
    return float(np.prod([1 + x for x in r]) - 1)

caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps["era3"] = pd.cut(caps["cq"].dt.year, [0, 1994, 2009, 9999],
                      labels=["1980-94", "1995-2009", "2010-23"])

for k in (4, 8):
    ev = [fwd_bench(w, q, k) for w, q in zip(caps["wficn"], caps["cq"])]
    ev = pd.Series(ev, index=caps.index).dropna()
    # baseline: same statistic across a 1-in-4 systematic sample of all
    # fund-quarters (full grid is expensive and unnecessary)
    base = []
    for w, g in list(PF.items())[::4]:
        for q in g.index[::4]:
            b = fwd_bench(w, q, k)
            if pd.notna(b):
                base.append(b)
    base = pd.Series(base)
    nq = caps.loc[ev.index, "cq"].nunique()
    log.append(f"\nforward {k}q benchmark return after capitulation:")
    log.append(f"  events {len(ev):,} (distinct event-quarters {nq}), "
               f"mean {ev.mean():+.1%}, median {ev.median():+.1%}")
    log.append(f"  baseline fund-quarters {len(base):,}: "
               f"mean {base.mean():+.1%}, median {base.median():+.1%}")
    log.append(f"  event minus baseline mean: "
               f"{ev.mean() - base.mean():+.1%}")
    for era in ["1980-94", "1995-2009", "2010-23"]:
        m = caps.loc[ev.index, "era3"] == era
        if m.sum():
            log.append(f"    {era}: event mean {ev[m].mean():+.1%} "
                       f"(n {int(m.sum())}, quarters "
                       f"{caps.loc[ev.index][m]['cq'].nunique()})")

log.append("\nINFERENCE CAVEAT (loud): capitulations cluster in calendar "
           "time (post-2000, post-2008 waves). The distinct-quarter counts "
           "above are the honest effective n; a positive gap here is "
           "suggestive, not significant, until a calendar-clustered test "
           "(quarter-level bootstrap) is run. v1 is descriptive by design.")
log.append("\nSTAGE 39 DONE - aggregates only. Positive gap = managers "
           "fold before rebounds (institutional behavior gap - mirrors "
           "the retail result and feeds the M2 option-abandonment story).")
P.write_report("referee_39_market_timing.txt", log)
print("\n".join(log))
