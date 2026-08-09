"""Stage 25c: the REAL break-at-gap test, and why 25(b) was a no-op.

25b's diagnostic proved the panel stores gaps as ABSENT rows, and that
pandas' asfreq on a period-indexed frame silently converts label frequency
instead of inserting missing periods. So stage 25(b)'s variant never
engaged (identical results were the tell), and throughout the pipeline
'trailing 4 quarters' has meant trailing 4 OBSERVED quarters.

This stage runs the real variants with an explicit period_range reindex:

 (a) BREAK-AT-GAP: gap quarters inserted as explicit missing rows, so
     spells censor at their first gap (ended_by counts will show
     'as_missing' firing for the first time).
 (b) CALENDAR-TRUE rel4q: relative performance recomputed on the calendar
     grid, so a gap poisons the trailing window for four quarters (the
     strictest possible convention), and spells re-extracted from scratch.

Baseline (observed-quarters, bridged) plus these two bracket every
defensible gap treatment. Output: output/referee_25c_gapfix.txt.
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["REAL BREAK-AT-GAP (stage 25c)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)

COLS = ["as_min", "qret", "bench_qret", "flowq", "rel4q"]

def cal_reindex(g):
    g = g.set_index("quarter")
    idx = pd.period_range(g.index.min(), g.index.max(), freq="Q")
    return g.reindex(idx).rename_axis("quarter").reset_index()

pan = (panel[["wficn", "quarter"] + COLS]
       .groupby("wficn", group_keys=True)[["quarter"] + COLS]
       .apply(cal_reindex).reset_index(level=0).reset_index(drop=True))
ins = len(pan) - len(panel)
log.append(f"inserted {ins:,} explicit gap rows "
           f"({len(panel):,} -> {len(pan):,} fund-quarters); if this is 0, "
           f"something is still wrong - stop and say so")

def report(p, label):
    sp = R.attach_death(PL.extract_spells(p, client_cut=None), death)
    R.summarize(sp, log, label)
    log.append("    ended_by: " + ", ".join(
        f"{k} {v:,}" for k, v in sp["ended_by"].value_counts().items()))
    pf = {w: g.set_index("quarter") for w, g in p.groupby("wficn")}
    dt = R.build_dt(sp, pf)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation")
    R.slim_fit(dt, R.SLIM, "event_die", log, "death")

R.section(log, "(a) BREAK-AT-GAP, properly engaged",
          lambda: report(pan, "BREAK-AT-GAP (gap rows inserted, spells "
                              "censor at gaps)"))

def sect_cal():
    def roll(g):
        g = g.set_index("quarter")
        fr = (1 + g["qret"]).rolling(4).apply(np.prod, raw=True) - 1
        br = (1 + g["bench_qret"]).rolling(4).apply(np.prod, raw=True) - 1
        g["rel4q"] = fr - br
        return g.reset_index()
    cols = ["quarter"] + COLS
    q = (pan.groupby("wficn", group_keys=True)[cols]
            .apply(roll).reset_index(level=0).reset_index(drop=True))
    report(q, "CALENDAR-TRUE rel4q (gaps poison the trailing window)")

R.section(log, "(b) CALENDAR-TRUE TRAILING WINDOW", sect_cal)

log.append("""
Reading guide: baseline bridges gaps and measures performance over observed
quarters; (a) censors at gaps; (b) additionally demands four consecutive
calendar quarters of data for the performance window. If the era decline
and both gradients appear under all three conventions, the gap machinery is
settled, and Appendix B states plainly that the paper's 'trailing four
quarters' means trailing four observed quarters.""")
log.append("STAGE 25c DONE - aggregates only.")
P.write_report("referee_25c_gapfix.txt", log)
print("\n".join(log))
