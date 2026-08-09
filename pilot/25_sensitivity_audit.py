"""Stage 25: FINAL SENSITIVITY AUDIT - reassignment tail + gap handling.

Two loose ends from the battery, both aimed at the capitulation definition:

 (a) REASSIGNMENT SENSITIVITY. 18c showed ~25% of benchmark-reassignment
     crossings have Active Share >= 0.75 against their ORIGINAL benchmark
     at the crossing - the style-migration-flavored tail. This rerun drops
     those events entirely (they become non-capitulations; their spells run
     on to their natural ends) and re-estimates the era table and slim
     hazards. If the era decline and gradients hold, the min-AS definition
     is robust to its most attackable edge.
 (b) GAP AUDIT. AS reporting gaps that drop out of the panel are silently
     bridged (durations count observed quarters). This section first
     quantifies the phenomenon (share of spells containing calendar gaps,
     gap sizes), then runs the strict variant: reindex every fund to a
     full quarterly calendar so gaps appear as explicit missing rows and
     spells CENSOR at the first gap. If the headline survives spells being
     broken at every gap, the bridging convention is immaterial.

Output: output/referee_25_sensitivity.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["FINAL SENSITIVITY AUDIT (stage 25)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp0 = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# ------------------------------------ (a) reassignment tail dropped ----
def sect_reassign():
    bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
    bp["quarter"] = pd.to_datetime(bp["month"]).dt.to_period("Q")
    bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
            .drop_duplicates(["wficn", "quarter"], keep="last")
            .set_index(["wficn", "quarter"]))
    dropped = 0
    sp = sp0.copy()
    for i, s in sp[sp["capitulated"]].iterrows():
        w, start = s["wficn"], s["start_p"]
        qc = start + int(s["m_dur"])
        if (w, start) not in bp.index or (w, qc) not in bp.index:
            continue
        col = "as_" + str(bp.at[(w, start), "bench_min"]).lower()
        if col not in bp.columns:
            continue
        v = bp.at[(w, qc), col]
        if pd.notna(v) and float(v) >= 0.75:
            sp.at[i, "capitulated"] = False
            sp.at[i, "m_dur"] = np.nan
            dropped += 1
    log.append(f"  crossings dropped (AS vs original benchmark >= 0.75 at "
               f"crossing): {dropped} of {int(sp0['capitulated'].sum())}")
    R.summarize(sp, log, "STRICT DEFINITION (migration-flavored crossings "
                         "removed)")
    dt = R.build_dt(sp, PF)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation, strict definition")
    R.slim_fit(dt, R.SLIM, "event_die", log, "death, strict definition")
    log.append("  compare: v4 baseline era caps 6.51/3.14/0.99, era HR 0.21, "
               "dur 2.79, depth 28.3.")

# ------------------------------------------------ (b) gap audit ----
def sect_gaps():
    log.append("  *** SUPERSEDED: this section's 'break-at-gap variant' is a "
               "NO-OP (asfreq on a PeriodIndex inserts nothing; see stage "
               "25b diagnostic). The real variants live in stage 25c. Kept "
               "only for the audit trail - the gap-share count below is "
               "still valid. ***")
    # quantify calendar gaps inside baseline spells
    n_gap, tot_gap_q = 0, 0
    for _, s in sp0.iterrows():
        g = PF.get(s["wficn"])
        if g is None:
            continue
        span = (s["end_p"] - s["start_p"]).n + 1
        obs = int(((g.index >= s["start_p"])
                   & (g.index <= s["end_p"])).sum())
        if span > obs:
            n_gap += 1
            tot_gap_q += span - obs
    log.append(f"  spells containing calendar gaps: {n_gap:,} of "
               f"{len(sp0):,} ({n_gap / len(sp0):.1%}); total bridged "
               f"quarters {tot_gap_q:,}")

    # strict variant: full quarterly calendar per fund, censor at gaps
    cols = ["quarter", "as_min", "qret", "bench_qret", "flowq", "rel4q"]
    def full_cal(g):
        return g.set_index("quarter").asfreq("Q").reset_index()
    pan = (panel[["wficn"] + cols]
           .groupby("wficn", group_keys=True)[cols]
           .apply(full_cal).reset_index(level=0).reset_index(drop=True))
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    R.summarize(sp, log, "BREAK-AT-GAP VARIANT (spells censor at any "
                         "missing quarter)")
    pf = {w: g.set_index("quarter") for w, g in pan.groupby("wficn")}
    dt = R.build_dt(sp, pf)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation, break-at-gap")
    R.slim_fit(dt, R.SLIM, "event_die", log, "death, break-at-gap")
    log.append("  reading: bridging biases durations DOWN for gappy funds "
               "(fatigue conservative); breaking biases them down harder. "
               "The truth sits between the two variants - if both show the "
               "gradients and the era decline, the gap rule is immaterial "
               "and Appendix B says so with numbers.")

R.section(log, "(a) REASSIGNMENT TAIL REMOVED (critique 3 sensitivity)",
          sect_reassign)
R.section(log, "(b) GAP AUDIT + BREAK-AT-GAP VARIANT", sect_gaps)

log.append("\nSTAGE 25 DONE - aggregates only. The coded robustness queue "
           "is now empty; remaining work waits on N-SAR and Morningstar.")
P.write_report("referee_25_sensitivity.txt", log)
print("\n".join(log))
