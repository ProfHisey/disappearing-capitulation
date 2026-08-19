"""Stage 37b: RECOVERY, DONE RIGHT - censoring-aware + quality-graded.

37 found 38.7% durable recovery but with two problems this stage fixes:
 (a) CENSORING: median recovery takes 14q, AS data ends 2023m9 - modern
     folds mechanically can't show recoveries yet. Fix: Kaplan-Meier on
     time-to-recovery, censoring at last observed quarter, by era.
 (b) QUALITY: 60% of recoverers re-fold -> boundary oscillation suspicion.
     Fix: grade recoveries by the bar re-crossed (0.70 / 0.75 / 0.80,
     2 consecutive quarters each) and report post-recovery peak AS.

Aggregates only; report: output/referee_37b_recovery_km.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["RECOVERY v2: KM + QUALITY (stage 37b)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps["era2"] = np.where(caps["cq"].dt.year <= 2009, "1995-2009",
                        "2010-23")

def first_recovery(w, cq, bar):
    """(time_to_event_q, observed_flag, followup_q, peak_after) with
    2-consecutive-quarter durability at `bar`."""
    g = PF.get(w)
    if g is None:
        return None
    post = g.loc[g.index > cq, "as_min"].dropna()
    if not len(post):
        return None
    run, start_q = 0, None
    for q, v in post.items():
        run = run + 1 if v >= bar else 0
        if run == 1:
            start_q = q
        if run == 2:
            return ((start_q - cq).n, 1, (post.index[-1] - cq).n,
                    float(post.loc[post.index >= start_q].max()))
    return ((post.index[-1] - cq).n, 0, (post.index[-1] - cq).n, np.nan)

for bar in (0.70, 0.75, 0.80):
    rows = []
    for _, s in caps.iterrows():
        r = first_recovery(s["wficn"], s["cq"], bar)
        if r:
            rows.append((s["era2"],) + r)
    df = pd.DataFrame(rows, columns=["era2", "t", "obs", "fup", "peak"])
    log.append(f"\nrecovery bar {bar:.2f} (2 consecutive quarters):")
    log.append(f"  raw: {int(df['obs'].sum()):,} recoveries / "
               f"{len(df):,} events; median follow-up "
               f"{df['fup'].median():.0f}q")
    for era in ("1995-2009", "2010-23"):
        d = df[df["era2"] == era]
        if len(d) < 10:
            continue
        km = KaplanMeierFitter().fit(d["t"], d["obs"])
        sf = km.survival_function_
        def surv_at(k):
            s_ = sf[sf.index <= k]
            return float(s_.iloc[-1, 0]) if len(s_) else 1.0
        log.append(f"    {era}: KM P(recovered by 8q) "
                   f"{1 - surv_at(8):.1%}, by 16q {1 - surv_at(16):.1%}, "
                   f"by 24q {1 - surv_at(24):.1%}  "
                   f"(events {len(d):,}, recoveries "
                   f"{int(d['obs'].sum()):,})")
    pk = df.loc[df["obs"] == 1, "peak"].dropna()
    if len(pk):
        log.append(f"  post-recovery PEAK AS: median {pk.median():.3f}, "
                   f"p25 {pk.quantile(.25):.3f}, p75 "
                   f"{pk.quantile(.75):.3f}")

log.append("\nreading: (1) compare the era gap at FIXED horizons (8/16q) "
           "- that's censoring-honest; if the modern deficit survives, "
           "re-conviction really did die alongside surrender (mechanism "
           "doc F-list gains a fact). (2) If recoveries at bar 0.70 "
           "vanish at 0.75/0.80 and peak AS sits ~0.72, 'recovery' is "
           "boundary oscillation and Paper 1's durable-crossing framing "
           "stands unbruised; if quality recoveries are real, the "
           "multi-state story (stage 22) gets a genuine third act.")
log.append("\nSTAGE 37b DONE - aggregates only.")
P.write_report("referee_37b_recovery_km.txt", log)
print("\n".join(log))
