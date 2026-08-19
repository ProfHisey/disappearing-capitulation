"""Stage 37d: RECOVERY VIA AALEN-JOHANSEN (audit round 5 / M6 closeout).

37b estimated recovery with Kaplan-Meier, censoring at last observation -
which treats death as independent censoring and estimates recovery "in a
world where nobody dies", inflating cumulative recovery most in the era
with more post-capitulation deaths (the wave). This stage redoes it with
death as a competing event (Aalen-Johansen), by era, at bars
0.70/0.75/0.80, and reports the death incidence that drives the gap.

Conventions kept from 37b for comparability (and disclosed): recovery =
2 consecutive OBSERVED quarters at/above the bar, timed in calendar
quarters from the crossing; death events use death_q even when it falls
after the fund's last AS observation (recovery in that gap is
unobservable - biases against recovery, symmetrically by era).

Aggregates only; report: output/referee_37d_recovery_aj.txt
Builds the panel - run alone or beside 35d.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["RECOVERY, COMPETING-RISKS VERSION (stage 37d)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter")["as_min"].dropna().sort_index()
      for w, g in panel.groupby("wficn")}
dd = death[death["died"] == 1]
DQ = dict(zip(dd["wficn"].astype("int64"),
              pd.PeriodIndex(dd["death_q"], freq="Q")))

caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps["era2"] = np.where(caps["cq"].dt.year <= 2009, "1995-2009",
                        "2010-23")
caps = caps[caps["cq"].dt.year >= 1995]
log.append(f"capitulation events 1995+: {len(caps):,}")

def episode(w, cq, bar):
    """(t, code) with code 0=censored, 1=recovered, 2=died."""
    w = int(w)
    s = PF.get(w)
    post = s[s.index > cq] if s is not None else None
    t_rec = None
    if post is not None and len(post):
        run, start_q = 0, None
        for q, v in post.items():
            run = run + 1 if v >= bar else 0
            if run == 1:
                start_q = q
            if run == 2:
                t_rec = (start_q - cq).n
                break
        t_last = (post.index[-1] - cq).n
    else:
        t_last = 0
    t_die = (DQ[w] - cq).n if w in DQ and DQ[w] > cq else None
    if t_rec is not None and (t_die is None or t_rec <= t_die):
        return t_rec, 1
    if t_die is not None:
        return t_die, 2
    return max(t_last, 1), 0

for bar in (0.70, 0.75, 0.80):
    log.append(f"\nbar {bar:.2f}:")
    for era in ("1995-2009", "2010-23"):
        sub = caps[caps["era2"] == era]
        rows = [episode(w, q, bar)
                for w, q in zip(sub["wficn"], sub["cq"])]
        d = pd.DataFrame(rows, columns=["t", "code"])
        d = d[d["t"] > 0]
        n_rec = int((d["code"] == 1).sum())
        n_die = int((d["code"] == 2).sum())
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(d["t"], d["code"], event_of_interest=1)
        cif = aj.cumulative_density_
        def at(k):
            c = cif[cif.index <= k]
            return float(c.iloc[-1, 0]) if len(c) else 0.0
        ajd = AalenJohansenFitter(calculate_variance=False)
        ajd.fit(d["t"], d["code"], event_of_interest=2)
        cifd = ajd.cumulative_density_
        def atd(k):
            c = cifd[cifd.index <= k]
            return float(c.iloc[-1, 0]) if len(c) else 0.0
        log.append(f"  {era}: n {len(d):,} (rec {n_rec}, died {n_die})")
        log.append(f"    P(recovered) by 8/16/24q: {at(8):.1%} / "
                   f"{at(16):.1%} / {at(24):.1%}   "
                   f"[37b KM said: see comparison note]")
        log.append(f"    P(died)      by 8/16/24q: {atd(8):.1%} / "
                   f"{atd(16):.1%} / {atd(24):.1%}")
log.append("\ncomparison (37b KM, bar 0.70): wave 18.6/30.2/40.8 vs "
           "modern 12.9/20.8/26.9; bar 0.80 inversion modern 8.1/13.0/"
           "16.5 vs wave 5.3/10.8/14.2. READING: if the AJ era gap at "
           "0.70 shrinks materially vs KM, the 'partial recovery died' "
           "leg of F9 was partly a competing-risk artifact and the "
           "ledger's F9 era language gets rewritten; if the 0.80 "
           "INVERSION survives AJ, the all-or-nothing finding stands on "
           "the corrected estimator. Quality leg (peak AS levels) is "
           "unaffected either way.")
log.append("\nSTAGE 37d DONE - aggregates only.")
P.write_report("referee_37d_recovery_aj.txt", log)
print("\n".join(log))
