"""Stage 22: RECOVERY AS A COMPETING RISK (referee critique 10).

The hazard machinery so far treats recovery as censoring, and recovery is
not innocent censoring: funds recover through the same mean reversion that
drives depth. This stage rebuilds the estimates with recovery as a third
competing outcome, two ways:

 (a) THREE-STATE Aalen-Johansen cumulative incidence: capitulate / die /
     recover as mutually competing events (only data-edge exits remain as
     censoring). Compare with Table 1's two-state version at the same
     horizons - this bounds how much the recovery-as-censoring assumption
     moves the paper's cumulative-incidence magnitudes.
 (b) DISCRETE-TIME COMPETING RISKS: a multinomial model on spell-quarters
     with four outcomes per quarter (continue / capitulate / die / recover).
     The capitulation branch's duration, depth, and era coefficients are
     compared with the binary cloglog headline. If signs and rough
     magnitudes agree, informative censoring is not driving the hazard
     results. (Multinomial SEs are unclustered - this is a structure check,
     not a headline re-estimate; the clustered binary model stays primary.)

Output: output/referee_22_recovery.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import AalenJohansenFitter

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["RECOVERY AS A COMPETING RISK (stage 22)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# event coding: 0 censored (data edge), 1 capitulate, 2 die, 3 recover
sp["etype"] = 0
sp.loc[sp["ended_by"] == "recovered", "etype"] = 3
sp.loc[sp["spell_died"], "etype"] = 2
sp.loc[sp["capitulated"], "etype"] = 1
sp["dur"] = np.where(sp["etype"] == 1, sp["m_dur"], sp["end_dur"])
sp["dur"] = pd.to_numeric(sp["dur"]).clip(lower=1)
log.append(f"spells {len(sp):,}: capitulate {(sp['etype'] == 1).sum():,}, "
           f"die {(sp['etype'] == 2).sum():,}, recover "
           f"{(sp['etype'] == 3).sum():,}, censored "
           f"{(sp['etype'] == 0).sum():,}")

# ------------------------------------------- (a) three-state AJ ----
def sect_aj():
    HOR = [8, 16, 24, 40]
    log.append(f"  {'event':12s} " + " ".join(f"{h:>3d}q" for h in HOR)
               + "   (three-state, recovery competing)")
    twostate = {1: [0.029, 0.068, 0.102, 0.144],
                2: [0.115, 0.231, 0.331, 0.429]}
    for evt, name in ((1, "capitulate"), (2, "die"), (3, "recover")):
        aj = AalenJohansenFitter(calculate_variance=False)
        aj.fit(sp["dur"], sp["etype"], event_of_interest=evt)
        c = aj.cumulative_density_
        vals = []
        for h in HOR:
            i = c.index[c.index <= h]
            vals.append(float(c.loc[i[-1]].iloc[0]) if len(i) else np.nan)
        log.append(f"  {name:12s} "
                   + " ".join(f"{v:.3f}" for v in vals))
        if evt in twostate:
            log.append(f"  {'(two-state)':12s} "
                       + " ".join(f"{v:.3f}" for v in twostate[evt]))
    log.append("  reading: the two-state Table 1 numbers treated recovery as "
               "censoring, which mechanically inflates long-horizon CIFs. "
               "The three-state numbers are the bounded, defensible version; "
               "if the era and depth CONTRASTS are what the paper claims "
               "(and they are estimated in the hazard models, not here), "
               "level shifts in the CIFs change exposition, not findings.")

# ------------------------- (b) discrete-time multinomial competing ----
def sect_mnl():
    rows = []
    for _, s in sp.iterrows():
        g = PF.get(s["wficn"])
        if g is None:
            continue
        T = int(s["dur"])
        start = s["start_p"]
        # audit fix A1: observed-row clock, not start + t calendar arithmetic
        idx = g.index
        p0 = idx.get_loc(start)
        dsf = 0.0
        for t in range(1, T + 1):
            q = idx[p0 + t - 1] if p0 + t - 1 < len(idx) else start + (t - 1)
            q_risk = idx[p0 + t] if p0 + t < len(idx) else q + 1
            rl = g.at[q, "rel4q"] if q in g.index else np.nan
            if pd.notna(rl):
                dsf = min(dsf, float(rl))
            out = 0
            if t == T:
                out = int(s["etype"]) if s["etype"] in (1, 2, 3) else 0
            rows.append({"y": out, "dur_5p": float(t >= 5), "depth": dsf,
                         "era_1023": float(q_risk.year >= 2010)})
    dt = pd.DataFrame(rows)
    log.append(f"  spell-quarters {len(dt):,}; outcomes "
               + ", ".join(f"{k}:{v:,}" for k, v in
                           dt["y"].value_counts().sort_index().items()))
    X = sm.add_constant(dt[["dur_5p", "depth", "era_1023"]].to_numpy(float))
    m = sm.MNLogit(dt["y"].to_numpy(), X).fit(disp=0, maxiter=200)
    names = ["const", "dur_5p", "depth", "era_1023"]
    branches = ["capitulate", "die", "recover"]
    log.append("  multinomial relative-risk ratios vs 'continue' "
               "(UNclustered SEs - structure check only):")
    for b in range(m.params.shape[1]):
        line = "  ".join(f"{n} {np.exp(m.params[i, b]):.2f}"
                         for i, n in enumerate(names) if n != "const")
        log.append(f"    {branches[b]:10s}: {line}")
    log.append("  binary cloglog headline for comparison (post-audit, stage "
               "26): capitulation dur_5p 2.71, depth 10.7, era 0.22; death "
               "dur 0.92, depth 0.44, era 0.78.")
    log.append("  reading: capitulation branch showing dur>1, depth>1 "
               "(shallow folds), era<1 with recovery explicitly in the "
               "outcome set = the headline hazards are not artifacts of "
               "treating recovery as censoring.")

R.section(log, "(a) THREE-STATE CUMULATIVE INCIDENCE", sect_aj)
R.section(log, "(b) MULTINOMIAL DISCRETE-TIME COMPETING RISKS", sect_mnl)

log.append("\nSTAGE 22 DONE - aggregates only.")
P.write_report("referee_22_recovery.txt", log)
print("\n".join(log))
