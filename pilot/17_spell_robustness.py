"""Stage 17: REFEREE BATTERY I — spell-machinery robustness.

Answers referee critiques 1, 9, 11, 12, 20, 23 (triage in the project doc
claude/referee-preempt-plan.md):

 (a) FEES (critique 1). Spell entry compares net fund returns to a cost-free
     index, so fee drag alone can put funds "underwater." Reruns with gross
     returns and with a fee-sized entry buffer; reports the share of baseline
     spells whose max depth never exceeds the fund's own annual fee.
 (b) SPELL-DEFINITION GRID (12). Entry buffer -2%, 8-quarter trailing window,
     minimum-depth filter, plus the spell max-depth distribution.
 (c) INCUBATION / AGE SCREENS + LEFT TRUNCATION (9). Drop spells in a fund's
     first 2-3 years and spells already underwater at first observation.
 (d) DEATH-WINDOW SENSITIVITY (11). Attribution windows 0/1/2/4 quarters.
 (e) COVARIATE TIMING (20). Depth lagged two quarters instead of one.
 (f) CLEANING-RULE COUNTS by era (23) + the explicit recovery definition.

Reading guide: each variant prints spells, outcome shares, the three-era
table, and the slim hazard (dur_5p, depth, era_1023) for BOTH outcomes. The
findings survive a variant when (i) the era decline in capitulation persists,
(ii) depth's sign still flips between capitulation and death, and (iii) the
duration HR stays > 1. Output: output/referee_17_spells.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["REFEREE BATTERY I - SPELL MACHINERY", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}


def describe(pan, label, min_depth=None, fit_death=True):
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    if min_depth is not None:
        sp = sp[sp["depth"] <= min_depth]
    R.summarize(sp, log, label)
    pf = {w: g.set_index("quarter") for w, g in pan.groupby("wficn")}
    dt = R.build_dt(sp, pf)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation")
    if fit_death:
        R.slim_fit(dt, R.SLIM, "event_die", log, "death")
    return sp


log.append("\nV0 BASELINE (identical machinery to stages 14-16)")
sp0 = describe(panel, "V0 baseline")

# ------------------------------------------------------------ (a) fees ----
def sect_fees():
    pan = R.load_exp_ratio(panel, log)
    # V1: gross returns (add back fees), rel4q recomputed on gross basis
    pan["qret_g"] = pan["qret"] + pan["exp_ratio"] / 4
    keep = ["wficn", "quarter", "as_min", "qret_g", "bench_qret", "flowq"]
    pan_g = R.retrail(pan[keep].rename(columns={"qret_g": "qret"}))
    describe(pan_g, "V1 GROSS RETURNS")
    # fee-sized spells: baseline spells whose max depth is within the fee
    ent = sp0.merge(
        pan[["wficn", "quarter", "exp_ratio"]]
           .rename(columns={"quarter": "start_p"}),
        on=["wficn", "start_p"], how="left")
    feesized = (ent["depth"].abs() <= ent["exp_ratio"]).mean()
    log.append(f"  share of BASELINE spells with |max depth| <= own annual "
               f"fee: {feesized:.1%}  (the fee-artifact share the referee "
               f"asked for)")
    # V2: net returns, entry needs to clear a fee-sized buffer (~1.5%/yr)
    pan_b = panel.copy()
    pan_b["rel4q"] = pan_b["rel4q"] + 0.015
    describe(pan_b, "V2 FEE BUFFER (entry needs net rel4q < -1.5%)")

# ------------------------------------------------- (b) definition grid ----
def sect_grid():
    pan = panel.copy()
    pan["rel4q"] = pan["rel4q"] + 0.02
    describe(pan, "V3 ENTRY BUFFER -2%")
    keep = ["wficn", "quarter", "as_min", "qret", "bench_qret", "flowq"]
    pan8 = R.retrail(panel[keep], window=8)
    describe(pan8, "V4 8-QUARTER TRAILING WINDOW")
    describe(panel, "V5 MIN-DEPTH FILTER (drop spells shallower than -2%)",
             min_depth=-0.02)
    qs = sp0["depth"].quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    log.append("  baseline spell max-depth distribution: "
               + ", ".join(f"p{int(k * 100)} {v:+.1%}" for k, v in qs.items()))
    log.append("  gap rule (stated for the paper, post-25b/25c): quarters "
               "absent from the panel are BRIDGED - durations count observed "
               "quarters; 'as_missing' fires only on explicit missing-AS rows. "
               "The censor-at-gap and calendar-true variants are bracketed in "
               "stage 25c and disclosed in Appendix B.")

# ------------------------------------- (c) age screens, left truncation ----
def sect_age():
    fm = PL.get_fund_monthly([])
    first = (fm.groupby("wficn")["caldt"].min()
               .dt.to_period("Q").rename("first_q").reset_index())
    sp = sp0.merge(first, on="wficn", how="left")
    age_y = (sp["start_p"] - sp["first_q"]).map(
        lambda x: getattr(x, "n", np.nan)) / 4.0
    for yrs in (2, 3):
        ss = sp[age_y >= yrs]
        R.summarize(ss, log, f"V6 drop spells in fund's first {yrs} years "
                             f"(incubation screen)")
        if yrs == 3:
            dt = R.build_dt(ss, PF)
            R.slim_fit(dt, R.SLIM, "event", log, "V6(3y) capitulation")
            R.slim_fit(dt, R.SLIM, "event_die", log, "V6(3y) death")
    firstrel = {w: g["rel4q"].first_valid_index() for w, g in PF.items()}
    lt = sp0["start_p"] == sp0["wficn"].map(firstrel)
    log.append(f"\n  left-truncated spells (underwater already at the fund's "
               f"first usable quarter): {int(lt.sum()):,} of {len(sp0):,}")
    ss = sp0[~lt]
    R.summarize(ss, log, "V7 drop left-truncated spells")
    dt = R.build_dt(ss, PF)
    R.slim_fit(dt, R.SLIM, "event", log, "V7 capitulation")

# ------------------------------------------------ (d) death windows ----
def sect_deathwin():
    sp_raw = PL.extract_spells(panel, client_cut=None)
    for w in (0, 1, 2, 4):
        sp = R.attach_death(sp_raw.copy(), death, window=w)
        log.append(f"\n  death window {w}q: died "
                   f"{int(sp['spell_died'].sum()):,} spells "
                   f"({sp['spell_died'].mean():.2%})")
        dt = R.build_dt(sp, PF)
        R.slim_fit(dt, R.SLIM, "event_die", log, f"death, window {w}q")
    log.append("  reading: if the depth HR sign holds at 0-1q windows, the "
               "look-ahead critique costs magnitude at most, not the claim.")

# ----------------------------------------------- (e) depth double-lag ----
def sect_lag2():
    dt2 = R.build_dt(sp0, PF, lag=2)
    R.slim_fit(dt2, R.SLIM, "event", log,
               "capitulation, depth lagged 2q (vs 1q in V0)")
    log.append("  note for the paper: the stage-14 convention already lags "
               "depth one quarter behind the event quarter; this is the "
               "stricter check.")

# -------------------------------------------- (f) cleaning-rule counts ----
def sect_cleaning():
    m1 = PL.get_mflink1()
    ret = P.load_monthly_returns(log).merge(m1, on="crsp_fundno", how="inner")
    ret = ret.sort_values(["crsp_fundno", "caldt"])
    ret["w"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
    ret["w"] = ret["w"].fillna(ret["mtna"]).clip(lower=0)
    ret = ret.dropna(subset=["mret"])
    ret["wr"] = ret["w"] * ret["mret"]
    fm = (ret.groupby(["wficn", "caldt"])
             .agg(wr=("wr", "sum"), w=("w", "sum"),
                  tna=("mtna", "sum")).reset_index())
    fm["fret"] = np.where(fm["w"] > 0, fm["wr"] / fm["w"], np.nan)
    fm = fm.dropna(subset=["fret"])
    fm["bad_ret"] = fm["fret"].abs() > 2.0
    fm["bad_tna"] = fm["tna"].fillna(0) < 1.0
    fm["y5"] = (fm["caldt"].dt.year // 5) * 5
    log.append("  fund-months dropped by the two hygiene screens, by 5y bin:")
    for y5, g in fm.groupby("y5"):
        log.append(f"    {int(y5)}-{int(y5) + 4}: {len(g):9,} months | "
                   f"|ret|>200%: {g['bad_ret'].mean():6.3%} | "
                   f"TNA<$1M: {g['bad_tna'].mean():6.2%}")
    log.append("  recovery definition (stated for the paper): a spell ends in "
               "'recovery' the first quarter the trailing 4-quarter net "
               "return vs benchmark is >= 0.")

R.section(log, "(a) FEES: gross returns + fee buffer (critique 1)", sect_fees)
R.section(log, "(b) SPELL-DEFINITION GRID (critique 12)", sect_grid)
R.section(log, "(c) AGE SCREENS + LEFT TRUNCATION (critique 9)", sect_age)
R.section(log, "(d) DEATH-WINDOW SENSITIVITY (critique 11)", sect_deathwin)
R.section(log, "(e) DEPTH DOUBLE-LAG (critique 20)", sect_lag2)
R.section(log, "(f) CLEANING-RULE COUNTS (critique 23)", sect_cleaning)

log.append("\nBATTERY I DONE - aggregates only.")
P.write_report("referee_17_spells.txt", log)
print("\n".join(log))
