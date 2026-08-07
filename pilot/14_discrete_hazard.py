"""Stage 14: v2 SURVIVAL MACHINERY — discrete-time hazard (primary spec).

Why: our durations are inherently quarterly, the stage-13 Cox showed a
proportional-hazards violation, and the brief's line (vi) horse race (does
DURATION or DEPTH of underperformance drive capitulation?) needs time-varying
covariates. A discrete-time model handles all three at once: each at-risk
spell-quarter becomes one observation, the outcome is "capitulated this
quarter," the baseline hazard is duration-bin dummies, and covariates vary
quarter by quarter. Standard errors cluster by fund.

All covariates are LAGGED one quarter (information available at the start of
the quarter) to avoid mechanical simultaneity with the event.

Models:
  M1  1980-2023, no flows (flows only exist post-2000)
  M2  2000-2023, with lagged retail flows
Requires: statsmodels (pip install statsmodels), stage 11+ caches.
Outputs: output/dt_hazard_report.txt, dt_hazard_table.csv (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL

log = ["DISCRETE-TIME HAZARD (v2 primary specification)", "=" * 60]

panel = PL.build_panel(log)
sp = PL.extract_spells(panel, client_cut=None)
sp["start_p"] = pd.PeriodIndex(sp["start_q"], freq="Q")
sp["spell_id"] = np.arange(len(sp))

# death classification (as stages 07/13) for the M3 death-outcome model
death = PL.get_death(log)
sp = sp.merge(death, on="wficn", how="left")
sp["end_p"] = pd.PeriodIndex(sp["end_q"], freq="Q")
_dp = pd.PeriodIndex(sp["death_q"].where(
    sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
_gap = (_dp - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))
sp["spell_died"] = (sp["ended_by"].isin(["data_end", "as_missing"])
                    & sp["died"].fillna(False) & _gap.between(-1, 4)
                    & sp["m_dur"].isna())

# fund-level lookups
pf = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}
fm = PL.get_fund_monthly([])
fm["quarter"] = fm["caldt"].dt.to_period("Q")
tnaq = fm.groupby(["wficn", "quarter"])["tna"].last()
cov = pd.read_parquet(P.CACHE / "covars.parquet")
cov["quarter"] = pd.PeriodIndex(cov["quarter"], freq="Q")
covq = cov.set_index(["wficn", "quarter"])

def q_at(g, q, col):
    return g.at[q, col] if q in g.index else np.nan

rows = []
for _, s in sp.iterrows():
    w = s["wficn"]
    g = pf.get(w)
    if g is None:
        continue
    T = int(s["m_dur"]) if pd.notna(s["m_dur"]) else int(s["end_dur"])
    T = max(T, 1)
    start = s["start_p"]
    depth_so_far = 0.0
    for t in range(1, T + 1):
        qlag = start + (t - 1)          # info known entering quarter t
        rel_lag = q_at(g, qlag, "rel4q")
        if pd.notna(rel_lag):
            depth_so_far = min(depth_so_far, float(rel_lag))
        as_lag = q_at(g, qlag, "as_min")
        as_lag2 = q_at(g, qlag - 1, "as_min")
        rows.append({
            "spell_id": s["spell_id"], "wficn": w, "t": t,
            "event": int(pd.notna(s["m_dur"]) and t == int(s["m_dur"])),
            "event_die": int(bool(s["spell_died"]) and t == T),
            "as_lag": as_lag,
            "das_lag": (as_lag - as_lag2) if pd.notna(as_lag) and pd.notna(as_lag2) else np.nan,
            "depth": depth_so_far,
            "flow_lag": q_at(g, qlag, "flowq"),
            "yr": (start + t).year,
        })
dt = pd.DataFrame(rows)

# entry-level covariates (stable within spell)
sp_entry = sp[["spell_id", "wficn", "start_p"]].copy()
sp_entry = sp_entry.merge(tnaq.rename("tna0").reset_index(),
                          left_on=["wficn", "start_p"],
                          right_on=["wficn", "quarter"], how="left")
ent_cov = covq.reindex(pd.MultiIndex.from_frame(sp_entry[["wficn", "start_p"]]))
sp_entry["exp100"] = pd.to_numeric(ent_cov["exp_ratio"].values, errors="coerce") * 100
sp_entry["mgr_dt"] = ent_cov["mgr_dt"].values
sp_entry["tenure"] = ((sp_entry["start_p"].dt.to_timestamp()
                       - pd.to_datetime(sp_entry["mgr_dt"])).dt.days / 365.25)
sp_entry["tenure"] = sp_entry["tenure"].where(sp_entry["tenure"].between(0, 60))
sp_entry["ln_tna"] = np.log(sp_entry["tna0"].where(sp_entry["tna0"] > 0))
dt = dt.merge(sp_entry[["spell_id", "ln_tna", "exp100", "tenure"]],
              on="spell_id", how="left")

# duration bins (the baseline hazard) and eras
dt["dur_3_4"] = dt["t"].between(3, 4).astype(float)
dt["dur_5_8"] = dt["t"].between(5, 8).astype(float)
dt["dur_9_12"] = dt["t"].between(9, 12).astype(float)
dt["dur_13p"] = (dt["t"] >= 13).astype(float)
dt["era_9509"] = dt["yr"].between(1995, 2009).astype(float)
dt["era_1023"] = (dt["yr"] >= 2010).astype(float)
dt["depth_x_dur"] = dt["depth"] * (dt["t"] / 4.0)

log.append(f"\nspell-quarter panel: {len(dt):,} at-risk quarters, "
           f"{dt['spell_id'].nunique():,} spells, {int(dt['event'].sum()):,} events")

def fit(df, xcols, label, ycol="event"):
    d = df[[ycol, "wficn"] + xcols].dropna()
    y = d[ycol].to_numpy(float)
    X = sm.add_constant(d[xcols].to_numpy(float))
    m = sm.GLM(y, X, family=sm.families.Binomial(
        link=sm.families.links.CLogLog())).fit(
        cov_type="cluster", cov_kwds={"groups": d["wficn"].to_numpy()})
    log.append(f"\n{label}: n={len(d):,}, events={int(y.sum()):,}")
    log.append(f"  {'covariate':12s} {'HR':>8s} {'coef':>8s} {'clust z':>8s} {'p':>8s}")
    out = []
    for name, b, se, z, p in zip(["const"] + xcols, m.params, m.bse,
                                 m.tvalues, m.pvalues):
        log.append(f"  {name:12s} {np.exp(b):8.3f} {b:+8.3f} {z:8.2f} {p:8.4f}")
        out.append((label, name, np.exp(b), b, se, z, p))
    return out

log.append("\nmissing-data cost per covariate (share of spell-quarters missing):")
for c in ["depth", "as_lag", "das_lag", "ln_tna", "exp100", "tenure", "flow_lag"]:
    log.append(f"  {c}: {dt[c].isna().mean():.1%}")

# Reduced form = the headline (total effect of stress); mechanics model adds
# the AS path and is over-controlled BY DESIGN (as_lag is the proximate cause,
# so stress effects run through it - a mediation structure, not a bug).
CORE = ["dur_3_4", "dur_5_8", "dur_9_12", "dur_13p", "depth", "depth_x_dur",
        "ln_tna", "exp100", "tenure", "era_9509", "era_1023"]
MECH = CORE + ["as_lag", "das_lag"]
M2COLS = [c for c in CORE if c != "era_9509"] + ["flow_lag"]  # era_9509 is
# collinear with the constant in a 2000+ sample (baseline becomes 2000-2009)
res = []
res += fit(dt, CORE, "M1a 1980-2023 REDUCED FORM (headline: total effect)")
res += fit(dt, MECH, "M1b 1980-2023 + AS mechanics (mediation evidence)")
res += fit(dt[dt["yr"] >= 2000], M2COLS, "M2 2000-2023 reduced form + flows")
# M3: same specification, DEATH as the outcome. Prediction registered in
# advance: depth's sign flips relative to M1a (deep spells die; shallow-long
# spells capitulate). If confirmed, depth selects the failure MODE.
res += fit(dt, CORE, "M3 1980-2023 DEATH outcome (mode-selection test)",
           ycol="event_die")
pd.DataFrame(res, columns=["model", "covariate", "HR", "coef", "se", "z", "p"]) \
  .to_csv(P.OUT / "dt_hazard_table.csv", index=False)

log.append("""
Reading guide:
  dur_* bins ARE the baseline hazard (relative to quarters 1-2): the DURATION
    answer for the horse race, holding depth fixed.
  depth = worst trailing 4q relative return so far (lagged, negative number;
    HR < 1 means deeper underperformance RAISES the hazard, because depth is
    negative - read the sign carefully). depth_x_dur tests whether depth bites
    harder in long spells.
  das_lag = last quarter's change in Active Share: the leading-indicator test
    (grind-then-fold showed capitulators start sliding early).
  This specification replaces Cox as primary: no PH assumption, cluster-robust
    by fund, quarterly by construction. Cox (stage 13) becomes robustness.
Caveats: entry-level TNA/fees/tenure (time-varying versions later); CRSP
  tenure noisy; M2 sample limited to flow-covered spell-quarters.""")
log.append("DISCRETE-TIME HAZARD DONE - aggregates only.")
P.write_report("dt_hazard_report.txt", log)
print("\n".join(log))
