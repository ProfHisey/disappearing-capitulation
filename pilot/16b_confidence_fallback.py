"""Stage 16b: FALLBACK for stage 16 — use ONLY if 16 had to be killed.

Identical to 16_confidence_upgrades.py except:
  1. The report is written to disk BEFORE the slow mixed model starts, so a
     Ctrl+C during (d2) no longer loses sections (a)-(d1).
  2. The random-intercept model runs on a deterministic 25% subsample of
     funds (wficn % 4 == 0), cutting the fit roughly 16-fold. A frailty
     check does not need the full sample: if the duration gradient survives
     within the subsample with fund intercepts in, the conclusion carries.
  3. Adds the spell-order stratification the referee panel asked for
     (critique 8): first spells vs later spells, fitted separately.

Output: output/confidence_report_16b.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL

log = ["CONFIDENCE UPGRADES 16b (fallback)", "=" * 60]

panel = PL.build_panel(log)
sp = PL.extract_spells(panel, client_cut=None)
sp["start_p"] = pd.PeriodIndex(sp["start_q"], freq="Q")
sp["end_p"] = pd.PeriodIndex(sp["end_q"], freq="Q")
death = PL.get_death(log)
sp = sp.merge(death, on="wficn", how="left")
_dp = pd.PeriodIndex(sp["death_q"].where(
    sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
_gap = (_dp - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))
sp["capitulated"] = sp["m_dur"].notna()
sp["spell_died"] = (sp["ended_by"].isin(["data_end", "as_missing"])
                    & sp["died"].fillna(False) & _gap.between(-1, 4)
                    & ~sp["capitulated"])
pf = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# ------------------------------------------- spell-quarter frame builder ----
rows = []
for _, s in sp.iterrows():
    w = s["wficn"]
    g = pf.get(w)
    if g is None:
        continue
    T = int(s["m_dur"]) if s["capitulated"] else int(s["end_dur"])
    start = s["start_p"]
    dsf = 0.0
    for t in range(1, max(T, 1) + 1):
        q = start + (t - 1)
        rl = g.at[q, "rel4q"] if q in g.index else np.nan
        if pd.notna(rl):
            dsf = min(dsf, float(rl))
        rows.append({
            "wficn": w, "spell_id": s.name, "t": t, "depth": dsf,
            "event": int(s["capitulated"] and t == int(s["m_dur"])),
            "era_1023": float((start + t).year >= 2010),
        })
dt = pd.DataFrame(rows)
dt["dur_3_4"] = dt["t"].between(3, 4).astype(float)
dt["dur_5_8"] = dt["t"].between(5, 8).astype(float)
dt["dur_9_12"] = dt["t"].between(9, 12).astype(float)
dt["dur_13p"] = (dt["t"] >= 13).astype(float)
DUR = ["dur_3_4", "dur_5_8", "dur_9_12", "dur_13p"]

def fit(df, xcols, ycol, label):
    d = df[[ycol, "wficn"] + xcols].dropna()
    y = d[ycol].to_numpy(float)
    X = sm.add_constant(d[xcols].to_numpy(float))
    m = sm.GLM(y, X, family=sm.families.Binomial(
        link=sm.families.links.CLogLog())).fit(
        cov_type="cluster", cov_kwds={"groups": d["wficn"].to_numpy()})
    log.append(f"\n  {label} [n={len(d):,}, events={int(y.sum()):,}]")
    for name, b, z, p in zip(["const"] + xcols, m.params, m.tvalues, m.pvalues):
        if name == "const":
            continue
        log.append(f"    {name:10s} HR {np.exp(b):7.3f}  z {z:6.2f}  p {p:.4f}")
    return m

# ------------------------- (d1) first vs later spells (order strata) ----
log.append("\n(d1) SPELL-ORDER STRATIFICATION (frailty/composition check)")
order = (sp.sort_values("start_p").groupby("wficn").cumcount())
first_ids = set(sp.index[order == 0])
later_ids = set(sp.index[order >= 1])
fit(dt[dt["spell_id"].isin(first_ids)], DUR + ["depth", "era_1023"],
    "event", "CAPITULATION, FIRST spells only")
fit(dt[dt["spell_id"].isin(later_ids)], DUR + ["depth", "era_1023"],
    "event", "CAPITULATION, LATER spells only")
log.append("  reading: the duration gradient rising in BOTH strata is the "
           "transparent answer to unobserved-heterogeneity sorting.")

# interim write: nothing below can lose what is above
P.write_report("confidence_report_16b.txt", log)

# --------------------- (d2) random intercepts, 25% fund subsample ----
log.append("\n(d2) RANDOM-INTERCEPT MIXED MODEL, 25% deterministic subsample "
           "(wficn % 4 == 0)")
try:
    d = dt[["event", "wficn", "depth", "era_1023"] + DUR].dropna().copy()
    d = d[d["wficn"] % 4 == 0]
    d["wficn"] = d["wficn"].astype(str)
    log.append(f"  subsample: {len(d):,} spell-quarters, "
               f"{d['wficn'].nunique():,} funds, "
               f"{int(d['event'].sum()):,} events")
    mm = sm.BinomialBayesMixedGLM.from_formula(
        "event ~ " + " + ".join(DUR + ["depth", "era_1023"]),
        {"fund": "0 + C(wficn)"}, d).fit_vb()
    names = mm.model.exog_names
    log.append(f"  converged; fund-effect SD (posterior mean of vcp): "
               f"{float(np.exp(mm.vcp_mean[0])):.3f}")
    for n, b in zip(names, mm.fe_mean):
        if n != "Intercept":
            log.append(f"    {n:10s} OR {np.exp(b):7.3f}")
    log.append("  compare duration ORs to the stage-16 (c) HRs: still rising "
               "= fatigue survives fund-level unobserved heterogeneity.")
except Exception as e:  # noqa: BLE001
    log.append(f"  mixed model failed/skipped: {e}")
    log.append("  verdict rests on the spell-order strata above.")

log.append("\n16b DONE - local only.")
P.write_report("confidence_report_16b.txt", log)
print("\n".join(log))
