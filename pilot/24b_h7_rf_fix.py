"""Stage 24b: H7 CORRECTION — the risk-free double-subtraction bug.

Figure 6 exposed an internal contradiction: a -2.4%/yr spread alpha over
~30 years implies a cumulative spread near -70%, but the plotted cumulative
series ends near -2%. Diagnosis: the ff4 helper in stages 20 and 24
computed y = r - rf for EVERY series. That is correct for a single
portfolio (excess return) and WRONG for a long-short spread, where rf
cancels between legs; subtracting it anyway biases the spread alpha down
by roughly the sample-average T-bill rate.

This stage recomputes every spread with the correct specification
(y = spread, no rf subtraction) and prints the biased number beside it so
the audit trail is explicit. Group-level alphas (correctly excess) are
unchanged. The own-benchmark ruler in battery IV never subtracted rf and
was always correct (it showed a spread of zero).

Output: output/referee_24b_h7_fix.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["H7 CORRECTION - RF DOUBLE-SUBTRACTION", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

fm = PL.get_fund_monthly(log)
fm["m"] = fm["caldt"].dt.to_period("M")
fm["quarter"] = fm["caldt"].dt.to_period("Q")
cov = pd.read_parquet(P.CACHE / "covars.parquet")
cov["quarter"] = pd.PeriodIndex(cov["quarter"], freq="Q")
cov["exp_ratio"] = pd.to_numeric(cov["exp_ratio"], errors="coerce")
cov.loc[~cov["exp_ratio"].between(0, 0.05), "exp_ratio"] = np.nan
fm = fm.merge(cov[["wficn", "quarter", "exp_ratio"]],
              on=["wficn", "quarter"], how="left")
fm = fm.sort_values(["wficn", "caldt"])
fm["exp_ratio"] = fm.groupby("wficn")["exp_ratio"].transform(
    lambda s: s.ffill().bfill())
fm["exp_ratio"] = fm["exp_ratio"].fillna(fm["exp_ratio"].median())
fm["fret_g"] = fm["fret"] + fm["exp_ratio"] / 12

fac = PL.get_factors(log)
fac["m"] = fac["month"].dt.to_period("M")
FAC = fac.set_index("m")[["mktrf", "smb", "hml", "mom", "rf"]]


# audit fixes A1 + A4: deduped membership; entry quarters on the calendar-
# true clock (crossing stamp for capitulators, k-th OBSERVED underwater
# quarter for milestone formations), matching stage 26's FIXED convention.
def obs_q(w, start, k):
    g = PF.get(w)
    if g is None:
        return start + k
    qs = g.index[g.index >= start]
    return qs[k] if k < len(qs) else start + k


def calendar_port(ev, retcol):
    rows = []
    for _, s in ev.iterrows():
        m0 = s["entry_q"].asfreq("M", how="end") + 1
        rows += [(s["wficn"], m0 + k) for k in range(36)]
    mem = pd.DataFrame(rows, columns=["wficn", "m"]).drop_duplicates()
    d = mem.merge(fm[["wficn", "m", retcol]], on=["wficn", "m"], how="inner")
    g = d.groupby("m")[retcol].agg(["mean", "size"])
    return g[g["size"] >= 10]["mean"]


def ff4(r, excess):
    """FF4 alpha. excess=True subtracts rf (single portfolios);
    excess=False does not (long-short spreads)."""
    j = pd.concat([r.rename("r"), FAC], axis=1, join="inner").dropna()
    y = (j["r"] - j["rf"]).to_numpy() if excess else j["r"].to_numpy()
    X = sm.add_constant(j[["mktrf", "smb", "hml", "mom"]].to_numpy())
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return float(m.params[0]), float(m.bse[0]), len(j), j

def spread_report(folded, fighting, retcol, label):
    pf_ = calendar_port(folded, retcol)
    pg = calendar_port(fighting, retcol)
    if min(len(pf_), len(pg)) < 24:
        log.append(f"  {label}: too few portfolio months - skipped")
        return
    a1, s1, n1, _ = ff4(pf_, excess=True)
    a2, s2, n2, _ = ff4(pg, excess=True)
    sp_ = (pg - pf_).dropna()
    aC, sC, nC, j = ff4(sp_, excess=False)     # CORRECT
    aB, sB, nB, _ = ff4(sp_, excess=True)      # biased (old code)
    rfm = float(j["rf"].mean()) * 12
    log.append(f"\n  {label}")
    log.append(f"    folded   alpha {a1 * 12:+.2%}/yr (se {s1 * 12:.2%}, {n1}m)")
    log.append(f"    fighting alpha {a2 * 12:+.2%}/yr (se {s2 * 12:.2%}, {n2}m)")
    log.append(f"    SPREAD corrected: {aC * 12:+.2%}/yr (se {sC * 12:.2%}, "
               f"t {aC / sC:+.2f}) | MDE(80%) {2.80 * sC * 12:.2%}/yr")
    log.append(f"    [old, biased:     {aB * 12:+.2%}/yr - bias = mean rf "
               f"{rfm:.2%}/yr, as diagnosed]")

# ---------------- battery IV design (unmatched), corrected ----------------
def sect_battery4():
    caps = sp[sp["capitulated"]].copy()
    caps["entry_q"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
    res = sp[(sp["end_dur"] >= 8)
             & (sp["m_dur"].isna() | (sp["m_dur"] > 8))].copy()
    res["entry_q"] = [obs_q(w, s, 8)
                      for w, s in zip(res["wficn"], res["start_p"])]
    spread_report(caps, res, "fret", "UNMATCHED (battery IV design), NET")
    spread_report(caps, res, "fret_g", "UNMATCHED, GROSS")

# ---------------- matched milestones, corrected ----------------
def sect_matched():
    for K in (4, 8, 12):
        elig = sp[sp["end_dur"] >= K].copy()
        folded = elig[elig["m_dur"].notna() & (elig["m_dur"] <= K)].copy()
        fighting = elig[elig["m_dur"].isna() | (elig["m_dur"] > K)].copy()
        for g in (folded, fighting):
            g["entry_q"] = [obs_q(w, s, K)
                            for w, s in zip(g["wficn"], g["start_p"])]
        spread_report(folded, fighting, "fret", f"MATCHED K={K}, NET")
        if K == 8:
            spread_report(folded, fighting, "fret_g", "MATCHED K=8, GROSS")

R.section(log, "(a) BATTERY IV DESIGN, CORRECTED", sect_battery4)
R.section(log, "(b) MATCHED MILESTONES, CORRECTED", sect_matched)

log.append("""
Reading guide: the own-benchmark ruler already showed a spread of 0.00 and
was computed correctly all along; the corrected FF4 spreads should now be
judged against the MDE. Spread within +/- MDE of zero = Section 8 becomes
an adequately powered NULL (no conviction premium, no conviction penalty,
bounded within ~2%/yr), which is what the paper reports. Any residual
significant spread survives the correction and is real.""")
log.append("H7 CORRECTION DONE - aggregates only.")
P.write_report("referee_24b_h7_fix.txt", log)
print("\n".join(log))
