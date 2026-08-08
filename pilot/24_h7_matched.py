"""Stage 24: H7 MATCHED-DURATION CHECK — same milestone, fold vs fight.

The battery-IV result (resisters trail capitulators by ~2.4%/yr FF4-adjusted)
compared groups formed at different spell moments: capitulators enter at
their crossing (often quarters 2-6), resisters at quarter 8. This rerun
removes that asymmetry. Stand at quarter K of every spell that lasts at
least K quarters and split the funds standing there:

    FOLDED-BY-K:  crossed below 60% Active Share at or before quarter K
    FIGHTING-AT-K: still above 60% at quarter K

Both groups enter the calendar-time portfolio the month after quarter K of
their own spell, so time-in-spell at formation is identical by construction.
36-month hold, equal weight, >=10 members/month, FF4 adjustment, HAC errors.
Run at K = 4, 8, 12. At K = 8, also: gross returns, and a depth split at
the milestone (shallow vs deep at K), since the folded group can still
differ in how far underwater it is.

Reading: if the FIGHTING group still underperforms by ~2%/yr at the same
milestone, the battery-IV result hardens into 'folding beat fighting.' If
the spread collapses toward zero, the original result was formation-state
composition and the paper reports it as such.

Output: output/referee_24_h7_matched.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["H7 MATCHED-DURATION CHECK", "=" * 60]

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


def depth_at(s, K):
    g = PF.get(s["wficn"])
    if g is None:
        return np.nan
    vals = [g.at[s["start_p"] + t, "rel4q"]
            for t in range(0, K + 1) if (s["start_p"] + t) in g.index]
    vals = [float(v) for v in vals if pd.notna(v)]
    return min(vals) if vals else np.nan


def calendar_port(ev, retcol):
    rows = []
    for _, s in ev.iterrows():
        m0 = s["entry_q"].asfreq("M", how="end") + 1
        rows += [(s["wficn"], m0 + k) for k in range(36)]
    mem = pd.DataFrame(rows, columns=["wficn", "m"])
    d = mem.merge(fm[["wficn", "m", retcol]], on=["wficn", "m"], how="inner")
    g = d.groupby("m")[retcol].agg(["mean", "size"])
    return g[g["size"] >= 10]["mean"]


def ff4(r):
    j = pd.concat([r.rename("r"), FAC], axis=1, join="inner").dropna()
    y = (j["r"] - j["rf"]).to_numpy()
    X = sm.add_constant(j[["mktrf", "smb", "hml", "mom"]].to_numpy())
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return float(m.params[0]), float(m.bse[0]), len(j)


def spread_line(folded, fighting, retcol, label):
    pf_ = calendar_port(folded, retcol)
    pg = calendar_port(fighting, retcol)
    for nm, r in (("folded", pf_), ("fighting", pg)):
        if len(r) < 24:
            log.append(f"    {label} {nm}: only {len(r)} portfolio months - "
                       f"skipped")
            return
    a1, s1, n1 = ff4(pf_)
    a2, s2, n2 = ff4(pg)
    sp_ = (pg - pf_).dropna()
    a, s, n = ff4(sp_)
    log.append(f"    {label}: folded alpha {a1 * 12:+.2%}/yr (se {s1 * 12:.2%},"
               f" {n1}m) | fighting {a2 * 12:+.2%}/yr (se {s2 * 12:.2%}, {n2}m)")
    log.append(f"      SPREAD (fight - fold): {a * 12:+.2%}/yr "
               f"(se {s * 12:.2%}, t {a / s:+.2f}, {n}m) | "
               f"MDE(80%) {2.80 * s * 12:.2%}/yr")


for K in (4, 8, 12):
    def run(K=K):
        elig = sp[sp["end_dur"] >= K].copy()
        folded = elig[elig["m_dur"].notna() & (elig["m_dur"] <= K)].copy()
        fighting = elig[elig["m_dur"].isna() | (elig["m_dur"] > K)].copy()
        for g in (folded, fighting):
            g["entry_q"] = g["start_p"] + K
        d_f = folded.apply(lambda s: depth_at(s, K), axis=1) if len(folded) else pd.Series(dtype=float)
        d_g = fighting.apply(lambda s: depth_at(s, K), axis=1) if len(fighting) else pd.Series(dtype=float)
        log.append(f"\n  milestone K={K}q: folded-by-K {len(folded):,} "
                   f"(mean depth at K {d_f.mean():+.1%}) | fighting-at-K "
                   f"{len(fighting):,} (mean depth {d_g.mean():+.1%})")
        spread_line(folded, fighting, "fret", f"K={K} NET")
        if K == 8:
            spread_line(folded, fighting, "fret_g", "K=8 GROSS")
            folded["d8"], fighting["d8"] = d_f, d_g
            for lo, hi, tag in ((-0.15, 0.0, "shallow at K (0 to -15%)"),
                                (-9.9, -0.15, "deep at K (beyond -15%)")):
                fsub = folded[folded["d8"].between(lo, hi)]
                gsub = fighting[fighting["d8"].between(lo, hi)]
                log.append(f"    depth stratum: {tag} - folded {len(fsub):,},"
                           f" fighting {len(gsub):,}")
                spread_line(fsub, gsub, "fret", f"K=8 NET, {tag}")
    R.section(log, f"MILESTONE K = {K} QUARTERS", run)

log.append("""
Reading guide: battery IV's unmatched spread was -2.38%/yr net (fight minus
fold, t -3.99). If the matched spreads at K=8 land in the same range with
usable power, formation-state composition is excluded and the paper's
Section 8 claim stands as written. Attenuation toward zero means the
original spread partly reflected WHERE in their spells the two groups stood,
and Section 8's magnitude gets restated from this table.""")
log.append("MATCHED-DURATION CHECK DONE - aggregates only.")
P.write_report("referee_24_h7_matched.txt", log)
print("\n".join(log))
