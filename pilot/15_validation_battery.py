"""Stage 15: VALIDATION BATTERY — trust checks for the assembled pipeline.

Three checks, ordered by what failure mode they catch:

(A) FAMOUS-FUND TRACES. Funds with publicly known histories, located by ticker,
    traced through our panel: date range, Active Share path, spells, outcomes.
    If the panel retells verifiable stories (Legg Mason Value Trust's collapse,
    Magellan's index-hugging era, Fairholme's redemption run, Sequoia after
    Valeant), the joins are right. Vanguard 500 is the negative control: the
    passive screen should have removed it entirely.
(B) FINER BINS. Five-year cohorts for the era decline; depth deciles for the
    mode-selection flip (capitulation and death shares should run in opposite
    directions); quarter-by-quarter empirical hazard for the fatigue curve.
    Real trends survive re-binning; boundary artifacts don't.
(C) SPLIT-SAMPLE. Funds split by odd/even wficn (deterministic, reproducible).
    The depth sign flip and the era decline must appear in BOTH halves.

Output: output/validation_report.txt. Fund traces are brief summaries for
validation purposes; the report stays local like everything else.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL

log = ["VALIDATION BATTERY", "=" * 60]

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
sp["depth_final"] = sp["depth"]
sp["start_yr"] = sp["start_p"].dt.year

pf = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# ---------------------------------------------------- (A) famous funds ----
log.append("\n(A) FAMOUS-FUND TRACES")
FUNDS = {
    "FMAGX": "Fidelity Magellan (expect: huge TNA; low-AS/index-hugging era circa 1996-2005)",
    "LMVTX": "Legg Mason Value Trust (expect: streak through 2005, deep spell 2007-08+)",
    "FAIRX": "Fairholme (expect: deep spell + redemption run circa 2011)",
    "SEQUX": "Sequoia (expect: trouble after 2015, Valeant)",
    "CGMFX": "CGM Focus (expect: extreme swings, deep spells post-2008)",
    "VFINX": "Vanguard 500 NEGATIVE CONTROL (expect: ABSENT - passive screen)",
}
m1 = P.norm_cols(pd.read_csv(PL.MFLINK1))
m1["ticker"] = m1["ticker"].astype(str).str.strip().str.upper()
tmap = m1.dropna(subset=["wficn"]).drop_duplicates("ticker").set_index("ticker")

for tk, expect in FUNDS.items():
    log.append(f"\n  {tk}: {expect}")
    if tk not in tmap.index:
        log.append("    ticker not in mflink1 - no trace possible")
        continue
    w = int(tmap.loc[tk, "wficn"])
    g = pf.get(w)
    if g is None:
        log.append(f"    wficn {w} NOT IN PANEL"
                   + (" (as expected)" if tk == "VFINX" else " - INVESTIGATE"))
        continue
    asx = g["as_min"].dropna()
    log.append(f"    in panel {g.index.min()} to {g.index.max()} "
               f"({len(g)} quarters); AS first {asx.iloc[0]:.2f}, "
               f"min {asx.min():.2f} (at {asx.idxmin()}), last {asx.iloc[-1]:.2f}")
    wr = g["rel4q"].dropna()
    if len(wr):
        log.append(f"    worst trailing 4q vs benchmark: {wr.min():+.1%} at {wr.idxmin()}")
    fl = g["flowq"].dropna()
    if len(fl):
        log.append(f"    worst retail flow quarter: {fl.min():+.1%} at {fl.idxmin()}")
    ss = sp[sp["wficn"] == w]
    for _, s in ss.head(6).iterrows():
        out = ("CAPITULATED" if s["capitulated"] else
               "DIED" if s["spell_died"] else s["ended_by"])
        log.append(f"    spell {s['start_q']} -> {s['end_q']} "
                   f"({int(s['end_dur'])}q, depth {s['depth_final']:+.1%}): {out}")
    if not len(ss):
        log.append("    no spells (never simultaneously active>=70% and underwater)")

# -------------------------------------------------------- (B) finer bins ----
log.append("\n(B1) ERA DECLINE, five-year cohorts (spell capitulation / death rates):")
sp["cohort5"] = (sp["start_yr"] // 5) * 5
for c, s in sp.groupby("cohort5"):
    log.append(f"  {int(c)}-{int(c)+4}: {len(s):5,} spells | "
               f"capitulated {s['capitulated'].mean():6.2%} | "
               f"died {s['spell_died'].mean():6.2%}")

log.append("\n(B2) MODE SELECTION, depth deciles (opposite gradients expected):")
sp["ddec"] = pd.qcut(sp["depth_final"], 10, labels=False, duplicates="drop")
for d, s in sp.groupby("ddec"):
    log.append(f"  decile {int(d)} (mean depth {s['depth_final'].mean():+.1%}, "
               f"n={len(s):,}): capitulated {s['capitulated'].mean():6.2%} | "
               f"died {s['spell_died'].mean():6.2%}")
log.append("  (decile 0 = deepest underperformance)")

log.append("\n(B3) FATIGUE, quarter-by-quarter empirical capitulation hazard:")
maxd = 16
at_risk = {t: 0 for t in range(1, maxd + 2)}
ev = {t: 0 for t in range(1, maxd + 2)}
for _, s in sp.iterrows():
    T = int(s["m_dur"]) if s["capitulated"] else int(s["end_dur"])
    for t in range(1, min(T, maxd + 1) + 1):
        at_risk[min(t, maxd + 1)] += 1
    if s["capitulated"] and s["m_dur"] <= maxd:
        ev[int(s["m_dur"])] += 1
    elif s["capitulated"]:
        ev[maxd + 1] += 1
for t in range(1, maxd + 1):
    h = ev[t] / at_risk[t] if at_risk[t] else np.nan
    log.append(f"  q{t:2d}: at-risk {at_risk[t]:6,}  events {ev[t]:3d}  "
               f"hazard {h:7.3%}")

# ------------------------------------------------------ (C) split-sample ----
log.append("\n(C) SPLIT-SAMPLE (odd vs even wficn), slim reduced form:")
rows = []
for _, s in sp.iterrows():
    w = s["wficn"]
    g = pf.get(w)
    if g is None:
        continue
    T = int(s["m_dur"]) if s["capitulated"] else int(s["end_dur"])
    start = s["start_p"]
    depth_so_far = 0.0
    for t in range(1, max(T, 1) + 1):
        rl = g.at[start + (t - 1), "rel4q"] if (start + (t - 1)) in g.index else np.nan
        if pd.notna(rl):
            depth_so_far = min(depth_so_far, float(rl))
        rows.append({
            "wficn": w, "t": t, "depth": depth_so_far,
            "event": int(s["capitulated"] and t == int(s["m_dur"])),
            "event_die": int(bool(s["spell_died"]) and t == T),
            "dur_5p": float(t >= 5),
            "era_1023": float((start + t).year >= 2010),
            "half": "even" if w % 2 == 0 else "odd",
        })
dtv = pd.DataFrame(rows)

def slim_fit(df, ycol, label):
    xcols = ["dur_5p", "depth", "era_1023"]
    d = df[[ycol, "wficn"] + xcols].dropna()
    y = d[ycol].to_numpy(float)
    X = sm.add_constant(d[xcols].to_numpy(float))
    try:
        m = sm.GLM(y, X, family=sm.families.Binomial(
            link=sm.families.links.CLogLog())).fit(
            cov_type="cluster", cov_kwds={"groups": d["wficn"].to_numpy()})
        names = ["const"] + xcols
        msg = "  ".join(f"{n}: coef {b:+.2f} (z {z:.1f})"
                        for n, b, z in zip(names, m.params, m.tvalues)
                        if n != "const")
        log.append(f"  {label} [n={len(d):,}, ev={int(y.sum()):,}]  {msg}")
        return dict(zip(names, m.params))
    except Exception as e:  # noqa: BLE001
        log.append(f"  {label}: FIT FAILED ({e})")
        return {}

for half in ("odd", "even"):
    dd = dtv[dtv["half"] == half]
    c = slim_fit(dd, "event", f"{half} half, CAPITULATION")
    d_ = slim_fit(dd, "event_die", f"{half} half, DEATH")
    if c and d_:
        flip = np.sign(c.get("depth", 0)) != np.sign(d_.get("depth", 0))
        log.append(f"    -> depth sign flip between outcomes in {half} half: "
                   f"{'YES' if flip else 'NO'}")

log.append("\nReading guide: (A) any trace contradicting public history means a "
           "join/timing bug - stop and investigate before anything else. "
           "(B1) decline should be roughly monotone across 8 cohorts. "
           "(B2) capitulation share should FALL and death share RISE toward "
           "deeper deciles. (B3) hazard should rise roughly smoothly with t. "
           "(C) the flip and era signs must hold in both halves.")
log.append("VALIDATION BATTERY DONE - aggregates + brief fund traces, local only.")
P.write_report("validation_report.txt", log)
print("\n".join(log))
