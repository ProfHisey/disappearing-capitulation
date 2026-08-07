"""Stage 16c: recover sections (a)-(c) of the original stage 16.

The original 16_confidence_upgrades.py was killed during its slow mixed
model, and since it wrote its report only at the end, the finished early
sections were lost. 16b re-ran the frailty checks (d). This script re-runs
the other three, unchanged:

 (a) FAIRX INVESTIGATION - pipeline funnel for the famous funds.
 (b) FULL SPELL TRACES, untruncated.
 (c) DEPTH ENTERED FLEXIBLY - bins instead of a line, both outcomes.

No slow models anywhere; this one runs in minutes.
Output: output/confidence_report_16c.txt (aggregates + fund traces, local).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL

log = ["CONFIDENCE UPGRADES 16c (sections a-c)", "=" * 60]

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

# ------------------------------------------------- (a) pipeline funnel ----
log.append("\n(a) PIPELINE FUNNEL for the famous funds")
FUNDS = ["FMAGX", "LMVTX", "FAIRX", "SEQUX", "CGMFX", "VFINX"]
m1 = P.norm_cols(pd.read_csv(PL.MFLINK1))
m1["ticker"] = m1["ticker"].astype(str).str.strip().str.upper()
tmap = m1.dropna(subset=["wficn"]).drop_duplicates("ticker").set_index("ticker")

asp_raw = pd.read_parquet(P.CACHE / "as_panel.parquet").dropna(subset=["wficn"])
asp_raw["wficn"] = asp_raw["wficn"].astype("int64")
flags = pd.read_parquet(P.CACHE / "flags.parquet")
fm = PL.get_fund_monthly([])
panel_w = set(panel["wficn"].unique())

for tk in FUNDS:
    if tk not in tmap.index:
        log.append(f"  {tk}: not in mflink1")
        continue
    w = int(tmap.loc[tk, "wficn"])
    nd = asp_raw[asp_raw["wficn"] == w]
    fl = flags[flags["wficn"] == w]
    passive = bool(fl["passive"].any()) if len(fl) else None
    has_ret = w in set(fm["wficn"].unique())
    in_panel = w in panel_w
    log.append(f"  {tk} (wficn {w}): ND rows {len(nd):,}"
               f" | passive-flag {passive}"
               f" | returns {'Y' if has_ret else 'N'}"
               f" | final panel {'Y' if in_panel else 'N'}")
    if len(nd) and not in_panel and not passive:
        codes = nd["bench_min"].value_counts().head(3).to_dict()
        span = f"{nd['month'].min():%Y-%m} to {nd['month'].max():%Y-%m}"
        log.append(f"    dropped after ND: span {span}, top min-AS benchmarks "
                   f"{codes} (unmatched benchmark or failed returns/benchmark "
                   f"merge would explain the drop)")
    if len(nd) == 0:
        log.append("    -> absent from the ND Active Share data itself "
                   "(coverage, not a screen)")

# ------------------------------------------------------ (b) full traces ----
log.append("\n(b) FULL SPELL TRACES (untruncated)")
for tk in FUNDS:
    if tk not in tmap.index:
        continue
    w = int(tmap.loc[tk, "wficn"])
    ss = sp[sp["wficn"] == w]
    if not len(ss):
        continue
    log.append(f"  {tk}:")
    for _, s in ss.iterrows():
        out = ("CAPITULATED" if s["capitulated"] else
               "DIED" if s["spell_died"] else s["ended_by"])
        log.append(f"    {s['start_q']} -> {s['end_q']} "
                   f"({int(s['end_dur'])}q, depth {s['depth']:+.1%}): {out}")

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
            "event_die": int(bool(s["spell_died"]) and t == T),
            "era_1023": float((start + t).year >= 2010),
        })
dt = pd.DataFrame(rows)
dt["dur_3_4"] = dt["t"].between(3, 4).astype(float)
dt["dur_5_8"] = dt["t"].between(5, 8).astype(float)
dt["dur_9_12"] = dt["t"].between(9, 12).astype(float)
dt["dur_13p"] = (dt["t"] >= 13).astype(float)
dt["dep_5_15"] = dt["depth"].between(-0.15, -0.05, inclusive="left").astype(float)
dt["dep_15_25"] = dt["depth"].between(-0.25, -0.15, inclusive="left").astype(float)
dt["dep_25_40"] = dt["depth"].between(-0.40, -0.25, inclusive="left").astype(float)
dt["dep_40p"] = (dt["depth"] < -0.40).astype(float)
DUR = ["dur_3_4", "dur_5_8", "dur_9_12", "dur_13p"]
DEP = ["dep_5_15", "dep_15_25", "dep_25_40", "dep_40p"]

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

# ------------------------------------------------- (c) depth as bins ----
log.append("\n(c) DEPTH ENTERED FLEXIBLY (bins vs shallowest 0 to -5%)")
fit(dt, DUR + DEP + ["era_1023"], "event", "CAPITULATION hazard")
fit(dt, DUR + DEP + ["era_1023"], "event_die", "DEATH hazard")
log.append("  expected: capitulation HRs hump at dep_15_25; death HRs rise "
           "monotonically toward dep_40p.")

log.append("\n16c DONE - local only.")
P.write_report("confidence_report_16c.txt", log)
print("\n".join(log))
