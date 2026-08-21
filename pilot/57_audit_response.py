# 57_audit_response.py -- answers the three findings that decide what the
# paper is. Run this before anything else in the audit queue.
#
# F1. CRSP RETURNS ARE NET OF FEES, so capm/ff4 are NET alphas and
#         net gradient = gross gradient - [fee_gap(K=20) - fee_gap(K=3)]
#     The bracket is mechanically negative: the cheapest of 20 draws is far
#     cheaper than the cheapest of 3 (a minimum order statistic), while the
#     hot pick's fee does not fall with K. Stage 52 added fee_pen back;
#     stages 53-56 did not. This script reports NET, FEE-GAP and GROSS
#     separately. The GROSS gradient is the paper's actual claim.
#
# F2. OVERLAPPING OUTCOME WINDOWS. Forward 60-month alphas from consecutive
#     formation years share 48 of 60 months, but every interval so far
#     treated formation years as independent. Reported here three ways:
#     plain t, Newey-West HAC (lag 4), and a NON-OVERLAPPING subsample
#     (every 5th formation year), which is what a 60-month horizon really
#     supports.
#
# TIES. Every earlier stage dropped menus where the cheapest fund IS the
#     hottest (`if c == h: continue`). Those are not missing data - the
#     difference is exactly zero. Conditioning on disagreement changes the
#     estimand and the conditioning rate differs sharply by K. This reports
#     both A(K) (conditional, what we had) and D(K) (unconditional, what a
#     participant actually faces), plus the two-channel decomposition.
#
#   python 57_audit_response.py
import os, sys, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
KS, REPS, KBIG, NW_LAG = (3, 20), 600, 20, 4
MEAS = [("ff4", "4-factor"), ("capm", "CAPM")]

src = os.path.join(CACHE, "s54_formations_lipper.parquet")
if not os.path.exists(src):
    src = os.path.join(CACHE, "s53_formations_alpha.parquet")
    print("NOTE: lipper cache absent; crsp sleeves only")
ft = pd.read_parquet(src)
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
assert not ft.duplicated(["wficn", "year"]).any(), "duplicate fund-years in formations"
has_lip = "lip" in ft.columns and ft.lip.notna().any()
print(f"formations {len(ft):,}   lipper: {has_lip}")


def draws(df, catcol, K, index_free, seed):
    """One row per simulated menu. TIES KEPT, with differences set to 0."""
    rng = np.random.default_rng(seed)
    recs = []
    for (y, ct), g in df.groupby(["year", catcol]):
        if index_free:
            g = g[~g.passive.astype(bool)]
        if len(g) < KBIG:
            continue
        n = len(g)
        fee = g.exp_ratio.to_numpy(float)
        tr = g.trail12.to_numpy(float)
        vals = {c: g[c].to_numpy(float) for c, _ in MEAS}
        logw = np.log(np.maximum(g.tna.to_numpy(float), 1e-12))
        keys = rng.gumbel(size=(REPS, n)) + logw[None, :]
        idx = np.argpartition(-keys, K - 1, axis=1)[:, :K]
        rows = np.arange(REPS)
        cheap = idx[rows, np.argmin(fee[idx], axis=1)]
        hot = idx[rows, np.argmax(tr[idx], axis=1)]
        tie = cheap == hot
        rec = {"year": np.full(REPS, y), "tie": tie,
               "fee_pen": np.where(tie, 0.0, fee[hot] - fee[cheap])}
        for c, _ in MEAS:
            d = vals[c][hot] - vals[c][cheap]
            rec[f"net_{c}"] = np.where(tie, 0.0, d)
        recs.append(pd.DataFrame(rec))
    d = pd.concat(recs, ignore_index=True)
    for c, _ in MEAS:                       # gross = net + the fee handicap
        d[f"gross_{c}"] = d[f"net_{c}"] + d.fee_pen
    return d


def nw_se(x, lag):
    """Newey-West HAC standard error of a sample mean."""
    x = np.asarray(x, float); T = len(x); e = x - x.mean()
    s = (e @ e) / T
    for k in range(1, min(lag, T - 1) + 1):
        s += 2 * (1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / T
    return np.sqrt(max(s, 0) / T)


def series(d, col, conditional):
    """Year-level means. conditional=True drops ties (the old estimand)."""
    dd = d[~d.tie] if conditional else d
    return dd.groupby("year")[col].mean() * 10000


def report(name, s3, s20, label):
    yrs = s3.index.intersection(s20.index)
    g = (s20.loc[yrs] - s3.loc[yrs]).dropna()
    T = len(g)
    m = g.mean()
    se_iid = g.std(ddof=1) / np.sqrt(T)
    se_nw = nw_se(g.values, NW_LAG)
    ac1 = pd.Series(g.values).autocorr(1) if T > 3 else np.nan
    non = g.iloc[::5]                       # every 5th year: no window overlap
    se_non = non.std(ddof=1) / np.sqrt(len(non)) if len(non) > 2 else np.nan
    print(f"  {label:<10s} {m:+8.1f} bps | t(iid) {m/se_iid:+5.2f} "
          f"| t(NW{NW_LAG}) {m/se_nw:+5.2f} "
          f"| AC1 {ac1:+.2f} "
          f"| non-overlap n={len(non)} t {m/se_non if se_non else float('nan'):+5.2f}")
    return {"spec": name, "component": label, "gradient_bps": m, "n_years": T,
            "t_iid": m / se_iid, "t_nw": m / se_nw, "ac1": ac1,
            "n_nonoverlap": len(non), "t_nonoverlap": m / se_non if se_non else np.nan}


specs = [("crsp", "cat", False), ("crsp", "cat", True)]
if has_lip:
    specs += [("lipper", "lip", False), ("lipper", "lip", True)]

rows, ties = [], []
for sleeve, catcol, index_free in specs:
    df = ft.dropna(subset=[catcol])
    for sample, sub in [("all", df), ("post-2000", df[df.year >= 2000])]:
        d3 = draws(sub, catcol, 3, index_free, 101)
        d20 = draws(sub, catcol, KBIG, index_free, 202)
        name = f"{sleeve}/{'index-free' if index_free else 'all menus'} [{sample}]"
        t3, t20 = d3.tie.mean() * 100, d20.tie.mean() * 100
        ties.append({"spec": name, "tie_pct_K3": t3, "tie_pct_K20": t20})
        print("\n" + "=" * 96)
        print(f"{name}   ties: {t3:.1f}% at K=3, {t20:.1f}% at K=20")
        print("=" * 96)
        for c, lab in MEAS:
            print(f"  -- {lab} --")
            for comp, col, cond in [("A net", f"net_{c}", True),
                                    ("A gross", f"gross_{c}", True),
                                    ("D net", f"net_{c}", False),
                                    ("D gross", f"gross_{c}", False)]:
                rows.append(report(name, series(d3, col, cond),
                                   series(d20, col, cond), f"{lab[:4]} {comp}"))
            # the mechanical piece, on its own
            rows.append(report(name, series(d3, "fee_pen", False),
                               series(d20, "fee_pen", False), "fee-gap D"))
            break   # fee-gap is measure-independent; print once

        for c, lab in MEAS[1:]:
            print(f"  -- {lab} --")
            for comp, col, cond in [("A net", f"net_{c}", True),
                                    ("A gross", f"gross_{c}", True),
                                    ("D net", f"net_{c}", False),
                                    ("D gross", f"gross_{c}", False)]:
                rows.append(report(name, series(d3, col, cond),
                                   series(d20, col, cond), f"{lab[:4]} {comp}"))

res = pd.DataFrame(rows)
res.round(2).to_csv(os.path.join(OUT, "s57_audit_response.csv"), index=False)
pd.DataFrame(ties).round(1).to_csv(os.path.join(OUT, "s57_tie_rates.csv"), index=False)

print("""
================================================================================
PLAIN READING
================================================================================
  Read the GROSS rows. That is the paper's claim: does the recency rule
  degrade with menu size for reasons other than the fee it happens to pay?
    gross clearly negative, t(NW) surviving -> selection under skewness,
        the title stands, and the fee is a separate additive cost.
    gross near zero -> the whole effect was the fee-gap arithmetic, the
        mechanism claim dies, and the honest paper is about fee dispersion
        across menu sizes. Still true, still useful, different paper.

  Then compare t(iid) with t(NW4). Every interval reported before today was
  the iid one, and 60-month windows from adjacent formation years share 48
  months. If t(NW4) is roughly half of t(iid), that is the overlap showing
  up, and the non-overlap column is the floor the design really supports.

  Then compare A rows with D rows. A conditions on the two rules disagreeing;
  D is what a participant faces, counting the times both rules point at the
  same fund. D is the honest headline. The gap between them is the second
  channel of the mechanism: on a short menu the recency rule accidentally
  lands on the cheap fund a third of the time, and on a long menu it almost
  never does.
""")
