# 53_factor_adjusted_gradient.py -- is the gross gradient just a BETA
# gradient? This is the referee question for the whole mechanism claim.
#
# Bigger menus select more extreme trailing returns. Extreme trailing
# returns are partly high beta and high momentum exposure. If the gross
# edge falls with menu size only because the hot pick's beta rises, then
# "selection under skewness" is really "selection on factor loadings" and
# the paper must say so.
#
# Fama-French factors live in the BuyRisk library:
#   E:\Finance\BuyRisk\data\sources\french\F-F_Research_Data_Factors.csv
#   E:\Finance\BuyRisk\data\sources\french\F-F_Momentum_Factor.csv
#
#   python 53_factor_adjusted_gradient.py
import os, sys, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
BR = os.environ.get("BUYRISK_LIB", r"E:\Finance\BuyRisk\data\sources")
FF  = os.path.join(BR, "french", "F-F_Research_Data_Factors.csv")
MOM = os.path.join(BR, "french", "F-F_Momentum_Factor.csv")
KS, REPS, BOOT, SEED = (3, 5, 10, 15, 20), 120, 2000, 20260825
WINDOW = 60          # months of forward data used to estimate each alpha
rng = np.random.default_rng(SEED)

def read_ff(path, names):
    """French CSVs have a header block, a monthly table, then an annual table.
    Read the lines directly - pandas cannot be told to split on newlines."""
    rows = []
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < len(names) + 1:
                continue
            if not (parts[0].isdigit() and len(parts[0]) == 6):
                continue                      # 6 digits = YYYYMM (annual rows are 4)
            try:
                vals = [float(x) for x in parts[1:len(names) + 1]]
            except ValueError:
                continue
            if any(v <= -99.0 for v in vals):  # French missing code
                continue
            rows.append([int(parts[0])] + vals)
    if not rows:
        sys.exit(f"no monthly rows parsed from {path}")
    df = pd.DataFrame(rows, columns=["ym"] + names)
    df["ym"] = pd.PeriodIndex(pd.to_datetime(df.ym.astype(str), format="%Y%m"), freq="M")
    for c in names:
        df[c] = df[c] / 100.0
    return df.drop_duplicates("ym").set_index("ym")

for p in (FF, MOM):
    if not os.path.exists(p):
        sys.exit(f"MISSING factor file: {p}")
f3 = read_ff(FF, ["mktrf", "smb", "hml", "rf"])
mom = read_ff(MOM, ["umd"])
fac = f3.join(mom, how="left")
print(f"factors: {fac.index.min()} to {fac.index.max()}, {len(fac):,} months")

pn = pd.read_parquet(os.path.join(CACHE, "s45_panel_with_passive.parquet"))
pn = pn.dropna(subset=["ret"]).copy()
pn["ym"] = pd.PeriodIndex(pn.ym.astype(str), freq="M")
ft = pd.read_parquet(os.path.join(CACHE, "s52_formations_enriched.parquet"))
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[ft.cat != "UNK"]

# ---- forward alpha per fund-formation, CAPM and 4-factor ---------------
# Only fund-years that appear in a formation, and slice with searchsorted
# instead of slicing the DataFrame once per regression.
print(f"estimating forward {WINDOW}-month alphas")
pn = pn[pn.wficn.isin(ft.wficn.unique())].merge(fac, left_on="ym", right_index=True,
                                                how="inner")
pn["exret"] = pn.ret - pn.rf
pn = pn.sort_values(["wficn", "ym"]).reset_index(drop=True)

wf   = pn.wficn.values
ordn = pn.ym.apply(lambda p: p.ordinal).values
Y    = pn.exret.values
F4   = pn[["mktrf", "smb", "hml", "umd"]].values
uniq = np.unique(wf)
lo_i = np.searchsorted(wf, uniq, side="left")
hi_i = np.searchsorted(wf, uniq, side="right")
bounds = dict(zip(uniq, zip(lo_i, hi_i)))

recs = []
for w, y in ft[["wficn", "year"]].itertuples(index=False):
    b = bounds.get(w)
    if b is None: continue
    s0, s1 = b
    start = pd.Period(f"{y}-12", "M").ordinal
    i0 = s0 + np.searchsorted(ordn[s0:s1], start, side="right")
    i1 = s0 + np.searchsorted(ordn[s0:s1], start + WINDOW, side="right")
    n = i1 - i0
    if n < 36: continue
    yv, xv = Y[i0:i1], F4[i0:i1]
    ok = np.isfinite(yv) & np.isfinite(xv).all(axis=1)
    if ok.sum() < 36: continue
    yv, xv = yv[ok], xv[ok]
    one = np.ones((len(yv), 1))
    bc = np.linalg.lstsq(np.hstack([one, xv[:, :1]]), yv, rcond=None)[0]
    b4 = np.linalg.lstsq(np.hstack([one, xv]), yv, rcond=None)[0]
    recs.append((w, y, bc[0] * 12, b4[0] * 12, bc[1]))
a = pd.DataFrame(recs, columns=["wficn", "year", "capm", "ff4", "beta"])
ft = ft.merge(a, on=["wficn", "year"], how="left")
print(f"  alpha coverage: {100*ft.capm.notna().mean():.1f}% of formations")
ft.to_parquet(os.path.join(CACHE, "s53_formations_alpha.parquet"), index=False)

# ---- simulate menus, compare hot vs cheap on alpha and on beta ---------
res = []
for K in KS:
    for (y, ct), g in ft.dropna(subset=["capm"]).groupby(["year", "cat"]):
        if len(g) < K: continue
        w = g.tna.values / g.tna.values.sum(); idx = np.arange(len(g))
        for _ in range(REPS):
            m = g.iloc[rng.choice(idx, size=K, replace=False, p=w)]
            c, h = m.exp_ratio.idxmin(), m.trail12.idxmax()
            if c == h: continue
            res.append({"K": K, "year": y,
                        "d_capm": m.loc[h, "capm"] - m.loc[c, "capm"],
                        "d_ff4":  m.loc[h, "ff4"]  - m.loc[c, "ff4"]
                                  if m.loc[[c, h], "ff4"].notna().all() else np.nan,
                        "d_beta": m.loc[h, "beta"] - m.loc[c, "beta"],
                        "d_fee":  m.loc[h, "exp_ratio"] - m.loc[c, "exp_ratio"]})
r = pd.DataFrame(res)
r.to_parquet(os.path.join(CACHE, "s53_alpha_gradient.parquet"), index=False)
print(f"disagreeing menus: {len(r):,}")

def boot(d, col, scale=10000):
    py = d.dropna(subset=[col]).groupby("year")[col].median()
    if len(py) < 5: return None
    draws = [py.loc[rng.choice(py.index, len(py), replace=True)].median() * scale
             for _ in range(BOOT)]
    return py.median() * scale, *np.percentile(draws, [2.5, 97.5]), len(py)

for era, sub in [("ALL", r), ("POST-2000", r[r.year >= 2000])]:
    print("\n" + "=" * 78); print(f"{era}: hot minus cheap, by menu size"); print("=" * 78)
    rows = []
    for K, d in sub.groupby("K"):
        for col, lab, sc in [("d_capm", "CAPM alpha (bps/yr)", 10000),
                             ("d_ff4", "4-factor alpha (bps/yr)", 10000),
                             ("d_beta", "market beta", 1)]:
            b = boot(d, col, sc)
            if b: rows.append({"K": K, "measure": lab, "value": b[0],
                               "lo": b[1], "hi": b[2], "n_years": b[3],
                               "excl_zero": (b[1] < 0) == (b[2] < 0)})
    t = pd.DataFrame(rows)
    if len(t): print(t.round(2).to_string(index=False))

    print(f"\n{era} gradient, K=20 minus K=3:")
    for col, lab, sc in [("d_capm", "CAPM alpha", 10000),
                         ("d_ff4", "4-factor alpha", 10000),
                         ("d_beta", "market beta", 1)]:
        x = sub[sub.K == 3].dropna(subset=[col]).groupby("year")[col].median()
        z = sub[sub.K == 20].dropna(subset=[col]).groupby("year")[col].median()
        yrs = x.index.intersection(z.index)
        if len(yrs) < 5: continue
        diff = (z.loc[yrs] - x.loc[yrs]) * sc
        draws = [diff.loc[rng.choice(yrs, len(yrs), replace=True)].median() for _ in range(BOOT)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f"  {lab:16s}: {diff.median():+8.2f} [{lo:+8.2f}, {hi:+8.2f}]  "
              f"excludes zero: {'YES' if (lo<0)==(hi<0) else 'no'}")

print("""
PLAIN READING
  If the ALPHA gradient survives (CAPM and 4-factor both negative, CIs
  excluding zero), the mechanism is selection, not factor loading, and the
  paper's central claim holds after risk adjustment.
  If the alpha gradient vanishes while the BETA gradient is large and
  positive, then bigger menus simply select higher-beta funds, and the
  honest paper says the recency heuristic is a beta-escalation device.
  Both are publishable. Only one is the paper we think we have.
""")
