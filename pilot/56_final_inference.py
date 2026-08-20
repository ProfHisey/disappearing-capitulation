# 56_final_inference.py -- replace the estimator, not the compute budget.
#
# Stage 55 showed the four-factor gradient is precisely estimated (-27.2,
# -26.9, -26.6, -27.2, -25.5 across five seeds at 600 reps, a range of
# 1.7bp) but that its bootstrap CI is QUANTIZED: upper bounds recurred at
# exactly +6.8 and -7.3 across independent seeds, because a median of 23
# year-medians has a discrete sampling distribution and the 97.5th
# percentile keeps landing on one particular year. Significance was being
# decided by which side of zero a single year fell on.
#
# Fix: stop taking medians of medians. Build a PAIRED difference per
# sleeve-year (same sleeve, same year, K=20 menus minus K=3 menus), then do
# inference two ways that both handle few clusters honestly:
#
#   (1) Year-level t-test. Collapse to one number per formation year and run
#       a one-sample t-test on those ~23 observations. Simple, transparent,
#       and what a referee will replicate in their head.
#   (2) Wild cluster bootstrap by formation year (Rademacher weights) on the
#       sleeve-year observations. The standard fix when clusters are few
#       (Cameron, Gelbach & Miller). Reported alongside the naive
#       cluster-robust SE so the difference is visible.
#
#   python 56_final_inference.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
KSMALL, KBIG, REPS, WILD, SEED = 3, 20, 600, 9999, 20260827
MEASURES = [("ff4", "4-factor alpha", 10000), ("capm", "CAPM alpha", 10000),
            ("rein5", "raw net 5y (ann.)", 2000)]
rng = np.random.default_rng(SEED)

f = os.path.join(CACHE, "s54_formations_lipper.parquet")
ft = pd.read_parquet(f if os.path.exists(f)
                     else os.path.join(CACHE, "s53_formations_alpha.parquet"))
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
has_lip = "lip" in ft.columns and ft.lip.notna().any()
print(f"formations {len(ft):,}  lipper: {has_lip}")


def paired(df, catcol, index_free, weighted=True):
    """One paired K=20-minus-K=3 difference per sleeve-year."""
    rows = []
    for (y, ct), g in df.groupby(["year", catcol]):
        if index_free and "passive" in g:
            g = g[~g.passive]
        if len(g) < KBIG:
            continue
        n = len(g)
        fee = g.exp_ratio.to_numpy(float); tr = g.trail12.to_numpy(float)
        vals = {c: g[c].to_numpy(float) for c, _l, _s in MEASURES if c in g.columns}
        logw = (np.log(np.maximum(g.tna.to_numpy(float), 1e-12)) if weighted
                else np.zeros(n))
        rec = {"year": y, "sleeve": ct, "n_funds": n}
        for K, tag in ((KSMALL, "small"), (KBIG, "big")):
            keys = rng.gumbel(size=(REPS, n)) + logw[None, :]
            idx = np.argpartition(-keys, K - 1, axis=1)[:, :K]
            r = np.arange(REPS)
            cheap = idx[r, np.argmin(fee[idx], axis=1)]
            hot = idx[r, np.argmax(tr[idx], axis=1)]
            keep = cheap != hot
            for c, arr in vals.items():
                d = arr[hot[keep]] - arr[cheap[keep]]
                d = d[np.isfinite(d)]
                rec[f"{c}_{tag}"] = d.mean() if len(d) else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    for c, _l, _s in MEASURES:
        if f"{c}_big" in out:
            out[c] = out[f"{c}_big"] - out[f"{c}_small"]
    return out


def infer(d, col, scale, label):
    s = d.dropna(subset=[col])
    if s.year.nunique() < 5:
        return None
    # (1) year-level t-test
    ym = s.groupby("year")[col].mean() * scale
    n = len(ym); mean = ym.mean(); se = ym.std(ddof=1) / np.sqrt(n)
    tstat = mean / se if se > 0 else np.nan
    from math import sqrt
    crit = 2.0 + 2.0 / max(n - 6, 1)          # ~t(.975) for n in the 15-40 range
    lo_t, hi_t = mean - crit * se, mean + crit * se
    # (2) wild cluster bootstrap by year on sleeve-year observations
    v = s[col].to_numpy(float) * scale
    yrs = s.year.to_numpy()
    uy = np.unique(yrs)
    resid = v - v.mean()
    draws = np.empty(WILD)
    for b in range(WILD):
        w = rng.choice([-1.0, 1.0], size=len(uy))
        wmap = dict(zip(uy, w))
        vb = v.mean() + resid * np.array([wmap[y] for y in yrs])
        draws[b] = vb.mean()
    lo_w, hi_w = np.percentile(draws, [2.5, 97.5])
    return {"spec": label, "measure": col, "estimate": mean, "n_years": n,
            "n_sleeve_years": len(s), "t": tstat,
            "t_lo": lo_t, "t_hi": hi_t, "wild_lo": lo_w, "wild_hi": hi_w,
            "t_excl0": (lo_t < 0) == (hi_t < 0),
            "wild_excl0": (lo_w < 0) == (hi_w < 0)}


specs = [("cat", False, "crsp / all menus"), ("cat", True, "crsp / index-free")]
if has_lip:
    specs += [("lip", False, "lipper / all menus"), ("lip", True, "lipper / index-free")]

rows = []
for catcol, ifree, label in specs:
    df = ft.dropna(subset=[catcol])
    for sample, tag in [(df, "all"), (df[df.year >= 2000], "post-2000")]:
        p = paired(sample, catcol, ifree)
        if not len(p): continue
        for c, mlab, sc in MEASURES:
            if c not in p.columns: continue
            r = infer(p, c, sc, f"{label} [{tag}]")
            if r: rows.append(r)
    print(f"  done: {label}", flush=True)

res = pd.DataFrame(rows)
res.round(1).to_csv(os.path.join(OUT, "s56_final_inference.csv"), index=False)

print("\n" + "=" * 100)
print("K=20 MINUS K=3, PAIRED BY SLEEVE-YEAR (bps/yr)")
print("=" * 100)
show = res[["spec", "measure", "estimate", "n_years", "n_sleeve_years", "t",
            "t_lo", "t_hi", "t_excl0", "wild_lo", "wild_hi", "wild_excl0"]]
print(show.round(1).to_string(index=False))

print("""
PLAIN READING
  Compare t_lo/t_hi against wild_lo/wild_hi. If both intervals agree, the
  inference is not an artifact of the method and the estimate can be
  quoted. If the wild bootstrap is much wider, the few-cluster problem is
  real and the wild interval is the honest one to report.

  The estimate column should sit near -25 bps for the 4-factor spec, which
  is where stage 55 put it with a seed-to-seed range of 1.7bp. The question
  was never the point estimate; it was whether an interval built from 23
  formation years could support it.

  n_sleeve_years is the paired sample size. It is much larger than n_years,
  but the clusters are years, so the wild bootstrap is what governs.
""")
