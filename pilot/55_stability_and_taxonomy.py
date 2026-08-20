# 55_stability_and_taxonomy.py -- two things must be settled before writing.
#
#   1. SEED STABILITY. Between stages 53 and 54 a point estimate moved from
#      -32.7 to -59.6 and a CI went from excluding zero to spanning it. Some
#      of that was a design fix (53 filtered each menu size separately, so
#      K=3 and K=20 came from different category-years; 54 requires the same
#      category-years for both, which is correct). But how much is just
#      simulation noise? Run the headline gradient under several seeds and
#      look at the spread. If point estimates swing by tens of bps across
#      seeds, REPS must rise until they do not.
#
#   2. TAXONOMY vs SAMPLE. Lipper class (160 classes) gives a gradient that
#      clears zero everywhere; crsp_obj_cd (24 codes) mostly does not. Two
#      candidate explanations, and they need separating:
#        (a) Lipper is FINER, so the cheapest and hottest fund in a sleeve
#            are genuinely comparable -- a better measurement of the same
#            thing.
#        (b) Requiring >=20 funds per sleeve-year selects a DIFFERENT set of
#            sleeve-years under each taxonomy, and the samples differ.
#      Fix: restrict to sleeve-years that qualify under BOTH, then compare.
#
#   python 55_stability_and_taxonomy.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
SEEDS = (11, 22, 33, 44, 55)
REPS_GRID = (200, 600)
BOOT = 2000
KBIG = 20
MEASURES = [("ff4", "4-factor alpha", 10000), ("capm", "CAPM alpha", 10000)]

ft = pd.read_parquet(os.path.join(CACHE, "s54_formations_lipper.parquet")) \
     if os.path.exists(os.path.join(CACHE, "s54_formations_lipper.parquet")) else None
if ft is None:
    ft = pd.read_parquet(os.path.join(CACHE, "s53_formations_alpha.parquet"))
    print("NOTE: lipper column not cached by 54; re-deriving is skipped.")
    print("      If 'lip' is absent, only the crsp_obj_cd arm will run.")
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
has_lip = "lip" in ft.columns and ft.lip.notna().any()
print(f"formations {len(ft):,}   lipper available: {has_lip}")


def sim(df, catcol, K, rng, reps, weighted=True, years=None, tag=""):
    """Vectorised menu draws.

    The slow way is to slice the DataFrame once per draw. Instead, for each
    sleeve-year we draw ALL `reps` menus at once with the Gumbel top-k trick:
    adding Gumbel noise to log weights and taking the top K gives weighted
    sampling WITHOUT replacement (Efraimidis-Spirakis). Everything after that
    is array indexing.
    """
    recs, done = [], 0
    groups = [(k, g) for k, g in df.groupby(["year", catcol]) if len(g) >= KBIG]
    if years is not None:
        groups = [(k, g) for k, g in groups if k[0] in years]
    for (y, ct), g in groups:
        n = len(g)
        fee = g.exp_ratio.to_numpy(float)
        tr = g.trail12.to_numpy(float)
        vals = {c: g[c].to_numpy(float) for c, _l, _s in MEASURES if c in g.columns}
        logw = (np.log(np.maximum(g.tna.to_numpy(float), 1e-12)) if weighted
                else np.zeros(n))
        keys = rng.gumbel(size=(reps, n)) + logw[None, :]
        idx = np.argpartition(-keys, K - 1, axis=1)[:, :K]
        rows = np.arange(reps)
        cheap = idx[rows, np.argmin(fee[idx], axis=1)]
        hot = idx[rows, np.argmax(tr[idx], axis=1)]
        keep = cheap != hot
        if not keep.any():
            continue
        rec = {"year": np.full(int(keep.sum()), y)}
        for c, arr in vals.items():
            rec[c] = arr[hot[keep]] - arr[cheap[keep]]
        recs.append(pd.DataFrame(rec))
        done += 1
    if tag:
        print(f"    {tag}: {done} sleeve-years", flush=True)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


def grad(d3, d20, col, rng, scale=10000):
    a = d3.dropna(subset=[col]).groupby("year")[col].median()
    b = d20.dropna(subset=[col]).groupby("year")[col].median()
    yrs = a.index.intersection(b.index)
    if len(yrs) < 5: return None
    diff = (b.loc[yrs] - a.loc[yrs]) * scale
    draws = [diff.loc[rng.choice(yrs, len(yrs), replace=True)].median() for _ in range(BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return diff.median(), lo, hi, len(yrs)


print("\n" + "=" * 88); print("1. SEED STABILITY (crsp_obj_cd, post-2000, TNA-weighted)")
print("=" * 88)
rows = []
post = ft[ft.year >= 2000]
for reps in REPS_GRID:
    for sd in SEEDS:
        rng = np.random.default_rng(sd)
        print(f"  reps={reps} seed={sd}", flush=True)
        d3 = sim(post, "cat", 3, rng, reps)
        d20 = sim(post, "cat", KBIG, rng, reps)
        for col, lab, sc in MEASURES:
            g = grad(d3, d20, col, rng, sc)
            if g: rows.append({"reps": reps, "seed": sd, "measure": lab,
                               "gradient_bps": g[0], "lo": g[1], "hi": g[2],
                               "excl_zero": (g[1] < 0) == (g[2] < 0)})
st = pd.DataFrame(rows)
st.round(1).to_csv(os.path.join(OUT, "s55_seed_stability.csv"), index=False)
print(st.round(1).to_string(index=False))
for (reps, lab), d in st.groupby(["reps", "measure"]):
    print(f"\n  reps={reps} {lab}: point estimates span "
          f"{d.gradient_bps.min():.1f} to {d.gradient_bps.max():.1f} bps "
          f"(range {d.gradient_bps.max()-d.gradient_bps.min():.1f}); "
          f"{d.excl_zero.sum()}/{len(d)} seeds exclude zero")

if has_lip:
    print("\n" + "=" * 88); print("2. TAXONOMY vs SAMPLE -- same sleeve-years under both")
    print("=" * 88)
    p = ft[ft.year >= 2000].dropna(subset=["lip"])
    ok_cat = {(y, c) for (y, c), g in p.groupby(["year", "cat"]) if len(g) >= KBIG}
    ok_lip = {(y, l) for (y, l), g in p.groupby(["year", "lip"]) if len(g) >= KBIG}
    yrs_cat = {y for y, _ in ok_cat}; yrs_lip = {y for y, _ in ok_lip}
    common = yrs_cat & yrs_lip
    print(f"  sleeve-years qualifying: crsp {len(ok_cat)}, lipper {len(ok_lip)}")
    print(f"  formation years in common: {len(common)}")
    q = p[p.year.isin(common)]
    for catcol, lab in [("cat", "crsp_obj_cd"), ("lip", "lipper")]:
        rng = np.random.default_rng(777)
        d3 = sim(q, catcol, 3, rng, 400, years=common, tag=f"{lab} K=3")
        d20 = sim(q, catcol, KBIG, rng, 400, years=common, tag=f"{lab} K=20")
        for col, mlab, sc in MEASURES:
            g = grad(d3, d20, col, rng, sc)
            if g:
                print(f"  {lab:12s} {mlab:16s}: {g[0]:+7.1f} [{g[1]:+7.1f}, {g[2]:+7.1f}]"
                      f"  n={g[3]} yrs  excludes zero: "
                      f"{'YES' if (g[1]<0)==(g[2]<0) else 'no'}")
    print("\n  Same years, same funds, same filter -- only the sleeve definition")
    print("  differs. If lipper still wins here, it is granularity, and the")
    print("  paper should argue for it on ex-ante grounds (finer peer groups,")
    print("  the practitioner standard) and report crsp as the conservative")
    print("  bound. If they converge, the earlier gap was sample selection.")

print("""
PLAIN READING
  Block 1 first. If point estimates swing widely across seeds at reps=200
  but settle at reps=600, then every earlier number was under-simulated and
  the final run must use the higher setting. If they swing at BOTH, the
  formation-year median is too noisy and the estimator needs rethinking,
  not more draws.
""")
