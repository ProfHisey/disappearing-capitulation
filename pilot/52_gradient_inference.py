# 52_gradient_inference.py (v2) -- the menu-size gradient is the headline,
# so it needs its own confidence interval and a wider range of K.
#
# v2 FIX: stage 49 built the long-horizon return columns in memory and only
# saved the menu-level draws, so s45_formations.parquet has no rein5/rein10.
# v1 skipped them silently and then died. This rebuilds them, caches the
# enriched table, and ASSERTS before proceeding.
#
#   python 52_gradient_inference.py
import os, sys, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
KS, REPS, BOOT, SEED = (3, 5, 10, 15, 20), 120, 2000, 20260824
HORIZONS = (5, 10)
rng = np.random.default_rng(SEED)
ENRICHED = os.path.join(CACHE, "s52_formations_enriched.parquet")

# ---------- rebuild the long-horizon columns (stage 49 logic) -----------
if os.path.exists(ENRICHED):
    ft = pd.read_parquet(ENRICHED); print("formations: using enriched cache")
else:
    print("rebuilding long-horizon returns (once)")
    pn = pd.read_parquet(os.path.join(CACHE, "s45_panel_with_passive.parquet"))
    pn = pn.dropna(subset=["ret"]).sort_values(["wficn", "ym"])
    pn["cat"] = pn["cat"].fillna("UNK")
    pn["cum"] = (1 + pn.ret).groupby(pn.wficn).cumprod()
    catm = pn.groupby(["cat", "ym"], observed=True)["ret"].mean().reset_index()
    catm["ccum"] = (1 + catm.ret).groupby(catm.cat).cumprod()
    dec  = pn[pn.ym.dt.month == 12][["wficn", "year", "cat", "cum"]]
    cd   = catm[catm.ym.dt.month == 12][["cat", "ccum"]].assign(
             year=catm.loc[catm.ym.dt.month == 12, "ym"].dt.year.values)
    last = pn.groupby("wficn").agg(last_year=("year", "max"), last_cum=("cum", "last"),
                                   last_cat=("cat", "last")).reset_index()
    last = last.merge(cd.rename(columns={"year": "last_year", "ccum": "ccum_at_death",
                                         "cat": "last_cat"}),
                      on=["last_cat", "last_year"], how="left")

    def forward(h):
        a = dec.rename(columns={"cum": "cum0"})
        b = dec[["wficn", "year", "cum"]].rename(columns={"cum": "cumH"})
        b["year"] -= h
        m = a.merge(b, on=["wficn", "year"], how="left")
        m = m.merge(last[["wficn", "last_year", "last_cum", "last_cat", "ccum_at_death"]],
                    on="wficn", how="left")
        m["tgt_year"] = m.year + h
        m = m.merge(cd.rename(columns={"year": "tgt_year", "ccum": "ccum_tgt",
                                       "cat": "last_cat"}),
                    on=["last_cat", "tgt_year"], how="left")
        m[f"rein{h}"] = m.cumH / m.cum0 - 1
        dead = m.cumH.isna() & m.ccum_at_death.notna() & m.ccum_tgt.notna()
        m.loc[dead, f"rein{h}"] = (m.loc[dead, "last_cum"] / m.loc[dead, "cum0"]) * \
                                  (m.loc[dead, "ccum_tgt"] / m.loc[dead, "ccum_at_death"]) - 1
        return m[["wficn", "year", f"rein{h}"]]

    ft = pd.read_parquet(os.path.join(CACHE, "s45_formations.parquet"))
    ft = ft[["wficn", "year", "cat", "exp_ratio", "trail12", "tna", "passive"]]
    for h in HORIZONS:
        ft = ft.merge(forward(h), on=["wficn", "year"], how="left")
    ft.to_parquet(ENRICHED, index=False)

need = [f"rein{h}" for h in HORIZONS]
missing = [c for c in need if c not in ft.columns]
if missing:
    sys.exit(f"missing columns after rebuild: {missing}")
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[ft.cat != "UNK"]
for h in HORIZONS:
    print(f"  rein{h} coverage {100*ft[f'rein{h}'].notna().mean():5.1f}%")
print(f"formations {len(ft):,}")

# ---------- simulate ----------------------------------------------------
res = []
for K in KS:
    for (y, ct), g in ft.groupby(["year", "cat"]):
        if len(g) < K: continue
        w = g.tna.values / g.tna.values.sum(); idx = np.arange(len(g))
        for _ in range(REPS):
            m = g.iloc[rng.choice(idx, size=K, replace=False, p=w)]
            c, h = m.exp_ratio.idxmin(), m.trail12.idxmax()
            if c == h: continue
            row = {"K": K, "year": y,
                   "fee_pen": m.loc[h, "exp_ratio"] - m.loc[c, "exp_ratio"]}
            for hz in HORIZONS:
                col = f"rein{hz}"
                if m.loc[[c, h], col].notna().all():
                    row[f"net{hz}"] = (m.loc[h, col] - m.loc[c, col]) / hz
                    row[f"gross{hz}"] = row[f"net{hz}"] + row["fee_pen"]
            res.append(row)
r = pd.DataFrame(res)
for hz in HORIZONS:
    for kind in ("net", "gross"):
        if f"{kind}{hz}" not in r.columns: r[f"{kind}{hz}"] = np.nan
r.to_parquet(os.path.join(CACHE, "s52_gradient.parquet"), index=False)
print(f"disagreeing menus: {len(r):,}   K range {KS}")

def boot(d, col):
    py = d.dropna(subset=[col]).groupby("year")[col].median()
    if len(py) < 5: return None
    draws = [py.loc[rng.choice(py.index, len(py), replace=True)].median() * 10000
             for _ in range(BOOT)]
    return py.median() * 10000, *np.percentile(draws, [2.5, 97.5]), len(py)

for era_name, sub in [("ALL FORMATIONS", r), ("POST-2000", r[r.year >= 2000])]:
    print("\n" + "=" * 78); print(f"{era_name} -- level at each menu size (bps/yr)")
    print("=" * 78)
    rows = []
    for hz in HORIZONS:
        for K, d in sub.groupby("K"):
            for kind in ("gross", "net"):
                b = boot(d, f"{kind}{hz}")
                if b: rows.append({"horizon": hz, "K": K, "measure": kind,
                                   "bps": b[0], "lo": b[1], "hi": b[2], "n_years": b[3],
                                   "excl_zero": (b[1] < 0) == (b[2] < 0)})
    t = pd.DataFrame(rows)
    if len(t):
        print(t.round(1).to_string(index=False))
        t.round(1).to_csv(os.path.join(OUT,
            f"s52_levels_{era_name.split()[0].lower()}.csv"), index=False)

    print(f"\n{era_name} -- THE GRADIENT: K=20 minus K=3")
    for hz in HORIZONS:
        for kind in ("gross", "net"):
            a = sub[sub.K == 3].dropna(subset=[f"{kind}{hz}"]).groupby("year")[f"{kind}{hz}"].median()
            b2 = sub[sub.K == 20].dropna(subset=[f"{kind}{hz}"]).groupby("year")[f"{kind}{hz}"].median()
            yrs = a.index.intersection(b2.index)
            if len(yrs) < 5: continue
            diff = (b2.loc[yrs] - a.loc[yrs]) * 10000
            draws = [diff.loc[rng.choice(yrs, len(yrs), replace=True)].median()
                     for _ in range(BOOT)]
            lo, hi = np.percentile(draws, [2.5, 97.5])
            print(f"  {hz:2d}y {kind:5s}: {diff.median():+7.1f} bps "
                  f"[{lo:+7.1f}, {hi:+7.1f}]  n={len(yrs)} yrs  "
                  f"excludes zero: {'YES' if (lo<0)==(hi<0) else 'no'}")

print("""
PLAIN READING
  Does the GROSS edge fall with menu size, CI excluding zero? That is
  selection under skewness, independent of fees, and a contribution
  whatever fee levels do next.
  Does the NET gradient survive post-2000? If yes, a live recommendation
  for plan sponsors. If only gross survives, this is a mechanism paper
  with a policy implication, not a policy paper with a mechanism.
""")
