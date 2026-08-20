# 50_long_horizon_robustness.py -- can the long-horizon result carry weight?
#
# Stage 49 gave a $9,537 median gap on $10k over 20 years. Before that number
# goes anywhere, three things have to be checked, because they compound:
#
#   A. OVERLAP. 20-year windows exist only for formations 1990-2005. Sixteen
#      formation years, almost fully overlapping. The effective sample is a
#      handful of independent draws, not 176,128 menus. Block bootstrap BY
#      FORMATION YEAR is the only honest inference here.
#   B. ERA. If the whole result comes from formations in one decade, it is an
#      era study wearing a horizon costume.
#   C. ATTRITION. 70% of formations have no 20-year window. If the missing
#      ones differ systematically, the survivors are not the population.
#
#   python 50_long_horizon_robustness.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
BOOT, SEED = 2000, 20260822
rng = np.random.default_rng(SEED)

r  = pd.read_parquet(os.path.join(CACHE, "s49_long_horizon.parquet"))
ft = pd.read_parquet(os.path.join(CACHE, "s45_formations.parquet"))
print(f"disagreeing menus: {len(r):,}")

# ---------- A. block bootstrap by formation year ------------------------
print("\n" + "="*78); print("A. BLOCK BOOTSTRAP BY FORMATION YEAR"); print("="*78)
rows = []
for K in sorted(r.K.unique()):
    for h in (1, 5, 10, 20):
        col = f"gap_rein{h}"
        d = r[(r.K == K)].dropna(subset=[col])
        if len(d) < 500: continue
        yrs = d.year.unique()
        per_year = d.groupby("year")[col].median()
        obs = per_year.median() / h * 10000
        draws = []
        for _ in range(BOOT):
            pick = rng.choice(yrs, size=len(yrs), replace=True)
            draws.append(per_year.loc[pick].median() / h * 10000)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"K": K, "horizon": h, "n_formation_years": len(yrs),
                     "net_bps": obs, "ci_lo": lo, "ci_hi": hi,
                     "excludes_zero": (lo < 0) == (hi < 0)})
b = pd.DataFrame(rows)
b.round(1).to_csv(os.path.join(OUT, "s50_bootstrap.csv"), index=False)
print(b.round(1).to_string(index=False))
print("\n  net_bps is the hot pick's annualised net edge (negative = the")
print("  cheap pick wins). The CI is over FORMATION YEARS, which is the")
print("  unit that actually varies. If it spans zero at 20y, the headline")
print("  number cannot be asserted.")

# ---------- B. era split -------------------------------------------------
print("\n" + "="*78); print("B. ERA SPLIT -- is this one decade?"); print("="*78)
r["era"] = pd.cut(r.year, [1989, 1999, 2009, 2026],
                  labels=["1990-1999", "2000-2009", "2010-2025"])
for h in (5, 10, 20):
    col = f"gap_rein{h}"
    print(f"\n  horizon {h}y, annualised net bps by era and menu size:")
    t = (r.dropna(subset=[col]).groupby(["era", "K"], observed=True)[col]
           .median().unstack() / h * 10000)
    n = r.dropna(subset=[col]).groupby("era", observed=True).year.nunique()
    t["n_years"] = n
    print(t.round(1).to_string())

# ---------- C. attrition: who has no 20-year window? --------------------
print("\n" + "="*78); print("C. ATTRITION -- who is missing at 20 years?"); print("="*78)
f20 = ft[ft.year <= 2005].copy()
lh = pd.read_parquet(os.path.join(CACHE, "s49_long_horizon.parquet"))
have = set(zip(lh.year, lh.K))  # menu-level; use fund-level proxy instead
pn = pd.read_parquet(os.path.join(CACHE, "s45_panel_with_passive.parquet"))
last = pn.groupby("wficn").year.max().rename("last_year")
f20 = f20.merge(last, on="wficn", how="left")
f20["has20"] = f20.last_year >= f20.year + 20
print(f"  formations 1990-2005: {len(f20):,}; with a full 20y window: "
      f"{100*f20.has20.mean():.1f}%")
g = f20.groupby("has20").agg(n=("wficn", "size"), median_fee_bps=("exp_ratio", lambda s: s.median()*10000),
                             median_tna_musd=("tna", "median"),
                             mean_trail12_pct=("trail12", lambda s: s.mean()*100),
                             pct_passive=("passive", lambda s: 100*s.mean()))
print(g.round(1).to_string())
print("\n  If the funds WITH a 20-year window are systematically cheaper,")
print("  bigger, or more passive, then the 20-year comparison is between")
print("  survivors that differ from the menu the investor actually faced.")

print("""
PLAIN READING
  Report the horizon at which the confidence interval still excludes zero,
  and lead with THAT number, not the biggest one. My guess before running:
  1y through 10y survive, 20y does not, and the honest headline becomes a
  10-year figure with the 20-year shown as suggestive.
  If the era split shows the effect concentrated in 1990-1999, the paper
  has an era problem and the fix is to report all three eras separately
  rather than pooling.
""")
