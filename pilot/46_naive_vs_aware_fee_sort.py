# 46_naive_vs_aware_fee_sort.py -- what happens to the investor who just
# wants "stocks" and sorts the whole equity lineup by fee, ignoring style?
#
# Run AFTER 45 (reuses s45_formations.parquet).   python 46_naive_vs_aware_fee_sort.py
#
# THE QUESTION BEHIND THE QUESTION. A naive fee sort lands you in one fund,
# almost certainly a large-cap index fund. That beats a high-fee fund for
# TWO reasons that must be told apart:
#   (1) COST -- robust, arithmetic, always there.
#   (2) STYLE ERA LUCK -- you are 100% large-cap. Over 1990-2025 that was a
#       winning bet for long stretches. A referee will say you discovered
#       that large beat small, not that cheap beat expensive.
# This script separates them by conditioning on whether large-cap actually
# beat small-cap in each forward year.
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
CATS_PER_PLAN, FUNDS_PER_CAT, REPS, SEED = 4, 4, 150, 20260821
rng = np.random.default_rng(SEED)

ft = pd.read_parquet(os.path.join(CACHE, "s45_formations.parquet"))
ft = ft.dropna(subset=["fwd1", "exp_ratio", "trail12", "cat", "tna"])
print(f"formations: {len(ft):,} fund-years, {ft.cat.nunique()} categories")

# --- style era: did large-cap beat small-cap in the forward year? -------
LARGE, SMALL = ("EDCL", "EDYB", "EDYG"), ("EDCS",)
era = (ft.assign(grp=np.where(ft.cat.isin(LARGE), "L", np.where(ft.cat.isin(SMALL), "S", "X")))
         .query("grp != 'X'").groupby(["year", "grp"])["fwd1"].mean().unstack())
era["large_won"] = era["L"] > era["S"]
print(f"large-cap beat small-cap in {100*era.large_won.mean():.0f}% of forward years")

# --- simulate plan-like equity lineups ---------------------------------
res = []
for y, g in ft.groupby("year"):
    cats = [c for c, gg in g.groupby("cat") if len(gg) >= FUNDS_PER_CAT]
    if len(cats) < CATS_PER_PLAN: continue
    for _ in range(REPS):
        chosen = rng.choice(cats, size=CATS_PER_PLAN, replace=False)
        parts = []
        for c in chosen:
            gg = g[g.cat == c]
            w = gg.tna.values / gg.tna.values.sum()
            parts.append(gg.iloc[rng.choice(np.arange(len(gg)), size=FUNDS_PER_CAT,
                                            replace=False, p=w)])
        lineup = pd.concat(parts)
        naive_fee  = lineup.loc[lineup.exp_ratio.idxmin()]        # cheapest, any style
        naive_perf = lineup.loc[lineup.trail12.idxmax()]          # hottest, any style
        worst_fee  = lineup.loc[lineup.exp_ratio.idxmax()]        # priciest, any style
        aware_fee  = lineup.loc[lineup.groupby("cat").exp_ratio.idxmin()]  # cheapest per sleeve
        hi = lineup[lineup.exp_ratio >= lineup.exp_ratio.quantile(2/3)]
        rand_hi = hi.iloc[rng.integers(len(hi))]                  # a random expensive fund
        res.append({"year": y,
                    "naive_fee": naive_fee.fwd1, "naive_fee_cat": naive_fee.cat,
                    "naive_fee_passive": bool(naive_fee.passive),
                    "naive_fee_bps": naive_fee.exp_ratio * 10000,
                    "aware_fee": aware_fee.fwd1.mean(),
                    "naive_perf": naive_perf.fwd1, "worst_fee": worst_fee.fwd1,
                    "rand_hi_fee": rand_hi.fwd1, "lineup_mean": lineup.fwd1.mean()})
r = pd.DataFrame(res).merge(era[["large_won"]], left_on="year", right_index=True, how="left")
r.to_parquet(os.path.join(CACHE, "s46_lineups.parquet"), index=False)
print(f"simulated {len(r):,} plan lineups "
      f"({CATS_PER_PLAN} sleeves x {FUNDS_PER_CAT} funds)")

RULES = ["naive_fee", "aware_fee", "naive_perf", "rand_hi_fee", "worst_fee", "lineup_mean"]
print("\n" + "="*72); print("1. MEAN FORWARD-1Y RETURN BY RULE (%)"); print("="*72)
yr = r.groupby("year")[RULES].mean()
for c in RULES: print(f"  {c:<14} {yr[c].mean()*100:6.2f}%/yr")

print("\n" + "="*72); print("2. HEAD TO HEAD, LINEUP BY LINEUP"); print("="*72)
for a, b in [("naive_fee", "rand_hi_fee"), ("naive_fee", "worst_fee"),
             ("naive_fee", "naive_perf"), ("aware_fee", "naive_perf"),
             ("naive_fee", "aware_fee")]:
    print(f"  P({a} beats {b}) = {100*(r[a] > r[b]).mean():5.1f}%   "
          f"median gap {100*(r[a]-r[b]).median():+.2f}pp")

print("\n" + "="*72); print("3. IS IT COST, OR IS IT A STYLE BET?"); print("="*72)
for won, d in r.groupby("large_won"):
    lab = "years large-cap WON" if won else "years small-cap WON"
    print(f"\n  {lab}  ({d.year.nunique()} years, {len(d):,} lineups)")
    for a, b in [("naive_fee", "rand_hi_fee"), ("naive_fee", "naive_perf"),
                 ("naive_fee", "aware_fee")]:
        print(f"    P({a} beats {b}) = {100*(d[a] > d[b]).mean():5.1f}%   "
              f"median gap {100*(d[a]-d[b]).median():+.2f}pp")

print("\n" + "="*72); print("4. WHERE DOES THE NAIVE FEE SORT LAND YOU?"); print("="*72)
print(f"  it picks a passive fund {100*r.naive_fee_passive.mean():.1f}% of the time")
print(f"  median fee of the pick: {r.naive_fee_bps.median():.0f} bps")
print((r.naive_fee_cat.value_counts(normalize=True) * 100).head(8).round(1).to_string())

print("""
PLAIN READING
  Block 2 is your headline: how often the naive fee-sorter beats an
  expensive fund, and by how much, lineup by lineup.

  Block 3 is the one that decides whether the headline survives review.
  If the naive fee rule wins ~equally in years small-cap won and years
  large-cap won, the advantage is COST and the claim is clean. If it wins
  overwhelmingly only when large-cap won, you have measured a style era,
  and the honest rule becomes the allocation-AWARE one (aware_fee), which
  keeps the cost advantage without the concentrated style bet.

  Note aware_fee holds one cheap fund per sleeve, so it is diversified
  across styles by construction; naive_fee is a single fund. Comparing
  their spread tells you what the tilt decision was actually worth.
""")
