# 48_breakeven_persistence.py -- how good would the hot fund have to be?
#
# The investor who sorts by recent return is making an implicit bet: that
# the hot fund's skill advantage exceeds the extra fee they are not looking
# at. That bet decomposes exactly:
#
#     (hot fund return - cheap fund return) = GROSS EDGE - FEE PENALTY
#
# We observe the left side and the fee penalty, so we can back out the gross
# edge the hot pick actually delivered, and then ask the break-even question:
# how many times larger would it have to be to justify the choice?
#
# Stage 45 did not record the fee of each pick, so this re-simulates the
# menus (formations are cached, so it is quick).
#   python 48_breakeven_persistence.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
MENU_SIZES, REPS, SEED = (3, 5, 10), 200, 20260820      # same seed as stage 45
rng = np.random.default_rng(SEED)

ft = pd.read_parquet(os.path.join(CACHE, "s45_formations.parquet"))
print(f"formations: {len(ft):,} fund-years")

res = []
for K in MENU_SIZES:
    for (y, ct), g in ft.groupby(["year", "cat"]):
        if len(g) < K: continue
        w = g.tna.values / g.tna.values.sum()
        idx = np.arange(len(g))
        for _ in range(REPS):
            m = g.iloc[rng.choice(idx, size=K, replace=False, p=w)]
            c, h = m.exp_ratio.idxmin(), m.trail12.idxmax()
            if c == h: continue                      # rules agree; no bet to evaluate
            res.append({"K": K, "year": y,
                        "gap1": m.loc[h, "fwd1"] - m.loc[c, "fwd1"],
                        "gap5": m.loc[h, "fwd5"] - m.loc[c, "fwd5"],
                        "fee_pen": m.loc[h, "exp_ratio"] - m.loc[c, "exp_ratio"],
                        "hot_is_index": bool(m.loc[h, "passive"]),
                        "cheap_is_index": bool(m.loc[c, "passive"])})
r = pd.DataFrame(res)
r.to_parquet(os.path.join(CACHE, "s48_disagreements.parquet"), index=False)
print(f"menus where the rules disagree: {len(r):,}")

print("\n" + "=" * 78); print("1. DECOMPOSITION -- what the hot pick delivered, and what it cost")
print("=" * 78)
rows = []
for K, d in r.groupby("K"):
    for h, gap, yrs in [(1, "gap1", 1), (5, "gap5", 5)]:
        s = d.dropna(subset=[gap])
        # annualise the 5y gap so fee penalty is comparable
        obs = s[gap] / yrs
        fee = s.fee_pen
        gross = obs + fee                     # gross edge = observed + fee handicap
        rows.append({"K": K, "horizon_y": h, "n": len(s),
                     "fee_penalty_bps": fee.median() * 10000,
                     "observed_net_bps": obs.median() * 10000,
                     "implied_gross_bps": gross.median() * 10000,
                     "breakeven_multiple": (fee.median() / gross.median())
                     if gross.median() > 0 else np.nan})
t = pd.DataFrame(rows)
t.round(1).to_csv(os.path.join(OUT, "s48_breakeven.csv"), index=False)
print(t.round(1).to_string(index=False))
print("""
  fee_penalty_bps   how much more the hot fund charges, per year (median)
  observed_net_bps  how much the hot fund actually beat the cheap one by,
                    per year, net -- negative means it lost
  implied_gross_bps what the hot fund delivered BEFORE its fee handicap
  breakeven_multiple how many times larger the gross edge would need to be
                    for the bet to break even. Above 1 means the bet fails.
""")

print("=" * 78); print("2. THE SAME THING IN PLAIN NUMBERS"); print("=" * 78)
for _, x in t.iterrows():
    verdict = "LOSES" if x.observed_net_bps < 0 else "wins"
    print(f"  menu of {int(x.K):2d}, {int(x.horizon_y)}y: the hot pick charges "
          f"{x.fee_penalty_bps:5.0f}bp more, delivers {x.implied_gross_bps:6.0f}bp "
          f"gross, and {verdict} by {abs(x.observed_net_bps):4.0f}bp net")

print("\n" + "=" * 78); print("3. DOES THE HOT PICK EVER HAVE REAL SKILL?"); print("=" * 78)
for K, d in r.groupby("K"):
    s = d.dropna(subset=["gap1"])
    print(f"  menu of {K:2d}: hot pick's gross edge > 0 in "
          f"{100*((s.gap1 + s.fee_pen) > 0).mean():.1f}% of bets;  "
          f"net edge > 0 in {100*(s.gap1 > 0).mean():.1f}%")
    print(f"            hot pick is itself an index fund {100*s.hot_is_index.mean():.1f}% "
          f"of the time")

print("""
PLAIN READING
  This is the sentence for an investment committee. The person sorting by
  recent performance is paying the fee_penalty for the implied_gross edge.
  If the breakeven_multiple is comfortably above 1, they are paying more
  for the bet than the bet has ever been worth -- and the screen they are
  looking at shows them the bet and hides the price.
""")
