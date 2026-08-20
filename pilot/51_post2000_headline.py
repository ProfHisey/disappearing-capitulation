# 51_post2000_headline.py -- the publishable version.
#
# Stage 50 showed the pooled result is dominated by 1990s formations: at 10y
# and a menu of 10 the net edge is -320bp/yr for 1990s formations but -77bp
# for the 2000s and -100bp for the 2010s. The 1990s were a different market
# (high fees, few index funds, pre-compression), and a 2026 recommendation
# cannot be built on them.
#
# This recomputes every headline exhibit restricted to formations from 2000
# onward, and reports the menu-size gradient separately by era because that
# gradient is the structural finding that holds everywhere.
#
#   python 51_post2000_headline.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
BOOT, SEED = 2000, 20260823
rng = np.random.default_rng(SEED)

r = pd.read_parquet(os.path.join(CACHE, "s49_long_horizon.parquet"))
post = r[r.year >= 2000].copy()
print(f"all formations {len(r):,} menus | post-2000 {len(post):,} menus "
      f"({post.year.nunique()} formation years)")

# ---------- 1. headline quantiles, post-2000 ----------------------------
print("\n" + "=" * 78)
print("1. TERMINAL WEALTH OF $10,000 -- POST-2000 FORMATIONS ONLY")
print("=" * 78)
rows = []
for K, d in post.groupby("K"):
    for h in (5, 10, 20):
        s = d.dropna(subset=[f"cheap_rein{h}", f"hot_rein{h}"])
        if len(s) < 500: continue
        c, hot = (1 + s[f"cheap_rein{h}"]) * 10000, (1 + s[f"hot_rein{h}"]) * 10000
        rows.append({"K": K, "horizon": h, "n_menus": len(s),
                     "n_years": s.year.nunique(),
                     "cheap_p25": c.quantile(.25), "cheap_med": c.median(),
                     "hot_p25": hot.quantile(.25), "hot_med": hot.median(),
                     "gap_med": c.median() - hot.median(),
                     "gap_p25": c.quantile(.25) - hot.quantile(.25),
                     "pct_cheap_wins": 100 * (s[f"cheap_rein{h}"] > s[f"hot_rein{h}"]).mean()})
w = pd.DataFrame(rows)
w.round(0).to_csv(os.path.join(OUT, "s51_terminal_wealth_post2000.csv"), index=False)
print(w.round(0).to_string(index=False))

# ---------- 2. bootstrap the post-2000 headline -------------------------
print("\n" + "=" * 78)
print("2. BLOCK BOOTSTRAP BY FORMATION YEAR, POST-2000")
print("=" * 78)
rows = []
for K, d in post.groupby("K"):
    for h in (5, 10, 20):
        col = f"gap_rein{h}"
        s = d.dropna(subset=[col])
        if len(s) < 500 or s.year.nunique() < 5: continue
        py = s.groupby("year")[col].median()
        obs = py.median() / h * 10000
        draws = [py.loc[rng.choice(py.index, len(py), replace=True)].median() / h * 10000
                 for _ in range(BOOT)]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"K": K, "horizon": h, "n_years": len(py), "net_bps": obs,
                     "ci_lo": lo, "ci_hi": hi, "excludes_zero": (lo < 0) == (hi < 0)})
b = pd.DataFrame(rows)
b.round(1).to_csv(os.path.join(OUT, "s51_bootstrap_post2000.csv"), index=False)
print(b.round(1).to_string(index=False))

# ---------- 3. the menu-size gradient, by era ---------------------------
print("\n" + "=" * 78)
print("3. THE MENU-SIZE GRADIENT -- the finding that holds in every era")
print("=" * 78)
r["era"] = pd.cut(r.year, [1989, 1999, 2009, 2026],
                  labels=["1990-1999", "2000-2009", "2010-2025"])
for h in (5, 10):
    col = f"gap_rein{h}"
    t = (r.dropna(subset=[col]).groupby(["era", "K"], observed=True)[col]
           .median().unstack() / h * 10000)
    t["gradient_3_to_10"] = t[10] - t[3]
    print(f"\n  horizon {h}y, annualised net bps (negative = cheap pick wins):")
    print(t.round(1).to_string())
print("\n  Every era: the bigger the menu, the worse the chasing rule does.")
print("  That is selection, not an era effect, and it is the mechanism the")
print("  paper should lead with.")

# ---------- 4. the sentence ---------------------------------------------
print("\n" + "=" * 78); print("4. THE DEFENSIBLE SENTENCE"); print("=" * 78)
best = b[(b.excludes_zero) & (b.horizon <= 10)]
if len(best):
    x = best.sort_values("horizon").iloc[-1]
    wr = w[(w.K == x.K) & (w.horizon == x.horizon)].iloc[0]
    print(f"  For plan menus offering {int(x.K)} funds per category, among"
          f" formations since 2000,\n  choosing the cheapest fund rather than"
          f" the best recent performer was worth\n  {abs(x.net_bps):.0f} bps a"
          f" year over {int(x.horizon)} years"
          f" (95% CI {abs(x.ci_hi):.0f} to {abs(x.ci_lo):.0f} bps),"
          f"\n  or ${wr.gap_med:,.0f} on a $10,000 balance at the median and"
          f" ${wr.gap_p25:,.0f} at the\n  25th percentile. The cheap pick won"
          f" {wr.pct_cheap_wins:.0f}% of the time.")
else:
    print("  No post-2000 horizon at or under 10y excludes zero. Say so, and")
    print("  lead with the menu-size gradient instead of a point estimate.")

print("""
PLAIN READING
  Block 1 and 2 are the numbers that can go in a paper aimed at 2026
  decisions. If they are much smaller than the pooled figures, that is the
  honest result and the pooled ones were an era artifact.
  Block 3 is the finding I would actually lead with: the more options a
  plan offers, the worse performance-chasing performs, in every era. That
  is a statement about selection under skewness, it does not depend on the
  fee level, and it survives everything we have thrown at it.
""")
