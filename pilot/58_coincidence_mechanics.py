# 58_coincidence_mechanics.py -- is the mechanism just combinatorics?
#
# Stage 57 killed the "recency picks worse funds" claim and left this:
#   the coincidence rate (both rules pick the SAME fund) falls from ~35% at
#   K=3 to ~5% at K=20, and the resulting forgone fee saving is ~40bp/yr.
#
# THE OBSERVATION THAT MOTIVATES THIS SCRIPT. If fee rank and trailing-return
# rank were INDEPENDENT within a sleeve, the coincidence rate would be exactly
# 1/K -- given which fund is cheapest, it is also the hottest with probability
# 1/K. Observed: 35.6% vs 1/3 = 33.3%, and 5.3% vs 1/20 = 5.0%. Almost exactly
# 1/K. That would mean the entire effect is an order-statistic fact about
# menus, with a closed form, and momentum contributes nothing beyond chance.
#
# THE TEST. For every simulated menu, alongside the real hot pick we also take
# a UNIFORMLY RANDOM member of the same menu. That randomised pick is what the
# recency rule would deliver if trailing return carried no information about
# fees at all.
#
#     tau_real  vs  1/K  vs  tau_random     -> is the coincidence rate chance?
#     gap_real  vs  gap_random              -> does momentum pick funds that
#                                              are MORE expensive than chance?
#
# If gap_real == gap_random, the mechanism is combinatorics and the paper has
# a closed-form backbone. If gap_real > gap_random, momentum systematically
# steers toward expensive funds, which is a second, separate, behavioural
# channel worth its own section.
#
#   python 58_coincidence_mechanics.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
KS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25)
KMAX, REPS, SEED = max(KS), 400, 20260827
rng = np.random.default_rng(SEED)

src = os.path.join(CACHE, "s54_formations_lipper.parquet")
if not os.path.exists(src):
    src = os.path.join(CACHE, "s53_formations_alpha.parquet")
ft = pd.read_parquet(src)
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
has_lip = "lip" in ft.columns and ft.lip.notna().any()
print(f"formations {len(ft):,}   lipper: {has_lip}")

# ---- how correlated ARE fee and trailing return inside a sleeve? --------
def spearman_within(df, catcol):
    rs, ns = [], []
    for _, g in df.groupby(["year", catcol]):
        if len(g) < 10: continue
        a = g.exp_ratio.rank().to_numpy(); b = g.trail12.rank().to_numpy()
        a -= a.mean(); b -= b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        if d > 0:
            rs.append((a @ b) / d); ns.append(len(g))
    rs = np.array(rs)
    return rs.mean(), np.median(rs), len(rs)

print("\n" + "=" * 88)
print("1. RANK CORRELATION OF FEE vs TRAILING RETURN, WITHIN SLEEVE-YEAR")
print("=" * 88)
for sleeve, catcol in ([("crsp", "cat")] + ([("lipper", "lip")] if has_lip else [])):
    m, md, n = spearman_within(ft.dropna(subset=[catcol]), catcol)
    print(f"  {sleeve:7s}: mean Spearman {m:+.4f}, median {md:+.4f}, "
          f"{n:,} sleeve-years")
print("  Near zero means fee tells you nothing about who just won, and the")
print("  coincidence rate should be 1/K exactly.")


def simulate(df, catcol, index_free=False):
    recs = []
    for (y, ct), g in df.groupby(["year", catcol]):
        if index_free:
            g = g[~g.passive.astype(bool)]
        if len(g) < KMAX: continue
        n = len(g)
        fee = g.exp_ratio.to_numpy(float)
        tr = g.trail12.to_numpy(float)
        logw = np.log(np.maximum(g.tna.to_numpy(float), 1e-12))
        for K in KS:
            keys = rng.gumbel(size=(REPS, n)) + logw[None, :]
            idx = np.argpartition(-keys, K - 1, axis=1)[:, :K]
            rows = np.arange(REPS)
            cheap = idx[rows, np.argmin(fee[idx], axis=1)]
            hot = idx[rows, np.argmax(tr[idx], axis=1)]
            rand = idx[rows, rng.integers(0, K, size=REPS)]   # momentum-free pick
            recs.append(pd.DataFrame({
                "year": y, "K": K,
                "tie_real": cheap == hot,
                "tie_rand": cheap == rand,
                "fee_cheap": fee[cheap], "fee_hot": fee[hot], "fee_rand": fee[rand],
                "gap_real": np.where(cheap == hot, 0.0, fee[hot] - fee[cheap]),
                "gap_rand": np.where(cheap == rand, 0.0, fee[rand] - fee[cheap]),
            }))
    return pd.concat(recs, ignore_index=True)


for sleeve, catcol, ifree in ([("crsp", "cat", False), ("crsp", "cat", True)] +
                              ([("lipper", "lip", False), ("lipper", "lip", True)]
                               if has_lip else [])):
    d = simulate(ft.dropna(subset=[catcol]), catcol, ifree)
    lab = f"{sleeve}/{'index-free' if ifree else 'all menus'}"
    print("\n" + "=" * 88)
    print(f"2. {lab}   ({d.year.nunique()} formation years)")
    print("=" * 88)
    t = d.groupby("K").agg(
        tau_real=("tie_real", "mean"), tau_rand=("tie_rand", "mean"),
        fee_cheap=("fee_cheap", "mean"), fee_hot=("fee_hot", "mean"),
        fee_rand=("fee_rand", "mean"),
        gap_real=("gap_real", "mean"), gap_rand=("gap_rand", "mean"))
    t["one_over_K"] = 1.0 / t.index
    for c in ["tau_real", "tau_rand", "one_over_K"]:
        t[c] = t[c] * 100
    for c in ["fee_cheap", "fee_hot", "fee_rand", "gap_real", "gap_rand"]:
        t[c] = t[c] * 10000
    t["excess_tau_pp"] = t.tau_real - t.one_over_K
    t["gap_excess_bps"] = t.gap_real - t.gap_rand
    print(t[["tau_real", "one_over_K", "tau_rand", "excess_tau_pp",
             "fee_cheap", "fee_hot", "fee_rand",
             "gap_real", "gap_rand", "gap_excess_bps"]].round(2).to_string())
    t.round(3).to_csv(os.path.join(OUT, f"s58_{sleeve}_{'if' if ifree else 'all'}.csv"))

    # year-level inference on the K=20 vs K=3 gradient, real vs momentum-free
    print(f"\n  GRADIENT K=20 minus K=3, by formation year:")
    for col, name in [("gap_real", "real momentum"), ("gap_rand", "random pick")]:
        s = d.groupby(["K", "year"])[col].mean().unstack(0) * 10000
        g = (s[20] - s[3]).dropna()
        se = g.std(ddof=1) / np.sqrt(len(g))
        print(f"    {name:14s} {g.mean():+7.1f} bps  t {g.mean()/se:+6.2f}  n={len(g)}")
    s = d.groupby(["K", "year"])[["gap_real", "gap_rand"]].mean()
    diff = ((s.xs(20, level="K") - s.xs(3, level="K")) * 10000)
    dd = (diff.gap_real - diff.gap_rand).dropna()
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    print(f"    DIFFERENCE     {dd.mean():+7.1f} bps  t {dd.mean()/se:+6.2f}  "
          f"<- momentum's contribution beyond chance")

print("""
================================================================================
PLAIN READING
================================================================================
  Block 1: if the within-sleeve rank correlation between fee and trailing
  return is near zero, then knowing a fund just won tells you nothing about
  what it charges, and everything downstream is combinatorics.

  Block 2, the two columns that matter: tau_real against one_over_K, and
  gap_real against gap_rand.

    tau_real ~ 1/K and gap_real ~ gap_rand
        The mechanism is an order-statistic fact about menus. The recency
        rule is, for fee purposes, indistinguishable from picking at random,
        and the cost of a longer menu is fully predictable in closed form
        from the sleeve's fee distribution. That is a cleaner and more
        general result than the one we lost, and it generalises to ANY
        selection rule uncorrelated with fees.

    gap_real materially above gap_rand
        Momentum steers toward expensive funds beyond chance -- a real
        behavioural channel, separable and worth its own section.

  The DIFFERENCE line is the whole question in one number, with a t-stat.
""")
