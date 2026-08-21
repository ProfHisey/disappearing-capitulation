# 59_closed_form.py -- if the mechanism is combinatorics, we should be able to
# PREDICT it exactly, with no simulation.
#
# Stage 58: blindfolded picking reproduces 87-95% of the menu-size effect.
# Momentum adds only +2.4 to +5.4 bps beyond chance. So the object of study is
# not recency -- it is FEE-BLINDNESS, and its cost should follow from the
# sleeve's fee distribution alone.
#
# THE CLOSED FORM. Draw K funds uniformly without replacement from a sleeve
# holding n funds with fees f_(1) <= ... <= f_(n). The cheapest drawn fund is
# f_(i) exactly when the other K-1 draws all come from above i:
#
#       P(min = f_(i)) = C(n-i, K-1) / C(n, K)
#       E[min_K]       = sum_i f_(i) * C(n-i, K-1) / C(n, K)
#
# A fee-blind picker gets the sleeve mean. So the expected cost of ignoring
# fees, and the entire menu-size gradient, are:
#
#       cost(K)            = mean(f) - E[min_K]
#       cost(20) - cost(3) = E[min_3] - E[min_20]
#
# No behaviour, no simulation, no returns data. If the simulated numbers match
# these, the paper's core is a theorem with an empirical illustration rather
# than a simulation result.
#
# This script also runs the UNIFORM-draw simulation the audit asked for (M7),
# since the closed form assumes uniform sampling and the TNA-weighted sampler
# is a separate modelling choice that needs its own justification.
#
#   python 59_closed_form.py
import os
from math import comb
import numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
KS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25)
KMAX, REPS, SEED = max(KS), 400, 20260828
rng = np.random.default_rng(SEED)

src = os.path.join(CACHE, "s54_formations_lipper.parquet")
if not os.path.exists(src):
    src = os.path.join(CACHE, "s53_formations_alpha.parquet")
ft = pd.read_parquet(src)
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
has_lip = "lip" in ft.columns and ft.lip.notna().any()
print(f"formations {len(ft):,}   lipper: {has_lip}")


def exp_min(fees_sorted, K):
    """Exact E[min of K uniform draws without replacement]."""
    n = len(fees_sorted)
    if K > n: return np.nan
    denom = comb(n, K)
    tot = 0.0
    for i in range(1, n - K + 2):          # f_(i), 1-indexed
        tot += fees_sorted[i - 1] * comb(n - i, K - 1)
    return tot / denom


def analytic(df, catcol):
    """Closed-form cost of fee-blind selection, per sleeve-year, then averaged."""
    rows = []
    for (y, ct), g in df.groupby(["year", catcol]):
        f = np.sort(g.exp_ratio.to_numpy(float))
        if len(f) < KMAX: continue
        mu = f.mean()
        rec = {"year": y, "sleeve": ct, "n": len(f)}
        for K in KS:
            rec[f"cost{K}"] = mu - exp_min(f, K)
        rows.append(rec)
    return pd.DataFrame(rows)


def simulate_uniform(df, catcol):
    """Same thing by simulation, UNIFORM draws (the audit's M7 check)."""
    rows = []
    for (y, ct), g in df.groupby(["year", catcol]):
        f = g.exp_ratio.to_numpy(float)
        tr = g.trail12.to_numpy(float)
        n = len(f)
        if n < KMAX: continue
        rec = {"year": y, "sleeve": ct}
        for K in KS:
            keys = rng.random((REPS, n))
            idx = np.argpartition(keys, K - 1, axis=1)[:, :K]
            r = np.arange(REPS)
            cheap = idx[r, np.argmin(f[idx], axis=1)]
            hot = idx[r, np.argmax(tr[idx], axis=1)]
            rand = idx[r, rng.integers(0, K, size=REPS)]
            rec[f"sim_rand{K}"] = np.mean(f[rand] - f[cheap])
            rec[f"sim_hot{K}"] = np.mean(f[hot] - f[cheap])
        rows.append(rec)
    return pd.DataFrame(rows)


for sleeve, catcol in ([("crsp", "cat")] + ([("lipper", "lip")] if has_lip else [])):
    df = ft.dropna(subset=[catcol])
    a = analytic(df, catcol)
    s = simulate_uniform(df, catcol)
    if not len(a): continue
    m = a.merge(s, on=["year", "sleeve"])
    print("\n" + "=" * 92)
    print(f"{sleeve.upper()} SLEEVES -- {len(m):,} sleeve-years, "
          f"{m.year.nunique()} formation years, median n = {m.n.median():.0f} funds")
    print("=" * 92)
    tab = pd.DataFrame({
        "K": KS,
        "closed_form_bps": [m[f"cost{K}"].mean() * 10000 for K in KS],
        "sim_random_bps": [m[f"sim_rand{K}"].mean() * 10000 for K in KS],
        "sim_momentum_bps": [m[f"sim_hot{K}"].mean() * 10000 for K in KS],
    }).set_index("K")
    tab["formula_error_bps"] = tab.closed_form_bps - tab.sim_random_bps
    tab["momentum_excess_bps"] = tab.sim_momentum_bps - tab.sim_random_bps
    print(tab.round(2).to_string())
    tab.round(3).to_csv(os.path.join(OUT, f"s59_{sleeve}_closed_form.csv"))

    g_cf = (m["cost20"] - m["cost3"]).mean() * 10000
    g_sr = (m["sim_rand20"] - m["sim_rand3"]).mean() * 10000
    g_sm = (m["sim_hot20"] - m["sim_hot3"]).mean() * 10000
    print(f"\n  GRADIENT K=20 minus K=3, uniform draws:")
    print(f"    closed form (no simulation, no returns) {g_cf:+7.2f} bps")
    print(f"    simulated, blindfolded pick             {g_sr:+7.2f} bps")
    print(f"    simulated, recency pick                 {g_sm:+7.2f} bps")
    print(f"    formula error                           {g_cf - g_sr:+7.2f} bps")
    print(f"    momentum beyond chance                  {g_sm - g_sr:+7.2f} bps")

    yr = m.groupby("year").apply(
        lambda d: pd.Series({"cf": (d["cost20"] - d["cost3"]).mean() * 10000,
                             "sm": (d["sim_hot20"] - d["sim_hot3"]).mean() * 10000}),
        include_groups=False)
    for c, lab in [("cf", "closed form"), ("sm", "recency")]:
        se = yr[c].std(ddof=1) / np.sqrt(len(yr))
        print(f"    {lab:12s} by year: {yr[c].mean():+7.2f} bps  t {yr[c].mean()/se:+6.2f}"
              f"  n={len(yr)}")

# ---- worked example on the real plan menu ------------------------------
print("\n" + "=" * 92)
print("WORKED EXAMPLE -- the actual plan menu")
print("=" * 92)
for p in [os.path.join(HERE, "plan_menu_2026-08-19.csv"),
          r"E:\Finance\research-agenda\plan_menu_2026-08-19.csv"]:
    if os.path.exists(p):
        menu = pd.read_csv(p)
        eq = menu[menu.asset_class == "Stock Investments"]
        print(f"  source: {p}\n")
        tot = 0.0
        for cat, g in eq.groupby("category"):
            f = np.sort(g.gross_exp_ratio_pct.to_numpy(float) / 100)
            n = len(f)
            if n < 2: continue
            cost = f.mean() - exp_min(f, n)      # K = the whole sleeve
            tot += cost
            print(f"    {cat:<14} n={n}  mean fee {f.mean()*10000:6.1f}bp  "
                  f"cheapest {f[0]*10000:5.1f}bp  "
                  f"expected cost of a fee-blind pick {cost*10000:6.1f}bp")
        print(f"\n    Equal-weighted across sleeves: {tot / eq.category.nunique() * 10000:.1f} bps/yr")
        print("    That is what a participant gives up, on average, by choosing")
        print("    within each sleeve for any reason unrelated to cost.")
        break
else:
    print("  plan_menu_2026-08-19.csv not found - skipping")

print("""
================================================================================
PLAIN READING
================================================================================
  formula_error_bps is the test. If the closed form matches the blindfolded
  simulation to a fraction of a basis point, then the central quantity in
  this paper is a theorem about order statistics, and the simulation is an
  illustration rather than evidence. That is a much stronger position: no
  sampler to defend, no seed, no bootstrap, and it generalises to any
  fee-blind rule and any menu.

  momentum_excess_bps is then the ONLY behavioural quantity in the paper,
  and it is small. Report it honestly as what it is: recency picks funds
  about 2-5 bps more expensive than blind chance, reliably but modestly.

  The worked example is the practical payoff. A plan sponsor can compute the
  expected cost of fee-blind selection for their own menu from their own fee
  list, with no returns data and no model.
""")
