"""
Stage 42 - The certainty asymmetry, quantified by simulation.

    STATUS 2026-08-20: SUPERSEDED AS A PAPER EXHIBIT. Keep as a BuyRisk
    teaching artifact only. Two reasons, both from our own later work:

    1. Its null is alpha = 0. Stage 52 measured the actual gross edge of the
       recent winner and it is NOT zero in small sleeves: +38.4 bps/yr at
       K=3, 95% CI [7.1, 90.2], post-2000, 10-year horizon, four-factor.
       The fee roughly cancels it (net -11.7 bps, indistinguishable from
       zero). This menu's sleeves hold 2-5 funds each, so K=3 is the
       relevant regime and the zero-alpha null understates the hot pick.
       The honest story for a SMALL menu is "a fairly priced bet", not
       "a losing one". The bet only turns bad as the menu grows: gross
       edge -59.1 bps at K=20.
    2. The paper it was written for no longer exists under that title. The
       surviving paper is "Selection Under Skewness: Why Longer Fund Menus
       Punish Recency", and its claim is a GRADIENT across menu sizes, not
       a level for one menu.

    NAMING COLLISION - do not mix these up:
      stage 42 "break-even alpha"    = derived from ASSUMPTIONS (this file)
      stage 48 "break-even multiple" = derived from DATA
    Only the stage 48 number belongs in the paper.

    To rehabilitate this as a BuyRisk lab, replace ALPHA_SWEEP with the
    measured gross edge by menu size from output/s52_levels_post-2000.csv
    and let the user vary the number of funds in the sleeve. That version
    teaches the real finding instead of a simplified one.

Original header follows.
---------------------------------------------------------------------------
Paper: "Sort by Fees, Not Performance"

Run:  python 42_menu_monte_carlo.py
Input: plan_menu_2026-08-19.csv (the menu audit, in this folder or ../)

THE ARGUMENT THIS SCRIPT MAKES

Once a participant has chosen an asset allocation, the remaining choice inside
each category has exactly two consequences:

  1. A FEE DIFFERENCE - known today, to the basis point, and it compounds with
     certainty. It has no distribution. It is a point mass.
  2. AN ALPHA DIFFERENCE - unknowable today, and on the evidence roughly
     mean-zero with wide dispersion.

The screen shows the second one in nine return figures across two tabs, and
hides the first behind a third tab, after the daily NAV change.

The simulation asks the only question that matters to a committee: how much
persistent alpha would the performance-chosen funds have to deliver, every
year, forever, just to break even against the fee difference? Then it puts
that break-even next to what the persistence evidence actually says (S&P's
Persistence Scorecard: 0.00% of top-quartile funds stay top-quartile over
five consecutive years, in both the 2024 and 2025 editions).

WHAT IS DELIBERATELY NOT CLAIMED
  - No claim that cheap funds earn positive alpha. The null here is alpha = 0.
  - No claim about asset allocation. Allocation is held fixed by construction;
    both portfolios hold the same sleeves in the same weights.
  - The alpha dispersion parameter is an assumption, and the sweep shows how
    much the answer depends on it.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

CANDIDATES = [os.path.join(HERE, "plan_menu_2026-08-19.csv"),
              os.path.join(HERE, "..", "plan_menu_2026-08-19.csv"),
              r"E:\Finance\research-agenda\plan_menu_2026-08-19.csv"]

N_SIMS = 20000
YEARS = 35
ANNUAL_CONTRIB = 20000.0      # real dollars per year
GROSS_MU, GROSS_SIGMA = 0.065, 0.16   # real equity return assumptions
ALPHA_SIGMA = 0.02            # idiosyncratic dispersion of the active bet
ALPHA_SWEEP = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015]
SEED = 20260820


def say(m):
    print(m, flush=True)


def hr(t):
    say("\n" + "=" * 72)
    say(t)
    say("=" * 72)


def load_menu():
    for p in CANDIDATES:
        if os.path.exists(p):
            say(f"menu: {p}")
            return pd.read_csv(p)
    raise SystemExit("!! plan_menu_2026-08-19.csv not found - see CANDIDATES")


def build_rules(menu):
    eq = menu[(menu.asset_class == "Stock Investments") &
              (menu.category != "Specialty")].copy()
    rows = []
    for cat, s in eq.groupby("category"):
        perf = s.loc[s.ret_1y_pct.idxmax()]
        cheap = s.loc[s.gross_exp_ratio_pct.idxmin()]
        rows.append({"category": cat,
                     "perf_pick": perf.ticker, "perf_fee": perf.gross_exp_ratio_pct / 100,
                     "perf_1y": perf.ret_1y_pct,
                     "cheap_pick": cheap.ticker, "cheap_fee": cheap.gross_exp_ratio_pct / 100,
                     "cheap_1y": cheap.ret_1y_pct})
    return pd.DataFrame(rows)


def simulate(fee_a, fee_b, alpha_mu, rng):
    """Two portfolios, same gross market path, different fee and alpha."""
    gross = rng.normal(GROSS_MU, GROSS_SIGMA, size=(N_SIMS, YEARS))
    alpha = rng.normal(alpha_mu, ALPHA_SIGMA, size=(N_SIMS, YEARS))
    ra = gross + alpha - fee_a          # performance-chosen
    rb = gross - fee_b                  # fee-chosen
    wa = np.zeros(N_SIMS)
    wb = np.zeros(N_SIMS)
    for t in range(YEARS):
        wa = (wa + ANNUAL_CONTRIB) * (1 + ra[:, t])
        wb = (wb + ANNUAL_CONTRIB) * (1 + rb[:, t])
    return wa, wb


def main():
    rng = np.random.default_rng(SEED)
    menu = load_menu()
    rules = build_rules(menu)

    hr("WHAT EACH RULE PICKS FROM THIS MENU")
    say(rules.assign(perf_fee_bps=(rules.perf_fee * 10000).round(1),
                     cheap_fee_bps=(rules.cheap_fee * 10000).round(1))
        [["category", "perf_pick", "perf_fee_bps", "perf_1y",
          "cheap_pick", "cheap_fee_bps", "cheap_1y"]].to_string(index=False))
    fee_perf = rules.perf_fee.mean()
    fee_cheap = rules.cheap_fee.mean()
    delta = fee_perf - fee_cheap
    say(f"\nequal-weight portfolio fee, top-1yr rule : {fee_perf * 10000:.1f} bps")
    say(f"equal-weight portfolio fee, cheapest rule: {fee_cheap * 10000:.1f} bps")
    say(f"ANNUAL FEE DELTA                         : {delta * 10000:.1f} bps")

    hr("THE ASYMMETRY, IN ONE COMPARISON")
    det = (1 - fee_cheap) ** YEARS / (1 - fee_perf) ** YEARS - 1
    say(f"Over {YEARS} years the fee difference alone compounds to "
        f"{det * 100:.1f}% more terminal wealth for the cheap portfolio.")
    say("That number has NO distribution. It is known today.")
    say(f"The alpha difference is drawn from a distribution with standard")
    say(f"deviation {ALPHA_SIGMA * 100:.1f}%/yr and a mean nobody can observe in advance.")

    hr(f"MONTE CARLO: {N_SIMS:,} paths, {YEARS} years, "
       f"${ANNUAL_CONTRIB:,.0f}/yr contributions")
    rows = []
    for amu in ALPHA_SWEEP:
        wa, wb = simulate(fee_perf, fee_cheap, amu, rng)
        rows.append({
            "assumed_persistent_alpha_bps": amu * 10000,
            "P(cheap wins)_pct": 100 * np.mean(wb > wa),
            "median_gap_dollars": np.median(wb - wa),
            "p10_gap_dollars": np.percentile(wb - wa, 10),
            "p90_gap_dollars": np.percentile(wb - wa, 90),
            "median_wealth_cheap": np.median(wb),
            "median_wealth_perf": np.median(wa),
        })
    tab = pd.DataFrame(rows)
    tab.round(1).to_csv(os.path.join(OUT, "s42_monte_carlo_sweep.csv"), index=False)
    say(tab.round(1).to_string(index=False))

    lo, hi = 0.0, 0.05
    for _ in range(40):
        mid = (lo + hi) / 2
        wa, wb = simulate(fee_perf, fee_cheap, mid, rng)
        if np.median(wa) < np.median(wb):
            lo = mid
        else:
            hi = mid
    say(f"\nBREAK-EVEN: the performance-chosen funds need about "
        f"{(lo + hi) / 2 * 10000:.0f} bps/yr of PERSISTENT alpha, every year for "
        f"{YEARS} years, just to match the cheap portfolio's median outcome.")

    hr("PLAIN READING")
    say("The fee difference is a certainty and the alpha difference is a")
    say("coin-weighting exercise. At zero persistent alpha - which is what the")
    say("persistence evidence supports - the cheap portfolio wins in the large")
    say("majority of simulated lifetimes, and the median gap is real money.")
    say("Nobody's choice is removed by putting the fee column first. The")
    say("participant can still buy any fund on the menu. They would simply be")
    say("shown the number that is knowable before the ones that are not.")
    say(f"\nCSV: {os.path.join(OUT, 's42_monte_carlo_sweep.csv')}")


if __name__ == "__main__":
    main()
