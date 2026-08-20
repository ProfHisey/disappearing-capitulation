# 45b_horizon_and_survivorship.py -- two checks on the stage 45 result,
# which said sorting by 1-year return beats sorting by fee.
#
#   CHECK 1: the FIVE-YEAR horizon. Stage 45 computed fwd5 and never printed
#            it (my omission). Momentum is a ~1-year phenomenon (Carhart
#            1997); fees compound forever. If the ranking flips at 5 years,
#            stage 45 measured the wrong horizon, not the wrong rule.
#
#   CHECK 2: partial-year survivorship. Forward returns compound whatever
#            months exist. A fund that liquidates three months into the
#            forward year contributes a 3-month return as if it were an
#            annual one -- which is much closer to zero than a full year of
#            losses would be. That FLATTERS high-mortality picks, and the
#            performance rule picks the highest-mortality funds.
#
# Run after 45.   python 45b_horizon_and_survivorship.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
r  = pd.read_parquet(os.path.join(CACHE, "s45_menu_draws.parquet"))
ft = pd.read_parquet(os.path.join(CACHE, "s45_formations.parquet"))
pn = pd.read_parquet(os.path.join(CACHE, "s45_panel_with_passive.parquet"))

print("=" * 72); print("CHECK 1 -- FIVE-YEAR HORIZON"); print("=" * 72)
d5 = r.dropna(subset=["fee_fwd5", "perf_fwd5"])
for K, d in d5.groupby("K"):
    yr = d.groupby("year")[["fee_fwd5", "perf_fwd5"]].mean()
    print(f"\nmenu of {K}  ({len(d):,} menus with a full 5-year window)")
    print(f"  sort by FEE, 5y cumulative        {yr.fee_fwd5.mean()*100:7.2f}%")
    print(f"  sort by 1y RETURN, 5y cumulative  {yr.perf_fwd5.mean()*100:7.2f}%")
    print(f"  P(fee pick beats return pick) at 5y: "
          f"{100*(d.fee_fwd5 > d.perf_fwd5).mean():5.1f}%   "
          f"(was {100*(d.fee_fwd1 > d.perf_fwd1).mean():.1f}% at 1y)")
    print(f"  median gap at 5y: {100*(d.fee_fwd5 - d.perf_fwd5).median():+.2f}pp"
          f"   mean gap: {100*(d.fee_fwd5 - d.perf_fwd5).mean():+.2f}pp")

print("\n  MEAN vs MEDIAN at 1 year (the skewness question):")
for K, d in r.groupby("K"):
    print(f"    menu of {K:2d}:  fee mean {d.fee_fwd1.mean()*100:5.2f}% "
          f"median {d.fee_fwd1.median()*100:5.2f}%  |  "
          f"return mean {d.perf_fwd1.mean()*100:5.2f}% "
          f"median {d.perf_fwd1.median()*100:5.2f}%")

print("\n" + "=" * 72); print("CHECK 2 -- PARTIAL-YEAR SURVIVORSHIP"); print("=" * 72)
last = pn.groupby("wficn")["ym"].max().rename("last_ym")
ft = ft.merge(last, on="wficn", how="left")
ft["fwd_end"] = ft.year.apply(lambda y: pd.Period(f"{y+1}-12", "M"))
ft["complete"] = ft.last_ym >= ft.fwd_end
n_inc = (~ft.complete).sum()
print(f"  fund-years whose forward year is INCOMPLETE: {n_inc:,} "
      f"({100*(~ft.complete).mean():.1f}%)")
print(f"  mean fwd1, complete   {ft.loc[ft.complete,'fwd1'].mean()*100:6.2f}%")
print(f"  mean fwd1, incomplete {ft.loc[~ft.complete,'fwd1'].mean()*100:6.2f}%")
q = ft.assign(fee_q=pd.qcut(ft.exp_ratio.rank(method="first"), 4, labels=False) + 1,
              mom_q=pd.qcut(ft.trail12.rank(method="first"), 4, labels=False) + 1)
print("\n  incomplete-forward-year rate by fee quartile (1=cheapest):")
print((100 * q.groupby("fee_q").complete.apply(lambda s: 1 - s.mean())).round(1).to_string())
print("\n  incomplete-forward-year rate by trailing-return quartile (4=hottest):")
print((100 * q.groupby("mom_q").complete.apply(lambda s: 1 - s.mean())).round(1).to_string())

print("""
PLAIN READING
  CHECK 1: if the fee rule's win rate rises sharply from 1 year to 5, the
  stage 45 headline was a horizon artifact -- momentum paying off inside
  its own decay window. If it stays near 32-46%, the horizon is not the
  problem and the thesis is in real trouble.

  Watch the mean-vs-median line too. If the return rule's MEAN beats the
  fee rule while its MEDIAN does not, that is your skewness mechanism
  showing up: chasing performance buys a lottery ticket whose average is
  flattered by a few big winners, while the typical chooser does worse.

  CHECK 2: if incomplete forward years cluster in the expensive or the
  hottest quartiles, the survivorship handling is biased toward the rule
  that won, and stage 45 must be rebuilt requiring a full forward window
  (or filling dead months with the category return) before it means
  anything.
""")
