# 47_ties_and_skewness.py -- fix the broken win-rate statistic and pin down
# the finding that stage 45b surfaced.
#
# TWO PROBLEMS WITH THE STAGE 45 HEADLINE:
#  (1) TIES. In many menus the cheapest fund IS the best trailing performer,
#      so both rules pick the same fund. Stage 45 used a strict ">", which
#      scored every tie as a loss for the fee rule. The median gap of
#      exactly 0.00pp at all three menu sizes is the tell.
#  (2) MEAN vs MEDIAN. The return rule has the higher MEAN and the worse
#      MEDIAN. That is a skewness result, and it is the paper.
#
# Run after 45.   python 47_ties_and_skewness.py
import os, numpy as np, pandas as pd
from scipy import stats

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
r = pd.read_parquet(os.path.join(CACHE, "s45_menu_draws.parquet"))
EPS = 1e-9

rows = []
for K, d in r.groupby("K"):
    for h, f, p in [(1, "fee_fwd1", "perf_fwd1"), (5, "fee_fwd5", "perf_fwd5")]:
        s = d.dropna(subset=[f, p])
        gap = s[f] - s[p]
        tie = gap.abs() < EPS
        diff = s[~tie]
        rows.append({
            "K": K, "horizon_y": h, "n_menus": len(s),
            "pct_tie": 100 * tie.mean(),
            "pct_fee_wins": 100 * (gap > EPS).mean(),
            "pct_return_wins": 100 * (gap < -EPS).mean(),
            "pct_fee_wins_excl_ties": 100 * (diff[f] > diff[p]).mean(),
            "fee_mean": 100 * s[f].mean(), "fee_median": 100 * s[f].median(),
            "ret_mean": 100 * s[p].mean(), "ret_median": 100 * s[p].median(),
            "fee_skew": stats.skew(s[f].dropna()), "ret_skew": stats.skew(s[p].dropna()),
            "fee_p25": 100 * s[f].quantile(.25), "ret_p25": 100 * s[p].quantile(.25),
            "fee_p75": 100 * s[f].quantile(.75), "ret_p75": 100 * s[p].quantile(.75)})
t = pd.DataFrame(rows)
t.round(2).to_csv(os.path.join(OUT, "s47_ties_and_skewness.csv"), index=False)

print("=" * 78); print("1. THE WIN RATE, WITH TIES SEPARATED"); print("=" * 78)
print(t[["K", "horizon_y", "n_menus", "pct_tie", "pct_fee_wins",
         "pct_return_wins", "pct_fee_wins_excl_ties"]].round(1).to_string(index=False))
print("\n  pct_tie = both rules picked the SAME fund. Stage 45 scored these")
print("  as losses for the fee rule. pct_fee_wins_excl_ties is the honest")
print("  head-to-head: of the menus where the rules disagree, how often")
print("  does the cheap pick win?")

print("\n" + "=" * 78); print("2. MEAN, MEDIAN AND SKEW OF EACH RULE'S OUTCOME"); print("=" * 78)
print(t[["K", "horizon_y", "fee_mean", "fee_median", "fee_skew",
         "ret_mean", "ret_median", "ret_skew"]].round(2).to_string(index=False))
print("\n  If ret_mean > fee_mean while ret_median < fee_median, and")
print("  ret_skew > fee_skew, the performance rule is a lottery: better on")
print("  average, worse for the typical chooser.")

print("\n" + "=" * 78); print("3. THE QUARTILE VIEW -- who bears the downside?"); print("=" * 78)
print(t[["K", "horizon_y", "fee_p25", "ret_p25", "fee_median", "ret_median",
         "fee_p75", "ret_p75"]].round(2).to_string(index=False))
print("\n  Compare the p25 columns. That is the unlucky quarter of investors")
print("  under each rule -- the number an investment committee should care")
print("  about, and the one no fund screen has ever shown anyone.")

print("\n" + "=" * 78); print("4. WHERE THE RULES DISAGREE"); print("=" * 78)
for K, d in r.groupby("K"):
    s = d.dropna(subset=["fee_fwd1", "perf_fwd1"])
    tie = (s.fee_fwd1 - s.perf_fwd1).abs() < EPS
    dd = s[~tie]
    print(f"\n  menu of {K}: rules disagree in {100*(~tie).mean():.1f}% of menus")
    print(f"    when they disagree -- fee mean {dd.fee_fwd1.mean()*100:5.2f}% "
          f"median {dd.fee_fwd1.median()*100:5.2f}%  |  "
          f"return mean {dd.perf_fwd1.mean()*100:5.2f}% "
          f"median {dd.perf_fwd1.median()*100:5.2f}%")
    print(f"    cheap pick is an index fund in {100*dd.fee_is_index.mean():.1f}% "
          f"of disagreements")

print("""
PLAIN READING
  The claim that survives, if these numbers hold, is not "sorting by fee
  earns you more". It is:

     Sorting by past performance raises your AVERAGE outcome and lowers
     your TYPICAL one, because it buys a right-skewed payoff. Sorting by
     fee declines that lottery. Most participants are not average.

  That is a stronger practitioner paper than the original thesis, it is
  honest about the mean, and it is the fee paper and the skewness paper
  turning out to be the same paper.
""")
