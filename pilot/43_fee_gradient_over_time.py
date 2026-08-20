# 43_fee_gradient_over_time.py  -- run after 41. python 43_fee_gradient_over_time.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

ft = pd.read_parquet(os.path.join(CACHE, "s41_formations.parquet"))
fm = pd.read_parquet(os.path.join(CACHE, "fund_month_v3_tnafix.parquet"))
fl = pd.read_parquet(os.path.join(CACHE, "flags.parquet"))

# passive share of the domestic-equity fund universe, by year (Dec TNA)
fm["year"] = pd.DatetimeIndex(fm["caldt"]).year
dec = fm[pd.DatetimeIndex(fm["caldt"]).month == 12].merge(fl, on="wficn", how="inner")
dec = dec[dec["dom_eq"]]
ps = (dec.groupby("year")
        .apply(lambda d: d.loc[d["passive"], "tna"].sum() / max(d["tna"].sum(), 1),
               include_groups=False)
        .rename("passive_share"))

# yearly cross-sectional regression of forward 1y return on the fee
rows = []
for yr, g in ft.dropna(subset=["fwd1", "exp_ratio"]).groupby("form_year"):
    if len(g) < 100: continue
    d = g.copy()
    if d["cat"].nunique() > 1:                      # strip category means from both sides
        d["fee_d"]  = d["exp_ratio"] - d.groupby("cat")["exp_ratio"].transform("mean")
        d["fwd1_d"] = d["fwd1"]      - d.groupby("cat")["fwd1"].transform("mean")
    else:
        d["fee_d"], d["fwd1_d"] = d["exp_ratio"] - d["exp_ratio"].mean(), d["fwd1"] - d["fwd1"].mean()
    sd = d["fee_d"].std(ddof=1)
    if not sd > 0: continue
    for lab, x in [("raw_per_100bp", d["fee_d"] / 0.01), ("standardised_per_SD", d["fee_d"] / sd)]:
        X = np.column_stack([np.ones(len(d)), x.values])
        b, *_ = np.linalg.lstsq(X, d["fwd1_d"].values, rcond=None)
        r = d["fwd1_d"].values - X @ b
        se = np.sqrt((r @ r / (len(d) - 2)) * np.linalg.inv(X.T @ X)[1, 1])
        rows.append({"year": yr, "spec": lab, "beta_pct": b[1] * 100,
                     "t": b[1] / se, "n": len(d), "fee_sd_bps": sd * 10000})

g = pd.DataFrame(rows).merge(ps, left_on="year", right_index=True, how="left")
g.round(3).to_csv(os.path.join(OUT, "s43_fee_gradient_by_year.csv"), index=False)

for spec, d in g.groupby("spec"):
    print("\n" + "=" * 70); print(spec); print("=" * 70)
    print(d[["year", "beta_pct", "t", "n", "fee_sd_bps", "passive_share"]]
          .round(3).to_string(index=False))
    d = d.dropna(subset=["passive_share"])
    print(f"  mean beta {d.beta_pct.mean():+.3f} | share of years negative "
          f"{100*(d.beta_pct<0).mean():.0f}%")
    print(f"  corr(beta, passive share)  = {d.beta_pct.corr(d.passive_share):+.3f}")
    print(f"  corr(beta, fee dispersion) = {d.beta_pct.corr(d.fee_sd_bps):+.3f}")
    h = d[d.year >= 2010]; l = d[d.year < 2010]
    print(f"  mean beta pre-2010 {l.beta_pct.mean():+.3f}  |  2010+ {h.beta_pct.mean():+.3f}")

print("""
PLAIN READING
  beta is the extra forward-year return per 100bp of fee (raw) or per
  cross-sectional SD of fee (standardised). Berk-Green says beta = 0.
  The naive fee argument says beta < 0. Your hypothesis says beta has
  been RISING toward zero or above as passive share grew.

  Read the two specs together. If raw beta rises but standardised beta
  is flat, the change is fee COMPRESSION shrinking the spread, not
  economics. Only a rising STANDARDISED beta is evidence for your story.

  Caveat you cannot dodge: Sheng, Simutin & Zhang (RAPS 2023) show the
  high-fee 'outperformance' is a profitability and investment tilt. Until
  RMW and CMA are controlled, a positive beta is a style tilt, not skill.
""")
