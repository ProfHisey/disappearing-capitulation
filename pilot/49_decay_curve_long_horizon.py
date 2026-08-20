# 49_decay_curve_long_horizon.py -- the figure the paper is built around.
#
# Stage 48 showed the hot pick's GROSS edge decaying (51 -> 14bp for a menu
# of 3) while the fee penalty stays flat (~50-65bp every year). Investors
# do not rebalance: Vanguard reports 90% of DC participants made no
# exchanges in 2020 and 5% traded during 2026 volatility. So the horizon
# that matters is decades, not one year.
#
# This extends the horizon to 1/3/5/10/20 years and does it under TWO
# reinvestment rules, because dead funds must go somewhere and the choice
# moves the answer:
#   SURVIVORS  - fund must be alive at the horizon (survivorship-biased,
#                upward, and biased toward whichever rule picks hardier funds)
#   REINVEST   - on death, proceeds go into the category's equal-weight
#                return for the remainder (the honest default)
#
#   python 49_decay_curve_long_horizon.py
import os, numpy as np, pandas as pd

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
HORIZONS, MENU_SIZES, REPS, SEED = (1, 3, 5, 10, 20), (3, 5, 10), 120, 20260820
rng = np.random.default_rng(SEED)

pn = pd.read_parquet(os.path.join(CACHE, "s45_panel_with_passive.parquet"))
pn = pn.dropna(subset=["ret"]).sort_values(["wficn", "ym"])
pn["cat"] = pn["cat"].fillna("UNK")

# ---- growth indexes: per fund, and per category (the reinvestment vehicle)
pn["cum"] = (1 + pn.ret).groupby(pn.wficn).cumprod()
catm = pn.groupby(["cat", "ym"], observed=True)["ret"].mean().reset_index()
catm["ccum"] = (1 + catm.ret).groupby(catm.cat).cumprod()

dec = pn[pn.ym.dt.month == 12][["wficn", "year", "cat", "cum"]]
cdec = catm[catm.ym.dt.month == 12][["cat", "ccum"]].assign(year=catm.loc[catm.ym.dt.month == 12, "ym"].dt.year.values)
last = pn.groupby("wficn").agg(last_year=("year", "max"), last_cum=("cum", "last"),
                               last_cat=("cat", "last")).reset_index()
cdec_l = cdec.rename(columns={"year": "last_year", "ccum": "ccum_at_death", "cat": "last_cat"})
last = last.merge(cdec_l, on=["last_cat", "last_year"], how="left")

def forward(h):
    """Forward h-year return under both reinvestment rules."""
    a = dec.rename(columns={"cum": "cum0"})
    b = dec[["wficn", "year", "cum"]].rename(columns={"cum": "cumH"})
    b["year"] -= h
    m = a.merge(b, on=["wficn", "year"], how="left")           # NaN => fund died
    m[f"surv{h}"] = m.cumH / m.cum0 - 1
    # reinvestment path: fund value to death, then category growth to horizon
    m = m.merge(last[["wficn", "last_year", "last_cum", "last_cat", "ccum_at_death"]],
                on="wficn", how="left")
    tgt = cdec.rename(columns={"year": "tgt_year", "ccum": "ccum_tgt", "cat": "last_cat"})
    m["tgt_year"] = m.year + h
    m = m.merge(tgt, on=["last_cat", "tgt_year"], how="left")
    dead = m.cumH.isna() & m.ccum_at_death.notna() & m.ccum_tgt.notna()
    m[f"rein{h}"] = m[f"surv{h}"]
    m.loc[dead, f"rein{h}"] = (m.loc[dead, "last_cum"] / m.loc[dead, "cum0"]) * \
                              (m.loc[dead, "ccum_tgt"] / m.loc[dead, "ccum_at_death"]) - 1
    return m[["wficn", "year", f"surv{h}", f"rein{h}"]]

ft = pd.read_parquet(os.path.join(CACHE, "s45_formations.parquet"))
ft = ft[["wficn", "year", "cat", "exp_ratio", "trail12", "tna", "passive"]].copy()
for h in HORIZONS:
    ft = ft.merge(forward(h), on=["wficn", "year"], how="left")
print(f"formations {len(ft):,}; coverage by horizon (reinvest rule):")
for h in HORIZONS:
    print(f"   {h:2d}y  {100*ft[f'rein{h}'].notna().mean():5.1f}%  "
          f"(survivors-only {100*ft[f'surv{h}'].notna().mean():5.1f}%)")

# ---- simulate menus, record both picks at every horizon ----------------
res = []
for K in MENU_SIZES:
    for (y, ct), g in ft.groupby(["year", "cat"]):
        if len(g) < K or ct == "UNK": continue
        w = g.tna.values / g.tna.values.sum(); idx = np.arange(len(g))
        for _ in range(REPS):
            m = g.iloc[rng.choice(idx, size=K, replace=False, p=w)]
            c, hh = m.exp_ratio.idxmin(), m.trail12.idxmax()
            if c == hh: continue
            row = {"K": K, "year": y,
                   "fee_pen": m.loc[hh, "exp_ratio"] - m.loc[c, "exp_ratio"]}
            for h in HORIZONS:
                for rule in ("surv", "rein"):
                    row[f"gap_{rule}{h}"] = m.loc[hh, f"{rule}{h}"] - m.loc[c, f"{rule}{h}"]
                    row[f"cheap_{rule}{h}"] = m.loc[c, f"{rule}{h}"]
                    row[f"hot_{rule}{h}"] = m.loc[hh, f"{rule}{h}"]
            res.append(row)
r = pd.DataFrame(res)
r.to_parquet(os.path.join(CACHE, "s49_long_horizon.parquet"), index=False)
print(f"\ndisagreeing menus: {len(r):,}")

print("\n" + "="*78); print("THE DECAY CURVE -- gross edge vs the flat fee, annualised bps")
print("="*78)
out = []
for rule, lab in [("rein", "reinvest on death"), ("surv", "survivors only")]:
    for K, d in r.groupby("K"):
        for h in HORIZONS:
            s = d.dropna(subset=[f"gap_{rule}{h}"])
            if len(s) < 500: continue
            ann = (1 + s[f"gap_{rule}{h}"]).pow(0)  # placeholder, use simple /h
            net = s[f"gap_{rule}{h}"] / h
            out.append({"rule": lab, "K": K, "horizon": h, "n": len(s),
                        "fee_bps": s.fee_pen.median() * 10000,
                        "net_bps": net.median() * 10000,
                        "gross_bps": (net + s.fee_pen).median() * 10000,
                        "pct_hot_wins": 100 * (s[f"gap_{rule}{h}"] > 0).mean()})
t = pd.DataFrame(out)
t.round(1).to_csv(os.path.join(OUT, "s49_decay_curve.csv"), index=False)
for rule, d in t.groupby("rule"):
    print(f"\n--- {rule} ---")
    print(d.pivot_table(index="horizon", columns="K",
                        values=["gross_bps", "net_bps", "pct_hot_wins"]).round(1).to_string())

print("\n" + "="*78); print("TERMINAL WEALTH OF $10,000 (reinvest rule, median / p25)")
print("="*78)
for K, d in r.groupby("K"):
    print(f"\n menu of {K}")
    for h in (5, 10, 20):
        s = d.dropna(subset=[f"cheap_rein{h}", f"hot_rein{h}"])
        if len(s) < 500: continue
        cm, hm = (1 + s[f"cheap_rein{h}"]) * 10000, (1 + s[f"hot_rein{h}"]) * 10000
        print(f"   {h:2d}y  cheap pick  median ${cm.median():9,.0f}  p25 ${cm.quantile(.25):9,.0f}"
              f"   |  hot pick  median ${hm.median():9,.0f}  p25 ${hm.quantile(.25):9,.0f}"
              f"   | median gap ${cm.median()-hm.median():+8,.0f}")

print("""
PLAIN READING
  Read gross_bps down the horizon column. If it decays toward zero while
  fee_bps stays flat, that is the paper's central figure: a wasting asset
  bought with a permanent liability.
  Then read the two reinvestment rules against each other. If the picture
  holds under both, the result is not a survivorship artifact. If it only
  holds under survivors-only, we have a problem and must say so.
  The terminal-wealth block is the committee exhibit -- and the p25 column
  is the number a fiduciary is actually responsible for.
""")
