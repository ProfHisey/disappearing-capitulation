# 44_fee_size_double_sort.py -- is the expensive tenth bad because it is
# EXPENSIVE, or because it is SMALL? Fee and size are ~9x apart across the
# extreme deciles even within category, so they must be separated before
# "avoid the expensive tenth" can be claimed.
#
# Run after 41c + 41.   python 44_fee_size_double_sort.py
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

ft = pd.read_parquet(os.path.join(CACHE, "s41_formations.parquet"))
ft = ft.dropna(subset=["fwd1", "exp_ratio", "cat", "tna"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)].copy()
ft["ltna"] = np.log(ft.tna)
by = ["form_year", "cat"]
ft["excess1"] = ft.fwd1 - ft.groupby(by)["fwd1"].transform("mean")

def bucket(s, n):
    try: return pd.qcut(s.rank(method="first"), n, labels=False) + 1
    except ValueError: return pd.Series(np.nan, index=s.index)

ft["fee_dec"]  = ft.groupby(by)["exp_ratio"].transform(lambda s: bucket(s, 10))
ft["size_ter"] = ft.groupby(by)["ltna"].transform(lambda s: bucket(s, 3))
ft = ft.dropna(subset=["fee_dec", "size_ter"])
print(f"sample: {len(ft):,} fund-years, {ft.fundgrp.nunique():,} funds, "
      f"{ft.cat.nunique()} categories, UNK dropped")

# ---- 1. double sort: fee decile x size tercile -------------------------
print("\n" + "="*72); print("1. MEAN EXCESS FORWARD-1Y RETURN (%), BY FEE DECILE x SIZE TERCILE")
print("="*72)
cell = (ft.groupby(["fee_dec", "size_ter", "form_year"])["excess1"].mean()
          .groupby(["fee_dec", "size_ter"]).mean().unstack() * 100)
cell.columns = ["small", "mid", "large"]
print(cell.round(2).to_string())
cell.round(3).to_csv(os.path.join(OUT, "s44_fee_x_size.csv"))
d10 = cell.loc[10]
print(f"\ndecile 10 by size:  small {d10['small']:+.2f}  mid {d10['mid']:+.2f}  "
      f"large {d10['large']:+.2f}")
print("If decile 10 is bad in ALL THREE size buckets, the fee is doing the")
print("work. If it is bad only among small funds, you have found a size")
print("effect and 'avoid the expensive tenth' needs rewording.")

# ---- 2. Fama-MacBeth: fee vs size vs momentum --------------------------
print("\n" + "="*72); print("2. FAMA-MACBETH: what predicts next year, once everything is in?")
print("="*72)
ft["fee_z"]  = ft.groupby(by)["exp_ratio"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
ft["size_z"] = ft.groupby(by)["ltna"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
ft["mom_z"]  = ft.groupby(by)["trail12"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
ft["d10"]    = (ft.fee_dec == 10).astype(float)

specs = {"fee only": ["fee_z"], "fee + size": ["fee_z", "size_z"],
         "fee + size + momentum": ["fee_z", "size_z", "mom_z"],
         "top-decile dummy + size + mom": ["d10", "size_z", "mom_z"]}
rows = []
for name, xs in specs.items():
    coefs = {x: [] for x in xs}
    for yr, g in ft.dropna(subset=xs + ["excess1"]).groupby("form_year"):
        if len(g) < 100: continue
        X = np.column_stack([np.ones(len(g))] + [g[x].values for x in xs])
        b, *_ = np.linalg.lstsq(X, g["excess1"].values, rcond=None)
        for i, x in enumerate(xs): coefs[x].append(b[i + 1])
    for x in xs:
        a = np.array(coefs[x]); se = a.std(ddof=1) / np.sqrt(len(a))
        rows.append({"spec": name, "variable": x, "mean_coef_pct": a.mean() * 100,
                     "t_stat": a.mean() / se, "n_years": len(a)})
fm = pd.DataFrame(rows)
print(fm.round(3).to_string(index=False))
fm.round(3).to_csv(os.path.join(OUT, "s44_fama_macbeth.csv"), index=False)

# ---- 3. how much of the tenth decile is sector funds? ------------------
print("\n" + "="*72); print("3. WHAT KIND OF FUND IS IN THE EXPENSIVE TENTH?")
print("="*72)
mix = (pd.crosstab(ft.fee_dec, ft.cat.str[:3], normalize="index") * 100)
keep = [c for c in mix.columns if mix[c].max() >= 3]
print(mix[keep].round(1).loc[[1, 5, 10]].to_string())
print("\nEDC = cap-based, EDY = style, EDS = sector. If decile 10 is mostly")
print("EDS, 'expensive' may be standing in for 'sector fund'.")

print("""
PLAIN READING
  The within-category run replicated Nanigian: deciles 1-9 flat, decile 10
  bad. This script asks whether that tenth decile is really about cost.
  Three ways it could fail: it is small funds, it is momentum, or it is
  sector funds. If the fee coefficient survives all three controls, the
  claim 'avoid the expensive tenth' is yours to make.
""")
