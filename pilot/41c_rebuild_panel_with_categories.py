# 41c_rebuild_panel_with_categories.py
# Rebuilds s41_panel.parquet WITH fund categories. The previous build fell
# back to cat="ALL" because WRDS CSVs are latin-1 and the failure was
# swallowed by a try/except. No silent fallback here: if the category join
# fails, this stops and says why.
import os, sys, numpy as np, pandas as pd

DATA_LIB = os.environ.get("DATA_LIB", r"E:\Finance\data\sources")
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
FS    = os.path.join(DATA_LIB, "crsp_mf", "Fund Summary.csv")
MFL   = os.path.join(DATA_LIB, "mflinks", "mflink1.csv")
MIN_EXP, MAX_EXP, START_YEAR = 0.0001, 0.10, 1989

for p in (FS, MFL):
    if not os.path.exists(p):
        sys.exit(f"MISSING: {p}")

def cols_of(path):
    h = pd.read_csv(path, nrows=3, encoding="latin-1", low_memory=False)
    return {c.lower(): c for c in h.columns}

fs_c, ml_c = cols_of(FS), cols_of(MFL)
for name, c, label in [("crsp_fundno", fs_c, "Fund Summary"),
                       ("caldt", fs_c, "Fund Summary"),
                       ("crsp_obj_cd", fs_c, "Fund Summary"),
                       ("crsp_fundno", ml_c, "mflink1"),
                       ("wficn", ml_c, "mflink1")]:
    if name not in c:
        sys.exit(f"column '{name}' not found in {label}. available: {list(c.values())[:25]}")

link = pd.read_csv(MFL, encoding="latin-1",
                   usecols=[ml_c["crsp_fundno"], ml_c["wficn"]])
link = link.rename(columns={ml_c["crsp_fundno"]: "crsp_fundno",
                            ml_c["wficn"]: "wficn"}).dropna().drop_duplicates()
print(f"mflink1: {len(link):,} rows, {link.wficn.nunique():,} wficn")

# rename BY NAME - usecols returns columns in file order, not the order listed
ren = {fs_c["crsp_fundno"]: "crsp_fundno", fs_c["caldt"]: "date",
       fs_c["crsp_obj_cd"]: "obj"}
parts = []
for i, ch in enumerate(pd.read_csv(FS, encoding="latin-1", low_memory=False,
                                   usecols=list(ren.keys()), chunksize=2_000_000)):
    ch = ch.rename(columns=ren)
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch.dropna(subset=["date", "crsp_fundno"])
    ch["year"] = ch["date"].dt.year
    parts.append(ch.loc[ch.year >= START_YEAR, ["crsp_fundno", "year", "obj"]])
    print(f"  chunk {i+1}: {len(parts[-1]):,} kept")

s = pd.concat(parts, ignore_index=True).merge(link, on="crsp_fundno", how="inner")
s["cat"] = s["obj"].astype(str).str.upper().str.strip().str[:4]
s = s[s["cat"].str.startswith("ED")]
cat = (s.groupby(["wficn", "year"], as_index=False)
         .agg(cat=("cat", lambda x: x.mode().iat[0] if len(x.mode()) else np.nan)))
print(f"categories: {cat.cat.nunique()} codes on {cat.wficn.nunique():,} funds")
print(cat.cat.value_counts().head(12).to_string())

fm  = pd.read_parquet(os.path.join(CACHE, "fund_month_v3_tnafix.parquet"))
cov = pd.read_parquet(os.path.join(CACHE, "covars.parquet"))
flg = pd.read_parquet(os.path.join(CACHE, "flags.parquet"))
keep = flg.loc[flg.dom_eq & ~flg.passive, "wficn"].unique()
fm = fm[fm.wficn.isin(keep)].copy()
print(f"universe: {len(keep):,} active domestic-equity funds")

bad = ((cov.exp_ratio < MIN_EXP) | (cov.exp_ratio > MAX_EXP)).sum()
cov = cov.copy()
cov.loc[(cov.exp_ratio < MIN_EXP) | (cov.exp_ratio > MAX_EXP), "exp_ratio"] = np.nan
print(f"fee cleaning: {bad:,} values outside [{MIN_EXP:.2%},{MAX_EXP:.0%}] set missing")
cov["year"] = pd.PeriodIndex(cov.quarter.astype(str), freq="Q").year
fee = (cov.dropna(subset=["exp_ratio"]).sort_values(["wficn", "year"])
          .groupby(["wficn", "year"], as_index=False)
          .agg(exp_ratio=("exp_ratio", "last")))

fm["ym"] = pd.PeriodIndex(fm.caldt, freq="M")
fm["year"] = fm.ym.dt.year
panel = (fm.rename(columns={"fret": "ret"})[["wficn", "ym", "year", "ret", "tna"]]
           .dropna(subset=["ret"])
           .merge(fee, on=["wficn", "year"], how="left")
           .merge(cat, on=["wficn", "year"], how="left"))
panel["fundgrp"] = panel.wficn.astype(str)

cov_pct = 100 * panel.cat.notna().mean()
print(f"\nCATEGORY COVERAGE: {cov_pct:.1f}% of fund-months")
if cov_pct < 50:
    sys.exit("coverage too low - stopping rather than falling back to pooled")
panel["cat"] = panel.cat.fillna("UNK")
panel.to_parquet(os.path.join(CACHE, "s41_panel.parquet"), index=False)
print(f"wrote s41_panel.parquet: {panel.wficn.nunique():,} funds, {len(panel):,} rows")

# force 41 to rebuild its formation table off the new panel
f = os.path.join(CACHE, "s41_formations.parquet")
if os.path.exists(f):
    os.remove(f); print("removed stale s41_formations.parquet")
print("\nnow run:  python 41_fee_vs_performance_probe.py")
