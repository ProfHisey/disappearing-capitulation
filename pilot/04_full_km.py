"""Stage 4: FULL-PERIOD manager capitulation hazard, 1979-2023 (ND Active Share).

What this adds over the 03 pilot (1980-2009, Petajisto data):
  - MFLINKS join: ND Active Share panel (wficn) x CRSP returns (crsp_fundno),
    TNA-weighted across ALL share classes (drops the largest-class approximation).
  - Index-fund / ETF screen via CRSP fund summary flags (the ND data has none).
  - Benchmark core-index returns spliced: CPZ actual index returns through
    2011-02, then French 6-portfolio size proxies after (large=BIG, small=SMALL,
    mid=all six) - with the overlap correlation reported as a validity check.
  - Era cohorts (1980-94 / 1995-2009 / 2010-23) - first look at secular change.

Requires: 01_build_panel.py has run (uses its cached as_panel + monthly returns).
Outputs (aggregates only): output/full_km_report.txt, km_full.png,
km_full_survival.csv, km_full_by_era.csv.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import pilot_lib as P

MFLINK1 = P.SOURCES / "mflinks" / "mflink1.csv"
FRENCH_DIR = Path(r"E:\Finance\BuyRisk\data\sources\french")
F_6PORT = FRENCH_DIR / "6_Portfolios_2x3.csv"
SPLICE_END = pd.Period("2011Q1", freq="Q")  # CPZ covers through 2011-02

log = ["FULL-PERIOD KM REPORT (1979-2023, ND Active Share + MFLINKS)", "=" * 60]

# ------------------------------------------------------------- mflink1 ----
m1 = P.norm_cols(pd.read_csv(MFLINK1))
log.append(f"mflink1 columns: {', '.join(m1.columns)}")
fcol = next(c for c in m1.columns if "fundno" in c)
wcol = next(c for c in m1.columns if "wficn" in c)
m1 = (m1[[fcol, wcol]].dropna().astype("int64")
      .rename(columns={fcol: "crsp_fundno", wcol: "wficn"})
      .drop_duplicates("crsp_fundno"))
log.append(f"mflink1: {len(m1):,} share-class links to {m1['wficn'].nunique():,} wficn funds")

# ------------------------------- CRSP monthly returns -> wficn quarterly ----
ret = P.load_monthly_returns(log)
ret = ret.merge(m1, on="crsp_fundno", how="inner")
log.append(f"CRSP return rows matched to wficn: {len(ret):,}")

ret = ret.sort_values(["crsp_fundno", "caldt"])
ret["w"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
ret["w"] = ret["w"].fillna(ret["mtna"]).clip(lower=0)
ret = ret.dropna(subset=["mret"])
ret["wr"] = ret["w"] * ret["mret"]
g = ret.groupby(["wficn", "caldt"])
fund_m = g.agg(wr=("wr", "sum"), w=("w", "sum"), tna=("mtna", "sum")).reset_index()
fund_m["fret"] = np.where(fund_m["w"] > 0, fund_m["wr"] / fund_m["w"], np.nan)
fund_m = fund_m.dropna(subset=["fret"])
fund_m["quarter"] = fund_m["caldt"].dt.to_period("Q")
fq = (fund_m.assign(gross=lambda d: 1 + d["fret"])
            .groupby(["wficn", "quarter"])
            .agg(qret=("gross", lambda x: x.prod() - 1), nm=("gross", "size"),
                 tna=("tna", "last"))
            .reset_index())
fq = fq[fq["nm"] == 3].drop(columns="nm")
log.append(f"wficn-quarter returns (TNA-weighted across share classes): {len(fq):,}")

# --------------------------------- index-fund screen from Fund Summary ----
flags_pq = P.CACHE / "flags.parquet"
if flags_pq.exists():
    fl = pd.read_parquet(flags_pq)
else:
    flags = []
    usecols = ["crsp_fundno", "index_fund_flag", "et_flag", "crsp_obj_cd"]
    for chunk in pd.read_csv(P.F_SUMMARY, usecols=lambda c: c.strip().lower() in usecols,
                             chunksize=500_000, low_memory=False, encoding="latin-1"):
        chunk = P.norm_cols(chunk)
        flags.append(chunk)
    fl = pd.concat(flags, ignore_index=True)
    fl["passive"] = fl["index_fund_flag"].notna() | fl["et_flag"].notna()
    fl["dom_eq"] = fl["crsp_obj_cd"].astype(str).str.startswith("ED")
    fl = (fl.groupby("crsp_fundno")
            .agg(passive=("passive", "any"), dom_eq=("dom_eq", "any")).reset_index()
            .merge(m1, on="crsp_fundno", how="inner")
            .groupby("wficn").agg(passive=("passive", "any"), dom_eq=("dom_eq", "any"))
            .reset_index())
    fl.to_parquet(flags_pq, index=False)
log.append(f"flags: {int(fl['passive'].sum()):,} wficn funds flagged index/ETF "
           f"(screened out); {int(fl['dom_eq'].sum()):,} flagged domestic equity")

# --------------------------------------------- ND Active Share panel ----
as_pq = P.CACHE / "as_panel.parquet"
if not as_pq.exists():
    P.fail("cache/as_panel.parquet missing - run 01_build_panel.py first")
asp = pd.read_parquet(as_pq).dropna(subset=["wficn"])
asp["wficn"] = asp["wficn"].astype("int64")
asp["quarter"] = asp["month"].dt.to_period("Q")
asp = (asp.sort_values(["wficn", "quarter", "total_assets"])
          .drop_duplicates(["wficn", "quarter"], keep="last"))
n0 = asp["wficn"].nunique()
asp = asp.merge(fl, on="wficn", how="left")
asp = asp[asp["passive"] != True]  # noqa: E712  (NaN = unknown -> keep)
log.append(f"AS panel: {n0:,} funds before screen, {asp['wficn'].nunique():,} after "
           f"index/ETF screen; {len(asp):,} fund-quarters "
           f"({asp['quarter'].min()} to {asp['quarter'].max()})")

# ------------------- staleness diagnostic: is measured AS frozen? ----------
# Thomson holdings have documented quality problems ~2011+. A stale sensor shows
# up as quarter-to-quarter AS changes of exactly zero. Compare eras/sources.
d = asp.sort_values(["wficn", "quarter"]).copy()
d["das"] = d.groupby("wficn")["as_min"].diff()
d = d.dropna(subset=["das"])
d["bucket"] = pd.cut(d["quarter"].dt.year, [0, 1994, 2004, 2010, 2019, 9999],
                     labels=["1980-94", "1995-2004", "2005-10",
                             "2011-19 (TR suspect)", "2020-23 (CRSP)"])
log.append("\nAS staleness diagnostic (quarter-to-quarter changes in min-AS):")
for b, s in d.groupby("bucket", observed=True):
    log.append(f"  {b}: frozen (|dAS|<1e-6) {(s['das'].abs() < 1e-6).mean():.1%}; "
               f"median |dAS| {s['das'].abs().median():.4f}  (n={len(s):,})")

# ----------------------- benchmark core returns: CPZ + French splice ----
cpz = P.load_cpz_monthly(log)

def parse_french_first_block(path: Path) -> pd.DataFrame:
    rows, header, started = [], None, False
    for ln in path.read_text(encoding="latin-1", errors="replace").splitlines():
        cells = [c.strip() for c in ln.split(",")]
        if header is None:
            if len(cells) >= 6 and cells[0] == "" and sum(bool(c) for c in cells[1:]) >= 5:
                header = cells[1:]
            continue
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            started = True
            rows.append(cells)
        elif started:
            break
    df = pd.DataFrame(rows, columns=["ym"] + header)
    for c in header:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([-99.99, -999], np.nan) / 100
    df["month"] = P.parse_ym(df["ym"])
    return df

f6 = parse_french_first_block(F_6PORT)
small_cols = [c for c in f6.columns if "SMALL" in c.upper() or c.upper().startswith("ME1")]
big_cols = [c for c in f6.columns if "BIG" in c.upper() or c.upper().startswith("ME2")]
f6["p_s5"] = f6[big_cols].mean(axis=1)
f6["p_r2"] = f6[small_cols].mean(axis=1)
f6["p_rm"] = f6[big_cols + small_cols].mean(axis=1)
log.append(f"french 6-portfolios: {f6['month'].min():%Y-%m} to {f6['month'].max():%Y-%m} "
           f"(small={small_cols}, big={big_cols})")

ov = cpz.merge(f6[["month", "p_s5", "p_r2", "p_rm"]], on="month", how="inner")
for a, b in (("idx_s5", "p_s5"), ("idx_r2", "p_r2"), ("idx_rm", "p_rm")):
    log.append(f"  splice check corr({a},{b}) over {len(ov)} overlap months: "
               f"{ov[a].corr(ov[b]):.4f}")

def to_q(df, cols):
    d = df.copy()
    d["quarter"] = d["month"].dt.to_period("Q")
    return (d.set_index("quarter")[cols].add(1).groupby(level=0).prod().sub(1))

bq = pd.concat([
    to_q(cpz, ["idx_s5", "idx_r2", "idx_rm"]).loc[lambda d: d.index <= SPLICE_END],
    to_q(f6, ["p_s5", "p_r2", "p_rm"])
        .rename(columns={"p_s5": "idx_s5", "p_r2": "idx_r2", "p_rm": "idx_rm"})
        .loc[lambda d: d.index > SPLICE_END],
]).sort_index().reset_index()
log.append(f"benchmark core returns spliced at {SPLICE_END} "
           f"({bq['quarter'].min()} to {bq['quarter'].max()})")

# ----------------------------------------------------- join everything ----
asp["core"] = asp["bench_min"].astype(str).str.upper().map(
    {k: v.replace("idx_", "") for k, v in P.BENCH_TO_CORE.items()})
panel = (asp.merge(fq, on=["wficn", "quarter"], how="inner")
            .merge(bq, on="quarter", how="left"))
panel["bench_qret"] = np.select(
    [panel["core"] == "s5", panel["core"] == "r2", panel["core"] == "rm"],
    [panel["idx_s5"], panel["idx_r2"], panel["idx_rm"]], default=np.nan)
panel = panel.dropna(subset=["as_min", "qret", "bench_qret"])
match = panel["wficn"].nunique()
log.append(f"\nKM-usable: {len(panel):,} fund-quarters, {match:,} funds "
           f"({panel['quarter'].min()} to {panel['quarter'].max()})")

# --------------------------------------------- spells + KM (as in 03) ----
panel = panel.sort_values(["wficn", "quarter"])

def add_trailing(g):
    g = g.set_index("quarter").asfreq("Q")
    f = (1 + g["qret"]).rolling(4).apply(np.prod, raw=True) - 1
    b = (1 + g["bench_qret"]).rolling(4).apply(np.prod, raw=True) - 1
    g["rel4q"] = f - b
    return g.reset_index()

panel = (panel.groupby("wficn", group_keys=True)[
             ["quarter", "as_min", "qret", "bench_qret"]]
         .apply(add_trailing)
         .reset_index(level=0)              # restore wficn as a column
         .reset_index(drop=True))

spells, reldrops = [], 0
for wficn, g in panel.groupby("wficn"):
    g = g.reset_index(drop=True)
    in_spell = False
    for i in range(len(g)):
        row = g.loc[i]
        if not in_spell:
            if (pd.notna(row["rel4q"]) and row["rel4q"] < 0
                    and pd.notna(row["as_min"]) and row["as_min"] >= P.ACTIVE_START):
                in_spell, start_i, depth, asmax = True, i, float(row["rel4q"]), float(row["as_min"])
        else:
            depth = min(depth, float(row["rel4q"])) if pd.notna(row["rel4q"]) else depth
            if pd.notna(row["as_min"]):
                asmax = max(asmax, float(row["as_min"]))
                if asmax - row["as_min"] >= 0.20:
                    reldrops += 1  # info only: fund-relative conviction drop
            dur = i - start_i
            if pd.notna(row["as_min"]) and row["as_min"] < P.CLOSET_CUTOFF:
                spells.append((wficn, g.loc[start_i, "quarter"].year, dur, 1, depth))
                in_spell = False
            elif pd.notna(row["rel4q"]) and row["rel4q"] >= 0:
                spells.append((wficn, g.loc[start_i, "quarter"].year, dur, 0, depth))
                in_spell = False
            elif i == len(g) - 1 or pd.isna(row["as_min"]):
                spells.append((wficn, g.loc[start_i, "quarter"].year, max(dur, 1), 0, depth))
                in_spell = False

sp = pd.DataFrame(spells, columns=["wficn", "start_yr", "dur_q", "event", "depth"])
sp = sp[sp["dur_q"] > 0]
sp.to_parquet(P.CACHE / "spells.parquet", index=False)
sp["era"] = pd.cut(sp["start_yr"], [0, 1994, 2009, 9999],
                   labels=["1980-94", "1995-2009", "2010-23"])
fine = pd.cut(sp["start_yr"], [0, 1994, 2009, 2019, 9999],
              labels=["1980-94", "1995-2009", "2010-19 (TR sensor)",
                      "2020-23 (CRSP sensor)"])
log.append("\nfine era split (sensor-staleness check on the era result):")
for b, s in sp.groupby(fine, observed=True):
    log.append(f"  {b}: {len(s):,} spells, {int(s['event'].sum()):,} events "
               f"({s['event'].mean():.1%})")
log.append(f"spells: {len(sp):,}; events: {int(sp['event'].sum()):,} "
           f"({sp['event'].mean():.1%}); median dur {sp['dur_q'].median():.0f}q")
log.append(f"fund-relative >=20pt AS drops during spells (info): {reldrops:,}")
log.append("\nby era:")
for era, s in sp.groupby("era", observed=True):
    log.append(f"  {era}: {len(s):,} spells, {int(s['event'].sum()):,} events "
               f"({s['event'].mean():.1%})")

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7.5, 5))
kmf.fit(sp["dur_q"], sp["event"], label=f"All spells (n={len(sp):,})")
kmf.plot_survival_function(ax=ax, lw=2)
kmf.survival_function_.to_csv(P.OUT / "km_full_survival.csv")
era_tables = {}
for era, s in sp.groupby("era", observed=True):
    k = KaplanMeierFitter().fit(s["dur_q"], s["event"], label=f"{era} (n={len(s):,})")
    k.plot_survival_function(ax=ax, lw=1.2, alpha=0.85)
    era_tables[str(era)] = k.survival_function_.iloc[:, 0]
pd.DataFrame(era_tables).to_csv(P.OUT / "km_full_by_era.csv")
try:
    lr = multivariate_logrank_test(sp["dur_q"], sp["era"], sp["event"])
    log.append(f"\nlogrank across eras: p = {lr.p_value:.4f}")
    lr2 = multivariate_logrank_test(sp["dur_q"], pd.qcut(sp["depth"], 3,
                                    labels=["shallow", "mid", "deep"]), sp["event"])
    log.append(f"logrank across depth terciles: p = {lr2.p_value:.4f}")
except Exception as e:  # noqa: BLE001
    log.append(f"logrank failed: {e}")

ax.axvline(12, color="0.6", ls=":", lw=1)
ax.text(12.2, 0.05, "~3 years", fontsize=8, color="0.4")
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Share still genuinely active (min-AS >= 60%)")
ax.set_title("Survival of active conviction, 1980-2023 (full period, by era)")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "km_full.png", dpi=200)

log.append("\nFULL KM DONE - all outputs aggregate-only and shareable.")
P.write_report("full_km_report.txt", log)
print("\n".join(log))
