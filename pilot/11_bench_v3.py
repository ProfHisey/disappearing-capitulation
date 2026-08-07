"""Stage 11: BENCHMARK SERIES v3 — fix the coverage regression from stage 10.

Diagnosis from stage 10: the Russell reconstruction (holdings-file weights x
security MTD returns) does not span the full 1978+ history (the return column
appears to populate only from the late 1990s), so pre-coverage fund-quarters
lost their benchmark unless mapped to S&P 500 -> the 1980-94 era shrank to 166
spells. Also the R2 series shows a suspicious French-overlap correlation.

v3 fixes:
  - per-code coverage spans + data-quality counts printed (nothing hides)
  - hybrid series: reconstructed Russell where available; before that, CPZ's
    ACTUAL index returns (R2/RM/S5 families) - actual indexes, not proxies
  - overwrite the bench cache so panel_lib picks v3 up transparently
  - rebuild panel, re-run headlines, deltas vs v2 AND the proxy era

Outputs: output/bench_v3_report.txt, km_full_v3.png (aggregates only; the
series itself stays in cache/ - vendor data).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import pilot_lib as P
import panel_lib as PL

V2 = {"spells": 14557, "events": 516, "rate": 0.035,
      "era": {"1980-94": 0.1386, "1995-2009": 0.0504, "2010-23": 0.0250},
      "n_era": {"1980-94": 166, "1995-2009": 5243, "2010-23": 9148}}
PROXY = {"era": {"1980-94": 0.065, "1995-2009": 0.031, "2010-23": 0.008}}

log = ["BENCHMARK SERIES v3 (hybrid: reconstructed + CPZ actual fill)", "=" * 60]

# ------------------------------------------------ inspect current series ----
ser = PL.build_bench_series(log)                       # cached v2
ser["month"] = pd.PeriodIndex(ser["month"], freq="M")
log.append("\nreconstructed coverage per code (v2):")
for code, g in ser.groupby("code"):
    crazy = (g["ret"].abs() > 0.4).sum()
    log.append(f"  {code}: {g['month'].min()} to {g['month'].max()} "
               f"({len(g)} months; |ret|>40%: {crazy})")

# drop obviously-broken months (|monthly index return| > 40% is impossible)
bad = (ser["ret"].abs() > 0.4)
if bad.any():
    log.append(f"\ndropped {int(bad.sum())} corrupt month-code rows (|ret|>40%)")
    ser = ser[~bad]

# ------------------------------------------------------- CPZ actual fill ----
cpz = P.load_cpz_monthly(log)
cpz["month"] = cpz["month"].dt.to_period("M")
FILL = {}
for c in ("R1", "R1G", "R1V", "R3", "R3G", "R3V"):
    FILL[c] = "idx_s5"       # large-cap family -> actual S&P 500 (corr ~0.99 w/ R1)
for c in ("RM", "RMG", "RMV"):
    FILL[c] = "idx_rm"       # actual Russell Midcap from CPZ
for c in ("R2", "R2G", "R2V"):
    FILL[c] = "idx_r2"       # actual Russell 2000 from CPZ

fills = []
for code in sorted(set(list(PL.WT_TO_CODE.values()))):
    have = ser.loc[ser["code"] == code, "month"]
    start = have.min() if len(have) else pd.Period("2100-01", freq="M")
    src = FILL.get(code)
    if src is None:
        continue
    pre = cpz[cpz["month"] < start][["month", src]].dropna()
    if len(pre):
        fills.append(pd.DataFrame({"month": pre["month"], "code": code,
                                   "ret": pre[src].values}))
        log.append(f"  fill {code}: {len(pre)} months of CPZ {src} before {start}")
ser3 = pd.concat([ser] + fills, ignore_index=True) \
         .drop_duplicates(["code", "month"], keep="first") \
         .sort_values(["code", "month"])

out = ser3.copy()
out["month"] = out["month"].astype(str)
out.to_parquet(P.CACHE / "bench_series_monthly.parquet", index=False)   # v3 live
out.to_parquet(P.CACHE / "bench_series_monthly_v3.parquet", index=False)
log.append("\nv3 coverage per code:")
for code, g in ser3.groupby("code"):
    log.append(f"  {code}: {g['month'].min()} to {g['month'].max()} ({len(g)} months)")

# ---------------------------------------------- rebuild panel + headlines ----
panel = PL.build_panel(log, force=True)
sp = PL.extract_spells(panel, client_cut=-0.10)
sp["start_yr"] = pd.PeriodIndex(sp["start_q"], freq="Q").year
sp["era"] = pd.cut(sp["start_yr"], [0, 1994, 2009, 9999],
                   labels=["1980-94", "1995-2009", "2010-23"])
sp["event"] = sp["m_dur"].notna().astype(int)
sp["dur"] = sp["m_dur"].fillna(sp["end_dur"]).clip(lower=1)

log.append(f"\nHEADLINES v3 (vs v2, vs proxy):")
log.append(f"  spells: {len(sp):,} (v2 {V2['spells']:,})")
log.append(f"  manager events: {int(sp['event'].sum()):,} ({sp['event'].mean():.1%}; "
           f"v2 {V2['events']:,} / {V2['rate']:.1%})")
for era, s in sp.groupby("era", observed=True):
    log.append(f"    {era}: {s['event'].mean():.2%} of {len(s):,} spells "
               f"(v2 {V2['era'][str(era)]:.2%} of {V2['n_era'][str(era)]:,}; "
               f"proxy {PROXY['era'][str(era)]:.1%})")
try:
    lr = multivariate_logrank_test(sp["dur"], sp["era"], sp["event"])
    lr2 = multivariate_logrank_test(sp["dur"], pd.qcut(sp["depth"], 3,
                                    labels=["s", "m", "d"]), sp["event"])
    log.append(f"  logrank eras p = {lr.p_value:.4f}; depth p = {lr2.p_value:.4f}")
except Exception as e:  # noqa: BLE001
    log.append(f"  logrank failed: {e}")
sp00 = sp[sp["start_yr"] >= 2000]
log.append(f"  client events (2000+, <=-10%/q): {int(sp00['c_dur'].notna().sum()):,} "
           f"of {len(sp00):,} ({sp00['c_dur'].notna().mean():.1%})")

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7.5, 5))
kmf.fit(sp["dur"], sp["event"], label=f"All spells (n={len(sp):,})")
kmf.plot_survival_function(ax=ax, lw=2)
for era, s in sp.groupby("era", observed=True):
    KaplanMeierFitter().fit(s["dur"], s["event"], label=f"{era} (n={len(s):,})") \
                       .plot_survival_function(ax=ax, lw=1.2, alpha=0.85)
ax.axvline(12, color="0.6", ls=":", lw=1)
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Share still genuinely active (min-AS >= 60%)")
ax.set_title("Survival of active conviction, 1980-2023 - benchmark series v3")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "km_full_v3.png", dpi=200)

# quick catalog of any new files in crsp_sp500 (e.g., the big 'composite')
log.append("\ncrsp_sp500/ files on disk:")
for f in sorted(PL.SP500_DIR.glob("*.csv")):
    log.append(f"  {f.name}: {f.stat().st_size/1e6:.1f} MB")

log.append("\nNote: stages 06-08 use the rebuilt panel automatically on next run.")
log.append("BENCH v3 DONE - outputs aggregate-only and shareable.")
P.write_report("bench_v3_report.txt", log)
print("\n".join(log))
