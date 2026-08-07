"""Stage 10: REAL BENCHMARKS — retire the proxy splice, re-run the headlines.

1. Build actual monthly benchmark returns: 12 Russell indexes reconstructed
   from the FTSE Russell holdings file (sum of weight x security MTD return,
   1978-12 onward) + CRSP's S&P 500 value-weighted TOTAL return (1925-12 on).
2. Validate: reconstructed series vs the old CPZ/French proxies over their
   overlap; CRSP VW S&P vs the official composite.
3. Rebuild the panel with each fund's min-AS benchmark matched to ITS OWN
   index series (S&P 400/600 + S&P style codes approximated by the nearest
   Russell series until Morningstar Direct delivers them - mapping reported).
4. Re-run the headline results and report DELTAS vs the proxy-era numbers.

Licensing: the benchmark return series live in cache/ only (vendor data);
output/ gets aggregates and validation stats.
Outputs: output/real_benchmarks_report.txt, km_full_v2.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import pilot_lib as P
import panel_lib as PL

PRIOR = {"spells": 25728, "events": 474, "rate": 0.018,
         "era": {"1980-94": 0.065, "1995-2009": 0.031, "2010-23": 0.008}}

log = ["REAL BENCHMARKS UPGRADE", "=" * 60]

# ------------------------------------------------ build + validate series ----
ser = PL.build_bench_series(log, force=False)

old = P.load_cpz_monthly([])                       # actual CPZ to 2011
f6 = PL.parse_french_first_block(PL.F_6PORT)
small = [c for c in f6.columns if "SMALL" in str(c).upper() or str(c).upper().startswith("ME1")]
big = [c for c in f6.columns if "BIG" in str(c).upper() or str(c).upper().startswith("ME2")]
f6["p_s5"], f6["p_r2"] = f6[big].mean(axis=1), f6[small].mean(axis=1)
f6["month"] = f6["month"].dt.to_period("M")
old["month"] = old["month"].dt.to_period("M")

log.append("\nvalidation correlations (monthly, over overlap):")
wide = ser.pivot_table(index="month", columns="code", values="ret")
for code, ref, refname in (("S5", old.set_index("month")["idx_s5"], "CPZ S&P500"),
                           ("R2", old.set_index("month")["idx_r2"], "CPZ R2000"),
                           ("RM", old.set_index("month")["idx_rm"], "CPZ RMid"),
                           ("S5", f6.set_index("month")["p_s5"], "French BIG proxy"),
                           ("R2", f6.set_index("month")["p_r2"], "French SMALL proxy")):
    if code in wide.columns:
        j = pd.concat([wide[code], ref], axis=1, join="inner").dropna()
        log.append(f"  {code} vs {refname}: corr {j.iloc[:, 0].corr(j.iloc[:, 1]):.4f} "
                   f"({len(j)} months)")
try:
    comp = P.norm_cols(pd.read_csv(PL.F_SP_COMP_M))
    comp["month"] = pd.to_datetime(comp["mthcaldt"], errors="coerce").dt.to_period("M")
    j = pd.concat([wide["S5"], comp.set_index("month")["mthtotret"]],
                  axis=1, join="inner").dropna()
    log.append(f"  S5 (VW universe) vs official composite total return: "
               f"corr {j.iloc[:, 0].corr(j.iloc[:, 1]):.4f} ({len(j)} months)")
except Exception as e:  # noqa: BLE001
    log.append(f"  composite check skipped: {e}")
log.append(f"\nS&P-family codes approximated by Russell series until Direct "
           f"delivers them: {PL.BENCH_APPROX}")

# ------------------------------------------------------ rebuild the panel ----
panel = PL.build_panel(log, force=True)
sp = PL.extract_spells(panel, client_cut=-0.10)
sp["start_yr"] = pd.PeriodIndex(sp["start_q"], freq="Q").year
sp["era"] = pd.cut(sp["start_yr"], [0, 1994, 2009, 9999],
                   labels=["1980-94", "1995-2009", "2010-23"])
sp["event"] = sp["m_dur"].notna().astype(int)
sp["dur"] = sp["m_dur"].fillna(sp["end_dur"]).clip(lower=1)

log.append(f"\nHEADLINE RE-RUN (real benchmarks) vs proxy era:")
log.append(f"  spells: {len(sp):,} (was {PRIOR['spells']:,})")
log.append(f"  manager events: {int(sp['event'].sum()):,} "
           f"({sp['event'].mean():.1%}; was {PRIOR['events']:,} / {PRIOR['rate']:.1%})")
log.append("  era rates:")
for era, s in sp.groupby("era", observed=True):
    was = PRIOR["era"].get(str(era), float("nan"))
    log.append(f"    {era}: {s['event'].mean():.2%} of {len(s):,} spells "
               f"(was {was:.1%})")
try:
    lr = multivariate_logrank_test(sp["dur"], sp["era"], sp["event"])
    lr2 = multivariate_logrank_test(sp["dur"], pd.qcut(sp["depth"], 3,
                                    labels=["shallow", "mid", "deep"]), sp["event"])
    log.append(f"  logrank eras p = {lr.p_value:.4f}; depth terciles p = {lr2.p_value:.4f}")
except Exception as e:  # noqa: BLE001
    log.append(f"  logrank failed: {e}")

# client arm quick recount (2000+, -10% threshold, as stage 05)
sp2 = sp[(sp["start_yr"] >= 2000) & sp["c_dur"].notna() | (sp["start_yr"] >= 2000)]
sp00 = sp[sp["start_yr"] >= 2000]
log.append(f"  client events (2000+, flow<=-10%/q): "
           f"{int(sp00['c_dur'].notna().sum()):,} of {len(sp00):,} spells "
           f"({sp00['c_dur'].notna().mean():.1%}; was 27.2%)")

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7.5, 5))
kmf.fit(sp["dur"], sp["event"], label=f"All spells (n={len(sp):,})")
kmf.plot_survival_function(ax=ax, lw=2)
for era, s in sp.groupby("era", observed=True):
    KaplanMeierFitter().fit(s["dur"], s["event"], label=f"{era} (n={len(s):,})") \
                       .plot_survival_function(ax=ax, lw=1.2, alpha=0.85)
ax.axvline(12, color="0.6", ls=":", lw=1)
ax.text(12.2, 0.05, "~3 years", fontsize=8, color="0.4")
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Share still genuinely active (min-AS >= 60%)")
ax.set_title("Survival of active conviction, 1980-2023 - REAL benchmark returns")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "km_full_v2.png", dpi=200)

log.append("\nNote: stages 06-08 will automatically use the v2 panel on their "
           "next run (panel cache rebuilt). Old proxy panel retained as "
           "panel_full.parquet for comparison.")
log.append("REAL BENCHMARKS DONE - outputs aggregate-only and shareable.")
P.write_report("real_benchmarks_report.txt", log)
print("\n".join(log))
