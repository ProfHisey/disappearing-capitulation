"""Stage 3: first Kaplan-Meier capitulation curve (pilot, 1980-2009).

Design (pilot version of brief section 8):
  Risk set entry: fund is genuinely active (AS >= 70%) and enters an
    underperformance spell (trailing 4-quarter return < its benchmark's).
  Event: Active Share first crosses below 60% (closet-index territory).
  Censoring: spell ends (fund pulls back above benchmark on trailing 4q),
    fund leaves the sample, or data ends.
  Duration: quarters since spell start.

Outputs (aggregates only):
  output/km_pilot.png            KM curve, overall + by spell-depth tercile
  output/km_survival_table.csv   survival function table (for Prism later)
  output/km_report.txt           counts, event totals, logrank p
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

import pilot_lib as P

log = ["KM PILOT REPORT (1980-2009, Petajisto data)", "=" * 60]

df = pd.read_parquet(P.CACHE / "pet_panel.parquet")
df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
df = df.dropna(subset=["activeshare", "qret", "bench_qret"])
# guard: one row per fund-quarter (keep the latest report date in the quarter)
df = (df.sort_values(["wficn", "quarter", "rdate"])
        .drop_duplicates(["wficn", "quarter"], keep="last"))
if "indexfund" in df.columns:
    df = df[pd.to_numeric(df["indexfund"], errors="coerce").fillna(0) == 0]
df = df.sort_values(["wficn", "quarter"])
log.append(f"usable fund-quarters: {len(df):,}; funds: {df['wficn'].nunique():,}")

# trailing 4-quarter cumulative relative return per fund
def add_trailing(g: pd.DataFrame) -> pd.DataFrame:
    g = g.set_index("quarter").asfreq("Q")  # explicit gaps
    f = (1 + g["qret"]).rolling(4).apply(np.prod, raw=True) - 1
    b = (1 + g["bench_qret"]).rolling(4).apply(np.prod, raw=True) - 1
    g["rel4q"] = f - b
    return g.reset_index()

df = df.groupby("wficn", group_keys=False).apply(add_trailing)

spells, events = [], 0
for wficn, g in df.groupby("wficn"):
    g = g.reset_index(drop=True)
    in_spell = False
    for i in range(len(g)):
        row = g.loc[i]
        if not in_spell:
            if (pd.notna(row["rel4q"]) and row["rel4q"] < 0
                    and pd.notna(row["activeshare"])
                    and row["activeshare"] >= P.ACTIVE_START):
                in_spell, start_i, depth = True, i, float(row["rel4q"])
        else:
            depth = min(depth, float(row["rel4q"])) if pd.notna(row["rel4q"]) else depth
            dur = i - start_i
            if pd.notna(row["activeshare"]) and row["activeshare"] < P.CLOSET_CUTOFF:
                spells.append((wficn, dur, 1, depth)); events += 1; in_spell = False
            elif pd.notna(row["rel4q"]) and row["rel4q"] >= 0:
                spells.append((wficn, dur, 0, depth)); in_spell = False
            elif i == len(g) - 1 or pd.isna(row["activeshare"]):
                spells.append((wficn, max(dur, 1), 0, depth)); in_spell = False

sp = pd.DataFrame(spells, columns=["wficn", "dur_q", "event", "depth"])
sp = sp[sp["dur_q"] > 0]
log.append(f"underperformance spells (from genuinely-active start): {len(sp):,}")
log.append(f"capitulation events (AS crossed below {P.CLOSET_CUTOFF:.0%}): "
           f"{int(sp['event'].sum()):,} ({sp['event'].mean():.1%} of spells)")
log.append(f"median spell length: {sp['dur_q'].median():.0f} quarters; "
           f"max: {sp['dur_q'].max():.0f}")

if sp["event"].sum() < 10:
    log.append("\nVERDICT INPUT: fewer than 10 events - capitulation as defined is "
               "too rare in 1980-2009 data; revisit definitions before go/no-go.")

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(7.5, 5))
kmf.fit(sp["dur_q"], sp["event"], label=f"All spells (n={len(sp):,})")
kmf.plot_survival_function(ax=ax, lw=2)
kmf.survival_function_.to_csv(P.OUT / "km_survival_table.csv")

sp["depth_ter"] = pd.qcut(sp["depth"], 3, labels=["shallow", "mid", "deep"])
for lab, sub in sp.groupby("depth_ter", observed=True):
    KaplanMeierFitter().fit(sub["dur_q"], sub["event"],
                            label=f"{lab} (n={len(sub):,})") \
                       .plot_survival_function(ax=ax, lw=1.1, alpha=0.8)
try:
    lr = multivariate_logrank_test(sp["dur_q"], sp["depth_ter"], sp["event"])
    log.append(f"logrank across depth terciles: p = {lr.p_value:.4f}")
except Exception as e:  # noqa: BLE001
    log.append(f"logrank failed: {e}")

ax.axvline(12, color="0.6", ls=":", lw=1)
ax.text(12.2, 0.05, "~3 years", fontsize=8, color="0.4")
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Share still genuinely active (AS >= 60%)")
ax.set_title("Survival of active conviction under underperformance (pilot)")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "km_pilot.png", dpi=200)

log.append("\nKM DONE - km_pilot.png, km_survival_table.csv, km_report.txt "
           "are aggregate-only and shareable.")
P.write_report("km_report.txt", log)
print("\n".join(log))
