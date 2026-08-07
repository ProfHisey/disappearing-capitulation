"""Stage 7: COMPETING RISKS — did capitulation migrate from closet indexing
to fund death?

For every underperformance spell, classify how it ended:
  capitulated   Active Share collapsed below 60% (the stage-04 event)
  died          spell ran until the fund's data ended AND the fund's share
                classes are recorded dead, with death within 4 quarters of the
                spell's last observation (liquidation/merger = the other way
                to stop being a genuinely active fund)
  recovered     trailing 4q relative return came back above the benchmark
  censored      spell ran to the panel edge with the fund still alive

The era headline from stage 04 (capitulation rates 6.5% -> 3.1% -> 0.8%) is
re-examined with death as a second failure mode: if 'died' rises as
'capitulated' falls, the post-2010 slowdown is partly a change in the FORM of
capitulation, not its amount.

Requires stage 06 to have run (uses panel_full cache). Outputs (aggregates):
  output/competing_risks_report.txt, competing_risks.png, spell_outcomes.csv
Pilot caveats: 4-quarter death-attribution window is a design choice; mergers
mix distress and family reorganization (merged-flag split reported); the real
build uses a proper competing-risks estimator (Aalen-Johansen) and delist codes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

DEATH_WINDOW_Q = 4

log = ["COMPETING RISKS: capitulate vs die (1980-2023)", "=" * 60]

panel = PL.build_panel(log)
sp = PL.extract_spells(panel, client_cut=None)
death = PL.get_death(log)
sp = sp.merge(death, on="wficn", how="left")

sp["end_p"] = pd.PeriodIndex(sp["end_q"], freq="Q")
death_p = pd.PeriodIndex(
    sp["death_q"].where(sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
gap = (death_p - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))

sp["outcome"] = "censored"
sp.loc[sp["ended_by"] == "recovered", "outcome"] = "recovered"
died_mask = (sp["ended_by"].isin(["data_end", "as_missing"])
             & sp["died"].fillna(False)
             & gap.between(-1, DEATH_WINDOW_Q))
sp.loc[died_mask, "outcome"] = "died"
sp.loc[sp["m_dur"].notna(), "outcome"] = "capitulated"   # AS event takes priority

sp["start_yr"] = pd.PeriodIndex(sp["start_q"], freq="Q").year
sp["era"] = pd.cut(sp["start_yr"], [0, 1994, 2009, 9999],
                   labels=["1980-94", "1995-2009", "2010-23"])

log.append(f"\nspells: {len(sp):,}")
tab = (sp.groupby("era", observed=True)["outcome"]
         .value_counts(normalize=False).unstack(fill_value=0))
shares = tab.div(tab.sum(axis=1), axis=0)
log.append("\noutcome COUNTS by era:")
log.append(tab.to_string())
log.append("\noutcome SHARES by era:")
log.append((shares * 100).round(2).to_string())

log.append("\nthe key comparison (per spell, %):")
for era in tab.index:
    cap = shares.loc[era].get("capitulated", 0) * 100
    die = shares.loc[era].get("died", 0) * 100
    log.append(f"  {era}: capitulated {cap:.2f}%  died {die:.2f}%  "
               f"either {cap + die:.2f}%")

merged_died = sp[(sp["outcome"] == "died")]
if len(merged_died):
    log.append(f"\nof 'died' spells, ended via merger (family reorg possible, "
               f"not always distress): {merged_died['merged'].fillna(False).mean():.1%}")

out = tab.copy()
out.to_csv(P.OUT / "spell_outcomes.csv")

fig, ax = plt.subplots(figsize=(7.5, 4.8))
order = [c for c in ["capitulated", "died", "recovered", "censored"] if c in shares]
bottom = np.zeros(len(shares))
colors = {"capitulated": "#d62728", "died": "#7f7f7f",
          "recovered": "#2ca02c", "censored": "#c7c7c7"}
for c in order:
    ax.bar(shares.index.astype(str), shares[c], bottom=bottom,
           label=c, color=colors.get(c))
    bottom += shares[c].to_numpy()
ax.set_ylabel("Share of underperformance spells")
ax.set_title("How spells end, by era: capitulate, die, recover, or censor")
ax.legend(frameon=False, fontsize=8, ncol=4)
fig.tight_layout()
fig.savefig(P.OUT / "competing_risks.png", dpi=200)

log.append("\nInterpretation guide: if 'died' rises as 'capitulated' falls "
           "across eras, the post-2010 slowdown partly reflects a change in the "
           "FORM of capitulation (death replacing closet indexing). If both "
           "fall, the slowdown is a real increase in resilience/selection.")
log.append("COMPETING RISKS DONE - outputs aggregate-only and shareable.")
P.write_report("competing_risks_report.txt", log)
print("\n".join(log))
