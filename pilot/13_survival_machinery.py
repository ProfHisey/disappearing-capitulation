"""Stage 13: REAL SURVIVAL MACHINERY v1 — Aalen-Johansen + covariate Cox.

Upgrades the pilot's Kaplan-Meier curves to the estimators the paper needs:

1. AALEN-JOHANSEN cumulative incidence: capitulation and death treated as
   competing events (KM overstates each risk when the other exists). Reported
   overall and by era at 8/16/24/40-quarter horizons. This is the correct
   version of the stage-07 outcome shares (fixes the panel-edge censoring
   problem mechanically).
2. CAUSE-SPECIFIC COX for capitulation, spell-level covariates measured AT
   ENTRY (no look-ahead): Active Share level, trailing rel. return, retail
   flow, log TNA, expense ratio, manager tenure, era dummies. Death and
   recovery are censoring (standard cause-specific formulation).
3. Proportional-hazards test on the fitted model.

Outputs (aggregates only): output/survival_report.txt, cif_by_era.png,
cox_table.csv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test

import pilot_lib as P
import panel_lib as PL

log = ["SURVIVAL MACHINERY v1 (Aalen-Johansen + Cox)", "=" * 60]

panel = PL.build_panel(log)
sp = PL.extract_spells(panel, client_cut=-0.10)
death = PL.get_death(log)
sp = sp.merge(death, on="wficn", how="left")
sp["start_p"] = pd.PeriodIndex(sp["start_q"], freq="Q")
sp["end_p"] = pd.PeriodIndex(sp["end_q"], freq="Q")
death_p = pd.PeriodIndex(
    sp["death_q"].where(sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
gap = (death_p - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))

# multi-state event coding: 0 censored (incl. recovered), 1 capitulated, 2 died
sp["etype"] = 0
sp.loc[sp["ended_by"].isin(["data_end", "as_missing"])
       & sp["died"].fillna(False) & gap.between(-1, 4), "etype"] = 2
sp.loc[sp["m_dur"].notna(), "etype"] = 1
sp["dur"] = np.where(sp["etype"] == 1, sp["m_dur"], sp["end_dur"])
sp["dur"] = sp["dur"].clip(lower=1)
sp["start_yr"] = sp["start_p"].dt.year
sp["era"] = pd.cut(sp["start_yr"], [0, 1994, 2009, 9999],
                   labels=["1980-94", "1995-2009", "2010-23"])
log.append(f"spells {len(sp):,}: capitulated {(sp['etype'] == 1).sum():,}, "
           f"died {(sp['etype'] == 2).sum():,}, censored {(sp['etype'] == 0).sum():,}")

# ------------------------------------------- Aalen-Johansen CIFs ----------
HORIZONS = [8, 16, 24, 40]

def cif_at(df, event, horizons):
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(df["dur"], df["etype"], event_of_interest=event)
    c = aj.cumulative_density_
    out = []
    for h in horizons:
        idx = c.index[c.index <= h]
        out.append(float(c.loc[idx[-1]].iloc[0]) if len(idx) else np.nan)
    return out, aj

log.append("\nAalen-Johansen cumulative incidence (competing risks correct):")
log.append(f"{'group':14s} {'event':12s} " + " ".join(f"{h:>3d}q" for h in HORIZONS))
rows = {}
for label, df in [("overall", sp)] + [(str(e), g) for e, g in
                                      sp.groupby("era", observed=True)]:
    for ev, evname in ((1, "capitulate"), (2, "die")):
        vals, aj = cif_at(df, ev, HORIZONS)
        rows[(label, evname)] = (vals, aj if label != "overall" else aj)
        log.append(f"{label:14s} {evname:12s} "
                   + " ".join(f"{v:.3f}" if pd.notna(v) else "  na" for v in vals))

fig, ax = plt.subplots(figsize=(7.5, 5))
for era, g in sp.groupby("era", observed=True):
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(g["dur"], g["etype"], event_of_interest=1)
    aj.cumulative_density_.plot(ax=ax, label=f"capitulate, {era} (n={len(g):,})", lw=1.6)
aj_d = AalenJohansenFitter(calculate_variance=False)
aj_d.fit(sp["dur"], sp["etype"], event_of_interest=2)
aj_d.cumulative_density_.plot(ax=ax, label="die (overall)", lw=1.4, ls="--", color="0.4")
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Cumulative incidence")
ax.set_title("Competing risks: capitulation by era vs death (Aalen-Johansen)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "cif_by_era.png", dpi=200)

# --------------------------------------------- covariates at spell entry ----
cov_pq = P.CACHE / "covars.parquet"
if cov_pq.exists():
    cov = pd.read_parquet(cov_pq)
else:
    m1 = PL.get_mflink1()
    parts, use = [], ["crsp_fundno", "caldt", "exp_ratio", "mgr_dt"]
    for chunk in pd.read_csv(P.F_SUMMARY, usecols=lambda c: c.strip().lower() in use,
                             chunksize=500_000, low_memory=False, encoding="latin-1"):
        c = P.norm_cols(chunk)
        c["caldt"] = pd.to_datetime(c["caldt"], errors="coerce")
        c["mgr_dt"] = pd.to_datetime(c["mgr_dt"], errors="coerce")
        c["exp_ratio"] = pd.to_numeric(c["exp_ratio"], errors="coerce")
        parts.append(c.dropna(subset=["caldt"]))
    cov = pd.concat(parts, ignore_index=True).merge(m1, on="crsp_fundno", how="inner")
    cov["quarter"] = cov["caldt"].dt.to_period("Q").astype(str)
    cov = (cov.sort_values("caldt").groupby(["wficn", "quarter"])
              .agg(exp_ratio=("exp_ratio", "median"), mgr_dt=("mgr_dt", "max"))
              .reset_index())
    cov.to_parquet(cov_pq, index=False)
cov["quarter"] = pd.PeriodIndex(cov["quarter"], freq="Q")

fm = PL.get_fund_monthly([])
fm["quarter"] = fm["caldt"].dt.to_period("Q")
tnaq = fm.groupby(["wficn", "quarter"])["tna"].last().reset_index()

pf = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}
def at_entry(row, col):
    g = pf.get(row["wficn"])
    if g is None or row["start_p"] not in g.index:
        return np.nan
    return g.at[row["start_p"], col]

sp["as0"] = sp.apply(lambda r: at_entry(r, "as_min"), axis=1)
sp["rel0"] = sp.apply(lambda r: at_entry(r, "rel4q"), axis=1)
sp["flow0"] = sp.apply(lambda r: at_entry(r, "flowq"), axis=1)
sp = sp.merge(tnaq.rename(columns={"quarter": "start_p", "tna": "tna0"}),
              on=["wficn", "start_p"], how="left")
sp = sp.merge(cov.rename(columns={"quarter": "start_p"}),
              on=["wficn", "start_p"], how="left")
sp["ln_tna"] = np.log(sp["tna0"].where(sp["tna0"] > 0))
sp["tenure_yrs"] = (sp["start_p"].dt.to_timestamp() - sp["mgr_dt"]).dt.days / 365.25
sp["tenure_yrs"] = sp["tenure_yrs"].where(sp["tenure_yrs"].between(0, 60))
sp["era_9509"] = (sp["era"] == "1995-2009").astype(float)
sp["era_1023"] = (sp["era"] == "2010-23").astype(float)

COVARS = ["as0", "rel0", "flow0", "ln_tna", "exp_ratio", "tenure_yrs",
          "era_9509", "era_1023"]
cx = sp[["dur", "etype"] + COVARS].copy()
cx["event"] = (cx["etype"] == 1).astype(int)   # cause-specific: death censored
cx = cx.drop(columns="etype").dropna()
log.append(f"\nCox sample: {len(cx):,} spells with full covariates, "
           f"{int(cx['event'].sum()):,} capitulation events")

cph = CoxPHFitter()
cph.fit(cx, duration_col="dur", event_col="event")
summ = cph.summary[["coef", "exp(coef)", "se(coef)", "p"]]
summ.to_csv(P.OUT / "cox_table.csv")
log.append("\nCAUSE-SPECIFIC COX, capitulation hazard (HR = exp(coef)):")
for name, r in summ.iterrows():
    log.append(f"  {name:12s} HR {r['exp(coef)']:.3f}  coef {r['coef']:+.3f} "
               f"(se {r['se(coef)']:.3f})  p={r['p']:.4f}")
log.append(f"  concordance: {cph.concordance_index_:.3f}")

try:
    ph = proportional_hazard_test(cph, cx, time_transform="rank")
    worst = ph.summary["p"].min()
    log.append(f"\nproportional-hazards test: min p across covariates = {worst:.4f}"
               + ("  (PH violation flagged - stratify or add time-interactions "
                  "in real build)" if worst < 0.05 else "  (no violation flagged)"))
except Exception as e:  # noqa: BLE001
    log.append(f"PH test failed: {e}")

log.append("\nNotes: covariates measured at spell entry (no look-ahead); "
           "death/recovery censored (cause-specific formulation); tenure from "
           "CRSP mgr_dt (noisy - Morningstar upgrade later); flow0 limits the "
           "sample to flow-covered spells.")
log.append("SURVIVAL MACHINERY DONE - aggregates only.")
P.write_report("survival_report.txt", log)
print("\n".join(log))
