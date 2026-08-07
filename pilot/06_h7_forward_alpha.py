"""Stage 6: H7 FLAGSHIP — does resisting capitulation predict subsequent alpha?

Design (calendar-time portfolios, the standard look-ahead-safe approach):
  CAPITULATORS: funds enter the portfolio the month after their Active Share
    first collapses below 60% during an underperformance spell; held 36 months.
  RESISTERS: funds enter the month after reaching 8 consecutive quarters (2y)
    underwater while still genuinely active (no AS collapse yet); held 36 months.
    (Classified on information available at entry — no peeking at recovery.)
Each month, equal-weight the current members of each portfolio (>=10 members
required); regress excess returns on CAPM / FF3 / FF3+momentum; also the
long-short (resister minus capitulator) spread.

Requires stages 01 and 04 caches. Outputs (aggregates only):
  output/h7_report.txt, h7_cumulative.png, h7_portfolios.csv
Pilot caveats in report: classic OLS SEs (not Newey-West); style-cycle controls
limited to FF3+Mom; min-AS event definition.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

RESIST_QUARTERS = 8      # "long-suffering resister" threshold (2y underwater)
HOLD_MONTHS = 36
MIN_MEMBERS = 10

log = ["H7 FORWARD ALPHA (calendar-time portfolios)", "=" * 60]

panel = PL.build_panel(log)
sp = PL.extract_spells(panel, client_cut=None)
log.append(f"spells: {len(sp):,} across {sp['wficn'].nunique():,} funds")

# ------------------------------------------------- portfolio entry dates ----
def q_end_month(qstr: str, offset_q: int = 0) -> pd.Timestamp:
    return (pd.Period(qstr, freq="Q") + offset_q).to_timestamp("Q")

entries = []  # (portfolio, wficn, entry_month_end)
cap = sp[sp["m_dur"].notna()]
for _, r in cap.iterrows():
    entries.append(("capitulator", r["wficn"], q_end_month(r["start_q"], int(r["m_dur"]))))
res = sp[(sp["end_dur"] >= RESIST_QUARTERS)
         & (sp["m_dur"].isna() | (sp["m_dur"] >= RESIST_QUARTERS))]
for _, r in res.iterrows():
    entries.append(("resister", r["wficn"], q_end_month(r["start_q"], RESIST_QUARTERS)))
ent = pd.DataFrame(entries, columns=["port", "wficn", "entry"])
log.append(f"capitulator entries: {(ent['port'] == 'capitulator').sum():,}; "
           f"resister entries: {(ent['port'] == 'resister').sum():,} "
           f"(resist threshold = {RESIST_QUARTERS}q underwater, hold {HOLD_MONTHS}m)")

# --------------------------------------------- monthly portfolio returns ----
fm = PL.get_fund_monthly(log)
fm["month"] = fm["caldt"].dt.to_period("M")
ent["entry_m"] = ent["entry"].dt.to_period("M")

member_rows = []
for _, r in ent.iterrows():
    for k in range(1, HOLD_MONTHS + 1):
        member_rows.append((r["port"], r["wficn"], r["entry_m"] + k))
mem = pd.DataFrame(member_rows, columns=["port", "wficn", "month"]).drop_duplicates()
mem = mem.merge(fm[["wficn", "month", "fret"]], on=["wficn", "month"], how="inner")
port = (mem.groupby(["port", "month"])
           .agg(pret=("fret", "mean"), n=("fret", "size")).reset_index())
port = port[port["n"] >= MIN_MEMBERS]
wide = port.pivot(index="month", columns="port", values="pret").dropna()
nmem = port.pivot(index="month", columns="port", values="n")
log.append(f"portfolio months with both portfolios populated: {len(wide):,} "
           f"({wide.index.min()} to {wide.index.max()})")
log.append(f"avg members: capitulator {nmem['capitulator'].mean():.0f}, "
           f"resister {nmem['resister'].mean():.0f}")

# ---------------------------------------------------------- regressions ----
fac = PL.get_factors(log)
fac["month"] = fac["month"].dt.to_period("M")
df = wide.reset_index().merge(fac, on="month", how="inner").dropna(
    subset=["mktrf", "smb", "hml", "mom", "rf"])
df["xcap"] = df["capitulator"] - df["rf"]
df["xres"] = df["resister"] - df["rf"]
df["ls"] = df["resister"] - df["capitulator"]

MODELS = {"CAPM": ["mktrf"], "FF3": ["mktrf", "smb", "hml"],
          "FF3+Mom": ["mktrf", "smb", "hml", "mom"]}
log.append(f"\nalphas (monthly, annualized in brackets; {len(df)} months; "
           "classic OLS t-stats - pilot-grade):")
results = []
for name, xs in MODELS.items():
    X = df[xs].to_numpy()
    for label, y in (("resister", df["xres"]), ("capitulator", df["xcap"]),
                     ("resister-minus-capitulator", df["ls"])):
        b, se, t = PL.ols(y.to_numpy(), X)
        results.append((name, label, b[0], t[0]))
        log.append(f"  {name:8s} {label:28s} alpha={b[0]*100:6.3f}%/m "
                   f"[{b[0]*12*100:6.2f}%/yr]  t={t[0]:5.2f}")

# ------------------------------------------------------------- outputs ----
out = wide.copy()
out.index = out.index.astype(str)
out.to_csv(P.OUT / "h7_portfolios.csv")

fig, ax = plt.subplots(figsize=(8.5, 5))
idx = wide.index.to_timestamp()
ax.plot(idx, (1 + wide["resister"]).cumprod(), lw=1.8,
        label="Resisters (held conviction 2y+ underwater)")
ax.plot(idx, (1 + wide["capitulator"]).cumprod(), lw=1.8,
        label="Capitulators (AS collapsed)")
ax.plot(idx, (1 + wide["resister"] - wide["capitulator"]).cumprod(), lw=1.2,
        ls="--", color="0.4", label="Long-short (resister - capitulator)")
ax.set_yscale("log")
ax.set_ylabel("Growth of $1 (log scale)")
ax.set_title("H7: post-classification performance, calendar-time portfolios")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "h7_cumulative.png", dpi=200)

log.append("\nInterpretation guide: H7 predicts a positive resister-minus-"
           "capitulator alpha. Caveats: classic SEs (not Newey-West); factor "
           "controls FF3+Mom only (no CPZ/AQR yet); min-AS event definition; "
           "equal-weighted portfolios; entries are look-ahead-safe (classified "
           "on information available at entry month).")
log.append("H7 DONE - outputs aggregate-only and shareable.")
P.write_report("h7_report.txt", log)
print("\n".join(log))
