"""Stage 35d v2: GROSS-FLOW LEVEL RECONCILIATION (audit round 5 / M7).

v2 fixes the CRSP-TNA units bug (mtna is in $MILLIONS; N-PORT flows in
dollars - v1 divided dollars by millions and every rr_tna hit the
winsorization ceiling) and adds the decisive instrument-vs-instrument
test: per-fund-month SEC net flows vs CRSP-IMPUTED net flows
(tna - tna_lag*(1+ret)). Form semantics (SEC readme, checked):
B.6.a sales, B.6.b reinvestments (separate), B.6.c redemptions
INCLUDING EXCHANGES - gross fields carry intra-family exchange traffic.

Stage 35's headline (stressed and unstressed funds redeem identically;
the flow response is on the purchase margin) rests on flow LEVELS that
looked off: median net flow +0.64/+0.92%/month in a documented outflow
era. Three suspects: passive funds/ETFs in the matched sample, the SEC
filing's stale quarter-end net_assets denominator, and link errors.
This stage:
 (a) splits everything ACTIVE vs PASSIVE (the panel's own flags);
 (b) aggregates matched-sample net flows by calendar year, active-only,
     for eyeball reconciliation against published industry flows;
 (c) re-denominates redemption rates with CRSP lagged monthly TNA and
     reruns the stressed/unstressed comparison on both denominators;
 (d) the verdict line: does the ghosting result survive active-only +
     mtna denominators?

Aggregates only; report: output/referee_35d_reconciliation.txt
Uses caches (no full panel build) - safe beside 37d.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

SRC = Path(r"E:\Finance\data\sources")
DRV = SRC / "nport" / "derived"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["GROSS-FLOW RECONCILIATION (stage 35d)", "=" * 60]

# ---- flows at wficn-month ----------------------------------------------
fl = pd.read_csv(DRV / "monthly_gross_flows.csv", low_memory=False)
link = pd.read_csv(DRV / "series_crsp_link_v2.csv", low_memory=False)
lw = link[link["wficn"].notna() & ~link["ambiguous"]].copy()
multi = lw.groupby("series_id")["wficn"].nunique()
lw = (lw[~lw["series_id"].isin(set(multi[multi > 1].index))]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
fl = fl.merge(lw, on="series_id", how="inner")
fl["month"] = pd.PeriodIndex(fl["month"], freq="M")
g = (fl.groupby(["wficn", "month"])
       [["sales", "reinvestments", "redemptions", "net_assets"]]
       .sum(min_count=1).reset_index())
g = g[g["net_assets"] > 0]
g["wficn"] = g["wficn"].astype("int64")
g["netd"] = (g["sales"].fillna(0) + g["reinvestments"].fillna(0)
             - g["redemptions"].fillna(0))

# ---- (a) active vs passive ----------------------------------------------
flg = pd.read_parquet(P.CACHE / "flags.parquet")
flg["wficn"] = flg["wficn"].astype("int64")
g = g.merge(flg[["wficn", "passive"]], on="wficn", how="left")
g["active"] = g["passive"] != True  # noqa: E712
log.append(f"fund-months: {len(g):,}; flagged passive: "
           f"{(~g['active']).sum():,} ({(~g['active']).mean():.1%}); "
           f"unflagged treated active")

# ---- (b) annual aggregate net flow, active-only -------------------------
g["year"] = g["month"].dt.year
log.append("\n(b) aggregate net flow rate by year (sum dollars / mean "
           "aggregate NA), matched sample:")
for lab, m in (("active ", g["active"]), ("passive", ~g["active"])):
    d = g[m]
    for y, gy in d.groupby("year"):
        if y < 2020 or y > 2025:
            continue
        na_m = gy.groupby("month")["net_assets"].sum().mean()
        rate = gy["netd"].sum() / na_m
        log.append(f"    {lab} {y}: {rate:+.1%} of assets "
                   f"(funds {gy['wficn'].nunique():,})")
log.append("  eyeball vs published industry flows: US active equity ran "
           "persistent NET OUTFLOWS 2020-2025 (ICI/Morningstar). "
           "Active rows near zero or negative = instrument plausible; "
           "strongly positive = sales includes exchanges/conversions or "
           "link errors -> flow LEVELS unusable, shares/timing only.")

# ---- (c) mtna denominators ---------------------------------------------
fm = PL.get_fund_monthly(log)
fm["month"] = fm["caldt"].dt.to_period("M")
fm = fm.sort_values(["wficn", "month"])
fm["tna_lag"] = fm.groupby("wficn")["tna"].shift(1)
g = g.merge(fm[["wficn", "month", "tna_lag", "fret", "tna"]],
            on=["wficn", "month"], how="left")
g["tna_lag_d"] = g["tna_lag"] * 1e6          # $M -> $ (v2 units fix)
g["rr_sec"] = (g["redemptions"] / g["net_assets"]).clip(-1, 2)
g["rr_tna"] = (g["redemptions"] / g["tna_lag_d"]).where(
    g["tna_lag_d"] > 0).clip(-1, 2)
# CRSP-imputed net flow, dollars
g["imp_net_d"] = (g["tna"] - g["tna_lag"] * (1 + g["fret"])) * 1e6
g["nr_sec"] = (g["netd"] / g["net_assets"]).clip(-1, 2)
g["nr_imp"] = (g["imp_net_d"] / g["tna_lag_d"]).where(
    g["tna_lag_d"] > 0).clip(-1, 2)
log.append(f"\n(c) mtna coverage: {g['rr_tna'].notna().mean():.1%} of "
           f"fund-months")

panel = pd.read_parquet(P.CACHE / "panel_full_v3.parquet")
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
panel["wficn"] = panel["wficn"].astype("int64")
pq = (panel[["wficn", "quarter", "rel4q"]]
      .drop_duplicates(["wficn", "quarter"]))
g["quarter"] = pd.PeriodIndex(g["month"], freq="M").asfreq("Q")
g = g.merge(pq, on=["wficn", "quarter"], how="left")

ok = g[g["active"] & g["nr_imp"].notna()]
corr = ok["nr_sec"].corr(ok["nr_imp"])
dsec = ok["nr_sec"].median()
dimp = ok["nr_imp"].median()
log.append(f"\n(c2) INSTRUMENT vs INSTRUMENT, active fund-months "
           f"(n {len(ok):,}): monthly net flow rate median "
           f"SEC {dsec:+.2%} vs CRSP-imputed {dimp:+.2%}; "
           f"corr {corr:.3f}; median (SEC minus imputed) "
           f"{(ok['nr_sec'] - ok['nr_imp']).median():+.2%}")
log.append("  reading: CRSP-imputed is the paper's established "
           "instrument. A big positive SEC-minus-imputed median = "
           "the SEC sales field overstates net flows (exchanges/"
           "conversion traffic); levels then defer to CRSP and the "
           "SEC panel is used for gross STRUCTURE only.")
log.append("\n(d) stressed vs unstressed medians, ACTIVE funds only:")
s = g[g["active"] & g["rel4q"].notna()]
for lab, m in (("stressed  ", s["rel4q"] < 0),
               ("unstressed", s["rel4q"] >= 0)):
    d = s[m]
    nf_sec = (d["netd"] / d["net_assets"]).clip(-1, 2)
    log.append(f"    {lab}: rr(SEC-NA) {d['rr_sec'].median():.2%} | "
               f"rr(CRSP-TNA) {d['rr_tna'].median():.2%} | "
               f"net SEC {nf_sec.median():+.2%} | net imputed "
               f"{d['nr_imp'].median():+.2%}  (n {len(d):,})")
log.append("  VERDICT LINE: ghosting survives if the stressed and "
           "unstressed rr medians remain within ~0.1pp of each other on "
           "BOTH denominators in the active-only sample, while the net "
           "flow gap persists. If stressed rr now exceeds unstressed on "
           "the CRSP-TNA denominator, the original equality was the "
           "stale-denominator artifact the audit predicted, and the "
           "purchase-margin claim needs restating.")
log.append("\nSTAGE 35d DONE - aggregates only.")
P.write_report("referee_35d_reconciliation.txt", log)
print("\n".join(log))
