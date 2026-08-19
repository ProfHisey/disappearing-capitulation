"""Stage 35c: DEATH-PROFILE VERIFICATION (audit round 5, M-3/M-4/M-5).

Settles whether 35b's merger-drain vs liquidation-zeros contrast is real:
 (a) rebuild death info WITH delist_cd and death MONTH (both discarded
     by the cached death table): per-wficn last end_dt month, modal
     delist_cd prefix, merged flag; crosstab delist prefix vs the old
     merged proxy;
 (b) death-month vs last-flow-month distribution (does end_dt trail
     operations, manufacturing the zero window?);
 (c) all-zero flow triples (sales=reinv=redemptions=0): share of
     dying-fund months vs population months - fill-zeros vs dormancy;
 (d) profile v3: anchored on death MONTH, split by delist_cd prefix,
     with an all-zero-as-missing variant.

Aggregates only; report: output/referee_35c_death_verify.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["DEATH-PROFILE VERIFICATION (stage 35c)", "=" * 60]

# ---- (a) death info with delist_cd + month ------------------------------
parts, use = [], ["crsp_fundno", "end_dt", "dead_flag", "delist_cd",
                  "merge_fundno"]
for ch in pd.read_csv(P.F_SUMMARY,
                      usecols=lambda c: c.strip().lower() in use,
                      chunksize=500_000, low_memory=False,
                      encoding="latin-1"):
    parts.append(P.norm_cols(ch))
d = pd.concat(parts, ignore_index=True)
d["end_dt"] = pd.to_datetime(d["end_dt"], errors="coerce")
d["dead"] = d["dead_flag"].astype(str).str.upper().eq("Y")
d["merged"] = d["merge_fundno"].notna()
d["dcd"] = d["delist_cd"].astype(str).str.upper().str[0]
per = (d.sort_values("end_dt").groupby("crsp_fundno")
        .agg(end_dt=("end_dt", "max"), dead=("dead", "any"),
             merged=("merged", "any"),
             dcd=("dcd", lambda s: s.dropna().iloc[-1]
                  if len(s.dropna()) else "?")).reset_index())
m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                 encoding="latin-1")
m1.columns = [c.lower() for c in m1.columns]
m1 = (m1[["crsp_fundno", "wficn"]].dropna().astype("int64")
      .drop_duplicates("crsp_fundno"))
w = (per.merge(m1, on="crsp_fundno", how="inner").groupby("wficn")
     .agg(end_dt=("end_dt", "max"), n_dead=("dead", "sum"),
          n_cls=("dead", "size"), merged=("merged", "any"),
          dcd=("dcd", lambda s: s.value_counts().index[0]))
     .reset_index())
w["died"] = w["n_dead"] == w["n_cls"]
w = w[w["died"]]
w["death_m"] = w["end_dt"].dt.to_period("M")
log.append(f"(a) dead funds: {len(w):,}. delist_cd prefix vs merged "
           f"proxy crosstab:")
ct = pd.crosstab(w["dcd"], w["merged"])
for idx, row in ct.iterrows():
    log.append(f"    dcd={idx}: merged=False {row.get(False, 0):,} | "
               f"merged=True {row.get(True, 0):,}")
log.append("    reading: L=liquidation, M=merger. Off-diagonal mass = "
           "35b's proxy mislabeled that many funds.")

# ---- flows --------------------------------------------------------------
DRV = SRC / "nport" / "derived"
fl = pd.read_csv(DRV / "monthly_gross_flows.csv", low_memory=False)
link = pd.read_csv(DRV / "series_crsp_link_v2.csv", low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
fl = fl.merge(lw, on="series_id", how="inner")
fl["month"] = pd.PeriodIndex(fl["month"], freq="M")
g = (fl.groupby(["wficn", "month"])
       [["sales", "reinvestments", "redemptions", "net_assets"]]
       .sum(min_count=1).reset_index())
g = g[g["net_assets"] > 0]
g["wficn"] = g["wficn"].astype("int64")
g["allzero"] = ((g["sales"].fillna(0) == 0)
                & (g["reinvestments"].fillna(0) == 0)
                & (g["redemptions"].fillna(0) == 0))
g["rr"] = (g["redemptions"] / g["net_assets"]).clip(-1, 2)

dw = w[(w["death_m"] >= pd.Period("2020-07", freq="M"))
       & (w["death_m"] <= pd.Period("2026-03", freq="M"))]
dw = dw[dw["wficn"].isin(set(g["wficn"]))]
log.append(f"\ndeaths in flows window with flow data: {len(dw):,}")

# ---- (b) death month vs last flow month ---------------------------------
last_flow = g.groupby("wficn")["month"].max()
gap = [(dm - last_flow.get(wf)).n for wf, dm in
       zip(dw["wficn"], dw["death_m"]) if wf in last_flow.index]
gap = pd.Series(gap)
log.append(f"(b) death_m minus last flow month: median "
           f"{gap.median():.0f}m, p25 {gap.quantile(.25):.0f}, p75 "
           f"{gap.quantile(.75):.0f}; share >3m: {(gap > 3).mean():.1%} "
           f"(large positive = end_dt trails operations; the 'zero "
           f"window' may be post-operational)")

# ---- (c) all-zero shares ------------------------------------------------
gz = g.set_index(["wficn", "month"])
dead_months, pop_share = [], g["allzero"].mean()
for wf, dm in zip(dw["wficn"], dw["death_m"]):
    for k in range(-12, 1):
        try:
            dead_months.append(gz.loc[(wf, dm + k), "allzero"])
        except KeyError:
            pass
dead_share = float(np.mean(dead_months)) if dead_months else np.nan
log.append(f"(c) all-zero flow months: dying funds (t-12..t0) "
           f"{dead_share:.1%} vs population {pop_share:.1%} "
           f"(dying >> population = dormancy is real relative to the "
           f"instrument; population also high = fill-zero suspicion "
           f"for the instrument itself)")

# ---- (d) profile v3 -----------------------------------------------------
def profile(sub, lab, drop_allzero):
    prof = {}
    for wf, dm in zip(sub["wficn"], sub["death_m"]):
        for k in range(-12, 1):
            try:
                r = gz.loc[(wf, dm + k)]
            except KeyError:
                continue
            if drop_allzero and r["allzero"]:
                continue
            prof.setdefault(k, []).append(float(r["rr"]))
    line = [f"{lab}:"]
    for k in range(-12, 1, 3):
        vals = prof.get(k, [])
        line.append(f"t{k:+d} {np.median(vals):.2%}(n{len(vals)})"
                    if vals else f"t{k:+d} -")
    log.append("  " + "  ".join(line))

log.append("(d) median rr to DEATH MONTH, by delist_cd prefix:")
for pref in ("L", "M"):
    sub = dw[dw["dcd"] == pref]
    if len(sub) >= 30:
        profile(sub, f"dcd={pref} all months (n {len(sub)})", False)
        profile(sub, f"dcd={pref} excl all-zero  ", True)
log.append(f"population median rr: {g['rr'].median():.2%}")
log.append("\nreading: if dcd=L still shows zeros/dormancy in NON-zero "
           "months too, and (b) shows end_dt trailing operations, the "
           "honest claim is 'liquidated funds go administratively "
           "dormant before formal death' - a different (and still "
           "interesting) sentence than 'clients abandon them'. Quote "
           "nothing from 35b until this run is read.")
log.append("\nSTAGE 35c DONE - aggregates only.")
P.write_report("referee_35c_death_verify.txt", log)
print("\n".join(log))
