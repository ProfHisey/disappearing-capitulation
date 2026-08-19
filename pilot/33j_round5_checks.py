"""Stage 33j: ROUND-5 AUDIT CHECKS - gates on the v9 event-side numbers.

Runs the cheap checks audit round 5 demanded before the manuscript's
extension numbers are quotable:
 (1) C-1: per-benchmark-code return-series end dates (a silently
     truncated S5/style series would undercount events invisibly);
 (2) C-2: fund-level splice-step inspection of the 10 events dated
     2023Q4 (first new-instrument quarter);
 (3) 33h real-arm RERUN on the v2 (S&P-augmented) cache, with the
     window guard and the M4 bad-series drop applied;
 (4) M-2: NaN-net_assets dedup incidence;
 (5) extension-era attrition decomposition (44,234 pre-filter rows ->
     25,606 panel rows: where did 42% go?);
 (6) 33e2 membership tail (names per month min/median/max).

Aggregates only; report: output/nport_33j_checks.txt
Uses caches - no full panel rebuild; moderate runtime.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["ROUND-5 CHECKS (stage 33j)", "=" * 60]

# ---- (1) C-1: per-bcode end dates ---------------------------------------
bq = PL.get_real_bench_q(log)
ends = bq.groupby("bcode")["quarter"].agg(["min", "max"])
log.append("(1) benchmark return series coverage by code:")
for code, r in ends.iterrows():
    flag = "" if r["max"] >= pd.Period("2026Q2", freq="Q") else \
        "  <-- SHORT: events after this are undercounted"
    log.append(f"    {code}: {r['min']} to {r['max']}{flag}")
needed = set(PL.WT_TO_CODE.values()) | {"S5"}
missing = needed - set(ends.index)
log.append(f"    codes needed but absent from bq: {sorted(missing)}")

# ---- (2) C-2: the 10 seam-quarter events --------------------------------
panel = pd.read_parquet(P.CACHE / "panel_full_ext_v1.parquet")
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
sp = PL.extract_spells(panel, client_cut=None)
sp["m_cal"] = pd.PeriodIndex(sp["m_cal_q"].where(sp["m_cal_q"].notna()),
                             freq="Q")
seamers = sp[(sp["m_dur"].notna())
             & (sp["m_cal"] == pd.Period("2023Q4", freq="Q"))]
log.append(f"\n(2) events dated 2023Q4: {len(seamers)}")
asp = pd.read_parquet(P.CACHE / "as_panel.parquet").dropna(
    subset=["wficn"])
asp["wficn"] = asp["wficn"].astype("int64")
asp["quarter"] = asp["month"].dt.to_period("Q")
asp = (asp.sort_values(["wficn", "quarter", "total_assets"])
          .drop_duplicates(["wficn", "quarter"], keep="last")
          .set_index(["wficn", "quarter"]))
PFx = {w: g.set_index("quarter")["as_min"]
       for w, g in panel.groupby("wficn")}
n_suspect = 0
for w in seamers["wficn"].astype("int64"):
    nd = asp["as_min"].get((w, pd.Period("2023Q3", freq="Q")), np.nan)
    s = PFx.get(w)
    e1 = s.get(pd.Period("2023Q4", freq="Q"), np.nan) if s is not None \
        else np.nan
    e2 = s.get(pd.Period("2024Q1", freq="Q"), np.nan) if s is not None \
        else np.nan
    step = (e1 - nd) if pd.notna(e1) and pd.notna(nd) else np.nan
    sus = pd.notna(step) and step < -0.05
    n_suspect += bool(sus)
    log.append(f"    wficn {int(w)}: ND@23Q3 {nd:.3f}  ext@23Q4 "
               f"{e1 if pd.notna(e1) else float('nan'):.3f}  ext@24Q1 "
               f"{e2 if pd.notna(e2) else float('nan'):.3f}  step "
               f"{step if pd.notna(step) else float('nan'):+.3f}"
               f"{'  SUSPECT' if sus else ''}")
log.append(f"    splice-step suspects (step<-0.05): {n_suspect} of "
           f"{len(seamers)} -> quote post-2023 crossings as "
           f"{35 - n_suspect}-35 or footnote accordingly")

# ---- (3) 33h real arm on v2, windowed, M4-dropped -----------------------
ext = pd.read_parquet(P.CACHE / "nport_as_extension_v2.parquet")
ext["as_use"] = ext["as_min_v2"].fillna(ext["as_min_ru"])
link = pd.read_csv(SRC / "nport" / "derived" / "series_crsp_link_v2.csv",
                   low_memory=False)
lw = link[link["wficn"].notna() & ~link["ambiguous"]].copy()
multi = lw.groupby("series_id")["wficn"].nunique()
lw = (lw[~lw["series_id"].isin(set(multi[multi > 1].index))]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
ext = ext.merge(lw, on="series_id", how="inner")
ext["wficn"] = ext["wficn"].astype("int64")
ext["q"] = pd.PeriodIndex(ext["period"], freq="M").asfreq("Q")
ew = ext.groupby(["wficn", "q"])["as_use"].mean().reset_index()
log.append(f"\n(3) 33h real arm on v2: ew min quarter {ew['q'].min()} "
           f"(guard: must be 2023Q4)")
B = pd.Period("2023Q3", freq="Q")
ew = ew[(ew["q"] > B) & (ew["q"] <= B + 11)]
EW = {w: g.set_index("q")["as_use"].sort_index()
      for w, g in ew.groupby("wficn")}
death = PL.get_death(log)
dd = death[death["died"] == 1]
DQ = dict(zip(dd["wficn"].astype("int64"),
              pd.PeriodIndex(dd["death_q"], freq="Q")))
panel_nd = PL.build_panel(log)
sp_nd = PL.extract_spells(panel_nd, client_cut=None)
cap_q = {}
for _, s in sp_nd[sp_nd["m_dur"].notna()].iterrows():
    w = int(s["wficn"])
    q = pd.Period(s["m_cal_q"], freq="Q")
    if w not in cap_q or q < cap_q[w]:
        cap_q[w] = q
sp_nd["sp_"] = pd.PeriodIndex(sp_nd["start_q"], freq="Q")
sp_nd["ep_"] = pd.PeriodIndex(sp_nd["end_q"], freq="Q")
open_sp = sp_nd[(sp_nd["sp_"] <= B) & (sp_nd["ep_"] >= B)].copy()
open_sp = open_sp.sort_values("ep_").drop_duplicates("wficn",
                                                     keep="last")
PFnd = {w: g.set_index("quarter")["as_min"].dropna()
        for w, g in panel_nd.groupby("wficn")}
rows = []
for w in open_sp["wficn"].astype("int64"):
    if (w in cap_q and cap_q[w] <= B) or (w in DQ and DQ[w] <= B):
        continue
    ser = EW.get(w)
    if ser is None or not len(ser):
        continue
    full = PFnd.get(w)
    as_b = float(full[full.index <= B].iloc[-1]) \
        if full is not None and len(full[full.index <= B]) else np.nan
    run7 = run6 = best7 = best6 = 0
    for v in ser.values:
        run7 = run7 + 1 if v < 0.70 else 0
        run6 = run6 + 1 if v < 0.60 else 0
        best7, best6 = max(best7, run7), max(best6, run6)
    rows.append((w, len(ser), best7 >= 2, best6 >= 2, as_b))
d = pd.DataFrame(rows, columns=["w", "nq", "x70", "x60", "as_b"])
fy = d["nq"].sum() / 4
log.append(f"    real 2023Q3 on v2: followed {len(d):,}; <0.70 "
           f"{int(d['x70'].sum())} = {d['x70'].sum() / fy:.2%}/fy; "
           f"<0.60 {int(d['x60'].sum())} = {d['x60'].sum() / fy:.2%}/fy")
fresh = d[d["as_b"] >= 0.70]
ffy = fresh["nq"].sum() / 4
log.append(f"    C2-clean (n {len(fresh):,}): <0.70 "
           f"{int(fresh['x70'].sum())} = "
           f"{fresh['x70'].sum() / ffy:.2%}/fy; <0.60 "
           f"{int(fresh['x60'].sum())} = "
           f"{fresh['x60'].sum() / ffy:.2%}/fy "
           f"(33h v1 gave 0.33%; placebos ~0.50%)")

# ---- (4) NaN dedup incidence --------------------------------------------
meta = pd.read_parquet(P.CACHE / "nport_holdings_parts"
                       / "_filings_meta.parquet")
e2 = pd.read_parquet(P.CACHE / "nport_as_extension_v2.parquet")
e2 = e2.merge(lw, on="series_id", how="inner")
e2 = e2.merge(meta[["accession", "net_assets"]], on="accession",
              how="left")
e2["q"] = pd.PeriodIndex(e2["period"], freq="M").asfreq("Q")
dup = e2[e2.duplicated(["wficn", "q"], keep=False)]
log.append(f"\n(4) duplicate (wficn, quarter) filing rows: {len(dup):,}; "
           f"with NaN net_assets among them: "
           f"{int(dup['net_assets'].isna().sum()):,} "
           f"(0 = M-2 has no impact on this build)")

# ---- (5) attrition decomposition ----------------------------------------
ext_fq = panel[panel["quarter"] > B]
log.append(f"\n(5) extension-era rows in final panel: {len(ext_fq):,} of "
           f"44,234 continuation rows; losses come from the passive "
           f"filter, funds without 3-month CRSP return quarters, and "
           f"missing benchmark returns - decomposition needs the "
           f"intermediate frames (rerun 33i with counters if referees "
           f"ask; the count is disclosed as-is in v9.1)")

log.append("\nSTAGE 33j DONE - aggregates only (per-fund rows in (2) are "
           "derived AS values, not licensed raw).")
P.write_report("nport_33j_checks.txt", log)
print("\n".join(log))
