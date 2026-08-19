"""Stage 33h: THE DECISIVE F10 CHECK - placebo boundaries + honest recounts.

Audit round 4 (C1/C2/M3) wounded F10: the extension crossing rate (<0.70,
unconditional, 2q-observed) was compared to the paper's event rate (<0.60,
in-spell) - apples to oranges - and pre-existing below-0.70 stock at the
boundary was never excluded. This stage settles it:

 (A) PLACEBO: run the EXACT 33f cohort-and-crossing algorithm on the ND
     panel itself at boundaries 2005Q3 / 2010Q3 / 2015Q3 / 2019Q3,
     conditioning only on information available at each boundary.
     ~3%/fund-year at a known-quiet placebo = F10 was definitional
     artifact; ~0.7% = F10 resurrects as a genuine regime signal.
 (B) RECOUNTS on the real 2023Q3 boundary: fund-level dedup (M3 check),
     crossings at <0.60 (the paper's event threshold), and the C2
     decomposition - crossings excluding funds already below 0.70 at
     the boundary (stock vs flow).

Aggregates only; report: output/nport_33h_placebo.txt
Builds the panel - run alone.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["F10 DECISIVE CHECK: PLACEBOS + RECOUNTS (stage 33h)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
sp["wficn"] = sp["wficn"].astype("int64")
PF = {w: g.set_index("quarter")["as_min"].dropna().sort_index()
      for w, g in panel.groupby("wficn")}

# death quarters by wficn (info-at-boundary conditioning)
dq_col = next((c for c in ("death_q", "dq", "end_q")
               if hasattr(death, "columns") and c in death.columns), None)
DQ = {}
if dq_col:
    dd = death[death.get("died", 1) == 1] if "died" in death.columns \
        else death
    for w, q in zip(dd["wficn"].astype("int64"),
                    pd.PeriodIndex(dd[dq_col], freq="Q")):
        DQ[w] = q
log.append(f"death quarters available: {len(DQ):,} "
           f"(col={dq_col})")

cap_q = {}
for _, s in sp[sp["capitulated"] == True].iterrows():
    w = int(s["wficn"])
    q = pd.Period(s["m_cal_q"], freq="Q")
    if w not in cap_q or q < cap_q[w]:
        cap_q[w] = q

def run_boundary(B, series_getter, label, horizon=11):
    """Exact 33f logic, info-at-B conditioning, fund-level dedup."""
    open_sp = sp[(sp["start_p"] <= B) & (sp["end_p"] >= B)].copy()
    open_sp = open_sp.sort_values("end_p").drop_duplicates("wficn",
                                                           keep="last")
    keep = []
    for w in open_sp["wficn"]:
        w = int(w)
        if w in cap_q and cap_q[w] <= B:
            continue
        if w in DQ and DQ[w] <= B:
            continue
        keep.append(w)
    rows = []
    for w in keep:
        ser = series_getter(w, B, horizon)
        if ser is None or not len(ser):
            continue
        as_at_b = np.nan
        full = PF.get(w)
        if full is not None:
            upto = full[full.index <= B]
            if len(upto):
                as_at_b = float(upto.iloc[-1])
        run7 = run6 = best7 = best6 = 0
        for v in ser.values:
            run7 = run7 + 1 if v < 0.70 else 0
            run6 = run6 + 1 if v < 0.60 else 0
            best7, best6 = max(best7, run7), max(best6, run6)
        rows.append((w, len(ser), best7 >= 2, best6 >= 2, as_at_b))
    d = pd.DataFrame(rows, columns=["wficn", "nq", "x70", "x60", "as_b"])
    if not len(d):
        log.append(f"  {label}: no funds followed")
        return
    fy = d["nq"].sum() / 4
    log.append(f"  {label}: followed {len(d):,} funds, "
               f"{d['nq'].sum():,} fund-quarters")
    log.append(f"    <0.70 2q rule: {int(d['x70'].sum()):,} crossers = "
               f"{d['x70'].sum() / fy:.2%}/fund-year")
    log.append(f"    <0.60 2q rule: {int(d['x60'].sum()):,} crossers = "
               f"{d['x60'].sum() / fy:.2%}/fund-year")
    fresh = d[d["as_b"] >= 0.70]
    if len(fresh):
        ffy = fresh["nq"].sum() / 4
        log.append(f"    C2-clean (AS at boundary >=0.70, n "
                   f"{len(fresh):,}): <0.70 "
                   f"{int(fresh['x70'].sum()):,} = "
                   f"{fresh['x70'].sum() / ffy:.2%}/fund-yr; <0.60 "
                   f"{int(fresh['x60'].sum()):,} = "
                   f"{fresh['x60'].sum() / ffy:.2%}/fund-yr")
    below = d[d["as_b"] < 0.70]
    log.append(f"    already below 0.70 at boundary: {len(below):,} "
               f"funds ({len(below) / len(d):.1%}) - the C2 'stock'")

# ---- (A) placebos on the panel itself -----------------------------------
def panel_series(w, B, horizon):
    s = PF.get(w)
    if s is None:
        return None
    return s[(s.index > B) & (s.index <= B + horizon)]

log.append("\n(A) PLACEBO BOUNDARIES (exact 33f rule on the ND panel):")
for y in ("2005Q3", "2010Q3", "2015Q3", "2019Q3"):
    run_boundary(pd.Period(y, freq="Q"), panel_series, f"placebo {y}")

# ---- (B) real boundary, extension series, recounts ----------------------
ext = pd.read_parquet(P.CACHE / "nport_as_extension.parquet")
link = pd.read_csv(SRC / "nport" / "derived" / "series_crsp_link_v2.csv",
                   low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
ext = ext.merge(lw, on="series_id", how="inner")
ext["wficn"] = ext["wficn"].astype("int64")
ext["q"] = pd.PeriodIndex(ext["period"], freq="M").asfreq("Q")
ew = ext.groupby(["wficn", "q"])["as_min_ru"].mean().reset_index()
EW = {w: g.set_index("q")["as_min_ru"].sort_index()
      for w, g in ew.groupby("wficn")}

def ext_series(w, B, horizon):
    return EW.get(w)

log.append("\n(B) REAL BOUNDARY 2023Q3 (extension series, fund-deduped, "
           "info-at-B conditioning):")
run_boundary(pd.Period("2023Q3", freq="Q"), ext_series, "real 2023Q3")
# M3 delta for the record
open_raw = sp[(sp["capitulated"] == False) & (sp["died"] == 0)
              & (sp["end_p"] >= pd.Period("2023Q1", freq="Q"))]
log.append(f"  M3 note: 33f's spell-level cohort had "
           f"{len(open_raw):,} rows vs {open_raw['wficn'].nunique():,} "
           f"unique funds (delta = duplicate exposure in the original "
           f"3.12% figure)")

log.append("\nVERDICT KEY: compare the real-boundary C2-clean <0.60 rate "
           "against the SAME statistic at the placebos. That pair is the "
           "honest apples-to-apples F10 test. Placebo ~ real -> artifact; "
           "real >> placebo -> surrender genuinely returned.")
log.append("\nSTAGE 33h DONE - aggregates only.")
P.write_report("nport_33h_placebo.txt", log)
print("\n".join(log))
