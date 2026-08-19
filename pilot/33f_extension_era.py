"""Stage 33f: DID ANYONE SURRENDER AFTER THE SAMPLE ENDED? (extension era)

Follows spells that were OPEN (uncapitulated, alive) at the paper's
2023Q3 boundary through the 33e extension (2023Q4-2026Q2):
 (a) coverage: open spells with extension AS data (needs N-PORT link);
 (b) new durable crossings below 0.70 (2q consecutive; 4q variant);
 (c) new deaths among open-spell funds (CRSP death data runs to 2026);
 (d) back-of-envelope extension-era capitulation rate per fund-year vs
     the paper's 2010-23 rate.
APPROXIMATION NOTE: crossing detection here runs on the extension series
alone, not the fully merged panel - the real integration re-runs
extract_spells after the panel merge. This is the fast preview.

Aggregates only; report: output/nport_33f_extension_era.txt
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

log = ["EXTENSION-ERA SURRENDER WATCH (stage 33f)", "=" * 60]

# ---- extension AS series per wficn-quarter ------------------------------
ext = pd.read_parquet(P.CACHE / "nport_as_extension.parquet")
link = pd.read_csv(SRC / "nport" / "derived" / "series_crsp_link_v2.csv",
                   low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
ext = ext.merge(lw, on="series_id", how="inner")
ext["wficn"] = ext["wficn"].astype("int64")
ext["q"] = pd.PeriodIndex(ext["period"], freq="M").asfreq("Q")
# one obs per wficn-quarter (funds with multiple filing series: mean)
ew = (ext.groupby(["wficn", "q"])["as_min_ru"].mean().reset_index())
EW = {w: g.set_index("q")["as_min_ru"] for w, g in ew.groupby("wficn")}
log.append(f"extension series: {len(ew):,} wficn-quarters, "
           f"{len(EW):,} funds")

# ---- open spells at the boundary ----------------------------------------
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
B = pd.Period("2023Q3", freq="Q")
open_sp = sp[(sp["capitulated"] == False) & (sp["died"] == 0)
             & (sp["end_p"] >= B - 2)].copy()   # active near the boundary
log.append(f"spells open/uncapitulated near 2023Q3: {len(open_sp):,}")
open_sp["wficn"] = open_sp["wficn"].astype("int64")
cov = open_sp["wficn"].isin(EW)
log.append(f"  with extension AS coverage: {int(cov.sum()):,} "
           f"({cov.mean():.1%}) - uncovered funds are unlinked or "
           f"dropped by the id_share filter; note selection.")

# ---- (b) new crossings --------------------------------------------------
n2 = n4 = 0
fq_obs = 0
for w in open_sp.loc[cov, "wficn"]:
    s = EW[w].sort_index()
    fq_obs += len(s)
    run = best = 0
    for q, v in s.items():
        run = run + 1 if v < 0.70 else 0
        best = max(best, run)
    if best >= 2:
        n2 += 1
    if best >= 4:
        n4 += 1
log.append(f"new durable crossings 2023Q4-2026Q2: {n2:,} funds at 2q "
           f"({n4:,} at 4q) out of {int(cov.sum()):,} followed; "
           f"fund-quarters observed {fq_obs:,}")
rate = n2 / (fq_obs / 4) if fq_obs else np.nan
log.append(f"  extension-era crossing rate: {rate:.2%} per fund-YEAR")
log.append("  compare: the paper's 2010-23 era produced 175 events - "
           "compute the comparable per-fund-year rate from stage 14 "
           "outputs when integrating; this is the preview number.")

# ---- (c) new deaths -----------------------------------------------------
try:
    dw = death.copy()
    dw["dq"] = pd.to_datetime(dw["death_dt"]).dt.to_period("Q")
    dd = dw[dw["wficn"].astype("int64").isin(
        set(open_sp["wficn"])) & (dw["dq"] > B)]
    log.append(f"deaths among boundary-open funds after 2023Q3: "
               f"{len(dd):,}")
except Exception as e:
    log.append(f"death-table format differs ({e}); count deaths at "
               "integration instead.")

log.append("\nreading: if crossings stayed rare while deaths continued, "
           "'nobody surrenders anymore' extends through mid-2026 and the "
           "final build gains three more years of the modern regime. A "
           "SURGE in crossings would be a finding of its own (and would "
           "demand the S&P-benchmark completion first).")
log.append("\nSTAGE 33f DONE - aggregates only.")
P.write_report("nport_33f_extension_era.txt", log)
print("\n".join(log))
