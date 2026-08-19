"""Stage 34: BENCHMARK SELF-DECLARATION GAMES (ranked queue R2).

Question (Sensoy 2009, modernized with the SEC's own required
declarations): do funds change their DECLARED benchmark index after bad
runs? FUND_VAR_INFO (in newer N-PORT archives) carries each filing's
DESIGNATED_INDEX_NAME - a per-quarter panel of what the fund itself claims
to be measured against.

 (a) re-extract the FULL declaration history (33b kept only the latest);
 (b) normalize the free-text index names (noisy!) and detect changes;
 (c) condition change probability on trailing relative performance rel4q;
 (d) overlap changes with OUR min-AS benchmark reassignments (bench_min).

All aggregates; report: output/referee_34_declarations.txt
Light script - safe alongside 33d2/36.
"""
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

NP = Path(r"E:\Finance\data\sources\nport")
DRV = NP / "derived"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["BENCHMARK SELF-DECLARATION GAMES (stage 34)", "=" * 60]

# ---- (a) full declaration history ---------------------------------------
frames = []
first_zip = None
for z in sorted(NP.glob("*_nport.zip")):
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
        if "FUND_VAR_INFO.tsv" not in names:
            continue
        first_zip = first_zip or z.name
        v = pd.read_csv(zf.open("FUND_VAR_INFO.tsv"), sep="\t",
                        low_memory=False)
        sub = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t",
                          usecols=["ACCESSION_NUMBER", "REPORT_DATE",
                                   "FILING_DATE"], low_memory=False)
        info = pd.read_csv(zf.open("FUND_REPORTED_INFO.tsv"), sep="\t",
                           usecols=["ACCESSION_NUMBER", "SERIES_ID"],
                           low_memory=False)
        v = v.merge(sub, on="ACCESSION_NUMBER").merge(
            info, on="ACCESSION_NUMBER")
        frames.append(v)
decl = pd.concat(frames, ignore_index=True)
decl = decl[decl["SERIES_ID"].notna()
            & decl["DESIGNATED_INDEX_NAME"].notna()]
for c in ("REPORT_DATE", "FILING_DATE"):
    decl[c] = pd.to_datetime(decl[c], format="%d-%b-%Y", errors="coerce")
    if decl[c].isna().mean() > 0.1:
        decl[c] = pd.to_datetime(decl[c], errors="coerce")
decl["q"] = decl["REPORT_DATE"].dt.to_period("Q")
decl = (decl.sort_values("FILING_DATE")
            .drop_duplicates(["SERIES_ID", "q"], keep="last"))
log.append(f"declarations: {len(decl):,} series-quarters, "
           f"{decl['SERIES_ID'].nunique():,} series; first archive with "
           f"FUND_VAR_INFO: {first_zip}; span {decl['q'].min()} to "
           f"{decl['q'].max()}")

# ---- (b) normalize names, detect changes --------------------------------
STOP = r"\b(TOTAL RETURN|INDEX|THE|NET|GROSS|TR|USD|\(R\))\b"
def norm(s):
    s = str(s).upper()
    s = re.sub(r"[^\w\s&]", " ", s)
    s = re.sub(STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()

decl["bench"] = decl["DESIGNATED_INDEX_NAME"].map(norm)
decl = decl[decl["bench"].ne("") & decl["bench"].ne("0")]
decl = decl.sort_values(["SERIES_ID", "q"])
decl["prev"] = decl.groupby("SERIES_ID")["bench"].shift()
decl["chg"] = decl["prev"].notna() & (decl["bench"] != decl["prev"])
n_chg = int(decl["chg"].sum())
n_obs = int(decl["prev"].notna().sum())
log.append(f"declared-benchmark CHANGES: {n_chg:,} of {n_obs:,} "
           f"series-quarter transitions ({n_chg / n_obs:.2%}); series with "
           f">=1 change: {decl.groupby('SERIES_ID')['chg'].any().sum():,}")
top_pairs = (decl.loc[decl['chg']]
             .groupby(["prev", "bench"]).size().nlargest(10))
log.append("top change pairs (EYEBALL FOR FORMATTING NOISE - a rename of "
           "the same index is not a real switch):")
for (a, b), n in top_pairs.items():
    log.append(f"    {n:4d}  {a[:38]} -> {b[:38]}")

# ---- (c) condition on trailing performance ------------------------------
link = pd.read_csv(DRV / "series_crsp_link_v2.csv", low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
decl = decl.merge(lw, left_on="SERIES_ID", right_on="series_id",
                  how="left")
panel = PL.build_panel(log)
pq = panel.set_index(["wficn", "quarter"])["rel4q"]
decl["quarter"] = decl["q"]
decl["rel4q"] = [
    pq.get((w, q), np.nan) if pd.notna(w) else np.nan
    for w, q in zip(decl["wficn"], decl["quarter"])]
sub = decl[decl["prev"].notna() & decl["rel4q"].notna()]
log.append(f"transitions with wficn+rel4q: {len(sub):,} "
           f"({sub['chg'].sum():,.0f} changes)")
qq = pd.qcut(sub["rel4q"], 5, labels=False, duplicates="drop")
tab = sub.groupby(qq)["chg"].agg(["mean", "count"])
log.append("P(declared-benchmark change) by trailing rel4q quintile "
           "(0=worst performers):")
for i, r in tab.iterrows():
    log.append(f"    Q{int(i)}: {r['mean']:.2%}  (n {int(r['count']):,})")
log.append("  reading: monotone decline from Q0 = losers switch their "
           "declared benchmark - the modern Sensoy result.")

# ---- (d) overlap with min-AS benchmark reassignment ---------------------
bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
bp["quarter"] = pd.to_datetime(bp["month"]).dt.to_period("Q")
bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
        .drop_duplicates(["wficn", "quarter"], keep="last"))
bp = bp.sort_values(["wficn", "quarter"])
bp["b_prev"] = bp.groupby("wficn")["bench_min"].shift()
bp["reassign"] = bp["b_prev"].notna() & (bp["bench_min"] != bp["b_prev"])
re_q = set(map(tuple, bp.loc[bp["reassign"], ["wficn", "quarter"]]
               .itertuples(index=False)))
chg_rows = decl[decl["chg"] & decl["wficn"].notna()]
hit = 0
for w, q in zip(chg_rows["wficn"], chg_rows["quarter"]):
    if any((w, q + k) in re_q for k in (-2, -1, 0, 1, 2)):
        hit += 1
if len(chg_rows):
    log.append(f"declared changes with a min-AS bench reassignment within "
               f"±2q: {hit:,} of {len(chg_rows):,} "
               f"({hit / len(chg_rows):.1%})")
    base = bp["reassign"].mean()
    log.append(f"  (baseline unconditional reassignment rate "
               f"{base:.2%}/quarter - compare ~{1 - (1 - base) ** 5:.1%} "
               f"expected in a random 5-quarter window)")

log.append("\nSTAGE 34 DONE - aggregates only. If (c) shows the gradient, "
           "this graduates from descriptive to a paper section (Paper 2 "
           "candidate spine with the gross-flow test).")
P.write_report("referee_34_declarations.txt", log)
print("\n".join(log))
