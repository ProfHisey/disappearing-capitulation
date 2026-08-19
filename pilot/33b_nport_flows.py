"""Stage 33b: EXTRACT MONTHLY GROSS FLOWS FROM N-PORT ARCHIVES.

Streams the small series-level tables (SUBMISSION, FUND_REPORTED_INFO,
REGISTRANT, FUND_VAR_INFO where present) out of all 27 quarterly ZIPs -
never touching the huge holdings tables - and builds:

  derived\\monthly_gross_flows.csv : one row per SEC series per calendar
      month with gross SALES / REINVESTMENTS / REDEMPTIONS (Item B.6) plus
      period-end total/net assets. The modern-era "who breaks first" panel.
  derived\\designated_index.csv    : each series' self-declared benchmark
      (DESIGNATED_INDEX_NAME, newer filings only).

Amendment handling: filings are deduplicated per (SERIES_ID, report period),
keeping the latest FILING_DATE (amendments NPORT-P/A supersede originals).
Month mapping (CORRECTED after the v1 diagnostic fired): REPORT_DATE is the
fiscal QUARTER end this filing covers - MON1/2/3 are the three calendar
months ending there. REPORT_ENDING_PERIOD is the fund's fiscal YEAR end
(v1 wrongly used it as the quarter end; the 0/-3/-6/-9 month four-way split
in the diagnostic is the quarters-within-fiscal-year fingerprint that
caught it). The diagnostic stays in as a standing check.

Output: output/nport_33b_flows.txt (aggregates only; derived files stay in
the git-ignored library).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(r"E:\Finance\data\sources\nport")
DRV = SRC / "derived"
DRV.mkdir(exist_ok=True)
OUT = Path("output")
OUT.mkdir(exist_ok=True)

INFO_COLS = ["ACCESSION_NUMBER", "SERIES_NAME", "SERIES_ID", "SERIES_LEI",
             "TOTAL_ASSETS", "NET_ASSETS",
             "SALES_FLOW_MON1", "REINVESTMENT_FLOW_MON1",
             "REDEMPTION_FLOW_MON1",
             "SALES_FLOW_MON2", "REINVESTMENT_FLOW_MON2",
             "REDEMPTION_FLOW_MON2",
             "SALES_FLOW_MON3", "REINVESTMENT_FLOW_MON3",
             "REDEMPTION_FLOW_MON3"]

log = ["N-PORT MONTHLY GROSS FLOWS (stage 33b)", "=" * 60]
frames, var_frames = [], []

zips = sorted(SRC.glob("*_nport.zip"))
log.append(f"archives found: {len(zips)}")

for zpath in zips:
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        sub = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t",
                          low_memory=False)
        info = pd.read_csv(zf.open("FUND_REPORTED_INFO.tsv"), sep="\t",
                           usecols=lambda c: c in INFO_COLS,
                           low_memory=False)
        reg = pd.read_csv(zf.open("REGISTRANT.tsv"), sep="\t",
                          usecols=["ACCESSION_NUMBER", "CIK",
                                   "REGISTRANT_NAME"], low_memory=False)
        df = (info.merge(sub, on="ACCESSION_NUMBER", how="left")
                  .merge(reg, on="ACCESSION_NUMBER", how="left"))
        df["src_zip"] = zpath.name
        frames.append(df)
        if "FUND_VAR_INFO.tsv" in names:
            v = pd.read_csv(zf.open("FUND_VAR_INFO.tsv"), sep="\t",
                            low_memory=False)
            v = v.merge(sub[["ACCESSION_NUMBER", "FILING_DATE"]],
                        on="ACCESSION_NUMBER", how="left")
            v = v.merge(info[["ACCESSION_NUMBER", "SERIES_ID"]],
                        on="ACCESSION_NUMBER", how="left")
            var_frames.append(v)
    print(f"read {zpath.name}")

raw = pd.concat(frames, ignore_index=True)
log.append(f"filings read: {len(raw):,}")

# ---- dates + diagnostic on the two period fields ------------------------
def parse_dates(s):
    """Try known fixed formats first (fast, warning-free), else dateutil."""
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%m/%d/%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        if d.notna().mean() > 0.9:
            return d
    return pd.to_datetime(s, errors="coerce")

for c in ("FILING_DATE", "REPORT_DATE", "REPORT_ENDING_PERIOD"):
    raw[c] = parse_dates(raw[c])
rd = raw["REPORT_DATE"].dt.to_period("M")
rep = raw["REPORT_ENDING_PERIOD"].dt.to_period("M")
diff = (rd.astype("int64") - rep.astype("int64"))
log.append("REPORT_DATE minus REPORT_ENDING_PERIOD, in months "
           f"(value: share): "
           f"{ {int(k): f'{v:.1%}' for k, v in diff.value_counts(normalize=True).head(6).items()} }")
log.append("  -> expected pattern: ~equal shares at 0/-3/-6/-9 (quarters "
           "within a fiscal year). REPORT_DATE = quarter end (used); "
           "REPORT_ENDING_PERIOD = fiscal year end (NOT used). Any other "
           "pattern: STOP and investigate.")

# ---- drop rows with no series id; dedup amendments ----------------------
n0 = len(raw)
raw = raw[raw["SERIES_ID"].notna() & (raw["SERIES_ID"].astype(str) != "")]
log.append(f"rows dropped for missing SERIES_ID: {n0 - len(raw):,}")
raw["_period"] = raw["REPORT_DATE"].dt.to_period("M")  # quarter end (v2 fix)
raw = (raw.sort_values("FILING_DATE")
          .drop_duplicates(["SERIES_ID", "_period"], keep="last"))
log.append(f"filings after amendment dedup (latest FILING_DATE per "
           f"series-period): {len(raw):,}")
log.append(f"unique series: {raw['SERIES_ID'].nunique():,}   "
           f"unique registrants (CIK): {raw['CIK'].nunique():,}")

# ---- explode to one row per series-month --------------------------------
rows = []
for k in (1, 2, 3):
    part = raw[["SERIES_ID", "SERIES_NAME", "SERIES_LEI", "CIK",
                "REGISTRANT_NAME", "_period", "SUB_TYPE", "src_zip",
                "TOTAL_ASSETS", "NET_ASSETS",
                f"SALES_FLOW_MON{k}", f"REINVESTMENT_FLOW_MON{k}",
                f"REDEMPTION_FLOW_MON{k}"]].copy()
    part.columns = (["series_id", "series_name", "series_lei", "cik",
                     "registrant", "period_end", "sub_type", "src_zip",
                     "total_assets", "net_assets",
                     "sales", "reinvestments", "redemptions"])
    part["month"] = part["period_end"] - (3 - k)
    rows.append(part)
panel = pd.concat(rows, ignore_index=True)
panel = panel.sort_values(["series_id", "month"]).reset_index(drop=True)

dup = panel.duplicated(["series_id", "month"]).sum()
log.append(f"panel rows: {len(panel):,}   duplicate series-months after "
           f"dedup: {dup:,}")
if dup:
    panel = panel.drop_duplicates(["series_id", "month"], keep="last")
    log.append(f"  kept last -> {len(panel):,} rows")

span = panel["month"].agg(["min", "max"])
log.append(f"month span: {span['min']} to {span['max']}")
for c in ("sales", "reinvestments", "redemptions"):
    nz = panel[c].notna().mean()
    log.append(f"  {c}: non-missing {nz:.1%}, median of positives "
               f"${panel.loc[panel[c] > 0, c].median() / 1e6:,.1f}M")

out_csv = DRV / "monthly_gross_flows.csv"
panel.drop(columns=["period_end"]).to_csv(out_csv, index=False)
log.append(f"written: {out_csv}  ({out_csv.stat().st_size / 1e6:.0f} MB)")

# ---- designated index (newer filings) -----------------------------------
if var_frames:
    var = pd.concat(var_frames, ignore_index=True)
    var["FILING_DATE"] = parse_dates(var["FILING_DATE"])
    var = (var.sort_values("FILING_DATE")
              .drop_duplicates("SERIES_ID", keep="last"))
    var = var[var["SERIES_ID"].notna()]
    out_var = DRV / "designated_index.csv"
    var[["SERIES_ID", "DESIGNATED_INDEX_NAME",
         "DESIGNATED_INDEX_IDENTIFIER", "FILING_DATE"]].to_csv(
        out_var, index=False)
    log.append(f"designated-index table: {len(var):,} series "
               f"-> {out_var.name}")
    top = var["DESIGNATED_INDEX_NAME"].value_counts().head(8)
    log.append("  top declared benchmarks: "
               + "; ".join(f"{i} ({n})" for i, n in top.items()))
else:
    log.append("no FUND_VAR_INFO tables found (only in newer archives)")

log.append("\nSTAGE 33b DONE. NOTE: series_id is the SEC S000-style id - "
           "linking to CRSP needs the SEC series/class-ticker mapping "
           "(stage 33c candidate) or LEI/name matching. Holdings extraction "
           "for the Active Share extension is a separate stage (33d).")
(OUT / "nport_33b_flows.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
