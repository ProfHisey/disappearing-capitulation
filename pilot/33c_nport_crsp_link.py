"""Stage 33c: LINK N-PORT SERIES IDS TO CRSP (via SEC ticker mapping).

Chain: SEC series_id (S000...) -> class tickers (SEC company_tickers_mf.json,
free) -> CRSP crsp_fundno (Fund Summary ticker match) -> wficn (mflink1).

Downloads the SEC mapping (small JSON; reuses the stage-32 CONTACT line for
the fair-access User-Agent), builds the link table, and reports coverage of
the stage-33b flows panel. Ticker matching is deliberately conservative:
a CRSP ticker that maps to MULTIPLE fundnos is kept but flagged ambiguous;
downstream work can require the unambiguous subset.

Outputs (git-ignored library):
  nport\\company_tickers_mf.json          raw SEC mapping (provenance copy)
  nport\\derived\\series_crsp_link.csv     series_id -> ticker -> fundno -> wficn
Report: output/nport_33c_link.txt (aggregates only).
"""
import json
import re
from pathlib import Path

import pandas as pd
import requests

SRC = Path(r"E:\Finance\data\sources")
NP = SRC / "nport"
DRV = NP / "derived"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

# reuse the identity Colin already set in stage 32
CONTACT = re.search(r'CONTACT = "(.*)"',
                    Path("32_nport_download.py").read_text()).group(1)
assert "FILL ME IN" not in CONTACT

log = ["N-PORT -> CRSP LINK (stage 33c)", "=" * 60]

# ---- 1. SEC series/class/ticker mapping ---------------------------------
raw_path = NP / "company_tickers_mf.json"
if not raw_path.exists():
    r = requests.get("https://www.sec.gov/files/company_tickers_mf.json",
                     headers={"User-Agent": CONTACT}, timeout=60)
    r.raise_for_status()
    raw_path.write_bytes(r.content)
data = json.loads(raw_path.read_text())
sec = pd.DataFrame(data["data"], columns=data["fields"])
sec.columns = [c.lower() for c in sec.columns]  # cik, seriesId, classId, symbol
sec = sec.rename(columns={"seriesid": "series_id", "classid": "class_id",
                          "symbol": "ticker"})
sec["ticker"] = sec["ticker"].astype(str).str.strip().str.upper()
sec = sec[sec["ticker"].ne("") & sec["ticker"].ne("NAN")]
log.append(f"SEC mapping: {len(sec):,} class rows, "
           f"{sec['series_id'].nunique():,} series, "
           f"{sec['ticker'].nunique():,} tickers")

# ---- 2. CRSP ticker -> fundno (latest ticker per fundno) ----------------
fs_path = SRC / "crsp_mf" / "Fund Summary.csv"
parts, date_col = [], None
for ch in pd.read_csv(fs_path, chunksize=2_000_000, low_memory=False,
                      encoding="latin-1"):
    ch.columns = [c.lower() for c in ch.columns]
    if date_col is None:
        date_col = next((c for c in ("caldt", "summary_period2", "begdt")
                         if c in ch.columns), None)
        log.append(f"Fund Summary date column used: {date_col}")
    ch = ch[ch["ticker"].notna()]
    parts.append(ch[["crsp_fundno", "ticker", date_col]])
allt = pd.concat(parts, ignore_index=True)
allt["ticker"] = allt["ticker"].astype(str).str.strip().str.upper()
crsp = (allt.sort_values(date_col)
            .drop_duplicates("crsp_fundno", keep="last")  # latest ticker
            [["crsp_fundno", "ticker"]])
log.append(f"CRSP fundnos with a ticker: {len(crsp):,} "
           f"({crsp['ticker'].nunique():,} distinct tickers)")
amb = crsp.groupby("ticker")["crsp_fundno"].nunique()
log.append(f"tickers mapping to >1 fundno (ambiguous): "
           f"{(amb > 1).sum():,} of {len(amb):,}")

# ---- 3. join the chain --------------------------------------------------
link = sec.merge(crsp, on="ticker", how="inner")
link["ambiguous"] = link["ticker"].map(amb > 1)
m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                 encoding="latin-1")
m1.columns = [c.lower() for c in m1.columns]
link = link.merge(m1[["crsp_fundno", "wficn"]].drop_duplicates(),
                  on="crsp_fundno", how="left")
DRV.mkdir(exist_ok=True)
out_csv = DRV / "series_crsp_link.csv"
link.to_csv(out_csv, index=False)
log.append(f"link rows written: {len(link):,} -> {out_csv.name}")
log.append(f"  series linked to >=1 fundno: {link['series_id'].nunique():,}")
log.append(f"  series linked unambiguously: "
           f"{link.loc[~link['ambiguous'], 'series_id'].nunique():,}")
log.append(f"  series linked to a wficn: "
           f"{link.loc[link['wficn'].notna(), 'series_id'].nunique():,}")

# ---- 4. coverage of the flows panel -------------------------------------
flows = pd.read_csv(DRV / "monthly_gross_flows.csv",
                    usecols=["series_id"], low_memory=False)
fl_series = set(flows["series_id"].unique())
linked = set(link["series_id"])
linked_w = set(link.loc[link["wficn"].notna(), "series_id"])
log.append(f"flows-panel series: {len(fl_series):,}")
log.append(f"  with any CRSP link: {len(fl_series & linked):,} "
           f"({len(fl_series & linked) / len(fl_series):.1%})")
log.append(f"  with a wficn: {len(fl_series & linked_w):,} "
           f"({len(fl_series & linked_w) / len(fl_series):.1%})")
log.append("  reading: wficn coverage is what matters for joining gross "
           "flows onto the capitulation panel. Sub-100% is expected - the "
           "flows panel includes bond/international/money funds outside "
           "MFLINKS's equity focus; judge coverage after filtering to "
           "domestic equity in the analysis stage.")

log.append("\nSTAGE 33c DONE - aggregates only. Next: 33d holdings "
           "extraction (post-2023 Active Share extension).")
(OUT / "nport_33c_link.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
