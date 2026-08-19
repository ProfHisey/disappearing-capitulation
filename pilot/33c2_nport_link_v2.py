"""Stage 33c2: SURVIVORSHIP-FREE N-PORT -> CRSP LINK (v2).

v1 used the SEC's company_tickers_mf.json - a CURRENT-registrants snapshot,
so funds that died 2019-2026 were missing from the bridge (survivorship
bias aimed at exactly the deaths the client-arm tests care about). v2 fixes
this with the SEC's ANNUAL "Investment Company Series and Class Information"
lists (2019-2026): a fund alive in any year appears in that year's list,
dead or not by today. Union of all years -> series_id/ticker bridge ->
same chain as v1 (ticker -> CRSP fundno -> wficn).

Outputs: nport\\sec_series_class\\series_class_<year>.csv (raw provenance)
         nport\\derived\\series_crsp_link_v2.csv
Report:  output/nport_33c2_link.txt (aggregates only).
"""
import re
import time
from pathlib import Path

import pandas as pd
import requests

SRC = Path(r"E:\Finance\data\sources")
NP = SRC / "nport"
SC = NP / "sec_series_class"
SC.mkdir(exist_ok=True)
DRV = NP / "derived"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

CONTACT = re.search(r'CONTACT = "(.*)"',
                    Path("32_nport_download.py").read_text()).group(1)
HEAD = {"User-Agent": CONTACT}

# SEC's own URL scheme is inconsistent across years - try all combinations
BASES = ["https://www.sec.gov/files/investment/data/other/"
         "investment-company-series-class-information",
         "https://www.sec.gov/files/investment/data/other/"
         "investment-company-series-and-class-information"]
NAMES = ["investment-company-series-class-{y}.csv",
         "investment_company_series_class_{y}.csv"]

log = ["N-PORT -> CRSP LINK v2, SURVIVORSHIP-FREE (stage 33c2)", "=" * 60]

# ---- 1. download the yearly lists ---------------------------------------
for y in range(2019, 2027):
    dest = SC / f"series_class_{y}.csv"
    if dest.exists() and dest.stat().st_size > 10_000:
        log.append(f"  {y}: already on disk")
        continue
    got = False
    for base in BASES:
        for name in NAMES:
            url = f"{base}/{name.format(y=y)}"
            try:
                r = requests.get(url, headers=HEAD, timeout=60)
                if r.status_code == 200 and len(r.content) > 10_000:
                    dest.write_bytes(r.content)
                    log.append(f"  {y}: downloaded "
                               f"({len(r.content) / 1e6:.1f} MB)")
                    got = True
                    break
            except requests.RequestException:
                pass
        if got:
            break
    if not got:
        log.append(f"  {y}: NOT FOUND at any candidate URL - investigate")
    time.sleep(1)

# ---- 2. union into a series/ticker bridge -------------------------------
def norm(c):
    return re.sub(r"\W+", "_", str(c).strip().lower()).strip("_")

frames = []
for f in sorted(SC.glob("series_class_*.csv")):
    year = int(f.stem.split("_")[-1])
    df = pd.read_csv(f, encoding="latin-1", low_memory=False)
    df.columns = [norm(c) for c in df.columns]
    s_col = next((c for c in df.columns
                  if "series" in c and "id" in c), None)
    t_col = next((c for c in df.columns if "ticker" in c), None)
    c_col = next((c for c in df.columns
                  if c.startswith("class") and c.endswith("id")), None)
    if not (s_col and t_col):
        log.append(f"  {f.name}: columns not recognized "
                   f"({list(df.columns)[:6]}...) - SKIPPED")
        continue
    part = df[[s_col, t_col] + ([c_col] if c_col else [])].copy()
    part.columns = (["series_id", "ticker"]
                    + (["class_id"] if c_col else []))
    part["list_year"] = year
    frames.append(part)

bridge = pd.concat(frames, ignore_index=True)
bridge["ticker"] = bridge["ticker"].astype(str).str.strip().str.upper()
bridge = bridge[bridge["ticker"].ne("") & bridge["ticker"].ne("NAN")
                & bridge["series_id"].notna()]
years_seen = bridge.groupby("series_id")["list_year"].agg(["min", "max"])
bridge = (bridge.sort_values("list_year")
                .drop_duplicates(["series_id", "ticker"], keep="last"))
log.append(f"bridge: {len(bridge):,} series-ticker pairs, "
           f"{bridge['series_id'].nunique():,} series "
           f"(v1 snapshot had 12,011)")

# ---- 3. CRSP ticker map + chain (same as v1) ----------------------------
fs_path = SRC / "crsp_mf" / "Fund Summary.csv"
parts, date_col = [], None
for ch in pd.read_csv(fs_path, chunksize=2_000_000, low_memory=False,
                      encoding="latin-1"):
    ch.columns = [c.lower() for c in ch.columns]
    if date_col is None:
        date_col = next((c for c in ("caldt", "summary_period2", "begdt")
                         if c in ch.columns), None)
    ch = ch[ch["ticker"].notna()]
    parts.append(ch[["crsp_fundno", "ticker", date_col]])
allt = pd.concat(parts, ignore_index=True)
allt["ticker"] = allt["ticker"].astype(str).str.strip().str.upper()
crsp = (allt.sort_values(date_col)
            .drop_duplicates("crsp_fundno", keep="last")
            [["crsp_fundno", "ticker"]])
amb = crsp.groupby("ticker")["crsp_fundno"].nunique()

link = bridge.merge(crsp, on="ticker", how="inner")
link["ambiguous"] = link["ticker"].map(amb > 1)
m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                 encoding="latin-1")
m1.columns = [c.lower() for c in m1.columns]
link = link.merge(m1[["crsp_fundno", "wficn"]].drop_duplicates(),
                  on="crsp_fundno", how="left")
link = link.merge(years_seen.rename(columns={"min": "first_list_year",
                                             "max": "last_list_year"}),
                  on="series_id", how="left")
out_csv = DRV / "series_crsp_link_v2.csv"
link.to_csv(out_csv, index=False)
log.append(f"link rows: {len(link):,} -> {out_csv.name}")
log.append(f"  series linked to >=1 fundno: {link['series_id'].nunique():,}")
log.append(f"  series linked to a wficn: "
           f"{link.loc[link['wficn'].notna(), 'series_id'].nunique():,}")

# ---- 4. coverage vs flows panel, v1 vs v2 -------------------------------
flows = pd.read_csv(DRV / "monthly_gross_flows.csv",
                    usecols=["series_id"], low_memory=False)
fl = set(flows["series_id"].unique())
linked = set(link["series_id"])
linked_w = set(link.loc[link["wficn"].notna(), "series_id"])
log.append(f"flows-panel series: {len(fl):,}")
log.append(f"  any CRSP link: {len(fl & linked):,} "
           f"({len(fl & linked) / len(fl):.1%})  [v1: 60.6%]")
log.append(f"  with a wficn:  {len(fl & linked_w):,} "
           f"({len(fl & linked_w) / len(fl):.1%})  [v1: 40.2%]")
dead_ish = set(years_seen[years_seen["max"] <= 2024].index)
log.append(f"  series that VANISH from SEC lists by 2024 (death "
           f"candidates) now linkable: {len(fl & dead_ish & linked):,} "
           f"- these are what v1's survivor-only snapshot missed.")

log.append("\nSTAGE 33c2 DONE - aggregates only. v2 supersedes v1 for all "
           "downstream joins; last_list_year is a bonus death-year proxy.")
(OUT / "nport_33c2_link.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
