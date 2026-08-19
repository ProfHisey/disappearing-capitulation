"""Stage 33d: EXTRACT EQUITY HOLDINGS FROM N-PORT (post-ND-era, 2023q4+).

Purpose: raw material for computing our own Active Share past the Notre
Dame series' Sep-2023 end (Russell weights on disk run to 2026). This stage
extracts and filters only; the AS-vs-benchmark computation is stage 33e
(adapting the stage-21b machinery).

Two phases:
 1. Small tables from every archive 2023q4+ -> filings for wficn-linked
    series (link v2), amendment-deduped GLOBALLY (latest FILING_DATE per
    series-period, across zips - an amendment can sit in a later zip).
 2. Stream FUND_REPORTED_HOLDING per archive; keep equity rows (ASSET_CAT
    starting 'E') for kept accessions; write one parquet part per archive
    to cache\\nport_holdings_parts\\.

Report: output/nport_33d_holdings.txt (aggregates only).
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

import pilot_lib as P

SRC = Path(r"E:\Finance\data\sources")
NP = SRC / "nport"
DRV = NP / "derived"
PARTS = P.CACHE / "nport_holdings_parts"
PARTS.mkdir(parents=True, exist_ok=True)
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["N-PORT EQUITY HOLDINGS EXTRACT (stage 33d)", "=" * 60]

# only archives that can contain post-ND filings (ND ends 2023m9)
zips = sorted(z for z in NP.glob("*_nport.zip")
              if z.name >= "2023q4")
log.append(f"archives in scope (2023q4+): {len(zips)}")

link = pd.read_csv(DRV / "series_crsp_link_v2.csv", low_memory=False)
keep_series = set(link.loc[link["wficn"].notna(), "series_id"])
log.append(f"wficn-linked series (link v2): {len(keep_series):,}")

# ---- phase 1: global filing keep-list -----------------------------------
metas = []
for z in zips:
    with zipfile.ZipFile(z) as zf:
        sub = pd.read_csv(zf.open("SUBMISSION.tsv"), sep="\t",
                          low_memory=False)
        info = pd.read_csv(zf.open("FUND_REPORTED_INFO.tsv"), sep="\t",
                           usecols=["ACCESSION_NUMBER", "SERIES_ID",
                                    "NET_ASSETS", "TOTAL_ASSETS"],
                           low_memory=False)
        m = info.merge(sub[["ACCESSION_NUMBER", "FILING_DATE",
                            "REPORT_DATE", "SUB_TYPE"]],
                       on="ACCESSION_NUMBER", how="left")
        m["src_zip"] = z.name
        metas.append(m)
meta = pd.concat(metas, ignore_index=True)
meta = meta[meta["SERIES_ID"].isin(keep_series)]
for c in ("FILING_DATE", "REPORT_DATE"):
    meta[c] = pd.to_datetime(meta[c], format="%d-%b-%Y", errors="coerce")
    if meta[c].isna().mean() > 0.1:  # fallback if format differs
        meta[c] = pd.to_datetime(meta[c], errors="coerce")
meta["_period"] = meta["REPORT_DATE"].dt.to_period("M")
# post-ND only: quarters ending after 2023-09
meta = meta[meta["_period"] > pd.Period("2023-09", freq="M")]
n0 = len(meta)
meta = (meta.sort_values("FILING_DATE")
            .drop_duplicates(["SERIES_ID", "_period"], keep="last"))
log.append(f"filings: {n0:,} in scope -> {len(meta):,} after global "
           f"amendment dedup; {meta['SERIES_ID'].nunique():,} series; "
           f"periods {meta['_period'].min()} to {meta['_period'].max()}")
keep_acc = {}
for z in zips:
    keep_acc[z.name] = set(
        meta.loc[meta["src_zip"] == z.name, "ACCESSION_NUMBER"])
meta_out = meta[["ACCESSION_NUMBER", "SERIES_ID", "_period",
                 "NET_ASSETS", "TOTAL_ASSETS", "SUB_TYPE"]]
meta_out.columns = ["accession", "series_id", "period", "net_assets",
                    "total_assets", "sub_type"]
meta_out["period"] = meta_out["period"].astype(str)
meta_out.to_parquet(PARTS / "_filings_meta.parquet", index=False)

# ---- phase 2: stream holdings -------------------------------------------
USE = ["ACCESSION_NUMBER", "HOLDING_ID", "ISSUER_NAME", "ISSUER_CUSIP",
       "BALANCE", "CURRENCY_CODE", "CURRENCY_VALUE", "PERCENTAGE",
       "ASSET_CAT", "ISSUER_TYPE", "INVESTMENT_COUNTRY"]
tot_rows = tot_kept = 0
cat_dist = {}
for z in zips:
    acc = keep_acc[z.name]
    if not acc:
        log.append(f"  {z.name}: no kept filings, skipped")
        continue
    parts = []
    with zipfile.ZipFile(z) as zf:
        for ch in pd.read_csv(zf.open("FUND_REPORTED_HOLDING.tsv"),
                              sep="\t", usecols=USE, chunksize=1_000_000,
                              low_memory=False):
            tot_rows += len(ch)
            ch = ch[ch["ACCESSION_NUMBER"].isin(acc)]
            for k, v in ch["ASSET_CAT"].value_counts().items():
                cat_dist[k] = cat_dist.get(k, 0) + int(v)
            ch = ch[ch["ASSET_CAT"].astype(str).str.startswith("E")]
            if len(ch):
                parts.append(ch)
                tot_kept += len(ch)
    if parts:
        q = pd.concat(parts, ignore_index=True)
        q.to_parquet(PARTS / f"{z.name.split('_')[0]}.parquet",
                     index=False)
        log.append(f"  {z.name}: kept {len(q):,} equity rows "
                   f"({q['ACCESSION_NUMBER'].nunique():,} filings)")

log.append(f"\nholdings rows scanned: {tot_rows:,}; equity rows kept: "
           f"{tot_kept:,}")
log.append("ASSET_CAT distribution among linked funds' holdings "
           "(pre-equity-filter): "
           + str(dict(sorted(cat_dist.items(), key=lambda x: -x[1])[:12])))

# quick quality read on the kept rows (latest part)
qq = pd.read_parquet(sorted(PARTS.glob("2*.parquet"))[-1])
miss_cusip = qq["ISSUER_CUSIP"].isna().mean()
log.append(f"latest quarter: missing CUSIP share {miss_cusip:.1%} "
           "(HOLDING_ID retained - ISIN fallback via IDENTIFIERS.tsv is "
           "stage 33d2 if this is material)")
ps = (qq.groupby("ACCESSION_NUMBER")["PERCENTAGE"].sum())
log.append(f"sum of equity PERCENTAGE per filing, latest quarter: median "
           f"{ps.median():.1f}, p10 {ps.quantile(0.1):.1f}, "
           f"p90 {ps.quantile(0.9):.1f} (values near 90-100 = equity "
           f"funds; low values = hybrid/bond funds to be filtered in 33e)")

log.append("\nSTAGE 33d DONE - parts in cache\\nport_holdings_parts\\. "
           "Next: 33e computes Active Share vs Russell benchmarks "
           "(stage-21b machinery) and splices onto the ND series.")
(OUT / "nport_33d_holdings.txt").write_text("\n".join(log),
                                            encoding="utf-8")
print("\n".join(log))
