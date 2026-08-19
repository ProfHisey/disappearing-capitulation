"""Stage 33d2: ISIN/TICKER FALLBACK FOR MISSING CUSIPS (required pre-33e).

33d found 20% of kept equity holdings rows lack ISSUER_CUSIP. This stage
joins each quarter's IDENTIFIERS.tsv (by HOLDING_ID) to recover identity:
 - US ISINs embed the 9-char CUSIP (chars 3-11) -> fill cusip directly;
 - non-US ISINs are kept as-is (foreign holdings legitimately lack CUSIPs
   and mostly fall outside Russell benchmarks anyway);
 - tickers retained as a last-resort match key for 33e.
Parts are updated IN PLACE (new columns: isin, id_ticker, cusip_filled,
cusip_source). Report: output/nport_33d2_isin.txt (aggregates only).

Safe to run alongside 34/35/36 (only this script touches the parts).
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P

NP = Path(r"E:\Finance\data\sources\nport")
PARTS = P.CACHE / "nport_holdings_parts"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["N-PORT ISIN/TICKER FALLBACK (stage 33d2)", "=" * 60]

for part in sorted(PARTS.glob("2*.parquet")):
    qtr = part.stem
    zpath = NP / f"{qtr}_nport.zip"
    q = pd.read_parquet(part)
    if "cusip_source" in q.columns:
        log.append(f"  {qtr}: already processed, skipped")
        continue
    need = set(q.loc[q["ISSUER_CUSIP"].isna(), "HOLDING_ID"])
    idf = []
    with zipfile.ZipFile(zpath) as zf:
        for ch in pd.read_csv(zf.open("IDENTIFIERS.tsv"), sep="\t",
                              usecols=["HOLDING_ID", "IDENTIFIER_ISIN",
                                       "IDENTIFIER_TICKER"],
                              chunksize=2_000_000, low_memory=False):
            ch = ch[ch["HOLDING_ID"].isin(need)]
            if len(ch):
                idf.append(ch)
    ids = (pd.concat(idf, ignore_index=True)
             .groupby("HOLDING_ID").first().reset_index()
           if idf else pd.DataFrame(columns=["HOLDING_ID",
                                             "IDENTIFIER_ISIN",
                                             "IDENTIFIER_TICKER"]))
    ids.columns = ["HOLDING_ID", "isin", "id_ticker"]
    q = q.merge(ids, on="HOLDING_ID", how="left")
    isin = q["isin"].astype(str)
    us = isin.str.startswith("US") & (isin.str.len() == 12)
    cusip_from_isin = isin.str[2:11].where(us)
    q["cusip_filled"] = q["ISSUER_CUSIP"].fillna(cusip_from_isin)
    q["cusip_source"] = np.select(
        [q["ISSUER_CUSIP"].notna(), cusip_from_isin.notna()],
        ["reported", "us_isin"], default="none")
    q.to_parquet(part, index=False)

    n = len(q)
    n_miss0 = int((q["cusip_source"] != "reported").sum())
    n_rec = int((q["cusip_source"] == "us_isin").sum())
    n_still = int(q["cusip_filled"].isna().sum())
    n_tick = int((q["cusip_filled"].isna() & q["id_ticker"].notna()).sum())
    foreign = int((q["cusip_filled"].isna()
                   & q["isin"].notna()
                   & ~q["isin"].astype(str).str.startswith("US")).sum())
    log.append(f"  {qtr}: rows {n:,}; missing cusip {n_miss0:,} "
               f"-> recovered via US ISIN {n_rec:,}; still missing "
               f"{n_still:,} (of which foreign-ISIN {foreign:,}, "
               f"ticker-only {n_tick:,})")

log.append("\nreading: 'still missing & foreign' is fine - non-US names "
           "sit outside the Russell benchmark universe. If ticker-only is "
           "large, 33e adds a ticker match against s12type2/crsp_stock.")
log.append("STAGE 33d2 DONE - parts updated in place; 33e unblocked.")
(OUT / "nport_33d2_isin.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
