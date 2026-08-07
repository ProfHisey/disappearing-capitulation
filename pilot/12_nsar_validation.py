"""Stage 12: N-SAR VALIDATION — coverage, and which column is redemptions?

WRDS's N-SAR extract names Item 28's four monthly columns generically
(shr_sold1..4). The actual form's columns are: (1) new sales, (2) reinvested
dividends, (3) shares redeemed, (4) other/exchanges. Rather than trust labels,
identify them EMPIRICALLY: for funds matchable to CRSP (via ticker), the
imputed CRSP net flow should correlate best with [sales + reinvest - redeemed
+/- other] style combinations. The winning combination reveals the semantics.

Also reports: coverage span, filings/rows counts, month_of_period structure,
match rates. Aggregates only.
Outputs: output/nsar_report.txt
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

NSAR = P.SOURCES / "nsar"
log = ["N-SAR VALIDATION", "=" * 60]

# ------------------------------------------------------------- coverage ----
hd = P.norm_cols(pd.read_csv(NSAR / "headers.csv", encoding="latin-1",
                             low_memory=False))
hd["rdate"] = pd.to_datetime(hd["rdate"], errors="coerce")
log.append(f"headers: {len(hd):,} class-rows, {hd['accession'].nunique():,} "
           f"filings, {hd['series_id'].nunique():,} series; rdate "
           f"{hd['rdate'].min():%Y-%m} to {hd['rdate'].max():%Y-%m}")
tickers = hd["class_contract_ticker_symbol"].astype(str).str.strip().str.upper()
log.append(f"class rows with a ticker: {(tickers.str.len() > 1).sum():,} "
           f"({(tickers.str.len() > 1).mean():.0%})")

ms = P.norm_cols(pd.read_csv(NSAR / "monthly_sales_repurchases.csv",
                             encoding="latin-1", low_memory=False))
log.append(f"monthly_sales_repurchases: {len(ms):,} rows, "
           f"{ms['accession'].nunique():,} filings")
log.append(f"month_of_period values: {sorted(ms['month_of_period'].dropna().unique().tolist())}")
CAND = [c for c in ms.columns if "sold" in c or "purch" in c]
for c in CAND:
    v = pd.to_numeric(ms[c], errors="coerce")
    log.append(f"  {c}: non-null {v.notna().mean():.0%}, median {v.median():,.0f}, "
               f"negative share {(v < 0).mean():.1%}")

# --------------------------------------- match to CRSP via class ticker ----
# use single-series filings only (unambiguous accession -> ticker mapping)
ntk = hd.groupby("accession")["class_contract_ticker_symbol"].nunique()
single = set(ntk[ntk == 1].index)
h1 = hd[hd["accession"].isin(single)].drop_duplicates("accession")
h1["ticker"] = h1["class_contract_ticker_symbol"].astype(str).str.strip().str.upper()
log.append(f"\nsingle-ticker filings usable for validation: {len(h1):,}")

# ticker -> crsp_fundno (from mflink1, which carries tickers)
m1 = P.norm_cols(pd.read_csv(PL.MFLINK1))
m1["ticker"] = m1["ticker"].astype(str).str.strip().str.upper()
tmap = (m1[m1["ticker"].str.len() > 1]
        .drop_duplicates("ticker")[["ticker", "crsp_fundno"]])
h1 = h1.merge(tmap, on="ticker", how="inner")
log.append(f"matched to a crsp_fundno via mflink1 ticker: {len(h1):,}")

# imputed CRSP monthly net flow per share class ($M)
ret = P.load_monthly_returns([])
ret = ret.sort_values(["crsp_fundno", "caldt"])
ret["tna_lag"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
ret["iflow"] = ret["mtna"] - ret["tna_lag"] * (1 + ret["mret"])
ret["month"] = ret["caldt"].dt.to_period("M")
iflow = ret.dropna(subset=["iflow"])[["crsp_fundno", "month", "iflow"]]

# map filing months: month_of_period m of a filing ending rdate covers
# calendar month rdate - (maxm - m)
ms2 = ms.merge(h1[["accession", "crsp_fundno", "rdate"]], on="accession",
               how="inner")
ms2["maxm"] = ms2.groupby("accession")["month_of_period"].transform("max")
ms2["month"] = (ms2["rdate"].dt.to_period("M")
                - (ms2["maxm"] - ms2["month_of_period"]).astype(int))
ms2 = ms2.merge(iflow, on=["crsp_fundno", "month"], how="inner")
log.append(f"matched N-SAR fund-months with CRSP imputed flow: {len(ms2):,}")

if len(ms2) >= 500:
    for c in CAND:
        ms2[c] = pd.to_numeric(ms2[c], errors="coerce")
    log.append("\ncorrelation of candidate columns/combos with CRSP imputed "
               "net flow (identifies semantics; scale-free):")
    base = [c for c in CAND if c.startswith("shr_sold")]
    results = []
    for c in CAND:
        j = ms2[[c, "iflow"]].dropna()
        if len(j) > 200:
            results.append((f"+{c}", j[c].corr(j["iflow"]), len(j)))
    for i in base:
        for k in base:
            if i != k:
                d = (ms2[i] - ms2[k])
                j = pd.concat([d, ms2["iflow"]], axis=1).dropna()
                if len(j) > 200:
                    results.append((f"{i} - {k}", j.iloc[:, 0].corr(j["iflow"]),
                                    len(j)))
    for name, corr, n in sorted(results, key=lambda x: -abs(x[1]))[:12]:
        log.append(f"  {name:28s} corr {corr:+.3f}  (n={n:,})")
    log.append("\nreading: the top positive combo should be [sales - redemptions];"
               " the redemptions column is the SUBTRACTED one. The top single"
               " NEGATIVE correlate is likely redemptions itself.")
    # units check on the best single positive correlate
    best = max((r for r in results if r[0].startswith("+")),
               key=lambda x: abs(x[1]), default=None)
    if best:
        c = best[0][1:]
        j = ms2[[c, "iflow"]].dropna()
        j = j[(j["iflow"].abs() > 0.1) & (j[c].abs() > 0)]
        if len(j) > 100:
            ratio = (j[c] / (j["iflow"] * 1e6)).median()
            log.append(f"units hint: median {c} / (iflow in $): {ratio:,.2f} "
                       "(~1 => dollars; ~0.001 => $ thousands)")
else:
    log.append("TOO FEW matched fund-months - ticker match too thin; "
               "fallback: match via CUSIP in real build.")

log.append("\nNSAR VALIDATION DONE - aggregates only.")
P.write_report("nsar_report.txt", log)
print("\n".join(log))
