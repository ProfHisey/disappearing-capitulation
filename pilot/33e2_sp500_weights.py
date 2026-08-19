"""Stage 33e2: S&P 500 WEIGHTS -> AS_S5 FOR THE EXTENSION (audit follow-up).

The 33e extension computed min-AS over Russell indexes only; the paper's
benchmark vocabulary is ~half S&P families. This stage builds month-end
S&P 500 weights from crsp_sp500\\sp500_constituents_daily.csv (membership
spans + daily caps) for 2023-08+, computes as_s5 for every extension
filing, and writes an augmented extension cache. S&P style/mid/small
variants remain unavailable (Morningstar-gated) - BENCH_APPROX continues
to proxy those, as in the paper panel.

PROBE-FIRST: the constituents file's schema has never been read by this
pipeline generation. Phase 0 prints the detected columns and HARD-STOPS
with instructions if the needed fields can't be identified - paste the
schema output back rather than guessing.

Output: cache\\nport_as_extension_v2.parquet (adds as_s5, as_min_v2,
bench_min_v2) + report output/nport_33e2_s5.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P

SRC = Path(r"E:\Finance\data\sources")
CONS = SRC / "crsp_sp500" / "sp500_constituents_daily.csv"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["S&P 500 WEIGHTS FOR THE EXTENSION (stage 33e2)", "=" * 60]

# ---- phase 0: schema probe ----------------------------------------------
head = pd.read_csv(CONS, nrows=50_000, low_memory=False)
head.columns = [c.strip().lower() for c in head.columns]
log.append(f"constituents columns ({len(head.columns)}): "
           + ", ".join(head.columns))
dcol = next((c for c in head.columns
             if c in ("date", "caldt", "dlycaldt", "mthcaldt")
             or "caldt" in c or c == "datadate"), None)
ccol = next((c for c in head.columns if "cusip" in c), None)
pcol = next((c for c in head.columns if "permno" in c), None)
capcol = next((c for c in head.columns
               if "cap" in c or c in ("dlycap", "mktcap", "totcap")), None)
prccol = next((c for c in head.columns
               if c in ("dlyprc", "prc", "price")), None)
shrcol = next((c for c in head.columns
               if "shr" in c and "out" in c or c == "shrout"), None)
log.append(f"detected: date={dcol}, cusip={ccol}, permno={pcol}, "
           f"cap={capcol}, prc={prccol}, shrout={shrcol}")
if dcol is None or (capcol is None and (prccol is None or shrcol is None)):
    log.append("HARD STOP: cannot identify date + market-cap fields. "
               "Paste this schema output back for a targeted fix.")
    P.write_report("nport_33e2_s5.txt", log)
    raise SystemExit("schema not recognized - see report")

# ---- phase 1: month-end S5 weights, 2023-08+ ----------------------------
use = [c for c in (dcol, ccol, pcol, capcol, prccol, shrcol) if c]
parts = []
for ch in pd.read_csv(CONS, usecols=lambda c: c.strip().lower() in use,
                      chunksize=2_000_000, low_memory=False,
                      encoding="latin-1"):
    ch.columns = [c.strip().lower() for c in ch.columns]
    ch["d"] = pd.to_datetime(ch[dcol], errors="coerce")
    ch = ch[ch["d"] >= "2023-08-01"]
    if len(ch):
        parts.append(ch)
cons = pd.concat(parts, ignore_index=True)
if capcol:
    cons["cap"] = pd.to_numeric(cons[capcol], errors="coerce")
else:
    cons["cap"] = (pd.to_numeric(cons[prccol], errors="coerce").abs()
                   * pd.to_numeric(cons[shrcol], errors="coerce"))
cons = cons[cons["cap"] > 0]
cons["m"] = cons["d"].dt.to_period("M")
# last trading day per month
last_day = cons.groupby("m")["d"].transform("max")
cons = cons[cons["d"] == last_day]
log.append(f"month-ends kept: {cons['m'].nunique()} "
           f"({cons['m'].min()} to {cons['m'].max()}); median names/month "
           f"{cons.groupby('m').size().median():.0f} (expect ~500)")

# cusip key: direct, else permno -> ncusip map from crsp_stock
if ccol:
    cons["c8"] = cons[ccol].astype(str).str[:8]
else:
    log.append("no cusip column - mapping permno via crsp_stock")
    cm = pd.read_csv(SRC / "crsp_stock" / "crsp_monthly.csv",
                     usecols=lambda c: c.strip().lower() in
                     ("permno", "cusip", "ncusip", "date"),
                     low_memory=False, encoding="latin-1")
    cm.columns = [c.strip().lower() for c in cm.columns]
    kc = "ncusip" if "ncusip" in cm.columns else "cusip"
    cm = (cm.dropna(subset=[kc]).sort_values("date")
            .drop_duplicates("permno", keep="last"))
    pmap = dict(zip(cm["permno"], cm[kc].astype(str).str[:8]))
    cons["c8"] = cons[pcol].map(pmap)
    log.append(f"  permno->cusip mapped: {cons['c8'].notna().mean():.1%}")
cons = cons.dropna(subset=["c8"])
wt = (cons.groupby(["m", "c8"])["cap"].sum().reset_index())
wt["w"] = wt["cap"] / wt.groupby("m")["cap"].transform("sum")
S5 = {m: g.set_index("c8")["w"] for m, g in wt.groupby("m")}
chk = next(iter(S5.values()))
log.append(f"weight sanity: first month sum {chk.sum():.3f} "
           f"(names {len(chk)})")

# ---- phase 2: as_s5 per extension filing --------------------------------
PARTS = P.CACHE / "nport_holdings_parts"
meta = pd.read_parquet(PARTS / "_filings_meta.parquet")
meta["pend_m"] = pd.PeriodIndex(meta["period"], freq="M")
acc_meta = meta.set_index("accession")
s5_months = sorted(S5)

rows = []
for part in sorted(PARTS.glob("2*.parquet")):
    q = pd.read_parquet(part, columns=["ACCESSION_NUMBER", "cusip_filled",
                                       "CURRENCY_VALUE"])
    q = q[(q["CURRENCY_VALUE"] > 0) & q["cusip_filled"].notna()]
    q["c8"] = q["cusip_filled"].astype(str).str[:8]
    w = (q.groupby(["ACCESSION_NUMBER", "c8"])["CURRENCY_VALUE"]
         .sum().reset_index())
    for acc, g in w.groupby("ACCESSION_NUMBER"):
        if acc not in acc_meta.index:
            continue
        pm = acc_meta.loc[acc, "pend_m"]
        ms = [m for m in s5_months if m <= pm]
        if not ms:
            continue
        b = S5[ms[-1]]
        wv = g.set_index("c8")["CURRENCY_VALUE"]
        wv = wv / wv.sum()
        summin = np.minimum(wv, b.reindex(wv.index).fillna(0.0)).sum()
        rows.append((acc, 1.0 - float(summin)))
    print(f"done {part.stem}")
s5df = pd.DataFrame(rows, columns=["accession", "as_s5"])

ext = pd.read_parquet(P.CACHE / "nport_as_extension.parquet")
ext = ext.merge(s5df, on="accession", how="left")
as_cols = [c for c in ext.columns if c.startswith("as_")
           and c not in ("as_min_ru", "as_s5")]
ext["as_min_v2"] = ext[as_cols + ["as_s5"]].min(axis=1)
ext["bench_min_v2"] = (ext[as_cols + ["as_s5"]].idxmin(axis=1)
                       .str.replace("as_", "").str.upper()
                       .str.replace("S5", "S5"))
ext.to_parquet(P.CACHE / "nport_as_extension_v2.parquet", index=False)
log.append(f"\nas_s5 computed for {len(s5df):,} filings; v2 cache written")
log.append(f"as_min change (v2 minus v1): median "
           f"{(ext['as_min_v2'] - ext['as_min_ru']).median():+.4f}, "
           f"share changed >1pt "
           f"{((ext['as_min_ru'] - ext['as_min_v2']) > 0.01).mean():.1%}")
log.append(f"bench_min_v2 = S5 share: "
           f"{(ext['bench_min_v2'] == 'S5').mean():.1%}")
newly_below = ((ext["as_min_v2"] < 0.70)
               & (ext["as_min_ru"] >= 0.70)).sum()
log.append(f"filings newly below 0.70 under v2: {newly_below:,} "
           f"(the S&P-absence undercount, now measured)")
log.append("\nSTAGE 33e2 DONE - aggregates only. 33i integration uses "
           "the v2 cache automatically if present.")
P.write_report("nport_33e2_s5.txt", log)
print("\n".join(log))
