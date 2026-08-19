"""Stage 33e: ACTIVE SHARE EXTENSION 2023q4-2026q2 (N-PORT x Russell).

The build the whole N-PORT program aimed at: compute min-AS for the
post-ND era from N-PORT equity holdings vs Russell index weights, so
Paper 1's sample extends from Sep-2023 to mid-2026.

Conventions (documented, mirroring the paper's machinery where possible):
 - fund weights: identified-CUSIP equity sleeve, renormalized to 1
   (funds with <80% of equity dollars identified are excluded as
   international/unidentifiable - the 33d2 missingness-as-filter design);
 - benchmark weights: Russell *_wt columns, renormalized per index-date;
 - AS_j = 1 - sum_i min(w_i, b_ij); as_min_ru = min over 12 Russell
   indexes. NO S&P benchmarks in v1 (S5 from constituents file = 33e2);
   splice validation therefore restricts to Russell-benchmarked funds.
 - benchmark date: last Russell month-end <= filing's report month.

Output: cache\\nport_as_extension.parquet + aggregates report
        output/nport_33e_extension.txt
Heavy-ish (~10-20 min) but panel-free.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P

SRC = Path(r"E:\Finance\data\sources")
PARTS = P.CACHE / "nport_holdings_parts"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["ACTIVE SHARE EXTENSION 2023q4-2026q2 (stage 33e)", "=" * 60]

RUS_MAP = {"R3": "r3000_wt", "R3G": "r3000g_wt", "R3V": "r3000v_wt",
           "R1": "r1000_wt", "R1G": "r1000g_wt", "R1V": "r1000v_wt",
           "R2": "r2000_wt", "R2G": "r2000g_wt", "R2V": "r2000v_wt",
           "RM": "rmidc_wt", "RMG": "rmidcg_wt", "RMV": "rmidcv_wt"}

# ---- Russell weights, 2023-08 onward ------------------------------------
use = ["date", "cusip"] + list(RUS_MAP.values())
rus = []
for ch in pd.read_csv(SRC / "Russell" / "idx_holdings_us.csv",
                      usecols=lambda c: c.lower() in use,
                      chunksize=1_000_000, low_memory=False):
    ch.columns = [c.lower() for c in ch.columns]
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    ch = ch[ch["date"] >= "2023-08-01"]
    if len(ch):
        rus.append(ch)
rus = pd.concat(rus, ignore_index=True)
rus["c8"] = rus["cusip"].astype(str).str[:8]
BENCH = {}
for dt, g in rus.groupby("date"):
    m = g.groupby("c8")[list(RUS_MAP.values())].sum()
    tot = m.sum(axis=0)
    m = m.div(tot.where(tot > 0, np.nan), axis=1).fillna(0.0)
    BENCH[dt] = m
rus_dates = sorted(BENCH)
log.append(f"Russell weight matrices: {len(rus_dates)} month-ends, "
           f"{rus_dates[0].date()} to {rus_dates[-1].date()}")
chk = BENCH[rus_dates[-1]].sum(axis=0)
log.append("  latest-date column sums after renormalize (should be ~1 "
           "or 0 for empty): "
           + ", ".join(f"{k.split('_')[0]}:{v:.2f}"
                       for k, v in chk.items()))

# ---- filings meta -------------------------------------------------------
meta = pd.read_parquet(PARTS / "_filings_meta.parquet")
meta["pend"] = (pd.PeriodIndex(meta["period"], freq="M")
                .to_timestamp(how="end"))
acc_meta = meta.set_index("accession")

def bench_date_for(pend):
    ds = [d for d in rus_dates if d <= pend]
    return ds[-1] if ds else None

# ---- per-quarter AS computation -----------------------------------------
res = []
n_dropped_id = 0
for part in sorted(PARTS.glob("2*.parquet")):
    q = pd.read_parquet(part, columns=["ACCESSION_NUMBER", "cusip_filled",
                                       "CURRENCY_VALUE"])
    q = q[q["CURRENCY_VALUE"] > 0]
    tot = q.groupby("ACCESSION_NUMBER")["CURRENCY_VALUE"].sum()
    qid = q[q["cusip_filled"].notna()].copy()
    idv = qid.groupby("ACCESSION_NUMBER")["CURRENCY_VALUE"].sum()
    idshare = (idv / tot).rename("id_share")
    keep = idshare[idshare >= 0.80].index
    n_dropped_id += int((idshare < 0.80).sum())
    qid = qid[qid["ACCESSION_NUMBER"].isin(keep)]
    qid["c8"] = qid["cusip_filled"].astype(str).str[:8]
    w = (qid.groupby(["ACCESSION_NUMBER", "c8"])["CURRENCY_VALUE"]
         .sum().reset_index())
    for acc, g in w.groupby("ACCESSION_NUMBER"):
        if acc not in acc_meta.index:
            continue
        row = acc_meta.loc[acc]
        bd = bench_date_for(row["pend"])
        if bd is None:
            continue
        B = BENCH[bd]
        wv = g.set_index("c8")["CURRENCY_VALUE"]
        wv = wv / wv.sum()
        Bm = B.reindex(wv.index).fillna(0.0).to_numpy()
        summin = np.minimum(wv.to_numpy()[:, None], Bm).sum(axis=0)
        as_vec = 1.0 - summin
        res.append([acc, row["series_id"], row["period"], len(wv),
                    float(idshare.get(acc, np.nan))] + list(as_vec))
    print(f"done {part.stem}")

cols = (["accession", "series_id", "period", "n_holdings", "id_share"]
        + ["as_" + k.lower() for k in RUS_MAP])
ext = pd.DataFrame(res, columns=cols)
as_cols = ["as_" + k.lower() for k in RUS_MAP]
ext["as_min_ru"] = ext[as_cols].min(axis=1)
ext["bench_min_ru"] = (ext[as_cols].idxmin(axis=1)
                       .str.replace("as_", "").str.upper())
ext.to_parquet(P.CACHE / "nport_as_extension.parquet", index=False)
log.append(f"\nAS computed: {len(ext):,} filings "
           f"({ext['series_id'].nunique():,} series); excluded for "
           f"id_share<80%: {n_dropped_id:,} filings")
log.append("quarterly median as_min_ru and share below 0.70:")
ext["q"] = pd.PeriodIndex(ext["period"], freq="M").asfreq("Q")
for qq, g in ext.groupby("q"):
    log.append(f"    {qq}: median {g['as_min_ru'].median():.3f}, "
               f"<0.70 {(g['as_min_ru'] < 0.70).mean():5.1%}, "
               f"n {len(g):,}")
log.append("bench_min_ru distribution: "
           + str(ext["bench_min_ru"].value_counts().head(8).to_dict()))

# ---- splice validation vs the paper's AS panel (2023 Q3 boundary) -------
link = pd.read_csv(SRC / "nport" / "derived" / "series_crsp_link_v2.csv",
                   low_memory=False)
lw = (link[link["wficn"].notna() & ~link["ambiguous"]]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
ext = ext.merge(lw, on="series_id", how="left")
bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
bp["month"] = pd.to_datetime(bp["month"])
last = bp[bp["month"] == bp["month"].max()].copy()
ascols_bp = [c for c in last.columns if c.startswith("as_")]
last["as_min_nd"] = last[ascols_bp].min(axis=1)
last = last[["wficn", "as_min_nd", "bench_min"]].dropna()
first_q = ext[ext["q"] == ext["q"].min()]
val = first_q.merge(last, on="wficn", how="inner")
ru_set = set(RUS_MAP)
valr = val[val["bench_min"].isin(ru_set)]
log.append(f"\nsplice validation (our first quarter vs paper panel's "
           f"last month {bp['month'].max().date()}):")
log.append(f"  matched funds: {len(val):,}; Russell-benchmarked subset: "
           f"{len(valr):,}")
for lab, d in (("all matched", val), ("Russell-benchmarked", valr)):
    if len(d) > 10:
        corr = d["as_min_ru"].corr(d["as_min_nd"])
        diff = (d["as_min_ru"] - d["as_min_nd"])
        log.append(f"  {lab}: corr {corr:.3f}, median diff "
                   f"{diff.median():+.3f}, |diff|>0.10 share "
                   f"{(diff.abs() > 0.10).mean():.1%}")
log.append("  reading: corr >0.85 and small median diff on the "
           "Russell-benchmarked subset = splice is publication-viable; "
           "a positive median diff is expected (v1 lacks S&P benchmarks, "
           "so our min is over a smaller set).")

log.append("\nSTAGE 33e DONE - aggregates only. Next: 33e2 adds S5 from "
           "sp500 constituents; then the extension merges into the panel "
           "for the final-build sample to 2026.")
P.write_report("nport_33e_extension.txt", log)
print("\n".join(log))
