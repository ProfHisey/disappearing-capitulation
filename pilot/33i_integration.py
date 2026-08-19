"""Stage 33i: FULL PANEL INTEGRATION - the final-build sample to 2026.

Merges the N-PORT AS extension into the paper's panel under the paper's
EXACT conventions (build_panel + extract_spells, read this time, not
remembered - audit C1), with the round-4 M-fixes:
 - M4: drop series mapping to >1 wficn; per (wficn, quarter) keep the
   filing with the LARGEST net_assets (mirrors the panel's share-class
   dedup), not the mean;
 - M3: fund-level by construction;
 - uses the 33e2 v2 cache (S&P-augmented as_min) if present, else v1
   with a loud banner;
 - v1 scope: CONTINUATION ONLY - extension rows are added for funds
   already in the ND panel (new post-2023 entrants deferred; avoids
   unflagged passive funds entering the universe).

Then: extract_spells on the extended panel; final-build era table with
2024-26 as its own row; the paper-definition event count and rate for
the extension era; spell-continuation accounting at the 2023Q3 seam.

Output: cache\\panel_full_ext_v1.parquet + output/nport_33i_integration.txt
HEAVY (rebuilds panel components + spell extraction) - run alone.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["FULL PANEL INTEGRATION TO 2026 (stage 33i)", "=" * 60]

# ---- extension fund-quarter rows (M-fixed) ------------------------------
v2 = P.CACHE / "nport_as_extension_v2.parquet"
if v2.exists():
    ext = pd.read_parquet(v2)
    ext["as_use"] = ext["as_min_v2"].fillna(ext["as_min_ru"])
    ext["bench_use"] = ext["bench_min_v2"].fillna(ext["bench_min_ru"])
    log.append("using 33e2 v2 cache (S&P-augmented as_min)")
else:
    ext = pd.read_parquet(P.CACHE / "nport_as_extension.parquet")
    ext["as_use"] = ext["as_min_ru"]
    ext["bench_use"] = ext["bench_min_ru"]
    log.append("*** v1 cache only (Russell-only min) - run 33e2 first "
               "for the final build; proceeding with banner ***")

link = pd.read_csv(SRC / "nport" / "derived" / "series_crsp_link_v2.csv",
                   low_memory=False)
lw = link[link["wficn"].notna() & ~link["ambiguous"]].copy()
multi = lw.groupby("series_id")["wficn"].nunique()
bad_series = set(multi[multi > 1].index)                     # M4 fix
lw = (lw[~lw["series_id"].isin(bad_series)]
      [["series_id", "wficn"]].drop_duplicates("series_id"))
log.append(f"M4: series mapping to >1 wficn dropped: {len(bad_series):,}")
ext = ext.merge(lw, on="series_id", how="inner")
ext["wficn"] = ext["wficn"].astype("int64")

meta = pd.read_parquet(P.CACHE / "nport_holdings_parts"
                       / "_filings_meta.parquet")
ext = ext.merge(meta[["accession", "net_assets"]], on="accession",
                how="left")
ext["quarter"] = pd.PeriodIndex(ext["period"], freq="M").asfreq("Q")
ext = (ext.sort_values("net_assets")
          .drop_duplicates(["wficn", "quarter"], keep="last"))   # M4 fix
log.append(f"extension fund-quarters after largest-series dedup: "
           f"{len(ext):,} ({ext['wficn'].nunique():,} funds)")

# ---- splice onto the ND as_panel (continuation-only) --------------------
asp = pd.read_parquet(P.CACHE / "as_panel.parquet").dropna(
    subset=["wficn"])
asp["wficn"] = asp["wficn"].astype("int64")
asp["quarter"] = asp["month"].dt.to_period("Q")
asp = (asp.sort_values(["wficn", "quarter", "total_assets"])
          .drop_duplicates(["wficn", "quarter"], keep="last"))
nd_funds = set(asp["wficn"].unique())
B = pd.Period("2023Q3", freq="Q")
ext_rows = ext[(ext["quarter"] > B)
               & ext["wficn"].isin(nd_funds)].copy()
log.append(f"continuation rows added (funds already in ND panel): "
           f"{len(ext_rows):,} ({ext_rows['wficn'].nunique():,} funds); "
           f"extension-only new funds deferred: "
           f"{ext.loc[~ext['wficn'].isin(nd_funds), 'wficn'].nunique():,}")

add = pd.DataFrame({
    "wficn": ext_rows["wficn"],
    "month": ext_rows["quarter"].dt.asfreq("M", how="end")
             .dt.to_timestamp(),
    "total_assets": ext_rows["net_assets"],
    "bench_min": ext_rows["bench_use"],
    "as_min": ext_rows["as_use"],
})
asp_ext = pd.concat([asp[["wficn", "month", "total_assets", "bench_min",
                          "as_min", "quarter"]],
                     add.assign(quarter=lambda d: d["month"]
                                .dt.to_period("Q"))],
                    ignore_index=True)

# ---- replicate build_panel body on the extended AS panel ----------------
fl = pd.read_parquet(P.CACHE / "flags.parquet")
asp_ext = asp_ext.merge(fl, on="wficn", how="left")
asp_ext = asp_ext[asp_ext["passive"] != True]  # noqa: E712

fm = PL.get_fund_monthly(log)
fm["quarter"] = fm["caldt"].dt.to_period("Q")
fq = (fm.assign(g=lambda d: 1 + d["fret"]).groupby(["wficn", "quarter"])
        .agg(qret=("g", lambda x: x.prod() - 1),
             nm=("g", "size")).reset_index())
fq = fq[fq["nm"] == 3].drop(columns="nm")
log.append(f"fund quarterly returns reach {fq['quarter'].max()}")

bq = PL.get_real_bench_q(log)
log.append(f"benchmark returns reach {bq['quarter'].max()}")
flows = PL.get_retail_flows(log)
flows["quarter"] = pd.PeriodIndex(flows["quarter"], freq="Q")

asp_ext["bcode"] = (asp_ext["bench_min"].astype(str).str.upper()
                    .replace(PL.BENCH_APPROX))
panel = (asp_ext.merge(fq, on=["wficn", "quarter"], how="inner")
                .merge(bq, on=["quarter", "bcode"], how="left")
                .merge(flows, on=["wficn", "quarter"], how="left"))
panel["bench_qret"] = panel["bret"]
panel = panel.dropna(subset=["as_min", "qret", "bench_qret"])
panel = panel.sort_values(["wficn", "quarter"])

def add_trailing(g):
    g = g.set_index("quarter").asfreq("Q")
    f = (1 + g["qret"]).rolling(4).apply(np.prod, raw=True) - 1
    b = (1 + g["bench_qret"]).rolling(4).apply(np.prod, raw=True) - 1
    g["rel4q"] = f - b
    return g.reset_index()

panel = (panel.groupby("wficn", group_keys=True)[
             ["quarter", "as_min", "qret", "bench_qret", "flowq"]]
         .apply(add_trailing).reset_index(level=0).reset_index(drop=True))
out = panel.copy()
out["quarter"] = out["quarter"].astype(str)
out.to_parquet(P.CACHE / "panel_full_ext_v1.parquet", index=False)
panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
ext_fq = panel[panel["quarter"] > B]
log.append(f"extended panel: {len(panel):,} fund-quarters "
           f"({len(ext_fq):,} in the extension era, "
           f"{ext_fq['wficn'].nunique():,} funds)")

# ---- spells on the extended panel ---------------------------------------
sp = PL.extract_spells(panel, client_cut=None)
sp["m_cal"] = pd.PeriodIndex(sp["m_cal_q"].where(sp["m_cal_q"].notna()),
                             freq="Q")
caps = sp[sp["m_dur"].notna()]
log.append(f"\nspells: {len(sp):,}; capitulation events (paper "
           f"definition, <{P.CLOSET_CUTOFF} in-spell): {len(caps):,}")
era = pd.cut(caps["m_cal"].dt.year,
             [0, 1994, 2009, 2023, 9999],
             labels=["1980-94", "1995-2009", "2010-23", "2024-26"])
log.append("capitulation events by calendar era (FINAL-BUILD TABLE):")
for e, n in era.value_counts().sort_index().items():
    log.append(f"    {e}: {n}")
# at-risk in-spell fund-quarters by era for rates
risk = {k: 0 for k in ["1980-94", "1995-2009", "2010-23", "2024-26"]}
def era_of(y):
    return ("1980-94" if y <= 1994 else "1995-2009" if y <= 2009
            else "2010-23" if y <= 2023 else "2024-26")
for _, s in sp.iterrows():
    q0 = pd.Period(s["start_q"], freq="Q")
    end = int(s["m_dur"]) if pd.notna(s["m_dur"]) else int(s["end_dur"])
    for k in range(1, end + 1):
        risk[era_of((q0 + k).year)] += 1
log.append("event rate per 100 at-risk spell-quarters, by era:")
for e in ["1980-94", "1995-2009", "2010-23", "2024-26"]:
    n = int((era == e).sum())
    if risk[e]:
        log.append(f"    {e}: {100 * n / risk[e]:.2f}  "
                   f"(events {n}, at-risk {risk[e]:,})")
seam = sp[(pd.PeriodIndex(sp["start_q"], freq="Q") <= B)
          & (pd.PeriodIndex(sp["end_q"], freq="Q") > B)]
log.append(f"spells continuing across the 2023Q3 seam: {len(seam):,} "
           f"(previously censored at data_end)")

log.append("\nreading: the 2024-26 row IS the final-build result. If its "
           "rate sits at or below 2010-23's, the paper's ending becomes "
           "'through mid-2026, at the lowest rates on record' - quote "
           "ONLY after this run's numbers replace the 33h preview in the "
           "ledger. Known caveats carried: Russell-only min if 33e2 "
           "wasn't run; continuation-only universe; M5 renormalization "
           "variant still to be added as a robustness column.")
log.append("\nSTAGE 33i DONE - aggregates only.")
P.write_report("nport_33i_integration.txt", log)
print("\n".join(log))
