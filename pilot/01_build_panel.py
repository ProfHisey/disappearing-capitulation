"""Stage 1: build the Active Share panel and the CRSP return join.

Outputs (all local; cache/ holds intermediate parquet, output/ holds aggregates):
  cache/as_panel.parquet        fund-quarter Active Share (ND TR + ND CRSP union)
  cache/pet_panel.parquet       Petajisto 1980-2009 with quarterly fund returns
  output/build_report.txt       counts + match rates (AGGREGATES ONLY - shareable)
"""
import numpy as np
import pandas as pd

import pilot_lib as P

log = ["PANEL BUILD REPORT", "=" * 60]

# ---------------------------------------------------- ND Active Share union
nd_tr = P.load_nd(P.F_ND_TR, ["fundno", "wficn"], log)
nd_cr = P.load_nd(P.F_ND_CRSP, ["crsp_portno", "wficn"], log)
nd_tr["source"] = "TR"
nd_cr["source"] = "CRSP"
keep = ["wficn", "month", "as_min", "bench_min", "total_assets", "source"]
as_panel = pd.concat([nd_tr[[c for c in keep if c in nd_tr]],
                      nd_cr[[c for c in keep if c in nd_cr]]], ignore_index=True)
as_panel = as_panel.dropna(subset=["as_min"])
as_panel.to_parquet(P.CACHE / "as_panel.parquet", index=False)
log.append(f"\nND union: {len(as_panel):,} fund-quarters, "
           f"{as_panel['wficn'].nunique():,} wficn funds, "
           f"{as_panel['month'].min():%Y-%m} to {as_panel['month'].max():%Y-%m}")

# ---------------------------------------------------- Petajisto + CRSP join
pet = P.load_petajisto(log)
ret = P.load_monthly_returns(log)

ret["quarter"] = ret["caldt"].dt.to_period("Q")
qret = (ret.dropna(subset=["mret"])
           .assign(gross=lambda d: 1 + d["mret"])
           .groupby(["crsp_fundno", "quarter"])
           .agg(qret=("gross", lambda g: g.prod() - 1), nmonths=("gross", "size"))
           .reset_index())
qret = qret[qret["nmonths"] == 3]

pet["crsp_fundno"] = pd.to_numeric(pet["crsp_fundno"], errors="coerce").astype("Int64")
merged = pet.merge(qret, on=["crsp_fundno", "quarter"], how="left")
match = merged["qret"].notna().mean()
log.append(f"\nPetajisto x CRSP quarterly-return match rate: {match:.1%} "
           f"({merged['qret'].notna().sum():,} of {len(merged):,} fund-quarters)")
if match < 0.5:
    log.append("  WARNING: match rate under 50% - check crsp_fundno parsing "
               "before trusting 03_km_pilot output.")

# benchmark quarterly returns from CPZ core indexes
cpz = P.load_cpz_monthly(log)
cpz["quarter"] = cpz["month"].dt.to_period("Q")
bq = (cpz.set_index("quarter")[["idx_s5", "idx_r2", "idx_rm"]]
         .add(1).groupby(level=0).prod().sub(1).reset_index())

merged["core"] = merged["index"].astype(str).str.strip().str.upper().map(P.BENCH_TO_CORE)
unmapped = merged.loc[merged["core"].isna(), "index"].value_counts()
if len(unmapped):
    log.append(f"\nBenchmark codes not mapped to a core index (dropped in KM): "
               f"{dict(unmapped.head(10))}")
merged = merged.merge(bq, on="quarter", how="left")
merged["bench_qret"] = np.select(
    [merged["core"] == "idx_s5", merged["core"] == "idx_r2", merged["core"] == "idx_rm"],
    [merged["idx_s5"], merged["idx_r2"], merged["idx_rm"]], default=np.nan)

cols = ["wficn", "quarter", "rdate", "crsp_fundno", "index", "core", "activeshare",
        "activeshare_min", "trackingerror", "indexfund", "enhanced_index", "tna",
        "qret", "bench_qret"]
out = merged[[c for c in cols if c in merged.columns]].copy()
out["quarter"] = out["quarter"].astype(str)
out.to_parquet(P.CACHE / "pet_panel.parquet", index=False)
usable = out.dropna(subset=["activeshare", "qret", "bench_qret"])
log.append(f"\nKM-usable fund-quarters (AS + fund ret + bench ret): {len(usable):,} "
           f"across {usable['wficn'].nunique():,} funds")

log.append("\nBUILD DONE - output/build_report.txt is aggregate-only and shareable.")
P.write_report("build_report.txt", log)
print("\n".join(log))
