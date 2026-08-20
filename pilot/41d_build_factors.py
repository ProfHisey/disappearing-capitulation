# 41d_build_factors.py -- build a monthly CAPM factor file from the CRSP
# index data already on disk, so 41b can run without the Fama-French files.
#   market = CRSP NYSE/NYSEMKT/Nasdaq/Arca Value-Weighted Market Index (1000200)
#   rf     = CRSP 30-Day Bill Returns (1000708)
import os, sys, pandas as pd, numpy as np

DATA_LIB = os.environ.get("DATA_LIB", r"E:\Finance\data\sources")
IDX = os.path.join(DATA_LIB, "crsp_indexes", "monthly_indexes.csv")
TSY = os.path.join(DATA_LIB, "crsp_indexes", "treasury_inflation.csv")
OUTF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache",
                    "factors_monthly.csv")
MKT_PRIMARY, MKT_FALLBACK, RF_INDNO = 1000200, 1000080, 1000708

for p in (IDX, TSY):
    if not os.path.exists(p): sys.exit(f"MISSING: {p}")

idx = pd.read_csv(IDX, encoding="latin-1", low_memory=False,
                  usecols=["MthCalDt", "INDNO", "_NAME_", "COL1", "IndNm"])
idx = idx[idx["_NAME_"] == "MthTotRet"]
use = MKT_PRIMARY if (idx.INDNO == MKT_PRIMARY).any() else MKT_FALLBACK
mkt = idx[idx.INDNO == use][["MthCalDt", "COL1", "IndNm"]].copy()
print(f"market index {use}: {mkt.IndNm.iat[0]}  ({len(mkt):,} months)")
mkt = mkt.rename(columns={"COL1": "mkt"})[["MthCalDt", "mkt"]]

tsy = pd.read_csv(TSY, encoding="latin-1")
rf = tsy[tsy.INDNO == RF_INDNO][["MthCalDt", "MthTotRet"]].rename(
    columns={"MthTotRet": "rf"})
print(f"risk-free 1000708: CRSP 30-Day Bill Returns  ({len(rf):,} months)")

for d in (mkt, rf):
    d["m"] = pd.PeriodIndex(pd.to_datetime(d.MthCalDt), freq="M")
f = mkt.merge(rf, on="m", how="inner").sort_values("m")
f["mktrf"] = f["mkt"] - f["rf"]
out = pd.DataFrame({"date": f.m.dt.to_timestamp("M").dt.strftime("%Y-%m-%d"),
                    "mktrf": f.mktrf.values, "rf": f.rf.values}).dropna()
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
out.to_csv(OUTF, index=False)

print(f"\nwrote {OUTF}: {len(out):,} months, {out.date.min()} to {out.date.max()}")
print(f"  mean mktrf {out.mktrf.mean()*12*100:6.2f}%/yr   "
      f"sd {out.mktrf.std()*np.sqrt(12)*100:5.2f}%   "
      f"mean rf {out.rf.mean()*12*100:4.2f}%/yr")
print("\nSanity check those three numbers before using this. The equity")
print("premium should land near 6-8%/yr and market vol near 15-20%.")
print("\nNow run 41b with:")
print(f'  set FACTOR_CSV={OUTF}')
print("  python 41b_nanigian_replication.py")
print("\nNOTE: this is CAPM only - no SMB/HML/RMW/CMA/UMD. Sheng, Simutin &")
print("Zhang (RAPS 2023) show the high-fee 'outperformance' is a profitability")
print("and investment tilt, which needs RMW and CMA to test. If the Fama-French")
print("monthly factor file is in the Buy Risk library, copy it into")
print(f"  {os.path.join(DATA_LIB, 'french')}")
print("and 41b will find it on its own and run the multi-factor arm.")
