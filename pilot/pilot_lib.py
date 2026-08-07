"""Shared config + loaders for the capitulation-study feasibility pilot.

Everything here is defensive: column names are normalized to lowercase, date
formats are auto-detected, and Active Share units (0-1 vs 0-100) are inferred.
All outputs written by the pilot are AGGREGATES ONLY - never fund-level rows.

v2: Petajisto loader now prefers activeshare.sas7bdat (the CSV has unquoted
commas inside fund names); comma-tolerant CSV fallback included. parse_ym no
longer uses the deprecated PeriodIndex-from-fields constructor.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd

# ---------------------------------------------------------------- paths ----
# Scripts live in the Capitulation repo (E:\Finance\Capitulation\pilot); raw data
# lives in the shared cross-project library. Override per-machine with DATA_LIB
# (same env var Buy Risk's data-paths.mjs uses).
HERE = Path(__file__).resolve().parent          # ...\Capitulation\pilot
SOURCES = Path(os.environ.get("DATA_LIB", r"E:\Finance\data\sources"))
CRSP_DIR = SOURCES / "crsp_mf"
ND_DIR = SOURCES / "activeshare_nd"
PET_DIR = SOURCES / "petajisto" / "active share"
CPZ_DIR = SOURCES / "petajisto" / "INDEX-BASED FACTOR RETURNS FOR PERFORMANCE EVALUATION"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

F_MONTHLY = CRSP_DIR / "Monthly Returns.csv"
F_SUMMARY = CRSP_DIR / "Fund Summary.csv"
F_MAP = CRSP_DIR / "Fund-Portfolio Map.csv"
F_ND_TR = next(ND_DIR.glob("TR Active Share*.csv"), None)
F_ND_CRSP = next(ND_DIR.glob("CRSP Active Share*.csv"), None)
F_PET = PET_DIR / "activeshare.csv"
F_PET_SAS = PET_DIR / "activeshare.sas7bdat"
F_CPZ_M = CPZ_DIR / "factorret_cpz_monthly.csv"

# Optional (Buy Risk repo, committed provider) - recession shading; skipped if absent.
F_USREC = Path(r"E:\Finance\BuyRisk\data\sources\fred\USREC.csv")

# Thresholds (Petajisto 2013 conventions)
CLOSET_CUTOFF = 0.60     # closet-index territory
ACTIVE_START = 0.70      # a fund must start here to be "genuinely active"


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _month_end(year: pd.Series, month: pd.Series) -> pd.Series:
    dt = pd.to_datetime({"year": year.astype(int),
                         "month": month.astype(int),
                         "day": 1})
    return dt + MonthEnd(0)


def parse_ym(s: pd.Series) -> pd.Series:
    """Parse ND 'ym' year-month: handles '1979m6', '1979-06', YYYYMM ints,
    and Stata %tm elapsed-month ints. Returns month-end timestamps."""
    s = s.astype(str).str.strip().str.lower()
    sample = s.dropna().iloc[0]
    if re.fullmatch(r"\d{4}m\d{1,2}", sample):
        ext = s.str.extract(r"(\d{4})m(\d{1,2})")
        return _month_end(ext[0], ext[1])
    if re.fullmatch(r"\d{6}", sample):
        return _month_end(s.str[:4], s.str[4:6])
    if re.fullmatch(r"\d{4}-\d{1,2}", sample):
        ext = s.str.extract(r"(\d{4})-(\d{1,2})")
        return _month_end(ext[0], ext[1])
    if re.fullmatch(r"-?\d{1,4}(\.0)?", sample):  # Stata %tm: months since 1960m1
        n = pd.to_numeric(s, errors="coerce").astype("Int64")
        return _month_end(1960 + n // 12, n % 12 + 1)
    raise ValueError(f"Unrecognized ym format, e.g. {sample!r}")


def parse_anydate(s: pd.Series) -> pd.Series:
    """Parse date columns: passes through datetimes; handles ISO, YYYYMMDD,
    and SAS numeric (days since 1960-01-01)."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    sample = str(s.dropna().iloc[0]).strip()
    if re.fullmatch(r"\d{8}", sample):
        return pd.to_datetime(s.astype(str), format="%Y%m%d", errors="coerce")
    if re.fullmatch(r"-?\d{1,6}(\.0+)?", sample):  # SAS days since 1960-01-01
        return pd.to_datetime("1960-01-01") + pd.to_timedelta(
            pd.to_numeric(s, errors="coerce"), unit="D")
    return pd.to_datetime(s, errors="coerce")


def to_unit_interval(x: pd.Series, name: str, log: list) -> pd.Series:
    """Active Share as fraction: divide by 100 if it looks like percent."""
    mx = x.max(skipna=True)
    if pd.notna(mx) and mx > 1.5:
        log.append(f"  note: {name} looked like percent (max={mx:.1f}); divided by 100")
        return x / 100.0
    return x


def as_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("as_")]


def load_nd(path: Path, id_cols: list[str], log: list) -> pd.DataFrame:
    head = norm_cols(pd.read_csv(path, nrows=0))
    ascols = as_columns(head)
    usecols = [c for c in id_cols + ["ym", "total_assets"] if c in head.columns] + ascols
    df = norm_cols(pd.read_csv(path, usecols=lambda c: str(c).strip().lower() in usecols))
    df["month"] = parse_ym(df["ym"])
    for c in as_columns(df):
        df[c] = to_unit_interval(pd.to_numeric(df[c], errors="coerce"), c, log)
    arr = df[as_columns(df)].to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        df["as_min"] = np.nanmin(arr, axis=1)
        idx = np.where(np.all(np.isnan(arr), axis=1), -1,
                       np.nanargmin(np.where(np.isnan(arr), np.inf, arr), axis=1))
    names = np.array([c[3:].upper() for c in as_columns(df)] + ["NA"])
    df["bench_min"] = names[idx]
    log.append(f"  {path.name}: {len(df):,} fund-quarters, "
               f"{df[id_cols[0]].nunique():,} funds, {len(as_columns(df))} benchmarks, "
               f"{df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
    return df


def _read_pet_csv() -> pd.DataFrame:
    """Comma-tolerant fallback: fund_name is the LAST column, so split each
    line at most (ncols-1) times and let extra commas stay in fund_name."""
    with open(F_PET, "r", encoding="latin-1", errors="replace") as fh:
        header = [c.strip().lower() for c in fh.readline().rstrip("\n").split(",")]
        n = len(header)
        rows = [ln.rstrip("\n").split(",", n - 1) for ln in fh if ln.strip()]
    return pd.DataFrame(rows, columns=header)


def load_petajisto(log: list) -> pd.DataFrame:
    df = None
    if F_PET_SAS.exists():
        try:
            df = norm_cols(pd.read_sas(F_PET_SAS, format="sas7bdat",
                                       encoding="latin-1"))
            log.append("  petajisto: loaded from activeshare.sas7bdat")
        except Exception as e:  # noqa: BLE001
            log.append(f"  petajisto: sas7bdat load failed ({e}); using CSV fallback")
    if df is None:
        df = _read_pet_csv()
        log.append("  petajisto: loaded via comma-tolerant CSV parser")
    df["rdate"] = parse_anydate(df["rdate"])
    for c in ("activeshare", "activeshare_min", "trackingerror",
              "trackingerror_min", "tna", "indexfund", "enhanced_index"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("activeshare", "activeshare_min"):
        if c in df.columns:
            df[c] = to_unit_interval(df[c], c, log)
    for c in ("wficn", "fundno", "crsp_fundno"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["quarter"] = df["rdate"].dt.to_period("Q")
    log.append(f"  petajisto: {len(df):,} rows, {df['wficn'].nunique():,} funds, "
               f"{df['rdate'].min():%Y-%m} to {df['rdate'].max():%Y-%m}")
    return df


def load_monthly_returns(log: list) -> pd.DataFrame:
    pq = CACHE / "monthly_returns.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
    else:
        df = pd.read_csv(F_MONTHLY,
                         usecols=lambda c: c.strip().lower() in
                         ("caldt", "crsp_fundno", "mret", "mtna", "mnav"),
                         dtype={"mret": "str"})
        df = norm_cols(df)
        df["caldt"] = pd.to_datetime(df["caldt"], errors="coerce")
        df["mret"] = pd.to_numeric(df["mret"], errors="coerce")  # 'R'/'' -> NaN
        for c in ("mtna", "mnav"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["crsp_fundno"] = pd.to_numeric(df["crsp_fundno"], errors="coerce").astype("Int64")
        df.to_parquet(pq, index=False)
    log.append(f"  monthly returns: {len(df):,} rows, "
               f"{df['crsp_fundno'].nunique():,} share classes")
    return df


def load_cpz_monthly(log: list) -> pd.DataFrame:
    df = norm_cols(pd.read_csv(F_CPZ_M))
    datecol = next((c for c in df.columns if "date" in c or c in ("month", "ym", "caldt")),
                   df.columns[0])
    df["month"] = parse_anydate(df[datecol]).dt.to_period("M").dt.to_timestamp("M")
    for c in df.columns:
        if c not in (datecol, "month"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if df["rf"].abs().max() > 0.2:
        for c in df.columns:
            if c not in (datecol, "month"):
                df[c] = df[c] / 100.0
        log.append("  note: CPZ factors looked like percent; divided by 100")
    df["idx_s5"] = df["s5rf"] + df["rf"]
    df["idx_r2"] = df["r2s5"] + df["idx_s5"]
    df["idx_rm"] = df["rms5"] + df["idx_s5"]
    log.append(f"  cpz factors: {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}")
    return df[["month", "idx_s5", "idx_r2", "idx_rm", "rf"]]


# Pilot approximation: map every benchmark code to a reconstructible core index.
BENCH_TO_CORE = {}
for _code in ("S5", "S5G", "S5V", "R1", "R1G", "R1V", "R3", "R3G", "R3V", "DJ", "W5"):
    BENCH_TO_CORE[_code] = "idx_s5"
for _code in ("RM", "RMG", "RMV", "S4", "S4G", "S4V", "W4"):
    BENCH_TO_CORE[_code] = "idx_rm"
for _code in ("R2", "R2G", "R2V", "S6", "S6G", "S6V"):
    BENCH_TO_CORE[_code] = "idx_r2"


def write_report(name: str, lines: list) -> None:
    p = OUT / name
    p.write_text("\n".join(str(x) for x in lines), encoding="utf-8")
    print(f"\nwrote {p}")


def fail(msg: str) -> None:
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)
