"""Shared panel construction for stages 06+ (imports pilot_lib for paths/parsers).

Builds and caches the full fund-quarter panel (Active Share + TNA-weighted fund
returns + spliced benchmark returns + retail flows) and provides the common
spell extractor. Heavy intermediates cache under pilot/cache/ — delete the cache
folder to force a rebuild. All outputs derived from these are AGGREGATES ONLY.

Requires stages 01 and 04 to have run once (uses their caches: as_panel,
monthly_returns, flags, retail_flags).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P

MFLINK1 = P.SOURCES / "mflinks" / "mflink1.csv"
FRENCH_DIR = Path(r"E:\Finance\BuyRisk\data\sources\french")
F_6PORT = FRENCH_DIR / "6_Portfolios_2x3.csv"
F_FACTORS = FRENCH_DIR / "F-F_Research_Data_Factors.csv"
F_MOM = FRENCH_DIR / "F-F_Momentum_Factor.csv"
SPLICE_END = pd.Period("2011Q1", freq="Q")


def get_mflink1() -> pd.DataFrame:
    m1 = P.norm_cols(pd.read_csv(MFLINK1))
    fcol = next(c for c in m1.columns if "fundno" in c)
    wcol = next(c for c in m1.columns if "wficn" in c)
    return (m1[[fcol, wcol]].dropna().astype("int64")
            .rename(columns={fcol: "crsp_fundno", wcol: "wficn"})
            .drop_duplicates("crsp_fundno"))


def get_fund_monthly(log: list) -> pd.DataFrame:
    """wficn-month TNA-weighted fund returns (cached).
    v2: return hygiene — drop corrupt monthly returns (|ret|>200%) and
    micro-fund months (TNA < $1M), which otherwise poison EW portfolios."""
    pq = P.CACHE / "fund_month_v2.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    m1 = get_mflink1()
    ret = P.load_monthly_returns(log).merge(m1, on="crsp_fundno", how="inner")
    ret = ret.sort_values(["crsp_fundno", "caldt"])
    ret["w"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
    ret["w"] = ret["w"].fillna(ret["mtna"]).clip(lower=0)
    ret = ret.dropna(subset=["mret"])
    ret["wr"] = ret["w"] * ret["mret"]
    fm = (ret.groupby(["wficn", "caldt"])
             .agg(wr=("wr", "sum"), w=("w", "sum"), tna=("mtna", "sum")).reset_index())
    fm["fret"] = np.where(fm["w"] > 0, fm["wr"] / fm["w"], np.nan)
    fm = fm.dropna(subset=["fret"])[["wficn", "caldt", "fret", "tna"]]
    n0 = len(fm)
    fm = fm[(fm["fret"].abs() <= 2.0) & (fm["tna"].fillna(0) >= 1.0)]
    log.append(f"  return hygiene: dropped {n0 - len(fm):,} fund-months "
               f"(|ret|>200% or TNA<$1M)")
    fm.to_parquet(pq, index=False)
    return fm


def get_retail_flows(log: list) -> pd.DataFrame:
    """wficn-quarter retail net flow rate (cached). Requires retail_flags cache
    (stage 05) — rebuilt here if missing."""
    pq = P.CACHE / "retail_flow_q.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    rf_pq = P.CACHE / "retail_flags.parquet"
    if rf_pq.exists():
        rfl = pd.read_parquet(rf_pq)
    else:
        parts, use = [], ["crsp_fundno", "retail_fund", "inst_fund"]
        for chunk in pd.read_csv(P.F_SUMMARY,
                                 usecols=lambda c: c.strip().lower() in use,
                                 chunksize=500_000, low_memory=False,
                                 encoding="latin-1"):
            parts.append(P.norm_cols(chunk))
        rfl = pd.concat(parts, ignore_index=True)
        rfl["is_retail"] = rfl["retail_fund"].astype(str).str.upper().eq("Y")
        rfl = rfl.groupby("crsp_fundno").agg(is_retail=("is_retail", "any")).reset_index()
        rfl.to_parquet(rf_pq, index=False)
    m1 = get_mflink1()
    ret = (P.load_monthly_returns(log)
           .merge(rfl, on="crsp_fundno", how="left")
           .merge(m1, on="crsp_fundno", how="inner"))
    ret = ret[ret["is_retail"] == True]  # noqa: E712
    ret = ret.sort_values(["crsp_fundno", "caldt"])
    ret["tna_lag"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
    ret["flow"] = ret["mtna"] - ret["tna_lag"] * (1 + ret["mret"])
    fm = (ret.dropna(subset=["flow"])
             .groupby(["wficn", "caldt"])
             .agg(flow=("flow", "sum"), tna=("mtna", "sum")).reset_index())
    fm["quarter"] = fm["caldt"].dt.to_period("Q")
    fq = (fm.groupby(["wficn", "quarter"])
            .agg(flow=("flow", "sum"), tna_end=("tna", "last")).reset_index())
    fq = fq.sort_values(["wficn", "quarter"])
    fq["tna_prev"] = fq.groupby("wficn")["tna_end"].shift(1)
    fq["flowq"] = (fq["flow"] / fq["tna_prev"]).where(fq["tna_prev"] > 0)
    fq["flowq"] = fq["flowq"].clip(-1, 1)
    out = fq[["wficn", "quarter", "flowq"]].dropna().copy()
    out["quarter"] = out["quarter"].astype(str)
    out.to_parquet(pq, index=False)
    return out


def get_death(log: list) -> pd.DataFrame:
    """wficn-level death info from Fund Summary (cached): died flag, death
    quarter (last end_dt across share classes), merged flag."""
    pq = P.CACHE / "death.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    parts, use = [], ["crsp_fundno", "end_dt", "dead_flag", "delist_cd", "merge_fundno"]
    for chunk in pd.read_csv(P.F_SUMMARY, usecols=lambda c: c.strip().lower() in use,
                             chunksize=500_000, low_memory=False, encoding="latin-1"):
        parts.append(P.norm_cols(chunk))
    d = pd.concat(parts, ignore_index=True)
    d["end_dt"] = pd.to_datetime(d["end_dt"], errors="coerce")
    d["dead"] = d["dead_flag"].astype(str).str.upper().eq("Y")
    d["merged"] = d["merge_fundno"].notna()
    per_class = (d.sort_values("end_dt")
                   .groupby("crsp_fundno")
                   .agg(end_dt=("end_dt", "max"), dead=("dead", "any"),
                        merged=("merged", "any")).reset_index())
    m1 = get_mflink1()
    w = (per_class.merge(m1, on="crsp_fundno", how="inner")
                  .groupby("wficn")
                  .agg(end_dt=("end_dt", "max"), n_dead=("dead", "sum"),
                       n_cls=("dead", "size"), merged=("merged", "any"))
                  .reset_index())
    w["died"] = w["n_dead"] == w["n_cls"]          # every share class ended dead
    w["death_q"] = w["end_dt"].dt.to_period("Q").astype(str)
    out = w[["wficn", "died", "death_q", "merged"]]
    out.to_parquet(pq, index=False)
    log.append(f"death table: {int(out['died'].sum()):,} of {len(out):,} wficn "
               f"funds fully dead ({int(out['merged'].sum()):,} with merges)")
    return out


def parse_french_first_block(path: Path) -> pd.DataFrame:
    rows, header, started = [], None, False
    for ln in path.read_text(encoding="latin-1", errors="replace").splitlines():
        cells = [c.strip() for c in ln.split(",")]
        if header is None:
            if len(cells) >= 2 and cells[0] == "" and sum(bool(c) for c in cells[1:]) >= 1:
                header = [c if c else f"col{i}" for i, c in enumerate(cells[1:])]
            continue
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            started = True
            rows.append(cells[:len(header) + 1])
        elif started:
            break
    df = pd.DataFrame(rows, columns=["ym"] + header)
    for c in header:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([-99.99, -999], np.nan) / 100
    df["month"] = P.parse_ym(df["ym"])
    return df


def get_factors(log: list) -> pd.DataFrame:
    """Monthly FF3 + momentum + RF from the French library (decimals)."""
    ff = parse_french_first_block(F_FACTORS)
    ff.columns = [str(c).strip().lower().replace("-", "") for c in ff.columns]
    mom = parse_french_first_block(F_MOM)
    mom.columns = [str(c).strip().lower() for c in mom.columns]
    momcol = next(c for c in mom.columns if "mom" in c)
    out = ff.merge(mom[["month", momcol]].rename(columns={momcol: "mom"}),
                   on="month", how="left")
    keep = {"mktrf": "mktrf", "smb": "smb", "hml": "hml", "rf": "rf", "mom": "mom"}
    cols = {c: keep[c] for c in out.columns if c in keep}
    out = out[["month", *cols]].rename(columns=cols)
    log.append(f"factors: {out['month'].min():%Y-%m} to {out['month'].max():%Y-%m}")
    return out


def get_bench_q(log: list) -> pd.DataFrame:
    cpz = P.load_cpz_monthly(log)
    f6 = parse_french_first_block(F_6PORT)
    small = [c for c in f6.columns if "SMALL" in str(c).upper()
             or str(c).upper().startswith("ME1")]
    big = [c for c in f6.columns if "BIG" in str(c).upper()
           or str(c).upper().startswith("ME2")]
    f6["p_s5"], f6["p_r2"] = f6[big].mean(axis=1), f6[small].mean(axis=1)
    f6["p_rm"] = f6[big + small].mean(axis=1)

    def to_q(df, cols):
        d = df.copy()
        d["quarter"] = d["month"].dt.to_period("Q")
        return d.set_index("quarter")[cols].add(1).groupby(level=0).prod().sub(1)

    return pd.concat([
        to_q(cpz, ["idx_s5", "idx_r2", "idx_rm"]).loc[lambda d: d.index <= SPLICE_END],
        to_q(f6, ["p_s5", "p_r2", "p_rm"])
            .rename(columns={"p_s5": "idx_s5", "p_r2": "idx_r2", "p_rm": "idx_rm"})
            .loc[lambda d: d.index > SPLICE_END],
    ]).sort_index().reset_index()


# ------------------- real benchmark series (stage 10+) ---------------------
RUSSELL_DIR = P.SOURCES / "russell"
SP500_DIR = P.SOURCES / "crsp_sp500"
F_IDX_HOLD = RUSSELL_DIR / "idx_holdings_us.csv"
F_SP_VW_M = SP500_DIR / "value weighted monthly.csv"
F_SP_COMP_M = SP500_DIR / "composite monthly.csv"

WT_TO_CODE = {"r3000_wt": "R3", "r3000g_wt": "R3G", "r3000v_wt": "R3V",
              "r1000_wt": "R1", "r1000g_wt": "R1G", "r1000v_wt": "R1V",
              "r2000_wt": "R2", "r2000g_wt": "R2G", "r2000v_wt": "R2V",
              "rmidc_wt": "RM", "rmidcg_wt": "RMG", "rmidcv_wt": "RMV"}

# ND benchmark codes with no actual series on disk -> nearest actual series.
# (S&P 400/600 + S&P style variants await Morningstar Direct.)
BENCH_APPROX = {"S4": "RM", "S4G": "RMG", "S4V": "RMV",
                "S6": "R2", "S6G": "R2G", "S6V": "R2V",
                "S5G": "R1G", "S5V": "R1V"}


def build_bench_series(log: list, force: bool = False) -> pd.DataFrame:
    """Monthly TOTAL returns per benchmark code (long: month, code, ret) —
    Russell reconstructed from the holdings file (sum of weight x security
    mtd return), S&P 500 from CRSP's VW total-return file. Cached; series stay
    in cache/ (vendor-licensed), never in output/."""
    pq = P.CACHE / "bench_series_monthly.parquet"
    if pq.exists() and not force:
        df = pd.read_parquet(pq)
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
        return df
    usecols = ["date", "mtdreturn"] + list(WT_TO_CODE)
    num: dict = {}
    den: dict = {}
    for chunk in pd.read_csv(F_IDX_HOLD,
                             usecols=lambda c: c.strip().lower() in usecols,
                             chunksize=1_000_000, low_memory=False,
                             encoding="latin-1"):
        chunk = P.norm_cols(chunk)
        chunk["month"] = pd.to_datetime(chunk["date"], errors="coerce") \
                           .dt.to_period("M")
        r = pd.to_numeric(chunk["mtdreturn"], errors="coerce")
        for wcol, code in WT_TO_CODE.items():
            if wcol not in chunk.columns:
                continue
            w = pd.to_numeric(chunk[wcol], errors="coerce")
            m = w.notna() & r.notna() & (w > 0) & chunk["month"].notna()
            if not m.any():
                continue
            g = (pd.DataFrame({"month": chunk.loc[m, "month"],
                               "wr": (w * r)[m], "w": w[m]})
                 .groupby("month").sum())
            num[code] = g["wr"] if code not in num else num[code].add(g["wr"], fill_value=0)
            den[code] = g["w"] if code not in den else den[code].add(g["w"], fill_value=0)
    rows = []
    for code in num:
        s = (num[code] / den[code]).dropna()
        rows.append(pd.DataFrame({"month": s.index, "code": code, "ret": s.values}))
    ser = pd.concat(rows, ignore_index=True)
    if ser["ret"].abs().median() > 0.5:      # percent -> decimal
        ser["ret"] = ser["ret"] / 100.0
        log.append("  russell reconstructed returns looked like percent; /100")
    sp = P.norm_cols(pd.read_csv(F_SP_VW_M))
    sp["month"] = pd.to_datetime(sp["mthcaldt"], errors="coerce").dt.to_period("M")
    sp = sp.dropna(subset=["month"])
    rows_sp = pd.DataFrame({"month": sp["month"], "code": "S5",
                            "ret": pd.to_numeric(sp["mthtotret"], errors="coerce")})
    ser = pd.concat([ser, rows_sp.dropna(subset=["ret"])], ignore_index=True)
    out = ser.copy()
    out["month"] = out["month"].astype(str)
    out.to_parquet(pq, index=False)
    for code, g in ser.groupby("code"):
        log.append(f"  bench {code}: {g['month'].min()} to {g['month'].max()} "
                   f"({len(g)} months)")
    return ser


def get_real_bench_q(log: list) -> pd.DataFrame:
    """Quarterly compounded benchmark returns, long format: quarter, bcode, bret."""
    ser = build_bench_series(log)
    ser = ser.copy()
    ser["quarter"] = pd.PeriodIndex(ser["month"], freq="M").asfreq("Q")
    q = (ser.assign(g=lambda d: 1 + d["ret"])
            .groupby(["code", "quarter"])
            .agg(bret=("g", lambda x: x.prod() - 1), nm=("g", "size")).reset_index())
    q = q[q["nm"] == 3].drop(columns="nm").rename(columns={"code": "bcode"})
    return q


def build_panel(log: list, force: bool = False) -> pd.DataFrame:
    """Full fund-quarter panel with as_min, qret, bench_qret, flowq, rel4q.
    v2: per-benchmark ACTUAL index returns (Russell reconstructed + CRSP S&P
    total return), replacing the CPZ/French core-proxy splice.
    v3 cache: rebuilt on hygienic fund returns."""
    pq = P.CACHE / "panel_full_v3.parquet"
    if pq.exists() and not force:
        df = pd.read_parquet(pq)
        df["quarter"] = pd.PeriodIndex(df["quarter"], freq="Q")
        return df
    if not (P.CACHE / "as_panel.parquet").exists():
        P.fail("run 01_build_panel.py first (as_panel cache missing)")
    if not (P.CACHE / "flags.parquet").exists():
        P.fail("run 04_full_km.py first (flags cache missing)")

    fm = get_fund_monthly(log)
    fm["quarter"] = fm["caldt"].dt.to_period("Q")
    fq = (fm.assign(g=lambda d: 1 + d["fret"]).groupby(["wficn", "quarter"])
            .agg(qret=("g", lambda x: x.prod() - 1), nm=("g", "size")).reset_index())
    fq = fq[fq["nm"] == 3].drop(columns="nm")

    fl = pd.read_parquet(P.CACHE / "flags.parquet")
    asp = pd.read_parquet(P.CACHE / "as_panel.parquet").dropna(subset=["wficn"])
    asp["wficn"] = asp["wficn"].astype("int64")
    asp["quarter"] = asp["month"].dt.to_period("Q")
    asp = (asp.sort_values(["wficn", "quarter", "total_assets"])
              .drop_duplicates(["wficn", "quarter"], keep="last")
              .merge(fl, on="wficn", how="left"))
    asp = asp[asp["passive"] != True]  # noqa: E712

    bq = get_real_bench_q(log)
    flows = get_retail_flows(log)
    flows["quarter"] = pd.PeriodIndex(flows["quarter"], freq="Q")

    asp["bcode"] = (asp["bench_min"].astype(str).str.upper()
                    .replace(BENCH_APPROX))
    panel = (asp.merge(fq, on=["wficn", "quarter"], how="inner")
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
    out.to_parquet(pq, index=False)
    log.append(f"panel_full: {len(panel):,} fund-quarters, "
               f"{panel['wficn'].nunique():,} funds (cached)")
    panel["quarter"] = pd.PeriodIndex(panel["quarter"], freq="Q")
    return panel


def extract_spells(panel: pd.DataFrame, client_cut: float | None = -0.10,
                   start_year: int | None = None,
                   require_flow_at_start: bool = False,
                   emit_last_row_entry: bool = True) -> pd.DataFrame:
    """One row per underperformance spell (entry: rel4q<0 and as_min>=70%).
    Columns: wficn, start_q, end_q (str Periods), end_dur, m_dur, c_dur, depth,
    ended_by in {recovered, as_missing, data_end}, plus CALENDAR event stamps
    m_cal_q / c_cal_q (audit fix A1: for spells containing reporting gaps,
    start_q + m_dur lands BEFORE the true crossing quarter, because durations
    count observed rows; consumers that need the calendar quarter of an event
    must use these stamps, never start + dur arithmetic).
    emit_last_row_entry (audit fix A2): a spell entered on a fund's final
    observed quarter is emitted as a right-censored 1-quarter data_end spell
    instead of being silently dropped. Set False to reproduce the pre-audit
    behavior."""
    rows = []
    for wficn, g in panel.groupby("wficn"):
        g = g.reset_index(drop=True)
        in_spell = False
        for i in range(len(g)):
            r = g.loc[i]
            if not in_spell:
                ok = (pd.notna(r["rel4q"]) and r["rel4q"] < 0
                      and pd.notna(r["as_min"]) and r["as_min"] >= P.ACTIVE_START)
                if ok and start_year is not None and r["quarter"].year < start_year:
                    ok = False
                if ok and require_flow_at_start and pd.isna(r["flowq"]):
                    ok = False
                if ok:
                    in_spell, start_i, m_dur, c_dur = True, i, None, None
                    m_cal, c_cal = None, None
                    depth = float(r["rel4q"])
            else:
                if pd.notna(r["rel4q"]):
                    depth = min(depth, float(r["rel4q"]))
                dur = i - start_i
                if m_dur is None and pd.notna(r["as_min"]) and r["as_min"] < P.CLOSET_CUTOFF:
                    m_dur, m_cal = dur, str(r["quarter"])
                if (client_cut is not None and c_dur is None
                        and pd.notna(r["flowq"]) and r["flowq"] <= client_cut):
                    c_dur, c_cal = dur, str(r["quarter"])
                ended_by = None
                if pd.isna(r["as_min"]):
                    ended_by = "as_missing"
                elif pd.notna(r["rel4q"]) and r["rel4q"] >= 0:
                    ended_by = "recovered"
                elif i == len(g) - 1:
                    ended_by = "data_end"
                if ended_by:
                    rows.append((wficn, str(g.loc[start_i, "quarter"]),
                                 str(r["quarter"]), max(dur, 1), m_dur, c_dur,
                                 depth, ended_by, m_cal, c_cal))
                    in_spell = False
        if in_spell and emit_last_row_entry:
            # entry occurred on the fund's last observed row: right-censored
            # 1-quarter spell at the data edge (audit fix A2)
            rows.append((wficn, str(g.loc[start_i, "quarter"]),
                         str(g.loc[start_i, "quarter"]), 1, None, None,
                         depth, "data_end", None, None))
    return pd.DataFrame(rows, columns=["wficn", "start_q", "end_q", "end_dur",
                                       "m_dur", "c_dur", "depth", "ended_by",
                                       "m_cal_q", "c_cal_q"])


def ols(y: np.ndarray, X: np.ndarray):
    """Plain OLS with intercept; returns beta, se, t (classic SEs — pilot-grade;
    real build uses Newey-West)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    dof = max(len(y) - X1.shape[1], 1)
    s2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    return beta, se, beta / se
