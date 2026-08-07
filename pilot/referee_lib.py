"""Shared helpers for the referee-preemption battery (stages 17-20).

Small, deliberately boring functions reused across the four battery scripts:
outcome classification with a parameterized death window, the standard era
table, the spell-quarter frame builder (with a parameterized covariate lag),
the slim discrete-time hazard fit, and a section runner that logs a failure
and keeps going instead of killing the whole script.
"""
from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P

ERAS = [(1980, 1994), (1995, 2009), (2010, 2023)]
SLIM = ["dur_5p", "depth", "era_1023"]


def section(log: list, title: str, fn) -> None:
    """Run one battery section; on failure, log the error and continue."""
    log.append("\n" + title)
    print(title)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()
        log.append(f"  SECTION FAILED: {type(e).__name__}: {e}")
        log.append("  " + tb[-2].strip() if len(tb) > 1 else "")


def attach_death(sp: pd.DataFrame, death: pd.DataFrame,
                 window: int = 4) -> pd.DataFrame:
    """Add start_p/end_p, capitulated, spell_died (death within `window`
    quarters of spell end, no capitulation). Same rule as stages 07/13-16,
    with the window parameterized for the critique-11 sensitivity."""
    sp = sp.merge(death[["wficn", "died", "death_q"]], on="wficn", how="left")
    sp["start_p"] = pd.PeriodIndex(sp["start_q"], freq="Q")
    sp["end_p"] = pd.PeriodIndex(sp["end_q"], freq="Q")
    dp = pd.PeriodIndex(sp["death_q"].where(
        sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
    gap = (dp - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))
    sp["capitulated"] = sp["m_dur"].notna()
    sp["spell_died"] = (sp["ended_by"].isin(["data_end", "as_missing"])
                        & sp["died"].fillna(False)
                        & gap.between(-1, window)
                        & ~sp["capitulated"])
    return sp


def summarize(sp: pd.DataFrame, log: list, label: str) -> None:
    """Counts + outcome shares overall and by era (needs attach_death cols)."""
    log.append(f"\n  {label}: {len(sp):,} spells | capitulated "
               f"{sp['capitulated'].mean():.2%} | died {sp['spell_died'].mean():.2%}")
    for lo, hi in ERAS:
        s = sp[sp["start_p"].dt.year.between(lo, hi)]
        if not len(s):
            continue
        log.append(f"    {lo}-{hi}: n {len(s):6,} | cap {s['capitulated'].mean():6.2%}"
                   f" | died {s['spell_died'].mean():6.2%}")


def build_dt(sp: pd.DataFrame, pf: dict, lag: int = 1) -> pd.DataFrame:
    """Spell-quarter at-risk frame. Covariates (depth, flow) read from the
    quarter `lag` quarters before the at-risk quarter (lag=1 reproduces the
    stage 14-16 convention; lag=2 is the critique-20 double-lag check)."""
    rows = []
    for _, s in sp.iterrows():
        w = s["wficn"]
        g = pf.get(w)
        if g is None:
            continue
        T = int(s["m_dur"]) if s["capitulated"] else int(s["end_dur"])
        T = max(T, 1)
        start = s["start_p"]
        dsf = 0.0
        for t in range(1, T + 1):
            q = start + (t - lag)
            rl = g.at[q, "rel4q"] if q in g.index else np.nan
            if pd.notna(rl):
                dsf = min(dsf, float(rl))
            fl = (g.at[q, "flowq"]
                  if q in g.index and "flowq" in g.columns else np.nan)
            rows.append({
                "wficn": w, "spell_id": s.name, "t": t, "depth": dsf,
                "flow_lag": fl, "q_info": q, "yr": (start + t).year,
                "event": int(s["capitulated"] and t == int(s["m_dur"])),
                "event_die": int(bool(s["spell_died"]) and t == T),
            })
    dt = pd.DataFrame(rows)
    if not len(dt):
        return dt
    dt["era_1023"] = (dt["yr"] >= 2010).astype(float)
    dt["dur_5p"] = (dt["t"] >= 5).astype(float)
    dt["dur_3_4"] = dt["t"].between(3, 4).astype(float)
    dt["dur_5_8"] = dt["t"].between(5, 8).astype(float)
    dt["dur_9_12"] = dt["t"].between(9, 12).astype(float)
    dt["dur_13p"] = (dt["t"] >= 13).astype(float)
    return dt


def slim_fit(df: pd.DataFrame, xcols: list, ycol: str, log: list,
             label: str):
    """Cloglog discrete-time hazard, fund-clustered, one log line of HRs."""
    d = df[[ycol, "wficn"] + xcols].dropna()
    y = d[ycol].to_numpy(float)
    if len(d) < 50 or y.sum() < 5:
        log.append(f"    {label}: too few obs/events "
                   f"(n={len(d)}, ev={int(y.sum())}) - skipped")
        return None
    X = sm.add_constant(d[xcols].to_numpy(float))
    try:
        m = sm.GLM(y, X, family=sm.families.Binomial(
            link=sm.families.links.CLogLog())).fit(
            cov_type="cluster", cov_kwds={"groups": d["wficn"].to_numpy()})
    except Exception as e:  # noqa: BLE001
        log.append(f"    {label}: FIT FAILED ({e})")
        return None
    msg = "  ".join(f"{n} HR {np.exp(b):.2f} (z {z:+.1f})"
                    for n, b, z in zip(["const"] + xcols, m.params, m.tvalues)
                    if n != "const" and not n.startswith("c_"))
    log.append(f"    {label} [n={len(d):,}, ev={int(y.sum()):,}]  {msg}")
    return m


def retrail(pan: pd.DataFrame, retcol: str = "qret",
            window: int = 4) -> pd.DataFrame:
    """Recompute rel4q from `retcol` vs bench_qret with a rolling `window`
    (same groupby-apply pattern as panel_lib.build_panel)."""
    def f(g):
        g = g.set_index("quarter").asfreq("Q")
        fr = (1 + g[retcol]).rolling(window).apply(np.prod, raw=True) - 1
        br = (1 + g["bench_qret"]).rolling(window).apply(np.prod, raw=True) - 1
        g["rel4q"] = fr - br
        return g.reset_index()

    cols = [c for c in pan.columns if c != "wficn"]
    return (pan.sort_values(["wficn", "quarter"])
               .groupby("wficn", group_keys=True)[cols]
               .apply(f).reset_index(level=0).reset_index(drop=True))


def load_exp_ratio(panel: pd.DataFrame, log: list) -> pd.DataFrame:
    """Panel + exp_ratio column (annual, decimal), ffilled within fund; rows
    with implausible values (<0 or >5%) treated as missing."""
    cov = pd.read_parquet(P.CACHE / "covars.parquet")
    cov["quarter"] = pd.PeriodIndex(cov["quarter"], freq="Q")
    cov["exp_ratio"] = pd.to_numeric(cov["exp_ratio"], errors="coerce")
    cov.loc[~cov["exp_ratio"].between(0, 0.05), "exp_ratio"] = np.nan
    pan = panel.merge(cov[["wficn", "quarter", "exp_ratio"]],
                      on=["wficn", "quarter"], how="left")
    pan = pan.sort_values(["wficn", "quarter"])
    pan["exp_ratio"] = pan.groupby("wficn")["exp_ratio"].transform(
        lambda s: s.ffill().bfill())
    miss = pan["exp_ratio"].isna().mean()
    med = float(pan["exp_ratio"].median())
    pan["exp_ratio"] = pan["exp_ratio"].fillna(med)
    log.append(f"  expense ratio: {miss:.1%} of fund-quarters missing entirely"
               f" -> filled with panel median {med:.2%}")
    return pan
