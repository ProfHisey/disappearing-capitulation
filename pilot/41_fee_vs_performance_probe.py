r"""
Stage 41 (v2) - Fee vs. past-performance selection rules: rough hypothesis check.
Paper: "Sort by Fees, Not Performance"

Run:  E:
      cd \Finance\Capitulation\pilot
      python 41_fee_vs_performance_probe.py

v2 CHANGE: reuses the parquet caches the capitulation build already produced
(fund_month_v3_tnafix, covars, flags) instead of re-parsing the CRSP CSVs.
Seconds instead of an hour. Falls back to the CSVs only for fund category,
which is the one thing those caches do not carry.

FOUR ANGLES:
  A. FEE DISPERSION  - is there still enough spread to sort on in 2026?
  B. THE NANIGIAN TEST - gradient (title stands) or cliff (title changes to
     "Avoid the Expensive Tenth")?
  C. THE HORSE RACE - top-decile trailing 12m vs cheapest decile vs the
     universe, re-formed annually, chained into a $10k wealth path.
  D. PERSISTENCE - our own version of S&P's Persistence Scorecard.

DATA HYGIENE NOTE, and it matters here more than it did in Paper 1. The
expense ratio was a control variable in the capitulation study; in this paper
it IS the treatment. The cached covars field contains negative values, exact
zeros, and values above 100%. The cleaning rule below is explicit and the
script prints what each screen removes. Any of these choices can move the
answer, so they belong in the paper, not in a footnote.

ROUGH-VERSION LIMITS (fixed in later stages):
  - Portfolios are equal-weighted among SURVIVING members each month; a fund
    that dies has its weight spread across the rest. Stage 43 must report at
    least three reinvestment rules.
  - "Excess" is measured against the equal-weighted universe, not an
    investable index. Stage 42 fixes that.
  - No load adjustment.
"""

import os
import sys
import numpy as np
import pandas as pd

DATA_LIB = os.environ.get("DATA_LIB", r"E:\Finance\data\sources")
CRSP_DIR = os.path.join(DATA_LIB, "crsp_mf")
MFLINK1 = os.path.join(DATA_LIB, "mflinks", "mflink1.csv")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

FUND_MONTH = os.path.join(CACHE, "fund_month_v3_tnafix.parquet")
COVARS = os.path.join(CACHE, "covars.parquet")
FLAGS = os.path.join(CACHE, "flags.parquet")
PANEL_OUT = os.path.join(CACHE, "s41_panel.parquet")

START_YEAR, END_YEAR = 1990, 2025
MIN_EXP, MAX_EXP = 0.0001, 0.10     # 1bp to 1000bp; outside this is bad data
MIN_FUNDS_PER_CAT = 20
USE_CATEGORY = True                  # set False to skip the Fund Summary read


def say(m):
    print(m, flush=True)


def hr(t):
    say("\n" + "=" * 72)
    say(t)
    say("=" * 72)


def need(path, what):
    if not os.path.exists(path):
        say(f"!! MISSING {what}: {path}")
        say("   If the capitulation caches live elsewhere, edit CACHE at the top.")
        sys.exit(1)


# ------------------------------------------------------------------- loading
def load_panel():
    if os.path.exists(PANEL_OUT):
        say("panel: using s41_panel.parquet")
        return pd.read_parquet(PANEL_OUT)

    for p, w in [(FUND_MONTH, "fund-month panel"), (COVARS, "covariates"),
                 (FLAGS, "flags")]:
        need(p, w)

    fm = pd.read_parquet(FUND_MONTH)
    cov = pd.read_parquet(COVARS)
    flg = pd.read_parquet(FLAGS)
    say(f"fund-months {len(fm):,} | covars {len(cov):,} | flags {len(flg):,}")

    # --- universe: actively managed domestic equity
    keep = flg[flg["dom_eq"] & (~flg["passive"])]["wficn"].unique()
    fm = fm[fm["wficn"].isin(keep)].copy()
    say(f"universe: {len(keep):,} active domestic-equity funds "
        f"({fm['wficn'].nunique():,} with return data)")

    # --- fee cleaning, stated out loud
    n0 = len(cov)
    neg = (cov["exp_ratio"] < 0).sum()
    zero = (cov["exp_ratio"] == 0).sum()
    big = (cov["exp_ratio"] > MAX_EXP).sum()
    nan = cov["exp_ratio"].isna().sum()
    cov = cov.copy()
    cov.loc[(cov["exp_ratio"] < MIN_EXP) | (cov["exp_ratio"] > MAX_EXP),
            "exp_ratio"] = np.nan
    say(f"fee cleaning on {n0:,} rows: {nan:,} already missing, "
        f"{neg:,} negative, {zero:,} exactly zero, {big:,} above {MAX_EXP:.0%} "
        f"-> {cov['exp_ratio'].isna().sum():,} missing after cleaning")

    cov["q"] = pd.PeriodIndex(cov["quarter"].astype(str), freq="Q")
    cov = cov.dropna(subset=["exp_ratio"]).sort_values(["wficn", "q"])
    cov["year"] = cov["q"].dt.year
    fee = (cov.groupby(["wficn", "year"], as_index=False)
           .agg(exp_ratio=("exp_ratio", "last")))

    fm["ym"] = pd.PeriodIndex(fm["caldt"], freq="M")
    fm["year"] = fm["ym"].dt.year
    panel = fm.rename(columns={"fret": "ret"})[
        ["wficn", "ym", "year", "ret", "tna"]].dropna(subset=["ret"])
    panel = panel.merge(fee, on=["wficn", "year"], how="left")
    panel["fundgrp"] = panel["wficn"].astype(str)

    cat = load_categories(panel["wficn"].unique()) if USE_CATEGORY else None
    if cat is not None:
        panel = panel.merge(cat, on=["wficn", "year"], how="left")
    else:
        panel["cat"] = "ALL"

    panel = panel[(panel["year"] >= START_YEAR - 1) & (panel["year"] <= END_YEAR + 5)]
    panel.to_parquet(PANEL_OUT, index=False)
    say(f"panel: {panel['wficn'].nunique():,} funds, {len(panel):,} fund-months, "
        f"{panel['ym'].min()} to {panel['ym'].max()}")
    return panel


def load_categories(wficns):
    """crsp_obj_cd -> wficn-year category, via MFLINKS. Optional."""
    fs = os.path.join(CRSP_DIR, "Fund Summary.csv")
    if not (os.path.exists(fs) and os.path.exists(MFLINK1)):
        say("categories: Fund Summary or mflink1 not found - pooled sorts only")
        return None
    try:
        head = pd.read_csv(fs, nrows=3, encoding="latin-1")
        cols = {c.lower(): c for c in head.columns}
        c_fund = cols.get("crsp_fundno")
        c_date = cols.get("caldt") or cols.get("begdt")
        c_obj = cols.get("crsp_obj_cd")
        if not (c_fund and c_date and c_obj):
            say(f"categories: needed columns not in Fund Summary ({list(head.columns)[:20]})")
            return None

        link = pd.read_csv(MFLINK1, encoding="latin-1")
        lcols = {c.lower(): c for c in link.columns}
        link = link.rename(columns={lcols["crsp_fundno"]: "crsp_fundno",
                                    lcols["wficn"]: "wficn"})
        link = link[["crsp_fundno", "wficn"]].dropna().drop_duplicates()

        say("categories: reading Fund Summary (3 columns, a minute or two)")
        parts = []
        for ch in pd.read_csv(fs, usecols=[c_fund, c_date, c_obj],
                              chunksize=2_000_000, low_memory=False,
                              encoding="latin-1", on_bad_lines="skip"):
            ch = ch.rename(columns={c_fund: "crsp_fundno", c_date: "date",
                                    c_obj: "obj"})
            ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
            ch = ch.dropna(subset=["date", "crsp_fundno"])
            ch["year"] = ch["date"].dt.year
            ch = ch[ch["year"] >= START_YEAR - 2]
            parts.append(ch[["crsp_fundno", "year", "obj"]])
        s = pd.concat(parts, ignore_index=True)
        s = s.merge(link, on="crsp_fundno", how="inner")
        s = s[s["wficn"].isin(wficns)]
        s["cat"] = s["obj"].astype(str).str.upper().str[:4]
        cat = (s.sort_values(["wficn", "year"])
               .groupby(["wficn", "year"], as_index=False)
               .agg(cat=("cat", lambda x: x.mode().iat[0] if len(x.mode()) else np.nan)))
        say(f"categories: {cat['cat'].nunique()} codes on "
            f"{cat['wficn'].nunique():,} funds")
        return cat
    except Exception as e:
        say("")
        say("!! " + "*" * 66)
        say(f"!! CATEGORY JOIN FAILED: {type(e).__name__}: {e}")
        say("!! Every exhibit below will run POOLED across all active domestic")
        say("!! equity funds. Pooled fee sorts confound fee with style, because")
        say("!! cheap funds are large-cap heavy and expensive funds are small-cap")
        say("!! and sector heavy. DO NOT interpret a pooled result as a fee")
        say("!! result. Fix the join and re-run before drawing any conclusion.")
        say("!! " + "*" * 66)
        say("")
        return None


# ------------------------------------------------------------------ exhibits
def compound(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    return np.prod(1.0 + x.values) - 1.0 if len(x) else np.nan


def formation_table(panel):
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        past = panel[(panel["ym"] >= pd.Period(f"{year}-01", "M")) &
                     (panel["ym"] <= pd.Period(f"{year}-12", "M"))]
        t = past.groupby("fundgrp").agg(
            trail12=("ret", compound), n=("ret", "size"),
            exp_ratio=("exp_ratio", "last"), cat=("cat", "last"),
            tna=("tna", "last"))
        t = t[t["n"] >= 12]
        for h, lab in [(1, "fwd1"), (3, "fwd3"), (5, "fwd5")]:
            fut = panel[(panel["ym"] >= pd.Period(f"{year + 1}-01", "M")) &
                        (panel["ym"] <= pd.Period(f"{year + h}-12", "M"))]
            t = t.join(fut.groupby("fundgrp")["ret"].apply(compound).rename(lab))
        t["form_year"] = year
        rows.append(t.reset_index())
    ft = pd.concat(rows, ignore_index=True).dropna(subset=["exp_ratio", "trail12"])
    ft["cat"] = ft["cat"].fillna("ALL")
    say(f"formations: {len(ft):,} fund-year observations, "
        f"{ft['fundgrp'].nunique():,} funds")
    return ft


def decile(s, n=10):
    try:
        return pd.qcut(s.rank(method="first"), n, labels=False) + 1
    except ValueError:
        return pd.Series(np.nan, index=s.index)


def exhibit_A(ft):
    hr("A. FEE DISPERSION - is there still room to sort on fees?")
    g = ft.groupby("form_year")["exp_ratio"]
    tab = pd.DataFrame({"n_funds": g.size(), "median_bps": g.median() * 10000,
                        "p10_bps": g.quantile(.10) * 10000,
                        "p90_bps": g.quantile(.90) * 10000})
    tab["spread_bps"] = tab["p90_bps"] - tab["p10_bps"]
    if ft["cat"].nunique() > 1:
        w = (ft.groupby(["form_year", "cat"])["exp_ratio"]
             .agg(lambda x: x.quantile(.9) - x.quantile(.1))
             .groupby("form_year").median() * 10000)
        tab["within_cat_spread_bps"] = w
    tab.round(1).to_csv(os.path.join(OUT, "s41_A_fee_dispersion.csv"))
    say(tab.round(1).to_string())
    f, l = tab.index.min(), tab.index.max()
    col = "within_cat_spread_bps" if "within_cat_spread_bps" in tab else "spread_bps"
    say(f"\nPLAIN READING: p90-p10 fee spread went from "
        f"{tab.loc[f, col]:.0f} bps in {f} to {tab.loc[l, col]:.0f} bps in {l}.")
    say("Small today = the paper's national projection shrinks with it.")
    say("For contrast, the plan menu screenshots show 55-77 bps of spread")
    say("WITHIN category in 2026, so menu-level dispersion is alive even where")
    say("the industry average has compressed.")


def exhibit_B(ft):
    hr("B. THE NANIGIAN TEST - gradient or cliff?")
    ft = ft.copy()
    by = ["form_year", "cat"] if ft["cat"].nunique() > 1 else ["form_year"]
    ft["fee_dec"] = ft.groupby(by)["exp_ratio"].transform(decile)
    ft["excess1"] = ft["fwd1"] - ft.groupby(by)["fwd1"].transform("mean")
    ft = ft.dropna(subset=["fee_dec", "excess1"])
    yearly = ft.groupby(["fee_dec", "form_year"])["excess1"].mean().reset_index()
    rows = []
    for d, g in yearly.groupby("fee_dec"):
        x = g["excess1"].dropna()
        se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
        rows.append({"fee_decile": int(d), "n_years": len(x),
                     "mean_excess_fwd1_pct": x.mean() * 100,
                     "t_stat": x.mean() / se if se and se > 0 else np.nan})
    tab = pd.DataFrame(rows).set_index("fee_decile")
    tab.round(3).to_csv(os.path.join(OUT, "s41_B_fee_decile_gradient.csv"))
    say(tab.round(3).to_string())
    mid = tab.loc[2:9, "mean_excess_fwd1_pct"]
    say(f"\nPLAIN READING: decile 1 {tab.loc[1, 'mean_excess_fwd1_pct']:+.2f}%/yr, "
        f"decile 10 {tab.loc[10, 'mean_excess_fwd1_pct']:+.2f}%, "
        f"deciles 2-9 span {mid.min():+.2f} to {mid.max():+.2f}.")
    say("Flat middle with a bad tenth = Nanigian replicates, title changes.")
    say("Monotone steps = 'Sort by Fees' stands. 41b runs his exact design.")


def portfolio_path(panel, picks, label, hold_years=1):
    """hold_years>1 re-forms every hold_years and holds through. Momentum
    decays after about twelve months; a fee advantage does not."""
    rows, wealth = [], 1.0
    for year, members in sorted(picks.items()):
        if (year - min(picks)) % hold_years != 0:
            continue
        held = panel[(panel["fundgrp"].isin(members)) &
                     (panel["ym"] >= pd.Period(f"{year + 1}-01", "M")) &
                     (panel["ym"] <= pd.Period(f"{year + hold_years}-12", "M"))]
        if held.empty:
            continue
        yr = np.prod(1.0 + held.groupby("ym")["ret"].mean().values) - 1.0
        wealth *= (1.0 + yr)
        rows.append({"rule": label, "hold_years": hold_years,
                     "hold_year": year + hold_years, "n_funds": len(members),
                     "period_return_pct": yr * 100, "wealth_10k": wealth * 10000})
    return pd.DataFrame(rows)


def exhibit_B2(ft):
    """Who lives in decile 1 and decile 10? A U-shape usually means style."""
    hr("B2. WHO IS IN THE EXTREME FEE DECILES?")
    ft = ft.copy()
    by = ["form_year", "cat"] if ft["cat"].nunique() > 1 else ["form_year"]
    ft["fee_dec"] = ft.groupby(by)["exp_ratio"].transform(decile)
    d = ft[ft["fee_dec"].isin([1, 5, 10])]
    tab = d.groupby("fee_dec").agg(
        n_fund_years=("fundgrp", "size"),
        median_fee_bps=("exp_ratio", lambda x: x.median() * 10000),
        median_tna_musd=("tna", "median"),
        mean_trail12_pct=("trail12", lambda x: x.mean() * 100),
        sd_trail12_pct=("trail12", lambda x: x.std() * 100),
        mean_fwd1_pct=("fwd1", lambda x: x.mean() * 100))
    say(tab.round(2).to_string())
    say("\nPLAIN READING: if decile 1 holds funds an order of magnitude larger")
    say("than decile 10 with much lower return dispersion, the fee deciles are")
    say("sorting on SIZE and STYLE, not on cost. That is the pooled-sorting")
    say("problem, and it is why the category join matters.")


def exhibit_C(ft, panel):
    hr("C. THE HORSE RACE - three rules, one wealth path")
    ft = ft.copy()
    by = ["form_year", "cat"] if ft["cat"].nunique() > 1 else ["form_year"]
    ft["perf_dec"] = ft.groupby(by)["trail12"].transform(decile)
    ft["fee_dec"] = ft.groupby(by)["exp_ratio"].transform(decile)
    rules = {"R1_top_decile_trailing_12m": ft["perf_dec"] == 10,
             "R2_cheapest_decile_fee": ft["fee_dec"] == 1,
             "R3_whole_universe": pd.Series(True, index=ft.index)}
    paths, summ = [], []
    for lab, m in rules.items():
        sel = ft[m]
        picks = {y: set(g["fundgrp"]) for y, g in sel.groupby("form_year")}
        for h in (1, 3, 5):
            paths.append(portfolio_path(panel, picks, lab, hold_years=h))
        for h, c in [(1, "fwd1"), (3, "fwd3"), (5, "fwd5")]:
            summ.append({"rule": lab, "horizon_years": h,
                         "mean_net_return_pct": sel[c].mean() * 100,
                         "median_net_return_pct": sel[c].median() * 100,
                         "n_obs": int(sel[c].notna().sum())})
    paths = pd.concat(paths, ignore_index=True)
    summ = pd.DataFrame(summ)
    paths.round(3).to_csv(os.path.join(OUT, "s41_C_wealth_paths.csv"), index=False)
    summ.round(3).to_csv(os.path.join(OUT, "s41_C_rule_returns.csv"), index=False)
    say(summ.round(2).to_string(index=False))
    say("\nTERMINAL WEALTH, $10,000, BY HOLDING PERIOD:")
    fin = paths.sort_values("hold_year").groupby(["rule", "hold_years"]).tail(1)
    for hy, g in fin.groupby("hold_years"):
        say(f"  hold {hy} year(s):")
        for _, r in g.iterrows():
            say(f"    {r['rule']:<30s} ${r['wealth_10k']:>12,.0f}  "
                f"(through {int(r['hold_year'])})")
    say("\n  If R1's edge shrinks as the holding period lengthens, it is")
    say("  momentum decaying, not skill. That is the Carhart (1997) result.")
    say("\nPLAIN READING: the gap between R1 and R2 is the cost of sorting by")
    say("the wrong column. Dead funds' weight is spread across survivors -")
    say("stage 43 must show the answer survives other reinvestment rules.")


def exhibit_D(ft):
    hr("D. PERSISTENCE - does last year's winner stay a winner?")
    ft = ft.copy()
    by = ["form_year", "cat"] if ft["cat"].nunique() > 1 else ["form_year"]
    ft["q"] = ft.groupby(by)["trail12"].transform(lambda s: decile(s, 4))
    nxt = ft[["fundgrp", "form_year", "q"]].copy()
    nxt["form_year"] -= 1
    nxt = nxt.rename(columns={"q": "q_next"})
    j = ft.merge(nxt, on=["fundgrp", "form_year"]).dropna(subset=["q", "q_next"])
    trans = pd.crosstab(j["q"], j["q_next"], normalize="index") * 100
    trans.round(1).to_csv(os.path.join(OUT, "s41_D_transition_matrix.csv"))
    say("Transition matrix, % (row = this year quartile, col = next year):")
    say(trans.round(1).to_string())
    top = ft[ft["q"] == 4][["fundgrp", "form_year"]]
    rows = []
    for k in range(1, 6):
        ok = top.copy()
        for step in range(1, k):
            nx = ft[ft["q"] == 4][["fundgrp", "form_year"]].copy()
            nx["form_year"] -= step
            ok = ok.merge(nx, on=["fundgrp", "form_year"])
        rows.append({"consecutive_years_top_quartile": k, "n": len(ok),
                     "pct_of_starters": 100.0 * len(ok) / max(len(top), 1)})
    sd = pd.DataFrame(rows)
    sd.round(2).to_csv(os.path.join(OUT, "s41_D_persistence.csv"), index=False)
    say("\n" + sd.round(2).to_string(index=False))
    say("\nPLAIN READING: compare the 5-year row with S&P's Persistence")
    say("Scorecard, which reports 0.00% in both the 2024 and 2025 editions.")


def main():
    hr("STAGE 41 v2 - fee vs. past performance, rough hypothesis check")
    say(f"DATA_LIB = {DATA_LIB}")
    panel = load_panel()
    ft = formation_table(panel)
    ft.to_parquet(os.path.join(CACHE, "s41_formations.parquet"), index=False)
    exhibit_A(ft)
    exhibit_B(ft)
    exhibit_B2(ft)
    exhibit_C(ft, panel)
    exhibit_D(ft)
    hr("DONE")
    say(f"Tidy CSVs in {OUT} (s41_A_*, s41_B_*, s41_C_*, s41_D_*).")
    say("Send back the four PLAIN READING blocks and we decide on 2026.")


if __name__ == "__main__":
    main()
