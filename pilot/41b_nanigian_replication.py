"""
Stage 41b - Nanigian replication, and the sorting-level contrast that may
decide the title of "Sort by Fees, Not Performance".

Run AFTER 41_fee_vs_performance_probe.py (it reuses that script's parquet
caches). From the (capit) env:   python 41b_nanigian_replication.py

WHY THIS EXISTS

Nanigian (2016, Journal of Financial Planning) is the one paper that could
rename ours. His result: sort active US equity funds into ten expense buckets
and only the MOST EXPENSIVE tenth is bad (CAPM alpha -1.27%/yr). Deciles 1-9
range from -0.13% to +0.87% and none is statistically distinguishable from
zero. Drop just the top decile and the rest earns +0.43%. His cross-sectional
regression of alpha on expense ratio averages -1.21 over 2000-2015, but
-0.16 once the top decile is excluded.

If that replicates, "sort by fees" really means "avoid the priciest tenth".

BUT his deciles are formed across ALL active US equity funds POOLED. Expense
ratios are not comparable across categories - small-cap and sector funds cost
more than large-cap blend for reasons that have nothing to do with a manager
gouging anyone. So a pooled top-expense decile is stuffed with small-cap and
sector funds, and he scores it on CAPM alpha against a total-market index,
which does not adjust for that size tilt. His "expensive funds are bad" may
be partly "small-cap funds had a rough 2000-2015 against the Wilshire 5000".

THAT IS A TESTABLE DISTINCTION AND IT IS THE POINT OF THIS SCRIPT:

  ARM 1  Pooled deciles, CAPM alpha        -> his design, our data. Replicate?
  ARM 2  WITHIN-CATEGORY deciles, CAPM     -> does the cliff become a gradient
                                              once fees are compared only to
                                              peers who should cost the same?
  ARM 3  Within-category, multi-factor     -> does decile 10's damage survive
                                              size/value/momentum adjustment?
  ARM 4  Both, on 2000-2015 (his window)   -> and on the full sample, and on
                                              2010-2026, the era that matters
                                              for a 2026 recommendation.

If ARM 1 reproduces the cliff and ARM 2 produces a gradient, that contrast is
a genuine methodological contribution and the paper keeps its title. If the
cliff survives within category and after factor adjustment, the honest title
is "Avoid the Expensive Tenth" and we find that out this week.

Everything Nanigian does that we match deliberately: annual sorts on the
prior-year expense ratio, equal weighting, monthly rebalancing, and capital
from disappearing funds redistributed among the surviving members of the same
decile. That last rule is his and it is also what stage 41 does, so the two
scripts are comparable to each other as well as to him.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

DATA_LIB = os.environ.get("DATA_LIB", r"E:\Finance\data\sources")
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

NANIGIAN_START, NANIGIAN_END = 2000, 2015
FULL_START, FULL_END = 1990, 2025
MODERN_START = 2010
MIN_FUNDS_POOLED = 100
MIN_FUNDS_PER_CAT = 20


def say(m):
    print(m, flush=True)


def hr(t):
    say("\n" + "=" * 72)
    say(t)
    say("=" * 72)


# --------------------------------------------------------------- factor file
PREFERRED = ["F-F_Research_Data_Factors.csv", "F-F_Research_Data_5_Factors_2x3.csv"]
BAD_TOKENS = ("daily", "weekly", "_d.", "_w.")


def find_factors():
    """Locate a MONTHLY factor file. Frequency mistakes are silent and fatal:
    weekly returns collapsed onto monthly periods give a market factor about
    a quarter the right size, which drives beta down and dumps the remainder
    into alpha. That produced +7%/yr 'alpha' for every decile on a first
    run. Prefer exact monthly filenames, exclude daily/weekly outright, and
    validate the frequency after parsing."""
    env = os.environ.get("FACTOR_CSV")
    cands = [env] if env else []
    roots = [DATA_LIB, os.environ.get("BUYRISK_LIB", r"E:\Finance\BuyRisk\data\sources")]
    subs = ["french", "ff", "factors", "aqr", "crsp_indexes", "fred"]
    for root in roots:                       # exact preferred names first
        for sub in subs:
            for name in PREFERRED:
                p = os.path.join(root, sub, name)
                if os.path.exists(p):
                    cands.append(p)
    scored = []
    for root in roots:                       # then a scored fallback
        for sub in subs:
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                continue
            for ext in ("*.csv", "*.CSV", "*.txt"):
                for p in sorted(glob.glob(os.path.join(d, "**", ext), recursive=True)):
                    base = os.path.basename(p).lower()
                    if any(t in base for t in BAD_TOKENS):
                        continue             # never a daily or weekly file
                    sc = 0
                    if base.startswith("f-f"): sc += 4
                    if "research_data" in base: sc += 3
                    if "factor" in base: sc += 2
                    scored.append((sc, p))
    scored.sort(key=lambda x: (-x[0], x[1]))
    cands += [p for _, p in scored]

    for p in cands:
        if not p or not os.path.exists(p):
            continue
        try:
            df = load_factor_file(p)
        except Exception:
            continue
        if df is None:
            continue
        say(f"factors: using {p}")
        ann_mean, ann_sd = df.mktrf.mean() * 12 * 100, df.mktrf.std() * np.sqrt(12) * 100
        say(f"  sanity: market excess {ann_mean:.1f}%/yr mean, {ann_sd:.1f}% vol, "
            f"{df.index.min()} to {df.index.max()}, {len(df):,} months")
        if not (10 < ann_sd < 30):
            say(f"  !! annualised market vol {ann_sd:.1f}% is not plausible for a")
            say("     monthly series (expect roughly 15-20%). Wrong frequency?")
            sys.exit(1)
        return df

    say("\n!! No usable MONTHLY factor file found.")
    say(f"   Point at one directly, e.g.:")
    say('   set FACTOR_CSV=E:\\Finance\\BuyRisk\\data\\sources\\french\\F-F_Research_Data_Factors.csv')
    sys.exit(1)


def load_factor_file(path):
    """Monthly FF-style file -> DataFrame indexed by Period[M] with mktrf, rf
    (+ smb/hml/umd). Rejects anything that is not monthly."""
    rows, names = [], None
    with open(path, "r", encoding="latin-1") as fh:
        for line in fh:
            parts = [x.strip() for x in line.strip().split(",")]
            if names is None and len(parts) > 1 and any("mkt" in x.lower() for x in parts):
                names = [x.lower().replace("-", "").replace(" ", "") for x in parts[1:]]
                continue
            if names is None or len(parts) < 2:
                continue
            if not (parts[0].isdigit() and len(parts[0]) == 6):
                continue                     # 6 digits = YYYYMM; 8 = daily/weekly
            try:
                vals = [float(x) for x in parts[1:len(names) + 1]]
            except ValueError:
                continue
            if any(v <= -99.0 for v in vals):
                continue
            rows.append([int(parts[0])] + vals)
    if not rows or names is None:
        return None
    df = pd.DataFrame(rows, columns=["ym"] + names[:len(rows[0]) - 1])
    df["ym"] = pd.PeriodIndex(pd.to_datetime(df.ym.astype(str), format="%Y%m"), freq="M")
    if df.ym.duplicated().any():
        return None                          # more than one row per month
    ren = {}
    for c in df.columns:
        if "mkt" in c: ren[c] = "mktrf"
        elif c == "rf": ren[c] = "rf"
        elif c in ("smb", "hml"): ren[c] = c
        elif c in ("mom", "umd"): ren[c] = "umd"
    df = df.rename(columns=ren).set_index("ym")
    keep = [c for c in ["mktrf", "rf", "smb", "hml", "umd"] if c in df.columns]
    if "mktrf" not in keep or "rf" not in keep:
        return None
    df = df[keep] / 100.0
    return df.dropna(subset=["mktrf", "rf"]) if len(df) > 200 else None


# ------------------------------------------------------------ panel (fast)
def build_panel():
    """Prefer the panel stage 41 v2 writes; that is the whole build."""
    for name in ["s41_panel.parquet", "s41b_panel.parquet"]:
        p = os.path.join(CACHE, name)
        if os.path.exists(p):
            say(f"panel: using {name}")
            d = pd.read_parquet(p)
            if "ym" in d and not str(d["ym"].dtype).startswith("period"):
                d["ym"] = pd.PeriodIndex(d["ym"].astype(str), freq="M")
            if "fundgrp" not in d and "wficn" in d:
                d["fundgrp"] = d["wficn"].astype(str)
            if "cat" not in d:
                d["cat"] = "ALL"
            return d[["fundgrp", "ym", "year", "cat", "ret", "exp_ratio", "tna"]]
    say("!! Run 41_fee_vs_performance_probe.py first - this reuses its panel.")
    sys.exit(1)


# ------------------------------------------------------------------ engine
def decile_assignments(panel, within_category):
    """Prior-year expense ratio -> decile, assigned to the following year."""
    ann = (panel.dropna(subset=["exp_ratio"])
           .sort_values(["fundgrp", "ym"])
           .groupby(["fundgrp", "year"], observed=True)
           .agg(exp_ratio=("exp_ratio", "last"), cat=("cat", "last"))
           .reset_index())
    keys = ["year", "cat"] if within_category else ["year"]
    if within_category:
        ann = ann.groupby(keys, observed=True).filter(
            lambda d: len(d) >= MIN_FUNDS_PER_CAT)
    else:
        ann = ann.groupby(keys, observed=True).filter(
            lambda d: len(d) >= MIN_FUNDS_POOLED)
    ann["dec"] = (ann.groupby(keys, observed=True)["exp_ratio"]
                  .transform(lambda s: pd.qcut(s.rank(method="first"), 10,
                                               labels=False) + 1))
    ann["hold_year"] = ann["year"] + 1
    return ann[["fundgrp", "hold_year", "dec", "exp_ratio"]]


def decile_monthly_returns(panel, assign):
    """Equal weight among surviving members, monthly, per decile."""
    p = panel[["fundgrp", "ym", "year", "ret"]].rename(columns={"year": "hold_year"})
    j = p.merge(assign, on=["fundgrp", "hold_year"], how="inner")
    out = (j.groupby(["dec", "ym"], observed=True)["ret"]
           .mean().reset_index().rename(columns={"ret": "port_ret"}))
    counts = (j.groupby(["dec", "ym"], observed=True)["fundgrp"]
              .nunique().reset_index().rename(columns={"fundgrp": "n_funds"}))
    return out.merge(counts, on=["dec", "ym"]), j


def alpha(series, fac, factors=("mktrf",)):
    """OLS alpha of an excess-return series, annualised, with a t-stat."""
    df = pd.DataFrame({"r": series}).join(fac, how="inner").dropna()
    if len(df) < 24:
        return np.nan, np.nan, len(df)
    y = (df["r"] - df["rf"]).values
    X = np.column_stack([np.ones(len(df))] + [df[f].values for f in factors])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(df) - X.shape[1]
    s2 = resid @ resid / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(s2 * xtx_inv[0, 0])
    return beta[0] * 12 * 100, beta[0] / se, len(df)


def run_arm(panel, fac, label, within_category, y0, y1, factors):
    assign = decile_assignments(panel, within_category)
    dm, joined = decile_monthly_returns(panel, assign)
    dm = dm[(dm["ym"].dt.year >= y0) & (dm["ym"].dt.year <= y1)]
    rows = []
    for d, g in dm.groupby("dec", observed=True):
        s = g.set_index("ym")["port_ret"]
        a, t, n = alpha(s, fac, factors)
        rows.append({"arm": label, "decile": int(d), "alpha_pct_yr": a,
                     "t_stat": t, "months": n,
                     "avg_n_funds": g["n_funds"].mean()})
    # deciles 1-9 combined, Nanigian's headline construction
    sub = dm[dm["dec"] <= 9].groupby("ym", observed=True)["port_ret"].mean()
    a, t, n = alpha(sub, fac, factors)
    rows.append({"arm": label, "decile": 0, "alpha_pct_yr": a, "t_stat": t,
                 "months": n, "avg_n_funds": np.nan})
    return pd.DataFrame(rows)


def cross_sectional_beta(panel, fac, within_category, y0, y1):
    """Nanigian's regression of alpha on expense ratio, with and without D10."""
    assign = decile_assignments(panel, within_category)
    p = panel[["fundgrp", "ym", "year", "ret"]].rename(columns={"year": "hold_year"})
    j = p.merge(assign, on=["fundgrp", "hold_year"], how="inner")
    j = j[(j["ym"].dt.year >= y0) & (j["ym"].dt.year <= y1)]
    f = fac.reset_index().rename(columns={"index": "ym"})
    f.columns = ["ym"] + list(f.columns[1:])
    j = j.merge(f[["ym", "mktrf", "rf"]], on="ym", how="inner")
    j["exret"] = j["ret"] - j["rf"]

    res = []
    for excl in [False, True]:
        d = j[j["dec"] <= 9] if excl else j
        yearly = (d.groupby(["fundgrp", "hold_year"], observed=True)
                  .agg(exret=("exret", lambda x: np.prod(1 + x) - 1),
                       fee=("exp_ratio", "last")).reset_index().dropna())
        betas = []
        for yr, g in yearly.groupby("hold_year"):
            if len(g) < 50:
                continue
            X = np.column_stack([np.ones(len(g)), g["fee"].values])
            b, *_ = np.linalg.lstsq(X, g["exret"].values, rcond=None)
            betas.append(b[1])
        res.append({"spec": "excl. decile 10" if excl else "all funds",
                    "avg_coefficient": np.mean(betas) if betas else np.nan,
                    "n_years": len(betas),
                    "pct_years_negative": 100 * np.mean(np.array(betas) < 0)
                    if betas else np.nan})
    return pd.DataFrame(res)


def main():
    hr("STAGE 41b - Nanigian replication and the sorting-level contrast")
    fac = find_factors()
    have = [c for c in ["smb", "hml", "umd"] if c in fac.columns]
    say(f"factors available: mktrf, rf" + (f", {', '.join(have)}" if have else ""))
    panel = build_panel()

    arms = []
    arms.append(run_arm(panel, fac, "1_pooled_CAPM_2000_2015", False,
                        NANIGIAN_START, NANIGIAN_END, ("mktrf",)))
    arms.append(run_arm(panel, fac, "2_withincat_CAPM_2000_2015", True,
                        NANIGIAN_START, NANIGIAN_END, ("mktrf",)))
    arms.append(run_arm(panel, fac, "3_pooled_CAPM_full", False,
                        FULL_START, FULL_END, ("mktrf",)))
    arms.append(run_arm(panel, fac, "4_withincat_CAPM_full", True,
                        FULL_START, FULL_END, ("mktrf",)))
    arms.append(run_arm(panel, fac, "5_withincat_CAPM_2010on", True,
                        MODERN_START, FULL_END, ("mktrf",)))
    if len(have) >= 2:
        fl = tuple(["mktrf"] + have)
        arms.append(run_arm(panel, fac, "6_withincat_multifactor_full", True,
                            FULL_START, FULL_END, fl))
    tab = pd.concat(arms, ignore_index=True)
    tab.round(3).to_csv(os.path.join(OUT, "s41b_decile_alphas.csv"), index=False)

    for arm, g in tab.groupby("arm", sort=True):
        hr(arm)
        g = g.sort_values("decile")
        show = g[g["decile"] > 0]
        say(show[["decile", "alpha_pct_yr", "t_stat", "avg_n_funds"]]
            .round(2).to_string(index=False))
        row = g[g["decile"] == 0]
        if len(row):
            say(f"  deciles 1-9 combined: {row['alpha_pct_yr'].iat[0]:+.2f}%/yr "
                f"(t = {row['t_stat'].iat[0]:.2f})")
        d10 = show[show["decile"] == 10]["alpha_pct_yr"]
        d1 = show[show["decile"] == 1]["alpha_pct_yr"]
        mid = show[(show["decile"] >= 2) & (show["decile"] <= 9)]["alpha_pct_yr"]
        if len(d1) and len(d10) and len(mid):
            spread = d1.iat[0] - d10.iat[0]
            say(f"  decile 1 minus decile 10: {spread:+.2f}%/yr;  "
                f"deciles 2-9 span {mid.min():+.2f} to {mid.max():+.2f}")

    hr("NANIGIAN'S CROSS-SECTIONAL REGRESSION")
    for wc, name in [(False, "pooled"), (True, "within category")]:
        cs = cross_sectional_beta(panel, fac, wc, NANIGIAN_START, NANIGIAN_END)
        cs.insert(0, "sorting", name)
        say(cs.round(3).to_string(index=False))
        cs.round(3).to_csv(os.path.join(
            OUT, f"s41b_cross_sectional_{name.replace(' ', '_')}.csv"),
            index=False)
    say("\nHis published numbers, 2000-2015, pooled: average coefficient -1.21")
    say("across all funds, and -0.16 once the top decile is dropped.")

    hr("PLAIN READING")
    say("Compare arm 1 with arm 2. Arm 1 is his design on our data - if the")
    say("only bad decile is the tenth, we have replicated him. Arm 2 sorts")
    say("fees only against category peers, so a small-cap fund is compared to")
    say("other small-cap funds rather than to large-cap blend.")
    say("")
    say("  Cliff in arm 1, gradient in arm 2  -> his result was an artifact of")
    say("     pooled sorting, that contrast is our methodological contribution,")
    say("     and the paper keeps the title 'Sort by Fees, Not Performance'.")
    say("  Cliff in both, and it survives arm 6 -> he is right, the honest")
    say("     title is 'Avoid the Expensive Tenth', and we say so.")
    say("  Arm 5 is the one a 2026 committee cares about. If the gradient has")
    say("     flattened since 2010, the recommendation is weaker today than the")
    say("     full-sample number suggests, and the paper must say that too.")
    say(f"\nCSVs in {OUT}: s41b_decile_alphas.csv, s41b_cross_sectional_*.csv")


if __name__ == "__main__":
    main()
