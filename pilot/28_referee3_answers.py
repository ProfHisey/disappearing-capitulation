"""Stage 28: ANSWERS TO REFEREE ROUND 3 (the testable critiques).

Round 3 attacked draft v7. The two FATAL-IF-TRUE critiques and three MAJOR
ones are testable on existing data; this stage runs them all.

 (a) N1 DISTANCE-TO-THRESHOLD / COMPOSITION. Depth-suppresses-folding and
     the era decline could both be artifacts of entry Active Share: a fund
     entering at 95 sits far from the 60 line. Tests: entry-AS distribution
     by era; the slim hazard with entry AS as a covariate; the era HR
     estimated WITHIN entry-AS bands (70-80, 80-90, 90+).
 (b) N2 CONTINUOUS SUBSTITUTION. If funds now fold to 61 instead of 58,
     crossings vanish while the behavior survives (and partial folds get
     coded as recoveries). Tests, unconditional on any crossing: the
     distribution of within-spell Active Share declines by era; shares of
     spells with declines beyond 5/10/15 points; shares reaching below
     65 and 62.5 (threshold-free surrender intensity). Plus the Section 9
     lead-time claim: share of capitulators already under 70 and 65 two
     quarters before crossing.
 (c) N5/R6 TABLE 4 MATCHED-ROW RECONCILIATION. Matched K=8 group alphas
     differ by -0.06 while the spread is -0.66: print common-month group
     alphas so levels and spread reconcile exactly, plus the spread with
     its CI so the "fee-sized penalty inside the MDE" reading is explicit.
 (d) N12 POST-FOLD TRAJECTORY. Capitulators' own-benchmark shortfall and
     mean Active Share by post-formation year: do folded funds stay folded,
     and does the -2.39%/yr own-benchmark bleed decay toward -fees?
 (e) N7 STRICT-DEFINITION COHORT TABLE. Table 3's capitulation column with
     reassignment-flavored crossings (AS vs entry benchmark >= 0.75 at
     crossing) reclassified as non-events.

Output: output/referee_28_round3.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["REFEREE ROUND 3 ANSWERS (stage 28)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# entry AS and within-spell minimum AS per spell (observed-row walk)
as0, as_min_sp, lead2_as = [], [], []
for _, s in sp.iterrows():
    g = PF.get(s["wficn"])
    if g is None or s["start_p"] not in g.index:
        as0.append(np.nan); as_min_sp.append(np.nan); lead2_as.append(np.nan)
        continue
    idx = g.index
    p0 = idx.get_loc(s["start_p"])
    pend = min(p0 + int(s["end_dur"]), len(idx) - 1)
    vals = g["as_min"].iloc[p0:pend + 1]
    as0.append(float(vals.iloc[0]))
    as_min_sp.append(float(vals.min()))
    if s["capitulated"] and int(s["m_dur"]) >= 2 and p0 + int(s["m_dur"]) - 2 < len(idx):
        lead2_as.append(float(g["as_min"].iloc[p0 + int(s["m_dur"]) - 2]))
    else:
        lead2_as.append(np.nan)
sp["as0"], sp["as_min_sp"], sp["lead2_as"] = as0, as_min_sp, lead2_as
sp["das_sp"] = sp["as_min_sp"] - sp["as0"]
sp["era3"] = pd.cut(sp["start_p"].dt.year, [0, 1994, 2009, 9999],
                    labels=["1980-94", "1995-2009", "2010-23"])

# ------------------------------------------- (a) entry-AS confound ----
def sect_a():
    log.append("  entry Active Share by era (composition check):")
    for era, g in sp.groupby("era3", observed=True):
        d = g["as0"].dropna()
        log.append(f"    {era}: mean {d.mean():.3f} | p25 {d.quantile(.25):.3f}"
                   f" | p50 {d.median():.3f} | p75 {d.quantile(.75):.3f}")
    dt = R.build_dt(sp, PF)
    dt = dt.merge(sp["as0"].rename("as0"), left_on="spell_id",
                  right_index=True, how="left")
    R.slim_fit(dt, R.SLIM, "event", log, "baseline slim (reference)")
    R.slim_fit(dt, R.SLIM + ["as0"], "event", log,
               "slim + ENTRY AS covariate")
    log.append("  era HR within entry-AS bands:")
    for lo, hi, lab in [(0.70, 0.80, "70-80"), (0.80, 0.90, "80-90"),
                        (0.90, 1.01, "90+")]:
        d = dt[(dt["as0"] >= lo) & (dt["as0"] < hi)]
        R.slim_fit(d, R.SLIM, "event", log, f"  band {lab}")
    log.append("  reading: if the era HR stays far below 1 with entry AS "
               "controlled and within every band, neither the depth result "
               "nor the decline is a distance-to-threshold artifact.")

# --------------------------------------- (b) continuous substitution ----
def sect_b():
    log.append("  within-spell AS decline (as_min_sp - as0), by entry era, "
               "UNCONDITIONAL on crossing:")
    for era, g in sp.groupby("era3", observed=True):
        d = g["das_sp"].dropna()
        log.append(f"    {era}: p10 {d.quantile(.10):+.3f} | p25 "
                   f"{d.quantile(.25):+.3f} | p50 {d.median():+.3f} | "
                   f"p75 {d.quantile(.75):+.3f}")
        log.append(f"      share of spells with decline > 5pts "
                   f"{(d < -0.05).mean():6.1%} | > 10pts "
                   f"{(d < -0.10).mean():6.1%} | > 15pts "
                   f"{(d < -0.15).mean():6.1%}")
        m = g["as_min_sp"].dropna()
        log.append(f"      share reaching AS < 0.65: {(m < 0.65).mean():6.2%}"
                   f" | < 0.625: {(m < 0.625).mean():6.2%} | < 0.60: "
                   f"{(m < 0.60).mean():6.2%}")
    caps = sp[sp["capitulated"]]
    l2 = caps["lead2_as"].dropna()
    log.append(f"  Section 9 lead-time check ({len(l2):,} capitulations with "
               f"a t-2 observation):")
    log.append(f"    already below 70 two quarters before crossing: "
               f"{(l2 < 0.70).mean():.1%}; below 65: {(l2 < 0.65).mean():.1%}")
    log.append("  reading: if 10-point-plus declines and sub-65 visits "
               "collapsed across eras in step with crossings, the BEHAVIOR "
               "disappeared, not just the threshold event. If they did not, "
               "critique N2 has legs and the paper must report both.")

# ------------------------------- (c) Table 4 matched reconciliation ----
def sect_c():
    fm = PL.get_fund_monthly(log)
    fm["m"] = fm["caldt"].dt.to_period("M")
    fac = PL.get_factors(log)
    fac["m"] = fac["month"].dt.to_period("M")
    FAC = fac.set_index("m")[["mktrf", "smb", "hml", "mom", "rf"]]

    def obs_q(w, start, k):
        g = PF.get(w)
        if g is None:
            return start + k
        qs = g.index[g.index >= start]
        return qs[k] if k < len(qs) else start + k

    def port(ev):
        rows = []
        for _, s in ev.iterrows():
            m0 = s["entry_q"].asfreq("M", how="end") + 1
            rows += [(s["wficn"], m0 + k) for k in range(36)]
        mem = pd.DataFrame(rows, columns=["wficn", "m"]).drop_duplicates()
        d = mem.merge(fm[["wficn", "m", "fret"]], on=["wficn", "m"],
                      how="inner")
        g = d.groupby("m")["fret"].agg(["mean", "size"])
        return g[g["size"] >= 10]["mean"]

    def ff4(r, excess):
        j = pd.concat([r.rename("r"), FAC], axis=1, join="inner").dropna()
        y = (j["r"] - j["rf"]).to_numpy() if excess else j["r"].to_numpy()
        X = sm.add_constant(j[["mktrf", "smb", "hml", "mom"]].to_numpy())
        m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        return float(m.params[0]), float(m.bse[0]), len(j)

    elig = sp[sp["end_dur"] >= 8]
    folded = elig[elig["m_dur"].notna() & (elig["m_dur"] <= 8)].copy()
    fighting = elig[elig["m_dur"].isna() | (elig["m_dur"] > 8)].copy()
    for g in (folded, fighting):
        g["entry_q"] = [obs_q(w, s, 8)
                        for w, s in zip(g["wficn"], g["start_p"])]
    pf_, pg = port(folded), port(fighting)
    common = pf_.index.intersection(pg.index)
    a1, s1, n1 = ff4(pf_.loc[common], excess=True)
    a2, s2, n2 = ff4(pg.loc[common], excess=True)
    spr = (pg - pf_).dropna()
    aC, sC, nC = ff4(spr, excess=False)
    log.append(f"  matched K=8, COMMON MONTHS ONLY ({len(common)}m):")
    log.append(f"    folded   alpha {a1 * 12:+.2%}/yr")
    log.append(f"    fighting alpha {a2 * 12:+.2%}/yr")
    log.append(f"    difference     {(a2 - a1) * 12:+.2%}/yr  <- must equal "
               f"the spread below")
    ci_lo, ci_hi = (aC - 1.96 * sC) * 12, (aC + 1.96 * sC) * 12
    log.append(f"    spread alpha   {aC * 12:+.2%}/yr (se {sC * 12:.2%}), "
               f"95% CI [{ci_lo:+.2%}, {ci_hi:+.2%}]")
    log.append(f"  reading: v7's Table 4 printed FULL-WINDOW group alphas "
               f"beside a COMMON-MONTH spread; the referee's 10x gap is that "
               f"misalignment, not a computation error. v8 prints "
               f"common-month levels. The CI line replaces 'null' rhetoric: "
               f"the matched point estimate is "
               f"{abs(aC) / (2.80 * sC):.0%} of the MDE, bounded well "
               f"inside +/-2%/yr.")

# ----------------------------------- (d) post-fold trajectory ----
def sect_d():
    caps = sp[sp["capitulated"]].copy()
    caps["entry_q"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
    rel = panel.assign(rel=panel["qret"] - panel["bench_qret"]) \
               .set_index(["wficn", "quarter"])[["rel"]]
    asq = panel.set_index(["wficn", "quarter"])["as_min"]
    for lo, hi, lab in [(1, 4, "year 1"), (5, 8, "year 2"), (9, 12, "year 3")]:
        vals, asv = [], []
        for _, s in caps.iterrows():
            for k in range(lo, hi + 1):
                key = (s["wficn"], s["entry_q"] + k)
                if key in rel.index:
                    vals.append(float(rel.at[key, "rel"]))
                v = asq.get(key)
                if pd.notna(v):
                    asv.append(float(v))
        log.append(f"    {lab}: own-benchmark net {np.mean(vals) * 4:+.2%}/yr "
                   f"(n {len(vals):,} fund-qtrs) | mean AS {np.mean(asv):.3f}")
    log.append("  reading: if mean AS stays near or below 0.60 and the "
               "own-benchmark bleed decays toward minus-fees, folded funds "
               "stay folded and N12's implausibility worry is answered by "
               "the trajectory (early quarters still carry live-bet losses; "
               "later ones are index-plus-fees).")

# ------------------------------- (e) strict-definition cohort table ----
def sect_e():
    bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
    bp["quarter"] = pd.to_datetime(bp["month"]).dt.to_period("Q")
    bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
            .drop_duplicates(["wficn", "quarter"], keep="last")
            .set_index(["wficn", "quarter"]))
    strict = sp.copy()
    dropped = 0
    for i, s in strict[strict["capitulated"]].iterrows():
        w, start = s["wficn"], s["start_p"]
        qc = pd.Period(s["m_cal_q"], freq="Q")
        if (w, start) not in bp.index or (w, qc) not in bp.index:
            continue
        col = "as_" + str(bp.at[(w, start), "bench_min"]).lower()
        if col not in bp.columns:
            continue
        v = bp.at[(w, qc), col]
        if pd.notna(v) and float(v) >= 0.75:
            strict.at[i, "capitulated"] = False
            dropped += 1
    log.append(f"  reassignment-flavored crossings reclassified: {dropped} "
               f"of {int(sp['capitulated'].sum())}")
    yr = strict["start_p"].dt.year
    log.append(f"  {'cohort':10s} {'n':>7s} {'strict cap':>11s} "
               f"{'(loose cap)':>11s}")
    for lo, hi in [(1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009),
                   (2010, 2014), (2015, 2019), (2020, 2023)]:
        s_ = strict[yr.between(lo, hi)]
        l_ = sp[sp["start_p"].dt.year.between(lo, hi)]
        if len(s_):
            log.append(f"  {lo}-{hi}  {len(s_):7,} "
                       f"{s_['capitulated'].mean():11.1%} "
                       f"{l_['capitulated'].mean():11.1%}")
    log.append("  reading: v8 reports both columns in Table 3 (or its "
               "caption), so the loose-definition levels never stand alone.")

# ------------------------------- (f) N3: duration x era interaction ----
def sect_f():
    dt = R.build_dt(sp, PF)
    dt["dur5p_x_era"] = dt["dur_5p"] * dt["era_1023"]
    R.slim_fit(dt, R.SLIM + ["dur5p_x_era"], "event", log,
               "slim + duration x era interaction")
    log.append("  reading: an interaction HR near 1 says the duration "
               "gradient is stable across eras - the APC restriction the "
               "identification subsection will state (duration profile "
               "constant across eras) is then TESTED, not assumed.")

# ------------------------------- (g) N10: mid-hold attrition ----
def sect_g():
    def obs_q(w, start, k):
        g = PF.get(w)
        if g is None:
            return start + k
        qs = g.index[g.index >= start]
        return qs[k] if k < len(qs) else start + k

    dq = pd.PeriodIndex(death["death_q"].where(
        death["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
    dmap = dict(zip(death["wficn"], dq))
    died_flag = dict(zip(death["wficn"], death["died"].fillna(False)))
    elig = sp[sp["end_dur"] >= 8]
    folded = elig[elig["m_dur"].notna() & (elig["m_dur"] <= 8)].copy()
    fighting = elig[elig["m_dur"].isna() | (elig["m_dur"] > 8)].copy()
    for g_, lab in ((folded, "folded-by-q8"), (fighting, "fighting-at-q8")):
        n = died = 0
        for _, s in g_.iterrows():
            eq = obs_q(s["wficn"], s["start_p"], 8)
            n += 1
            d = dmap.get(s["wficn"])
            if died_flag.get(s["wficn"], False) and d is not pd.NaT \
                    and pd.notna(d) and 0 < (d - eq).n <= 12:
                died += 1
        log.append(f"    {lab}: {n:,} entries | fund dies within the "
                   f"36-month hold: {died:,} ({died / max(n, 1):.1%})")
    log.append("  reading: calendar-time members simply stop contributing "
               "return months at death (no delisting return is imputed); "
               "this prints the differential-attrition exposure the "
               "convention carries so v8 can state the rule and the counts "
               "(referee N10).")

R.section(log, "(a) N1: ENTRY-AS CONFOUND / COMPOSITION", sect_a)
R.section(log, "(b) N2: CONTINUOUS SUBSTITUTION + LEAD TIME", sect_b)
R.section(log, "(c) N5: TABLE 4 MATCHED-ROW RECONCILIATION", sect_c)
R.section(log, "(d) N12: POST-FOLD TRAJECTORY", sect_d)
R.section(log, "(e) N7: STRICT-DEFINITION COHORT TABLE", sect_e)
R.section(log, "(f) N3: DURATION x ERA INTERACTION", sect_f)
R.section(log, "(g) N10: MID-HOLD ATTRITION BY GROUP", sect_g)

log.append("\nSTAGE 28 DONE - aggregates only. Feed the results back for "
           "draft v8's rewrite decisions (title framing, abstract, "
           "Section 9 lead time).")
P.write_report("referee_28_round3.txt", log)
print("\n".join(log))
