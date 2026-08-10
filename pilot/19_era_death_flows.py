"""Stage 19: REFEREE BATTERY III — the era decline, deaths, and flows.

Answers referee critiques 4, 5, 6 (partial), 13 (partial), 15 (partial), 16
(triage in the project doc claude/referee-preempt-plan.md):

 (a) COVERAGE TABLE (4iv). Funds with AS / funds in the final panel / CRSP
     return universe, by 5-year bin, so coverage drift is visible.
 (b) FUND-LAUNCH COHORT FIXED EFFECTS (4i). The era HR re-estimated holding
     the fund's launch cohort fixed; identification comes from funds
     observed across eras.
 (c) SEMIANNUAL DOWNSAMPLING (4ii). AS observed only in Q2/Q4 everywhere,
     equalizing crossing-detection frequency across the 2004 reporting
     change. If the era step survives, it is not a detection artifact.
 (d) LIQUIDATION vs MERGER DEATHS (5). Delist-code split, distressed vs
     administrative mergers, and the death hazard rerun liquidation-only.
 (e) FLOW-ARTIFACT PURGE (6). Flows within +/-2 quarters of any share-class
     birth/closure set to missing, threshold recalibrated, who-breaks-first
     recomputed. Plus the same on TOTAL fund flows (16).
 (f) RETAIL SHARE BY YEAR (16). How much of fund TNA the retail flow
     measure actually covers, by era.
 (g) FLOWS BEFORE DEATH, LAGGED (13). Outflows at 3-8 quarters before death
     (outside any plausible announcement window) vs the final 2 quarters.
 (h) TENURE vs FUND AGE (15, partial). Joint model + missingness by era.

Output: output/referee_19_era_death_flows.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["REFEREE BATTERY III - ERA / DEATHS / FLOWS", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp0 = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# ------------------------------------------------- (a) coverage table ----
def sect_coverage():
    asp = pd.read_parquet(P.CACHE / "as_panel.parquet")
    asp = asp.dropna(subset=["wficn"])
    asp["yr"] = pd.to_datetime(asp["month"]).dt.year
    fm = PL.get_fund_monthly([])
    fm["yr"] = fm["caldt"].dt.year
    pan = panel.copy()
    pan["yr"] = pan["quarter"].dt.year
    log.append("  5y bin: funds w/ AS | funds in panel | share AS>=70 | "
               "CRSP return funds")
    for y5 in range(1980, 2024, 5):
        a = asp[asp["yr"].between(y5, y5 + 4)]
        p = pan[pan["yr"].between(y5, y5 + 4)]
        f = fm[fm["yr"].between(y5, y5 + 4)]
        if not (len(a) or len(p)):
            continue
        hi = (p.groupby("wficn")["as_min"].max() >= P.ACTIVE_START).mean() \
            if len(p) else np.nan
        log.append(f"    {y5}-{y5 + 4}: {a['wficn'].nunique():6,} | "
                   f"{p['wficn'].nunique():6,} | {hi:6.1%} | "
                   f"{f['wficn'].nunique():6,}")
    log.append("  reading: if panel coverage relative to the CRSP universe "
               "collapses late-sample, the era result needs the coverage "
               "caveat front and center.")

# ------------------------------------------- (b) launch-cohort FE ----
def sect_cohort():
    fm = PL.get_fund_monthly([])
    launch = (fm.groupby("wficn")["caldt"].min().dt.year // 5 * 5)
    launch = launch.rename("cohort5").reset_index()
    dt = R.build_dt(sp0, PF).merge(launch, on="wficn", how="left")
    dums = pd.get_dummies(dt["cohort5"], prefix="c", drop_first=True)
    dt = pd.concat([dt, dums.astype(float)], axis=1)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation, NO cohort FE")
    R.slim_fit(dt, R.SLIM + list(dums.columns), "event", log,
               "capitulation, WITH launch-cohort FE")
    log.append("  reading: era_1023 HR stable across the two rows = the "
               "decline is not a launch-cohort composition artifact. "
               "(Cohort dummy HRs suppressed from the log line for width.)")

# -------------------------------------- (c) semiannual downsampling ----
def sect_semiannual():
    pan = panel.sort_values(["wficn", "quarter"]).copy()
    qtr = pan["quarter"].dt.quarter
    pan["as_min"] = pan["as_min"].where(qtr.isin([2, 4]))
    pan["as_min"] = pan.groupby("wficn")["as_min"].transform(
        lambda s: s.ffill(limit=1))
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    R.summarize(sp, log, "AS OBSERVED SEMIANNUALLY EVERYWHERE")
    R.summarize(sp0, log, "baseline (for comparison)")
    log.append("  reading: detection frequency is now equal on both sides of "
               "2004. If the era decline persists here, the reporting-"
               "frequency critique is answered.")

# ------------------------------------ (d) liquidation vs merger deaths ----
def get_death_v2():
    pq = P.CACHE / "death_v2.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    use = ["crsp_fundno", "end_dt", "dead_flag", "delist_cd", "merge_fundno"]
    parts = []
    for chunk in pd.read_csv(P.F_SUMMARY,
                             usecols=lambda c: c.strip().lower() in use,
                             chunksize=500_000, low_memory=False,
                             encoding="latin-1"):
        parts.append(P.norm_cols(chunk))
    d = pd.concat(parts, ignore_index=True)
    d["end_dt"] = pd.to_datetime(d["end_dt"], errors="coerce")
    d["dead"] = d["dead_flag"].astype(str).str.upper().eq("Y")
    d["code"] = d["delist_cd"].astype(str).str.strip().str.upper().str[:1]
    d["mrg"] = d["merge_fundno"].notna() | d["code"].eq("M")
    d["liq"] = d["code"].eq("L")
    per_class = (d.sort_values("end_dt").groupby("crsp_fundno")
                   .agg(end_dt=("end_dt", "max"), dead=("dead", "any"),
                        mrg=("mrg", "any"), liq=("liq", "any")).reset_index())
    m1 = PL.get_mflink1()
    w = (per_class.merge(m1, on="crsp_fundno", how="inner")
                  .groupby("wficn")
                  .agg(end_dt=("end_dt", "max"), n_dead=("dead", "sum"),
                       n_cls=("dead", "size"), any_mrg=("mrg", "any"),
                       any_liq=("liq", "any")).reset_index())
    w["died"] = w["n_dead"] == w["n_cls"]
    w["death_q"] = w["end_dt"].dt.to_period("Q").astype(str)
    w["dtype"] = np.select([w["any_mrg"], w["any_liq"]],
                           ["merger", "liquidation"], default="other")
    out = w[["wficn", "died", "death_q", "dtype"]]
    out.to_parquet(pq, index=False)
    return out

def sect_deathsplit():
    dv = get_death_v2()
    dd = dv[dv["died"]]
    log.append("  dead funds by delist type: "
               + ", ".join(f"{k} {v:,}" for k, v in
                           dd["dtype"].value_counts().items()))
    # distressed vs administrative mergers: trailing rel4q at the last
    # observed panel quarter before death, and TNA trajectory
    fm = PL.get_fund_monthly([])
    fm["quarter"] = fm["caldt"].dt.to_period("Q")
    tnaq = fm.groupby(["wficn", "quarter"])["tna"].last()
    n_dis = n_adm = 0
    distressed = {}
    for _, r in dd[dd["dtype"] == "merger"].iterrows():
        w = r["wficn"]
        g = PF.get(w)
        dis = False
        if g is not None and len(g):
            rl = g["rel4q"].dropna()
            if len(rl) and rl.iloc[-1] < -0.05:
                dis = True
        try:
            dq = pd.Period(r["death_q"], freq="Q")
            t_end = tnaq.get((w, dq), np.nan)
            t_pre = tnaq.get((w, dq - 8), np.nan)
            if pd.notna(t_end) and pd.notna(t_pre) and t_pre > 0 \
                    and t_end / t_pre < 0.5:
                dis = True
        except Exception:  # noqa: BLE001
            pass
        distressed[w] = dis
        n_dis += int(dis)
        n_adm += int(not dis)
    log.append(f"  mergers classified: distressed {n_dis:,} "
               f"(underwater at end or TNA halved over 2y), "
               f"administrative {n_adm:,}")
    # death hazard reruns under three definitions
    for label, keepfun in [
        ("liquidation-only", lambda r: r["dtype"] == "liquidation"),
        ("liquidation + distressed mergers",
         lambda r: r["dtype"] == "liquidation"
         or (r["dtype"] == "merger" and distressed.get(r["wficn"], False))),
        ("all deaths (baseline)", lambda r: True),
    ]:
        dvx = dv.copy()
        keep = dv.apply(lambda r: bool(r["died"]) and keepfun(r), axis=1)
        dvx["died"] = keep
        sp = R.attach_death(PL.extract_spells(panel, client_cut=None), dvx)
        log.append(f"\n  {label}: died {int(sp['spell_died'].sum()):,} spells")
        R.summarize(sp, log, label)
        dt = R.build_dt(sp, PF)
        R.slim_fit(dt, R.SLIM, "event_die", log, f"death hazard, {label}")
    log.append("  reading: the depth gradient and the flat era pattern must "
               "hold for liquidation-only for finding 2/3's death half to "
               "survive as stated.")

# ----------------------------------------------- (e) flow purge ----
def class_event_quarters():
    """Quarters with a share-class birth or closure inside the fund's life."""
    m1 = PL.get_mflink1()
    ret = P.load_monthly_returns([]).merge(m1, on="crsp_fundno", how="inner")
    ret = ret.dropna(subset=["mret"])
    ret["q"] = ret["caldt"].dt.to_period("Q")
    cls = ret.groupby(["wficn", "crsp_fundno"])["q"].agg(["min", "max"])
    fund = ret.groupby("wficn")["q"].agg(["min", "max"])
    ev = set()
    for (w, _), r in cls.iterrows():
        f = fund.loc[w]
        if r["min"] > f["min"]:                      # class born mid-life
            ev.add((w, r["min"]))
        if r["max"] < f["max"]:                      # class closed mid-life
            ev.add((w, r["max"]))
    ev2 = set()
    for w, q in ev:
        for dq in range(-2, 3):
            ev2.add((w, q + dq))
    return ev2

def who_first(pan, label):
    pf = {w: g.set_index("quarter") for w, g in pan.groupby("wficn")}
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    p_base = sp["capitulated"].mean()
    minfl = []
    for _, s in sp.iterrows():
        g = pf.get(s["wficn"])
        vals = []
        if g is not None and s["start_p"] in g.index:
            # audit round 2: observed clock, so t aligns with m_dur (A1)
            idx = g.index
            p0 = idx.get_loc(s["start_p"])
            for t in range(1, int(s["end_dur"]) + 1):
                if p0 + t >= len(idx):
                    break
                f = g.at[idx[p0 + t], "flowq"]
                if pd.notna(f):
                    vals.append(float(f))
        minfl.append(min(vals) if vals else np.nan)
    minfl = pd.Series(minfl, index=sp.index)
    tstar = float(np.nanquantile(minfl.dropna(), p_base))
    log.append(f"  {label}: manager base rate {p_base:.2%} -> calibrated "
               f"client threshold t* = {tstar:+.1%}/q")
    mgr = cli = tie = both = 0
    for i, s in sp.iterrows():
        g = pf.get(s["wficn"])
        if g is None or not s["capitulated"]:
            continue
        c_dur = None
        if s["start_p"] in g.index:
            idx = g.index
            p0 = idx.get_loc(s["start_p"])
            for t in range(1, int(s["end_dur"]) + 1):
                if p0 + t >= len(idx):
                    break
                f = g.at[idx[p0 + t], "flowq"]
                if pd.notna(f) and f <= tstar:
                    c_dur = t          # observed-clock t, comparable to m_dur
                    break
        if c_dur is None:
            continue
        both += 1
        m = int(s["m_dur"])
        if m < c_dur:
            mgr += 1
        elif c_dur < m:
            cli += 1
        else:
            tie += 1
    if both:
        log.append(f"    spells with both events: {both:,} | manager first "
                   f"{mgr / both:.1%} | client first {cli / both:.1%} | "
                   f"same quarter {tie / both:.1%}")
    else:
        log.append("    no spells with both events")

def sect_flowpurge():
    ev = class_event_quarters()
    log.append(f"  share-class event windows: {len(ev):,} fund-quarter "
               f"cells flagged")
    who_first(panel, "V0 retail flows, unpurged (baseline)")
    idx = pd.MultiIndex.from_arrays([panel["wficn"], panel["quarter"]])
    mask = idx.isin(pd.MultiIndex.from_tuples(list(ev)))
    n_ext0 = int((panel["flowq"] <= -0.30).sum())
    pan = panel.copy()
    pan.loc[mask, "flowq"] = np.nan
    n_ext1 = int((pan["flowq"] <= -0.30).sum())
    log.append(f"  extreme outflow quarters (<= -30%): {n_ext0:,} before "
               f"purge, {n_ext1:,} after "
               f"({1 - n_ext1 / max(n_ext0, 1):.0%} removed)")
    who_first(pan, "V1 retail flows, PURGED near class events")

def sect_totalflows():
    pq = P.CACHE / "total_flow_q.parquet"
    if pq.exists():
        tf = pd.read_parquet(pq)
    else:
        m1 = PL.get_mflink1()
        ret = P.load_monthly_returns([]).merge(m1, on="crsp_fundno",
                                               how="inner")
        ret = ret.sort_values(["crsp_fundno", "caldt"])
        ret["tna_lag"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
        ret["flow"] = ret["mtna"] - ret["tna_lag"] * (1 + ret["mret"])
        fmm = (ret.dropna(subset=["flow"])
                  .groupby(["wficn", "caldt"])
                  .agg(flow=("flow", "sum"), tna=("mtna", "sum"))
                  .reset_index())
        fmm["quarter"] = fmm["caldt"].dt.to_period("Q")
        fq = (fmm.groupby(["wficn", "quarter"])
                 .agg(flow=("flow", "sum"), tna_end=("tna", "last"))
                 .reset_index().sort_values(["wficn", "quarter"]))
        fq["tna_prev"] = fq.groupby("wficn")["tna_end"].shift(1)
        fq["flowq"] = (fq["flow"] / fq["tna_prev"]).where(fq["tna_prev"] > 0)
        fq["flowq"] = fq["flowq"].clip(-1, 1)
        tf = fq[["wficn", "quarter", "flowq"]].dropna().copy()
        tf["quarter"] = tf["quarter"].astype(str)
        tf.to_parquet(pq, index=False)
    tf = tf.copy()
    tf["quarter"] = pd.PeriodIndex(tf["quarter"], freq="Q")
    pan = panel.drop(columns=["flowq"]).merge(tf, on=["wficn", "quarter"],
                                              how="left")
    who_first(pan, "V2 TOTAL fund flows (all share classes)")

# ------------------------------------------ (f) retail share by year ----
def sect_retailshare():
    rfl = pd.read_parquet(P.CACHE / "retail_flags.parquet")
    m1 = PL.get_mflink1()
    ret = (P.load_monthly_returns([])
           .merge(rfl, on="crsp_fundno", how="left")
           .merge(m1, on="crsp_fundno", how="inner"))
    ret["yr"] = ret["caldt"].dt.year
    ret["ret_tna"] = ret["mtna"].where(ret["is_retail"] == True)  # noqa: E712
    g = ret.groupby("yr").agg(tot=("mtna", "sum"), rt=("ret_tna", "sum"))
    g["share"] = g["rt"] / g["tot"]
    log.append("  retail share of TNA (what the retail-flow measure covers):")
    for y5 in range(1980, 2024, 5):
        s = g.loc[g.index.to_series().between(y5, y5 + 4), "share"].mean()
        if pd.notna(s):
            log.append(f"    {y5}-{y5 + 4}: {s:.1%}")

# -------------------------------------- (g) flows before death, lagged ----
def sect_deathflows():
    dv = death[death["died"]
               & death["death_q"].astype(str).str.match(r"\d{4}Q\d")]
    pan = panel.merge(dv[["wficn", "death_q"]], on="wficn", how="inner")
    dqp = pd.PeriodIndex(pan["death_q"], freq="Q")
    pan["dq"] = (dqp - pd.PeriodIndex(pan["quarter"], freq="Q")).map(
        lambda x: getattr(x, "n", np.nan))
    early = pan.loc[pan["dq"].between(3, 8), "flowq"]
    late = pan.loc[pan["dq"].between(1, 2), "flowq"]
    base = panel["flowq"]
    log.append(f"  mean retail flow/quarter: all fund-quarters "
               f"{base.mean():+.2%} (n={base.notna().sum():,}) | dying funds "
               f"3-8q before death {early.mean():+.2%} "
               f"(n={early.notna().sum():,}) | final 2q {late.mean():+.2%} "
               f"(n={late.notna().sum():,})")
    log.append("  reading: outflows already elevated at 3-8 quarters out sit "
               "outside any plausible liquidation-announcement window - the "
               "association is not just announcement-triggered redemptions. "
               "Causal language still comes out of the draft (critique 13).")

# ------------------------------------------ (h) tenure vs fund age ----
def sect_tenure():
    cov = pd.read_parquet(P.CACHE / "covars.parquet")
    cov["quarter"] = pd.PeriodIndex(cov["quarter"], freq="Q")
    covq = cov.set_index(["wficn", "quarter"])
    fm = PL.get_fund_monthly([])
    first = fm.groupby("wficn")["caldt"].min().dt.to_period("Q")
    sp = sp0.copy()
    sp["spell_id"] = sp.index
    ent = covq.reindex(pd.MultiIndex.from_frame(sp[["wficn", "start_p"]]))
    sp["mgr_dt"] = ent["mgr_dt"].values
    sp["tenure"] = ((sp["start_p"].dt.to_timestamp()
                     - pd.to_datetime(sp["mgr_dt"])).dt.days / 365.25)
    sp["tenure"] = sp["tenure"].where(sp["tenure"].between(0, 60))
    sp["age"] = (sp["start_p"] - sp["wficn"].map(first)).map(
        lambda x: getattr(x, "n", np.nan)) / 4.0
    sp["ten_miss"] = sp["tenure"].isna().astype(float)
    log.append("  tenure missingness by era of spell start:")
    for lo, hi in R.ERAS:
        s = sp[sp["start_p"].dt.year.between(lo, hi)]
        if len(s):
            log.append(f"    {lo}-{hi}: {s['ten_miss'].mean():.1%} missing")
    dt = R.build_dt(sp, PF).merge(
        sp[["spell_id", "tenure", "age", "ten_miss"]],
        on="spell_id", how="left")
    dt["tenure0"] = dt["tenure"].fillna(0)
    R.slim_fit(dt, R.SLIM + ["age"], "event", log,
               "capitulation, fund AGE only")
    R.slim_fit(dt, R.SLIM + ["age", "tenure0", "ten_miss"], "event", log,
               "capitulation, age + tenure + missing-indicator")
    log.append("  reading: if tenure's HR collapses once age is in, H3 is "
               "confounded and stays flagged preliminary until Morningstar "
               "manager histories arrive.")

R.section(log, "(a) COVERAGE TABLE (critique 4iv)", sect_coverage)
R.section(log, "(b) LAUNCH-COHORT FIXED EFFECTS (critique 4i)", sect_cohort)
R.section(log, "(c) SEMIANNUAL DOWNSAMPLING (critique 4ii)", sect_semiannual)
R.section(log, "(d) LIQUIDATION vs MERGER (critique 5)", sect_deathsplit)
R.section(log, "(e) FLOW PURGE + WHO-FIRST (critique 6)", sect_flowpurge)
R.section(log, "(e2) TOTAL FLOWS WHO-FIRST (critique 16)", sect_totalflows)
R.section(log, "(f) RETAIL TNA SHARE BY ERA (critique 16)", sect_retailshare)
R.section(log, "(g) FLOWS BEFORE DEATH, LAGGED (critique 13)",
          sect_deathflows)
R.section(log, "(h) TENURE vs FUND AGE (critique 15)", sect_tenure)

log.append("\nBATTERY III DONE - aggregates only.")
P.write_report("referee_19_era_death_flows.txt", log)
print("\n".join(log))
