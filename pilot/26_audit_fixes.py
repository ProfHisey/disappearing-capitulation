"""Stage 26: AUDIT FIXES - quantify A1/A2/A4 and diff the headline numbers.

The overnight adversarial code audit found two structural cousins of the
gap-bridging issue plus a portfolio-membership inconsistency:

 A1  durations count OBSERVED rows but were converted back to calendar
     quarters everywhere downstream (start + dur), misdating events for
     spells containing reporting gaps. Fixed by stamping the event's actual
     calendar quarter in extract_spells (m_cal_q / c_cal_q) and walking the
     observed clock in build_dt.
 A2  a spell entered on a fund's FINAL observed quarter was silently
     dropped (the end-check lived in the else branch), undercounting
     right-censored spells and, more importantly, deaths at the panel edge.
     Fixed: extract_spells now emits these as 1-quarter data_end spells.
 A4  calendar-time H7 portfolios weighted repeat-entry funds 2-3x within a
     portfolio-month (stage 06 deduped membership; 20/23/24/24b did not).
     Rerun here with drop_duplicates + calendar-true entry quarters.

Sections: (a) A2 magnitude + era table old vs new; (b) A1 magnitude;
(c) headline hazards old vs new; (d) H7 spreads old vs fixed.
Output: output/referee_26_audit_fixes.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["AUDIT FIXES A1/A2/A4 (stage 26)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

sp_old = R.attach_death(
    PL.extract_spells(panel, client_cut=None, emit_last_row_entry=False),
    death)
sp_new = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

# ------------------------------------------ (a) A2: recovered spells ----
def sect_a2():
    add = len(sp_new) - len(sp_old)
    log.append(f"  last-row-entry spells recovered: {add:,} "
               f"(+{add / len(sp_old):.2%} on the old count)")
    log.append(f"  of the recovered spells, spell_died: "
               f"{int(sp_new['spell_died'].sum()) - int(sp_old['spell_died'].sum()):,} "
               f"additional edge-deaths now counted")
    R.summarize(sp_old, log, "OLD (last-row entries dropped)")
    R.summarize(sp_new, log, "NEW (A2 fixed)")

# ------------------------------------------ (b) A1: misdated events ----
def sect_a1():
    caps = sp_new[sp_new["capitulated"]]
    mis, gaps = 0, []
    for _, s in caps.iterrows():
        naive = s["start_p"] + int(s["m_dur"])
        true = pd.Period(s["m_cal_q"], freq="Q")
        if naive != true:
            mis += 1
            gaps.append((true - naive).n)
    log.append(f"  capitulations misdated by start+dur arithmetic: {mis:,} "
               f"of {len(caps):,} ({mis / max(len(caps), 1):.1%})")
    if gaps:
        log.append(f"  misdating size (quarters the true crossing is later): "
                   f"median {np.median(gaps):.0f} | p90 "
                   f"{np.percentile(gaps, 90):.0f} | max {max(gaps)}")

# ------------------------------- (c) headline hazards, old vs new ----
def sect_hazard():
    dt_o = R.build_dt(sp_old, PF, observed_clock=False)
    dt_n = R.build_dt(sp_new, PF)
    R.slim_fit(dt_o, R.SLIM, "event", log, "capitulation OLD (pre-audit)")
    R.slim_fit(dt_n, R.SLIM, "event", log, "capitulation NEW (A1+A2 fixed)")
    R.slim_fit(dt_o, R.SLIM, "event_die", log, "death OLD (pre-audit)")
    R.slim_fit(dt_n, R.SLIM, "event_die", log, "death NEW (A1+A2 fixed)")
    log.append("  v4 baseline for reference: cap era HR 0.21, dur 2.79, "
               "depth 28.3; death depth 0.29, era 0.76")

# ----------------------------------- (d) A4: H7 membership dedup ----
def sect_h7():
    fm = PL.get_fund_monthly(log)
    fm["m"] = fm["caldt"].dt.to_period("M")
    fac = PL.get_factors(log)
    fac["m"] = fac["month"].dt.to_period("M")
    FAC = fac.set_index("m")[["mktrf", "smb", "hml", "mom", "rf"]]

    def ff4_spread(r):
        j = pd.concat([r.rename("r"), FAC], axis=1, join="inner").dropna()
        X = sm.add_constant(j[["mktrf", "smb", "hml", "mom"]].to_numpy())
        m = sm.OLS(j["r"].to_numpy(), X).fit(cov_type="HAC",
                                             cov_kwds={"maxlags": 6})
        return float(m.params[0]), float(m.bse[0]), len(j)

    def obs_q(w, start, k):
        g = PF.get(w)
        if g is None:
            return start + k
        qs = g.index[g.index >= start]
        return qs[k] if k < len(qs) else start + k

    def port(ev, dedup):
        rows = []
        for _, s in ev.iterrows():
            m0 = s["entry_q"].asfreq("M", how="end") + 1
            rows += [(s["wficn"], m0 + k) for k in range(36)]
        mem = pd.DataFrame(rows, columns=["wficn", "m"])
        dup = mem.duplicated().mean()
        if dedup:
            mem = mem.drop_duplicates()
        d = mem.merge(fm[["wficn", "m", "fret"]], on=["wficn", "m"],
                      how="inner")
        g = d.groupby("m")["fret"].agg(["mean", "size"])
        return g[g["size"] >= 10]["mean"], dup

    def run(caps, res, label):
        p1, d1 = port(caps, "FIXED" in label)
        p2, d2 = port(res, "FIXED" in label)
        spr = (p2 - p1).dropna()
        if len(spr) < 24:
            log.append(f"    {label}: too few months - skipped")
            return
        a, se, n = ff4_spread(spr)
        log.append(f"    {label}: spread {a * 12:+.2%}/yr (se {se * 12:.2%}, "
                   f"t {a / se:+.2f}) | MDE(80%) {2.80 * se * 12:.2%}/yr | "
                   f"{n}m | dup member-months folded {d1:.1%} / "
                   f"fighting {d2:.1%}")

    # unmatched (battery IV design)
    caps = sp_new[sp_new["capitulated"]].copy()
    res = sp_new[(sp_new["end_dur"] >= 8)
                 & (sp_new["m_dur"].isna() | (sp_new["m_dur"] > 8))].copy()
    log.append("  UNMATCHED, NET:")
    caps["entry_q"] = caps["start_p"] + caps["m_dur"].astype(int)
    res["entry_q"] = res["start_p"] + 8
    run(caps, res, "OLD  (event-weighted, start+dur clock)")
    caps["entry_q"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
    res["entry_q"] = [obs_q(w, s, 8)
                      for w, s in zip(res["wficn"], res["start_p"])]
    run(caps, res, "FIXED (deduped, calendar-true clock)")

    # matched K=8
    log.append("  MATCHED K=8, NET:")
    elig = sp_new[sp_new["end_dur"] >= 8]
    folded = elig[elig["m_dur"].notna() & (elig["m_dur"] <= 8)].copy()
    fighting = elig[elig["m_dur"].isna() | (elig["m_dur"] > 8)].copy()
    for g in (folded, fighting):
        g["entry_q"] = g["start_p"] + 8
    run(folded, fighting, "OLD  (event-weighted, start+dur clock)")
    for g in (folded, fighting):
        g["entry_q"] = [obs_q(w, s, 8)
                        for w, s in zip(g["wficn"], g["start_p"])]
    run(folded, fighting, "FIXED (deduped, calendar-true clock)")
    log.append("  reading: if OLD and FIXED agree within a fraction of the "
               "MDE, the powered-null H7 conclusion is insensitive to the "
               "membership convention (A4) and to the clock fix (A1), and "
               "the paper cites the FIXED numbers.")

R.section(log, "(a) A2: LAST-ROW SPELLS RECOVERED", sect_a2)
R.section(log, "(b) A1: EVENT MISDATING QUANTIFIED", sect_a1)
R.section(log, "(c) HEADLINE HAZARDS, OLD vs NEW", sect_hazard)
R.section(log, "(d) A4: H7 SPREADS, OLD vs FIXED", sect_h7)

log.append("\nSTAGE 26 DONE - aggregates only. If (c) moves at the second "
           "decimal and (d) stays inside the MDE, the audit findings are "
           "closed and the draft cites the NEW numbers with an Appendix B "
           "note on the observed-quarters clock.")
P.write_report("referee_26_audit_fixes.txt", log)
print("\n".join(log))
