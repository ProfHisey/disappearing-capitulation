"""Stage 20: REFEREE BATTERY IV — H7 with teeth, and the Berk-Green horse race.

Answers referee critiques 7, 14 (partial), 19 (triage in the project doc
claude/referee-preempt-plan.md):

 (a) H7 POWER + RULERS (7). The conviction-premium test rerun three ways,
     each with the number the referee demanded: the minimum detectable
     effect (MDE), not just the t-stat.
       - net returns, FF4 alpha (replicates stage 06)
       - GROSS returns, FF4 alpha (removes the fee-mechanical equalizer:
         post-capitulation a closet indexer's net alpha ~ -fees by
         construction, so net-vs-net cannot detect skill)
       - own-benchmark-adjusted quarterly returns (each group measured
         against its own min-AS index, removing benchmark mismatch)
     Formation rules are stated in the log for the look-ahead audit.
 (b) BERK-GREEN HORSE RACE (19) + per-10%-flow HRs (14). Capacity story:
     capitulation follows INFLOWS and SIZE GROWTH (a rational fund hitting
     capacity de-activates). Surrender story: capitulation follows OUTFLOWS
     and DURATION of underperformance. Both enter one hazard; the data
     picks. Flow HRs reported per 10% quarterly flow.

Output: output/referee_20_h7_berkgreen.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["REFEREE BATTERY IV - H7 POWER + BERK-GREEN", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp0 = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

fm = PL.get_fund_monthly(log)
fm["m"] = fm["caldt"].dt.to_period("M")
fm["quarter"] = fm["caldt"].dt.to_period("Q")

# expense ratios for the gross-return leg
cov = pd.read_parquet(P.CACHE / "covars.parquet")
cov["quarter"] = pd.PeriodIndex(cov["quarter"], freq="Q")
cov["exp_ratio"] = pd.to_numeric(cov["exp_ratio"], errors="coerce")
cov.loc[~cov["exp_ratio"].between(0, 0.05), "exp_ratio"] = np.nan
fm = fm.merge(cov[["wficn", "quarter", "exp_ratio"]],
              on=["wficn", "quarter"], how="left")
fm = fm.sort_values(["wficn", "caldt"])
fm["exp_ratio"] = fm.groupby("wficn")["exp_ratio"].transform(
    lambda s: s.ffill().bfill())
fm["exp_ratio"] = fm["exp_ratio"].fillna(fm["exp_ratio"].median())
fm["fret_g"] = fm["fret"] + fm["exp_ratio"] / 12

fac = PL.get_factors(log)
fac["m"] = fac["month"].dt.to_period("M")
FAC = fac.set_index("m")[["mktrf", "smb", "hml", "mom", "rf"]]

# formation (stated for the look-ahead audit, critique 7):
#   capitulators enter the calendar-time portfolio the MONTH AFTER the
#   quarter in which AS first closed below 60%; resisters enter the month
#   after quarter 8 of a spell, conditional only on information available
#   then (spell still open, AS still >= 60%). 36-month holding, equal
#   weights, >= 10 members per month.
caps = sp0[sp0["capitulated"]].copy()
caps["entry_q"] = caps["start_p"] + caps["m_dur"].astype(int)
res = sp0[(sp0["end_dur"] >= 8)
          & (sp0["m_dur"].isna() | (sp0["m_dur"] > 8))].copy()
res["entry_q"] = res["start_p"] + 8
log.append(f"\nformation: {len(caps):,} capitulator entries, "
           f"{len(res):,} resister entries (36m hold, EW, >=10 members)")

def membership(ev, months=36):
    rows = []
    for _, s in ev.iterrows():
        m0 = s["entry_q"].asfreq("M", how="end") + 1
        for k in range(months):
            rows.append((s["wficn"], m0 + k))
    return pd.DataFrame(rows, columns=["wficn", "m"])

MEM = {"capitulators": membership(caps), "resisters": membership(res)}

def calendar_port(mem, retcol):
    d = mem.merge(fm[["wficn", "m", retcol]], on=["wficn", "m"], how="inner")
    g = d.groupby("m")[retcol].agg(["mean", "size"])
    return g[g["size"] >= 10]["mean"]

def ff4_alpha(r):
    j = pd.concat([r.rename("r"), FAC], axis=1, join="inner").dropna()
    y = (j["r"] - j["rf"]).to_numpy()
    X = sm.add_constant(j[["mktrf", "smb", "hml", "mom"]].to_numpy())
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    return float(m.params[0]), float(m.bse[0]), len(j)

def h7_leg(retcol, label):
    ports = {k: calendar_port(mem, retcol) for k, mem in MEM.items()}
    for k, r in ports.items():
        a, se, n = ff4_alpha(r)
        log.append(f"  {label} {k:13s}: alpha {a * 12:+.2%}/yr "
                   f"(se {se * 12:.2%}) over {n} months")
    spread = (ports["resisters"] - ports["capitulators"]).dropna()
    a, se, n = ff4_alpha(spread)
    mde = 2.80 * se * 12          # 5% two-sided, 80% power
    log.append(f"  {label} SPREAD (resist - capit): {a * 12:+.2%}/yr "
               f"(se {se * 12:.2%}, t {a / se:+.2f}, n {n}m) | 95% CI "
               f"[{(a - 1.96 * se) * 12:+.2%}, {(a + 1.96 * se) * 12:+.2%}] "
               f"| MDE(80% power) {mde:.2%}/yr")

def sect_h7():
    h7_leg("fret", "NET")
    h7_leg("fret_g", "GROSS")
    log.append("  reading: if the MDE is materially wider than any premium "
               "worth writing about (say 2%/yr), H7 is 'uninformative', not "
               "'null' - and the paper says so. The GROSS leg removes the "
               "fee-arithmetic equalizer.")

# own-benchmark quarterly ruler
def sect_h7_bench():
    def qport(ev):
        rows = []
        for _, s in ev.iterrows():
            for k in range(1, 13):
                rows.append((s["wficn"], s["entry_q"] + k))
        mem = pd.DataFrame(rows, columns=["wficn", "quarter"])
        d = mem.merge(panel[["wficn", "quarter", "qret", "bench_qret"]],
                      on=["wficn", "quarter"], how="inner")
        d["ra"] = d["qret"] - d["bench_qret"]
        g = d.groupby("quarter")["ra"].agg(["mean", "size"])
        return g[g["size"] >= 10]["mean"]

    ports = {"capitulators": qport(caps), "resisters": qport(res)}
    for k, r in ports.items():
        X = np.ones((len(r), 1))
        m = sm.OLS(r.to_numpy(), X).fit(cov_type="HAC",
                                        cov_kwds={"maxlags": 4})
        log.append(f"  OWN-BENCH {k:13s}: {float(m.params[0]) * 4:+.2%}/yr "
                   f"(se {float(m.bse[0]) * 4:.2%}) over {len(r)} quarters")
    spread = (ports["resisters"] - ports["capitulators"]).dropna()
    X = np.ones((len(spread), 1))
    m = sm.OLS(spread.to_numpy(), X).fit(cov_type="HAC",
                                         cov_kwds={"maxlags": 4})
    a, se = float(m.params[0]), float(m.bse[0])
    log.append(f"  OWN-BENCH SPREAD: {a * 4:+.2%}/yr (se {se * 4:.2%}, "
               f"t {a / se:+.2f}) | MDE(80%) {2.80 * se * 4:.2%}/yr")
    log.append("  note: 12-quarter window vs each fund's own min-AS "
               "benchmark; caveat that post-capitulation the benchmark is "
               "nearly the portfolio, so this leg mostly prices the "
               "RESISTERS' live bets - exactly the object H7 cares about.")

# ------------------------------------------- (b) Berk-Green horse race ----
def sect_berkgreen():
    tnaq = fm.groupby(["wficn", "quarter"])["tna"].last()
    lt = np.log(tnaq.where(tnaq > 0))
    g4 = lt.groupby(level="wficn").diff(4)      # 1y log size growth
    g4d, ltd = g4.to_dict(), lt.to_dict()
    dt = R.build_dt(sp0, PF)
    dt = dt[dt["yr"] >= 2000].copy()            # flows exist post-2000
    dt["inflow"] = dt["flow_lag"].clip(lower=0) * 10   # per 10% flow
    dt["outflow"] = dt["flow_lag"].clip(upper=0) * 10  # per 10% flow
    dt["g4"] = [g4d.get((w, q), np.nan)
                for w, q in zip(dt["wficn"], dt["q_info"])]
    dt["ln_tna"] = [ltd.get((w, q), np.nan)
                    for w, q in zip(dt["wficn"], dt["q_info"])]
    xc = ["dur_5p", "depth", "inflow", "outflow", "g4", "ln_tna"]
    R.slim_fit(dt, xc, "event", log, "capitulation 2000-23, horse race")
    log.append("""  reading (flows are PER 10% OF TNA per quarter - critique 14):
    Berk-Green capacity story predicts: inflow HR > 1 and g4 HR > 1
      (funds de-activate as money and size arrive).
    Surrender story predicts: outflow HR < 1 (more outflow -> higher
      hazard, since outflow is negative) and dur_5p HR > 1.
    Both can be partly true; what matters for positioning (critique 19)
    is which margin carries the era decline and the duration gradient.""")

R.section(log, "(a) H7 CALENDAR-TIME, NET AND GROSS, WITH MDE (critique 7)",
          sect_h7)
R.section(log, "(a2) H7 OWN-BENCHMARK RULER (critique 7)", sect_h7_bench)
R.section(log, "(b) BERK-GREEN HORSE RACE + PER-10%-FLOW HRs "
               "(critiques 19, 14)", sect_berkgreen)

log.append("\nBATTERY IV DONE - aggregates only.")
P.write_report("referee_20_h7_berkgreen.txt", log)
print("\n".join(log))
