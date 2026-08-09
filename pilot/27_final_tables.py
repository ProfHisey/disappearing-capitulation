"""Stage 27: FINAL TABLE NUMBERS for draft v7 (post-audit machinery).

Reprints, on the fixed spell machinery (A1 calendar stamps + A2 last-row
spells), the exhibit numbers that still traced to pre-audit runs:

 (a) Table 3: outcome shares by five-year entry cohort (n, capitulated,
     died all-cause, died liquidation-only), plus the pre-1990 exclusion
     count.
 (b) Section 7 death accounting: spell-level death split into liquidation /
     merger / other, using the death_v2 typing from stage 19.
 (c) The liquidation-only era contrast (the "death did not get rarer" rate
     pair) on the new spell set.
 (d) A quarterly own-benchmark calendar-time ruler for Table 4's third row,
     groups formed exactly as stage 26's FIXED convention (calendar-true
     entries, deduped membership).

Output: output/referee_27_final_tables.txt (aggregates only).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["FINAL TABLE NUMBERS (stage 27)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# death typing from stage 19's cache
try:
    dv = pd.read_parquet(P.CACHE / "death_v2.parquet")
    dcol = next(c for c in dv.columns if "dtype" in c)
    sp = sp.merge(dv[["wficn", dcol]].rename(columns={dcol: "dtype"}),
                  on="wficn", how="left")
except Exception as e:  # noqa: BLE001
    sp["dtype"] = np.nan
    log.append(f"  (death_v2 cache unavailable: {e} - run stage 19 first; "
               f"liquidation splits will be empty)")
sp["died_liq"] = sp["spell_died"] & sp["dtype"].eq("liquidation")

COHORTS = [(1990, 1994), (1995, 1999), (2000, 2004), (2005, 2009),
           (2010, 2014), (2015, 2019), (2020, 2023)]

def sect_table3():
    yr = sp["start_p"].dt.year
    log.append(f"  spells before 1990 (excluded from era claims): "
               f"{int((yr < 1990).sum()):,} of {len(sp):,}")
    log.append(f"  {'cohort':10s} {'n':>7s} {'capitulated':>12s} "
               f"{'died (all)':>11s} {'died (liq)':>11s}")
    for lo, hi in COHORTS:
        s = sp[yr.between(lo, hi)]
        if not len(s):
            continue
        log.append(f"  {lo}-{hi}  {len(s):7,} "
                   f"{s['capitulated'].mean():12.1%} "
                   f"{s['spell_died'].mean():11.1%} "
                   f"{s['died_liq'].mean():11.1%}")

def sect_deathsplit():
    d = sp[sp["spell_died"]]
    log.append(f"  spell-level deaths: {len(d):,} of {len(sp):,} spells")
    for k, v in d["dtype"].value_counts(dropna=False).items():
        log.append(f"    {str(k):14s} {v:,}")

def sect_liq_era():
    yr = sp["start_p"].dt.year
    for lo, hi in [(1990, 1994), (1995, 2009), (2010, 2023)]:
        s = sp[yr.between(lo, hi)]
        log.append(f"  {lo}-{hi}: died-liquidation share "
                   f"{s['died_liq'].mean():.2%}  (n {len(s):,})")

def sect_ownbench():
    def obs_q(w, start, k):
        g = PF.get(w)
        if g is None:
            return start + k
        qs = g.index[g.index >= start]
        return qs[k] if k < len(qs) else start + k

    caps = sp[sp["capitulated"]].copy()
    caps["entry_q"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
    res = sp[(sp["end_dur"] >= 8)
             & (sp["m_dur"].isna() | (sp["m_dur"] > 8))].copy()
    res["entry_q"] = [obs_q(w, s, 8)
                      for w, s in zip(res["wficn"], res["start_p"])]
    rel = panel.assign(rel=panel["qret"] - panel["bench_qret"]) \
               .set_index(["wficn", "quarter"])["rel"]

    def port(ev):
        rows = []
        for _, s in ev.iterrows():
            rows += [(s["wficn"], s["entry_q"] + k) for k in range(1, 13)]
        mem = pd.DataFrame(rows, columns=["wficn", "quarter"]) \
                .drop_duplicates()
        mem["rel"] = rel.reindex(pd.MultiIndex.from_frame(mem)).to_numpy()
        d = mem.dropna(subset=["rel"])
        g = d.groupby("quarter")["rel"].agg(["mean", "size"])
        return g[g["size"] >= 10]["mean"]

    pc, pr = port(caps), port(res)
    for name, s in [("capitulators", pc), ("resisters", pr)]:
        log.append(f"    {name}: own-benchmark net "
                   f"{s.mean() * 4:+.2%}/yr over {len(s)} quarters")
    spr = (pr - pc).dropna()
    m = sm.OLS(spr.to_numpy(),
               np.ones((len(spr), 1))).fit(cov_type="HAC",
                                           cov_kwds={"maxlags": 4})
    a, se = float(m.params[0]), float(m.bse[0])
    log.append(f"    SPREAD (R - C): {a * 4:+.2%}/yr (se {se * 4:.2%}, "
               f"t {a / se:+.2f}) over {len(spr)} common quarters "
               f"(quarterly own-benchmark ruler, FIXED formation)")

R.section(log, "(a) TABLE 3: OUTCOME SHARES BY ENTRY COHORT", sect_table3)
R.section(log, "(b) SPELL-LEVEL DEATH SPLIT", sect_deathsplit)
R.section(log, "(c) LIQUIDATION-ONLY ERA CONTRAST", sect_liq_era)
R.section(log, "(d) OWN-BENCHMARK RULER (Table 4 row 3)", sect_ownbench)

log.append("\nSTAGE 27 DONE - aggregates only. These plus stages 22/24b/26 "
           "are the complete v7 number set.")
P.write_report("referee_27_final_tables.txt", log)
print("\n".join(log))
