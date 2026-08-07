"""Stage 15b: v4 POST-2014 VALIDATION — why did two famous late spells vanish?

Under v3, LMVTX had a 2014+ spell and SEQUX a 2015Q4 (Valeant) spell; under
v4 both are gone, and both sat in the post-2014 segment where the official
extraction is UNVALIDATED (the CPZ cross-check ends 2014-01). Three checks:

 (a) INTERNAL CONSISTENCY of the official series: Russell 3000 must be
     approximately the cap-weighted blend of R1 and R2 (~92/8) month by
     month, and style pairs must blend to their cores. A jump in error
     size after 2014 = extraction bug in that segment.
 (b) FRENCH TRIANGULATION split at 2014: official R2/R1/R1G/R1V vs the
     French size portfolios, before vs after.
 (c) FUND TRACES (local only): quarterly as_min, rel4q, and min-AS benchmark
     for SEQUX and LMVTX 2013Q1-2018Q4, to see exactly why no spell starts.

Output: output/referee_15b_v4check.txt (a-b aggregates; c stays local).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["V4 POST-2014 VALIDATION", "=" * 60]

ser = pd.read_parquet(P.CACHE / "bench_series_monthly.parquet")
ser["month"] = pd.PeriodIndex(ser["month"], freq="M")
W = ser.pivot_table(index="month", columns="code", values="ret")

# ------------------------------------------------ (a) internal blends ----
def sect_internal():
    pairs = [("R3", 0.92, "R1", 0.08, "R2"),
             ("R1", 0.50, "R1G", 0.50, "R1V"),
             ("R2", 0.50, "R2G", 0.50, "R2V")]
    for lo, hi, tag in ((2008, 2014, "2008-2013"), (2014, 2027, "2014-2026")):
        w = W[(W.index.year >= lo) & (W.index.year < hi)]
        for tgt, wa, a, wb, b in pairs:
            if not all(c in w.columns for c in (tgt, a, b)):
                continue
            d = (w[tgt] - (wa * w[a] + wb * w[b])).dropna()
            if len(d):
                log.append(f"  {tag} {tgt} vs {wa:.2f}*{a}+{wb:.2f}*{b}: "
                           f"mean {d.mean() * 1e4:+.1f} bps/m | sd "
                           f"{d.std() * 1e4:.1f} | max|.| "
                           f"{d.abs().max() * 1e4:.0f} bps ({len(d)}m)")
    log.append("  reading: blend weights drift, so nonzero errors are normal;"
               " what matters is whether sd/max JUMP after 2014.")

# ------------------------------------------- (b) French triangulation ----
def sect_french():
    f6 = PL.parse_french_first_block(PL.F_6PORT)
    small = [c for c in f6.columns if "SMALL" in str(c).upper()
             or str(c).upper().startswith("ME1")]
    big = [c for c in f6.columns if "BIG" in str(c).upper()
           or str(c).upper().startswith("ME2")]
    f6["p_small"] = f6[small].mean(axis=1)
    f6["p_big"] = f6[big].mean(axis=1)
    f6["m"] = f6["month"].dt.to_period("M")
    fi = f6.set_index("m")
    for code, col in (("R2", "p_small"), ("R1", "p_big"),
                      ("R1G", "p_big"), ("R1V", "p_big")):
        for lo, hi, tag in ((2008, 2014, "2008-2013"),
                            (2014, 2024, "2014-2023")):
            if code not in W.columns:
                continue
            a = W[code].dropna()
            a = a[(a.index.year >= lo) & (a.index.year < hi)]
            j = pd.concat([a.rename("v4"), fi[col]], axis=1,
                          join="inner").dropna()
            if len(j) < 12:
                log.append(f"  {code} {tag}: <12 overlap months")
                continue
            d = j["v4"] - j[col]
            log.append(f"  {code} vs French {col} {tag}: corr "
                       f"{j['v4'].corr(j[col]):.4f} | mean "
                       f"{d.mean() * 1e4:+.1f} bps/m | sd "
                       f"{d.std() * 1e4:.1f} ({len(j)}m)")
    log.append("  reading: post-2014 stats should look like pre-2014 stats. "
               "A sign change or doubled sd marks the broken segment.")

# ------------------------------------------------- (c) fund traces ----
def sect_traces():
    panel = PL.build_panel(log)
    m1 = P.norm_cols(pd.read_csv(PL.MFLINK1))
    m1["ticker"] = m1["ticker"].astype(str).str.strip().str.upper()
    tmap = (m1.dropna(subset=["wficn"]).drop_duplicates("ticker")
              .set_index("ticker"))
    bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
    bp["quarter"] = pd.to_datetime(bp["month"]).dt.to_period("Q")
    bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
            .drop_duplicates(["wficn", "quarter"], keep="last")
            .set_index(["wficn", "quarter"]))
    for tk in ("SEQUX", "LMVTX"):
        w = int(tmap.loc[tk, "wficn"])
        g = panel[panel["wficn"] == w].set_index("quarter")
        log.append(f"\n  {tk} 2013Q1-2018Q4 (AS, trailing rel4q, min-AS "
                   f"benchmark):")
        for q in pd.period_range("2013Q1", "2018Q4", freq="Q"):
            asv = g.at[q, "as_min"] if q in g.index else np.nan
            rl = g.at[q, "rel4q"] if q in g.index else np.nan
            bc = (str(bp.at[(w, q), "bench_min"])
                  if (w, q) in bp.index else "?")
            s_as = "  na" if pd.isna(asv) else f"{asv:.2f}"
            s_rl = "    na" if pd.isna(rl) else f"{rl:+.1%}"
            log.append(f"    {q}: AS {s_as} | rel4q {s_rl} | bench {bc}")
    log.append("  reading: rel4q hovering just above zero through 2015-16 "
               "for SEQUX = the Valeant spell is a near-miss under corrected "
               "benchmarks (legitimate). rel4q missing, or positive by a "
               "wide margin while the fund publicly trailed by 20%+, = the "
               "post-2014 official segment is wrong and needs re-extraction.")

R.section(log, "(a) INTERNAL CONSISTENCY (index blends, pre vs post 2014)",
          sect_internal)
R.section(log, "(b) FRENCH TRIANGULATION, pre vs post 2014", sect_french)
R.section(log, "(c) SEQUX / LMVTX TRACES 2013-2018", sect_traces)

log.append("\n15b DONE - traces local only.")
P.write_report("referee_15b_v4check.txt", log)
print("\n".join(log))
