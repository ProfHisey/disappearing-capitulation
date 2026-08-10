"""Stage 18c: BENCHMARK SERIES v4 — official Russell returns + sharpened
reassignment test.

18b's verdicts: (1) the Russell reconstruction is the outlier (ours-vs-French
+119 bps/m where CPZ-vs-French is -10), so it must be replaced; (2)
ftse_russell_us.csv contains official index-level mtd_return by index_name,
roughly 2008-2020; (3) 57% of capitulation crossings happen against a
different benchmark than at entry, stable across eras — this section
measures how different those benchmarks really are.

 (a) EXTRACT official monthly returns from ftse_russell_us.csv. The file's
     row semantics are uncertain (total rows vs sector rows x layers), so
     several extraction strategies are built and each is validated against
     CPZ actuals on the 2008-2014 overlap; the best one wins. If none gets
     within 25 bps/m mean abs difference, NOTHING is overwritten and the
     script says so.
 (b) BUILD series v4 per Russell code: CPZ actuals before the official span,
     official returns within it, reconstruction only after it ends (2021+,
     flagged - known upward bias applies only to those ~11 quarters). S&P
     500 stays CRSP VW total return (validated at 6.9 bps TE). Backup
     written to bench_series_monthly_v4.parquet; live cache overwritten
     ONLY if validation passed.
 (c) REBUILD the panel on v4 and re-run the headline era table + slim
     hazards: the new V0 anchor for batteries III and IV.
 (d) REASSIGNMENT SHARPENING. For the 397 reassignment crossings: how far
     above 60% is AS against the entry benchmark (median/quartiles), and is
     the new min-AS benchmark a same-size-segment sibling (style shuffle)
     or a different segment (real migration)? Near-60 values + same-segment
     siblings = the crossings are economically the same event; far-above-60
     + cross-segment = a different phenomenon and the event definition
     needs a joint criterion.

Run this BEFORE batteries III/IV so they inherit corrected benchmarks.
Output: output/referee_18c_bench_v4.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["BENCHMARK SERIES v4 + REASSIGNMENT SHARPENING", "=" * 60]

F_FTSE = P.SOURCES / "russell" / "ftse_russell_us.csv"
NAME_TO_CODE = {
    "RUSSELL 3000": "R3", "RUSSELL 3000 GROWTH": "R3G",
    "RUSSELL 3000 VALUE": "R3V",
    "RUSSELL 1000": "R1", "RUSSELL 1000 GROWTH": "R1G",
    "RUSSELL 1000 VALUE": "R1V",
    "RUSSELL 2000": "R2", "RUSSELL 2000 GROWTH": "R2G",
    "RUSSELL 2000 VALUE": "R2V",
    "RUSSELL MIDCAP": "RM", "RUSSELL MIDCAP GROWTH": "RMG",
    "RUSSELL MIDCAP VALUE": "RMV",
}

STATE = {"official": None, "validated": False}

# ------------------------- (a) extract + validate official returns ----
def sect_extract():
    raw = P.norm_cols(pd.read_csv(F_FTSE, low_memory=False))
    raw["date"] = pd.to_datetime(raw["effective_date"], errors="coerce")
    raw["month"] = raw["date"].dt.to_period("M")
    raw["code"] = (raw["index_name"].astype(str).str.strip().str.upper()
                   .map(NAME_TO_CODE))
    raw = raw.dropna(subset=["code", "month"])
    raw["mtd_return"] = pd.to_numeric(raw["mtd_return"], errors="coerce")
    raw["mtd_contribution"] = pd.to_numeric(raw["mtd_contribution"],
                                            errors="coerce")
    # month-end snapshot rows only
    last = raw.groupby(["code", "month"])["date"].transform("max")
    raw = raw[raw["date"] == last]
    log.append(f"  usable rows {len(raw):,}; span "
               f"{raw['month'].min()}..{raw['month'].max()}; codes "
               f"{sorted(raw['code'].unique())}")

    cands = {}
    # A: explicit total rows (blank security or ~100% weight)
    secblank = (raw["security"].isna()
                | raw["security"].astype(str).str.strip()
                     .isin(["", "-", "--", "nan", "None"]))
    wt = pd.to_numeric(raw.get("daily_weight"), errors="coerce")
    totA = raw[secblank | (wt > 99.5)]
    if len(totA):
        cands["A_total_rows"] = totA.groupby(["code", "month"])[
            "mtd_return"].median()
    # B: sum of mtd_contribution within each classification layer
    for lay, g in raw.groupby(raw["classification_layer_code"].astype(str)):
        s = g.groupby(["code", "month"])["mtd_contribution"].sum()
        if len(s):
            cands[f"B_layer{lay}"] = s
    # C: median mtd_return of rows per (code, month) if the file repeats the
    # index-level number on every sector row
    cands["C_median_mtd"] = raw.groupby(["code", "month"])[
        "mtd_return"].median()

    cpz = P.load_cpz_monthly(log)
    cpz["m"] = cpz["month"].dt.to_period("M")
    cpzi = cpz.set_index("m")
    best_name, best_err, best_ser = None, np.inf, None
    for name, ser in cands.items():
        ser = ser.dropna()
        if not len(ser):
            continue
        if ser.abs().median() > 1.5:          # percent -> decimal
            ser = ser / 100.0
        errs = []
        for code, col in (("R2", "idx_r2"), ("RM", "idx_rm")):
            a = ser.xs(code, level="code") if code in \
                ser.index.get_level_values("code") else pd.Series(dtype=float)
            j = pd.concat([a.rename("off"), cpzi[col]], axis=1,
                          join="inner").dropna()
            if len(j) >= 24:
                errs.append((j["off"] - j[col]).abs().mean())
        err = np.mean(errs) if errs else np.inf
        log.append(f"  candidate {name:14s}: "
                   f"{'no CPZ overlap' if not np.isfinite(err) else f'mean abs diff vs CPZ {err * 1e4:.1f} bps/m'}"
                   f" ({ser.index.get_level_values('month').nunique()} months)")
        if err < best_err:
            best_name, best_err, best_ser = name, err, ser
    if best_ser is None or best_err > 0.0025:
        log.append("  NO candidate within 25 bps/m of CPZ - caches left "
                   "untouched. Paste this report back for a schema rethink.")
        return
    log.append(f"  WINNER: {best_name} ({best_err * 1e4:.1f} bps/m vs CPZ) - "
               f"official series accepted")
    STATE["official"] = best_ser
    STATE["validated"] = True

# ------------------------------------------------- (b) build v4 ----
def sect_build_v4():
    if not STATE["validated"]:
        log.append("  skipped (no validated official series)")
        return
    off = STATE["official"]
    ser = pd.read_parquet(P.CACHE / "bench_series_monthly.parquet")
    ser["month"] = pd.PeriodIndex(ser["month"], freq="M")
    cpz = P.load_cpz_monthly(log)
    cpz["m"] = cpz["month"].dt.to_period("M")
    FILL = {}
    for c in ("R1", "R1G", "R1V", "R3", "R3G", "R3V"):
        FILL[c] = "idx_s5"
    for c in ("RM", "RMG", "RMV"):
        FILL[c] = "idx_rm"
    for c in ("R2", "R2G", "R2V"):
        FILL[c] = "idx_r2"
    parts = [ser[ser["code"] == "S5"]]                  # CRSP S&P 500 stays
    for code in sorted(set(ser["code"].unique()) - {"S5"}):
        segs = []
        if code in off.index.get_level_values("code"):
            o = off.xs(code, level="code").dropna()
            o0, o1 = o.index.min(), o.index.max()
            segs.append(pd.DataFrame({"month": o.index, "code": code,
                                      "ret": o.values, "src": "official"}))
        else:
            o0, o1 = pd.Period("2100-01", freq="M"), pd.Period("1900-01",
                                                               freq="M")
        src = FILL.get(code)
        if src is not None:
            pre = cpz[(cpz["m"] < o0)][["m", src]].dropna()
            segs.append(pd.DataFrame({"month": pre["m"], "code": code,
                                      "ret": pre[src].values, "src": "cpz"}))
        rec = ser[(ser["code"] == code) & (ser["month"] > o1)]
        if len(rec):
            segs.append(rec.assign(src="reconstruction")[
                ["month", "code", "ret", "src"]])
        v4c = (pd.concat(segs, ignore_index=True)
                 .drop_duplicates(["month"], keep="first")
                 .sort_values("month"))
        parts.append(v4c[["month", "code", "ret"]].assign(src=v4c["src"]))
        n_by = v4c["src"].value_counts().to_dict()
        log.append(f"  {code}: {v4c['month'].min()}..{v4c['month'].max()} "
                   f"({n_by})")
    v4 = pd.concat(parts, ignore_index=True)
    v4.loc[v4["code"] == "S5", "src"] = "crsp"
    out = v4[["month", "code", "ret"]].copy()
    out["month"] = out["month"].astype(str)
    out.to_parquet(P.CACHE / "bench_series_monthly_v4.parquet", index=False)
    out.to_parquet(P.CACHE / "bench_series_monthly.parquet", index=False)
    log.append("  live benchmark cache overwritten with v4 (v3 backup "
               "remains as bench_series_monthly_v3.parquet)")

# ------------------------------------- (c) rebuild panel + headlines ----
def sect_rebuild():
    if not STATE["validated"]:
        log.append("  skipped (no validated official series)")
        return
    panel = PL.build_panel(log, force=True)
    death = PL.get_death(log)
    sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
    R.summarize(sp, log, "V0-v4 BASELINE (official Russell benchmarks)")
    pf = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}
    dt = R.build_dt(sp, pf)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation")
    R.slim_fit(dt, R.SLIM, "event_die", log, "death")
    log.append("  compare to v3 V0 (18,094 spells; era caps 6.61/5.04/2.46; "
               "cap HRs dur 2.60 depth 117 era 0.37). This table is the new "
               "anchor; batteries III/IV inherit it automatically.")

# ------------------------------------ (d) reassignment sharpening ----
def sect_reassign():
    SEG = {}
    for c in ("R3", "R3G", "R3V", "W5"):
        SEG[c] = "broad"
    for c in ("R1", "R1G", "R1V", "S5", "S5G", "S5V", "DJ"):
        SEG[c] = "large"
    for c in ("RM", "RMG", "RMV", "S4", "S4G", "S4V", "W4"):
        SEG[c] = "mid"
    for c in ("R2", "R2G", "R2V", "S6", "S6G", "S6V"):
        SEG[c] = "small"
    panel = PL.build_panel(log)
    death = PL.get_death(log)
    sp0 = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
    bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
    bp["month"] = pd.to_datetime(bp["month"])
    bp["quarter"] = bp["month"].dt.to_period("Q")
    bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
            .drop_duplicates(["wficn", "quarter"], keep="last"))
    BPI = bp.set_index(["wficn", "quarter"]).sort_index()
    vals, same_seg, sibling, n = [], 0, 0, 0
    for _, s in sp0[sp0["capitulated"]].iterrows():
        w, start = s["wficn"], s["start_p"]
        qc = pd.Period(s["m_cal_q"], freq="Q")  # audit fix A1 (round 2)
        if (w, start) not in BPI.index or (w, qc) not in BPI.index:
            continue
        b0 = str(BPI.at[(w, start), "bench_min"])
        col = "as_" + b0.lower()
        if col not in BPI.columns or pd.isna(BPI.at[(w, qc), col]):
            continue
        v = float(BPI.at[(w, qc), col])
        if v < P.CLOSET_CUTOFF:
            continue                       # frozen-confirmed, not reassigned
        bc = str(BPI.at[(w, qc), "bench_min"])
        vals.append(v)
        n += 1
        if SEG.get(b0) == SEG.get(bc):
            same_seg += 1
        sibling += int(b0.rstrip("GV") == bc.rstrip("GV"))
    v = pd.Series(vals)
    if len(v):
        log.append(f"  {n} reassignment crossings measured:")
        log.append(f"    AS vs ENTRY benchmark at crossing: p25 {v.quantile(.25):.3f} "
                   f"| median {v.median():.3f} | p75 {v.quantile(.75):.3f} "
                   f"| share < 0.65: {(v < 0.65).mean():.1%} "
                   f"| share < 0.70: {(v < 0.70).mean():.1%}")
        log.append(f"    new min-AS benchmark in SAME size segment: "
                   f"{same_seg / n:.1%}; same index family (style sibling, "
                   f"e.g. R1 vs R1V): {sibling / n:.1%}")
        log.append("  reading: median near 0.60-0.65 + mostly same-segment = "
                   "these are the same economic event measured against a "
                   "nearby ruler, and the min-AS definition stands with a "
                   "reported sensitivity. Median near 0.75+ or mostly "
                   "cross-segment = style migration contaminates the event "
                   "and a joint AS+TE definition becomes necessary.")
    else:
        log.append("  no measurable reassignment crossings (unexpected - "
                   "check as_bench_panel cache)")

R.section(log, "(a) EXTRACT + VALIDATE official Russell returns",
          sect_extract)
R.section(log, "(b) BUILD SERIES v4 (CPZ -> official -> reconstruction)",
          sect_build_v4)
R.section(log, "(c) REBUILD PANEL + NEW HEADLINE ANCHOR", sect_rebuild)
R.section(log, "(d) REASSIGNMENT CROSSINGS: HOW DIFFERENT IS THE NEW "
               "BENCHMARK?", sect_reassign)

log.append("\n18c DONE - aggregates only.")
P.write_report("referee_18c_bench_v4.txt", log)
print("\n".join(log))
