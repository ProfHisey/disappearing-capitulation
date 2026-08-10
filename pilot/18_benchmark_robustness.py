"""Stage 18: REFEREE BATTERY II — the measuring stick.

Answers referee critiques 3, 17 (partial), 18, 22 (triage in the project doc
claude/referee-preempt-plan.md). The theme: does anything survive when the
benchmark machinery is pinned down?

 (a) Build a per-benchmark Active Share panel (all as_* columns from the ND
     files, quarterly, cached) — the raw material for (b), (c), (f).
 (b) FROZEN BENCHMARK (3i). Capitulation re-detected against the benchmark
     assigned at spell entry, so within-spell benchmark reassignment cannot
     manufacture a 70->60 crossing.
 (c) CONSTANT BENCHMARK SET (3ii). Full panel rebuild with as_min taken over
     only S&P 500 + Russell 1000/2000/Midcap — indexes that exist for the
     whole sample — so a growing minimand cannot fake the era decline.
 (d) PETAJISTO OFFICIAL-BENCHMARK RERUN, 1980-2009 (3iii) + the systematic
     AS cross-check the panel demanded instead of anecdotes (22).
 (e) RECONSTRUCTED vs CPZ ACTUAL index returns, overlap years (18).
 (f) PERCENTILE THRESHOLDS (17, partial). Entry/exit defined within
     benchmark-year AS terciles instead of fixed 70/60, neutralizing
     cap-size and secular-concentration effects on AS levels.

Output: output/referee_18_benchmarks.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["REFEREE BATTERY II - BENCHMARK MACHINERY", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp0 = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

# ------------------------------------ (a) per-benchmark AS panel (cached) ----
def load_bench_panel():
    pq = P.CACHE / "as_bench_panel.parquet"
    if pq.exists():
        bp = pd.read_parquet(pq)
    else:
        nd_tr = P.load_nd(P.F_ND_TR, ["fundno", "wficn"], log)
        nd_cr = P.load_nd(P.F_ND_CRSP, ["crsp_portno", "wficn"], log)
        bp = pd.concat([nd_tr, nd_cr], ignore_index=True)
        bp = bp.dropna(subset=["wficn", "as_min"])
        bp["wficn"] = bp["wficn"].astype("int64")
        keep = list(dict.fromkeys(
            ["wficn", "month", "total_assets", "as_min", "bench_min"]
            + P.as_columns(bp)))
        bp = bp[[c for c in keep if c in bp.columns]]
        bp.to_parquet(pq, index=False)
    bp["month"] = pd.to_datetime(bp["month"])
    bp["quarter"] = bp["month"].dt.to_period("Q")
    bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
            .drop_duplicates(["wficn", "quarter"], keep="last"))
    return bp

BP = load_bench_panel()
BPI = BP.set_index(["wficn", "quarter"]).sort_index()
# audit round 2 (fix A1): observed-clock quarter lookup per fund, so scan
# indexes t align with the observed-row durations from extract_spells
PFI = {w: g.set_index("quarter").index.sort_values()
       for w, g in panel.groupby("wficn")}

def obs_q18(w, start, t):
    qs = PFI.get(w)
    if qs is None or start not in qs:
        return start + t
    j = qs.get_loc(start) + t
    return qs[j] if 0 <= j < len(qs) else start + t
log.append(f"per-benchmark AS panel: {len(BP):,} fund-quarters, "
           f"{len(P.as_columns(BP))} benchmark columns")

# --------------------------------------------- (b) frozen-at-entry bench ----
def sect_frozen():
    n_base = int(sp0["capitulated"].sum())
    res = []
    for _, s in sp0.iterrows():
        w, start = s["wficn"], s["start_p"]
        if (w, start) not in BPI.index:
            res.append(np.nan)
            continue
        b0 = str(BPI.at[(w, start), "bench_min"])
        col = "as_" + b0.lower()
        if col not in BPI.columns:
            res.append(np.nan)
            continue
        m_frozen = np.nan
        for t in range(1, int(s["end_dur"]) + 1):
            k = (w, obs_q18(w, start, t))     # audit fix A1 (round 2)
            if k in BPI.index:
                v = BPI.at[k, col]
                if pd.notna(v) and float(v) < P.CLOSET_CUTOFF:
                    m_frozen = t
                    break
        res.append(m_frozen)
    sp = sp0.copy()
    sp["m_frozen"] = res
    covered = sp["m_frozen"].notna() | sp["capitulated"]
    both = (sp["capitulated"] & sp["m_frozen"].notna()).sum()
    log.append(f"  baseline capitulations {n_base:,} | frozen-benchmark "
               f"capitulations {int(sp['m_frozen'].notna().sum()):,} | "
               f"events under both definitions {int(both):,}")
    sp_f = sp.copy()
    sp_f["capitulated"] = sp_f["m_frozen"].notna()
    sp_f["spell_died"] = sp_f["spell_died"] & ~sp_f["capitulated"]
    R.summarize(sp_f, log, "frozen-benchmark definition")
    R.summarize(sp0, log, "baseline definition (for comparison)")
    log.append("  reading: if the era decline appears in both tables, "
               "within-spell benchmark reassignment is not driving it.")
    _ = covered

# ------------------------------------------------ (c) constant bench set ----
def sect_constant():
    CONST = ["as_s5", "as_r1", "as_r2", "as_rm"]
    avail = [c for c in CONST if c in BP.columns]
    log.append(f"  constant set available on disk: {avail}")
    arr = BP[avail].to_numpy(float)
    ok = ~np.all(np.isnan(arr), axis=1)
    with np.errstate(all="ignore"):
        as_min_c = np.nanmin(arr, axis=1)
        idx = np.where(ok, np.nanargmin(
            np.where(np.isnan(arr), np.inf, arr), axis=1), -1)
    names = np.array([c[3:].upper() for c in avail] + ["NA"])
    aspc = BP.loc[ok, ["wficn", "quarter"]].copy()
    aspc["as_min"] = as_min_c[ok]
    aspc["bench_min"] = names[idx[ok]]

    fm = PL.get_fund_monthly(log)
    fm["quarter"] = fm["caldt"].dt.to_period("Q")
    fq = (fm.assign(g=lambda d: 1 + d["fret"]).groupby(["wficn", "quarter"])
            .agg(qret=("g", lambda x: x.prod() - 1),
                 nm=("g", "size")).reset_index())
    fq = fq[fq["nm"] == 3].drop(columns="nm")
    fl = pd.read_parquet(P.CACHE / "flags.parquet")
    aspc = aspc.merge(fl, on="wficn", how="left")
    aspc = aspc[aspc["passive"] != True]  # noqa: E712
    bq = PL.get_real_bench_q(log)
    flows = PL.get_retail_flows(log)
    flows["quarter"] = pd.PeriodIndex(flows["quarter"], freq="Q")
    aspc["bcode"] = aspc["bench_min"]
    pan = (aspc.merge(fq, on=["wficn", "quarter"], how="inner")
               .merge(bq, on=["quarter", "bcode"], how="left")
               .merge(flows, on=["wficn", "quarter"], how="left"))
    pan["bench_qret"] = pan["bret"]
    pan = pan.dropna(subset=["as_min", "qret", "bench_qret"])
    pan = R.retrail(pan[["wficn", "quarter", "as_min", "qret",
                         "bench_qret", "flowq"]])
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    R.summarize(sp, log, "CONSTANT BENCHMARK SET (S5/R1/R2/RM only)")
    pf = {w: g.set_index("quarter") for w, g in pan.groupby("wficn")}
    dt = R.build_dt(sp, pf)
    R.slim_fit(dt, R.SLIM, "event", log, "capitulation")
    R.slim_fit(dt, R.SLIM, "event_die", log, "death")
    log.append("  reading: the minimand no longer grows across eras; if the "
               "era HR still < 1 the growing-benchmark-set critique is "
               "answered.")

# --------------------------- (d) Petajisto official AS, 1980-2009 + checks ----
def sect_petajisto():
    pet = pd.read_parquet(P.CACHE / "pet_panel.parquet")
    pet["quarter"] = pd.PeriodIndex(pet["quarter"], freq="Q")
    for c in ("indexfund", "enhanced_index", "tna"):
        if c in pet.columns:
            pet[c] = pd.to_numeric(pet[c], errors="coerce")
    for c in ("indexfund", "enhanced_index"):
        if c in pet.columns:
            pet = pet[pet[c].fillna(0) != 1]
    pet = (pet.sort_values(["wficn", "quarter", "tna"])
              .drop_duplicates(["wficn", "quarter"], keep="last"))

    # systematic cross-check (critique 22): our ND min-AS vs Petajisto's
    nd = panel[["wficn", "quarter", "as_min"]].dropna()
    j = nd.merge(pet[["wficn", "quarter", "activeshare",
                      "activeshare_min"]], on=["wficn", "quarter"],
                 how="inner").dropna(subset=["activeshare"])
    if len(j):
        log.append(f"  overlap with Petajisto file: {len(j):,} fund-quarters, "
                   f"{j['wficn'].nunique():,} funds")
        log.append(f"    corr(our as_min, Petajisto official AS)  "
                   f"{j['as_min'].corr(j['activeshare']):.3f} | "
                   f"mean abs diff {(j['as_min'] - j['activeshare']).abs().mean():.3f}")
        jm = j.dropna(subset=["activeshare_min"])
        if len(jm):
            log.append(f"    corr(our as_min, Petajisto min AS)       "
                       f"{jm['as_min'].corr(jm['activeshare_min']):.3f} | "
                       f"mean abs diff "
                       f"{(jm['as_min'] - jm['activeshare_min']).abs().mean():.3f}")

    # official-benchmark spell rerun, 1980-2009
    pm = pet[["wficn", "quarter", "activeshare", "qret", "bench_qret"]].copy()
    pm = pm.rename(columns={"activeshare": "as_min"})
    pm = pm.dropna(subset=["as_min", "qret", "bench_qret"])
    pm["flowq"] = np.nan
    pan = R.retrail(pm)
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    R.summarize(sp, log, "PETAJISTO OFFICIAL-BENCHMARK AS, 1980-2009")
    log.append("  compare the 1980-94 vs 1995-2009 capitulation rates to the "
               "same two rows of the baseline table: same direction = the "
               "era trend is not a min-AS artifact.")

# ------------------------------------- (e) reconstructed vs CPZ actuals ----
def sect_validate_bench():
    ser = pd.read_parquet(P.CACHE / "bench_series_monthly.parquet")
    ser["month"] = pd.PeriodIndex(ser["month"], freq="M")
    cpz = P.load_cpz_monthly(log)
    cpz["monthp"] = cpz["month"].dt.to_period("M")
    cpzi = cpz.set_index("monthp")
    log.append("  reconstructed (ours) vs CPZ actual index returns, "
               "overlap months:")
    for code, col in (("S5", "idx_s5"), ("R2", "idx_r2"), ("RM", "idx_rm")):
        a = ser[ser["code"] == code].set_index("month")["ret"]
        j = pd.concat([a, cpzi[col]], axis=1, join="inner").dropna()
        j.columns = ["ours", "cpz"]
        if not len(j):
            log.append(f"    {code}: no overlap")
            continue
        d = j["ours"] - j["cpz"]
        log.append(f"    {code}: {len(j):3d} months "
                   f"({j.index.min()}..{j.index.max()}) | corr "
                   f"{j['ours'].corr(j['cpz']):.4f} | mean diff "
                   f"{d.mean() * 1e4:+.1f} bps/m | TE {d.std() * 1e4:.1f} "
                   f"bps/m | max |diff| {d.abs().max() * 1e4:.0f} bps")
    log.append("  reading: TE under ~30 bps/m and near-1 correlation puts "
               "reconstruction error an order of magnitude below spell-entry "
               "thresholds. Publishing-grade check vs vendor-published "
               "series still worth doing when Morningstar Direct is back.")

# ------------------------------------------ (f) percentile thresholds ----
def sect_percentile():
    bm = BPI["bench_min"].rename("bmin").reset_index()
    pan = panel.merge(bm, on=["wficn", "quarter"], how="left")
    pan["yr"] = pan["quarter"].dt.year
    pan["pr"] = (pan.groupby(["bmin", "yr"])["as_min"].rank(pct=True))
    # map percentile onto the fixed thresholds: tercile 2/3 -> 0.70 (entry),
    # tercile 1/3 -> 0.60 (capitulation), monotone linear in between
    pan["as_min"] = 0.60 + (pan["pr"] - 1.0 / 3.0) * 0.30
    pan = pan[["wficn", "quarter", "as_min", "qret", "bench_qret",
               "flowq", "rel4q"]]
    sp = R.attach_death(PL.extract_spells(pan, client_cut=None), death)
    R.summarize(sp, log, "WITHIN BENCHMARK-YEAR TERCILE THRESHOLDS "
                         "(entry top tercile, capitulation bottom tercile)")
    log.append("  reading: thresholds are now relative to the same "
               "benchmark-year peer group, so cap-size and concentration "
               "effects on AS levels cannot drive the era pattern. TE-based "
               "joint definition still to come (Petajisto TE 1980-2009; "
               "daily-return TE for the ND era).")

R.section(log, "(b) FROZEN-AT-ENTRY BENCHMARK (critique 3i)", sect_frozen)
R.section(log, "(c) CONSTANT BENCHMARK SET (critique 3ii)", sect_constant)
R.section(log, "(d) PETAJISTO OFFICIAL AS + SYSTEMATIC CROSS-CHECK "
               "(critiques 3iii, 22)", sect_petajisto)
R.section(log, "(e) RECONSTRUCTED vs CPZ ACTUALS (critique 18)",
          sect_validate_bench)
R.section(log, "(f) PERCENTILE THRESHOLDS (critique 17, partial)",
          sect_percentile)

log.append("\nBATTERY II DONE - aggregates only.")
P.write_report("referee_18_benchmarks.txt", log)
print("\n".join(log))
