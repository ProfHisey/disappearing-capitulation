"""Stage 18b: BATTERY II FOLLOW-UP — frozen-benchmark forensics + the R2 gap.

Battery II left two loose ends that need diagnosis before any conclusion:

 (i) FROZEN BENCHMARK. The frozen definition kept only 338 of 693
     capitulations and flattened the early era. Two very different stories
     fit that pattern: genuine benchmark-reassignment crossings (the
     referee's story), or the entry benchmark's AS column simply not being
     observed at the crossing quarter (a tracking artifact). This section
     classifies every baseline crossing into frozen-confirmed /
     reassignment / untracked, by era, and re-runs the frozen era table on
     well-tracked spells only.
 (ii) R2/RM RECONSTRUCTION GAP. Ours-vs-CPZ showed mean +44 bps/m on R2.
     The pre-fill months are literal CPZ copies (diff = 0), so the gap is
     concentrated in the reconstructed segment. This section splits the
     comparison at the fill boundary (identified empirically by diff == 0),
     reports the reconstructed segment's stats, and triangulates with the
     French size portfolios to see whether ours or CPZ is the outlier.
 (iii) SCHEMA PROBE of the other files in russell/ (ftse_russell_us,
     rgs_sw_us, ...). If official published index RETURNS exist on disk, a
     v4 benchmark series can use them directly and retire the
     reconstruction question entirely. Prints column names and small-sample
     value ranges only (no data leaves the machine).

Output: output/referee_18b_diagnostics.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["BATTERY II FOLLOW-UP - FROZEN FORENSICS + R2 GAP", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp0 = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet")
bp["month"] = pd.to_datetime(bp["month"])
bp["quarter"] = bp["month"].dt.to_period("Q")
bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
        .drop_duplicates(["wficn", "quarter"], keep="last"))
BPI = bp.set_index(["wficn", "quarter"]).sort_index()
# audit round 2 (fix A1): observed-clock quarter lookup per fund
PFI = {w: g.set_index("quarter").index.sort_values()
       for w, g in panel.groupby("wficn")}

def obs_q18(w, start, t):
    qs = PFI.get(w)
    if qs is None or start not in qs:
        return start + t
    j = qs.get_loc(start) + t
    return qs[j] if 0 <= j < len(qs) else start + t

# ------------------------------- (i) frozen-benchmark forensics ----------
def sect_frozen_forensics():
    caps = sp0[sp0["capitulated"]]
    cls = {"frozen_cross": 0, "reassignment_cross": 0,
           "untracked_at_crossing": 0, "entry_missing": 0}
    by_era = {}
    for _, s in caps.iterrows():
        w, start = s["wficn"], s["start_p"]
        qc = pd.Period(s["m_cal_q"], freq="Q")  # audit fix A1 (round 2)
        era = next((f"{lo}-{hi}" for lo, hi in R.ERAS
                    if lo <= start.year <= hi), "?")
        if (w, start) not in BPI.index:
            kind = "entry_missing"
        else:
            b0 = str(BPI.at[(w, start), "bench_min"])
            col = "as_" + b0.lower()
            if col not in BPI.columns:
                kind = "entry_missing"
            elif (w, qc) in BPI.index and pd.notna(BPI.at[(w, qc), col]):
                v = float(BPI.at[(w, qc), col])
                kind = ("frozen_cross" if v < P.CLOSET_CUTOFF
                        else "reassignment_cross")
            else:
                kind = "untracked_at_crossing"
        cls[kind] += 1
        by_era.setdefault(era, {k: 0 for k in cls})[kind] += 1
    log.append(f"  {len(caps):,} baseline capitulation crossings classified:")
    for k, v in cls.items():
        log.append(f"    {k:22s} {v:4d} ({v / len(caps):.1%})")
    log.append("  by era (frozen / reassignment / untracked / entry-missing):")
    for era in sorted(by_era):
        d = by_era[era]
        tot = sum(d.values())
        log.append(f"    {era}: n {tot:3d} | "
                   f"{d['frozen_cross'] / tot:5.1%} / "
                   f"{d['reassignment_cross'] / tot:5.1%} / "
                   f"{d['untracked_at_crossing'] / tot:5.1%} / "
                   f"{d['entry_missing'] / tot:5.1%}")
    log.append("  reading: a large 'untracked' or 'entry_missing' share, "
               "concentrated early, means battery II's frozen row was a "
               "coverage artifact there, not reassignment.")

    # frozen scan with tracking coverage, era table on well-tracked spells
    rows = []
    for _, s in sp0.iterrows():
        w, start = s["wficn"], s["start_p"]
        if (w, start) not in BPI.index:
            continue
        b0 = str(BPI.at[(w, start), "bench_min"])
        col = "as_" + b0.lower()
        if col not in BPI.columns:
            continue
        T = int(s["end_dur"])
        n_obs, m_frozen = 0, np.nan
        for t in range(1, T + 1):
            k = (w, obs_q18(w, start, t))     # audit fix A1 (round 2)
            if k in BPI.index and pd.notna(BPI.at[k, col]):
                n_obs += 1
                if (np.isnan(m_frozen)
                        and float(BPI.at[k, col]) < P.CLOSET_CUTOFF):
                    m_frozen = t
        rows.append({"idx": s.name, "track": n_obs / max(T, 1),
                     "frozen_cap": pd.notna(m_frozen)})
    fr = pd.DataFrame(rows).set_index("idx")
    sp = sp0.join(fr, how="inner")
    well = sp[sp["track"] >= 0.8].copy()
    log.append(f"\n  spells with entry benchmark trackable: {len(sp):,}; "
               f"well-tracked (>=80% of quarters observed): {len(well):,}")
    well["capitulated"] = well["frozen_cap"]
    well["spell_died"] = well["spell_died"] & ~well["capitulated"]
    R.summarize(well, log, "FROZEN definition, WELL-TRACKED spells only")
    log.append("  compare to battery II's unrestricted frozen table: if the "
               "era decline reappears here, coverage was the culprit.")

# --------------------------------- (ii) R2/RM gap: split + triangulate ----
def sect_r2_gap():
    ser = pd.read_parquet(P.CACHE / "bench_series_monthly.parquet")
    ser["month"] = pd.PeriodIndex(ser["month"], freq="M")
    cpz = P.load_cpz_monthly(log)
    cpz["m"] = cpz["month"].dt.to_period("M")
    cpzi = cpz.set_index("m")
    f6 = PL.parse_french_first_block(PL.F_6PORT)
    small = [c for c in f6.columns if "SMALL" in str(c).upper()
             or str(c).upper().startswith("ME1")]
    big = [c for c in f6.columns if "BIG" in str(c).upper()
           or str(c).upper().startswith("ME2")]
    f6["p_small"] = f6[small].mean(axis=1)
    f6["p_big"] = f6[big].mean(axis=1)
    f6["m"] = f6["month"].dt.to_period("M")
    f6i = f6.set_index("m")

    for code, cpzcol, frcol in (("R2", "idx_r2", "p_small"),
                                ("RM", "idx_rm", None),
                                ("S5", "idx_s5", "p_big")):
        a = ser[ser["code"] == code].set_index("month")["ret"]
        j = pd.concat([a.rename("ours"), cpzi[cpzcol].rename("cpz")],
                      axis=1, join="inner").dropna()
        d = j["ours"] - j["cpz"]
        fill = d.abs() < 1e-12
        rec = j[~fill]
        log.append(f"\n  {code}: {int(fill.sum())} fill months (identical to "
                   f"CPZ), {len(rec)} genuinely reconstructed overlap months")
        if len(rec):
            dr = rec["ours"] - rec["cpz"]
            log.append(f"    reconstructed segment "
                       f"{rec.index.min()}..{rec.index.max()}: mean diff "
                       f"{dr.mean() * 1e4:+.1f} bps/m | TE {dr.std() * 1e4:.1f}"
                       f" bps/m | corr {rec['ours'].corr(rec['cpz']):.4f}")
            if frcol is not None:
                fj = f6i[frcol].reindex(rec.index)
                log.append(f"    triangulation vs French {frcol}: "
                           f"corr(ours, French) "
                           f"{rec['ours'].corr(fj):.4f} vs corr(CPZ, French) "
                           f"{rec['cpz'].corr(fj):.4f} | mean(ours-French) "
                           f"{(rec['ours'] - fj).mean() * 1e4:+.1f} bps/m | "
                           f"mean(CPZ-French) "
                           f"{(rec['cpz'] - fj).mean() * 1e4:+.1f} bps/m")
    log.append("\n  reading: if ours-vs-French shows the same positive bias "
               "while CPZ-vs-French does not, the reconstruction is the "
               "outlier and needs replacing with official returns (see iii). "
               "Note CPZ's post-2011 extension is itself less authoritative.")

# ----------------------------- (iii) probe for official return files ----
def sect_probe():
    rdir = P.SOURCES / "russell"
    for f in sorted(rdir.glob("*")):
        if f.is_dir():
            continue
        sz = f.stat().st_size / 1e6
        log.append(f"\n  {f.name} ({sz:.1f} MB)")
        if f.suffix.lower() != ".csv":
            log.append("    (non-csv, skipped)")
            continue
        if f.name == "idx_holdings_us.csv":
            log.append("    (holdings file, already characterized)")
            continue
        try:
            head = P.norm_cols(pd.read_csv(f, nrows=0))
            log.append(f"    columns: {list(head.columns)[:40]}")
            samp = P.norm_cols(pd.read_csv(f, nrows=5000, low_memory=False))
            retcols = [c for c in samp.columns
                       if "ret" in c or "tr_" in c or c.endswith("_tr")]
            idcols = [c for c in samp.columns
                      if any(k in c for k in ("index", "idx", "code", "name"))]
            for c in idcols[:3]:
                vals = samp[c].astype(str).value_counts().head(12)
                log.append(f"    {c} top values: {list(vals.index)}")
            for c in retcols[:6]:
                v = pd.to_numeric(samp[c], errors="coerce")
                log.append(f"    {c}: non-null {v.notna().mean():.0%}, "
                           f"range [{v.min():.4g}, {v.max():.4g}]")
            dcols = [c for c in samp.columns if "date" in c or c == "caldt"]
            for c in dcols[:2]:
                dd = pd.to_datetime(samp[c], errors="coerce")
                log.append(f"    {c}: {dd.min()} .. {dd.max()} (first 5k rows)")
        except Exception as e:  # noqa: BLE001
            log.append(f"    probe failed: {e}")
    log.append("\n  reading: if an official monthly (or daily) total-return "
               "column exists per Russell index, benchmark series v4 uses it "
               "directly and the reconstruction question is retired.")

R.section(log, "(i) FROZEN-BENCHMARK FORENSICS", sect_frozen_forensics)
R.section(log, "(ii) R2/RM GAP: FILL SPLIT + FRENCH TRIANGULATION",
          sect_r2_gap)
R.section(log, "(iii) SCHEMA PROBE: official Russell return files?",
          sect_probe)

log.append("\n18b DONE - aggregates only.")
P.write_report("referee_18b_diagnostics.txt", log)
print("\n".join(log))
