"""Stage 21a: HOLDINGS EXTRACTION — the overnight half of the decomposition.

The referee panel's one mandatory remaining test (critique 2) decomposes
each pre-capitulation Active Share drop into TRADING (the fund selling its
active positions) versus DRIFT (losing positions shrinking toward benchmark
weights by price alone). That needs portfolio holdings, and the CRSP
holdings file is ~53 GB, far too slow to touch repeatedly.

This stage reads it exactly once and caches the rows for a target set of
funds:

  group A: all capitulators (the 496 funds with a crossing)
  group B: a deterministic sample of fighters (still active at quarter 8)
  group C: a deterministic sample of other panel funds (controls)

plus schema probes of the stock-return files, so stage 21b (the actual
decomposition) can be written against known columns and run in minutes.

RESUMABLE: progress is saved per chunk under cache/holdings_parts/. If the
run is interrupted, rerun the same command and it continues where it left
off. Expect one to several hours on the full file; disk-heavy, safe to
leave unattended. Output report: output/holdings_extract_report.txt.
"""
import glob
import os

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["HOLDINGS EXTRACTION (stage 21a)", "=" * 60]

PARTS = P.CACHE / "holdings_parts"
PARTS.mkdir(exist_ok=True)
CHUNK = 2_000_000

# ------------------------------------------------ target fund groups ----
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

caps = set(sp.loc[sp["capitulated"], "wficn"])
fight = set(sp.loc[(sp["end_dur"] >= 8)
                   & (sp["m_dur"].isna() | (sp["m_dur"] > 8)), "wficn"])
fight_s = {w for w in fight if w % 7 == 0} - caps
others = {int(w) for w in panel["wficn"].unique() if w % 13 == 0}
others = others - caps - fight_s
TARGET_W = caps | fight_s | others
log.append(f"target funds: {len(caps):,} capitulators + {len(fight_s):,} "
           f"fighter sample + {len(others):,} controls = {len(TARGET_W):,}")

# wficn -> crsp_portno via mflink1 (fundno) + Fund-Portfolio Map
m1 = PL.get_mflink1()
fmap = P.norm_cols(pd.read_csv(P.F_MAP, low_memory=False, encoding="latin-1"))
fcol = next(c for c in fmap.columns if "fundno" in c)
pcol = next(c for c in fmap.columns if "portno" in c)
fmap = fmap[[fcol, pcol]].dropna()
fmap[fcol] = pd.to_numeric(fmap[fcol], errors="coerce").astype("Int64")
fmap[pcol] = pd.to_numeric(fmap[pcol], errors="coerce").astype("Int64")
link = fmap.merge(m1, left_on=fcol, right_on="crsp_fundno", how="inner")
link = link[link["wficn"].isin(TARGET_W)]
PORT_W = dict(zip(link[pcol].astype("int64"), link["wficn"].astype("int64")))
TARGET_P = set(PORT_W)
log.append(f"mapped to {len(TARGET_P):,} portfolio numbers "
           f"({link['wficn'].nunique():,} of {len(TARGET_W):,} target funds "
           f"have a portno)")
pd.DataFrame({"portno": list(PORT_W), "wficn": [PORT_W[k] for k in PORT_W]}) \
  .to_parquet(P.CACHE / "holdings_portno_map.parquet", index=False)

# ------------------------------------------------ locate the big file ----
cands = [f for f in glob.glob(str(P.CRSP_DIR / "*.csv"))
         if "olding" in os.path.basename(f).lower()
         and "map" not in os.path.basename(f).lower()]
if not cands:
    P.fail("no holdings csv found in crsp_mf (looked for *olding*.csv)")
F_HOLD = max(cands, key=os.path.getsize)
log.append(f"holdings file: {os.path.basename(F_HOLD)} "
           f"({os.path.getsize(F_HOLD) / 1e9:.1f} GB)")

head = P.norm_cols(pd.read_csv(F_HOLD, nrows=0, encoding="latin-1"))
log.append(f"holdings columns: {list(head.columns)}")
hp = next((c for c in head.columns if "portno" in c), None)
if hp is None:
    P.fail("no portno column in holdings file - paste the columns line back")

# ------------------------------------------------ chunked filter pass ----
done = sorted(PARTS.glob("part_*.parquet"))
start_chunk = len(done)
if start_chunk:
    log.append(f"RESUME: {start_chunk} chunks already done, continuing")

kept_total = sum(len(pd.read_parquet(f, columns=[])) for f in done) \
    if done else 0
reader = pd.read_csv(F_HOLD, chunksize=CHUNK, low_memory=False,
                     encoding="latin-1")
for i, chunk in enumerate(reader):
    if i < start_chunk:
        continue
    chunk = P.norm_cols(chunk)
    chunk[hp] = pd.to_numeric(chunk[hp], errors="coerce")
    keep = chunk[chunk[hp].isin(TARGET_P)]
    kept_total += len(keep)
    keep.to_parquet(PARTS / f"part_{i:05d}.parquet", index=False)
    if i % 10 == 0:
        msg = (f"  chunk {i}: scanned ~{(i + 1) * CHUNK / 1e6:.0f}M rows, "
               f"kept {kept_total:,}")
        print(msg, flush=True)
        log.append(msg)
log.append(f"filter pass complete: {kept_total:,} holdings rows kept")

# ------------------------------------------------ consolidate + report ----
parts = sorted(PARTS.glob("part_*.parquet"))
hold = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
hold["wficn"] = hold[hp].map(PORT_W)
# audit hardening: prefer report_dt explicitly, never by column order
dtc = None
for pref in ("report_dt", "eff_dt", "caldt"):
    dtc = next((c for c in hold.columns if pref in c), None)
    if dtc:
        break
if dtc is None and "date" in hold.columns:
    dtc = "date"
log.append(f"holdings snapshot date column chosen: {dtc!r} "
           f"(preference order report_dt > eff_dt > caldt > date, hardened "
           f"post-audit)")
if dtc is not None:
    hold["rq"] = pd.to_datetime(hold[dtc], errors="coerce").dt.to_period("Q")
hold.to_parquet(P.CACHE / "holdings_target.parquet", index=False)
log.append(f"wrote cache/holdings_target.parquet "
           f"({len(hold):,} rows, {hold['wficn'].nunique():,} funds)")

if dtc is not None:
    grp = pd.Series("control", index=hold.index)
    grp[hold["wficn"].isin(fight_s)] = "fighter"
    grp[hold["wficn"].isin(caps)] = "capitulator"
    hold["grp"] = grp
    log.append("\ncoverage by group:")
    for g, d in hold.groupby("grp"):
        nrep = d.groupby(["wficn", "rq"]).size()
        log.append(f"  {g}: {d['wficn'].nunique():,} funds | "
                   f"{nrep.index.get_level_values('rq').min()}..".replace("NaT", "?")
                   + f"{nrep.index.get_level_values('rq').max()} | "
                   f"median positions/report {nrep.median():.0f}")
    caps_cov = 0
    for _, s in sp[sp["capitulated"]].iterrows():
        qc = pd.Period(s["m_cal_q"], freq="Q")  # audit fix A1 (round 2)
        h = hold[(hold["wficn"] == s["wficn"])
                 & hold["rq"].isin([qc - 2, qc - 1, qc])]
        if h["rq"].nunique() >= 2:
            caps_cov += 1
    log.append(f"\ncapitulations with holdings at >=2 of the 3 quarters "
               f"around the crossing: {caps_cov} of "
               f"{int(sp['capitulated'].sum())} - this is stage 21b's "
               f"usable sample")

# ------------------------------------------------ stock-file probes ----
log.append("\nSTOCK-RETURN FILE PROBES (for stage 21b):")
sdir = P.SOURCES / "crsp_stock"
if sdir.exists():
    for f in sorted(sdir.glob("*.csv")):
        try:
            h = P.norm_cols(pd.read_csv(f, nrows=5000, low_memory=False,
                                        encoding="latin-1"))
            log.append(f"  {f.name} ({os.path.getsize(f) / 1e6:.0f} MB): "
                       f"{list(h.columns)[:20]}")
            dc = next((c for c in h.columns if "date" in c or c == "caldt"),
                      None)
            if dc:
                dd = pd.to_datetime(h[dc], errors="coerce")
                log.append(f"    {dc}: {dd.min()}..{dd.max()} (first 5k rows)")
        except Exception as e:  # noqa: BLE001
            log.append(f"  {f.name}: probe failed ({e})")
else:
    log.append("  crsp_stock folder not found - stage 21b will need "
               "security returns from another source")

log.append("\nEXTRACTION DONE. Stage 21b (the decomposition itself) will be "
           "written against this cache and these schemas.")
P.write_report("holdings_extract_report.txt", log)
print("\n".join(log[-25:]))
