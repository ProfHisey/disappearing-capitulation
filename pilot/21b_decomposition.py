"""Stage 21b: THE DECOMPOSITION — trading versus drift (referee critique 2).

The last mandatory test. For each capitulation with holdings around the
crossing, three Active Share numbers are computed from the same machinery:

  AS_obs(t-1)  the fund's actual portfolio vs the benchmark, quarter before
  AS_drift(t)  the NO-TRADE counterfactual: hold every share from t-1, let
               prices move it, compare to the benchmark at t
  AS_obs(t)    the actual portfolio at the crossing quarter t

Then the observed drop splits exactly:
  drift component   = AS_drift(t) - AS_obs(t-1)   (what prices alone did)
  trading component = AS_obs(t)  - AS_drift(t)    (what the fund chose)

Placebos: the same computation for (i) the same capitulators two quarters
earlier (before the plunge) and (ii) fighters at their eighth underwater
quarter. If capitulation is chosen, the trading component should dominate
at the crossing and be near zero in both placebos.

Validation: our computed AS_obs is correlated against the Notre Dame
as_min at the same fund-quarters (levels can differ by benchmark set;
the correlation should be high).

Inputs: cache/holdings_target.parquet (21a), crsp_stock/crsp_monthly.csv,
russell/idx_holdings_us.csv, the panel, and the per-benchmark AS panel.
Runtime: ~10-30 minutes. Output: output/decomposition_report.txt
(aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["TRADING vs DRIFT DECOMPOSITION (stage 21b)", "=" * 60]

# --------------------------------------------------- events ----
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

bp = pd.read_parquet(P.CACHE / "as_bench_panel.parquet",
                     columns=["wficn", "month", "total_assets", "bench_min"])
bp["quarter"] = pd.to_datetime(bp["month"]).dt.to_period("Q")
bp = (bp.sort_values(["wficn", "quarter", "total_assets"])
        .drop_duplicates(["wficn", "quarter"], keep="last")
        .set_index(["wficn", "quarter"])["bench_min"])

# ND as_min lookup for validation
ndas = panel.set_index(["wficn", "quarter"])["as_min"]

# benchmark code -> Russell weight column family
TO_RUSS = {"R1": "R1", "R1G": "R1G", "R1V": "R1V", "R2": "R2", "R2G": "R2G",
           "R2V": "R2V", "R3": "R3", "R3G": "R3G", "R3V": "R3V",
           "RM": "RM", "RMG": "RMG", "RMV": "RMV",
           "S5": "R1", "S5G": "R1G", "S5V": "R1V", "S4": "RM", "S4G": "RMG",
           "S4V": "RMV", "S6": "R2", "S6G": "R2G", "S6V": "R2V",
           "DJ": "R1", "W5": "R3", "W4": "RM"}
CODE_WT = {"R3": "r3000_wt", "R3G": "r3000g_wt", "R3V": "r3000v_wt",
           "R1": "r1000_wt", "R1G": "r1000g_wt", "R1V": "r1000v_wt",
           "R2": "r2000_wt", "R2G": "r2000g_wt", "R2V": "r2000v_wt",
           "RM": "rmidc_wt", "RMG": "rmidcg_wt", "RMV": "rmidcv_wt"}

# audit fix A1: crossing quarters come from the CALENDAR stamp recorded by
# extract_spells (m_cal_q), never start + m_dur arithmetic, which lands
# before the true crossing for spells containing reporting gaps. Fighter
# milestones use the fund's 8th OBSERVED underwater quarter's calendar label.
PFQ = {w: pd.PeriodIndex(g["quarter"].sort_values())
       for w, g in panel.groupby("wficn")}

def obs_q(w, start, k):
    qs = PFQ.get(w)
    if qs is None:
        return start + k
    qs = qs[qs >= start]
    return qs[k] if k < len(qs) else start + k

events = []
n_misdate = 0
for _, s in sp[sp["capitulated"]].iterrows():
    qc = pd.Period(s["m_cal_q"], freq="Q")
    if qc != s["start_p"] + int(s["m_dur"]):
        n_misdate += 1
    events.append((s["wficn"], qc - 1, qc, "crossing"))
    events.append((s["wficn"], qc - 3, qc - 2, "cap_pre_placebo"))
fight = sp[(sp["end_dur"] >= 8) & (sp["m_dur"].isna() | (sp["m_dur"] > 8))]
for _, s in fight.iterrows():
    if s["wficn"] % 7 == 0:                       # the 21a fighter sample
        q8 = obs_q(s["wficn"], s["start_p"], 8)
        events.append((s["wficn"], q8 - 1, q8, "fighter_placebo"))
log.append(f"crossings whose start+dur arithmetic misdated the event "
           f"(A1, now fixed): {n_misdate:,} of "
           f"{int(sp['capitulated'].sum()):,}")
ev = pd.DataFrame(events, columns=["wficn", "q0", "q1", "grp"])
log.append(f"candidate events: "
           + ", ".join(f"{g} {n}" for g, n in ev["grp"].value_counts().items()))

# --------------------------------------------------- holdings ----
hold = pd.read_parquet(
    P.CACHE / "holdings_target.parquet",
    columns=["wficn", "crsp_portno", "report_dt", "eff_dt", "percent_tna",
             "market_val", "permno", "cusip", "rq"])
hold = hold.dropna(subset=["wficn", "rq", "permno"])
hold["permno"] = pd.to_numeric(hold["permno"], errors="coerce")
hold = hold.dropna(subset=["permno"])
hold["permno"] = hold["permno"].astype("int64")
hold["market_val"] = pd.to_numeric(hold["market_val"], errors="coerce")
hold["percent_tna"] = pd.to_numeric(hold["percent_tna"], errors="coerce")
hold["report_dt"] = pd.to_datetime(hold["report_dt"], errors="coerce")
hold["eff_dt"] = pd.to_datetime(hold["eff_dt"], errors="coerce")
hold["rq"] = pd.PeriodIndex(hold["rq"], freq="Q")
# one snapshot per fund-quarter: latest report, then latest eff, then the
# portno with the largest reported value if several portnos coexist
snap_key = (hold.groupby(["wficn", "rq"])
                .agg(rd=("report_dt", "max")).reset_index())
hold = hold.merge(snap_key, on=["wficn", "rq"])
hold = hold[hold["report_dt"] == hold["rd"]]
# audit fix A3: keep NaT-eff_dt rows ONLY when the whole fund-quarter is
# undated; the old rule kept them alongside the max-eff_dt vintage, so a
# mixed fund-quarter summed two report vintages per position.
_na = hold.assign(_isna=hold["eff_dt"].isna()) \
          .groupby(["wficn", "rq"])["_isna"].agg(["any", "all"])
log.append(f"fund-quarters mixing dated and undated eff_dt vintages "
           f"(A3, now excluded from the dated snapshot): "
           f"{(_na['any'] & ~_na['all']).mean():.2%}")
ed = hold.groupby(["wficn", "rq"])["eff_dt"].transform("max")
hold = hold[(hold["eff_dt"] == ed) | (hold["eff_dt"].isna() & ed.isna())]
pv = (hold.groupby(["wficn", "rq", "crsp_portno"])["market_val"]
          .sum().reset_index())
pv = pv.sort_values("market_val").drop_duplicates(["wficn", "rq"],
                                                  keep="last")
hold = hold.merge(pv[["wficn", "rq", "crsp_portno"]],
                  on=["wficn", "rq", "crsp_portno"])
hold = (hold.groupby(["wficn", "rq", "permno"])
            .agg(mv=("market_val", "sum"), pt=("percent_tna", "sum"),
                 rd=("report_dt", "max"),
                 cusip=("cusip", "first")).reset_index())
cus = hold["cusip"].astype(str).str.strip().str.upper().str[:8]
hold["sid"] = np.where(cus.str.len() >= 6, cus,
                       "P" + hold["permno"].astype(str))
log.append(f"holdings snapshots: "
           f"{hold.groupby(['wficn', 'rq']).ngroups:,} fund-quarters after "
           f"dedup")
HG = {k: g for k, g in hold.groupby(["wficn", "rq"])}

# --------------------------------------------------- stock returns ----
_stk_path = P.SOURCES / "crsp_stock" / "crsp_monthly.csv"
_stk_head = pd.read_csv(_stk_path, nrows=0, encoding="latin-1")
_stk_map = {str(c).strip().lower(): c for c in _stk_head.columns}
stk = pd.read_csv(_stk_path,
                  usecols=[_stk_map["permno"], _stk_map["mthcaldt"],
                           _stk_map["mthret"]],
                  low_memory=False, encoding="latin-1")
stk = P.norm_cols(stk)
stk["m"] = pd.to_datetime(stk["mthcaldt"], errors="coerce").dt.to_period("M")
stk["mthret"] = pd.to_numeric(stk["mthret"], errors="coerce")
stk = stk.dropna(subset=["m", "mthret"])
stk = stk.drop_duplicates(["permno", "m"], keep="last")
RET = stk.set_index(["permno", "m"])["mthret"].sort_index()
log.append(f"stock returns: {len(stk):,} permno-months")

# --------------------------------------------------- bench weights ----
need_q = set(ev["q0"]) | set(ev["q1"])
need_m = {q.asfreq("M", how="end") for q in need_q}
idx_head = P.norm_cols(pd.read_csv(P.SOURCES / "russell" /
                                   "idx_holdings_us.csv", nrows=0,
                                   encoding="latin-1"))
idc = next((c for c in idx_head.columns if c == "permno"), None)
if idc is None:
    idc = next((c for c in idx_head.columns if "cusip" in c), None)
log.append(f"index constituent id column: {idc}")
use = ["date", idc] + list(CODE_WT.values())
parts = []
for chunk in pd.read_csv(P.SOURCES / "russell" / "idx_holdings_us.csv",
                         usecols=lambda c: str(c).strip().lower() in use,
                         chunksize=1_000_000, low_memory=False,
                         encoding="latin-1"):
    chunk = P.norm_cols(chunk)
    chunk["m"] = pd.to_datetime(chunk["date"], errors="coerce") \
                   .dt.to_period("M")
    parts.append(chunk[chunk["m"].isin(need_m)])
idx = pd.concat(parts, ignore_index=True)
if idc == "permno":
    idx["sid"] = pd.to_numeric(idx["permno"], errors="coerce")
else:
    idx["sid"] = idx[idc].astype(str).str.strip().str.upper().str[:8]
idx = idx.dropna(subset=["sid"])
log.append(f"benchmark constituent rows kept: {len(idx):,} across "
           f"{idx['m'].nunique()} months")

def bench_weights(code, q):
    wcol = CODE_WT.get(code)
    if wcol is None or wcol not in idx.columns:
        return None
    m = q.asfreq("M", how="end")
    d = idx[idx["m"] == m]
    w = pd.to_numeric(d[wcol], errors="coerce")
    keep = w.notna() & (w > 0)
    if keep.sum() < 50:
        return None
    s = pd.Series(w[keep].values, index=d.loc[keep, "sid"].values)
    s = s.groupby(level=0).sum()
    return s / s.sum()

# --------------------------------------------------- the decomposition ----
def fund_weights(g):
    v = g[["sid", "permno", "mv", "pt"]].copy()
    v["val"] = v["mv"]
    if v["val"].notna().sum() < 5 or v["val"].fillna(0).sum() <= 0:
        v["val"] = v["pt"]
    v = v.dropna(subset=["val"])
    v = v[v["val"] > 0]
    if len(v) < 5:
        return None
    v = v.copy()
    v["w"] = v["val"] / v["val"].sum()
    return v[["sid", "permno", "w"]]

def collapse(v):
    s = v.groupby("sid")["w"].sum()
    return s / s.sum()

def active_share(wf, wb):
    u = wf.index.union(wb.index)
    return 0.5 * float((wf.reindex(u, fill_value=0.0)
                        - wb.reindex(u, fill_value=0.0)).abs().sum())

rows, val_pairs = [], []
miss_ret_shares, part_shares = [], []
for _, e in ev.iterrows():
    w, q0, q1, grp = e["wficn"], e["q0"], e["q1"], e["grp"]
    g0, g1 = HG.get((w, q0)), HG.get((w, q1))
    if g0 is None or g1 is None:
        continue
    vf0, vf1 = fund_weights(g0), fund_weights(g1)
    if vf0 is None or vf1 is None:
        continue
    code = TO_RUSS.get(str(bp.get((w, q1), bp.get((w, q0), "NA"))).upper())
    wb0, wb1 = bench_weights(code, q0), bench_weights(code, q1)
    if wb0 is None or wb1 is None:
        continue
    m0 = g0["rd"].iloc[0].to_period("M")
    m1 = g1["rd"].iloc[0].to_period("M")
    months = pd.period_range(m0 + 1, m1, freq="M")
    if len(months) == 0 or len(months) > 8:
        continue
    gross = np.ones(len(vf0))
    missing = partial = 0
    for j, pn in enumerate(vf0["permno"].to_numpy()):
        try:
            r = RET.loc[pn].reindex(months)
        except KeyError:
            missing += 1
            continue
        if r.isna().any():                 # audit A8: flat-price fill months
            partial += 1
        gross[j] = float((1 + r.fillna(0)).prod())
    miss_ret_shares.append(missing / len(vf0))
    part_shares.append(partial / len(vf0))
    vdrift = vf0.copy()
    vdrift["w"] = vdrift["w"].to_numpy() * gross
    vdrift["w"] = vdrift["w"] / vdrift["w"].sum()
    wf0s, wf1s, wds = collapse(vf0), collapse(vf1), collapse(vdrift)
    as0 = active_share(wf0s, wb0)
    as_drift = active_share(wds, wb1)
    as1 = active_share(wf1s, wb1)
    rows.append({"grp": grp, "wficn": w, "das": as1 - as0,
                 "drift": as_drift - as0, "trade": as1 - as_drift})
    for q, a in ((q0, as0), (q1, as1)):
        nd = ndas.get((w, q))
        if pd.notna(nd):
            val_pairs.append((a, float(nd)))

dc = pd.DataFrame(rows)
log.append(f"\nevents decomposed: "
           + ", ".join(f"{g} {n}" for g, n in dc["grp"].value_counts().items())
           if len(dc) else "\nNO events decomposed - check id matching")
log.append(f"mean share of positions with permno absent from CRSP: "
           f"{np.mean(miss_ret_shares):.1%}" if miss_ret_shares else "")
log.append(f"mean share of positions with a PARTIALLY missing return roll "
           f"(gap months filled at 0% - audit A8 disclosure): "
           f"{np.mean(part_shares):.1%}" if part_shares else "")

if len(dc):
    vp = pd.DataFrame(val_pairs, columns=["ours", "nd"])
    log.append(f"validation: corr(our AS, ND as_min) = "
               f"{vp['ours'].corr(vp['nd']):.3f} over {len(vp):,} "
               f"fund-quarters | mean ours {vp['ours'].mean():.2f} vs ND "
               f"{vp['nd'].mean():.2f}")
    if vp["ours"].corr(vp["nd"]) < 0.5:
        log.append("  *** WARNING: validation correlation < 0.5 - the "
                   "fund/index identifier matching likely broke (audit "
                   "landmine: permno-vs-cusip sid branch). DO NOT use "
                   "these numbers. ***")
    log.append("\nDECOMPOSITION (means; AS points, negative = toward index):")
    log.append(f"  {'group':18s} {'n':>5s} {'dAS_obs':>9s} {'drift':>9s} "
               f"{'trading':>9s} {'trade share':>12s}")
    for g, d in dc.groupby("grp"):
        drops = d[d["das"] < -0.005]
        tsh = (drops["trade"] / drops["das"]).clip(-1, 2)
        log.append(f"  {g:18s} {len(d):5,} {d['das'].mean():+9.3f} "
                   f"{d['drift'].mean():+9.3f} {d['trade'].mean():+9.3f} "
                   f"{tsh.median() if len(drops) else float('nan'):12.1%}")
    cx = dc[dc["grp"] == "crossing"]
    if len(cx):
        drops = cx[cx["das"] < -0.005]
        dom = (drops["trade"] < drops["drift"]).mean() if len(drops) else np.nan
        log.append(f"\n  crossing events with a real AS drop: {len(drops):,}; "
                   f"trading component exceeds drift in {dom:.0%} of them")
        log.append(f"  trading share of the drop: p25 "
                   f"{(drops['trade'] / drops['das']).quantile(.25):.0%} | "
                   f"median {(drops['trade'] / drops['das']).median():.0%} | "
                   f"p75 {(drops['trade'] / drops['das']).quantile(.75):.0%}")
    log.append("""
Reading guide: 'drift' is what the Active Share drop would have been had the
fund not traded at all (prices and index reconstitution only); 'trading' is
the remainder, attributable to the fund's own transactions. The critique-2
claim dies if trading dominates at crossings while both placebos show small,
drift-sized changes. Levels of our recomputed AS need not match Notre Dame's
(different benchmark handling); the correlation is the validity check.""")

log.append("DECOMPOSITION DONE - aggregates only.")
P.write_report("decomposition_report.txt", log)
print("\n".join(log))
