"""Stage 5: WHO BREAKS FIRST — manager vs. clients of the same fund (F1 draft).

For every underperformance spell (same definition as stage 04, restricted to
spells starting 2000+ where CRSP retail/institutional flags exist), track TWO
events on the same clock:
  MANAGER capitulation: min-AS first crosses below 60% (closet indexing).
  CLIENT capitulation:  retail share classes' quarterly net flow first hits
                        -10% or worse of retail TNA (a redemption spike).
                        (Sensitivity counts at -5% and -15% reported.)
Both arms share the spell's censoring (recovery above benchmark or data end).

Requires stage 04 to have run at least once (uses its caches). Outputs
(aggregates only): output/wbf_report.txt, who_breaks_first.png, wbf_survival.csv.

Pilot-grade caveats printed in the report: the two arms are PAIRED (same funds),
so the overlay logrank is descriptive; the real build uses paired/competing-risk
methods. Client arm uses share-class flags that begin Dec 1999.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

import pilot_lib as P

MFLINK1 = P.SOURCES / "mflinks" / "mflink1.csv"
FRENCH_DIR = Path(r"E:\Finance\BuyRisk\data\sources\french")
F_6PORT = FRENCH_DIR / "6_Portfolios_2x3.csv"
SPLICE_END = pd.Period("2011Q1", freq="Q")
CLIENT_CUT = -0.10          # primary client-capitulation threshold
CLIENT_SENS = (-0.05, -0.15)

log = ["WHO BREAKS FIRST (2000-2023, same funds, same spells)", "=" * 60]

# ---------------------------------------------------------------- links ----
m1 = P.norm_cols(pd.read_csv(MFLINK1))
fcol = next(c for c in m1.columns if "fundno" in c)
wcol = next(c for c in m1.columns if "wficn" in c)
m1 = (m1[[fcol, wcol]].dropna().astype("int64")
      .rename(columns={fcol: "crsp_fundno", wcol: "wficn"})
      .drop_duplicates("crsp_fundno"))

# --------------------------------------------- retail share-class flags ----
rf_pq = P.CACHE / "retail_flags.parquet"
if rf_pq.exists():
    rfl = pd.read_parquet(rf_pq)
else:
    parts = []
    use = ["crsp_fundno", "retail_fund", "inst_fund"]
    for chunk in pd.read_csv(P.F_SUMMARY, usecols=lambda c: c.strip().lower() in use,
                             chunksize=500_000, low_memory=False, encoding="latin-1"):
        parts.append(P.norm_cols(chunk))
    rfl = pd.concat(parts, ignore_index=True)
    rfl["is_retail"] = rfl["retail_fund"].astype(str).str.upper().eq("Y")
    rfl = rfl.groupby("crsp_fundno").agg(is_retail=("is_retail", "any")).reset_index()
    rfl.to_parquet(rf_pq, index=False)
log.append(f"retail-flagged share classes: {int(rfl['is_retail'].sum()):,} "
           f"of {len(rfl):,}")

# ------------------------------------- retail flows (share class level) ----
ret = P.load_monthly_returns(log)
ret = ret.merge(rfl, on="crsp_fundno", how="left").merge(m1, on="crsp_fundno", how="inner")
ret = ret[ret["is_retail"] == True]  # noqa: E712
ret = ret.sort_values(["crsp_fundno", "caldt"])
ret["tna_lag"] = ret.groupby("crsp_fundno")["mtna"].shift(1)
ret["flow"] = ret["mtna"] - ret["tna_lag"] * (1 + ret["mret"])
fm = (ret.dropna(subset=["flow"])
         .groupby(["wficn", "caldt"])
         .agg(flow=("flow", "sum"), tna=("mtna", "sum")).reset_index())
fm["quarter"] = fm["caldt"].dt.to_period("Q")
fq_flow = (fm.groupby(["wficn", "quarter"])
             .agg(flow=("flow", "sum"), tna_end=("tna", "last")).reset_index())
fq_flow = fq_flow.sort_values(["wficn", "quarter"])
fq_flow["tna_prev"] = fq_flow.groupby("wficn")["tna_end"].shift(1)
fq_flow["flowq"] = (fq_flow["flow"] / fq_flow["tna_prev"]).where(fq_flow["tna_prev"] > 0)
fq_flow["flowq"] = fq_flow["flowq"].clip(-1, 1)
fq_flow = fq_flow[["wficn", "quarter", "flowq"]].dropna()
log.append(f"retail flow observations: {len(fq_flow):,} wficn-quarters, "
           f"{fq_flow['wficn'].nunique():,} funds")

# ---------------- fund returns, AS panel, benchmarks (as in stage 04) ------
ret_all = P.load_monthly_returns([])
ret_all = ret_all.merge(m1, on="crsp_fundno", how="inner")
ret_all = ret_all.sort_values(["crsp_fundno", "caldt"])
ret_all["w"] = ret_all.groupby("crsp_fundno")["mtna"].shift(1)
ret_all["w"] = ret_all["w"].fillna(ret_all["mtna"]).clip(lower=0)
ret_all = ret_all.dropna(subset=["mret"])
ret_all["wr"] = ret_all["w"] * ret_all["mret"]
fund_m = (ret_all.groupby(["wficn", "caldt"])
                 .agg(wr=("wr", "sum"), w=("w", "sum")).reset_index())
fund_m["fret"] = np.where(fund_m["w"] > 0, fund_m["wr"] / fund_m["w"], np.nan)
fund_m = fund_m.dropna(subset=["fret"])
fund_m["quarter"] = fund_m["caldt"].dt.to_period("Q")
fq = (fund_m.assign(g=lambda d: 1 + d["fret"]).groupby(["wficn", "quarter"])
            .agg(qret=("g", lambda x: x.prod() - 1), nm=("g", "size")).reset_index())
fq = fq[fq["nm"] == 3].drop(columns="nm")

fl = pd.read_parquet(P.CACHE / "flags.parquet")
asp = pd.read_parquet(P.CACHE / "as_panel.parquet").dropna(subset=["wficn"])
asp["wficn"] = asp["wficn"].astype("int64")
asp["quarter"] = asp["month"].dt.to_period("Q")
asp = (asp.sort_values(["wficn", "quarter", "total_assets"])
          .drop_duplicates(["wficn", "quarter"], keep="last")
          .merge(fl, on="wficn", how="left"))
asp = asp[asp["passive"] != True]  # noqa: E712

import re  # noqa: E402

def parse_french_first_block(path: Path) -> pd.DataFrame:
    rows, header, started = [], None, False
    for ln in path.read_text(encoding="latin-1", errors="replace").splitlines():
        cells = [c.strip() for c in ln.split(",")]
        if header is None:
            if len(cells) >= 6 and cells[0] == "" and sum(bool(c) for c in cells[1:]) >= 5:
                header = cells[1:]
            continue
        if re.fullmatch(r"\d{6}", cells[0] or ""):
            started = True
            rows.append(cells)
        elif started:
            break
    df = pd.DataFrame(rows, columns=["ym"] + header)
    for c in header:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([-99.99, -999], np.nan) / 100
    df["month"] = P.parse_ym(df["ym"])
    return df

cpz = P.load_cpz_monthly(log)
f6 = parse_french_first_block(F_6PORT)
small = [c for c in f6.columns if "SMALL" in c.upper() or c.upper().startswith("ME1")]
big = [c for c in f6.columns if "BIG" in c.upper() or c.upper().startswith("ME2")]
f6["p_s5"], f6["p_r2"], f6["p_rm"] = (f6[big].mean(axis=1), f6[small].mean(axis=1),
                                      f6[big + small].mean(axis=1))

def to_q(df, cols):
    d = df.copy()
    d["quarter"] = d["month"].dt.to_period("Q")
    return d.set_index("quarter")[cols].add(1).groupby(level=0).prod().sub(1)

bq = pd.concat([
    to_q(cpz, ["idx_s5", "idx_r2", "idx_rm"]).loc[lambda d: d.index <= SPLICE_END],
    to_q(f6, ["p_s5", "p_r2", "p_rm"])
        .rename(columns={"p_s5": "idx_s5", "p_r2": "idx_r2", "p_rm": "idx_rm"})
        .loc[lambda d: d.index > SPLICE_END],
]).sort_index().reset_index()

asp["core"] = asp["bench_min"].astype(str).str.upper().map(
    {k: v.replace("idx_", "") for k, v in P.BENCH_TO_CORE.items()})
panel = (asp.merge(fq, on=["wficn", "quarter"], how="inner")
            .merge(bq, on="quarter", how="left")
            .merge(fq_flow, on=["wficn", "quarter"], how="left"))
panel["bench_qret"] = np.select(
    [panel["core"] == "s5", panel["core"] == "r2", panel["core"] == "rm"],
    [panel["idx_s5"], panel["idx_r2"], panel["idx_rm"]], default=np.nan)
panel = panel.dropna(subset=["as_min", "qret", "bench_qret"])
panel = panel.sort_values(["wficn", "quarter"])

def add_trailing(g):
    g = g.set_index("quarter").asfreq("Q")
    f = (1 + g["qret"]).rolling(4).apply(np.prod, raw=True) - 1
    b = (1 + g["bench_qret"]).rolling(4).apply(np.prod, raw=True) - 1
    g["rel4q"] = f - b
    return g.reset_index()

panel = (panel.groupby("wficn", group_keys=True)[
             ["quarter", "as_min", "qret", "bench_qret", "flowq"]]
         .apply(add_trailing).reset_index(level=0).reset_index(drop=True))

# ------------------------------------------- paired two-event spell scan ----
rows = []
sens = {t: 0 for t in CLIENT_SENS}
for wficn, g in panel.groupby("wficn"):
    g = g.reset_index(drop=True)
    in_spell = False
    for i in range(len(g)):
        r = g.loc[i]
        if not in_spell:
            if (pd.notna(r["rel4q"]) and r["rel4q"] < 0
                    and pd.notna(r["as_min"]) and r["as_min"] >= P.ACTIVE_START
                    and r["quarter"].year >= 2000 and pd.notna(r["flowq"])):
                in_spell = True
                start_i, m_dur, c_dur = i, None, None
        else:
            dur = i - start_i
            if m_dur is None and pd.notna(r["as_min"]) and r["as_min"] < P.CLOSET_CUTOFF:
                m_dur = dur
            if c_dur is None and pd.notna(r["flowq"]):
                if r["flowq"] <= CLIENT_CUT:
                    c_dur = dur
                for t in CLIENT_SENS:
                    if r["flowq"] <= t:
                        sens[t] += 1
            ended = ((pd.notna(r["rel4q"]) and r["rel4q"] >= 0)
                     or i == len(g) - 1 or pd.isna(r["as_min"]))
            if ended:
                end_dur = max(dur, 1)
                rows.append((wficn, g.loc[start_i, "quarter"].year, end_dur,
                             m_dur, c_dur))
                in_spell = False

sp = pd.DataFrame(rows, columns=["wficn", "start_yr", "end_dur", "m_dur", "c_dur"])
sp["m_event"] = sp["m_dur"].notna().astype(int)
sp["c_event"] = sp["c_dur"].notna().astype(int)
sp["m_time"] = sp["m_dur"].fillna(sp["end_dur"]).clip(lower=1)
sp["c_time"] = sp["c_dur"].fillna(sp["end_dur"]).clip(lower=1)
log.append(f"\nspells (2000+, both arms observable at start): {len(sp):,} "
           f"across {sp['wficn'].nunique():,} funds")
log.append(f"MANAGER capitulation events (min-AS < 60%): {int(sp['m_event'].sum()):,} "
           f"({sp['m_event'].mean():.1%} of spells)")
log.append(f"CLIENT capitulation events (retail flow <= {CLIENT_CUT:.0%}/q): "
           f"{int(sp['c_event'].sum()):,} ({sp['c_event'].mean():.1%})")
log.append(f"  sensitivity (quarter-events at thresholds, info): "
           + ", ".join(f"<= {t:.0%}: {n:,}" for t, n in sens.items()))

both = sp[(sp["m_event"] == 1) & (sp["c_event"] == 1)]
any_ = sp[(sp["m_event"] == 1) | (sp["c_event"] == 1)]
first = np.select(
    [sp["m_event"].eq(1) & (sp["c_event"].eq(0) | (sp["m_dur"] < sp["c_dur"])),
     sp["c_event"].eq(1) & (sp["m_event"].eq(0) | (sp["c_dur"] < sp["m_dur"])),
     sp["m_event"].eq(1) & sp["c_event"].eq(1) & (sp["m_dur"] == sp["c_dur"])],
    ["manager", "client", "tie"], default="neither")
log.append(f"\nWHO BREAKS FIRST (among {len(any_):,} spells with >=1 event):")
for lab in ("client", "manager", "tie"):
    n = (first == lab).sum()
    log.append(f"  {lab} first: {n:,} ({n / max(len(any_), 1):.1%})")
log.append(f"  both events in same spell: {len(both):,}")
if sp["m_event"].sum():
    log.append(f"median time to manager event: "
               f"{sp.loc[sp['m_event'] == 1, 'm_dur'].median():.0f}q")
if sp["c_event"].sum():
    log.append(f"median time to client event: "
               f"{sp.loc[sp['c_event'] == 1, 'c_dur'].median():.0f}q")

fig, ax = plt.subplots(figsize=(7.5, 5))
km_m = KaplanMeierFitter().fit(sp["m_time"], sp["m_event"],
                               label=f"Manager: AS collapse ({int(sp['m_event'].sum()):,} events)")
km_m.plot_survival_function(ax=ax, lw=2)
km_c = KaplanMeierFitter().fit(sp["c_time"], sp["c_event"],
                               label=f"Clients: redemption spike ({int(sp['c_event'].sum()):,} events)")
km_c.plot_survival_function(ax=ax, lw=2)
pd.concat({"manager": km_m.survival_function_.iloc[:, 0],
           "clients": km_c.survival_function_.iloc[:, 0]}, axis=1) \
  .to_csv(P.OUT / "wbf_survival.csv")
ax.axvline(12, color="0.6", ls=":", lw=1)
ax.text(12.2, 0.05, "~3 years", fontsize=8, color="0.4")
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Share not yet capitulated")
ax.set_title("Who breaks first? Manager vs. clients of the same funds, 2000-2023")
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "who_breaks_first.png", dpi=200)

log.append("\nCaveats: paired arms (same funds) - overlay is descriptive, real build "
           "uses paired/competing-risk tests; retail flags begin Dec 1999; client "
           "threshold is a pre-specified pilot definition (-10%/quarter).")
log.append("WBF DONE - outputs aggregate-only and shareable.")
P.write_report("wbf_report.txt", log)
print("\n".join(log))
