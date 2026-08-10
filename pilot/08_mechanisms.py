"""Stage 8: MECHANISMS — four analyses from the same spell panel.

(a) GAMBLE-THEN-FOLD: within-spell Active Share trajectories. Do capitulators
    RAISE activeness early in the spell (tournament doubling-down) before the
    collapse? Mean AS path (relative to spell start) for capitulator vs
    long-resister spells.
(b) CALIBRATED WHO-FIRST: set the client redemption threshold so client events
    have the SAME base rate as manager events (fixes the stage-05 asymmetric-
    tripwire problem), then recount who breaks first.
(c) PACIFY OR PANIC: retail flows in the 4 quarters after the manager's AS
    collapse vs the 4 quarters before. Does the manager's surrender calm the
    clients or accelerate the exit?
(d) DO CLIENTS KILL THE FUND: retail flow path over the last 8 quarters of
    spells that end in fund DEATH vs spells that end in recovery.

Requires stages 06/07 caches (panel_full, death). Outputs (aggregates only):
  output/mechanisms_report.txt, gamble_then_fold.png, flows_before_death.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

HORIZON = 12          # quarters of trajectory to trace
log = ["MECHANISMS (a-d)", "=" * 60]

panel = PL.build_panel(log)
sp = PL.extract_spells(panel, client_cut=None)
death = PL.get_death(log)
sp = sp.merge(death, on="wficn", how="left")
sp["start_p"] = pd.PeriodIndex(sp["start_q"], freq="Q")
sp["end_p"] = pd.PeriodIndex(sp["end_q"], freq="Q")

# outcome classification (as stage 07)
death_p = pd.PeriodIndex(
    sp["death_q"].where(sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
gap = (death_p - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))
sp["outcome"] = "censored"
sp.loc[sp["ended_by"] == "recovered", "outcome"] = "recovered"
sp.loc[sp["ended_by"].isin(["data_end", "as_missing"])
       & sp["died"].fillna(False) & gap.between(-1, 4), "outcome"] = "died"
sp.loc[sp["m_dur"].notna(), "outcome"] = "capitulated"

pf = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

def path(row, col, t0, t1):
    """Values of `col` at spell-relative OBSERVED quarters t0..t1 (audit
    round 2, fix A1: the relative index counts observed rows, aligning with
    m_dur/end_dur for spells containing reporting gaps)."""
    g = pf[row["wficn"]]
    idx = g.index
    if row["start_p"] not in idx:
        return [np.nan] * (t1 - t0 + 1)
    p0 = idx.get_loc(row["start_p"])
    out = []
    for t in range(t0, t1 + 1):
        j = p0 + t
        out.append(g[col].iloc[j] if 0 <= j < len(idx) else np.nan)
    return out

def mean_se(mat):
    m = np.nanmean(mat, axis=0)
    n = np.sum(~np.isnan(mat), axis=0)
    se = np.nanstd(mat, axis=0) / np.sqrt(np.maximum(n, 1))
    return m, se, n

# ---------------------------------------------------- (a) gamble-then-fold --
cap_sp = sp[(sp["m_dur"].notna()) & (sp["m_dur"] >= 2)]
res_sp = sp[(sp["m_dur"].isna()) & (sp["end_dur"] >= HORIZON)]
cap_paths, gambled = [], 0
for _, r in cap_sp.iterrows():
    vals = path(r, "as_min", 0, min(int(r["m_dur"]), HORIZON))
    if vals and pd.notna(vals[0]):
        rel = [v - vals[0] if pd.notna(v) else np.nan for v in vals]
        rel += [np.nan] * (HORIZON + 1 - len(rel))
        cap_paths.append(rel)
        mid = [v for v in vals[1:-1] if pd.notna(v)]
        if mid and max(mid) > vals[0] + 0.02:
            gambled += 1
res_paths = []
for _, r in res_sp.iterrows():
    vals = path(r, "as_min", 0, HORIZON)
    if vals and pd.notna(vals[0]):
        res_paths.append([v - vals[0] if pd.notna(v) else np.nan for v in vals])
cm, cs, cn = mean_se(np.array(cap_paths, dtype=float))
rm, rs, rn = mean_se(np.array(res_paths, dtype=float))
log.append(f"\n(a) gamble-then-fold: {len(cap_paths):,} capitulator spells "
           f"(dur>=2), {len(res_paths):,} resister spells (dur>={HORIZON})")
log.append(f"    capitulators whose AS ROSE >2pts above start mid-spell before "
           f"collapsing: {gambled:,} ({gambled / max(len(cap_paths), 1):.1%})")
log.append(f"    mean dAS at t=1,2,3 (capitulators): "
           f"{', '.join(f'{cm[t]:+.3f}' for t in (1, 2, 3))}")

t = np.arange(HORIZON + 1)
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.axhline(0, color="0.8", lw=1)
ax.plot(t, cm, lw=2, color="#d62728", label=f"Capitulators (n={len(cap_paths):,})")
ax.fill_between(t, cm - 1.96 * cs, cm + 1.96 * cs, color="#d62728", alpha=0.15)
ax.plot(t, rm, lw=2, color="#1f77b4", label=f"Resisters (n={len(res_paths):,})")
ax.fill_between(t, rm - 1.96 * rs, rm + 1.96 * rs, color="#1f77b4", alpha=0.15)
ax.set_xlabel("Quarters since underperformance spell began")
ax.set_ylabel("Active Share minus its value at spell start")
ax.set_title("Within-spell Active Share trajectory: gamble, grind, or fold?")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(P.OUT / "gamble_then_fold.png", dpi=200)

# ---------------------------------------------- (b) calibrated who-first ----
flow_sp = []
for _, r in sp.iterrows():
    vals = [v for v in path(r, "flowq", 0, int(r["end_dur"])) if pd.notna(v)]
    if vals:
        flow_sp.append((r["wficn"], r["start_q"], r["m_dur"], r["end_dur"],
                        min(vals), r["start_p"].year))
fs = pd.DataFrame(flow_sp, columns=["wficn", "start_q", "m_dur", "end_dur",
                                    "min_flow", "yr"])
fs = fs[fs["yr"] >= 2000]
p_m = fs["m_dur"].notna().mean()
t_star = float(np.quantile(fs["min_flow"], p_m))
fs["c_event"] = fs["min_flow"] <= t_star
log.append(f"\n(b) calibrated who-first (2000+, {len(fs):,} spells with flows):")
log.append(f"    manager base rate {p_m:.2%} -> calibrated client threshold "
           f"t* = {t_star:.1%} per quarter (vs -10% ad hoc in stage 05)")
# recompute first-event timing at t*
firsts = {"client": 0, "manager": 0, "tie": 0}
for _, r in fs[fs["m_dur"].notna() | fs["c_event"]].iterrows():
    row = sp[(sp["wficn"] == r["wficn"]) & (sp["start_q"] == r["start_q"])].iloc[0]
    fvals = path(row, "flowq", 0, int(r["end_dur"]))
    c_dur = next((i for i, v in enumerate(fvals)
                  if pd.notna(v) and v <= t_star), None)
    m_dur = int(r["m_dur"]) if pd.notna(r["m_dur"]) else None
    if c_dur is None and m_dur is None:
        continue
    if m_dur is None or (c_dur is not None and c_dur < m_dur):
        firsts["client"] += 1
    elif c_dur is None or m_dur < c_dur:
        firsts["manager"] += 1
    else:
        firsts["tie"] += 1
tot = sum(firsts.values())
log.append("    " + ", ".join(f"{k} first: {v:,} ({v / max(tot, 1):.1%})"
                              for k, v in firsts.items()))

# ------------------------------------------------- (c) pacify or panic ----
diffs = []
for _, r in sp[sp["m_dur"].notna()].iterrows():
    g = pf[r["wficn"]]
    eq = pd.Period(r["m_cal_q"], freq="Q")  # audit fix A1 (round 2)
    pre = [g.at[eq - k, "flowq"] for k in range(1, 5) if (eq - k) in g.index]
    post = [g.at[eq + k, "flowq"] for k in range(1, 5) if (eq + k) in g.index]
    pre = [v for v in pre if pd.notna(v)]
    post = [v for v in post if pd.notna(v)]
    if len(pre) >= 2 and len(post) >= 2:
        diffs.append(np.mean(post) - np.mean(pre))
d = np.array(diffs)
if len(d):
    tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    log.append(f"\n(c) pacify or panic ({len(d):,} manager events with flows): "
               f"mean retail flow (post-4q) minus (pre-4q) = {d.mean():+.2%}/q, "
               f"t = {tstat:.2f}")
    log.append("    positive = flows improve after surrender (pacified); "
               "negative = exit accelerates (panic). Caveat: no control for the "
               "secular downtrend in active flows; matched controls in real build.")

# ------------------------------------------- (d) do clients kill the fund? --
def tail_paths(rows):
    out = []
    for _, r in rows.iterrows():
        g = pf[r["wficn"]]
        vals = []
        for k in range(8, 0, -1):
            q = r["end_p"] - k + 1
            vals.append(g.at[q, "flowq"] if q in g.index else np.nan)
        out.append(vals)
    return np.array(out, dtype=float)

died_t = tail_paths(sp[sp["outcome"] == "died"])
rec_t = tail_paths(sp[(sp["outcome"] == "recovered") & (sp["end_dur"] >= 4)])
dm, ds, dn = mean_se(died_t)
rm2, rs2, rn2 = mean_se(rec_t)
log.append(f"\n(d) flows before death: {died_t.shape[0]:,} died spells vs "
           f"{rec_t.shape[0]:,} recovered spells")
log.append(f"    mean retail flow in final 4 quarters: died "
           f"{np.nanmean(dm[-4:]):+.2%}/q vs recovered {np.nanmean(rm2[-4:]):+.2%}/q")

x = np.arange(-8, 0) + 1
fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
ax2.axhline(0, color="0.8", lw=1)
ax2.plot(x, dm, lw=2, color="#7f7f7f", label=f"Spells ending in death (n={died_t.shape[0]:,})")
ax2.fill_between(x, dm - 1.96 * ds, dm + 1.96 * ds, color="#7f7f7f", alpha=0.2)
ax2.plot(x, rm2, lw=2, color="#2ca02c", label=f"Spells ending in recovery (n={rec_t.shape[0]:,})")
ax2.fill_between(x, rm2 - 1.96 * rs2, rm2 + 1.96 * rs2, color="#2ca02c", alpha=0.2)
ax2.set_xlabel("Quarters before spell end")
ax2.set_ylabel("Mean retail net flow rate (per quarter)")
ax2.set_title("Do clients kill the fund? Retail flows into death vs recovery")
ax2.legend(frameon=False, fontsize=8)
fig2.tight_layout()
fig2.savefig(P.OUT / "flows_before_death.png", dpi=200)

log.append("\nMECHANISMS DONE - outputs aggregate-only and shareable.")
P.write_report("mechanisms_report.txt", log)
print("\n".join(log))
