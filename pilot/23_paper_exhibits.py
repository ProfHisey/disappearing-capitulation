"""Stage 23: PAPER EXHIBITS — figures 1, 3, 4, 5, 6 and Table 3 Panel B.

Produces the exhibits the round-2 writing review demanded (v4 panel):

  fig1_cohort_decline.png   capitulation vs liquidation-only death, by cohort
  fig3_two_clocks.png       duration + depth hazard profiles, both outcomes
  fig4_famous_traces.png    Active Share paths, Magellan / Value Trust / Sequoia
  fig5_grind.png            mean AS in event time before capitulation
  fig6_h7_cumulative.png    cumulative FF4-adjusted spread, resisters - capitulators
  output/cohort_descriptives.txt   Table 3 Panel B numbers

Figure 2 (CIF curves) already exists as output/cif_by_era.png from stage 13.
All outputs are aggregates (fig4 shows three named public funds' AS paths,
consistent with published fund-level exhibits in the Active Share literature).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["PAPER EXHIBITS", "=" * 60]
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_CAP = "#1f4e79"   # capitulation: dark blue
C_DIE = "#8c1c13"   # death: dark red

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# ------------------------------------------------- fig 1 + Panel B ----
def sect_fig1():
    dv2 = pd.read_parquet(P.CACHE / "death_v2.parquet")
    liq = dv2[dv2["died"] & (dv2["dtype"] == "liquidation")]
    liq_q = liq.set_index("wficn")["death_q"]
    dp = pd.PeriodIndex(sp["death_q"].where(
        sp["death_q"].astype(str).str.match(r"\d{4}Q\d")), freq="Q")
    gap = (dp - sp["end_p"]).map(lambda x: getattr(x, "n", np.nan))
    is_liq = sp["wficn"].isin(set(liq["wficn"]))
    sp["died_liq"] = sp["spell_died"] & is_liq
    sp["c5"] = (sp["start_p"].dt.year // 5) * 5
    rows, xs, cap, dql = [], [], [], []
    fm = PL.get_fund_monthly([])
    fm["quarter"] = fm["caldt"].dt.to_period("Q")
    # audit round 2 fix: fm is MONTHLY, so a (wficn, quarter) index holds
    # ~3 rows per key and .get() returned a Series - the old "median entry
    # TNA" was a pooled fund-month median. Collapse to quarter-end first.
    tnaq = (fm.sort_values("caldt")
              .groupby(["wficn", "quarter"])["tna"].last())
    for c5, g in sp[sp["c5"] >= 1990].groupby("c5"):
        lbl = f"{int(c5)}-{str(int(c5) + 4)[2:]}"
        xs.append(lbl)
        cap.append(g["capitulated"].mean() * 100)
        dql.append(g["died_liq"].mean() * 100)
        # Panel B descriptives
        as0 = [PF[w].at[q, "as_min"] if q in PF.get(w, pd.DataFrame()).index
               else np.nan
               for w, q in zip(g["wficn"], g["start_p"])] if len(g) else []
        tna0 = [tnaq.get((w, q), np.nan)
                for w, q in zip(g["wficn"], g["start_p"])]
        rows.append(f"  {lbl}: n {len(g):5,} | median dur "
                    f"{g['end_dur'].median():.0f}q | mean depth "
                    f"{g['depth'].mean():+.1%} | mean entry AS "
                    f"{np.nanmean(as0):.2f} | median entry TNA "
                    f"${np.nanmedian(tna0):,.0f}M")
    log.append("\nTABLE 3 PANEL B (cohort descriptives):")
    log.extend(rows)
    (P.OUT / "cohort_descriptives.txt").write_text("\n".join(rows),
                                                   encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(xs))
    ax.plot(x, cap, "o-", color=C_CAP, lw=1.8, label="Capitulated")
    ax.plot(x, dql, "s--", color=C_DIE, lw=1.6,
            label="Died (liquidation only)")
    ax.annotate("break ~2000", xy=(2, cap[2]), xytext=(2.2, max(cap) * 0.75),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(len(xs) - 1, max(cap) * 0.9, "final cohort\nright-truncated",
            fontsize=7, ha="right", color="0.4")
    ax.set_xticks(x, xs)
    ax.set_ylabel("Share of spells ending this way (%)")
    ax.set_xlabel("Five-year window in which the spell began")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(P.OUT / "fig1_cohort_decline.png", dpi=300)
    log.append("  wrote fig1_cohort_decline.png")

# ------------------------------------------------------- fig 3 ----
def sect_fig3():
    dt = R.build_dt(sp, PF)
    dt["dep_5_15"] = dt["depth"].between(-0.15, -0.05, inclusive="left").astype(float)
    dt["dep_15_25"] = dt["depth"].between(-0.25, -0.15, inclusive="left").astype(float)
    dt["dep_25_40"] = dt["depth"].between(-0.40, -0.25, inclusive="left").astype(float)
    dt["dep_40p"] = (dt["depth"] < -0.40).astype(float)
    X = ["dur_3_4", "dur_5_8", "dur_9_12", "dur_13p",
         "dep_5_15", "dep_15_25", "dep_25_40", "dep_40p", "era_1023"]

    def hrs(ycol):
        d = dt[[ycol, "wficn"] + X].dropna()
        m = sm.GLM(d[ycol].to_numpy(float),
                   sm.add_constant(d[X].to_numpy(float)),
                   family=sm.families.Binomial(
                       link=sm.families.links.CLogLog())).fit(
            cov_type="cluster", cov_kwds={"groups": d["wficn"].to_numpy()})
        b, se = m.params[1:], m.bse[1:]
        return np.exp(b), np.exp(b - 1.96 * se), np.exp(b + 1.96 * se)

    hc, lc, uc = hrs("event")
    hd, ld, ud = hrs("event_die")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    labels = [["1", "1.5-2", "2.25-3", "3.25+"],
              ["5-15", "15-25", "25-40", ">40"]]
    titles = ["Years underwater (vs first half-year)",
              "Depth, % behind (vs 0-5%)"]
    for k, (sl, ttl) in enumerate(zip([slice(0, 4), slice(4, 8)], titles)):
        ax = axes[k]
        x = np.arange(4)
        for i, (h, lo, hi, col, mk, nm) in enumerate(
                [(hc, lc, uc, C_CAP, "o", "Capitulation"),
                 (hd, ld, ud, C_DIE, "s", "Death")]):
            ax.errorbar(x + (i - 0.5) * 0.15, h[sl],
                        yerr=[h[sl] - lo[sl], hi[sl] - h[sl]],
                        fmt=mk, color=col, ms=5, lw=1.2, capsize=2,
                        label=nm if k == 0 else None)
        ax.axhline(1.0, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("log")
        ax.set_xticks(x, labels[k])
        ax.set_xlabel(ttl)
        if k == 0:
            ax.set_ylabel("Hazard ratio (log scale)")
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(P.OUT / "fig3_two_clocks.png", dpi=300)
    log.append("  wrote fig3_two_clocks.png")

# ------------------------------------------------------- fig 4 ----
def sect_fig4():
    m1 = P.norm_cols(pd.read_csv(PL.MFLINK1))
    m1["ticker"] = m1["ticker"].astype(str).str.strip().str.upper()
    tmap = (m1.dropna(subset=["wficn"]).drop_duplicates("ticker")
              .set_index("ticker"))
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 6.4), sharex=True)
    NAMES = {"FMAGX": "Fidelity Magellan (capitulated 2012)",
             "LMVTX": "Legg Mason Value Trust (never folded)",
             "SEQUX": "Sequoia (never folded; Valeant spell 2015-21)"}
    for ax, tk in zip(axes, NAMES):
        w = int(tmap.loc[tk, "wficn"])
        g = PF[w]["as_min"].dropna()
        x = g.index.to_timestamp()
        ax.plot(x, g.values, color=C_CAP, lw=1.3)
        ax.axhline(0.70, color="0.5", lw=0.8, ls="--")
        ax.axhline(0.60, color=C_DIE, lw=0.8, ls="--")
        ax.set_ylim(0.2, 1.0)
        ax.set_title(NAMES[tk], fontsize=9, loc="left")
        ax.set_ylabel("Active Share")
    axes[-1].text(axes[-1].get_xlim()[0], 0.705, " 70% activity line",
                  fontsize=7, color="0.4", va="bottom")
    axes[-1].text(axes[-1].get_xlim()[0], 0.605, " 60% closet-index line",
                  fontsize=7, color=C_DIE, va="bottom")
    fig.tight_layout()
    fig.savefig(P.OUT / "fig4_famous_traces.png", dpi=300)
    log.append("  wrote fig4_famous_traces.png")

# ------------------------------------------------------- fig 5 ----
def sect_fig5():
    caps = sp[sp["capitulated"]]
    paths = []
    for _, s in caps.iterrows():
        g = PF.get(s["wficn"])
        if g is None:
            continue
        qc = pd.Period(s["m_cal_q"], freq="Q")      # audit fix A1
        row = [g.at[qc + k, "as_min"] if (qc + k) in g.index else np.nan
               for k in range(-8, 1)]
        paths.append(row)
    A = np.array(paths, dtype=float)
    mean_path = np.nanmean(A, axis=0)
    n_path = np.sum(~np.isnan(A), axis=0)
    # control (audit fix A6): ALL at-risk spell-quarters, computed in full
    # (the old [:200000] slice silently kept only low-wficn funds), and
    # EXCLUDING quarters within 2 quarters of that spell's own crossing,
    # which the old code included (dragging the control toward treatment).
    vals = []
    for _, s in sp.iterrows():
        g = PF.get(s["wficn"])
        if g is None:
            continue
        T = int(s["m_dur"]) if s["capitulated"] else int(s["end_dur"])
        lim = int(s["m_dur"]) - 3 if s["capitulated"] else T
        qs = g.index[g.index >= s["start_p"]]
        for t in range(1, min(T, min(lim, len(qs) - 1)) + 1):
            v = g.at[qs[t], "as_min"]
            if pd.notna(v):
                vals.append(float(v))
    ctrl = float(np.mean(vals))
    log.append(f"  fig5 control: {len(vals):,} at-risk spell-quarters "
               f"(full sample, >=3q before any crossing)")
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(-8, 1)
    ax.plot(x, mean_path, "o-", color=C_CAP, lw=1.8,
            label=f"Capitulating funds (n={len(A):,})")
    ax.axhline(ctrl, color="0.5", lw=1.0, ls="--",
               label="At-risk spell-quarters $\\geq$3q before any crossing (mean)")
    ax.axhline(0.60, color=C_DIE, lw=0.8, ls=":")
    ax.set_xlabel("Quarters before the capitulation crossing")
    ax.set_ylabel("Mean Active Share")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(P.OUT / "fig5_grind.png", dpi=300)
    log.append(f"  wrote fig5_grind.png (min per-point n = {n_path.min()})")

# ------------------------------------------------------- fig 6 ----
def sect_fig6():
    fm = PL.get_fund_monthly([])
    fm["m"] = fm["caldt"].dt.to_period("M")
    fac = PL.get_factors(log)
    fac["m"] = fac["month"].dt.to_period("M")
    FAC = fac.set_index("m")[["mktrf", "smb", "hml", "mom", "rf"]]
    # audit fixes A1 + A4: calendar-true entry quarters (crossing stamp for
    # capitulators, 8th OBSERVED underwater quarter for resisters) and
    # deduped portfolio membership, matching stage 26's FIXED convention.
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

    def port(ev):
        rows = []
        for _, s in ev.iterrows():
            m0 = s["entry_q"].asfreq("M", how="end") + 1
            rows += [(s["wficn"], m0 + k) for k in range(36)]
        mem = pd.DataFrame(rows, columns=["wficn", "m"]).drop_duplicates()
        d = mem.merge(fm[["wficn", "m", "fret"]], on=["wficn", "m"],
                      how="inner")
        g = d.groupby("m")["fret"].agg(["mean", "size"])
        return g[g["size"] >= 10]["mean"]

    spread = (port(res) - port(caps)).dropna()
    j = pd.concat([spread.rename("s"), FAC], axis=1, join="inner").dropna()
    X = sm.add_constant(j[["mktrf", "smb", "hml", "mom"]].to_numpy())
    m = sm.OLS(j["s"].to_numpy(), X).fit()
    abn = m.params[0] + m.resid          # alpha + residual each month
    cum = np.cumsum(abn)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.plot(j.index.to_timestamp(), cum * 100, color=C_CAP, lw=1.4)
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_ylabel("Cumulative factor-adjusted spread (%)")
    ax.set_title("Resisters minus capitulators, calendar time", fontsize=9,
                 loc="left")
    fig.tight_layout()
    fig.savefig(P.OUT / "fig6_h7_cumulative.png", dpi=300)
    log.append(f"  wrote fig6_h7_cumulative.png "
               f"(alpha {m.params[0] * 12:+.2%}/yr over {len(j)} months)")

R.section(log, "FIG 1 + TABLE 3 PANEL B (cohort decline + descriptives)",
          sect_fig1)
R.section(log, "FIG 3 (two clocks: duration and depth profiles)", sect_fig3)
R.section(log, "FIG 4 (famous-fund Active Share traces)", sect_fig4)
R.section(log, "FIG 5 (the grind before the fold)", sect_fig5)
R.section(log, "FIG 6 (cumulative resister-capitulator spread)", sect_fig6)

log.append("\nEXHIBITS DONE. Figure 2 = existing cif_by_era.png (stage 13).")
P.write_report("paper_exhibits_report.txt", log)
print("\n".join(log))
