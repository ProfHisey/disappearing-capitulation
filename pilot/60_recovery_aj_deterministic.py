"""Stage 60: RECOVERY, DETERMINISTIC AALEN-JOHANSEN + INFERENCE (audit round 6).

Answers three audit findings against stage 37d at once.

FA-2 (SAMPLE vs ESTIMATOR). 37d added `caps[caps["cq"].dt.year >= 1995]`;
37b and 37c have no such line, so every pre-1995 capitulation sat in their
wave arm. The M6 write-up compared 37d against 37b and attributed the whole
KM->AJ move to competing risks. Two things changed. This stage computes the
full 2x2 -- {KM, AJ} x {all events, 1995+} -- so the estimator effect and the
sample effect are separated and each is reported on its own.

FA-3 / MA-1 (NO VARIANCE, NON-DETERMINISTIC). 37d ran
AalenJohansenFitter(calculate_variance=False) with no seed. lifelines jitters
every non-censored duration by +/-1e-4 seeded from OS entropy, which moves
CIF(16q) by up to ~1.5pp across runs with a systematic downward bias (about
half the boundary events get pushed past the horizon), and reports no
intervals at all. Durations here are integer quarters, so the jitter is
unnecessary: this stage computes Aalen-Johansen directly from the event
table on the integer grid. Fully deterministic, no ties problem, no bias.
It then bootstraps the ERA DIFFERENCE, which is the quantity F9 actually
claims and which has never had an interval.

MA-2 (DEATH PAST LAST AS OBSERVATION). 37d codes a death at death_q even
when the Active Share series ended years earlier, so recovery is
unobservable over that gap. The docstring claims this is era-symmetric; it
cannot be, because the ND panel ends 2023Q3 for everyone while deaths run to
2026. This stage prints the gap distribution by era and re-runs everything
under a strict alternative that censors at the last AS observation.

Conventions preserved from 37b/37d for comparability: recovery = 2
consecutive OBSERVED quarters at/above the bar, timed in calendar quarters
from the crossing; ties broken toward recovery.

Aggregates only; report: output/referee_60_recovery_aj_deterministic.txt

  python 60_recovery_aj_deterministic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
OUT.mkdir(exist_ok=True)
BARS = (0.70, 0.75, 0.80)
HORIZONS = (8, 16, 24)
BOOT = 2000
SEED = 20260821
rng = np.random.default_rng(SEED)

log = ["RECOVERY: DETERMINISTIC AJ + INFERENCE (stage 60)", "=" * 64]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter")["as_min"].dropna().sort_index()
      for w, g in panel.groupby("wficn")}
dd = death[death["died"] == 1].copy()

# MI-2: death_q can be missing or malformed. Count rather than swallow.
dq_raw = dd["death_q"].astype(str)
dq_par = pd.PeriodIndex(pd.to_datetime(dq_raw, errors="coerce"), freq="Q")
bad = dq_par.isna().sum()
if bad:
    log.append(f"NOTE: {bad:,} death rows with unparseable death_q dropped")
ok = ~dq_par.isna()
DQ = dict(zip(dd.loc[ok, "wficn"].astype("int64"), dq_par[ok]))
log.append(f"deaths with usable death_q: {len(DQ):,}")

caps_all = sp[sp["capitulated"] == True].copy()
caps_all["cq"] = pd.PeriodIndex(caps_all["m_cal_q"], freq="Q")
caps_all = caps_all[caps_all["cq"].notna()]
caps_all["era2"] = np.where(caps_all["cq"].dt.year <= 2009,
                            "1995-2009", "2010-23")
n_pre95 = int((caps_all["cq"].dt.year < 1995).sum())
log.append(f"capitulation events, all: {len(caps_all):,}")
log.append(f"  of which pre-1995 (mislabelled into the wave arm by "
           f"37b/37c): {n_pre95:,}")
log.append(f"  1995+: {len(caps_all) - n_pre95:,}")


def episode(w, cq, bar, censor_at_last_as=False):
    """(t, code): 0 censored, 1 recovered, 2 died. Ties -> recovery."""
    w = int(w)
    s = PF.get(w)
    post = s[s.index > cq] if s is not None else None
    t_rec, t_last = None, 0
    if post is not None and len(post):
        run, start_q = 0, None
        for q, v in post.items():
            run = run + 1 if v >= bar else 0
            if run == 1:
                start_q = q
            if run == 2:
                t_rec = (start_q - cq).n
                break
        t_last = (post.index[-1] - cq).n
    t_die = (DQ[w] - cq).n if w in DQ and DQ[w] > cq else None
    if censor_at_last_as and t_die is not None and t_die > t_last:
        t_die = None                    # death is past the AS wall: censor
    if t_rec is not None and (t_die is None or t_rec <= t_die):
        return t_rec, 1
    if t_die is not None:
        return t_die, 2
    return max(t_last, 1), 0


# ---------------------------------------------------------------- estimators
def _grid(t, code):
    """Counts on the integer quarter grid. t must be >= 1 integers."""
    tmax = int(t.max())
    n = len(t)
    cnt = np.bincount(t, minlength=tmax + 1)
    at_risk = n - np.concatenate([[0], np.cumsum(cnt)[:-1]])
    d1 = np.bincount(t[code == 1], minlength=tmax + 1)
    d2 = np.bincount(t[code == 2], minlength=tmax + 1)
    return at_risk.astype(float), d1.astype(float), d2.astype(float), tmax


def aj_cif(t, code, ks):
    """Aalen-Johansen CIFs for events 1 and 2, read at each k. Exact."""
    at_risk, d1, d2, tmax = _grid(t, code)
    safe = np.where(at_risk > 0, at_risk, 1.0)
    haz_any = (d1 + d2) / safe
    surv = np.cumprod(1.0 - haz_any)
    s_prev = np.concatenate([[1.0], surv[:-1]])
    c1 = np.cumsum(s_prev * d1 / safe)
    c2 = np.cumsum(s_prev * d2 / safe)
    return ({k: float(c1[min(k, tmax)]) for k in ks},
            {k: float(c2[min(k, tmax)]) for k in ks})


def km_recovery(t, code, ks):
    """Kaplan-Meier for recovery treating DEATH AS CENSORING (37b's estimator)."""
    at_risk, d1, _d2, tmax = _grid(t, code)
    safe = np.where(at_risk > 0, at_risk, 1.0)
    surv = np.cumprod(1.0 - d1 / safe)
    return {k: float(1.0 - surv[min(k, tmax)]) for k in ks}


def build(caps, bar, censor_at_last_as=False):
    out = {}
    for era in ("1995-2009", "2010-23"):
        sub = caps[caps["era2"] == era]
        rows = [episode(w, q, bar, censor_at_last_as)
                for w, q in zip(sub["wficn"], sub["cq"])]
        d = pd.DataFrame(rows, columns=["t", "code"])
        d = d[d["t"] > 0]
        out[era] = (d["t"].to_numpy(np.int64), d["code"].to_numpy(np.int64))
    return out


# ------------------------------------------------- 1. the 2x2: sample x estimator
log.append("\n" + "=" * 64)
log.append("1. SEPARATING THE SAMPLE CHANGE FROM THE ESTIMATOR CHANGE")
log.append("=" * 64)
log.append("  37b = KM on ALL events (pre-1995 mislabelled into the wave)")
log.append("  37d = AJ on 1995+ only. Both changed at once. Here is each.")

samples = {"all events": caps_all,
           "1995+": caps_all[caps_all["cq"].dt.year >= 1995]}
grid_rows = []
for bar in BARS:
    log.append(f"\nbar {bar:.2f}   P(recovered) at 8/16/24q")
    for sname, caps in samples.items():
        ev = build(caps, bar)
        for era in ("1995-2009", "2010-23"):
            t, c = ev[era]
            km = km_recovery(t, c, HORIZONS)
            cif1, cif2 = aj_cif(t, c, HORIZONS)
            log.append(
                f"  {sname:<11s} {era}  n {len(t):>5,}  "
                f"rec {int((c == 1).sum()):>4}  died {int((c == 2).sum()):>4}")
            log.append(
                f"      KM {km[8]:6.1%} {km[16]:6.1%} {km[24]:6.1%}   "
                f"AJ {cif1[8]:6.1%} {cif1[16]:6.1%} {cif1[24]:6.1%}   "
                f"AJ death {cif2[24]:6.1%}")
            for k in HORIZONS:
                grid_rows.append({"bar": bar, "sample": sname, "era": era,
                                  "k": k, "km": km[k], "aj": cif1[k],
                                  "aj_death": cif2[k], "n": len(t)})
pd.DataFrame(grid_rows).round(4).to_csv(OUT / "s60_grid.csv", index=False)

log.append("\n  READ: compare KM->AJ WITHIN a sample row (pure estimator "
           "effect) against 'all events'->'1995+' WITHIN an estimator "
           "(pure sample effect). 37d's reported attenuation mixed the two.")

# --------------------------------------------- 2. bootstrap the era difference
log.append("\n" + "=" * 64)
log.append("2. ERA DIFFERENCE WITH A CONFIDENCE INTERVAL (never reported before)")
log.append("=" * 64)

boot_rows = []
for bar in BARS:
    caps = samples["1995+"]
    ev = build(caps, bar)
    pt = {}
    for era in ("1995-2009", "2010-23"):
        t, c = ev[era]
        pt[era], _ = aj_cif(t, c, HORIZONS)
    for k in HORIZONS:
        obs = pt["2010-23"][k] - pt["1995-2009"][k]
        draws = np.empty(BOOT)
        for b in range(BOOT):
            vals = {}
            for era in ("1995-2009", "2010-23"):
                t, c = ev[era]
                idx = rng.integers(0, len(t), len(t))
                cif1, _ = aj_cif(t[idx], c[idx], (k,))
                vals[era] = cif1[k]
            draws[b] = vals["2010-23"] - vals["1995-2009"]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        sig = "YES" if (lo < 0) == (hi < 0) else "no"
        log.append(f"  bar {bar:.2f} at {k:>2}q: modern minus wave "
                   f"{obs:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  "
                   f"excludes zero: {sig}")
        boot_rows.append({"bar": bar, "k": k, "diff": obs, "lo": lo, "hi": hi,
                          "excl_zero": (lo < 0) == (hi < 0)})
pd.DataFrame(boot_rows).round(4).to_csv(OUT / "s60_era_difference.csv",
                                        index=False)
log.append("\n  The bar-0.80 INVERSION is the F9 claim. If its CI spans "
           "zero, the honest sentence is 'recovery at the high bar is no "
           "LOWER in the modern era', which still supports all-or-nothing.")

# ------------------------------------------- 3. MA-2: the Active Share wall
log.append("\n" + "=" * 64)
log.append("3. DEATHS RECORDED AFTER THE ACTIVE SHARE PANEL ENDS (MA-2)")
log.append("=" * 64)
gaps = []
for w, q, era in zip(caps_all["wficn"], caps_all["cq"], caps_all["era2"]):
    w = int(w)
    s = PF.get(w)
    if s is None or not len(s) or w not in DQ:
        continue
    post = s[s.index > q]
    if not len(post):
        continue
    gaps.append({"era": era, "gap_q": (DQ[w] - post.index[-1]).n})
g = pd.DataFrame(gaps)
if len(g):
    log.append(g.groupby("era")["gap_q"].describe().round(2).to_string())
    log.append("  share with death recorded >4q after the last AS obs:")
    log.append((g.assign(far=g.gap_q > 4).groupby("era")["far"].mean() * 100)
               .round(1).to_string())

log.append("\n  strict alternative: censor at the last AS observation")
for bar in BARS:
    ev = build(samples["1995+"], bar, censor_at_last_as=True)
    line = [f"  bar {bar:.2f}"]
    for era in ("1995-2009", "2010-23"):
        t, c = ev[era]
        cif1, cif2 = aj_cif(t, c, HORIZONS)
        line.append(f"{era} rec16 {cif1[16]:.1%} death24 {cif2[24]:.1%}")
    log.append("   ".join(line))
log.append("  Compare death24 against block 1. The 'one third die in both "
           "eras' fact heading for Paper 2 must survive this.")

log.append("\nSTAGE 60 DONE - deterministic, seeded, aggregates only.")
P.write_report("referee_60_recovery_aj_deterministic.txt", log)
print("\n".join(log))
