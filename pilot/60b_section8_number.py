"""Stage 60b: THE PAPER 1 SECTION 8 NUMBER, AND THE M6 REWRITE.

Stage 60 settled three things and opened two more. This stage closes them.

WHAT 60 SETTLED
  - FA-2 is refuted. Only 2 capitulations predate 1995, so 37d's `>= 1995`
    filter is a 0.1pp nuisance, not the explanation for anything. The audit
    calibrated that finding against assumed counts of 250-800 events per arm;
    the real counts are 321 and 175. Recorded as a wrong prediction.
  - The real difference between 37b and 37d is the DEATH CLOCK (MA-2), which
    nobody listed as a difference at all. 37b censored at the last Active
    Share observation; 37d kept funds at risk of DEATH through quarters in
    which their RECOVERY could not be observed, because the AS panel had
    already ended for them.
  - The bar-0.80 inversion does not survive an interval.

WHAT THIS STAGE ADDS
  1. The four-step decomposition with each step sized, so the M6 paragraph
     can be rewritten to say what the code actually did.
  2. The Section 8 number itself: P(rebuild full conviction) POOLED across
     eras, at 8/16/24q, under both death conventions, with a fund-clustered
     confidence interval. The draft says "roughly a fifth"; stage 60 suggests
     that is too low and horizon-dependent. This settles the wording.
  3. The era difference under the STRICT convention, which 60 reported only
     under the extended one. Fixing MA-2 should WIDEN the decline; confirm.
  4. The "about a third die in both eras" fact, with an interval, under both
     conventions -- it is entirely a creature of the extended clock and the
     paper has to say so.

Bootstrap clusters on wficn, not on the event, because a fund can contribute
more than one spell and those spells are not independent draws.

Aggregates only; report: output/referee_60b_section8_number.txt

  python 60b_section8_number.py
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

log = ["SECTION 8 NUMBER + M6 REWRITE (stage 60b)", "=" * 64]


def say(s):
    log.append(s)
    print(s, flush=True)


panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter")["as_min"].dropna().sort_index()
      for w, g in panel.groupby("wficn")}

dd = death[death["died"] == 1].copy()
dq = pd.PeriodIndex(pd.to_datetime(dd["death_q"].astype(str),
                                   errors="coerce"), freq="Q")
DQ = dict(zip(dd.loc[~dq.isna(), "wficn"].astype("int64"), dq[~dq.isna()]))

caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps = caps[caps["cq"].notna()].copy()
caps["era2"] = np.where(caps["cq"].dt.year <= 2009, "1995-2009", "2010-23")
say(f"capitulation events: {len(caps):,} across "
    f"{caps['wficn'].nunique():,} distinct funds "
    f"({len(caps) / max(caps['wficn'].nunique(), 1):.2f} spells per fund)")
say("  (if that ratio is above 1, event-level resampling overstates "
    "precision; everything below clusters on the fund)")


def episode(w, cq, bar, strict):
    """(t, code): 0 censored, 1 recovered, 2 died. Ties -> recovery.

    strict=True censors at the last Active Share observation, so a death
    recorded after the AS panel ended for that fund is NOT counted -- the
    convention 37b used. strict=False extends death to death_q -- 37d's.
    """
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
    if strict and t_die is not None and t_die > t_last:
        t_die = None
    if t_rec is not None and (t_die is None or t_rec <= t_die):
        return t_rec, 1
    if t_die is not None:
        return t_die, 2
    return max(t_last, 1), 0


def frame(d, bar, strict):
    rows = [episode(w, q, bar, strict) + (int(w), e)
            for w, q, e in zip(d["wficn"], d["cq"], d["era2"])]
    f = pd.DataFrame(rows, columns=["t", "code", "wficn", "era"])
    return f[f["t"] > 0].reset_index(drop=True)


def _cif(t, code, ks):
    if len(t) == 0:
        return {k: np.nan for k in ks}, {k: np.nan for k in ks}
    tmax = int(t.max())
    cnt = np.bincount(t, minlength=tmax + 1)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(cnt)[:-1]])
    d1 = np.bincount(t[code == 1], minlength=tmax + 1).astype(float)
    d2 = np.bincount(t[code == 2], minlength=tmax + 1).astype(float)
    safe = np.where(at_risk > 0, at_risk, 1.0).astype(float)
    surv = np.cumprod(1.0 - (d1 + d2) / safe)
    s_prev = np.concatenate([[1.0], surv[:-1]])
    c1 = np.cumsum(s_prev * d1 / safe)
    c2 = np.cumsum(s_prev * d2 / safe)
    return ({k: float(c1[min(k, tmax)]) for k in ks},
            {k: float(c2[min(k, tmax)]) for k in ks})


def _km(t, code, ks):
    """Recovery with death treated as censoring -- 37b's estimator."""
    if len(t) == 0:
        return {k: np.nan for k in ks}
    tmax = int(t.max())
    cnt = np.bincount(t, minlength=tmax + 1)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(cnt)[:-1]])
    d1 = np.bincount(t[code == 1], minlength=tmax + 1).astype(float)
    safe = np.where(at_risk > 0, at_risk, 1.0).astype(float)
    surv = np.cumprod(1.0 - d1 / safe)
    return {k: float(1.0 - surv[min(k, tmax)]) for k in ks}


class Boot:
    """Fund-clustered resampling over one event frame."""

    def __init__(self, f):
        self.t = f["t"].to_numpy(np.int64)
        self.c = f["code"].to_numpy(np.int64)
        self.funds = f["wficn"].to_numpy(np.int64)
        self.uf = np.unique(self.funds)
        self.idx = {int(u): np.where(self.funds == u)[0] for u in self.uf}

    def draw(self):
        pick = rng.choice(self.uf, len(self.uf), replace=True)
        ii = np.concatenate([self.idx[int(p)] for p in pick])
        return self.t[ii], self.c[ii]


def ci(stat_fn, boots, n=BOOT):
    """stat_fn takes a dict of drawn (t, code) keyed like `boots`."""
    out = np.full(n, np.nan)
    for b in range(n):
        out[b] = stat_fn({k: v.draw() for k, v in boots.items()})
    out = out[np.isfinite(out)]
    if len(out) < 100:
        return np.nan, np.nan
    return tuple(np.percentile(out, [2.5, 97.5]))


# ======================================================================
# 1. the four-step decomposition
# ======================================================================
say("\n" + "=" * 64)
say("1. WHAT ACTUALLY CHANGED BETWEEN 37b AND 37d (the M6 rewrite)")
say("=" * 64)
say("  Each row changes ONE thing from the row above it.")
dec = []
for bar in BARS:
    say(f"\n  bar {bar:.2f}, P(recovered), by era, at 8/16/24q")
    steps = [
        ("KM  | wall-censored | all  (= 37b)", True, "km", caps),
        ("AJ  | wall-censored | all  (estimator only)", True, "aj", caps),
        ("AJ  | death-extended| all  (the death clock)", False, "aj", caps),
        ("AJ  | death-extended| 1995+ (= 37d)", False, "aj",
         caps[caps["cq"].dt.year >= 1995]),
    ]
    prev = {}
    for label, strict, est, d in steps:
        line = f"  {label:<44s}"
        for era in ("1995-2009", "2010-23"):
            f = frame(d[d["era2"] == era], bar, strict)
            t = f["t"].to_numpy(np.int64)
            c = f["code"].to_numpy(np.int64)
            v = _km(t, c, HORIZONS) if est == "km" else _cif(t, c, HORIZONS)[0]
            delta = ""
            key = (era, bar)
            if key in prev:
                delta = f"({v[16] - prev[key]:+.1f}pp)".replace("0.0", "0.0")
                delta = f"({(v[16] - prev[key]) * 100:+.1f}pp)"
            prev[key] = v[16]
            line += (f"  {era} {v[8]:5.1%}/{v[16]:5.1%}/{v[24]:5.1%} "
                     f"{delta:<10s}")
            dec.append({"bar": bar, "step": label, "era": era,
                        **{f"k{k}": v[k] for k in HORIZONS}})
        say(line)
pd.DataFrame(dec).round(4).to_csv(OUT / "s60b_decomposition.csv", index=False)
say("\n  The (+/-pp) figures are the change in the 16-quarter estimate "
    "caused by that one step. The M6 paragraph currently attributes the "
    "whole 37b->37d move to competing risks; write it from this table "
    "instead.")

# ======================================================================
# 2. the Section 8 number
# ======================================================================
say("\n" + "=" * 64)
say("2. SECTION 8: P(REBUILD FULL CONVICTION), POOLED, WITH AN INTERVAL")
say("=" * 64)
say("  Draft sentence: 'roughly a fifth of capitulators eventually rebuild "
    "full conviction; the modal capitulator never does.'")
say("  'Eventually' has no horizon in the draft. Here is every horizon.")
s8 = []
for bar in BARS:
    for strict, cname in ((True, "wall-censored"), (False, "death-extended")):
        f = frame(caps, bar, strict)
        t = f["t"].to_numpy(np.int64)
        c = f["code"].to_numpy(np.int64)
        rec, die = _cif(t, c, HORIZONS)
        bo = {"all": Boot(f)}
        parts = []
        for k in HORIZONS:
            lo, hi = ci(lambda dr, k=k: _cif(*dr["all"], (k,))[0][k], bo)
            parts.append(f"{k:>2}q {rec[k]:5.1%} [{lo:.1%},{hi:.1%}]")
            s8.append({"bar": bar, "convention": cname, "k": k,
                       "recover": rec[k], "lo": lo, "hi": hi,
                       "death": die[k], "n": len(t)})
        say(f"  bar {bar:.2f} {cname:<15s} n {len(t):>4,}  " +
            "   ".join(parts))
pd.DataFrame(s8).round(4).to_csv(OUT / "s60b_section8.csv", index=False)
say("\n  Read the bar-0.70 row. 'Full conviction' in the draft is the same "
    "70% bar that defines an active fund, so that is the row Section 8 is "
    "about. If 24q sits near a third rather than a fifth, the sentence "
    "needs both a horizon and a bigger fraction, and it should say the "
    "curve is still rising at the end of the window.")

# ======================================================================
# 3. era difference under the strict convention
# ======================================================================
say("\n" + "=" * 64)
say("3. ERA DIFFERENCE UNDER WALL-CENSORING (60 reported only the extended)")
say("=" * 64)
say("  MA-2's differential gap (3.9q in the wave era vs 1.4q modern) "
    "penalises the WAVE arm's recovery more, so removing it should make "
    "the decline LARGER. If it does, the finding is robust to the "
    "convention, which is the sentence a referee wants.")
er = []
for bar in BARS:
    for strict, cname in ((True, "wall-censored"), (False, "death-extended")):
        boots, pt = {}, {}
        for era in ("1995-2009", "2010-23"):
            f = frame(caps[caps["era2"] == era], bar, strict)
            boots[era] = Boot(f)
            pt[era], _ = _cif(f["t"].to_numpy(np.int64),
                              f["code"].to_numpy(np.int64), HORIZONS)
        for k in HORIZONS:
            obs = pt["2010-23"][k] - pt["1995-2009"][k]
            lo, hi = ci(lambda dr, k=k: (_cif(*dr["2010-23"], (k,))[0][k]
                                         - _cif(*dr["1995-2009"], (k,))[0][k]),
                        boots)
            excl = "YES" if np.isfinite(lo) and (lo < 0) == (hi < 0) else "no"
            say(f"  bar {bar:.2f} {cname:<15s} {k:>2}q: "
                f"modern minus wave {obs:+6.1%}  "
                f"95% CI [{lo:+.1%}, {hi:+.1%}]  excludes zero: {excl}")
            er.append({"bar": bar, "convention": cname, "k": k, "diff": obs,
                       "lo": lo, "hi": hi})
pd.DataFrame(er).round(4).to_csv(OUT / "s60b_era_diff.csv", index=False)

# ======================================================================
# 4. the death fact
# ======================================================================
say("\n" + "=" * 64)
say("4. 'ABOUT A THIRD DIE IN BOTH ERAS' -- WITH AN INTERVAL")
say("=" * 64)
for bar in (0.70,):
    for strict, cname in ((False, "death-extended"), (True, "wall-censored")):
        for era in ("1995-2009", "2010-23"):
            f = frame(caps[caps["era2"] == era], bar, strict)
            t = f["t"].to_numpy(np.int64)
            c = f["code"].to_numpy(np.int64)
            _, die = _cif(t, c, HORIZONS)
            bo = {"a": Boot(f)}
            lo, hi = ci(lambda dr: _cif(*dr["a"], (24,))[1][24], bo)
            say(f"  {cname:<15s} {era}: P(die within 24q) {die[24]:.1%} "
                f"95% CI [{lo:.1%}, {hi:.1%}]  (n {len(t):,})")
say("  The wall-censored figures are NOT a competing estimate of mortality. "
    "They are what is OBSERVABLE once you refuse to follow a fund past the "
    "end of its Active Share record. The gap between the two conventions "
    "is the size of the unobservable window, and the paper should report "
    "the extended figure while stating that window explicitly.")

say("\nSTAGE 60b DONE - deterministic, seeded, fund-clustered, aggregates "
    "only.")
P.write_report("referee_60b_section8_number.txt", log)
