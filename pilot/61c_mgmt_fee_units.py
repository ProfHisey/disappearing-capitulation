"""Stage 61c: FIX THE MGMT-FEE UNITS, RERUN THE DECISION TEST.

Stage 61b block 4 was invalid as run: CRSP stores exp_ratio as a decimal
fraction but mgmt_fee in PERCENT, so mg_bps was inflated ~100x. The tell
was printed in 61b's own log: median 'other' (er - mgmt) = -5510bps, and
'mgmt fee cut' rates of 53-62% (a 10bps threshold on a 100x series fires
on 0.1bp wiggles). Every mgmt/other number in 61b block 4 is unusable.

This stage: (a) detects and fixes the scale from the 61b cache;
(b) reruns the three-series detector (exp ratio / mgmt fee / other) with
correct units; (c) reruns the flat-assets test on the management fee.
After 61b, Finding 4 hangs on one number: the modern-era flat-assets gap
is +2.9pp [+0.8, +5.0]. If that margin lives in the MANAGEMENT FEE, it is
a board decision (price concession as the modern surrender margin). If it
lives only in 'other', it is waivers and breakpoints, and Finding 4
reduces to the mechanical result.

CAVEAT to verify before the paper quotes this: CRSP's mgmt_fee may be
reported net of waivers in some periods, which blurs the decision line.

  python 61c_mgmt_fee_units.py
Requires output/s61b_size_panel.parquet (run 61b first).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

OUT = Path("output")
CACHE = OUT / "s61b_size_panel.parquet"

CUT_BPS = 10.0
WIN = 8
SEED = 20260822
ERAS = ["1980-94", "1995-2009", "2010-23"]
rng = np.random.default_rng(SEED)

log = ["MGMT-FEE UNITS FIX + DECISION TEST (stage 61c)", "=" * 64]


def say(s):
    log.append(s)
    print(s, flush=True)


if not CACHE.exists():
    raise SystemExit("run 61b first: cache missing")
fw = pd.read_parquet(CACHE)
say(f"loaded 61b cache: {len(fw):,} wficn-quarters")

# ---- scale detection and fix ----------------------------------------
both = fw[(fw["er_bps"] > 0) & (fw["mg_bps"] > 0)]
ratio = (both["mg_bps"] / both["er_bps"]).median()
say(f"median mg_bps/er_bps as cached: {ratio:.1f}")
if 20 < ratio < 500:
    fw["mg_bps"] = fw["mg_bps"] / 100.0
    say("  -> mgmt_fee was in PERCENT while exp_ratio was a fraction; "
        "divided mg_bps by 100.")
elif 0.2 < ratio < 5:
    say("  -> scales already consistent; no fix applied (61b block 4 "
        "would then need a different explanation - stop and inspect).")
else:
    raise SystemExit(f"unexpected scale ratio {ratio:.2f}; inspect before "
                     "trusting any fix")
fw["other_bps"] = fw["er_bps"] - fw["mg_bps"]
say(f"  after fix: median er {fw['er_bps'].median():.0f}bps, "
    f"mg {fw['mg_bps'].median():.0f}bps, other "
    f"{fw['other_bps'].median():.0f}bps  (other should now be a plausible "
    "non-advisory cost level, roughly 20-80bps)")

# ---- matrices (identical construction to 61b) -----------------------
QS = pd.period_range(fw["quarter"].min(), fw["quarter"].max(), freq="Q")
QORD = {q: i for i, q in enumerate(QS)}
FUNDS = np.sort(fw["wficn"].unique())
FORD = {int(w): i for i, w in enumerate(FUNDS)}
NF, NQ = len(FUNDS), len(QS)
ri = fw["wficn"].map(FORD).to_numpy()
ci = fw["quarter"].map(QORD).to_numpy()


def mat(col):
    M = np.full((NF, NQ), np.nan)
    M[ri, ci] = fw[col].to_numpy()
    return M


ER, MG, OTH, LT = (mat("er_bps"), mat("mg_bps"), mat("other_bps"),
                   mat("log_tna"))
OBS = np.isfinite(ER)
LAST_OBS = np.where(OBS.any(1), NQ - 1 - np.argmax(OBS[:, ::-1], axis=1), -1)

panel = PL.build_panel(log)
death = PL.get_death(log)
dd = death[death["died"] == 1].copy()
dq = pd.PeriodIndex(pd.to_datetime(dd["death_q"].astype(str),
                                   errors="coerce"), freq="Q")
DEATH_ORD = np.full(NF, np.iinfo(np.int32).max, dtype=np.int64)
for w, q in zip(dd.loc[~dq.isna(), "wficn"].astype("int64"), dq[~dq.isna()]):
    r = FORD.get(int(w))
    if r is not None and q in QORD:
        DEATH_ORD[r] = QORD[q]


def windows(M, rows, q0):
    idx = q0[:, None] + np.arange(1, WIN + 1)[None, :]
    inb = idx < NQ
    W = M[rows[:, None], np.where(inb, idx, 0)]
    return M[rows, q0], np.where(inb, W, np.nan)


def code_events(M, rows, q0):
    base, W = windows(M, rows, q0)
    hit = np.isfinite(W) & (W <= (base - CUT_BPS)[:, None])
    has_cut = hit.any(1)
    t_cut = np.where(has_cut, hit.argmax(1) + 1, WIN + 1)
    t_die = DEATH_ORD[rows] - q0
    has_die = (t_die >= 1) & (t_die <= WIN)
    t_die = np.where(has_die, t_die, WIN + 1)
    t_last = np.maximum(np.minimum(LAST_OBS[rows] - q0, WIN), 0)
    code = np.zeros(len(rows), dtype=np.int64)
    t = np.minimum(t_last, WIN)
    cut_first = has_cut & (t_cut <= np.minimum(t_die, WIN))
    die_first = (~cut_first) & has_die
    code[cut_first] = 1
    t[cut_first] = t_cut[cut_first]
    code[die_first] = 2
    t[die_first] = t_die[die_first]
    keep = (t >= 1) & np.isfinite(base)
    return t[keep].astype(np.int64), code[keep], keep


def cif(t, code, k=WIN):
    if len(t) < 20:
        return np.nan
    tmax = int(t.max())
    cnt = np.bincount(t, minlength=tmax + 1)
    at_risk = len(t) - np.concatenate([[0], np.cumsum(cnt)[:-1]])
    d1 = np.bincount(t[code == 1], minlength=tmax + 1).astype(float)
    d2 = np.bincount(t[code == 2], minlength=tmax + 1).astype(float)
    safe = np.where(at_risk > 0, at_risk, 1.0).astype(float)
    surv = np.cumprod(1.0 - (d1 + d2) / safe)
    s_prev = np.concatenate([[1.0], surv[:-1]])
    c1 = np.cumsum(s_prev * d1 / safe)
    return float(c1[min(k, tmax)])


def pcut(M, rows, q0):
    return cif(*code_events(M, rows, q0)[:2])


# ---- fund-year sample, identical to 61b -----------------------------
pan = panel.sort_values(["wficn", "quarter"]).copy()
pan["rel_prev"] = pan.groupby("wficn")["rel4q"].shift()
pan = pan[pan["rel4q"].notna() & pan["rel_prev"].notna()]
pan = pan[pan["quarter"].dt.quarter == 4]
pan["stressed"] = (pan["rel4q"] < 0) & (pan["rel_prev"] < 0)
pan["unstressed"] = (pan["rel4q"] >= 0) & (pan["rel_prev"] >= 0)
pan["row"] = pan["wficn"].astype("int64").map(FORD)
pan["q0"] = pan["quarter"].map(QORD)
s = pan[pan["row"].notna() & pan["q0"].notna()].copy()
s["row"] = s["row"].astype(np.int64)
s["q0"] = s["q0"].astype(np.int64)
s["era3"] = pd.cut(s["quarter"].dt.year, [0, 1994, 2009, 9999], labels=ERAS)
R_, Q_ = s["row"].to_numpy(np.int64), s["q0"].to_numpy(np.int64)
fwd = np.minimum(Q_ + WIN, NQ - 1)
s["g_post"] = LT[R_, fwd] - LT[R_, Q_]

# ---- 1. corrected three-series table --------------------------------
say("\n" + "=" * 64)
say("1. DECISION TEST, CORRECT UNITS (P(cut >= 10bps within 8q), CIF)")
say("=" * 64)
say("  era        arm          exp ratio   mgmt fee   other")
for era in ERAS:
    d0 = s[s["era3"] == era]
    for arm in ("stressed", "unstressed"):
        z = d0[d0[arm]]
        if len(z) < 100:
            continue
        r_, q_ = z["row"].to_numpy(np.int64), z["q0"].to_numpy(np.int64)
        say(f"  {era:<10s} {arm:<11s} {pcut(ER, r_, q_):9.1%} "
            f"{pcut(MG, r_, q_):10.1%} {pcut(OTH, r_, q_):8.1%}")
    a, b = d0[d0["stressed"]], d0[d0["unstressed"]]
    if len(a) >= 100 and len(b) >= 100:
        ra, qa = a["row"].to_numpy(np.int64), a["q0"].to_numpy(np.int64)
        rb, qb = b["row"].to_numpy(np.int64), b["q0"].to_numpy(np.int64)
        say(f"  {era:<10s} {'GAP s-u':<11s} "
            f"{pcut(ER, ra, qa) - pcut(ER, rb, qb):+9.1%} "
            f"{pcut(MG, ra, qa) - pcut(MG, rb, qb):+10.1%} "
            f"{pcut(OTH, ra, qa) - pcut(OTH, rb, qb):+8.1%}")
say("  Sanity anchors: mgmt-fee cut rates should now sit BELOW exp-ratio "
    "cut rates (boards move rarely; waivers move often).")

# ---- 2. flat-assets decision test -----------------------------------
say("\n" + "=" * 64)
say("2. FLAT-ASSETS SUBSAMPLE (|8q growth| < 0.10): where does the modern "
    "+2.9pp live?")
say("=" * 64)
flat = s[s["g_post"].notna() & (s["g_post"].abs() < 0.10)]
say(f"  n = {len(flat):,} fund-years")
for era in ("1995-2009", "2010-23"):
    d0 = flat[flat["era3"] == era]
    a, b = d0[d0["stressed"]], d0[d0["unstressed"]]
    if len(a) < 100 or len(b) < 100:
        say(f"  {era}: too few (stressed {len(a):,}, unstressed {len(b):,})")
        continue
    ra, qa = a["row"].to_numpy(np.int64), a["q0"].to_numpy(np.int64)
    rb, qb = b["row"].to_numpy(np.int64), b["q0"].to_numpy(np.int64)
    for lab, M in (("exp ratio", ER), ("mgmt fee ", MG), ("other    ", OTH)):
        pa, pb = pcut(M, ra, qa), pcut(M, rb, qb)
        funds = np.unique(d0["row"].to_numpy(np.int64))
        ia = {f: np.where(ra == f)[0] for f in funds}
        ib = {f: np.where(rb == f)[0] for f in funds}
        dr = []
        for _ in range(500):
            pick = rng.choice(funds, len(funds), replace=True)
            sa = np.concatenate([ia[f] for f in pick if len(ia[f])]) \
                if any(len(ia[f]) for f in pick) else np.array([], np.int64)
            sb = np.concatenate([ib[f] for f in pick if len(ib[f])]) \
                if any(len(ib[f]) for f in pick) else np.array([], np.int64)
            if len(sa) < 50 or len(sb) < 50:
                continue
            v = pcut(M, ra[sa], qa[sa]) - pcut(M, rb[sb], qb[sb])
            if np.isfinite(v):
                dr.append(v)
        lo, hi = (np.percentile(dr, [2.5, 97.5]) if len(dr) > 100
                  else (np.nan, np.nan))
        say(f"  {era:<10s} {lab} stressed {pa:6.1%}  unstressed {pb:6.1%}  "
            f"gap {pa - pb:+6.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]")
say("  READING: modern-era mgmt-fee gap positive with CI excluding zero = "
    "boards of stressed funds genuinely concede on price (the decision "
    "margin). Gap only in 'other' = waivers/breakpoints, and Finding 4 "
    "reduces to the mechanical story.")
say("\n  VERIFY BEFORE QUOTING: whether CRSP mgmt_fee nets out waivers "
    "(check CRSP MF documentation); if it does, part of 'decision' is "
    "still mechanical.")

say("\nSTAGE 61c DONE - aggregates only.")
P.write_report("referee_61c_mgmt_fee_fixed.txt", log)
