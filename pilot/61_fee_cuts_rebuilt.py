"""Stage 61: FEE CUTS AS PRICE-SURRENDER, REBUILT (audit round 6).

Stage 38 asked whether managers who stopped surrendering with the PORTFOLIO
started surrendering on PRICE. The audit found four defects that between them
put the sign of that finding in play. This stage rebuilds the measurement.

FA-4 (THE DETECTOR IS INCREASING IN SURVIVAL). 38 took `min` over whatever
future quarters happened to exist -- eight chances for a survivor, two for a
dying fund -- and `if not fut: continue` deleted funds that died before any
future fee was observed. Stressed funds die more often, so their measured
P(cut) was biased DOWN, which is the exact direction that manufactures the
null 38 reported. Three estimands are reported here instead of one:
  (A) BALANCED WINDOW: only fund-years with all eight future quarters
      observed AND the fund alive throughout. Clean, but conditions on
      survival, so it answers "among survivors" and is labelled that way.
  (B) COMPETING RISKS (primary): time to first >=10bp cut within 8q, with
      death as a competing event and censoring at the last observed fee
      quarter. Aalen-Johansen, computed on the integer quarter grid, exact
      and deterministic (same estimator as stage 60). Nothing is deleted.
  (C) The all-or-nothing check: P(cut OR death within 8q), so a fund that
      exits instead of repricing is not silently scored as "did not cut."

MA-9 (EQUAL-WEIGHTED FEE). 38 averaged expense ratios across share classes
equally, so the fund's fee moved when a class opened or closed. Everything is
computed twice, TNA-weighted and equal-weighted, and both are printed. If
they disagree the weighting is the finding.

MA-10 (ASYMMETRIC DEFINITIONS). 38 defined stressed as two consecutive
negative quarters but unstressed as a single non-negative quarter. Symmetric
here: two consecutive on both sides. A five-bin dose-response on trailing
relative return is added, because a binary split throws away the gradient
that would tell us whether this is a real behavioural margin.

MA-11 (POST-OUTCOME SELECTION). 38's section (c) selected "resisting" spells
on `end_dur >= 12`, an outcome realised AFTER t0, then evaluated them at
quarter 8 -- inside the window that defined their selection. Replaced with a
LANDMARK design: at quarter 8, take every spell still alive and still above
the bar, using only information available at quarter 8, and follow it forward.

Also new: the event-time fee path is now differenced against the industry
path anchored at the same calendar base quarter, because fees fell
industry-wide and the level was never the statistic.

Inference: fund-clustered bootstrap (funds resampled with replacement, all of
a fund's rows moving together) on every headline difference. 38 had none.

Aggregates only; report: output/referee_61_fee_cuts_rebuilt.txt
First run streams Fund Summary (~1.5GB, several minutes) and caches the fee
panel; later runs load the cache in seconds.

  python 61_fee_cuts_rebuilt.py
  python 61_fee_cuts_rebuilt.py --rebuild     (force re-stream)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)
CACHE = OUT / "s61_fee_panel.parquet"

CUT_BPS = 10.0          # a "cut" is a fall of at least this many bps
WIN = 8                 # quarters of follow-up
BOOT = 2000
SEED = 20260821
ERAS = ["1980-94", "1995-2009", "2010-23"]
rng = np.random.default_rng(SEED)

log = ["FEE CUTS, REBUILT (stage 61)", "=" * 64]


def say(s):
    log.append(s)
    print(s, flush=True)


# ======================================================================
# 1. fee panel: share class -> wficn-quarter, BOTH weightings, cached
# ======================================================================
if CACHE.exists() and "--rebuild" not in sys.argv:
    fw = pd.read_parquet(CACHE)
    say(f"loaded cached fee panel: {CACHE}  ({len(fw):,} rows)")
else:
    fs_path = SRC / "crsp_mf" / "Fund Summary.csv"
    say(f"streaming {fs_path} (first run only) ...")
    TNA_CANDIDATES = ("tna_latest", "tna", "mtna", "tna_lag")
    parts, tna_col = [], None
    for ch in pd.read_csv(fs_path, chunksize=2_000_000, low_memory=False,
                          encoding="latin-1"):
        ch.columns = [c.lower() for c in ch.columns]
        if tna_col is None:
            if "exp_ratio" not in ch.columns:
                raise SystemExit(f"exp_ratio not in Fund Summary; saw "
                                 f"{list(ch.columns)[:25]}")
            tna_col = next((c for c in TNA_CANDIDATES if c in ch.columns), "")
        keep = ["crsp_fundno", "caldt", "exp_ratio"] + ([tna_col] if tna_col
                                                        else [])
        ch = ch[[c for c in keep if c in ch.columns]]
        ch = ch[ch["exp_ratio"].notna() & (ch["exp_ratio"] > 0)]
        parts.append(ch)
    fees = pd.concat(parts, ignore_index=True)
    del parts
    if not tna_col:
        say("  !! WARNING: no TNA column in Fund Summary. MA-9 CANNOT be "
            "answered on this run; TNA-weighted results will be missing.")
    else:
        # CRSP writes blanks and occasional text into tna_latest, so the
        # column arrives as object dtype. Coerce and COUNT the losses --
        # never let a silent dtype decide which funds get a TNA weight.
        raw = fees[tna_col]
        if raw.dtype == object:
            num = pd.to_numeric(raw, errors="coerce")
            lost = int(num.isna().sum() - raw.isna().sum())
            say(f"  {tna_col} arrived as text; {lost:,} non-numeric values "
                f"coerced to missing ({lost / max(len(raw), 1):.3%} of rows)")
            fees[tna_col] = num
        say(f"  TNA column in use: {tna_col}; "
            f"{fees[tna_col].notna().mean():.1%} of share-class rows have it")

    n0 = len(fees)
    fees = fees[fees["exp_ratio"] < 0.25]          # CRSP junk guard
    say(f"  dropped {n0 - len(fees):,} rows with exp_ratio >= 25% "
        f"({(n0 - len(fees)) / max(n0, 1):.3%} of rows)")

    fees["quarter"] = pd.to_datetime(fees["caldt"]).dt.to_period("Q")
    m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                     encoding="latin-1")
    m1.columns = [c.lower() for c in m1.columns]
    link = m1[["crsp_fundno", "wficn"]].drop_duplicates()
    n_multi = int(link.groupby("crsp_fundno")["wficn"].nunique().gt(1).sum())
    if n_multi:
        say(f"  NOTE: {n_multi:,} crsp_fundno map to >1 wficn; those "
            f"share classes are dropped (MA-7 discipline)")
        good = link.groupby("crsp_fundno")["wficn"].nunique().eq(1)
        link = link[link["crsp_fundno"].isin(good[good].index)]
    fees = fees.merge(link, on="crsp_fundno", how="inner")

    fees["wficn"] = fees["wficn"].astype("int64")
    g = fees.groupby(["wficn", "quarter"])
    ew = g["exp_ratio"].mean().rename("er_ew")
    if tna_col:
        fees["_w"] = fees[tna_col].where(fees[tna_col] > 0)
        fees["_num"] = fees["exp_ratio"] * fees["_w"]
        gg = fees.groupby(["wficn", "quarter"])[["_num", "_w"]].sum(
            min_count=1)
        vw = (gg["_num"] / gg["_w"]).rename("er_vw")
        cov = float(vw.notna().mean())
        say(f"  TNA weights available for {cov:.1%} of wficn-quarters")
    else:
        vw = pd.Series(np.nan, index=ew.index, name="er_vw")
    nclass = g["exp_ratio"].size().rename("n_class")
    fw = pd.concat([ew, vw, nclass], axis=1).reset_index()
    # materialise every derived column IN the cache (round-5 lesson: never
    # recompute a derived quantity downstream, it drifts between stages)
    fw["er_ew_bps"] = fw["er_ew"] * 1e4
    fw["er_vw_bps"] = fw["er_vw"] * 1e4
    fw.to_parquet(CACHE, index=False)
    say(f"  cached -> {CACHE}")

say(f"fee panel: {len(fw):,} wficn-quarters, {fw['wficn'].nunique():,} funds, "
    f"{fw['quarter'].min()} to {fw['quarter'].max()}")
say(f"  median share classes per fund-quarter: "
    f"{fw['n_class'].median():.0f}, max {fw['n_class'].max():.0f}")
_d = (fw["er_vw_bps"] - fw["er_ew_bps"]).dropna()
if len(_d):
    say(f"  TNA-wt minus equal-wt fee: median {_d.median():+.1f}bps, "
        f"IQR [{_d.quantile(.25):+.1f}, {_d.quantile(.75):+.1f}], "
        f"|diff| > 5bps in {(_d.abs() > 5).mean():.1%} of fund-quarters")
    say("  (if that is large, every stage-38 number was measuring share-class "
        "composition as much as price)")

# ---- dense matrices: rows = fund, cols = quarter ordinal ---------------
QS = pd.period_range(fw["quarter"].min(), fw["quarter"].max(), freq="Q")
QORD = {q: i for i, q in enumerate(QS)}
FUNDS = np.sort(fw["wficn"].unique())
FORD = {int(w): i for i, w in enumerate(FUNDS)}
NF, NQ = len(FUNDS), len(QS)

ri = fw["wficn"].map(FORD).to_numpy()
ci = fw["quarter"].map(QORD).to_numpy()
MATS = {}
for tag, col in (("tna-wt", "er_vw_bps"), ("equal-wt", "er_ew_bps")):
    M = np.full((NF, NQ), np.nan)
    M[ri, ci] = fw[col].to_numpy()
    if np.isfinite(M).any():
        MATS[tag] = M
say(f"fee matrix: {NF:,} funds x {NQ} quarters; weightings available: "
    f"{list(MATS)}")

# ======================================================================
# 2. panel, spells, deaths
# ======================================================================
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)

dd = death[death["died"] == 1].copy()
dq = pd.PeriodIndex(pd.to_datetime(dd["death_q"].astype(str),
                                   errors="coerce"), freq="Q")
bad = int(dq.isna().sum())
if bad:
    say(f"NOTE: {bad:,} death rows with unparseable death_q dropped")
DEATH_ORD = np.full(NF, np.iinfo(np.int32).max, dtype=np.int64)
nd = 0
for w, q in zip(dd.loc[~dq.isna(), "wficn"].astype("int64"), dq[~dq.isna()]):
    r = FORD.get(int(w))
    if r is not None and q in QORD:
        DEATH_ORD[r] = QORD[q]
        nd += 1
say(f"deaths mapped onto the fee grid: {nd:,}")

OBS = np.isfinite(next(iter(MATS.values())))
LAST_OBS = np.where(OBS.any(1), NQ - 1 - np.argmax(OBS[:, ::-1], axis=1), -1)


# ======================================================================
# 3. the three estimands
# ======================================================================
def windows(M, rows, q0):
    """Return base, the WIN-quarter forward fee block, and its validity."""
    idx = q0[:, None] + np.arange(1, WIN + 1)[None, :]
    inb = idx < NQ
    safe = np.where(inb, idx, 0)
    W = M[rows[:, None], safe]
    W = np.where(inb, W, np.nan)
    return M[rows, q0], W


def code_events(M, rows, q0):
    """(t, code) with code 1=cut, 2=death, 0=censored. Nothing dropped.

    t is in quarters after q0. Censoring is at the last quarter in which the
    fund's fee is OBSERVED (a fund cannot be seen to cut a fee we never see),
    capped at WIN. Death outranks a later cut and vice versa; a cut in the
    same quarter as death counts as a cut, which is conservative for the
    null we are testing.
    """
    base, W = windows(M, rows, q0)
    hit = np.isfinite(W) & (W <= (base - CUT_BPS)[:, None])
    has_cut = hit.any(1)
    t_cut = np.where(has_cut, hit.argmax(1) + 1, WIN + 1)

    t_die = DEATH_ORD[rows] - q0
    has_die = (t_die >= 1) & (t_die <= WIN)
    t_die = np.where(has_die, t_die, WIN + 1)

    # last quarter with an observed fee, in event time, capped at WIN
    t_last = np.minimum(LAST_OBS[rows] - q0, WIN)
    t_last = np.maximum(t_last, 0)

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


def aj_cif(t, code, k):
    """Aalen-Johansen CIF for event 1 (and 2) at k. Integer grid, exact."""
    if len(t) == 0:
        return np.nan, np.nan
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
    kk = min(k, tmax)
    return float(c1[kk]), float(c2[kk])


def balanced(M, rows, q0):
    """(A) survivors with a complete WIN-quarter fee record."""
    base, W = windows(M, rows, q0)
    full = np.isfinite(W).all(1) & np.isfinite(base)
    alive = (DEATH_ORD[rows] - q0) > WIN
    ok = full & alive
    if ok.sum() == 0:
        return np.nan, 0
    drop = (W[ok].min(1) - base[ok])
    return float((drop <= -CUT_BPS).mean()), int(ok.sum())


# ---- build the stressed / unstressed samples (MA-10: symmetric) --------
pan = panel.sort_values(["wficn", "quarter"]).copy()
pan["rel_prev"] = pan.groupby("wficn")["rel4q"].shift()
pan = pan[pan["rel4q"].notna() & pan["rel_prev"].notna()]
pan = pan[pan["quarter"].dt.quarter == 4]        # one obs per fund-year
pan["stressed"] = (pan["rel4q"] < 0) & (pan["rel_prev"] < 0)
pan["unstressed"] = (pan["rel4q"] >= 0) & (pan["rel_prev"] >= 0)
pan["row"] = pan["wficn"].astype("int64").map(FORD)
pan["q0"] = pan["quarter"].map(QORD)
samp = pan[pan["row"].notna() & pan["q0"].notna()].copy()
samp["row"] = samp["row"].astype(np.int64)
samp["q0"] = samp["q0"].astype(np.int64)
samp["year"] = samp["quarter"].dt.year
samp["era3"] = pd.cut(samp["year"], [0, 1994, 2009, 9999], labels=ERAS)
say(f"\nfund-years on the fee grid: {len(samp):,} "
    f"(stressed {int(samp['stressed'].sum()):,}, "
    f"unstressed {int(samp['unstressed'].sum()):,}, "
    f"mixed/ambiguous {int((~samp['stressed'] & ~samp['unstressed']).sum()):,})")
say("  MA-10: BOTH labels now require two consecutive quarters. The "
    "'mixed' group is the sign-flippers stage 38 silently put in the "
    "unstressed arm.")


def run_arm(M, d):
    rows = d["row"].to_numpy(np.int64)
    q0 = d["q0"].to_numpy(np.int64)
    t, code, keep = code_events(M, rows, q0)
    cif_cut, cif_die = aj_cif(t, code, WIN)
    pa, na = balanced(M, rows, q0)
    both = np.nan
    if len(t):
        both = float(((code == 1) | (code == 2)).mean())
    return {"n": len(d), "n_risk": len(t), "aj_cut": cif_cut,
            "aj_die": cif_die, "bal": pa, "n_bal": na, "cut_or_die": both}


def _blocks(starts, counts):
    """Concatenate the integer ranges [s, s+c) without a Python loop."""
    keep = counts > 0
    s_, c_ = starts[keep], counts[keep]
    if len(s_) == 0:
        return np.empty(0, dtype=np.int64)
    tot = int(c_.sum())
    out = np.ones(tot, dtype=np.int64)
    out[0] = s_[0]
    if len(s_) > 1:
        pos = np.cumsum(c_)[:-1]
        out[pos] = s_[1:] - (s_[:-1] + c_[:-1] - 1)
    return np.cumsum(out)


class Arm:
    """One cell of the design, indexed for O(n) fund-clustered resampling.

    A fund contributes several fund-years; the cluster bootstrap has to move
    all of a fund's rows together or the SE is the independent-observations
    SE, which is the mistake FA-1 flags elsewhere in this paper.
    """

    def __init__(self, d, funds):
        self.rows = d["row"].to_numpy(np.int64)
        self.q0 = d["q0"].to_numpy(np.int64)
        self.order = np.argsort(self.rows, kind="stable")
        pos = np.searchsorted(funds, self.rows[self.order])
        self.cnt = np.bincount(pos, minlength=len(funds))
        self.st = np.concatenate([[0], np.cumsum(self.cnt)[:-1]])

    def draw(self, pick):
        sel = self.order[_blocks(self.st[pick], self.cnt[pick])]
        return self.rows[sel], self.q0[sel]

    def cif(self, M, rows=None, q0=None):
        r = self.rows if rows is None else rows
        q = self.q0 if q0 is None else q0
        if len(r) < 20:
            return np.nan
        t, c, _ = code_events(M, r, q)
        return aj_cif(t, c, WIN)[0]


def boot_gap(M, arm_s, arm_u, funds, n_boot=BOOT):
    """Fund-clustered bootstrap of CIF(stressed) - CIF(unstressed)."""
    out = np.full(n_boot, np.nan)
    nf = len(funds)
    for b in range(n_boot):
        pick = rng.integers(0, nf, nf)
        a1 = arm_s.cif(M, *arm_s.draw(pick))
        a2 = arm_u.cif(M, *arm_u.draw(pick))
        out[b] = a1 - a2
    return out[np.isfinite(out)]


FUND_IDS = np.unique(samp["row"].to_numpy(np.int64))

say("\n" + "=" * 64)
say("A/B/C. P(FEE CUT >= 10bps WITHIN 8q), THREE ESTIMANDS, BY ERA")
say("=" * 64)
rows_out = []
for tag, M in MATS.items():
    say(f"\n--- fee weighting: {tag} ---")
    say("  era        arm          AJ cut   AJ death  balanced(A)  "
        "cut-or-die   n@risk")
    for era in ERAS:
        d = samp[samp["era3"] == era]
        for arm in ("stressed", "unstressed"):
            dd_ = d[d[arm]]
            if len(dd_) < 30:
                continue
            r = run_arm(M, dd_)
            say(f"  {era:<10s} {arm:<11s} {r['aj_cut']:7.1%} "
                f"{r['aj_die']:9.1%} {r['bal']:11.1%} "
                f"{r['cut_or_die']:11.1%} {r['n_risk']:8,}")
            rows_out.append({"weight": tag, "era": era, "arm": arm, **r})
        ds, du = d[d["stressed"]], d[d["unstressed"]]
        if len(ds) >= 30 and len(du) >= 30:
            arm_s, arm_u = Arm(ds, FUND_IDS), Arm(du, FUND_IDS)
            a1, a2 = arm_s.cif(M), arm_u.cif(M)
            dr = boot_gap(M, arm_s, arm_u, FUND_IDS)
            lo, hi = (np.percentile(dr, [2.5, 97.5]) if len(dr) > 100
                      else (np.nan, np.nan))
            excl = "YES" if np.isfinite(lo) and (lo < 0) == (hi < 0) else "no"
            say(f"  {era:<10s} {'GAP s-u':<11s} {a1 - a2:+7.1%}   "
                f"95% CI [{lo:+.1%}, {hi:+.1%}]  excludes zero: {excl}"
                f"   ({len(dr)} usable draws)")
            rows_out.append({"weight": tag, "era": era, "arm": "gap",
                             "aj_cut": a1 - a2, "lo": lo, "hi": hi,
                             "n": len(ds) + len(du)})
pd.DataFrame(rows_out).to_csv(OUT / "s61_cut_probabilities.csv", index=False)

say("\n  READ: the STRESSED-MINUS-UNSTRESSED GAP is the statistic, not the "
    "level -- fees fell industry-wide, so a rising level across eras is "
    "the industry trend, not price-surrender. M1/M2 predict the GAP widens "
    "across eras as portfolio-surrender dies.")
say("  Compare column A against the AJ column. If A is materially higher, "
    "stage 38's survivor-only design was the bias FA-4 described, and by "
    "how much.")

# ---- the era trend in the gap, with an interval -----------------------
say("\n" + "=" * 64)
say("D. DID PRICE-SURRENDER RISE? (difference in differences, bootstrapped)")
say("=" * 64)
for tag, M in MATS.items():
    late = samp[samp["era3"] == "2010-23"]
    early = samp[samp["era3"] == "1995-2009"]
    cells, ok = {}, True
    for nm, d in (("late", late), ("early", early)):
        for arm in ("stressed", "unstressed"):
            dd_ = d[d[arm]]
            if len(dd_) < 30:
                ok = False
            cells[(nm, arm)] = Arm(dd_, FUND_IDS)
    if not ok:
        say(f"  {tag}: too few observations for the DiD")
        continue

    def did(pick=None):
        v = {}
        for k, a in cells.items():
            v[k] = a.cif(M) if pick is None else a.cif(M, *a.draw(pick))
        return ((v[("late", "stressed")] - v[("late", "unstressed")])
                - (v[("early", "stressed")] - v[("early", "unstressed")]))

    obs = did()
    nf = len(FUND_IDS)
    draws = np.array([did(rng.integers(0, nf, nf)) for _ in range(BOOT)])
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) > 100
              else (np.nan, np.nan))
    say(f"  {tag}: DiD (2010-23 gap minus 1995-2009 gap) = {obs:+.1%}  "
        f"95% CI [{lo:+.1%}, {hi:+.1%}]  ({len(draws)} draws)")
say("  A positive DiD is the M1/M2 prediction: as the portfolio margin "
    "closed, the price margin opened. A CI spanning zero means the paper "
    "reports a bounded null, with the bound stated.")

# ======================================================================
# 4. dose-response, not a binary split
# ======================================================================
say("\n" + "=" * 64)
say("E. DOSE-RESPONSE ON TRAILING RELATIVE RETURN (replaces the binary)")
say("=" * 64)
for tag, M in MATS.items():
    say(f"  --- {tag} ---")
    for era in ERAS:
        d = samp[samp["era3"] == era].copy()
        if len(d) < 500:
            continue
        d["qtile"] = pd.qcut(d["rel4q"], 5, labels=False, duplicates="drop")
        line = []
        for qv in sorted(d["qtile"].dropna().unique()):
            dd_ = d[d["qtile"] == qv]
            t_, c_, _ = code_events(M, dd_["row"].to_numpy(np.int64),
                                    dd_["q0"].to_numpy(np.int64))
            a1, _ = aj_cif(t_, c_, WIN)
            line.append(f"Q{int(qv) + 1} {a1:.1%}")
        say(f"    {era:<10s} (Q1 = worst trailing)  " + "  ".join(line))
say("  A monotone fall from Q1 to Q5 is the behavioural signature. A flat "
    "line says fee changes are scheduled, not reactive, whatever the "
    "binary split shows.")

# ======================================================================
# 5. event-time fee path, differenced against the industry
# ======================================================================
say("\n" + "=" * 64)
say("F. FEE PATH AROUND CAPITULATION, MINUS THE INDUSTRY PATH")
say("=" * 64)
caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
caps = caps[caps["cq"].notna()]
caps["row"] = caps["wficn"].astype("int64").map(FORD)
caps["b"] = caps["cq"].map(lambda q: QORD.get(q - WIN, np.nan))
cap2 = caps[caps["row"].notna() & caps["b"].notna()].copy()
cap2["row"] = cap2["row"].astype(np.int64)
cap2["b"] = cap2["b"].astype(np.int64)
say(f"capitulation events on the fee grid: {len(cap2):,} of {len(caps):,}")

OFFS = np.arange(0, 2 * WIN + 1)
for tag, M in MATS.items():
    exc = {k: [] for k in OFFS}
    raw = {k: [] for k in OFFS}
    for b, grp in cap2.groupby("b"):
        cols = b + OFFS
        cols = cols[cols < NQ]
        if len(cols) < 2:
            continue
        base_all = M[:, b]
        diff_all = M[:, cols] - base_all[:, None]      # industry, same anchor
        med = np.nanmedian(diff_all, axis=0)
        r = grp["row"].to_numpy(np.int64)
        mine = M[np.ix_(r, cols)] - M[r, b][:, None]
        for j, k in enumerate(OFFS[:len(cols)]):
            v = mine[:, j]
            v = v[np.isfinite(v)]
            if len(v):
                raw[k].extend(v.tolist())
                if np.isfinite(med[j]):
                    exc[k].extend((v - med[j]).tolist())
    say(f"  --- {tag} --- median fee change vs t-8 (bps); "
        f"'excess' nets out the industry path from the same base quarter")
    for label, dct in (("raw   ", raw), ("excess", exc)):
        parts = []
        for k in range(0, 2 * WIN + 1, 2):
            v = dct.get(k, [])
            parts.append(f"t{k - WIN:+d}:{np.median(v):+.0f}(n{len(v)})"
                         if v else f"t{k - WIN:+d}:-")
        say(f"    {label}  " + "  ".join(parts))
say("  Only the EXCESS row is evidence. The raw row will fall in every era "
    "because industry fees fell; stage 38 printed only the raw row.")

# ======================================================================
# 6. MA-11: landmark design replaces the post-outcome selection
# ======================================================================
say("\n" + "=" * 64)
say("G. CAPITULATORS vs RESISTERS -- LANDMARK AT QUARTER 8 (MA-11)")
say("=" * 64)
say("  Stage 38 kept spells with end_dur >= 12 (an outcome realised after "
    "t0) and then measured them at quarter 8, inside the window that "
    "selected them. Here the landmark uses only what is knowable at "
    "quarter 8: the spell is still running and has not capitulated yet.")

sp2 = sp.copy()
sp2["row"] = sp2["wficn"].astype("int64").map(FORD)
sp2 = sp2[sp2["row"].notna()].copy()
sp2["row"] = sp2["row"].astype(np.int64)
sp2["start_o"] = pd.PeriodIndex(sp2["start_p"], freq="Q").map(
    lambda q: QORD.get(q, np.nan))
sp2 = sp2[sp2["start_o"].notna()].copy()
sp2["start_o"] = sp2["start_o"].astype(np.int64)
sp2["cap_o"] = pd.PeriodIndex(sp2["m_cal_q"], freq="Q").map(
    lambda q: QORD.get(q, np.nan) if pd.notna(q) else np.nan)

lm = sp2[(sp2["end_dur"] >= WIN)
         & (sp2["cap_o"].isna() | (sp2["cap_o"] > sp2["start_o"] + WIN))]
lm_row = lm["row"].to_numpy(np.int64)
lm_q0 = (lm["start_o"].to_numpy(np.int64) + WIN)
lm_ok = lm_q0 < NQ
capr = cap2["row"].to_numpy(np.int64)
capq = (cap2["b"].to_numpy(np.int64) + WIN)          # = the crossing quarter

for tag, M in MATS.items():
    t1, c1, _ = code_events(M, capr, capq)
    a1, d1_ = aj_cif(t1, c1, WIN)
    t2, c2, _ = code_events(M, lm_row[lm_ok], lm_q0[lm_ok])
    a2, d2_ = aj_cif(t2, c2, WIN)
    say(f"  {tag}: P(cut >=10bps within 8q) at capitulation {a1:.1%} "
        f"(n {len(t1):,}, death {d1_:.1%}) vs landmark resisters {a2:.1%} "
        f"(n {len(t2):,}, death {d2_:.1%})")
say("  Higher at capitulation = the two surrenders are BUNDLED (a fund in "
    "trouble does both). Higher among resisters = SUBSTITUTES (cut price "
    "instead of conviction). Stage 38 could not tell these apart because "
    "its resister set was chosen using the future.")

say("\nSTAGE 61 DONE - aggregates only.")
say("Superseded stage 38 entirely: its detector deleted deaths, its fee was "
    "equal-weighted, its arms were defined asymmetrically, and its "
    "resister comparison selected on the outcome. Any stage-38 number "
    "still in the draft should be replaced by the matching number here.")
P.write_report("referee_61_fee_cuts_rebuilt.txt", log)
