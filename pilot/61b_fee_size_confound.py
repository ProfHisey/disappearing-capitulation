"""Stage 61b: IS THE FEE GAP A DECISION OR IS IT ARITHMETIC?

Stage 61 found that in 1995-2009 stressed funds were LESS likely to cut fees
than unstressed funds (-3.8pp, CI [-5.2, -2.5]), and that by 2010-23 the
difference is gone (+1.0pp, CI [-0.1, +2.0]). The DiD of +4.8pp is real. What
it MEANS is not established, because there is a mechanical channel that
predicts exactly that pattern with nobody deciding anything.

THE CONFOUND. An expense ratio is a rate: costs divided by assets. Three
features of fund contracts make it fall when assets rise.
  1. Advisory fee BREAKPOINTS lower the rate as assets cross thresholds.
  2. Fixed costs (transfer agency, audit, legal, registration) spread over a
     growing base.
  3. Expense CAPS and WAIVERS shrink as assets grow toward the cap.
Unstressed funds take in money, so their expense ratio falls without anyone
deciding to cut price. Stressed funds bleed assets, so theirs rises. That
alone can produce a negative stressed-minus-unstressed gap, and "the fee-size
link weakened over time" is a sufficient explanation for the DiD.

Stage 61's dose-response already points this way: in 1995-2009 the gradient
runs the WRONG way for a behavioural story (Q1 worst 18.9%, Q5 best 23.5%).
The best performers cut fees most, which is what mechanics predicts and
surrender does not.

WHAT THIS STAGE DOES
  1. Measurement first. How often does CRSP actually update an expense ratio?
     If it is annual, an 8-quarter window contains about two real
     observations, not eight, and every interval in stage 61 is too narrow.
  2. Sizes the mechanical gradient directly: P(cut) and median fee change by
     quintile of contemporaneous asset growth.
  3. DECOMPOSES the 1995-2009 gap by standardisation (DiNardo-Fortin-Lemieux
     reweighting): what would the stressed-minus-unstressed gap be if stressed
     funds had had unstressed funds' asset-growth distribution? Whatever
     survives that is the part not explained by size. This is the same move
     that showed 87-95% of the menu-size effect in the other paper was
     combinatorics.
  4. Splits the fee into MANAGEMENT FEE and everything else. A management fee
     cut needs board approval and a prospectus supplement. That is a decision.
     An expense-ratio fall driven by waivers and breakpoints is not. If the
     gap lives in the expense ratio but not the management fee, the finding
     is arithmetic.
  5. A constant-assets subsample: fund-years whose assets barely moved. If the
     gap vanishes there, it was never about price.

IDENTIFICATION WARNING, STATED UP FRONT. Contemporaneous asset growth is a
MEDIATOR, not a confounder: stress causes outflows. Conditioning on it does
not estimate the causal effect of stress on fee-setting, and a referee will
say so. It estimates a DECOMPOSITION -- how much of the observed gap runs
through the size channel. Both the pre-determined stratifier (asset growth
BEFORE the window, a legitimate control) and the contemporaneous one (the
mediator) are reported, labelled, and must not be read the same way.

Aggregates only; report: output/referee_61b_fee_size_confound.txt
First run re-streams Fund Summary to capture TNA and mgmt_fee, which the
stage-61 cache does not carry, then caches its own panel.

  python 61b_fee_size_confound.py
  python 61b_fee_size_confound.py --rebuild
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
CACHE = OUT / "s61b_size_panel.parquet"

CUT_BPS = 10.0
WIN = 8
BOOT = 2000
SEED = 20260821
ERAS = ["1980-94", "1995-2009", "2010-23"]
rng = np.random.default_rng(SEED)

log = ["FEE GAP: DECISION OR ARITHMETIC? (stage 61b)", "=" * 64]


def say(s):
    log.append(s)
    print(s, flush=True)


# ======================================================================
# 0. panel with TNA and management fee
# ======================================================================
if CACHE.exists() and "--rebuild" not in sys.argv:
    fw = pd.read_parquet(CACHE)
    say(f"loaded cached panel: {CACHE}  ({len(fw):,} rows)")
else:
    say("streaming Fund Summary for tna_latest and mgmt_fee (first run) ...")
    want = ["crsp_fundno", "caldt", "exp_ratio", "mgmt_fee", "tna_latest"]
    parts = []
    for ch in pd.read_csv(SRC / "crsp_mf" / "Fund Summary.csv",
                          chunksize=2_000_000, low_memory=False,
                          encoding="latin-1"):
        ch.columns = [c.lower() for c in ch.columns]
        cols = [c for c in want if c in ch.columns]
        missing = set(want) - set(cols)
        if "exp_ratio" not in cols or "tna_latest" not in cols:
            raise SystemExit(f"need exp_ratio and tna_latest; missing "
                             f"{missing}; saw {list(ch.columns)[:25]}")
        parts.append(ch[cols])
    fees = pd.concat(parts, ignore_index=True)
    del parts
    if "mgmt_fee" not in fees.columns:
        say("  !! mgmt_fee absent from Fund Summary. Block 4 (the decision "
            "test) CANNOT run on this data and will be skipped.")
        fees["mgmt_fee"] = np.nan

    for c in ("exp_ratio", "mgmt_fee", "tna_latest"):
        if fees[c].dtype == object:
            n_before = int(fees[c].notna().sum())
            fees[c] = pd.to_numeric(fees[c], errors="coerce")
            lost = n_before - int(fees[c].notna().sum())
            say(f"  {c}: text column, {lost:,} values coerced to missing")
    say(f"  coverage: exp_ratio {fees['exp_ratio'].notna().mean():.1%}, "
        f"mgmt_fee {fees['mgmt_fee'].notna().mean():.1%}, "
        f"tna_latest {fees['tna_latest'].notna().mean():.1%}")

    fees = fees[fees["exp_ratio"].notna() & (fees["exp_ratio"] > 0)
                & (fees["exp_ratio"] < 0.25)]
    fees["quarter"] = pd.to_datetime(fees["caldt"]).dt.to_period("Q")
    m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                     encoding="latin-1")
    m1.columns = [c.lower() for c in m1.columns]
    link = m1[["crsp_fundno", "wficn"]].drop_duplicates()
    uni = link.groupby("crsp_fundno")["wficn"].nunique().eq(1)
    link = link[link["crsp_fundno"].isin(uni[uni].index)]
    fees = fees.merge(link, on="crsp_fundno", how="inner")
    fees["wficn"] = fees["wficn"].astype("int64")

    fees["_w"] = fees["tna_latest"].where(fees["tna_latest"] > 0)
    fees["_er_n"] = fees["exp_ratio"] * fees["_w"]
    fees["_mg_n"] = fees["mgmt_fee"] * fees["_w"]
    g = fees.groupby(["wficn", "quarter"])
    agg = g[["_er_n", "_mg_n", "_w"]].sum(min_count=1)
    out = pd.DataFrame({
        "er_bps": (agg["_er_n"] / agg["_w"]) * 1e4,
        "mg_bps": (agg["_mg_n"] / agg["_w"]) * 1e4,
        "tna": agg["_w"],
        "n_class": g["exp_ratio"].size(),
    }).reset_index()
    # every derived quantity is materialised HERE, never recomputed later
    out["other_bps"] = out["er_bps"] - out["mg_bps"]
    out["log_tna"] = np.log(out["tna"].where(out["tna"] > 0))
    fw = out
    fw.to_parquet(CACHE, index=False)
    say(f"  cached -> {CACHE}")

say(f"panel: {len(fw):,} wficn-quarters, {fw['wficn'].nunique():,} funds")
say(f"  mgmt_fee present in {fw['mg_bps'].notna().mean():.1%} of "
    f"fund-quarters; 'other' (er - mgmt) median "
    f"{fw['other_bps'].median():.0f}bps")
HAS_MG = fw["mg_bps"].notna().mean() > 0.30
if not HAS_MG:
    say("  !! mgmt_fee coverage too thin for the decision test; block 4 "
        "will report what it can and flag the coverage.")

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


ER, MG, LT = mat("er_bps"), mat("mg_bps"), mat("log_tna")
OTH = mat("other_bps")
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

# ======================================================================
# 1. MEASUREMENT: how often does CRSP actually move an expense ratio?
# ======================================================================
say("\n" + "=" * 64)
say("1. MEASUREMENT FIRST -- how many real observations are in 8 quarters?")
say("=" * 64)
d = np.diff(ER, axis=1)
seen = np.isfinite(d)
say(f"  quarter-to-quarter expense ratio: unchanged in "
    f"{np.nansum(np.abs(d[seen]) < 0.05) / max(seen.sum(), 1):.1%} of "
    f"consecutive observed pairs")
qtr_of_yr = np.array([q.quarter for q in QS])[1:]
moved = seen & (np.abs(d) >= 0.05)
say("  share of all fee CHANGES landing in each calendar quarter:")
tot = moved.sum()
say("    " + "  ".join(
    f"Q{k}: {moved[:, qtr_of_yr == k].sum() / max(tot, 1):5.1%}"
    for k in (1, 2, 3, 4)))
say("  If changes cluster in one calendar quarter, CRSP is carrying an "
    "annual number forward and an 8-quarter window holds ~2 independent "
    "observations, not 8. Every stage-61 interval is then too narrow and "
    "the paper must say the unit is fund-YEARS.")

# ======================================================================
# 2. helpers
# ======================================================================
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


# ---- the fund-year sample, identical construction to stage 61 ---------
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
bwd = np.maximum(Q_ - WIN, 0)
s["g_post"] = LT[R_, fwd] - LT[R_, Q_]      # MEDIATOR (stress causes this)
s["g_pre"] = LT[R_, Q_] - LT[R_, bwd]       # pre-determined, legit control
s["d_er"] = ER[R_, fwd] - ER[R_, Q_]
say(f"\nfund-years: {len(s):,}; asset growth observable for "
    f"post {s['g_post'].notna().mean():.1%}, pre {s['g_pre'].notna().mean():.1%}")
say("  funds that die inside the window have no end-of-window TNA, so the "
    "growth-conditioned blocks below run on survivors. That is a real "
    "restriction and it is why block 5 exists.")

# ======================================================================
# 3. how big is the mechanical gradient?
# ======================================================================
say("\n" + "=" * 64)
say("2. THE MECHANICAL GRADIENT: fee change by ASSET GROWTH, ignoring stress")
say("=" * 64)
for era in ERAS:
    d0 = s[(s["era3"] == era) & s["g_post"].notna()]
    if len(d0) < 500:
        continue
    d0 = d0.assign(gq=pd.qcut(d0["g_post"], 5, labels=False,
                              duplicates="drop"))
    parts, med = [], []
    for q in sorted(d0["gq"].dropna().unique()):
        z = d0[d0["gq"] == q]
        parts.append(f"G{int(q) + 1} {pcut(ER, z['row'].to_numpy(np.int64), z['q0'].to_numpy(np.int64)):.1%}")
        med.append(f"G{int(q) + 1} {z['d_er'].median():+.0f}")
    say(f"  {era:<10s} P(cut) by asset-growth quintile (G1 = shrank most)")
    say("             " + "  ".join(parts))
    say(f"             median 8q fee change (bps): " + "  ".join(med))
    say(f"             mean growth G1 {d0[d0.gq == d0.gq.min()]['g_post'].mean():+.2f} "
        f"log points, G5 {d0[d0.gq == d0.gq.max()]['g_post'].mean():+.2f}")
say("  A steep rise from G1 to G5 IS the confound. Funds that grow cut "
    "fees; funds that shrink do not. No decision required.")

# ======================================================================
# 4. standardisation: how much of the gap survives equal asset growth?
# ======================================================================
say("\n" + "=" * 64)
say("3. DECOMPOSITION -- the gap if stressed funds had grown like unstressed")
say("=" * 64)
say("  Reweighting (DiNardo-Fortin-Lemieux): within each asset-growth decile "
    "take the stressed-minus-unstressed difference, then average those "
    "differences using the UNSTRESSED growth distribution as weights. "
    "MEDIATOR CAVEAT: this is a decomposition, not a causal effect.")
dec = []
for era in ERAS:
    d0 = s[(s["era3"] == era) & s["g_post"].notna()
           & (s["stressed"] | s["unstressed"])]
    if len(d0) < 1000:
        continue
    d0 = d0.assign(gq=pd.qcut(d0["g_post"], 10, labels=False,
                              duplicates="drop"))
    ds, du = d0[d0["stressed"]], d0[d0["unstressed"]]
    raw = (pcut(ER, ds["row"].to_numpy(np.int64), ds["q0"].to_numpy(np.int64))
           - pcut(ER, du["row"].to_numpy(np.int64),
                  du["q0"].to_numpy(np.int64)))
    num, den, filled = 0.0, 0.0, 0
    for q in sorted(d0["gq"].dropna().unique()):
        a = ds[ds["gq"] == q]
        b = du[du["gq"] == q]
        if len(a) < 40 or len(b) < 40:
            continue
        pa = pcut(ER, a["row"].to_numpy(np.int64), a["q0"].to_numpy(np.int64))
        pb = pcut(ER, b["row"].to_numpy(np.int64), b["q0"].to_numpy(np.int64))
        if not (np.isfinite(pa) and np.isfinite(pb)):
            continue
        w = len(b)
        num += w * (pa - pb)
        den += w
        filled += 1
    adj = num / den if den else np.nan
    expl = (1 - adj / raw) if (np.isfinite(adj) and abs(raw) > 1e-9) else np.nan
    say(f"  {era:<10s} raw gap {raw:+.1%}   size-standardised gap "
        f"{adj:+.1%}   share explained by asset growth "
        f"{expl:6.1%}   ({filled} deciles usable)")
    say(f"             mean growth: stressed {ds['g_post'].mean():+.2f}, "
        f"unstressed {du['g_post'].mean():+.2f} log points")
    dec.append({"era": era, "raw": raw, "adj": adj, "explained": expl})
pd.DataFrame(dec).round(4).to_csv(OUT / "s61b_decomposition.csv", index=False)
say("  If 'share explained' is near 100%, Finding 4 is arithmetic and the "
    "paper says so plainly. If a stable remainder survives in both eras, "
    "that remainder is the behavioural margin and it is the only number "
    "worth reporting.")

say("\n  same decomposition on PRE-PERIOD growth (a legitimate control, "
    "determined before the window opens):")
for era in ERAS:
    d0 = s[(s["era3"] == era) & s["g_pre"].notna()
           & (s["stressed"] | s["unstressed"])]
    if len(d0) < 1000:
        continue
    d0 = d0.assign(gq=pd.qcut(d0["g_pre"], 10, labels=False,
                              duplicates="drop"))
    ds, du = d0[d0["stressed"]], d0[d0["unstressed"]]
    raw = (pcut(ER, ds["row"].to_numpy(np.int64), ds["q0"].to_numpy(np.int64))
           - pcut(ER, du["row"].to_numpy(np.int64),
                  du["q0"].to_numpy(np.int64)))
    num, den = 0.0, 0.0
    for q in sorted(d0["gq"].dropna().unique()):
        a, b = ds[ds["gq"] == q], du[du["gq"] == q]
        if len(a) < 40 or len(b) < 40:
            continue
        pa = pcut(ER, a["row"].to_numpy(np.int64), a["q0"].to_numpy(np.int64))
        pb = pcut(ER, b["row"].to_numpy(np.int64), b["q0"].to_numpy(np.int64))
        if np.isfinite(pa) and np.isfinite(pb):
            num += len(b) * (pa - pb)
            den += len(b)
    say(f"  {era:<10s} raw {raw:+.1%}  pre-growth-standardised "
        f"{num / den if den else np.nan:+.1%}")

# ======================================================================
# 5. management fee: the part that requires a decision
# ======================================================================
say("\n" + "=" * 64)
say("4. MANAGEMENT FEE vs THE REST -- separating decisions from mechanics")
say("=" * 64)
say("  A management fee cut needs board approval and a prospectus "
    "supplement. A fall in 'other' expenses can be waivers, breakpoints or "
    "fixed costs spread over a bigger base. Same detector, three series.")
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
say("  THE DISCRIMINATOR: a gap that lives in 'other' but not in the "
    "management fee is arithmetic. A gap in the management fee is a "
    "decision, and that is the only version of Finding 4 worth defending.")

# ======================================================================
# 6. constant-assets subsample
# ======================================================================
say("\n" + "=" * 64)
say("5. FUND-YEARS WHOSE ASSETS BARELY MOVED (|8q growth| < 10 log pts)")
say("=" * 64)
flat = s[s["g_post"].notna() & (s["g_post"].abs() < 0.10)]
say(f"  n = {len(flat):,} fund-years ({len(flat) / max(len(s), 1):.1%})")
for era in ERAS:
    d0 = flat[flat["era3"] == era]
    a, b = d0[d0["stressed"]], d0[d0["unstressed"]]
    if len(a) < 100 or len(b) < 100:
        say(f"  {era:<10s} too few (stressed {len(a):,}, unstressed {len(b):,})")
        continue
    pa = pcut(ER, a["row"].to_numpy(np.int64), a["q0"].to_numpy(np.int64))
    pb = pcut(ER, b["row"].to_numpy(np.int64), b["q0"].to_numpy(np.int64))
    funds = np.unique(d0["row"].to_numpy(np.int64))
    ia = {f: np.where(a["row"].to_numpy(np.int64) == f)[0] for f in funds}
    ib = {f: np.where(b["row"].to_numpy(np.int64) == f)[0] for f in funds}
    ra, qa = a["row"].to_numpy(np.int64), a["q0"].to_numpy(np.int64)
    rb, qb = b["row"].to_numpy(np.int64), b["q0"].to_numpy(np.int64)
    dr = []
    for _ in range(500):
        pick = rng.choice(funds, len(funds), replace=True)
        sa = np.concatenate([ia[f] for f in pick if len(ia[f])]) \
            if any(len(ia[f]) for f in pick) else np.array([], np.int64)
        sb = np.concatenate([ib[f] for f in pick if len(ib[f])]) \
            if any(len(ib[f]) for f in pick) else np.array([], np.int64)
        if len(sa) < 50 or len(sb) < 50:
            continue
        v = pcut(ER, ra[sa], qa[sa]) - pcut(ER, rb[sb], qb[sb])
        if np.isfinite(v):
            dr.append(v)
    lo, hi = (np.percentile(dr, [2.5, 97.5]) if len(dr) > 100
              else (np.nan, np.nan))
    say(f"  {era:<10s} stressed {pa:.1%}  unstressed {pb:.1%}  "
        f"gap {pa - pb:+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  "
        f"(n {len(a):,}/{len(b):,})")
say("  Assets held roughly fixed, the size channel is shut off. A gap that "
    "survives here is the real thing. A gap that disappears was never "
    "about anyone's pricing decision.")

say("\nSTAGE 61b DONE - aggregates only.")
say("Finding 4 cannot be written until blocks 3, 4 and 5 agree. If they say "
    "arithmetic, the honest paper reports the mechanical result, which is "
    "itself worth knowing: an expense ratio is a rate, so underperforming "
    "funds get MORE expensive without anyone raising a fee.")
P.write_report("referee_61b_fee_size_confound.txt", log)
