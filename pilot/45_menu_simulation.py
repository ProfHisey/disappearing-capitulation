# 45_menu_simulation.py -- the practical question: in a realistic menu
# containing BOTH index and active funds, what does sorting by fee get you
# versus sorting by recent return? Decomposes into (A) composition -- the
# rule surfaces an index fund -- and (B) within-active selection.
import os, sys, numpy as np, pandas as pd

DATA_LIB = os.environ.get("DATA_LIB", r"E:\Finance\data\sources")
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
PANEL = os.path.join(CACHE, "s45_panel_with_passive.parquet")
MENU_SIZES, REPS, SEED = (3, 5, 10), 200, 20260820
MIN_EXP, MAX_EXP = 0.0001, 0.10
rng = np.random.default_rng(SEED)

# ---------- panel: same build as 41c but KEEPING passive funds ----------
if os.path.exists(PANEL):
    panel = pd.read_parquet(PANEL); print("panel: using cache")
else:
    FS  = os.path.join(DATA_LIB, "crsp_mf", "Fund Summary.csv")
    MFL = os.path.join(DATA_LIB, "mflinks", "mflink1.csv")
    c = lambda p: {k.lower(): k for k in pd.read_csv(p, nrows=3, encoding="latin-1").columns}
    fs_c, ml_c = c(FS), c(MFL)
    link = pd.read_csv(MFL, encoding="latin-1", usecols=[ml_c["crsp_fundno"], ml_c["wficn"]])
    link = link.rename(columns={ml_c["crsp_fundno"]: "crsp_fundno", ml_c["wficn"]: "wficn"}).dropna().drop_duplicates()
    ren = {fs_c["crsp_fundno"]: "crsp_fundno", fs_c["caldt"]: "date", fs_c["crsp_obj_cd"]: "obj"}
    parts = []
    for ch in pd.read_csv(FS, encoding="latin-1", low_memory=False,
                          usecols=list(ren), chunksize=2_000_000):
        ch = ch.rename(columns=ren); ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
        ch = ch.dropna(subset=["date", "crsp_fundno"]); ch["year"] = ch["date"].dt.year
        parts.append(ch.loc[ch.year >= 1989, ["crsp_fundno", "year", "obj"]])
    s = pd.concat(parts).merge(link, on="crsp_fundno", how="inner")
    s["cat"] = s.obj.astype(str).str.upper().str.strip().str[:4]
    s = s[s.cat.str.startswith("ED")]
    cat = s.groupby(["wficn", "year"], as_index=False).agg(
        cat=("cat", lambda x: x.mode().iat[0] if len(x.mode()) else np.nan))

    fm  = pd.read_parquet(os.path.join(CACHE, "fund_month_v3_tnafix.parquet"))
    cov = pd.read_parquet(os.path.join(CACHE, "covars.parquet"))
    flg = pd.read_parquet(os.path.join(CACHE, "flags.parquet"))
    fm = fm[fm.wficn.isin(flg.loc[flg.dom_eq, "wficn"])].copy()   # KEEP passive
    cov = cov.copy()
    cov.loc[(cov.exp_ratio < MIN_EXP) | (cov.exp_ratio > MAX_EXP), "exp_ratio"] = np.nan
    cov["year"] = pd.PeriodIndex(cov.quarter.astype(str), freq="Q").year
    fee = (cov.dropna(subset=["exp_ratio"]).sort_values(["wficn", "year"])
              .groupby(["wficn", "year"], as_index=False).agg(exp_ratio=("exp_ratio", "last")))
    fm["ym"] = pd.PeriodIndex(fm.caldt, freq="M"); fm["year"] = fm.ym.dt.year
    panel = (fm.rename(columns={"fret": "ret"})[["wficn", "ym", "year", "ret", "tna"]]
               .dropna(subset=["ret"]).merge(fee, on=["wficn", "year"], how="left")
               .merge(cat, on=["wficn", "year"], how="left")
               .merge(flg[["wficn", "passive"]], on="wficn", how="left"))
    panel.to_parquet(PANEL, index=False)
print(f"panel: {panel.wficn.nunique():,} funds "
      f"({panel.loc[panel.passive, 'wficn'].nunique():,} passive), {len(panel):,} rows")

# ---------- formation table: fee + trailing 12m + forward 1y and 5y ----------
comp = lambda x: np.prod(1 + pd.to_numeric(x, errors="coerce").dropna().values) - 1
rows = []
for y in range(1990, 2026):
    past = panel[(panel.ym >= pd.Period(f"{y}-01", "M")) & (panel.ym <= pd.Period(f"{y}-12", "M"))]
    t = past.groupby("wficn").agg(trail12=("ret", comp), n=("ret", "size"),
                                  exp_ratio=("exp_ratio", "last"), cat=("cat", "last"),
                                  tna=("tna", "last"), passive=("passive", "last"))
    t = t[t.n >= 12]
    for h, lab in [(1, "fwd1"), (5, "fwd5")]:
        fut = panel[(panel.ym >= pd.Period(f"{y+1}-01", "M")) & (panel.ym <= pd.Period(f"{y+h}-12", "M"))]
        t = t.join(fut.groupby("wficn")["ret"].apply(comp).rename(lab))
    t["year"] = y; rows.append(t.reset_index())
ft = pd.concat(rows, ignore_index=True).dropna(subset=["exp_ratio", "trail12", "fwd1", "cat", "tna"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
ft.to_parquet(os.path.join(CACHE, "s45_formations.parquet"), index=False)
print(f"formations: {len(ft):,} fund-years; passive share of rows "
      f"{100*ft.passive.mean():.1f}%")

# ---------- simulate menus ----------
res = []
for K in MENU_SIZES:
    for (y, ct), g in ft.groupby(["year", "cat"]):
        if len(g) < K: continue
        w = g.tna.values / g.tna.values.sum()          # bigger funds likelier on a menu
        idx = np.arange(len(g))
        for _ in range(REPS):
            pick = rng.choice(idx, size=K, replace=False, p=w)
            m = g.iloc[pick]
            cheap, perf = m.exp_ratio.idxmin(), m.trail12.idxmax()
            res.append({"K": K, "year": y, "cat": ct,
                        "has_index": bool(m.passive.any()),
                        "fee_fwd1": m.loc[cheap, "fwd1"], "perf_fwd1": m.loc[perf, "fwd1"],
                        "rand_fwd1": m.fwd1.mean(),
                        "fee_fwd5": m.loc[cheap, "fwd5"], "perf_fwd5": m.loc[perf, "fwd5"],
                        "fee_is_index": bool(m.loc[cheap, "passive"]),
                        "perf_is_index": bool(m.loc[perf, "passive"])})
r = pd.DataFrame(res)
r.to_parquet(os.path.join(CACHE, "s45_menu_draws.parquet"), index=False)

print("\n" + "=" * 72); print("MENU SIMULATION -- mean forward returns by rule (%)"); print("=" * 72)
for K, d in r.groupby("K"):
    yr = d.groupby("year")[["fee_fwd1", "perf_fwd1", "rand_fwd1"]].mean()
    print(f"\nmenu of {K} per category   ({len(d):,} simulated menus)")
    print(f"  sort by FEE        {yr.fee_fwd1.mean()*100:6.2f}%/yr")
    print(f"  sort by 1y RETURN  {yr.perf_fwd1.mean()*100:6.2f}%/yr")
    print(f"  pick at random     {yr.rand_fwd1.mean()*100:6.2f}%/yr")
    print(f"  P(fee pick beats return pick), menu by menu: "
          f"{100*(d.fee_fwd1 > d.perf_fwd1).mean():.1f}%")
    print(f"  P(fee rule better), by year: "
          f"{100*(yr.fee_fwd1 > yr.perf_fwd1).mean():.0f}% of years")
    print(f"  fee rule lands on an index fund: {100*d.fee_is_index.mean():.1f}% of menus"
          f"   | return rule: {100*d.perf_is_index.mean():.1f}%")

print("\n" + "=" * 72); print("DECOMPOSITION -- composition vs within-active"); print("=" * 72)
for K, d in r.groupby("K"):
    a = d[d.has_index]; b = d[~d.has_index]
    print(f"\nmenu of {K}:")
    for lab, sub in [("menus WITH an index fund", a), ("menus with NO index fund", b)]:
        if len(sub) < 100: continue
        yr = sub.groupby("year")[["fee_fwd1", "perf_fwd1"]].mean()
        print(f"  {lab:<26} fee {yr.fee_fwd1.mean()*100:6.2f}%  "
              f"return {yr.perf_fwd1.mean()*100:6.2f}%  "
              f"edge {(yr.fee_fwd1-yr.perf_fwd1).mean()*100:+.2f}pp  "
              f"| P(fee wins) {100*(sub.fee_fwd1 > sub.perf_fwd1).mean():.1f}%  ({len(sub):,} menus)")

print("""
PLAIN READING
  Top block: the practical question, answered. Two rules, realistic menus.
  The headline number you wanted is P(fee pick beats return pick), computed
  menu by menu rather than averaged first.
  Bottom block: WHERE the edge comes from. If it is large on menus with an
  index fund and near zero without one, "sort by fees" is "buy the index
  fund" in disguise -- true and useful, but SPIVA owns the finding and the
  paper's contribution has to be the decision architecture.
""")
