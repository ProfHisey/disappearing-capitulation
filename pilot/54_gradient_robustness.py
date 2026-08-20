# 54_gradient_robustness.py -- the three questions a referee asks first
# about the menu-size gradient, answered in one pass.
#
#   A. DRAW RULE. Menus were drawn TNA-weighted (big funds likelier to be on
#      a plan menu). Does the gradient survive uniform draws?
#   B. INDEX-FREE MENUS. The cheap pick is a passive fund most of the time.
#      Is the gradient just "cheapest = index fund"? Re-run on menus that
#      contain NO passive option.
#   C. CATEGORY DEFINITION. Sleeves are defined by 4-char crsp_obj_cd. Does
#      the result hold under Lipper class, an independent taxonomy?
#
# Only K=3 and K=20 are simulated -- the gradient is their difference and
# the middle sizes are not needed here.
#
#   python 54_gradient_robustness.py
import os, sys, numpy as np, pandas as pd

DATA_LIB = os.environ.get("DATA_LIB", r"E:\Finance\data\sources")
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE, OUT = os.path.join(HERE, "cache"), os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)
KS, REPS, BOOT, SEED = (3, 20), 200, 2000, 20260826
rng = np.random.default_rng(SEED)

ft = pd.read_parquet(os.path.join(CACHE, "s53_formations_alpha.parquet"))
ft = ft.dropna(subset=["exp_ratio", "trail12", "tna", "cat"])
ft = ft[(ft.cat != "UNK") & (ft.tna > 0)]
print(f"formations {len(ft):,}, {ft.wficn.nunique():,} funds")

# ---------- C. Lipper class as an alternative sleeve definition ---------
FS = os.path.join(DATA_LIB, "crsp_mf", "Fund Summary.csv")
MFL = os.path.join(DATA_LIB, "mflinks", "mflink1.csv")
lip = None
try:
    hd = pd.read_csv(FS, nrows=3, encoding="latin-1", low_memory=False)
    cl = {c.lower(): c for c in hd.columns}
    if "lipper_class" in cl:
        link = pd.read_csv(MFL, encoding="latin-1",
                           usecols=[c for c in pd.read_csv(MFL, nrows=1).columns
                                    if c.lower() in ("crsp_fundno", "wficn")])
        link.columns = [c.lower() for c in link.columns]
        ren = {cl["crsp_fundno"]: "crsp_fundno", cl["caldt"]: "date",
               cl["lipper_class"]: "lip"}
        parts = []
        for ch in pd.read_csv(FS, encoding="latin-1", low_memory=False,
                              usecols=list(ren), chunksize=2_000_000):
            ch = ch.rename(columns=ren)
            ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
            ch = ch.dropna(subset=["date", "crsp_fundno", "lip"])
            ch["year"] = ch["date"].dt.year
            parts.append(ch.loc[ch.year >= 1989, ["crsp_fundno", "year", "lip"]])
        s = pd.concat(parts).merge(link, on="crsp_fundno", how="inner")
        lip = (s.groupby(["wficn", "year"], as_index=False)
                 .agg(lip=("lip", lambda x: x.mode().iat[0] if len(x.mode()) else np.nan)))
        ft = ft.merge(lip, on=["wficn", "year"], how="left")
        print(f"lipper: {ft.lip.notna().mean()*100:.1f}% coverage, "
              f"{ft.lip.nunique()} classes")
        ft.to_parquet(os.path.join(CACHE, "s54_formations_lipper.parquet"),
                      index=False)
    else:
        print("lipper_class not in Fund Summary - skipping check C")
except Exception as e:
    print(f"lipper check skipped ({type(e).__name__}: {e})")

MEASURES = [("ff4", "4-factor alpha", 10000), ("capm", "CAPM alpha", 10000),
            ("rein5", "raw net 5y (ann.)", 10000 / 5)]


def simulate(df, catcol, weighted, index_free):
    out = []
    for (y, ct), g in df.groupby(["year", catcol]):
        if index_free and g.passive.any():
            g = g[~g.passive]
        if len(g) < max(KS): continue
        p = g.tna.values / g.tna.values.sum() if weighted else None
        idx = np.arange(len(g))
        for K in KS:
            for _ in range(REPS):
                m = g.iloc[rng.choice(idx, size=K, replace=False, p=p)]
                c, h = m.exp_ratio.idxmin(), m.trail12.idxmax()
                if c == h: continue
                row = {"K": K, "year": y}
                for col, _lab, _sc in MEASURES:
                    if col in m and m.loc[[c, h], col].notna().all():
                        row[col] = m.loc[h, col] - m.loc[c, col]
                out.append(row)
    return pd.DataFrame(out)


def gradient(d, col, scale):
    a = d[d.K == KS[0]].dropna(subset=[col]).groupby("year")[col].median()
    b = d[d.K == KS[1]].dropna(subset=[col]).groupby("year")[col].median()
    yrs = a.index.intersection(b.index)
    if len(yrs) < 5: return None
    diff = (b.loc[yrs] - a.loc[yrs]) * scale
    draws = [diff.loc[rng.choice(yrs, len(yrs), replace=True)].median()
             for _ in range(BOOT)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return diff.median(), lo, hi, len(yrs)


specs = []
for weighted in (True, False):
    for index_free in (False, True):
        specs.append(("crsp_obj_cd", weighted, index_free))
if lip is not None:
    specs.append(("lip", True, False))
    specs.append(("lip", True, True))

rows = []
for catcol_name, weighted, index_free in specs:
    catcol = "cat" if catcol_name == "crsp_obj_cd" else "lip"
    df = ft.dropna(subset=[catcol])
    d = simulate(df, catcol, weighted, index_free)
    if not len(d): continue
    for post, lab in [(False, "all"), (True, "post-2000")]:
        dd = d[d.year >= 2000] if post else d
        for col, mlab, sc in MEASURES:
            if col not in dd.columns: continue
            g = gradient(dd, col, sc)
            if not g: continue
            rows.append({"sleeve": catcol_name,
                         "draw": "TNA-wtd" if weighted else "uniform",
                         "menus": "index-free" if index_free else "all",
                         "sample": lab, "measure": mlab,
                         "gradient_bps": g[0], "lo": g[1], "hi": g[2],
                         "n_years": g[3], "excl_zero": (g[1] < 0) == (g[2] < 0)})
res = pd.DataFrame(rows)
res.round(1).to_csv(os.path.join(OUT, "s54_gradient_robustness.csv"), index=False)

for sample in ("post-2000", "all"):
    print("\n" + "=" * 92); print(f"K=20 MINUS K=3 GRADIENT -- {sample}"); print("=" * 92)
    t = res[res["sample"] == sample]
    if len(t):
        print(t.drop(columns=["sample"]).round(1).to_string(index=False))

print("""
PLAIN READING
  Read down the 'menus' column first. If the gradient holds on INDEX-FREE
  menus, then it is not "cheapest = index fund" -- it is selection among
  active funds, which is the claim.
  Then the 'draw' column: if uniform draws give the same answer, the
  TNA-weighting assumption is not carrying the result.
  Then 'sleeve': if Lipper class reproduces it, the finding does not depend
  on the CRSP objective taxonomy.
  A gradient that survives all three is ready to write up. One that only
  appears TNA-weighted, on index-containing menus, under one taxonomy, is
  not a finding -- it is a specification.
""")
