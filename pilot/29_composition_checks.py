"""Stage 29: THE LAST OPEN COMPOSITION TEST (referee round 4, MAJOR-4).

Round 4 accepted the threshold-free collapse of within-spell Active Share
declines but named one remaining compositional channel: the retail share of
assets fell from 91 to 58 percent over the sample, and institutional
mandates police tracking error differently. This stage runs the named test:

 (a) the share of spells with a within-spell AS decline of 10+ points, by
     era WITHIN entry-AS bands (70-80 / 80-90 / 90+), so rising entry AS
     cannot drive the collapse inside any band;
 (b) the same collapse split RETAIL vs INSTITUTIONAL funds (a fund is
     retail if any share class carries CRSP's retail flag), so the
     institutional-mix shift cannot drive it either - unless the collapse
     lives only in the institutional half, in which case the paper says so.

Output: output/referee_29_composition.txt (aggregates only).
"""
import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

log = ["COMPOSITION CHECKS (stage 29)", "=" * 60]

panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
PF = {w: g.set_index("quarter") for w, g in panel.groupby("wficn")}

# entry AS + within-spell min AS (same walk as stage 28)
as0, as_min_sp = [], []
for _, s in sp.iterrows():
    g = PF.get(s["wficn"])
    if g is None or s["start_p"] not in g.index:
        as0.append(np.nan); as_min_sp.append(np.nan)
        continue
    idx = g.index
    p0 = idx.get_loc(s["start_p"])
    pend = min(p0 + int(s["end_dur"]), len(idx) - 1)
    vals = g["as_min"].iloc[p0:pend + 1]
    as0.append(float(vals.iloc[0]))
    as_min_sp.append(float(vals.min()))
sp["as0"], sp["as_min_sp"] = as0, as_min_sp
sp["das_sp"] = sp["as_min_sp"] - sp["as0"]
sp["big_slide"] = sp["das_sp"] < -0.10
sp["era3"] = pd.cut(sp["start_p"].dt.year, [0, 1994, 2009, 9999],
                    labels=["1980-94", "1995-2009", "2010-23"])

# fund-level retail flag (any share class retail)
rfl = pd.read_parquet(P.CACHE / "retail_flags.parquet")
m1 = PL.get_mflink1()
ret_w = (rfl.merge(m1, on="crsp_fundno", how="inner")
            .groupby("wficn")["is_retail"].any())
sp["retail"] = sp["wficn"].map(ret_w)

def sect_bands():
    log.append("  share of spells with a 10+ point within-spell AS decline "
               "(threshold-free), by era WITHIN entry-AS bands:")
    for lo, hi, lab in [(0.70, 0.80, "70-80"), (0.80, 0.90, "80-90"),
                        (0.90, 1.01, "90+  ")]:
        row = [lab]
        for era in ["1980-94", "1995-2009", "2010-23"]:
            d = sp[(sp["as0"] >= lo) & (sp["as0"] < hi)
                   & (sp["era3"] == era)]["big_slide"]
            row.append(f"{d.mean():6.1%} (n {len(d):5,})" if len(d) else "  -")
        log.append("    band " + " | ".join(row))
    log.append("  reading: the collapse must appear inside every band for "
               "the composition-in-entry-AS story to stay dead.")

def sect_retail():
    log.append("  same statistic, retail vs institutional funds:")
    for flag, lab in [(True, "retail       "), (False, "institutional")]:
        row = [lab]
        for era in ["1980-94", "1995-2009", "2010-23"]:
            d = sp[(sp["retail"] == flag)
                   & (sp["era3"] == era)]["big_slide"]
            row.append(f"{d.mean():6.1%} (n {len(d):5,})" if len(d) else "  -")
        log.append("    " + " | ".join(row))
    n_uncl = int(sp["retail"].isna().sum())
    log.append(f"  (spells with no retail classification: {n_uncl:,})")
    log.append("  reading: if the collapse appears in BOTH halves, the "
               "retail-to-institutional mix shift cannot explain it and the "
               "round-4 channel closes. If it lives in one half only, the "
               "paper reports that honestly and interprets accordingly.")

R.section(log, "(a) BIG SLIDES BY ERA WITHIN ENTRY-AS BANDS", sect_bands)
R.section(log, "(b) BIG SLIDES BY ERA, RETAIL vs INSTITUTIONAL", sect_retail)

log.append("\nSTAGE 29 DONE - aggregates only. This closes the final open "
           "referee channel that existing data can close.")
P.write_report("referee_29_composition.txt", log)
print("\n".join(log))
