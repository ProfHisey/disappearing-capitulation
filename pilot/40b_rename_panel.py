"""Stage 40b: RENAMING x STRESS x FLOWS (panel join for stage 40).

Are renamers post-stress funds, and does renaming work? Joins base-name
changes (Fund Summary, flip-flop pairs excluded) to the capitulation
panel via mflink1:
 (a) trailing rel4q in the year BEFORE a rename vs population;
 (b) P(rename within +-4q of a capitulation crossing) vs baseline;
 (c) within-fund net flow, 4q after vs 4q before the rename.

Aggregates only; report: output/referee_40b_rename_panel.txt
Builds the panel - run alone.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["RENAMING x STRESS x FLOWS (stage 40b)", "=" * 60]

# ---- rebuild rename events (as stage 40, + flip-flop filter) ------------
fs_path = SRC / "crsp_mf" / "Fund Summary.csv"
parts = []
for ch in pd.read_csv(fs_path, chunksize=2_000_000, low_memory=False,
                      encoding="latin-1"):
    ch.columns = [c.lower() for c in ch.columns]
    ch = ch[["crsp_fundno", "caldt", "fund_name"]].dropna(
        subset=["fund_name"])
    parts.append(ch)
nm = pd.concat(parts, ignore_index=True)
nm["caldt"] = pd.to_datetime(nm["caldt"], errors="coerce")

SUFFIX = re.compile(
    r"[;/].*$|\b(CL(ASS)?\s+[A-Z0-9]{1,4}|INST(ITUTIONAL)?|INV(ESTOR)?|"
    r"ADM(IRAL|IN)?|RET(AIL|IREMENT)?|ADV(ISOR)?|[A-Z]\s*SHARES?|"
    r"SHARES?)\b\.?$")
def base_name(s):
    s = str(s).upper().strip()
    prev = None
    while prev != s:
        prev = s
        s = SUFFIX.sub("", s).strip(" -,;/")
    return re.sub(r"\s+", " ", s)

nm["base"] = nm["fund_name"].map(base_name)
nm = nm.sort_values(["crsp_fundno", "caldt"])
nm["prev"] = nm.groupby("crsp_fundno")["base"].shift()
chg = nm[nm["prev"].notna() & (nm["base"] != nm["prev"])
         & nm["base"].ne("") & nm["prev"].ne("")].copy()
pairs = set(map(tuple, chg[["prev", "base"]].drop_duplicates()
                .itertuples(index=False)))
chg["flipflop"] = [((b, a) in pairs) for a, b in
                   zip(chg["prev"], chg["base"])]
log.append(f"rename events: {len(chg):,}; flip-flop pairs excluded: "
           f"{int(chg['flipflop'].sum()):,}")
chg = chg[~chg["flipflop"]]

m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                 encoding="latin-1")
m1.columns = [c.lower() for c in m1.columns]
chg = chg.merge(m1[["crsp_fundno", "wficn"]].drop_duplicates(),
                on="crsp_fundno", how="inner")
chg["quarter"] = chg["caldt"].dt.to_period("Q")
# fund-level: one rename event per wficn-quarter
ev = (chg.drop_duplicates(["wficn", "quarter"])
      [["wficn", "quarter"]].copy())
ev["wficn"] = ev["wficn"].astype("int64")
log.append(f"fund-level rename events (wficn-quarters): {len(ev):,}")

# ---- panel + spells -----------------------------------------------------
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
panel = panel.copy()
panel["wficn"] = panel["wficn"].astype("int64")
univ = set(panel["wficn"].unique())
ev = ev[ev["wficn"].isin(univ)]
log.append(f"rename events inside capitulation universe: {len(ev):,}")

pq = (panel[["wficn", "quarter", "rel4q", "flowq"]]
      .drop_duplicates(["wficn", "quarter"]))
ev = ev.merge(pq, on=["wficn", "quarter"], how="left")

# ---- (a) are renamers stressed? -----------------------------------------
pop = panel["rel4q"].dropna()
evr = ev["rel4q"].dropna()
log.append(f"trailing rel4q at rename: median {evr.median():+.2%} "
           f"(n {len(evr):,}) vs population median {pop.median():+.2%}")
log.append(f"  share with rel4q<0 at rename: {(evr < 0).mean():.1%} vs "
           f"population {(pop < 0).mean():.1%}")

# ---- (b) rename around capitulation -------------------------------------
caps = sp[sp["capitulated"] == True].copy()
caps["cq"] = pd.PeriodIndex(caps["m_cal_q"], freq="Q")
ev_set = set(map(tuple, ev[["wficn", "quarter"]].itertuples(index=False)))
hit = sum(any((int(w), c + k) in ev_set for k in range(-4, 5))
          for w, c in zip(caps["wficn"], caps["cq"]))
# baseline: probability a random 9q window of a universe fund has a rename
n_fq = len(pq)
p_q = len(ev) / n_fq
log.append(f"capitulations with a rename within ±4q: {hit} of "
           f"{len(caps):,} ({hit / len(caps):.1%}); naive baseline for a "
           f"9q window {1 - (1 - p_q) ** 9:.1%}")

# ---- (c) does renaming attract flows? -----------------------------------
PFq = {w: g.set_index("quarter")["flowq"]
       for w, g in panel.groupby("wficn")}
pre, post = [], []
for w, q in zip(ev["wficn"], ev["quarter"]):
    f = PFq.get(w)
    if f is None:
        continue
    a = [f.get(q - k, np.nan) for k in range(1, 5)]
    b = [f.get(q + k, np.nan) for k in range(1, 5)]
    a = [x for x in a if pd.notna(x)]
    b = [x for x in b if pd.notna(x)]
    if len(a) >= 2 and len(b) >= 2:
        pre.append(float(np.mean(a)))
        post.append(float(np.mean(b)))
pre, post = pd.Series(pre), pd.Series(post)
d = post - pre
log.append(f"within-fund quarterly net flow, 4q after minus 4q before "
           f"rename: median {d.median():+.2%}, mean {d.mean():+.2%} "
           f"(n {len(d):,} events)")
log.append(f"  levels: pre median {pre.median():+.2%}, post median "
           f"{post.median():+.2%}")
log.append("  caveat: renames cluster at acquisitions/mergers - a flow "
           "jump may be reorganization, not marketing. LLM classification "
           "of rename TYPE (rebrand vs strategy shift vs merger) is the "
           "scale-up that separates these.")

log.append("\nSTAGE 40b DONE - aggregates only.")
P.write_report("referee_40b_rename_panel.txt", log)
print("\n".join(log))
