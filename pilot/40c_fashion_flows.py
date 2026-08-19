"""Stage 40c: FASHION-RENAME FLOW RESPONSE (the sharp version of 40b).

40b showed generic renaming is administrative noise: renamers aren't
stressed, renames don't cluster at capitulations, and flows don't
respond. The live hypothesis (Cooper-Gulen-Rau) is about STYLE-CHASING
renames only. This stage reruns the stress and flow tests on the
keyword-ADOPTION subset from stage 40: funds whose new name gains a
tech/ESG/AI-crypto vocabulary word.

Aggregates only; report: output/referee_40c_fashion_flows.txt
Builds the panel - run when a slot frees.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["FASHION-RENAME FLOW RESPONSE (stage 40c)", "=" * 60]

# ---- rename events + keyword adoption flags (as stage 40) ---------------
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
    r"[;/].*$|\b(?:CL(?:ASS)?\s+[A-Z0-9]{1,4}|INST(?:ITUTIONAL)?|"
    r"INV(?:ESTOR)?|ADM(?:IRAL|IN)?|RET(?:AIL|IREMENT)?|ADV(?:ISOR)?|"
    r"[A-Z]\s*SHARES?|SHARES?)\b\.?$")
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

FASH = {
    "tech":  r"\b(?:INTERNET|TECHNOLOGY|TELECOM|NET|E-COMMERCE|"
             r"INFORMATION)\b",
    "ESG":   r"\b(?:ESG|SUSTAINAB\w*|CLIMATE|IMPACT|RESPONSIBLE|GREEN|"
             r"SOCIAL(?:LY)?|CLEAN)\b",
    "AI":    r"\b(?:AI|ARTIFICIAL INTELLIGENCE|INNOVAT\w*|DISRUPT\w*|"
             r"BLOCKCHAIN|CRYPTO|DIGITAL ASSETS?|MACHINE LEARNING)\b",
}
m1 = pd.read_csv(SRC / "mflinks" / "mflink1.csv", low_memory=False,
                 encoding="latin-1")
m1.columns = [c.lower() for c in m1.columns]

panel = PL.build_panel(log)
panel = panel.copy()
panel["wficn"] = panel["wficn"].astype("int64")
PFq = {w: g.set_index("quarter")[["flowq", "rel4q"]]
       for w, g in panel.groupby("wficn")}
pop = panel["rel4q"].dropna()

for lab, pat in FASH.items():
    rx = re.compile(pat)
    ad = chg[~chg["prev"].str.contains(rx) & chg["base"]
             .str.contains(rx)].copy()
    ad = ad.merge(m1[["crsp_fundno", "wficn"]].drop_duplicates(),
                  on="crsp_fundno", how="inner")
    ad["quarter"] = ad["caldt"].dt.to_period("Q")
    ev = ad.drop_duplicates(["wficn", "quarter"])
    ev = ev[ev["wficn"].astype("int64").isin(PFq)]
    log.append(f"\n{lab} adoptions linked to universe: {len(ev):,} "
               f"fund-events")
    if len(ev) < 10:
        log.append("  too few - skipped")
        continue
    stress, pre, post = [], [], []
    for w, q in zip(ev["wficn"].astype("int64"), ev["quarter"]):
        f = PFq.get(w)
        if f is None:
            continue
        r = f["rel4q"].get(q, np.nan)
        if pd.notna(r):
            stress.append(float(r))
        a = [f["flowq"].get(q - k, np.nan) for k in range(1, 5)]
        b = [f["flowq"].get(q + k, np.nan) for k in range(1, 5)]
        a = [x for x in a if pd.notna(x)]
        b = [x for x in b if pd.notna(x)]
        if len(a) >= 2 and len(b) >= 2:
            pre.append(float(np.mean(a)))
            post.append(float(np.mean(b)))
    st = pd.Series(stress)
    log.append(f"  trailing rel4q at adoption: median {st.median():+.2%} "
               f"(n {len(st)}) vs population {pop.median():+.2%}; "
               f"share <0: {(st < 0).mean():.1%} vs "
               f"{(pop < 0).mean():.1%}")
    pre, post = pd.Series(pre), pd.Series(post)
    d = post - pre
    log.append(f"  net flow 4q-after minus 4q-before: median "
               f"{d.median():+.2%}, mean {d.mean():+.2%} (n {len(d)}); "
               f"levels pre {pre.median():+.2%} -> post "
               f"{post.median():+.2%}")

log.append("\nreading: Cooper-Gulen-Rau found style renames attract "
           "flows; if the ESG/tech adoption rows show positive flow "
           "shifts against 40b's negative generic baseline, the fashion "
           "channel is real and the LLM rename-type classifier is worth "
           "building. Small n per fashion - descriptive.")
log.append("\nSTAGE 40c DONE - aggregates only.")
P.write_report("referee_40c_fashion_flows.txt", log)
print("\n".join(log))
