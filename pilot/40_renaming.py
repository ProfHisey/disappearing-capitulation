"""Stage 40: FUND RENAMING - detection + fashion timing (ranked R13 v1).

Capitulation by rebranding, first pass: from CRSP Fund Summary name
histories, when do funds change names, and toward what? v1 is LLM-free:
 (a) rename rate per year (share-class suffix noise stripped; top pairs
     printed for eyeballing, lesson of stage 34);
 (b) adoption curves for fashion keywords (tech/dot-com, ESG/sustainable,
     AI/innovation/crypto) - both in NEW names adopted and in the living
     population of fund names;
 (c) de-branding too: keywords DROPPED (post-2022 ESG retreat?).
Flow/stress response needs the panel - that's stage 40b once a slot frees.

Streams Fund Summary only (no panel build) - safe alongside 37c.
Aggregates only; report: output/referee_40_renaming.txt
"""
import re
from pathlib import Path

import pandas as pd

import pilot_lib as P

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["FUND RENAMING (stage 40)", "=" * 60]

# ---- name histories -----------------------------------------------------
fs_path = SRC / "crsp_mf" / "Fund Summary.csv"
parts = []
for ch in pd.read_csv(fs_path, chunksize=2_000_000, low_memory=False,
                      encoding="latin-1"):
    ch.columns = [c.lower() for c in ch.columns]
    keep = [c for c in ("crsp_fundno", "caldt", "fund_name")
            if c in ch.columns]
    if "fund_name" not in keep:
        raise SystemExit(f"fund_name not found; saw "
                         f"{list(ch.columns)[:20]}")
    ch = ch[keep].dropna(subset=["fund_name"])
    parts.append(ch)
nm = pd.concat(parts, ignore_index=True)
nm["caldt"] = pd.to_datetime(nm["caldt"], errors="coerce")
nm["year"] = nm["caldt"].dt.year

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
         & nm["base"].ne("") & nm["prev"].ne("")]
n_fund_yr = nm.groupby("year")["crsp_fundno"].nunique()
log.append(f"name observations: {len(nm):,} rows, "
           f"{nm['crsp_fundno'].nunique():,} share classes; detected "
           f"base-name changes: {len(chg):,}")
log.append("rename rate (changes / share classes observed), by 5yr:")
chg_yr = chg.groupby("year").size()
for y0 in range(1965, 2026, 5):
    ys = range(y0, y0 + 5)
    c = sum(int(chg_yr.get(y, 0)) for y in ys)
    n = sum(int(n_fund_yr.get(y, 0)) for y in ys)
    if n:
        log.append(f"    {y0}-{y0 + 4}: {c / n:6.1%}  ({c:,} changes)")
log.append("top changed pairs (eyeball for normalization noise):")
top = chg.groupby(["prev", "base"]).size().nlargest(10)
for (a, b), n in top.items():
    log.append(f"    {n:4d}  {a[:36]} -> {b[:36]}")

# ---- fashion keywords ---------------------------------------------------
FASH = {
    "tech/dot-com": r"\b(INTERNET|TECHNOLOGY|TELECOM|NET|E-COMMERCE|"
                    r"INFORMATION)\b",
    "ESG":          r"\b(ESG|SUSTAINAB\w*|CLIMATE|IMPACT|RESPONSIBLE|"
                    r"GREEN|SOCIAL(LY)?|CLEAN)\b",
    "AI/crypto":    r"\b(AI|ARTIFICIAL INTELLIGENCE|INNOVAT\w*|"
                    r"DISRUPT\w*|BLOCKCHAIN|CRYPTO|DIGITAL ASSETS?|"
                    r"MACHINE LEARNING)\b",
    "index/passive": r"\b(INDEX|PASSIVE)\b",
}
log.append("\nkeyword ADOPTIONS via rename (keyword absent before, "
           "present after), by 5yr:")
for lab, pat in FASH.items():
    rx = re.compile(pat)
    adopt = chg[~chg["prev"].str.contains(rx) & chg["base"]
                .str.contains(rx)]
    drop = chg[chg["prev"].str.contains(rx) & ~chg["base"]
               .str.contains(rx)]
    ay = adopt.groupby("year").size()
    dy = drop.groupby("year").size()
    line = []
    for y0 in range(1990, 2026, 5):
        a = sum(int(ay.get(y, 0)) for y in range(y0, y0 + 5))
        d = sum(int(dy.get(y, 0)) for y in range(y0, y0 + 5))
        line.append(f"{y0}s:+{a}/-{d}")
    log.append(f"  {lab:14s}: " + "  ".join(line)
               + f"   (total +{len(adopt)}/-{len(drop)})")
log.append("  (+adopted / -dropped; watch tech peaking ~1995-2004, ESG "
           "~2015-2021 with drops after 2022, AI post-2023, and "
           "active->'index' conversions as literal surrender-by-renaming)")

log.append("\nshare of LIVING fund names containing each keyword, "
           "selected years:")
for y in (1995, 2000, 2008, 2015, 2021, 2025):
    d = nm[nm["year"] == y].drop_duplicates("crsp_fundno")
    if not len(d):
        continue
    row = [f"{y} (n {len(d):,})"]
    for lab, pat in FASH.items():
        row.append(f"{lab} {d['base'].str.contains(pat, regex=True).mean():.1%}")
    log.append("    " + " | ".join(row))

log.append("\nSTAGE 40 DONE - aggregates only. 40b (panel join: are "
           "renamers post-stress funds? do adoptions attract flows?) "
           "runs when a panel slot frees; LLM trendiness grading replaces "
           "keywords at scale-up.")
P.write_report("referee_40_renaming.txt", log)
print("\n".join(log))
