"""Stage 31: PROBE ROUND-2 PULLS (Thomson s12 completions, CRSP indexes,
Thomson s34 as files land).

Generic content fingerprint for each new file: schema, row count (streamed
for big files), date span, key identifier cardinality. Rerun as more files
arrive - missing files are reported and skipped, so one script covers the
whole batch. Prints aggregates only; nothing licensed leaves disk.

Output: output/probe_31_new_sources.txt
"""
from pathlib import Path

import pandas as pd

SRC = Path(r"E:\Finance\data\sources")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

TARGETS = [
    SRC / "thomson_s12" / "s12type2.csv",
    SRC / "thomson_s12" / "s12type4.csv",
    SRC / "crsp_indexes" / "monthly_indexes.csv",
    SRC / "crsp_indexes" / "monthly_indexes_levels.csv",
    SRC / "crsp_indexes" / "treasury_inflation.csv",
    SRC / "thomson_s34" / "s34type1.csv",
    SRC / "thomson_s34" / "s34type2.csv",
    SRC / "thomson_s34" / "s34type3.csv",
    SRC / "thomson_s34" / "s34type4.csv",
    SRC / "compustat" / "fundq_fs.csv",  # retired from queue; probed if present
]

# columns whose unique counts are worth reporting, by lowercase name
ID_COLS = ("fundno", "mgrno", "indno", "cusip", "gvkey", "permno", "ticker")
BIG = 500_000_000  # bytes; above this, stream instead of full read

log = ["NEW SOURCE PROBE (stage 31)", "=" * 70]


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            n += chunk.count(b"\n")
    return n - 1


def date_cols(cols):
    return [c for c in cols
            if any(k in c for k in ("date", "caldt", "fdate", "rdate"))
            or c.endswith("dt")]


def probe(path):
    log.append(f"\n{'-' * 70}\nFILE: {path.parent.name}\\{path.name}")
    if not path.exists():
        log.append("  NOT PRESENT YET - skipped")
        return
    log.append(f"  size: {path.stat().st_size / 1e9:.2f} GB")
    head = pd.read_csv(path, nrows=5000, low_memory=False)
    head.columns = [c.lower() for c in head.columns]
    cols = list(head.columns)
    log.append(f"  columns ({len(cols)}): {', '.join(cols[:40])}"
               + (" ..." if len(cols) > 40 else ""))

    dcols = date_cols(cols)
    idc = [c for c in cols if c in ID_COLS]
    small = path.stat().st_size < BIG

    if small:
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        log.append(f"  rows: {len(df):,}")
        for c in dcols[:3]:
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().any():
                log.append(f"  {c} span: {d.min().date()} to {d.max().date()}"
                           f"  (missing {d.isna().mean():.1%})")
        for c in idc:
            log.append(f"  unique {c}: {df[c].nunique():,}")
        # low-cardinality columns are usually format/type flags - show them
        for c in cols:
            if c in dcols or c in idc:
                continue
            nu = df[c].nunique(dropna=True)
            if 1 <= nu <= 8 and not pd.api.types.is_float_dtype(df[c]):
                log.append(f"  {c}: {df[c].value_counts().head(8).to_dict()}")
    else:
        log.append(f"  rows (streamed count): {count_lines(path):,}")
        want = set(dcols[:2]) | set(idc)
        if want:
            mins, maxs, uniq = {}, {}, {c: set() for c in idc}
            for ch in pd.read_csv(path, usecols=lambda c: c.lower() in want,
                                  chunksize=2_000_000, low_memory=False):
                ch.columns = [c.lower() for c in ch.columns]
                for c in dcols[:2]:
                    if c in ch.columns:
                        d = pd.to_datetime(ch[c], errors="coerce")
                        lo, hi = d.min(), d.max()
                        if pd.notna(lo):
                            mins[c] = min(mins.get(c, lo), lo)
                            maxs[c] = max(maxs.get(c, hi), hi)
                for c in idc:
                    if c in ch.columns:
                        uniq[c].update(ch[c].dropna().unique())
            for c in mins:
                log.append(f"  {c} span: {mins[c].date()} to {maxs[c].date()}")
            for c in idc:
                log.append(f"  unique {c}: {len(uniq[c]):,}")


for p in TARGETS:
    probe(p)

log.append("\nSTAGE 31 DONE - rerun after each new file lands; the final "
           "clean run is the manifest of record for the acquisition log.")
(OUT / "probe_31_new_sources.txt").write_text("\n".join(log),
                                              encoding="utf-8")
print("\n".join(log))
