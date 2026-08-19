"""Stage 33: INSPECT N-PORT ARCHIVE STRUCTURE (probe before parse).

Opens the FIRST (2019q4) and LATEST (2026q2) N-PORT structured data set ZIPs
and fingerprints their contents: member tables, compressed/uncompressed
sizes, and every table's column header (read directly from the archive - no
extraction). The stage-33b extractor gets written against THIS output, not
against table names from memory. Also flags any schema drift between the
first and latest quarter.

Output: output/nport_33_inspect.txt (structure only - no filer data).
"""
import io
import zipfile
from pathlib import Path

SRC = Path(r"E:\Finance\data\sources\nport")
OUT = Path("output")
OUT.mkdir(exist_ok=True)

PROBES = ["2019q4_nport.zip", "2026q2_nport.zip"]

log = ["N-PORT ARCHIVE INSPECTION (stage 33)", "=" * 70]
schemas = {}  # member name -> {probe: [cols]}


def header_of(zf, info, n_lines=1):
    """Read just the first line(s) of a zip member without extracting it."""
    with zf.open(info) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        return [text.readline().rstrip("\n") for _ in range(n_lines)]


for name in PROBES:
    path = SRC / name
    log.append(f"\n{'-' * 70}\nARCHIVE: {name}")
    if not path.exists():
        log.append("  NOT PRESENT - skipped")
        continue
    log.append(f"  zip size: {path.stat().st_size / 1e6:.0f} MB")
    with zipfile.ZipFile(path) as zf:
        members = sorted(zf.infolist(), key=lambda i: -i.file_size)
        log.append(f"  members: {len(members)}")
        for info in members:
            log.append(f"\n  {info.filename}  "
                       f"({info.file_size / 1e6:,.1f} MB uncompressed)")
            if info.file_size == 0 or info.filename.endswith("/"):
                continue
            try:
                (head,) = header_of(zf, info)
            except Exception as e:
                log.append(f"    [could not read header: {e}]")
                continue
            sep = "\t" if "\t" in head else ","
            cols = [c.strip().strip('"') for c in head.split(sep)]
            if info.filename.lower().endswith((".tsv", ".csv", ".txt")):
                schemas.setdefault(info.filename, {})[name] = cols
                log.append(f"    delimiter: "
                           f"{'TAB' if sep == chr(9) else 'comma'}   "
                           f"columns: {len(cols)}")
                for i in range(0, len(cols), 8):
                    log.append("      " + ", ".join(cols[i:i + 8]))
            else:
                log.append(f"    [non-tabular member; first 120 chars: "
                           f"{head[:120]}]")

# ---- schema drift between first and latest quarter ----------------------
log.append(f"\n{'=' * 70}\nSCHEMA DRIFT (first vs latest quarter):")
drift = False
for member, by_probe in sorted(schemas.items()):
    if len(by_probe) < 2:
        only = next(iter(by_probe))
        log.append(f"  {member}: present only in {only}")
        drift = True
        continue
    a, b = (by_probe[p] for p in PROBES if p in by_probe)
    gone, new = set(a) - set(b), set(b) - set(a)
    if gone or new:
        drift = True
        log.append(f"  {member}:")
        if gone:
            log.append(f"    dropped since 2019q4: {sorted(gone)}")
        if new:
            log.append(f"    added since 2019q4:   {sorted(new)}")
if not drift:
    log.append("  none - identical table sets and columns in both quarters.")

log.append("\nSTAGE 33 DONE - structure only. Paste this back; the 33b "
           "extractor (monthly gross flows + holdings) gets written "
           "against these exact table and column names.")
(OUT / "nport_33_inspect.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
