"""Which reports in output/ are older than the script that made them?

A stale report is the quiet failure mode of a pipeline edited across sessions
and machines: the script was fixed, the report was not regenerated, and the
number in the draft came from the old one. Run this whenever you are unsure
what a report reflects.

  python check_freshness.py
"""
import hashlib
import re
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"


def stamp(p):
    return datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


scripts = sorted(
    (p for p in HERE.glob("*.py") if re.match(r"^\d+[a-z]?_", p.name)),
    key=lambda p: (int(re.match(r"^(\d+)", p.name).group(1)), p.name))

print(f"{'script':<40s} {'modified':<17s} {'report':<17s} status")
print("-" * 92)
stale = miss = 0
for s in scripts:
    tag = re.match(r"^(\d+[a-z]?)_", s.name).group(1)
    reps = sorted(OUT.glob(f"referee_{tag}_*.txt")) if OUT.exists() else []
    sha = hashlib.sha256(s.read_bytes()).hexdigest()[:16]
    if not reps:
        print(f"{s.name:<40s} {stamp(s):<17s} {'-':<17s} NO REPORT   {sha}")
        miss += 1
        continue
    r = max(reps, key=lambda p: p.stat().st_mtime)
    old = r.stat().st_mtime < s.stat().st_mtime
    if old:
        stale += 1
    print(f"{s.name:<40s} {stamp(s):<17s} {stamp(r):<17s} "
          f"{'*** STALE ***' if old else 'ok':<13s} {sha}")

print("-" * 92)
print(f"{len(scripts)} scripts, {stale} stale, {miss} never run")
if stale:
    print("\nSTALE means the script was edited after its report was written. "
          "Any number taken from that report is from the older code.")
print("\nsha256 (first 16) of the round-6 scripts as delivered:")
for n, h in (("60_recovery_aj_deterministic.py", "388eb5a3f119eb22"),
             ("60b_section8_number.py", "0e7217f522e3918d"),
             ("61_fee_cuts_rebuilt.py", "81e97a809d0fae51"),
             ("61b_fee_size_confound.py", "9c3a2cb48f4d04c6")):
    p = HERE / n
    if not p.exists():
        print(f"  {n:<40s} MISSING")
        continue
    got = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    print(f"  {n:<40s} {'match' if got == h else 'DIFFERS: ' + got}")
print("  (a mismatch on a file you did not edit usually means CRLF line "
      "endings from a browser download, which is harmless)")
