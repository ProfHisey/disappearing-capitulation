"""Stage 32: DOWNLOAD SEC N-PORT STRUCTURED DATA SETS (free, public domain).

Fetches every quarterly Form N-PORT data set ZIP from sec.gov (2019q4 through
the present; ~230-420 MB each, ~9+ GB total). Resumable: already-downloaded
files are skipped, so rerunning after an interruption just picks up where it
left off. Future quarters that 404 are reported and skipped.

STORAGE NOTE: we KEEP the ZIPs as-is (no bulk extraction). Fully extracted
they would be 100+ GB of TSV; the stage-33 parser reads what it needs
directly from each archive instead.

BEFORE RUNNING: fill in CONTACT below - the SEC's fair-access policy asks
automated clients to identify themselves via the User-Agent header.

Output: files in E:\\Finance\\data\\sources\\nport\\ + a manifest report in
output/nport_download_log.txt
"""
import time
from pathlib import Path

import requests

# --- fill this in: "Your Name your.email@domain.edu" --------------------
CONTACT = "Colin Hisey, Northwestern University, colin.hisey@northwestern.edu"
# ------------------------------------------------------------------------

DEST = Path(r"E:\Finance\data\sources\nport")
DEST.mkdir(parents=True, exist_ok=True)
OUT = Path("output")
OUT.mkdir(exist_ok=True)

BASE = "https://www.sec.gov/files/dera/data/form-n-port-data-sets"
FIRST = (2019, 4)
LAST = (2026, 4)   # generous end; missing future quarters 404 harmlessly

assert "FILL ME IN" not in CONTACT, (
    "Edit the CONTACT line first - the SEC asks automated downloads to "
    "identify themselves (name + email in the User-Agent header).")

HEADERS = {"User-Agent": CONTACT,
           "Accept-Encoding": "gzip, deflate",
           "Host": "www.sec.gov"}

quarters = [(y, q) for y in range(FIRST[0], LAST[0] + 1) for q in (1, 2, 3, 4)
            if (y, q) >= FIRST and (y, q) <= LAST]

log = ["N-PORT DOWNLOAD LOG (stage 32)", "=" * 60]
ok = skipped = missing = failed = 0

for y, q in quarters:
    name = f"{y}q{q}_nport.zip"
    dest = DEST / name
    url = f"{BASE}/{name}"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.append(f"  SKIP (already on disk, "
                   f"{dest.stat().st_size / 1e6:.0f} MB): {name}")
        skipped += 1
        continue
    for attempt in (1, 2, 3):
        try:
            print(f"downloading {name} (attempt {attempt}) ...")
            with requests.get(url, headers=HEADERS, stream=True,
                              timeout=120) as r:
                if r.status_code == 404:
                    log.append(f"  404 (not published yet): {name}")
                    missing += 1
                    break
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
                log.append(f"  OK ({dest.stat().st_size / 1e6:.0f} MB): "
                           f"{name}")
                ok += 1
            break
        except requests.RequestException as e:
            print(f"    error: {e}")
            if attempt == 3:
                log.append(f"  FAILED after 3 attempts: {name} ({e})")
                failed += 1
            else:
                time.sleep(10 * attempt)
    time.sleep(1.5)  # stay well under the SEC's 10 requests/sec cap

log.append(f"\nSUMMARY: {ok} downloaded, {skipped} already on disk, "
           f"{missing} not yet published, {failed} failed")
total = sum(f.stat().st_size for f in DEST.glob("*.zip"))
log.append(f"library now holds {len(list(DEST.glob('*.zip')))} zips, "
           f"{total / 1e9:.2f} GB")
log.append("\nSTAGE 32 DONE. Keep the zips; stage 33 parses directly from "
           "them. Rerun any quarter's 'FAILED' by just rerunning the script.")
(OUT / "nport_download_log.txt").write_text("\n".join(log), encoding="utf-8")
print("\n".join(log))
