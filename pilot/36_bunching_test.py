"""Stage 36: THRESHOLD BUNCHING TEST (ranked R7 / backlog #12).

If managers manage the METRIC, Active Share should pile up just above the
salient thresholds after they became famous: 60% ("closet indexer" label,
Cremers-Petajisto 2009) and 80% ("truly active" label). A McCrary-style
density check: compare mass just below vs just above each threshold,
before and after 2009. Also checks 70% - OUR spell threshold - where
bunching would bear on the event definition itself.

Unit note: fund-quarter observations are serially correlated, so the
binomial CIs are optimistic - treat as descriptive; the fund-level panel
(each fund's mean AS) is the conservative cut reported alongside.

Aggregates only; report: output/referee_36_bunching.txt
Light - safe alongside anything.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pilot_lib as P
import panel_lib as PL

OUT = Path("output")
OUT.mkdir(exist_ok=True)

log = ["THRESHOLD BUNCHING TEST (stage 36)", "=" * 60]

panel = PL.build_panel(log)
panel = panel[panel["as_min"].notna()].copy()
panel["era"] = np.where(panel["quarter"].dt.year <= 2009,
                        "pre-2009", "post-2009")

def density_check(df, thr, w, label):
    lo = df[(df["as_min"] >= thr - w) & (df["as_min"] < thr)]
    hi = df[(df["as_min"] >= thr) & (df["as_min"] < thr + w)]
    n_lo, n_hi = len(lo), len(hi)
    tot = n_lo + n_hi
    if tot < 50:
        log.append(f"    {label}: too few obs ({tot})")
        return
    share_hi = n_hi / tot
    se = np.sqrt(0.25 / tot)  # null p=0.5
    z = (share_hi - 0.5) / se
    log.append(f"    {label}: below {n_lo:,} vs above {n_hi:,} "
               f"(share above {share_hi:.1%}, z={z:+.1f} vs smooth-null "
               f"50%)")

for thr in (0.60, 0.70, 0.80):
    log.append(f"\nthreshold {thr:.0%} (±3pt window):")
    for era in ("pre-2009", "post-2009"):
        density_check(panel[panel["era"] == era], thr, 0.03,
                      f"fund-quarters, {era}")
    fund_lvl = (panel.groupby(["wficn", "era"])["as_min"]
                .mean().reset_index())
    for era in ("pre-2009", "post-2009"):
        density_check(fund_lvl[fund_lvl["era"] == era], thr, 0.03,
                      f"fund-level means, {era}")

# fine-grained histogram around 60% post-2009, for eyeballing shape
post = panel.loc[panel["era"] == "post-2009", "as_min"]
bins = np.arange(0.50, 0.71, 0.01)
hist, _ = np.histogram(post, bins=bins)
log.append("\npost-2009 fund-quarter counts, 1pt bins 50-70%:")
log.append("  " + " ".join(f"{int(b * 100)}:{n}"
                           for b, n in zip(bins[:-1], hist)))
log.append("\nreading: bunching = a jump in mass at/just above the "
           "threshold relative to just below, appearing POST-2009 only "
           "and stronger at 60/80 (public labels) than at 70 (our "
           "internal threshold). Bunching at 70 too would suggest funds "
           "manage AS generally, and our crossing events partly measure "
           "metric management - flag for the paper if so.")
log.append("\nSTAGE 36 DONE - aggregates only.")
P.write_report("referee_36_bunching.txt", log)
print("\n".join(log))
