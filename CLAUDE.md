# CLAUDE.md

Guidance for working in the **Capitulation** study repo.

## What this project is

Empirical finance study toward publication (FAJ/JPM register): survival/hazard
analysis of **capitulation under sustained underperformance**, compared within
the same mutual funds between the **manager** (Active Share collapsing into
closet indexing, < 60% threshold from a genuinely-active ≥ 70% start) and the
**clients** (retail share-class redemptions, active→passive exit). Headline
question: *who breaks first — the manager or the clients?*

The full research design lives in the claude.ai Project **"Active share/closet
indexing/capitulation study"** (brief, novelty scan, verified citations, WRDS
pull specs, pilot results, data gap analysis). Read those before redesigning
anything; record durable decisions there, code here.

## Data licensing firewall (non-negotiable)

- Raw licensed data (CRSP via WRDS, and terms-bearing files: AQR, ICI, ND Active
  Share, Petajisto, JST, SPIVA) **never leaves this machine** — never uploaded
  to claude.ai, never committed to git, never pasted into chats.
- Only **derived, aggregated outputs** (counts, shares, coefficients, curves,
  figures — never fund-level rows) may be shared or committed.
- Every WRDS pull gets a committed **manifest** (query, date, row counts, span);
  raw files are git-ignored. Pattern copied from the Buy Risk repo.
- CRSP citation: "Calculated from CRSP data, © Center for Research in Security
  Prices, LLC, via WRDS." (CRSP acquired by Morningstar 2026 — watch renames.)

## Data locations

- Shared raw library: `E:\Finance\data\sources` (env override `DATA_LIB` —
  same variable the Buy Risk repo uses). Providers: `crsp_mf`, `crsp_stock`,
  `activeshare_nd`, `petajisto`, `aqr`, `jst`, `ici`, `spiva`.
- The library is shared with `E:\Finance\BuyRisk` (Astro site). One vintage per
  source; refreshing raw data means updating its manifest and re-running BOTH
  projects' reducers. Never refresh silently.
- `pilot\cache\` holds rebuildable parquet intermediates (git-ignored).

## Stack & conventions

- Python (conda env `capit`): pandas, numpy, pyarrow, matplotlib, lifelines
  (KM/Cox), later linearmodels (panel FE) and ruptures (change-points).
- GraphPad Prism renders the publication survival figures — export tidy CSVs
  (see `pilot\output\km_survival_table.csv` for the shape).
- The 52.7 GB `crsp_mf\Portfolio Holdings.csv` must never be loaded whole:
  convert to year-partitioned parquet first.
- Colin prefers intuition-first explanations of econometrics, one step at a
  time. Verify citations against primary sources before they enter any draft.

## Status (2026-08-05)

Pilot returned **GO**: 99.8% Petajisto×CRSP match; 6,496 spells; 235 capitulation
events; depth-tercile logrank p=0.0195; Petajisto F3 replicated and extended to
2023. Next: MFLINKS pull (WRDS) to join 2010–2019 Active Share; Morningstar
Direct trial exports (benchmarks, manager history, rating history) → library
`morningstar\`; then the real panel build in `src\`.
