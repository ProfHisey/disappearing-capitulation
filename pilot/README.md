# Capitulation study — feasibility pilot

Scripts live in `E:\Finance\Capitulation\pilot\` and read raw data from the shared
library at `E:\Finance\data\sources` (override with the `DATA_LIB` env var — same
variable Buy Risk uses). Expected library folders: `crsp_mf\`, `activeshare_nd\`,
`petajisto\`. All outputs go to `pilot\output\` and are **aggregates only**
(counts, shares, curves — no fund-level rows), so the output folder is safe to
share back to Claude for interpretation.

## One-time setup (~5 min)

1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html
2. In "Anaconda Prompt": `conda env create -f ..\environment.yml` (or, if the env
   already exists from the pilot era, just `conda activate capit`).

## Run order (in Anaconda Prompt)

```
cd /d E:\Finance\Capitulation\pilot
conda activate capit
python 00_probe.py          # schema check; output\schema_report.txt
python 01_build_panel.py    # AS + returns panels (parquet in pilot\cache\)
python 02_f3_replication.py # closet-indexing share over time (Petajisto F3)
python 03_km_pilot.py       # Kaplan-Meier capitulation curve, 1980-2009
```

## Reference results (2026-08-05 pilot run — the smoke-test baseline)

Petajisto x CRSP match rate 99.8%; 6,496 underperformance spells; 235 capitulation
events (3.6%); median spell 4 quarters; logrank across depth terciles p = 0.0195.
A re-run after any path/data change should reproduce these exactly.

Known pilot approximations (fixed in the real build): benchmark returns proxied by
CPZ core indexes (to 2011); 2010-2019 AS not yet joined to returns (needs MFLINKS);
fund returns use the largest share class; ND series can't screen index funds yet.
