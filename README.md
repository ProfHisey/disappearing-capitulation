# Capitulation — "Who breaks first?"

Empirical finance study: the hazard of **capitulation under sustained
underperformance**, compared within the same funds between the **manager**
(Active Share collapse into closet indexing) and the **clients** (redemption /
active→passive exit). Target venues: FAJ / JPM / Journal of Investing.

Study design, literature scan, verified citations, and data specs live in the
claude.ai Project *"Active share/closet indexing/capitulation study"* — this repo
is the code and pipeline.

## Layout

```
pilot\           feasibility pilot (GO verdict 2026-08-05; see pilot\README.md)
src\             real pipeline (panel construction, hazard models) — to come
notebooks\       exploration
output\          aggregate results, figures, Prism export tables
environment.yml  conda env (capit)
```

## Data

Raw sources live in the shared cross-project library `E:\Finance\data\sources`
(override: `DATA_LIB` env var), shared with the Buy Risk project. This repo
commits **no raw data** — see CLAUDE.md for the licensing firewall. Key library
folders: `crsp_mf` (WRDS CRSP Mutual Fund pulls), `activeshare_nd` (Notre Dame
Active Share), `petajisto` (Petajisto Active Share + CPZ factors), `crsp_stock`,
`aqr`, `jst`, `ici`, `spiva`.

## Setup

```
conda env create -f environment.yml
conda activate capit
```
