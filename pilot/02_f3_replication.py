"""Stage 2: replicate Petajisto (2013) Figure-3-style closet-indexing series.

Share of funds (and of TNA) with Active Share below 60%, quarterly, 1980-2023:
Petajisto official-benchmark AS for 1980-2009, ND min-AS for 1979-2023 (shown as
two lines - they use different benchmark conventions, so levels differ; the
SHAPE (rise around 2000 and 2007-09) is the replication target).

Outputs: output/f3_closet_share.csv + output/f3_closet_share.png (aggregates).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import pilot_lib as P

as_panel = pd.read_parquet(P.CACHE / "as_panel.parquet")
pet = pd.read_parquet(P.CACHE / "pet_panel.parquet")

# ND min-AS series (all years)
g = as_panel.assign(closet=lambda d: d["as_min"] < P.CLOSET_CUTOFF).groupby("month")
nd_series = g.agg(share_funds=("closet", "mean"), n=("closet", "size"))
if "total_assets" in as_panel.columns:
    w = (as_panel.assign(cta=lambda d: d["total_assets"].where(d["as_min"] < P.CLOSET_CUTOFF, 0))
                 .groupby("month")[["cta", "total_assets"]].sum())
    nd_series["share_tna"] = w["cta"] / w["total_assets"]

# Petajisto official-benchmark series (1980-2009), excluding index funds
pq = pet.copy()
pq["month"] = pd.PeriodIndex(pq["quarter"], freq="Q").to_timestamp("Q")
if "indexfund" in pq.columns:
    pq = pq[pd.to_numeric(pq["indexfund"], errors="coerce").fillna(0) == 0]
pg = pq.dropna(subset=["activeshare"]) \
       .assign(closet=lambda d: d["activeshare"] < P.CLOSET_CUTOFF).groupby("month")
pet_series = pg.agg(share_funds_official=("closet", "mean"), n_official=("closet", "size"))

outdf = nd_series.join(pet_series, how="outer")
outdf.to_csv(P.OUT / "f3_closet_share.csv")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(nd_series.index, nd_series["share_funds"], lw=1.6,
        label="ND data, min-AS benchmark (1979-2023)")
if "share_tna" in nd_series.columns:
    ax.plot(nd_series.index, nd_series["share_tna"], lw=1.2, ls="--",
            label="ND, TNA-weighted")
ax.plot(pet_series.index, pet_series["share_funds_official"], lw=1.6,
        label="Petajisto data, official benchmark (1980-2009)")
try:  # recession shading if Buy Risk FRED file is reachable
    rec = pd.read_csv(P.F_USREC, parse_dates=[0])
    rec.columns = ["date", "usrec"]
    on = rec[rec["usrec"] == 1]["date"]
    for _, grp in on.groupby((on.diff().dt.days > 45).cumsum()):
        ax.axvspan(grp.min(), grp.max(), color="0.85", zorder=0)
except Exception:  # noqa: BLE001
    pass
ax.set_ylabel(f"Share of funds with Active Share < {P.CLOSET_CUTOFF:.0%}")
ax.set_title("Closet indexing over time (Petajisto 2013 F3 replication, pilot)")
ax.legend(frameon=False, fontsize=8)
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(P.OUT / "f3_closet_share.png", dpi=200)
print(f"wrote {P.OUT / 'f3_closet_share.csv'} and .png")
print(outdf.tail(8).round(3))
