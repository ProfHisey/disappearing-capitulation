"""Stage 22b: regenerate Figure 2 as the three-state version.

Replaces the two-state cif_by_era.png with fig2_cif_threestate.png:
left panel, cumulative incidence of all three endings (recover / die /
capitulate), full sample; right panel, the capitulation incidence alone by
era, which is the disappearance in cumulative form. Runs in a couple of
minutes. Output: output/fig2_cif_threestate.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter

import pilot_lib as P
import panel_lib as PL
import referee_lib as R

plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
C_CAP, C_DIE, C_REC = "#1f4e79", "#8c1c13", "#5a6b5d"

log = []
panel = PL.build_panel(log)
death = PL.get_death(log)
sp = R.attach_death(PL.extract_spells(panel, client_cut=None), death)
sp["etype"] = 0
sp.loc[sp["ended_by"] == "recovered", "etype"] = 3
sp.loc[sp["spell_died"], "etype"] = 2
sp.loc[sp["capitulated"], "etype"] = 1
sp["dur"] = np.where(sp["etype"] == 1, sp["m_dur"], sp["end_dur"])
sp["dur"] = pd.to_numeric(sp["dur"]).clip(lower=1)
sp["era"] = pd.cut(sp["start_p"].dt.year, [0, 1994, 2009, 9999],
                   labels=["1980-94", "1995-2009", "2010-23"])  # audit A7

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

ax = axes[0]
for evt, name, col in ((3, "Recover", C_REC), (2, "Die", C_DIE),
                       (1, "Capitulate", C_CAP)):
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(sp["dur"], sp["etype"], event_of_interest=evt)
    c = aj.cumulative_density_
    ax.step(c.index, c.iloc[:, 0], where="post", color=col, lw=1.6,
            label=name)
ax.set_xlim(0, 40)
ax.set_xlabel("Quarters since spell entry")
ax.set_ylabel("Cumulative incidence")
ax.set_title("All spells, three competing endings", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8)

ax = axes[1]
shades = {"1980-94": "0.15", "1995-2009": "0.45", "2010-23": C_CAP}
for era, g in sp.groupby("era", observed=True):
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(g["dur"], g["etype"], event_of_interest=1)
    c = aj.cumulative_density_
    ax.step(c.index, c.iloc[:, 0], where="post", lw=1.6,
            color=shades[str(era)], label=f"{era} (n={len(g):,})")
ax.set_xlim(0, 40)
ax.set_xlabel("Quarters since spell entry")
ax.set_title("Capitulation only, by entry era", fontsize=9, loc="left")
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig(P.OUT / "fig2_cif_threestate.png", dpi=300)
print("wrote output/fig2_cif_threestate.png")
