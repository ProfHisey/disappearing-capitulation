"""Stage 25b: settle how the cached panel actually stores AS gaps.

Stage 25(b)'s break-at-gap variant returned results IDENTICAL to baseline,
which is only possible if the panel already contains gap quarters as
explicit missing rows (baseline already censors) or if the variant failed.
This prints the facts. No report file; output to screen only.
"""
import pandas as pd

import panel_lib as PL

log = []
panel = PL.build_panel(log)

res = []
for w, g in panel.groupby("wficn"):
    q = g["quarter"]
    span = (q.max() - q.min()).n + 1
    res.append((w, len(g), span, int(g["as_min"].isna().sum())))
df = pd.DataFrame(res, columns=["w", "rows", "span", "nan_as"])
print(f"funds: {len(df):,}")
print(f"share of funds with contiguous quarterly rows (rows == span): "
      f"{(df['rows'] == df['span']).mean():.1%}")
print(f"share of funds with >=1 explicit NaN as_min row: "
      f"{(df['nan_as'] > 0).mean():.1%} (mean {df['nan_as'].mean():.1f} rows)")
print(f"total missing quarters NOT present as rows: "
      f"{(df['span'] - df['rows']).clip(lower=0).sum():,}")
print(f"duplicate wficn-quarter rows: "
      f"{panel.duplicated(['wficn', 'quarter']).sum():,}")

g = panel[panel["wficn"] == 103013].set_index("quarter").sort_index()
print(f"\nSEQUX: {len(g)} rows, span "
      f"{(g.index.max() - g.index.min()).n + 1} quarters")
print("SEQUX as_min 2015Q1-2017Q4 (NaN = explicit missing row; absent "
      "quarters simply won't print):")
print(g.loc["2015Q1":"2017Q4", "as_min"])
