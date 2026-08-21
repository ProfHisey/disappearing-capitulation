# Paper 2 — adversarial code audit, 2026-08-21

Scope: stages 35, 35b, 35c, 35d, 37, 37b, 37c, 37d, 38 — the entire evidence
base for Paper 2. Independent agent, read-and-reason (no data), with numerical
checks executed where possible: lifelines 0.30.3 source read for the jitter
contract, the AJ estimator reimplemented to measure run-to-run variability,
hazard models calibrated to the reported inputs, pandas 3.0.2 semantics tested
for the NaN paths.

## VERDICT: Paper 2 is not ready. Three of four findings have live defects.

---

## FATAL

**FA-1. Ghosting has no inference of any kind.** The headline is
`d["rr_sec"].median()` and `d["rr_tna"].median()` for two subsets of pooled
fund-months. No SE, no CI, no test, no clustering, no MDE. `scipy` and
`statsmodels` are not imported in 35/35b/35c/35d/38. `n` is fund-months, which
overstates information by the design effect.

Worse: the comparison is BETWEEN funds, not within. Clientele (institutional
vs retail mix, platform, channel) is the largest determinant of gross
redemption rates. Simulated: a true stress effect of exactly zero plus a
0.2 log-point clientele difference prints a +0.17pp gap — 17x the observed
0.01pp. Composition can equally cancel a real effect.

**But the null is probably well powered.** Simulated fund-clustered bootstrap
SE of the median gap: 0.013-0.037pp → MDE at 80% power ≈ 0.04-0.10pp, i.e.
4-10% of the 1.00%/month base. Two additions make Finding 1 defensible:
(a) fund-clustered bootstrap of the median gap with a stated MDE;
(b) a within-fund estimate — `rr` on a stressed dummy with wficn and month
fixed effects, SEs clustered by fund. That converts "stressed funds look like
unstressed funds" into "the same fund is not redeemed harder when it goes
underwater," which is the sentence the paper wants.

**FA-2. Stage 37d silently changed the SAMPLE as well as the estimator.**
`37d:46` has `caps = caps[caps["cq"].dt.year >= 1995]`. 37b and 37c have no
equivalent, so every pre-1995 capitulation sits in their wave arm (this is
round-4's MINOR-3, fixed in 37d by deletion). The M6 write-up compares 37d
against 37b and attributes the entire KM→AJ move to competing risks. Two
things changed; one is disclosed.

Calibrated to the reported inputs, the estimator change alone implies:

| arm | KM(16q) | AJ(16q) implied | AJ(16q) reported | residual |
|---|---|---|---|---|
| wave 1995-2009 | 30.2% | 26.5% | 24.5% | **−2.0 pp** |
| modern 2010-23 | 20.8% | 18.1% | 18.2% | −0.1 pp |

Modern reproduces to a tenth of a point; the wave — the arm whose sample
changed — is 2pp off. A sweep of 9 hazard-shape combinations found none that
fits both arms. Same pattern at bar 0.80 (implied wave 9.6 vs reported 7.9).

**Confirmation is one line:** diff the per-era event count between
`referee_37b_recovery_km.txt` and `referee_37d_recovery_aj.txt` for 1995-2009.

Consequence: the F9 rewrite may be right but its stated REASON is not what the
code did, and restoring pre-1995 events would shrink the 0.80 inversion from
2.6pp to roughly 1.1pp.

**FA-3. 37d reports no intervals; the 0.80 inversion is noise-consistent.**
`AalenJohansenFitter(calculate_variance=False)` at `:85, 91`. Simulated
sampling distribution of the 2.6pp era gap:

| events/arm (modern/wave) | SE of gap | P(gap ≤ 0) |
|---|---|---|
| 150/250 | 3.34pp | 21.5% |
| 300/500 | 2.37pp | 12.5% |
| 500/800 | 1.86pp | 8.8% |

Even at generous counts the gap is ~1.4 SE from zero (two-sided p ≈ 0.17).

**FA-4. Stage 38's fee-cut detector is mechanically increasing in survival.**
`38:104-108` takes `min` over whatever future quarters exist — 8 chances for a
survivor, 2 for a dying fund — and `if not fut: continue` drops deaths
entirely. Stressed funds die more, so their measured P(cut) is biased DOWN,
which is the direction that manufactures the null. Finding 4's sign is in play.

---

## MAJOR (abbreviated; full list with the agent)

- **MA-1.** 37d jitter is unseeded despite the replication note saying a seed
  "must be passed." lifelines jitters every non-censored duration by
  ±1e-4 and reseeds from OS entropy. Measured: CIF(16q) sd 0.18-0.30pp,
  range up to 1.46pp over 200 runs, with a systematic −0.2 to −0.5pp bias
  (half the boundary events get pushed past the horizon). **Better than a
  seed: the data are discrete quarters — compute AJ directly from the event
  table, five lines, deterministic, no jitter, no bias.**
- **MA-2.** 37d extends death cases past the last Active Share observation.
  The docstring claims this is era-symmetric; it is not — the ND panel ends
  2023Q3 for everyone while deaths run to 2026, so modern funds are far more
  likely to have a death recorded after the AS wall. Works AGAINST the 0.80
  inversion (conservative, say so) but IN FAVOUR of the "one third die in
  both eras" fact headed for Paper 2.
- **MA-3.** `35d:92` uses `groupby("wficn")["tna"].shift(1)` — a ROW lag, not
  a calendar lag. With month gaps the imputed flow becomes
  `tna_t − tna_{t−k}(1+r_t)`. Gaps are not random with respect to stress.
  This is the one place a stress-correlated artifact could move the
  non-null half of Finding 1 (−0.62 vs −0.43).
- **MA-4.** 35c's "98-99% accurate where coded" is a base-rate statistic. The
  proxy (`merge_fundno.notna()`) is near-mechanically implied by a merger
  delist code; "always predict merged" would score in the high 90s. What
  matters for Finding 2 is **recall for `L`** (liquidations), the minority
  class. The 8% coded subsample is never checked for representativeness, yet
  the 0.07-0.47% liquidation redemption profile is computed only on it.
- **MA-5.** 35c's dormancy statistic scores an all-NaN month as dormant
  (`fillna(0)` after `sum(min_count=1)`). Dying funds file incomplete N-PORT
  items as they wind down, which manufactures exactly the 19.4% vs 6.6% gap
  the finding rests on. **Needs the per-field NaN rate printed before the
  claim stands.**
- **MA-6.** Numerator and denominator summed over different series within a
  wficn-month; `rr_sec` drops NaN months while `netd` fillna(0)s them, so the
  flat gross statistic and the differing net statistic are computed on
  DIFFERENT samples — the exact contrast Finding 1 rests on.
- **MA-7.** 35/35b/35c keep the first wficn for multiply-mapped series
  (`drop_duplicates("series_id")`); only 35d excludes them. Finding 2 runs on
  the contaminated link, Finding 1 on the clean one.
- **MA-8.** Every gross-redemption statistic is a median. Accelerated exit is
  a TAIL event — a run shows up as a few months at 5-20%, not as a shifted
  median. Report p75/p90/p95 and the share above 4% and 8%.
- **MA-9/10/11.** Stage 38: equal-weighted expense ratio across share classes
  (changes when classes are added or closed); asymmetric stressed/unstressed
  definitions; and section (c) selects "long-resisting spells" on
  `end_dur >= 12`, a post-t0 outcome, then evaluates them at quarter 8 inside
  the window that defines their selection.

## VERIFIED CLEAN — checked and could not break

- **The $M vs $ units fix is complete.** All nine files grepped; `tna` appears
  nowhere outside 35d's converted sites; the `1e6` cancels exactly in
  `nr_imp`. Corroborated independently: `rr_sec` and `rr_tna` agree at
  0.99-1.01% on completely different denominators, which a residual 10⁶ error
  could not produce.
- **Winsorization did not create the ghosting null.** `clip(-1, 2)` on a
  non-negative ratio truncates only above 200%/month and the statistics are
  medians, which cannot move unless the clip binds on half the sample.
- **The era split conditions only on the capitulation quarter** in all four
  37-series scripts. The definition is clean; only the sample filter differs.
- **37d's competing events are mutually exclusive and exhaustive**, ties
  broken toward recovery symmetrically, censoring coded correctly, horizon
  reads correct.
- **35b correctly implements the round-4 C3 fix** (death-table event clock,
  one row per fund, 12-month pre-period requirement).
- **35c's end_dt statistic is biased AGAINST its own conclusion** — a
  strength worth stating in the paper.
- **The "stressed" label's mild look-ahead attenuates toward the null**, so it
  works in the paper's favour. (Conditional on `rel4q` being trailing —
  `panel_lib` was out of scope. If `rel4q` has any forward component this
  reclassifies as MAJOR.)

## SPILLOVER INTO PAPER 1 — check before submission

v9.2 §8 says "roughly a fifth of capitulators eventually rebuild full
conviction, a pattern a companion paper takes up; the modal capitulator never
does." That sentence rests on 37d's bar-0.70 AJ numbers (24.5/18.2), which
are produced with unseeded jitter (±0.5pp, downward-biased) and on a sample
that differs from 37b's. "Roughly a fifth" survives ±0.5pp, and the implied
correction moves the wave arm UP (26.5 vs 24.5), which does not threaten the
sentence. **Judgement: the §8 sentence stands, but re-run 37d deterministically
and confirm before submitting.** One run, five minutes.

## ACTION ORDER

1. Re-run 37d deterministically (discrete-time AJ, no jitter) → confirm the
   Paper 1 §8 sentence. **Blocks submission.**
2. Re-run 37b/37c with the `>= 1995` filter so KM and AJ share a sample;
   rewrite the M6 attribution.
3. Bootstrap the era difference in 37d; do not quote the 0.80 inversion until
   it has an interval.
4. Rebuild stage 38's cut detector on a balanced window or a censored
   time-to-event; TNA-weight the fee.
5. Fund-clustered bootstrap + fund-FE for ghosting; add tail quantiles.
6. Calendar-lag the CRSP TNA in 35d.
7. Finding 2 cluster: separate missing from zero (MA-5), per-class recall
   (MA-4), re-run 35b/35c on 35d's clean link (MA-7).
