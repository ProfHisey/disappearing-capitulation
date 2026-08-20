// Capitulation paper, draft v9.2. Same numbers as v9.1; presentation only.
// v9.2 changes: (a) the five figure placeholders are replaced by the real
// PNGs from figs/, (b) Figure 4 (famous-fund validation traces) is newly
// placed in Section 3, where the validation is described, and (c) the
// Section 8 durability sentence is softened per stage 37d: the fold is
// durable for years, but roughly a fifth of capitulators eventually rebuild
// full conviction, flagged as a companion paper. Figure files expected in
// figs/ as fig1..fig6.png (1 = cohort decline, 2 = three-state CIF,
// 3 = two clocks, 4 = famous traces, 5 = grind, 6 = H7 cumulative).
// Run from the paper folder:  node make_draft_v9_2.js
// Requires the docx npm package (npm i docx@9); built and byte-verified
// against docx 9.7.1. Output goes next to this script.
//
// Capitulation paper, draft v9.1. Applies referee round 5 + code audit
// round 5: seam-quarter events excluded (5/10 instrument-suspect, 33j),
// Poisson CI, seam attrition, renormalization direction, Table 6, stress
// framing. v9 notes: Adds the out-of-sample extension to mid-2026
// (Section 6.1: N-PORT-computed Active Share, seam-validated, placebo-tested),
// the no-bunching robustness result, and Appendix A/B entries. Core 1990-2023
// numbers unchanged from v8. Inherited v7/v8 notes: Post-audit rebuild: all numbers from the
// fixed observed-clock machinery (stages 22/24b/26/27), per-10pp depth
// framing, cohort-vs-calendar reconciliation, audit confession in 3.1,
// expanded Appendix B. Derived from v6, which applied referee round 2:
// fund-level attribution, calculator-proof abstract, estimand reconciliations,
// consolidated confessions, restructured sections, practitioner implications,
// figure placeholders, verified new citations, unit-labeled death counts.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, PageBreak, ImageRun,
} = require("docx");

const FONT = "Georgia";

function p(text) {
  return new Paragraph({
    spacing: { after: 160, line: 276 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}
function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: "1a1a1a" })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: "1a1a1a" })],
  });
}
function caption(text) {
  return new Paragraph({
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, font: FONT, size: 18, italics: true, color: "444444" })],
  });
}
function figbox(text) {
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    alignment: AlignmentType.CENTER,
    border: {
      top: { style: BorderStyle.SINGLE, size: 6, color: "888888" },
      bottom: { style: BorderStyle.SINGLE, size: 6, color: "888888" },
      left: { style: BorderStyle.SINGLE, size: 6, color: "888888" },
      right: { style: BorderStyle.SINGLE, size: 6, color: "888888" },
    },
    children: [new TextRun({ text, font: FONT, size: 20, color: "555555" })],
  });
}

// Real figure images. The text column is 6.2 inches wide; figures are capped
// at that width and at 4.6 inches tall, and scaled to fit while preserving
// aspect ratio. docx sizes images in pixels at 96 dpi.
const PX_PER_IN = 96;
const FIG_MAX_W = 6.2 * PX_PER_IN;
const FIG_MAX_H = 4.6 * PX_PER_IN;
function pngSize(buf) {
  // PNG IHDR: width and height are big-endian uint32 at bytes 16 and 20.
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}
function fig(file) {
  const full = path.join(__dirname, file);
  const data = fs.readFileSync(full);
  const { w, h } = pngSize(data);
  const scale = Math.min(FIG_MAX_W / w, FIG_MAX_H / h);
  return new Paragraph({
    spacing: { before: 160, after: 60 },
    alignment: AlignmentType.CENTER,
    children: [new ImageRun({
      type: "png",
      data,
      transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
    })],
  });
}
function table(cols, rows, headerRows = 1) {
  const total = cols.reduce((a, b) => a + b, 0);
  const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: cols,
    rows: rows.map((r, i) =>
      new TableRow({
        tableHeader: i < headerRows,
        children: r.map((c, j) =>
          new TableCell({
            width: { size: cols[j], type: WidthType.DXA },
            margins: { top: 60, bottom: 60, left: 100, right: 100 },
            borders: { top: border, bottom: border, left: border, right: border },
            children: [new Paragraph({
              children: [new TextRun({ text: c, font: FONT, size: 18, bold: i < headerRows })],
              alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.RIGHT,
            })],
          })),
      })),
  });
}

const children = [];

// ------------------------------------------------------------ title page
children.push(new Paragraph({
  spacing: { before: 2200, after: 240 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: "Nobody Surrenders Anymore",
    font: FONT, size: 48, bold: true,
  })],
}));
children.push(new Paragraph({
  spacing: { after: 480 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: "The disappearance of capitulation in US active equity funds, 1990–2026",
    font: FONT, size: 26, italics: true,
  })],
}));
children.push(new Paragraph({
  spacing: { after: 120 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Colin Hisey", font: FONT, size: 24 })],
}));
children.push(new Paragraph({
  spacing: { after: 720 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: "Working draft v9.2, August 2026. Comments welcome. Please do not cite without permission.",
    font: FONT, size: 20, italics: true, color: "555555",
  })],
}));

children.push(h2("Abstract"));
children.push(p(
  "An active fund that falls behind its benchmark has three exits. It can recover, it can die, or it can quietly fold the portfolio toward the index. I measure folding, capitulation, as Active Share crossing from at least 70 percent to below 60 percent during a sustained underperformance episode, and I treat capitulation and fund death as competing failure modes across 25,990 episodes in 9,096 US active equity funds from 1990 to 2023. Three regularities emerge. Time drives capitulation: the quarterly hazard of folding roughly quintuples between the first six months underwater and the fourth year, a gradient that survives giving every fund its own baseline propensity. The two failure modes select different funds: death is the exit of small funds in severe shortfalls, folding the exit of large funds already near the closet-indexing line; severe underperformance predicts liquidation, not surrender. And capitulation has vanished, from a peak rather than from a timeless norm: the per-quarter folding hazard was near zero before 1995, crested with the closet-indexing wave of 1995 to 2009, and now runs at a fifth of that rate, a collapse that also appears threshold-free as the disappearance of large within-episode Active Share declines. Episodes begun in the 1990s folded about 7 percent of the time; the 2020s cohort stands at half a percent so far (right-truncated). Fund death declined little on any definition, and hardly at all for outright liquidations. Resisting was neither clearly rewarded nor clearly punished: funds still fighting after two years underwater went on to perform within roughly 2 percent a year of the funds that folded, an adequately powered bound, with matched point estimates small and mostly negative; both groups earned roughly zero gross alpha and fees-sized losses net. An out-of-sample extension through mid-2026, on Active Share computed independently from SEC N-PORT holdings and validated against the original panel where the two meet, finds the folding rate still at the floor: the 2024 to 2026 era runs at 0.18 events per 100 at-risk spell-quarters against 0.20 for 2010 to 2023, indistinguishable given 25 events."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ------------------------------------------------------------ 1 intro
children.push(h1("1. The exit nobody advertises"));
children.push(p(
  "An active stock fund exists to be different from its benchmark index. Difference is the product being sold. It is also the only source of embarrassment, because a fund that differs can trail, and a fund that trails for years faces a choice no prospectus describes. It can hold the line and keep explaining. It can quietly rebuild the portfolio to look like the index, which ends the underperformance almost by definition and also ends the reason for the fund's fees. Or events can take the choice away when the fund is liquidated or merged out of existence."));
children.push(p(
  "I came to this problem from engineering rather than finance, and the framing I could not shake is reliability analysis. A part under stress can fail in more than one way, and the useful questions are always the same. Which failure mode dominates at which stress level? Does the hazard rise with time under load? Has the failure surface shifted across generations of the part? This paper asks those questions about active funds under the stress of underperformance. Folding toward the index is one failure mode. Death is the other. Recovery is the part surviving the test."));
children.push(p(
  "Measuring the fold takes one tool from the academic literature. Active Share, introduced by Cremers and Petajisto, is the fraction of a portfolio that differs from its benchmark index. A pure index fund sits near zero, a concentrated stock picker above 90 percent, and the literature's convention calls a fund below 60 percent a closet indexer, charging active fees for an index-like portfolio. I define capitulation as a fund that entered an underperformance episode unambiguously active, at 70 percent or above, and closed below 60 percent before the episode resolved. The fund kept existing. The bet inside it did not."));
children.push(p(
  "One caution belongs here rather than buried in the limitations. Capitulation as measured is a fund-level event. When Active Share collapses mid-episode, the data cannot yet distinguish a manager losing conviction from a board losing patience, firing the manager, and installing someone tamer. Both are surrender by the fund; only one is surrender by the manager. Manager-level histories that separate the two are the top item on this paper's pending list, and until they arrive, the paper's claims are about what funds do. Where the prose speculates about the people inside, it says so."));
children.push(p(
  "Three findings organize the paper. First, the hazard of capitulation rises steeply with time underwater. Holding severity fixed, a fund in the third year of a losing streak is nearly five times more likely to fold in the next quarter than a fund in its first six months. The gradient holds within first episodes, within repeat episodes, and after giving every fund its own baseline propensity, so it is not the artifact of stubborn and fragile funds sorting over time. Fatigue accumulates inside the same fund."));
children.push(p(
  "Second, the two failure modes select different funds. Depth of underperformance strongly predicts death, and it does not predict folding once a fund's starting distance from the closet-indexing line is held fixed. An earlier draft reported that shallow episodes fold; an adversarial review caught what that mostly was, funds entering near the line having both shallow episodes and short trips to the threshold, and the claim is corrected here (Section 5). What survives correction is the selection pattern: small funds die and large funds fold, deep shortfalls kill without driving the fold, and time underwater drives folding while showing no robust pull on dying. Depth selects death; time and proximity select surrender. That is why the competing-risks structure matters beyond econometric hygiene."));
children.push(p(
  "Third, and this is the headline, capitulation has vanished, and it vanished from a peak rather than from a timeless norm. By entry cohort, roughly 7 percent of episodes that began in the early 1990s ended with the fund folding; for the 2010s the figure is close to 1 percent, and for the right-truncated 2020s half a percent. On the calendar clock the same history is sharper: folding was nearly absent before 1995, crested during the closet-indexing wave of 1995 to 2009, and has since collapsed to a fifth of the peak rate. The behavior, not merely the threshold event, went away: large within-episode Active Share declines, measured with no reference to any crossing, collapsed in step. Fund death shows no comparable trend: restricted to outright liquidations, the cleanest definition, the death rate is flat from the mid 1990s to today, and even the broadest distress-inclusive definition declined by far less than folding did. Whatever changed in active management changed the behavior of funds under stress much more than their mortality. The finding now extends past the original panel: holdings filed with the SEC through mid-2026, measured with independently computed Active Share, show the fold still as rare as ever (Section 6.1)."));
children.push(p(
  "The last substantive result asks whether folding was a mistake, and the answer is that the choice did not matter for future returns. Funds still fighting after eight quarters underwater went on to perform indistinguishably from the funds that had folded, adjusted for market, size, value, and momentum exposure, in net and gross returns alike, over the following three years. The design would have detected a conviction premium (or penalty) of about 1.5 percent a year with standard power, so this is an adequately powered null, not an inconclusive one. Both groups earned roughly zero alpha gross of fees. What the fold changed was not future performance but what investors were paying active fees for."));

// ------------------------------------------------------------ 2 related
children.push(h1("2. What was already known"));
children.push(p(
  "Six strands of prior work border this one, and the borders are worth drawing precisely. Cremers and Petajisto built the Active Share measure and documented that closet indexing exists and is costly to investors. Their question was about skill and fees in the cross-section. Mine is about the dynamics of a single fund under stress, when in a losing streak the closet indexing begins. Cremers, Ferreira, Matos, and Starks showed that explicit and closet indexing move together across countries as index competition disciplines active managers. That work measures levels across markets; this paper documents a behavioral transition within one market that level comparisons would not detect."));
children.push(p(
  "The career-concerns literature supplies the natural behavioral mechanism. Chevalier and Ellison showed that termination risk shapes the risk-taking of mutual fund managers, especially young ones, and Brown, Harlow, and Starks showed that mid-year losers gamble, increasing risk in the second half of the year to catch up. Kempf, Ruenzi, and Thiele then showed the other half: when employment risk dominates compensation incentives, losing managers reduce risk within the year. This paper is not the first to find de-risking under pressure, and does not claim to be. What it adds to that strand is the horizon and the object: a multi-year, portfolio-dissolving version of the retreat, measured in holdings rather than volatility, that ends the strategy itself, and the fact that this terminal form of retreat has almost disappeared. The two behaviors coexist in the data here: about 28 percent of eventual capitulators first raise Active Share mid-episode, the tournament gamble, before dissolving the portfolio. Short losing streaks invite gambles; long ones, historically, invited surrender. The tenure result in Section 8, where seasoned management protects against folding, is consistent with career concerns doing real work."));
children.push(p(
  "Lynch and Musto modeled strategy changes by poorly performing funds and predicted that the change follows sustained pressure, which matches the fatigue gradient here. What their framework does not predict is the secular disappearance of the behavior. On the death side, Lunde, Timmermann, and Blake estimated hazard models of fund closure and established that sustained underperformance kills funds; this paper adds the second, competing exit and shows the two exits respond to different inputs. Berk and Green's capacity logic offers the rational alternative reading, in which a fund that has grown large optimally sheds active risk. Their model does not speak of Active Share directly, but a natural extension of its logic predicts de-activation following success, inflows, and growth. Section 8 races that prediction against the pressure story directly, and pressure wins. Finally, Frazzini, Friedman, and Pomorski criticized Active Share as a predictor of skill, and nothing here disputes that. Active Share serves this paper as a measure of what a portfolio is doing, not of whether its manager is good."));

// ------------------------------------------------------------ 3 data
children.push(h1("3. Data and how the measurements were checked"));
children.push(p(
  "The spine of the data is the CRSP Survivor-Bias-Free US Mutual Fund database for returns, assets, share classes, and fund exits, joined at the fund level (share classes aggregated) to the Notre Dame Active Share database, which computes quarterly Active Share for US equity funds against roughly twenty candidate benchmark indexes. A fund's benchmark each quarter is the index against which its Active Share is lowest, the standard convention, since a closet indexer should be judged against whatever index it actually hugs. Section 6 reports how results change when this choice is frozen, restricted, or replaced."));
children.push(p(
  "Benchmark returns are official throughout: actual index return series from Cremers, Petajisto, and Zitzewitz for 1979 to 2008, official FTSE Russell monthly index returns from 2008 onward, extracted from the index provider's own files, and the CRSP S&P 500 total return. Where the segments overlap they agree to the basis point, and internal checks (the Russell 3000 must equal the cap-weighted blend of the Russell 1000 and 2000 every month) hold across the sample. The Active Share measurements were validated against the published literature directly: over the 49,881 fund-quarters where this panel overlaps Petajisto's published data, the correlation is 0.98 with a mean absolute difference of one percentage point."));
children.push(p(
  "The pipeline was also checked against history, and Figure 4 shows the result. Fidelity Magellan's slide into index-hugging appears where the record says it should and is classified as a capitulation. Bill Miller's Value Trust rides a fourteen-quarter, 23 percent deep episode through the financial crisis without folding, which is exactly his reputation. Sequoia's Valeant episode runs nineteen quarters and reaches 28 percent behind. CGM Focus, run by the famously stubborn Ken Heebner, shows six deep episodes and zero capitulations. The Vanguard 500 index fund, planted as a negative control, is correctly removed by the passive-fund screen."));
children.push(fig("figs/fig4.png"));
children.push(caption(
  "Figure 4. Validation against the record: minimum-benchmark Active Share for three famous funds, with the 70 percent activity line and the 60 percent capitulation line marked. Fidelity Magellan slides across in 2012 and is classified as a capitulation; Bill Miller's Value Trust and Sequoia ride deep, multi-year losing episodes without folding."));
children.push(p(
  "An underperformance episode, a spell, begins when a fund with Active Share of at least 70 percent falls behind its benchmark over the trailing four quarters, net of fees. It ends in one of four ways. Recovery, when trailing relative performance turns non-negative. Capitulation, when Active Share closes below 60. Death, defined as every share class of the fund terminating within four quarters of the spell's end (results are insensitive to windows of one to four quarters, and a zero-quarter window fails mechanically because funds stop reporting holdings shortly before they legally die). Or censoring at the edge of the data. The sample holds 25,990 spells across 9,096 funds. The panel formally extends back to 1980 but is thin before 1990 because the fund-linking tables are, so results are stated for 1990 through 2023 and nothing is claimed about the 1980s."));
children.push(h2("3.1 What changed during the analysis"));
children.push(p(
  "This project ran an adversarial review of its own design before drafting, twenty-three critiques answered by the battery in Appendix A, plus two rounds of adversarial line-by-line review of the pipeline code itself, and six things changed as a result. They are collected here once rather than sprinkled through the paper. First, an early version reconstructed Russell index returns from a constituent holdings file, and a triangulation test against the French size portfolios caught the reconstruction running about 1.2 percent per month hot. Replacing it with official index returns made the era decline stronger, which is worth savoring: the paper's headline was being muted, not manufactured, by its worst measurement error. Second, an earlier claim that the decline extends back through the 1980s was withdrawn; the early panel is too thin, and the one external ruler available for that period shows no decline before the mid 1990s. Third, a planned comparison of manager capitulation against client capitulation, redemptions calibrated to be exactly as rare as manager folds, was shelved when the calibration exposed the flow data's limits: at the calibrated threshold, only nine to fifteen spells in three decades contain both events, depending on the flow convention, and imputed flows cannot distinguish client panic from share-class bookkeeping at that severity. That comparison waits for regulatory filings with true gross redemptions. Fourth, a first version of the forward-return comparison in Section 8 reported that resisters trailed capitulators by 2.4 percent a year; drawing the cumulative-spread figure exposed an arithmetic error, the risk-free rate subtracted from a long-short spread in which it cancels, and the corrected spread is zero. The bias equaled the sample-average T-bill rate almost exactly. Section 8 reports the corrected numbers, and the episode is one reason every headline claim in this paper now has a figure. Fifth, the code audit found two structural defects in the spell machinery: durations were counted over observed quarters but converted back to calendar dates downstream, misdating the crossing quarter for two thirds of capitulations (by a median of four quarters, because reporting gaps are common inside long spells), and episodes beginning on a fund's final observed quarter were silently dropped, undercounting deaths at the panel edge by 336. Fixing both moved the headline hazard ratios at the second decimal, raised the death share by about one percentage point, and cut the depth coefficient roughly in half; it also revealed that the direction of the death-duration gradient is not robust to sample and controls, so this paper claims nothing about it. Every number here comes from the corrected machinery, and both audit reports are in the replication package. Sixth, a third referee round identified a distance-to-threshold confound in the depth result: capitulation requires Active Share to travel from above 70 to below 60, and the distance to be traveled was never controlled. With entry Active Share in the model the apparent depth-suppression of folding disappears, and Sections 1 and 5 state the corrected finding; the era decline survives the same control within every entry band. A cleaning-rule correction with no effect on results is described in Appendix B."));

// ------------------------------------------------------------ 4 how spells end
children.push(h1("4. How losing streaks end"));
children.push(p(
  "Counting endings honestly requires treating recovery as the competing outcome it is, since funds recover through the same mean reversion that drives the depth variable. In simple shares, 80 percent of spells end in recovery, 8.6 percent in death, 1.9 percent in capitulation, and 9.5 percent are censored at the data's edge. Most losing streaks just end, and they end quickly: nearly seventy percent of spells recover within two years. Table 1 reports the forward-looking version, the cumulative chance of each ending as a spell wears on, with all three endings competing, since a fund that has recovered or died can no longer fold. Death outruns capitulation four-to-five to one at every horizon. Capitulation was always the rare exit, which is precisely why its disappearance needed a long panel to see, and why the hazard models of Section 5, which condition on the spell still running, are the right instrument for studying it."));
children.push(table(
  [3200, 1500, 1500, 1500, 1500],
  [
    ["Cumulative incidence (competing risks)", "2 yrs", "4 yrs", "6 yrs", "10 yrs"],
    ["Fund recovers", "68.7%", "84.2%", "87.5%", "88.3%"],
    ["Fund dies", "7.4%", "9.0%", "9.4%", "9.5%"],
    ["Fund capitulates", "1.4%", "2.0%", "2.1%", "2.2%"],
  ]));
children.push(caption(
  "Table 1. Aalen-Johansen cumulative incidence with recovery, death, and capitulation as mutually competing outcomes; only data-edge exits are censored. Horizons in years since spell entry. The earlier convention treating recovery as censoring inflates the death and capitulation columns several-fold; the multinomial competing-risks model confirms the Section 5 hazard results are unaffected by the choice."));
children.push(fig("figs/fig2.png"));
children.push(caption(
  "Figure 2. Cumulative incidence of the three endings as a spell lengthens (Aalen-Johansen, competing risks). The capitulation curve flattens dramatically for spells beginning after 2010; the death curve does not."));

// ------------------------------------------------------------ 5 fatigue + mode
children.push(h1("5. Time drives folding, depth drives death"));
children.push(p(
  "The workhorse is a discrete-time hazard model. Each quarter a fund spends inside a spell is one observation, the outcome is whether the fund folded that quarter, and the covariates are the spell's duration so far, its depth so far (the worst trailing four-quarter shortfall to date, lagged one quarter so the event cannot explain its own cause), and era, with standard errors clustered by fund. Table 2 reports this reduced form on the full sample, once for capitulation and once for death. Adding fund size, fees, and manager tenure costs a third of the sample (manager dates are thin early) and moves the era ratio from 0.20 to 0.22; adding the fund's Active Share at spell entry, the single most important covariate this design admits, moves it to 0.27. Entry Active Share matters enormously for folding, and honestly interpreting Table 2 requires it: a fund entering a spell at 72 percent has a twelve-point trip to the closet-indexing line, a fund entering at 95 a thirty-five-point one, and the folding hazard falls off steeply in that distance (funds above 90 percent almost never fold: 55 events in 71,326 spell-quarters). The era decline survives inside every entry band, 0.28 for funds entering at 70 to 80, 0.18 at 80 to 90, 0.42 above 90, so the disappearance is not the marginal near-line funds exiting the sample. One estimand distinction is stated rather than blurred: entry Active Share is chosen by the fund, and it rose across eras plausibly in response to the same competitive pressure that killed folding, so it is partly a mediator, not only a confounder. The uncontrolled 0.20 is then the total decline in folding; the controlled 0.27 is the decline holding the industry\u2019s repositioning fixed, a conservative lower bound on the change. The paper reports both and claims only what both support."));
children.push(table(
  [3400, 1800, 1800],
  [
    ["Hazard ratio vs. baseline", "Capitulation", "Death"],
    ["Quarters 3–4 underwater", "2.01", "0.66"],
    ["Quarters 5–8", "3.18", "0.78"],
    ["Quarters 9–12", "4.83", "0.90"],
    ["Quarters 13+", "5.26", "0.87"],
    ["10 pp deeper shortfall", "0.74", "1.15"],
    ["Era 1990–94 (vs 1995–2009)", "0.09", "0.98"],
    ["Era 2010–23 (vs 1995–2009)", "0.20", "0.79"],
  ]));
children.push(caption(
  "Table 2. Discrete-time hazard ratios, both outcomes, full-sample reduced form: 130,714 spell-quarters, 25,990 spells, 496 capitulations, 2,238 deaths, fund-clustered. Duration baseline is quarters 1–2; depth enters continuously and is reported per 10 percentage points of additional trailing shortfall. All capitulation duration effects, both depth effects, and the capitulation era effects are significant at the 1% level. The era hazard ratio for capitulation is 0.196 with a 95% confidence interval of roughly 0.16 to 0.24 (0.22 with size, fee, and tenure controls; 0.27 with entry Active Share); the pre-1995 row rests on only two calendar-clock events and its interval is wide. The death duration bins are shown for completeness but their direction is not robust to sample and controls (Section 3.1), and the paper makes no claim on them. The capitulation depth row is a reduced-form coefficient that Section 5 shows to be largely a distance-to-threshold effect; with entry Active Share controlled it is 0.44 per unit with reversed reading (z \u22122.4), and the paper claims only the death-side depth gradient."));
children.push(fig("figs/fig3.png"));
children.push(caption(
  "Figure 3. The two clocks. Capitulation hazard rises with time underwater; its apparent depth profile is largely proximity to the closet-indexing line plus competition from death (Section 5). Death hazard rises with depth, monotonically, and shows no comparable duration ramp."));
children.push(p(
  "Read the capitulation column first. The duration gradient rises monotonically to above five, and it survives everything thrown at it: entry-Active-Share controls (2.0 to 2.2 within every band), first spells and repeat spells separately, a random-intercept model that gives every fund its own baseline, and both halves of a deterministic sample split. The depth row requires the correction promised in Section 1. In the reduced form, deeper shortfalls appear to suppress folding (0.74 per ten percentage points); with entry Active Share controlled, that suppression disappears and mildly reverses. The reduced-form coefficient was mostly proximity, near-line funds having both shallow episodes and short trips to the threshold, compounded by deep spells competing against death, and this draft retires the earlier 'shallow spells fold' claim accordingly. Depth's real work is in the death column, where no threshold mechanics exist to confound it: each ten points of deeper shortfall raises the death hazard by 15 percent, monotonically across bins, sharpening further when death is restricted to liquidations. Size completes the split, from the controlled specification: each log point of assets cuts the death hazard by a third and nudges the capitulation hazard up. Small funds die, deep funds die; big funds near the line fold, and they fold on the clock, not the depth gauge."));
children.push(p(
  "The standard statistical objection is that rising duration effects can be manufactured by funds differing from one another, fragile types exiting early and tough types accumulating at long durations. Three tests meet it. The gradient holds within first spells only and within repeat spells only. It holds in a mixed model that gives every fund its own baseline hazard, the direct answer, and the estimated spread of those baselines is wide, so the objection's premise was correct even though its conclusion was not. And it holds in each half of a deterministic split of the sample. Fatigue operates within funds: the same fund becomes more likely to fold the longer its streak drags on."));
children.push(p(
  "Two texture results earn a sentence each. Capitulation arrives as a grind and then a plunge: mean Active Share slides about five points over the two years before the crossing, leaving the average eventual capitulator sitting at the 70 percent activity line two quarters out, and then collapses fourteen points in the crossing quarter itself (Figure 5; the holdings-measured drop in Table 5's subsample is ten points on a different instrument). (Once that final slide enters the model the other covariates lose force; the paper reads this as mechanical decomposition, the slide being the crossing in progress, not as independent evidence of mediation.) A holdings-level decomposition settles what the shape suggests: for every crossing with portfolio reports around it, the prior quarter's portfolio is rolled forward at market prices with no trades, giving the Active Share the fund would have had by sitting still. Table 5 reports the split. Price drift explains essentially none of the drop (a tenth of a point of a ten-point mean decline), the fund's own trades carry a median 99 percent of it, and both placebos, the same funds two quarters earlier and the fighters at quarter eight, show next to nothing. The recomputed Active Share correlates 0.93 with the Notre Dame values, validating the machinery. Capitulation is executed, not endured. A desensitization pattern, survivors of a previous spell appearing less depth-sensitive in the next, appears in the raw stratified fits but inherits the same distance-to-threshold confound as the depth row, so it is noted and not claimed."));
children.push(fig("figs/fig5.png"));
children.push(caption(
  "Figure 5. The grind before the fold: mean Active Share in the quarters preceding a capitulation crossing, against the mean across all at-risk spell-quarters at least three quarters before any crossing. The drop concentrates in the final two quarters; Table 5 shows it is executed through trades."));
children.push(table(
  [3400, 1200, 1900, 1400, 1600],
  [
    ["Event group", "n", "Observed dAS", "Drift", "Trading"],
    ["Capitulation crossings", "127", "\u221210.2 pts", "\u22120.1", "\u221210.1"],
    ["Same funds, 2q earlier", "115", "\u22120.8 pts", "+0.0", "\u22120.8"],
    ["Fighters at quarter 8", "380", "+0.0 pts", "+0.1", "\u22120.0"],
  ]));
children.push(caption(
  "Table 5. Trading-versus-drift decomposition of Active Share changes (means, percentage points). Drift is the no-trade counterfactual change (prior holdings rolled forward at market returns, compared to the current benchmark); trading is the remainder. Among the 112 crossings with a material drop, trading exceeds drift in 93 percent; the trading share of the drop is 90 percent at the 25th percentile, 99 at the median, 107 at the 75th. Sample: events since 2002 with holdings in adjacent quarters (127 of 496 crossings, dated at the true calendar crossing); 27 percent of positions, mostly non-common-stock holdings, lack matched CRSP returns in the roll and are held flat, and 2.5 percent have partial gaps; recomputed Active Share correlates 0.926 with published values."));

// ------------------------------------------------------------ 6 disappearance
children.push(h1("6. The disappearance"));
children.push(p(
  "Sort the same spells by the five-year window in which they began. Capitulation falls by an order of magnitude, with one sharp break around 2000 and a long slide after. Total deaths, in the rightmost column, include mergers and carry no trend, and the truncated final cohort has simply had less time to die; Section 7 does the death accounting properly."));
children.push(table(
  [2600, 1500, 1700, 2000, 1600],
  [
    ["Spells beginning", "Spells", "Capitulated", "Capitulated (strict)", "Died (all)"],
    ["1990–1994", "1,047", "6.8%", "6.1%", "8.9%"],
    ["1995–1999", "1,875", "7.6%", "6.9%", "7.9%"],
    ["2000–2004", "3,409", "2.1%", "1.8%", "11.3%"],
    ["2005–2009", "3,540", "1.6%", "1.6%", "8.8%"],
    ["2010–2014", "5,358", "1.4%", "1.1%", "9.1%"],
    ["2015–2019", "5,947", "1.0%", "0.9%", "9.4%"],
    ["2020–2023", "4,756", "0.5%", "0.5%", "5.3%"],
  ]));
children.push(caption(
  "Table 3. Outcome shares by entry cohort. The seven cohorts hold 25,932 spells; the remaining 58 began before 1990 and are excluded from era claims throughout. The strict column reclassifies crossings still arguably active against the entry benchmark (54 events) as non-events; the entry-benchmark-only (frozen) definition, lower in every cohort, declines identically (Section 6). The 2020–23 cohort is right-truncated for both outcomes; the Section 6.1 extension partially relieves that truncation, following 1,524 of the episodes censored at the 2023 edge."));
children.push(fig("figs/fig1.png"));
children.push(caption(
  "Figure 1. The paper in one picture. Capitulation by entry cohort collapses after 2000; liquidation-only death rates stay flat. Annotations mark the 2000 break and the truncated final cohort."));
children.push(p(
  "In hazard terms, spells active after 2010 fold at 0.20 times the 1995 to 2009 rate, with a confidence interval of roughly 0.16 to 0.24, conditioning on duration and depth; adding size, fees, and tenure moves the ratio to 0.22. The covariates neither create nor explain the decline. And the identical specification run with death as the outcome puts the post-2010 death ratio at 0.79, a real but modest decline (0.92 and indistinguishable from no change once entry controls are added). That pair of numbers, 0.20 against 0.79, is the paper's sharpest single contrast: the folding hazard fell four times further than the death hazard, in the same model, on the same spells."))
children.push(p(
  "One reconciliation belongs here, because Tables 2 and 3 tell the era story on two different clocks and a careful reader will notice the tension. By entry cohort (Table 3), the earliest spells capitulated most. By calendar quarter (Table 2), capitulation was nearly absent before 1995: just two events, against 319 in 1995 to 2009 and 175 after, and the pre-1995 hazard ratio of 0.09 says the per-quarter rate was lowest exactly when the cohort rate was highest. Both are true, and the resolution is the history: spells entered in the early 1990s ran long and crossed the closet-indexing line during the late-1990s and 2000s wave that Cremers and Petajisto documented. Capitulation was not an eternal constant that faded. It rose with the closet-indexing era and died with it, which sharpens the question the conclusion takes up: the exit did not just close, it opened and then closed within one professional generation."))
children.push(p(
  "A threshold event can disappear while the behavior it proxies survives, funds learning to stop at 61 instead of 58, so the decisive exhibit is threshold-free: the distribution of within-episode Active Share declines, measured with no reference to any crossing. It collapsed in step. The share of episodes containing a decline of ten points or more fell from 15 percent (pre-1995 entries) to 9 percent (1995 to 2009) to 3.5 percent (2010 onward); the share ever touching 65 percent, five points above the closet-indexing line, fell from 10 to 5 to 2 percent. Entry Active Share rose across the same eras, giving later funds more room to fall, so composition works against the decline-size metric (the touch-a-level metrics partly reflect the higher starting points, which is why the paper leads with the former). The remaining compositional channels were then tested directly. Within every entry-Active-Share band the collapse holds (22 to 7 percent of spells with ten-point slides in the 70-to-80 band, 19 to 5 at 80 to 90, 8 to 2 above 90). Split by clientele, the collapse is concentrated where the behavior lived: retail funds slid often and stopped (16.9 to 3.5 percent), while institutional funds rarely slid in any era (5.5 to 3.3), the two converging to the same low rate; since the decline is largest within the retail half, the falling retail share of assets (91 to 58 percent) cannot explain it, though the contrast itself is telling, and Section 11 returns to it: surrender was a retail-fund behavior, and retail funds converged to the standard institutional mandates had enforced all along. On the evidence in hand, funds did not learn to stop short of the line. They stopped sliding."))
children.push(p(
  "One identification note, stated rather than buried. Duration, calendar era, and entry cohort are linearly dependent, so their joint appearance in the hazard model rests on functional-form restrictions: duration in bins, era in three blocks. The key cross-restriction is testable, and tested: interacting the duration gradient with the modern era gives 0.64 (z \u22122.3), so fatigue is attenuated but far from absent after 2010, and the era decline is not an artifact of a changing duration profile. Launch-cohort fixed effects, the third axis, move the era ratio from 0.22 to 0.25."));
children.push(p(
  "A decline this large invites suspicion, and the suspicion was outsourced before being answered. Each mundane mechanism that could fake Table 3 became a test, summarized in one sentence each here and in full in Appendix A. Fee drag entering the spell definition: rerun gross and with fee-sized entry buffers, decline intact (era ratio 0.20 both ways), and under 8 percent of spells are shallow enough to be fee artifacts at all. A benchmark menu that grows over time, mechanically lowering minimum Active Share later: rerun with the menu frozen at four indexes that exist throughout, decline steeper. Fixed 70/60 thresholds interacting with market concentration: redefine both thresholds relative to same-benchmark peers in the same year, decline intact. Holdings reported semiannually early and quarterly later: force semiannual observation everywhere, decline intact. Cohort composition: launch-cohort controls move the hazard ratio from 0.22 to 0.25. Incubation bias, left truncation, death-window definitions, and cleaning rules get the same treatment with the same outcome. One further mundane mechanism, managers managing the metric itself by bunching just above the salient thresholds, was tested directly: the distribution of Active Share is smooth through 60, 70, and 80 percent, before and after the measure became famous in 2009, so the crossings in this paper are not contaminated by threshold management, and Active Share appears never to have become a managed number the way tracking error did."));
children.push(p(
  "One measurement issue deserves its own paragraph. The fund's nearest index can change between entry and crossing, so part of a measured drop could reflect relabeling rather than retreat. Self-designated benchmarks are known to be gameable (Sensoy); the min-Active-Share convention removes the fund\u2019s say but introduces the reassignment mechanics measured here. On correctly dated crossings the accounting is: 60 percent of capitulations are frozen-confirmed, the fund crosses below 60 against the very benchmark it started with; 39 percent involve reassignment; fewer than one percent are untrackable. Among the reassigned minority, the median Active Share against the original benchmark at the crossing is 0.66, and two thirds sit below the 70 percent activity line against the ruler they started with, so the typical reassigned fold is no longer meaningfully active against any nearby index; four in five reassignments stay within the same size segment. Two honest wrinkles are reported rather than argued away: the reassignment share drifts upward across eras (32 to 46 percent), and only a third of reassignments land on a style sibling of the original index. Neither can carry the result, for two reasons the reader can check. Reclassifying every reassigned crossing that was still arguably active (Active Share of 0.75 or more against the entry benchmark) as a non-event removes 54 of 496 events and leaves the era hazard ratio at 0.20. And rebuilding the event entirely inside the frozen definition, each fund judged only against its entry benchmark, reproduces the decline on its own: 5.1, 2.3, and 0.6 percent across the three eras on well-tracked spells. Table 3 reports cohort rates under both definitions."));


// ------------------------------------------------------------ 6.1 extension
children.push(h2("6.1 Out of sample: the fold stayed rare through mid-2026"));
children.push(p(
  "The original panel ends where its Active Share source ends, September 2023. To see past that edge, this draft adds an extension computed from primary sources: portfolio holdings filed with the SEC on Form N-PORT, linked to the panel's funds through the SEC's own registration lists (a union of the annual lists for 2019 through 2026, so funds that died along the way are not dropped from the bridge), with Active Share computed against FTSE Russell index weights and CRSP S&P 500 constituent weights month by month. The result is a second, independent instrument for the same quantity: 51,866 fund-quarters of Active Share across 5,547 funds, of which 44,234 quarters belonging to 4,685 funds already in the panel enter the analysis (862 post-2023 entrants are deferred, as described below), covering the fourth quarter of 2023 through the second quarter of 2026; after the panel's own screens, 25,606 of those quarters reach the spell machinery."));
children.push(p(
  "The two instruments agree where they meet. For funds observed by both at the seam, the correlation between the extension's Active Share and the panel's final observation is 0.97 among Russell-benchmarked funds, with a median difference of a tenth of a percentage point; across all matched funds the figures are 0.93 and half a point. The extension follows 1,524 of the 2,358 episodes that were open and uncapitulated at the original panel's edge; the remaining third belong to funds the extension cannot link or identify, an attrition disclosed here because linkage could in principle select on fund type. Because the seam comparison spans two adjacent quarters on two instruments, the 0.97 conflates instrument disagreement with one true quarter of change, and so understates the agreement."));
children.push(p(
  "The question the extension exists to answer is whether the disappearance held. It did. Re-running the full spell machinery on the extended panel yields 25 capitulation crossings in the 2024 to 2026 calendar era, entirely outside the original data, a rate of 0.18 events per 100 at-risk spell-quarters, with an exact Poisson 95 percent interval of 0.12 to 0.27, a width that understates uncertainty to the extent events cluster within funds. The comparable rate is 0.20 for 2010 to 2023 and 0.82 for 1995 to 2009, so the new era is statistically indistinguishable from the modern floor and far below the wave: at the wave rate, the extension's at-risk quarters would have produced roughly 112 events against the 25 observed. Ten further crossings are dated in the seam quarter itself, 2023Q4, the first quarter measured on the new instrument, and they are excluded from these era claims: half of them show instrument-level disagreements at the seam large enough to manufacture a crossing, so the paper counts none of the ten as behavior (in the extended build's bookkeeping they raise the 2010 to 2023 calendar count from 175 to 185 without moving its rate from 0.20). This is a rate comparison, not the hazard ratio of Table 2, whose era coefficient of 0.20 is a different quantity that happens to share its digits. The extension speaks only to folding, since deaths after 2023 are not yet accounted, and the rate describes incumbent funds, because the extension follows the panel's own universe. The test is a stressed one: more than half of incumbent active fund-quarters in 2024 to 2026 sat inside an underperformance spell, so the fold stayed rare not for lack of funds under pressure but for lack of funds choosing it. Table 6 collects the comparison."));
children.push(table(
  [2600, 1400, 2400, 2000],
  [
    ["Calendar era", "Events", "At-risk spell-qtrs", "Rate per 100"],
    ["1995\u20132009", "319", "39,016", "0.82"],
    ["2010\u20132023", "185*", "90,991", "0.20"],
    ["2024\u20132026", "25", "13,722", "0.18"],
  ]));
children.push(caption(
  "Table 6. Capitulation events and at-risk exposure by calendar era on "
  + "the extended panel, paper definition throughout (crossing below 60 "
  + "percent inside an underperformance episode). *Includes the ten "
  + "2023Q4 seam-quarter crossings excluded from the extension-era "
  + "claims in the text; excluding their at-risk quarters as well "
  + "leaves the 2010\u20132023 rate at 0.20. The 2024\u20132026 exact "
  + "Poisson 95 percent interval is 0.12 to 0.27 per 100."));
children.push(p(
  "One false alarm during this extension belongs in the Section 3.1 tradition. A preliminary comparison, run before the extension was integrated into the paper's machinery, appeared to show surrender returning at wave-era rates. An adversarial audit traced the appearance to definitional mismatches, the preliminary count having used a different threshold and no spell conditioning, and a placebo test, running the identical procedure at boundaries in known-quiet years, produced the same apparent surge at every boundary; on the like-for-like statistic the real boundary in fact came in below every placebo, so the correction turned a false alarm into affirmative evidence. The integrated numbers above are what survived. Two scope limits are disclosed rather than hidden: the extension follows funds already in the panel, deferring post-2023 entrants until they can be screened for passive management the same way, and its Active Share is computed on the portion of each portfolio with identifiable US securities, renormalized to sum to one. That is the low-Active-Share convention, biased if anything toward recording more crossings, so the finding that folding stayed rare runs against the grain of the measurement choice; the alternative convention, carrying unidentified value as active weight, is a pending robustness column."));

// ------------------------------------------------------------ 7 death flat
children.push(h1("7. Death did not get rarer the way surrender did"));
children.push(p(
  "Counting deaths requires stating the unit, because two different counts appear in this section. At the fund level, across all linked funds in the CRSP universe, terminal exits split into 2,453 liquidations, 3,712 mergers, and 1,802 exits with ambiguous codes. At the spell level, which is what Table 3 counts, 2,238 of the 25,990 spells end in the fund's death, of which 693 are liquidations, 1,173 mergers, and 372 exits with ambiguous codes, and classifying mergers by the acquired fund's condition, trailing performance underwater or assets halved over two years, marks 45 percent of merger exits, at the fund level, as distressed."));
children.push(p(
  "The paper's contrast survives with one refinement this draft adds rather than hides. On the strictest definition, liquidation rates for spells beginning in 1995 through 2009 versus 2010 through 2023 are 2.8 and 2.7 percent, flat (hazard ratio 0.85, marginal), across exactly the period in which capitulation's hazard fell by four fifths; if anything, liquidation was rarer still in the early 1990s, at 1.6 percent (Figure 1). Broadening death to liquidations plus distressed mergers, it did decline after 2010, by about a third (hazard ratio 0.65). So the accurate statement is graded, not absolute: the strictest form of death is flat, the distress-inclusive form fell modestly, and no definition of death comes near capitulation's fivefold collapse. The one exit that vanished is the chosen one. The depth gradient also sharpens on liquidations only (0.27 per unit of depth against 0.44 all-cause), administrative mergers diluting the all-cause signal; in raw decile terms, the deepest tenth of spells dies at 12.5 percent against 3.4 percent for the shallowest. Redemption pressure fits the same picture with a long fuse: funds that eventually die bleed about 2 percent of assets per quarter beginning two years before the end, far outside any liquidation-announcement window, worsening to 5 percent per quarter in the final two. The paper states that as timing, not causation. Put together: the market still removes failing funds at the old rate, while the funds' own preemptive surrender has nearly stopped."));

// ------------------------------------------------------------ 8 folding + horse race
children.push(h1("8. Was folding a mistake, and what drives it?"));
children.push(p(
  "If conviction is valuable, funds that resist through long losing streaks should eventually be paid for it. The test is a calendar-time comparison, run two ways. In the unmatched design, capitulators enter a portfolio the month after their Active Share crosses below 60 and resisters enter the month after their eighth consecutive underwater quarter, using only information available at formation. In the matched design, both groups are formed standing at the same milestone of their spells, quarter eight, split by whether they had folded by then, so time-in-spell cannot drive the comparison; milestones at quarters four and twelve give the same answer. Both portfolios hold three years, equal weighted. Table 4 reports the results under three rulers."));
children.push(table(
  [3800, 1700, 1700, 1900],
  [
    ["Annualized, 36-month hold", "Capitulators", "Resisters", "Spread (R − C)"],
    ["Four-factor alpha, net", "−1.04%", "−1.17%", "−0.06% (t −0.1)"],
    ["Four-factor alpha, gross", "+0.18%", "+0.17%", "+0.06% (t +0.1)"],
    ["Return vs own benchmark, net", "−2.39%", "−2.43%", "−0.00% (t −0.0)"],
    ["Matched at quarter 8, net*", "−1.11%", "−1.77%", "−0.66% (t −1.0)"],
  ]));
children.push(caption(
  "Table 4. Calendar-time portfolios: 496 capitulator entries and the resisters still fighting at quarter eight, entries dated at the true calendar crossing or milestone, membership deduplicated. Unmatched rows report each group over its full window; the matched row (*) reports both groups on their 278 common months, so the level difference equals the spread by construction. The matched spread's 95% confidence interval is [−1.9%, +0.6%]; milestones at quarters four and twelve give −0.6% and +0.4%. Minimum detectable spread at 80% power: about 1.5% per year unmatched, 1.8% matched. Funds dying mid-hold simply stop contributing months (no delisting return is imputed); 21.6% of folded and 15.2% of fighting entries die within the 36-month window. HAC standard errors."));
children.push(p(
  "Every ruler agrees within its precision, and the bound is the finding. The unmatched factor-adjusted spread is a few basis points with a t-statistic near zero, net and gross, and the own-benchmark ruler is an exact zero. The matched designs, the cleanest comparison, produce small negative point estimates, minus 0.6 to 0.7 percent a year at two of three milestones with confidence intervals spanning zero, so the honest statement is a bound, not a declaration: resisting neither earned nor cost more than about 1.5 to 2 percent a year, and if anything the point estimates lean toward a small, statistically indistinguishable penalty of roughly fee size. What the data rule out is the thing worth writing home about, a conviction premium. The levels are as telling as the spread. Both groups earn approximately zero gross alpha after formation (+0.2 percent a year for each) and approximately minus-fees net, and both bleed the same 2.4 percent a year against their own chosen benchmarks, so whatever information the fold-or-fight choice carried, it was not information about future returns. The fold itself is durable: capitulators’ mean Active Share sits at 0.55 to 0.58 for three years after the crossing, so a folded fund is not a fund catching its breath before re-activating, it is an index-hugger for years. Over longer horizons the door is not quite shut: roughly a fifth of capitulators eventually rebuild full conviction, a pattern a companion paper takes up; the modal capitulator never does. The one economic difference the choice made is what the fee bought afterward: capitulators charged active fees for index-like portfolios, resisters charged them for live bets that generated no alpha. The claim is about averages, not individuals; Bill Miller's crisis spell ended in recovery, and anyone holding him in 2005 had been paid handsomely for years of resistance."));
children.push(fig("figs/fig6.png"));
children.push(caption(
  "Figure 6. Cumulative factor-adjusted return of resisters minus capitulators in calendar time, ending near zero after three decades. This figure exposed the arithmetic error described in Section 3.1: the originally reported spread implied a cumulative gap near seventy points, which the picture flatly contradicted."));
children.push(p(
  "What, then, drives the fold? It is not the risk-shifting documented by Huang, Sialm, and Zhang, which operates within a live strategy; the fold ends the strategy. The rational alternative to a behavioral reading is capacity logic in the spirit of Berk and Green, in which a successful fund grows to the point where shedding active risk is optimal. That story and the pressure story make opposite predictions about what precedes a capitulation, so they can race in the hazard model. Pressure wins cleanly. A year of asset shrinkage raises the odds of folding by roughly 70 percent (hazard ratio 0.59 per unit of four-quarter growth, z −4.1); last quarter's flow predicts folding on its own but carries no weight once cumulative shrinkage is in the model, so the pressure is the accumulated bleed, not the latest statement. Folding follows sustained failure and shrinkage, not success and growth. Managerial seasoning also matters: each year of manager tenure cuts the capitulation hazard by about 8 percent, an estimate that agrees across three specifications and survives controlling for fund age. It carries a flag, since CRSP's manager-date field is missing for a third of the sample, and it awaits confirmation in cleaner manager data. Taken together with the career-concerns literature, the picture is of an exit chosen under duress by those with the least accumulated standing, an exit that has now largely closed."));

// ------------------------------------------------------------ 9 practitioner
children.push(h1("9. What this means for allocators and boards"));
children.push(p(
  "Three practical readings follow from the failure-mode structure, offered as descriptions of what monitoring would have seen in this sample rather than as validated signals. First, the warning is a level, not a slide. The fold itself concentrates in the final quarter, too fast for a change-based alarm once filing lags are counted. But the state signal was informative: 48 percent of eventual capitulators had already left the active zone, Active Share below 70 percent, two quarters before the crossing, and 24 percent were within five points of the closet-indexing line. An allocator tracking holdings-based Active Share against the 70 line, which is computable from public filings, would historically have seen roughly half of impending folds standing in the doorway two quarters early, and the other half only as they happened. Second, the failure modes sort by observable state. A large fund in a long, shallow losing streak was historically the capitulation-risk profile; a small fund deeply underwater was the mortality-risk profile, with redemptions running about 2 percent of assets a quarter as much as two years ahead of the end. The due-diligence question differs accordingly: for the first profile, is the portfolio still meaningfully different from the index; for the second, will the fund exist in two years. Third, the performance evidence deflates both romances at once. Backing long-suffering conviction earned nothing extra on average, and firing it cost nothing: funds that kept fighting past two years underwater and funds that folded went on to perform the same, roughly zero gross alpha either way. What differed is what the fee purchased. The practical question a capitulated holding poses is therefore not performance but product: an investor holding a folded fund owns an index fund at active prices, and the performance data say there is no hidden compensation for it."));

// ------------------------------------------------------------ 10 pending
children.push(h1("10. What this paper cannot yet say"));
children.push(p(
  "Two items are pending, ranked by how much of the paper each could overturn. First, surrender versus succession: manager-level histories will show how many capitulations coincide with manager replacement. This threatens interpretation everywhere the prose gestures at psychology, though not the fund-level facts. Second, the shelved manager-versus-client comparison. Monthly gross sales and redemptions from SEC Form N-PORT are now assembled for 2019 onward, and the historical N-SAR archive that would extend them back to 1994 is being built, so that comparison is next in line rather than indefinitely deferred. None of these touches the core descriptive facts: the fatigue gradient, the depth split, the disappearance, and now the traded nature of the fold."));

// ------------------------------------------------------------ 11 conclusion
children.push(h1("11. Conclusion"));
children.push(p(
  "For one professional generation, roughly 1995 through 2009, a fund deep into a losing streak had three exits, and all three saw regular traffic. Today one of them is nearly boarded up, and before the closet-indexing wave it had barely opened. Funds that fall behind now recover or die trying, and the quiet middle path, dissolving the bet into the index while keeping the fund alive, has almost vanished from American active management. The timing of the vanishing is at least suggestive, coinciding with the flood of cheap index funds and ETFs that turned index-hugging from a survivable compromise into a product with a superior competitor at a tenth of the price. A closet indexer in 1995 was mediocre. A closet indexer in 2015 was redundant, and clients could see it. The clientele split of Section 6 says the same thing from the fund side: the sliding was always a retail-fund behavior, rare under institutional mandates that policed activeness by contract, and it ended when retail investors gained what institutions always had, a cheap way to punish it."));
children.push(p(
  "The reliability engineer's summary is this. The failure imposed from outside, death, arrives at the same rate it always did. The failure that requires a decision inside the fund, surrender, has nearly stopped occurring. Funds did not stop breaking because breaking became impossible. One way of breaking stopped being chosen, or stopped being permitted, and telling those two apart is the next paper. As of the middle of 2026, on data the original panel never saw, the boarded-up exit remains boarded up."));

children.push(new Paragraph({ children: [new PageBreak()] }));

// ------------------------------------------------------------ appendix A
children.push(h1("Appendix A. The robustness battery"));
children.push(p(
  "The design was reviewed adversarially before drafting, generating twenty-three critiques; the answerable ones became the tests below. Each row names the concern, the test, the headline claim it protects, and the outcome."));
children.push(table(
  [3000, 3100, 1900, 1800],
  [
    ["Concern", "Test", "Protects", "Outcome"],
    ["Fee drag manufactures spells", "Gross returns; fee buffer; fee-sized share", "All", "Intact; <8% fee-sized"],
    ["Spell definition arbitrary", "Entry buffers; 8q window; depth filters", "All", "Intact"],
    ["Incubation/backfill bias", "Drop funds' first 2–3 years", "Fatigue", "Intact"],
    ["Left truncation", "Drop in-progress spells (17%)", "Fatigue", "Intact"],
    ["Death-window look-ahead", "Windows 0/1/2/4 quarters", "Depth-kills", "Holds at 1q+"],
    ["Depth-event simultaneity", "Depth lagged two quarters", "Fatigue, depth", "Intact"],
    ["Cleaning rules bite by era", "Dropped-count tables; corrected TNA rule", "Disappearance", "Unchanged"],
    ["Unobserved heterogeneity", "First/repeat spells; random intercepts; split sample", "Fatigue", "Intact"],
    ["Benchmark menu grows", "Frozen 4-index menu, rebuild", "Disappearance", "Intact (HR 0.25)"],
    ["Benchmark reassignment (round 3)", "Frozen-only event definition, well-tracked spells", "Disappearance", "Intact (5.1/2.3/0.6%)"],
    ["Benchmark reassignment", "Freeze at entry; classify all crossings", "Disappearance", "Levels, not trend"],
    ["Fixed thresholds vs cap size", "Peer-relative terciles", "Disappearance", "Intact"],
    ["Reporting frequency 2004", "Force semiannual everywhere", "Disappearance", "Intact"],
    ["Launch-cohort composition", "Cohort fixed effects", "Disappearance", "HR 0.22→0.25"],
    ["Mergers pollute death", "Liquidation-only; distressed split", "Death-flat", "Sharpens"],
    ["Benchmark returns wrong", "Official series; 3-way calibration", "Disappearance", "Strengthens"],
    ["AS idiosyncratic", "Corr 0.98 vs published Petajisto", "All", "Validated"],
    ["Pipeline errors", "Six famous-fund traces + negative control", "All", "Match record"],
    ["Section 8 underpowered", "MDE; gross; own-benchmark; matched milestones; rf fix", "Folding result", "Powered null"],
    ["Capacity story explains it", "Inflow/shrinkage horse race", "Interpretation", "Pressure wins"],
    ["AS drop is passive drift", "No-trade counterfactual decomposition", "Capitulation construct", "Trading = ~100%"],
    ["Recovery censoring informative", "Three-state CIFs; multinomial competing risks", "Table 1 levels", "Levels recast; hazards intact"],
    ["Reassignment tail in definition", "Drop crossings still active vs entry benchmark", "Capitulation construct", "Intact (HR 0.20)"],
    ["Gap-bridging convention", "Censor at gaps; calendar-true windows", "Disappearance, fatigue", "Intact (HR 0.36\u20130.37); depth attenuates"],
    ["Code errors survive review", "Two adversarial line-by-line audits + reruns", "All", "2nd-decimal moves; depth halved; death +1pp"],
    ["Distance-to-threshold confound", "Entry-AS control; era HR within entry bands", "Disappearance, depth", "Era intact (0.18\u20130.42); depth claim retired"],
    ["Threshold vs behavior", "Within-spell \u0394AS distributions, threshold-free", "Disappearance", "Behavior collapsed in step"],
    ["Duration profile shifts by era", "Duration \u00d7 era interaction", "Fatigue, era", "Attenuated (0.64), present"],
    ["Managers game the metric at thresholds", "AS density at 60/70/80, pre/post 2009", "Event definition", "No bunching"],
    ["Extension instrument differs from ND", "Seam r 0.97/0.93; placebos; seam-quarter events excluded", "Extension (6.1)", "Validated"],
  ]));
children.push(caption(
  "Table A1. Summary of the robustness battery. Full specifications and numerical results are in the replication package."));

children.push(h1("Appendix B. Definitions and cleaning rules"));
children.push(p(
  "Spell entry: trailing four-quarter fund return net of fees below the trailing four-quarter benchmark return, with Active Share at or above 70 percent. Recovery: trailing relative return turns non-negative. Capitulation: minimum Active Share closes below 60 percent. Death: all share classes terminated within four quarters of the spell's end; sensitivity at zero, one, and two quarters in Appendix A. Outcome precedence when endings coincide or follow closely: recovery is recorded if trailing relative performance turned non-negative before the fund stopped reporting; a capitulation crossing overrides other codings of the same spell; death is attributed only to spells that ended by censoring, so a fund that recovers and later dies contributes a recovered spell. Depth: worst trailing four-quarter relative return so far in the spell, lagged one quarter. Benchmark: the index with lowest Active Share, per the Notre Dame convention, with official index returns as described in Section 3. Return hygiene: fund-months with absolute returns above 200 percent are dropped as data errors, as are months with observed assets under one million dollars. An earlier version of that rule treated missing assets as zero and silently discarded almost half of pre-1990 fund-months, since early CRSP reports assets quarterly; a robustness test caught it, the corrected rule (missing is not zero) changed no reported number to the second decimal, and the episode is why cleaning rules appear in this appendix as a table rather than a footnote. Flows: imputed from assets and returns at the share-class level, aggregated over retail classes, with total-fund flows as an alternative. Reporting gaps and the clock: quarters missing from the Active Share panel are bridged, so every duration in this paper counts observed quarters, and every event is dated at its true calendar quarter (the code audit of Section 3.1 found and fixed downstream conversions that assumed the two clocks agree; they diverge for a third of spells, and one spell bridges a hole of 110 quarters). The bracketing conventions were run in full: censoring every spell at its first gap gives an era hazard ratio of 0.37, and additionally requiring four consecutive calendar quarters for the performance window gives 0.36, both at more than six standard errors, so the disappearance and the duration gradient are convention-proof, while the capitulation-depth magnitude attenuates toward insignificance under censoring and is flagged accordingly in Section 5. Benchmark approximations: before 2008, official style-index returns (growth and value variants) do not exist in the assembled series, and style-benchmarked funds are measured against their core or S&P counterpart; the mismeasurement is largest around 1999 to 2002 and moves individual spells' depths, not the era contrast, which survives on a frozen four-index menu. Calendar-time portfolios deduplicate fund-month membership; expense ratios missing after within-fund filling are set to the panel median (gross legs only). In the holdings decomposition, positions without CRSP return histories, 27 percent, mostly bonds, foreign lines, and cash instruments, are held at flat prices in the no-trade counterfactual. Processing scripts, aggregate outputs, and both code-audit reports are in the replication package; licensed source data (CRSP, Notre Dame, FTSE Russell) cannot be redistributed."));

children.push(p(
  "Extension (Section 6.1): fund-level Active Share from SEC Form N-PORT holdings, quarterly from 2023Q4 through 2026Q2. Equity positions are identified by CUSIP, with US ISINs decoded where the CUSIP field is empty; filings are deduplicated to the latest amendment per fund-quarter and, where several SEC series map to one fund, to the largest by net assets; series whose share classes map to more than one fund are dropped. Benchmark weights come from the FTSE Russell holdings file (twelve indexes) and CRSP S&P 500 constituents, taken at the last month-end on or before each filing's report date. Funds with less than 80 percent of equity value identified are excluded as international or unidentifiable. Fund and benchmark weights are each renormalized to one before computing Active Share; the alternative convention, carrying unidentified value as active weight, is a pending robustness column. S&P 500 constituent identities use CRSP header CUSIPs, a shortcut immaterial over the extension's three-year window; 2026Q2 is only partially filed at the extension's construction date and its cross-section is smaller accordingly."));

children.push(h1("References"));
children.push(p(
  "Brown, K., W. Harlow, and L. Starks, 1996, Of tournaments and temptations: An analysis of managerial incentives in the mutual fund industry, Journal of Finance 51, 85–110. Chevalier, J., and G. Ellison, 1999, Career concerns of mutual fund managers, Quarterly Journal of Economics 114, 389–432. Cremers, M., and A. Petajisto, 2009, How active is your fund manager? A new measure that predicts performance, Review of Financial Studies 22, 3329–3365. Cremers, M., M. Ferreira, P. Matos, and L. Starks, 2016, Indexing and active fund management: International evidence, Journal of Financial Economics 120, 539–560. Frazzini, A., J. Friedman, and L. Pomorski, 2016, Deactivating active share, Financial Analysts Journal 72, 14–21. Huang, J., C. Sialm, and H. Zhang, 2011, Risk shifting and mutual fund performance, Review of Financial Studies 24, 2575–2616. Kempf, A., S. Ruenzi, and T. Thiele, 2009, Employment risk, compensation incentives, and managerial risk taking: Evidence from the mutual fund industry, Journal of Financial Economics 92, 92–108. Lunde, A., A. Timmermann, and D. Blake, 1999, The hazards of mutual fund underperformance: A Cox regression analysis, Journal of Empirical Finance 6, 121–152. Lynch, A., and D. Musto, 2003, How investors interpret past fund returns, Journal of Finance 58, 2033–2058. Petajisto, A., 2013, Active share and mutual fund performance, Financial Analysts Journal 69, 73–93. Sensoy, B., 2009, Performance evaluation and self-designated benchmark indexes in the mutual fund industry, Journal of Financial Economics 92, 25–39. [All citations verified against primary sources for author, title, journal, volume, and pages for the 2009–2016 additions; the remainder re-verified at final formatting.]"));

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 22 } } } },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1620, right: 1620 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(__dirname, "capitulation_draft_v9_2.docx"), buf);
  console.log("written", buf.length, "bytes");
});
