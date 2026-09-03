# #44a-r2 -- AD feedback positive control on the Yahoo geometry

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  a development positive control: the spikes are ours, not the data's.  A reading about our injected contamination says nothing about whether natural Yahoo carries removable contamination, and never enters a Yahoo capability claim

## Verdict

- **PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED**
- secondary: **INVERTED_EFFECT_OBSERVED**
- routing: neither rate passes B1, so the pre-registered branch is PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED.  The branch's pre-registered *reason* -- that the Consumer cannot read this contamination -- is falsified and must not be adopted: the Consumer reads the injection extremely well (+0.124 and +0.247 macro event F1, monotone in dose) and responds with the opposite sign.  The oracle repair fails because there was a gain to undo, not a loss to recover
- B1 by rate: {'r01': 'EFFECT_NOT_CONFIRMED', 'r03': 'EFFECT_NOT_CONFIRMED'}
- predictive Support signals: none evaluated

## Instrument gates

- anchor (#42g-b identity eval F1): **REPRODUCED**, 48/48 pairs bitwise equal, max gap 0.0
- non-saturation pre-gate: **True** -- median natural-arm flagged share 0.1339 (min 0.0278, max 0.9838, 2/24 series ≥ 0.5); bar is < 0.30.  #44a's NOAA geometry read a median flagged share of 0.877 with 11/12 series above 0.5; that is the saturation this gate exists to exclude
- determinism: substrate **True**, model recheck **True**, work originals SHA unchanged **True**

## Three arms x two rates: eval macro event F1

| rate | spikes/series | mean mask fraction | natural | injected | injected+masked |
|---|---|---|---|---|---|
| r01 | 10–10 | 0.1845 | 0.3227 | 0.4467 | 0.3252 |
| r03 | 30–31 | 0.4795 | 0.3227 | 0.5694 | 0.3259 |

## B1: did the injection hurt, and did the oracle mask recover it?

| rate | harm (injected − natural) | recovery (masked − injected) | harmed | worst | recovered share | verdict |
|---|---|---|---|---|---|---|
| r01 | +0.124078 | -0.121523 | 14 | -0.7778 | n/a | EFFECT_NOT_CONFIRMED |
| r03 | +0.246709 | -0.243497 | 15 | -1.0000 | n/a | EFFECT_NOT_CONFIRMED |

gate: macro Δ ≥ +0.005, harmed ≤ 2/24, worst ≥ -0.020

### Dose response: **MONOTONE_INVERTED**

- harm (positive = the injection hurt): -0.124078 at the low rate, -0.246709 at the high rate.  Both are negative, i.e. the injection *helped* at both doses, and the magnitude grows with dose: sign consistent **True**, magnitude grows **True**.
- repair recovery: -0.121523 at the low rate, -0.243497 at the high rate -- the oracle mask removes the gain, and removes more of it at the higher dose.
- noise red flag: **False**.  NON_MONOTONE is the noise-dominated red flag (#44a's NOAA run); MONOTONE_INVERTED is not noise -- it is a real, dose-responsive effect running opposite to the hypothesis, and it has to be explained rather than dismissed

### What the repair does and does not undo

| rate | mean scale inflation from injection | mean scale after mask | scale left unrepaired |
|---|---|---|---|
| r01 | 1.0729 | 1.0729 | True |
| r03 | 1.1983 | 1.1983 | True |

## The inverted effect, decomposed

| rate | arm | macro F1 | precision | recall | flagged share | predicted events | true events | fit windows | std ÷ natural | forest offset |
|---|---|---|---|---|---|---|---|---|---|---|
| r01 | natural | 0.3227 | 0.2525 | 0.8347 | 0.2345 | 10.71 | 1.58 | 987 | 1.0000 | -0.50153 |
| r01 | injected | 0.4467 | 0.3904 | 0.7931 | 0.1850 | 6.62 | 1.58 | 987 | 1.0729 | -0.45968 |
| r01 | injected_masked | 0.3252 | 0.2476 | 0.8597 | 0.2426 | 11.54 | 1.58 | 805 | 1.0729 | -0.49660 |
| r03 | natural | 0.3227 | 0.2525 | 0.8347 | 0.2345 | 10.71 | 1.58 | 987 | 1.0000 | -0.50153 |
| r03 | injected | 0.5694 | 0.5504 | 0.7208 | 0.1376 | 4.00 | 1.58 | 987 | 1.1983 | -0.44362 |
| r03 | injected_masked | 0.3259 | 0.2615 | 0.8389 | 0.2469 | 12.46 | 1.58 | 514 | 1.1983 | -0.49225 |

**The threshold move, measured directly.**

| rate | mean offset_ injected | mean offset_ masked | mean shift | series where injected is stricter |
|---|---|---|---|---|
| r01 | -0.45968 | -0.49660 | +0.03692 | 23/24 |
| r03 | -0.44362 | -0.49225 | +0.04863 | 23/24 |

the forest's own offset_, compared between the injected and the masked arm.  Those two arms share standardization constants exactly, so any difference here is caused only by which windows entered the fit matrix.

**What the masked arm isolates.** the masked arm carries the SAME standardization constants as the injected arm (this repair policy computes them from the full block) yet reads like the natural arm, so scale inflation is not the driver.  The only thing that differs between the injected and the masked arm is which windows entered the fit matrix -- and therefore where IsolationForest's contamination=0.1 threshold lands

**Not a sample-size artefact.** the masked arm fits on roughly half the windows the natural arm does and still reproduces the natural arm's reading, so the effect is not driven by how many windows entered the fit

**Hypothesis (MECHANISTIC_HYPOTHESIS_SUPPORTED_BY_THREE_ARMS_TWO_DOSES).** contamination=0.1 declares that a tenth of the training windows are anomalies.  On a training block that carries almost none, the frozen threshold is set among ordinary windows, and the Consumer over-alarms on the Query (natural arm: 13% of the eval region flagged, 10.7 predicted events against 1.6 true ones).  Injecting genuine outliers gives that budget something to spend on and moves the threshold outward; masking them out again moves it back

**What this is not.** this is not evidence that contaminated training data is better data.  It is evidence that on this Consumer any training-side operation that changes the training outlier rate acts first as a threshold knob

## Cross-check against #43 M0-C (CONSISTENT)

| operation | direction on training outliers | macro eval Δ |
|---|---|---|
| M0-C outlier_iqr | removes | -0.029757 |
| M0-C outlier_mad | removes | -0.060229 |
| M0-C hampel_filter | removes | -0.057184 |
| M0-C winsorize | removes | -0.091949 |
| #44a-r2 injection r01 | adds | +0.124078 |
| #44a-r2 injection r03 | adds | +0.246709 |

M0-C's five programs all REMOVE outliers from the training block and all read negative on this Consumer; this book ADDS outliers and reads positive, monotonically in dose.  Same Consumer, same roster, same split, same metric, opposite operation, opposite sign -- which is what the threshold-knob hypothesis predicts

*Caution.* consistency is not proof: M0-C's programs also change the data in ways an injection does not, so this is a converging reading, not an attribution

## Per-series eval event F1

### rate r01

| series | natural | injected | masked | masked − injected | injected − natural |
|---|---|---|---|---|---|
| real_1.csv | 0.0952 | 0.1333 | 0.1000 | -0.0333 | +0.0381 |
| real_10.csv | 0.2500 | 0.6667 | 0.2000 | -0.4667 | +0.4167 |
| real_11.csv | 1.0000 | 1.0000 | 0.5000 | -0.5000 | +0.0000 |
| real_12.csv | 0.3333 | 1.0000 | 0.2222 | -0.7778 | +0.6667 |
| real_13.csv | 0.2222 | 0.4000 | 0.2222 | -0.1778 | +0.1778 |
| real_14.csv | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| real_15.csv | 0.2000 | 0.4000 | 0.2857 | -0.1143 | +0.2000 |
| real_16.csv | 0.1333 | 0.1053 | 0.1333 | +0.0281 | -0.0281 |
| real_17.csv | 0.5714 | 1.0000 | 1.0000 | +0.0000 | +0.4286 |
| real_18.csv | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| real_19.csv | 0.5714 | 1.0000 | 0.4444 | -0.5556 | +0.4286 |
| real_2.csv | 0.1818 | 0.2857 | 0.2222 | -0.0635 | +0.1039 |
| real_20.csv | 0.2857 | 0.0000 | 0.5714 | +0.5714 | -0.2857 |
| real_21.csv | 0.0976 | 0.1053 | 0.0889 | -0.0164 | +0.0077 |
| real_22.csv | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| real_23.csv | 0.5000 | 0.5000 | 0.5000 | +0.0000 | +0.0000 |
| real_24.csv | 0.1111 | 0.2222 | 0.1429 | -0.0794 | +0.1111 |
| real_25.csv | 0.6667 | 0.6667 | 0.6667 | +0.0000 | +0.0000 |
| real_26.csv | 0.1818 | 0.2857 | 0.3636 | +0.0779 | +0.1039 |
| real_27.csv | 0.1081 | 0.1667 | 0.0488 | -0.1179 | +0.0586 |
| real_28.csv | 0.8000 | 0.5000 | 0.6667 | +0.1667 | -0.3000 |
| real_29.csv | 0.0870 | 0.1176 | 0.0833 | -0.0343 | +0.0307 |
| real_3.csv | 0.2222 | 0.6667 | 0.1429 | -0.5238 | +0.4444 |
| real_30.csv | 0.1250 | 0.5000 | 0.2000 | -0.3000 | +0.3750 |

### rate r03

| series | natural | injected | masked | masked − injected | injected − natural |
|---|---|---|---|---|---|
| real_1.csv | 0.0952 | 0.1053 | 0.0870 | -0.0183 | +0.0100 |
| real_10.csv | 0.2500 | 1.0000 | 0.1333 | -0.8667 | +0.7500 |
| real_11.csv | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| real_12.csv | 0.3333 | 0.0000 | 0.3333 | +0.3333 | -0.3333 |
| real_13.csv | 0.2222 | 1.0000 | 0.3333 | -0.6667 | +0.7778 |
| real_14.csv | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| real_15.csv | 0.2000 | 1.0000 | 0.1667 | -0.8333 | +0.8000 |
| real_16.csv | 0.1333 | 0.2857 | 0.0952 | -0.1905 | +0.1524 |
| real_17.csv | 0.5714 | 1.0000 | 1.0000 | +0.0000 | +0.4286 |
| real_18.csv | 0.0000 | 1.0000 | 0.0000 | -1.0000 | +1.0000 |
| real_19.csv | 0.5714 | 0.8000 | 0.6667 | -0.1333 | +0.2286 |
| real_2.csv | 0.1818 | 0.5714 | 0.1667 | -0.4048 | +0.3896 |
| real_20.csv | 0.2857 | 0.0000 | 0.0000 | +0.0000 | -0.2857 |
| real_21.csv | 0.0976 | 0.1538 | 0.0645 | -0.0893 | +0.0563 |
| real_22.csv | 1.0000 | 1.0000 | 1.0000 | +0.0000 | +0.0000 |
| real_23.csv | 0.5000 | 0.6667 | 0.3333 | -0.3333 | +0.1667 |
| real_24.csv | 0.1111 | 0.3333 | 0.1818 | -0.1515 | +0.2222 |
| real_25.csv | 0.6667 | 1.0000 | 0.6667 | -0.3333 | +0.3333 |
| real_26.csv | 0.1818 | 0.6154 | 0.2963 | -0.3191 | +0.4336 |
| real_27.csv | 0.1081 | 0.0000 | 0.1176 | +0.1176 | -0.1081 |
| real_28.csv | 0.8000 | 0.1333 | 0.8000 | +0.6667 | -0.6667 |
| real_29.csv | 0.0870 | 0.0000 | 0.0952 | +0.0952 | -0.0870 |
| real_3.csv | 0.2222 | 1.0000 | 0.2000 | -0.8000 | +0.7778 |
| real_30.csv | 0.1250 | 1.0000 | 0.0833 | -0.9167 | +0.8750 |

## B2: not reached

B1 confirmed no effect at either rate, so the Support signal stream was not run.

## Budget

- LLM: 0; AD fits: 130 / 180
- by arm: natural=24, injected=48, injected_masked=48, determinism_recheck=10

## Obligation self-report

- anchor_reproduction: REPRODUCED
- eval_region_bytes_injected: 0
- fit_budget_cap: 180
- fit_budget_respected: True
- fit_budget_used: 130
- fits_by_arm: {'natural': 24, 'injected': 48, 'injected_masked': 48, 'determinism_recheck': 10}
- llm_calls: 0
- mask_policy_default_cap_exceeded: True
- mask_policy_deviation_recorded: True
- model_determinism: True
- noaa_nab_smd_beyond_17520_reads: 0
- non_saturation_pre_gate: True
- rate_cherry_picking: False
- rates_reported: ['r01', 'r03']
- substrate_determinism: True
- work_originals_untouched: True
- yahoo_exposed_24_reads: 24
- yahoo_sealed_41_reads: 0

---

*Everything above this line is generated by the runner from the artifact.
Everything below it is the executor's written reading, appended by hand.*

## Executor's reading

### The short version

Moving the exam onto the Yahoo geometry worked exactly as the mainline
intended: the judge is non-saturated (median 13% of the eval region flagged
against #44a's 88%), the anchor reproduces bitwise, and the effect is now
enormous and perfectly readable.

It also runs the wrong way.  Injecting 6×MAD point spikes into the training
substrate **raised** macro eval event F1 from 0.3227 to 0.4467 at 1% and to
0.5694 at 3%.  The oracle mask — removing exactly those known positions from
the fit — put it back to 0.3252 and 0.3259, i.e. back to the natural arm.

So B1 fails at both rates, but not for the pre-registered reason.  The book's
branch text said this outcome would mean "the Consumer still cannot read this
contamination".  That is falsified and I am not adopting it: the Consumer
reads the contamination superbly.  It just responds to it as an improvement.
The repair "fails" because there was a gain to undo, not a loss to recover.

I kept the pre-registered label `PROGRAM_CONSUMER_LAYER_FAULT_CONFIRMED`
because that is what "neither rate passes B1" was defined to produce, and
added the residual classification `INVERTED_EFFECT_OBSERVED` for the part no
pre-registered gate was built to receive.

### Why it runs the wrong way — measured, not inferred

The natural arm over-alarms badly: it flags 23% of the eval region and
predicts **10.7 events per series against 1.6 true ones**.  Precision is
0.2525 while recall is already 0.8347.  On this cohort the Consumer's problem
was never sensitivity; it was that it cries wolf about nine times out of ten.

`IsolationForest(contamination=0.1)` sets its decision threshold at fit time
as a quantile of the *training* score distribution — it is told that a tenth
of the training windows are anomalies.  On a training block that carries
almost none, that tenth is taken out of ordinary windows, so the threshold
lands among normal behaviour and the Query gets shredded.

The three arms let this be measured rather than argued, because the masked
arm and the injected arm share standardization constants **exactly** (the
in-service mask policy computes constants from the full block, which #44a
flagged as an incompleteness and which turns out to be exactly the control I
needed here).  The only difference between them is which windows entered the
fit matrix.  The forest's own `offset_`:

| arm | r01 | r03 |
|---|---|---|
| natural | −0.50153 | −0.50153 |
| injected | −0.45968 | −0.44362 |
| injected + oracle mask | −0.49660 | −0.49225 |

The injected arm's threshold is stricter than the masked arm's on **23 of 24
series**, the shift grows with dose (+0.0369 → +0.0486), and masking returns
it to within 0.005 of the natural arm.  Scale inflation is not the driver:
the masked arm carries the injected arm's inflated std (1.073x / 1.198x) and
still reads like the natural arm.  Sample size is not the driver either: the
masked arm fits on roughly half the windows (514 vs 987 at r03) and still
reproduces the natural arm.

**On this Consumer, a training-side operation that changes the training
outlier rate acts first as a threshold knob, and only second as anything to
do with data quality.**

### What this does to the M0-C reading

M0-C's five programs — iqr, mad, hampel, winsorize — all *remove* outliers
from the training block, and all four read negative on this same Consumer,
same roster, same split, same metric.  This book *adds* outliers and reads
positive, monotonically in dose.  The signs are exactly what the threshold
mechanism predicts, and the artifact carries the cross-check.

I want to be careful about how far that goes:

- it does **not** invalidate M0-C's numbers.  They are correct
  measurements and they reproduce bitwise;
- it does raise a serious question about what they *measure*.  "All five
  cleaning programs are harmful to this Consumer" and "all five cleaning
  programs move a miscalibrated threshold in the harmful direction" are
  observationally identical on this evidence, and only the second one is
  about data quality;
- consistency is not attribution.  M0-C's programs change the data in ways an
  injection does not, and I did not decompose their effect into a threshold
  part and a data part.  That decomposition is the obvious next probe and it
  is not mine to authorise.

This is decision-relevant for the mainline's stop-probe ruling, which rests
on "identity is the ready-made correct answer for this cohort × menu".  That
may still be true, but the evidence for it now has a competing explanation
that was not on the table when it was ruled.  I report; I do not overturn.

### Why this is not "contamination is good data"

The injected arm finds *fewer* true events (recall 0.8347 → 0.7208 at r03).
It scores better because it stops producing nine false events for every true
one.  Nothing here says corrupted training data is better training data; it
says the frozen `contamination=0.1` is wrong for this cohort by enough that
it dominates everything else the training substrate does.  A book that wanted
the detector to actually improve would change the threshold honestly, not
inject garbage to move it sideways.

### B2

Not reached, by rule.  No Support signal is reported predictive or
unpredictive — the round does not know, and the Support machinery was built
and left unexercised.

## Out-of-book findings (reported, not fixed)

1. **The in-service AD Consumer is miscalibrated on the Yahoo EXPOSED 24 in
   a way that is measurable and large.**  10.7 predicted events per 1.6 true
   events, precision 0.25 at recall 0.83.  Every AD utility reading this line
   has taken on this cohort — #42g-b, #42h, #42j, M0-C — was taken through
   that miscalibration.  This is a bigger scope condition than #44a's
   saturation finding, because it applies to the *Yahoo* geometry, i.e. to
   the readings the line actually uses.

2. **The in-service mask fit policy does not repair the standardization.**
   `contamination_mask_refit_v1` computes constants from the full block and
   filters only the fit matrix, so scale inflation from contamination
   survives the repair.  Here that turned out to be a useful control; in
   service it means the policy's repair is partial by construction.  #44a
   raised this from the other direction and it is now measured: std ratio
   after masking is identical to before, to machine precision, at both rates.

3. **`contamination` is an unaudited free parameter with more leverage than
   the entire five-program menu.**  The menu's largest macro effect in M0-C
   was −0.092 (winsorize).  Moving the effective threshold by injecting
   outliers moved macro F1 by +0.247.  If a knob nobody is studying is three
   times more powerful than the thing being studied, the readings about the
   thing being studied are hard to interpret.  Not touched here — it is
   frozen, and unfreezing it is a mainline decision.

4. **The mask fraction ran to 48% of training windows at r03**, far past the
   fit policy's 1% default cap.  The book authorised widening it and the
   artifact records the deviation, but it is worth noting that at 3% point
   contamination with a 20-point window, nearly half the windows are touched
   — window-level masking is a very blunt repair at any realistic point
   contamination rate.  A point-level repair would be needed to test point
   contamination properly, and that is a different primitive than the one in
   service.

5. **Two of the 24 series are saturated even on this geometry**
   (`real_1.csv` and `real_28.csv`, flagged share up to 0.984).  The median
   passes the pre-gate comfortably, but the per-series spread is 0.028 to
   0.984, so per-series readings on the saturated ones carry the #44a
   pathology inside an otherwise healthy cohort.  Flagged, not excluded — the
   roster is frozen.

## Executor's obligation self-report (beyond the machine section)

- Executions: four full runs of 130 fits each.  The first produced the
  readings; the second added the dose-shape classification, the inverted-
  effect decomposition and the M0-C cross-check; the third fixed two
  presentation defects (a meaningless cross-series mean of raw standard
  deviations, and a dose-response line that printed "monotone: False" for
  what is actually a monotone inverted response); the fourth added the direct
  `offset_` measurement that turned the mechanism from a hypothesis into a
  measurement.  Every run produced identical readings — the macro figures are
  unchanged across all four.  Only the last artifact is delivered.  The
  budget cap is per run; 130 of 180.
- B1's pre-registered gate was applied exactly as written and reported
  exactly as it fell.  The pre-registered *narrative* for the failing branch
  was falsified by the data and is explicitly rejected in the artifact rather
  than quietly reused.
- Both rates reported in full; no rate dropped, none added.
- The eval region carries zero injected bytes and zero processing: no
  injected index is at or after the cut, and the array handed to the Consumer
  for eval scoring is the untouched raw series in all three arms, so the
  scored bytes and their trailing windows are bit-identical across arms.
- The 24 work CSVs were hashed before and after the run and are unchanged.
  All injected copies live under `_scratch/yahoo_positive_control/
  yahoo_m44a_r2_v1/`, with guards refusing to write anywhere under `data/`.
- The repair reuses the in-service mask fit policy's mechanics; both
  deviations (oracle window selection instead of first-forest scores, and the
  mask fraction past the 1% default cap) are recorded in the artifact.
- Zero LLM calls, zero network access, zero reads of the 41 sealed Yahoo
  series, zero new NOAA / NAB / SMD / beyond_17520 reads.
- This report is **not** committed, per Part C.  The runner is committed.
- Part 0: the #44a artifacts and the mainline's C33 ledger increment to
  `STAGE_REPORT_BATCH_RECIPE_LINE_2026-08-21.md` were committed as `ae8821b`.
  `AGENTS.md`, `README.md`, `PROJECT_STATE_AND_DATA_MAP_2026-08-23.md` and
  `SUCCESSOR_BRIEF_2026-08-22.md` carry other lines' in-flight edits and were
  left untouched and uncommitted.

