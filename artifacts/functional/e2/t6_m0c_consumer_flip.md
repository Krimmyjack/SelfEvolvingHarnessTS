# #43 M0-C -- three-Consumer processing-utility response matrix

evidence class: MECHANISM (development, EXPOSED 24). 41 sealed series unread.

## Verdicts

- C1 **CONSUMER_UTILITY_FLIP_NOT_CONFIRMED** (flipped programs: none)
- C2 **RECONSTRUCTION_HEADROOM_NOT_QUALIFIED** (qualified programs: none)
- side flag **SUPERVISED_EVIDENCE_PRESERVATION_NOT_CONFIRMED** (programs: none)

Claim scope: processing utility responds to the Consumer protocol; the three protocols differ in fit protocol as well as model structure, so this is NOT a 'only the model structure changed' reading, and the C-c arm speaks only for the deterministic low-rank reconstruction family (no AE / TimesNet extrapolation)

## Aggregate by Consumer

### C-a IForest (per-series, unlabeled, w20)

| program | macro F1 | macro Δ | macro recall Δ | harmed | improved | worst |
|---|---|---|---|---|---|---|
| identity | 0.322665 | +0.000000 | +0.000000 | 0 | 0 | +0.000000 |
| outlier_iqr | 0.292908 | -0.029757 | -0.041667 | 5 | 6 | -0.333333 |
| outlier_mad | 0.262436 | -0.060229 | -0.025000 | 7 | 5 | -0.666667 |
| hampel_filter | 0.265481 | -0.057184 | -0.020833 | 12 | 7 | -0.750000 |
| winsorize | 0.230716 | -0.091949 | -0.133333 | 14 | 4 | -0.714286 |

### C-b supervised v3 (pooled, labelled, w49)

| program | macro F1 | macro Δ | macro recall Δ | harmed | improved | worst |
|---|---|---|---|---|---|---|
| identity | 0.199149 | +0.000000 | +0.000000 | 0 | 0 | +0.000000 |
| outlier_iqr | 0.185305 | -0.013844 | +0.000000 | 15 | 1 | -0.126984 |
| outlier_mad | 0.187429 | -0.011721 | +0.000000 | 11 | 0 | -0.126984 |
| hampel_filter | 0.196089 | -0.003060 | +0.000000 | 5 | 0 | -0.019608 |
| winsorize | 0.123160 | -0.075989 | +0.008333 | 21 | 1 | -0.800000 |

### C-c PCA rank-3 (per-series, unlabeled, w20)

| program | macro F1 | macro Δ | macro recall Δ | harmed | improved | worst |
|---|---|---|---|---|---|---|
| identity | 0.378300 | +0.000000 | +0.000000 | 0 | 0 | +0.000000 |
| outlier_iqr | 0.333440 | -0.044860 | +0.000000 | 9 | 1 | -0.600000 |
| outlier_mad | 0.336470 | -0.041830 | +0.000000 | 8 | 2 | -0.600000 |
| hampel_filter | 0.304967 | -0.073333 | -0.008333 | 13 | 3 | -0.666667 |
| winsorize | 0.294657 | -0.083642 | +0.000000 | 14 | 4 | -0.666667 |

## Per-program flip contrast (C1)

| program | C-a IForest macro Δ | C-c PCA macro Δ | C-b supervised macro Δ | flip |
|---|---|---|---|---|
| outlier_iqr | -0.029757 | -0.044860 | -0.013844 | no |
| outlier_mad | -0.060229 | -0.041830 | -0.011721 | no |
| hampel_filter | -0.057184 | -0.073333 | -0.003060 | no |
| winsorize | -0.091949 | -0.083642 | -0.075989 | no |

## Reconstruction safety gate (C2)

| program | macro Δ | harmed /24 | worst | qualified |
|---|---|---|---|---|
| outlier_iqr | -0.044860 | 9 | -0.600000 | no |
| outlier_mad | -0.041830 | 8 | -0.600000 | no |
| hampel_filter | -0.073333 | 13 | -0.666667 | no |
| winsorize | -0.083642 | 14 | -0.666667 | no |

## Anchor reproduction (#42g-b C-a)

- status: **REPRODUCED**
- pairs compared: 240, bitwise equal: 240, max abs gap: 0.0

## Pre-registered predictions

| arm | prediction | held |
|---|---|---|
| c_a_iforest | cleaning programs are macro-negative | YES |
| c_b_supervised | cleaning damages the positive class (recall drops) | NO |
| c_c_pca | cleaning is macro-positive (literature prior) | NO |

## Full per-series matrix

### C-a IForest (per-series, unlabeled, w20)

| series | identity F1 | outlier_iqr Δ | outlier_mad Δ | hampel_filter Δ | winsorize Δ |
|---|---|---|---|---|---|
| real_1.csv | 0.0952 | +0.0000 | +0.0100 | -0.0083 | -0.0212 |
| real_10.csv | 0.2500 | +0.0357 | -0.0278 | -0.1447 | +0.0833 |
| real_11.csv | 1.0000 | +0.0000 | -0.6667 | -0.7500 | -0.7143 |
| real_12.csv | 0.3333 | -0.3333 | -0.3333 | -0.0833 | -0.3333 |
| real_13.csv | 0.2222 | +0.0000 | +0.0000 | +0.0000 | -0.2222 |
| real_14.csv | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_15.csv | 0.2000 | +0.0000 | +0.0000 | +0.0222 | -0.2000 |
| real_16.csv | 0.1333 | +0.0095 | +0.0000 | -0.0644 | -0.0333 |
| real_17.csv | 0.5714 | +0.0000 | +0.2286 | +0.0952 | +0.0952 |
| real_18.csv | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_19.csv | 0.5714 | +0.0000 | -0.2857 | -0.0714 | -0.1270 |
| real_2.csv | 0.1818 | +0.0000 | +0.0535 | +0.0682 | -0.0280 |
| real_20.csv | 0.2857 | +0.0000 | +0.0000 | -0.2857 | +0.0476 |
| real_21.csv | 0.0976 | -0.0142 | -0.0023 | +0.0050 | +0.0024 |
| real_22.csv | 1.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_23.csv | 0.5000 | -0.1667 | -0.2143 | -0.1000 | +0.0000 |
| real_24.csv | 0.1111 | +0.0317 | +0.0317 | -0.0111 | -0.0421 |
| real_25.csv | 0.6667 | -0.2667 | -0.2667 | -0.2667 | -0.2667 |
| real_26.csv | 0.1818 | -0.0642 | +0.0791 | -0.0568 | -0.1818 |
| real_27.csv | 0.1081 | +0.0000 | +0.0000 | -0.0081 | +0.0000 |
| real_28.csv | 0.8000 | +0.0000 | +0.0000 | +0.2000 | -0.1333 |
| real_29.csv | 0.0870 | +0.0083 | +0.0040 | +0.0307 | +0.0130 |
| real_3.csv | 0.2222 | +0.0278 | -0.0556 | +0.0000 | -0.1313 |
| real_30.csv | 0.1250 | +0.0179 | +0.0000 | +0.0568 | -0.0139 |

### C-b supervised v3 (pooled, labelled, w49)

| series | identity F1 | outlier_iqr Δ | outlier_mad Δ | hampel_filter Δ | winsorize Δ |
|---|---|---|---|---|---|
| real_1.csv | 0.1111 | -0.0111 | -0.0058 | -0.0030 | -0.0327 |
| real_10.csv | 0.0435 | -0.0018 | -0.0035 | +0.0000 | -0.0057 |
| real_11.csv | 0.0571 | -0.0045 | -0.0031 | -0.0031 | -0.0171 |
| real_12.csv | 0.0690 | -0.0134 | -0.0044 | +0.0000 | -0.0177 |
| real_13.csv | 0.1250 | -0.0074 | -0.0074 | -0.0074 | -0.0341 |
| real_14.csv | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_15.csv | 0.1176 | +0.0000 | +0.0000 | +0.0000 | -0.0224 |
| real_16.csv | 0.0714 | +0.0000 | +0.0000 | -0.0025 | -0.0069 |
| real_17.csv | 0.3077 | -0.0220 | -0.0220 | +0.0000 | -0.0577 |
| real_18.csv | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_19.csv | 0.1600 | -0.0221 | +0.0000 | +0.0000 | -0.0310 |
| real_2.csv | 0.2667 | -0.0444 | -0.0167 | -0.0167 | -0.0444 |
| real_20.csv | 0.2222 | -0.0556 | +0.0000 | +0.0000 | -0.0684 |
| real_21.csv | 0.1905 | -0.0238 | -0.0238 | -0.0087 | -0.0614 |
| real_22.csv | 0.1429 | -0.0179 | -0.0095 | +0.0000 | -0.0376 |
| real_23.csv | 0.0625 | -0.0084 | -0.0084 | -0.0019 | -0.0225 |
| real_24.csv | 0.1212 | -0.0159 | -0.0131 | -0.0069 | -0.0396 |
| real_25.csv | 0.0800 | -0.0059 | -0.0031 | +0.0000 | -0.0059 |
| real_26.csv | 0.1250 | -0.0139 | -0.0139 | -0.0038 | +0.0114 |
| real_27.csv | 0.1818 | +0.0000 | +0.0000 | +0.0000 | -0.0390 |
| real_28.csv | 0.5714 | -0.1270 | -0.1270 | +0.0000 | -0.2637 |
| real_29.csv | 0.3529 | -0.0372 | -0.0196 | -0.0196 | -0.1129 |
| real_3.csv | 1.0000 | +0.0000 | +0.0000 | +0.0000 | -0.8000 |
| real_30.csv | 0.4000 | +0.1000 | +0.0000 | +0.0000 | -0.1143 |

### C-c PCA rank-3 (per-series, unlabeled, w20)

| series | identity F1 | outlier_iqr Δ | outlier_mad Δ | hampel_filter Δ | winsorize Δ |
|---|---|---|---|---|---|
| real_1.csv | 0.1667 | -0.0128 | +0.0000 | -0.0238 | +0.0152 |
| real_10.csv | 0.1667 | -0.0128 | +0.0000 | +0.0556 | +0.0556 |
| real_11.csv | 0.1818 | -0.0642 | -0.0766 | -0.0568 | -0.0766 |
| real_12.csv | 0.6667 | -0.2667 | -0.2667 | -0.3333 | -0.3810 |
| real_13.csv | 0.6667 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_14.csv | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_15.csv | 0.6667 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_16.csv | 0.1667 | +0.0000 | +0.0000 | -0.0833 | -0.0797 |
| real_17.csv | 0.5714 | +0.4286 | +0.4286 | +0.0952 | +0.0000 |
| real_18.csv | 0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_19.csv | 0.5714 | +0.0000 | -0.0714 | +0.0000 | -0.0714 |
| real_2.csv | 0.4444 | +0.0000 | -0.1587 | +0.0000 | -0.1111 |
| real_20.csv | 0.5714 | +0.0000 | +0.0000 | +0.0952 | +0.2286 |
| real_21.csv | 0.1905 | -0.0305 | +0.0000 | -0.0655 | -0.0614 |
| real_22.csv | 1.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| real_23.csv | 0.5000 | -0.2778 | -0.1000 | -0.1667 | -0.2500 |
| real_24.csv | 0.4000 | -0.1333 | -0.0667 | -0.1647 | -0.2750 |
| real_25.csv | 1.0000 | -0.6000 | -0.6000 | -0.6667 | -0.6667 |
| real_26.csv | 0.2500 | -0.1071 | -0.1071 | -0.2500 | -0.0682 |
| real_27.csv | 0.1250 | +0.0000 | +0.0000 | +0.0000 | -0.0074 |
| real_28.csv | 0.1212 | +0.0000 | +0.0000 | -0.0101 | +0.0326 |
| real_29.csv | 0.2353 | +0.0000 | +0.0147 | -0.0614 | -0.0871 |
| real_3.csv | 0.2500 | +0.0000 | +0.0000 | -0.0682 | -0.1324 |
| real_30.csv | 0.1667 | +0.0000 | +0.0000 | -0.0556 | -0.0714 |

## Frozen reconstruction parameters and their basis

- rank = 3. a column-centered 20-point window of a locally smooth or quasi-periodic series is spanned by a level/slope-residual mode plus one periodic pair; rank 3 is the smallest rank carrying that family.  Fixture-gated at explained-variance ratio >= 0.90 on a clean synthetic substrate, then frozen -- never scanned on Yahoo
- threshold quantile = 0.9. parity with the in-service IForest Consumer's contamination = 0.1: both Consumers carry the same 10% frozen training-time alarm budget, so a delta between them is not an alarm-rate artefact
- scanned on Yahoo: False
- realized explained-variance ratio over 120 fits: min 0.3062, median 0.8301, max 0.9932; 71.7% of fits sit below the 0.90 fixture floor.  reported as a diagnostic only; the rank stays 3 and is not re-picked from these numbers

## Descriptive readings (not judgments)

- per-cell sign contrast (POST_HOC_DESCRIPTIVE): 14 of 96 (series, cleaning program) cells put the IForest and the reconstruction Consumer on materially opposite sides.  C1 is judged on the pre-registered macro bar only; this per-cell count was not pre-registered, opens nothing and closes nothing

- supervised training-evidence census (POST_HOC_DESCRIPTIVE): the pre-registered side flag reads eval-side macro recall; this is the training-side positive-row census and was not pre-registered

| program | rows entering pooled fit | positive rows | Δ vs identity |
|---|---|---|---|
| identity | 22964 | 369 | +0 |
| outlier_iqr | 22735 | 187 | -182 |
| outlier_mad | 22504 | 184 | -185 |
| hampel_filter | 22964 | 369 | +0 |
| winsorize | 22868 | 302 | -67 |

## Budget

- LLM: 0; forecasting retrains: 0
- AD fits: 245 / 280 -- c_a_iforest=120, c_b_supervised=5, c_c_pca=120
- mask fit policy: not run this book; #42j artifact read only

## Obligation self-report

- anchor_reproduction: REPRODUCED
- fit_budget_cap: 280
- fit_budget_respected: True
- fit_budget_used: 245
- forecast_retrains: 0
- frozen_before_yahoo: ['pca rank', 'pca threshold quantile']
- llm_calls: 0
- mask_fit_policy_run: False
- methods_package_touched: False
- noaa_nab_smd_beyond_17520_reads: 0
- preregistered_predictions_missed: ['c_b_supervised', 'c_c_pca']
- preregistered_predictions_scored: 3
- rescanned_on_yahoo: False
- roster_size: 24
- sealed_series_note: the roster is the first 24 files of the frozen Yahoo list in lexicographic order; the remaining 41 are never opened here
- sealed_series_read: 0

## Menu

menu: ['identity', 'outlier_iqr', 'outlier_mad', 'hampel_filter', 'winsorize'] (cleaning: ['outlier_iqr', 'outlier_mad', 'hampel_filter', 'winsorize'])

---

*Everything above this line is generated by the runner from the artifact.
Everything below it is the executor's written reading, appended by hand.*

## Executor's reading

### What the matrix says

Twelve cleaning readings were taken (three Consumers x four cleaning
programs).  All twelve are macro-negative.  C1 therefore fails for a reason
worth stating precisely: it is **not** that the IForest and the
reconstruction Consumer disagreed in magnitude but agreed in sign near the
bar -- it is that **no Consumer in the matrix showed positive utility for any
cleaning program at all**.  The flip test needs one side at or above +0.005;
the most positive macro delta anywhere in the matrix is -0.00306
(hampel_filter on the supervised arm).  There was nothing for the other side
to flip against.

C2 fails for the same underlying reason and independently of C1, as the book
requires: the reconstruction arm's best program (outlier_mad, macro -0.0418,
8/24 harmed, worst -0.60) misses all three safety conditions, not just one.

The pre-registered predictions are scored honestly: the IForest prediction
held, and **both** of the other two failed.  The literature prior that a
reconstruction detector benefits from a cleaned training substrate is
falsified on this roster, under this frozen reconstruction specification.
The supervised prediction failed in its pre-registered form (eval-side macro
recall did not drop; three of four programs moved it by exactly 0.0).

### Why this is a credible negative rather than an unreadable instrument

- The anchor obligation was met at the strongest possible level: all 240
  C-a (series x program x {eval_f1, eval_delta}) pairs reproduce the landed
  #42g-b artifact **bitwise**, max absolute gap 0.0.  The C-a arm is
  literally the same measurement the earlier book took.
- The full 245-fit matrix was executed three times end to end.  All shared
  artifact sections are byte-identical across executions, so the reading is
  deterministic on natural data, not only on the fixture.  Only one
  measurement's worth of evidence is claimed; the repeats are repeat
  observations, not independent evidence.
- No arm is degenerate.  Every Consumer produces predicted events on every
  series, identity macro F1 is 0.3227 (C-a), 0.1991 (C-b) and 0.3783 (C-c),
  and per-series F1 spans the full range including exact 1.0 and exact 0.0
  cells.
- The three arms disagree strongly at the per-series level -- 14 of 96
  (series, cleaning program) cells put the IForest and the reconstruction
  Consumer on materially opposite sides -- so the instrument is not simply
  reporting one Consumer three times.  That count is post-hoc and descriptive
  and decides nothing here.

### What this closes and what it does not

Closes: the specific hypothesis that swapping the in-service IForest for a
deterministic low-rank reconstruction Consumer turns the five-program menu's
utility positive on the Yahoo EXPOSED 24.  It does not.

Does not close: M0 itself.  The claim under test was "processing utility
responds to the Consumer protocol"; this round found agreement, not
response, across the three protocols it could afford.  Nothing here licenses
"processing utility is Consumer-invariant" -- three protocols on one roster
with one frozen menu is not the space.  It also does not close the
reconstruction family: this Consumer is one frozen point in the
deterministic low-rank reconstruction family, and no reading may be carried
to autoencoders, TimesNet or any learned/iterative reconstruction model.

### First fault, in the AGENTS.md ladder

`no readable positive effect -> Consumer / evaluator / training protocol`.
The menu, not the Observation or the selection layer, is where this round
runs out: three independently built Consumers with three different inductive
biases and three different fit protocols all read the same five programs as
harmful.  A fourth Consumer is the cheaper next probe than a sixth program,
and #42j already closed adding programs to fit these 24.

## Out-of-book findings (reported, not fixed)

1. **The clipping programs destroy labelled positive evidence on the
   training side, even though eval-side recall did not move.**  The pooled
   supervised fit keeps 369 positive rows under identity, but only 187 under
   `outlier_iqr` and 184 under `outlier_mad` -- roughly half the positive
   training evidence stops entering the fit.  `hampel_filter` loses none and
   `winsorize` loses 67.  The mechanism is visible in the v3 abstention
   accounting: clipping flattens the neighbourhood around an anomaly, the
   trailing robust-z becomes undefined (zero-scale), and the row is excluded
   rather than mislabelled.  The book's pre-registered side flag reads
   eval-side recall and correctly did not fire; this training-side census was
   not pre-registered and is reported as an observation only.  It is the
   sharpest "cleaning damages evidence" signal in the round and deserves its
   own pre-registration if the supervised arm is used again.

2. **Rank 3 is a stricter low-rank assumption on natural data than the
   fixture implied.**  The fixture gate required the top-3 explained-variance
   ratio to clear 0.90 on a clean synthetic substrate, and it did.  On the
   120 Yahoo fits the realized ratio has median 0.8301 and minimum 0.3062,
   with 71.7% of fits below the fixture floor.  The rank was **not**
   re-picked -- the book forbids scanning it on Yahoo and it stays 3 -- but
   this means the reconstruction residual on natural data carries
   substantially more ordinary structure than on the fixture, and the C-c
   readings should be read as "rank-3 residual detector", not as "the
   reconstruction subspace of these series".

3. **The pooled supervised protocol has a roster-size dependency the
   per-series protocols do not have.**  The 4-series smoke was reached only
   after a 2-series attempt stopped: the first two roster entries carry zero
   held-in anomaly points, so the pooled fit had one class and could not be
   built.  This is now a clean `INSTRUMENT_UNREADABLE` stop rather than a
   bare traceback.  It is a real protocol asymmetry inside what the book
   calls the model axis, and it is one more reason the claim wording must
   stay "Consumer protocol", never "only the model structure changed".

4. **Two of the 24 series carry no eval-region event at all** (`real_14.csv`,
   `real_18.csv`).  Under the frozen event-F1 edge rule they score 0.0 for
   every Consumer and every program, so their delta is structurally 0.0 in
   all 12 readings.  They dilute every macro by 1/12 without biasing any
   sign, and they affect all three arms identically.  Not fixed; noted so
   that the macro magnitudes are read for what they are.

5. **The supervised arm is the weakest baseline despite being the only arm
   that sees labels** (identity macro F1 0.1991 vs 0.3227 and 0.3783).  Its
   pooled positive-class weight runs 61x-121x, and it predicts many more
   events per series than the unlabeled arms on most series.  Not
   investigated here.

## Executor's obligation self-report (beyond the machine section)

- Executions this book: one 4-series smoke (45 fits) and three full 24-series
  runs (245 fits each).  Only the last full run's artifact is delivered; the
  first two were the same measurement and are cited only as the determinism
  check described above.  A 2-series smoke attempt stopped before spending
  its supervised fit.
- Pre-registered predictions: 3 registered, 3 scored, 2 missed.  Both misses
  are reported above without reinterpretation of the prediction.
- Frozen-before-Yahoo parameters: PCA rank and threshold quantile.  Neither
  was re-picked after seeing Yahoo numbers.  The fixture gate passed on the
  first configuration proposed, so no "fixture retry" happened either.
- Forbidden-edit list: `runtime/public_features.py`,
  `tests/.../test_public_feature_calibration.py`, `run_e2_m0a_*` and the
  other line's untracked `run_t233_supply_obs_ab.py` were not opened for
  writing.  `methods/` has zero changes.  Delivered code is three paths:
  the new Consumer module, its test, and the new runner entry.
- The parallel line's tracked modifications to `AGENTS.md`, `README.md`,
  `docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md` and
  `docs/SUCCESSOR_BRIEF_2026-08-22.md` were present at start and were left
  untouched and uncommitted by this book.
- This report is **not** committed, per the book.  The code was committed as
  `79ef5c0`.
- Zero LLM calls, zero network access, zero reads of the 41 sealed series,
  and zero reads of NOAA / NAB / SMD / beyond_17520.

