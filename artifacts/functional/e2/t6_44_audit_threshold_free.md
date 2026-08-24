# #44-audit -- threshold-free reread of M0-C IForest and PCA

evidence class: INSTRUMENT / EVIDENCE_INTEGRITY (development).  a development evidence-integrity audit of the M0-C IForest and PCA event-F1 negatives.  It decides whether those negatives are a threshold artefact or data harm.  It is not a headroom probe, does not retune contamination, and never enters a Yahoo capability claim

## Verdict

- **c_a_iforest**: **DATA_HARM_CONFIRMED**
  - at least one program has AUPRC macro Δ ≤ −0.005 and harmed series > 2/24
  - per-program: {'outlier_iqr': 'HARM', 'outlier_mad': 'HARM', 'hampel_filter': 'HARM', 'winsorize': 'HARM'}
- **c_c_pca**: **MIXED**
  - some programs sit in the artefact band, some meet the DATA_HARM bar; labelled per program
  - per-program: {'outlier_iqr': 'WEAK_NEGATIVE', 'outlier_mad': 'ARTIFACT_BAND', 'hampel_filter': 'HARM', 'winsorize': 'ARTIFACT_BAND'}

## Instrument gates

- companion event-F1 vs landed M0-C: **REPRODUCED**, 480/480 pairs bitwise equal, max gap 0.0
- two-run numeric fingerprint: **BITWISE_IDENTICAL**
- work originals SHA unchanged: **True**
- identity AUPRC identity: **NEW_ANCHOR** (no prior AUPRC anchor on this roster)

## New identity AUPRC anchors

| Consumer | identity | macro AUPRC | n scored / event-bearing | zero-event excluded |
|---|---|---|---|---|
| c_a_iforest | NEW_ANCHOR | 0.451323 | 22 / 22 | real_14.csv, real_18.csv |
| c_c_pca | NEW_ANCHOR | 0.576031 | 22 / 22 | real_14.csv, real_18.csv |

## AUPRC macro (event-bearing series only; zero-event excluded)

### c_a_iforest

| program | macro AUPRC | macro Δ | harmed /24 | improved | worst | label |
|---|---|---|---|---|---|---|
| identity | 0.451323 | +0.000000 | 0 | 0 | +0.0000 | identity |
| outlier_iqr | 0.415705 | -0.035618 | 11 | 2 | -0.2845 | HARM |
| outlier_mad | 0.439032 | -0.012292 | 9 | 3 | -0.5635 | HARM |
| hampel_filter | 0.419929 | -0.031394 | 11 | 4 | -0.2573 | HARM |
| winsorize | 0.429178 | -0.022145 | 9 | 7 | -0.4123 | HARM |

harmed series by program:
- outlier_iqr (HARM): real_1.csv, real_11.csv, real_12.csv, real_16.csv, real_17.csv, real_19.csv, real_2.csv, real_21.csv, real_23.csv, real_24.csv, real_25.csv
- outlier_mad (HARM): real_1.csv, real_11.csv, real_12.csv, real_17.csv, real_19.csv, real_21.csv, real_23.csv, real_25.csv, real_3.csv
- hampel_filter (HARM): real_1.csv, real_11.csv, real_12.csv, real_13.csv, real_15.csv, real_19.csv, real_2.csv, real_21.csv, real_23.csv, real_25.csv, real_26.csv
- winsorize (HARM): real_12.csv, real_13.csv, real_15.csv, real_17.csv, real_21.csv, real_23.csv, real_24.csv, real_25.csv, real_26.csv

### c_c_pca

| program | macro AUPRC | macro Δ | harmed /24 | improved | worst | label |
|---|---|---|---|---|---|---|
| identity | 0.576031 | +0.000000 | 0 | 0 | +0.0000 | identity |
| outlier_iqr | 0.570583 | -0.005449 | 1 | 2 | -0.1402 | WEAK_NEGATIVE |
| outlier_mad | 0.573739 | -0.002292 | 1 | 1 | -0.0703 | ARTIFACT_BAND |
| hampel_filter | 0.549258 | -0.026774 | 3 | 0 | -0.5447 | HARM |
| winsorize | 0.573981 | -0.002050 | 2 | 1 | -0.0529 | ARTIFACT_BAND |

harmed series by program:
- outlier_iqr (WEAK_NEGATIVE): real_26.csv
- outlier_mad (ARTIFACT_BAND): real_26.csv
- hampel_filter (HARM): real_16.csv, real_26.csv, real_30.csv
- winsorize (ARTIFACT_BAND): real_17.csv, real_26.csv

## Companion event-F1 (same fits; not a judgment input)

### c_a_iforest

| program | macro F1 | macro Δ |
|---|---|---|
| identity | 0.351998 | +0.000000 |
| outlier_iqr | 0.319536 | -0.032462 |
| outlier_mad | 0.286294 | -0.065705 |
| hampel_filter | 0.289616 | -0.062382 |
| winsorize | 0.251690 | -0.100309 |

### c_c_pca

| program | macro F1 | macro Δ |
|---|---|---|
| identity | 0.412691 | +0.000000 |
| outlier_iqr | 0.363753 | -0.048938 |
| outlier_mad | 0.367058 | -0.045632 |
| hampel_filter | 0.332691 | -0.080000 |
| winsorize | 0.321445 | -0.091246 |

## AUPRC Δ vs F1 Δ sign agreement (descriptive)

| Consumer | agree | disagree | both zero | rate | skipped |
|---|---|---|---|---|---|
| c_a_iforest | 53 | 35 | 17 | 0.602 | 0 |
| c_c_pca | 43 | 45 | 31 | 0.489 | 0 |

sign agreement is reported because the book asks for it; it does not open or close a verdict

## Per-series AUPRC

### c_a_iforest

| series | identity | iqr Δ | mad Δ | hampel Δ | winsorize Δ | events |
|---|---|---|---|---|---|---|
| real_1.csv | 0.032563 | -0.0053 | -0.0117 | -0.0053 | -0.0029 | 2 |
| real_10.csv | 1.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_11.csv | 0.864507 | -0.0288 | -0.0918 | -0.2573 | +0.0318 | 1 |
| real_12.csv | 0.071509 | -0.0541 | -0.0527 | -0.0081 | -0.0596 | 1 |
| real_13.csv | 0.187454 | +0.0000 | +0.0000 | -0.0964 | -0.1427 | 1 |
| real_14.csv | excluded (zero-event) | n/a | n/a | n/a | n/a | 0 |
| real_15.csv | 0.109239 | +0.0000 | +0.0000 | -0.0680 | -0.0805 | 1 |
| real_16.csv | 0.091756 | -0.0269 | +0.0000 | +0.1331 | +0.0373 | 1 |
| real_17.csv | 0.898688 | -0.2383 | -0.2039 | +0.0042 | -0.1025 | 2 |
| real_18.csv | excluded (zero-event) | n/a | n/a | n/a | n/a | 0 |
| real_19.csv | 0.819200 | -0.2602 | -0.5635 | -0.0152 | +0.0081 | 2 |
| real_2.csv | 0.112546 | -0.0118 | +0.7471 | -0.0125 | +0.7204 | 2 |
| real_20.csv | 0.106772 | +0.0000 | +0.0000 | +0.0113 | +0.0412 | 2 |
| real_21.csv | 0.067604 | -0.0411 | -0.0075 | -0.0318 | -0.0511 | 2 |
| real_22.csv | 1.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_23.csv | 0.348718 | -0.2845 | -0.2294 | -0.2287 | -0.2945 | 2 |
| real_24.csv | 0.901131 | -0.0383 | +0.0191 | +0.0117 | -0.1941 | 2 |
| real_25.csv | 0.981040 | -0.0547 | -0.0451 | -0.0079 | -0.0380 | 1 |
| real_26.csv | 0.704734 | +0.2471 | +0.1807 | -0.1308 | -0.4123 | 5 |
| real_27.csv | 0.014874 | +0.0000 | +0.0000 | +0.0024 | +0.0005 | 2 |
| real_28.csv | 0.005482 | +0.0000 | +0.0000 | -0.0000 | +0.0004 | 2 |
| real_29.csv | 0.014575 | +0.0001 | +0.0013 | -0.0006 | +0.0010 | 3 |
| real_3.csv | 0.963054 | +0.0040 | -0.0132 | +0.0000 | +0.0086 | 1 |
| real_30.csv | 0.633667 | +0.0091 | +0.0000 | +0.0091 | +0.0417 | 1 |

### c_c_pca

| series | identity | iqr Δ | mad Δ | hampel Δ | winsorize Δ | events |
|---|---|---|---|---|---|---|
| real_1.csv | 0.545455 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 2 |
| real_10.csv | 1.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_11.csv | 0.618482 | +0.0123 | +0.0175 | +0.0035 | +0.0123 | 1 |
| real_12.csv | 0.333333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_13.csv | 0.933333 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_14.csv | excluded (zero-event) | n/a | n/a | n/a | n/a | 0 |
| real_15.csv | 0.860000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_16.csv | 0.118867 | +0.0000 | +0.0000 | -0.0114 | +0.0000 | 1 |
| real_17.csv | 0.706997 | +0.0070 | +0.0045 | -0.0009 | -0.0051 | 2 |
| real_18.csv | excluded (zero-event) | n/a | n/a | n/a | n/a | 0 |
| real_19.csv | 0.679965 | +0.0011 | +0.0007 | -0.0001 | +0.0004 | 2 |
| real_2.csv | 0.291837 | +0.0000 | -0.0010 | +0.0000 | -0.0008 | 2 |
| real_20.csv | 0.136165 | +0.0000 | +0.0000 | +0.0000 | -0.0004 | 2 |
| real_21.csv | 0.111095 | +0.0000 | +0.0000 | +0.0025 | +0.0015 | 2 |
| real_22.csv | 1.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_23.csv | 0.148276 | +0.0000 | -0.0019 | +0.0023 | +0.0000 | 2 |
| real_24.csv | 0.995833 | +0.0000 | +0.0000 | -0.0044 | +0.0000 | 2 |
| real_25.csv | 1.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_26.csv | 0.727228 | -0.1402 | -0.0703 | -0.5447 | -0.0529 | 5 |
| real_27.csv | 0.088933 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 2 |
| real_28.csv | 0.116883 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 2 |
| real_29.csv | 0.367113 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 3 |
| real_3.csv | 1.000000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 1 |
| real_30.csv | 0.892895 | +0.0000 | +0.0000 | -0.0357 | +0.0000 | 1 |

## Zero-event eval series (listed; not in any macro)

- real_14.csv: truth_events=0; identity AUPRC {'c_a_iforest': None, 'c_c_pca': None}; identity F1 {'c_a_iforest': 0.0, 'c_c_pca': 0.0}
- real_18.csv: truth_events=0; identity AUPRC {'c_a_iforest': None, 'c_c_pca': None}; identity F1 {'c_a_iforest': 0.0, 'c_c_pca': 0.0}

## Budget

- LLM: 0; official AD fits: 240 / 280; verification fits: 240
- official by arm: c_a_iforest=120, c_c_pca=120

## Obligation self-report

- contamination_parameter_edited: False
- eval_region_bytes_processed: 0
- fit_budget_cap: 280
- fit_budget_respected: True
- fit_budget_used_official: 240
- fits_by_arm_official: {'c_a_iforest': 120, 'c_c_pca': 120}
- freeze_roster_n: 65
- gates_rewritten_after_seeing_numbers: False
- identity_auprc_anchor_identity: NEW_ANCHOR
- llm_calls: 0
- m0c_f1_reproduction: REPRODUCED
- methods_package_touched: False
- new_threshold_introduced: False
- noaa_nab_smd_beyond_17520_reads: 0
- supervised_v3_rerun: False
- two_run_bitwise: BITWISE_IDENTICAL
- verification_fits: 240
- work_originals_untouched: True
- yahoo_exposed_24_reads: 24
- yahoo_sealed_41_reads: 0
- zero_event_excluded_from_macro: True
- zero_event_series: ['real_14.csv', 'real_18.csv']

## Outside findings (reported, not repaired)

- **score_samples_vs_decision_function**: IForest AUPRC uses the existing -decision_function ranking.  That ranking is identical to -score_samples because decision_function = score_samples - offset_ is a per-model constant shift.  No new threshold was introduced to break the tie.
- **pca_point_mapping_is_window_end**: PCA maps the window residual to the window-ending point via the existing score_region alignment; this book does not invent a second mapping.
- **companion_f1_still_thresholded**: event-F1 still uses each Consumer's in-service threshold (IForest decision < 0; PCA residual > training 0.90 quantile).  That is the old-calibre companion, taken from the same fit, and is not an input to the verdict.
- **contamination_left_untouched**: aegists_iforest_v1.FOREST_KWARGS['contamination'] remains 0.1.  This audit does not retune a Consumer in service.

