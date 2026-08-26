# CLS-DEV-WINE precheck -- v2 injection legality + headroom

protocol: `t6_cls_dev_wine_v1`  target: **Wine**  evidence grade: **development**

## Gate

**FAMILY_CLOSURE_RECOMMENDED**

PASS iff hampel_filter is legal under the cohort verifier (cap 0.10) and its held-in headroom vs identity is >= max(0.005, 1/n_heldin).  Otherwise FAMILY_CLOSURE_RECOMMENDED; do not spend LLM, do not change the substrate.

- hampel_filter legal: True
- hampel headroom vs identity: 0.0 (line 0.058823529411764705, n_heldin=17)
- hampel cohort modified fraction: 0.0297008547008547
- hampel rejection: -

## Honesty constraint

Wine was previously used by the action_credit line (run_e2_action_credit_candidate_ordering.py) under the same impulse condition pair (audit: artifacts/functional/e2/t6_cls_conf_r3_selection.json). This run is therefore not an independent confirmation. Every judgement stays at evidence_grade=development. The label CLS_CHAIN_CONFIRMED must not be used.

This artifact is **development** evidence.  It is not an independent confirmation and must not be cited as CLS_CHAIN_CONFIRMED.

## v2 scaling

- formula: `round(1/150 * L)`
- v1 source constants: evaluation/functional/run_e2_task_context_label_evidence_witness.py:37 SPIKE_FRACTIONS=(0.08, 0.20, 0.80, 0.92); :38 SPIKE_AMPLITUDE=16.0; :95-100 _inject writes one point per position so v1 segment length = 1
- invariance at L=150: **True**  checks={'segment_length_equals_v1': True, 'positions_equal': True, 'amplitude_equals_v1': True, 'fractions_equal_v1': True, 'injected_values_equal': True}

### Wine (L=234)

- v2 segment length: 2
- positions: [19, 47, 186, 214]
- injection: 8/234 = 0.03418803418803419 (below 0.10: True)
- hampel theoretical (window=3, halo/spike=4): 16/234 = 0.06837606837606838 (below 0.10: True)
- segments overlap: False (min gap 28)

### ECG200 reference (L=96)

- v2 segment length: 1
- positions: [8, 19, 76, 87]
- injection: 4/96 = 0.041666666666666664 (below 0.10: True)
- hampel theoretical (window=3, halo/spike=3): 12/96 = 0.125 (below 0.10: False)
- segments overlap: False (min gap 11)

### GunPoint positive-control length (L=150)

- v2 segment length: 1
- positions: [12, 30, 119, 137]
- injection: 4/150 = 0.02666666666666667 (below 0.10: True)
- hampel theoretical (window=3, halo/spike=3): 12/150 = 0.08 (below 0.10: True)
- segments overlap: False (min gap 18)

## Substrate

- archive: `data/ucr_task_context/Wine.zip` (388159 bytes)
- TRAIN rows × length: 57 × 234; classes: 2
- held-in n (full support pool): 17
- consumer / metric: ridge-raw-plus-difference-v1 / accuracy

## Operator table

| program | legal | modified fraction | rejection | no-op | headroom vs identity | worst class Δrecall |
|---|---|---|---|---|---|---|
| identity | True | 0.0 | - | True | +0.0000 | +0.0000 |
| impute_linear | True | 0.0 | - | True | +0.0000 | +0.0000 |
| impute_fft | True | 0.0 | - | True | +0.0000 | +0.0000 |
| impute_ema | True | 0.0 | - | True | +0.0000 | +0.0000 |
| period_complete | True | 0.0 | - | True | +0.0000 | +0.0000 |
| period_median_complete | True | 0.0 | - | True | +0.0000 | +0.0000 |
| impute_ssm | True | 0.0 | - | True | +0.0000 | +0.0000 |
| impute_ar | True | 0.0 | - | True | +0.0000 | +0.0000 |
| denoise_savgol | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| denoise_wavelet | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| denoise_median | True | 0.0 | - | True | +0.0000 | +0.0000 |
| smooth_ma | False | 0.9995726495726496 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| denoise_stl | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| winsorize | False | 0.10256410256410256 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| outlier_iqr | False | 0.16367521367521368 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| outlier_mad | False | 0.24476495726495726 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| hampel_filter | True | 0.0297008547008547 | - | False | +0.0000 | -0.5556 |
| repair_level_shift | False | 0.5511752136752137 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| repair_burst_segment | False | 0.193482905982906 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| stl_decompose | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| fft_decompose | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| smooth_ema | False | 0.9971153846153846 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| resample_uniform | True | 0.0 | - | True | +0.0000 | +0.0000 |
| znorm | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |
| minmax_norm | False | 1.0 | COHORT_MODIFICATION_FRACTION_EXCEEDED | False | n/a | n/a |

## Cost

- LLM: 0 / 0
- Consumer fits: 2 / 200
- wall clock: 1.2 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **cap_not_raised**: True
- **no_scan_no_tune**: True
- **zero_llm**: True
- **downloads**: 0
- **ucr_conf_downloaded_not_opened**: True
- **other_line_files_untouched**: True
- **not_an_independent_confirmation**: True
- **cls_chain_confirmed_label_not_used**: True
- **existing_entries_unchanged**: True
