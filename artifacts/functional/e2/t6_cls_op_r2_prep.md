# CLS-OP-r2-prep -- verifier fix, smoke, Program headroom

protocol: `t6_cls_op_r2_prep_v1`  evidence grade: **DEVELOPMENT**  LLM: 0

## Verdict

**HEADROOM_EXISTS**

HEADROOM_EXISTS iff at least one shared-menu candidate is materially positive on at least one Target held-in surface (delta accuracy >= max(0.005, 1/n) for that surface's own n) while no per-class recall falls more than 0.05.  Source cells are reported for context and cannot carry the verdict.

next: r2 three-arm launch condition met

## Part A -- verifier fix

- before: verify_candidate is called per training window with maximum_modified_fraction; any window whose own modified_fraction exceeds the cap sets MODIFICATION_FRACTION_EXCEEDED, and passed = not rejected, so one window vetoes the candidate for the whole cohort
- after: modification_fraction_scope='cohort' passes 1.0 as the per-window cap so the fraction gate cannot fire per window, accumulates len(modified_indices) and window.size over every window, and rejects once if the ratio exceeds the cap.  Every other gate still vetoes per window.
- switch: modification_fraction_scope defaults to 'per_window', so forecasting, AD and minipipe callers are unchanged; only this book's executor opts in
- why not a number change: 0.20 and 'share of rows over the line' were both suggested by the CLS-OP diagnostic table, so adopting either would import a result-derived constant.  Changing which quantity the existing constant measures imports nothing.

### Zero regression

- before: 40 failed, 170 passed, 3 skipped, 1 xfailed
- after: 40 failed, 170 passed, 3 skipped, 1 xfailed
- identical failure set: True (sha 9947c9ed623279f4)
- pre-existing cause: 38 of the 40 are ValueError 'snapshot lock mismatch; run compiler with --write-lock' from methods/ttha/harness/compiler.py; the h0 lock was last regenerated at 29bed7e (2026-08-24 16:27) and operators/registry.py changed at 5ef9726 (2026-08-25 11:55), after it.  The lock carries an operator_bundle_sha, so the CLS-4 operator commit is the likely origin.  Not repaired here: it is another line's surface and this book is single-face.
- new unit tests: 8 passed

## Part B -- smoke: what survives the fixed verifier

| cell | menu | survivors after fix | survivors before fix | unblocked by the fix | numeric no-ops | verifier-rejected |
|---|---|---|---|---|---|---|
| ProximalPhalanxOutlineCorrect/fit_only_artifact | 24 | winsorize, outlier_iqr, outlier_mad, hampel_filter, repair_level_shift | winsorize, outlier_iqr, outlier_mad | hampel_filter, repair_level_shift | 10 | 9 |
| MiddlePhalanxOutlineCorrect/fit_only_artifact | 24 | winsorize, outlier_iqr, outlier_mad, hampel_filter, repair_level_shift | winsorize, outlier_iqr, outlier_mad | hampel_filter, repair_level_shift | 10 | 9 |
| Lightning2/fit_only_artifact | 24 | winsorize | winsorize | (none) | 9 | 14 |
| PhalangesOutlinesCorrect/fit_only_artifact | 24 | winsorize, outlier_iqr, outlier_mad, hampel_filter, repair_level_shift | winsorize, outlier_mad | outlier_iqr, hampel_filter, repair_level_shift | 10 | 9 |
| GunPointAgeSpan/fit_only_artifact | 24 | outlier_iqr, hampel_filter | (none) | outlier_iqr, hampel_filter | 9 | 13 |

## Part C -- Program headroom census

| cell | surface | n | material line | program | d accuracy | worst class recall d | material+ | guard |
|---|---|---|---|---|---|---|---|---|
| ProximalPhalanxOutlineCorrect | support | 46 | 0.0217 | winsorize | -0.0435 | -0.3548 | False | False |
| ProximalPhalanxOutlineCorrect | delayed | 46 | 0.0217 | winsorize | -0.1739 | -0.4516 | False | False |
| ProximalPhalanxOutlineCorrect | support | 46 | 0.0217 | outlier_iqr | +0.2826 | -0.3333 | True | False |
| ProximalPhalanxOutlineCorrect | delayed | 46 | 0.0217 | outlier_iqr | +0.1739 | -0.5333 | True | False |
| ProximalPhalanxOutlineCorrect | support | 46 | 0.0217 | outlier_mad | +0.3043 | -0.4000 | True | False |
| ProximalPhalanxOutlineCorrect | delayed | 46 | 0.0217 | outlier_mad | +0.1957 | -0.5333 | True | False |
| ProximalPhalanxOutlineCorrect | support | 46 | 0.0217 | hampel_filter | +0.1522 | +0.0000 | True | True |
| ProximalPhalanxOutlineCorrect | delayed | 46 | 0.0217 | hampel_filter | -0.0217 | -0.2000 | False | False |
| ProximalPhalanxOutlineCorrect | support | 46 | 0.0217 | repair_level_shift | +0.0435 | +0.0323 | True | True |
| ProximalPhalanxOutlineCorrect | delayed | 46 | 0.0217 | repair_level_shift | -0.0217 | -0.0323 | False | True |
| MiddlePhalanxOutlineCorrect | support | 45 | 0.0222 | winsorize | +0.1333 | +0.0690 | True | True |
| MiddlePhalanxOutlineCorrect | delayed | 45 | 0.0222 | winsorize | -0.1111 | -0.2069 | False | False |
| MiddlePhalanxOutlineCorrect | support | 45 | 0.0222 | outlier_iqr | +0.0000 | -0.1250 | False | False |
| MiddlePhalanxOutlineCorrect | delayed | 45 | 0.0222 | outlier_iqr | -0.1778 | -0.2414 | False | False |
| MiddlePhalanxOutlineCorrect | support | 45 | 0.0222 | outlier_mad | +0.2000 | -0.5000 | True | False |
| MiddlePhalanxOutlineCorrect | delayed | 45 | 0.0222 | outlier_mad | +0.0222 | -0.5000 | True | False |
| MiddlePhalanxOutlineCorrect | support | 45 | 0.0222 | hampel_filter | +0.0222 | +0.0000 | True | True |
| MiddlePhalanxOutlineCorrect | delayed | 45 | 0.0222 | hampel_filter | -0.0222 | -0.1379 | False | False |
| MiddlePhalanxOutlineCorrect | support | 45 | 0.0222 | repair_level_shift | +0.0222 | +0.0000 | True | True |
| MiddlePhalanxOutlineCorrect | delayed | 45 | 0.0222 | repair_level_shift | -0.0444 | -0.0690 | False | False |
| Lightning2 | support | 5 | 0.2000 | winsorize | +0.0000 | +0.0000 | False | True |
| Lightning2 | delayed | 5 | 0.2000 | winsorize | +0.0000 | -0.5000 | False | False |
| PhalangesOutlinesCorrect | support | 135 | 0.0074 | winsorize | -0.0148 | -0.1932 | False | False |
| PhalangesOutlinesCorrect | delayed | 135 | 0.0074 | winsorize | -0.0741 | -0.2159 | False | False |
| PhalangesOutlinesCorrect | support | 135 | 0.0074 | outlier_iqr | +0.0889 | -0.0568 | True | False |
| PhalangesOutlinesCorrect | delayed | 135 | 0.0074 | outlier_iqr | +0.0000 | -0.1023 | False | False |
| PhalangesOutlinesCorrect | support | 135 | 0.0074 | outlier_mad | +0.0222 | -0.2159 | True | False |
| PhalangesOutlinesCorrect | delayed | 135 | 0.0074 | outlier_mad | -0.0519 | -0.2386 | False | False |
| PhalangesOutlinesCorrect | support | 135 | 0.0074 | hampel_filter | +0.0222 | -0.0568 | True | False |
| PhalangesOutlinesCorrect | delayed | 135 | 0.0074 | hampel_filter | +0.0222 | -0.0341 | True | True |
| PhalangesOutlinesCorrect | support | 135 | 0.0074 | repair_level_shift | +0.0148 | -0.0213 | True | True |
| PhalangesOutlinesCorrect | delayed | 135 | 0.0074 | repair_level_shift | +0.0222 | -0.0213 | True | True |
| GunPointAgeSpan | support | 10 | 0.1000 | outlier_iqr | +0.1000 | -0.2000 | False | False |
| GunPointAgeSpan | delayed | 10 | 0.1000 | outlier_iqr | +0.2000 | -0.4000 | True | False |
| GunPointAgeSpan | support | 10 | 0.1000 | hampel_filter | +0.5000 | +0.4000 | True | True |
| GunPointAgeSpan | delayed | 10 | 0.1000 | hampel_filter | +0.3000 | +0.2000 | True | True |

## Budget

- LLM: 0
- Consumer fits: 46 of 120
- wall clock: 49.7 s

## Obligations

- **llm_calls**: 0
- **downloads**: 0
- **forbidden_data_untouched**: no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened; the only data root is data/ucr_task_context
- **single_face**: maximum_modified_fraction stays 0.10, maximum_candidates stays 3, selectable semantics and effect distinctness are untouched; the verifier is modified once and only in the fraction scope
- **no_new_runner**: Part B and Part C are entries on the CLS-OP runner, which already owns the cell builder, the executor subclass and the Consumer adapter; a second runner would be a second dialect
- **artifact_not_committed**: True
- **known_debts_not_paid_here**: ["verify_candidate.selectable still does not require effect distinctness, so a numeric no-op is still 'actionable' to the candidate supply; this book only excludes no-ops from its own survivor list", 'run_online_round still records a verifier rejection without its rejection code']
