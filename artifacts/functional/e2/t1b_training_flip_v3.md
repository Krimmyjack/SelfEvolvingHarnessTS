# T1b -- training-side task flip (POSITIVE_CONTROL)

- verdict: **TRAINING_SIDE_TASK_FLIP_CONFIRMED_POSITIVE_CONTROL**
- evidence grade: POSITIVE_CONTROL (permanent)
- LLM 0/0, forecasting retrains 0/0, AD evaluations 78/120
- Part 0 checkpoint: `a6ba53d` (6 files)

## v3 conventions (same book, continued)

- feature family: the only change vs v2: the feature is the single statistic z_t from anomaly_detection_v1.detect(values, window=49, threshold=3.5)['scores'] -- explicit 49/3.5 (T0's frozen fallback parameters), never the 25/4.0 file defaults; no re-standardization.  Head, labels, P(B), arms, Queries, scoring and ordering all frozen along v2
- single shot: no fallback: a gate miss closes the supervised-AD positive-control family (SUPERVISED_AD_PC_FAMILY_CLOSED)
- abstention: T0 semantics: undefined z is excluded from the fit (counted by warm_up / zero_scale / non_finite), forced to not flag at Query time, excluded from the AUPRC ranking with the count reported, never zeroed
- oracle: B4 readings cited from record (Qcal 0.7458 / Qf 0.6977), not re-run this slice
- relation to v1/v2: v1/v2 (raw trailing windows x linear ridge, both geometries) closed by credible negative; v3 changes only the feature family

## Mandatory calibration notes carried by the confirmed verdict

- the AD Consumer is a threshold head learned on the task-native sufficient statistic (a learnable robust-z in effect); a flip verdict speaks only for this Consumer family
- this result only proves that the training-data utility flip is readable by an instrument when the task-native sufficient statistic is visible -- it does not prove the Harness discovered that representation by itself, and it does not claim generalization to natural anomaly data

## Part A -- the trainable AD Consumer

- instrument: `evaluation/functional/consumers/anomaly_detection_trainable_v3.py`
- feature window used: 49
- A3 readability gate: {"rule": "identity(B)-trained classifier, macro-averaged per-series event F1 on Qcal >= 0.50; the formal Query never participates", "gate_metric": "macro_f1", "pooled_f1": 0.6301369863013699, "macro_f1": 0.6109126984126984, "fallback_pooled_f1": null, "fallback_macro_f1": null, "passed": true}

## Part B -- the two Query regions

- QF [2100, 2560): 4 events/series, 48 total, 1 skips; ledger frozen before any training or scoring
- QCAL [2600, 3060): 4 events/series, 48 total, 0 skips; ledger frozen before any training or scoring

B4 oracle smoke (robust-z 49/3.5, injection visibility only, never a verdict input): Qf pooled F1 0.6976744186046512, Qcal pooled F1 0.7457627118644068 (cited from record, not re-run this slice)

## Part C -- both Consumers trained on the same P(B)

- C2 same-byte: 600 comparisons, all_equal=True, reproducible=True
- B3 query-never-processed: True
- C3 guard: {"t1_copy_shas_unchanged": true, "pbuffers_recomputed_twice_byte_identical": true, "rebuilt_gains_match_recorded_c1": true}
- forecasting readings: artifacts/functional/e2/t1_flip_control_v1.json, read-only (retrains 0)

| program | forecasting delayed agg | AD macro F1 | AD pooled F1 | AD train gain | mean AUPRC | undef pts | flip |
|---|---|---|---|---|---|---|---|
| hampel_filter | +0.0648 | +0.7962 | +0.7961 | +0.0413 | +0.8878 | 249 | — |
| identity | — | +0.7549 | +0.7816 | — | +0.8878 | 249 | — |
| outlier_iqr | +0.2723 | +0.7635 | +0.7636 | +0.0086 | +0.8878 | 249 | — |
| outlier_mad | +0.3255 | +0.7962 | +0.7961 | +0.0413 | +0.8878 | 249 | — |
| winsorize | +0.4059 | +0.5877 | +0.5660 | -0.1672 | +0.8878 | 249 | forecasting_up_ad_down |

Per-series vectors (forecasting delayed and AD gain) are in the JSON under `part_d`.

## Discipline

- NOAA 2025 / beyond_17520 / SMD test+labels: zero reads; robust-z (49/3.5) served as injection-visibility oracle only, never as the main AD Consumer
- deliverables not committed (the Part 0 checkpoint excepted); no spawn; the other line untouched
- originals unchanged post-run: True; T1 copies unchanged: True; Query copies unchanged: True

## Ambiguities (reported, not self-adjudicated)

- the cycle counter restarts from slot 0 per Query region (the T1 convention: each freeze is its own seeded draw); the T1b book does not name the counter rule, only the seeds
- Query features for the first scored points read pristine bytes preceding the region (up to 48 back for window 49) -- the same trailing geometry the T0 detector uses, and never P-processed bytes; the v2 isolation block asserts the 168-step pre-region prefix byte-equal to pristine in every scored copy
- the v2-line metric ruling holds: the gate and the judgment read the macro average of per-series event F1; the pooled F1 is kept as a secondary reading alongside
- the A3 gate reads the same macro average on Qcal; a per-series reading of the gate was not pre-registered
- '168-step history isolation' is implemented as (a) each injected event's sigma prefix recomputed from pristine bytes and compared to the ledger, and (b) the 168-step pre-region prefix of every scored copy asserted byte-equal to pristine; if the main line meant a different isolation, the assertion block is the single place to adjust
- the z feature for the training block reads no bytes before the block (detect's warm-up starts at the block's own index 0, so training rows begin at block index 49), while Query features read the 49 pristine pre-region bytes -- the same training/query asymmetry canon as v1/v2
- the detect() threshold 3.5 is passed explicitly per the book but is inert for the feature path (only ['scores'] is used, never ['flags']); it is recorded for provenance

## Errata

- 2026-08-22: `part0_checkpoint` recorded `a6ba53d` (6 files) -- that was the T1b-era Part 0 (T2 wiring + #37). The v3 slice's own Part 0 is `359eec5` (9 files: trainable v1/v2 consumers, the runner, the v1/v2 artifacts, two main-line docs). Corrected by this appended erratum per the #39 pre-distribution revision; the frozen field is not rewritten, no reading changed, no re-run.
