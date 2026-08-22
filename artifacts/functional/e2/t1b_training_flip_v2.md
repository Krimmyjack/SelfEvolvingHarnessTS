# T1b -- training-side task flip (POSITIVE_CONTROL)

- verdict: **AD_TRAINABLE_SPEC_DEFECT** (pre-registered stop; the first block ends the run)
- reason: the identity-trained classifier reads Qcal at macro-averaged per-series event F1 0.17646022730867628 under the primary window and 0.29416579019672207 under the fallback; both below 0.50
- stopped at stage: `a3_readability_gate`
- evidence grade: POSITIVE_CONTROL (permanent)
- LLM 0/0, forecasting retrains 0/40, AD evaluations 38/300
- Part 0 checkpoint: `a6ba53d` (6 files)

## Part A -- the A3 readability gate

- instrument: `evaluation/functional/consumers/anomaly_detection_trainable_v2.py`
- rule: identity(B)-trained classifier, macro-averaged per-series event F1 on Qcal >= 0.50; the formal Query never participates
- gate metric: macro_f1; primary window: 0.17646022730867628 (pooled 0.17338709677419353); fallback window: 0.29416579019672207 (pooled 0.27627627627627627); passed: False

## Part B -- the two Query regions

- QF [2100, 2560): 4 events/series, 48 total, 1 skips; ledger frozen before any training or scoring
- QCAL [2600, 3060): 4 events/series, 48 total, 0 skips; ledger frozen before any training or scoring
- B4 oracle smoke (robust-z 49/3.5, visibility only): Qf pooled F1 None, Qcal pooled F1 0.7457627118644068 (Qf never read -- the run stopped before the gate released it)

## Part C / Part D

- C2 same-byte: {"comparisons": 600, "all_equal": true, "failures": [], "assertion": "np.array_equal between the bytes handed to the ridge stack and the bytes handed to the AD detector, on every shared index of the one P(B) buffer, per (series, program, anchor)", "note": "both Consumers slice one buffer by construction; this pass is the online guard that no code path re-derives, re-applies P, or hands over a copy that is not the same bytes"}
- B3 query-never-processed: {"rule": "v6._apply_program is called only inside t1.build_pbuffers, on [120, 900) slices; the Query arrays are never an argument", "program_calls": 120, "expected_program_calls": 120, "holds": true}
- AD arms: not reached
- Part D: not_reached -- the run stopped at a3_readability_gate before the aggregate-layer judgment; no flip claim is made or implied

## Discipline

- NOAA 2025 / beyond_17520 / SMD test+labels: zero reads; robust-z (49/3.5) served as injection-visibility oracle only, never as the main AD Consumer
- deliverables not committed (the Part 0 checkpoint excepted); no spawn; the other line untouched
- originals unchanged post-run: True; T1 copies unchanged: True; Query copies unchanged: True

## Ambiguities (reported, not self-adjudicated)

- the cycle counter restarts from slot 0 per Query region (the T1 convention: each freeze is its own seeded draw); the T1b book does not name the counter rule, only the seeds
- Query features for the first scored points read pristine bytes preceding the region (up to 48 back for window 49) -- the same trailing geometry the T0 detector uses, and never P-processed bytes; the v2 isolation block asserts the 168-step pre-region prefix byte-equal to pristine in every scored copy
- the v2 gate and judgment read the macro average of per-series event F1 (main-line ruling); the pooled F1 is kept as a secondary reading alongside, and both are in the artifact
- the A3 gate reads the same macro average on Qcal; a per-series reading of the gate was not pre-registered
- '168-step history isolation' is implemented as (a) each injected event's sigma prefix recomputed from pristine bytes and compared to the ledger, and (b) the 168-step pre-region prefix of every scored copy asserted byte-equal to pristine; if the main line meant a different isolation, the assertion block is the single place to adjust
