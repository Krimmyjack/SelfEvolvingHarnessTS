# T1b -- training-side task flip (POSITIVE_CONTROL)

- verdict: **AD_TRAINABLE_SPEC_DEFECT** (pre-registered stop; the first block ends the run)
- reason: the identity-trained classifier reads Qcal at pooled event F1 0.170873786407767 under the primary window and 0.26062322946175637 under the fallback; both below 0.50
- stopped at stage: `a3_readability_gate`
- evidence grade: POSITIVE_CONTROL (permanent)
- LLM 0/0, forecasting retrains 0/40, AD evaluations 50/300
- Part 0 checkpoint: `a6ba53d` (6 files)

## Part A -- the A3 readability gate

- instrument: `evaluation/functional/consumers/anomaly_detection_trainable_v1.py`
- rule: identity(B)-trained classifier, pooled event F1 on Qcal >= 0.50; the formal Query never participates
- primary window pooled F1: 0.170873786407767; fallback window pooled F1: 0.26062322946175637; passed: False

## Part B -- the two Query regions

- QF [2100, 2560): 4 events/series, 48 total, 1 skips; ledger frozen before any training or scoring
- QCAL [2600, 3060): 4 events/series, 48 total, 0 skips; ledger frozen before any training or scoring
- B4 oracle smoke (robust-z 49/3.5, visibility only): Qf pooled F1 0.6976744186046512, Qcal pooled F1 0.7457627118644068

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
- the aggregate AD gain is the pooled event-F1 difference over Qf's 48 events, matching the book's quantization note; the per-series F1 differences are reported alongside
- the A3 gate reads pooled Qcal F1 (the 48-event granularity), consistent with the aggregate-layer judgment; a per-series reading of the gate was not pre-registered
- Query features for t in [2100, 2100+49) read the 49 pristine bytes preceding the region -- the same trailing geometry the T0 detector uses, and never P-processed bytes
