# CLS-1-r2 classification qualification gate

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  development positive control on an injected row-subset contiguous-gap defect; not a natural UCR capability claim

## Verdict

- **INJURY_NOT_READABLE_BOTH**
- GunPoint and ECG200 both failed the pre-registered injury bar; stop for mainline/sol to reopen the defect family
- ladder: first=GunPoint, second=ECG200, trigger=INJURY_NOT_READABLE on GunPoint

## Binding

- Consumer / Support split / four-arm frame / gates reused from CLS-1
- injection: 50% class-stratified held-in rows; 2 contiguous gaps of 10–15 points, gap ≥20; clean rows untouched
- identity still drops any NaN training row
- TEST is Query/delayed only (byte-zero-touch)

## Exam: GunPoint

- local verdict: **INJURY_NOT_READABLE** — clean vs injected+identity delayed Δacc is 0.0; pre-registered bar is <= -0.050
- TRAIN n=50, TEST n=150, L=150, fit=35, Support=15
- TRAIN class counts: {'0': 24, '1': 26}; TEST: {'0': 76, '1': 74}
- inject seed 202608252; hit 25/50 held-in rows (fit 18/35, Support 7/15)
- mean missing fraction on hit rows: 0.1637; ledger `_scratch/cls1/cls1_r2_v1/gunpoint/injection_ledger.json`

### Observation

| surface | coverage | max_missing_run | missing_signal |
|---|---:|---:|---|
| clean TRAIN | 1.000000 | 0 | False |
| injected held-in | 0.918133 | 15 | True |

- Fast `_MISSING_ONLY_OPS` would skip impute: **False**
- mean per-series public `missing_fraction`: 0.081867

### Four-arm delayed (TEST) and Support

| arm | workflow | n_fit (dropped) | delayed acc | Support acc | delayed Δacc | Support Δacc |
|---|---|---:|---:|---:|---:|---:|
| clean_reference | identity_on_clean | 35 (0) | 0.820000 | 0.933333 | +0.000000 | +0.000000 |
| injected_identity | identity_drop_nan_training_rows | 17 (18) | 0.820000 | 1.000000 | +0.000000 | +0.066667 |
| injected_impute_linear | impute_linear | 35 (0) | 0.820000 | 0.933333 | +0.000000 | +0.000000 |
| injected_impute_ema | impute_ema | 35 (0) | 0.840000 | 0.866667 | +0.020000 | -0.066667 |

### Per-class recall (delayed / Support)

**clean_reference**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 7 | 1.000000 |
| 1 | 74 | 0.702703 | 8 | 0.875000 |

**injected_identity**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 3 | 1.000000 |
| 1 | 74 | 0.702703 | 5 | 1.000000 |

**injected_impute_linear**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 7 | 1.000000 |
| 1 | 74 | 0.702703 | 8 | 0.875000 |

**injected_impute_ema**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 7 | 0.857143 |
| 1 | 74 | 0.743243 | 8 | 0.875000 |

### B1 / B2

- injury Δacc: **+0.000000** (readable=False, bar=−0.05, 7.50 steps)
- legal headroom: **False**; best impute None (recovery null)
- Support vs delayed full order: False; identity/best-impute direction: None; B2: **False**

| arm | recovery fraction | recall guard | qualifies |
|---|---:|---|---|
| injected_impute_linear | — | — | INJURY_UNDEFINED |
| injected_impute_ema | — | — | INJURY_UNDEFINED |

- TEST n=150, step=0.006667; two-run **BITWISE_IDENTICAL**

## Exam: ECG200

- local verdict: **INJURY_NOT_READABLE** — clean vs injected+identity delayed Δacc is 0.019999999999999907; pre-registered bar is <= -0.050
- TRAIN n=100, TEST n=100, L=96, fit=70, Support=30
- TRAIN class counts: {'0': 31, '1': 69}; TEST: {'0': 36, '1': 64}
- inject seed 202608253; hit 50/100 held-in rows (fit 34/70, Support 16/30)
- mean missing fraction on hit rows: 0.2571; ledger `_scratch/cls1/cls1_r2_v1/ecg200/injection_ledger.json`

### Observation

| surface | coverage | max_missing_run | missing_signal |
|---|---:|---:|---|
| clean TRAIN | 1.000000 | 0 | False |
| injected held-in | 0.871458 | 15 | True |

- Fast `_MISSING_ONLY_OPS` would skip impute: **False**
- mean per-series public `missing_fraction`: 0.128542

### Four-arm delayed (TEST) and Support

| arm | workflow | n_fit (dropped) | delayed acc | Support acc | delayed Δacc | Support Δacc |
|---|---|---:|---:|---:|---:|---:|
| clean_reference | identity_on_clean | 70 (0) | 0.800000 | 0.733333 | +0.000000 | +0.000000 |
| injected_identity | identity_drop_nan_training_rows | 36 (34) | 0.820000 | 0.785714 | +0.020000 | +0.052381 |
| injected_impute_linear | impute_linear | 70 (0) | 0.770000 | 0.733333 | -0.030000 | +0.000000 |
| injected_impute_ema | impute_ema | 70 (0) | 0.770000 | 0.666667 | -0.030000 | -0.066667 |

### Per-class recall (delayed / Support)

**clean_reference**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 36 | 0.805556 | 9 | 0.666667 |
| 1 | 64 | 0.796875 | 21 | 0.761905 |

**injected_identity**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 36 | 0.638889 | 5 | 0.400000 |
| 1 | 64 | 0.921875 | 9 | 1.000000 |

**injected_impute_linear**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 36 | 0.833333 | 9 | 0.777778 |
| 1 | 64 | 0.734375 | 21 | 0.714286 |

**injected_impute_ema**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 36 | 0.805556 | 9 | 0.666667 |
| 1 | 64 | 0.750000 | 21 | 0.666667 |

### B1 / B2

- injury Δacc: **+0.020000** (readable=False, bar=−0.05, 5.00 steps)
- legal headroom: **False**; best impute None (recovery null)
- Support vs delayed full order: False; identity/best-impute direction: None; B2: **False**

| arm | recovery fraction | recall guard | qualifies |
|---|---:|---|---|
| injected_impute_linear | — | — | INJURY_UNDEFINED |
| injected_impute_ema | — | — | INJURY_UNDEFINED |

- TEST n=100, step=0.010000; two-run **BITWISE_IDENTICAL**

## Fit ledger (shared cap, ladder + recompute)

- used 16 / 50: {'GunPoint/official:clean_reference': 1, 'GunPoint/official:injected_identity': 1, 'GunPoint/official:injected_impute_linear': 1, 'GunPoint/official:injected_impute_ema': 1, 'GunPoint/verify:clean_reference': 1, 'GunPoint/verify:injected_identity': 1, 'GunPoint/verify:injected_impute_linear': 1, 'GunPoint/verify:injected_impute_ema': 1, 'ECG200/official:clean_reference': 1, 'ECG200/official:injected_identity': 1, 'ECG200/official:injected_impute_linear': 1, 'ECG200/official:injected_impute_ema': 1, 'ECG200/verify:clean_reference': 1, 'ECG200/verify:injected_identity': 1, 'ECG200/verify:injected_impute_linear': 1, 'ECG200/verify:injected_impute_ema': 1}

## Obligation self-report

- agent_invoked: False
- beyond_17520_reads: 0
- clean_rows_untouched: True
- fit_budget_cap: 50
- fit_budget_respected: True
- fit_budget_used: 16
- fits_by_arm: {'GunPoint/official:clean_reference': 1, 'GunPoint/official:injected_identity': 1, 'GunPoint/official:injected_impute_linear': 1, 'GunPoint/official:injected_impute_ema': 1, 'GunPoint/verify:clean_reference': 1, 'GunPoint/verify:injected_identity': 1, 'GunPoint/verify:injected_impute_linear': 1, 'GunPoint/verify:injected_impute_ema': 1, 'ECG200/official:clean_reference': 1, 'ECG200/official:injected_identity': 1, 'ECG200/official:injected_impute_linear': 1, 'ECG200/official:injected_impute_ema': 1, 'ECG200/verify:clean_reference': 1, 'ECG200/verify:injected_identity': 1, 'ECG200/verify:injected_impute_linear': 1, 'ECG200/verify:injected_impute_ema': 1}
- flying_files_untouched: ['AGENTS.md', 'README.md', 'docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md', 'docs/SUCCESSOR_BRIEF_2026-08-22.md']
- injection_after_load: True
- llm_calls: 0
- loader_output_unmutated: True
- missing_signal_after_inject: True
- nab_reads: 0
- noaa_2025_reads: 0
- part0_repeated: False
- preregistered_gates_rewritten: False
- rate_scan: False
- smd_reads: 0
- test_bytes_touched: False
- test_sha_unchanged: True
- third_impute: False
- two_run: True
- yahoo_all_reads: 0
- zip_bytes_unchanged: True

## Out-of-book findings (report only, not repaired)

- CLS-1 per-row point MCAR × drop-row identity collision is closed by construction: only the stratified 50% hit rows carry NaNs.
- Segment lengths stay absolute 10–15 (not rescaled to 13–20% of L). On ECG200 L=96 a hit row therefore misses 20.8–31.3% of its length, above the GunPoint ~13–20% sketch.  This is not a scan.
- CohortHistoryPublicToolGateway still needs 2*192 points; both substrates are shorter.  Missing signal uses the same _window_summary coverage / max-run formulas as CLS-1.
- Part 0 was not repeated; CLS-1 and CLS-1-r2 artifacts stay uncommitted for the next book to collect.
- GunPoint identity kept 17/35 fit rows (dropped 18 disaster rows).
- GunPoint clean vs identity delayed Δacc=+0.000000 (bar=−0.05, step=0.006667).
- GunPoint injected_impute_linear delayed acc=0.820000 vs clean 0.820000 (Δ=+0.000000).
- GunPoint injected_impute_ema delayed acc=0.840000 vs clean 0.820000 (Δ=+0.020000).
- ECG200 identity kept 36/70 fit rows (dropped 34 disaster rows).
- ECG200 clean vs identity delayed Δacc=+0.020000 (bar=−0.05, step=0.010000).
- ECG200 injected_impute_linear delayed acc=0.770000 vs clean 0.800000 (Δ=-0.030000).
- ECG200 injected_impute_ema delayed acc=0.770000 vs clean 0.800000 (Δ=-0.030000).
- ECG200 identity raised delayed acc (+0.02) while both imputes lowered it (−0.03).  The extra accuracy is majority-class recall (0.797→0.922) paid for by minority recall (0.806→0.639).  Not a gate input: B1 did not open.
