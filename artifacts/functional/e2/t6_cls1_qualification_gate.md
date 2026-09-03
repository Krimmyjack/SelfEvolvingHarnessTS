# CLS-1 classification qualification gate

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  development positive control on an injected MCAR defect; not a natural UCR capability claim

## Verdict

- **INSTRUMENT_UNREADABLE**
- at least one arm could not produce a delayed accuracy; the B1 injury contrast is undefined

## Dataset choice

- selected: **GunPoint** (rejected ECG200)
- GunPoint TRAIN is class-balanced (24/26 after official labels mapped 1,2 -> 0,1) against ECG200 TRAIN 31/69; TEST n=150 gives quantization floor 1/150≈0.00667 so the 0.05 injury bar is 7.5 steps, versus ECG200 TEST n=100 (floor 0.01, 5 steps). Both zips are already on the in-service loader; Support 30% still leaves >=7 series per class.
- TRAIN n=50, TEST n=150, L=150
- TRAIN class counts: {'0': 24, '1': 26}
- TEST class counts: {'0': 76, '1': 74}

## Site

- Consumer: ridge-raw-plus-difference-v1 (RidgeClassifier alpha=1, features = raw || first difference; reused from run_e2_task_context_label_evidence_witness.py)
- quality contract: classification-global-coarse-quality-v1
- held-in = official TRAIN; Query/delayed = official TEST (byte-zero-touch)
- Support = per-class 30% of TRAIN, min 3/class, seed 2026082501; remainder = fit
- fit n=35, Support n=15
- injection: held-in only, point_mcar, rate 0.15, seed 20260825, 22 missing points/row
- ledger: `_scratch/cls1/cls1_v1/injection_ledger.json`

## Observation missing signal

| surface | coverage | max_missing_run | missing_signal |
|---|---:|---:|---|
| clean TRAIN | 1.000000 | 0 | False |
| injected held-in | 0.853333 | 5 | True |

- Fast `_MISSING_ONLY_OPS` would skip impute: **False** (need coverage<1 or max_run>0)
- mean per-series public `missing_fraction`: 0.146667

## Four-arm delayed (TEST) and Support

| arm | workflow | n_fit (dropped) | delayed acc | Support acc | delayed Δacc | Support Δacc |
|---|---|---:|---:|---:|---:|---:|
| clean_reference | identity_on_clean | 35 (0) | 0.820000 | 0.933333 | +0.000000 | +0.000000 |
| injected_identity | identity_drop_nan_training_rows | 0 (35) | null | null | null | null |
| injected_impute_linear | impute_linear | 35 (0) | 0.793333 | 0.933333 | -0.026667 | +0.000000 |
| injected_impute_ema | impute_ema | 35 (0) | 0.793333 | 0.866667 | -0.026667 | -0.066667 |

### Per-class recall (delayed / Support)

**clean_reference**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 7 | 1.000000 |
| 1 | 74 | 0.702703 | 8 | 0.875000 |

**injected_identity**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| — | 0 | null | 0 | null |

**injected_impute_linear**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.921053 | 7 | 1.000000 |
| 1 | 74 | 0.662162 | 8 | 0.875000 |

**injected_impute_ema**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.868421 | 7 | 1.000000 |
| 1 | 74 | 0.716216 | 8 | 0.750000 |

## B1 / B2

- injury Δacc (identity − clean, delayed): **null** (readable=False, bar=-0.050, bar/floor=7.50 steps)
- legal headroom: **False**
- best impute: None (recovery fraction null)
- Support vs delayed full order match: False
- identity / best-impute direction match: None
- B2 passed: **False**

### Impute recovery detail

| arm | recovery fraction | recall guard | qualifies |
|---|---:|---|---|
| injected_impute_linear | — | — | INJURY_UNDEFINED |
| injected_impute_ema | — | — | INJURY_UNDEFINED |

## n and quantization floor

- TEST n=150, one step=1/n=0.006667
- injury bar 0.05 is 7.50 steps (above floor=True)
- recall-harm bar 0.05 above floor: True

## Fit ledger and determinism

- official fits 3 / 50: {'clean_reference': 1, 'injected_impute_linear': 1, 'injected_impute_ema': 1}
- verification fits 3 (reported separately)
- two-run numeric fingerprint: **BITWISE_IDENTICAL**
- TEST SHA unchanged: True
- zip SHA unchanged: True

## Obligation self-report

- agent_invoked: False
- beyond_17520_reads: 0
- fit_budget_cap: 50
- fit_budget_respected: True
- fit_budget_used: 3
- fits_by_arm: {'clean_reference': 1, 'injected_impute_linear': 1, 'injected_impute_ema': 1}
- flying_files_untouched: ['AGENTS.md', 'README.md', 'docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md', 'docs/SUCCESSOR_BRIEF_2026-08-22.md']
- injection_after_load: True
- injection_ledger_path: _scratch/cls1/cls1_v1/injection_ledger.json
- llm_calls: 0
- loader_output_unmutated: True
- missing_signal_after_inject: True
- missing_signal_on_clean: False
- nab_reads: 0
- noaa_2025_reads: 0
- preregistered_gates_rewritten: False
- rate_scan: False
- smd_reads: 0
- test_bytes_touched: False
- test_sha_unchanged: True
- third_impute: False
- two_run: BITWISE_IDENTICAL
- verification_fits: 3
- yahoo_all_reads: 0
- zip_bytes_unchanged: True

## Out-of-book findings (report only, not repaired)

- per-row 15% point MCAR on L=150 makes P(complete series)=(0.85)^150≈2.587e-11; identity-as-drop-rows therefore kept 0 fit rows.  This is a structural collision between the pre-registered identity NaN policy and the pre-registered injection, not a rate scan.
- CohortHistoryPublicToolGateway requires 2*192 points; GunPoint L=150 cannot enter that forecast history window.  Missing signal was read with the same _window_summary coverage / max-run formulas Fast Agent gates on, treating each held-in series as one window.
- injected_impute_linear delayed acc=0.793333 vs clean 0.820000 (Δ=-0.026667); reported as a diagnostic only because identity delayed is null.
- injected_impute_ema delayed acc=0.793333 vs clean 0.820000 (Δ=-0.026667); reported as a diagnostic only because identity delayed is null.
- evaluation/functional/run_e2_t6_45_frep_a5a3_replay.py remains untracked; Part 0 allowlist listed the 45-frep artifacts and the 44-audit runner, not that replay runner.
