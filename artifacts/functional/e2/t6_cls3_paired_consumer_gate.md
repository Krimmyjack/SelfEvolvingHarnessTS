# CLS-3 paired Consumer qualification gate

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  development paired-Consumer eligibility on the frozen CLS-2 burst; not a natural UCR capability claim and not a CLS-2 judge swap

## Verdict

- **KNN_INJURED_NO_REPAIR**
- kNN injury is readable but no frozen Workflow recovered >=50% without a class-recall drop > 0.05 (Program Supply gap; no Agent)
- role: paired Consumer qualification; not a judge swap on the CLS-2 ridge result

## Five conditions

- (1) kNN delayed injury ≤ −0.05: **True**
- (2) ≥1 Workflow recovers ≥50%: **False**
- (3) that repair does not drop class recall >0.05: **False**
- (4) kNN Support order matches delayed: **False**
- (5) ridge identity Δacc still > −0.05: **True**
- all five: **False**

## Consumer contrast

- kNN identity Δacc: **-0.120000**
- ridge identity Δacc: **-0.013333**
- kNN − ridge injury: -0.106667

## Site

- dataset: GunPoint development (no ladder)
- injection: CLS-2 ledger replay (`_scratch/cls2/cls2_v1/injection_ledger.json`)
- seed/segments/amplitude: frozen; zero redraw
- features: raw || first difference (shared)
- Consumers: ridge-raw-plus-difference-v1; knn k=3 Euclidean uniform
- Workflows: identity + hampel_filter + outlier_mad (no extras)
- held-in = official TRAIN; TEST = delayed, byte-zero-touch
- hit 25/50 held-in (fit 15/35, Support 10/15)

## Eight-arm delayed (TEST) and Support

| consumer | workflow | n_fit | delayed acc | Support acc | delayed Δacc | Support Δacc |
|---|---|---:|---:|---:|---:|---:|
| ridge | identity_on_clean | 35 | 0.820000 | 0.933333 | +0.000000 | +0.000000 |
| ridge | identity_on_corrupted | 35 | 0.806667 | 0.733333 | -0.013333 | -0.200000 |
| ridge | hampel_filter | 35 | 0.806667 | 0.666667 | -0.013333 | -0.266667 |
| ridge | outlier_mad | 35 | 0.693333 | 0.733333 | -0.126667 | -0.200000 |
| knn | identity_on_clean | 35 | 0.780000 | 0.800000 | +0.000000 | +0.000000 |
| knn | identity_on_corrupted | 35 | 0.660000 | 0.600000 | -0.120000 | -0.200000 |
| knn | hampel_filter | 35 | 0.660000 | 0.666667 | -0.120000 | -0.133333 |
| knn | outlier_mad | 35 | 0.606667 | 0.733333 | -0.173333 | -0.066667 |

### Per-class recall (delayed / Support)

**ridge / identity_on_clean**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 7 | 1.000000 |
| 1 | 74 | 0.702703 | 8 | 0.875000 |

**ridge / identity_on_corrupted**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.868421 | 7 | 0.857143 |
| 1 | 74 | 0.743243 | 8 | 0.625000 |

**ridge / hampel_filter**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.881579 | 7 | 0.857143 |
| 1 | 74 | 0.729730 | 8 | 0.500000 |

**ridge / outlier_mad**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 1.000000 | 7 | 0.714286 |
| 1 | 74 | 0.378378 | 8 | 0.750000 |

**knn / identity_on_clean**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.868421 | 7 | 0.857143 |
| 1 | 74 | 0.689189 | 8 | 0.750000 |

**knn / identity_on_corrupted**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.750000 | 7 | 0.714286 |
| 1 | 74 | 0.567568 | 8 | 0.500000 |

**knn / hampel_filter**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.750000 | 7 | 0.714286 |
| 1 | 74 | 0.567568 | 8 | 0.625000 |

**knn / outlier_mad**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 1.000000 | 7 | 0.714286 |
| 1 | 74 | 0.202703 | 8 | 0.750000 |

## kNN recoveries

| workflow | recovery fraction | recall guard | qualifies |
|---|---:|---|---|
| corrupted_hampel | 0.0000 | False | False |
| corrupted_outlier_mad | -0.4444 | False | False |

## n, fit, determinism

- TEST n=150, step=0.006667; injury bar 7.50 steps (above floor=True)
- official fits 8 / 30: {'ridge.clean_reference': 1, 'ridge.corrupted_identity': 1, 'ridge.corrupted_hampel': 1, 'ridge.corrupted_outlier_mad': 1, 'knn.clean_reference': 1, 'knn.corrupted_identity': 1, 'knn.corrupted_hampel': 1, 'knn.corrupted_outlier_mad': 1}
- verification fits 8; two-run **BITWISE_IDENTICAL**
- injected SHA matches CLS-2: **True**

## Obligation self-report

- agent_invoked: False
- amplitude_scan: False
- beyond_17520_reads: 0
- clean_rows_untouched: True
- cls2_ledger_rewritten: False
- dataset_scan: False
- distance_scan: False
- fit_budget_cap: 30
- fit_budget_respected: True
- fit_budget_used: 8
- flying_files_untouched: ['AGENTS.md', 'README.md', 'docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md', 'docs/SUCCESSOR_BRIEF_2026-08-22.md']
- injection_from_frozen_ledger: True
- judge_swap: False
- k_scan: False
- llm_calls: 0
- loader_output_unmutated: True
- nab_reads: 0
- noaa_2025_reads: 0
- part0: False
- preregistered_gates_rewritten: False
- smd_reads: 0
- test_bytes_touched: False
- test_sha_unchanged: True
- third_repair: False
- two_run: True
- yahoo_all_reads: 0
- zip_bytes_unchanged: True

## Out-of-book findings (report only, not repaired)

- Paired-Consumer eligibility, not a judge swap to overturn CLS-2 INJURY_NOT_READABLE.
- Injection was ledger replay only; seed/segments/noise were not redrawn.
- k=3 / Euclidean / uniform and GunPoint / amplitude 5 were not scanned.
- Part 0 is empty; CLS-2 artifacts stay with the CLS-OP book.
- This CLS-3 artifact stays uncommitted.
- Ridge four-arm delayed acc reproduced CLS-2 bitwise.
- kNN identity delayed Δacc=-0.120000 (bar=−0.05, step=0.006667).
- kNN Support vs delayed full order failed (delayed ['clean_reference', 'corrupted_hampel', 'corrupted_identity', 'corrupted_outlier_mad']; Support ['clean_reference', 'corrupted_outlier_mad', 'corrupted_hampel', 'corrupted_identity']; pair-direction False).
