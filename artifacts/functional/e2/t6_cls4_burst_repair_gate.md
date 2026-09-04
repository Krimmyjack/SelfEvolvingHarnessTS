# CLS-4 burst-repair Program Supply + paired-Consumer re-judgment

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  development Program Supply on the frozen CLS-2 burst; not a natural UCR capability claim

## Verdict

- **REPAIR_INSUFFICIENT**
- repair_burst_segment did not recover >=50% of the kNN injury without a class-recall drop > 0.05 (Program Supply still open)

## Five conditions

- (1) kNN delayed injury ≤ −0.05: **True** (cited CLS-3 -0.120)
- (2) repair_burst_segment recovers ≥50%: **False** (fraction -0.4444)
- (3) that arm recall vs clean not worse >0.05: **False**
- (4) kNN Support order matches delayed (all arms): **False**
- (5) ridge identity remains numb: **True** (cited CLS-3 -0.0133)
- all five: **False**

## New operator

- `repair_burst_segment`: series-level robust-z, |z|>3.5 and run≥8, endpoint linear interpolation; identity if no hit
- targeting_mode=intrinsic; allowed_tasks=(classification,); destructive=True
- unit tests: {'requested': ['deterministic', 'clean_identity', 'boundary_head_tail', 'synthetic_index_exact', 'repair_mae_beats_corruption'], 'module': 'tests/operators/test_repair_burst_segment.py'}

## Detection quality vs CLS-2 ledger

- mean IoU on hit rows: **0.1422**
- detection recall / precision: 0.328358 / 0.159536
- hit rows with any detection: 14/25
- clean-row identity: 11/25 (rate 0.440000)

## Ten-arm delayed (TEST) and Support

| consumer | workflow | source | delayed acc | Support acc | delayed Δacc | Support Δacc |
|---|---|---|---:|---:|---:|---:|
| ridge | identity_on_clean | cited-CLS-3 | 0.820000 | 0.933333 | +0.000000 | +0.000000 |
| ridge | identity_on_corrupted | cited-CLS-3 | 0.806667 | 0.733333 | -0.013333 | -0.200000 |
| ridge | hampel_filter | cited-CLS-3 | 0.806667 | 0.666667 | -0.013333 | -0.266667 |
| ridge | outlier_mad | cited-CLS-3 | 0.693333 | 0.733333 | -0.126667 | -0.200000 |
| ridge | repair_burst_segment | new-fit | 0.666667 | 0.533333 | -0.153333 | -0.400000 |
| knn | identity_on_clean | cited-CLS-3 | 0.780000 | 0.800000 | +0.000000 | +0.000000 |
| knn | identity_on_corrupted | cited-CLS-3 | 0.660000 | 0.600000 | -0.120000 | -0.200000 |
| knn | hampel_filter | cited-CLS-3 | 0.660000 | 0.666667 | -0.120000 | -0.133333 |
| knn | outlier_mad | cited-CLS-3 | 0.606667 | 0.733333 | -0.173333 | -0.066667 |
| knn | repair_burst_segment | new-fit | 0.606667 | 0.733333 | -0.173333 | -0.066667 |

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

**ridge / repair_burst_segment**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.855263 | 7 | 0.571429 |
| 1 | 74 | 0.472973 | 8 | 0.500000 |

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

**knn / repair_burst_segment**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 1.000000 | 7 | 0.714286 |
| 1 | 74 | 0.202703 | 8 | 0.750000 |

## Ridge on the new repair

- delayed acc: 0.666667 (Δ vs clean -0.153333, Δ vs identity -0.140000)

## n, fit, determinism

- TEST n=150, step=0.006667; kNN recovery fraction -0.4444
- official new fits 2 / 30: {'ridge.corrupted_repair_burst_segment': 1, 'knn.corrupted_repair_burst_segment': 1}
- verification fits 2; two-run **BITWISE_IDENTICAL**
- injected SHA matches CLS-2: **True**

## Obligation self-report

- agent_invoked: False
- amplitude_scan: False
- beyond_17520_reads: 0
- cls2_ledger_rewritten: False
- dataset_scan: False
- fit_budget_cap: 30
- fit_budget_respected: True
- fit_budget_used: 2
- flying_files_untouched: ['AGENTS.md', 'README.md', 'docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md', 'docs/SUCCESSOR_BRIEF_2026-08-22.md']
- k_scan: False
- ledger_redraw: False
- llm_calls: 0
- loader_output_unmutated: True
- mad_refit: False
- methods_edited: False
- min_run_scan: False
- nab_reads: 0
- noaa_2025_reads: 0
- part0_is_cls3_collection: True
- second_new_operator: False
- smd_reads: 0
- test_bytes_touched: False
- test_sha_unchanged: True
- threshold_scan: False
- two_run: True
- yahoo_all_reads: 0
- zip_bytes_unchanged: True

## Out-of-book findings (report only, not repaired)

- Recon before Part A: no registry operator already did contiguous-burst interpolation.  hampel/mad are pointwise; repair_level_shift is a step geometry.
- Detection uses series-level median/MAD (book-allowed global simplification of rolling robust-z).  Gaussian holes inside a 5σ burst can split runs below min_run=8.
- outlier_mad / hampel / identity / clean delayed numbers are cited from CLS-3; only repair_burst_segment was newly fit.
- methods/ was not edited.  CLS-4 JSON/MD stay uncommitted.
- Part 0 is the CLS-3 artifact collection, not this gate.
- Ledger IoU mean=0.1422; detection recall=0.3284; clean-row identity=0.440.
- Ridge burst vs identity delayed Δacc=-0.140000.
- kNN Support vs delayed order: delayed ['clean_reference', 'corrupted_hampel', 'corrupted_identity', 'corrupted_outlier_mad', 'corrupted_repair_burst_segment']; Support ['clean_reference', 'corrupted_outlier_mad', 'corrupted_repair_burst_segment', 'corrupted_hampel', 'corrupted_identity'].
