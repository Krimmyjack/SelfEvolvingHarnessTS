# CLS-2 classification value-corruption qualification gate

evidence class: INSTRUMENT / POSITIVE_CONTROL (development).  development positive control on an injected contiguous burst; not a natural UCR capability claim

## Verdict

- **INJURY_NOT_READABLE**
- clean vs corrupted+identity delayed Δacc is -0.013333333333333308; pre-registered bar is <= -0.050

## Dataset choice

- selected: **GunPoint** (rejected ECG200; no ladder)
- GunPoint clean delayed acc was 0.82 in CLS-1-r2 versus chance ~0.50 (0.32 drop room).  ECG200 clean was 0.80 versus a 0.64 majority floor (only 0.16 room), and r2 already showed that imbalance can manufacture a +0.02 identity gain.  The Consumer is raw || first difference: a 15–20% burst on L=150 corrupts 23–30 raw coordinates plus two ~5σ difference spikes at the boundaries, more feature mass than ECG200 L=96 (14–19 points).  Injury bar 0.05 is 7.5 TEST steps here versus 5 on ECG200.  No substrate ladder.

## Menu reconnaissance

- mandated: `hampel_filter`
- W2: **`outlier_mad`**
- Contiguous 5×std burst on 15–20% of a z-normed row is a tail against the remaining 80–85% clean mass.  outlier_mad (intrinsic, global robust clip, k=3.5) is the matching family: it can see the burst without a window longer than the burst.  denoise_median default window=5 cannot eat a 23–30 point GunPoint burst.  Hampel's window=7 local MAD inflates inside the burst, so the second slot must not duplicate that local-window family.  winsorize/outlier_iqr are the same global-clip family as outlier_mad (registry docstring); one representative is enough.  repair_level_shift targets a two-boundary level geometry, not additive burst noise.

| name | category | tags | targeting_mode | destructive |
|---|---|---|---|---|
| `denoise_savgol` | denoise | smoothing | global | False |
| `denoise_wavelet` | denoise | smoothing | global | False |
| `denoise_median` | denoise | smoothing | global | False |
| `smooth_ma` | denoise | smoothing | global | False |
| `denoise_stl` | denoise | smoothing | global | False |
| `winsorize` | outlier | destructive | intrinsic | True |
| `outlier_iqr` | outlier | destructive | intrinsic | True |
| `outlier_mad` | outlier | destructive | intrinsic | True |
| `hampel_filter` | outlier | destructive | intrinsic | True |
| `repair_level_shift` | structural | destructive | intrinsic | True |
| `stl_decompose` | decompose | — | global | False |
| `fft_decompose` | decompose | — | global | False |
| `smooth_ema` | decompose | smoothing | global | False |

Impute / shape / scale operators were listed as classification-legal but excluded from the repair menu (see JSON).

## Site

- Consumer: ridge-raw-plus-difference-v1 (reused)
- quality contract: classification-global-coarse-quality-v1
- held-in = official TRAIN; TEST = delayed, byte-zero-touch
- 50% class-stratified hit rows; one contiguous burst, 15–20% of L; additive N(0, (5×row std)²); seed 202608254
- identity = fit the corrupted substrate (no row drop)
- ledger: `_scratch/cls2/cls2_v1/injection_ledger.json`
- hit 25/50 held-in (fit 15/35, Support 10/15)

## Observation visibility (value-corruption variant)

- coverage clean/corrupted: 1.000000 / 1.000000
- max_missing_run clean/corrupted: 0 / 0
- missing signal present: **False** (expected False)
- hit-row local_robust_z_peak mean: 18.2262 → 67.2302 (max corrupted 368.8610)
- hit rows whose z-peak rose: 23/25
- value corruption visible in public features: **True**
- value corruption does not move coverage; visibility is carried by local_robust_z_peak / region features, which is what a future Fast Agent could read

| feature | clean mean (hit) | corrupted mean (hit) | Δ |
|---|---:|---:|---:|
| `missing_fraction` | 0.000000 | 0.000000 | +0.000000 |
| `longest_missing_run_fraction` | 0.000000 | 0.000000 | +0.000000 |
| `local_robust_z_peak` | 18.226199 | 67.230200 | +49.004001 |
| `estimated_region_start_fraction` | 0.240800 | 0.221067 | -0.019733 |
| `estimated_region_end_fraction` | 0.731200 | 0.769333 | +0.038133 |
| `outlier_region_end_fraction` | 0.429067 | 0.686667 | +0.257600 |
| `level_region_fraction` | 0.407733 | 0.384800 | -0.022933 |
| `level_excursion_score` | 43.062245 | 35.435911 | -7.626334 |
| `estimated_level_offset` | 2.173942 | 2.221632 | +0.047690 |
| `period_change_score` | 2.274656 | 1.431710 | -0.842946 |
| `period_reliability` | 1.000000 | 1.000000 | +0.000000 |

## Four-arm delayed (TEST) and Support

| arm | workflow | n_fit (dropped) | delayed acc | Support acc | delayed Δacc | Support Δacc |
|---|---|---:|---:|---:|---:|---:|
| clean_reference | identity_on_clean | 35 (0) | 0.820000 | 0.933333 | +0.000000 | +0.000000 |
| corrupted_identity | identity_on_corrupted | 35 (0) | 0.806667 | 0.733333 | -0.013333 | -0.200000 |
| corrupted_hampel | hampel_filter | 35 (0) | 0.806667 | 0.666667 | -0.013333 | -0.266667 |
| corrupted_outlier_mad | outlier_mad | 35 (0) | 0.693333 | 0.733333 | -0.126667 | -0.200000 |

### Per-class recall (delayed / Support)

**clean_reference**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.934211 | 7 | 1.000000 |
| 1 | 74 | 0.702703 | 8 | 0.875000 |

**corrupted_identity**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.868421 | 7 | 0.857143 |
| 1 | 74 | 0.743243 | 8 | 0.625000 |

**corrupted_hampel**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 0.881579 | 7 | 0.857143 |
| 1 | 74 | 0.729730 | 8 | 0.500000 |

**corrupted_outlier_mad**

| class | delayed n | delayed recall | Support n | Support recall |
|---|---:|---:|---:|---:|
| 0 | 76 | 1.000000 | 7 | 0.714286 |
| 1 | 74 | 0.378378 | 8 | 0.750000 |

## B1 / B2

- injury Δacc: **-0.013333** (readable=False, bar=−0.05, 7.50 steps)
- legal headroom: **False**; best repair corrupted_hampel (recovery 0.0000)
- Support vs delayed full order: False; identity/best-repair direction: True; B2: **True**

| arm | recovery fraction | recall guard | qualifies |
|---|---:|---|---|
| corrupted_hampel | 0.0000 | False | False |
| corrupted_outlier_mad | -8.5000 | False | False |

## n, fit, determinism

- TEST n=150, step=0.006667; injury bar 7.50 steps (above floor=True)
- official fits 4 / 50: {'clean_reference': 1, 'corrupted_identity': 1, 'corrupted_hampel': 1, 'corrupted_outlier_mad': 1}
- verification fits 4; two-run **BITWISE_IDENTICAL**

## Obligation self-report

- agent_invoked: False
- amplitude_scan: False
- beyond_17520_reads: 0
- clean_rows_untouched: True
- fit_budget_cap: 50
- fit_budget_respected: True
- fit_budget_used: 4
- flying_files_untouched: ['AGENTS.md', 'README.md', 'docs/PROJECT_STATE_AND_DATA_MAP_2026-08-23.md', 'docs/SUCCESSOR_BRIEF_2026-08-22.md']
- injection_after_load: True
- llm_calls: 0
- loader_output_unmutated: True
- nab_reads: 0
- noaa_2025_reads: 0
- part0_sha: f1fe3a004c959934e8f4cea72df8107d45616e2e
- preregistered_gates_rewritten: False
- rate_scan: False
- smd_reads: 0
- substrate_ladder: False
- test_bytes_touched: False
- test_sha_unchanged: True
- third_repair: False
- two_run: True
- yahoo_all_reads: 0
- zip_bytes_unchanged: True

## Out-of-book findings (report only, not repaired)

- Loader z-norms each TRAIN row, so 5×row std is 5.0 on the unit-variance series.  Amplitude was not rescaled.
- Hampel default window=7 may treat the burst interior as a new local regime; that is a mechanism risk, not a parameter scan.
- No missingness family operators were added.  Impute ops remain no-ops on this finite substrate.
- Part 0 collected CLS-replay only; this CLS-2 artifact stays uncommitted.
- As required: coverage stayed 1.0 and max_missing_run stayed 0 after value corruption.
- clean vs identity delayed Δacc=-0.013333 (bar=−0.05, step=0.006667).
