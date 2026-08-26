# PS-0b confirmation-surface audit

protocol: `ps0b_confirmation_surface_audit_v1`  evidence grade: **development**  git: `bbd5fc5d225383a70181200ab532cb5f89e096c5`

**SECOND_SOURCE_AVAILABLE**

ROBUST dual source exists in: hampel_filter

本文件不得进入任何臂的 prompt/store/检索视野

Sealed oracles were read only as exam keys.  This artifact must not enter any arm prompt, store, or retrieval view.

## 1. Method

- Object: every r1+r3 census unit whose oracle set is non-identity (ties kept).
- Same cell construction as the sealed oracle (`_r3_build_cell`); slice_rows verified against the sealed file.
- Same consumer/metric: ridge-raw-plus-difference-v1 / accuracy.  The workflow is applied to the fit cohort once; each slice is scored with that one model (identity fit shared per unit).
- Slice materiality = max(0.005, 1/n_slice).
- Frozen grades: ROBUST_LEARNABLE ≥3/4; FRAGILE 1–2/4; UNREADABLE 0/4.
- Margin multiplier = pooled reading ÷ coarsest-slice materiality (source-qualification reproducibility: ≥2×).
- Dual source = same program + family independence (name prefix or byte-equal pattern_view) + five-axis Scope usable (pattern intersection has leaves beyond task_kind).

## 2. Unit × operator four-slice table

| unit | family | program | census | r1s / r1d / r2s / r2d | meet | grade | pooled | coarsest n | margin | ≥2× |
|---|---|---|---|---|---|---|---|---|---|---|
| GunPointAgeSpan__impulse_v2 | GunPointFamily | hampel_filter | LEARNABLE | 0.5000* / 0.3000* / 0.3000* / 0.4000* | 4/4 | **ROBUST_LEARNABLE** | 0.3750 | 10 | 3.75 | yes |
| GunPoint__impulse_v2 | GunPointFamily | hampel_filter | LEARNABLE | 0.2500* / 0.7500* / 0.5000* / 0.3333* | 4/4 | **ROBUST_LEARNABLE** | 0.4667 | 3 | 1.40 | no |
| ECG200__impulse_v2 | ECGFamily | repair_burst_segment | HELDOUT_ONLY | 0.1111* / 0.0000 / 0.0000 / -0.1429 | 1/4 | **FRAGILE** | 0.0000 | 7 | 0.00 | no |
| ToeSegmentation1__impulse_v2 | ToeSegmentationFamily | repair_burst_segment | LEARNABLE | 0.0000 / 0.0000 / -0.5000 / 1.0000* | 1/4 | **FRAGILE** | 0.0833 | 2 | 0.17 | no |
| Lightning2__impulse_v2 | Lightning2 | repair_burst_segment | LEARNABLE | 0.2000 / 0.0000 / 0.5000* / 0.0000 | 1/4 | **FRAGILE** | 0.1667 | 4 | 0.67 | no |
| Herring__impulse_v2 | Herring | hampel_filter | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0000 | 5 | 0.00 | no |
| GunPoint__burst_cls2 | GunPointFamily | outlier_iqr | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0000 | 3 | 0.00 | no |
| BeetleFly__burst_cls2 | BeetleFly | outlier_iqr | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / — | 0/4 | **UNREADABLE** | 0.0000 | 2 | 0.00 | no |
| BeetleFly__burst_cls2 | BeetleFly | hampel_filter | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / — | 0/4 | **UNREADABLE** | 0.0000 | 2 | 0.00 | no |
| DistalPhalanxOutlineCorrect__impulse_v2 | PhalanxFamily | outlier_mad | HELDOUT_ONLY | 0.0435* / 0.0444* / -0.0889 / 0.0227 | 2/4 | **FRAGILE** | 0.0056 | 44 | 0.24 | no |
| DistalPhalanxOutlineCorrect__burst_cls2 | PhalanxFamily | outlier_iqr | LEARNABLE | 0.0217 / 0.0889* / 0.0000 / 0.0227* | 2/4 | **FRAGILE** | 0.0333 | 44 | 1.47 | no |
| DistalPhalanxOutlineCorrect__burst_cls2 | PhalanxFamily | outlier_mad | LEARNABLE | 0.0217 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0056 | 44 | 0.24 | no |
| ECGFiveDays__impulse_v2 | ECGFamily | repair_burst_segment | LEARNABLE | 0.0000 / 1.0000* / 0.5000* / 1.0000* | 3/4 | **ROBUST_LEARNABLE** | 0.5714 | 1 | 0.57 | no |
| GunPointMaleVersusFemale__impulse_v2 | GunPointFamily | hampel_filter | LEARNABLE | 0.1818* / 0.0000 / 0.2000* / 0.2222* | 3/4 | **ROBUST_LEARNABLE** | 0.1500 | 9 | 1.35 | no |
| GunPointMaleVersusFemale__burst_cls2 | GunPointFamily | outlier_iqr | LEARNABLE | 0.0000 / 0.2000* / 0.1000 / 0.1111 | 1/4 | **FRAGILE** | 0.1000 | 9 | 0.90 | no |
| GunPointMaleVersusFemale__burst_cls2 | GunPointFamily | hampel_filter | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0000 | 9 | 0.00 | no |
| GunPointMaleVersusFemale__burst_cls2 | GunPointFamily | repair_level_shift | LEARNABLE | 0.0000 / 0.2000* / 0.2000* / 0.0000 | 2/4 | **FRAGILE** | 0.1000 | 9 | 0.90 | no |
| GunPointOldVersusYoung__impulse_v2 | GunPointFamily | hampel_filter | LEARNABLE | 0.6364* / 0.3000* / 0.5000* / 0.2000* | 4/4 | **ROBUST_LEARNABLE** | 0.4146 | 10 | 4.15 | yes |
| MiddlePhalanxOutlineCorrect__impulse_v2 | PhalanxFamily | repair_level_shift | HELDOUT_ONLY | 0.0222* / -0.0444 / 0.0000 / 0.0222 | 1/4 | **FRAGILE** | 0.0000 | 45 | 0.00 | no |
| MoteStrain__impulse_v2 | MoteStrain | hampel_filter | HELDOUT_ONLY | -0.5000 / 0.5000* / 0.0000 / — | 1/4 | **FRAGILE** | 0.0000 | 2 | 0.00 | no |
| PowerCons__impulse_v2 | PowerCons | hampel_filter | LEARNABLE | 0.1429* / 0.4286* / 0.2143* / 0.0000 | 3/4 | **ROBUST_LEARNABLE** | 0.2037 | 12 | 2.44 | yes |
| PowerCons__burst_cls2 | PowerCons | hampel_filter | LEARNABLE | 0.1429* / 0.3571* / 0.2143* / 0.0000 | 3/4 | **ROBUST_LEARNABLE** | 0.1852 | 12 | 2.22 | yes |
| ProximalPhalanxOutlineCorrect__burst_cls2 | PhalanxFamily | winsorize | LEARNABLE | 0.0217 / 0.0217 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0111 | 44 | 0.49 | no |
| ProximalPhalanxOutlineCorrect__burst_cls2 | PhalanxFamily | outlier_iqr | LEARNABLE | 0.0217 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0056 | 44 | 0.24 | no |
| ProximalPhalanxOutlineCorrect__burst_cls2 | PhalanxFamily | outlier_mad | LEARNABLE | 0.0217 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0056 | 44 | 0.24 | no |
| SonyAIBORobotSurface2__burst_cls2 | SonyAIBOFamily | hampel_filter | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0000 | 1 | 0.00 | no |
| ToeSegmentation2__burst_cls2 | ToeSegmentationFamily | hampel_filter | HELDOUT_ONLY | 0.0000 / 0.0000 / 0.0000 / 0.0000 | 0/4 | **UNREADABLE** | 0.0000 | 2 | 0.00 | no |
| ToeSegmentation2__burst_cls2 | ToeSegmentationFamily | repair_burst_segment | HELDOUT_ONLY | 0.2500* / -0.5000 / -0.5000 / 0.0000 | 1/4 | **FRAGILE** | -0.1000 | 2 | -0.20 | no |
| TwoLeadECG__impulse_v2 | TwoLeadECG | outlier_mad | LEARNABLE | 0.0000 / 0.5000* / -0.5000 / 1.0000* | 2/4 | **FRAGILE** | 0.1429 | 1 | 0.14 | no |

A trailing `*` on a slice reading means it met that slice's materiality line.  GPA is the designated hampel ROBUST anchor from the PS-0 re-earn; the table still shows the recomputed grade.

### Slice sizes (sealed = rebuilt)

| unit | r1_support | r1_delayed | r2_support | r2_delayed |
|---|---|---|---|---|
| GunPointAgeSpan__impulse_v2 | 10 | 10 | 10 | 10 |
| GunPoint__impulse_v2 | 4 | 4 | 4 | 3 |
| ECG200__impulse_v2 | 9 | 7 | 7 | 7 |
| ToeSegmentation1__impulse_v2 | 4 | 4 | 2 | 2 |
| Lightning2__impulse_v2 | 5 | 5 | 4 | 4 |
| Herring__impulse_v2 | 5 | 5 | 5 | 5 |
| GunPoint__burst_cls2 | 4 | 4 | 4 | 3 |
| BeetleFly__burst_cls2 | 2 | 2 | 2 | 0 |
| DistalPhalanxOutlineCorrect__impulse_v2 | 46 | 45 | 45 | 44 |
| DistalPhalanxOutlineCorrect__burst_cls2 | 46 | 45 | 45 | 44 |
| ECGFiveDays__impulse_v2 | 2 | 2 | 2 | 1 |
| GunPointMaleVersusFemale__impulse_v2 | 11 | 10 | 10 | 9 |
| GunPointMaleVersusFemale__burst_cls2 | 11 | 10 | 10 | 9 |
| GunPointOldVersusYoung__impulse_v2 | 11 | 10 | 10 | 10 |
| MiddlePhalanxOutlineCorrect__impulse_v2 | 45 | 45 | 45 | 45 |
| MoteStrain__impulse_v2 | 2 | 2 | 2 | 0 |
| PowerCons__impulse_v2 | 14 | 14 | 14 | 12 |
| PowerCons__burst_cls2 | 14 | 14 | 14 | 12 |
| ProximalPhalanxOutlineCorrect__burst_cls2 | 46 | 46 | 44 | 44 |
| SonyAIBORobotSurface2__burst_cls2 | 3 | 2 | 2 | 1 |
| ToeSegmentation2__burst_cls2 | 4 | 2 | 2 | 2 |
| TwoLeadECG__impulse_v2 | 2 | 2 | 2 | 1 |

## 3. Clusters (ROBUST members, independence, Scope)

### `hampel_filter`

- ROBUST 6 / FRAGILE 1 / UNREADABLE 5 (of 12 oracle pairs)
- independent ROBUST families **2**: GunPointFamily, PowerCons
- five-axis Scope: **SCOPE_INTERSECTION_USABLE** (leaves beyond task_kind: estimated_level_offset, estimated_region_end_fraction, level_excursion_score, level_region_end_fraction, level_region_fraction, local_robust_z_peak, longest_missing_run_fraction, missing_fraction, outlier_region_end_fraction, period_evidence_status, period_reliability)
- dual-source eligible: **yes**
- ROBUST: GunPointAgeSpan__impulse_v2, GunPoint__impulse_v2, GunPointMaleVersusFemale__impulse_v2, GunPointOldVersusYoung__impulse_v2, PowerCons__impulse_v2, PowerCons__burst_cls2
- FRAGILE: MoteStrain__impulse_v2
- UNREADABLE: Herring__impulse_v2, BeetleFly__burst_cls2, GunPointMaleVersusFemale__burst_cls2, SonyAIBORobotSurface2__burst_cls2, ToeSegmentation2__burst_cls2

### `repair_burst_segment`

- ROBUST 1 / FRAGILE 4 / UNREADABLE 0 (of 5 oracle pairs)
- independent ROBUST families **1**: ECGFamily
- five-axis Scope: **SCOPE_INTERSECTION_USABLE** (leaves beyond task_kind: estimated_level_offset, estimated_region_end_fraction, estimated_region_start_fraction, level_excursion_score, level_region_end_fraction, level_region_fraction, local_robust_z_peak, longest_missing_run_fraction, missing_fraction, outlier_region_end_fraction, period_change_score, period_evidence_status, period_reliability)
- dual-source eligible: **no**
- ROBUST: ECGFiveDays__impulse_v2
- FRAGILE: ECG200__impulse_v2, ToeSegmentation1__impulse_v2, Lightning2__impulse_v2, ToeSegmentation2__burst_cls2
- UNREADABLE: —

### `outlier_iqr`

- ROBUST 0 / FRAGILE 2 / UNREADABLE 3 (of 5 oracle pairs)
- independent ROBUST families **0**: —
- five-axis Scope: **SCOPE_INTERSECTION_TOO_WIDE** (leaves beyond task_kind: none)
- dual-source eligible: **no**
- ROBUST: —
- FRAGILE: DistalPhalanxOutlineCorrect__burst_cls2, GunPointMaleVersusFemale__burst_cls2
- UNREADABLE: GunPoint__burst_cls2, BeetleFly__burst_cls2, ProximalPhalanxOutlineCorrect__burst_cls2

### `outlier_mad`

- ROBUST 0 / FRAGILE 2 / UNREADABLE 2 (of 4 oracle pairs)
- independent ROBUST families **0**: —
- five-axis Scope: **SCOPE_INTERSECTION_TOO_WIDE** (leaves beyond task_kind: none)
- dual-source eligible: **no**
- ROBUST: —
- FRAGILE: DistalPhalanxOutlineCorrect__impulse_v2, TwoLeadECG__impulse_v2
- UNREADABLE: DistalPhalanxOutlineCorrect__burst_cls2, ProximalPhalanxOutlineCorrect__burst_cls2

### `repair_level_shift`

- ROBUST 0 / FRAGILE 2 / UNREADABLE 0 (of 2 oracle pairs)
- independent ROBUST families **0**: —
- five-axis Scope: **SCOPE_INTERSECTION_TOO_WIDE** (leaves beyond task_kind: none)
- dual-source eligible: **no**
- ROBUST: —
- FRAGILE: GunPointMaleVersusFemale__burst_cls2, MiddlePhalanxOutlineCorrect__impulse_v2
- UNREADABLE: —

### `winsorize`

- ROBUST 0 / FRAGILE 0 / UNREADABLE 1 (of 1 oracle pairs)
- independent ROBUST families **0**: —
- five-axis Scope: **SCOPE_INTERSECTION_TOO_WIDE** (leaves beyond task_kind: none)
- dual-source eligible: **no**
- ROBUST: —
- FRAGILE: —
- UNREADABLE: ProximalPhalanxOutlineCorrect__burst_cls2

## 4. Named-cluster readout

### hampel

GPA (`GunPointAgeSpan__impulse_v2`) recomputed grade: **ROBUST_LEARNABLE** (4/4, pooled +0.375, margin 3.75×).  PS-0 re-earn remains the live-source fact.
ROBUST members: GunPointAgeSpan__impulse_v2, GunPoint__impulse_v2, GunPointMaleVersusFemale__impulse_v2, GunPointOldVersusYoung__impulse_v2, PowerCons__impulse_v2, PowerCons__burst_cls2.
Independent ROBUST families: GunPointFamily, PowerCons.

PowerCons impulse is **ROBUST_LEARNABLE 3/4** on the *oracle operator* (readings +0.143 / +0.429 / +0.214 / 0.000, margin 2.44×).  That is a different object from the cancelled S1c episode (live Support +0.0714 = 1/14, re-earn Support 0.0).  This book does not recycle that episode.  The unit-level confirmation surface is what the 3/4 rule scores.

### repair_burst (Toe1 / Lightning2 / ECGFiveDays focus)

- `ToeSegmentation1__impulse_v2`: grade **FRAGILE**, meet 1/4, pooled 0.0833, margin 0.17, slices 0.0000 / 0.0000 / -0.5000 / 1.0000, half-grade FRAGILE
- `Lightning2__impulse_v2`: grade **FRAGILE**, meet 1/4, pooled 0.1667, margin 0.67, slices 0.2000 / 0.0000 / 0.5000 / 0.0000, half-grade FRAGILE
- `ECGFiveDays__impulse_v2`: grade **ROBUST_LEARNABLE**, meet 3/4, pooled 0.5714, margin 0.57, slices 0.0000 / 1.0000 / 0.5000 / 1.0000, half-grade ROBUST_LEARNABLE

Cluster ROBUST: ECGFiveDays__impulse_v2.  Independent families: ECGFamily.

ECGFiveDays is ROBUST by the frozen 3/4 count, but the coarsest slice is **1 row** (materiality 1.0) so the reproducibility margin is 0.57×.  Two of the three hits are 1.0 on n=1 or n=2.  Do not treat it as a high-quality source.  Toe1 and Lightning2 are one-slice FRAGILE (the +0.083 / +0.167 census LEARNABLE labels were pooled-pool illusions at this resolution).

## 5. Dual-source verdict

**SECOND_SOURCE_AVAILABLE**

ROBUST dual source exists in: hampel_filter

### PS-1 unlock path

- cluster `hampel_filter`
- sources: GunPointAgeSpan__impulse_v2 (GunPointFamily), PowerCons__impulse_v2 (PowerCons)
- suggested exam: **GunPointOldVersusYoung__impulse_v2** — original PS-1 exam, still in-cluster and ROBUST 4/4; same name-family as GPA so the report cannot claim a cross-family capability
- Available cluster is still hampel.  Keep GPOVY as the exam.  GPA stays source A (re-earned episode).  PowerCons is the independent second *unit*; the S1c PowerCons episode stays cancelled and is not recycled.

Layer split: this verdict is **unit × oracle-operator** confirmation-surface robustness.  It does not restore the cancelled S1c PowerCons episode.  A PS-1 that requires two live Episodes still needs a new PowerCons earn of the oracle-default (or a stable) hampel; hypothesis cards cannot be compiled from the sealed oracle itself.

## 6. Protocol variant (report only, not adopted)

If the four quarter slices are collapsed to two halves (r1_support+r1_delayed and r2_support+r2_delayed; Support surface doubles) and graded 2/2 ROBUST / 1/2 FRAGILE / 0/2 UNREADABLE:

| grade | quarters (frozen) | halves (variant) |
|---|---|---|
| ROBUST_LEARNABLE | 7 | 10 |
| FRAGILE | 11 | 8 |
| UNREADABLE | 11 | 11 |

Transitions (quarter → half):

| unit × program | quarter | half |
|---|---|---|
| DistalPhalanxOutlineCorrect__burst_cls2 × outlier_iqr | FRAGILE | **ROBUST_LEARNABLE** |
| GunPointMaleVersusFemale__burst_cls2 × outlier_iqr | FRAGILE | **ROBUST_LEARNABLE** |
| GunPointMaleVersusFemale__burst_cls2 × repair_level_shift | FRAGILE | **ROBUST_LEARNABLE** |
| MiddlePhalanxOutlineCorrect__impulse_v2 × repair_level_shift | FRAGILE | **UNREADABLE** |
| MoteStrain__impulse_v2 × hampel_filter | FRAGILE | **UNREADABLE** |
| ProximalPhalanxOutlineCorrect__burst_cls2 × winsorize | UNREADABLE | **FRAGILE** |
| ProximalPhalanxOutlineCorrect__burst_cls2 × outlier_iqr | UNREADABLE | **FRAGILE** |
| ProximalPhalanxOutlineCorrect__burst_cls2 × outlier_mad | UNREADABLE | **FRAGILE** |
| ToeSegmentation2__burst_cls2 × repair_burst_segment | FRAGILE | **UNREADABLE** |

Halving the slices (doubling each confirmation surface) is a protocol change, not a finding about the current course.  Under halves, hampel stays dual-source and outlier_iqr newly becomes dual-source eligible (Distal burst + GPMVF burst).  Burst does not: Toe1 and Lightning2 stay FRAGILE.  Adopt only if sol wants the confirmation surface itself enlarged; it does not create a second hampel family, and it does not rescue the burst +0.571 as a high-quality source.

## 7. Cost

- Fast LLM: 0 / 0
- Consumer fits: 51 / 300
- wall clock: 9.55 s / 3600 s
- downloads: 0
- pairs scored: 29

## 8. Obligations

- **no_llm**: True
- **no_downloads**: True
- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **existing_runners_unmodified**: True
- **sealed_oracles_read_only**: True
- **oracle_isolated**: True
- **artifacts_isolated_from_arm_view**: True
- **curriculum_and_budgets_unmodified**: True
- **full_repo_pytest_not_run**: True
- **fit_budget_held**: True
- **wall_clock_held**: True
- **slice_rows_verified_against_sealed**: True
- **protocol_variant_not_adopted**: True

## 9. Outside the book

- Adapter scoring applies the workflow to the fit cohort only; slices stay unprocessed.  That is the live confirmation surface for fit_only_artifact, not a new instrument.
- ECGFiveDays held-in pool is 7 rows (slices 2/2/2/1).  A pooled +0.571 can still be UNREADABLE or FRAGILE at slice resolution because 1/n on a 1-row slice is 1.0.
- Half-protocol grades are report-only.  They were not used for the frozen verdict.
- GPA is listed as the designated hampel anchor from the PS-0 re-earn (Support +0.40 / delayed +0.40).  The table reports the recomputed four-slice grade without overriding it.
