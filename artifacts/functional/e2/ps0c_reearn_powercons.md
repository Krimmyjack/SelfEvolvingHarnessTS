# PS-0c -- re-earn PowerCons source B

protocol: `ps0c_reearn_powercons_v1`  evidence grade: **development-mechanism**  git: `346d30c2b895c952f4ace5719359b86aba7f92e3`  backend: **gpt-5.6-sol**

**SOURCES_REEARNED_SCOPE_USABLE**

all five axes intersect and the Pattern intersection carries 19 leaf or leaves beyond the eligibility gate

source A' (GPA, ps0_srcA_1) was earned on gpt-5.6-sol@agicto.  Episode validity is the consumer Support / delayed reading and does not depend on the relay.  Proposal-behaviour differences are absorbed by the PS-1 three-arm same-backend contrast.  This book does not fall back to the old relay.

## Protocol

- unit: `PowerCons__impulse_v2`
- arm: A3-reset, isomorphic to PS-0 / S1c unit protocol
- run-ids: ps0_srcB_3 / ps0_srcB_4 (≤2, stop on first earn)
- no hinting of prompt, budget, or candidate cap

## Per-run proposal ledger

#### `ps0_srcB_3` -- miss

| round | candidate id | operators | family | chosen | outcome | gain |
|---|---|---|---|---|---|---|
| r1 | `identity` | - | identity | True | dropped_before_probe_no_compiled_steps | - |
| r1 | `iqr_local_deviation_repair` | outlier_iqr | outlier_threshold | False | probe | -0.3571 |
| r2 | `identity` | - | identity | True | dropped_before_probe_no_compiled_steps | - |
| r2 | `repair_localized_level_excursion` | repair_level_shift | level_shift | False | probe | -0.3571 |

- families proposed: ['level_shift', 'outlier_threshold']
- target family proposed: False
- cost: LLM 7, fits 5, 173.8 s

| round | workflow | relation | Support | delayed |
|---|---|---|---|---|
| r1 | `outlier_iqr` | NEGATIVE | -0.3571428571428572 | None |
| r2 | `repair_level_shift` | NEGATIVE | -0.3571428571428572 | None |

#### `ps0_srcB_4` -- earned

| round | candidate id | operators | family | chosen | outcome | gain |
|---|---|---|---|---|---|---|
| r1 | `identity` | - | identity | True | dropped_before_probe_no_compiled_steps | - |
| r1 | `local_hampel_repair` | hampel_filter | hampel | False | probe | 0.0714 |
| r2 | `identity` | - | identity | False | dropped_before_probe_no_compiled_steps | - |
| r2 | `repair_local_level_shift` | repair_level_shift | level_shift | False | probe | -0.3571 |
| r2 | `cand_skill_fast_winner_classification_ridge_raw_plus_difference_v1_accuracy_hampel_filter` | hampel_filter | hampel | True | probe | 0.0000 |

- families proposed: ['hampel', 'level_shift']
- target family proposed: True
- cost: LLM 7, fits 9, 180.9 s

| round | workflow | relation | Support | delayed |
|---|---|---|---|---|
| r1 | `hampel_filter` | POSITIVE | 0.0714285714285714 | 0.49999999999999994 |
| r2 | `hampel_filter` | NEUTRAL | 0.0 | None |
| r2 | `repair_level_shift` | NEGATIVE | -0.3571428571428572 | None |

## Part 0 re-verification

- verdict: **SCOPE_INTERSECTION_USABLE**
- all five axes intersect and the Pattern intersection carries 19 leaf or leaves beyond the eligibility gate

| axis | intersection | agree |
|---|---|---|
| task_kind | classification | True |
| consumer_id | ridge-raw-plus-difference-v1 | True |
| metric | accuracy | True |
| program_geometry | hampel_filter | True |
| deployment_visible_pattern_intersection | ['clipping_probe_direction', 'denoising_probe_direction', 'estimated_level_offset', 'estimated_region_end_fraction', 'estimated_region_start_fraction', 'imputation_probe_direction', 'level_excursion_score', 'level_only_post_shift_support_sufficient', 'level_probe_direction', 'level_region_end_fraction', 'level_region_fraction', 'local_robust_z_peak', 'longest_missing_run_fraction', 'missing_fraction', 'outlier_region_end_fraction', 'period_evidence_status', 'period_reliability', 'period_repair_available', 'post_shift_support_sufficient'] | True |

## Cost (book so far)

- LLM: 81 / 180
- Consumer fits: 26 / 160
- wall clock: 5729.1 s / 9000 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **production_governance_unmodified**: True
- **protocol_isomorphic_to_ps0_s1c_unit**: True
- **no_hinting_of_prompt_budget_or_candidate_cap**: True
- **scenes_stopped_on_first_earn**: True
- **live_backend**: gpt-5.6-sol
- **new_relay_only_no_agicto_fallback**: True
- **secret_key_not_written**: True
- **downloads**: 0
- **sealed_artifacts_not_read**: True
- **oracle_isolation_holds**: True
- **stage_report_not_written**: False
- **full_repo_pytest_not_run**: True

## Outside the book

- PS-1 resumed via --ps1-only after a runner-level applicability shape fix; PowerCons was not re-earned
