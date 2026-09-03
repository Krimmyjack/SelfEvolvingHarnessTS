# PS-0 -- record-layer repair and dual-source re-earn

protocol: `ps0_reearn_sources_v1`  evidence grade: **development-mechanism**  git: `94cf58eb5bef6ad27c3f1392d1d2ae2f22c20b86`  backend: **gpt-5.6-sol**

**PS1_SOURCES_NOT_REEARNED**

only 1 of 2 scenes re-earned the target family; a two-source hypothesis needs both

## Part 1 -- what the round record now keeps

- **field_1_fast_features_binned**: the binned observable-contract projection of the exact mapping _run_round hands to run_online_round as fast_features; not a recomputation
- **field_2_proposals**: every candidate the proposal stage named, with its compiled steps, operators, family, whether select chose it and how it ended: probed, verifier_rejected, dropped without compiled steps, or never reached because the Support budget ran out
- **family_tagging_fix**: family now comes from compiled steps.  S1c tagged from the candidate id, and the ids the Fast Agent invents carry no operator word, so every S1c probe recorded an empty operator list
- **behaviour_change**: none; both fields are additive
- **cross_check**: tests/functional/test_round_record_persists_context_and_proposals.py

## Part 2 -- re-earn, take what comes

| scene | unit | outcome | attempts | earned in | Support | delayed | target family proposed |
|---|---|---|---|---|---|---|---|
| source_A_prime | GunPointAgeSpan__impulse_v2 | **EARNED** | 1 | ps0_srcA_1 | 0.4 | 0.4 | True |
| source_B_prime | PowerCons__impulse_v2 | **MISS** | 2 | - | - | - | True |

### Per-run proposal ledger

#### `ps0_srcA_1` -- GunPointAgeSpan__impulse_v2 (earned)

| round | candidate id | operators | family | chosen | outcome | gain |
|---|---|---|---|---|---|---|
| r1 | `identity` | - | identity | False | dropped_before_probe_no_compiled_steps | - |
| r1 | `intrinsic-extreme-deviation-repair` | winsorize | outlier_threshold | False | verifier_rejected | - |
| r1 | `intrinsic-level-excursion-repair` | repair_level_shift | level_shift | True | verifier_rejected | - |
| r2 | `identity` | - | identity | False | dropped_before_probe_no_compiled_steps | - |
| r2 | `repair_local_level_shift` | repair_level_shift | level_shift | True | verifier_rejected | - |
| r2 | `hampel_extreme_deviation` | hampel_filter | hampel | False | probe | 0.4000 |

- families proposed: ['hampel', 'level_shift', 'outlier_threshold']
- deploy: FROZEN_ACTIVE_SKILL_RECALL, gain 0.2689873417721519
- cost: LLM 10, fits 6, 249.4 s

#### `ps0_srcB_1` -- PowerCons__impulse_v2 (miss)

| round | candidate id | operators | family | chosen | outcome | gain |
|---|---|---|---|---|---|---|
| r1 | `identity` | - | identity | True | dropped_before_probe_no_compiled_steps | - |
| r1 | `localized_level_shift_repair` | repair_level_shift | level_shift | False | probe | -0.3571 |
| r2 | `identity` | - | identity | True | dropped_before_probe_no_compiled_steps | - |
| r2 | `localized_level_shift_repair` | repair_level_shift | level_shift | False | probe | -0.3571 |

- families proposed: ['level_shift']
- deploy: FROZEN_LEDGER_NO_INCUMBENT_IDENTITY, gain 0.0
- cost: LLM 9, fits 5, 97.9 s

#### `ps0_srcB_2` -- PowerCons__impulse_v2 (miss)

| round | candidate id | operators | family | chosen | outcome | gain |
|---|---|---|---|---|---|---|
| r1 | `identity` | - | identity | False | dropped_before_probe_no_compiled_steps | - |
| r1 | `repair_local_level_excursion` | repair_level_shift | level_shift | False | probe | -0.3571 |
| r1 | `repair_extreme_deviation_iqr` | outlier_iqr | outlier_threshold | True | probe | -0.3571 |
| r2 | `identity` | - | identity | True | dropped_before_probe_no_compiled_steps | - |
| r2 | `repair_localized_level_shift` | repair_level_shift | level_shift | False | probe | -0.3571 |
| r2 | `hampel_localized_extreme_deviation` | hampel_filter | hampel | False | probe | 0.0000 |

- families proposed: ['hampel', 'level_shift', 'outlier_threshold']
- deploy: FROZEN_LEDGER_NO_INCUMBENT_IDENTITY, gain 0.0
- cost: LLM 8, fits 7, 130.5 s

## Part 0 re-verification (axis 5 from the fresh records)

- verdict: **PS1_SOURCES_NOT_REEARNED**
- only 1 of 2 scenes re-earned the target family; a two-source hypothesis needs both

## Cost

- LLM: 27 / 220
- Consumer fits: 18 / 200
- wall clock: 492.7 s / 10800 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **production_governance_unmodified**: True
- **record_repair_is_additive_only**: True
- **protocol_isomorphic_to_s1c_unit**: True
- **no_hinting_of_prompt_budget_or_candidate_cap**: True
- **scenes_stopped_on_first_earn**: True
- **live_backend**: gpt-5.6-sol
- **downloads**: 0
- **sealed_artifacts_not_read**: True
- **oracle_isolation_holds**: True
- **stage_report_not_written**: True
- **full_repo_pytest_not_run**: True
