# PS-1 -- proposal shift under a Scoped hypothesis (pilot)

protocol: `ps1_proposal_shift_v2_pilot`  evidence grade: **development-mechanism (pilot)**  git: `346d30c2b895c952f4ace5719359b86aba7f92e3`  backend: **gpt-5.6-sol**

**NO_PROPOSAL_SHIFT**

the three arms' proposal distributions overlap: {'A3': '0/4', 'A5-neutral': '0/4', 'A5-scoped': '0/4'}

> Pilot grade.  This result freezes no production design.  GunPointOldVersusYoung__impulse_v2 shares GunPointFamily with source A, so it isolates a mechanism and is not a cross-family transfer claim.  A positive earned under the card is a Target-local Skill and counts zero toward any cross-domain authorization for the Source Skill.

## Cards (SkillEntry, existing authority fields)

| field | A5-scoped | A5-neutral |
|---|---|---|
| skill_id | `ps1_source_hypothesis_scoped_v1` | `ps1_source_hypothesis_neutral_v1` |
| schema_version | `skill-entry/1` | `skill-entry/1` |
| skill_kind | `capability` | `capability` |
| revision | `1` | `1` |
| allowed_tools | [] | [] |
| authority.reorders_supplied_candidates | **False** | **False** |
| authority.supplies_candidates | **True** | **False** |
| authority.suppresses_operators | **False** | **False** |
| authority.grants_execution | **False** | **False** |
| observable_applicability | {"all": [{"feature": "task_kind", "op": "==", "value": "classification"}, {"feature": "clipping_probe_direction", "op": "==", "value": "unknown"}, {"feature": "denoising_probe_direction", "op": "==", "value": "unknown"}, {"feature": "estimated_level_offset", "op": "==", "value": "low"}, {"feature": "estimated_region_end_fraction", "op": "==", "value": "high"}, {"feature": "estimated_region_start_fraction", "op": "==", "value": "very_low"}, {"feature": "imputation_probe_direction", "op": "==", "value": "unknown"}, {"feature": "level_excursion_score", "op": "==", "value": "high"}, {"feature": "level_probe_direction", "op": "==", "value": "unknown"}, {"feature": "local_robust_z_peak", "op": "==", "value": "high"}, {"feature": "longest_missing_run_fraction", "op": "==", "value": "zero"}, {"feature": "missing_fraction", "op": "==", "value": "zero"}, {"feature": "period_evidence_status", "op": "==", "value": "OK"}, {"feature": "period_reliability", "op": "==", "value": "high"}, {"feature": "period_repair_available", "op": "==", "value": false}, {"feature": "post_shift_support_sufficient", "op": "==", "value": false}]} | identical |

### Card audit

- **scoped_body_tokens**: 185
- **neutral_body_tokens**: 170
- **token_ratio**: 0.9189
- **token_ratio_within_tolerance**: True
- **neutral_names_no_operator**: False
- **neutral_operator_hits**: ['hampel_filter']
- **neutral_names_no_program_family**: False
- **neutral_family_hits**: ['hampel']
- **neutral_all_authority_false**: True
- **scoped_authority**: {'reorders_supplied_candidates': False, 'supplies_candidates': True, 'suppresses_operators': False, 'grants_execution': False}
- **neutral_authority**: {'reorders_supplied_candidates': False, 'supplies_candidates': False, 'suppresses_operators': False, 'grants_execution': False}
- **scoped_opens_only_supplies_candidates**: True
- **neither_card_supplies_a_frozen_program**: True
- **identical_applicability**: True
- **same_schema_and_kind**: True
- **machine_applicability_leaf_count**: 16
- **pattern_leaves_dropped_as_uncontracted_for_edit_schema**: ['level_only_post_shift_support_sufficient', 'level_region_end_fraction', 'level_region_fraction', 'outlier_region_end_fraction']
- **dropped_leaves_remain_in_body_and_scope_v1**: True

## Budget equality across the three arms

- all equal: **True**

- maximum_candidates equal: True (value 3)
- maximum_modified_fraction equal: True (value 0.1)
- support_trial_budget_per_round equal: True (value 2)
- rounds equal: True (value ['r1', 'r2'])
- llm_cap_per_run equal: True (value 12)
- fit_cap_per_run equal: True (value 10)

- the card carries no frozen program and no allowed tool, so it cannot reach _skill_frozen_candidates or _frozen_recall.  Any candidate it inspires is proposed by the same proposal stage and counts inside the same maximum_candidates cap; nothing is added outside it.

## Per-run readout

| run | arm | card served | proposed | selected | verifier | Support | delayed | deployed | gain | worst-class | LLM | fits | probes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ps1_run1 | A3 | - | False | False | False | False | False | False | +0.0000 | +0.0000 | 6 | 1 | 1 |
| ps1_run2 | A5-neutral | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 6 | 1 | 2 |
| ps1_run3 | A5-scoped | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 3 | 1 | 0 |
| ps1_run4 | A3 | - | False | False | False | False | False | False | +0.0000 | +0.0000 | 6 | 1 | 2 |
| ps1_run5 | A5-neutral | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 3 | 1 | 0 |
| ps1_run6 | A5-scoped | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 5 | 1 | 1 |
| ps1_run7 | A3 | - | False | False | False | False | False | False | +0.0000 | +0.0000 | 7 | 1 | 2 |
| ps1_run8 | A5-neutral | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 4 | 1 | 0 |
| ps1_run9 | A5-scoped | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 6 | 1 | 1 |
| ps1_run10 | A3 | - | False | False | False | False | False | False | +0.0000 | +0.0000 | 8 | 1 | 2 |
| ps1_run11 | A5-neutral | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 8 | 1 | 2 |
| ps1_run12 | A5-scoped | yes | False | False | False | False | False | False | +0.0000 | +0.0000 | 5 | 1 | 0 |

## Three-arm aggregate

| arm | proposal rate | selected | verifier | Support | delayed | deployed | harm runs | mean LLM | mean fits | mean probes | other families |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A3 | **0/4** | 0 | 0 | 0 | 0 | 0 | 0 | 6.75 | 1.0 | 1.75 | burst, outlier_threshold |
| A5-neutral | **0/4** | 0 | 0 | 0 | 0 | 0 | 0 | 5.25 | 1.0 | 1.0 | burst, outlier_threshold |
| A5-scoped | **0/4** | 0 | 0 | 0 | 0 | 0 | 0 | 4.75 | 1.0 | 0.5 | outlier_threshold |

### Cost to the first effective Skill

- **A3**: 0 of 4 runs reached one
- **A5-neutral**: 0 of 4 runs reached one
- **A5-scoped**: 0 of 4 runs reached one

## Cost

- LLM: 81 / 180
- Consumer fits: 26 / 160
- wall clock: 5729.1 s / 9000 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **production_governance_unmodified**: True
- **no_new_skill_class_or_permission_platform**: True
- **card_is_a_plain_skill_entry**: True
- **experimental_prior_slot**: True
- **budgets_equal_across_arms**: True
- **neither_card_supplies_a_frozen_program**: True
- **guided_positive_counts_zero_toward_cross_domain_authorization**: True
- **pilot_grade_freezes_no_production_design**: True
- **gray_zone_appends_no_batch**: True
- **arms_run**: 12
- **downloads**: 0
- **oracle_isolation_holds**: True
- **stage_report_not_written**: False
- **full_repo_pytest_not_run**: True
- **new_relay_only_no_agicto_fallback**: True
- **secret_key_not_written**: True

## Outside the book

- runner-level fix: machine applicability drops leaves that contracts/observables.py lists but observable_feature_v1.json does not; body and scope_v1 still carry the full intersection
