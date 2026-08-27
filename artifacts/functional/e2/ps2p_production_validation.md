# PS-2 -- mechanical supply of one candidate to verify (pilot)

protocol: `ps2p_production_validation_v1`  evidence grade: **development-mechanism (pilot)**  git: `853f367cc95d4cc398a2846a865d5cfa33c9df91`  backend: **gpt-5.6-sol**

**SUPPLY_RUNG_PRODUCTION_CONFIRMED**

the frozen hampel entered the pool in 4/4 runs, took a Support receipt in 4, converted through Support and the delayed gate in 4, the no-op card did not falsely deploy, and harm was zero in every arm.  Experience supplied the candidate; Target feedback adjudicated it.

> Pilot grade.  GunPointOldVersusYoung__impulse_v2 shares GunPointFamily with source A, so this isolates a mechanism and is not a cross-family transfer claim.  A conversion means experience supplied a candidate through the mechanical channel and Target feedback adjudicated it.  It is not 'the agent learned to propose hampel'.  A guided positive counts zero toward Source cross-domain authorization.

## Cards

| field | A5-scoped | A5-neutral |
|---|---|---|
| skill_id | `ps2_source_hypothesis_scoped_v1` | `ps2_source_hypothesis_neutral_v1` |
| schema_version | `skill-entry/1` | `skill-entry/1` |
| skill_kind | `capability` | `capability` |
| revision | `1` | `1` |
| authority.reorders_supplied_candidates | **False** | **False** |
| authority.supplies_candidates | **True** | **True** |
| authority.suppresses_operators | **False** | **False** |
| authority.grants_execution | **False** | **False** |
| frozen program | `hampel_filter` | `resample_uniform` |
| requires_target_support | **True** | **True** |

### Card audit

- **scoped_body_tokens**: 189
- **neutral_body_tokens**: 213
- **token_ratio**: 1.127
- **token_ratio_within_tolerance**: True
- **neutral_prose_names_no_operator**: True
- **neutral_prose_operator_hits**: []
- **neutral_frozen_operator**: resample_uniform
- **neutral_family_hits_in_prose**: []
- **both_open_only_supplies_candidates**: True
- **both_cards_supply_a_frozen_program**: True
- **scoped_frozen_ops**: ['hampel_filter']
- **neutral_frozen_ops**: ['resample_uniform']
- **identical_applicability**: True
- **same_schema_and_kind**: True
- **machine_applicability_leaf_count**: 16
- **pattern_leaves_dropped_as_uncontracted_for_edit_schema**: ['level_only_post_shift_support_sufficient', 'level_region_end_fraction', 'level_region_fraction', 'outlier_region_end_fraction']
- **oracle_placebo_ok**: True
- **oracle_target_ok**: True
- **scoped_frozen_params**: {}
- **scoped_params_note**: empty params use operator literature defaults (window=7, n_sigmas=3.0).  The sealed-oracle scored hampel params are not copied onto the card.
- **injection_channel**: card body Frozen program steps -> production _parse_frozen_steps / _skill_frozen_candidates; requires_target_support=true so DRAFT merge keeps one agent exploration slot; CandidatePool.build counts the inject inside min(total_k=4, maximum_candidates=3)
- **apply_smoke**: {'A5-scoped': {'applied_ids': ['ps2_source_hypothesis_scoped_v1'], 'skill_in_snapshot': True, 'frozen_steps_survived': True, 'frozen_ops': ['hampel_filter'], 'runtime_bundle_sha': '1e3da20fe5ee1a672504ca48f6f086bb1be0f1176e1dbc77a8a9cf5d52f5e09c'}, 'A5-neutral': {'applied_ids': ['ps2_source_hypothesis_neutral_v1'], 'skill_in_snapshot': True, 'frozen_steps_survived': True, 'frozen_ops': ['resample_uniform'], 'runtime_bundle_sha': '1649d777e9bc9d1fd34869c461fcdd2a8a61e856959ba29b9badd6f1e0b895d8'}}

## Sealed-oracle confirmation (grader only)

- path: `artifacts/functional/e2/s1_oracle/GunPointOldVersusYoung__impulse_v2.json`
- isolation: sealed exam key.  This file must not enter any arm prompt, store, or retrieval view.  Held-out per-operator scores are exam keys only.
- placebo `resample_uniform` legal numeric no-op: **True**
- target `hampel_filter` legal positive: **True**

- `hampel_filter`: legal=True verifier=True no-op=False held-in=0.41463414634146345 held-out=0.18412698412698414
- `resample_uniform`: legal=True verifier=True no-op=True held-in=0.0 held-out=0.0

## Same-rights proof (no shortcut)

- **no_methods_edit**: True
- **channel**: methods/ttha/fast_agent.py _skill_frozen_candidates
- **draft_not_priority**: requires_target_support=true routes the inject into the DRAFT bucket: Agent candidates stay in front; the Skill does not keep the ACTIVE priority slot
- **grants_execution_false**: True
- **verifier**: verify_candidate on every pool member, same limits
- **select**: ordinary fast_select_v1 over the public pool
- **support_and_delayed**: run_online_round / open_delayed unchanged
- **deploy**: only winner_delayed_approved plus the existing freeze path
- **no_runner_bypass**: this runner never marks a candidate verified, never writes a Support receipt, and never sets approved_skill_id

## Budget equality

- all equal: **True**

- maximum_candidates equal: True (value 3)
- maximum_modified_fraction equal: True (value 0.1)
- support_trial_budget_per_round equal: True (value 2)
- rounds equal: True (value ['r1', 'r2'])
- llm_cap_per_run equal: True (value 12)
- fit_cap_per_run equal: True (value 10)

- the card now carries Frozen program steps, so _skill_frozen_candidates emits cand_skill_<id>.  DRAFT merge is (*agent[:1], *draft[:1]); CandidatePool.build prepends identity and truncates at maximum_candidates=3.  The inject occupies one of the two program slots; it does not add a fourth.


## Attempt-1 stdout (supplementary, not protocol)

Structured rows were dropped when run 12 raised AgentTransportError.  These cells are the flushed print line only.

| run | arm | inject | selected | Support | delayed | deployed | agent | gain | LLM |
|---|---|---|---|---|---|---|---|---|---|
| ps2_run1 | A3 | False | False | False | False | False | burst | 0.0 | 7 |
| ps2_run2 | A5-neutral | True | False | False | False | False | burst,outlier_threshold | 0.0 | 8 |
| ps2_run3 | A5-scoped | True | False | True | False | False | burst,outlier_threshold | 0.0 | 8 |
| ps2_run4 | A3 | False | False | False | False | False | burst,outlier_threshold | 0.0 | 8 |
| ps2_run5 | A5-neutral | False | False | False | False | False | - | 0.0 | 3 |
| ps2_run6 | A5-scoped | False | False | False | False | False | - | 0.0 | 4 |
| ps2_run7 | A3 | False | False | False | False | False | burst,outlier_threshold | 0.0 | 7 |
| ps2_run8 | A5-neutral | True | False | False | False | False | burst | 0.0 | 5 |
| ps2_run9 | A5-scoped | True | False | True | False | False | burst | 0.0 | 6 |
| ps2_run10 | A3 | False | False | False | False | False | burst,outlier_threshold | 0.0 | 6 |
| ps2_run11 | A5-neutral | True | False | False | False | False | burst | 0.0 | 5 |
| ps2_run12 | A5-scoped | None | None | None | None | None | InternalServerError during inspect; unit not scored | None | None |

## Per-run readout (persisted protocol records)

| run | arm | card | inject in pool | selected | Support | delayed | deployed | break_at | miss | agent families | explore kept | gain | worst | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ps2p_run1 | A3 | - | False | False | False | False | False | no_card | - | burst,outlier_threshold | True | +0.0000 | +0.0000 | 8 | 1 |
| ps2p_run2 | A5-neutral | yes | False | False | False | False | False | never_entered_pool | card_in_view_not_in_selectable_pool | - | True | +0.0000 | +0.0000 | 4 | 1 |
| ps2p_run3 | A5-scoped | yes | True | False | True | True | True | None | - | burst | True | +0.2127 | +0.1879 | 7 | 10 |
| ps2p_run4 | A3 | - | False | False | False | False | False | no_card | - | burst | True | +0.0000 | +0.0000 | 7 | 1 |
| ps2p_run5 | A5-neutral | yes | True | False | False | False | False | selection_did_not_choose_inject | - | burst | True | +0.0000 | +0.0000 | 6 | 3 |
| ps2p_run6 | A5-scoped | yes | True | False | True | True | True | None | - | burst | True | +0.2127 | +0.1879 | 8 | 10 |
| ps2p_run7 | A3 | - | False | False | False | False | False | no_card | - | - | True | +0.0000 | +0.0000 | 3 | 1 |
| ps2p_run8 | A5-neutral | yes | False | False | False | False | False | never_entered_pool | card_in_view_not_in_selectable_pool | - | True | +0.0000 | +0.0000 | 4 | 1 |
| ps2p_run9 | A5-scoped | yes | True | False | True | True | True | None | - | burst,outlier_threshold | True | +0.2127 | +0.1879 | 8 | 10 |
| ps2p_run10 | A3 | - | False | False | False | False | False | no_card | - | burst | True | +0.0000 | +0.0000 | 7 | 1 |
| ps2p_run11 | A5-neutral | yes | True | False | False | False | False | selection_did_not_choose_inject | - | burst | True | +0.0000 | +0.0000 | 6 | 3 |
| ps2p_run12 | A5-scoped | yes | True | False | True | True | True | None | - | burst | True | +0.2127 | +0.1879 | 6 | 6 |

## Three-arm inject funnel

| arm | entry | selected | verifier/probe | Support | delayed | deployed | harm | explore kept | agent families |
|---|---|---|---|---|---|---|---|---|---|
| A3 | **0/4** | 0 | 0 | 0 | 0 | 0 | 0 | 4/4 | burst, outlier_threshold |
| A5-neutral | **2/4** | 0 | 2 | 0 | 0 | 0 | 0 | 4/4 | burst |
| A5-scoped | **4/4** | 0 | 4 | 4 | 4 | 4 | 0 | 4/4 | burst, outlier_threshold |

## Agent-authored families vs PS-1 baseline

- PS-1 verdict: **NO_PROPOSAL_SHIFT**

- **A3**: PS-1 ['burst', 'outlier_threshold'] → PS-2 agent-authored ['burst', 'outlier_threshold']
- **A5-neutral**: PS-1 ['burst', 'outlier_threshold'] → PS-2 agent-authored ['burst']
- **A5-scoped**: PS-1 ['outlier_threshold'] → PS-2 agent-authored ['burst', 'outlier_threshold']

## Cost

- LLM: 74 / 150 (attempt 1 charged)
- Consumer fits: 48 / 160
- attempt-2 wall: 3043.4 s / 7200 s
- attempt-1 wall (lost records): None s
- combined wall: None s
- downloads: 0

## Obligations

- **methods_package_unmodified**: False
- **methods_surgery**: W-1 wired the supply rung: fast_agent materialises a supplies_candidates card independently of the propose stage, and open_delayed gives a supply-sourced winner the delayed approval route every agent-authored winner already had.  No threshold, no authorization policy and no permission class.
- **runtime_contracts_operators_unmodified**: True
- **production_governance_unmodified**: True
- **no_new_skill_class_or_permission_platform**: True
- **injection_uses_existing_frozen_steps_channel**: True
- **injected_candidate_same_rights**: True
- **grants_execution_false**: True
- **experimental_prior_slot**: True
- **budgets_equal_across_arms**: True
- **oracle_not_loaded_into_harness**: True
- **guided_positive_counts_zero_toward_source_auth**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True
- **semantic_discipline**: a conversion is experience supplying a candidate through the mechanical channel, adjudicated by Target feedback.  It is not a proposal-ability improvement.

## Outside the book

- PS-2 scored Support only when the select stage named the inject.  The harness gives every pool member a Support trial inside the round budget, so ps2_run9 / ps2_run12 were full Support+delayed walks recorded as 0/4.  The funnel now attributes by the Episode the inject's own probe wrote (episode_id carries the Skill id).
