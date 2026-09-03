# P0 -- supply-tier production reachability

protocol: `p0_supply_tier_reachability_v1`  evidence grade: **infrastructure (deterministic; zero fit, zero LLM)**  git: `010c0d1fc40ab6442891bd9fdfa53c7ad53056c0`

**SUPPLY_TIER_PRODUCTION_REACHABLE**

the two re-earned Episodes compile into a supply-tier card by mechanical template, the card survives the frozen edit path, retrieval serves it only in Scope, and the W-1 reader materialises its frozen program into the candidate pool.  Both Target gates are on that path and neither is bypassed.

## Sources

| task_episode_id | run | program | relation | Support | delayed | pattern leaves | conditioned |
|---|---|---|---|---|---|---|---|
| `GunPointAgeSpan__impulse_v2` | `ps0_srcA_1` | `hampel_filter` | POSITIVE | +0.4000 | +0.4000 | 21 | False |
| `PowerCons__impulse_v2` | `ps0_srcB_4` | `hampel_filter` | POSITIVE | +0.0714 | +0.5000 | 21 | False |

## Reachability chain

| link | reached | witness |
|---|---|---|
| 1. persisted Episode | **True** | `scenes[].earned` |
| 2. supply-tier audit and template compile | **True** | `evaluation/functional/task_episode_harness/agentic/source_skill.py:566` |
| 3. card shape | **True** | `evaluation/functional/task_episode_harness/agentic/source_skill.py:466` |
| 4. EditController apply and recompile | **True** | `evaluation/functional/run_e2_s1_curriculum_four_arms.py:1058` |
| 5. resolve_harness_view(role='fast') | **True** | `methods/ttha/retrieval.py:241` |
| 6. _supply_rung_candidates (dry pool entry) | **True** | `methods/ttha/fast_agent.py:386` |
| 7. both gates exist for a supplied winner | **True** | `methods/ttha/online_loop.py:435`; `methods/ttha/online_loop.py:833` |

## Two tiers, one shared vocabulary

- supply tier: 2 distinct unguided positive Tasks, no leave-one-out
- TRY tier: `authorization_audit (unchanged)` -- unguided positives, leave-one-out floor at min_distinct_tasks, opposing evidence blocks -- so two unguided positives authorize no TRY clause
- shared: unguided evidence only, opposing evidence blocks, distinct task_episode_id is the unit
- only difference: the count, and whether leave-one-out applies

## Compiled card

- skill_id: `p0_supply_tier_hampel_v1`
- authority: {"grants_execution": false, "reorders_supplied_candidates": false, "supplies_candidates": true, "suppresses_operators": false}
- requires_target_support: True
- frozen program: `hampel_filter`
- machine AST leaves: 17
- dropped as uncontracted: ['level_only_post_shift_support_sufficient', 'level_region_end_fraction', 'level_region_fraction', 'outlier_region_end_fraction']

## Cost

- LLM: 0
- Consumer fits: 0
- wall: 0.3 s
- downloads: 0

## Obligations

- **no_llm**: True
- **no_consumer_fits**: True
- **thresholds_unmodified**: MATERIAL, the TRY tier's leave-one-out, the T1 inert predicate and the ledger incumbent rule are untouched; the only addition is the supply-tier exit
- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **no_new_skill_class_or_permission_platform**: True
- **grants_execution_false**: True
- **guided_positive_counts_zero_toward_source_auth**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True

## Outside the book

- the compiler drops pattern leaves that contracts/observables accepts but contracts/schemas/observable_feature_v1.json does not, and records them; that schema-vs-code drift is PS-1's finding and is still open.
