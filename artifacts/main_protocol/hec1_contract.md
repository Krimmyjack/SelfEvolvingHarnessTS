# HEC-1 contract (P4U-v4) -- FROZEN 2026-09-03

Written from `docs/HEC1_CONTRACT_SKELETON_2026-09-03.md` with every `[D2]` field re-derived from `p4ac_hec1_course_supply.json` at read time, so the contract and the audit cannot drift apart. sol confirmed all 13 mainline defaults and added 6 rulings; the user released four budget envelopes.

## Status

| check | state |
| --- | --- |
| mechanical drift (`assert_frozen`) | see the JSON `assert_frozen` block |
| sol ratification | **confirmed** (13 defaults, 6 rulings) |
| user budget release | **released** (6 envelopes) |
| still human, always | `phase_f.seal_opening` |

## Autonomy envelope

> Phase S -> K0 freeze -> Forward -> Reverse -> Interleaved -> one frozen 0-LLM course readout -> stop before Phase F is opened

Never autonomous: opening the Phase F seal: a human release, every time; the verdict: the mainline writes it and sol confirms it.

## Course

| item | value |
| --- | --- |
| Phase S units | 13 |
| Phase T units | 26 (all-usable 30 not taken) |
| Phase S blocks | [160:200], [200:239] |
| Phase T blocks | [0:40], [40:80], [80:120], [120:160] |
| blocks disjoint | True |
| held-out intersection empty | True |
| pattern-sparse units | 0 |

HEC-1 therefore does **not** test silence when the pattern is absent: every unit has at least six series over `z_peak >= 3`, so the frozen initialiser barely filters anyone and all narrowing comes from Slow's clause. Recorded as a limit, not discovered in the report.

## Budget arithmetic

| item | value |
| --- | --- |
| LLM per unit-arm | 5 |
| outer steps per ordering | 5 (k = 5) |
| Forward, full K0 | **410** / cap 500 |
| Forward, empty K0 | 270 |
| Phase S estimate | 69 / cap 120 |
| released envelopes | phase_s 120, phase_t_forward 500, phase_t_interleaved 500, phase_t_reverse 500 (sum 1620, hard cap 1620) |
| Best-Safe-Global fits | 1820 |

## sol's rulings

| ruling | enforced by |
| --- | --- |
| all thirteen mainline defaults confirmed | `CONFIRMED_BY_SOL and assert_frozen` |
| the 20/19 cut is confirmed, but every denominator must take the actual served count at run time; a hard-coded 20 anywhere in the denominator path means the cut becomes 19/19 instead | `SERVED_DENOMINATOR and assert_frozen's scan` |
| Phase F's 0-LLM recall is confirmed, but runs only on a supported verdict and only after a separate human seal release | `PHASE_F and assert_launchable('phase_f')` |
| the Runtime calibrates the threshold and records LLM_THRESHOLD_IGNORED when Slow returns a number | `scope_threshold_tool.clause_from_slow` |
| census grouping by program is approved, but the key must carry the full operator sequence, its order and its parameters; the behaviour fingerprint may only fold aliases | `outer_loop._program_signature and CENSUS_KEY` |
| an unimplemented live arm loop is not a completion state; it must be finished and pass a 0-LLM end-to-end test before the freeze | `run_hec1.run_course and the end-to-end test` |

## Confirmed defaults

| field | value | authority |
| --- | --- | --- |
| `outer_loop.period_k_units` | 5 | MAINLINE_PLAN 8-7a |
| `unit.inner_loop_immediate_slow` | closed | MAINLINE_PLAN 8-7b |
| `budget.replay_fits_share_of_course_fits` | 0.25 | MAINLINE_PLAN 8-7c |
| `scope_tool.threshold_rule` | widest feasible frozen bin edge; ties take the coarser box | MAINLINE_PLAN 8-8a |
| `scope_tool.scopefit_only_control` | shadow | MAINLINE_PLAN 8-8b |
| `budget.llm_per_unit_arm` | 5 | CONTRACT_SKELETON 7; makes Forward <= 500 reachable |
| `unit.evaluation_face_offset` | 144 | CONTRACT_SKELETON 5; scored only, never fed back |
| `course.block_200_239_cut` | A=20 / B=19 | p4ac; a 20/20 cell does not form from 39 series |
| `course.target_held_in_origins` | 7 | p4ac; p4u's 5 plus 1176 and 1416, both earlier than held-in and not time-adjacent to held-out |
| `arms.frozen_arm_name_when_k0_empty` | A3-frozen | MAINLINE_PLAN 4.2 mainline note |
| `arms.k0_empty_uses_existing_audited_cards` | False | CONTRACT_SKELETON 4; strictly empty |
| `phase_f.deployment` | Fast-only, 0 LLM mechanical recall | CONTRACT_SKELETON 10 |
| `best_safe_global.menu` | frozen single operators plus the period_median_complete->outlier_* family | CONTRACT_SKELETON 7 |

## Auto-continuation conditions

1. the contract is frozen and the live loop passed its 0-LLM end-to-end test
2. Phase S completed all 13 units
3. the K0 mechanical audit passed; an empty K0 is a legal freeze too
4. the eight mechanical instrument checks passed for the finished ordering
5. whether to continue reads instrument health only, never the effect's sign
6. a fault may only be repaired as an instrument and resumed from a checkpoint; no scientific re-throw
7. no change to the contract, thresholds, features, menu, course or prompts
8. the readout runs exactly once and then stops, reading no Phase F held-out

## Boundary

Every counter is zero: `artifacts_overwritten`, `coverage_floor_changed`, `gates_added`, `held_out_reads`, `manifests_added`, `methods_ttha_files_changed`, `observation_features_added`, `operators_added`, `risk_faces_added`, `risk_thresholds_changed`, `shas_added`, `stage_schemas_added`, `ucr_test_outcome_reads`, `window_verifier_changed`.

## Freeze procedure

1. D2 landed and the [D2] fields were filled -- done
2. sol confirmed all 13 defaults and added 6 rulings -- done 2026-09-03
3. the user released Phase S 120 and three orderings at 500 each -- done
4. the live arm loop was implemented and passed its 0-LLM end-to-end test -- sol's ruling 6, done
5. assert_frozen and assert_launchable pass; Phase F stays refused -- done
6. the 0-LLM smoke passes -- done
7. frozen; from here only an appended erratum, and findings go to HEC-2

This file now takes only an appended erratum section; no field changes, and anything the run turns up goes to HEC-2.

