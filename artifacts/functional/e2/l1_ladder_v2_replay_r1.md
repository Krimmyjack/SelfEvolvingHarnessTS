# L1 -- ladder revision v2 replay from the v4 boundary

**Core positive effect moved: YES, +0.2127 cumulative regret against the v4 A5 tail (0.7710 -> 0.5583), gate 0.088462**

protocol: `l1_ladder_v2_replay_v1`  git: `9894a5da2081a4a35d8a5cb6b75d20e0d9d39a00`  verdict: **L1_SIGNAL**

supply-sourced conversion produced a material regret improvement with zero harm

> v2: supply-tier evidence price 2 -> 1 strong positive (Support and delayed both POSITIVE).  TRY tier LOO, RISK tier, execution and deployment gates, MATERIAL and the prompt/model/budget protocol are all untouched.

## T1 offline gate

| check | pass | evidence |
|---|---|---|
| 1. supply-tier price constant is 1 | **True** | `evaluation/functional/task_episode_harness/agentic/source_skill.py:319` |
| 2. unit-3 boundary compiles the single-Episode card | **True** | `evaluation/functional/run_e2_s1v2_forward_course.py:1627` |
| 3. T1 inert predicate does not withhold the supply card | **True** | `methods/ttha/retrieval.py:195` |
| 4. Scope match precheck over the tail five units | **True** | `` |
| 5. injection dry run materialises and verifies | **True** | `methods/ttha/fast_agent.py:386` |
| 6. guided positives count zero | **True** | `evaluation/functional/task_episode_harness/agentic/source_skill.py:366` |

### Scope match precheck

| # | unit | role | machine match | served in Fast view |
|---|---|---|---|---|
| 4 | `GunPoint__impulse_v2` | producer_C_backup | **False** | False |
| 5 | `GunPointOldVersusYoung__impulse_v2` | beneficiary_strong | **True** | True |
| 6 | `PowerCons__impulse_v2` | beneficiary_weak | **False** | False |
| 7 | `Herring__impulse_v2` | heldout_only | **False** | False |
| 8 | `BirdChicken__burst_cls2` | identity_B | **False** | False |

## Boundary resume

- kind: **boundary_replay**
- carried: the supply card compiled from the recorded unit-3 Episode, installed on K0 through the frozen edit path
- not carried: A5's in-memory Episode objects and its unit-3 Target-local capability.  The Episode rows survive as the card's evidence block; the Target-local Skill is domain-stamped and could not apply to any tail unit anyway.  Stated so the attribution is not overclaimed.
- producer stage re-run: True

## A5 tail, per unit

| # | role | unit | supplied in pool | self-proposed | supplied probed | deployed | held-out | regret | worst class | probes | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | producer_C_backup | GunPoint | 0 | 2 | 0 | `identity` | +0.0000 | +0.4067 | +0.0000 | 2 | 5 | 1 |
| 5 | beneficiary_strong | GunPointOldVersusYoung | 1 | 1 | 1 | `hampel_filter` | +0.2127 | -0.0286 | +0.1879 | 1 | 4 | 6 |
| 6 | beneficiary_weak | PowerCons | 0 | 2 | 0 | `identity` | +0.0000 | +0.1333 | +0.0000 | 2 | 4 | 4 |
| 7 | heldout_only | Herring | 0 | 1 | 0 | `identity` | +0.0000 | +0.0469 | +0.0000 | 1 | 3 | 3 |
| 8 | identity_B | BirdChicken | 0 | 1 | 0 | `identity` | +0.0000 | +0.0000 | +0.0000 | 1 | 4 | 3 |

## Pre-registered predictions

| prediction | expected | observed | held |
|---|---|---|---|
| card compiles at the unit-3 boundary | True | True | **True** |
| Scope matches GunPoint__impulse_v2 | True | False | **False** |
| Scope matches GunPointOldVersusYoung__impulse_v2 | True | True | **True** |
| Scope matches PowerCons__impulse_v2 | True | False | **False** |
| Scope matches Herring__impulse_v2 | True | False | **False** |
| Scope does not match BirdChicken__burst_cls2 | False | False | **True** |
| GunPoint__impulse_v2 converts | True | False | **False** |
| GunPointOldVersusYoung__impulse_v2 converts | True | True | **True** |
| PowerCons__impulse_v2 does not deploy | True | True | **True** |
| Herring__impulse_v2 does not deploy | True | True | **True** |
| harm events = 0 | 0 | 0 | **True** |
| A5 tail regret +0.7710 -> <= 0.20 | <= 0.20 | 0.5583 | **False** |

## v4 replay-grade control (tail units)

- v4 frozen readings, used as a replay-grade control.  The other three arms are not re-run in L1, so these are not a fresh contemporaneous comparison and must not be reported as one.

| arm | units | cumulative regret | harm | probes | LLM | fits |
|---|---|---|---|---|---|---|
| Static | 5 | +0.7710 | 0 | 0 | 0 | 5 |
| A3-reset | 5 | +0.7710 | 0 | 7 | 26 | 12 |
| K0-fixed | 5 | +0.7710 | 0 | 7 | 23 | 13 |
| A5-online | 5 | +0.7710 | 0 | 8 | 23 | 13 |

## Cost

- LLM: 20 / 120
- fits: 17 / 300
- wall: 709.8 s
- downloads: 0

## Obligations

- **methods_package_unmodified**: True
- **try_tier_loo_untouched**: True
- **risk_tier_untouched**: True
- **execution_and_deployment_gates_untouched**: True
- **material_threshold_untouched**: True
- **prompt_model_budget_protocol_untouched**: True
- **sealed_artifacts_untouched**: Epilepsy2 and s1_oracle never enter any arm view
- **no_new_units_operators_or_consumers**: True
- **producer_stage_not_rerun**: True
- **other_three_arms_are_v4_replay_grade_control**: True
- **guided_positive_counts_zero**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True
