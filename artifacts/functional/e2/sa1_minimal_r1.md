# SA-1 minimal r1 -- Skill as an updatable hypothesis

**核心正效果移动:是 + probe 省 2 / 避免挨拒 2 / regret 差 +0.0000**

Of those raw differences, **0 probes and 1 refusals are attributable to the narrowing itself** (the position where A5's Scope stopped admitting a unit that K0's still admitted); the rest is the Fast agent's own proposal set.  Carrying the Scope-v2 card at all -- either arm -- is worth **+0.6860** cumulative regret against carrying none, which is a Part 0.5 result and not the mechanism under test here.

判词 **SA1_DEVELOPMENT_SIGNAL** -- a refusal produced a structured narrowing, the narrowed card stopped supplying the domain that refused it while every other domain kept its candidate, and the arm that revises beat the arm that cannot on the mechanism readout with non-inferior regret and zero harm

## Offline gates

| part | check | pass |
|---|---|---|
| part_0 | P0. the four fields backfill correctly on the L1 record | True |
| part_0_5 | P0.5 the family-axis card matches the four impulse units and not the burst control | True |
| part_1 | (a) R1 -- L1's recorded conversion appends one ledger row | True |
| part_1 | (b) R2 and R3 -- two historical negatives each drive one revision, and every version is recoverable | True |
| part_1 | (c) the narrowed card excludes the refusing unit and nothing else | True |

## Card version chain

v0 `00503481edac` -> v1 `0eae563f76ce` -> v2 `3ef7202e73ce` -> v3 `89728a4af566`

## Per unit, per arm

| # | unit | arm | deploy | regret | probes | supplied | converted | refused | harm |
|---|---|---|---|---|---|---|---|---|---|
| 1 | GunPoint | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.4067 | 1 | 0 | 0 | 0 | False |
| 1 | GunPoint | A5-adaptive | FROZEN_ACTIVE_SKILL_RECALL | -0.0667 | 2 | 1 | 1 | 0 | False |
| 1 | GunPoint | K0-fixed | FROZEN_ACTIVE_SKILL_RECALL | -0.0667 | 2 | 1 | 1 | 0 | False |
| 2 | GunPointOldVersusYoung | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1841 | 1 | 0 | 0 | 0 | False |
| 2 | GunPointOldVersusYoung | A5-adaptive | FROZEN_ACTIVE_SKILL_RECALL | -0.0286 | 2 | 1 | 1 | 0 | False |
| 2 | GunPointOldVersusYoung | K0-fixed | FROZEN_ACTIVE_SKILL_RECALL | -0.0286 | 2 | 1 | 1 | 0 | False |
| 3 | PowerCons | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 0 | 0 | 0 | False |
| 3 | PowerCons | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 1 | 0 | 1 | False |
| 3 | PowerCons | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 1 | 0 | 1 | False |
| 4 | Herring | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0469 | 1 | 0 | 0 | 0 | False |
| 4 | Herring | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0469 | 1 | 0 | 0 | 0 | False |
| 4 | Herring | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0469 | 2 | 1 | 0 | 1 | False |
| 5* | PowerCons | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 1 | 0 | 0 | 0 | False |
| 5* | PowerCons | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 0 | 0 | 0 | False |
| 5* | PowerCons | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.1333 | 2 | 1 | 0 | 1 | False |
| 6 | BirdChicken | A3-reset | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0000 | 1 | 0 | 0 | 0 | False |
| 6 | BirdChicken | A5-adaptive | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0000 | 1 | 0 | 0 | 0 | False |
| 6 | BirdChicken | K0-fixed | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | +0.0000 | 2 | 0 | 0 | 0 | False |

`*` = the re-encounter slot; a mechanism readout that does not count toward cumulative regret.

## Arm totals (distinct units only for regret)

| arm | regret | probes | supplied | converted | refused | harm | llm | fit |
|---|---|---|---|---|---|---|---|---|
| A3-reset | +0.7710 | 7 | 0 | 0 | 0 | 0 | 28 | 15 |
| K0-fixed | +0.0850 | 12 | 5 | 2 | 3 | 0 | 25 | 28 |
| A5-adaptive | +0.0850 | 10 | 3 | 2 | 1 | 0 | 25 | 26 |

## Pre-registered predictions

| id | claim | held | observed |
|---|---|---|---|
| P1 | the Scope-v2 card matches GunPoint, GPOvY, PowerCons and Herring, and does not match BirdChicken | yes | matched ['GunPointOldVersusYoung__impulse_v2', 'GunPoint__impulse_v2', 'Herring__impulse_v2', 'PowerCons__impulse_v2'] |
| P2 | GunPoint and GPOvY convert the supplied candidate through both gates | yes | A5-adaptive conversions at positions [1, 2] |
| P3 | PowerCons #1 refuses the supplied candidate and A5-adaptive emits exactly one narrowing PATCH (card v1, content sha versioned) | yes | PowerCons#1 refused=1, narrowing PATCHes at position 3 = 1, version chain = ['v0', 'v1', 'v2', 'v3'] |
| P4 | Herring is either refused or already excluded by v1; both are legal and which one is reported | **no** | Herring scope_match=True supplied=0 refused=0 |
| P5 | PowerCons #2: A5-adaptive supplies nothing and is refused nothing; K0-fixed supplies again and is refused again | yes | A5 supplied=0 refused=0 / K0 supplied=1 refused=1 |
| P6 | A5-adaptive saves >= 1 probe against K0-fixed and its cumulative regret is non-inferior | raw yes, attributable **no** | probes saved 2 raw / **0 attributable to the narrowing**, regret gap +0.0000 |
| P7 | harm events are zero in every arm | yes | harm events {'A3-reset': 0, 'K0-fixed': 0, 'A5-adaptive': 0} |

## A5-adaptive vs K0-fixed, position by position

| # | unit | A5 scope | K0 scope | A5 supplied | K0 supplied | probe delta | refusal delta | cause |
|---|---|---|---|---|---|---|---|---|
| 1 | GunPoint | True | True | 1 | 1 | +0 | +0 | - |
| 2 | GunPointOldVersusYoung | True | True | 1 | 1 | +0 | +0 | - |
| 3 | PowerCons | True | True | 1 | 1 | +0 | +0 | - |
| 4 | Herring | True | True | 0 | 1 | +1 | +1 | agent_pool_composition |
| 5 | PowerCons | False | True | 0 | 1 | +0 | +1 | scope_narrowing |
| 6 | BirdChicken | False | False | 0 | 0 | +1 | +0 | agent_pool_composition |

## Revisions

| after # | unit | rules | excluded | card sha |
|---|---|---|---|---|
| 1 | GunPoint | R1 | - | `0eae563f76ce` |
| 2 | GunPointOldVersusYoung | R1 | - | `3ef7202e73ce` |
| 3 | PowerCons | R2 | period_change_score==very_low | `89728a4af566` |
| 4 | Herring | - | - | `89728a4af566` |
| 5 | PowerCons | - | - | `89728a4af566` |
| 6 | BirdChicken | - | - | `89728a4af566` |

## What broke, and what the numbers do not say

- **P4 broke, and not in either of the two ways it allowed for.**  Herring matched the narrowed card's Scope and the card was in A5-adaptive's view, but no `cand_skill_` candidate reached the pool: the Fast agent had already proposed the same frozen program itself, so the mechanical supply was deduplicated against it.  Herring was therefore neither refused-as-supplied nor excluded by v1.  The outcome is identical to K0-fixed (identity deployed, regret +0.0469); only the probe count differs.

- **The probe saving is not the narrowing's.**  Of 2 raw probes saved against K0-fixed, 0 are attributable to the narrowing.  At the pre-registered re-encounter slot the narrowing did remove the supplied candidate and the refusal that came with it, but the Fast agent spent the freed slot on a proposal of its own, so the probe count there is unchanged.  The clean, attributable mechanism difference in this run is **one avoided refusal**, at exactly the position the freeze named.

- **Regret is identical between the two card arms (+0.0000).**  With one refusal avoided and no conversion gained or lost, revision bought cost and not quality here; the pre-registered claim was non-inferiority, which holds, and nothing stronger is claimed.

- **The big regret number belongs to Part 0.5, not to revision.**  Both card arms finish at +0.0850 cumulative regret over the five distinct units against +0.7710 with no card -- a gap of +0.6860.  The five units and the identity baseline are the same ones the v4 tail scored at +0.7710, so this is comparable to L1's +0.2127 on the same substrate: the family Pattern axis, not the revision loop, is what moved 1-of-5 Scope coverage to 4-of-5.

- **R3 wrote a demotion but no exclusion in the offline replay.**  The ECG200 outlier_mad harm reading agrees with the card's own evidence on every contracted axis, so nothing distinguishes it and the rule refused to invent an axis.  The structured demotion note landed; the narrowing did not.  The live course produced zero harm events, so R3 was never exercised live -- as the freeze expected.

- **Q7 stands: the effective Scope is wider than the recorded one.**  The edit schema cannot carry 3 of the family's leaves (level_region_end_fraction, level_region_fraction, outlier_region_end_fraction), so they are declared on the card and absent from its machine AST.  Any exclusion compiled on this surface inherits the same blind spot.

- **Development grade, one run.**  Every unit was already exposed, the seed Episode is a recorded one rather than a live re-earn, and the wording stays `SA1_DEVELOPMENT_SIGNAL`; a compound claim needs a sampling replicate, which this book does not authorize.


## Cost

LLM 78/150, consumer fits 69/300, wall 2227 s / 15120 s, downloads 0.

## Obligations

- **course_frozen_before_any_live_run**: True
- **downloads**: 0
- **fault_routes_and_router_unmodified**: True
- **full_repo_pytest_not_run**: True
- **minipipe_fault_routing_not_touched**: True
- **new_data_units_operators_consumers**: 0
- **q1_residue**: the applicability surface is authorized by RETRIEVAL_MISS alone, a cause named for the widening direction; SA-1 used it as the token for a narrowing PATCH rather than mint a code, and this stays open as Q1
- **sealed_material_not_opened**: Epilepsy2 and the s1_oracle keys were not read; every Pattern view in this book comes from extract_public_features on the built cell, which is the production path
- **single_run**: True
- **subagents_spawned**: 0
- **thresholds_and_authorization_unmodified**: MATERIAL, the TRY tier leave-one-out, the RISK tier, the supply tier count, the execution and deployment gates and the prompt/model/budget protocol are untouched
