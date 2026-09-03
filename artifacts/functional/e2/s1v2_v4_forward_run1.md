# S1-v2 forward course, seed r1

protocol: `s1v2_forward_course_v1`  git: `a1f51341b35d246ba755f478ce6cb73ef38c6e6e`  run label: `20260827` (sampling replicate)

**TREATMENT_EMPTY**

the course produced no Fast-visible knowledge: no supply card compiled and no supplied candidate ever reached a pool.  Stop; the second seed is not started.

> The book asked for two forward runs on different injection seeds.  This family's injection has no RNG to seed -- run_e2_t6_cls_op_shared_harness.py:3896-3901 records it: a fixed signed template at positions derived from the series length, and a deterministic evenly-spaced fit/support split.  A 'fresh injection seed' would therefore be a fiction.  The two runs are honest *sampling* replicates: identical substrate and identical protocol, with the Fast Agent as the only stochastic element.  The seed label is a run id, not an injection parameter.

> Both beneficiaries are units whose hampel convertibility is already known from PS-2 / W-1 (GPOvY) and M-1 (GPMvF).  There is no leakage into any arm: A5-online starts from K0 and its only Source knowledge is what this course compiles from its own Episodes.  The novel claim is therefore the end-to-end ITT compounding -- knowledge produced inside the course changing later units -- and explicitly NOT that these units are convertible, which is prior work.

> ITT: a Scope-qualified unit whose supplied candidate failed to enter the pool counts as an A5 system failure.  The conditional conversion rate given successful injection is reported separately and is not the main analysis.

## Per-unit, per-arm

| # | role | unit | arm | deployed | held-out | regret | worst class | probes | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | producer_A | GunPointAgeSpan | Static | `identity` | +0.0000 | +0.2627 | +0.0000 | 0 | 0 | 1 |
| 1 | producer_A | GunPointAgeSpan | A3-reset | `identity` | +0.0000 | +0.2627 | +0.0000 | 1 | 4 | 1 |
| 1 | producer_A | GunPointAgeSpan | K0-fixed | `hampel_filter` | +0.2690 | -0.0063 | +0.2564 | 2 | 6 | 6 |
| 1 | producer_A | GunPointAgeSpan | A5-online | `identity` | +0.0000 | +0.2627 | +0.0000 | 1 | 5 | 1 |
| 2 | identity_A | BeetleFly | Static | `identity` | +0.0000 | +0.0000 | +0.0000 | 0 | 0 | 1 |
| 2 | identity_A | BeetleFly | A3-reset | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 4 | 3 |
| 2 | identity_A | BeetleFly | K0-fixed | `identity` | +0.0000 | +0.0000 | +0.0000 | 1 | 5 | 1 |
| 2 | identity_A | BeetleFly | A5-online | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 5 | 3 |
| 3 | producer_B | GunPointMaleVersusFemale | Static | `identity` | +0.0000 | +0.1930 | +0.0000 | 0 | 0 | 1 |
| 3 | producer_B | GunPointMaleVersusFemale | A3-reset | `identity` | +0.0000 | +0.1930 | +0.0000 | 1 | 4 | 3 |
| 3 | producer_B | GunPointMaleVersusFemale | K0-fixed | `hampel_filter` | +0.1867 | +0.0063 | +0.1400 | 2 | 5 | 7 |
| 3 | producer_B | GunPointMaleVersusFemale | A5-online | `hampel_filter` | +0.1867 | +0.0063 | +0.1400 | 1 | 5 | 6 |
| 4 | producer_C_backup | GunPoint | Static | `identity` | +0.0000 | +0.4067 | +0.0000 | 0 | 0 | 1 |
| 4 | producer_C_backup | GunPoint | A3-reset | `identity` | +0.0000 | +0.4067 | +0.0000 | 2 | 5 | 1 |
| 4 | producer_C_backup | GunPoint | K0-fixed | `identity` | +0.0000 | +0.4067 | +0.0000 | 2 | 7 | 1 |
| 4 | producer_C_backup | GunPoint | A5-online | `identity` | +0.0000 | +0.4067 | +0.0000 | 1 | 5 | 1 |
| 5 | beneficiary_strong | GunPointOldVersusYoung | Static | `identity` | +0.0000 | +0.1841 | +0.0000 | 0 | 0 | 1 |
| 5 | beneficiary_strong | GunPointOldVersusYoung | A3-reset | `identity` | +0.0000 | +0.1841 | +0.0000 | 1 | 6 | 1 |
| 5 | beneficiary_strong | GunPointOldVersusYoung | K0-fixed | `identity` | +0.0000 | +0.1841 | +0.0000 | 0 | 4 | 1 |
| 5 | beneficiary_strong | GunPointOldVersusYoung | A5-online | `identity` | +0.0000 | +0.1841 | +0.0000 | 2 | 4 | 1 |
| 6 | beneficiary_weak | PowerCons | Static | `identity` | +0.0000 | +0.1333 | +0.0000 | 0 | 0 | 1 |
| 6 | beneficiary_weak | PowerCons | A3-reset | `identity` | +0.0000 | +0.1333 | +0.0000 | 2 | 5 | 4 |
| 6 | beneficiary_weak | PowerCons | K0-fixed | `identity` | +0.0000 | +0.1333 | +0.0000 | 2 | 4 | 4 |
| 6 | beneficiary_weak | PowerCons | A5-online | `identity` | +0.0000 | +0.1333 | +0.0000 | 2 | 5 | 4 |
| 7 | heldout_only | Herring | Static | `identity` | +0.0000 | +0.0469 | +0.0000 | 0 | 0 | 1 |
| 7 | heldout_only | Herring | A3-reset | `identity` | +0.0000 | +0.0469 | +0.0000 | 1 | 5 | 3 |
| 7 | heldout_only | Herring | K0-fixed | `identity` | +0.0000 | +0.0469 | +0.0000 | 2 | 4 | 4 |
| 7 | heldout_only | Herring | A5-online | `identity` | +0.0000 | +0.0469 | +0.0000 | 2 | 5 | 4 |
| 8 | identity_B | BirdChicken | Static | `identity` | +0.0000 | +0.0000 | +0.0000 | 0 | 0 | 1 |
| 8 | identity_B | BirdChicken | A3-reset | `identity` | +0.0000 | +0.0000 | +0.0000 | 1 | 5 | 3 |
| 8 | identity_B | BirdChicken | K0-fixed | `identity` | +0.0000 | +0.0000 | +0.0000 | 1 | 4 | 3 |
| 8 | identity_B | BirdChicken | A5-online | `identity` | +0.0000 | +0.0000 | +0.0000 | 1 | 4 | 3 |

## Arm summary

| arm | units | cumulative regret | mean held-out | worst class | harm | probes | LLM | fits | fit wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| Static | 8 | +1.2267 | +0.0000 | +0.0000 | 0 | 0 | 0 | 8 | 52.0 |
| A3-reset | 8 | +1.2267 | +0.0000 | +0.0000 | 0 | 11 | 38 | 19 | 1068.2 |
| K0-fixed | 8 | +0.7710 | +0.0570 | +0.0000 | 0 | 12 | 39 | 27 | 1102.0 |
| A5-online | 8 | +1.0400 | +0.0233 | +0.0000 | 0 | 12 | 38 | 23 | 965.4 |

## Supply / guard timeline

| after # | unit | rows | card compiled | installed | withheld because |
|---|---|---|---|---|---|
| 1 | GunPointAgeSpan | 0 | False | False | no_program_family_met_the_supply_rule |
| 2 | BeetleFly | 0 | False | False | no_program_family_met_the_supply_rule |
| 3 | GunPointMaleVersusFemale | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 4 | GunPoint | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 5 | GunPointOldVersusYoung | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 6 | PowerCons | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 7 | Herring | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 8 | BirdChicken | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |

## Gates

- Delta_material = 0.088462
- regret gap vs A3-reset = +0.1867; vs K0-fixed = -0.2690
- probe gap vs A3-reset = -1
- beneficiaries with an injected candidate: 0 / 2

## Cost

- LLM: 115 / 280
- fits: 77 / 900
- wall: 3206.0 s / 21600 s
- downloads: 0

## Obligations

- **course_frozen_before_any_live_run**: True
- **thresholds_and_authorization_unmodified**: MATERIAL, the TRY tier's leave-one-out, the supply tier's count, the T1 predicate and the ledger incumbent rule are untouched
- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **k0_carries_no_dual_source_card**: True
- **a5_knowledge_is_course_produced_only**: True
- **oracle_read_as_exam_key_only**: True
- **guided_positive_counts_zero**: True
- **itt_main_analysis**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True
