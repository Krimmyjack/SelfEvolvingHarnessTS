# S1-v2 forward course, seed r1

protocol: `s1v2_forward_course_v1`  git: `1966fe49ab0ffeec43f6d210a26982ac7088a33f`  run label: `20260827` (sampling replicate)

**TREATMENT_EMPTY**

the course produced no Fast-visible knowledge: no supply card compiled and no supplied candidate ever reached a pool.  Stop; the second seed is not started.

> The book asked for two forward runs on different injection seeds.  This family's injection has no RNG to seed -- run_e2_t6_cls_op_shared_harness.py:3896-3901 records it: a fixed signed template at positions derived from the series length, and a deterministic evenly-spaced fit/support split.  A 'fresh injection seed' would therefore be a fiction.  The two runs are honest *sampling* replicates: identical substrate and identical protocol, with the Fast Agent as the only stochastic element.  The seed label is a run id, not an injection parameter.

> Both beneficiaries are units whose hampel convertibility is already known from PS-2 / W-1 (GPOvY) and M-1 (GPMvF).  There is no leakage into any arm: A5-online starts from K0 and its only Source knowledge is what this course compiles from its own Episodes.  The novel claim is therefore the end-to-end ITT compounding -- knowledge produced inside the course changing later units -- and explicitly NOT that these units are convertible, which is prior work.

> ITT: a Scope-qualified unit whose supplied candidate failed to enter the pool counts as an A5 system failure.  The conditional conversion rate given successful injection is reported separately and is not the main analysis.

## Per-unit, per-arm

| # | role | unit | arm | deployed | held-out | regret | worst class | probes | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | producer_A | PowerCons | Static | `identity` | +0.0000 | +0.1167 | +0.0000 | 0 | 0 | 1 |
| 1 | producer_A | PowerCons | A3-reset | `identity` | +0.0000 | +0.1167 | +0.0000 | 2 | 4 | 5 |
| 1 | producer_A | PowerCons | K0-fixed | `identity` | +0.0000 | +0.1167 | +0.0000 | 1 | 5 | 1 |
| 1 | producer_A | PowerCons | A5-online | `outlier_iqr` | +0.0444 | +0.0722 | -0.0667 | 1 | 6 | 6 |
| 2 | identity_A | BeetleFly | Static | `identity` | +0.0000 | +0.0000 | +0.0000 | 0 | 0 | 1 |
| 2 | identity_A | BeetleFly | A3-reset | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 5 | 3 |
| 2 | identity_A | BeetleFly | K0-fixed | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 5 | 3 |
| 2 | identity_A | BeetleFly | A5-online | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 6 | 3 |
| 3 | producer_B | GunPoint | Static | `identity` | +0.0000 | +0.4067 | +0.0000 | 0 | 0 | 1 |
| 3 | producer_B | GunPoint | A3-reset | `identity` | +0.0000 | +0.4067 | +0.0000 | 1 | 5 | 1 |
| 3 | producer_B | GunPoint | K0-fixed | `identity` | +0.0000 | +0.4067 | +0.0000 | 2 | 4 | 1 |
| 3 | producer_B | GunPoint | A5-online | `identity` | +0.0000 | +0.4067 | +0.0000 | 2 | 5 | 3 |
| 4 | beneficiary_strong | GunPointOldVersusYoung | Static | `identity` | +0.0000 | +0.1841 | +0.0000 | 0 | 0 | 1 |
| 4 | beneficiary_strong | GunPointOldVersusYoung | A3-reset | `identity` | +0.0000 | +0.1841 | +0.0000 | 2 | 4 | 1 |
| 4 | beneficiary_strong | GunPointOldVersusYoung | K0-fixed | `identity` | +0.0000 | +0.1841 | +0.0000 | 2 | 6 | 5 |
| 4 | beneficiary_strong | GunPointOldVersusYoung | A5-online | `identity` | +0.0000 | +0.1841 | +0.0000 | 1 | 4 | 1 |
| 5 | beneficiary_weak | GunPointMaleVersusFemale | Static | `identity` | +0.0000 | +0.1930 | +0.0000 | 0 | 0 | 1 |
| 5 | beneficiary_weak | GunPointMaleVersusFemale | A3-reset | `identity` | +0.0000 | +0.1930 | +0.0000 | 0 | 3 | 1 |
| 5 | beneficiary_weak | GunPointMaleVersusFemale | K0-fixed | `hampel_filter` | +0.1867 | +0.0063 | +0.1400 | 1 | 5 | 6 |
| 5 | beneficiary_weak | GunPointMaleVersusFemale | A5-online | `identity` | +0.0000 | +0.1930 | +0.0000 | 0 | 3 | 1 |
| 6 | heldout_only | Herring | Static | `identity` | +0.0000 | +0.0469 | +0.0000 | 0 | 0 | 1 |
| 6 | heldout_only | Herring | A3-reset | `identity` | +0.0000 | +0.0469 | +0.0000 | 0 | 4 | 1 |
| 6 | heldout_only | Herring | K0-fixed | `identity` | +0.0000 | +0.0469 | +0.0000 | 2 | 4 | 4 |
| 6 | heldout_only | Herring | A5-online | `identity` | +0.0000 | +0.0469 | +0.0000 | 2 | 5 | 4 |
| 7 | identity_B | BirdChicken | Static | `identity` | +0.0000 | +0.0000 | +0.0000 | 0 | 0 | 1 |
| 7 | identity_B | BirdChicken | A3-reset | `identity` | +0.0000 | +0.0000 | +0.0000 | 1 | 5 | 3 |
| 7 | identity_B | BirdChicken | K0-fixed | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 4 | 4 |
| 7 | identity_B | BirdChicken | A5-online | `identity` | +0.0000 | +0.0000 | +0.0000 | 2 | 4 | 4 |

## Arm summary

| arm | units | cumulative regret | mean held-out | worst class | harm | probes | LLM | fits | fit wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| Static | 7 | +0.9474 | +0.0000 | +0.0000 | 0 | 0 | 0 | 7 | 88.9 |
| A3-reset | 7 | +0.9474 | +0.0000 | +0.0000 | 0 | 8 | 30 | 15 | 1094.6 |
| K0-fixed | 7 | +0.7607 | +0.0267 | +0.0000 | 0 | 12 | 33 | 24 | 1052.8 |
| A5-online | 7 | +0.9029 | +0.0063 | -0.0667 | 1 | 10 | 33 | 22 | 1038.4 |

## Supply / guard timeline

| after # | unit | rows | card compiled | installed | withheld because |
|---|---|---|---|---|---|
| 1 | PowerCons | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 2 | BeetleFly | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 3 | GunPoint | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 4 | GunPointOldVersusYoung | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 5 | GunPointMaleVersusFemale | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 6 | Herring | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |
| 7 | BirdChicken | 1 | False | False | fewer_than_2_distinct_unguided_positive_tasks |

## Gates

- Delta_material = 0.102632
- regret gap vs A3-reset = +0.0444; vs K0-fixed = -0.1423
- probe gap vs A3-reset = -2
- beneficiaries with an injected candidate: 0 / 2

## Cost

- LLM: 96 / 250
- fits: 68 / 900
- wall: 3284.5 s / 21600 s
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
