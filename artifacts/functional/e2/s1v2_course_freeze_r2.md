# S1-v2 course freeze r2 (Part 0 after the main line's rulings)

protocol: `s1v2_forward_course_v1_r2`  git: `2e7c52758fe7bf32cb2f0c567b2476c21aaadd21`

**S1V2_COURSE_FROZEN_R2**

two producers with distinct task_episode_id and a non-empty five-axis Scope, followed by two held-in learnable beneficiaries that machine-match that Scope at separated margin bands.  The treatment group exists on arithmetic.

> Both beneficiaries are units whose hampel convertibility is already known from PS-2 / W-1 (GPOvY) and M-1 (GPMvF).  There is no leakage into any arm: A5-online starts from K0 and its only Source knowledge is what this course compiles from its own Episodes.  The novel claim is therefore the end-to-end ITT compounding -- knowledge produced inside the course changing later units -- and explicitly NOT that these units are convertible, which is prior work.

## Course (frozen forward order)

| # | role | unit | menu oracle | half margin | census | coarsest half n |
|---|---|---|---|---|---|---|
| 1 | producer_A | `PowerCons__burst_cls2` | `hampel_filter` | 5.00 | LEARNABLE | 26 |
| 2 | identity_A | `BeetleFly__impulse_v2` | `identity` | - | None | None |
| 3 | producer_B | `GunPoint__impulse_v2` | `hampel_filter` | 3.00 | LEARNABLE | 7 |
| 4 | beneficiary_strong | `GunPointOldVersusYoung__impulse_v2` | `hampel_filter` | 5.00 | LEARNABLE | 20 |
| 5 | beneficiary_weak | `GunPointMaleVersusFemale__impulse_v2` | `hampel_filter` | 2.00 | LEARNABLE | 19 |
| 6 | heldout_only | `Herring__impulse_v2` | `hampel_filter` | - | HELDOUT_ONLY | 10 |
| 7 | identity_B | `BirdChicken__burst_cls2` | `identity` | - | None | None |

- seven units; the book's 'eight' counts the Slow boundary between producer B and the strong beneficiary as a step

## Beneficiaries (ruling a)

| unit | band | Scope match | census | half margin | material line | prior exposure |
|---|---|---|---|---|---|---|
| `GunPointOldVersusYoung__impulse_v2` | **strong** | True | LEARNABLE | 5.00 | 0.0500 | PS-2 / W-1 exam unit |
| `GunPointMaleVersusFemale__impulse_v2` | **weak** | True | LEARNABLE | 2.00 | 0.0526 | M-1 margin-gate unit |

- stratified prediction: pre-registered: A5's advantage should concentrate on the strong-margin beneficiary and be marginal on the weak one

## Gates (ruling b)

- regret gate `Delta_material` = 0.050000 + 0.052632 = 0.102632
- cost gate: convertible units average >= 1 probe saved

## Transfer graph

- `PowerCons__burst_cls2` + `GunPoint__impulse_v2` --Slow boundary after position 3 -> compile_supply_tier--> `GunPointOldVersusYoung__impulse_v2`, `GunPointMaleVersusFemale__impulse_v2` (carrier: supplies_candidates card (grants_execution=false))

## Precheck

- five-axis Scope non-empty: True (19 leaves)
- both beneficiaries machine-match: True
- expected card boundary: after position 3
- expected first divergence: position 4 (`GunPointOldVersusYoung__impulse_v2`)
- seeds: {"r1": 20260827, "r2": 20260828}

