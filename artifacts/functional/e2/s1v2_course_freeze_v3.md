# S1-v2 course freeze v3 -- discovery-reliable development curriculum

protocol: `s1v2_forward_course_v1_v3`  git: `cf06343ee32b653c5c74bf019a3bce4cd49cde85`

**S1V2_COURSE_FROZEN_V3**

two producers with demonstrated cold-discovery on their own unit, a non-empty five-axis Scope, and two held-in learnable beneficiaries at separated margin bands, none of which is a producer.

- producers are selected on demonstrated cold-discovery rate; the natural-bootstrap course r1 is retained as the discovery module's control

> **Exclusion semantics, revised.** Revised, and the revision is what releases the former dual-source pair as producers.  The constraint that protects 'the card is earned inside the course' is not 'this unit was ever a source elsewhere'; it is (i) K0 carries no card, so A5 starts with nothing, and (ii) no beneficiary is also a producer, so nothing is graded on the unit that taught it.  Re-earning the family on a producer *inside* this course is exactly what the course is supposed to do, and it does not make the compiled card 'brought in' -- the Episodes it compiles from are this course's own.  Checked with sol.

> **Prior exposure.** Both beneficiaries are units whose hampel convertibility is already known from PS-2 / W-1 (GPOvY) and M-1 (GPMvF).  There is no leakage into any arm: A5-online starts from K0 and its only Source knowledge is what this course compiles from its own Episodes.  The novel claim is therefore the end-to-end ITT compounding -- knowledge produced inside the course changing later units -- and explicitly NOT that these units are convertible, which is prior work.

> **Family overlap.** GunPointAgeSpan (producer A) and GunPointOldVersusYoung / GunPointMaleVersusFemale (beneficiaries) share the GunPoint name family. The units themselves are disjoint and no beneficiary is a producer, but this is a within-family transfer at the substrate level and must not be reported as cross-family capability.

> **Control.** Course r1 (natural-bootstrap producers, chosen on sealed margin alone) returned TREATMENT_EMPTY: the arm never proposed the family on either producer, so no card compiled.  That run is retained as the discovery module's control -- it is the measurement of what happens when producer selection ignores proposability.  Artifact: artifacts/functional/e2/s1v2_forward_run1.json

> **Replicates.** the injection has no RNG to seed (run_e2_t6_cls_op_shared_harness.py:3896-3901), so a second run is a sampling replicate: identical substrate and protocol, Fast Agent the only stochastic element

## Course (frozen forward order)

| # | role | unit | menu oracle | half margin | cold discovery | census |
|---|---|---|---|---|---|---|
| 1 | producer_A | `GunPointAgeSpan__impulse_v2` | `hampel_filter` | 7.00 | 2/2 (PS-0 re-earn) | LEARNABLE |
| 2 | identity_A | `BeetleFly__impulse_v2` | `identity` | - | - | None |
| 3 | producer_B | `PowerCons__impulse_v2` | `hampel_filter` | 5.00 | 2/3 (PS-0c re-earn) | LEARNABLE |
| 4 | beneficiary_strong | `GunPointOldVersusYoung__impulse_v2` | `hampel_filter` | 5.00 | - | LEARNABLE |
| 5 | beneficiary_weak | `GunPointMaleVersusFemale__impulse_v2` | `hampel_filter` | 2.00 | - | LEARNABLE |
| 6 | heldout_only | `Herring__impulse_v2` | `hampel_filter` | - | - | HELDOUT_ONLY |
| 7 | identity_B | `BirdChicken__burst_cls2` | `identity` | - | - | None |

## Producer selection (the only change from r2)

- rule: demonstrated cold-discovery rate on this unit, then sealed half-protocol margin
- why r1 failed: r1 picked producers on sealed margin alone; a margin says a reading would be legible if the family were probed, not that the arm will propose it

## Beneficiaries

| unit | band | Scope match | census | half margin | material line | also a producer |
|---|---|---|---|---|---|---|
| `GunPointOldVersusYoung__impulse_v2` | **strong** | True | LEARNABLE | 5.00 | 0.0500 | False |
| `GunPointMaleVersusFemale__impulse_v2` | **weak** | True | LEARNABLE | 2.00 | 0.0526 | False |

- stratified prediction: pre-registered: A5's advantage should concentrate on the strong-margin beneficiary and be marginal on the weak one

## Gates

- regret gate `Delta_material` = 0.050000 + 0.052632 = 0.102632
- cost gate: convertible units average >= 1 probe saved

## Transfer graph

- `GunPointAgeSpan__impulse_v2` + `PowerCons__impulse_v2` --Slow boundary after position 3 -> compile_supply_tier--> `GunPointOldVersusYoung__impulse_v2`, `GunPointMaleVersusFemale__impulse_v2` (carrier: supplies_candidates card (grants_execution=false))

## Precheck

- five-axis Scope non-empty: True (20 leaves)
- both beneficiaries machine-match: True
- no beneficiary is a producer: True
- K0 purity: asserted at run time by compile_k0 purity
- expected card boundary: after position 3
- expected first divergence: position 4 (`GunPointOldVersusYoung__impulse_v2`)

