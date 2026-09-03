# S1-v2 course freeze v4 -- discovery-and-support-reliable development curriculum

protocol: `s1v2_forward_course_v1_v4`  git: `a1f51341b35d246ba755f478ce6cb73ef38c6e6e`

**S1V2_COURSE_FROZEN_V4**

three producers -- two selected on cold discovery *and* live Support pass rate, plus one backup -- a non-empty five-axis Scope, and two held-in learnable beneficiaries at separated bands, neither of which is a producer.  Final throw.

> **FINAL THROW.** HARD CAP, written into the freeze: this is S1-v2's last course attempt.  A third empty treatment group is not a fourth reshuffle -- it is a systematic result.  If this run returns TREATMENT_EMPTY, everything stops and the mechanism goes back for review; no v5 is compiled.

> **Exclusion semantics.** Revised, and the revision is what releases the former dual-source pair as producers.  The constraint that protects 'the card is earned inside the course' is not 'this unit was ever a source elsewhere'; it is (i) K0 carries no card, so A5 starts with nothing, and (ii) no beneficiary is also a producer, so nothing is graded on the unit that taught it.  Re-earning the family on a producer *inside* this course is exactly what the course is supposed to do, and it does not make the compiled card 'brought in' -- the Episodes it compiles from are this course's own.  Checked with sol.

> **Prior exposure.** Both beneficiaries are units whose hampel convertibility is already known from PS-2 / W-1 (GPOvY) and M-1 (GPMvF).  There is no leakage into any arm: A5-online starts from K0 and its only Source knowledge is what this course compiles from its own Episodes.  The novel claim is therefore the end-to-end ITT compounding -- knowledge produced inside the course changing later units -- and explicitly NOT that these units are convertible, which is prior work.

> **Family.** If the card compiles from A and B, its two Episodes are GunPointAgeSpan and GunPointMaleVersusFemale -- both GunPoint name family.  The strong beneficiary GPOvY is the same family again.  This is therefore a within-family transfer at the substrate level: development-mechanism grade, and it must not be reported as cross-family capability.  The weak beneficiary PowerCons is the one genuinely outside that family, which is part of why it is worth keeping despite its thin live Support.

> **Backup producer.** Producer C is a third chance at the second positive, not a third positive.  The supply tier compiles as soon as any boundary holds two distinct unguided positives, so if A and B both land the card is written after position 3 and C still runs -- but by then the card is in A5's view, so C's own positive is Harness-conditioned and counts zero toward authorization.  That is the existing UNGUIDED rule, not an exception carved for this course, and it is why C cannot inflate the evidence.

> **PowerCons sealed vs live.** PowerCons__impulse_v2 carries a sealed half-protocol margin of 5.00x and live Support readings of +0.0714 (PS-0c) and +0.0357 (v3), the latter graded CONFLICT.  Sealed margin and live reading disagree, and the attribution is the proposal's parameter binding rather than the substrate: the sealed oracle scores the operator at its own tuned parameters, while the arm proposes it at whatever the contract binds.  Recorded as an honest weak stratum, not as a substrate defect.

> **Control.** Course r1 (natural-bootstrap producers, chosen on sealed margin alone) returned TREATMENT_EMPTY: the arm never proposed the family on either producer, so no card compiled.  That run is retained as the discovery module's control -- it is the measurement of what happens when producer selection ignores proposability.  Artifact: artifacts/functional/e2/s1v2_forward_run1.json

> **Replicates.** the injection has no RNG to seed (run_e2_t6_cls_op_shared_harness.py:3896-3901), so a second run is a sampling replicate: identical substrate and protocol, Fast Agent the only stochastic element

## Course (frozen forward order)

| # | role | unit | menu oracle | half margin | cold discovery | live Support pass |
|---|---|---|---|---|---|---|
| 1 | producer_A | `GunPointAgeSpan__impulse_v2` | `hampel_filter` | 7.00 | 3/3 | 3/3 |
| 2 | identity_A | `BeetleFly__impulse_v2` | `identity` | - | - | - |
| 3 | producer_B | `GunPointMaleVersusFemale__impulse_v2` | `hampel_filter` | 2.00 | 3/4 | 3/4 |
| 4 | producer_C_backup | `GunPoint__impulse_v2` | `hampel_filter` | 3.00 | unmeasured | unmeasured |
| 5 | beneficiary_strong | `GunPointOldVersusYoung__impulse_v2` | `hampel_filter` | 5.00 | - | - |
| 6 | beneficiary_weak | `PowerCons__impulse_v2` | `hampel_filter` | 5.00 | 2/4 | 0/2 at the material line |
| 7 | heldout_only | `Herring__impulse_v2` | `hampel_filter` | - | - | - |
| 8 | identity_B | `BirdChicken__burst_cls2` | `identity` | - | - | - |

## Producer selection (third criterion)

- rule: three criteria: demonstrated cold discovery, live Support pass rate under the half protocol, then sealed margin; plus one backup producer
- why the third criterion exists: v3 showed cold discovery is not sufficient: PowerCons proposed hampel and read +0.0357, graded CONFLICT

| unit | role | cold discovery | live Support pass | readings |
|---|---|---|---|---|
| `GunPointAgeSpan__impulse_v2` | producer_A | 3/3 | 3/3 | PS-0 re-earn +0.4000; S1-v2 v3 r1 +0.4500 POSITIVE |
| `GunPointMaleVersusFemale__impulse_v2` | producer_B | 3/4 | 3/4 | M-1 half protocol: supply conversion 2/4 -> 3/4 after the wiring, +0.1867 deployed; S1-v2 v3 r1 A3/K0 both earned +0.1867 on this unit |
| `GunPoint__impulse_v2` | producer_C_backup | unmeasured | unmeasured | no live earn on record; carried as the backup producer precisely because the other two are single points |
| `PowerCons__impulse_v2` | beneficiary_weak | 2/4 | 0/2 at the material line | PS-0c +0.0714; S1-v2 v3 r1 +0.0357 -> CONFLICT.  Sealed half-protocol margin is 5.00x, which the live readings do not reproduce |

## Beneficiaries

| unit | band | Scope match | census | half margin | material line | live Support | also a producer |
|---|---|---|---|---|---|---|---|
| `GunPointOldVersusYoung__impulse_v2` | **strong** | True | LEARNABLE | 5.00 | 0.0500 | None | False |
| `PowerCons__impulse_v2` | **weak** | True | LEARNABLE | 5.00 | 0.0385 | 0/2 at the material line | False |

- stratified prediction: pre-registered: A5's advantage should concentrate on the strong-margin beneficiary (GPOvY) and be marginal or absent on the weak one (PowerCons), whose live Support has not cleared the material line in two attempts

## Gates

- regret gate `Delta_material` = 0.050000 + 0.038462 = 0.088462
- cost gate: convertible units average >= 1 probe saved

## Transfer graph

- `GunPointAgeSpan__impulse_v2` + `GunPointMaleVersusFemale__impulse_v2` + `GunPoint__impulse_v2 (backup)` --first Slow boundary holding two distinct unguided positives -> compile_supply_tier--> `GunPointOldVersusYoung__impulse_v2`, `PowerCons__impulse_v2` (carrier: supplies_candidates card (grants_execution=false))

## Precheck

- five-axis Scope non-empty: True (20 leaves)
- both beneficiaries machine-match: True
- no beneficiary is a producer: True
- expected card boundary: 3 if A and B both land, otherwise 4 via the backup producer
- expected first divergence: position 5 (`GunPointOldVersusYoung__impulse_v2`)

