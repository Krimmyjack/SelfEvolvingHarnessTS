# G-3 -- three-field course for the supply rung

protocol: `g3_three_field_course_v1`  evidence grade: **development-mechanism (pilot)**  git: `70b6d8d814feb7f2dfc6861337339a82769cb3a9`

**FIELD1_NO_CONVERSION**

the Scope-matched field converted 1/4 with mean held-out gain +0.0467; the mechanism did not reproduce off the ps2p exam unit.

> a conversion is experience supplying a candidate through the mechanical channel, adjudicated by Target feedback.  It is not evidence that the agent learned to propose the family, and a positive earned under the card counts zero toward cross-domain authorization for the Source Skill.

## 1. Field selection (mechanical, before any live run)

| field | unit | WHEN leaves met | machine match | hampel in oracle set | ps0b grade | margin | menu oracle |
|---|---|---|---|---|---|---|---|
| field1_scope_matched_positive | `GunPointMaleVersusFemale__impulse_v2` | 16/16 | **True** | True | ROBUST_LEARNABLE | 1.3500000000000003 | `hampel_filter` |
| field2_scope_mismatched_inert | `ShapeletSim__impulse_v2` | 13/16 | **False** | False | None | None | `identity` |
| field3_veto | `ToeSegmentation2__impulse_v2` | 16/16 | **True** | False | None | None | `identity` |

Missed leaves for the mismatched field: `estimated_level_offset, estimated_region_start_fraction, level_excursion_score`

## 2. Pre-registration

- **field1_scope_matched_positive** (`GunPointMaleVersusFemale__impulse_v2`): inject 4/4 in A5-scoped, 0/4 in A3 (no card); conversions >= 2/4 deployed hampel with positive held-out gain; deploy hampel_filter; harm 0
- **field2_scope_mismatched_inert** (`ShapeletSim__impulse_v2`): inject 0/4 -- the card is out of Scope and must not be retrieved or injected; conversions 0; deploy identity; harm 0
- **field3_veto** (`ToeSegmentation2__impulse_v2`): inject 4/4 -- Scope matches, so the candidate is supplied and probed; conversions 0/4 -- Support is not a material positive, so nothing drafts; deploy identity; harm 0

## 3. Sixteen-run protocol table

| run | field | arm | unit | inject | Support+ | delayed | deployed | applied | held-out gain | worst class | LLM | fits |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| g3_f1_a3_1 | field1 | A3 | GunPointMaleVersusFemale | False | False | False | False | `identity` | +0.0000 | +0.0000 | 7 | 3 |
| g3_f1_a5_1 | field1 | A5-scoped | GunPointMaleVersusFemale | True | False | False | False | `identity` | +0.0000 | +0.0000 | 4 | 5 |
| g3_f1_a3_2 | field1 | A3 | GunPointMaleVersusFemale | False | False | False | False | `hampel_filter` | +0.1867 | +0.1400 | 6 | 6 |
| g3_f2_1 | field2 | A5-scoped | ShapeletSim | False | False | False | False | `identity` | +0.0000 | +0.0000 | 8 | 5 |
| g3_f2_2 | field2 | A5-scoped | ShapeletSim | False | False | False | False | `identity` | +0.0000 | +0.0000 | 8 | 3 |
| g3_f2_3 | field2 | A5-scoped | ShapeletSim | False | False | False | False | `identity` | +0.0000 | +0.0000 | 8 | 5 |
| g3_f2_4 | field2 | A5-scoped | ShapeletSim | False | False | False | False | `identity` | +0.0000 | +0.0000 | 8 | 1 |
| g3_f3_1 | field3 | A5-scoped | ToeSegmentation2 | True | False | False | False | `identity` | +0.0000 | +0.0000 | 8 | 7 |
| g3_f3_2 | field3 | A5-scoped | ToeSegmentation2 | True | False | False | False | `identity` | +0.0000 | +0.0000 | 10 | 6 |
| g3_f3_3 | field3 | A5-scoped | ToeSegmentation2 | True | False | False | False | `identity` | +0.0000 | +0.0000 | 8 | 7 |
| g3_f3_4 | field3 | A5-scoped | ToeSegmentation2 | True | False | False | False | `identity` | +0.0000 | +0.0000 | 7 | 7 |
| g3_f1_a5_2 | field1 | A5-scoped | GunPointMaleVersusFemale | True | False | False | False | `identity` | +0.0000 | +0.0000 | 7 | 7 |
| g3_f1_a5_3 | field1 | A5-scoped | GunPointMaleVersusFemale | True | False | False | False | `identity` | +0.0000 | +0.0000 | 6 | 6 |
| g3_f1_a3_3 | field1 | A3 | GunPointMaleVersusFemale | False | False | False | False | `identity` | +0.0000 | +0.0000 | 6 | 3 |
| g3_f1_a5_4 | field1 | A5-scoped | GunPointMaleVersusFemale | False | False | True | True | `hampel_filter` | +0.1867 | +0.1400 | 9 | 10 |
| g3_f1_a3_4 | field1 | A3 | GunPointMaleVersusFemale | False | False | False | False | `identity` | +0.0000 | +0.0000 | 6 | 5 |

## 4. Field summaries

| field | arm | runs | entered | Support+ | deployed family | identity deploys | mean held-out | harm | LLM | fits | probes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| field1 / A5-scoped | A5-scoped | 4 | 3 | 0 | 1 | 3 | +0.0467 | 0 | 26 | 28 | 7 |
| field1 / A3 | A3 | 4 | 0 | 0 | 0 | 3 | +0.0467 | 0 | 25 | 17 | 4 |
| field2 / A5-scoped | A5-scoped | 4 | 0 | 0 | 0 | 4 | +0.0000 | 0 | 32 | 14 | 8 |
| field3 / A5-scoped | A5-scoped | 4 | 4 | 0 | 0 | 4 | +0.0000 | 0 | 33 | 27 | 15 |

## 5. Cost

- LLM: 116 / 150
- Consumer fits: 86 / 120
- wall: 4050.8 s / 10800 s
- downloads: 0

## 6. Obligations

- **fields_selected_before_any_live_run**: True
- **card_unchanged_from_w1**: True
- **thresholds_and_authorization_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **no_new_skill_class_or_permission_platform**: True
- **grants_execution_false**: True
- **oracle_read_as_exam_key_only**: True
- **oracle_not_loaded_into_harness**: True
- **guided_positive_counts_zero_toward_source_auth**: True
- **downloads**: 0
- **full_repo_pytest_not_run**: True
- **semantic_discipline**: a conversion is experience supplying a candidate through the mechanical channel, adjudicated by Target feedback.  It is not evidence that the agent learned to propose the family, and a positive earned under the card counts zero toward cross-domain authorization for the Source Skill.

## 7. Outside the book

