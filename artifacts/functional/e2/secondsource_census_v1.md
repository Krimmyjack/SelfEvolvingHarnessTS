# Second-source UNGUIDED census v1

Instrument / input for mainline R2. Not a Capability or Claim.
Zero LLM, zero new Outcome.

## Sources

- Weather UNGUIDED: `w1.weather_a5a3_autonomous_guidance` A3 (consumed the base guidance verbatim) plus `g2_shakedown_weather_report.json` A3 (cold arm, `source_prior_retrieval.matched=false`).
- Weather GUIDANCE_CONDITIONED: the same w1 block's A5 arm (consumed the patched autonomous guidance). Counted only in FULL, never toward an active-clause threshold.
- e31: **not included**. Frozen-roster eval series all hit the scale floor; existing `w1_e31_*` reports have no instrument-valid Episode rows. Second source is Weather only.
- T233 / `g1_agentic_pipeline_report*.json` are the first source and are not mixed into this census.

## UNGUIDED census

| program | post_shift_support_sufficient | relation | distinct_task_count | attempt_count |
| --- | --- | --- | ---: | ---: |
| `repair_level_shift` | True | NEGATIVE | 1 | 1 |
| `repair_level_shift` | True | POSITIVE | 3 | 3 |
| `repair_level_shift` | False | NEGATIVE | 7 | 7 |
| `repair_level_shift` | False | POSITIVE | 12 | 17 |
| `hampel_filter` | False | NEGATIVE | 2 | 2 |

## FULL census (UNGUIDED + GUIDANCE_CONDITIONED)

| program | post_shift_support_sufficient | relation | distinct_task_count | attempt_count |
| --- | --- | --- | ---: | ---: |
| `repair_level_shift` | True | NEGATIVE | 1 | 2 |
| `repair_level_shift` | True | POSITIVE | 3 | 6 |
| `repair_level_shift` | False | NEGATIVE | 7 | 13 |
| `repair_level_shift` | False | POSITIVE | 12 | 25 |
| `hampel_filter` | True | POSITIVE | 1 | 1 |
| `hampel_filter` | False | NEGATIVE | 8 | 10 |
| `hampel_filter` | False | POSITIVE | 4 | 4 |

## Outlier-family repeat (UNGUIDED, active-clause threshold)

T233 reference: `outlier_iqr` POSITIVE, `post_shift_support_sufficient=false`, distinct_task_count=6, NEGATIVE=0. That cell is the only T233 cell that is cleanly above the ≥2 distinct-task active-clause threshold.

**Answer: no.**

Weather UNGUIDED contains **zero** `outlier_iqr` or `outlier_mad` POSITIVE cells. The Weather UNGUIDED mass is `repair_level_shift` (both POSITIVE and NEGATIVE) and `hampel_filter` NEGATIVE. An unconditioned `prefer outlier_iqr` clause is not authorized by the second source under the frozen UNGUIDED rule.

GUIDANCE_CONDITIONED Weather A5 does contain `hampel_filter` POSITIVE cells; those may weaken a global hampel ban but cannot authorize a new active clause.

## Contract

- Evidence unit is `distinct_task_count`; `attempt_count` is diagnostic.
- Active-clause threshold is ≥2 UNGUIDED distinct tasks.
- Task ids are namespaced `weather:<id>` so they cannot collapse with T233 ids of the same roster label.

