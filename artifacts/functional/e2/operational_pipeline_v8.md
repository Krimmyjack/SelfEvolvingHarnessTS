# A third guard action, on banked evidence

**Overall: `RESCOPE_PRESERVES_GAIN_ELIMINATES_HARM`** -- the deterministic contrast decides the headline; the Slow decision is reported under its own verdict (SLOW_PROPOSES_RESCOPE_MASK_HARMED_SERIES) because choosing a veto with both actions on the table is a reading, not a failure

Slow decision: `SLOW_PROPOSES_RESCOPE_MASK_HARMED_SERIES`.

DEVELOPMENT, and banked on top of that: nothing here is measured.  Every Consumer reading is replayed from a delivered artifact, so this round produces no fresh evidence, no A5-over-A3 result, and no live behaviour reading.

## B1 -- the false-positive half

`NO_FALSE_POSITIVE`: 1 clean banked adoptions, 0 false positives.

| kind | source | step | plan | delayed | min per-series | VETO fires | RESCOPE fires |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| clean | `v1` | `task_A` | outlier_mad | +0.306380 | +0.017726 | no | no |
| crossing | `v1` | `task_C` | outlier_mad | +0.029688 | -0.125557 | yes | yes |
| crossing | `v3` | `task_A` | outlier_iqr | +0.066941 | -0.062068 | yes | yes |
| crossing | `v7` | `task_D` | outlier_mad | +0.049504 | -0.102763 | yes | yes |
| crossing | `slow_scope_update_v2` | `task_C_pooled_A5` | outlier_mad | +0.029688 | -0.125557 | yes | yes |
| crossing | `slow_scope_update_v2` | `task_C_pooled_A3_replay` | outlier_mad | +0.029688 | -0.125557 | yes | yes |

## B3 -- three arms

### `task_C`

| arm | plan after | delayed | harmed | series taken out |
| --- | --- | ---: | --- | --- |
| `no_guard` | outlier_mad | +0.029688 | 99999904140 | -- |
| `VETO` | identity | +0.000000 | none | -- |
| `RESCOPE` | outlier_mad | +0.061077 | none | 99999904140 |

### `task_D`

| arm | plan after | delayed | harmed | series taken out |
| --- | --- | ---: | --- | --- |
| `no_guard` | outlier_mad | +0.049504 | 99999904140, 99999963862 | -- |
| `VETO` | identity | +0.000000 | none | -- |
| `RESCOPE` | outlier_mad | +0.095879 | none | 99999904140, 99999963862 |

## B2 -- the Slow decision

`SLOW_PROPOSES_RESCOPE_MASK_HARMED_SERIES` -- min_per_series_gain lt -0.005000 on the delayed window -> RESCOPE_MASK_HARMED_SERIES

- draw 1: PROPOSAL

Guard `rescope-recalled-plan-series-harm`: min_per_series_gain `lt` -0.005000 -> RESCOPE_MASK_HARMED_SERIES on the delayed window.

> The observed recalled plan had positive aggregate delayed gain while one evaluation series lost -0.1255567, far below the declared -0.005 harm line. Rescoping only evaluation series below that line preserves the plan elsewhere, uses the measured per-series vector, and avoids an additional Consumer retrain.

## B4 -- the guard as proposed, over the whole v7 trajectory

`AS_PROPOSED_GUARD_CONTAINS_WITHOUT_COLLATERAL`.  applies_to: the model narrowed it to reused_skill_adoption_only, so a fresh-search adoption is not submitted to the guard at all

| step | mode | reused | checked | fired | plan after | delayed | harmed after |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `task_A` | FULL_PRICE_SEARCH | None | False | False | outlier_mad | +0.306380 | none |
| `task_B` | DIRECT_RECALL | False | False | False | identity | +0.000000 | none |
| `task_C` | DIRECT_RECALL | True | True | True | outlier_mad | +0.061077 | none |
| `task_D` | DIRECT_RECALL | True | True | True | outlier_mad | +0.095879 | none |

## Cost and integrity

- LLM calls: 1 / 6.
- Consumer retrains: 0 / 50.
- Frozen surface: 39 files, drift [].
- Wall seconds: 31.1.
