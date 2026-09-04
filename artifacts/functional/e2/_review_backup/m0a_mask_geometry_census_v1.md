# M0a outlier / level mask geometry census v1

0 LLM calls. 0 Support probes. 0 Outcome opened. `OBSERVABLE_FEATURES` unchanged. `extract_public_features` unchanged. KDD W3 T211-T230 not read (INSTANCE_UNSEEN preserved).

Frozen offline diagnostic only. Not a Capability, not a Claim, not a Promotion. No threshold is fitted anywhere in this report.

- decision point = one train-series public prefix `values[uid][:support_origins[0]]` at one Task Episode origin
- cohorts with the full report: T233, electricity
- cohorts with field distribution and coverage only: weather (no Utility / gain symbol is read or computed; its aggregate utility is METRIC_UNREADABLE)
- pss constants imported from `runtime.public_features` (`_DOWNSTREAM_WINDOW_POINTS`, `_POST_SHIFT_SUPPORT_MIN_POINTS`), never copied as literals

## 0. Coverage

| cohort | tasks | train series | decision points | cutoffs |
| --- | ---: | ---: | ---: | --- |
| T233 | 19 | 12 | 228 | 3072..8256 |
| electricity | 9 | 12 | 108 | 3072..5376 |
| weather | 19 | 11 | 209 | 3072..8256 |

Sanity check (`union_pss` == public `post_shift_support_sufficient`, union mask reconstruction, union end fraction, field finiteness) over all 545 rows: **PASS**.

## a) Field non-degeneracy

Full-report cohorts pooled (T233 + electricity).

| field | min | max | mean | distinct | all_zero | all_one | non_degenerate |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `outlier_region_fraction` | 0.000000 | 0.320417 | 0.054849 | 296 | False | False | True |
| `level_region_fraction` | 0.000000 | 0.031250 | 0.010146 | 171 | False | False | True |
| `outlier_region_end_fraction` | 0.000000 | 1.000000 | 0.585223 | 289 | False | False | True |
| `level_region_end_fraction` | 0.000000 | 0.995722 | 0.219416 | 293 | False | False | True |
| `outlier_point_fraction` | 0.000000 | 0.213216 | 0.030811 | 285 | False | False | True |
| `missing_region_end_fraction` | 0.000000 | 0.000000 | 0.000000 | 1 | True | False | False |
| `union_region_end_fraction` | 0.000000 | 1.000000 | 0.620856 | 310 | False | False | True |
| `union_region_fraction` | 0.000000 | 0.323242 | 0.061352 | 320 | False | False | True |
| `missing_fraction` | 0.000000 | 0.000000 | 0.000000 | 1 | True | False | False |

All 336 pooled rows have finite values on every reported field: True.

Per-cohort field stats, including the field-distribution-only cohort, are in the JSON under `per_cohort.<cohort>.field_stats`.

## b) T233 `task_01` (hampel NEGATIVE) vs `task_13..19` (outlier POSITIVE)

Labels are taken verbatim from the frozen M0a instruction. Values are the representative series of each Task under the frozen `public_context` scope rule. Descriptive reading only.

| task | label | representative | mask_class | outlier_region_frac | level_region_frac | outlier_end | level_end | union_end | union_pss | level_only_pss |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| e1v2_task_01 | hampel NEGATIVE | T234 | MIXED | 0.133138 | 0.018555 | 0.985677 | 0.182617 | 0.985677 | False | True |
| e1v2_task_13 | outlier POSITIVE | T256 | MIXED | 0.057751 | 0.006127 | 0.990962 | 0.319240 | 0.990962 | False | True |
| e1v2_task_14 | outlier POSITIVE | T256 | MIXED | 0.060886 | 0.005869 | 0.949090 | 0.305751 | 0.949090 | False | True |
| e1v2_task_15 | outlier POSITIVE | T256 | MIXED | 0.056166 | 0.005631 | 0.984938 | 0.293356 | 0.984938 | False | True |
| e1v2_task_16 | outlier POSITIVE | T256 | MIXED | 0.059524 | 0.005411 | 0.997159 | 0.281926 | 0.997159 | False | True |
| e1v2_task_17 | outlier POSITIVE | T256 | MIXED | 0.057292 | 0.005208 | 0.959766 | 0.271354 | 0.959766 | False | True |
| e1v2_task_18 | outlier POSITIVE | T256 | MIXED | 0.055848 | 0.005020 | 0.993599 | 0.261546 | 0.993599 | False | True |
| e1v2_task_19 | outlier POSITIVE | T256 | MIXED | 0.055354 | 0.004845 | 0.958939 | 0.252422 | 0.958939 | False | True |

Observed-range overlap (no threshold fitted):

| field | task_01 | task_13..19 min | task_13..19 max | task_01 inside range | direction |
| --- | ---: | ---: | ---: | --- | --- |
| `outlier_region_fraction` | 0.133138 | 0.055354 | 0.060886 | False | negative_above_positive_range |
| `level_region_fraction` | 0.018555 | 0.004845 | 0.006127 | False | negative_above_positive_range |
| `outlier_region_end_fraction` | 0.985677 | 0.949090 | 0.997159 | True | overlapping |
| `level_region_end_fraction` | 0.182617 | 0.252422 | 0.319240 | False | negative_below_positive_range |

Fields on which `task_01` falls outside the positive group's observed range: `level_region_end_fraction`, `level_region_fraction`, `outlier_region_fraction`.

Caveat, and it is load-bearing for how this table may be read: the frozen public_context representative rule selects a different train series for the negative Task than for the positive Tasks, so the representative-level contrast mixes a Task-origin difference with a series-identity difference; the train-scope-mean columns are reported as the cross-check (negative Task representative `T234`, positive Task representatives `T256`).

Same overlap reading on the train-scope mean over all train series of each Task, which removes the representative-series identity change:

| field | task_01 mean | task_13..19 min | task_13..19 max | task_01 inside range | direction |
| --- | ---: | ---: | ---: | --- | --- |
| `outlier_region_fraction` | 0.041287 | 0.041586 | 0.045616 | False | negative_below_positive_range |
| `level_region_fraction` | 0.015625 | 0.005582 | 0.007353 | False | negative_above_positive_range |
| `outlier_region_end_fraction` | 0.607775 | 0.581059 | 0.794982 | True | overlapping |
| `level_region_end_fraction` | 0.332764 | 0.149978 | 0.218902 | False | negative_above_positive_range |

Fields that separate in the **same direction** under both the representative reading and the train-scope-mean reading: `level_region_fraction`. Fields that separate only under the representative reading, or that flip direction between the two readings, are not a stable descriptive separation and are reported as such.

## c) mixed / ambiguous shares

| cohort | MIXED | OUTLIER_ONLY | LEVEL_ONLY | AMBIGUOUS | mixed_frac | ambiguous_frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T233 | 211 | 0 | 17 | 0 | 0.9254 | 0.0000 |
| electricity | 80 | 15 | 4 | 9 | 0.7407 | 0.0833 |
| weather | 70 | 0 | 139 | 0 | 0.3349 | 0.0000 |

`MIXED` = expanded outlier region and `level_mask` are both non-empty, so the union folds two mechanisms into one region. `AMBIGUOUS` = both are empty.

## d) `union_pss` != `level_only_pss`

| cohort | decision points | divergent | fraction | source breakdown |
| --- | ---: | ---: | ---: | --- |
| T233 | 228 | 89 | 0.3904 | {"OUTLIER": 89} |
| electricity | 108 | 27 | 0.2500 | {"OUTLIER": 27} |
| weather | 209 | 29 | 0.1388 | {"OUTLIER": 29} |

Full-report cohorts pooled: 116 / 336 = 0.3452 divergent, sources {"OUTLIER": 116}.

`OUTLIER` / `MISSING` / `BOTH` name the region whose expanded tail attains the union's last True index, i.e. the region that pushed the union end fraction up and flipped pss away from the level-only reading. When `missing_fraction > 0` the frozen extractor forces `level_mask` to all-zero, so a `MISSING` divergence is a structural consequence of that branch, not an independent measurement.

## e) Verdict

**INFORMATIVE**

Pre-stated rule: non-degenerate split-geometry fields AND (range separation on >=1 field OR non-zero pss divergence).

- all four split-geometry fields non-degenerate: True
- criterion (i) range separation: True (`level_region_end_fraction`, `level_region_fraction`, `outlier_region_fraction`)
- criterion (ii) pss divergence fraction: True (0.3452)

this verdict concerns only the outlier/level mask-geometry candidate Observation; it is not a Capability family termination.

### M0b minimal field-set suggestion (not implemented here)

The descriptive separation is a coverage difference (`level_region_fraction`); the tail positions of the two masks overlap on this cohort. `level_region_end_fraction`, `outlier_region_fraction` separate only on the representative series and lose the separation or flip direction on the train-scope mean, so they are not carried into the suggestion. `post_shift_support_sufficient` disagrees with its level-only reading on 34.52% of the full-report decision points (sources {"OUTLIER": 116}), so the field as currently wired reports a support margin that the level mechanism does not own. Minimal M0b field set to consider wiring, in this order: `level_region_end_fraction`, `level_region_fraction`, `outlier_region_end_fraction`. A level-only pss reading is the single decision-relevant derived field; the outlier tail field is only needed to explain why the union reading differs. Nothing beyond these is justified by M0a, and M0a does not implement any of them.

## Provenance

- extractor: `SelfEvolvingHarnessTS.runtime.public_features.extract_public_features (unmodified)`
- expansion: `runtime.public_features._expand (radius=2, the same helper the union uses)`
- pss formula: `max(0, (1 - end_fraction) * _DOWNSTREAM_WINDOW_POINTS) >= _POST_SHIFT_SUPPORT_MIN_POINTS, constants imported`
- rosters: {"T233": ["e1v2_task_01", "e1v2_task_02", "e1v2_task_03", "e1v2_task_04", "e1v2_task_05", "e1v2_task_06", "e1v2_task_07", "e1v2_task_08", "e1v2_task_09", "e1v2_task_10", "e1v2_task_11", "e1v2_task_12", "e1v2_task_13", "e1v2_task_14", "e1v2_task_15", "e1v2_task_16", "e1v2_task_17", "e1v2_task_18", "e1v2_task_19"], "electricity": ["e1v2_task_01", "e1v2_task_02", "e1v2_task_03", "e1v2_task_04", "e1v2_task_05", "e1v2_task_06", "e1v2_task_07", "e1v2_task_08", "e1v2_task_09"], "weather": ["e1v2_task_01", "e1v2_task_02", "e1v2_task_03", "e1v2_task_04", "e1v2_task_05", "e1v2_task_06", "e1v2_task_07", "e1v2_task_08", "e1v2_task_09", "e1v2_task_10", "e1v2_task_11", "e1v2_task_12", "e1v2_task_13", "e1v2_task_14", "e1v2_task_15", "e1v2_task_16", "e1v2_task_17", "e1v2_task_18", "e1v2_task_19"]}
- roster reconstruction: rosters come from the existing agentic.runner.load_cohort. For electricity and Weather that helper applies the repo's frozen outcome-blind substrate preflights to pick the same exposed columns; those preflights are the pre-Outcome Judge-readability guards, not an Outcome read. The census itself only reads values[uid][:support_origins[0]].
- not read: KDD W3 T211-T230 (INSTANCE_UNSEEN preserved, no Context read), any sealed Outcome (NOAA, g3_final_query_outcome, delayed truth), any Weather Utility or gain symbol (METRIC_UNREADABLE)

