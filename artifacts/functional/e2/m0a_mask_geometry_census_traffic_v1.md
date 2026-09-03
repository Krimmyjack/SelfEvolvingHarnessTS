# M0a mask geometry census -- `traffic` v1

**Descriptive geometry census. Not authorization evidence. Does not modify the frozen M0a census.**

Every field below is produced by `census_row` of `evaluation/functional/run_e2_m0a_mask_geometry_census.py`, imported unmodified, so a traffic row is field-for-field the same object as a row of the frozen `m0a_mask_geometry_census_v1` artifact. That artifact is not opened for writing by this run; the batch-recipe tool keeps reading it verbatim by provenance (sha256 after this run `284cad38fb205b42ee49ff9fe029157b5a05773f041c0599b6a5f5d753dd7f10`).

0 LLM calls. 0 Support probes. 0 Outcome opened. `OBSERVABLE_FEATURES` unchanged, `extract_public_features` unchanged, **no threshold is fitted anywhere in this report**. No Skill, Episode, Gate, Schema or execution right follows from it.

## 0. Why traffic

The batch recipe adopted `outlier_iqr` on traffic with training series `6` reverted to identity, and its reverted-series geometry table came out empty: the frozen census covers T233, electricity and Weather and has no traffic rows. This run fills that hole so the question *is the geometry of a dropped series observable in advance* has a third batch of samples. The census is computed **after** the fact and had no part in the decision it describes.

## 1. Coverage and window

| item | value |
| --- | --- |
| Task Episode | `e1v2_task_01` |
| train series censused | 12 (`0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`) |
| eval series censused | 8 (reported separately, in no summary) |
| prefix rule | `values[uid][:support_origins[0]]` |
| recipe Support origins | [1104, 1368] |
| recipe delayed origins | [1800] |
| census cutoff | 1104 (farthest index read 1103) |
| recipe farthest index read | 1848 |
| `sealed_from_index` | 3072 |
| CSV rows loaded from disk | 1104 |

The CSV reader is called with `max_rows=1104`, so the window is structural: no traffic row past index 1103 is loaded at all, which is well inside both the recipe's own farthest read (1848) and the sealed boundary (3072).

Exposure: STRUCTURALLY_ACCEPTED_BUT_SOURCE_FAMILY_EXPOSURE_UNRESOLVED: PeMS SF Bay Area / monash:traffic_hourly family has unresolved prior exposure; this census reads the public prefix values[uid][:1104] only and does not open a sealed Outcome

The roster and both origin sets are checked against the frozen recipe artifact `artifacts/functional/e2/batch_recipe_traffic_v1.json` before anything is computed; all 6 checks pass.

## 2. Sanity

The same four checks the frozen census runs -- `union_pss` reproduces the public `post_shift_support_sufficient`, the union mask reconstructs `region_mask`, the union end fraction matches `estimated_region_end_fraction`, every field is finite -- over the 12 train rows: **PASS**. Including the 8 eval rows: **PASS**.

Comparability with the frozen census: `census_row` was re-run on the 24 frozen `e1v2_task_01` train rows of the full-report cohorts and compared field for field at exact equality -- **PASS**, 0 mismatches. This matters because `runtime/public_features.py` carries uncommitted M0b edits; the check makes their additivity a measured fact rather than a reading of the diff, and it is what licenses putting a traffic row next to a frozen one.

## 3. The 12 train series

| series | mask_class | outlier_region_frac | level_region_frac | outlier_end | level_end | union_frac | union_end | outlier_point_frac | z_peak | level_excursion | union_pss | level_only_pss | divergent |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `0` | MIXED | 0.119565 | 0.043478 | 0.923007 | 0.082428 | 0.163043 | 0.923007 | 0.039855 | 10.629918 | 3.282334 | False | True | True |
| `1` | MIXED | 0.129529 | 0.058877 | 0.923913 | 0.145833 | 0.178442 | 0.923913 | 0.049819 | 14.723737 | 3.791035 | False | True | True |
| `2` | MIXED | 0.009058 | 0.036232 | 0.932971 | 0.945652 | 0.040761 | 0.945652 | 0.001812 | 5.128660 | 2.671078 | False | False | False |
| `3` | MIXED | 0.032609 | 0.053442 | 0.932065 | 0.140399 | 0.086051 | 0.932065 | 0.007246 | 9.989594 | 4.800175 | False | True | True |
| `4` | MIXED | 0.192935 | 0.061594 | 0.944746 | 0.365942 | 0.242754 | 0.944746 | 0.091486 | 13.326181 | 4.058918 | False | True | True |
| `5` | MIXED | 0.177536 | 0.039855 | 0.953804 | 0.139493 | 0.206522 | 0.953804 | 0.076993 | 10.321200 | 3.052609 | False | True | True |
| `6` **(reverted)** | OUTLIER_ONLY | 0.192935 | 0.000000 | 0.953804 | 0.000000 | 0.192935 | 0.953804 | 0.084239 | 6.569200 | 0.000000 | False | True | True |
| `7` | MIXED | 0.093297 | 0.069746 | 0.952899 | 0.161232 | 0.149457 | 0.952899 | 0.031703 | 8.345807 | 2.925502 | False | True | True |
| `8` | LEVEL_ONLY | 0.000000 | 0.036232 | 0.000000 | 0.076993 | 0.036232 | 0.076993 | 0.000000 | 3.566757 | 3.341514 | True | True | False |
| `9` | OUTLIER_ONLY | 0.187500 | 0.000000 | 0.952899 | 0.000000 | 0.187500 | 0.952899 | 0.093297 | 18.888661 | 0.000000 | False | True | True |
| `10` | MIXED | 0.101449 | 0.036232 | 0.944746 | 0.076087 | 0.137681 | 0.944746 | 0.021739 | 8.195571 | 3.818160 | False | True | True |
| `11` | MIXED | 0.125906 | 0.036232 | 0.923007 | 0.075181 | 0.162138 | 0.923007 | 0.038949 | 10.432430 | 4.253421 | False | True | True |

## 4. Field non-degeneracy (12 train rows)

Degeneracy is read exactly as the frozen census reads it: a field is non-degenerate when it is finite everywhere and neither all-zero nor all-one.

| field | min | max | mean | distinct | all_zero | all_one | non_degenerate |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `outlier_region_fraction` | 0.000000 | 0.192935 | 0.113527 | 11 | False | False | True |
| `level_region_fraction` | 0.000000 | 0.069746 | 0.039327 | 8 | False | False | True |
| `outlier_region_end_fraction` | 0.000000 | 0.953804 | 0.861489 | 8 | False | False | True |
| `level_region_end_fraction` | 0.000000 | 0.945652 | 0.184103 | 11 | False | False | True |
| `outlier_point_fraction` | 0.000000 | 0.093297 | 0.044761 | 12 | False | False | True |
| `missing_region_end_fraction` | 0.000000 | 0.000000 | 0.000000 | 1 | True | False | False |
| `union_region_end_fraction` | 0.076993 | 0.953804 | 0.868961 | 8 | False | False | True |
| `union_region_fraction` | 0.036232 | 0.242754 | 0.148626 | 12 | False | False | True |
| `missing_fraction` | 0.000000 | 0.000000 | 0.000000 | 1 | True | False | False |
| `local_robust_z_peak` | 3.566757 | 18.888661 | 10.009810 | 12 | False | False | True |
| `level_excursion_score` | 0.000000 | 4.800175 | 2.999562 | 11 | False | False | True |

Degenerate on this batch: `missing_fraction`, `missing_region_end_fraction`.

## 5. `mask_class` distribution

| class | count | fraction |
| --- | ---: | ---: |
| `MIXED` | 9 | 0.7500 |
| `OUTLIER_ONLY` | 2 | 0.1667 |
| `LEVEL_ONLY` | 1 | 0.0833 |
| `AMBIGUOUS` | 0 | 0.0000 |

`MIXED` = expanded outlier region and `level_mask` both non-empty; `AMBIGUOUS` = both empty.

## 6. `union_pss` vs `level_only_pss`

| quantity | value |
| --- | ---: |
| decision points | 12 |
| `union_pss` true | 1 |
| `level_only_pss` true | 11 |
| divergent | 10 |
| divergent fraction | 0.8333 |
| divergence sources | {"OUTLIER": 10} |

`OUTLIER` / `MISSING` / `BOTH` name the region whose expanded tail attains the union's last True index, i.e. the region that pushed the union end fraction up and flipped pss away from the level-only reading.

## 7. Descriptive contrast: the reverted series vs the retained ones

**DESCRIPTIVE ONLY. No threshold is fitted, no Observation is wired, and nothing in this block feeds any adoption rule, Gate or verdict.**

Group split: `6` ({"OUTLIER_ONLY": 1}) vs 11 retained ({"LEVEL_ONLY": 1, "MIXED": 9, "OUTLIER_ONLY": 1}). Selection basis: chosen by the batch recipe's greedy Support-window mask search, every step validated by a real retrain; the geometry below played no part in that decision and is read off afterwards.

| field | excluded mean | retained min | retained median | retained max | ranges overlap | direction | excluded rank (of 12) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `outlier_region_fraction` | 0.192935 | 0.000000 | 0.119565 | 0.192935 | True | excluded_higher | 6:2 |
| `level_region_fraction` | 0.000000 | 0.000000 | 0.039855 | 0.069746 | True | excluded_lower | 6:11 |
| `outlier_region_end_fraction` | 0.953804 | 0.000000 | 0.932971 | 0.953804 | True | excluded_higher | 6:2 |
| `level_region_end_fraction` | 0.000000 | 0.000000 | 0.139493 | 0.945652 | True | excluded_lower | 6:11 |
| `union_region_fraction` | 0.192935 | 0.036232 | 0.162138 | 0.242754 | True | excluded_higher | 6:3 |
| `union_region_end_fraction` | 0.953804 | 0.076993 | 0.944746 | 0.953804 | True | excluded_higher | 6:2 |
| `outlier_point_fraction` | 0.084239 | 0.000000 | 0.038949 | 0.093297 | True | excluded_higher | 6:3 |
| `local_robust_z_peak` | 6.569200 | 3.566757 | 10.321200 | 18.888661 | True | excluded_lower | 6:10 |
| `level_excursion_score` | 0.000000 | 0.000000 | 3.341514 | 4.800175 | True | excluded_lower | 6:11 |

Rank is descending within the whole 12-series batch (1 = highest value); ties are broken by series uid, so a rank can understate a series that holds a shared maximum.

Exact-equality ties with retained series: `6` ties `4` exactly on `outlier_region_fraction` (0.192935); `6` ties `9` exactly on `level_region_fraction` (0.000000); `6` ties `5` exactly on `outlier_region_end_fraction` (0.953804); `6` ties `9` exactly on `level_region_end_fraction` (0.000000); `6` ties `5` exactly on `union_region_end_fraction` (0.953804); `6` ties `9` exactly on `level_excursion_score` (0.000000).

Fields whose observed ranges do not overlap between the two groups: none.

## 8. Descriptive contrast: unadopted masked candidate `outlier_mad`, would have reverted 5, 6, 7 (adoption trace: NOT_REACHED)

**DESCRIPTIVE ONLY. No threshold is fitted, no Observation is wired, and nothing in this block feeds any adoption rule, Gate or verdict.**

Group split: `5`, `6`, `7` ({"MIXED": 2, "OUTLIER_ONLY": 1}) vs 9 retained ({"LEVEL_ONLY": 1, "MIXED": 7, "OUTLIER_ONLY": 1}). Selection basis: the other masked plan the same search produced; `NOT_REACHED` means the adoption rule stopped at the first candidate that passed the delayed stability check and never judged this one. It is reported so the descriptive reading is not conditioned on the winning mask alone.

The excluded group has more than one member, so its per-series values are listed before the group summary:

| field | `5` | `6` | `7` |
| --- | ---: | ---: | ---: |
| `outlier_region_fraction` | 0.177536 | 0.192935 | 0.093297 |
| `level_region_fraction` | 0.039855 | 0.000000 | 0.069746 |
| `outlier_region_end_fraction` | 0.953804 | 0.953804 | 0.952899 |
| `level_region_end_fraction` | 0.139493 | 0.000000 | 0.161232 |
| `union_region_fraction` | 0.206522 | 0.192935 | 0.149457 |
| `union_region_end_fraction` | 0.953804 | 0.953804 | 0.952899 |
| `outlier_point_fraction` | 0.076993 | 0.084239 | 0.031703 |
| `local_robust_z_peak` | 10.321200 | 6.569200 | 8.345807 |
| `level_excursion_score` | 3.052609 | 0.000000 | 2.925502 |

| field | excluded mean | retained min | retained median | retained max | ranges overlap | direction | excluded rank (of 12) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `outlier_region_fraction` | 0.154589 | 0.000000 | 0.119565 | 0.192935 | True | excluded_higher | 6:2, 5:4, 7:9 |
| `level_region_fraction` | 0.036534 | 0.000000 | 0.036232 | 0.061594 | True | excluded_lower | 7:1, 5:6, 6:11 |
| `outlier_region_end_fraction` | 0.953502 | 0.000000 | 0.932065 | 0.952899 | True | excluded_higher | 5:1, 6:2, 7:3 |
| `level_region_end_fraction` | 0.100242 | 0.000000 | 0.082428 | 0.945652 | True | excluded_lower | 7:3, 5:6, 6:11 |
| `union_region_fraction` | 0.182971 | 0.036232 | 0.162138 | 0.242754 | True | excluded_higher | 5:2, 6:3, 7:8 |
| `union_region_end_fraction` | 0.953502 | 0.076993 | 0.932065 | 0.952899 | True | excluded_higher | 5:1, 6:2, 7:3 |
| `outlier_point_fraction` | 0.064312 | 0.000000 | 0.038949 | 0.093297 | True | excluded_higher | 6:3, 5:4, 7:8 |
| `local_robust_z_peak` | 8.412069 | 3.566757 | 10.432430 | 18.888661 | True | excluded_lower | 5:6, 7:8, 6:10 |
| `level_excursion_score` | 1.992704 | 0.000000 | 3.791035 | 4.800175 | True | excluded_lower | 5:8, 7:9, 6:11 |

Rank is descending within the whole 12-series batch (1 = highest value); ties are broken by series uid, so a rank can understate a series that holds a shared maximum.

Exact-equality ties with retained series: `6` ties `4` exactly on `outlier_region_fraction` (0.192935); `6` ties `9` exactly on `level_region_fraction` (0.000000); `7` ties `9` exactly on `outlier_region_end_fraction` (0.952899); `6` ties `9` exactly on `level_region_end_fraction` (0.000000); `7` ties `9` exactly on `union_region_end_fraction` (0.952899); `6` ties `9` exactly on `level_excursion_score` (0.000000).

Fields whose observed ranges do not overlap between the two groups: none.

## 9. Screening `local_robust_z_peak`, side by side

The screening artifact `artifacts/functional/e2/g3_candidate_screening_v2.json` computed `local_robust_z_peak` on `values[uid][:3072] (frozen roster support_origins[0])`; this census computes it on `values[uid][:1104] (recipe development origin)`. **Different windows: the two columns are printed side by side and are never differenced, ranked together or thresholded.**

| series | screening z_peak (prefix 3072) | census z_peak (prefix 1104) |
| --- | ---: | ---: |
| `0` | 12.781086 | 10.629918 |
| `1` | 15.717203 | 14.723737 |
| `2` | 4.036137 | 5.128660 |
| `3` | 11.025587 | 9.989594 |
| `4` | 13.787171 | 13.326181 |
| `5` | 12.250520 | 10.321200 |
| `6` **(reverted)** | 6.572979 | 6.569200 |
| `7` | 8.205898 | 8.345807 |
| `8` | 3.698289 | 3.566757 |
| `9` | 15.314490 | 18.888661 |
| `10` | 7.995596 | 8.195571 |
| `11` | 14.584142 | 10.432430 |

## 10. What this does not say

- It does not say that any field predicts exclusion. One batch, one reverted series, and a rank is not a threshold.
- It does not fit, propose or imply a threshold, cut point or rule on any field, and nothing here is wired into a Gate, Schema, Observation or adoption rule.
- It does not revisit the adopted plan. That plan was chosen by real retrains on the Support window and is unchanged by this artifact.
- It does not touch the frozen census. The batch-recipe tool still reads `artifacts/functional/e2/m0a_mask_geometry_census_v1.json` verbatim; this run wrote `artifacts/functional/e2/m0a_mask_geometry_census_traffic_v1.json` instead.
- It is not authorization evidence: no Skill is written, no Episode is formed, no Fast or Slow path is entered, no Gate or Schema is proposed, and no execution right is granted or implied.

## Provenance

- row function: `run_e2_m0a_mask_geometry_census.census_row, imported unmodified, so every field is computed exactly as in m0a_mask_geometry_census_v1`
- extractor: `SelfEvolvingHarnessTS.runtime.public_features.extract_public_features (unmodified)`
- expansion: `runtime.public_features._expand (radius=2)`
- pss formula: `max(0, (1 - end_fraction) * _DOWNSTREAM_WINDOW_POINTS) >= _POST_SHIFT_SUPPORT_MIN_POINTS, constants imported from the extractor module`
- recipe artifact read for roster, windows and plan: `artifacts/functional/e2/batch_recipe_traffic_v1.json` (`batch_recipe_v1`)
- CSV rows read from disk: 1104
- not read: KDD W3 T211-T230 (INSTANCE_UNSEEN preserved); any sealed Outcome (NOAA, g3_final_query_outcome, delayed truth); traffic rows at or past index 3072, and in fact any traffic row at or past index 1104

