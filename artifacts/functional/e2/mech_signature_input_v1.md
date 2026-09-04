# Mechanism-signature input v1 (T3 → mainline R2 evidence view)

Instrument / input only. Not a Skill, not a clause, not a Claim.
Source: already-exposed T233 development tool receipts in
`g3d1_t233_source_extension.json` (19-task agentic run) and
`g1_agentic_pipeline_report_T233.json` (9-task corroboration for task_01).
Public tools confirmed in `agentic/gateway.py` and `methods/ttha/public_tools.py`:
`summarize_series` and `localize_regions`.

Slow Path may PATCH or ABSTAIN. This file does not compile a threshold or
write a rule from these outcomes.

## Candidate signatures and existing tool fields

| candidate signature | existing public field | tool | in closed vocabulary? |
| --- | --- | --- | --- |
| extreme deviation exists (peak robust-z) | `local_robust_z_peak` | `summarize_series` → `features` | yes |
| affected interval (union of missing / outlier / level masks) | `estimated_region_start_fraction`, `estimated_region_end_fraction` | both tools | yes |
| level-excursion co-occurrence | `level_excursion_score`, `estimated_level_offset` | `summarize_series` → `features` | yes |
| post-shift support on that series | `post_shift_support_sufficient` | `summarize_series` → `features` | yes |
| missingness | `missing_fraction`, `longest_missing_run_fraction` | `summarize_series` → `features` | yes |

`contracts/observables.py` already names `OUTLIER_Z_THRESHOLD = 4.0` as the
public robust-z vocabulary shared by the extractor. That constant is **not**
re-fit from the T233 cells below. Whether Slow conditions a clause on
`local_robust_z_peak`, and with what wording, is out of scope here.

Not returned by either public tool (so **not** available as a clause
condition unless Observation is added later):

- `outlier_indices` / outlier count (computed inside the extractor, dropped
  before the tool receipt)
- IQR fences, Tukey k, Hampel window, MAD-at-window
- point-level residual series

On this T233 slice, `local_robust_z_peak` **can** express “a large robust-z
peak was observed”. It cannot express “the peak is an IQR outlier rather than
a level excursion”. Every listed task also shows a large
`level_excursion_score` on the same series. That co-occurrence is visible in
the table; it is not a fitted interaction rule.

## Census cells in scope

From `g3d1_source_derived_skill.json` (T233, 19 Task Episodes):

- `outlier_iqr` POSITIVE, `post_shift_support_sufficient=false`:
  e1v2_task_13/14/15/16/17/19
- `hampel_filter` POSITIVE, `post_shift_support_sufficient=false`:
  e1v2_task_15/16/18/19
- `hampel_filter` NEGATIVE, `post_shift_support_sufficient=false`:
  e1v2_task_01

Numbers below are from `summarize_series` receipts actually served to the
Agent (A3 arm; A5 saw the same peak series). Task-level
`post_shift_support_sufficient` is False on all eight tasks. Series-level
`post_shift_support_sufficient` can differ inside the same task.

## Per-task peak series (the observable the Agent actually fetched)

| task | census cell | series at max z | `local_robust_z_peak` | `level_excursion_score` | `estimated_level_offset` | series `pss` | region start–end | `missing_fraction` | n series summarized |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |
| e1v2_task_01 | hampel NEGATIVE | T234 | 28.3129 | 14.0937 | 35.6 | false | 0.0000–0.9857 | 0.0 | 5 |
| e1v2_task_13 | outlier_iqr POSITIVE | T234 | 35.9861 | 17.8244 | 35.6 | false | 0.0000–0.9914 | 0.0 | 4 |
| e1v2_task_14 | outlier_iqr POSITIVE | T234 | 38.2774 | 18.9384 | 35.6 | false | 0.0000–0.9507 | 0.0 | 5 |
| e1v2_task_15 | outlier_iqr POSITIVE and hampel POSITIVE | T234 | 38.2774 | 18.9384 | 35.6 | false | 0.0000–0.9848 | 0.0 | 1 |
| e1v2_task_16 | outlier_iqr POSITIVE and hampel POSITIVE | T234 | 37.0970 | 18.3645 | 35.6 | false | 0.0000–1.0000 | 0.0 | 4 |
| e1v2_task_17 | outlier_iqr POSITIVE | T234 | 37.0970 | 18.3645 | 35.6 | false | 0.0000–0.9970 | 0.0 | 5 |
| e1v2_task_18 | hampel POSITIVE | T234 | 37.0970 | 18.3645 | 35.6 | false | 0.0000–0.9888 | 0.0 | 3 |
| e1v2_task_19 | outlier_iqr POSITIVE and hampel POSITIVE | T234 | 37.0970 | 18.3645 | 35.6 | false | 0.0000–0.9771 | 0.0 | 5 |

`localize_regions` on the same series returns the same
`estimated_region_start_fraction` / `estimated_region_end_fraction` pair.
No additional extreme-deviation field exists there.

Min `local_robust_z_peak` among summarized series on these tasks is still
above 6 (task_13 T236 = 6.1454 is the lowest). Missingness is 0 on every
listed receipt.

## What this does **not** distinguish

task_01 (hampel NEGATIVE) and the outlier_iqr POSITIVE tasks share the same
peak series family (T234), the same order of `local_robust_z_peak` (28–38),
and a large `level_excursion_score`. A clause that says only “when
`local_robust_z_peak` is large, prefer `outlier_iqr`” would be reading a
quantity that is also present on the hampel-negative cell. Slow must see that
overlap; this input does not resolve it.

## Observation gap (if Slow needs more than the current tools)

If the intended mechanism is “IQR-style point outliers rather than a level
step”, current public tools **cannot** say that. That would be
`OBSERVATION_REQUIRED`. This task does not add a field or a tool.

## Declaration

This input contains no threshold fitted from T233 (or any other) Outcome.
`OUTLIER_Z_THRESHOLD = 4.0` is the pre-existing public-feature contract, not
a number estimated from the table above.
