# Weather per-task Judge readability v1

Instrument ledger only. 0 LLM, zero experiment, no new Outcome, no Consumer retrain.

## Question

The frozen **aggregate** Weather Utility ruling is `METRIC_UNREADABLE` (`g2_shakedown_weather_report.json` `weather_utility_ruling`, `aggregate_utility_readouts`). That does not automatically make every Weather Task unreadable. This file only asks: which already-exposed Weather Tasks have a Judge that is readable under the **pre-registered** gates, so that a later citation could put their sign on an authorization ledger.

## Pre-registered gates (not moved)

Authority: `evaluation/functional/task_episode_harness/agentic/g3_sourcing.py` `development_judge_readability` (lines 156–204).

- `eval_loss_spread = max(losses) / min(losses)` if `min > 0`, else `inf`. Pass iff `<= MAX_EVAL_LOSS_SPREAD` (`5.0`).
- `largest_single_series_loss_share = max(losses) / sum(losses)` if `sum > 0`, else `1.0`. Pass iff `<= MAX_SINGLE_SERIES_LOSS_SHARE` (`0.40`).
- `losses[i]` is the mean identity `per_view_smase` of eval channel `i` over the development origins `(1104, 1368, 1800)`.

Only these two gates. Fail → `TASK_UNREADABLE`. Missing reusable per-channel identity loss → `INSUFFICIENT_TRACE`. No substitute metric, no gain inversion, no cohort-24.4× applied as a per-task verdict.

## Exposed Weather Task roster

Nineteen Tasks, `e1v2_task_01` … `e1v2_task_19`.

| Source | What it shows |
| --- | --- |
| `w1_task_episode_harness_report.json` `weather_feasibility.context_census` (249692–249890) | all 19 ids listed as valid |
| same file `weather_a5a3_autonomous_guidance` (`available_task_count` / `n` = 19 at 251190 / 251197; `stage_decomposition.per_task` from 251381) | all 19 opened |
| `g2_shakedown_weather_report.json` (`task_count` 9 at line 7; ids 01–09 at 592–600) | subset of the same roster |
| `secondsource_census_v1.json` / `.md` | same 19 ids, namespaced `weather:e1v2_task_XX` |

Shared eval channels (file order, 11 train / 8 eval): `wv (m/s)`, `max. wv (m/s)`, `wd (deg)`, `SWDR`, `PAR`, `max. PAR`, `Tlog (degC)`, `OT`. Cited: g2 `eval_substrate_preflight.eval_series` 26–35; w1 `weather_a5a3` `eval_series` 251154–251163.

`g3_sourcing.py` development-block path (`DEVELOPMENT_ORIGINS` 49–50, `SEALED_FROM_INDEX = 3072`) was used to screen later candidates. Weather is not a screened candidate in `g3_candidate_screening*.json`; those files only mention Weather’s 24.4× as rationale. No Weather `per_series_identity_smase` vector is stored there.

## Per-channel identity loss search

`development_judge_readability` needs identity `per_view_smase` per eval channel. Persisted Weather artifacts do not have it.

- g2 / w1 delayed blocks store `per_series_mean_gain` (program minus identity). Example: g2 task_01 delayed at 2020–2028. Gains are not identity losses and are not inverted here.
- Grep of `g2_shakedown_weather_report.json` and the `weather_a5a3` / `weather_feasibility` blocks of `w1_task_episode_harness_report.json`: no `per_view_smase`, no `per_series_identity_smase`, no `identity_smase`.
- The 24.4× figure and “sMASE about 17” in `weather_utility_ruling` (8247–8263) are cohort-level prose. The eight channel values were not written down. That aggregate number is not applied to any Task.
- Calling `_evaluate_origins` / `development_judge_readability` would retrain the ridge Consumer (`run_e2_autonomous_natural_workflow_generation.py` `_evaluate`). Not done.

Every Task is therefore `INSUFFICIENT_TRACE`.

## Counts

| verdict | n | ids |
| --- | ---: | --- |
| READABLE | 0 | — |
| TASK_UNREADABLE | 0 | — |
| INSUFFICIENT_TRACE | 19 | `e1v2_task_01` … `e1v2_task_19` |

**Even if a READABLE Task existed, its sign must not be written back into the current Source authorization ledger unless a separate one-shot pre-registration is opened. This report is instrument accounting only.**

## Ruling

`WEATHER_PER_TASK_READABILITY_NO_USABLE_SYMBOLS`

No Task has a reusable identity-loss vector, so no Task can pass or fail the two pre-registered gates. There are no usable per-task symbols for a later authorization citation.

This does **not** require another Weather Metric repair. The aggregate protocol is already `METRIC_UNREADABLE`; this ledger only records that the per-task Judge-readability question also has no usable symbol under the frozen gates.
