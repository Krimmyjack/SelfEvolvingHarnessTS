# closing the runtime loop: an out-of-selection delayed probe

**Overall: `LIFECYCLE_CLOSES`** -- 3 of 5 Drafts changed state through the real path: T1_A5 -> LOCAL_ACTIVE, T2_A3 -> LOCAL_ACTIVE, T2_A5 -> LOCAL_ACTIVE; T3_A3, T3_A5 had no legal probe window.

The integration slice formed five `LOCAL_DRAFT` Episodes on the real channel and stopped there, honestly: its delayed reading had set the adoption bar, so it was in-selection and could not serve as promotion evidence.  This slice supplies what was missing -- one delayed probe per Draft on a window that took part in no selection -- and hands it to the existing update path.

**No direction is presumed.**  `_update_delayed` grades three bands; a probe that restricts a Draft closes the loop exactly as well as one that promotes it.

0 LLM calls.  The probe grades an already-adopted plan and chooses nothing: no plan re-proposed, no shortlist re-run, no mask re-searched, no threshold touched.  State changes land under `_scratch/skill_store/lifecycle_probe_v1` only.

## Per Draft

| draft | plan | probe window | origins | source | probe delayed | status before -> after | outcome |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `T1_A5` | `outlier_iqr` full batch | `e1v2_task_05` | 4368, 4416, 4464 | quoted from the frozen roster | +0.093583 | `LOCAL_DRAFT` -> `LOCAL_ACTIVE` | `TRANSITIONED` |
| `T2_A3` | `repair_level_shift` minus 0, 1, 10, 11, 3 | `e1v2_task_05` | 4368, 4416, 4464 | quoted from the frozen roster | +0.056720 | `LOCAL_DRAFT` -> `LOCAL_ACTIVE` | `TRANSITIONED` |
| `T2_A5` | `outlier_iqr` full batch | `e1v2_task_05` | 4368, 4416, 4464 | quoted from the frozen roster | +0.040501 | `LOCAL_DRAFT` -> `LOCAL_ACTIVE` | `TRANSITIONED` |
| `T3_A3` | `hampel_filter` full batch | `W4_traffic_shift_probe_shift_480` | 3384 | chosen | -- | `--` -> `--` | `PROBE_WINDOW_UNAVAILABLE` |
| `T3_A5` | `outlier_iqr` full batch | `W4_traffic_shift_probe_shift_480` | 3384 | chosen | -- | `--` -> `--` | `PROBE_WINDOW_UNAVAILABLE` |

## Evidence fields, before and after

| draft | evidence level | relation | delayed evaluated | delayed gain | se_block | gain/se | block origins |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `T1_A5` before | `SUPPORT` | `POSITIVE` | True | +0.244845 | -- | -- | 4080, 4128, 4176 |
| `T1_A5` after | `DELAYED` | `POSITIVE` | True | +0.093583 | 0.017572 | 5.326 | 4368, 4416, 4464 |
| `T2_A3` before | `SUPPORT` | `POSITIVE` | True | +0.024745 | -- | -- | 4080, 4128, 4176 |
| `T2_A3` after | `DELAYED` | `POSITIVE` | True | +0.056720 | 0.014913 | 3.803 | 4368, 4416, 4464 |
| `T2_A5` before | `SUPPORT` | `POSITIVE` | True | +0.034455 | -- | -- | 4080, 4128, 4176 |
| `T2_A5` after | `DELAYED` | `POSITIVE` | True | +0.040501 | 0.007002 | 5.784 | 4368, 4416, 4464 |

## The update path

- call: `evaluation/functional/task_episode_harness/e1.py::_update_delayed`
- signature: `_update_delayed(episode, delayed_probe, delayed_origins)`
- it reads: episode.support_response['gain'] and delayed_probe['macro_gain'], 'se_block', 'gain_over_se'
- the bands: Support below the material threshold grades EPISODE_ONLY/NEGATIVE; otherwise delayed at or above +threshold grades LOCAL_ACTIVE/POSITIVE, at or below -threshold grades RESTRICTED/CONFLICT, and anything between grades LOCAL_DRAFT/ABSTAIN with evidence_level raised to DELAYED
- the probe is built by `evaluation/functional/task_episode_harness/runner.py::_arm_metrics`, the same cluster-unit metric the real path uses, over rows from `evaluation/functional/run_batch_composition_headroom.py::_evaluate_variant / _evaluate_assignment`.
- nothing is set by hand: the status is whatever that function returns.

## Probe windows

- **T1_A5** (T233, pooled): `e1v2_task_05`, origins [4368, 4416, 4464], task_episode_harness.e1._frozen_task_roster()[4], e1v2_task_05, delayed_origins verbatim; the Draft was graded on e1v2_task_04.  Farthest read 4512; inside the boundary.
- **T2_A3** (electricity, per_channel): `e1v2_task_05`, origins [4368, 4416, 4464], task_episode_harness.e1._frozen_task_roster()[4], e1v2_task_05, delayed_origins verbatim; the Draft was graded on e1v2_task_04.  Farthest read 4512; inside the boundary.
- **T2_A5** (electricity, per_channel): `e1v2_task_05`, origins [4368, 4416, 4464], task_episode_harness.e1._frozen_task_roster()[4], e1v2_task_05, delayed_origins verbatim; the Draft was graded on e1v2_task_04.  Farthest read 4512; inside the boundary.
- **T3_A3** (traffic, pooled): `W4_traffic_shift_probe_shift_480`, origins [3384], no next task exists in the frozen roster for this window, so the rule's fallback applies: the Draft's own delayed origins [2904] shifted by +480, same count and spacing.  Farthest read 3432; **the probe would read to index 3432, past the tightest frozen sealed boundary 3072 for this cohort; the window is left where the rule put it and not shortened**.
- **T3_A5** (traffic, pooled): `W4_traffic_shift_probe_shift_480`, origins [3384], no next task exists in the frozen roster for this window, so the rule's fallback applies: the Draft's own delayed origins [2904] shifted by +480, same count and spacing.  Farthest read 3432; **the probe would read to index 3432, past the tightest frozen sealed boundary 3072 for this cohort; the window is left where the rule put it and not shortened**.

### Where it stopped

- **T3_A3** -- probe window layer: the probe would read to index 3432, past the tightest frozen sealed boundary 3072 for this cohort; the window is left where the rule put it and not shortened
- **T3_A5** -- probe window layer: the probe would read to index 3432, past the tightest frozen sealed boundary 3072 for this cohort; the window is left where the rule put it and not shortened

## Cost

5 evaluation calls of a budget of 8, 15 Consumer retrains (one retrain per origin per evaluation call, the convention the bridge and integration slices used), 0 LLM calls.

