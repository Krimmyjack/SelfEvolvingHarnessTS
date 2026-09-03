# batch recipe across development windows v1

Two extra development windows per cell for the frozen v2 batch recipe, and the **first out-of-selection reading** of the plans the v2 rule adopted on window 1.

**Engineering effect measurement, not authorization evidence.** No Skill is written, no Episode is formed, no Fast or Slow path is entered, and no execution right is granted or implied. 0 LLM calls, deterministic.

Window 1 is `artifacts/functional/e2/batch_recipe_v2_all_cells_v1.json` (`batch_recipe_v2_all_cells_v1`), **read verbatim and not re-run**. This run writes one new stem and overwrites nothing.

Headline: window 1's plan holds its delayed gain at or above zero on **6 of 6 cells** across both new windows; the adopted program is stable across all three windows on 3 cells and the exclusion mask on 1.

## 0. Why this is the first honest reading

The v2 rule adopts a masked plan only if its delayed aggregate gain clears `max(best full-batch delayed, 0)`. That puts the delayed window **inside** the selection, which the recipe artifacts have said all along. Until now no adopted plan had been read on a window that took no part in adopting it. Section 2 is that reading.

## 1. Windows and where their origins come from

| cohort | window | Task | support origins | delayed origins | farthest read | origin source |
| --- | --- | --- | --- | --- | ---: | --- |
| electricity | W1 | `e1v2_task_01` | [3072, 3120, 3168] | [3216, 3264, 3312] | 3360 | quoted from the frozen roster |
| electricity | W2 | `e1v2_task_02` | [3360, 3408, 3456] | [3504, 3552, 3600] | 3648 | quoted from the frozen roster |
| electricity | W3 | `e1v2_task_03` | [3648, 3696, 3744] | [3792, 3840, 3888] | 3936 | quoted from the frozen roster |
| T233 | W1 | `e1v2_task_01` | [3072, 3120, 3168] | [3216, 3264, 3312] | 3360 | quoted from the frozen roster |
| T233 | W2 | `e1v2_task_02` | [3360, 3408, 3456] | [3504, 3552, 3600] | 3648 | quoted from the frozen roster |
| T233 | W3 | `e1v2_task_03` | [3648, 3696, 3744] | [3792, 3840, 3888] | 3936 | quoted from the frozen roster |
| traffic | W1 | `e1v2_task_01` | [1104, 1368] | [1800] | 1848 | quoted from the screening record |
| traffic | W2 | `e1v2_task_01` | [1584, 1848] | [2280] | 2328 | chosen inside the declared pre-sealed region |
| traffic | W3 | `e1v2_task_01` | [2064, 2328] | [2760] | 2808 | chosen inside the declared pre-sealed region |

Provenance, verbatim:

- `electricity` W1: frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), Task e1v2_task_01, support and delayed origins verbatim; this Task is inside the roster the g1 agentic pipeline and the M0a census already ran on this cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)
- `electricity` W2: frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), Task e1v2_task_02, support and delayed origins verbatim; this Task is inside the roster the g1 agentic pipeline and the M0a census already ran on this cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)
- `electricity` W3: frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), Task e1v2_task_03, support and delayed origins verbatim; this Task is inside the roster the g1 agentic pipeline and the M0a census already ran on this cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)
- `T233` W1: frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), Task e1v2_task_01, support and delayed origins verbatim; this Task is inside the roster the g1 agentic pipeline and the M0a census already ran on this cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)
- `T233` W2: frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), Task e1v2_task_02, support and delayed origins verbatim; this Task is inside the roster the g1 agentic pipeline and the M0a census already ran on this cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)
- `T233` W3: frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), Task e1v2_task_03, support and delayed origins verbatim; this Task is inside the roster the g1 agentic pipeline and the M0a census already ran on this cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)
- `traffic` W1: artifacts/functional/e2/g3_candidate_screening_v2.json, criteria.development_origins = [1104, 1368, 1800]; the same triple the batch recipe and the M0a traffic census already ran on
- `traffic` W2: chosen, not quoted: window 1's shape and spacing shifted by +480 inside the same pre-sealed development region declared by g3_candidate_screening_v2.json (sealed_from_index = 3072)
- `traffic` W3: chosen, not quoted: window 1's shape and spacing shifted by +960 inside the same pre-sealed development region declared by g3_candidate_screening_v2.json (sealed_from_index = 3072)

> traffic has exactly one triple of development origins on record (1104/1368/1800). Windows 2 and 3 keep its shape and spacing and are shifted forward inside the same pre-sealed region; they are chosen rather than quoted, they are the only chosen origins in this run, and the farthest index any of them reads is 2808 against sealed_from_index=3072.

> run_batch_composition_headroom is imported and not modified; the traffic window is selected by rebinding its module-level _TRAFFIC_DEVELOPMENT_ORIGINS for the duration of one call and restoring it afterwards, with _TRAFFIC_SEALED_FROM_INDEX left untouched so the module's own boundary guard stays live

## 2. Window 1's adopted plan, read out of selection

Pre-stated: per cell, window 1's adopted plan passes iff its delayed aggregate gain is >= 0.0 on both new windows; the v2 rule's own bar is max(best full-batch delayed, 0), so identity at zero is the honest bar. A second column reports the stricter > MATERIAL_THRESHOLD=0.005 reading for information only.

| cell | W1 plan | W1 delayed (in selection) | W2 support | W2 delayed | W3 support | W3 delayed | delayed >= 0 on both | verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| electricity x pooled | `outlier_iqr` minus 1, 2, 6 | +0.016343 | +0.014368 | +0.026595 | +0.018998 | +0.019688 | True | `W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION` |
| electricity x per_channel | `denoise_median` minus 10, 3, 4 | +0.138523 | +0.085848 | +0.138044 | +0.120348 | +0.108850 | True | `W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION` |
| T233 x pooled | `winsorize` minus none | +0.116627 | +0.250575 | +0.270423 | +0.267724 | +0.161501 | True | `W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION` |
| T233 x per_channel | `winsorize` minus T233, T234, T241, T247, T256 | +0.030410 | +0.001731 | +0.060038 | -0.001033 | +0.012082 | True | `W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION` |
| traffic x pooled | `outlier_iqr` minus 6 | +1.047075 | +1.317590 | +0.598790 | +0.841134 | +0.812342 | True | `W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION` |
| traffic x per_channel | `outlier_iqr` minus 3, 8 | +0.355472 | +0.283379 | +0.437642 | +0.513358 | +0.387062 | True | `W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION` |

The stricter `delayed > 0.005` reading, for information only: electricity x pooled W2=True W3=True; electricity x per_channel W2=True W3=True; T233 x pooled W2=True W3=True; T233 x per_channel W2=True W3=True; traffic x pooled W2=True W3=True; traffic x per_channel W2=True W3=True.

## 3. What each window adopted

| cell | window | kind | program | mask | support | delayed | harmed eval | adoption path |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| electricity x pooled | W1 | MASKED_PLAN | `outlier_iqr` | 1, 2, 6 | +0.034643 | +0.016343 | 1 | masked plan cleared the delayed stability check (bar=max(b |
| electricity x pooled | W2 | MASKED_PLAN | `winsorize` | 1, 10, 11, 4, 5, 7, 8 | +0.068383 | +0.063502 | 0 | masked plan cleared the delayed stability check (bar=max(b |
| electricity x pooled | W3 | BEST_FULL_BATCH | `winsorize` | none | +0.057660 | +0.016103 | 2 | no masked plan cleared the delayed stability check |
| electricity x per_channel | W1 | MASKED_PLAN | `denoise_median` | 10, 3, 4 | +0.129359 | +0.138523 | 1 | masked plan cleared the delayed stability check (bar=max(b |
| electricity x per_channel | W2 | BEST_FULL_BATCH | `denoise_median` | none | +0.091124 | +0.134171 | 1 | no masked plan cleared the delayed stability check |
| electricity x per_channel | W3 | BEST_FULL_BATCH | `denoise_median` | none | +0.119808 | +0.100672 | 0 | no masked plan cleared the delayed stability check |
| T233 x pooled | W1 | BEST_FULL_BATCH | `winsorize` | none | +0.072156 | +0.116627 | 2 | no masked plan cleared the delayed stability check |
| T233 x pooled | W2 | BEST_FULL_BATCH | `winsorize` | none | +0.250575 | +0.270423 | 0 | no masked plan cleared the delayed stability check |
| T233 x pooled | W3 | BEST_FULL_BATCH | `outlier_iqr` | none | +0.316813 | +0.225790 | 1 | no masked plan cleared the delayed stability check |
| T233 x per_channel | W1 | MASKED_PLAN | `winsorize` | T233, T234, T241, T247, T256 | +0.019022 | +0.030410 | 2 | masked plan cleared the delayed stability check (bar=max(b |
| T233 x per_channel | W2 | MASKED_PLAN | `outlier_iqr` | T235, T244, T256 | +0.007416 | +0.051322 | 1 | masked plan cleared the delayed stability check (bar=max(b |
| T233 x per_channel | W3 | IDENTITY | `identity` | none | +0.000000 | +0.000000 | 0 | no masked plan cleared the delayed stability check and the |
| traffic x pooled | W1 | MASKED_PLAN | `outlier_iqr` | 6 | +0.665277 | +1.047075 | 0 | masked plan cleared the delayed stability check (bar=max(b |
| traffic x pooled | W2 | BEST_FULL_BATCH | `outlier_iqr` | none | +1.300699 | +0.547008 | 0 | no masked plan cleared the delayed stability check |
| traffic x pooled | W3 | MASKED_PLAN | `outlier_iqr` | 3, 5, 7 | +0.949579 | +1.037101 | 0 | masked plan cleared the delayed stability check (bar=max(b |
| traffic x per_channel | W1 | MASKED_PLAN | `outlier_iqr` | 3, 8 | +0.303184 | +0.355472 | 0 | masked plan cleared the delayed stability check (bar=max(b |
| traffic x per_channel | W2 | MASKED_PLAN | `outlier_iqr` | 8 | +0.289403 | +0.439241 | 0 | masked plan cleared the delayed stability check (bar=max(b |
| traffic x per_channel | W3 | MASKED_PLAN | `outlier_iqr` | 3 | +0.514752 | +0.386972 | 0 | masked plan cleared the delayed stability check (bar=max(b |

## 4. Stability across windows

| cell | programs W1/W2/W3 | masks W1/W2/W3 | program stable | mask stable | kind stable |
| --- | --- | --- | --- | --- | --- |
| electricity x pooled | `outlier_iqr` / `winsorize` / `winsorize` | [1, 2, 6] / [1, 10, 11, 4, 5, 7, 8] / [none] | False | False | False |
| electricity x per_channel | `denoise_median` / `denoise_median` / `denoise_median` | [10, 3, 4] / [none] / [none] | True | False | False |
| T233 x pooled | `winsorize` / `winsorize` / `outlier_iqr` | [none] / [none] / [none] | False | True | True |
| T233 x per_channel | `winsorize` / `outlier_iqr` / `identity` | [T233, T234, T241, T247, T256] / [T235, T244, T256] / [none] | False | False | False |
| traffic x pooled | `outlier_iqr` / `outlier_iqr` / `outlier_iqr` | [6] / [none] / [3, 5, 7] | True | False | False |
| traffic x per_channel | `outlier_iqr` / `outlier_iqr` / `outlier_iqr` | [3, 8] / [8] / [3] | True | False | True |

## 5. What this does not say

- It does not authorize anything and it does not promote the v2 rule. It is one more reading of the same engineering measurement.
- Three windows on one Task family per cohort is a small sample. A per-cell pass is two numbers, not a rate.
- The new windows are development windows: quoted roster origins for electricity and T233, and for traffic origins inside an already-declared pre-sealed development region. Nothing sealed was opened and no claim here is a fresh-window claim.
- Window 1's plan being non-negative out of selection is not the same as it being the best plan for the new window. Sections 3 and 4 show where the recipe itself would have chosen differently.

## Provenance

- recipe: `run_batch_composition_headroom.make_batch_recipe`, adoption_rule_version `v2`, imported and not modified
- scoring: the same `_evaluate_assignment` + `_gain_rows`, identity baseline recomputed once per (cell, window)
- new recipe runs in this artifact: 12
- LLM calls: 0
- wall seconds: 222.0

