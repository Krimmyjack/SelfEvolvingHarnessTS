# T233 NEW_OBS supply, three clean executions pooled

## What changed before these runs, and what did not

Two instrument fixes were made, both narrow:

- The per-stage repair budget is now a parameter. **Its default is unchanged at 1**, the value that was previously hard-coded, so every other caller in the repository is unaffected and no existing readout moves. Only this driver passes 2, and only on request. `exec1` ran at 1; `exec2` and `exec3` ran at 2.
- The driver's ungrounded-citation counter no longer reads 0 when the stage died. It previously counted only from `stage_validation`, which is empty whenever a stage raised, so a fatal grounding rejection was invisible to the one counter written to catch it. Recovered and fatal are now reported separately as well as summed.

No M0b working-tree change, no historical artifact and no threshold was touched, and nothing was committed.

## Merge rule

The unit of evidence is the **distinct Task**, not the execution. A Task positive in the same cell in two executions is one distinct positive Task, because re-sampling one Task on the same already-exposed data is not independent evidence. A Task that comes out positive in one execution and negative in another is reported as a **conflict** for that cell and counted in neither column. Both a distinct negative Task and a conflict Task count as **opposing**, and either blocks a precheck.

## Protocol-error funnel, retry 1 against retry 2

| execution | retries | arm runs | protocol errors | rate | `REQUEST_OBSERVATION` | probed | rescued by 2nd retry |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `exec1` | 1 | 19 | 5 | 26.3% | 11 (57.9%) | 14 | 0 |
| `exec2` | 2 | 19 | 8 | 42.1% | 11 (57.9%) | 11 | 1 |
| `exec3` | 2 | 19 | 7 | 36.8% | 8 (42.1%) | 12 | 3 |

`exec1` here is its NEW_OBS arm only, so it is comparable with the single-arm supplementary runs; the 34.2% in the forensics report was both arms pooled.

- retry 1, `exec1` NEW_OBS only: 5 / 19 = 26.3%
- retry 2, `exec2` + `exec3` pooled: 15 / 38 = 39.5%

### Verdict on the retry fix

Two things are true at once here, and collapsing them into one number would misreport the fix.

**The second retry did work as designed.** 4 arm runs across the two retry-2 executions had their inspect stage return only on the second repair attempt (`e1v2_task_08`, `e1v2_task_04`, `e1v2_task_11`, `e1v2_task_14`). A stage only records a retry count when it returned, so at a budget of 1 every one of those would have exited as `AGENT_PROTOCOL_ERROR`. Holding the observed first-pass behaviour fixed, retry 2 converted 4 would-be fatal errors into completed stages: 50.0% would have become 39.5%.

**And the net rate still rose against exec1.** It went from 26.3% at retry 1 to 39.5% at retry 2, and every retry-2 execution individually sits above the retry-1 baseline. The repair budget is not what dominates this rate. The first-pass slip rate itself moved more between executions than one extra repair attempt could offset, and the errors landed on different Tasks each time. exec2 newly failed on 4 Tasks that exec1 completed (`e1v2_task_02`, `e1v2_task_03`, `e1v2_task_07`, `e1v2_task_16`); exec3 newly failed on 7 Tasks that exec1 completed (`e1v2_task_01`, `e1v2_task_05`, `e1v2_task_06`, `e1v2_task_13`, `e1v2_task_16`, `e1v2_task_17`, `e1v2_task_18`), and not one Task failed in all three executions. That is what a variance-dominated process looks like, not a stable task-geometry effect, which also weakens the exec1 reading that tied these errors to a low `estimated_region_start_fraction`.

So the fix is worth keeping available and is not worth promoting to a default on this evidence: it buys a real but small recovery, against a noise floor large enough that three executions of 19 Tasks cannot resolve a rate difference of this size. The default of 1 stays the default.

## Pooled cells

| program | context | distinct + | distinct - | conflict | opposing | executions |
| --- | --- | --- | --- | --- | --- | --- |
| `hampel_filter` | False | 2 | 3 | 2 | 5 | exec1, exec2, exec3 |
| `hampel_filter` | True | 1 | 2 | 0 | 2 | exec2, exec3 |
| `outlier_mad+repair_level_shift` | False | 1 | 0 | 0 | 0 | exec1 |
| `period_median_complete` | False | 0 | 0 | 0 | 0 | exec1 |
| `repair_level_shift` | False | 5 | 7 | 0 | 7 | exec1, exec2, exec3 |
| `repair_level_shift` | True | 1 | 0 | 0 | 0 | exec1, exec3 |
| `repair_level_shift+denoise_median` | False | 0 | 1 | 0 | 1 | exec1 |
| `repair_level_shift+hampel_filter` | False | 1 | 0 | 0 | 0 | exec1 |
| `repair_level_shift+winsorize` | True | 1 | 0 | 0 | 0 | exec1 |
| `resample_uniform` | True | 0 | 0 | 0 | 0 | exec1 |

## Authorization precheck, precheck only

**No authorization action was taken.** No TRY and no Skill was written, no authorization artifact was modified and nothing was promoted. This section is a precheck and nothing more.

Cells reaching >= 3 distinct UNGUIDED positive Tasks with zero opposing: **0**.

Closest three cells to the threshold:

| program | context | distinct + | short by | opposing | blocked |
| --- | --- | --- | --- | --- | --- |
| `repair_level_shift` | False | 5 | 0 | 7 | yes |
| `hampel_filter` | False | 2 | 1 | 5 | yes |
| `outlier_mad+repair_level_shift` | False | 1 | 2 | 0 | no |

## `outlier_mad` recurrence

The family appears only as `outlier_mad+repair_level_shift`, and only in exec1 -- the two retry-2 executions never probed an `outlier_mad` program at all. Its pooled count is 1 distinct positive Task, so it does **not** reach the 3-Task threshold the discarded execution appeared to clear. The discarded execution's `outlier_mad` cell does not reproduce.

## Standing limits

- Every Task here is already-exposed T233 development data. Pooling three executions raises confidence about this cohort under this budget and nothing else; it does not make any cell correct, useful downstream or transferable.
- Sealed sources were not read. KDD W3, NOAA and `g3_final_query_outcome` were not opened.
