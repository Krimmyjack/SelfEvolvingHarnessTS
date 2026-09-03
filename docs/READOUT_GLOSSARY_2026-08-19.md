# Readout glossary (2026-08-19)

Rule: after this file, any report or conversation that cites these quantities
must name the口径. Mixing them is how 0.626 and 0.759 were treated as the
same number.

Sources recomputed here, not rewritten:

- D0 = `artifacts/functional/e2/g4_source_experience_electricity.json`
- D1 = `artifacts/functional/e2/g3d1_electricity_skill_only_ab.json`

`MATERIAL_THRESHOLD = 0.005`. `B = 3`. Support gain is macro sMASE gain over
identity on the frozen Support origins (higher is better).

## Named口径

### `real_support_probe_count`

Count of Support evaluations that actually ran (`status == PROBED`). This is
the only quantity that may be called a probe count.

### `charged_probe_cost`

Budget-penalty bookkeeping, **never** a probe count. Per arm-task:

- if a `LOCAL_ACTIVE` Skill formed on that task: `charged = real_support_probe_count`
- otherwise: `charged = B + 1 = 4`

A nine-task arm with zero `LOCAL_ACTIVE` therefore has `charged_probe_cost = 36`
even if it only probed 9 or 11 times.

### material-harm

Negative side of the current Draft gate. A probe is material-harm iff
`support_gain < -0.005`. Cumulative material-harm is
`sum(-support_gain)` over those probes. This is what
`metrics.harmful_probe_count` and `metrics.cumulative_support_harm` store
in the agentic runner.

### all-negative

Every probe with `support_gain < 0`, including immaterial negatives in
`(-0.005, 0)`. Cumulative all-negative harm is `sum(-support_gain)` over
those probes.

### `distinct_task_count`

Evidence unit for Skill / guidance clauses. One Task Episode id counts once
inside a census cell.

### `attempt_count`

Diagnostic only. A3 and A5 probing the same Task Episode over the same frozen
Outcome cell are two attempts, one task.

### Paired-comparable vs all-probes

The G1/G3 paired readout drops a Task when **either** arm is a mechanical
exit (`AGENT_PROTOCOL_ERROR`, instrument-unreadable, infrastructure failure).
That pair is excluded from means, first-positive index-after-drop, and
paired harm. It is **not** the same as summing every probe in the JSON.

---

## Worked examples

### D0 electricity, A3 (cold, no Source inlet)

| quantity | value | how |
| --- | ---: | --- |
| `real_support_probe_count` | 8 | `cost_by_arm.A3` |
| `charged_probe_cost` | 27 | 3 ACTIVE tasks charged 1 + 6 non-ACTIVE charged 4 |
| `LOCAL_ACTIVE` | 3 | tasks 02, 06, 08 |
| abstention | 3 | |
| first material-positive task index | 2 | e1v2_task_02 |
| material-harm count | 3 | `g < -0.005` |
| material-harm sum | 0.083247 | matches the design-doc 0.083 |
| all-negative count | 3 | no immaterial negatives |
| all-negative sum | 0.083247 | same as material-harm on this arm |
| task_05 | protocol error, 0 probes | A3 mechanical exit |

### D0 electricity, A5 (raw Source Episode inlet; rejected interface)

This is the 0.626 vs 0.759 event.

| quantity | value |口径 |
| --- | ---: | --- |
| `real_support_probe_count` | 11 | all probes |
| `charged_probe_cost` | 36 | 9 × 4, zero ACTIVE |
| `LOCAL_ACTIVE` | 0 | |
| abstention | 8 | |
| first material-positive task index (all 9 tasks) | 9 | e1v2_task_09 |
| first material-positive among paired-comparable tasks | 8 | task_05 dropped, remaining order 01,02,03,04,06,07,08,**09** |
| material-harm count, all probes | 9 | |
| material-harm sum, all probes | 0.754979 | |
| material-harm count, paired-comparable (drop task_05) | 8 | design-doc “harmful probes = 8” |
| material-harm sum, paired-comparable (drop task_05) | **0.626226** | design-doc “cumulative harm = 0.626” |
| all-negative count, all probes | 10 | 9 material + 1 immaterial (`-0.004055` on task_02 hampel) |
| all-negative sum, all probes | **0.759034** | the 0.759 figure |
| all-negative sum, paired-comparable | 0.630281 | not a number that was cited |

So 0.626 is **material-harm, paired-comparable** (Draft gate, drop the
protocol-error pair). 0.759 is **all-negative, all probes** (no Draft gate,
keep task_05 and the immaterial negative). They differ by two口径 choices
at once. Neither is wrong; they are not interchangeable.

Per-probe A5 gains that produce those sums:

```
task_01  -0.014879   material
task_02  -0.054777   material
task_02  -0.004055   all-negative only
task_03  -0.196955   material
task_04  -0.072520   material
task_05  -0.128753   material, dropped from paired-comparable
task_06  -0.093511   material
task_07  -0.043176   material
task_07  -0.042358   material
task_08  -0.108050   material
task_09  +0.029081   material-positive
```

`0.754979 - 0.128753 = 0.626226`. `0.754979 + 0.004055 = 0.759034`.

### D1 electricity skill-only AB (post-c166b63; both arms 0 LOCAL_ACTIVE)

No mechanical-exit pair. Paired-comparable = all 9 tasks.

| quantity | D1 A3 | D1 A5 |
| --- | ---: | ---: |
| `real_support_probe_count` | 11 | 9 |
| `charged_probe_cost` | 36 | 36 |
| `LOCAL_ACTIVE` | 0 | 0 |
| abstention | 9 | 8 |
| first material-positive task index | 6 (task_06, gain +0.0419, then REQUEST_OBSERVATION) | 5 (task_05, +0.1144 after a harmful first probe; delayed RESTRICTED) |
| material-harm count | 9 | 8 |
| material-harm sum | 0.720992 | 0.683576 |
| all-negative count | 9 | 8 |
| all-negative sum | 0.720992 | 0.683576 |

D1 A3 also has one immaterial `0.0` probe (task_03); it is neither
material-harm nor all-negative.

Pooled across both D1 arms: 17 material-harm (also 17 all-negative) out of
20 real probes. That is the R1 baseline “all-negative 17/20”. It is
**not** an A3-only rate. Same-family first-probe repeat 7/9 is a separate
readout and must be named as such.

### Charged vs real, side by side

| run | arm | real probes | charged | LOCAL_ACTIVE |
| --- | --- | ---: | ---: | ---: |
| D0 | A3 | 8 | 27 | 3 |
| D0 | A5 | 11 | 36 | 0 |
| D1 | A3 | 11 | 36 | 0 |
| D1 | A5 | 9 | 36 | 0 |

Citing “36 probes” for D1 is wrong. That is `charged_probe_cost`.

---

## Citation template

Write the口径 in the same clause as the number, for example:

- “D0 A5 cumulative harm 0.626 (**material-harm, paired-comparable**)”
- “D0 A5 cumulative harm 0.759 (**all-negative, all probes**)”
- “D1 A3 `real_support_probe_count` 11; `charged_probe_cost` 36”
- “T233 `outlier_iqr` POSITIVE `distinct_task_count` 6 (`attempt_count` 6 is diagnostic)”
