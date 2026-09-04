# T233 UNGUIDED supply under two observation contracts

The driver is a rewrite. The original v1 driver is unrecoverable, so no
parameter-identical replay is claimed. Both arms run on this one driver with the
same roster, budgets, Judge, Runtime and stage contracts, and the only declared
difference is whether the four M0b mechanism-geometry fields reach the Agent, so
the primary contrast carries no driver confound. The comparison against the
historical v1 rows is reference only and is labelled driver-confounded
throughout. KDD W3, NOAA, `g3_final_query_outcome` and every other sealed source
were not read; only already-exposed T233 was touched. No authorization action was
taken: no TRY or Skill was written, no authorization artifact was modified and
nothing was promoted. The authorization section below is a precheck and nothing
more.

## What ran

19 Tasks x 2 arms = 38 arm runs, all completed, no driver exception and no
mechanical stop.

- LLM calls: 84 `OLD_OBS` + 89 `NEW_OBS` = 173. The per-Task-per-arm guardrail
  of 20 never bound.
- Support probes: 11 `OLD_OBS` / 15 `NEW_OBS`; Tasks reaching a probe at all:
  11 / 14.
- `AGENT_PROTOCOL_ERROR`: 8 `OLD_OBS` / 5 `NEW_OBS`, almost all the same schema
  slip (`pattern_hypotheses[0].region_fractions has too few items`).
- Mask assertions: zero violations across the run. 29 `OLD_OBS` receipts were
  served with the four names deleted; 25 `NEW_OBS` `summarize_series` receipts
  carried all four.
- Independence: every Task and every arm began from the same h0
  (`c4cb24b8...`) with zero learned Skills, zero Source prior and zero retrieved
  Target-local or risk Skills, with zero violations. Every probe on both sides
  is therefore UNGUIDED by construction.
- Mask-artifact check: ungrounded-citation rejections were 0 in *both* arms. M0b
  left `harness_content_sha` unchanged and moved only the observable contract and
  the feature extractor, so the h0 instruction text names none of the four
  fields; under the mask the Agent never sees the names and cannot be broken by
  being told to cite them. The `OLD_OBS` protocol errors are ordinary schema
  slips, not mask damage.

## Primary contrast, NEW_OBS vs OLD_OBS

The load-bearing result is not the headline share but the **width of the
explored action space**:

| readout | `OLD_OBS` | `NEW_OBS` |
| --- | --- | --- |
| distinct program x context cells probed | 3 | 9 |
| `repair_level_shift` probe share | 0.909 (10/11) | 0.667 (10/15) |
| outlier-family probes | 1 | 6 |
| cells passing the precheck | 0 | 0 |
| Tasks citing an M0b field | 0 / 19 | 13 / 19 |

`OLD_OBS` only ever probed two bare programs, `repair_level_shift` and
`hampel_filter`. `NEW_OBS` probed nine cells including four composites the masked
arm never proposed at all (`outlier_mad+repair_level_shift`,
`repair_level_shift+hampel_filter`, `repair_level_shift+winsorize`,
`repair_level_shift+denoise_median`) plus `period_median_complete` and
`resample_uniform`. `NEW_OBS` still spent 10 probes on level-shift repair, but as
part of composites rather than as the bare program: bare `repair_level_shift` at
`context_condition=false` went +3/-4 under the mask and +0/-4 without it.

Citation is close to saturated where it can be measured: 14 `NEW_OBS` Tasks
reached a grounded hypothesis and 13 of those cited at least one of the four new
fields, against 0 of 19 under the mask, which is the asserted expectation.

Outlier-family exploration inside `task_13..19` occurred on **both** sides
(`OLD_OBS` on task_19; `NEW_OBS` on task_13 and task_18), so the boolean
"did outlier exploration appear" did not change; the count did.

## Authorization precheck only

No program x context cell in either arm reached 3 distinct UNGUIDED positive
Tasks with no opposing evidence, so **zero cells** are precheck-eligible on
either side. The repository's own `general_evidence_min_distinct_tasks` of 2 is
unchanged and was not used to lower the bar. Nothing here earns a TRY.

## Stability across executions, and why it matters

A harness double-spawn ran this identical protocol twice more before the
delivered run; those two executions were discarded and their state directory was
deleted, but their headline readouts are recorded here because they bound the
confidence of everything above. This section was compiled by the operator across
runs; the driver only ever sees its own.

| execution | `OLD_OBS` RLS share | `NEW_OBS` RLS share | precheck cells | outlier in 13..19 |
| --- | --- | --- | --- | --- |
| discarded A | 0.833 (10/12) | 0.583 (7/12) | 0 / 0 | none / none |
| discarded B | 0.692 (9/13) | 0.250 (3/12) | 0 / 1 | 2 / 3 Tasks |
| delivered C | 0.909 (10/11) | 0.667 (10/15) | 0 / 0 | 1 / 2 Tasks |

The direction is consistent 3 out of 3: the new observation fields always reduce
`repair_level_shift` concentration, by 0.24 to 0.44. Everything cell-level is
not stable -- one execution produced an `outlier_mad` cell at 3 distinct
positives and the other two produced none. With 19 Tasks, one probe budget and a
25-40% protocol-error rate, the effective sample per cell is a handful of
Episodes, which is too thin to carry a cell-level claim even though it is enough
to see the concentration effect.

## Two corrections to the brief's premises

- The v1 census does **not** show zero outlier exploration on `task_13..19`: v1
  probed the outlier family on task_15, task_16 and task_18. v1's
  `repair_level_shift` dominance is a share statement (14/19 = 0.737), not an
  absence of outlier probing. So "does outlier exploration appear in 13..19" was
  already yes before M0b, and is the wrong discriminator; the share and the cell
  width are the readouts that move.
- The pre-M0b public surface here is 17 feature keys, not 13. The mask removes
  exactly 4 of the 21 keys the current extractor serves, verified by direct
  gateway probe before the run.

## Standing uncertainty

- The paired contrast measures that the four fields changed exploration on this
  one already-exposed cohort under one probe budget. It does not show that any
  resulting cell is correct, useful downstream, or transferable anywhere.
- Wider exploration is not by itself an improvement. `NEW_OBS` spent 4 more
  probes and its bare-`repair_level_shift` cell lost the 3 positives the masked
  arm found, so part of the effect is redistribution of a fixed budget into
  less-tested programs.
- The v1 comparison stays reference-only: the driver, the h0 runtime bundle and
  the model era all differ.
