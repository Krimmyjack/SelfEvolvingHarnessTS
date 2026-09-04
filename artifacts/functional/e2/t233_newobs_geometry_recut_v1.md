# T233 NEW_OBS supply, geometry recut by `mask_class`

Zero-LLM reanalysis. No Episode was run, no Support probe was spent, no code
was changed, no Outcome was opened, and no sealed source was read. The only
writes are this file and its companion
`t233_newobs_geometry_recut_v1.json`.

## Pre-registration, restated before the numbers

The one pre-declared context variable is `mask_class`, four-way
(`OUTLIER_ONLY` / `LEVEL_ONLY` / `MIXED` / `AMBIGUOUS`), taken from the frozen
`artifacts/functional/e2/m0a_mask_geometry_census_v1.json`
(`protocol_version: m0a_mask_geometry_census_v1`, file mtime
`2026-08-20T01:39:39`, which precedes exec1's report at `12:04:15` and both
supplementary executions, so it is not fitted to this batch).

No other feature was tried, no class boundary was adjusted, and no combination
search was run. `pss` appears once, at the end, as a secondary descriptive
split only, and takes no part in any authorization reading.

The evidence is the same three clean NEW_OBS executions pooled in
`t233_newobs_supply_pooled_v1.json`, under that report's own unchanged
distinct-Task and conflict rule: the unit is the distinct Task; a Task positive
in one execution and negative in another for the same cell is a **conflict** and
counts in neither column; **opposing = distinct negatives + conflicts**, and
either blocks a precheck. Threshold `material_threshold = 0.005`
(`pooled_census.counting_rule.material_threshold`), precheck minimum 3
(`pooled_census.counting_rule.precheck_min_distinct_positive_tasks`).

## The recut is degenerate, and the reason is structural

Every one of the 19 T233 Tasks carries representative `mask_class = MIXED`.
The class is **constant on the probed data**, so the recut has exactly one
non-empty column and cannot separate anything — this is not a low-power result,
it is a degenerate partition.

The reason is worth stating precisely, because it is a property of T233's
construction rather than of this sample size. The only `LEVEL_ONLY` series in
the whole T233 census is series uid **`T233` itself**, the target series, which
appears as `LEVEL_ONLY` in 17 of the 19 Task rows
(`rows[24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204,
216]`, all with `outlier_region_fraction = 0.0000`). All 11 other series
(`T234, T235, T236, T239, T240, T241, T244, T246, T254, T256`) are `MIXED` in
every Task. And series `T233` is **never in scope** for any Task: the probed
scopes are the train series only, verified identical across all three
executions (`rows[].scope_series_uids`, 8 to 10 series per Task, `T233` absent
from all 19).

So no additional T233 Task at this cohort and cutoff rule could introduce
variation in the pre-declared variable. The M0 union-collapse hypothesis is not
refuted here; it is **untestable on T233's UNGUIDED supply data**.

## `mask_class` share

Representative level, the basis for the authorization reading, from
`task_representatives["T233|e1v2_task_NN"]` joined to its census row
(`rows[1, 13, 25, 37, 49, 61, 73, 85, 97, 119, 131, 143, 155, 167, 179, 191,
203, 215, 227]`; representatives are `T234` for Tasks 01-09 and `T256` for
Tasks 10-19):

| mask_class | Tasks | share of 19 |
| --- | --- | --- |
| `MIXED` | 19 | 100.0% |
| `LEVEL_ONLY` | 0 | 0.0% |
| `OUTLIER_ONLY` | 0 | 0.0% |
| `AMBIGUOUS` | 0 | 0.0% |

Decision-point level, all 228 T233 census rows
(`per_cohort.T233.mask_class_counts`, `per_cohort.T233.mask_class_fractions`):

| mask_class | decision points | share of 228 |
| --- | --- | --- |
| `MIXED` | 211 | 92.5% |
| `LEVEL_ONLY` | 17 | 7.5% |
| `OUTLIER_ONLY` | 0 | 0.0% |
| `AMBIGUOUS` | 0 | 0.0% |

Restricted to in-scope series, which is what the probes actually acted on: 175
in-scope decision points (`per_cohort.T233.representative_coverage
.in_scope_decision_points`), **100% `MIXED`**, because all 17 `LEVEL_ONLY` rows
belong to the out-of-scope target series.

Reported without dressing up: T233 is a single-class cohort for this variable,
so the recut was always going to be low-powered, and is in fact fully
degenerate.

## Recut table: program x `mask_class`

38 real probes across 57 Task-executions
(`per_execution_task_rows[].probed_cells`). Families are read from the probe
receipts (`rows[].arms.NEW_OBS.probes[].families`), not inferred from names.

| program | families | mask_class | distinct + | distinct - | conflict | immaterial | opposing | meets >=3 & 0 opposing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hampel_filter` | outlier | `MIXED` | 3 | 5 | 2 | 0 | 7 | no |
| `outlier_mad+repair_level_shift` | outlier, structural | `MIXED` | 1 | 0 | 0 | 0 | 0 | no |
| `period_median_complete` | impute | `MIXED` | 0 | 0 | 0 | 1 | 0 | no |
| `repair_level_shift` | structural | `MIXED` | 6 | 7 | 0 | 1 | 7 | no |
| `repair_level_shift+denoise_median` | denoise, structural | `MIXED` | 0 | 1 | 0 | 0 | 1 | no |
| `repair_level_shift+hampel_filter` | outlier, structural | `MIXED` | 1 | 0 | 0 | 0 | 0 | no |
| `repair_level_shift+winsorize` | outlier, structural | `MIXED` | 1 | 0 | 0 | 0 | 0 | no |
| `resample_uniform` | align | `MIXED` | 0 | 0 | 0 | 1 | 0 | no |

Merge audit against the previous `pss` key, so the change in the `repair_level_shift`
count is traceable and not a recount: the old key split that program into
`pss=False` at 5+/7- (`pooled_census.cells`) and `pss=True` at 1+/0-. Dropping
`pss` for `mask_class` unions those two disjoint Task groups into 6+/7-. The
same union takes `hampel_filter` from 2+/3-/2conflict and 1+/2-/0conflict to
3+/5-/2conflict. No Task changed sign; only the partition changed.

## The three pre-registered questions

**1. Does the `repair_level_shift` 5+/7- split by geometry?**

No, and it cannot. All 19 Tasks are `MIXED`, so the positives and the negatives
sit in the same single cell: `repair_level_shift x MIXED` is 6 distinct positive
and 7 distinct negative, opposing 7, still blocked. There is no
`LEVEL_ONLY`-dominated or outlier-dominated column for them to separate into.
The pre-declared recut therefore returns no information about whether the
union-collapse mechanism explains the mixed sign — it is silent, not negative.

- positives: `e1v2_task_04`, `e1v2_task_06`, `e1v2_task_09`, `e1v2_task_10`, `e1v2_task_11`, `e1v2_task_19`
- negatives: `e1v2_task_01`, `e1v2_task_05`, `e1v2_task_08`, `e1v2_task_14`, `e1v2_task_15`, `e1v2_task_16`, `e1v2_task_17`
- conflicts: none

**2. Does any recut cell reach >=3 distinct UNGUIDED positive Tasks with zero opposing?**

No. Zero cells qualify. Two cells clear the count and fail on opposing
(`repair_level_shift` at 6 positive against 7 opposing; `hampel_filter` at 3
positive against 7 opposing). The three cells with zero opposing all sit at 1
positive Task, two short. The recut moved no cell across the line in either
direction.

**3. How is the outlier family distributed after the recut?**

The four programs whose receipts carry the `outlier` family are
`hampel_filter`, `outlier_mad+repair_level_shift`,
`repair_level_shift+hampel_filter` and `repair_level_shift+winsorize`. As a
union over distinct Tasks — not a sum of cells, since one Task can appear in
several — the family is **5 distinct positive, 5 distinct negative, 2
conflict**, opposing 7, so it does not qualify:

- positives: `e1v2_task_03`, `e1v2_task_06`, `e1v2_task_08`, `e1v2_task_17`, `e1v2_task_18`
- negatives: `e1v2_task_01`, `e1v2_task_04`, `e1v2_task_07`, `e1v2_task_09`, `e1v2_task_14`
- conflicts: `e1v2_task_12`, `e1v2_task_13`

The mixed sign is carried almost entirely by `hampel_filter` alone (3+/5-/2
conflict); the three combination programs are 1+/0- each on single Tasks. Both
conflicts in the whole recut belong to this family.

## Sensitivity version

The brief asks for a second version if the specific series a probe acted on can
be recovered and differs from the representative. Two findings:

- The Support probe has no per-series attribution to recover. It is scored as
  one macro gain over the whole Scope (`support_gain`, `se_block`,
  `modified_point_count` per probe), so the probe acts on the Scope, not on a
  nameable series.
- The recoverable substitute is the `mask_class` composition of each Task's
  in-scope series set. Computed from `rows[].scope_series_uids` joined to the
  census rows, **every in-scope series in every one of the 19 Tasks is
  `MIXED`**, so this version assigns the identical class to every Task.

**The sensitivity version and the representative version agree exactly**, cell
for cell and Task for Task. The authorization reading rests on the
representative version, as pre-declared; the sensitivity version changes no
number in it.

## Secondary descriptive split: `pss` (takes no part in the authorization reading)

Included only because the previous census used it as the context key. Of the 14
Tasks carrying a `repair_level_shift` probe, 13 are `pss=False` and 1
(`e1v2_task_09`) is `pss=True`. That single `pss=True` Task is positive in two
executions. Within `pss=False` the sign is mixed 5 positive against 7 negative.
`pss` therefore does not separate this cell either, and this observation is
descriptive only — it authorizes nothing and is not offered as a replacement
context variable.

## Verdict

**`GEOMETRY_RECUT_NOT_SEPARATING_AT_CURRENT_SAMPLE`**

No cell reached >=3 distinct UNGUIDED positive Tasks with zero opposing, so
nothing is marked `GEOMETRY_CONDITIONED_TRY_CANDIDATE`. This stops at the
precheck: no TRY was written, no Skill was composed, no authorization artifact
was modified, nothing was promoted.

Scope of this verdict, stated narrowly on purpose:

- It falsifies **this recut on this sample** and nothing more. It is not a
  judgment on the outlier or structural Capability family, and it is not a
  refutation of the M0 union-collapse hypothesis.
- In this instance the recut could not even be a fair test: the pre-declared
  variable is constant across all probed data, for the structural reason given
  above. Reporting it as evidence against the hypothesis would be a
  misreading.
- Per the pre-registration, no further recut follows. The context variable is
  not being swapped, the class boundaries are not being adjusted, and no
  additional feature will be tried on this data.
- What a real test of the same pre-declared variable would require is a cohort
  whose **in-scope** series vary in `mask_class`. The same frozen census shows
  that variation exists elsewhere — `electricity` at
  `{MIXED: 80, OUTLIER_ONLY: 15, AMBIGUOUS: 9, LEVEL_ONLY: 4}` and `weather` at
  `{LEVEL_ONLY: 139, MIXED: 70}` (`per_cohort.*.mask_class_counts`). Naming
  that requirement is a design consequence, not a new feature to cut on, and no
  such run is authorized or implied here.
- All Tasks here remain already-exposed T233 development data. Nothing in this
  file makes any cell correct, useful downstream or transferable.
