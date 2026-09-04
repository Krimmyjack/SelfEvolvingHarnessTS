# Per-series geometry response -- engineering diagnosis

*Artifact* `artifacts/functional/e2/per_series_geometry_response_v1.md`  
*Companion* `artifacts/functional/e2/per_series_geometry_response_v1.json`  
*LLM calls* **0**.  No Episode was run, no code was changed, no sealed data was read.  Every number below comes from frozen artifacts that were already paid for, plus the frozen geometry census.

## What this is, and what it is explicitly not

The `mask_class` recut of the T233 supply census returned `GEOMETRY_RECUT_NOT_SEPARATING_AT_CURRENT_SAMPLE` because the pre-declared geometry variable turned out to be constant across the Tasks that were actually probed.  Moving the same check to `electricity` did not help either: the roster has real class variation, but Step 0 of that attempt found the Agent has no protocol mechanism to narrow a probe's scope, so the unit of analysis collapsed back onto whole-Task scope.  This diagnosis takes the one step left that costs nothing: drop to the **series** the probes actually act on and ask whether utility responds to geometry at that level.

**Status of these readings.**  Series-level readings are an *engineering diagnosis only*.  They do not enter any authorization ledger.  No TRY was written, no Skill was compiled, no authorization artifact was touched, and the `electricity` numbers below come from Skill-accumulating arms which are not UNGUIDED and therefore could never be authorization evidence in the first place.

## Pre-registration, restated before the numbers

* **Scope-class rule.**  If every series in a probe's scope carries the same `mask_class` in the frozen census, the probe's scope takes that class; if they differ, the scope is `MIXED_SCOPE`.  Fixed in advance, not adjusted afterwards.
* **Degeneracy exit.**  If the scope class never varies, the result is `SCOPE_CLASS_DEGENERATE` -- a *dumb* result, not a negative one.  It falsifies the measurement, not the mechanism hypothesis.
* **The three questions.**  (1) Does level-repair utility split by scope geometry?  (2) Does the outlier family concentrate positive on outlier geometry?  (3) Is there any `program x scope-class` cell with >= 3 distinct positive Tasks and zero opposing?
* Anything computed after seeing that the pre-registered variable was constant is labelled **diagnosis-only** and is never presented as a substitute pre-registered variable.

## Step 0: can per-series utility be geometry-labelled at all?

No, and this is structural rather than a gap in logging.

| cohort | series carrying a utility reading (`per_series_mean_gain` keys) | series carrying a geometry label (census rows) | intersection |
| --- | --- | --- | --- |
| T233 | `T257`, `T259`, `T260`, `T261`, `T262`, `T264`, `T265`, `T266` | `T233`, `T234`, `T235`, `T236`, `T239`, `T240`, `T241`, `T244`, `T246`, `T247`, `T254`, `T256` | **empty** |
| electricity | `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19` | `0`, `1`, `10`, `11`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9` | **empty** |

Utility is only ever measured downstream, on the **eval** series (`per_series_mean_gain`, 34 occurrences in T233 artifacts and 28 in electricity artifacts).  Geometry is only ever labelled upstream, on the **train** series -- the census row unit is literally `cohort x task_episode_id x train series uid at observation_cutoff = support_origins[0]`.  The two uid sets are disjoint by construction, so a true per-series response x geometry join does not exist in the paid traces.

The one per-series block that *is* keyed on train series is a substrate preflight, not a utility reading: `train_substrate_preflight.per_series` contains `{"floor_hit_anchor_count": 0, "clean": true}`.

**What is joinable.**  Each Task row records `scope_series_uids`, the train series the probe acts on, and it matches the census `in_scope` flag exactly -- 145 of 145 Task rows across every artifact checked, zero mismatches.  So the geometry of the acted-on series is recoverable per Task even though the utility cannot be decomposed onto individual series.

## The pre-registered variable is constant on probed series -- in every cohort

| cohort | in-scope rows | `mask_class` of in-scope rows | `mask_class` of out-of-scope rows | non-MIXED rows | of those, in scope |
| --- | --- | --- | --- | --- | --- |
| T233 | 175 | MIXED 175 | LEVEL_ONLY 17, MIXED 36 | 17 | **0** |
| electricity | 52 | MIXED 52 | AMBIGUOUS 9, LEVEL_ONLY 4, MIXED 28, OUTLIER_ONLY 15 | 28 | **0** |
| weather | 31 | MIXED 31 | LEVEL_ONLY 139, MIXED 39 | 139 | **0** |

Every `(Task, series)` row that is in scope is `MIXED`, in all three cohorts, with no exception.  All 184 non-MIXED rows in the census -- every `OUTLIER_ONLY`, every `LEVEL_ONLY`, every `AMBIGUOUS` -- sit outside every probe scope.  The in-scope rows are also uniformly `scope_bin = high`.

This is the mechanism behind the earlier silent recut, and it generalises: the scope selector admits a series only when it has enough support to be worth repairing, and a series with enough of *both* an outlier signature and a level signature to clear that bar is, by the census definition, `MIXED`.  A pure-geometry series is exactly the kind the selector never admits.  `electricity` was chosen because its roster shows real class variation (`MIXED 80 / OUTLIER_ONLY 15 / AMBIGUOUS 9 / LEVEL_ONLY 4` rows), and that variation is entirely out of scope.

Scope-class distribution over the probes actually analysed:

| evidence set | probes | scope classes seen |
| --- | --- | --- |
| `t233_newobs_unguided` | 38 | MIXED 38 |
| `t233_oldobs_unguided_descriptive` | 11 | MIXED 11 |
| `electricity_sequential_descriptive` | 140 | MIXED 140 |

One class, 189 probes, zero contrast.  The pre-declared degeneracy exit fires.

## Evidence base

| evidence set | cohort | UNGUIDED | artifacts | arm-runs | probes | distinct Tasks | protocol errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `t233_newobs_unguided` | T233 | yes | 3 | 57 | 38 | 19 | 20 |
| `t233_oldobs_unguided_descriptive` | T233 | yes | 1 | 19 | 11 | 11 | 8 |
| `electricity_sequential_descriptive` | electricity | no | 9 | 138 | 140 | 9 | 2 |

Replay duplication is real and is accounted for: of 73 distinct probe fingerprints in the electricity set, 29 appear in more than one artifact (the same `(arm, Task, episode_id, program, support_gain)` replayed).  The distinct-Task rule already collapses these, so they inflate the probe count but not the cell counts.

Counting rules are inherited unchanged from the pooled census: material threshold `0.005`, one distinct Task per cell however many executions confirm it, a sign flip across executions is a **conflict** counted in neither column, and both a distinct negative and a conflict count as opposing.

## Pre-registered recut table (`program x scope-class`)

### T233, three clean NEW_OBS executions (UNGUIDED)

| program | scope-class | distinct + | distinct - | conflict | immaterial | opposing | precheck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `hampel_filter` | MIXED | 3 | 5 | 2 | 0 | 7 | no |
| `outlier_mad+repair_level_shift` | MIXED | 1 | 0 | 0 | 0 | 0 | no |
| `period_median_complete` | MIXED | 0 | 0 | 0 | 1 | 0 | no |
| `repair_level_shift` | MIXED | 6 | 7 | 0 | 1 | 7 | no |
| `repair_level_shift+denoise_median` | MIXED | 0 | 1 | 0 | 0 | 1 | no |
| `repair_level_shift+hampel_filter` | MIXED | 1 | 0 | 0 | 0 | 0 | no |
| `repair_level_shift+winsorize` | MIXED | 1 | 0 | 0 | 0 | 0 | no |
| `resample_uniform` | MIXED | 0 | 0 | 0 | 1 | 0 | no |

### electricity, task-sequential arms (NOT UNGUIDED, descriptive)

| program | scope-class | distinct + | distinct - | conflict | immaterial | opposing | precheck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `denoise_median` | MIXED | 0 | 0 | 0 | 1 | 0 | no |
| `fft_decompose` | MIXED | 0 | 1 | 0 | 0 | 1 | no |
| `hampel_filter` | MIXED | 3 | 2 | 3 | 0 | 5 | no |
| `impute_ar` | MIXED | 0 | 0 | 0 | 1 | 0 | no |
| `impute_linear` | MIXED | 0 | 0 | 0 | 3 | 0 | no |
| `outlier_iqr` | MIXED | 4 | 1 | 0 | 0 | 1 | no |
| `outlier_mad` | MIXED | 6 | 2 | 0 | 0 | 2 | no |
| `period_complete` | MIXED | 0 | 0 | 0 | 2 | 0 | no |
| `period_median_complete` | MIXED | 0 | 0 | 0 | 2 | 0 | no |
| `repair_level_shift` | MIXED | 1 | 8 | 0 | 0 | 8 | no |
| `repair_level_shift+impute_ar` | MIXED | 0 | 1 | 0 | 0 | 1 | no |
| `repair_level_shift+impute_linear` | MIXED | 0 | 1 | 0 | 0 | 1 | no |
| `smooth_ema` | MIXED | 0 | 1 | 0 | 0 | 1 | no |
| `smooth_ma+repair_level_shift` | MIXED | 0 | 1 | 0 | 0 | 1 | no |

Every scope-class column reads `MIXED`, so this table is the old program-only census with a constant column bolted on.  It cannot separate anything, which is the point.

## Diagnosis-only continuous readout

Since the four-way class is constant, the only geometry contrast left inside the probed data is *within* `MIXED`: how much of each scope is outlier-shaped versus level-shaped.  This is **post-hoc and diagnosis-only**.

It is also very weak, and the report says so up front.  Outlier dominance (the share of scope series whose outlier region exceeds their level region) takes only 3 distinct values in T233 (0.889, 0.900, 1.000) and 4 in electricity (0.800, 0.833, 0.857, 1.000), and every one of them is above 0.79.  Every scope in every cohort is predominantly outlier-shaped; there is no level-dominated scope anywhere to contrast against.

**The correlations are confounded with Task position.**  Scope size grows and the region fractions drift monotonically as the observation cutoff advances, so geometry is largely a proxy for where the Task sits in the roster:

| cohort | outlier dominance vs Task index | level region fraction vs Task index | level excursion vs Task index | scope size vs Task index |
| --- | --- | --- | --- | --- |
| T233 | -0.6728 | -0.9629 | 0.8424 | 0.7273 |
| electricity | -0.5512 | -0.9898 | 0.8872 | 0.9494 |

With the level region fraction tracking the Task index at `r = -0.9898` in electricity and `r = -0.9629` in T233, scope geometry is very nearly a deterministic function of Task position, so no gain-vs-geometry correlation in either cohort can be separated from a gain-vs-Task-position effect at this sample size.  The numbers below are reported for completeness, not as evidence of a geometry response.

| evidence set | group | n | gain vs outlier dominance | gain vs outlier region fraction | gain vs level region fraction | gain vs level excursion |
| --- | --- | --- | --- | --- | --- | --- |
| `t233_newobs_unguided` | all | 38 | 0.1531 | 0.2716 | 0.2153 | -0.0845 |
| `t233_newobs_unguided` | level_repair_family | 19 | 0.0621 | 0.4497 | 0.2974 | -0.0066 |
| `t233_newobs_unguided` | outlier_family | 14 | 0.0595 | 0.0964 | -0.0573 | 0.1078 |
| `t233_oldobs_unguided_descriptive` | all | 11 | 0.0246 | 0.1187 | -0.2442 | 0.3116 |
| `t233_oldobs_unguided_descriptive` | level_repair_family | 10 | 0.034 | 0.1308 | -0.2593 | 0.3383 |
| `electricity_sequential_descriptive` | all | 140 | 0.149 | 0.0667 | -0.0404 | 0.1259 |
| `electricity_sequential_descriptive` | level_repair_family | 69 | 0.6746 | 0.5191 | 0.2176 | 0.139 |
| `electricity_sequential_descriptive` | outlier_family | 60 | -0.0074 | -0.0236 | 0.0538 | -0.0167 |

## The three pre-registered questions

### 1. Does level-repair utility split by scope geometry?

**No, and the question is not answerable as posed.**  In the three clean UNGUIDED T233 executions `repair_level_shift` stands at 6 distinct positive and 7 distinct negative Tasks, and the entire split sits in one geometry cell (`MIXED`) because there is no other cell.  Forcing the diagnosis-only dominance split does not separate it either: 5+/6- below the dominance median against 1+/1- above it -- the mixed record is reproduced inside both halves rather than resolving into a level-positive and an outlier-negative side.

Where a correlation is large it points the wrong way for the mechanism hypothesis.  In electricity, `repair_level_shift` gain rises with outlier dominance (`r = 0.6841`, n = 66) and with outlier region fraction (`r = 0.5334`) -- level repair does *less* badly on more outlier-shaped scopes, the opposite of a level-repair-suits-level-geometry signature.  Given the Task-index confound above, the honest reading is that this is a Task-position effect, not a geometry effect.

### 2. Does the outlier family concentrate positive on outlier geometry?

**There is no outlier geometry to concentrate on, so the question is unanswerable -- but the family split itself is real and large.**  In electricity the outlier family runs 41 positive against 18 negative probes (mean gain +0.0149) while level repair runs 2 positive against 67 negative (mean gain -0.0767) over the same 9 Tasks.  That is a strong, reproducible family-level signature.

But it is **not** geometry-conditioned: within the outlier family the correlation between gain and outlier dominance is `r = -0.0074` (n = 60), i.e. flat.  The family advantage is uniform across the geometry range that was probed, not concentrated on the outlier-shaped end of it.

The same family split does **not** hold in T233, where the outlier family is 6+/8- and level repair is 7+/10- -- both mixed.  So the electricity family preference is cohort-specific, which is precisely why it cannot serve as a cross-domain prior.

### 3. Any cell with >= 3 distinct positive Tasks and zero opposing?

**None -- 0 eligible cells.**  The cells with enough positive volume are all blocked by opposing evidence, in both cohorts:

| evidence set | program | distinct + | distinct - | conflict | opposing | blocked by |
| --- | --- | --- | --- | --- | --- | --- |
| `t233_newobs_unguided` | `repair_level_shift` | 6 | 7 | 0 | 7 | opposing evidence |
| `t233_newobs_unguided` | `hampel_filter` | 3 | 5 | 2 | 7 | opposing evidence |
| `t233_newobs_unguided` | `outlier_mad+repair_level_shift` | 1 | 0 | 0 | 0 | short 2 Task(s) |
| `t233_newobs_unguided` | `repair_level_shift+hampel_filter` | 1 | 0 | 0 | 0 | short 2 Task(s) |
| `electricity_sequential_descriptive` | `outlier_mad` | 6 | 2 | 0 | 2 | opposing evidence |
| `electricity_sequential_descriptive` | `outlier_iqr` | 4 | 1 | 0 | 1 | opposing evidence |
| `electricity_sequential_descriptive` | `hampel_filter` | 3 | 2 | 3 | 5 | opposing evidence |
| `electricity_sequential_descriptive` | `repair_level_shift` | 1 | 8 | 0 | 8 | opposing evidence |

The closest thing to a clean cell anywhere is electricity `outlier_mad` at 6 distinct positive Tasks, and it still carries 2 distinct negatives.  It is also from Skill-accumulating arms, so it is not UNGUIDED evidence and could not be a precheck candidate even at zero opposing.

## Verdict

* **`SCOPE_CLASS_DEGENERATE_ALL_FULL_SCOPE_MIXED`** -- the pre-registered scope-class variable is constant (`MIXED`) on every series any probe has ever acted on, in all three cohorts.  Per the pre-declared exit this is a **dumb** result: it falsifies this measurement, not the mechanism hypothesis, and no further feature substitution is attempted here.
* **`PER_SERIES_UTILITY_NOT_JOINABLE_TRAIN_EVAL_DISJOINT`** -- independently of the above, per-series utility is measured only on eval series while geometry is labelled only on train series, and the two uid sets are disjoint by construction.  A genuine per-series response cannot be recovered from the paid traces at all.
* **Mechanism-precheck eligible cells: 0.**  No authorization action was taken, and none was available to take.

## What would make this testable, if it is worth testing

Two independent blockers, each with a concrete fix:

1. **No geometric contrast in scope.**  Every probed series is `MIXED`.  Testing a geometry-conditioned mechanism needs the scope selector to admit pure-geometry series, or a roster whose scope-eligible series are not all mixed.  Note this is a change to what gets selected, not to how it is measured.
2. **No per-series utility.**  Gain exists only as a downstream eval-series quantity.  Attributing utility to the geometry of an acted-on series needs per-train-series utility to be produced in the first place, for example by scoring a probe applied to one scope series at a time.  That is new measurement, not re-analysis, and it costs Episodes.

The cheaper and better-supported finding from this pass is the family-level one in question 2: in electricity the outlier family is materially better than level repair (41+/18- against 2+/67- probes), reproducibly across 9 artifacts, with no geometry conditioning required to state it.  That is directly actionable for in-domain program ordering, and it needs no cross-domain authorization to use.

## Provenance and discipline

* 0 LLM calls; no Episode executed; no code changed; nothing committed.
* Reads only frozen artifacts under `artifacts/functional/e2` plus the frozen census `m0a_mask_geometry_census_v1.json`, whose row unit is `cohort x task_episode_id x train series uid at observation_cutoff = support_origins[0]`.
* Sealed data read: **no**.
* `electricity` is an A5-vs-A3 development target with prior Outcome exposure. Nothing here may shape Skill clauses used for an electricity-Target contrast; confirmation of any mechanism signature would require a fresh Target.
* Series-level readings are engineering diagnosis and do not enter an authorization ledger.  Exactly two files were written: this report and its companion JSON.

