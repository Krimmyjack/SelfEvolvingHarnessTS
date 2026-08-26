# S1 four-arm evolution curriculum -- frozen course (r2)

protocol: `s1_curriculum_four_arms_v1`  revision: **r2**  evidence grade: **DEVELOPMENT**  git: `dbd840a50bddfb0b9e74b0284e2e2f88c3396776`

supersedes `artifacts/functional/e2/s1_curriculum_frozen_r1.json` (kept, not discarded)

The course below was selected mechanically from the sealed census and oracle artifacts by the rules in section 1.  No unit was chosen by hand and no outcome reordered anything after the rules ran.  The readability screen reads `cell.slice_rows` off the sealed oracle files, so it costs zero new Consumer fits.

## 1. Selection rules (declared before scoring)

- **revision**: r2.  The r1 rule ranked every group by smallest total points, which selected *against* the feedback surface: six of seven r1 units had a held-in slice of at most two rows and three had an empty r2 delayed slice, so no relation but NEUTRAL was reachable and the guard channel could never compile.  r2 replaces the ranking with a readability floor plus a readability ranking.  The r1 course is kept at s1_curriculum_frozen_r1.json/.md.
- **min_slice_rows**: min(cell.slice_rows) over r1_support, r1_delayed, r2_support, r2_delayed: the coarsest surface the frozen two-round protocol will read on that unit.  A slice of n rows moves accuracy only in steps of 1/n
- **slice_readability_floor**: admission gate for all four groups: min_slice_rows >= L, L walking the ladder [5, 4, 3].  A group that cannot fill its quota at L steps down one rung and the step is written into ladder_trace; nothing is reselected silently.
- **necessary_condition**: |key held-in readout| >= 1 / min_slice_rows, computed from the sealed oracle numbers already on disk.  The key readout is the one the group is defined by: for harm units the largest-magnitude qualifying outlier harm, for learnable units the oracle program's held-in headroom.  See necessary_condition_scope for why the other two groups are not screened by it.
- **necessary_condition_scope**: the condition binds only on the two groups whose defining held-in readout is non-zero (harm, learnable).  identity units are defined by an identity oracle set and HELDOUT_ONLY units by a held-in reading of zero, so |readout| >= 1/n is unsatisfiable for them by construction and applying it literally empties both groups at every rung of the ladder -- see literal_application_counterfactual.  For those two groups the informative requirement is that a material reading *would* have been visible had one existed, which is exactly the slice floor.  Flagged for main-line confirmation.
- **within_group_ranking**: descending min_slice_rows; ties broken by descending |key held-in readout|, then ascending unit_id.  No outcome may reorder.
- **family_deduplication**: cross-course, not per-group: the seven units should carry seven distinct family_key values.  When a group cannot fill its quota without a repeat, the repeat is taken best-ranked-first and named in family_census.repeated_families and in the group's ladder_trace.
- **relaxation_ladder**: per group, in order: (floor 5, 4, 3 with strict cross-course family distinctness), then (floor 5, 4, 3 with family repeats allowed but still preferring a fresh family within each rung).  The first rung that fills the quota wins.  The floor is relaxed before family distinctness is *not* the order: keeping the floor high is the whole point of r2, so the strict-family rungs are tried across the whole ladder first, and only then are repeats allowed starting again at floor 5.  Every rung tried is recorded.
- **group_selection_order**: harm -> learnable -> identity -> HELDOUT_ONLY.  Earlier groups consume families and units; the order is fixed here, before any candidate is scored.
- **harm_evidence**: an oracle-scored unit qualifies when outlier_mad or outlier_iqr is legal on it (verifier passed and cohort modified fraction <= 0.10) and its held-in headroom is <= -0.005, i.e. materially harmful on held-in.  Two units, ranked by within_group_ranking.
- **learnable_positive**: learnability == LEARNABLE.  Two units, ranked by within_group_ranking.
- **identity**: oracle_set is exactly identity (or empty).  Two units, ranked by within_group_ranking.
- **heldout_only_temptation**: learnability == HELDOUT_ONLY.  One unit, ranked by within_group_ranking.
- **unit_disjointness**: no unit id may appear twice in the course
- **forward_order**: harm A -> learnable A -> harm B -> identity A -> learnable B -> HELDOUT_ONLY -> identity B.  Design intent: the guard should be compilable after the second harm unit, so every unit after it tests whether the guard actually fires.
- **reverse_order**: the exact reverse of the forward order
- **domain_namespace**: unit_id (dataset__injection).  The whole course runs at one condition (fit_only_artifact), so dataset alone would collapse two curriculum units of the same substrate into one counted Task and the guard census (risk_skill._task_of) would undercount.

## 2. The seven units, forward order

| # | unit | group | family | learnability | oracle set | slice rows (r1s/r1d/r2s/r2d) | min slice | resolution | key held-in readout | why |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | MiddlePhalanxOutlineCorrect__impulse_v2 | harm_evidence | PhalanxFamily | HELDOUT_ONLY | repair_level_shift | 45/45/45/45 | 45 | 0.0222 | 0.0944 | materially harmful legal outlier program on held-in: outlier_iqr(held-in -0.0944).  smallest held-in slice 45 rows, so the surface resolves 0.0222, and the harm magnitude 0.0944 clears it |
| 2 | DistalPhalanxOutlineCorrect__burst_cls2 | learnable_positive | PhalanxFamily (repeat) | LEARNABLE | outlier_iqr,outlier_mad | 46/45/45/44 | 44 | 0.0227 | 0.0333 | LEARNABLE, held-in headroom +0.0333.  smallest held-in slice 44 rows, so the surface resolves 0.0227, and the headroom clears it; family repeat, named because the group could not fill its quota with a fresh family |
| 3 | PowerCons__impulse_v2 | harm_evidence | PowerCons | LEARNABLE | hampel_filter | 14/14/14/12 | 12 | 0.0833 | 0.1852 | materially harmful legal outlier program on held-in: outlier_iqr(held-in -0.1852).  smallest held-in slice 12 rows, so the surface resolves 0.0833, and the harm magnitude 0.1852 clears it |
| 4 | FreezerRegularTrain__burst_cls2 | identity | FreezerFamily | N/A | identity | 12/12/10/10 | 10 | 0.1000 | 0.0000 | oracle set is identity, so the correct end state is to change nothing.  smallest held-in slice 10 rows, so the surface resolves 0.1000, so 'nothing helps' is a reading rather than a blind spot; the largest legal held-in magnitude on this unit is 0.2500 |
| 5 | GunPointOldVersusYoung__impulse_v2 | learnable_positive | GunPointFamily | LEARNABLE | hampel_filter | 11/10/10/10 | 10 | 0.1000 | 0.4146 | LEARNABLE, held-in headroom +0.4146.  smallest held-in slice 10 rows, so the surface resolves 0.1000, and the headroom clears it |
| 6 | ECG200__impulse_v2 | heldout_only_temptation | ECGFamily | HELDOUT_ONLY | repair_burst_segment | 9/7/7/7 | 7 | 0.1429 | 0.0000 | HELDOUT_ONLY: the oracle-set program helps held-out (+0.0400) but held-in cannot approve it (headroom +0.0000) -- the abstention temptation.  smallest held-in slice 7 rows, so the surface resolves 0.1429, so the held-in zero is measured, not missing |
| 7 | Ham__impulse_v2 | identity | Ham | N/A | identity | 9/8/8/8 | 8 | 0.1250 | 0.0000 | oracle set is identity, so the correct end state is to change nothing.  smallest held-in slice 8 rows, so the surface resolves 0.1250, so 'nothing helps' is a reading rather than a blind spot; the largest legal held-in magnitude on this unit is 0.0303 |

### Readability ladder actually walked

| group | quota | rung used | family repeats | downgraded from floor 5 | short by | rungs tried |
|---|---|---|---|---|---|---|
| harm_evidence | 2 | floor 5, repeats forbidden | none | False | 0 | floor 5/strict -> 2 |
| learnable_positive | 2 | floor 5, repeats allowed | ['DistalPhalanxOutlineCorrect__burst_cls2'] | False | 0 | floor 5/strict -> 1 ; floor 4/strict -> 1 ; floor 3/strict -> 1 ; floor 5/repeat -> 2 |
| identity | 2 | floor 5, repeats forbidden | none | False | 0 | floor 5/strict -> 2 |
| heldout_only_temptation | 1 | floor 5, repeats forbidden | none | False | 0 | floor 5/strict -> 1 |

forward: `['MiddlePhalanxOutlineCorrect__impulse_v2', 'DistalPhalanxOutlineCorrect__burst_cls2', 'PowerCons__impulse_v2', 'FreezerRegularTrain__burst_cls2', 'GunPointOldVersusYoung__impulse_v2', 'ECG200__impulse_v2', 'Ham__impulse_v2']`

reverse: `['Ham__impulse_v2', 'ECG200__impulse_v2', 'GunPointOldVersusYoung__impulse_v2', 'FreezerRegularTrain__burst_cls2', 'PowerCons__impulse_v2', 'DistalPhalanxOutlineCorrect__burst_cls2', 'MiddlePhalanxOutlineCorrect__impulse_v2']`

Design intent of the forward order: the guard should become compilable after the second harm unit, so every unit after it is a test of whether the guard actually fires.

## 3. Families and substrates

- distinct families in course: **6** (ECGFamily, FreezerFamily, GunPointFamily, Ham, PhalanxFamily, PowerCons)
- repeated families: ['PhalanxFamily']
- repeated substrates: none

## 4. Arms

- **Static**: no adaptation; identity frozen and deployed on every unit; only the scoring fit is spent
- **A3-reset**: cold start from h0 on every unit; zero carry between units
- **K0-fixed**: every unit starts from the same K0; normal in-unit held-in adaptation; no write-back between units
- **A5-online**: same K0 start and in-unit protocol; full Slow integration between units including the risk lifecycle; the pool evolves with the course

## 5. K0

- base: methods/ttha/harness/h0 (the three bootstrap Skills)
- bootstrap Skills: inspect_and_localize, build_contrastive_candidates, select_or_identity_and_verify
- inert Slow card: `source_investigation_cls_v1` from `artifacts/functional/e2/t6_cls_op_r2_three_arms.json`; TRY = `NO_AUTHORIZED_ACTIVE_RECOMMENDATION`; allowed_tools = []; carries frozen steps = False
- **excluded on purpose**: the C40 Target-local hampel capability is NOT in K0.  It is a frozen-steps capability bound to one Source domain; placing it in K0 would leak an answer across domains and contaminate both K0-fixed and A5-online.

## 6. Domain-binding hooks

- **1_stamp**: unit_id (dataset__injection).  The whole course runs at one condition (fit_only_artifact), so dataset alone would collapse two curriculum units of the same substrate into one counted Task and the guard census (risk_skill._task_of) would undercount.
- **2_carry_wall**: a Target-local capability (frozen program steps; not an experience card) is dropped at the unit boundary unless its stamp equals the next unit's domain_namespace
- **3_scope_v1**: a Source-derived experience card reaches the next unit's Fast surface only when task_kind, consumer_id and metric match, its authorizing pattern-view intersection is non-empty and is satisfied by the next unit's binned deployment-visible pattern view, and its Program geometry is a real operator.  Dataset name is not an axis.

## 7. Budgets

- llm_per_unit_per_adaptive_arm: 15
- fit_per_unit_per_arm: 25
- llm_per_slow_integration: 6
- llm_total_cap: 400
- fit_total_cap: 900
- wall_seconds_cap: 10800
- over_cap_verdict: COMPUTE_BUDGET_EXCEEDED

## 8. Pre-registered readout (judged in S1c, not here)

- **primary**: A5-online must be non-inferior to both A3-reset and K0-fixed on quality (cumulative held-out utility) and on harm (harm events, worst-class harm), and must improve at least one of cumulative regret or cumulative total cost by a material margin
- **material_threshold**: 0.005
- **verdict_ceiling_for_a_single_order_single_run**: S1_DEVELOPMENT_EVOLUTION_SIGNAL
- **judged_in**: S1c.  This book freezes the readout and does not judge it.

## 9. Shortfalls and substitutions

- learnable_positive: found 2 -- quota could not be filled at slice floor 5 with strict cross-course family distinctness; the rung actually used was floor 5 with family repeats allowed -- repeated: ['DistalPhalanxOutlineCorrect__burst_cls2']

## Oracle isolation

- builtins.open, io.open, os.open, Path.open, Path.read_text and Path.read_bytes are wrapped at module import; any path containing artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach while the phase is 'arm'
- arm-phase attempts: 0 (blocked 0, leaks 0)

## Not in this book

- the full course is not run here.  This entry freezes the course and the readout only; S1c runs it.

