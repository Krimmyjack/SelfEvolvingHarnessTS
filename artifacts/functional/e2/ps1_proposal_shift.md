# PS-1 -- proposal shift under a Scoped hypothesis

protocol: `ps1_proposal_shift_v1`  evidence grade: **development-mechanism**  git: `258c28d66fb1029b1ab8eaf552c4136368f463c5`

**SOURCE_PROVENANCE_INSUFFICIENT**

items 1 through 4 pass on both sources: both are real executed Episodes, both unguided, both materially positive on Support and on delayed, and the two families are independent.  Item 5 fails: neither execution record persists any deployment-visible Pattern, so a five-axis Scope cannot be intersected from stored fields and the hypothesis card would have no machine-evaluable WHEN clause.  Stopped before Part 1 as the book directs.

## Part 0 -- dual-source provenance gate

| item | source | pass | evidence |
|---|---|---|---|
| 0_records_located | - | PASS | both records located |
| 1_real_executed_episode | source_A | PASS | `GunPointAgeSpan/fit_only_artifact_target_hampel_filter_a3_GunPointAgeSpan_r1_p1`, DELAYED / LOCAL_ACTIVE |
| 2_unguided | source_A | PASS | retrieved only 3; beyond bootstrap: none |
| 3_material_positive_on_support_and_delayed | source_A | PASS | Support +0.5000, delayed +0.4000, relation POSITIVE, Skill `fast_winner_classification_ridge_raw_plus_differ` |
| 1_real_executed_episode | source_B | PASS | `PowerCons__impulse_v2_target_hampel_filter_a3-reset_PowerCons__impulse_v2_r1_p1`, DELAYED / LOCAL_ACTIVE |
| 2_unguided | source_B | PASS | retrieved only 3; beyond bootstrap: none |
| 3_material_positive_on_support_and_delayed | source_B | PASS | Support +0.0714, delayed +0.5000, relation POSITIVE, Skill `fast_winner_classification_ridge_raw_plus_differ` |
| 4_family_independence | - | PASS | GunPointFamily vs PowerCons |
| 5_machine_executable_five_axis_scope | - | **FAIL** | axes available ['task_kind', 'consumer_id', 'metric', 'program_geometry']; missing ['deployment_visible_pattern_intersection'] |

### Deployment outcomes of the two sources

| source | deploy source | program | held-out gain |
|---|---|---|---|
| source_A | FROZEN_ACTIVE_SKILL_RECALL | hampel_filter | 0.2689873417721519 |
| source_B | FROZEN_ACTIVE_SKILL_RECALL | hampel_filter | 0.08333333333333326 |

### Axis 5 in detail

| axis | intersection | agree | source |
|---|---|---|---|
| task_kind | classification | True | stored field |
| consumer_id | ridge-raw-plus-difference-v1 | True | stored field |
| metric | accuracy | True | stored field |
| deployment_visible_pattern_intersection | {} | False | stored field |
| program_geometry | hampel_filter | True | stored field |

- neither provenance record persists a single one of the 20 non-task_kind observable-contract leaves.  The only Context the records hold is witness statistics, support_reproduces_fit_signal, observer node positions and slice sizes, and none of those is in contracts/observables.OBSERVABLE_FEATURES, so none can become an applicability leaf that retrieval.evaluate_applicability could ever read.

| source | Pattern leaves stored | Context fields the record does keep |
|---|---|---|
| source_A | **none** | controlled_impulse_positions, fit_rows, legacy_scope_decision, legacy_scope_reasons, observation_rows, observer_localized_nodes, observer_recovered_all_nodes, official_train_rows, series_length, slice_rows, support_pool_rows, support_reproduces_fit_signal, witness |
| source_B | **none** | consumer, family_key, family_repeat, forward_position, group, harmful_outlier_operators, injection, key_heldin_readout, largest_legal_heldin_magnitude, learnability, menu_oracle_program, metric, min_slice_rows, n_heldin, n_heldout, official_train_rows, oracle_set, series_length, slice_resolution, slice_rows, total_points, unit_id |

## The gap, precisely

the provenance chain for a legal cross-domain hypothesis is broken at the persistence layer, not at the evidence layer.  Both sources are real, unguided, materially positive on both gates and independent by family -- items 1 through 4 all pass.  What neither execution record kept is the deployment-visible Pattern the Scope has to be built out of.

Where the Pattern does exist today:

- `artifacts/functional/e2/s1a_r3_pool_census.json units[].pattern_view` -- usable: **False**.  sealed audit artifact.  Its own isolation banner forbids it entering any arm prompt, store or retrieval view, and a hypothesis card is exactly a Fast-visible surface
- `artifacts/functional/e2/s1_oracle/*.json pattern_view / public_features_binned` -- usable: **False**.  sealed exam key; same banner, stronger reason
- `recomputation via _build_cell + extract_public_features on the same frozen cell` -- usable: **False**.  deterministic and outcome-free, but the book's rule 5 says to intersect from the fields the records already store and to stop rather than reconstruct.  Not done here; named so the main line can decide

**Smallest thing that would close it**: persist the binned deployment-visible pattern view on the round or cell record at write time.  The Fast path already computes it every round -- run_e2_s1_curriculum_four_arms._run_round builds `features = extract_public_features(block, task_kind=...)` and hands it to run_online_round as fast_features -- so this is a record-keeping change, not a new computation.  Once a run persists it, this gate passes on that run's own fields and PS-1 proceeds without any reconstruction.

- closing it needs one fresh source-B-shaped run, or a re-run of both, with the field persisted.  Source A predates this runner entirely and would have to be re-earned rather than re-read.

## Frozen protocol for Parts 1-3 (pre-registered, not run)

The gate did not pass, so no card was compiled and no arm ran.  The protocol below is frozen now, before any outcome of this experiment has been seen, so that closing the gap does not reopen the design.

- exam substrate: `GunPointOldVersusYoung__impulse_v2`
- arms: A3, A5-neutral, A5-scoped
- replicates per arm: 4; run ids `ps1_run1` .. `ps1_run12`
- budgets: LLM <= 12/run and <= 150 total; fit <= 10/run and <= 120 total; wall <= 9000 s
- verdict set: PROPOSAL_SHIFT_CONFIRMED, SHIFT_WITHOUT_CONVERSION, NO_PROPOSAL_SHIFT, PLACEBO_EFFECT, SOURCE_PROVENANCE_INSUFFICIENT, COMPUTE_BUDGET_EXCEEDED

- **per_run**:
  - every raw proposal each round, tagged with its Program family
  - hampel family: proposed / selected / passed the verifier / earned Support / earned delayed / deployed
  - probes and wasted probes
  - LLM and Consumer-fit cost
  - non-hampel proposal diversity (crowding-out check)
  - harm events and worst-class recall delta
- **aggregate_by_arm**:
  - hampel-family proposal rate at run granularity (how many of the 4 runs proposed it at all)
  - conversion funnel proposed -> selected -> verifier -> Support -> delayed -> deployed
  - mean cost
  - deployed utility distribution
- **verdicts**:
  - `PROPOSAL_SHIFT_CONFIRMED`: A5-scoped's hampel proposal rate separates from both controls (order of >=3/4 against <=1/4) AND at least one run completes proposal -> Support -> delayed -> deployment AND harm does not rise AND A5-neutral shows no systematic separation from A3
  - `SHIFT_WITHOUT_CONVERSION`: proposal rate separates but nothing converts; the report names the layer it stopped at
  - `NO_PROPOSAL_SHIFT`: the three arms' proposal distributions overlap
  - `PLACEBO_EFFECT`: A5-neutral departs systematically from A3, i.e. the presence of a card changes behaviour on its own and any scoped effect has to be reinterpreted against that baseline
  - `SOURCE_PROVENANCE_INSUFFICIENT`: the Part 0 gate did not pass; no card was compiled and no arm ran
  - `COMPUTE_BUDGET_EXCEEDED`: a cap was hit; completed runs are kept and reported
- **statistics**: n=4 per arm is reported as counts and effect sizes.  No p-values.
- **scope_caveat**: GunPointOldVersusYoung__impulse_v2 shares GunPointFamily with source A.  This experiment isolates the proposal-shift mechanism and must not be cited as cross-family transfer capability.

## Oracle isolation

- builtins.open, io.open, os.open, Path.open, Path.read_text and Path.read_bytes are wrapped at module import; any path containing artifacts/functional/e2/s1_oracle/ raises OracleIsolationBreach while the phase is 'arm'
- unblocked reads by phase: {'setup': 0, 'select': 0, 'judge': 0}
- arm-phase attempts: 0, leaks 0

## Cost

- LLM: 0 (the gate spends none and Parts 1-3 did not start)
- Consumer fits: 0
- wall clock: 0.07 s
- downloads: 0

## Obligations

- **part0_items**: {'0_records_located': True, '1_real_executed_episode:source_A': True, '2_unguided:source_A': True, '3_material_positive_on_support_and_delayed:source_A': True, '1_real_executed_episode:source_B': True, '2_unguided:source_B': True, '3_material_positive_on_support_and_delayed:source_B': True, '4_family_independence': True, '5_machine_executable_five_axis_scope': False}
- **prior_slot_implementation**: not reached: no card was compiled and no arm ran.  The slot is declared as a runner-layer experiment mechanism only -- an independent paragraph in the agent's construction-time context, supplying no frozen candidate and changing no budget
- **production_governance_unmodified**: True
- **methods_package_unmodified**: True
- **runtime_contracts_operators_unmodified**: True
- **t1_authorization_retrieval_unmodified**: True
- **llm_calls**: 0
- **consumer_fits**: 0
- **downloads**: 0
- **arms_run**: 0
- **exam_substrate_opened**: False
- **sealed_artifacts_not_read**: True
- **sealed_artifacts_not_rewritten**: True
- **full_repo_pytest_not_run**: True
- **stage_report_not_written**: this book does not touch docs/STAGE_REPORT; another diagnostic book is in flight and the main line records for it

## Outside the book

- the persistence gap is systemic, not specific to these two runs: the Fast path computes the binned pattern view every single round and hands it to run_online_round as fast_features, and no runner on the classification line writes it to its artifact.  Every future cross-domain hypothesis will hit this same gate.
- the two sealed places the pattern view does live -- the r3 census and the s1_oracle keys -- both carry isolation banners, so the only legal supply is a live run that records its own Context.  A card built from either would be an oracle leak wearing a Scope.
- source A predates the S1 runner line and its artifact shape has no slot for the field, so closing the gap for source A means earning the positive again rather than re-reading it.
- second, independent reason source B's cell record cannot be a card source: the course record it lives in is the frozen curriculum entry, and that entry carries oracle-derived selection metadata -- oracle_set, menu_oracle_program, key_heldin_readout, largest_legal_heldin_magnitude, harmful_outlier_operators.  That is correct for a judging-side curriculum freeze and no arm ever reads it, but it means a card compiler pointed at course[] would be reading answers.  The Context a card may legally quote has to come from the round record, which is exactly the record that does not keep it.
- S1c's own GPOvY record is worth keeping next to this: A3 proposed nothing in r1 and one verifier-rejected candidate in r2, K0-fixed probed outlier_iqr to +0.0909 and still deployed identity, and A5-online proposed only outlier_mad twice and had both rejected by the verifier.  The +0.184 hampel headroom was never proposed by any arm, which is the observation PS-1 exists to explain.
