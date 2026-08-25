# CLS-OP -- second Task on the shared Harness

protocol: `t6_cls_op_r2_three_arms_v1`  evidence grade: **DEVELOPMENT**

## Verdict

**CLS_LIFECYCLE_OK_NO_ADVANTAGE**



- real Episodes formed: None (None)
- three-arm cells: None
- deployment purity: True

DEVELOPMENT.  Controlled impulse injection on UCR splits that W48/W55/W56 already opened.  A5 > A3 here is a development-grade reading and needs the frozen confirmation on an unused local UCR Target before it can be called classification transfer.

## Budget

- LLM: 69 of 90 (fast 68, slow 1)
- Consumer fits: 41 of 600
- wall clock: 1225.9 s

## Source Experience

Episodes: 9  by relation: {'CONFLICT': 2, 'NEGATIVE': 1, 'NEUTRAL': 6}

Source-derived Skill written: True

- **WHEN**: When task_kind == classification in a new cohort and a comparable context_condition can be observed.
- **OBSERVE**: Inspect task_kind and the applicable context_condition in the current Workspace, then check whether the available programs have distinct-task positive, negative, or immaterial Target Support.
- **TRY**: NO_AUTHORIZED_ACTIVE_RECOMMENDATION
- **RISK**: The census provides no unguided positive support for any listed program, and the lone negative result for repair_level_shift is not repeated; do not infer a transferable preference from this evidence.
- **VERIFY**: Require this Task's own Target Support to establish a positive relation under the observed context_condition across distinct tasks before believing any active recommendation.
- **FALLBACK**: If observation does not support a guarded hypothesis, retain no active recommendation and gather task-local Target Support rather than selecting an operator.

## Three-arm table

| dataset | arm | deploy source | applied | held-out acc | gain vs identity | A5-A3 | A5-A4 | harmed classes |
|---|---|---|---|---|---|---|---|---|
| GunPointAgeSpan | A3 | FROZEN_ACTIVE_SKILL_RECALL | [{'op': 'hampel_filter', 'params': {}}] | 0.8513 | +0.2690 |  |  | [] |
| GunPointAgeSpan | A4 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.5823 | +0.0000 |  |  | [] |
| GunPointAgeSpan | A5 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.5823 | +0.0000 | -0.2689873417721519 | 0.0 | [] |
| PhalangesOutlinesCorrect | A3 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.4697 | +0.0000 |  |  | [] |
| PhalangesOutlinesCorrect | A4 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.4697 | +0.0000 |  |  | [] |
| PhalangesOutlinesCorrect | A5 | FROZEN_LEDGER_NO_INCUMBENT_IDENTITY | identity | 0.4697 | +0.0000 | 0.0 | 0.0 | [] |

## Full readout set

| cell | Skill formed | first-Skill LLM | first-Skill executions | held-in delayed | held-out acc | vs identity | worst class harm | abstained rounds | Support/delayed agree:disagree |
|---|---|---|---|---|---|---|---|---|---|
| GunPointAgeSpan/A3 | True | 6 | 1 | +0.4000 | 0.8513 | +0.2690 | 0.0000 | 0 | 2:0 |
| GunPointAgeSpan/A4 | False | - | - | n/a | 0.5823 | +0.0000 | 0.0000 | 0 | 0:0 |
| GunPointAgeSpan/A5 | False | - | - | n/a | 0.5823 | +0.0000 | 0.0000 | 2 | 0:0 |
| PhalangesOutlinesCorrect/A3 | False | - | - | n/a | 0.4697 | +0.0000 | 0.0000 | 2 | 0:0 |
| PhalangesOutlinesCorrect/A4 | False | - | - | n/a | 0.4697 | +0.0000 | 0.0000 | 0 | 0:0 |
| PhalangesOutlinesCorrect/A5 | False | - | - | n/a | 0.4697 | +0.0000 | 0.0000 | 2 | 0:0 |

### Contrasts

- **GunPointAgeSpan**: A5-A3 accuracy -0.2690; A5-A4 accuracy +0.0000; A5-A3 first-Skill executions None; A4 beats A5: False
- **PhalangesOutlinesCorrect**: A5-A3 accuracy +0.0000; A5-A4 accuracy +0.0000; A5-A3 first-Skill executions None; A4 beats A5: False

## Pre-registration, scored

- **P1** HELD -- after the fix, hampel_filter or repair_level_shift enters the Source probe order (['hampel_filter', 'repair_level_shift'])
- **P2** FALSIFIED -- and may form a POSITIVE Source Episode ({'CONFLICT': 2, 'NEGATIVE': 1, 'NEUTRAL': 6})
- **P3** FALSIFIED -- the Slow audit may then authorize a non-empty TRY, giving A5 a real prior ([])
- **P4** HELD -- on the Target, hampel_filter can become a non-identity candidate and compete for a Target-local Skill (None)

## Why A5 fell below A3

classification: **prior_delivered_but_steered_the_proposal_elsewhere**

the Source card was retrieved into every A5 round and carries no frozen Workflow, so it supplied no candidate and could only act on the proposal stage as text.  A5's proposals went to the level-shift family on both rounds and the deployment constraint rejected them, while A3 reached the local-median family in one round.  This is neither a retrieval miss nor feedback bias: the prior was delivered, read, and unhelpful.

- Source card retrieved in every A5 round: True
- Source card supplied an executable candidate: False
- Source card execution right: withheld_no_authorized_try_operator

## Obligations

- **methods_package_unmodified**: True
- **new_files**: ['evaluation/functional/consumers/cls_scope_adapter.py (the one permitted thin evaluate_fn adapter)', 'evaluation/functional/run_e2_t6_cls_op_shared_harness.py (the runner; a book that writes an artifact needs one, and the adapter deliberately holds no protocol -- reported for ruling)']
- **forbidden_data_untouched**: no Yahoo, NOAA 2025, beyond_17520, NAB or SMD path is opened by this runner; the only data root is data/ucr_task_context
- **legacy_capability_card_not_injected**: the W56 promoted Capability card was read for archaeology only and was never written into experience_memory or a snapshot
- **fast_never_saw_raw_source_episodes**: Source Episodes stayed in their own per-cell Method instances; the Target arms construct with empty Memory and receive Source evidence only as the audited Skill on the snapshot
- **deviations**: ["The family's fit/support split is reused byte-for-byte, but the legacy support pool is quartered into per-round Support and delayed surfaces, because the shared lifecycle needs a delayed surface the family never produced.", 'The deployment-visible observation is a fixed ~3200-point window of the fit cohort rather than the whole block; the executor still acts on every fit row.  Without it, one actionability probe cost 215 s.', 'Slow Path is off inside the held-in rounds; the only Slow call is the Source consolidation that authors the six-section card.', "The per-view axis is the class axis, so CONFLICT means 'accuracy rose while a class recall fell'.  This is stricter than the family's accuracy-only gate and is the main reason POSITIVE is rare here."]
- **backend**: live Fast Agent

## Part 0b confirmation (post-run)

Regression subset after the h0 lock regen: 2 failed, 216 passed, against 40 failed / 170 passed before it. All 38 snapshot-lock-mismatch failures cleared. The two that remain (test_fault_cases sequential ids, test_history_window_observation training-window assertion) failed identically on both sides of the verifier byte-swap baseline and touch neither ScopeExecutor.verify nor verify_candidate, so they stay on the mainline ledger as pre-existing breakage from another line.
