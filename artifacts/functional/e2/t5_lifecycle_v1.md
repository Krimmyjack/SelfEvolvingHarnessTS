# T5 -- one Harness entry, two Consumers (t5_lifecycle_v1)

Evidence grade: **POSITIVE_CONTROL**.

## Verdict

**INCOMPLETE_LLM_BUDGET**

the live trajectory stopped: INCOMPLETE_LLM_BUDGET

## Part 0

- HEAD `eb4a03c` -- checkpoint: T4b abstain channel (TASK_SEPARATION_REGRESSION, C16) + T4 close-out and #41 T5 authorization
- tracked-modified at start of Part A: ['ethods/ttha/method.py', 'methods/ttha/online_loop.py', 'operators/registry.py']

## Part A

- [x] `A1_ad_supply_non_empty` -- AD candidate supply is 4 programs
- [x] `A1_allowlists_identical` -- F=['hampel_filter', 'outlier_iqr', 'outlier_mad', 'winsorize'] AD=['hampel_filter', 'outlier_iqr', 'outlier_mad', 'winsorize']
- [x] `A1_contract_id_sets_identical` -- ['hampel_filter', 'outlier_iqr', 'outlier_mad', 'winsorize']
- [x] `A1_forbidden_sets_identical` -- 22 forbidden entries, one function
- [x] `A1_identity_reserved_not_registered` -- identity is absent from OPERATOR_METADATA
- [x] `A1_other_ad_bans_untouched` -- smoothing/decompose AD bans left as they were
- [x] `A3_legacy_fixture_key_matches_old_literal` -- task_consumer_key(forecast/ridge/sMASE) == 'forecast|ridge|sMASE'
- [x] `A3_actual_f_key_is_minted_not_literal` -- actual F key 'forecast|pooled_ridge_a1|sMASE'; old literal 'forecast|ridge|sMASE' (differs, as expected)
- [x] `A3_ad_key_distinct` -- AD key 'anomaly_detection|ad_ridge_train_v3|macro_event_f1'
- [x] `A3_no_hardcoded_key_left_in_online_loop` -- no quoted forecast|ridge|sMASE remains in online_loop; the only surviving mention is the comment recording the removal
- [x] `A3_fallback_still_available_for_none_spec` -- task_spec=None keeps the historical default

### Key migration

| reading | value |
| --- | --- |
| legacy fixture (forecast/ridge/sMASE) | `forecast|ridge|sMASE` |
| actual T5 forecasting key | `forecast|pooled_ridge_a1|sMASE` |
| anomaly detection key | `anomaly_detection|ad_ridge_train_v3|macro_event_f1` |

the old literal was itself a dialect: the real forecasting Consumer on this substrate is the pooled ridge, so the minted key differs from the literal by design.  A byte-equal assertion here would only have forced a false green

## Part B (0 LLM)

- [x] `B1_support_positive_draft` -- outlier_iqr episode written
- [x] `B1_winner_formed` -- winner=[{'op': 'outlier_iqr', 'params': {}}]
- [x] `B1_draft_pending` -- fast winner stage=pending
- [x] `B1_delayed_positive_active` -- episode relation=POSITIVE status=LOCAL_ACTIVE
- [x] `B1_delayed_approved` -- delayed stage=approved
- [x] `B5_forecasting_positive_delayed_still_approves` -- the tightened gate still approves an unambiguously positive delayed
- [x] `B2_ad_candidate_supplied` -- AD probed 1 candidate(s): [{'candidate_id': 'cand_hampel_filter', 'gain': 0.050000000000000044}]
- [x] `B2_conflict_recorded` -- AD episode relation=CONFLICT (aggregate up, one series harmed)
- [x] `B2_conflict_grants_no_execution` -- winner=None approved=None status=EPISODE_ONLY
- [x] `B2_no_ad_skill_in_snapshot` -- no AD fast_winner skill was written
- [x] `B3_forecasting_key_correct` -- F episodes keyed forecast|pooled_ridge_a1|sMASE
- [x] `B3_ad_key_correct` -- AD episodes keyed anomaly_detection|ad_ridge_train_v3|macro_event_f1
- [x] `B3_zero_cross_task` -- the two arms' key sets are disjoint
- [x] `B4_f_skill_written` -- learned skills after B1: ['fast_winner_forecast_pooled_ridge_a1_smase_outlier_iqr']
- [x] `B4_next_f_round_retrieves` -- forecast view carries ['fast_winner_forecast_pooled_ridge_a1_smase_outlier_iqr']
- [x] `B4_next_ad_round_cannot_read_f_skill` -- anomaly_detection view carries ['build_contrastive_candidates', 'inspect_and_localize', 'select_or_identity_and_verify']
- [x] `B4_skill_id_is_task_scoped` -- task-scoped, hash-free ids: ['fast_winner_forecast_pooled_ridge_a1_smase_outlier_iqr']
- [x] `B6_support_positive_then_draft` -- Support POSITIVE reached pending Draft
- [x] `B6_delayed_conflict` -- episode=CONFLICT gate=CONFLICT
- [x] `B6_not_approved_pending_discarded` -- stage=delayed_rejected approved=None pending=None
- [x] `B6_skill_restricted` -- episode local_status=RESTRICTED
- [x] `B6_raw_readings_retained` -- aggregate=0.06000000000000005 series_read=4 harmed=1

## Part C -- live trajectory

| round | arm | chosen | winner | delayed | relation | skill |
| --- | --- | --- | --- | --- | --- | --- |
| r1 | forecasting | mad_outlier_repair | hampel_filter | 0.04500688582382373 | NEGATIVE,CONFLICT | - |
| r1 | anomaly_detection | outlier_mad_localized_extreme_deviation | none | None | CONFLICT | - |
| r2 | forecasting | hampel_extreme_deviation | hampel_filter | 0.04500688582382373 | CONFLICT | - |
| r2 | anomaly_detection | ERROR | AgentCallBudgetExceeded: Agent call budget exhausted at 16 | | | |

LLM calls: 1 / 16


## What the completed segment shows

- F r1: the Agent proposed and probed outlier_mad first (-0.2166, three of four series harmed -> NEGATIVE), then hampel_filter (+0.1920, no series harmed -> POSITIVE), which became the winner and reached a pending Draft Skill.
- F r1 delayed: aggregate +0.0450 but one of four series at -0.0714 -> CONFLICT -> not approved, pending discarded, the Episode written RESTRICTED. Under the gate this round replaced (dg >= -0.005) that same reading would have been APPROVED and written into the active snapshot.
- AD r1: the AD arm was supplied candidates at all, which was the first blocker; it proposed outlier_mad, whose Support aggregate was +0.2032 with one of twelve series at -0.1905 -> CONFLICT -> no winner, no Draft, Episode only.
- F r2: memory_resolution moved no_memory -> rendered, the candidate pool changed, and the Agent no longer probed the harmful outlier_mad at all (1 Support receipt instead of 2, harm_count 0 instead of 1) -- with TaskSpec, public features and Consumer key byte-identical to r1.

> no Skill was ever approved in the live trajectory, because every delayed window came back CONFLICT. Target-local Skill update and live cross-task Skill retrieval were therefore not exercised; they hold only at the Part B (scripted-reading) level. The AD arm's second round never ran.

## LLM budget arithmetic

three rounds and the first stage of a fourth consumed the whole cap, so the live entry costs about five agent calls per round, not the three the book's arithmetic assumed. fast prepare has exactly three stages (inspect / propose / select), each carrying validation_retries=1, so the four calls the book set aside as retry headroom were spent as ordinary traffic rather than left spare. A four-round trajectory needs roughly 20, not 16. The cap is the book's and was not raised.

## Collateral on the other line

the A5 rename to fast_winner_{task}_{model}_{metric}_{op} is a naming contract five existing functional tests depend on: three are literal expectations of the old id string. The other two are not: evaluation/functional/task_episode_harness/e1.py detects an already-present arm-local Skill by the prefix _LOCAL_SKILL_PREFIX = 'fast_winner_e1v2_', and the task scope now sits between 'fast_winner_' and the operator segment, so the prefix stops matching, the reuse path is not taken, and the re-ADD collides with the ABSENT precondition (AddTargetExistsError).

- e1.py is a FROZEN_SURFACE_V9 member and this round authorized method.py only. Reported for the main line to route; not self-adjudicated, not worked around, and the book's ID format was implemented as written rather than bent to keep the prefix alive.

  - `tests/functional/test_skill_revocation.py::test_delayed_harm_revokes_retrieved_skill`
  - `tests/functional/test_skill_evolution_e0.py::test_e0_add_compile_retrieval_delayed_and_revocation`
  - `tests/functional/test_e1_v2_protocol_repair.py::test_e1_v2_arm_isolation_window_non_overlap_and_local_skill_reuse`
  - `tests/functional/test_g1_proposal_guidance.py::test_next_task_reuses_instead_of_colliding`
  - `tests/functional/test_f1_forecast_pilot.py::test_f1_pilot_runs_on_frozen_h2_without_promoting_harness`


## Ambiguities (reported, not self-adjudicated)

- Part B is a wiring acceptance on scripted readings, not new evidence: its numbers are fixtures chosen to put each lifecycle cell under test, and they travel the real entry point unchanged.
- The T1 injected copy carries only the twelve training stations, so the forecasting arm splits them 8 train / 4 eval rather than reach outside the injected copy for T1's four original eval series. The AD arm keeps all twelve because its scored surface is the Query region, which the training block never touches. The two arms therefore see the same series but not the same roster roles; the book fixes the Harness, Memory, Agent, DSL and menu as shared, not the roster.
- The AD adapter's action region is the concatenation of the three verified windows ([120,840)), not the full T1 block [120,900). The 44 of 48 injected training events that fall inside it carry the label signal; the four outside it are not seen by the fit.
- The AD reading is reported to the executor as -macro_f1 so the executor's own gain arithmetic (baseline - candidate) yields candidate_F1 - baseline_F1. No executor code was changed to do it, but the negation is a convention this adapter chose and it is recorded here rather than buried.
- Both rounds of a task run at the same origin on purpose (the book requires byte-identical Context between r1 and r2), so the AD adapter's memo makes the second round's identical readings free. The fit and scoring are closed-form and deterministic, so a re-run returns identical numbers by construction; only cache misses are counted against the AD evaluation budget.
- A4 tightens handle_feedback_delayed for every task, not just AD: NEUTRAL delayed no longer extends privilege. That is the authorized behaviour change of this round, and it is what B5 re-checks on the forecasting side rather than a regression to be worked around.
- handle_feedback_support (the Slow path) was NOT given the same relation gate. Part C runs with allow_slow=False so it is never reached here; extending the gate there is a second wiring surface and was left for the main line to route.
