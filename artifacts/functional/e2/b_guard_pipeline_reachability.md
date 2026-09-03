# B -- guard pipeline reachability differential

Zero LLM.  Evidence grade: INFRASTRUCTURE.

Real Episode: `ECG200/fit_only_artifact_target_outlier_mad_a3_ECG200_r2_p1` from `artifacts/functional/e2/t6_cls_conf_dev_ecg200.json` (support_gain -0.1429, relation NEGATIVE).
Synthetic second unit: `Wine/fit_only_artifact` (same family, harm).

## The three repairs

- **1** `methods/ttha/online_loop.py:_write_target_episode` -- context_summary gains task_episode_id = the `domain` argument, which is the Episode's own domain_namespace
- **2** `evaluation/functional/task_episode_harness/agentic/source_skill.py:risk_guard_rows + build_skill_payload` -- risk_guards gains evidence_distinct_task_count and a structured deprioritized_scoped_evidence row (operators + context scope + count); the body is byte-identical and no free text is added
- **3** `evaluation/functional/run_e2_t6_cls_op_shared_harness.py:_risk_lifecycle, called from _run_round` -- the existing agentic/runner.run_risk_skill_lifecycle is called on this arm's own Episodes after every round; no new lifecycle, no new census, no new rule

## Safety-guard chain, before vs after

| stage | reader | before | after | before reachable | after reachable |
| --- | --- | --- | --- | --- | --- |
| 1. Episode carries the unit id | `risk_skill._task_of (risk_skill.py:72-74)` | ["", ""] | ["ECG200/fit_only_artifact", "Wine/fit_only_artifact"] | False | True |
| 2. distinct harm Tasks for the family | `risk_skill.census (risk_skill.py:83-112)` | 1 | 2 | False | True |
| 3. guard candidate at MIN_DISTINCT=2 | `risk_skill.risk_candidates (risk_skill.py:153-184)` | [] | ["outlier_mad"] | False | True |
| 4. classification line compiles it | `run_e2_t6_cls_op_shared_harness._risk_lifecycle -> agentic/runner.run_risk_skill_lifecycle` | "no call site existed" | ["target_risk_outlier_mad"] | False | True |
| 5. served into the Fast view | `retrieval.resolve_harness_view(role='fast')` | [] | ["target_risk_outlier_mad"] | False | True |
| 6. form is a structured avoid, not a candidate | `risk_skill.risk_skill_payload + fast_agent._skill_frozen_candidates` | null | {"skill_id": "target_risk_outlier_mad", "skill_kind": "safety", "allowed_tools": [], "risk_guards": {"deprioritize_only": true, "evidence_distinct_task_count... | False | True |

## Experience-card chain (repair 2)

| card scope | count before | count after | inert before | inert after |
| --- | --- | --- | --- | --- |
| as_shipped_scope | None | 2 | True | True |
| with_a_scope_finer_than_the_eligibility_gate | None | 2 | True | False |

## Structural findings

- **the shipped source card is scoped only to the eligibility gate** -- run_e2_t6_cls_op_shared_harness.SOURCE_APPLICABILITY is task_kind == classification, and retrieval._scopes_beyond_task_kind requires something finer, so repair 2 is necessary but not sufficient for the *experience card* branch of the middle tier. Narrowing that constant is a Scope change and is outside this book.
- **the classification census condition is not an observable feature** -- CENSUS_CONDITION_KEY = support_reproduces_fit_signal is absent from contracts/observables.OBSERVABLE_FEATURES, so it cannot appear in observable_applicability at all; the guard row records it inside risk_guards, which is free-form JSON. Giving the middle tier a real Context scope on this line therefore needs an observable-contract addition, not a wiring fix.
- **classification Episodes carry no task_signature** -- online_loop._write_target_episode writes no context_summary.task_signature, so risk_skill.applicability_from returns {'const': True} and the minted guard is unconditioned within the arm. The frozen risk_skill rule says that is the correct reading of a family that failed under every observed Context; narrowing it means writing a signature, which is a Scope change and is outside this book.
- **the guard body names the units the harm happened in** -- risk_skill.risk_skill_payload renders negative_task_ids into the body ('Tasks: ...'), so with dataset/condition ids a later unit's Fast prompt sees earlier cohort names as provenance. Applicability is still decided by observable_applicability, never by the names. Changing the body is a semantics change and is outside this book.
- **tests/functional/test_skill_revocation.py does not parse on Python 3.10** -- pre-existing: a multi-line f-string expression at line 166 is 3.12+ syntax. Not touched, not caused here.

## Verdict

Every stage of the safety-guard chain was unreachable before the three repairs and is reachable after them at n=2 distinct units, on one real ECG200 outlier_mad harm Episode plus one synthetic second-unit harm.  The experience-card branch now carries the count the predicate reads, but stays withheld from Fast because the shipped card's Scope is the eligibility gate itself -- a structural finding, not a repair.
