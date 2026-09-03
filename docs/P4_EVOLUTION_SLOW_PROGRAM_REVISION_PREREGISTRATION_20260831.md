# P4-Evolution Forecast Slow Program Revision Preregistration

Status: **FROZEN BEFORE THE NEW LIVE TRAJECTORY**  
Evidence role: exposed-development mechanism evidence only  
Existing P4-Performance artifacts: unchanged and interpreted as Fast+Runtime,
without Slow  
P4-Evolution gate before this experiment: `HELD`  
Natural Final / UCR TEST / sealed AD outcomes: closed

## 1. Question and claim ceiling

This experiment tests the missing A5 path:

```text
an already-active Skill is naturally retrieved and used
-> current held-in feedback exposes a local fault
-> Slow proposes one bounded Program PATCH
-> current Support replay creates pending
-> one independent Support-B evaluation approves or rejects it
-> a later independent held-in window makes Fast retrieve the approved revision
-> Runtime verifies whether the revised Skill improves over the frozen old Skill
```

It does not retell the completed Fast-only P4 result and does not use a
controlled card as performance evidence. One successful trajectory can support
only `EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD`.
It cannot by itself release H3, establish cross-domain transfer, or release
P4-Performance, Classification, AD performance, or Final.

## 2. Frozen old Skill and legal Domain

The old Skill is fixed before any call in this trajectory:

- id: `fast_winner_forecast_pooled_ridge_a1_smase_e1v2_outlier_iqr`;
- revision: `1`;
- Program: one Common-DSL step, `outlier_iqr`;
- applicability: `task_kind == forecast`;
- Task / Consumer / Metric: `forecast|pooled_ridge_a1|sMASE`;
- lifecycle: `LOCAL_ACTIVE`, with current-Target Support still required;
- formation evidence: natural Fast selection on exposed NOAA development,
  positive Support, positive delayed feedback, an out-of-selection activation
  probe, and promotion;
- exclusion: it is not a positive-control or manually authored answer card.

The machine source is the already materialized `a5_pooled` Skill from
`t6_45_frep_a5a3_replay`. Selection is outcome-independent for this new
trajectory: among Skills present before this preregistration, it is the unique
entry that has the exact Task/Consumer/Metric, is `LOCAL_ACTIVE`, carries one
executable Common-DSL Program, requires Target confirmation, and has the
natural formation chain above.

This is a Target-local NOAA Skill, so the trajectory remains in the same NOAA
Domain. It must not be moved into KDD merely because its current machine Scope
is broad. Such a move would bypass the project's Source-derived Skill rule.

## 3. Frozen development trajectory

Only the already-exposed NOAA-2024 development array `[0, 8760)` is readable.
The 2025 confirmation interval `[8760, 17520)`, data beyond 17520, Natural
Final, and all other sealed outcomes remain unread.

The first trajectory uses the first three chronological, horizon-sized windows
of the pre-existing exposed tail block:

| Stage | origin | role |
|---|---:|---|
| 1 | 8472 | old-Skill use, feedback, and at most one Slow PATCH |
| 2 | 8520 | the sole independent Support-B approval/rejection |
| 3 | 8568 | independent re-encounter and v2-versus-v1 comparison |

Horizon is 48. Outcome intervals do not overlap. The origins, roster, order,
and stopping rules do not change if a result is inconvenient. The only
permitted repeat, and only after a complete first-chain pass, is already fixed
as origins `8616 / 8664 / 8712` with the same roles and budgets. No other
origin search is allowed.

The frozen Task, pooled Ridge Consumer, sMASE, Common DSL, material threshold,
Prompt, model setting, and preprocessing constraints are unchanged.

## 4. Arms and causal isolation

- `K0-fixed`: starts with v1 and cannot write back.
- `A5-Slow`: starts from the same semantic snapshot and may write back at most
  one approved revision.

The two arms have independent stores, backend counters, caches, evaluators,
and Target Episode lists. Both begin with empty Target Episodes and no pending
update. Raw Source or prior-trajectory Episodes never enter Fast. The only
authorization difference is A5's writeback permission.

The old Skill is merely made retrievable. It is not forced into the Fast
choice or Runtime winner. Failure to retrieve or select it is a terminal
result, not permission to reorder candidates or choose a new Skill/window.

## 5. Stage semantics

### Stage 1 -- natural local fault

Both arms run the normal Fast path with `allow_fast_skill=True`. A5 continues
only if all of these are observed mechanically:

1. v1 is in the resolved Fast view;
2. Fast explicitly selects the candidate supplied by v1;
3. Runtime executes exactly v1's frozen steps;
4. current Support-A records a material negative or `CONFLICT` response for
   that attributed candidate.

The constrained-proposal fact is derived after the frozen Fast candidate pool
has been processed, not hard-coded:

- a fully evaluated non-Skill proposal is materially positive -> `True` and
  the Program-fault hypothesis is falsified;
- every non-Skill proposal in the frozen pool is attempted/rejected and none
  succeeds -> `False`;
- incomplete evidence -> `Unknown`, so Slow abstains.

Only `False` may route to `SKILL_CONTENT_GAP`.

### Stage 1 PATCH -- exactly one Program surface

The sole writable surface is:

```text
skill_library.entries/fast_winner_forecast_pooled_ridge_a1_smase_e1v2_outlier_iqr.body
```

Only `PATCH` is legal. `ADD`, `REVOKE`, Scope, independently selected Risk,
candidate policy, retrieval, bootstrap instructions, and other surfaces are
forbidden. Slow may select one verifier-earned typed Program from the existing
B=8 inventory (seven non-identity Common-DSL Programs plus identity) or
abstain. Identity is not a legal replacement PATCH. Runtime binds the selected
`patch_id` to the exact verified steps; the model cannot author executable parameters. A
byte/semantic no-op relative to v1 is rejected. The proposed Program must
replay positively on Stage-1 Support before it can become pending.

The old Skill stores the same Program in three representations: executable
`body`, `allowed_tools`, and `risk_guards.frozen_plan.program`. The latter two
are machine-owned Program mirrors, not independently editable Slow surfaces.
The one atomic Program PATCH therefore changes `body`, increments `revision`,
and makes Runtime derive those two mirrors from the bound steps. Scope and all
non-Program Risk fields must remain unchanged. A revision whose three Program
representations disagree is invalid and cannot become evidence.

### Stage 2 -- independent approval

Pending v2 is not exposed to Fast. Stage 2 performs no LLM call. The exact
pending Program is evaluated once on origin 8520 and classified by the frozen
Consumer rule. Only `POSITIVE` approves it. Rejection or unavailable evidence
discards pending and leaves v1 active.

Approval must preserve Skill id, Scope, and every non-Program Risk field while
incrementing the positive integer revision exactly `1 -> 2`. The Program
mirrors must agree with the revised `body`. K0 evaluates v1 on the same stage
for a matched descriptive control but cannot update.

### Stage 3 -- independent re-encounter

A5 starts with approved v2; K0 starts with v1. Neither Fast input contains the
Stage-1 raw Episode, Slow rationale, or Stage-2 outcome. The causal comparison
is valid only if A5 retrieves/selects v2 and Runtime's actual winner is v2,
while K0 retrieves/selects and wins with v1.

A complete pass requires:

- `utility(v2) - utility(v1) >= 0.005` on origin 8568;
- v2 is itself at least `+0.005` over identity;
- A5 harm count and magnitude do not exceed K0;
- the active A5 entry is revision 2 and K0 remains revision 1.

If v2 was approved but these re-encounter conditions fail, A5 rolls back to
the saved v1 snapshot. No v3 is created.

## 6. One whole-trajectory B=8 ledger per arm

Stage transitions never reset a budget counter.

| Resource | K0 whole trajectory | A5 whole trajectory |
|---|---:|---:|
| LLM calls | <= 8 | <= 8, including Slow |
| input + output tokens | <= 60,000 | <= 60,000 |
| full Consumer evaluations | <= 8 | <= 8 |
| Stage-1 Support-A | <= 4 | <= 4, including PATCH replay |
| Stage-2 Support-B | <= 1 | <= 1 |
| Stage-3 Support-A | <= 3 | <= 3 |
| cheap structural/verifier probes | <= 24 | <= 24 |
| accepted updates | 0 | <= 1 |
| wall clock | <= 2,700 s | <= 2,700 s |

The two-arm live ceiling is 16 LLM calls, 120,000 tokens, 16 treatment
Consumer evaluations, and 48 cheap probes. The three shared identity reference
fits are reported separately, so the absolute Consumer-fit ceiling is 19.

The ninth attempted arm call is blocked before reaching the backend and is not
charged. Budget exhaustion ends thinking for that arm, produces identity with
`BUDGET_EXHAUSTED_ABSTAIN_IDENTITY`, preserves the consumed-cost record, and
does not justify a scientific method verdict. Transport, global budget, token,
time, protocol, and data failures remain fail-closed.

## 7. Terminal branches

- no legal old Skill or same-Domain data ->
  `OLD_SKILL_QUALIFICATION_FAILED__H3_HELD`;
- old Skill not retrieved/selected/executed ->
  `OLD_SKILL_NOT_CAUSALLY_USED__H3_HELD`;
- v1 has no local fault -> `NO_LOCAL_V1_FAULT__H3_HELD`;
- a non-Skill proposal succeeds ->
  `PROGRAM_HYPOTHESIS_FALSIFIED__H3_HELD`;
- evidence incomplete or Slow abstains -> `SLOW_SAFE_ABSTAIN__H3_HELD`;
- unauthorized/no-op/invalid PATCH ->
  `UNAUTHORIZED_OR_INVALID_PATCH__H3_HELD`;
- Support replay rejects -> `PATCH_SUPPORT_REJECTED__H3_HELD`;
- independent B rejects -> `INDEPENDENT_SUPPORT_B_REJECTED__H3_HELD`;
- id/revision/Program-mirror/Scope/non-Program-Risk invariants fail ->
  `VERSION_CHAIN_INVALID__NO_SCIENTIFIC_VERDICT`;
- v2 or v1 is not causally used at re-encounter ->
  `REVISED_SKILL_NOT_CAUSALLY_USED__H3_HELD`;
- approved v2 has no improvement or more harm -> rollback and
  `REVISION_REENCOUNTER_FAILED_ROLLED_BACK__H3_HELD`;
- arm LLM ceiling is reached before completion ->
  `LLM_BUDGET_EXHAUSTED_BEFORE_CHAIN__H3_HELD`;
- all gates pass ->
  `EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD`.

Only the pre-frozen repeat may follow a first pass. Two complete passes permit
a separate gate review; they do not silently alter historical P3/P4 artifacts.
If the first trajectory reaches safe abstention, B rejection, or failed
re-encounter, sampling stops as specified by the user.

## 8. After this mechanism slice

Only after the full chain and its pre-frozen repeat pass may a new Forecast
performance experiment be launched with `Static / A3 / K0-fixed / A5-Slow`.
That future A5 must contain Fast -> real feedback -> Slow revision -> later
Fast. The completed Slow-disabled P4 remains a separate Fast+Runtime control
and is never pooled with it. Classification follows with Macro-F1 and adequate
sample counts; AD remains conditioning/safety only. Final stays one-shot and
sealed until the development gates are reviewed.

No Operator, Consumer, metric, threshold, Scope rule, Prompt, feedback budget,
SHA, Manifest, or Hash infrastructure is added by this preregistration.
