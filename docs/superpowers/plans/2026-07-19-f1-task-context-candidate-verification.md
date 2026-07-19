# F1 Implementation Plan: TaskContext and Candidate Verification

**Status:** active implementation plan

**Parent release:** `m0-agent-harness-v0.1.0`

**Purpose:** make the existing forecast Agent consume an explicit, immutable
task objective and runtime-generated candidate facts before selection, without
changing the task, downstream model, Harness content, or promotion policy.

## 1. Fixed experiment boundary

F1 keeps all current scientific axes fixed:

    task                 forecast
    downstream model     frozen Chronos / fixed:m0
    Harness content      released H2, unchanged
    corpus               small M0-compatible forecast slice
    Harness promotion    disabled
    model selection      disabled

F1 changes two interfaces only:

1. implicit forecast intent becomes a resolved, hashed TaskContext; and
2. proposed candidates are executed once and receive public-safe mechanical
   receipts before the Agent selects.

F1 is successful when these interfaces are truthful, stable, and used by the
real Agent. Utility improvement is not an F1 gate.

## 2. Contract design

### 2.1 Stable task identity

Add immutable contracts in `contracts/task.py`:

- `TaskQualityContract`: closed-vocabulary objective, preservation, harm,
  evidence, verification, and abstention semantics;
- `DeploymentConstraintSpec`: fixed model policy plus bounded candidate and
  modification constraints; and
- `TaskContext`: `TaskSpec` + the two contracts with a canonical SHA.

The default F1 forecast contract contains no operator, case, injection, private
metric threshold, or candidate instruction. Structural validation rejects an
unsupported revision, task mismatch, unknown vocabulary, or content/SHA
mismatch before any Agent call.

### 2.2 Run dependency identity

Add `contracts/run_context.py` with `RunDependencyBinding`:

    task_context_sha
    evaluator_adapter_id
    instrument_epoch
    corpus_epoch
    capability_bundle_sha
    runtime_sha
    harness_sha
    code_commit
    provider_id
    model_id

Its SHA is run provenance. Instrument, corpus, runtime, or provider changes do
not alter TaskContext identity and are not task predicates.

### 2.3 Compatibility bridge

Extend `PreparationRequest` with optional resolved `task_context` and
`run_dependency_binding` fields.

- legacy callers may omit both and retain the M0 prompt/cache path;
- an explicit TaskContext must contain exactly the existing `task_spec`;
- a provided RunDependencyBinding must reference the same TaskContext SHA; and
- F1 experiment entry points require both bindings.

The released H2 authoring files are never modified. Post-M0 compilation may
produce a new runtime bundle with the same Harness content SHA; this is a
compatibility migration, not a Harness edit.

## 3. Candidate receipt design

Add `runtime/candidate_verification.py` with:

- `CandidateVerificationReceipt`: public-safe facts and canonical receipt SHA;
- `CandidateExecutionArtifact`: internal output plus the receipt; and
- one verifier that executes each candidate at most once.

The public receipt includes only:

    candidate identity/kind and Program SHA
    operator legality
    compile/execute status
    shape and finite checks
    effect-equivalence to identity
    modified fraction and normalized modified regions
    outside-inspected-region modification
    bounded warning/rejection codes

It explicitly excludes U/J, clean/reference comparisons, candidate ranking,
oracle positions, injection family, and judge recommendations.

Lifecycle:

    propose
      -> compile/legality validation
      -> execute once
      -> receipt
      -> risk filter
      -> select-visible pool
      -> select
      -> reuse selected execution artifact

Identity receives a runtime-owned receipt. Hard-invalid candidates are excluded
from selection but leave rejection receipts in the trace. A valid no-op remains
selectable with an `EFFECT_EQUIVALENT_TO_IDENTITY` warning so selection behavior
is observable rather than silently rewritten.

## 4. Agent and cache binding

Update `TTHAFastAgent` so every F1 stage receives the same public TaskContext
payload. The select payload contains each selectable candidate and its receipt.

Update `AgentRequest` with `task_context_sha` and provenance-only
`run_context_sha`. The semantic request hash includes TaskContext SHA, while the
already-resolved stage messages provide stage-specific identities:

- inspect: TaskContext + public input + effective Harness view;
- propose: TaskContext + inspection/tools + effective Harness view;
- select: TaskContext + candidate/receipt sets + effective Harness view.

RunContext SHA is recorded but excluded from semantic cache identity unless it
changes actual messages, schemas, tools, or model identity.

Update `DecisionTrace` with:

    task_context_sha
    run_context_sha
    candidate_receipt_shas
    selectable_candidate_ids
    rejection_receipts

Behavior signatures include receipt-mediated selection facts but not private
execution artifacts.

## 5. Implementation order

1. Add task/run contracts, canonical identities, default forecast factories,
   and mismatch validation.
2. Add the compatibility fields to `PreparationRequest` without changing
   legacy callers.
3. Add candidate verification receipts and deterministic fixtures.
4. Refactor the fast path to execute candidates once, expose receipts, and
   reuse selected artifacts.
5. Bind TaskContext to Agent requests, prompts, cache identity, and trace.
6. Recompile released H2 under the F1 runtime and record that Harness content is
   unchanged while runtime dependency identity changes.
7. Run deterministic and compatibility tests.
8. Run one minimal live forecast-only F1 slice after preregistration.

## 6. Deterministic acceptance

Tests must prove:

1. TaskContext SHA is canonical and independent of instrument/corpus epochs;
2. TaskSpec/contract mismatch fails before Agent invocation;
3. unknown contract vocabulary and unsupported revisions fail mechanically;
4. legacy M0 requests remain executable;
5. explicit F1 requests bind one TaskContext SHA across inspect/propose/select;
6. identity, valid, no-op, illegal, overflow, and execution-failure receipts
   are deterministic and contain no forbidden private fields;
7. each Program executes once even when receipt, risk, selection, and final
   result all consume it;
8. rejected candidates cannot be selected but remain auditable;
9. select-stage cache changes when the receipt set changes, while inspect and
   propose identities do not depend on future receipts; and
10. released H2 source content remains byte-for-byte unchanged.

## 7. Minimal live F1 experiment

Create a preregistration before using the relay. Use 6-8 existing forecast
cases: a small mixture of H2 in-scope level shifts, identity/risk cases, and one
no-op or rejected-candidate diagnostic.

Run:

- explicit correct forecast TaskContext on all 6-8 cases;
- neutral contract on 2-4 matched report-only cases;
- no Harness promotion and no second cycle.

Primary protocol readings:

    stage TaskContext-SHA agreement                 required 100%
    selectable candidates with public receipts      required 100%
    rejected candidates with trace receipts          required 100%
    candidate duplicate execution count              required 0
    utility/private fields in Agent-visible receipts required 0
    protocol/transport failure rate                  required 0
    resume/cache replay                              exact on repeated requests

Secondary report-only readings:

    identity/program selection
    no-op avoidance
    receipt warning use in verification actions
    calls and prompt/completion tokens
    correct-versus-neutral behavior delta

F1 passes even if utility does not improve. It fails if the task input is not
bound, receipts are untruthful or leak private value, legacy behavior breaks, or
the real Agent cannot consume the new select payload.

## 8. Exit and next stage

On pass, publish an `f1-integration-receipt/1` and advance to F2a instrument
maintenance. F2a first calibrates the flat public level probe and missing-family
readability without modifying H2. F2b then adds the frozen-classifier query-only
classification adapter defined in the F-track specification.

On failure, emit a typed resolution:

- protocol/cache/prompt binding failure: framework defect;
- untruthful receipt: runtime/instrument defect;
- real Agent rejects a valid schema: interface defect and bounded retry review;
- no behavioral response despite correct delivery: report-only
  `TASK_CONTEXT_UNUSED` cause for later F3, not an F1 Harness edit.
