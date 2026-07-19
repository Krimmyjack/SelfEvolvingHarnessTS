# Post-M0 Functional Evolution Design: Task-Conditioned TTHA

**Status:** M0-R released as `m0-agent-harness-v0.1.0`; F1 functional
integration passed; F2a instrument maintenance is the active stage

**Date:** 2026-07-18

**Primary objective:** grow the existing M0 minipipe into the project’s main
task-conditioned data-readiness system, one working capability at a time

**Scope:** M0 closure, TaskContext integration, task/evaluator adapters, shared
Harness evolution, functional qualification, downstream-model selection, and
natural-data integration

**Non-goal for the current build:** a publication-grade confirmatory experiment,
large arm matrix, exhaustive statistical proof, or a second algorithm branch

## 1. Decision

This document specifies the next functional evolution of the existing TTHA
codebase. It does not replace the M0 minipipe and does not create an unrelated
task-conditioned implementation.

The main line is:

    M0-R  close and freeze the live M0 growth loop
      -> F1  bind task goals and mechanical candidate verification
      -> F2  add reusable task/evaluator adapters
      -> F3  evolve one shared task-conditioned Harness
      -> F4  qualify and release a useful frozen Harness snapshot
      -> F5  add downstream-model selection as a separate capability layer
      -> F6  run the same serving path on natural data

M0 remains the mechanism-optimization kernel:

    case
      -> Agent behavior
      -> outcome and intervention feedback
      -> first actionable fault
      -> one bounded Harness edit or an external change request
      -> replay or calibration
      -> versioned capability growth

F1-F6 add inputs, adapters, capabilities, and release gates around this kernel.
They must not duplicate the Agent, Harness, candidate runtime, operator registry,
feedback router, replay controller, or lineage store.

These phases are construction milestones, not exploratory branches competing to
replace the architecture. Each phase has a predetermined capability output. A
failure identifies the framework component to repair before the same main line
continues.

## 2. Current project state

### 2.1 What offline M0 already establishes

The deterministic M0 path has exercised the intended mechanical loop:

1. public fast-path execution;
2. behavior recording;
3. private outcome and intervention evaluation;
4. first-actionable-fault attribution;
5. bounded slow-path edit generation;
6. paired replay and risk checks; and
7. supported snapshot lineage.

This is sufficient evidence that the components and contracts can form a closed
loop under controlled responses.

### 2.2 What live M0 has and has not established

Early live runs validated the relay path, cache, information walls, four-view
feedback, failure-pattern construction, and slow-Agent invocation while exposing
contract and instrument defects. The final M0 sequence then closed the real
growth loop:

- H0 autonomously proposed and promoted an `ADD SkillEntry` edit to H1;
- a new held-out slice exposed selection conservatism rather than another
  library gap;
- the slow path autonomously proposed a one-surface Skill-body PATCH to H2;
- H2 produced positive incremental reuse on three of four selected held-out
  targets; and
- 44 out-of-scope risk cases retained identical effective views and behavior.

The released project state is:

    M0 mechanism gate: CLOSED_PASS
    M0 release: COMMITTED_RELEASE_VALIDATED
    release tag: m0-agent-harness-v0.1.0
    implementation commit: b2b799dbf352b564551b8706a2366cfac685f980
    receipt commit: 95e4e76
    F1 functional integration: CLOSED_PASS
    F2a instrument maintenance: READY

The release exports H2 as a tracked, recompilable snapshot, keeps a redacted
evidence manifest and derived CapabilityLedger in Git, and binds private
raw/cache artifacts to a restore-tested archive SHA. M0 establishes one real
scoped forecasting capability-growth trajectory; it does not establish broad
level-shift recall, task conditioning, model selection, or natural transfer.

## 3. M0-R advancement gate: close M0 before adding the task axis

M0-R was the project-advancement gate, not a publication experiment and not a
retroactive redefinition of every M0 unit test. It is now closed and frozen.

### 3.1 Required repairs

Finish the active M0 fixes without adding TaskQualityContract or second-task
scope. At minimum, resolve the blockers already exposed by the live run:

- slow-edit schema and observable-AST agreement;
- effect-equivalent/no-op candidate accounting;
- misleading or unreadable public observables;
- current fault-route semantics;
- the invalid or unreadable missing-family corpus behavior;
- replay and receipt regeneration after those changes; and
- durable run artifact retention.

### 3.2 Small live-closure run

Run a small, deliberately readable live M0 corpus. A full 36-case scientific run
is not required for this gate. Six to twelve target/risk cases across two small
cycles are sufficient when they exercise every load-bearing transition.

The run must demonstrate:

1. at least one effect-distinct executable Program candidate;
2. one slow-Agent edit that passes schema and observable-AST validation;
3. exact resolution to one declared Harness surface;
4. actual paired replay;
5. the predicted Agent behavior change;
6. positive target recovery on the supported scope;
7. no material regression on the automatic risk set;
8. one `SUPPORTED_EDIT` promotion from H0 to H1; and
9. positive reuse of the promoted capability on a new seed, location, or
   severity in the next cycle or a small holdout slice.

This gate does not require every degradation family to be repairable, a positive
mean across the entire corpus, or statistical significance.

### 3.3 M0-R outputs

M0-R produces:

- a tagged M0 code state;
- a frozen H0 and promoted diagnostic H1 snapshot;
- a live-closure receipt;
- the run/corpus/rules/model/runtime SHAs;
- restorable raw/cache artifact references;
- a short list of unresolved instrument and capability requests; and
- a new implementation branch or worktree for F1.

No F1 code is merged into the M0 tag.

The release used a two-commit protocol to avoid a self-referential receipt:

1. Commit A contains the implementation and H2 snapshot and is the exact tree
   exercised by the full `268 passed` acceptance suite.
2. Commit B contains the acceptance receipt, public evidence, release manifest,
   restore receipt, and CapabilityLedger and is the annotated tag target.

Post-M0 compatibility recompiles the same H2 authoring content against newer
runtime contracts. It never edits the tagged H2 release in place and never
records a runtime compatibility migration as Harness capability growth.

The new Git branch/worktree is only implementation isolation. Its reusable
changes are merged back into the same TTHA architecture; it is not a second
method or a long-lived product fork.

## 4. Architectural invariants

F1-F6 are allowed to extend the framework only while the following invariants
hold.

### 4.1 One serving path

There remains one public method path:

    PreparationRequest
      -> TTHAMethod
      -> TTHAFastAgent
      -> CandidatePool
      -> Program executor
      -> PreparationResult + DecisionTrace

F1-F6 may add fields, adapters, and later stages through versioned contracts.
They must not create a second forecast Agent, classification Agent, natural-data
Agent, or model-selection Agent.

### 4.2 One Agent core

Fast and slow roles continue to use the same `TTHAAgentCore` backend and typed
stage schemas. New task behavior is expressed through TaskContext, retrieved
Harness content, task adapters, and versioned output contracts.

### 4.3 One Harness lineage

The system retains one `HarnessSnapshot`, `SnapshotStore`, surface registry,
edit controller, replay mechanism, and lineage model.

The functional target is one shared H* containing:

- cross-task skills;
- task-scoped applicability where behavior genuinely conflicts;
- safety and abstention rules;
- retrieval/selection improvements; and
- scoped memory.

F3 does not create separately deployed `H_forecast` and `H_classification`
systems.

### 4.4 One Program and operator runtime

Existing `Program`, `Candidate`, `CandidatePool`, executor, and canonical
operator registry remain authoritative. Task adapters may evaluate outputs but
cannot invent a parallel operator implementation.

New operators enter through a capability release with registry and runtime
version bumps. They are never written by the slow Agent.

### 4.5 M0 compatibility remains executable

The frozen M0 tag remains reproducible. On the post-M0 branch, the existing M0
forecast composition remains available as a compatibility scenario while the
generic engine is extracted incrementally.

No big-bang rewrite of `evaluation/minipipe/cycle.py` is permitted. Shared
components are extracted behind compatibility wrappers only when a second task
actually needs the seam.

## 5. Current-code integration map

The implementation extends existing ownership boundaries as follows.

| Existing area | Reused responsibility | Additive post-M0 change |
| --- | --- | --- |
| `contracts/task.py` | canonical `TaskSpec` | add `TaskQualityContract`, `DeploymentConstraintSpec`, and `TaskContext` |
| `contracts/run_context.py` | new stable dependency boundary | add `RunDependencyBinding` separate from task semantics |
| `contracts/method.py` | `PreparationRequest/Result` | add a compatibility bridge to resolved TaskContext |
| `methods/ttha/agent_core.py` | one typed Agent backend | bind TaskContext SHA and contract payload |
| `methods/ttha/fast_agent.py` | inspect/propose/select/execute | consume resolved TaskContext; keep Program path |
| `methods/ttha/slow_agent.py` | one-surface Harness edit | consume sanitized task-aware failure evidence |
| `methods/ttha/harness/` | snapshots, skills, surfaces | reuse task predicates in applicability; no new Harness type |
| `runtime/candidate_pool.py` | candidate identity and choice | expose runtime-generated verification receipts |
| `runtime/decision_trace.py` | behavior lineage | add TaskContext and verification receipt references |
| `runtime/executor.py` | canonical Program execution | unchanged except versioned receipt additions if needed |
| `operators/registry.py` | task legality and operator metadata | unchanged until a capability release |
| `evaluation/minipipe/feedback/` | first fault, patterns, routing | add task scope and external change-request outcomes |
| `evaluation/minipipe/replay/` | single-surface replay and lineage | reused without task-specific forks |
| `evaluation/minipipe/valuation/` | forecast valuator | wrapped by the forecast task adapter |
| `evaluation/minipipe/tasks/` | new extension seam | task case/valuator adapters for forecast and classification |

The `evaluation/minipipe/tasks/` directory is an adapter layer, not a second
pipeline. If a generic cycle engine is extracted, the intended structure is:

    evaluation/minipipe/engine.py
      reusable evolution orchestration

    evaluation/minipipe/cycle.py
      M0-compatible forecast composition

    evaluation/minipipe/task_cycle.py
      multi-task composition using the same engine

Copying the full M0 cycle into a classification-specific runner is forbidden.

## 6. Ownership and change authority

The project distinguishes deployable Harness learning from framework
maintenance.

| Object | Owner | Slow Agent editable | Change timing |
| --- | --- | ---: | --- |
| TaskQualityContract | task owner | no | between epochs |
| instruction/skill/applicability/risk/retrieval/memory | Harness loop | yes, one declared surface | within a Harness cycle |
| evaluator/probe/observable schema/fault rules | instrument owner | no | between instrument epochs |
| operator/runtime/model adapter | capability owner | no | versioned capability release |
| corpus/curriculum | experiment owner | no | between corpus epochs |

### 6.1 TaskQualityContract is not an editable Harness surface

TaskQualityContract defines the objective, structures to preserve, harms to
avoid, evidence expectations, verification dimensions, and abstention
conditions for a task. Dynamic cost, latency, provider, and call/token ceilings
belong to `DeploymentConstraintSpec` or the run budget, not to the quality
contract.

It must not contain:

- degradation or injection family names;
- corpus case IDs;
- operator names;
- defect-to-Program mappings;
- concrete parameters;
- private thresholds or oracle locations; or
- instructions to choose a particular candidate.

Both Agent roles receive it as immutable input. The slow Agent may learn how to
meet the contract by editing a legal Harness surface, but it cannot edit the
contract itself.

If the contract is unclear, the system emits a
`TASK_CONTRACT_CLARIFICATION_REQUEST`. Only the task owner can publish a new
contract revision and SHA, and that change starts a new project epoch.

### 6.2 TaskContext, DeploymentConstraintSpec, and RunDependencyBinding

Task semantics and execution dependencies are separate identities:

    task_context_sha = SHA(
      TaskSpec,
      TaskQualityContract,
      fixed/selectable downstream-model semantic policy,
      DeploymentConstraintSpec
    )

    run_context_sha = SHA(
      task_context_sha,
      evaluator_adapter_id,
      instrument_epoch,
      corpus_epoch,
      capability/operator bundle SHA,
      runtime SHA,
      Harness SHA,
      code commit,
      provider/model identity
    )

Changing a probe, evaluator implementation, operator bundle, corpus, runtime,
or provider does not change what the task means. These changes therefore bump
the run dependency identity, not `task_context_sha`. Instrument/capability
epochs must not become task predicates or serving-time proxies.

The tagged M0 H2 snapshot remains immutable:

    H2@M0
      harness_content_sha = X
      runtime_bundle_sha  = M0-runtime-X

    H2@F1-compatible-runtime
      harness_content_sha = X
      runtime_bundle_sha  = F1-runtime-Y

This compatibility compile is a runtime migration, not an H2-to-H3 HarnessEdit.

### 6.3 TaskContext compatibility bridge

The existing `PreparationRequest.task_spec` field is retained during F1 so
current call sites do not break.

F1 adds an optional resolved TaskContext binding. When present:

- `task_context.task_spec` must equal the existing `task_spec`;
- a mismatch fails before Agent invocation;
- F1+ internal code reads the resolved TaskContext;
- the TaskContext SHA is recorded in requests and traces and enters each stage's
  effective request identity;
- the separate RunDependencyBinding records evaluator/instrument/corpus/runtime
  dependencies; and
- F1+ run rules require TaskContext even though legacy M0 fixtures may use the
  compatibility path.

This bridge is additive. Replacing `task_spec` in every constructor is deferred
until all active call sites have migrated and requires a major protocol version.

## 7. Feedback-driven project evolution

The broader project improves more than Harness text, but only the Harness loop
is automatically editable during a run.

### 7.1 Harness Evolution Loop

This is the existing M0 slow path:

    failure pattern
      -> one declared Harness surface
      -> edit proposal
      -> paired replay
      -> behavior/effect/risk receipt
      -> promote or reject

It may edit instruction, skill, applicability, risk guards, retrieval,
candidate control, verification rules, or scoped memory according to the
surface registry and fault routes.

### 7.2 Instrument Maintenance Loop

Instrument problems include unreadable public evidence, evaluator
insensitivity, probe ambiguity, faulty localization, and invalid corpus damage.

The cycle emits an `INSTRUMENT_CHANGE_REQUEST` containing:

- supporting fault cluster;
- affected instrument component;
- predicted measurement improvement;
- small calibration cases;
- regression cases; and
- required dependency/version bumps.

The request is implemented by a human or controlled engineering workflow after
the current run. It is checked on fixed calibration cases, assigned a new
instrument epoch, and then returned to the M0 engine.

Instrument changes are not ordinary paired Harness edits because they change
the measurement system itself.

### 7.3 Capability Release Loop

Capability gaps include missing operators, task judges, public tools, model
adapters, and data-type support.

The cycle emits a `CAPABILITY_RELEASE_REQUEST`. A controlled engineering change
implements and tests the new capability, updates the registry/runtime bundle,
and starts a new capability epoch.

The slow Agent never writes Python, evaluator code, operator code, or model
execution code inside a Harness evolution run.

### 7.4 Cycle resolution

Every cycle ends with one typed project resolution:

    HARNESS_EDIT
    INSTRUMENT_CHANGE_REQUEST
    CAPABILITY_RELEASE_REQUEST
    CORPUS_CHANGE_REQUEST
    TASK_CONTRACT_CLARIFICATION_REQUEST
    NO_ACTIONABLE_CHANGE
    INVALID_OR_INFRA_CYCLE

An external request does not count as a failed Harness edit. It identifies the
next framework component to improve.

### 7.5 Epoch boundary

Evaluator, observable schema, corpus, operator bundle, TaskQualityContract, and
model adapter changes may occur only between evolution runs. Every run records:

- code commit;
- Harness snapshot SHA;
- runtime/operator bundle SHA;
- instrument epoch;
- corpus epoch;
- TaskContext SHA; and
- capability bundle SHA.

Cross-cycle replay is interpreted only when these dependencies remain fixed.

M0 and F-track use the functional stage numbers in this document. A later
publication-grade confirmatory composition uses separate `C0-C4` identifiers;
its arms, replicas, and sealed queries do not redefine these construction gates.

## 8. F1: TaskContext and candidate-verification integration

### 8.1 Functional purpose

F1 makes task goals a real input to the existing Agent path and makes candidate
execution facts visible before selection.

F1 does not evolve the Harness. It verifies that the new input and receipt
plumbing work without hiding failures behind fallback behavior.

### 8.2 Additions

F1 adds:

- `TaskQualityContract`, `DeploymentConstraintSpec`, `TaskContext`, and
  `RunDependencyBinding` schemas;
- TaskContext resolution and SHA binding;
- the compatibility bridge in `PreparationRequest`;
- runtime-generated `CandidateVerificationReceipt` objects;
- receipt IDs in select-stage input and `DecisionTrace`;
- model/call/cost metadata already required for reliable live operation; and
- a report-only correct/neutral/shuffled contract check.

The receipt lifecycle is fixed:

    Agent propose
      -> schema/operator validation
      -> execute or dry-run once
      -> CandidateVerificationReceipt
      -> risk filtering
      -> select-visible CandidatePool
      -> Agent select

The execution artifact is reused by the receipt, risk filter, selection, and
final result. Receipt generation must not execute the same Program twice.

Receipt visibility is also fixed:

- identity receives a runtime-owned identity receipt;
- every candidate entering the select pool has a public-safe verification
  receipt;
- schema-invalid, unauthorized, or execution-failed candidates receive an
  auditable rejection receipt in `DecisionTrace` but are not selectable; and
- a valid candidate with a non-fatal warning remains selectable with that
  warning visible.

Candidate verification covers the facts the runtime can establish cheaply:

- schema validity;
- allowed operators;
- compilation and execution status;
- output shape and finite-value validity;
- semantic equivalence to identity;
- modified fraction;
- outside-inspected-region modification; and
- declared warnings.

The Agent may explain warnings, but it cannot self-certify these facts.
Receipts never contain candidate utility, clean-reference comparisons, private
rankings, oracle locations, or judge-derived recommendations. Uncalibrated
phase/extrema diagnostics remain warnings in F1; promotion to a hard gate
requires an F2 instrument receipt.

Cache identity is stage-specific because receipts do not exist before proposal:

    inspect
      TaskContext SHA + public input + effective Harness view

    propose
      TaskContext SHA + inspection/tools/probes + effective Harness view

    select
      TaskContext SHA + candidate-set SHA + receipt-set SHA
      + effective Harness view

    slow
      sanitized failure evidence + task scope + surface catalog
      + dependency SHAs

Putting future-stage receipts into inspect/propose identities is forbidden.

### 8.3 Functional acceptance

F1 is complete when one small live command demonstrates:

1. every Agent stage that consumes task semantics is bound to the same
   TaskContext SHA, while the run records a separate dependency binding;
2. the existing M0 request path still works through the compatibility bridge;
3. identity remains present;
4. illegal or no-op Programs are explicit rather than silently accepted;
5. select receives runtime receipts;
6. `DecisionTrace` records what the Agent saw and what the runtime verified;
7. resume/cache behavior remains correct; and
8. no TaskQualityContract field is editable by either Agent role.

Contract validation is structural and semantic, not a prohibited-word grep.
Before any API call it rejects task-type mismatch, content/SHA mismatch,
unsupported revision, unknown preserve/harm vocabulary, operator/case/injection
terms, and private-evaluator terminology.

F1 live use is deliberately small: deterministic fixtures prove contract,
receipt, rejection, cache, and mismatch behavior; one 6-8 case forecast slice
proves that the real Agent consumes the new task input and receipts. Neutral
contract diagnostics need only 2-4 matched cases. F1 does not require utility
improvement and does not promote a Harness edit.

The correct/neutral/shuffled comparison is diagnostic. H0 is not required to
show a large zero-shot task effect before F2 or F3. A weak H0 response may be the
capability that Harness evolution must later improve.

### 8.4 F1 non-goals

F1 does not add classification execution, promote task-conditioned skills,
construct multiple experimental arms, or run a large live corpus.

## 9. F2: Reusable task and evaluator adapters

### 9.1 Functional purpose

F2 makes the current forecast-specific evaluation composition extensible to a
second task without forking the main pipeline.

The first added task is classification because the existing repository history
contains a deterministic classification rig and because forecast and
classification can expose the same legal smoothing choices with different
utility.

Anomaly support remains a later safety adapter unless it is cheaper to restore
than classification. It is not required for the first functional multi-task
slice.

### 9.2 Adapter interfaces

Add small internal protocols rather than task conditionals throughout the
cycle:

    TaskCaseAdapter
      task_type
      build_preparation_request(case)
      build_public_case_view(case)
      private_evaluation_payload(case)

    TaskValuator
      task_type
      evaluate_raw(case)
      evaluate_prepared(case, result)
      summarize_effect(raw, prepared)

The existing forecast corpus and Chronos valuator become the forecast adapter.
The classification adapter may reuse compatible logic from the historical
`grounded_classify`, `rocket_probe`, runner, and tests, but must bind to the
current public/private wall and current contracts.

### 9.3 Minimal classification slice

The first classification slice uses:

- one fixed deterministic classifier family;
- three or four readable base-pattern groups;
- private labels unavailable to the Agent;
- a small matched support/risk corpus;
- the same Program and operator runtime as forecast; and
- at least one preparation whose direction differs between tasks.

The minimal classification execution protocol is frozen as follows:

1. fit one deterministic ROCKET/Ridge classifier on a frozen clean/support
   training split using private labels;
2. freeze the fitted classifier artifact across all preparation candidates;
3. let the Agent prepare corrupt/query series only, one series at a time;
4. evaluate raw, corrupt, and prepared query inputs against the same classifier;
5. keep training and query labels entirely behind the private evaluator wall;
6. split validation/test by base-generator group rather than randomly splitting
   variants from the same template; and
7. record classifier fit split, preprocessing scope, seed, metric direction,
   and artifact SHA in the adapter manifest.

Preparing and retraining the full training set per Program is a later protocol,
not part of the minimal adapter.

The required environment check is functional, not a large statistical study:

    one legal preparation
      -> improves a readable forecast case group
      -> damages a readable classification case group

This proves that the two adapters provide a real task conflict for F3. H0 does
not need to solve that conflict zero-shot. A plumbing smoke may start with one
group, but F2 qualification requires the direction to reproduce on at least two
independent base-pattern groups.

### 9.4 Functional acceptance

F2 is complete when:

1. forecast still runs through its compatibility composition;
2. classification runs through the same TTHA method and Program executor;
3. private labels never enter public Agent input;
4. both adapters produce comparable typed outcome receipts without forcing raw
   metric equality;
5. at least one readable cross-task conflict exists;
6. instrument failures become typed change requests rather than misleading
   Harness edits; and
7. no live task-conditioned promotion occurs before the adapters are readable.

## 10. F3: Shared task-conditioned Harness evolution

### 10.1 Functional purpose

F3 is the first phase that adds the intended task-conditioned learning
capability.

The same H0-derived Harness receives mixed forecast and classification support
cases. It may learn:

- a shared skill when behavior is useful for both tasks;
- a task-scoped applicability rule when the tasks conflict;
- a narrower risk guard when a skill over-triggers;
- improved retrieval/selection/verification behavior; or
- an external change request when the missing component is not Harness-editable.

### 10.2 Reuse current editable surfaces

F3 does not introduce a new task-policy surface. It uses the existing
`SkillEntry.observable_applicability` vocabulary, including `task_kind`, and
the existing one-surface edit contract.

The Agent edits its response to the immutable TaskContext; it never edits the
TaskQualityContract.

Task-related failures preserve the existing first-stage/cause/surface model:

| Observation | Resolution |
| --- | --- |
| TaskContext SHA is absent from a required stage | `TASK_CONTEXT_UNBOUND`: protocol/infra defect; no Harness edit |
| TaskContext is delivered but behavior ignores a real conflict | `TASK_CONTEXT_UNUSED` secondary cause; route the first actionable proposal/skill/selection stage |
| a task-scoped Skill is not retrieved or the wrong task Skill is retrieved | applicability/retrieval misrouting |
| candidate supply is correct but task-specific choice is wrong | `SELECTION_MISS` |
| TaskQualityContract is ambiguous | `TASK_CONTRACT_CLARIFICATION_REQUEST` |
| task judge or operator is missing | `CAPABILITY_RELEASE_REQUEST` |

Identical behavior across tasks is not automatically a Harness fault. The
controller first proves context delivery, cache separation, a real evaluator
conflict, and select-visible evidence.

### 10.3 Functional schedule

Use a small staged curriculum:

    cycle 0
      clear single-task repair and risk cases
      -> at most one promotion

    cycle 1
      new geometry/severity plus the second task
      -> verify shared reuse or learn one task-scoped rule

    cycles 2-3
      matched conflict cases
      -> narrow applicability, improve selection, or emit a change request

At most one Harness edit is promoted per cycle. Instrument and capability
dependencies remain fixed within the run.

### 10.4 Valid no-edit versus invalid cycle

The controller distinguishes:

    SCIENTIFIC_NO_SUPPORTED_EDIT
      valid candidates reached replay but none passed effect/risk gates

    INVALID_PROPOSAL_CYCLE
      schema/AST/protocol failure prevented replay

    INFRASTRUCTURE_INCONCLUSIVE
      API, budget, cache, or evaluator failure invalidated the cycle

Only the first status counts toward a consecutive-no-edit stop rule. Protocol
and infrastructure failures are repaired or retried under a new run receipt;
they are not interpreted as lack of learning.

### 10.5 Functional acceptance

F3 is complete when one shared H* demonstrates all of the following on small
held-out slices:

1. at least one promoted capability is reused beyond its source cases;
2. at least one cross-task shared behavior works for both tasks or one genuine
   conflict is handled by task-scoped applicability;
3. the expected skill is retrieved and changes Program behavior;
4. task-conflicting behavior does not leak into the wrong task beyond the risk
   margin;
5. identity/abstention remains available when evidence is insufficient; and
6. remaining operator/instrument gaps are emitted as typed requests.

This is a functional milestone. It does not require multiple independent
evolution replicas, a six-arm experiment, or publication-level significance.

## 11. F4: Functional qualification and Harness release

### 11.1 Purpose

F4 decides whether the current H* is useful and stable enough to become the
project’s next released Harness baseline.

It is not the place to add new capabilities. F4 freezes the candidate snapshot
and runs a compact qualification suite.

### 11.2 Qualification suite

Use held-out synthetic cases covering:

- both implemented tasks;
- new seeds and geometry;
- clean and genuine-event risks;
- one or two task-conflict groups;
- known operator gaps requiring correct abstention; and
- cost/latency/error accounting.

Primary target scope is the EditManifest's preregistered predicted scope. Cases
where a Skill happened to trigger are a post-treatment mechanism view and do
not replace the primary denominator.

The minimum comparisons are:

- released candidate H* versus H0;
- released candidate H* versus raw/identity; and
- risk behavior before versus after the promoted skills.

A deterministic reference policy may be included when cheap, but the full
task-blind, placebo, oracle, multi-replica, and power-calibrated arm matrix is
deferred until stronger scientific evidence is actually needed.

### 11.3 Release acceptance

Release H* when:

1. it improves a meaningful number of held-out target cases over H0;
2. learned skills are retrieved and execute as intended;
3. clean/genuine risks remain inside the declared functional margin;
4. task-scoped behavior does not systematically cross-trigger;
5. known gaps abstain or produce typed capability requests;
6. the snapshot, dependencies, and lineage are restorable; and
7. runtime cost and failure rate remain acceptable for continued development.

Qualification thresholds are versioned per task/adapter:

    qualification_thresholds:
      forecast:
        material_gain
        harm_margin
        recovery_fraction
      classification:
        material_gain
        harm_margin
        recovery_fraction
      shared:
        cost_ceiling
        failure_ceiling
        cross_trigger_ceiling

Repeatability establishes the evaluator noise floor. Clustered variance informs
uncertainty and detectable effect size. `material_gain` additionally encodes a
practically meaningful improvement, while `harm_margin` encodes task-owner risk
tolerance. Neither is generated from repeatability alone, and raw forecast and
classification metrics are never compared using one shared numeric threshold.

A deterministic policy may run as a non-blocking shadow report so the project
can detect when a simple legal rule still dominates the Agent, without turning
F4 into the later confirmatory arm matrix.

If qualification fails, record the responsible component and return to F2 or
F3. Any Support, corpus, instrument, contract, or runtime component that changes
receives a new version. Do not patch the frozen candidate in place.

### 11.4 Optional later evidence package

If the project later needs a confirmatory scientific claim, F4 can grow an
optional evidence package containing task-blind evolution, deterministic
serving, negative controls, multiple evolution replicas, conflict-sensitive
estimands, and sealed queries.

That package is deliberately outside the current functional critical path.

## 12. F5: Downstream-model selection capability

### 12.1 Purpose

F5 extends the Agent from selecting only a preparation Program to selecting a
bounded downstream model configuration as a second decision.

F5 starts only after F4 releases a useful data-preparation Harness. Otherwise
preparation and model-selection failures become impossible to separate during
development.

### 12.2 Additive architecture

F5 keeps `Program` unchanged and adds separate contracts:

    ModelSpec
    ModelRegistry
    ModelSelectionReceipt
    PreparationPlan
      program
      downstream_model_id
      bounded_configuration

Candidate identity becomes:

    (PreparationKind.IDENTITY | PROGRAM,
     downstream_model_id,
     bounded_configuration_sha)

Identity means raw preparation paired with an explicit ModelID. Pool de-dup,
trace, and effect-equivalence therefore distinguish the same Program on
different models and different Programs on the same model.

The same Agent core gains a versioned model-selection stage or output schema.
The existing preparation result remains usable by callers that do not request
model selection.

Model execution belongs to a bounded adapter/registry layer. Candidate model
scores remain private evaluator information and are not passed to the Agent for
direct argmax.

### 12.3 Functional acceptance

F5 is complete when:

1. legacy Program-only preparation still works;
2. at least two registered downstream models can be executed through one
   bounded interface;
3. the Agent makes an explicit model choice or abstains to the default;
4. model execution produces a typed receipt;
5. model-library and model-execution gaps are distinguishable; and
6. one small held-out slice shows that joint selection can improve over the
   fixed default without breaking preparation safety.

## 13. F6: Natural-data integration

### 13.1 Purpose

F6 applies the released H* through the same serving path on natural time series.
It is an integration milestone, not a new natural-data algorithm.

### 13.2 Natural-data rules

Natural serving input has:

- no injection manifest;
- no clean reference;
- no oracle repair location;
- no private candidate ranking; and
- no assumption that every irregularity is corruption.

The Agent receives only TaskContext, observed values, public features/tools,
retrieved Harness content, and runtime candidate receipts.

Development feedback may use a separately declared natural support set. Final
or production data is not fed back into the same frozen release decision.
Target-local Support adaptations, if enabled, are isolated from the global
Harness lineage. The final natural Query is use-once: opening it freezes Skill,
Memory, threshold, operator, and model updates for that release decision.

Natural evaluation does not fabricate synthetic D/G/NRR or oracle-location
fields. It reports downstream utility, do-no-harm, abstention, modification
fraction, structural preservation, latency/cost, and failure behavior.

### 13.3 Functional acceptance

F6 is complete when:

1. one natural dataset enters through the normal `PreparationRequest` path;
2. no synthetic-only fields are required by the fast Agent;
3. the released H* executes or abstains without a special natural-data Agent;
4. modifications, latency, failures, and downstream utility are recorded;
5. obvious clean/genuine behavior is not systematically damaged; and
6. natural failures route to Harness, instrument, or capability requests.

## 14. Capability ledger

The project needs a readable view of what the Harness has actually learned.

Add a derived `CapabilityLedger` with entries such as:

    capability_id
    source_pattern_id
    created_by_edit_id
    first_supported_cycle
    observable_applicability
    task_scope
    predicted_behavior
    held_out_reuse_count
    positive_reuse_count
    false_trigger_count
    risk_regression_count
    superseded_by
    current_status

The ledger is computed from snapshot lineage, EditManifest, paired replay, and
DecisionTrace. It is not a second manually editable truth source and is not an
Agent memory channel.

Its functional purpose is to answer:

- what new capability was created;
- whether it was reused;
- whether its scope was narrowed or generalized;
- whether it causes false triggers; and
- which unresolved gaps block the next feature.

## 15. Artifact retention

Every paid or decision-bearing run must be reproducible enough to resume,
inspect, and replay.

### 15.1 Tracked manifest

Git tracks a redacted run manifest containing:

- run ID and purpose;
- code commit;
- rules/corpus/TaskContext/Harness/runtime/evaluator SHAs;
- requested and returned model identity;
- aggregate protocol and effect summary;
- artifact bundle SHA and location;
- cache bundle SHA when replay depends on it;
- secret-scan status; and
- restore-test receipt.

### 15.2 Private raw bundle

Raw responses, private case outcomes, prompts, caches, and replay inputs may
remain outside ordinary Git when they are large or sensitive. They must be
stored as immutable private artifacts, Git LFS objects, or encrypted archives
with at least one independent backup.

Untracked local files are not a durable scientific or engineering record.
Conversely, sensitive raw caches must not be committed blindly merely to obtain
version control.

### 15.3 Restore check

Before deleting or moving a run, a clean worktree must be able to:

1. resolve the tracked manifest;
2. retrieve or locate the raw/cache bundle;
3. verify its SHA;
4. reconstruct the frozen inputs and dependencies; and
5. run the supported offline/cache replay path.

## 16. Verification philosophy

This project prioritizes working functionality and measurable capability growth.
Verification protects load-bearing boundaries; it is not an end product.

For each phase, add only:

- focused contract tests for new schemas or ownership rules;
- one integration slice proving the new capability through the real path;
- regression checks for M0 compatibility and information walls;
- replay or calibration evidence for behavior-changing updates; and
- a small held-out functional check.

Defer exhaustive arm matrices, broad benchmark suites, low-risk edge-case
combinatorics, and high-powered statistics until the feature path works and the
project actually needs that level of evidence.

Information walls, immutable task goals, exact surface ownership, replay before
promotion, and risk checks are not optional research ceremony. They prevent the
framework from learning the wrong capability or corrupting its own objective.

## 17. Stage gates and active order

| Stage | Entry gate | Functional output | Next-stage gate |
| --- | --- | --- | --- |
| M0-R | completed | tagged H2 release and restorable evidence | `m0-agent-harness-v0.1.0` |
| F1 | tagged M0 | TaskContext and candidate receipts in existing path | `f1-integration-receipt/1` pass |
| F2a | F1 path stable | calibrated forecast evidence instruments | level/missing calibration pass |
| F2b | F2a instrument stable | forecast/classification adapters | readable cross-task conflict |
| F3 | adapters readable | one shared task-conditioned H* | held-out reuse and bounded risk |
| F4 | candidate H* frozen | released Harness snapshot | functional qualification pass |
| F5 | preparation release useful | bounded model selection | joint path works without regressions |
| F6 | stable serving path | natural-data integration | natural functional acceptance |

The active implementation order is strict:

1. preserve the completed M0-R tag and immutable H2 release;
2. preserve the closed F1 TaskContext/receipt contracts and evidence;
3. calibrate the public forecast instruments in F2a without editing H2;
4. build the smallest forecast/classification adapter slice in F2b;
5. evolve one shared H* in F3;
6. qualify that H* in F4;
7. add model selection in F5; and
8. integrate natural data in F6.

Planning or code archaeology for later phases may occur in parallel. Later-phase
code must not be merged into the active path before its entry gate passes.

## 18. Explicitly deferred

The following are not on the current functional critical path:

- six-arm confirmatory evaluation;
- multiple powered evolution replicas;
- a task-blind primary comparison arm;
- placebo/sham prompt controls;
- publication-level causal mediation claims;
- a large eight-pattern factorial corpus;
- full anomaly-task implementation when classification is sufficient;
- Harness Tree, autonomous branch merging, or Pareto routing;
- learned embedding retrieval;
- unrestricted runtime/code edits by the slow Agent;
- joint evaluator/model/Harness co-evolution;
- model weight training inside the Harness loop; and
- online adaptation on final natural data.

These may be added after the main framework produces useful, reusable
task-conditioned capabilities.

## 19. Functional acceptance checklist

Before calling the post-M0 framework complete, verify:

1. the frozen M0 path is still reproducible;
2. a live M0 edit has reached replay, promotion, and held-out reuse;
3. TaskQualityContract is immutable to both Agent roles;
4. TaskContext is bound to every F1+ Agent stage and cache key;
5. existing `PreparationRequest.task_spec` callers remain compatible;
6. candidate verification is runtime-generated and visible at selection;
7. forecast and classification use the same Agent, Program, executor, and
   Harness types;
8. task-specific evaluation lives behind adapters rather than task branches;
9. one shared H* contains reusable shared or task-scoped capability;
10. slow-Agent edits remain limited to declared Harness surfaces;
11. instrument and capability gaps produce versioned requests;
12. dependency-changing updates occur only between epochs;
13. CapabilityLedger is derived from primary receipts and lineage;
14. a qualified Harness release improves useful held-out behavior without
    material risk regression;
15. F5 model selection is additive and Program-only callers still work;
16. F6 natural data uses the same serving method path; and
17. every paid or decision-bearing run has a restorable artifact manifest.

## 20. Final functional success statement

This development program succeeds when the current TTHA framework has grown
from M0 into one reusable system that can:

1. receive an explicit downstream TaskContext;
2. inspect a time series through deployment-visible evidence;
3. generate, verify, and select a bounded preparation Program or identity;
4. evaluate failures and route them to the correct project component;
5. improve its Harness through supported one-surface edits;
6. accumulate shared and task-scoped capabilities in one snapshot lineage;
7. release a useful H* after compact held-out qualification;
8. optionally select a downstream model through a separate bounded layer; and
9. run the same serving path on natural data.

The result is not a new branch beside mini_pipeline. It is the main framework
grown from the mini_pipeline’s feedback kernel through additive, versioned
capabilities.

## 21. Normative references

This design is read with:

- [Agent-Centric Minipipe M0 Design](2026-07-17-agent-centric-minipipe-m0-design.md);
- [Architecture Convergence Design](2026-07-17-architecture-convergence-design.md);
- `contracts/task.py` as the current canonical TaskSpec source;
- `contracts/method.py` as the public preparation request/result boundary;
- `methods/ttha/` as the single Agent and Harness implementation;
- `runtime/` as the canonical candidate and execution runtime;
- `operators/registry.py` as the operator/task-legality source; and
- `evaluation/minipipe/` as the feedback, replay, and functional-evolution
  kernel.

When this document conflicts with the frozen M0 behavior, the M0 tag wins for
the M0 compatibility path. Post-M0 changes require an explicit contract or
dependency version and must preserve the architectural invariants above.
