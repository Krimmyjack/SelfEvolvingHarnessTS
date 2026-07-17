# Agent-Centric Minipipe M0 Design

**Status:** approved initial design

**Date:** 2026-07-17

**Scope:** TTHA Agent fast path, minipipe diagnosis, and the first closed Harness-evolution loop

## 1. Decision and relationship to the convergence design

The target system is Agent-centric. TTHA is the only active method and the same
Agent core participates in both the fast and slow paths. `H_ref` is not an Agent
capability and is not part of the target runtime, minipipe, method registry, or
evaluation composition root.

Historical reproducibility for `H_ref` is preserved through a Git tag, frozen
artifacts, environment metadata, and archived reports. It must not remain as an
importable active method. This decision supersedes the parts of
`2026-07-17-architecture-convergence-design.md` that proposed an active
`methods/h_ref_v02/` package or H_ref/TTHA dual-method integration tests.

This document deliberately specifies a small M0. It is the first executable
feedback loop, not the final self-evolution architecture.

## 2. Goals and non-goals

M0 must demonstrate this causal loop:

```text
synthetic case
  -> Agent fast path
  -> mechanically recorded behavior
  -> private outcome and intervention evaluation
  -> first actionable fault
  -> sanitized recurring failure pattern
  -> one-surface Harness edit contract
  -> paired replay
  -> supported/rejected/scoped edit plus lineage
```

M0 goals are:

1. make the Agent, rather than a frozen reference harness, the method under test;
2. distinguish data mechanisms from Agent/Harness failure stages;
3. diagnose program supply and program selection separately;
4. route feedback to an editable Harness surface without leaking oracle data;
5. require an intervention receipt before confirming surface responsibility;
6. verify predicted behavior change before attributing an outcome improvement;
7. retain scoped improvements without forcing every edit into a global prompt.

M0 does not implement:

- a full Harness Tree, Pareto router, or autonomous branch merging;
- arbitrary slow-path edits to Python/runtime code;
- direct candidate-value access by the fast Agent;
- downstream-model search, although candidate contracts reserve a model field;
- full retraining of downstream models;
- H_ref execution or active H_ref compatibility layers.

## 3. System roles and dependency direction

### 3.1 TTHA Agent

TTHA is the sole active method. It owns the deployable `HarnessSnapshot`, including
instructions, skills, retrieval/applicability rules, memories, proposal controls,
verification policies, and permitted scalar configuration.

The same Agent core serves two roles:

- **fast path:** observes a case, retrieves skills, uses tools, supplies candidate
  programs, selects a candidate, and executes or abstains;
- **slow path:** consumes sanitized recurring-failure evidence and proposes a
  minimal edit to one declared Harness surface.

They share a model family and Harness state but use different role prompts,
inputs, tools, and permissions. The fast path cannot edit the Harness. The slow
path cannot read case-private oracle records or directly promote its own edit.

### 3.2 Minipipe

`evaluation/minipipe/` is an offline development and diagnostic environment. It
owns synthetic clean references, injection manifests, frozen evaluator calls,
candidate replay, fixed intervention probes, fault attribution, clustering,
risk-set construction, and paired replay.

Minipipe may call TTHA through canonical method contracts. TTHA must never import
minipipe.

### 3.3 Harness Evolution Controller

A deterministic controller owns immutable snapshots, validates edit manifests,
checks surface ownership and preconditions, runs paired replay, derives verdicts,
promotes at most one edit per cycle, and writes lineage. The slow Agent proposes;
the controller applies.

Dependency direction is therefore:

```text
CLI -> evaluation/minipipe -> canonical Method -> methods/ttha -> runtime/operators
                     |
                     +-> grader/controller/lineage
```

## 4. Fast-path behavior evidence

### 4.1 Do not ask the Agent to certify itself

The Agent does not emit trusted `PASS` or `FAIL` receipts. An Agent can truthfully
report that it made a selection without knowing that a better candidate existed.
M0 separates mechanically recorded actions from grader-derived judgments.

### 4.2 DecisionTrace

`DecisionTrace` contains facts recorded by the runtime, retriever, tool gateway,
compiler, and executor. It is not hidden chain-of-thought. At minimum it records:

- public observation identifiers and inspected regions;
- tool calls and public tool results;
- retrieved skill/memory identifiers and applicability matches;
- supplied candidates and their public program/model metadata;
- `chosen_candidate_id` or an explicit abstention;
- compiled canonical program and compilation result;
- execution result, modified indices/regions, and verification actions.

Every case must expose a candidate pool with stable identifiers. A one-candidate
run cannot distinguish a supply failure from a selection failure. M0 therefore
uses a small configurable pool with at least two candidates; three is the default
starting configuration. A future candidate is `(Program, DownstreamModel)`; in M0
the model field is fixed to the configured default.

### 4.3 StageAssessment

The grader derives a `StageAssessment` with:

```text
stage
status: PASS | FAIL | UNKNOWN | NOT_APPLICABLE
evidence_refs
decision_rule_id
suspect_surfaces
```

`UNKNOWN` is a valid scientific result. It prevents weak evidence from being
turned into a false causal label.

The ordered stages are:

1. case/critic eligibility;
2. observation;
3. localization, when local evidence exists;
4. mechanism/evidence discrimination, when identifiable;
5. retrieval and policy routing;
6. candidate supply;
7. candidate selection;
8. compilation;
9. execution;
10. outcome and risk control.

The first failed, actionable stage becomes `first_actionable_fault`. Eligibility
failures such as `NON_ACTIONABLE_CASE` and `CRITIC_BLIND` are pre-Agent outcomes
and route to corpus/evaluator instrumentation, not to an Agent skill edit.

For selection, the grader privately evaluates the frozen candidate pool. If an
acceptable candidate was present but not chosen, the fault is `SELECTION_MISS`.
If no acceptable candidate was supplied, it is `CANDIDATE_SUPPLY_GAP`. Thus M0 can
diagnose a wrong choice even when the Agent is selecting rather than reasoning.

`UPDATE_MISATTRIBUTION` is not inferred from correlation. It is written only when
a single-surface replay shows that changing the suspected surface did not produce
the predicted behavior.

## 5. CaseFeedback: four views

The grader writes one private `CaseFeedback` per case.

### 5.1 Outcome view

Use one sign convention throughout:

```text
U = -J                         # higher is better
D = U(clean) - U(corrupt)     # corruption damage
G = U(prepared) - U(corrupt)  # repair gain
NRR = G / D                   # only when D exceeds a noise threshold
```

The outcome view also records target-window gain, outside-window change,
clean/contextual-counterpart change, non-target collateral, and candidate regret:

```text
selection_regret = max_c U(candidate_c) - U(chosen)
```

`OverRestoration = U(prepared) - U(clean) > 0` is a judge-pleasing warning, not by
itself proof of harm. NRR is null when `D` is zero, negative, or below the declared
measurement-noise threshold.

### 5.2 Mechanism view

Mechanism evidence includes block attribution, localization receipts, observable
statistics, optional context-alignment signals, and fixed operator-response
curves. It does not assert an injection label in deployable feedback.

Each probe exposes a unified monotonic intervention strength:

```text
beta in {0.25, 0.50, 0.75}
R_operator(beta) = U(probe(operator, beta)) - U(corrupt)
```

Larger `beta` always means a more aggressive intervention. `ProbeSpec` versions
the mapping from beta to operator-native parameters. Beta makes response shapes
comparable; it does not claim equal physical dose across different operators.
All comparisons use common random numbers and the same evaluator configuration.

The M0 panel contains imputation, clipping, denoising, level correction, and
period repair. This panel is a fixed instrument, not an arbitrary J-query API.

### 5.3 Behavior view

This view is derived from `DecisionTrace`: whether relevant regions were inspected,
which discriminative tools ran, which skills were retrieved, what candidates were
supplied, what was chosen, whether the compiled program matched the proposal, and
whether the execution scope exceeded the selected region.

### 5.4 Update-attribution view

Before replay, this view may contain `suspect_surfaces` only. A
`confirmed_surface` can be written only after single-surface paired replay. The
intervention receipt, not an explanatory narrative, signs the attribution.

## 6. Two independent walls

### 6.1 Oracle information wall

Clean series, injection type/location, clean-derived measurements, oracle labels,
and exact success rankings are private to the grader. They never enter fast-path
requests, deployable Harness artifacts, public failure cards, or edit
applicability predicates.

### 6.2 Judge access wall

Natural computability is not sufficient for Agent access. Exact `J(corrupt)`,
per-candidate J values, and arbitrary candidate comparisons remain grader-private
even though they could be computed on natural data. Otherwise selection regret
would collapse by construction and the selector would be optimized to please the
judge, including through deletion and over-smoothing.

The fast Agent's only M0 access to evaluator evidence is the declared fixed
`ProbeAPI`. Candidate-J-assisted selection is reserved for an explicit M1 A/B arm
with risk monitoring; it is not a default capability.

### 6.3 Mechanical enforcement

Artifacts are physically separated:

```text
evaluation/minipipe/artifacts/
  private/   # CaseFeedback, oracle labels, candidate values, full replay details
  public/    # PublicCaseView, sanitized pattern cards, safe edit inputs
```

Prompt constructors accept typed public objects and may read only `public/`.
Private records are converted by a deterministic sanitizer into typed public
records. `FailurePatternCard` prose and `EditManifest.observable_applicability`
cannot accept arbitrary predicates: applicability is a validated AST over a
closed vocabulary of deployment-observable features.

Oracle labels may help the grader measure cluster purity or discover a cluster,
but public descriptions and routing conditions must be expressed only through
observable signatures.

## 7. Failure mining

M0 uses deterministic attribution followed by constrained discovery:

1. find the first actionable fault in the fixed stage order;
2. bucket cases by fault, suspect surface, and observable signature;
3. cluster numerical behavior within a bucket when needed;
4. attach a matched success contrast;
5. let the slow Agent name and summarize the mechanism, without changing the
   coarse stage label.

A public `FailurePatternCard` contains:

- pattern ID and support count;
- observable signature and contexts;
- first actionable fault;
- common normalized behavior signature;
- sanitized intervention receipt;
- matched successful cases and counterexamples;
- suspect surface(s);
- whether capability appears missing or misrouted.

The grader separately records oracle mechanism purity, best-intervention purity,
and target-surface purity. Low mechanism purity is an instrumentation signal, but
does not automatically imply an observation gap when the same intervention works
consistently across mechanisms.

## 8. Harness surfaces and edit contract

### 8.1 Editable surfaces

M0 allows one edit to one declared textual, structured-rule, or scalar-config
surface, such as:

- instruction sections;
- one skill body;
- one skill applicability predicate;
- retrieval thresholds or `top_k`;
- proposal/candidate controls;
- risk, abstention, or verification rules;
- scoped memory entries.

Python runtime, compiler implementation, arbitrary tool code, and operator code
are read-only to the slow Agent in M0.

`harness_surfaces.json` maps semantic surfaces to owned file fields or JSON
pointers. Each editable region has exactly one owner. Shared compiler/runtime
regions are declared read-only. A workspace diff must resolve to exactly one
surface before replay is accepted.

### 8.2 EditManifest

Each slow-path proposal is a falsifiable contract:

```text
edit_id
base_harness_sha
target_pattern_id
target_surface_id
surface_precondition_sha
dependency_precondition_shas
minimal_patch
observable_applicability
predicted_agent_behavior_change
predicted_data_effect
automatically_selected_risk_cases
falsification_condition
```

At application time the controller rechecks the base snapshot, target surface,
and declared dependency hashes. A stale edit is rebased and replayed; it is never
silently applied using an old receipt.

The automatic risk set contains:

1. clean cases with the same observable appearance;
2. genuine-event cases with similar signatures;
3. adjacent-severity cases;
4. cases the current Harness already handles correctly;
5. cases opposite to the edit's action direction.

## 9. Paired replay and verdicts

Replay compares `Agent(H_t)` with
`Agent(H_t + single_surface_edit)` under common seeds and the same case inputs.

Stage A runs the target cluster and automatic risk set. Only an edit that passes
Stage A advances to Stage B, the full M0 core regression suite. The controller
checks, in order:

1. whether the predicted normalized `BehaviorSignature` occurred;
2. whether target outcomes improved across a required majority, not merely in the
   mean due to one outlier;
3. whether risk cases remained stable;
4. whether observable applicability limited the behavior change to its scope.

Out-of-scope deterministic cases require no semantic behavior diff. Stochastic
cases use common seeds and declared repeat/tolerance rules. In-scope clean or
genuine-event cases may trigger the edit, but require `delta U >= -epsilon`.

The report stores orthogonal facts before deriving a label:

```text
evaluation_status
prediction_verified
behavior_change_status
target_outcome_status
risk_status
scope_status
```

Derived labels are:

- `DEAD_EDIT`: the surface changed but the predicted behavior did not;
- `BEHAVIOR_CHANGED_NO_GAIN`: predicted behavior changed, outcome did not improve;
- `TARGET_RECOVERED_WITH_HARM`: target improved but a risk case regressed;
- `PARTIAL_RECOVERY`: direction is supported but coverage/strength is incomplete;
- `SUPPORTED_EDIT`: predicted behavior occurred, target improved, and risk/scope
  checks passed;
- `UNEXPECTED_GAIN`: outcome improved through behavior not predicted by the
  contract; retain the evidence but do not promote until the explanation is
  rewritten and replayed;
- `INCONCLUSIVE`: judge or infrastructure failure prevents a scientific verdict.

Malformed Agent output, invalid programs, and compilation errors are Agent
behavior failures, not `INCONCLUSIVE`. A true infrastructure failure is retried
once with the identical request; if it repeats, the edit remains pending and the
incident enters the infrastructure backlog.

If Stage B finds a regression missing from the automatic risk set, the edit is
marked harmful and the risk-set generator receives a separate instrumentation
defect. If individually supported edits regress only after composition, record
`EDIT_INTERACTION_REGRESSION`, not `UPDATE_MISATTRIBUTION`.

## 10. M0 versioning and cycle limits

M0 maintains:

- one active immutable `HarnessSnapshot`;
- supported scoped candidates;
- rejected and inconclusive edits;
- parent snapshot, manifest, and replay receipt for every transition.

At most three edits enter replay per cycle, prioritized by support, impact,
attribution confidence, and actionability. At most one edit is promoted per
cycle. Before promotion its preconditions are rechecked. After sequential
promotion, the final snapshot runs Stage B again to detect composition effects.

An edit that helps one regime and harms another remains a `scoped_candidate` with
an observable applicability contract. It is not forced into the global
instruction. M1 may consume these candidates in a Pareto frontier or branch
router.

## 11. Project placement

M0 extends the convergence architecture rather than creating another operator or
executor stack:

```text
src/selfevolving/
  contracts/                 # candidate, trace, assessment, snapshot/edit types
  runtime/                   # canonical execution and mechanical trace recording
  operators/                 # the single operator registry
  methods/ttha/
    method.py                # canonical Method implementation
    fast_agent.py
    slow_agent.py
    harness/
    harness_surfaces.json
  evaluation/minipipe/
    corpus/                   # synthetic and matched cases
    valuation/                # private U/J and outcome views
    probes/                   # fixed ProbeAPI and beta mappings
    feedback/                 # assessments, sanitizer, patterns
    replay/                   # controller, risk sets, verdicts, lineage
    schemas/
  cli/                       # composition root and cycle command
```

Exact filenames may follow the refactor's final package names, but ownership and
dependency direction are invariant. Minipipe must use the canonical operator
registry and executor; it may not copy either.

## 12. One M0 cycle

1. Build a small controlled corpus spanning clean/genuine events and the initial
   missing, impulsive/outlier, level-shift, and period-change families.
2. Construct typed `PublicCaseView` objects and run the Agent fast path.
3. Record `DecisionTrace` mechanically.
4. Privately evaluate corruption, selected output, candidate pool, collateral,
   and fixed probe curves.
5. Derive stage assessments and the first actionable fault.
6. Produce private `CaseFeedback` and sanitized public pattern cards.
7. Select recurring high-impact patterns and request at most three one-surface
   edit manifests.
8. Run Stage A and, when eligible, Stage B paired replay.
9. Promote at most one supported edit or retain/reject it with a receipt.
10. Append snapshot lineage and run the next cycle from the resulting active
    snapshot.

Primary artifacts remain small:

```text
private/case_feedback.jsonl
public/failure_patterns.json
public/failure_patterns.md
public/edit_manifest.json
private/paired_replay_report.json
harness_lineage.jsonl
```

## 13. Minimum acceptance criteria

M0 is complete when one command can run at least two consecutive cycles and:

1. the active method registry and composition root contain TTHA but no H_ref;
2. the same TTHA Agent core executes both permission-separated roles;
3. the public prompt path cannot import/read private artifacts;
4. candidate J values never appear in fast-path inputs or public traces;
5. constructed fixtures distinguish candidate-supply and candidate-selection
   failures;
6. probe beta mappings are monotonic, versioned, and replayed under common seeds;
7. every accepted diff maps to exactly one owned Harness surface;
8. stale surface/dependency preconditions force replay;
9. paired fixtures exercise at least dead, supported, harmful, unexpected-gain,
   and infrastructure-inconclusive outcomes;
10. every promoted snapshot has a parent, manifest, paired-replay receipt, and
    final core-regression result;
11. repeated fixed-seed runs reproduce normalized behavior signatures and
    scientific verdicts within declared tolerances.

These criteria establish the causal feedback skeleton. More sophisticated
clustering, learned model routing, candidate-J A/B experiments, Pareto selection,
and Harness branching remain M1 work inside the same boundaries.
