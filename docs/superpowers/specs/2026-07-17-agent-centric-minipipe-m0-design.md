# Agent-Centric Minipipe M0 Design

**Status:** frozen M0 design after design and worktree audit

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

#### 3.1.1 H0 is procedurally complete and domain-naive

H0 is TTHA's first formal snapshot. It is not derived from H_ref. It must be able
to complete the Agent protocol while deliberately containing no defect-family to
program mapping.

H0 contains:

```text
Core Instruction
  - public/deployment-observable evidence only
  - no oracle, clean reference, or candidate J access
  - canonical operator registry only
  - typed candidate selection and DecisionTrace protocol

Bootstrap procedural skills, always injected
  - inspect_and_localize
  - build_contrastive_candidates
  - select_or_identity_and_verify

Capability skill library: empty
Memory: empty
Retrieval
  - bootstrap procedures bypass retrieval
  - capability skills use observable applicability
Candidate Policy
  - total K defaults to 3
  - one runtime-owned identity slot
  - at most two Agent PROGRAM slots
Generic Verification/Risk
  - identity cannot be filtered
  - task/operator compatibility
  - no future information
  - scope and destructive-action limits
  - no silent fallback
```

Bootstrap procedures describe how to inspect, construct, select, and verify.
They must not encode mappings such as `missing -> impute_linear` or
`outlier -> winsorize`. Domain-naive means that the Harness has no learned
family policy; the Agent can still see the canonical operator names and their
deployment contracts.

The skill schema distinguishes `bootstrap_procedure`, `capability`, and
`safety`. The fault router, not the slow Agent, controls which kind a fault may
edit. Protocol and observation-procedure causes may target bootstrap procedure;
library/content causes may target capability; risk causes may target safety or
verification. Capability knowledge cannot be placed in an always-injected
bootstrap surface.

Ownership is non-overlapping. The Core Instruction owns walls, role permissions,
and output protocol. Bootstrap skills own general procedures. Candidate Policy
owns identity and slot counts. Deterministic validators own compatibility and
risk invariants. Prompt text that exposes policy values is compiled from these
sources rather than repeated manually in instruction prose.

H0 authoring inputs are explicit and checked in:

```text
methods/ttha/harness/h0/
  instruction.md
  skills/
    bootstrap/                 # exactly three procedural SkillEntries
    learned/                   # no SkillEntry files
  memories.jsonl               # empty
  retrieval.json
  candidate_policy.json
  verification.json
  snapshot.lock.json
```

The compiler must fail at startup when the resolved H0 content or any locked
dependency differs from `snapshot.lock.json`. H0 is immutable after freeze;
subsequent H_t snapshots are controller-created copies, not edits to H0.

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
- the required `chosen_candidate_id`;
- compiled canonical program and compilation result;
- execution result, modified indices/regions, verification actions, and whether a
  PROGRAM was effect-equivalent to identity.

Every case exposes a tagged candidate pool with stable identifiers:

```text
CandidateKind.IDENTITY
  candidate_id = "identity"
  program = null
  source = "runtime"

CandidateKind.PROGRAM
  program = a non-empty canonical Program
```

The runtime injects exactly one identity candidate. It cannot be deleted by the
RiskGate and counts against the total candidate budget. M0 defaults to `K=3`:
identity plus at most two Agent-supplied PROGRAM candidates. K is a versioned
configuration default, not a permanent law; M1 may change it after inspecting
realized-pool and regret distributions.

The Agent must choose one candidate. Choosing identity maps to
`PreparationStatus.ABSTAINED`, an unchanged prepared series, and `program=null`,
while retaining `chosen_candidate_id="identity"` in the trace. Failure to provide
a choice is a protocol failure, not implicit abstention.

For deterministic execution, a PROGRAM is `effect_equivalent_to_identity` when
its prepared array and the raw input are equal in shape, dtype, and bytes. A
component may declare a numerical tolerance only when bit equality is not a valid
contract; that tolerance and reason are versioned. Effect-equivalent PROGRAMs
remain in the proposal trace but do not count as distinct candidate supply.

The grader sets `U(identity)=U(corrupt)` and includes identity in selection
regret. This measures both over-repair and over-abstention. A future candidate is
`(Program, DownstreamModel)`; in M0 the model field is fixed to the configured
default.

### 4.3 StageAssessment

The grader derives a `StageAssessment` with:

```text
stage
status: PASS | FAIL | UNKNOWN | NOT_APPLICABLE
evidence_refs
decision_rule_id
suspect_surface_templates
```

The ordered assessments are then folded into one case-level `FaultAttribution`:

```text
first_stage
fault_code
cause_code
actionability
suspect_surface_templates
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

Harness component causes are not execution stages. In particular,
`SKILL_LIBRARY_GAP` is a cause of `CANDIDATE_SUPPLY_GAP`, not a new stage. Before
replay, attribution names a surface template rather than inventing an edit ID:

```json
{
  "first_stage": "CANDIDATE_SUPPLY",
  "fault_code": "CANDIDATE_SUPPLY_GAP",
  "cause_code": "SKILL_LIBRARY_GAP",
  "suspect_surface_template": "skill_library.entries/{skill_id}",
  "actionability": "EDITABLE_M0"
}
```

The common supply and selection routes are:

| Evidence | Cause | Route |
| --- | --- | --- |
| deployment-observable program witness succeeds and no capability skill exists | `SKILL_LIBRARY_GAP` | atomic `ADD SkillEntry` |
| an existing skill works when forced but normal retrieval misses it | `RETRIEVAL_MISS` | applicability/retrieval |
| a retrieved skill still produces no effective candidate | `SKILL_CONTENT_GAP` or `PROPOSAL_CONTROL_GAP` | skill body or proposal control |
| an effective candidate is present but unchosen | `SELECTION_MISS` | selection control |
| the required transformation class is proven absent | `OPERATOR_GAP` | non-editable M0 backlog |
| expressibility is neither proven nor disproven | `EXPRESSIBILITY_UNKNOWN` | more evidence, no Harness edit |

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

Agent decision quality and system capability are separate ledgers. On a proven
`OPERATOR_GAP` case, choosing identity can be an Agent-side success even though
the system cannot repair the defect:

```text
agent_decision_status = CORRECT_IDENTITY
system_capability_status = OPERATOR_GAP
```

Such a case enters the operator backlog but not the Agent-failure denominator.
Otherwise feedback pressure would reward unsafe attempts to repair an unavailable
capability.

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

The repair-response panel contains imputation, clipping, denoising, and level
correction at three beta values. Period change has no repair-response curve in
M0 because the canonical registry has no period/frequency/phase correction
operator. It instead exposes a deployment-observable `period_change_diagnostic`
containing pre/post period estimates, a change score, spectral/ACF consistency,
and `repair_available=false`. It is presented only as a diagnostic.
The panel is a fixed instrument, not an arbitrary J-query API.

Expressibility uses two distinct witnesses:

- an **oracle-parameterized witness** may use private injection location or other
  oracle parameters; success proves only that an intervention direction can
  produce value;
- an **observable-parameterized witness** derives every location and parameter
  from deployment-visible features; only its success proves that an Agent can
  construct the program in deployment conditions.

`SKILL_LIBRARY_GAP` requires a successful observable witness. Oracle success with
an explicitly demonstrated failure to derive the required public parameterization
is `OBSERVABLE_PARAMETERIZATION_GAP` and routes to observation/feature extraction,
not to a skill. Merely failing to find an observable witness remains
`EXPRESSIBILITY_UNKNOWN`; search failure is not proof of impossibility.

Expressibility status is three-valued:

```text
PROVEN_EXPRESSIBLE
PROVEN_UNAVAILABLE
EXPRESSIBILITY_UNKNOWN
```

A finite fixed-program arm can provide a positive witness but cannot prove
unavailability. `PROVEN_UNAVAILABLE` requires either a versioned complete
transformation-category contract showing that the required class is absent, or
an exhaustive search over a declared bounded grammar. M0 adds a versioned
mechanism-family to required-transformation-class map and declares the category
enumeration complete. This makes the missing period-correction class auditable
without treating an unsuccessful heuristic search as proof.

### 5.3 Behavior view

This view is derived from `DecisionTrace`: whether relevant regions were inspected,
which discriminative tools ran, which skills were retrieved, what candidates were
supplied, what was chosen, whether the compiled program matched the proposal, and
whether the execution scope exceeded the selected region.

### 5.4 Update-attribution view

Before replay, this view may contain `suspect_surface_templates` only. A
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

### 6.4 Effective-request LLM cache

LLM replay identity is based on the resolved request the model can actually
observe, not on the full source Harness snapshot:

```text
cache namespace =
  (case_id, role, stage, call_index, replicate_id, semantic_request_hash)
```

`semantic_request_hash` covers:

- provider, requested model identity, reported revision/fingerprint when
  available, decoding parameters, and provider seed;
- exact resolved system/user/tool messages;
- tool and output schemas;
- `public_case_view_sha` and every versioned tool/data context reachable through
  the request;
- `effective_harness_view_sha`, covering only instruction sections, retrieved
  skills/memories, and controls consumed by that call;
- cache schema version.

The cache record separately stores `source_harness_snapshot_sha`, raw response,
parsed response, parse status, response hash, and provider metadata as
provenance. Successful transport responses are immutable even when parsing fails,
because malformed output is Agent behavior. Transient transport/provider errors
are not cached.

Identical effective requests replay deterministically. A scoped edit that changes
the full snapshot but leaves an out-of-scope case's effective view and tool
context unchanged reuses the baseline response. A prompt-changing edit can share
a provider seed where supported, but cache reuse does not make different prompts
strict common-random-number draws.

For out-of-scope paired replay, the baseline arm is materialized first. The edit
arm must have the same `effective_harness_view_sha` and reuse every eligible LLM
call from the baseline cache. A new skill that displaces an old skill through
`top_k` therefore fails scope even when its own applicability predicate says the
case is out of scope.

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
- suspect surface template(s), never a confirmed surface;
- whether capability appears missing or misrouted.

The grader separately records oracle mechanism purity, best-intervention purity,
and target-surface purity. Low mechanism purity is an instrumentation signal, but
does not automatically imply an observation gap when the same intervention works
consistently across mechanisms.

The grader owns a frozen positive-witness arm:

```text
evaluation/minipipe/baselines/fixed_program_baseline_v1.json
```

It is not an active method and is never imported by TTHA. Its programs do not
enter Agent prompts, H0, the Agent candidate pool, or Agent-visible selection
regret. A successful deployment-observable program supplies positive
expressibility evidence. Failure of this finite arm supplies no negative proof.
Oracle-parameterized programs are recorded separately and cannot sign
`SKILL_LIBRARY_GAP`.

The fault router contains a versioned, executable cause-to-target table. M0's
minimum mapping is:

| Cause code | Allowed target class |
| --- | --- |
| `PROTOCOL_GAP`, `OBSERVATION_PROCEDURE_GAP`, `LOCALIZATION_PROCEDURE_GAP` | `bootstrap_procedure` |
| `SKILL_LIBRARY_GAP`, `SKILL_CONTENT_GAP` | `capability` |
| `RISK_GAP` | `safety` or deterministic verification |
| `RETRIEVAL_MISS` | retrieval/applicability |
| `PROPOSAL_CONTROL_GAP` | proposal control |
| `SELECTION_MISS` | selection control |
| `OBSERVABLE_PARAMETERIZATION_GAP` | observation feature/tool surface |
| `OPERATOR_GAP` | no M0 edit; operator capability backlog |
| `EXPRESSIBILITY_UNKNOWN` | no edit; evidence backlog |

An EditManifest whose surface class is not allowed for its confirmed cause is
rejected before replay. This mechanically prevents scoped family knowledge from
being hidden in an always-injected bootstrap procedure.

## 8. Harness surfaces and edit contract

### 8.1 Editable surfaces

M0 allows one edit to one declared textual, structured-rule, or scalar-config
surface, such as:

- instruction sections;
- one existing skill body, applicability predicate, or risk guard;
- one atomic new capability `SkillEntry`;
- retrieval thresholds or `top_k`;
- proposal/candidate controls;
- risk, abstention, or verification rules;
- scoped memory entries.

Python runtime, compiler implementation, arbitrary tool code, and operator code
are read-only to the slow Agent in M0.

`harness_surfaces.json` maps semantic surfaces to fields or JSON pointers relative
to an immutable snapshot root. Each editable region has exactly one owner.
Shared compiler/runtime regions are read-only. A snapshot source diff must resolve
to exactly one surface before replay is accepted.

A learned capability is one structured object and its existence is membership;
there is no separately editable membership list:

```text
SkillEntry
  schema_version: skill-entry/1
  skill_id
  skill_kind: capability
  revision
  body
  observable_applicability
  allowed_tools
  risk_guards
```

Case IDs, injection labels/indices, D/G/J values, private receipts, and source
FailurePatternCard IDs are forbidden in the deployable entry. They remain in the
EditManifest and lineage provenance.

The surface registry declares a dynamic template:

```text
surface_template_id = skill_library.entries/{skill_id}
path_template = skills/learned/{skill_id}.json
surface_type = structured_entry
allowed_operations = [ADD]
precondition = ABSENT
value_schema = skill-entry/1
derived_outputs = [retrieval_index]
```

The parent ADD surface applies only while the entry is absent. Once present, the
parent surface is inapplicable and the mutually exclusive child surfaces
`.body`, `.observable_applicability`, and `.risk_guards` own later revisions.
One edit cannot change two child fields.

The retrieval index is a deterministic, read-only compilation of SkillEntry
sources, sorted by stable skill ID. It is not an EditManifest target and does not
count as a second surface. Its hash and compiler version are recorded in the
snapshot receipt. M0 uses a deterministic rule/lexical index; learned embedding
indexes are outside scope.

### 8.2 EditManifest

Each slow-path proposal is a falsifiable contract:

```text
edit_id
base_harness_sha
target_pattern_id
target_surface_id
operation: PATCH | ADD
surface_precondition
dependency_precondition_shas
minimal_patch | new_value
observable_applicability
predicted_agent_behavior_change
predicted_data_effect
automatically_selected_risk_cases
falsification_condition
```

At application time the controller rechecks the base snapshot, target surface,
and declared dependency hashes. A stale edit is rebased and replayed; it is never
silently applied using an old receipt.

For `ADD SkillEntry`, the controller mechanically verifies:

1. `skill_id` is canonical, resolves beneath the learned-skill snapshot directory,
   and is currently absent;
2. the new value conforms to `skill-entry/1` and has `skill_kind=capability`;
3. every allowed tool is a canonical registry member and task-compatible;
4. applicability uses only the closed public feature vocabulary;
5. no oracle/private/provenance-only field appears in deployable content;
6. the source diff contains exactly one new SkillEntry;
7. skill, operator, feature, candidate, and compiler dependency hashes are current;
8. the retrieval index rebuild is deterministic.

The predicted behavior for a library edit includes retrieval of the new skill,
at least one effect-distinct PROGRAM using allowed tools, retention of identity,
and unchanged effective Harness views for out-of-scope cases. Skill creation that
does not change retrieval/candidate behavior is `DEAD_EDIT`; changed supply
without gain is `BEHAVIOR_CHANGED_NO_GAIN`; recovery with risk regression is
`TARGET_RECOVERED_WITH_HARM`; only predicted behavior, gain, and stable risk/scope
can yield `SUPPORTED_EDIT`.

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

Out-of-scope deterministic cases require all three of: identical effective
Harness-view SHA, reuse of all eligible baseline LLM cache records, and no
normalized semantic behavior diff. Applicability alone is insufficient: a new
skill that changes `top_k` retrieval has changed scope even if its own predicate
does not match. Stochastic cases use common seeds and declared repeat/tolerance
rules. In-scope clean or genuine-event cases may trigger the edit, but require
`delta U >= -epsilon`.

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

Period-change fixtures use the same replay machinery but keep their two ledgers
separate. When upstream observation/localization/mechanism assessments pass, the
registry contract proves period correction unavailable, and the Agent selects
identity, the Agent decision is successful while `OPERATOR_GAP` is appended to
the system capability backlog. If an earlier stage fails, that earlier Agent
fault remains first; period injection type never overrides stage attribution.

## 10. M0 versioning and cycle limits

M0 maintains:

- one active pointer to an immutable `HarnessSnapshot`;
- coexisting copy-on-write H_t and H_t+edit snapshot roots for paired replay;
- supported scoped candidates;
- rejected and inconclusive edits;
- parent snapshot, manifest, and replay receipt for every transition.

Snapshot and run identity are separated:

```text
harness_content_sha
  = resolved instruction + skills + memory + retrieval/applicability
    + candidate controls + risk/verification

runtime_bundle_sha
  = harness_content_sha
    + operator contract and implementation bundle SHA
    + skill/observable/candidate schemas
    + prompt and retrieval compiler SHAs

run_context_sha
  = runtime_bundle_sha
    + model/provider identity
    + environment/dependency manifest
    + evaluator/probe configuration
```

`created_at`, `parent_sha`, `edit_log`, and `cycle_id` are provenance and do not
enter `harness_content_sha`. The lock records them separately. An operator
metadata hash alone is insufficient because implementation bytes can change
without a registry-name change.

`EditManifest.base_harness_sha` means `harness_content_sha`; optimistic
concurrency additionally checks the declared dependency preconditions. Every
paired-replay report records all three identities, so equal Harness content is
never mistaken for an equal runtime or experimental context.

Checked-in H0 authoring files and lock are immutable. Controller-created snapshots
live under the run/state store, for example
`runs/minipipe/harness_snapshots/<runtime_bundle_sha>/`, and contain their own
learned SkillEntries. The controller never mutates a global
`methods/ttha/harness/skills/learned/` directory during replay. A later explicit
export may promote a supported snapshot into checked-in authoring content, but
that is not part of scientific paired replay.

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
contracts/
  candidate.py
  harness.py
  schemas/
    skill_entry_v1.json
    harness_snapshot_v1.json
    edit_manifest_v2.json
runtime/
  executor.py                # canonical operator execution
  candidate_pool.py          # generic identity/PROGRAM pool
  decision_trace.py
  llm_cache.py
operators/                   # the single canonical operator registry
methods/ttha/
  method.py
  fast_agent.py
  slow_agent.py
  harness/
    h0/
    harness_surfaces.json
    compiler.py
evaluation/minipipe/
  baselines/fixed_program_baseline_v1.json
  corpus/
  valuation/
  probes/
  feedback/
  replay/
  schemas/
cli/                         # composition root and cycle command
runs/minipipe/harness_snapshots/  # ignored controller-managed immutable state
```

These top-level ownership boundaries and dependency directions are required.
Modules may be split internally without moving responsibility across a boundary.
Minipipe must use the canonical operator registry and executor; it may not copy
either.

The current `runtime/fast_path.py` is a migration source, not the target generic
runtime: it directly imports H_ref state, grammar, and defaults. Generic candidate
pool mechanics must accept injected supplier, selector, risk policy, and canonical
Candidate contracts. TTHA supplies the Agent implementation. H_ref fixed programs
are frozen into the private JSON baseline, after which `methods/h_ref_v02/` is
removed from the target active tree rather than retained as compatibility code.

## 12. One M0 cycle

1. Compile H0, validate its lock and dependency hashes, and create the immutable
   cycle-0 snapshot.
2. Build a controlled corpus spanning clean/genuine events and four families:
   missing, impulsive/outlier, level shift, and period change.
3. Construct typed `PublicCaseView` objects and run the Agent fast path with
   runtime identity plus at most two PROGRAM candidates.
4. Record `DecisionTrace` mechanically.
5. Privately evaluate corruption, selected output, effective candidate pool,
   collateral, repair-response curves, and period diagnostics.
6. Run oracle- and observable-parameterized witness checks where required and
   assign three-valued expressibility status.
7. Derive stage/fault/cause assessments and the first actionable fault.
8. Produce private `CaseFeedback` and sanitized public pattern cards.
9. Select recurring high-impact actionable patterns and request at most three
   one-surface edit manifests.
10. Materialize H_t and H_t+edit, then run Stage A and, when eligible, Stage B
    paired replay.
11. Promote at most one supported edit or retain/reject it with a receipt.
12. Append snapshot lineage and run the next cycle from the resulting active
    snapshot.

The family balance is intentional. Missing has six canonical imputation
operators, impulsive/outlier has four local/global alternatives, and level shift
has `repair_level_shift`; these three families exercise supply and selection.
Period change deliberately lacks a correction operator and exercises correct
identity selection plus `OPERATOR_GAP` routing.

Primary artifacts remain small:

```text
private/case_feedback.jsonl
public/failure_patterns.json
public/failure_patterns.md
public/edit_manifest.json
private/paired_replay_report.json
harness_lineage.jsonl
private/operator_capability_backlog.jsonl
```

## 13. Minimum acceptance criteria

M0 is complete when one command can run at least two consecutive cycles and:

1. the active method registry and composition root contain TTHA but no H_ref;
2. TTHA and generic runtime code do not import `methods.h_ref_v02`;
3. the same TTHA Agent core executes both permission-separated roles;
4. repeated H0 compilation produces the same content and bundle SHAs, with empty
   capability library and memory;
5. H0 lock mismatch or stale schema/operator/compiler dependency fails loudly;
6. the public prompt path cannot import/read private artifacts and candidate J
   values never appear in fast-path inputs or public traces;
7. identity is always present, cannot be risk-filtered, occupies one configured
   slot, and a missing `chosen_candidate_id` is a protocol failure;
8. deterministic effect-equivalence to identity follows the declared byte or
   numerical-tolerance contract;
9. fixtures distinguish `SKILL_LIBRARY_GAP`, `RETRIEVAL_MISS`,
   `SKILL_CONTENT_GAP`, `SELECTION_MISS`, `OPERATOR_GAP`, and
   `EXPRESSIBILITY_UNKNOWN`;
10. oracle-only success cannot sign a library edit, while an observable witness
    can;
11. the fault router rejects skill kinds/surface classes not allowed for the
    confirmed cause;
12. an `ADD SkillEntry` changes one absent structured source surface, and the
    retrieval index changes only as a deterministic derived artifact;
13. every accepted diff maps to exactly one owned surface and stale
    surface/dependency preconditions force replay;
14. repair-probe beta mappings are monotonic/versioned and run under common
    seeds, while period is diagnostic-only;
15. a correctly abstained period-change fixture records Agent success and a
    separate system `OPERATOR_GAP` backlog event;
16. identical effective LLM requests, including public case/tool-context hashes,
    replay the same cache record; full-snapshot-only changes do not force misses;
17. out-of-scope replay requires equal effective-view SHA, eligible cache reuse,
    and no normalized behavior diff;
18. paired fixtures exercise at least dead, supported, harmful, unexpected-gain,
    and infrastructure-inconclusive outcomes;
19. every promoted snapshot has a parent, manifest, paired-replay receipt, and
    final core-regression result;
20. repeated fixed-seed runs reproduce normalized behavior signatures and
    scientific verdicts within declared tolerances.

These criteria establish the causal feedback skeleton. More sophisticated
clustering, learned model routing, candidate-J A/B experiments, Pareto selection,
and Harness branching remain M1 work inside the same boundaries.
