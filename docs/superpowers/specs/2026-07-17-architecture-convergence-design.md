# SelfEvolvingHarnessTS Architecture Convergence Design

> **Status:** Approved in design review on 2026-07-17.
>
> **Scope:** Converge the active project architecture without reinterpreting frozen scientific evidence. Historical implementations are preserved by Git tag and evidence manifests, not by keeping every old import path alive in the current source tree.

> **Agent-centric update:**
> [`2026-07-17-agent-centric-minipipe-m0-design.md`](2026-07-17-agent-centric-minipipe-m0-design.md)
> supersedes this document wherever it proposes H_ref as an active/importable
> method or as a minipipe execution arm. H_ref is historical evidence only; TTHA
> is the sole active method in the target architecture.

> **Benchmark ownership update:**
> [`2026-07-17-active-tree-cleanup-design.md`](2026-07-17-active-tree-cleanup-design.md)
> supersedes this document's `method_compat.py`-only benchmark sketch. The complete
> benchmark-v0.2 package now has one active owner under
> `evaluation/benchmark_v02/`. Its 25 relocated frozen evidence files are protected
> by moved-path `.gitattributes` rules and SHA-256 entries in
> `artifacts/manifests/active_tree_cleanup.json`, enforced by the frozen-protocol
> smoke test.

## 1. Problem Statement

`SelfEvolvingHarnessTS` grew by adding a new implementation surface for each research phase. The repository now contains multiple harness state models, multiple fast-path entry points, phase-specific runners, frozen benchmark code, experiment diagnostics, and tracked results in the same active package.

The primary problem is not Git branching. It is architectural branching:

- `harness.HarnessState`, `policy.PolicyBundle`, and `p6.P6HarnessState` represent overlapping state concepts;
- `fast_path.process`, `policy.routed_process`, and `p6.run_fast_path` represent overlapping execution paths;
- benchmark code imports a concrete P6 implementation instead of depending only on a method contract;
- 43 top-level `run_*.py` files expose chronological experiment phases as if they were current product entry points;
- the proposed minipipe would copy operators and execution logic outside the canonical project, creating another live implementation branch.

The refactor must produce one understandable current architecture. It must not preserve ineffective research paths as permanent runtime dependencies merely because they existed previously.

## 2. Goals

1. Establish TTHA as the only method line that receives new capabilities.
2. Establish one operator registry, one runtime executor, and one canonical `runtime/fast_path.py`.
3. Make benchmark-v0.2 and minipipe consume a stable method contract rather than concrete method internals.
4. Preserve H_ref/P6 as a reproducible frozen baseline without retaining `p6/` as an active dependency.
5. Preserve scientific evidence through frozen artifacts, manifests, environment records, and a pre-refactor Git tag.
6. Remove superseded implementation branches from the current working tree after their useful behavior has been promoted and verified.
7. Keep generated runs, caches, and temporary artifacts outside the tracked source tree.

## 3. Non-Goals

- Redesigning the H_ref or TTHA algorithms during structural migration.
- Changing benchmark-v0.2 data roles, metrics, split semantics, or previously signed verdicts.
- Guaranteeing permanent import compatibility for every historical runner or experiment module.
- Keeping full copies of P1-P6 implementations under an `archive/` package.
- Moving to a `src/` layout before logical dependencies have converged.
- Treating directory movement alone as successful refactoring.

## 4. Preservation Policy

Preservation is tiered. Scientific traceability and runtime compatibility are separate concerns.

### 4.1 Tier A: Scientific sources of truth

The following remain strictly protected:

- benchmark method and prepared-series contracts;
- benchmark-v0.2 dataset roles, split manifests, metrics, trainer semantics, and resource discipline;
- canonical operator identities and provenance;
- frozen verdicts, digests, manifests, and the evidence required to interpret them.

Tier A content may be relocated only after its bytes or behavior are proven equivalent and its frozen references are reconciled explicitly.

### 4.2 Tier B: Executable reference baselines

H_ref/P6 remains executable as a comparison baseline. Its active representation is migrated into `methods/h_ref_v02/` and uses the canonical runtime. It is frozen: no new method capability is added to it.

The new architecture does not retain an adapter that imports `p6.fast_path`. Instead, the general mechanics are promoted into the canonical runtime and H_ref-specific grammar and configuration are represented as frozen method configuration.

### 4.3 Tier C: Completed or ineffective research paths

Old harness, slow-path, policy, conditioning, memory, P1-P6, E32, confirmatory, and diagnostic implementations are candidates for retirement. Reusable contracts or mechanics are extracted first. The remaining code is removed from the active tree once no active module imports it.

Historical recovery is provided by the pre-refactor Git tag, environment record, manifests, verdicts, and focused reproduction notes. The current tree does not retain a full source-code archive.

### 4.4 Tier D: Generated and duplicated material

Caches, temporary logs, zip copies, duplicate results, compiled files, and ordinary run outputs are removed from the tracked project. Required large artifacts live in an external artifact location; only their manifests and content digests remain in Git.

## 5. Target Structure

```text
SelfEvolvingHarnessTS/
  contracts/
    method.py
    program.py
    task.py
    harness.py
  operators/
    registry.py
    provenance.py
    common.py
    impute.py
    denoise.py
    outlier.py
    structural.py
    align.py
    shape.py
  runtime/
    executor.py
    fast_path.py
    trace.py
    errors.py
  methods/
    h_ref_v02/
      config.py
      method.py
    ttha/
      method.py
      harness.py
  evaluation/
    benchmark_v02/
      method_compat.py
    minipipe/
  experiments/
    archive/
      README.md
  artifacts/
    frozen/
    manifests/
  cli/
    main.py
  tests/
    contracts/
    runtime/
    integration/
    frozen_protocol/
    architecture/
```

`experiments/archive/` contains concise experiment descriptions, verdict pointers, and any small dedicated reproduction launcher that remains necessary. It does not contain a second importable copy of the old project.

`runs/` is a repository-local ignored directory for traces, caches, generated programs, and per-case output. It is not part of the importable package.

## 6. Module Responsibilities

### 6.1 Contracts

`contracts/` defines stable project language and contains no concrete method, evaluation, filesystem, or network logic.

It owns:

- `TaskSpec` and task identity;
- visible series/request views;
- `PreparedSeries` and `PreparationResult`;
- the `Method` protocol;
- `Program` and `ProgramStep`;
- `ExecutionReceipt` and result status;
- `HarnessSpec` and `HarnessEdit` contracts.

The frozen `benchmark.method_api.BenchmarkMethod` API is not modified in place. `evaluation/benchmark_v02/method_compat.py` translates between its existing call shape and the canonical result envelope while benchmark-v0.2 remains frozen. This compatibility module translates contracts only; it does not import an old execution branch.

### 6.2 Operators

`operators/` is the single operator implementation and registry. Every operator declares its identity, parameters, dependencies, target-space behavior, risk properties, and fallback policy.

No method or evaluation package may copy a project operator. External vendoring is allowed only for genuinely external source material and requires a source manifest containing origin, version or commit, copied paths, local modifications, and license metadata.

### 6.3 Runtime

`runtime/` is the only execution surface. It resolves programs through the operator registry, executes steps, records provenance and traces, validates outputs, and produces typed receipts.

`runtime/fast_path.py` owns general candidate generation, canonicalization, deduplication, risk filtering, selection, and execution mechanics. It does not own H_ref grammar literals or TTHA adaptation policy.

### 6.4 Methods

`methods/h_ref_v02/` owns only the frozen H_ref method definition: grammar, selector configuration, slot budget, and the method wrapper that invokes the canonical runtime.

`methods/ttha/` is the only active method line. New harness behavior, adaptation logic, and method capabilities are added here. It may depend on contracts, operators through runtime, and explicitly approved shared services. It may not import historical experiment modules.

### 6.5 Evaluation

`evaluation/benchmark_v02/` owns formal, low-frequency, frozen judgment. It privately owns targets, role policies, metrics, and signed evaluation semantics. Ownership is logical, not physical: the frozen benchmark-v0.2 package is not relocated during Phases 0-4, and `evaluation/benchmark_v02/` contains only the compatibility layer; any later relocation follows the Tier A byte/behavior equivalence rule.

`evaluation/minipipe/` owns high-frequency mechanism development: synthetic oracle generation, the information wall, intervention receipts, attribution, proposal review, and paired replay.

Both evaluations consume the canonical `Method` contract. They do not import H_ref, TTHA, or P6 internals. Concrete objects are injected by the CLI composition root.

### 6.6 CLI

`cli/` is the composition root. It selects a method, evaluation environment, runtime configuration, and output directory. It replaces valuable top-level runners with named subcommands but contains no algorithmic logic.

### 6.7 TTHA and minipipe are different system roles

TTHA is the active method being developed. It observes only method-visible inputs, decides whether and how to adapt, invokes the canonical runtime, and returns a preparation result. It is the component that may eventually run outside the development environment.

Minipipe is an offline development and diagnostic environment for methods. It generates or loads controlled cases, privately owns synthetic clean references and injection manifests, invokes H_ref or TTHA through the method contract, attributes failures, and produces feedback for the next method edit.

Minipipe may call TTHA; TTHA may not import minipipe. Oracle values, grader logic, failure labels, and clean-derived measurements owned by minipipe must never enter TTHA requests, runtime state, or deployable method artifacts. Minipipe is a development capability suite for TTHA, not a serving component of TTHA.

## 7. Dependency Rules

The stable direction is:

```text
contracts
   |-- operators --> runtime --> methods
   `---------------------------> evaluation

cli imports and composes methods + evaluation + runtime
```

The following imports are forbidden in active code:

- evaluation importing a concrete method implementation;
- runtime importing evaluation or a concrete method;
- TTHA importing P6, archived experiments, or frozen result files;
- minipipe copying or privately registering canonical operators;
- active packages importing `experiments/archive`;
- methods reading evaluation-owned oracle or future data.

Temporary migration imports are allowed only inside the migration branch, must be named in the implementation plan, and must be removed before the corresponding migration task is accepted.

## 8. Runtime Data Flow

```text
CLI
  |-- selects Evaluation: benchmark_v02 or minipipe
  |-- selects Method: h_ref_v02 or ttha
  `-- injects Runtime, configuration, and run directory
          |
Evaluation constructs a visible PreparationRequest
          |
Method.prepare(request)
  |-- analyzes visible inputs
  |-- produces a decision or Program
  `-- invokes the canonical Runtime
          |
PreparationResult
  |-- prepared_series
  |-- status: PREPARED, ABSTAINED, or FAILED
  |-- program
  |-- execution_receipt
  `-- provenance
          |
Evaluation privately reads target/oracle material and scores the result
```

The request contains only method-visible information. Benchmark future values and split roles, and minipipe clean series and injection manifests, remain evaluation-private.

## 9. Error Model

The canonical runtime distinguishes four error classes:

- `ContractError`: invalid request, output shape, units, schema, or operator identity. The case is failed.
- `ExecutionError`: operator failure, declared dependency failure, or execution timeout. The receipt records the failure. Fallback occurs only when the operator contract explicitly permits it.
- `ProtocolViolation`: oracle access, role-policy violation, or frozen digest drift. The entire run aborts immediately.
- `InfrastructureError`: missing files, unavailable model or device, or external service failure. The run is marked as a technical abort and is not mixed into scientific metrics.

Silent fallback is forbidden. `ABSTAINED` is an explicit method decision and is not represented as an execution failure or implicit identity fallback.

## 10. Artifact Boundaries

- `runs/` stores ordinary traces, caches, generated programs, per-case output, and temporary notebooks; it is ignored by Git.
- `artifacts/frozen/` stores signed verdicts, frozen manifests, digests, and the minimum evidence needed to interpret formal results.
- `artifacts/manifests/` stores dataset, program, environment, and external-vendor provenance.
- Method code may not read historical verdicts during a run unless a manifest explicitly declares the artifact as an allowed training or memory input.

## 11. Migration Strategy

The migration uses promote, verify, switch, and remove. Copying is a temporary migration operation, not a long-term duplication strategy.

For a superseded module such as `p6/fast_path.py`:

1. characterize the old externally relevant behavior;
2. copy and reshape general mechanics into `runtime/fast_path.py`;
3. move H_ref-specific literals into `methods/h_ref_v02/config.py`;
4. run equivalence and frozen-protocol tests;
5. switch active callers to the canonical modules;
6. prove that active code no longer imports the old module;
7. remove the old module from the current tree.

There is no permanent `h_ref_v02 -> p6.fast_path` adapter chain.

### 11.1 Phase 0: Reversible baseline

- record the current commit, environment, test outcomes, and frozen digests;
- create tag `pre-architecture-convergence-2026-07-17`;
- create a module disposition inventory with keep, promote, replace, and remove decisions;
- make no runtime behavior changes.

### 11.2 Phase 1: Canonical foundation

- create contracts and runtime packages;
- promote executor, trace, error, and fast-path mechanics;
- add contract, runtime, architecture, and characterization tests before switching callers.

### 11.3 Phase 2: H_ref and benchmark convergence

- create the frozen H_ref method package;
- make the benchmark consume the canonical method boundary;
- verify old/new H_ref behavior on fixed probes;
- remove active imports of P6.

### 11.4 Phase 3: Active TTHA line

- create `methods/ttha/`, initializing its harness from the frozen H_ref configuration (fingerprint-verified equivalent), so that minipipe edit cycles start from an H_ref-equivalent H0 while `methods/h_ref_v02/` remains frozen as the comparison baseline;
- create the `evaluation/minipipe/` package boundary: directory skeleton, information-wall layout, and architecture tests. Minipipe functionality is implemented on its own track per the minipipe plan document and does not gate Phase 4;
- use canonical contracts, operators, and runtime without project-code vendoring.

### 11.5 Phase 4: Historical cleanup

- replace valuable runners with CLI subcommands;
- remove superseded P6, old fast path, slow path, and ineffective policy implementations;
- remove generated and duplicate material from the tracked source tree;
- retain focused verdict, manifest, and reproduction documentation.

### 11.6 Phase 5: Physical package layout

After active legacy imports reach zero, decide whether to move the logically converged package into a standard `src/` layout. This phase changes physical layout only; it does not redesign runtime semantics.

## 12. Testing Strategy

1. **Contract tests** verify schemas, result states, and input/output constraints.
2. **Runtime tests** verify the canonical executor, tracing, declared fallback, abstention, and typed failures.
3. **Characterization tests** pin externally relevant old behavior before extraction.
4. **H_ref equivalence tests** compare old and canonical paths on fixed inputs before the old path is removed.
5. **Frozen-protocol tests** protect benchmark splits, metrics, digests, resource roles, and signed behavior.
6. **Integration tests** run at least one H_ref and one TTHA/minipipe case through the canonical contract and runtime.
7. **Architecture tests** parse imports and reject forbidden dependency edges.

Two standing execution rules apply to every phase: all test and equivalence evidence is produced under the project's canonical interpreter (conda `project`) only; and any file move or rename must re-verify `.gitattributes` path matching and line-ending policy before digests are compared, because path-scoped attribute rules stop matching silently after moves and CRLF conversion has previously corrupted frozen digests.

Archived code is not part of the active test matrix. Its reproducibility is owned by the pre-refactor tag and its recorded environment.

## 13. Retirement Gate

A module may be removed from the current tree only when all applicable conditions hold:

1. its preservation tier and replacement are recorded;
2. no active module imports it;
3. reusable contracts or mechanics have been promoted;
4. Tier A/B equivalence and frozen-protocol tests pass;
5. the pre-refactor tag and environment record can identify the historical implementation;
6. no current research conclusion depends on an unmanifested artifact stored only beside that module.

## 14. Completion Criteria

The architecture convergence is complete when:

- exactly one operator registry is active;
- exactly one runtime executor is active;
- exactly one `runtime/fast_path.py` implements fast-path mechanics;
- TTHA is the only method line receiving new capabilities;
- benchmark-v0.2 and minipipe depend on the canonical method boundary;
- active code has no imports from P6 or archived experiments;
- fixed-probe H_ref outputs match the characterized P6 baseline;
- benchmark-v0.2 data, metrics, roles, and verdict meaning remain unchanged;
- historical implementations are recoverable through the pre-refactor tag and environment records;
- generated caches, logs, zip copies, and ordinary runs are absent from the tracked source tree;
- no implementation remains in the active tree solely because it may be useful someday.
