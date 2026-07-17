# Active Tree Cleanup Design

**Date:** 2026-07-17  
**Status:** Approved direction; implementation pending  
**Recovery tag:** `pre-architecture-convergence-2026-07-17`  
**Foundation branch:** `refactor/architecture-convergence-foundation`

## 1. Objective

Turn `SelfEvolvingHarnessTS` into a small active development tree centered on one method
line and one runtime. Historical P1–P6, E32, confirmatory, ablation, and long-run code no
longer needs to execute from the current checkout. It remains recoverable from the frozen
Git tag and the architecture baseline manifest.

The cleanup optimizes for:

1. functional completeness of the active H_ref/TTHA path;
2. clear component ownership;
3. faster navigation, iteration, and Agent reasoning;
4. preserving only scientific evidence that still defines the active frozen baseline.

The cleanup does not preserve source compatibility for retired experiment runners.

## 2. Chosen Strategy

Use a clean active tree with Git-based recovery. Historical source and results are removed
from the current branch, not copied into an importable `experiments/archive/` subtree.

Recovery remains available through:

- tag `pre-architecture-convergence-2026-07-17`;
- `artifacts/manifests/architecture_convergence_baseline.json`;
- `experiments/archive/README.md`;
- normal Git history.

This is preferable to an in-repository source archive because a copied archive would still
pollute search, dependency analysis, packaging, and future Agent decisions.

## 3. Target Active Tree

The active checkout should converge to this logical structure:

```text
SelfEvolvingHarnessTS/
├── AGENTS.md
├── README.md
├── __init__.py
├── contracts/
├── operators/
├── conditioning/
├── runtime/
├── methods/
│   ├── h_ref_v02/
│   └── ttha/                 # created by the later TTHA implementation track
├── evaluation/
│   └── benchmark_v02/
├── benchmark/                    # temporarily retains the frozen physical evaluator
├── data/
│   └── benchmark_v0/
├── artifacts/
│   └── manifests/
├── results/
│   └── Benchmark_v0_2/
├── experiments/
│   └── archive/README.md
├── docs/
├── tests/
│   ├── architecture/
│   ├── contracts/
│   ├── runtime/
│   ├── integration/
│   └── frozen_protocol/
└── run_benchmark.py              # temporary until the unified CLI track
```

`conditioning/` stays because canonical operators currently use its period estimator.
The physical `benchmark/` package stays until benchmark-v0.2 is either retired or moved
behind a fully self-contained evaluation package. Neither directory is a historical branch
in the cleaned tree.

## 4. Required Promotions Before Deletion

Two active imports still cross into legacy packages and must be resolved first:

1. `benchmark/method_api.py` and `benchmark/baselines.py` must import `TaskSpec` from
   `contracts.task`, after which `policy/` can be removed.
2. `benchmark/trainers.py` must use a frozen benchmark-owned model module under
   `evaluation/benchmark_v02/` (or a canonical model package), after which the legacy
   `evaluators/` package can be removed.

No other canonical package currently imports `p6`, `harness`, `policy`, `sandbox`,
`slow_path`, `memory`, or `llm`.

The transitional modules `p6/fast_path.py`, `p6/harness_state.py`,
`policy/task_spec.py`, and `sandbox/executor.py` are deleted rather than retained as
permanent shims. Active callers already have canonical replacements.

## 5. Removal Scope

### 5.1 Historical root files

Remove historical root-level runner and experiment modules, including the P1–P6, E32,
confirmatory, family, gym, long-run, ablation, proposer, updater, replication, transfer,
and calibration entry points. Keep only the package initializer and the temporary active
benchmark entry point.

Remove tracked root logs and historical narrative files (`BUILD.md`, `EXECUTION_LOG.md`,
and `ONBOARDING.md`) after their still-relevant usage information is replaced by a concise
active `README.md`.

### 5.2 Retired packages

Remove these legacy implementation packages after the two promotions above:

- `config/`
- `diagnostics/`
- `evaluators/`
- `fast_path/`
- `harness/`
- `llm/`
- `memory/`
- `models/`
- `p6/`
- `policy/`
- `sandbox/`
- `slow_path/`

Potentially useful LLM, retrieval, memory, or edit logic is not retained speculatively.
When TTHA needs a mechanism, it is recovered from the tag, characterized if necessary,
and promoted intentionally into the active architecture.

### 5.3 Data and results

Keep:

- `data/benchmark_v0/` and the benchmark data instructions needed to acquire its sources;
- `results/Benchmark_v0_2/` as the current frozen judgment evidence;
- architecture manifests and concise frozen evidence pointers.

Remove:

- legacy `data/_artifacts/` and P1–P6 data loaders that only serve retired runners;
- all result trees other than `results/Benchmark_v0_2/`;
- tracked experiment logs and generated caches.

The cleanup does not add large historical archives to `artifacts/`.

### 5.4 Tests

Retain focused tests that prove active functionality:

- canonical contracts;
- operator registry/integrity and shared period behavior;
- runtime executor and fast-path fingerprints;
- H_ref integration;
- architecture dependency rules;
- a small benchmark-v0.2 smoke/frozen-protocol set.

Remove tests whose sole purpose is to keep retired P1–P6, E32, slow-path, policy, LLM,
memory, updater, proposer, or historical runner surfaces executable.

Tests are selected by functional coverage, not by preserving the old pass count. The
cleanup will not recreate exhaustive benchmark infrastructure merely to keep historical
tests alive.

## 6. Dependency Rules After Cleanup

- `contracts` depends only on the standard library, NumPy, and canonical operators where
  task-level canonicalization is required.
- `operators` may depend on `conditioning`, but not on methods, evaluation, or history.
- `runtime` depends on contracts/operators and the injected or frozen method definition;
  it never imports retired packages.
- `methods` depends on contracts/runtime/operators, not benchmark internals.
- `evaluation` may adapt a canonical method to the frozen benchmark.
- active code never imports `experiments` or a removed namespace.

An architecture test must reject imports of every removed top-level namespace.

## 7. Execution Order

1. Record a machine-readable cleanup manifest with the recovery tag and removal groups.
2. Promote the two remaining active dependencies out of `policy/` and `evaluators/`.
3. Add or consolidate the minimal active smoke tests and observe them passing.
4. Remove legacy root files, packages, results, data, and historical tests in explicit
   reviewed groups.
5. Replace historical onboarding material with an active `README.md`.
6. Tighten `.gitignore`, `.gitattributes`, and architecture rules to the remaining tree.
7. Run the focused active suite and fixed H_ref/benchmark fingerprints.
8. Report the final file-count reduction and any intentionally retained temporary edge.

Deletion is performed with explicit Git paths. Broad recursive deletion against the
workspace root is forbidden.

## 8. Verification and Success Criteria

The cleanup is complete when:

- the active worktree contains no retired package listed in section 5.2;
- root-level historical runners and tracked logs are gone;
- only `results/Benchmark_v0_2/` remains under `results/`;
- `rg` finds no active imports of retired namespaces;
- canonical H_ref state SHA remains `4e7e4ac5b40c941d`;
- deterministic ladder SHAs and fixed-probe artifact digests remain unchanged;
- the canonical H_ref method and benchmark adapter smoke tests pass;
- focused operator/runtime/contract/architecture tests pass under the project interpreter;
- the tracked file count is materially reduced from the current 1,132 files, with a target
  below 200 unless benchmark evidence alone makes that impossible;
- the cleanup branch is Git-clean and the other Agent's main-worktree changes remain
  untouched.

The historical full-suite pass count is explicitly not a success criterion because most of
that suite is deleted together with the functionality it tested.

## 9. Deferred Work

This cleanup does not implement TTHA, minipipe, a unified CLI, or method-performance
improvements. It creates the small active tree in which those features can be built without
competing with historical branches.
