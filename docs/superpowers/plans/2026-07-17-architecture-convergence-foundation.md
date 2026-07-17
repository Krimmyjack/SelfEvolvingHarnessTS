# Architecture Convergence Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete architecture-convergence Phases 0–2: capture a reproducible baseline, create canonical contracts and runtime modules, promote H_ref out of `p6/`, and remove every active benchmark dependency on P6 while preserving frozen behavior.

**Architecture:** The migration uses promote, verify, switch, and temporarily shim. Canonical implementations live in `contracts/`, `runtime/`, and `methods/h_ref_v02/`; `p6/fast_path.py`, `p6/harness_state.py`, `policy/task_spec.py`, and `sandbox/executor.py` become compatibility shims only after identity and behavior tests prove equivalence. Frozen benchmark files stay physically in `benchmark/`; `evaluation/benchmark_v02/` contains the compatibility boundary that connects the frozen evaluator to the canonical H_ref implementation.

**Tech Stack:** Python 3.10.19, NumPy 2.2.6, SciPy 1.15.2, scikit-learn 1.7.2, pytest, stdlib `dataclasses`/`typing`/`ast`, Git.

## Global Constraints

- Run every test with `D:\Anaconda_envs\envs\project\python.exe`; from WSL this is `/mnt/d/Anaconda_envs/envs/project/python.exe`.
- Run test commands from `/mnt/c/Users/辉/desktop/agent`, not from the package directory.
- Always pass `--basetemp=SelfEvolvingHarnessTS/_pytest_architecture` because the Windows default pytest temp root is not accessible in this environment.
- Do not stage, revert, or overwrite pre-existing worktree changes from another Agent. Before each commit, stage only the files listed by that task.
- Keep benchmark-v0.2 physically under `benchmark/` and `results/Benchmark_v0_2/` throughout this plan.
- Preserve `SCHEMA_VERSION = "p6-harness-state/1"`, H0 SHA `4e7e4ac5b40c941d`, deterministic ladder SHAs, and fixed-probe output digests exactly.
- Do not implement TTHA behavior or minipipe functionality in this plan. Those are separate implementation tracks after the canonical foundation exists.
- Do not vendor or copy canonical project operators.
- New and moved Python files must match `.gitattributes` as `text: set` and `eol: lf` before digest comparison.
- Temporary legacy shims may point to canonical modules; canonical modules may not import legacy `p6`, `harness`, `policy`, `sandbox`, or `experiments` modules.
- Treat the session's automatic approval setting as a tool-execution preference, not as a code-review workflow. Platform-enforced escalation boundaries still apply.

## Plan Decomposition

This plan intentionally stops after Phase 2. Later work is split into independent plans:

1. TTHA H0 skeleton and fingerprint-equivalent initialization.
2. Minipipe package boundary and information-wall tests.
3. Historical runner/slow-path/policy retirement and unified CLI.
4. Optional physical `src/` migration after active legacy imports reach zero.

## File Map

**Created in this plan:**

- `artifacts/manifests/architecture_convergence_baseline.json` — immutable pre-refactor code, environment, test, and H_ref fingerprints.
- `experiments/archive/README.md` — retirement policy and Git-tag recovery instructions; no archived source copy.
- `contracts/__init__.py` — canonical contract exports.
- `contracts/task.py` — canonical `TaskSpec` and `MetricSpec` implementation.
- `contracts/program.py` — canonical immutable program identity.
- `contracts/method.py` — canonical request/result/method protocol.
- `runtime/__init__.py` — canonical runtime exports.
- `runtime/errors.py` — typed runtime/protocol error taxonomy.
- `runtime/trace.py` — execution trace schema.
- `runtime/executor.py` — sole operator pipeline executor.
- `runtime/fast_path.py` — sole fast-path mechanics implementation.
- `methods/__init__.py` — method namespace.
- `methods/h_ref_v02/__init__.py` — frozen H_ref public surface.
- `methods/h_ref_v02/config.py` — frozen H_ref state, grammar, defaults, and fingerprints.
- `methods/h_ref_v02/method.py` — canonical H_ref method implementation.
- `evaluation/__init__.py` — evaluation namespace.
- `evaluation/benchmark_v02/__init__.py` — frozen benchmark compatibility namespace.
- `evaluation/benchmark_v02/method_compat.py` — old benchmark contract and batch-run compatibility.
- `tests/contracts/test_task_contract.py` — legacy/canonical task identity tests.
- `tests/contracts/test_method_contract.py` — canonical method/program validation tests.
- `tests/runtime/test_executor_contract.py` — canonical/legacy executor identity tests.
- `tests/runtime/test_fast_path_equivalence.py` — fixed H_ref state, ladder, choice, and artifact digests.
- `tests/integration/test_h_ref_method.py` — canonical and benchmark H_ref equivalence.
- `tests/architecture/test_dependency_rules.py` — forbidden import edges.

**Replaced with compatibility shims:**

- `policy/task_spec.py`
- `sandbox/executor.py`
- `p6/harness_state.py`
- `p6/fast_path.py`

**Modified without relocation:**

- `benchmark/baselines.py`
- `benchmark/dev_eval.py`

---

### Task 1: Freeze the Pre-Refactor Baseline

**Files:**

- Create: `artifacts/manifests/architecture_convergence_baseline.json`
- Create: `experiments/archive/README.md`

**Interfaces:**

- Consumes: commit `1e75305770815c256d5b295b7ad6b8cb6cffe4b4`, frozen H_ref state and fast path.
- Produces: tag `pre-architecture-convergence-2026-07-17` and a machine-readable fingerprint record used by all later equivalence tests.

- [ ] **Step 1: Verify the targeted baseline under the canonical interpreter**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/test_task_spec.py \
  SelfEvolvingHarnessTS/tests/test_p6_surfaces.py \
  SelfEvolvingHarnessTS/tests/test_benchmark_baselines.py \
  SelfEvolvingHarnessTS/tests/test_frozen_action_surfaces.py \
  SelfEvolvingHarnessTS/tests/test_benchmark_dev_eval.py \
  -q --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: `73 passed`, exit code 0. The same tests were observed as 71 passes plus two passes after correcting the pytest temp directory; no assertion failed.

- [ ] **Step 2: Create the immutable pre-refactor tag**

Run:

```bash
git -C /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS tag -a \
  pre-architecture-convergence-2026-07-17 \
  1e75305770815c256d5b295b7ad6b8cb6cffe4b4 \
  -m "Baseline before architecture convergence"
git -C /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS rev-list -n 1 \
  pre-architecture-convergence-2026-07-17
```

Expected final line:

```text
1e75305770815c256d5b295b7ad6b8cb6cffe4b4
```

- [ ] **Step 3: Add the exact baseline manifest**

Create `artifacts/manifests/architecture_convergence_baseline.json` with:

```json
{
  "schema_version": "architecture-convergence-baseline/1",
  "source_commit": "1e75305770815c256d5b295b7ad6b8cb6cffe4b4",
  "source_tag": "pre-architecture-convergence-2026-07-17",
  "interpreter": "D:/Anaconda_envs/envs/project/python.exe",
  "python": "3.10.19",
  "dependencies": {
    "numpy": "2.2.6",
    "scipy": "1.15.2",
    "sklearn": "1.7.2"
  },
  "targeted_test_result": {
    "passed": 73,
    "failed": 0,
    "errors": 0
  },
  "h_ref": {
    "state_sha": "4e7e4ac5b40c941d",
    "det_ladder_sha": [
      "a6a6db644a7b61c0",
      "c0f66a51e987f8a7",
      "bee33065e1b25757"
    ],
    "fixed_probe": {
      "u0": {
        "choice_sha": "a6a6db644a7b61c0",
        "artifact_sha256": "395193575038668d833b9cbba32b1f2a6ba486cac492ac19e226c2498da41c00",
        "realized_pool_size": 8
      },
      "u1": {
        "choice_sha": "a6a6db644a7b61c0",
        "artifact_sha256": "3734fc053b74086f7d49b18d13ce6b7f0452b1fc24980467870afb1ff2816b19",
        "realized_pool_size": 8
      },
      "u2": {
        "choice_sha": "a6a6db644a7b61c0",
        "artifact_sha256": "23f7c814d104764da16d8a0c62649bf0f97e8dbed95f2b36d4d2997a005dde15",
        "realized_pool_size": 8
      }
    }
  }
}
```

- [ ] **Step 4: Add the historical recovery policy**

Create `experiments/archive/README.md` with:

```markdown
# Historical experiment recovery

The current tree contains only active code and concise evidence pointers. Full P0-P6,
E32, confirmatory, and legacy harness implementations are recovered from Git tag
`pre-architecture-convergence-2026-07-17`.

Archived code must not be copied back into an importable package. Promote a still-useful
contract or mechanism into the canonical architecture with characterization tests instead.
Frozen scientific evidence remains under `artifacts/frozen/`, `artifacts/manifests/`, and
the existing signed benchmark result directories until their physical artifact migration
is separately approved.
```

- [ ] **Step 5: Validate and commit only the baseline files**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS
python -m json.tool artifacts/manifests/architecture_convergence_baseline.json >/dev/null
git diff --check -- artifacts/manifests/architecture_convergence_baseline.json experiments/archive/README.md
git add artifacts/manifests/architecture_convergence_baseline.json experiments/archive/README.md
git commit -m "chore: record architecture convergence baseline"
```

Expected: commit succeeds; unrelated `.gitignore`, spec, zip, or cache changes are not staged.

---

### Task 2: Promote TaskSpec into the Canonical Contract Package

**Files:**

- Create: `contracts/__init__.py`
- Create: `contracts/task.py`
- Modify: `policy/task_spec.py`
- Test: `tests/contracts/test_task_contract.py`

**Interfaces:**

- Consumes: current `policy.task_spec` classes and helper constructors.
- Produces: `contracts.task.MetricSpec`, `contracts.task.TaskSpec`, `forecast_task_spec_v1`, `classification_task_spec_v1`, and `anomaly_task_spec_v1`; legacy imports resolve to the same class objects.

- [ ] **Step 1: Write the failing identity test**

Create `tests/contracts/test_task_contract.py`:

```python
from SelfEvolvingHarnessTS.contracts.task import (
    MetricSpec,
    TaskSpec,
    anomaly_task_spec_v1,
    classification_task_spec_v1,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.policy import task_spec as legacy


def test_legacy_task_contract_is_the_canonical_contract():
    assert legacy.MetricSpec is MetricSpec
    assert legacy.TaskSpec is TaskSpec
    assert legacy.forecast_task_spec_v1 is forecast_task_spec_v1
    assert legacy.classification_task_spec_v1 is classification_task_spec_v1
    assert legacy.anomaly_task_spec_v1 is anomaly_task_spec_v1


def test_canonical_task_sha_matches_legacy_semantics():
    task = forecast_task_spec_v1(horizon=12)
    assert task.to_dict() == {
        "task_type": "forecast",
        "target_semantics": "future_values",
        "label_availability": "history_only",
        "metric": {"name": "nRMSE", "direction": "lower_is_better"},
        "horizon": 12,
        "downstream_model_class": "dlinear_shared",
        "forbidden_modifications": [],
    }
    assert len(task.sha()) == 16
```

- [ ] **Step 2: Run the test and verify the canonical package is absent**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/contracts/test_task_contract.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: collection fails with `ModuleNotFoundError: No module named 'SelfEvolvingHarnessTS.contracts'`.

- [ ] **Step 3: Promote the implementation and replace the old module with re-exports**

Copy `policy/task_spec.py` byte-for-byte to `contracts/task.py`. Its relative operator import remains valid because `contracts/` and `policy/` are sibling packages.

Create `contracts/__init__.py`:

```python
"""Stable contracts shared by methods, runtime, and evaluation."""

from .task import (
    LABEL_AVAILABILITY,
    TARGET_SEMANTICS_BY_TASK,
    TASK_TYPES,
    MetricSpec,
    TaskSpec,
    anomaly_task_spec_v1,
    classification_task_spec_v1,
    forecast_task_spec_v1,
)

__all__ = [
    "LABEL_AVAILABILITY",
    "TARGET_SEMANTICS_BY_TASK",
    "TASK_TYPES",
    "MetricSpec",
    "TaskSpec",
    "anomaly_task_spec_v1",
    "classification_task_spec_v1",
    "forecast_task_spec_v1",
]
```

Replace `policy/task_spec.py` with:

```python
"""Compatibility import for the canonical task contract.

New code imports :mod:`SelfEvolvingHarnessTS.contracts.task` directly.
"""

from ..contracts.task import *  # noqa: F401,F403
from ..contracts.task import __all__
```

Add the same explicit `__all__` list shown in `contracts/__init__.py` to the bottom of `contracts/task.py`.

- [ ] **Step 4: Run canonical and legacy task tests**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/contracts/test_task_contract.py \
  SelfEvolvingHarnessTS/tests/test_task_spec.py \
  SelfEvolvingHarnessTS/tests/test_policy_contract.py \
  SelfEvolvingHarnessTS/tests/test_benchmark_method_api.py \
  -q --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the task**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS
git diff --check -- contracts policy/task_spec.py tests/contracts/test_task_contract.py
git add contracts/__init__.py contracts/task.py policy/task_spec.py tests/contracts/test_task_contract.py
git commit -m "refactor: promote canonical task contract"
```

---

### Task 3: Define Canonical Program and Method Result Contracts

**Files:**

- Create: `contracts/program.py`
- Create: `contracts/method.py`
- Modify: `contracts/__init__.py`
- Test: `tests/contracts/test_method_contract.py`

**Interfaces:**

- Consumes: `contracts.task.TaskSpec`, NumPy arrays.
- Produces: `Program`, `ProgramStep`, `PreparationRequest`, `PreparedSeries`, `PreparationResult`, `PreparationStatus`, `ExecutionReceipt`, and `Method.prepare(request) -> PreparationResult`.

- [ ] **Step 1: Write failing contract tests**

Create `tests/contracts/test_method_contract.py`:

```python
import numpy as np
import pytest

from SelfEvolvingHarnessTS.contracts.method import (
    ExecutionReceipt,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1


def test_program_identity_is_mapping_order_independent():
    left = Program.from_steps([("denoise_savgol", {"window": 11, "order": 3})], source="det")
    right = Program.from_steps([("denoise_savgol", {"order": 3, "window": 11})], source="det")
    assert left.sha() == right.sha()
    assert left.execution_steps() == [("denoise_savgol", {"order": 3, "window": 11})]


def test_request_and_result_own_array_copies():
    raw = np.array([1.0, np.nan, 3.0])
    request = PreparationRequest("u0", raw, forecast_task_spec_v1(horizon=1), {})
    raw[0] = 99.0
    assert request.values[0] == 1.0

    prepared = PreparedSeries("u0", np.array([1.0, 2.0, 3.0]), (), "original_units")
    result = PreparationResult(
        status=PreparationStatus.PREPARED,
        prepared=prepared,
        program=None,
        receipt=ExecutionReceipt(ok=True),
    )
    assert result.status is PreparationStatus.PREPARED


def test_failed_result_cannot_carry_prepared_series():
    prepared = PreparedSeries("u0", np.ones(3), (), "original_units")
    with pytest.raises(ValueError, match="FAILED"):
        PreparationResult(
            status=PreparationStatus.FAILED,
            prepared=prepared,
            program=None,
            receipt=ExecutionReceipt(ok=False, error="boom"),
        )
```

- [ ] **Step 2: Run the tests and verify missing modules**

Run the new test with the canonical interpreter and repository basetemp. Expected: collection fails because `contracts.method` and `contracts.program` do not exist.

- [ ] **Step 3: Implement the immutable program contract**

Create `contracts/program.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProgramStep:
    op: str
    params: tuple[tuple[str, Any], ...] = ()

    @classmethod
    def from_mapping(cls, op: str, params: Mapping[str, Any] | None = None) -> "ProgramStep":
        if not isinstance(op, str) or not op or op != op.strip():
            raise ValueError("ProgramStep.op must be a canonical non-empty string")
        return cls(op=op, params=tuple(sorted(dict(params or {}).items())))

    def execution_pair(self) -> tuple[str, dict[str, Any]]:
        return self.op, dict(self.params)


@dataclass(frozen=True)
class Program:
    steps: tuple[ProgramStep, ...]
    source: str

    @classmethod
    def from_steps(
        cls,
        steps: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        source: str,
    ) -> "Program":
        if not isinstance(source, str) or not source or source != source.strip():
            raise ValueError("Program.source must be a canonical non-empty string")
        normalized = tuple(ProgramStep.from_mapping(op, params) for op, params in steps)
        if not normalized:
            raise ValueError("Program must contain at least one step")
        return cls(normalized, source)

    def execution_steps(self) -> list[tuple[str, dict[str, Any]]]:
        return [step.execution_pair() for step in self.steps]

    def sha(self) -> str:
        payload = [[step.op, dict(step.params)] for step in self.steps]
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


__all__ = ["Program", "ProgramStep"]
```

- [ ] **Step 4: Implement the canonical method envelope**

Create `contracts/method.py` with these complete public rules:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

import numpy as np

from .program import Program
from .task import TaskSpec


def _array_copy(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = array.copy()
    result.setflags(write=False)
    return result


def _uid(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("series_uid must be a canonical non-empty string")
    return value


class PreparationStatus(str, Enum):
    PREPARED = "prepared"
    ABSTAINED = "abstained"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionReceipt:
    ok: bool
    error: str = ""
    trace: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class PreparationRequest:
    series_uid: str
    values: np.ndarray
    task_spec: TaskSpec
    observed_pattern_spec: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_uid", _uid(self.series_uid))
        object.__setattr__(self, "values", _array_copy(self.values, "values"))
        object.__setattr__(self, "observed_pattern_spec", dict(self.observed_pattern_spec))
        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("task_spec must be TaskSpec")


@dataclass(frozen=True)
class PreparedSeries:
    series_uid: str
    values: np.ndarray
    operators: tuple[str, ...]
    units: str = "original_units"

    def __post_init__(self) -> None:
        object.__setattr__(self, "series_uid", _uid(self.series_uid))
        object.__setattr__(self, "values", _array_copy(self.values, "values"))
        object.__setattr__(self, "operators", tuple(self.operators))
        if self.units != "original_units":
            raise ValueError("units must be original_units")


@dataclass(frozen=True)
class PreparationResult:
    status: PreparationStatus
    prepared: PreparedSeries | None
    program: Program | None
    receipt: ExecutionReceipt

    def __post_init__(self) -> None:
        if self.status is PreparationStatus.FAILED and self.prepared is not None:
            raise ValueError("FAILED result cannot carry a prepared series")
        if self.status in {PreparationStatus.PREPARED, PreparationStatus.ABSTAINED} and self.prepared is None:
            raise ValueError(f"{self.status.name} result requires a prepared series")


class Method(Protocol):
    method_id: str

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        ...


__all__ = [
    "ExecutionReceipt",
    "Method",
    "PreparationRequest",
    "PreparationResult",
    "PreparationStatus",
    "PreparedSeries",
]
```

Update `contracts/__init__.py` to export every name in `contracts.method.__all__` and `contracts.program.__all__`.

- [ ] **Step 5: Run contract tests**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/contracts/test_task_contract.py \
  SelfEvolvingHarnessTS/tests/contracts/test_method_contract.py \
  -q --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: all contract tests pass.

- [ ] **Step 6: Commit the task**

Stage only `contracts/` and `tests/contracts/test_method_contract.py`; commit with `feat: add canonical method contracts`.

---

### Task 4: Promote the Canonical Runtime Executor

**Files:**

- Create: `runtime/__init__.py`
- Create: `runtime/errors.py`
- Create: `runtime/trace.py`
- Create: `runtime/executor.py`
- Modify: `sandbox/executor.py`
- Test: `tests/runtime/test_executor_contract.py`

**Interfaces:**

- Consumes: canonical operator registry and provenance recorder.
- Produces: `runtime.executor.ExecutionResult` and `run_pipeline`; legacy `sandbox.executor` exports the identical objects.

- [ ] **Step 1: Write the failing executor identity tests**

Create `tests/runtime/test_executor_contract.py`:

```python
import numpy as np

from SelfEvolvingHarnessTS.runtime import executor as canonical
from SelfEvolvingHarnessTS.runtime.errors import ContractError, ExecutionError, RuntimeFailure
from SelfEvolvingHarnessTS.sandbox import executor as legacy


def test_legacy_executor_is_canonical_executor():
    assert legacy.ExecutionResult is canonical.ExecutionResult
    assert legacy.run_pipeline is canonical.run_pipeline


def test_executor_records_unknown_operator_without_silent_fallback():
    result = canonical.run_pipeline([("does_not_exist", {})], np.arange(4.0))
    assert result.ok is False
    assert result.artifact is None
    assert result.error == "unknown op 'does_not_exist'"
    assert result.trace[-1]["error"] == "op not in registry"


def test_runtime_error_taxonomy_is_typed():
    assert issubclass(ContractError, RuntimeFailure)
    assert issubclass(ExecutionError, RuntimeFailure)
```

- [ ] **Step 2: Run the test and verify the runtime package is absent**

Expected: collection fails with `ModuleNotFoundError` for `SelfEvolvingHarnessTS.runtime`.

- [ ] **Step 3: Promote the existing executor without changing behavior**

Copy `sandbox/executor.py` to `runtime/executor.py`, then change only its two relative imports to:

```python
from ..operators.registry import TOOL_REGISTRY, canonicalize
from ..operators._provenance import record as _prov_record
```

Create `runtime/errors.py`:

```python
class RuntimeFailure(RuntimeError):
    """Base class for canonical runtime failures."""


class ContractError(RuntimeFailure):
    """A request, program, or prepared result violates a public contract."""


class ExecutionError(RuntimeFailure):
    """A declared runtime operation failed."""


class ProtocolViolation(RuntimeFailure):
    """A frozen or information-boundary protocol was violated."""


class InfrastructureError(RuntimeFailure):
    """Required infrastructure was unavailable."""
```

Create `runtime/trace.py`:

```python
from typing import TypedDict


class ExecutionTraceRecord(TypedDict):
    op: str
    canonical: str
    source: str
    ok: bool
    error: str


__all__ = ["ExecutionTraceRecord"]
```

Create `runtime/__init__.py`:

```python
"""Canonical program execution runtime."""

from .executor import ExecutionResult, run_pipeline

__all__ = ["ExecutionResult", "run_pipeline"]
```

Replace `sandbox/executor.py` with:

```python
"""Compatibility import for the canonical runtime executor."""

from ..runtime.executor import ExecutionResult, run_pipeline

__all__ = ["ExecutionResult", "run_pipeline"]
```

- [ ] **Step 4: Run executor and downstream tests**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/runtime/test_executor_contract.py \
  SelfEvolvingHarnessTS/tests/test_p6_surfaces.py \
  SelfEvolvingHarnessTS/tests/test_fast_path.py \
  -q --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: all selected tests pass.

- [ ] **Step 5: Verify line-ending attributes and commit**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS
git check-attr text eol -- runtime/executor.py runtime/errors.py runtime/trace.py
git diff --check -- runtime sandbox/executor.py tests/runtime/test_executor_contract.py
```

Expected: each Python file reports `text: set` and `eol: lf`. Commit the listed files with `refactor: promote canonical runtime executor`.

---

### Task 5: Promote the Frozen H_ref State and Configuration

**Files:**

- Create: `methods/__init__.py`
- Create: `methods/h_ref_v02/__init__.py`
- Create: `methods/h_ref_v02/config.py`
- Modify: `p6/harness_state.py`
- Test: extend `tests/runtime/test_fast_path_equivalence.py`

**Interfaces:**

- Consumes: frozen `p6.harness_state` behavior and the H_ref literals currently embedded in `p6.fast_path`/`harness.layers`.
- Produces: `HRefState`, `HRefEditError`, `default_state`, grammar constants, deterministic program literals, and operator defaults. Legacy P6 state names are aliases to the canonical classes.

- [ ] **Step 1: Write the failing state fingerprint test**

Create `tests/runtime/test_fast_path_equivalence.py` initially with:

```python
from SelfEvolvingHarnessTS.methods.h_ref_v02.config import (
    DET_PROGRAM_STEPS,
    H0_ALLOCATION,
    H0_EXPECTED_TOTAL_K,
    HRefState,
    default_state,
)
from SelfEvolvingHarnessTS.p6.harness_state import P6HarnessState


def test_h_ref_state_identity_and_frozen_fingerprint():
    state = default_state()
    assert P6HarnessState is HRefState
    assert state.sha() == "4e7e4ac5b40c941d"
    assert state.sampler.allocation == {"det": 3, "random": 5, "llm": 0} == H0_ALLOCATION
    assert state.sampler.expected_total == 8 == H0_EXPECTED_TOTAL_K
    assert DET_PROGRAM_STEPS == (
        (("impute_linear", {}),),
        (("impute_linear", {}), ("winsorize", {}), ("denoise_savgol", {})),
        (("impute_linear", {}), ("denoise_median", {"window": 9})),
    )
```

- [ ] **Step 2: Run the test and verify `methods.h_ref_v02` is absent**

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Promote and rename the frozen state implementation**

Copy `p6/harness_state.py` to `methods/h_ref_v02/config.py`. Apply these mechanical name changes throughout the copied file only:

```text
P6HarnessState -> HRefState
P6EditError -> HRefEditError
```

Do not change `SCHEMA_VERSION = "p6-harness-state/1"`; it is part of SHA identity.

Add these frozen literals after the existing H0 constants:

```python
HREF_OPERATOR_DEFAULTS = {
    "denoise_savgol": {"window": 11, "order": 3},
    "denoise_median": {"window": 5},
    "smooth_ma": {"window": 5},
    "stl_decompose": {"period": 0},
}
GUARD_OPS = ("winsorize", "outlier_iqr", "outlier_mad")
GRAMMAR_IMPUTERS = ("impute_linear", "impute_ema")
GRAMMAR_OUTLIERS = ("winsorize", "outlier_iqr", "outlier_mad")
GRAMMAR_DENOISERS = ("denoise_median", "smooth_ma", "denoise_savgol")
GRAMMAR_WINDOWS = (5, 9, 15, 25)
DET_PROGRAM_STEPS = (
    (("impute_linear", {}),),
    (("impute_linear", {}), ("winsorize", {}), ("denoise_savgol", {})),
    (("impute_linear", {}), ("denoise_median", {"window": 9})),
)

# Transitional names used only by legacy P6 imports.
P6HarnessState = HRefState
P6EditError = HRefEditError
```

Ensure `__all__` exports canonical names, the frozen literals, and the two transitional aliases.

Create `methods/__init__.py` as an empty namespace docstring. Create `methods/h_ref_v02/__init__.py` exporting `HRefState` and `default_state`.

- [ ] **Step 4: Replace the legacy state module with a true module alias**

Replace `p6/harness_state.py` with:

```python
"""Compatibility module alias for the canonical frozen H_ref configuration."""

import sys

from ..methods.h_ref_v02 import config as _canonical

sys.modules[__name__] = _canonical
```

This module alias preserves monkeypatch/module identity semantics; it is temporary and is removed with P6 in the historical-cleanup plan.

- [ ] **Step 5: Run state, edit-surface, cycle, and fingerprint tests**

Run the new equivalence test plus `test_p6_surfaces.py`, `test_p6_cycle.py`, `test_p6_miner.py`, and `test_frozen_action_surfaces.py` with the canonical interpreter and basetemp.

Expected: all tests pass; H0 SHA remains `4e7e4ac5b40c941d`.

- [ ] **Step 6: Commit the task**

Stage only the new method package, `p6/harness_state.py`, and the equivalence test. Commit with `refactor: promote frozen h-ref configuration`.

---

### Task 6: Promote the Sole Fast-Path Mechanics Implementation

**Files:**

- Create: `runtime/fast_path.py`
- Modify: `runtime/__init__.py`
- Modify: `p6/fast_path.py`
- Modify: `tests/runtime/test_fast_path_equivalence.py`

**Interfaces:**

- Consumes: `runtime.executor`, `methods.h_ref_v02.config`, canonical operators.
- Produces: the sole `Candidate`, candidate generation, selection, risk filtering, paired validators, `prepared_artifact`, and `run_fast_path` implementation. Legacy P6 fast-path imports resolve to the canonical module object.

- [ ] **Step 1: Add fixed behavior tests before moving the implementation**

Extend `tests/runtime/test_fast_path_equivalence.py`:

```python
import hashlib

import numpy as np

from SelfEvolvingHarnessTS.runtime.fast_path import det_ladder, prepared_artifact, run_fast_path


def _views():
    views = {}
    for i in range(3):
        t = np.arange(160, dtype=float)
        rng = np.random.default_rng(100 + i)
        values = np.sin(2.0 * np.pi * t / 24.0 + 0.7 * i) + 0.3 * i
        values = values + rng.normal(0.0, 0.05, t.size)
        if i == 1:
            values[10:14] = np.nan
        views[f"u{i}"] = values
    return views


def _sha(values):
    payload = np.ascontiguousarray(values, dtype="<f8").tobytes()
    return hashlib.sha256(payload).hexdigest()


def test_fast_path_matches_pre_refactor_fingerprints():
    assert [candidate.sha for candidate in det_ladder()] == [
        "a6a6db644a7b61c0",
        "c0f66a51e987f8a7",
        "bee33065e1b25757",
    ]
    views = _views()
    state = default_state()
    result = run_fast_path(views, state, state.sampler.expected_total)
    expected = {
        "u0": "395193575038668d833b9cbba32b1f2a6ba486cac492ac19e226c2498da41c00",
        "u1": "3734fc053b74086f7d49b18d13ce6b7f0452b1fc24980467870afb1ff2816b19",
        "u2": "23f7c814d104764da16d8a0c62649bf0f97e8dbed95f2b36d4d2997a005dde15",
    }
    for uid, choice in result.items():
        assert choice.sha == "a6a6db644a7b61c0"
        assert result.pool_stats[uid]["realized_pool_size"] == 8
        assert _sha(prepared_artifact(choice, views[uid])) == expected[uid]
```

Run this test now. Expected: collection fails because `runtime.fast_path` does not exist.

- [ ] **Step 2: Copy the complete fast-path mechanics into the runtime**

Copy `p6/fast_path.py` to `runtime/fast_path.py`. In the copied file:

1. replace `P6HarnessState` with `HRefState`;
2. replace `P6PairingError` with `FastPathPairingError`;
3. import `ExecutionResult` and `run_pipeline` from `.executor`;
4. import state types, `canonical_json`, grammar literals, deterministic program literals, and `HREF_OPERATOR_DEFAULTS` from `..methods.h_ref_v02.config`; this is the Phase-2 frozen-baseline composition edge, not a legacy dependency, and the later TTHA H0 plan must replace it with an injected fast-path definition before TTHA gains new grammar;
5. remove the `harness.layers.minimal_l2` import;
6. make `_operator_defaults()` return a copied `HREF_OPERATOR_DEFAULTS` mapping;
7. make `det_ladder()` build candidates from `DET_PROGRAM_STEPS`;
8. add transitional alias `P6PairingError = FastPathPairingError` and export both names.

The resulting key snippets must be:

```python
from .executor import ExecutionResult, run_pipeline
from ..methods.h_ref_v02.config import (
    DET_PROGRAM_STEPS,
    GRAMMAR_DENOISERS,
    GRAMMAR_IMPUTERS,
    GRAMMAR_OUTLIERS,
    GRAMMAR_WINDOWS,
    GUARD_OPS,
    HREF_OPERATOR_DEFAULTS,
    P0_FEATURE_ALLOWLIST,
    PRESET_SCOPE_FEATURE,
    HRefState,
    RiskRuleSpec,
    canonical_json,
)


@lru_cache(maxsize=1)
def _operator_defaults() -> Mapping[str, Dict[str, Any]]:
    return {name: dict(params) for name, params in HREF_OPERATOR_DEFAULTS.items()}


def det_ladder() -> List[Candidate]:
    return [make_candidate(list(program), source="det") for program in DET_PROGRAM_STEPS]
```

- [ ] **Step 3: Replace `p6.fast_path` with a module alias**

Replace `p6/fast_path.py` with:

```python
"""Compatibility module alias for the canonical fast-path runtime."""

import sys

from ..runtime import fast_path as _canonical

sys.modules[__name__] = _canonical
```

Update `runtime/__init__.py` to continue exporting executor names; do not wildcard-import fast path because that would make importing the executor eagerly construct H_ref dependencies.

- [ ] **Step 4: Run fast-path equivalence and all P6 compatibility tests**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/runtime/test_fast_path_equivalence.py \
  SelfEvolvingHarnessTS/tests/test_p6_surfaces.py \
  SelfEvolvingHarnessTS/tests/test_p6_cycle.py \
  SelfEvolvingHarnessTS/tests/test_p6_fixwave.py \
  SelfEvolvingHarnessTS/tests/test_p6_runners.py \
  SelfEvolvingHarnessTS/tests/test_frozen_action_surfaces.py \
  -q --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: all selected tests pass; fixed digests match exactly; monkeypatch tests continue working because the legacy path aliases the canonical module object.

- [ ] **Step 5: Verify canonical dependency and line-ending constraints**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS
rg -n '^(from|import) .*\b(p6|harness|policy|sandbox|evaluation)\b' runtime/fast_path.py
git check-attr text eol -- runtime/fast_path.py methods/h_ref_v02/config.py
git diff --check -- runtime/fast_path.py p6/fast_path.py tests/runtime/test_fast_path_equivalence.py
```

Expected: the first command returns no import or dependency edge to legacy packages; both Python files report `text: set`, `eol: lf`; diff check is silent.

- [ ] **Step 6: Commit the task**

Commit the task files with `refactor: promote canonical fast path`.

---

### Task 7: Add the Canonical H_ref Method and Benchmark Compatibility Boundary

**Files:**

- Create: `methods/h_ref_v02/method.py`
- Modify: `methods/h_ref_v02/__init__.py`
- Create: `evaluation/__init__.py`
- Create: `evaluation/benchmark_v02/__init__.py`
- Create: `evaluation/benchmark_v02/method_compat.py`
- Modify: `benchmark/baselines.py`
- Modify: `benchmark/dev_eval.py`
- Test: `tests/integration/test_h_ref_method.py`

**Interfaces:**

- Consumes: canonical Method contract, H_ref config, and runtime fast path.
- Produces: `HRefV02Method.prepare`, benchmark `BenchmarkMethodAdapter`, and `run_h_ref_batch`; benchmark modules no longer import P6.

- [ ] **Step 1: Write failing canonical/benchmark equivalence tests**

Create `tests/integration/test_h_ref_method.py`:

```python
import numpy as np

from SelfEvolvingHarnessTS.benchmark.method_api import MethodSeriesView
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationStatus
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.method_compat import BenchmarkMethodAdapter
from SelfEvolvingHarnessTS.methods.h_ref_v02.method import HRefV02Method


def test_canonical_h_ref_and_benchmark_adapter_are_equivalent():
    values = np.sin(np.arange(160, dtype=float) / 8.0)
    values[10:14] = np.nan
    task = forecast_task_spec_v1(horizon=12)
    method = HRefV02Method()

    canonical = method.prepare(PreparationRequest("u0", values, task, {}))
    benchmark = BenchmarkMethodAdapter(method).prepare(
        MethodSeriesView("u0", values), task, {}
    )

    assert canonical.status is PreparationStatus.PREPARED
    assert np.array_equal(canonical.prepared.values, benchmark.values, equal_nan=True)
    assert canonical.prepared.operators == benchmark.operators
    assert benchmark.units == "original_units"
```

Run the test. Expected: collection fails because the method/evaluation packages do not yet exist.

- [ ] **Step 2: Implement `HRefV02Method`**

Create `methods/h_ref_v02/method.py` so that:

- `method_id = "h_ref_v02"`;
- the default state is `default_state()`;
- `prepare()` invokes canonical `run_fast_path` for one UID;
- abstention returns an explicit `ABSTAINED` identity result;
- selected candidates execute through `execute_candidate` exactly once for the returned receipt;
- execution failure returns `FAILED` without a prepared series;
- a successful candidate returns canonical `Program`, `PreparedSeries`, and `ExecutionReceipt`.

Use this complete method body:

```python
class HRefV02Method:
    method_id = "h_ref_v02"

    def __init__(self, state: HRefState | None = None) -> None:
        self._state = default_state() if state is None else state

    def prepare(self, request: PreparationRequest) -> PreparationResult:
        budget = self._state.sampler.expected_total
        choices = run_fast_path({request.series_uid: request.values}, self._state, budget)
        choice = choices[request.series_uid]
        if choice is None:
            prepared = PreparedSeries(request.series_uid, request.values, (), "original_units")
            return PreparationResult(
                PreparationStatus.ABSTAINED,
                prepared,
                None,
                ExecutionReceipt(ok=True),
            )

        execution = execute_candidate(choice, request.values)
        receipt = ExecutionReceipt(
            ok=execution.ok,
            error=execution.error,
            trace=tuple(dict(row) for row in execution.trace),
        )
        program = Program.from_steps(choice.program_steps, source=choice.source)
        if not execution.ok or execution.artifact is None:
            return PreparationResult(PreparationStatus.FAILED, None, program, receipt)

        prepared = PreparedSeries(
            request.series_uid,
            execution.artifact,
            choice.op_names(),
            "original_units",
        )
        return PreparationResult(PreparationStatus.PREPARED, prepared, program, receipt)
```

Add the imports required by this body from `contracts.method`, `contracts.program`, `runtime.fast_path`, and `methods.h_ref_v02.config`. Export `HRefV02Method` from the method package.

- [ ] **Step 3: Implement the frozen benchmark compatibility module**

Create namespace `__init__.py` files and `evaluation/benchmark_v02/method_compat.py` with:

```python
from __future__ import annotations

from typing import Mapping

from ...benchmark.method_api import MethodSeriesView, PreparedSeries as BenchmarkPreparedSeries
from ...contracts.method import Method, PreparationRequest, PreparationStatus
from ...contracts.task import TaskSpec
from ...methods.h_ref_v02.config import default_state
from ...runtime.fast_path import run_fast_path


class BenchmarkMethodAdapter:
    def __init__(self, method: Method) -> None:
        self._method = method
        self.method_id = method.method_id

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: TaskSpec,
        observed_pattern_spec: Mapping[str, float],
    ) -> BenchmarkPreparedSeries:
        result = self._method.prepare(
            PreparationRequest(
                series_view.series_uid,
                series_view.degraded_inner_train,
                task_spec,
                observed_pattern_spec,
            )
        )
        if result.status is PreparationStatus.FAILED or result.prepared is None:
            raise RuntimeError(f"canonical method {self.method_id} failed: {result.receipt.error}")
        return BenchmarkPreparedSeries(
            series_uid=result.prepared.series_uid,
            values=result.prepared.values,
            operators=result.prepared.operators,
            units=result.prepared.units,
        )


def run_h_ref_batch(views):
    state = default_state()
    return run_fast_path(views, state, state.sampler.expected_total)


__all__ = ["BenchmarkMethodAdapter", "run_h_ref_batch"]
```

This is the one explicitly allowed concrete-method compatibility boundary. General benchmark code imports only `run_h_ref_batch` from this module.

- [ ] **Step 4: Remove P6 imports from benchmark modules**

In `benchmark/baselines.py`, replace:

```python
from ..p6.fast_path import prepared_artifact, run_fast_path
```

with:

```python
from ..runtime.fast_path import prepared_artifact, run_fast_path
```

In `benchmark/dev_eval.py`, replace both P6 imports with:

```python
from ..evaluation.benchmark_v02.method_compat import run_h_ref_batch
```

Replace:

```python
state = default_state()
started = time.perf_counter()
h_ref_choices = run_fast_path(inner, state, state.sampler.expected_total)
```

with:

```python
started = time.perf_counter()
h_ref_choices = run_h_ref_batch(inner)
```

Do not otherwise edit benchmark evaluation logic.

- [ ] **Step 5: Run method, baseline, dev-eval, and frozen tests**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/integration/test_h_ref_method.py \
  SelfEvolvingHarnessTS/tests/test_benchmark_baselines.py \
  SelfEvolvingHarnessTS/tests/test_benchmark_dev_eval.py \
  SelfEvolvingHarnessTS/tests/test_run_benchmark.py \
  SelfEvolvingHarnessTS/tests/test_frozen_action_surfaces.py \
  -q --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: all selected tests pass.

- [ ] **Step 6: Prove active benchmark code no longer imports P6**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS
rg -n '(^|\.)p6(\.|\s|$)|from \.\.p6' benchmark evaluation
```

Expected: no matches.

- [ ] **Step 7: Commit the task**

Commit only the method, evaluation compatibility, benchmark import changes, and integration test with `refactor: route benchmark through canonical h-ref`.

---

### Task 8: Enforce Dependency Rules and Sign Off Phase 2

**Files:**

- Create: `tests/architecture/test_dependency_rules.py`
- Modify: `artifacts/manifests/architecture_convergence_baseline.json` only if review fixes require an appended phase-2 verification record; do not rewrite baseline fields.

**Interfaces:**

- Consumes: canonical package structure completed in Tasks 2–7.
- Produces: executable architecture constraints and Phase 2 verification evidence.

- [ ] **Step 1: Write the architecture test**

Create `tests/architecture/test_dependency_rules.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            dots = "." * node.level
            found.add(dots + (node.module or ""))
    return found


@pytest.mark.parametrize(
    ("area", "forbidden"),
    [
        ("runtime", {"p6", "harness", "policy", "sandbox", "evaluation"}),
        ("methods", {"p6", "experiments", "evaluation"}),
        ("evaluation", {"p6", "experiments"}),
        ("benchmark", {"p6"}),
    ],
)
def test_active_packages_do_not_import_forbidden_architecture_branches(area, forbidden):
    violations = []
    for path in sorted((ROOT / area).rglob("*.py")):
        for name in _imports(path):
            parts = {part for part in name.lstrip(".").split(".") if part}
            blocked = sorted(parts & forbidden)
            if blocked:
                violations.append(f"{path.relative_to(ROOT)}: {name} -> {blocked}")
    assert violations == []


def test_archived_experiments_are_not_imported_by_active_python():
    violations = []
    for area in ("contracts", "runtime", "methods", "evaluation", "benchmark", "operators"):
        for path in sorted((ROOT / area).rglob("*.py")):
            for name in _imports(path):
                if "experiments" in name.split("."):
                    violations.append(f"{path.relative_to(ROOT)}: {name}")
    assert violations == []
```

- [ ] **Step 2: Run architecture and focused equivalence tests**

Run the new architecture test, all tests under `tests/contracts`, `tests/runtime`, and `tests/integration`, plus `test_p6_surfaces.py`, `test_benchmark_baselines.py`, `test_benchmark_dev_eval.py`, and `test_frozen_action_surfaces.py` with the canonical interpreter and basetemp.

Expected: all selected tests pass; architecture violations list is empty.

- [ ] **Step 3: Run the complete suite**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_architecture
```

Expected: exit code 0 with zero failures and zero errors. Record the actual pass count in the Phase 2 handoff; do not copy the stale count from documentation.

- [ ] **Step 4: Verify byte and import invariants**

Run:

```bash
cd /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS
git diff --check
git check-attr text eol -- \
  contracts/task.py contracts/program.py contracts/method.py \
  runtime/executor.py runtime/fast_path.py \
  methods/h_ref_v02/config.py methods/h_ref_v02/method.py
rg -n 'from .*p6|import .*p6' contracts runtime methods evaluation benchmark
git status --short
```

Expected: diff check is silent; all listed Python files report LF; the import scan has no matches; status contains only intentional task files and pre-existing user-owned untracked artifacts.

- [ ] **Step 5: Commit the architecture gate**

Stage only `tests/architecture/test_dependency_rules.py` and commit with `test: enforce architecture dependency rules`.

- [ ] **Step 6: Run the final engineering review gate**

At implementation completion, invoke `superpowers:requesting-code-review` once over the entire range from the Task 1 baseline commit through Task 8. This is an engineering correctness review, unrelated to the session's automatic approval mode. Review against the approved design spec, with explicit checks for:

- one active executor and one active fast-path implementation;
- H_ref state/candidate/artifact fingerprints;
- benchmark-v0.2 semantic and information-wall preservation;
- canonical modules importing no legacy branches;
- compatibility shims containing no independent implementation;
- unrelated user/Agent changes preserved;
- no generated cache, zip, or pytest temp content staged.

Resolve every High and Medium finding, rerun Steps 2–4, and commit fixes before declaring Phase 2 complete.

## Phase 2 Handoff

The handoff report must contain:

- commit range and pre-refactor tag target;
- focused and full-suite test counts from the canonical interpreter;
- final engineering-review findings and fix commits;
- H0 SHA, ladder SHAs, and fixed-probe output digest confirmation;
- remaining compatibility shims and their deletion phase;
- confirmation that TTHA/minipipe functionality was not implemented in this plan.
