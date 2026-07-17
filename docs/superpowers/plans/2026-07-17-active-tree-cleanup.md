# Active Tree Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the implementation baseline from 1,159 tracked files to fewer than 180 while preserving the canonical H_ref runtime, operators, and one runnable Benchmark-v0.2 evaluator.

**Architecture:** Historical source and results are recovered from Git tag `pre-architecture-convergence-2026-07-17`, not copied into an importable archive. Benchmark code is physically consolidated under `evaluation/benchmark_v02/`; its required bundled inputs live under that package, and immutable result/protocol evidence lives under `artifacts/frozen/benchmark_v02/`.

**Tech Stack:** Python 3.10.19, NumPy 2.2.6, SciPy 1.15.2, scikit-learn 1.7.2, PyTorch, pytest, Git.

## Global Constraints

- Functional completeness, component clarity, and iteration speed take priority over preserving historical test counts.
- Run tests with `/mnt/d/Anaconda_envs/envs/project/python.exe` from the parent directory containing `SelfEvolvingHarnessTS`.
- Always pass `--basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup` to pytest.
- Work only in branch `refactor/architecture-convergence-foundation` and its isolated worktree.
- Do not stage, revert, or overwrite the other Agent's changes in the main worktree.
- Full historical recovery must remain available at tag `pre-architecture-convergence-2026-07-17`, target commit `1e75305770815c256d5b295b7ad6b8cb6cffe4b4`.
- Preserve H_ref state SHA `4e7e4ac5b40c941d`, ladder SHAs `a6a6db644a7b61c0`, `c0f66a51e987f8a7`, and `bee33065e1b25757`, and all three fixed-probe artifact digests.
- Move frozen evidence byte-for-byte. Do not rewrite historical signed documents to update their embedded old paths.
- Delete only explicit repository paths or a Git-produced list whose count and contents were reviewed immediately before deletion.
- Do not create a second source archive under `experiments/archive/`; it contains only recovery instructions.
- The retained test suite is a functional suite. The former 1,035-test historical suite is not a completion criterion.

## Final File Map

**Active code:**

- `contracts/` — task, program, and method contracts.
- `operators/` — sole operator registry and implementations.
- `conditioning/` — shared period/key functionality required by operators.
- `runtime/` — sole executor and fast path.
- `methods/h_ref_v02/` — frozen reference method.
- `evaluation/benchmark_v02/` — complete auxiliary evaluator, package CLI, adapter, models, and required bundled inputs.

**Evidence and recovery:**

- `artifacts/manifests/architecture_convergence_baseline.json`
- `artifacts/manifests/active_tree_cleanup.json`
- `artifacts/frozen/benchmark_v02/`
- `experiments/archive/README.md`
- current architecture/cleanup design and plan documents.

**Focused tests:**

- `tests/architecture/`
- `tests/contracts/`
- `tests/operators/`
- `tests/runtime/`
- `tests/integration/`
- `tests/frozen_protocol/`

---

### Task 1: Record the Cleanup Boundary

**Files:**

- Create: `artifacts/manifests/active_tree_cleanup.json`

**Interfaces:**

- Consumes: recovery tag and approved cleanup design.
- Produces: machine-readable ownership, deletion, and relocation boundary for later tasks.

- [ ] **Step 1: Reconfirm the recovery tag and pre-cleanup count**

Run:

```bash
git rev-list -n 1 pre-architecture-convergence-2026-07-17
git ls-files | wc -l
git status --short
```

Expected: tag target `1e75305770815c256d5b295b7ad6b8cb6cffe4b4`, tracked count `1159`, and a clean worktree.

- [ ] **Step 2: Create the cleanup manifest**

Create `artifacts/manifests/active_tree_cleanup.json` with:

```json
{
  "schema_version": "active-tree-cleanup/1",
  "recovery_tag": "pre-architecture-convergence-2026-07-17",
  "recovery_commit": "1e75305770815c256d5b295b7ad6b8cb6cffe4b4",
  "foundation_commit": "bd0bd1bbf00014da2775b5eca95665a4fbd39b4b",
  "before_tracked_files": 1159,
  "target_max_tracked_files": 179,
  "active_packages": [
    "contracts",
    "operators",
    "conditioning",
    "runtime",
    "methods",
    "evaluation"
  ],
  "retired_namespaces": [
    "benchmark",
    "config",
    "diagnostics",
    "evaluators",
    "fast_path",
    "harness",
    "llm",
    "memory",
    "models",
    "p6",
    "policy",
    "sandbox",
    "slow_path"
  ],
  "benchmark_relocation": {
    "code": {
      "from": "benchmark",
      "to": "evaluation/benchmark_v02"
    },
    "evidence": {
      "from": "results/Benchmark_v0_2",
      "to": "artifacts/frozen/benchmark_v02",
      "content_policy": "git-rename-byte-identical",
      "files": [
        "TD_VERDICT.json",
        "TD_VERDICT.md",
        "TD_VERDICT_ADDENDUM.md",
        "baseline_report.md",
        "benchmark_manifest_v0.yaml",
        "corruption_grid.json",
        "covid_sensitivity.json",
        "data_card.md",
        "dataset_manifest.json",
        "dev_discrimination_report.json",
        "dev_per_dose_report.json",
        "dev_program_losses.jsonl",
        "dev_repeat_losses.jsonl",
        "h_ref_self_harm_diagnosis.json",
        "metr_la_spatial_blocks.json",
        "pool_code_pin_reconciliation.json",
        "probe_summary.json",
        "program_pool.json",
        "series_registry.jsonl",
        "split_manifest.json",
        "support_a_subsplit.json",
        "training_evaluation_protocol.md",
        "virgin_ledger.jsonl"
      ]
    },
    "bundled_data": {
      "to": "evaluation/benchmark_v02/data/legacy",
      "files": [
        "monash_clean.meta.jsonl",
        "monash_clean.npz",
        "u_admission_v2_traffic_hourly.json"
      ]
    }
  }
}
```

- [ ] **Step 3: Validate and commit the manifest**

Run:

```bash
python3 -m json.tool artifacts/manifests/active_tree_cleanup.json >/dev/null
git diff --check -- artifacts/manifests/active_tree_cleanup.json
git add artifacts/manifests/active_tree_cleanup.json
git commit -m "chore: record active tree cleanup boundary"
```

Expected: JSON validation and commit succeed.

---

### Task 2: Consolidate Benchmark Code and Required Inputs

**Files:**

- Create: `tests/frozen_protocol/test_benchmark_v02_smoke.py`
- Create: `evaluation/benchmark_v02/models.py`
- Create: `evaluation/benchmark_v02/__main__.py` by moving `run_benchmark.py`
- Modify: `evaluation/benchmark_v02/__init__.py`
- Modify/Move: every Python module under `benchmark/` into `evaluation/benchmark_v02/`
- Modify: `tests/integration/test_h_ref_method.py`
- Move: `data/benchmark_v0/acquisition_manifest.json` to `evaluation/benchmark_v02/data/acquisition_manifest.json`
- Move: `data/benchmark_v0/incoming/README.md` to `evaluation/benchmark_v02/data/incoming/README.md`
- Move: `data/_artifacts/monash_clean.meta.jsonl` and `monash_clean.npz` to `evaluation/benchmark_v02/data/legacy/`
- Move: `results/Stage2/P6Probes/u_admission_v2_traffic_hourly.json` to `evaluation/benchmark_v02/data/legacy/`

**Interfaces:**

- Consumes: current frozen `benchmark` package and `BenchmarkMethodAdapter`.
- Produces: `python -m SelfEvolvingHarnessTS.evaluation.benchmark_v02`, local `BenchmarkMethodAdapter`, and a self-contained benchmark module closure.

- [ ] **Step 1: Write the failing package-ownership smoke test**

Create `tests/frozen_protocol/`, then create
`tests/frozen_protocol/test_benchmark_v02_smoke.py`:

```python
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np

from SelfEvolvingHarnessTS.evaluation import benchmark_v02
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.__main__ import build_parser, main
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.programs import apply_program


MODULES = (
    "aggregate",
    "baselines",
    "corruption",
    "datasets",
    "dev_eval",
    "ingestion",
    "ledger",
    "materialize",
    "method_api",
    "method_compat",
    "metrics",
    "models",
    "power",
    "probe",
    "programs",
    "registry",
    "report",
    "runner",
    "sources",
    "spatial",
    "split",
    "trainers",
    "workspace",
)


def test_benchmark_has_one_importable_package_owner():
    prefix = "SelfEvolvingHarnessTS.evaluation.benchmark_v02"
    for module in MODULES:
        assert importlib.import_module(f"{prefix}.{module}") is not None


def test_package_cli_exposes_the_frozen_phases_and_dispatches(tmp_path):
    commands = build_parser()._subparsers._group_actions[0].choices
    assert set(commands) == {
        "acquire",
        "probe",
        "freeze",
        "dry-run",
        "confirm",
        "dev-eval",
        "campaign-freeze",
        "final-eval",
    }
    calls = []
    handlers = {"probe": lambda args: calls.append((args.root, args.out)) or 0}
    assert main(
        ["probe", "--root", str(tmp_path / "data"), "--out", str(tmp_path / "out")],
        handlers=handlers,
    ) == 0
    assert calls == [(str(tmp_path / "data"), str(tmp_path / "out"))]


def test_benchmark_bundled_inputs_and_program_smoke():
    data = Path(benchmark_v02.__file__).resolve().parent / "data"
    assert (data / "acquisition_manifest.json").is_file()
    assert (data / "legacy" / "monash_clean.meta.jsonl").is_file()
    assert (data / "legacy" / "monash_clean.npz").is_file()
    assert (data / "legacy" / "u_admission_v2_traffic_hourly.json").is_file()
    values = np.array([1.0, np.nan, 3.0, 4.0])
    raw = apply_program("raw", values, period=2)
    assert np.array_equal(raw, values, equal_nan=True)
```

- [ ] **Step 2: Run the smoke test and verify RED**

Run from the worktree parent:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_benchmark_v02_smoke.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: collection fails because `evaluation.benchmark_v02.__main__` and `.programs` do not exist.

- [ ] **Step 3: Move benchmark modules and bundled inputs**

Run from the repository root:

```bash
mkdir -p evaluation/benchmark_v02/data/incoming evaluation/benchmark_v02/data/legacy tests/frozen_protocol
git mv benchmark/aggregate.py benchmark/baselines.py benchmark/corruption.py \
  benchmark/datasets.py benchmark/dev_eval.py benchmark/ingestion.py benchmark/ledger.py \
  benchmark/materialize.py benchmark/method_api.py benchmark/metrics.py benchmark/power.py \
  benchmark/probe.py benchmark/programs.py benchmark/registry.py benchmark/report.py \
  benchmark/runner.py benchmark/sources.py benchmark/spatial.py benchmark/split.py \
  benchmark/trainers.py benchmark/workspace.py evaluation/benchmark_v02/
git mv run_benchmark.py evaluation/benchmark_v02/__main__.py
git mv data/benchmark_v0/acquisition_manifest.json \
  evaluation/benchmark_v02/data/acquisition_manifest.json
git mv data/benchmark_v0/incoming/README.md \
  evaluation/benchmark_v02/data/incoming/README.md
git mv data/_artifacts/monash_clean.meta.jsonl data/_artifacts/monash_clean.npz \
  evaluation/benchmark_v02/data/legacy/
git mv results/Stage2/P6Probes/u_admission_v2_traffic_hourly.json \
  evaluation/benchmark_v02/data/legacy/u_admission_v2_traffic_hourly.json
git rm benchmark/__init__.py
```

Expected: every move is registered by Git; only the now-empty top-level `benchmark/` directory disappears.

- [ ] **Step 4: Make the consolidated package canonical**

Replace `evaluation/benchmark_v02/__init__.py` with the constants from the former benchmark initializer plus the adapter exports:

```python
"""Frozen Benchmark-v0.2 evaluator and canonical-method adapter."""
from __future__ import annotations

BENCHMARK_VERSION = "benchmark-v0.2"
KNOWN_BENCHMARK_VERSIONS = ("benchmark-v0", "benchmark-v0.1", "benchmark-v0.2")

HEADLINE_LOOKBACK = 48
HEADLINE_HORIZON = 48
HEADLINE_MIN_LENGTH = 207

MODEL_SEEDS = (0, 1, 2)
CORRUPTION_REPLICATES = (0, 1)

HARM_THRESHOLD = 0.05
HARM_THRESHOLD_KIND = "conventional"
SATURATION_GAP = 0.02
SATURATION_GAP_KIND = "conventional"

BOOTSTRAP_B = 2000
BOOTSTRAP_MASTER_SEED = 20260713

DESIGN_COMMIT = "9e57da9"
EXTERNAL_ADDENDUM_SHA256 = (
    "468c65fbcb36f48a47a351597f99d9ccebd876fff39d3378923500a8c3ed45ff"
)

from .method_compat import BenchmarkMethodAdapter, run_h_ref_batch  # noqa: E402

__all__ = [
    "BENCHMARK_VERSION",
    "BOOTSTRAP_B",
    "BOOTSTRAP_MASTER_SEED",
    "BenchmarkMethodAdapter",
    "CORRUPTION_REPLICATES",
    "DESIGN_COMMIT",
    "EXTERNAL_ADDENDUM_SHA256",
    "HARM_THRESHOLD",
    "HARM_THRESHOLD_KIND",
    "HEADLINE_HORIZON",
    "HEADLINE_LOOKBACK",
    "HEADLINE_MIN_LENGTH",
    "KNOWN_BENCHMARK_VERSIONS",
    "MODEL_SEEDS",
    "SATURATION_GAP",
    "SATURATION_GAP_KIND",
    "run_h_ref_batch",
]
```

Create `evaluation/benchmark_v02/models.py` with only the models the frozen trainer consumes:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinear(nn.Module):
    def __init__(self, L: int, H: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.lin_trend = nn.Linear(L, H)
        self.lin_season = nn.Linear(L, H)

    def forward(self, x):
        xp = F.pad(x.unsqueeze(1), (self.kernel // 2, self.kernel // 2), mode="replicate")
        trend = F.avg_pool1d(xp, self.kernel, stride=1).squeeze(1)[:, : x.size(1)]
        season = x - trend
        return self.lin_trend(trend) + self.lin_season(season)


class LSTMForecaster(nn.Module):
    def __init__(self, L: int, H: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, H)

    def forward(self, x):
        out, _ = self.lstm(x.unsqueeze(-1))
        return self.head(out[:, -1, :])


__all__ = ["DLinear", "LSTMForecaster"]
```

Apply these exact import/path changes:

```text
evaluation/benchmark_v02/baselines.py:
  ..operators -> ...operators
  ..runtime -> ...runtime
  ..policy.task_spec -> ...contracts.task

evaluation/benchmark_v02/method_api.py:
  ..operators -> ...operators
  ..policy.task_spec -> ...contracts.task

evaluation/benchmark_v02/programs.py:
  ..operators -> ...operators
  _project_root() parents[1] -> parents[2]
  "benchmark/programs.py" -> "evaluation/benchmark_v02/programs.py"

evaluation/benchmark_v02/trainers.py:
  ..evaluators._torch_models -> .models

evaluation/benchmark_v02/dev_eval.py:
  ..evaluation.benchmark_v02.method_compat -> .method_compat
  ..runtime.fast_path -> ...runtime.fast_path

evaluation/benchmark_v02/datasets.py:
  "benchmark/datasets.py" -> "evaluation/benchmark_v02/datasets.py"

evaluation/benchmark_v02/method_compat.py:
  ...benchmark.method_api -> .method_api

evaluation/benchmark_v02/__main__.py:
  .benchmark.runner -> .runner
  parser prog -> "python -m SelfEvolvingHarnessTS.evaluation.benchmark_v02"

tests/integration/test_h_ref_method.py:
  SelfEvolvingHarnessTS.benchmark.method_api ->
  SelfEvolvingHarnessTS.evaluation.benchmark_v02.method_api
```

In `evaluation/benchmark_v02/workspace.py`, define and use package-owned bundled data:

```python
_BUNDLED_LEGACY = Path(__file__).resolve().parent / "data" / "legacy"


def _probe_consumed_traffic() -> set[str]:
    path = _BUNDLED_LEGACY / "u_admission_v2_traffic_hourly.json"
    if not path.is_file():
        return set()
    payload = json.loads(path.read_text("utf-8"))
    return {str(value) for value in payload.get("all_probe_consumed_item_ids", [])}
```

Replace the old legacy metadata path with:

```python
metadata_path = _BUNDLED_LEGACY / "monash_clean.meta.jsonl"
```

Replace `_probe_consumed_traffic(project_root)` with `_probe_consumed_traffic()` and remove the now-unused `project_root` local.

- [ ] **Step 5: Run focused benchmark/H_ref tests and verify GREEN**

Run:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_benchmark_v02_smoke.py \
  SelfEvolvingHarnessTS/tests/integration/test_h_ref_method.py \
  SelfEvolvingHarnessTS/tests/runtime/test_fast_path_equivalence.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: all tests pass and fixed H_ref fingerprints remain unchanged.

- [ ] **Step 6: Prove the old package is no longer imported and commit**

Run:

```bash
rg -n 'SelfEvolvingHarnessTS\.benchmark|from \.{1,3}(benchmark|policy|evaluators)(\.| import)' \
  evaluation/benchmark_v02 tests/frozen_protocol tests/integration
git diff --check -- evaluation/benchmark_v02 tests/frozen_protocol
git add evaluation/benchmark_v02 tests/frozen_protocol tests/integration/test_h_ref_method.py
git commit -m "refactor: consolidate frozen benchmark evaluator"
```

Expected: the scan has no old-package imports; commit records moves plus focused changes.

---

### Task 3: Relocate Frozen Benchmark Evidence

**Files:**

- Create directory: `artifacts/frozen/benchmark_v02/`
- Move: all 23 files from `results/Benchmark_v0_2/`
- Move: benchmark design addendum and v0.2 preregistration document
- Modify: `tests/frozen_protocol/test_benchmark_v02_smoke.py`
- Modify: `.gitattributes` before staging the moves so their bytes remain unchanged.

**Interfaces:**

- Consumes: relocation list in `active_tree_cleanup.json`.
- Produces: one immutable evidence owner under `artifacts/frozen/benchmark_v02/`.

- [ ] **Step 1: Add the failing evidence-location test**

Append to `tests/frozen_protocol/test_benchmark_v02_smoke.py`:

```python
def test_frozen_benchmark_evidence_has_one_owner():
    root = Path(__file__).resolve().parents[2]
    cleanup = json.loads(
        (root / "artifacts" / "manifests" / "active_tree_cleanup.json").read_text("utf-8")
    )
    expected = set(cleanup["benchmark_relocation"]["evidence"]["files"])
    frozen = root / "artifacts" / "frozen" / "benchmark_v02"
    assert expected <= {path.name for path in frozen.iterdir() if path.is_file()}
    assert not (root / "results").joinpath("Benchmark_v0_2").exists()
```

- [ ] **Step 2: Run the test and verify RED**

Run from the worktree parent:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_benchmark_v02_smoke.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: failure because `artifacts/frozen/benchmark_v02/` does not exist.

- [ ] **Step 3: Move the evidence without rewriting it**

Before staging the moves, add this exact rule to `.gitattributes`:

```gitattributes
artifacts/frozen/benchmark_v02/** -text
```

Then run:

```bash
mkdir -p artifacts/frozen
git mv results/Benchmark_v0_2 artifacts/frozen/benchmark_v02
git mv docs/benchmark/Benchmark_v0_Forecast_Design_v3_Addendum_2026-07-13.md \
  artifacts/frozen/benchmark_v02/design_addendum.md
git mv docs/superpowers/specs/2026-07-14-benchmark-v0_2-prereg.md \
  artifacts/frozen/benchmark_v02/prereg.md
```

Expected: Git reports renames; the 23 result files retain byte-identical content.

- [ ] **Step 4: Verify evidence and commit**

Run:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/frozen_protocol/test_benchmark_v02_smoke.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
git diff --check -- artifacts/frozen tests/frozen_protocol
git diff --cached --summary -M100% | \
  rg 'results/Benchmark_v0_2|docs/benchmark|benchmark-v0_2-prereg'
test "$(git diff --cached --summary -M100% | rg -c 'rename .* \(100%\)')" -eq 25
git add .gitattributes artifacts/frozen tests/frozen_protocol
git commit -m "refactor: centralize frozen benchmark evidence"
```

Expected: smoke test passes; diff summary reports renames rather than rewritten artifacts.

---

### Task 4: Establish the Minimal Active Documentation and Operator Suite

**Files:**

- Create: `README.md`
- Create: `tests/operators/test_registry.py`
- Create: `tests/operators/test_conditioning.py`
- Create: `conditioning/thresholds.py`
- Modify: `conditioning/binning.py`
- Modify: `conditioning/distance.py`
- Modify: `conditioning/key.py`
- Modify: `tests/contracts/test_task_contract.py`
- Modify: `tests/runtime/test_executor_contract.py`
- Modify: `tests/runtime/test_fast_path_equivalence.py`
- Move: `tests/test_operator_integrity.py` to `tests/operators/test_operator_integrity.py`
- Move: `tests/test_period_shared.py` to `tests/operators/test_period_shared.py`
- Move: `tests/test_boundary_semantics.py` to `tests/operators/test_boundary_semantics.py`
- Modify: `experiments/archive/README.md`

**Interfaces:**

- Consumes: canonical package structure and recovery tag.
- Produces: concise onboarding, a focused operator contract suite, and no retained test or
  conditioning dependency on a compatibility package.

- [ ] **Step 1: Move focused operator tests and add registry coverage**

Run:

```bash
mkdir -p tests/operators
git mv tests/test_operator_integrity.py tests/operators/test_operator_integrity.py
git mv tests/test_period_shared.py tests/operators/test_period_shared.py
git mv tests/test_boundary_semantics.py tests/operators/test_boundary_semantics.py
```

Create `tests/operators/test_registry.py`:

```python
import numpy as np

from SelfEvolvingHarnessTS.operators.registry import TOOL_REGISTRY, canonicalize
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline


def test_registry_alias_and_executor_share_the_canonical_operator():
    assert canonicalize("fill_gaps") == "impute_linear"
    assert TOOL_REGISTRY["fill_gaps"] is TOOL_REGISTRY["impute_linear"]
    result = run_pipeline([("impute_linear", {})], np.array([1.0, np.nan, 3.0]))
    assert result.ok is True
    assert np.isfinite(result.artifact).all()
```

- [ ] **Step 2: Give active conditioning thresholds a local owner**

Create `conditioning/thresholds.py`:

```python
"""Thresholds required by the active conditioning feature layer."""

STRUCT_FEATS_DIM = 10
ALPHA_DISTANCE = 0.5
BIN_SNR_SPLIT_DB = 4.0
BIN_MISSING_ANY = 0.0
OUTLIER_MAD_K = 3.5
SNR_DB_NOISY = 10.0
TREND_STRENGTH_DRIFT = 0.3
ADF_NONSTATIONARY_P = 0.05
```

Apply these exact import changes:

```text
conditioning/binning.py:
  from ..config import thresholds as TH -> from . import thresholds as TH

conditioning/distance.py:
  from ..config.thresholds import ALPHA_DISTANCE -> from .thresholds import ALPHA_DISTANCE

conditioning/key.py:
  from ..config import thresholds as TH -> from . import thresholds as TH
```

Create `tests/operators/test_conditioning.py`:

```python
from SelfEvolvingHarnessTS.conditioning.binning import pattern_bin
from SelfEvolvingHarnessTS.conditioning.distance import distance


def test_conditioning_thresholds_are_locally_owned_and_functional():
    assert pattern_bin({"SNR": 3.0, "missing_rate": 0.1}) == "snrLow|miss"
    key = {
        "pattern": {
            "struct_feats": {},
            "quality_profile": {"problem_types": {}, "urgency": 0.0},
        }
    }
    assert distance(key, key) == 0.0
```

- [ ] **Step 3: Remove compatibility-only assertions from retained tests**

Replace `tests/contracts/test_task_contract.py` with:

```python
from SelfEvolvingHarnessTS.contracts.task import forecast_task_spec_v1


def test_canonical_task_sha_and_semantics():
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

Replace `tests/runtime/test_executor_contract.py` with:

```python
import numpy as np

from SelfEvolvingHarnessTS.runtime import executor as canonical
from SelfEvolvingHarnessTS.runtime.errors import ContractError, ExecutionError, RuntimeFailure


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

In `tests/runtime/test_fast_path_equivalence.py`, remove the `P6HarnessState` import and
replace:

```python
assert P6HarnessState is HRefState
```

with:

```python
assert isinstance(state, HRefState)
```

- [ ] **Step 4: Create concise active README**

Create `README.md`:

````markdown
# SelfEvolvingHarnessTS

SelfEvolvingHarnessTS is an active preparation-method harness with one contract layer,
one operator registry, one runtime, and one evolving method line.

## Active structure

- `contracts/` — task, program, and method contracts.
- `operators/` and `conditioning/` — canonical preparation mechanics.
- `runtime/` — the sole executor and fast path.
- `methods/h_ref_v02/` — frozen reference method.
- `methods/ttha/` — the next active method line when implemented.
- `evaluation/benchmark_v02/` — auxiliary frozen evaluator.
- `artifacts/frozen/benchmark_v02/` — immutable benchmark evidence.

## Verification

From the directory containing `SelfEvolvingHarnessTS`:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Benchmark CLI:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe \
  -m SelfEvolvingHarnessTS.evaluation.benchmark_v02 --help
```

## Historical recovery

P1–P6, E32, confirmatory runs, former runners, and historical result trees are available
from Git tag `pre-architecture-convergence-2026-07-17`. They are intentionally absent from
the active checkout.
````

- [ ] **Step 5: Update archive instructions**

Replace `experiments/archive/README.md` with:

```markdown
# Historical experiment recovery

The active tree does not store importable historical source. Full P0-P6, E32,
confirmatory, and former harness implementations are recovered from Git tag
`pre-architecture-convergence-2026-07-17` at commit
`1e75305770815c256d5b295b7ad6b8cb6cffe4b4`.

The exact cleanup boundary and benchmark relocation map are recorded in
`artifacts/manifests/active_tree_cleanup.json`. Frozen Benchmark-v0.2 evidence lives under
`artifacts/frozen/benchmark_v02/`.

Do not copy an old package back wholesale. Recover the needed mechanism, characterize its
required behavior, and promote it into the canonical contracts/operators/runtime/methods
structure.
```

- [ ] **Step 6: Run the retained suite before deletion and commit**

Run:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/architecture \
  SelfEvolvingHarnessTS/tests/contracts \
  SelfEvolvingHarnessTS/tests/operators \
  SelfEvolvingHarnessTS/tests/runtime \
  SelfEvolvingHarnessTS/tests/integration \
  SelfEvolvingHarnessTS/tests/frozen_protocol -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: all retained tests pass.

Commit:

```bash
git add README.md conditioning experiments/archive/README.md tests/contracts tests/operators \
  tests/runtime
git commit -m "docs: define the active project surface"
```

---

### Task 5: Remove Historical Code, Results, and Tests

**Files:**

- Delete: all retired root runners/modules/logs.
- Delete directories: `config/`, `diagnostics/`, `evaluators/`, `fast_path/`, `harness/`, `llm/`, `memory/`, `models/`, `p6/`, `policy/`, `sandbox/`, `slow_path/`, remaining `data/`, remaining `results/`.
- Delete: all remaining top-level `tests/test_*.py` after the three operator tests have moved.
- Delete: superseded benchmark design/plan documents.

**Interfaces:**

- Consumes: canonical replacements and passing focused suite from Tasks 2–4.
- Produces: a tree containing only active functionality and frozen evidence.

- [ ] **Step 1: Preview and count every bulk deletion set**

Run:

```bash
git ls-files 'run_*.py'
git ls-files '_*.log'
git ls-files 'tests/test_*.py'
git ls-files config diagnostics evaluators fast_path harness llm memory models p6 policy sandbox slow_path data results
```

Expected after prior moves:

- 42 historical `run_*.py` files;
- 9 tracked root log files;
- 96 remaining top-level legacy test files;
- no required benchmark file in top-level `data/` or `results/`.

Stop if any count differs or any listed path belongs to the active file map.

- [ ] **Step 2: Delete historical root files**

Delete the 42 Git-listed `run_*.py` files and 9 Git-listed `_*.log` files only after the
count gates above pass. Then delete these exact historical modules/documents:

```text
BUILD.md
EXECUTION_LOG.md
ONBOARDING.md
augment_corpus.py
confirmatory_corpus.py
confirmatory_freeze.py
confirmatory_reporter.py
e32_nested.py
e32_policy.py
family0_actions.py
freeze_init_harness.py
nested_supply.py
readiness_gym.py
s2_corpus.py
stage2_protocol.py
tensor_runner.py
```

Use `git rm --` with the reviewed Git-produced lists for the runner/log groups and explicit
paths for the second group.

Run:

```bash
git ls-files -z 'run_*.py' | git rm --pathspec-from-file=- --pathspec-file-nul
git ls-files -z '_*.log' | git rm --pathspec-from-file=- --pathspec-file-nul
git rm -- BUILD.md EXECUTION_LOG.md ONBOARDING.md augment_corpus.py \
  confirmatory_corpus.py confirmatory_freeze.py confirmatory_reporter.py \
  e32_nested.py e32_policy.py family0_actions.py freeze_init_harness.py \
  nested_supply.py readiness_gym.py s2_corpus.py stage2_protocol.py tensor_runner.py
```

- [ ] **Step 3: Delete retired packages and remaining historical data/results**

Run only after `git status --short` confirms the benchmark/data/artifact moves:

```bash
git rm -r -- config diagnostics evaluators fast_path harness llm memory models p6 policy \
  sandbox slow_path data results
git status --short --ignored benchmark config data evaluators fast_path harness llm memory \
  models p6 policy sandbox slow_path
rm -r -- benchmark/__pycache__ config/__pycache__ data/__pycache__ \
  evaluators/__pycache__ fast_path/__pycache__ harness/__pycache__ llm/__pycache__ \
  memory/__pycache__ models/__pycache__ p6/__pycache__ policy/__pycache__ \
  sandbox/__pycache__ slow_path/__pycache__
```

Expected: none of `contracts`, `operators`, `conditioning`, `runtime`, `methods`,
`evaluation`, `artifacts`, or `experiments` is listed. Before running `rm`, confirm with
`git status --short --ignored` that these exact ignored paths contain only generated Python
bytecode; stop if any other ignored/untracked content appears under a retired directory.

- [ ] **Step 4: Delete the remaining top-level historical tests**

Re-run `git ls-files 'tests/test_*.py'`, verify the count is exactly 96 and every path is at
the top level of `tests/`, then remove precisely that Git-produced list. Nested focused test
directories must not be touched.

Run:

```bash
git ls-files -z 'tests/test_*.py' | git rm --pathspec-from-file=- --pathspec-file-nul
```

- [ ] **Step 5: Delete superseded documentation**

Delete these exact paths:

```text
docs/superpowers/plans/2026-07-13-benchmark-v0-data-metrics-pipeline.md
docs/superpowers/plans/2026-07-13-benchmark-v0-plan-amendment-1.md
docs/superpowers/plans/2026-07-13-benchmark-v0-plan-amendment-2.md
docs/superpowers/specs/2026-07-13-benchmark-data-metrics-pipeline-design.md
docs/superpowers/specs/2026-07-13-benchmark-design-amendment-1.md
```

Run:

```bash
git rm -- \
  docs/superpowers/plans/2026-07-13-benchmark-v0-data-metrics-pipeline.md \
  docs/superpowers/plans/2026-07-13-benchmark-v0-plan-amendment-1.md \
  docs/superpowers/plans/2026-07-13-benchmark-v0-plan-amendment-2.md \
  docs/superpowers/specs/2026-07-13-benchmark-data-metrics-pipeline-design.md \
  docs/superpowers/specs/2026-07-13-benchmark-design-amendment-1.md
```

- [ ] **Step 6: Scan the deletion and commit**

Run:

```bash
git status --short
git diff --stat
git diff --check
```

Confirm the diff contains only approved deletions/moves and no zip, cache, pytest temp, or
other Agent files. Commit:

```bash
git add -u
git commit -m "refactor: remove historical architecture branches"
```

---

### Task 6: Tighten Repository Metadata and Architecture Gates

**Files:**

- Replace: `.gitignore`
- Replace: `.gitattributes`
- Modify: `tests/architecture/test_dependency_rules.py`

**Interfaces:**

- Consumes: final active-tree layout.
- Produces: metadata and executable rules that prevent retired namespaces from returning.

- [ ] **Step 1: Replace `.gitignore` with active paths**

Use:

```gitignore
# Benchmark source and derived data are local; manifests and bundled legacy inputs remain tracked.
evaluation/benchmark_v02/data/raw/
evaluation/benchmark_v02/data/clean_base/
evaluation/benchmark_v02/data/probe_cache/
evaluation/benchmark_v02/data/incoming/**/*.zip
evaluation/benchmark_v02/data/incoming/**/*.zip.asset.json
evaluation/benchmark_v02/data/incoming/**/*.csv
evaluation/benchmark_v02/data/incoming/**/*.csv.asset.json

# Runtime/test noise.
__pycache__/
*.py[cod]
.pytest_cache/
_pytest_*/
*.zip
```

- [ ] **Step 2: Replace `.gitattributes` with active byte rules**

Use:

```gitattributes
* text=auto
*.py text eol=lf

# Frozen evidence and bundled benchmark inputs must round-trip byte-exactly.
artifacts/frozen/benchmark_v02/** -text
evaluation/benchmark_v02/data/acquisition_manifest.json -text
evaluation/benchmark_v02/data/legacy/** -text
```

- [ ] **Step 3: Strengthen the architecture test**

Replace `tests/architecture/test_dependency_rules.py` with:

```python
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ACTIVE_AREAS = ("contracts", "operators", "conditioning", "runtime", "methods", "evaluation")
REMOVED_TOP_LEVEL = {
    "benchmark",
    "config",
    "diagnostics",
    "evaluators",
    "fast_path",
    "harness",
    "llm",
    "memory",
    "models",
    "p6",
    "policy",
    "sandbox",
    "slow_path",
}


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.add("." * node.level + (node.module or ""))
    return found


def _top_level(path: Path, name: str) -> str:
    if name.startswith("."):
        level = len(name) - len(name.lstrip("."))
        module_parts = [part for part in name.lstrip(".").split(".") if part]
        package = list(path.relative_to(ROOT).with_suffix("").parts[:-1])
        keep = max(0, len(package) - (level - 1))
        resolved = package[:keep] + module_parts
    else:
        resolved = [part for part in name.split(".") if part]
    if resolved and resolved[0] == ROOT.name:
        resolved = resolved[1:]
    return resolved[0] if resolved else ""


def test_retired_top_level_namespaces_are_absent():
    assert sorted(name for name in REMOVED_TOP_LEVEL if (ROOT / name).exists()) == []


def test_active_code_does_not_import_retired_namespaces():
    violations = []
    for area in ACTIVE_AREAS:
        for path in sorted((ROOT / area).rglob("*.py")):
            for name in _imports(path):
                top_level = _top_level(path, name)
                if top_level in REMOVED_TOP_LEVEL:
                    violations.append(f"{path.relative_to(ROOT)}: {name} -> {top_level}")
    assert violations == []


def test_historical_experiments_are_not_importable_source():
    python_files = sorted((ROOT / "experiments").rglob("*.py"))
    assert python_files == []
```

- [ ] **Step 4: Run metadata/architecture verification and commit**

Run:

```bash
git check-attr text eol -- contracts/task.py runtime/fast_path.py \
  evaluation/benchmark_v02/programs.py artifacts/frozen/benchmark_v02/program_pool.json
rg -n 'from .*\b(benchmark|p6|policy|sandbox|harness|slow_path|llm|memory|evaluators)\b|import .*\b(benchmark|p6|policy|sandbox|harness|slow_path|llm|memory|evaluators)\b' \
  contracts operators conditioning runtime methods evaluation
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/architecture -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: source files report LF, frozen evidence reports `text: unset`, import scan returns
no matches, architecture tests pass.

Commit:

```bash
git add .gitignore .gitattributes tests/architecture/test_dependency_rules.py
git commit -m "chore: enforce the active tree boundary"
```

---

### Task 7: Verify the Clean Active Tree

**Files:**

- No production file changes expected.
- Modify `artifacts/manifests/active_tree_cleanup.json` only if the measured pre-cleanup count was incorrect; do not invent a post-cleanup count field.

**Interfaces:**

- Consumes: all prior cleanup tasks.
- Produces: final functional, fingerprint, dependency, and size evidence.

- [ ] **Step 1: Run the complete retained suite**

Run from the worktree parent:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: exit code 0. Record the actual retained pass count; do not compare it to the old
historical suite count.

- [ ] **Step 2: Reconfirm the fixed H_ref fingerprints explicitly**

Run:

```bash
/mnt/d/Anaconda_envs/envs/project/python.exe -m pytest \
  SelfEvolvingHarnessTS/tests/runtime/test_fast_path_equivalence.py \
  SelfEvolvingHarnessTS/tests/integration/test_h_ref_method.py -q \
  --basetemp=SelfEvolvingHarnessTS/_pytest_active_cleanup
```

Expected: state SHA, ladder SHAs, choices, and all three artifact digests pass unchanged.

- [ ] **Step 3: Verify final structure and size**

Run:

```bash
git ls-files | wc -l
git ls-files '*.py' | awk 'index($0,"/")==0' | sort
find . -maxdepth 1 -type d -printf '%f\n' | sort
rg -n 'SelfEvolvingHarnessTS\.(benchmark|p6|policy|sandbox|harness|slow_path|llm|memory|evaluators)' \
  contracts operators conditioning runtime methods evaluation tests README.md
git diff --check
git status --short
```

Expected:

- tracked file count is at most 179;
- no historical root Python runner remains;
- no retired top-level directory remains;
- no retired import remains;
- diff check is silent and worktree is clean after temporary pytest output is removed.

- [ ] **Step 4: Confirm the main worktree was preserved**

Run:

```bash
git -C /mnt/c/Users/辉/desktop/agent/SelfEvolvingHarnessTS status --short
```

Expected: the other Agent's pre-existing `.gitignore`, design-spec, zip, and cache changes
remain present and were never staged on the cleanup branch.

## Handoff

Report:

- branch and commit range;
- recovery tag target;
- before/after tracked file counts;
- retained-suite pass count;
- H_ref fingerprint confirmation;
- final benchmark code/evidence locations;
- confirmation that TTHA/minipipe/performance work remains the next functional track.
