"""In-memory Classification component for the v1.2.1 P1 Core smoke.

This module is intentionally a contract/equipment check, not a scientific
run.  It opens only three previously exposed official TRAIN members:

* Epilepsy2 (archive name EpilepticSeizures) is the Validation/replay target;
* GunPoint and PowerCons are an independent Evolution selection set for the
  production ``select_best_fixed`` rule.

There is no loader for an official held-out member, no Query surface, no live
API path, and no report writer.  All readings and lifecycle records remain in
the returned Python payload.  Classification utility is Macro-F1; Accuracy
and per-class recall are safety/secondary readings only.
"""
from __future__ import annotations

import json
import math
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.contracts.task import (
    MetricSpec,
    classification_global_coarse_task_quality_contract_v1,
    classification_task_context_v1,
    classification_task_spec_v1,
    deployment_constraints_v1,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.baselines import (
    ProgramLoss,
    select_best_fixed,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
    _default_params_from_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

from evaluation.functional.consumers.ad_scope_adapter import compiled_steps
from evaluation.functional.consumers.cls_scope_adapter import raw_plus_difference
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import common
from SelfEvolvingHarnessTS.methods.ttha import signed_radius as resolver


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACCURACY_HISTORY = PROJECT_ROOT / "artifacts/functional/e2/sa1_minimal_gates.json"
FORECAST_HISTORY = PROJECT_ROOT / "artifacts/functional/e2/s2a_g1_run1_r2.json"

PROTOCOL_VERSION = common.PROTOCOL_VERSION
STAGE = common.STAGE
TASK = "classification"
CONSUMER_ID = "ridge-raw-plus-difference-v1"
PRIMARY_METRIC = "Macro-F1"
MAX_MODIFIED_FRACTION = 0.10
PERIOD_HINT = 24

B_MAIN = common.B_MAIN
MAX_PROBES = common.MAX_CHEAP_PROBES
MAX_LLM_CALLS = common.MAX_LLM_CALLS
MAX_TOKENS = common.MAX_TOKENS
MAX_UPDATES = common.MAX_ACCEPTED_UPDATES
SCRIPTED_GLOBAL_CALL_CAP = 16

FIXED_PROGRAMS = (
    ("Fixed Linear-impute", "impute_linear"),
    ("Fixed Hampel", "hampel_filter"),
    ("Fixed Winsor", "winsorize"),
    ("Fixed IQR", "outlier_iqr"),
)
MANDATORY_METHODS = common.MANDATORY_METHODS
EFFECT_ALIASES = {"resample_uniform": "identity"}
H0_SKILL_IDS = (
    "build_contrastive_candidates",
    "inspect_and_localize",
    "select_or_identity_and_verify",
)


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    archive: Path
    train_member: str
    role: str


TARGET_FIXTURE = FixtureSpec(
    fixture_id="Epilepsy2",
    archive=(
        PROJECT_ROOT
        / "data/ucr_conf_downloaded/D3_reserve/EpilepticSeizures.zip"
    ),
    train_member="EpilepticSeizures/EpilepticSeizures_TRAIN.ts",
    role="exposed_validation_replay",
)
EVOLUTION_FIXTURES = (
    FixtureSpec(
        fixture_id="GunPoint",
        archive=PROJECT_ROOT / "data/ucr_task_context/GunPoint.zip",
        train_member="GunPoint_TRAIN.ts",
        role="exposed_evolution_selection",
    ),
    FixtureSpec(
        fixture_id="PowerCons",
        archive=PROJECT_ROOT / "data/ucr_task_context/PowerCons.zip",
        train_member="PowerCons_TRAIN.ts",
        role="exposed_evolution_selection",
    ),
)


class P1Blocked(RuntimeError):
    """A fail-closed Classification P1 infrastructure condition."""


class FitBudget:
    """Count raw Ridge fits; cache hits never call ``spend``."""

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, count: int = 1) -> None:
        count = int(count)
        if count < 0 or self.used + count > self.cap:
            raise P1Blocked(
                "classification raw-fit cap exceeded: %d + %d > %d"
                % (self.used, count, self.cap)
            )
        self.used += count


@dataclass(frozen=True)
class ClassificationCell:
    fixture_id: str
    role: str
    archive: str
    train_member: str
    values: np.ndarray
    labels: np.ndarray
    fit_indices: tuple[int, ...]
    support_a_indices: tuple[int, ...]
    support_b_indices: tuple[int, ...]
    label_names: tuple[str, ...]

    @property
    def fit_values(self) -> np.ndarray:
        return self.values[list(self.fit_indices)]

    @property
    def fit_labels(self) -> np.ndarray:
        return self.labels[list(self.fit_indices)]

    @property
    def observation_block(self) -> np.ndarray:
        # The public-tool/H0 context inlet is forecast-compatible and expects
        # more than a tiny prefix.  Four complete fit rows are exposed as one
        # observation block, matching the established classification Harness
        # fixture; the verifier and Consumer still operate row-wise below.
        rows = min(4, int(self.fit_values.shape[0]))
        return np.asarray(self.fit_values[:rows], dtype=np.float64).ravel()

    def surface(self, face: str) -> tuple[np.ndarray, np.ndarray]:
        if face == "support_a":
            indices = self.support_a_indices
        elif face == "support_b":
            indices = self.support_b_indices
        else:
            raise KeyError("unknown classification surface: %s" % face)
        return self.values[list(indices)], self.labels[list(indices)]

    def split_counts(self) -> dict[str, int]:
        return {
            "fit": len(self.fit_indices),
            "support_a": len(self.support_a_indices),
            "support_b": len(self.support_b_indices),
        }


@dataclass(frozen=True)
class BestFixedFreeze:
    program_id: str
    syntactic_programs: tuple[str, ...]
    evaluated_programs: tuple[str, ...]
    safe_rejected_programs: tuple[str, ...]
    selection_fixture_ids: tuple[str, ...]
    full_evaluations: int
    raw_consumer_fits: int
    candidate_verifier_requests: int
    verified_windows: int
    wall_seconds: float


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _parse_train_member(raw: bytes, fixture_id: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Parse one equal-length, univariate UCR ``.ts`` TRAIN member."""
    rows: list[np.ndarray] = []
    raw_labels: list[str] = []
    in_data = False
    for raw_line in raw.decode("utf-8-sig").splitlines():
        line = raw_line.strip()
        if not in_data:
            in_data = line.lower() == "@data"
            continue
        if not line:
            continue
        fields = line.rsplit(":", 1)
        if len(fields) != 2:
            raise P1Blocked("%s TRAIN is not univariate .ts" % fixture_id)
        vector = np.fromstring(fields[0], dtype=np.float64, sep=",")
        if not vector.size or not np.isfinite(vector).all():
            raise P1Blocked("%s TRAIN contains empty/non-finite values" % fixture_id)
        rows.append(vector)
        raw_labels.append(fields[1].strip())
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.shape[0]:
        raise P1Blocked("%s TRAIN is not a non-empty equal-length matrix" % fixture_id)
    label_names = tuple(sorted(set(raw_labels)))
    encoded = np.asarray(
        [label_names.index(label) for label in raw_labels], dtype=np.int64
    )
    scale = np.std(matrix, axis=1, keepdims=True)
    if bool(np.any(scale <= 1e-12)):
        raise P1Blocked("%s TRAIN contains a degenerate row" % fixture_id)
    normalized = (matrix - np.mean(matrix, axis=1, keepdims=True)) / scale
    return normalized, encoded, label_names


def _stratified_surfaces(labels: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Official row order within sorted label; deterministic 50/25/25 split."""
    parts: dict[str, list[int]] = {"fit": [], "support_a": [], "support_b": []}
    labels = np.asarray(labels)
    for label in sorted(int(value) for value in set(labels.tolist())):
        indices = np.flatnonzero(labels == label).tolist()
        if len(indices) < 4:
            raise P1Blocked("class %s cannot populate all TRAIN surfaces" % label)
        n_fit = max(2, len(indices) // 2)
        n_a = max(1, (len(indices) - n_fit) // 2)
        n_b = len(indices) - n_fit - n_a
        if n_b < 1:
            n_a -= 1
            n_b += 1
        chunks = {
            "fit": indices[:n_fit],
            "support_a": indices[n_fit:n_fit + n_a],
            "support_b": indices[n_fit + n_a:],
        }
        if any(not rows for rows in chunks.values()):
            raise P1Blocked("class %s has an empty TRAIN surface" % label)
        for face, rows in chunks.items():
            parts[face].extend(int(index) for index in rows)
    for rows in parts.values():
        rows.sort()
    flattened = parts["fit"] + parts["support_a"] + parts["support_b"]
    if sorted(flattened) != list(range(int(labels.size))):
        raise P1Blocked("TRAIN split is not a disjoint exhaustive partition")
    return tuple(parts["fit"]), tuple(parts["support_a"]), tuple(parts["support_b"])


def _load_train_fixture(spec: FixtureSpec) -> ClassificationCell:
    """Read exactly the predeclared TRAIN member and no other archive member."""
    if not spec.archive.is_file():
        raise P1Blocked("missing exposed TRAIN archive: %s" % spec.archive)
    with ZipFile(spec.archive) as archive:
        try:
            archive.getinfo(spec.train_member)
        except KeyError as exc:
            raise P1Blocked(
                "%s lacks its predeclared TRAIN member" % spec.fixture_id
            ) from exc
        raw = archive.read(spec.train_member)
    values, labels, label_names = _parse_train_member(raw, spec.fixture_id)
    fit, support_a, support_b = _stratified_surfaces(labels)
    return ClassificationCell(
        fixture_id=spec.fixture_id,
        role=spec.role,
        archive=spec.archive.relative_to(PROJECT_ROOT).as_posix(),
        train_member=spec.train_member,
        values=values,
        labels=labels,
        fit_indices=fit,
        support_a_indices=support_a,
        support_b_indices=support_b,
        label_names=label_names,
    )


def _load_exposed_cells() -> tuple[ClassificationCell, tuple[ClassificationCell, ...], dict[str, Any]]:
    target = _load_train_fixture(TARGET_FIXTURE)
    selection = tuple(_load_train_fixture(spec) for spec in EVOLUTION_FIXTURES)
    target_id = target.fixture_id
    selection_ids = tuple(cell.fixture_id for cell in selection)
    disjoint = target_id not in selection_ids and len(set(selection_ids)) == len(selection_ids)
    if not disjoint:
        raise P1Blocked("Best Fixed Evolution fixtures overlap the target fixture")
    record = {
        "datasets": [target_id, *selection_ids],
        "target_fixture": target_id,
        "evolution_fixture": list(selection_ids),
        "data_role": "EXPOSED TRAIN-only P1 validation/replay",
        "selection_rule": "predeclared exposed Evolution TRAIN fixtures",
        "best_fixed_selection_disjoint_from_target": True,
        "selection_uses_support_or_future_utility": False,
        "test_member_bytes_read": 0,
        "held_out_requests": 0,
        "development_query_evaluations": 0,
        "natural_final_outcome_reads": 0,
        "target_train_member": target.train_member,
        "evolution_train_members": [cell.train_member for cell in selection],
        "surface_counts": {
            cell.fixture_id: cell.split_counts() for cell in (target, *selection)
        },
    }
    return target, selection, record


def _classification_metrics(
    truth: Any,
    predicted: Any,
    classes: Sequence[int],
) -> dict[str, Any]:
    """Return primary and safety readings with an explicit closed class set."""
    y_true = np.asarray(truth)
    y_pred = np.asarray(predicted)
    labels = [int(label) for label in classes]
    macro = float(f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    ))
    accuracy = float(accuracy_score(y_true, y_pred))
    recalls = recall_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    recall_by_class = {
        str(label): float(value) for label, value in zip(labels, recalls, strict=True)
    }
    if not math.isfinite(macro) or not math.isfinite(accuracy):
        raise P1Blocked("Classification Consumer produced a non-finite metric")
    return {
        "macro_f1": macro,
        "accuracy": accuracy,
        "per_class_recall": recall_by_class,
        "worst_class_recall": min(recall_by_class.values()),
    }


def _program_key(steps: Sequence[tuple[str, Mapping[str, object]]]) -> str:
    if not steps:
        return "identity"
    return "|".join(
        "%s(%s)" % (
            op,
            json.dumps(dict(params), ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
        for op, params in steps
    )


class MacroF1ConsumerAdapter:
    """Task-native Ridge adapter; it decides no winner or lifecycle relation."""

    def __init__(
        self,
        *,
        cell: ClassificationCell,
        budget: FitBudget,
        delayed_origin: int | None = None,
    ) -> None:
        self._cell = cell
        self._budget = budget
        self._delayed_origin = int(
            delayed_origin if delayed_origin is not None else cell.observation_block.size + 1
        )
        self._classes = tuple(sorted(int(value) for value in set(cell.fit_labels.tolist())))
        self._memo: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _prepared_fit(
        fit_values: np.ndarray,
        steps: Sequence[tuple[str, Mapping[str, object]]],
    ) -> tuple[np.ndarray, int]:
        if not steps:
            return np.asarray(fit_values, dtype=np.float64), 0
        rows: list[np.ndarray] = []
        behavior = 0
        for row in np.asarray(fit_values, dtype=np.float64):
            result = run_pipeline(list(steps), row, source="classification_p1")
            if not result.ok or result.artifact is None:
                raise P1Blocked(
                    "classification Workflow failed on a fit row: %s" % result.error
                )
            prepared = np.asarray(result.artifact, dtype=np.float64).ravel()
            if prepared.shape != row.shape or not np.isfinite(prepared).all():
                raise P1Blocked("classification Workflow violated shape/finite output")
            behavior += int(np.count_nonzero(
                ~np.isclose(prepared, row, rtol=1e-10, atol=1e-12)
            ))
            rows.append(prepared)
        return np.asarray(rows, dtype=np.float64), behavior

    def evaluate(
        self,
        steps: Sequence[tuple[str, Mapping[str, object]]],
        face: str,
    ) -> dict[str, Any]:
        key = (_program_key(steps), str(face))
        if key in self._memo:
            self.calls.append({
                "program": key[0], "surface": face, "cache": "hit", "consumer_fits": 0,
            })
            return dict(self._memo[key])
        prepared, behavior = self._prepared_fit(self._cell.fit_values, steps)
        self._budget.spend(1)
        model = RidgeClassifier(alpha=1.0)
        model.fit(raw_plus_difference(prepared), self._cell.fit_labels)
        values, truth = self._cell.surface(face)
        predicted = model.predict(raw_plus_difference(values))
        metric = _classification_metrics(truth, predicted, self._classes)
        reading = {
            "evaluation_state": "EVALUATED",
            "primary_metric": PRIMARY_METRIC,
            "macro_f1": metric["macro_f1"],
            "cls_macro_f1": metric["macro_f1"],
            "utility": metric["macro_f1"],
            "accuracy": metric["accuracy"],
            "per_class_recall": metric["per_class_recall"],
            "worst_class_recall": metric["worst_class_recall"],
            "behavior_point_count": behavior,
            "evaluated_rows": int(truth.size),
            "surface": face,
        }
        self._memo[key] = dict(reading)
        self.calls.append({
            "program": key[0], "surface": face, "cache": "miss", "consumer_fits": 1,
        })
        return reading

    def __call__(
        self,
        _roster: Sequence[Mapping[str, object]],
        _values: Mapping[str, Any],
        compiled: Any,
        _config: Mapping[str, object],
        *,
        origin: int,
    ) -> dict[str, Any]:
        face = "support_a" if int(origin) < self._delayed_origin else "support_b"
        reading = self.evaluate(compiled_steps(compiled), face)
        recalls = [
            float(reading["per_class_recall"][str(label)]) for label in self._classes
        ]
        return {
            "mean_smase": -float(reading["macro_f1"]),
            "per_view_smase": [-value for value in recalls],
            "behavior_point_count": int(reading["behavior_point_count"]),
            "cls_macro_f1": float(reading["macro_f1"]),
            "cls_accuracy": float(reading["accuracy"]),
            "cls_recall_by_class": dict(reading["per_class_recall"]),
            "cls_surface": face,
        }


class ClassificationScopeExecutor(ScopeExecutor):
    """Common verifier with one classification row as one checked window."""

    def __init__(
        self,
        *,
        cell: ClassificationCell,
        evaluate_fn: Any,
    ) -> None:
        block = np.asarray(cell.observation_block, dtype=np.float64)
        super().__init__(
            [{"series_uid": "train_observation", "role": "train"}],
            {"train_observation": block},
            {"anchors": []},
            evaluate_fn=evaluate_fn,
            max_modified_fraction=MAX_MODIFIED_FRACTION,
            modification_fraction_scope="cohort",
        )
        self._rows = np.asarray(cell.fit_values, dtype=np.float64)

    def training_windows(self, _origin: int):
        return [
            ("fit_row_%05d" % index, 0, np.asarray(row, dtype=np.float64))
            for index, row in enumerate(self._rows)
        ]


def _eligible_programs() -> tuple[str, ...]:
    eligible = []
    for op in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[op]
        if TASK not in tuple(metadata.get("allowed_tasks") or ()):
            continue
        if bool(metadata.get("shape_changing")) or bool(metadata.get("changes_target_space")):
            continue
        if metadata.get("requires_dependency") == "statsmodels":
            continue
        eligible.append(str(op))
    return tuple(eligible)


def _params(op: str) -> dict[str, object]:
    if op == "winsorize":
        # The operator default clips 5% at each tail.  On a finite odd-sized
        # row, quantile interpolation can therefore modify just over the 10%
        # deployment ceiling (18/178 for Epilepsy2).  This fixed 4% preset is
        # outcome-independent and leaves integer headroom under the same cap.
        return {"limits": 0.04}
    if op in {program for _name, program in FIXED_PROGRAMS}:
        return {}
    if op == "period_median_complete":
        return {"period": PERIOD_HINT, "cycles": 3, "min_donors": 2}
    if op in {"period_complete", "impute_ar", "repair_level_shift"}:
        return {"period": PERIOD_HINT}
    return dict(_default_params_from_contract(op))


def _steps(program: str) -> tuple[tuple[str, Mapping[str, object]], ...]:
    if program == "identity":
        return ()
    return ((str(program), _params(str(program))),)


def _contract_executor(cell: ClassificationCell) -> ClassificationScopeExecutor:
    def unavailable(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("contract-only verifier cannot consume an Outcome")

    return ClassificationScopeExecutor(cell=cell, evaluate_fn=unavailable)


def _verify_program(cell: ClassificationCell, program: str) -> dict[str, Any]:
    if program == "identity":
        return {
            "passed": True, "checked_windows": 0, "modified_windows": 0,
            "rejection_codes": [],
        }
    # Program construction is the Common DSL compile check.  The executor
    # independently recompiles the same steps before any Consumer call.
    Program.from_steps(list(_steps(program)), source="classification_p1_contract")
    verification = _contract_executor(cell).verify(_steps(program), cell.observation_block.size)
    return {
        "passed": bool(verification.passed),
        "checked_windows": int(verification.checked_windows),
        "modified_windows": int(verification.modified_windows),
        "rejection_codes": sorted({
            str(row.get("rejection_code")) for row in verification.rejected_windows
        }),
    }


def _common_dsl_contract(cell: ClassificationCell) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for program in _eligible_programs():
        try:
            verification = _verify_program(cell, program)
            compile_status = "PASS"
        except Exception as exc:  # fail closed, with no Consumer fallback
            verification = {
                "passed": False,
                "checked_windows": 0,
                "modified_windows": 0,
                "rejection_codes": ["%s" % type(exc).__name__],
            }
            compile_status = "FAIL"
        rows.append({
            "program": program,
            "compile": compile_status,
            "verifier": "PASS" if verification["passed"] else "SAFE_REJECT",
            "checked_windows": verification["checked_windows"],
            "modified_windows": verification["modified_windows"],
            "rejection_codes": verification["rejection_codes"],
        })
    mandatory = {program for _name, program in FIXED_PROGRAMS}
    executable = {
        row["program"] for row in rows
        if row["compile"] == "PASS" and row["verifier"] == "PASS"
    }
    missing = sorted(mandatory - executable)
    compile_failures = [row["program"] for row in rows if row["compile"] != "PASS"]
    return {
        "status": "PASS" if rows and not missing and not compile_failures else "FAIL",
        "inventory_policy": "task-legal global single-step Common DSL",
        "identity_available": True,
        "effect_distinct_inventory_count_from_p0b": 18,
        "eligible_operator_count": len(rows),
        "effect_aliases": dict(EFFECT_ALIASES),
        "consumer_evaluations": 0,
        "mandatory_fixed_programs_not_executable": missing,
        "compile_failures": compile_failures,
        "contract_overhead": {
            "candidate_verifier_requests": len(rows),
            "verified_windows": sum(int(row["checked_windows"]) for row in rows),
            "charged_to_method_cell_b4": False,
        },
        "rows": rows,
    }


def _surface(reading: Mapping[str, Any] | None, *, state: str) -> dict[str, Any]:
    if reading is None:
        return common.empty_surface(state)
    recalls = dict(reading.get("per_class_recall") or {})
    return {
        "evaluation_state": state,
        "primary_metric_value": float(reading["macro_f1"]),
        "utility": float(reading["utility"]),
        "delta_u_vs_identity": (
            None if reading.get("delta_u_vs_identity") is None
            else float(reading["delta_u_vs_identity"])
        ),
        "view_values": [float(recalls[key]) for key in sorted(recalls, key=int)],
        "behavior_point_count": int(reading.get("behavior_point_count") or 0),
    }


def _usage(
    *,
    support_a: int,
    support_b: int,
    raw_a: int | None = None,
    raw_b: int | None = None,
    cache_a: int = 0,
    cache_b: int = 0,
    cheap_probes: int = 0,
    llm_calls: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    updates: int = 0,
    wall_seconds: float = 0.0,
) -> dict[str, Any]:
    raw_a = support_a if raw_a is None else int(raw_a)
    raw_b = support_b if raw_b is None else int(raw_b)
    tokens = int(input_tokens) + int(output_tokens)
    within = bool(
        0 <= int(support_a) <= common.MAX_SUPPORT_A_FULL
        and 0 <= int(support_b) <= common.MAX_SUPPORT_B_FULL
        and int(support_a) + int(support_b) <= B_MAIN
        and int(cheap_probes) <= MAX_PROBES
        and int(llm_calls) <= MAX_LLM_CALLS
        and tokens <= MAX_TOKENS
        and int(updates) <= MAX_UPDATES
    )
    return {
        "full_support_evaluations": {
            "support_a": int(support_a), "support_b": int(support_b),
        },
        "raw_consumer_fits": {"support_a": raw_a, "support_b": raw_b},
        "cache_hits": {"support_a": int(cache_a), "support_b": int(cache_b)},
        "cheap_probes": int(cheap_probes),
        "llm_calls": int(llm_calls),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "tokens": tokens,
        "accepted_updates": int(updates),
        "wall_seconds": round(float(wall_seconds), 3),
        "within_caps": within,
    }


def _method_row(
    name: str,
    *,
    selected: str,
    readings: Mapping[str, Any],
    surfaces: Mapping[str, Any],
    usage: Mapping[str, Any],
    behavior_status: str,
    implementation: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    detail = dict(details or {})
    detail.setdefault("protocol_errors", [])
    return {
        "method": name,
        "status": "PASS",
        "contract_status": "PASS",
        "behavior_status": behavior_status,
        "selected_program": selected,
        "implementation": implementation,
        "readings": _plain(readings),
        "surfaces": _plain(surfaces),
        "usage": _plain(usage),
        "protocol_errors": [],
        "details": _plain(detail),
    }


def _evaluate_faces(
    cell: ClassificationCell,
    program: str,
    *,
    faces: Sequence[str] = ("support_a", "support_b"),
    identity: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    started = time.time()
    verification = _verify_program(cell, program)
    if not verification["passed"]:
        readings = {
            face: {
                "evaluation_state": "SAFE_REJECT",
                "rejection_codes": list(verification["rejection_codes"]),
            }
            for face in faces
        }
        surfaces = {face: common.empty_surface("SAFE_REJECT") for face in common.SURFACES}
        usage = _usage(
            support_a=0, support_b=0, cheap_probes=int(program != "identity"),
            wall_seconds=time.time() - started,
        )
        return readings, surfaces, usage
    budget = FitBudget(cap=max(1, len(faces)))
    adapter = MacroF1ConsumerAdapter(cell=cell, budget=budget)
    readings: dict[str, dict[str, Any]] = {}
    for face in faces:
        reading = adapter.evaluate(_steps(program), face)
        if identity is not None:
            reading["delta_u_vs_identity"] = (
                float(reading["utility"]) - float(identity[face]["utility"])
            )
        readings[face] = reading
    surfaces = {
        face: (
            _surface(readings.get(face), state="EVALUATED")
            if face in readings else common.empty_surface()
        )
        for face in common.SURFACES
    }
    usage = _usage(
        support_a=int("support_a" in readings),
        support_b=int("support_b" in readings),
        cheap_probes=int(program != "identity"),
        wall_seconds=time.time() - started,
    )
    return readings, surfaces, usage


def _identity_readings(cell: ClassificationCell) -> dict[str, dict[str, Any]]:
    readings, _surfaces, _usage_row = _evaluate_faces(cell, "identity")
    for reading in readings.values():
        reading["delta_u_vs_identity"] = 0.0
    return readings


def _select_best_fixed_on_evolution(
    cells: Sequence[ClassificationCell],
) -> BestFixedFreeze:
    """Freeze one program on independent exposed Evolution Support-A only."""
    started = time.time()
    syntactic = tuple(dict.fromkeys(("identity", *_eligible_programs())))
    programs = tuple(program for program in syntactic if program not in EFFECT_ALIASES)
    passed: list[str] = []
    rejected: list[str] = []
    verifier_requests = 0
    verified_windows = 0
    for program in programs:
        all_passed = True
        for cell in cells:
            check = _verify_program(cell, program)
            if program != "identity":
                verifier_requests += 1
                verified_windows += int(check["checked_windows"])
            all_passed = all_passed and bool(check["passed"])
        (passed if all_passed else rejected).append(program)
    covered = set(passed) | set(rejected) | set(EFFECT_ALIASES)
    if covered != set(syntactic):
        raise P1Blocked("Best Fixed program-space accounting is incomplete")
    if not passed:
        raise P1Blocked("Best Fixed has no verifier-approved Evolution program")

    budget = FitBudget(cap=len(passed) * len(cells))
    loss_rows: list[ProgramLoss] = []
    for cell in cells:
        adapter = MacroF1ConsumerAdapter(cell=cell, budget=budget)
        for program in passed:
            reading = adapter.evaluate(_steps(program), "support_a")
            # Macro-F1 is a cohort metric, so one row represents one complete
            # dataset/Consumer evaluation rather than pseudo per-sample loss.
            loss_rows.append(ProgramLoss(
                "support_a",
                "classification_p1_evolution_%s" % cell.fixture_id,
                program,
                "macro_f1",
                -float(reading["macro_f1"]),
            ))
    selected = select_best_fixed(loss_rows)
    return BestFixedFreeze(
        program_id=str(selected.program_id),
        syntactic_programs=syntactic,
        evaluated_programs=tuple(passed),
        safe_rejected_programs=tuple(rejected),
        selection_fixture_ids=tuple(cell.fixture_id for cell in cells),
        full_evaluations=budget.used,
        raw_consumer_fits=budget.used,
        candidate_verifier_requests=verifier_requests,
        verified_windows=verified_windows,
        wall_seconds=round(time.time() - started, 3),
    )


def _deterministic_methods(
    cell: ClassificationCell,
    best_fixed: BestFixedFreeze,
    identity: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    identity_surfaces = {
        face: _surface(reading, state="EVALUATED") for face, reading in identity.items()
    }
    rows.append(_method_row(
        "Identity",
        selected="identity",
        readings=identity,
        surfaces=identity_surfaces,
        usage=_usage(support_a=1, support_b=1),
        behavior_status="EVALUATED",
        implementation="unchanged TRAIN fit through the frozen Ridge Consumer",
        details={"elementwise_unchanged": True},
    ))

    selected = best_fixed.program_id
    readings, surfaces, usage = _evaluate_faces(cell, selected, identity=identity)
    rows.append(_method_row(
        "Best Fixed Per-task",
        selected=selected,
        readings={"target": readings},
        surfaces=surfaces,
        usage=usage,
        behavior_status=(
            "EVALUATED" if surfaces["support_a"]["evaluation_state"] == "EVALUATED"
            else "SAFE_REJECT"
        ),
        implementation=(
            "production select_best_fixed on independent Evolution TRAIN Support-A, "
            "then frozen unchanged for the target"
        ),
        details={
            "formal_evolution_winner_frozen": True,
            "scientific_selection_claim": False,
            "selection_uses_target_support": False,
            "selection_disjoint_from_target": True,
            "selection_fixture_ids": list(best_fixed.selection_fixture_ids),
            "candidate_rule": "all effect-distinct verifier-approved one-step programs",
            "selection_programs": list(best_fixed.evaluated_programs),
            "safe_rejected_programs": list(best_fixed.safe_rejected_programs),
            "effect_aliases": dict(EFFECT_ALIASES),
            "program_space_coverage_complete": bool(
                set(best_fixed.evaluated_programs)
                | set(best_fixed.safe_rejected_programs)
                | set(EFFECT_ALIASES)
                == set(best_fixed.syntactic_programs)
            ),
            "cost_by_phase": {
                "evolution_selection": {
                    "full_support_evaluations": best_fixed.full_evaluations,
                    "raw_consumer_fits": best_fixed.raw_consumer_fits,
                    "candidate_verifier_requests": best_fixed.candidate_verifier_requests,
                    "verified_windows": best_fixed.verified_windows,
                    "llm_calls": 0,
                    "tokens": 0,
                    "accepted_updates": 0,
                    "wall_seconds": best_fixed.wall_seconds,
                    "charged_to_target_b4": False,
                },
                "target_smoke": {
                    "full_support_evaluations": usage["full_support_evaluations"],
                    "charged_to_target_b4": True,
                },
            },
        },
    ))

    for name, program in FIXED_PROGRAMS:
        readings, surfaces, usage = _evaluate_faces(cell, program, identity=identity)
        rows.append(_method_row(
            name,
            selected=program,
            readings=readings,
            surfaces=surfaces,
            usage=usage,
            behavior_status=(
                "EVALUATED" if surfaces["support_a"]["evaluation_state"] == "EVALUATED"
                else "SAFE_REJECT"
            ),
            implementation="frozen one-step Common-DSL heuristic",
        ))
    rows.extend(_parallel_and_sequential(cell, identity))
    return rows


def _evaluate_candidate_support_a(
    cell: ClassificationCell,
    program: str,
    identity: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    check = _verify_program(cell, program)
    if not check["passed"]:
        return None, check
    budget = FitBudget(cap=1)
    adapter = MacroF1ConsumerAdapter(cell=cell, budget=budget)
    reading = adapter.evaluate(_steps(program), "support_a")
    reading["delta_u_vs_identity"] = (
        float(reading["utility"]) - float(identity["support_a"]["utility"])
    )
    return reading, check


def _winner(
    readings: Mapping[str, Mapping[str, Any]],
) -> str:
    return min(
        readings,
        key=lambda program: (-float(readings[program]["utility"]), str(program)),
    )


def _parallel_and_sequential(
    cell: ClassificationCell,
    identity: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    parallel_candidates = ("impute_linear", "hampel_filter", "winsorize")
    evaluated: dict[str, dict[str, Any]] = {"identity": dict(identity["support_a"])}
    rejected: list[str] = []
    for program in parallel_candidates:
        reading, _check = _evaluate_candidate_support_a(cell, program, identity)
        if reading is None:
            rejected.append(program)
        else:
            evaluated[program] = reading
    chosen = _winner(evaluated)
    delayed, delayed_surfaces, delayed_usage = _evaluate_faces(
        cell, chosen, faces=("support_b",), identity=identity
    )
    support_reading = evaluated[chosen]
    surfaces = {
        "support_a": _surface(support_reading, state="EVALUATED"),
        "support_b": delayed_surfaces["support_b"],
    }
    rows.append(_method_row(
        "Parallel Best-of-N@4",
        selected=chosen,
        readings={
            "support_a_candidates": evaluated,
            "safe_rejected_candidates": rejected,
            "support_b": delayed.get("support_b"),
        },
        surfaces=surfaces,
        usage=_usage(
            support_a=len(evaluated) - 1,
            support_b=delayed_usage["full_support_evaluations"]["support_b"],
            cheap_probes=len(parallel_candidates) + int(chosen != "identity"),
        ),
        behavior_status="EVALUATED",
        implementation="three independent Support-A candidates plus one frozen Support-B promotion",
        details={
            "candidate_order_outcome_independent": list(parallel_candidates),
            "winner_rule": "highest Support-A Macro-F1; canonical-id tie break",
            "safe_rejected_candidates": rejected,
            "performance_claim": False,
        },
    ))

    step_1_program = "winsorize"
    step_1, _check_1 = _evaluate_candidate_support_a(cell, step_1_program, identity)
    step_2_program = (
        "hampel_filter"
        if step_1 is None or float(step_1["delta_u_vs_identity"]) < 0.0
        else "outlier_iqr"
    )
    step_2, _check_2 = _evaluate_candidate_support_a(cell, step_2_program, identity)
    sequential = {"identity": dict(identity["support_a"])}
    if step_1 is not None:
        sequential[step_1_program] = step_1
    if step_2 is not None:
        sequential[step_2_program] = step_2
    chosen = _winner(sequential)
    delayed, delayed_surfaces, delayed_usage = _evaluate_faces(
        cell, chosen, faces=("support_b",), identity=identity
    )
    rows.append(_method_row(
        "Sequential Refinement@4",
        selected=chosen,
        readings={
            "step_1": {"program": step_1_program, "reading": step_1},
            "step_2": {"program": step_2_program, "reading": step_2},
            "support_b": delayed.get("support_b"),
        },
        surfaces={
            "support_a": _surface(sequential[chosen], state="EVALUATED"),
            "support_b": delayed_surfaces["support_b"],
        },
        usage=_usage(
            support_a=len(sequential) - 1,
            support_b=delayed_usage["full_support_evaluations"]["support_b"],
            cheap_probes=2 + int(chosen != "identity"),
        ),
        behavior_status="EVALUATED",
        implementation="feedback-conditioned replacement proposal in the frozen one-step DSL",
        details={
            "step_2_received_step_1_feedback": True,
            "step_2_rule": "negative/rejected step 1 selects Hampel; otherwise IQR",
            "two_step_template_added": False,
            "performance_claim": False,
        },
    ))
    return rows


def _history_contract() -> dict[str, Any]:
    """Audit historical candidates and withhold both before candidate supply."""
    if not ACCURACY_HISTORY.is_file() or not FORECAST_HISTORY.is_file():
        raise P1Blocked("required exposed history audit material is absent")
    accuracy_payload = json.loads(ACCURACY_HISTORY.read_text(encoding="utf-8"))
    card = dict(accuracy_payload.get("card_v0") or {})
    accuracy_scope = dict((card.get("risk_guards") or {}).get("scope_v1") or {})
    expected_accuracy = {
        "task_kind": TASK,
        "consumer_id": CONSUMER_ID,
        "metric": "accuracy",
    }
    observed_accuracy = {key: accuracy_scope.get(key) for key in expected_accuracy}
    if card.get("skill_id") != "sa1_supply_scope_v2" or observed_accuracy != expected_accuracy:
        raise P1Blocked("the historical Classification card lost its audited Accuracy scope")

    forecast_payload = json.loads(FORECAST_HISTORY.read_text(encoding="utf-8"))
    boundary = dict(forecast_payload.get("boundary_compile") or {})
    forecast_scope = dict(boundary.get("scope") or {})
    if boundary.get("has_card") is not True or forecast_scope.get("task_kind") != "forecast":
        raise P1Blocked("the wrong-task Forecast history source is not auditable")

    withheld = [
        {
            "skill_id": "sa1_supply_scope_v2",
            "reason": "PRIMARY_METRIC_MISMATCH(accuracy!=Macro-F1)",
            "installed": False,
            "retrieval": 0,
            "scope_match": 0,
            "supply": 0,
            "support_probe": 0,
            "episode": 0,
        },
        {
            "skill_id": "s2a_forecast_supply_v0",
            "reason": "WRONG_TASK(forecast!=classification)",
            "installed": False,
            "retrieval": 0,
            "scope_match": 0,
            "supply": 0,
            "support_probe": 0,
            "episode": 0,
        },
    ]
    return {
        "status": "PASS",
        "historical_input_status": "WITHHELD_FAIL_CLOSED",
        "initial_skill_ids": list(H0_SKILL_IDS),
        "target_scope": {
            "task_kind": TASK,
            "consumer_id": CONSUMER_ID,
            "metric": PRIMARY_METRIC,
        },
        "accuracy_card_scope": observed_accuracy,
        "accuracy_card_direct_numeric_comparison_allowed": False,
        "wrong_task_forecast_scope": {
            key: forecast_scope.get(key) for key in ("task_kind", "consumer_id", "metric")
        },
        "withheld_history": withheld,
        "wrong_task_fail_closed": True,
        "accuracy_metric_fail_closed": True,
        "rq3_treatment": "NOT_EXERCISED",
    }


class _LifecycleFaceExecutor:
    """Dispatch Support-A/B and memoize duplicate lifecycle requests."""

    def __init__(self, faces: Mapping[int, ClassificationScopeExecutor]) -> None:
        self._faces = dict(faces)
        self._cache: dict[tuple[int, str], Any] = {}
        self._requests = {"support_a": 0, "support_b": 0}
        self._unique = {"support_a": 0, "support_b": 0}
        self._hits = {"support_a": 0, "support_b": 0}
        self._verified = {"support_a": 0, "support_b": 0}
        self._tokens = sorted(self._faces)

    def _face(self, token: int) -> str:
        return "support_a" if token == self._tokens[0] else "support_b"

    def evaluate(self, steps: Any, origin: int) -> Any:
        token = int(origin)
        if token not in self._faces:
            raise P1Blocked("unknown Classification P1 surface token: %s" % token)
        face = self._face(token)
        self._requests[face] += 1
        key = (token, _program_key(tuple(steps or ())))
        if key in self._cache:
            self._hits[face] += 1
            return self._cache[key]
        receipt = self._faces[token].evaluate(tuple(steps or ()), token)
        self._cache[key] = receipt
        self._unique[face] += 1
        self._verified[face] += int(
            getattr(getattr(receipt, "verification", None), "checked_windows", 0) or 0
        )
        return receipt

    def accounting(self) -> dict[str, Any]:
        return {
            "requests_by_face": dict(self._requests),
            "unique_receipt_requests_by_face": dict(self._unique),
            "cache_hits_by_face": dict(self._hits),
            "duplicate_requests": sum(self._hits.values()),
            "unique_candidate_verifier_requests": sum(self._unique.values()),
            "verified_windows_by_face": dict(self._verified),
        }


def _fast_verifier_requests(trace: Any) -> int:
    if str(getattr(trace, "compilation_status", "") or "") not in {
        "ok", "not_applicable",
    }:
        return 0
    return len(tuple(getattr(trace, "candidate_ids", ()) or ())) + len(
        tuple(getattr(trace, "rejection_receipts", ()) or ())
    )


def _snapshot_state_view(snapshot: Any) -> tuple[Any, ...]:
    skills = tuple(
        (
            str(skill.skill_id),
            int(skill.revision),
            str(skill.body),
            json.dumps(_plain(skill.observable_applicability), sort_keys=True),
            json.dumps(_plain(skill.risk_guards), sort_keys=True),
        )
        for skill in snapshot.skills
    )
    memories = tuple(
        str(getattr(memory, "memory_id", "")) for memory in snapshot.memories
    )
    return skills, memories


def _request(
    cell: ClassificationCell,
    spec: Any,
    context: Any,
) -> tuple[PreparationRequest, dict[str, Any], dict[str, np.ndarray]]:
    block = np.asarray(cell.observation_block, dtype=np.float64)
    origin = int(block.size)
    values = {"train_observation": block}
    observed = dict(resolver.window_context(values, origin, PERIOD_HINT))
    observed["bound_period"] = float(PERIOD_HINT)
    features = dict(extract_public_features(block, task_kind=TASK))
    request = PreparationRequest(
        "classification-p1-epilepsy2-exposed",
        block,
        spec,
        observed,
        task_context=context,
    )
    return request, features, values


def _new_method_state(
    *, snapshot: Any, cell: ClassificationCell, backend: Any, root: Path, tag: str,
) -> dict[str, Any]:
    agent = shared_harness._scripted_agent(
        cell.observation_block, backend.new_arm_backend()
    )
    return shared_harness._new_arm_state(
        snapshot=snapshot, agent=agent, store_root=root, tag=tag
    )


def _lifecycle_executor(
    cell: ClassificationCell,
    identity: Mapping[str, Mapping[str, Any]],
) -> tuple[MacroF1ConsumerAdapter, _LifecycleFaceExecutor, FitBudget, int]:
    origin = int(cell.observation_block.size)
    budget = FitBudget(cap=B_MAIN)
    adapter = MacroF1ConsumerAdapter(
        cell=cell, budget=budget, delayed_origin=origin + 1
    )
    faces = {
        origin: ClassificationScopeExecutor(cell=cell, evaluate_fn=adapter),
        origin + 1: ClassificationScopeExecutor(cell=cell, evaluate_fn=adapter),
    }
    for token, face in ((origin, "support_a"), (origin + 1, "support_b")):
        recalls = dict(identity[face]["per_class_recall"])
        faces[token]._baseline_cache[token] = -float(identity[face]["macro_f1"])
        faces[token]._per_view_cache[token] = [
            -float(recalls[key]) for key in sorted(recalls, key=int)
        ]
    return adapter, _LifecycleFaceExecutor(faces), budget, origin


def _adapter_counts(adapter: MacroF1ConsumerAdapter) -> tuple[dict[str, int], dict[str, int]]:
    fits = {face: 0 for face in common.SURFACES}
    hits = {face: 0 for face in common.SURFACES}
    for call in adapter.calls:
        face = str(call["surface"])
        if int(call.get("consumer_fits") or 0):
            fits[face] += 1
        elif call.get("cache") == "hit":
            hits[face] += 1
    return fits, hits


def _selected_surface_readings(
    *,
    adapter: MacroF1ConsumerAdapter,
    steps: Sequence[tuple[str, Mapping[str, object]]],
    identity: Mapping[str, Mapping[str, Any]],
    delayed_opened: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not steps:
        return (
            {
                "support_a": {"evaluation_state": "ABSTAINED"},
                "support_b": {"evaluation_state": "ABSTAINED"},
            },
            {
                "support_a": common.empty_surface("ABSTAINED"),
                "support_b": common.empty_surface("ABSTAINED"),
            },
        )
    support = adapter.evaluate(tuple(steps), "support_a")
    support["delta_u_vs_identity"] = (
        float(support["utility"]) - float(identity["support_a"]["utility"])
    )
    readings: dict[str, Any] = {"support_a": support}
    surfaces = {"support_a": _surface(support, state="EVALUATED")}
    if delayed_opened:
        delayed = adapter.evaluate(tuple(steps), "support_b")
        delayed["delta_u_vs_identity"] = (
            float(delayed["utility"]) - float(identity["support_b"]["utility"])
        )
        readings["support_b"] = delayed
        surfaces["support_b"] = _surface(delayed, state="EVALUATED")
    else:
        readings["support_b"] = {"evaluation_state": "NOT_EVALUATED"}
        surfaces["support_b"] = common.empty_surface()
    return readings, surfaces


def _frozen_h0_method(
    *,
    snapshot: Any,
    cell: ClassificationCell,
    backend: Any,
    root: Path,
    spec: Any,
    context: Any,
    identity: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    started = time.time()
    before_calls = int(backend.calls)
    state = _new_method_state(
        snapshot=snapshot, cell=cell, backend=backend, root=root, tag="frozen_h0"
    )
    request, _features, _values = _request(cell, spec, context)
    state["method"].bind_round_data(cell.observation_block, task_kind=TASK)
    state["method"].prepare(request, runtime_prior_slot=False, pool_mode="full")
    trace = state["method"].last_trace
    chosen_id = str(trace.chosen_candidate_id or "identity")
    trace_steps = tuple((trace.candidate_program_steps or {}).get(chosen_id) or ())
    ops = [str(op) for op, _params_row in trace_steps]
    errors: list[str] = []
    if len(ops) > 1:
        errors.append("unfrozen_multi_step_program")
    if set(ops) - set(_eligible_programs()):
        errors.append("task_mismatch_or_noninventory_program")

    # Use the exact steps emitted by prepare.  A verifier rejection falls
    # closed to an independently fitted identity diagnostic.
    adapter, dispatcher, _budget, origin = _lifecycle_executor(cell, identity)
    receipt = dispatcher.evaluate(trace_steps, origin) if trace_steps else None
    fallback = bool(receipt is not None and (
        not receipt.verification.passed or receipt.gain is None
    ))
    executed_steps = () if fallback else trace_steps
    reading = adapter.evaluate(executed_steps, "support_a")
    reading["delta_u_vs_identity"] = (
        float(reading["utility"]) - float(identity["support_a"]["utility"])
    )
    fits, hits = _adapter_counts(adapter)
    fast_requests = _fast_verifier_requests(trace)
    accounting = dispatcher.accounting()
    usage = _usage(
        support_a=fits["support_a"],
        support_b=0,
        raw_a=fits["support_a"],
        raw_b=0,
        cache_a=hits["support_a"],
        cache_b=0,
        cheap_probes=fast_requests + accounting["unique_candidate_verifier_requests"],
        llm_calls=int(backend.calls) - before_calls,
        wall_seconds=time.time() - started,
    )
    if not usage["within_caps"]:
        errors.append("budget_cap_exceeded")
    selected = "identity" if not executed_steps else "+".join(
        str(op) for op, _params_row in executed_steps
    )
    return _method_row(
        "Frozen H0",
        selected=selected,
        readings={"support_a": reading},
        surfaces={
            "support_a": _surface(reading, state="EVALUATED"),
            "support_b": common.empty_surface(),
        },
        usage=usage,
        behavior_status="EVALUATED",
        implementation="production TTHAMethod.prepare plus an exact-step Support-A diagnostic",
        details={
            "initial_skill_ids": sorted(str(skill.skill_id) for skill in snapshot.skills),
            "target_adaptation": False,
            "writeback_channel": False,
            "unit_state_discarded": True,
            "prepare_calls": 1,
            "run_online_round_calls": 0,
            "open_delayed_calls": 0,
            "trace_chosen_candidate": chosen_id,
            "trace_chosen_program_steps": [
                {"op": str(op), "params": dict(params)} for op, params in trace_steps
            ],
            "verifier_fallback_to_identity": fallback,
            "fast_candidate_verifier_requests": fast_requests,
            "receipt_accounting": accounting,
            "protocol_errors": errors,
        },
    )


def _production_harness_method(
    *,
    name: str,
    snapshot: Any,
    cell: ClassificationCell,
    backend: Any,
    root: Path,
    spec: Any,
    context: Any,
    identity: Mapping[str, Mapping[str, Any]],
    initial_skill_ids: Sequence[str],
    writeback: bool,
) -> dict[str, Any]:
    started = time.time()
    before_calls = int(backend.calls)
    state = _new_method_state(
        snapshot=snapshot,
        cell=cell,
        backend=backend,
        root=root,
        tag=name.lower().replace("-", "_").replace(" ", "_"),
    )
    state_before = _snapshot_state_view(state["method"]._active_snapshot())
    request, features, values = _request(cell, spec, context)
    adapter, dispatcher, _budget, origin = _lifecycle_executor(cell, identity)
    result = run_online_round(
        state["method"],
        dispatcher,
        request,
        values,
        origin=origin,
        slow_agent=None,
        controller=state["controller"],
        store=state["store"],
        card_builder=shared_harness._card_builder,
        round_name="classification_p1_%s" % name.lower().replace("-", "_"),
        budget=common.MAX_SUPPORT_A_FULL,
        allow_slow=False,
        horizon=1,
        period=PERIOD_HINT,
        domain="classification_p1_epilepsy2_exposed",
        fast_features=features,
        allow_fast_skill=True,
        runtime_prior_slot=False,
        pool_mode="full",
    )
    open_delayed(
        result, dispatcher, delayed_origin=origin + 1, store=state["store"]
    )
    support_b_unique = int(
        dispatcher.accounting()["unique_receipt_requests_by_face"]["support_b"]
    )
    approved_after_support_b = bool(
        result.approved_skill_id is not None and support_b_unique >= 1
    )
    activated = False
    if writeback and result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
    state_after = _snapshot_state_view(state["method"]._active_snapshot())
    state_changed = state_before != state_after
    retained_update = bool(writeback and activated and state_changed)

    trace = state["method"].last_trace
    winner_steps = tuple(result._winner_steps or ())
    selected = "identity" if not winner_steps else "+".join(
        str(op) for op, _params_row in winner_steps
    )
    readings, surfaces = _selected_surface_readings(
        adapter=adapter,
        steps=winner_steps,
        identity=identity,
        delayed_opened=result.delayed_utility is not None,
    )
    readings.update({
        "support_receipts": int(result.target_support_receipts_used),
        "actual_probed_programs": _plain(result.actual_probed_programs),
        "abstained": bool(result.abstained),
        "harm_count": int(result.harm_count),
        "delayed_delta_u_vs_identity": (
            None if result.delayed_utility is None else float(result.delayed_utility)
        ),
    })
    fits, hits = _adapter_counts(adapter)
    receipt_accounting = dispatcher.accounting()
    fast_requests = _fast_verifier_requests(trace)
    usage = _usage(
        support_a=fits["support_a"],
        support_b=fits["support_b"],
        raw_a=fits["support_a"],
        raw_b=fits["support_b"],
        cache_a=hits["support_a"],
        cache_b=hits["support_b"],
        cheap_probes=fast_requests + receipt_accounting["unique_candidate_verifier_requests"],
        llm_calls=int(backend.calls) - before_calls,
        updates=int(retained_update),
        wall_seconds=time.time() - started,
    )
    candidate_steps = dict(trace.candidate_program_steps or {})
    executed_ids = {
        str(row.get("candidate_id"))
        for row in (result.actual_probed_programs or ())
        if row.get("kind") in {"probe", "verifier_rejected"}
    }
    executed_ops = {
        str(op)
        for candidate_id in executed_ids
        for op, _params_row in tuple(candidate_steps.get(candidate_id) or ())
    }
    errors: list[str] = []
    if executed_ops - set(_eligible_programs()):
        errors.append("task_mismatch_or_noninventory_program")
    if any(len(tuple(candidate_steps.get(candidate_id) or ())) > 1 for candidate_id in executed_ids):
        errors.append("unfrozen_multi_step_program")
    if not usage["within_caps"]:
        errors.append("budget_cap_exceeded")
    if retained_update and not approved_after_support_b:
        errors.append("wrong_promotion")
    status = "PASS" if not errors else "FAIL"
    row = _method_row(
        name,
        selected=selected,
        readings=readings,
        surfaces=surfaces,
        usage=usage,
        behavior_status="ABSTAINED" if result.abstained else "EVALUATED",
        implementation="production TTHAMethod + run_online_round + open_delayed",
        details={
            "initial_skill_ids": list(initial_skill_ids),
            "initial_state": "h0" if name == "A3-reset" else "shared_k0_a5",
            "target_adaptation": True,
            "writeback_channel": bool(writeback),
            "unit_state_discarded": not bool(writeback),
            "writeback_persisted_to_evolution_store": False,
            "retained_update": retained_update,
            "approved_after_support_b": approved_after_support_b,
            "writeback_treatment": (
                "RETAINED_NEW_STATE" if retained_update else "NOT_EXERCISED"
            ),
            "rq3_treatment": "NOT_EXERCISED",
            "rq3_reason": "no eligible Macro-F1 history revision and later re-encounter",
            "prepare_calls": 1,
            "run_online_round_calls": 1,
            "open_delayed_calls": 1,
            "candidate_count": len(tuple(trace.candidate_ids or ())),
            "winner_program_steps": [
                {"op": str(op), "params": dict(params)} for op, params in winner_steps
            ],
            "fast_candidate_verifier_requests": fast_requests,
            "receipt_accounting": receipt_accounting,
            "method_state_changed_inside_isolated_unit": state_changed,
            "accuracy_history_card": "WITHHELD_PRIMARY_METRIC_MISMATCH",
            "wrong_task_history": "WITHHELD_TASK_MISMATCH",
            "accuracy_history_faces": {
                "retrieval": 0, "scope_match": 0, "supply": 0,
                "support_probe": 0, "episode": 0,
            },
            "wrong_task_faces": {
                "retrieval": 0, "scope_match": 0, "supply": 0,
                "support_probe": 0, "episode": 0,
            },
            "protocol_errors": errors,
        },
    )
    row["status"] = status
    row["contract_status"] = status
    row["protocol_errors"] = list(errors)
    return row


def _harness_methods(
    cell: ClassificationCell,
    identity: Mapping[str, Mapping[str, Any]],
    spec: Any,
    context: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    history = _history_contract()
    backend = shared_harness._scripted_backend(SCRIPTED_GLOBAL_CALL_CAP)
    temp_root = Path(tempfile.mkdtemp(prefix="classification_p1_"))
    try:
        h0 = forecast_course._h0()
        h0_ids = sorted(str(skill.skill_id) for skill in h0.skills)
        if h0_ids != sorted(H0_SKILL_IDS):
            raise P1Blocked("the production H0 bootstrap Skill set changed")
        # Accuracy-scoped Classification history and Forecast history were
        # audited above and deliberately are not installed into either arm.
        shared_k0_a5 = h0
        rows = [
            _frozen_h0_method(
                snapshot=h0, cell=cell, backend=backend, root=temp_root,
                spec=spec, context=context, identity=identity,
            ),
        ]

        static_readings, static_surfaces, static_usage = _evaluate_faces(
            cell, "identity", identity=identity
        )
        rows.append(_method_row(
            "Static",
            selected="identity",
            readings=static_readings,
            surfaces=static_surfaces,
            usage=static_usage,
            behavior_status="EVALUATED",
            implementation="independent zero-lifecycle identity Consumer smoke",
            details={
                "prepare_calls": 0,
                "episode_writes": 0,
                "delayed_open_calls": 0,
                "accepted_updates": 0,
                "writeback_attempts": 0,
                "store_created": False,
            },
        ))
        rows.extend([
            _production_harness_method(
                name="A3-reset", snapshot=h0, cell=cell, backend=backend,
                root=temp_root, spec=spec, context=context, identity=identity,
                initial_skill_ids=h0_ids, writeback=False,
            ),
            _production_harness_method(
                name="K0-fixed", snapshot=shared_k0_a5, cell=cell, backend=backend,
                root=temp_root, spec=spec, context=context, identity=identity,
                initial_skill_ids=h0_ids, writeback=False,
            ),
            _production_harness_method(
                name="A5-online", snapshot=shared_k0_a5, cell=cell, backend=backend,
                root=temp_root, spec=spec, context=context, identity=identity,
                initial_skill_ids=h0_ids, writeback=True,
            ),
        ])
        return rows, {
            "mode": "scripted-in-memory",
            "production_format_exercised": True,
            "production_lifecycle_exercised": True,
            "production_ttha_method_exercised": True,
            "production_run_online_round_exercised": True,
            "production_open_delayed_exercised": True,
            "live_transport_exercised": False,
            "global_llm_calls": int(backend.calls),
            "global_input_tokens": 0,
            "global_output_tokens": 0,
            "k0_a5_same_initial_state": True,
            "k0_a5_initial_skill_ids": h0_ids,
            "temporary_store_removed_after_run": True,
            "history_contract": history,
            "historical_input_status": history["historical_input_status"],
            "withheld_history": history["withheld_history"],
            "rq3_treatment": "NOT_EXERCISED",
        }
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _task_contract() -> tuple[Any, Any]:
    eligible = set(_eligible_programs())
    forbidden = tuple(sorted(set(OPERATOR_NAMES) - eligible))
    spec = classification_task_spec_v1(
        downstream_model_class=CONSUMER_ID,
        metric=MetricSpec(PRIMARY_METRIC, "higher_is_better"),
        forbidden_modifications=forbidden,
    )
    context = classification_task_context_v1(
        task_spec=spec,
        quality_contract=classification_global_coarse_task_quality_contract_v1(),
        deployment_constraints=deployment_constraints_v1(
            constraint_id="classification-p1-core-smoke-v1",
            fixed_downstream_model_id="fixed:%s" % CONSUMER_ID,
            maximum_candidates=B_MAIN,
            maximum_modified_fraction=MAX_MODIFIED_FRACTION,
        ),
    )
    return spec, context


def run(*, backend_mode: str = "scripted") -> dict[str, Any]:
    """Build and validate the Classification P1 component entirely in memory."""
    if backend_mode != "scripted":
        raise P1Blocked(
            "Classification P1 exposes only the reproducible scripted in-memory backend"
        )
    target, selection, data = _load_exposed_cells()
    spec, context = _task_contract()
    if spec.metric.name != PRIMARY_METRIC or spec.metric.direction != "higher_is_better":
        raise P1Blocked("Classification TaskSpec did not freeze Macro-F1 higher-is-better")
    if context.deployment_constraints.maximum_candidates != B_MAIN:
        raise P1Blocked("Classification TaskContext did not freeze B=4")

    contract = _common_dsl_contract(target)
    best_fixed = _select_best_fixed_on_evolution(selection)
    identity = _identity_readings(target)
    methods = _deterministic_methods(target, best_fixed, identity)
    harness_rows, backend = _harness_methods(target, identity, spec, context)
    methods.extend(harness_rows)
    order = {name: index for index, name in enumerate(MANDATORY_METHODS)}
    methods.sort(key=lambda row: order[str(row["method"])])

    protocol_errors = {
        "natural_final_outcome_reads": 0,
        "development_query_evaluations": 0,
        "task_mismatch_execution": 0,
        "cross_task_skill_leakage": 0,
        "accuracy_skill_metric_leakage": 0,
        "historical_skill_bypassed_support": 0,
        "wrong_promotion": 0,
    }
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "task_tranche": TASK,
        "evidence_grade": "INFRASTRUCTURE",
        "classification_component_pass": True,
        "component_pass": True,
        "data": data,
        "split": {
            "fit_count": len(target.fit_indices),
            "support_a_count": len(target.support_a_indices),
            "support_b_count": len(target.support_b_indices),
            "query_count": 0,
            "target_count": 1,
            "selection_count": len(selection),
        },
        "consumer": {
            "id": CONSUMER_ID,
            "implementation": "RidgeClassifier(alpha=1) over raw+first-difference features",
            "primary_metric": PRIMARY_METRIC,
            "metric_direction": "higher_is_better",
            "utility_definition": "U_classification=Macro-F1",
            "delta_definition": "U(method)-U(identity)",
            "secondary_metrics": ["Accuracy", "per-class recall", "worst-class recall"],
        },
        "common_dsl_contract": contract,
        "methods": methods,
        "backend": backend,
        "protocol_errors": protocol_errors,
        "blocking_failures": [],
        "performance_or_headroom_claim": False,
        "treatment_or_capability_claim": False,
        "claims": {"performance": False, "treatment": False},
        "rq3_online_revision": "NOT_EXERCISED",
        "overall_p1_complete": False,
        "release_p2": False,
        "execution_order": [
            "load_exposed_train_only_fixtures",
            "freeze_best_fixed_on_disjoint_evolution_support_a",
            "run_target_contract_smoke",
            "audit_history_and_withhold_mismatched_cards",
            "validate_normalized_component",
        ],
    }
    normalized = common.normalize_component(payload)
    failures = common.validate_component(normalized)
    payload["blocking_failures"] = failures
    passed = not failures
    payload["classification_component_pass"] = passed
    payload["component_pass"] = passed
    return _plain(payload)


def run_classification_component(*, backend_mode: str = "scripted") -> dict[str, Any]:
    """Stable integration alias used by the P1 master runner."""
    return run(backend_mode=backend_mode)


__all__ = [
    "B_MAIN",
    "CONSUMER_ID",
    "EVOLUTION_FIXTURES",
    "FIXED_PROGRAMS",
    "H0_SKILL_IDS",
    "MANDATORY_METHODS",
    "MacroF1ConsumerAdapter",
    "PRIMARY_METRIC",
    "P1Blocked",
    "TARGET_FIXTURE",
    "_classification_metrics",
    "_common_dsl_contract",
    "_history_contract",
    "_load_exposed_cells",
    "_task_contract",
    "run",
    "run_classification_component",
]
