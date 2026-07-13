"""Public benchmark Method surface, feedback accounting, and support gates."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from ..operators.registry import OPERATOR_METADATA, canonicalize
from ..policy.task_spec import TaskSpec

__all__ = [
    "BenchmarkMethod",
    "ConfirmationArtifact",
    "ContractVerdict",
    "DryRunArtifact",
    "FeedbackAPI",
    "FeedbackBudgetError",
    "FeedbackCallRecord",
    "FeedbackHandle",
    "MethodGateError",
    "MethodSeriesView",
    "PreparedSeries",
    "PrivateSeriesEpisode",
    "require_final_eligibility",
    "run_support_b_confirmation",
    "run_support_dry_run",
    "validate_prepared",
]


class MethodGateError(RuntimeError):
    """A method attempted to cross a frozen visibility or phase boundary."""


class FeedbackBudgetError(MethodGateError):
    """The frozen method feedback budget has been exhausted."""


def _canonical_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string")
    return value


def _sha256_string(value: Any, name: str) -> str:
    value = _canonical_string(value, name)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise MethodGateError(f"{name} must be a lowercase SHA256 digest")
    return value


def _readonly_array(value: Any) -> np.ndarray:
    try:
        array = np.asarray(value, dtype="<f8").copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("series values must be numeric") from exc
    array[np.isnan(array)] = np.nan
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class PrivateSeriesEpisode:
    """Runner-owned episode; private fields never cross into MethodSeriesView."""

    series_uid: str
    degraded_inner_train: np.ndarray
    future: np.ndarray
    regime_tag: str
    split_role: str

    def __post_init__(self) -> None:
        _canonical_string(self.series_uid, "series_uid")
        _canonical_string(self.regime_tag, "regime_tag")
        _canonical_string(self.split_role, "split_role")
        object.__setattr__(
            self, "degraded_inner_train", _readonly_array(self.degraded_inner_train)
        )
        object.__setattr__(self, "future", _readonly_array(self.future))


@dataclass(frozen=True)
class MethodSeriesView:
    series_uid: str
    degraded_inner_train: np.ndarray

    def __post_init__(self) -> None:
        _canonical_string(self.series_uid, "series_uid")
        object.__setattr__(
            self, "degraded_inner_train", _readonly_array(self.degraded_inner_train)
        )

    @classmethod
    def from_episode(cls, episode: PrivateSeriesEpisode) -> "MethodSeriesView":
        if not isinstance(episode, PrivateSeriesEpisode):
            raise TypeError("episode must be PrivateSeriesEpisode")
        return cls(episode.series_uid, episode.degraded_inner_train)


@dataclass(frozen=True)
class PreparedSeries:
    series_uid: str
    values: np.ndarray
    operators: tuple[str, ...]
    units: str

    def __post_init__(self) -> None:
        _canonical_string(self.series_uid, "series_uid")
        object.__setattr__(self, "values", _readonly_array(self.values))
        if not isinstance(self.operators, tuple):
            raise ValueError("operators must be an immutable tuple")
        if not all(isinstance(op, str) and op for op in self.operators):
            raise ValueError("operators must contain non-empty strings")
        _canonical_string(self.units, "units")


@runtime_checkable
class BenchmarkMethod(Protocol):
    method_id: str

    def prepare(
        self,
        series_view: MethodSeriesView,
        task_spec: TaskSpec,
        observed_pattern_spec: Mapping[str, float],
    ) -> PreparedSeries: ...


@dataclass(frozen=True)
class ContractVerdict:
    valid: bool
    code: str


def validate_prepared(
    prepared: PreparedSeries, *, expected_length: int
) -> ContractVerdict:
    if not isinstance(prepared, PreparedSeries):
        return ContractVerdict(False, "invalid_prepared_type")
    values = prepared.values
    if values.ndim != 1:
        return ContractVerdict(False, "dimensionality_changed")
    if len(values) != expected_length:
        return ContractVerdict(False, "length_changed")
    if prepared.units != "original_units":
        return ContractVerdict(False, "units_changed")
    if np.isinf(values).any() or np.isnan(values).all():
        return ContractVerdict(False, "non_finite_output")
    for requested in prepared.operators:
        canonical = canonicalize(requested)
        metadata = OPERATOR_METADATA.get(canonical)
        if metadata is None:
            return ContractVerdict(False, "unknown_operator")
        if bool(metadata.get("changes_target_space")):
            return ContractVerdict(False, "forbidden_target_space_transform")
    return ContractVerdict(True, "ok")


@dataclass(frozen=True)
class FeedbackHandle:
    handle_id: str
    split_role: str
    channel: str

    def __post_init__(self) -> None:
        _canonical_string(self.handle_id, "handle_id")
        _canonical_string(self.split_role, "split_role")
        _canonical_string(self.channel, "channel")


@dataclass(frozen=True)
class FeedbackCallRecord:
    call_index: int
    handle_id: str
    prepared_sha: str
    status: str
    result: float | None


def _prepared_sha(prepared: PreparedSeries) -> str:
    payload = hashlib.sha256()
    payload.update(prepared.series_uid.encode("utf-8"))
    payload.update(prepared.values.astype("<f8", copy=False).tobytes())
    payload.update(
        json.dumps(
            [list(prepared.operators), prepared.units],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return payload.hexdigest()


class FeedbackAPI:
    def __init__(
        self,
        *,
        budget: int,
        evaluator: Callable[[FeedbackHandle, PreparedSeries], float],
    ) -> None:
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer")
        if not callable(evaluator):
            raise TypeError("evaluator must be callable")
        self._budget = budget
        self._evaluator = evaluator
        self._records: list[FeedbackCallRecord] = []

    @property
    def call_records(self) -> tuple[FeedbackCallRecord, ...]:
        return tuple(self._records)

    @property
    def remaining_budget(self) -> int:
        consumed = {"complete", "invalid_result", "evaluator_error"}
        return self._budget - len(
            [record for record in self._records if record.status in consumed]
        )

    def evaluate(self, handle: FeedbackHandle, prepared: PreparedSeries) -> float:
        if not isinstance(handle, FeedbackHandle):
            raise TypeError("handle must be FeedbackHandle")
        if not isinstance(prepared, PreparedSeries):
            raise TypeError("prepared must be PreparedSeries")
        prepared_sha = _prepared_sha(prepared)
        call_index = len(self._records)
        if handle.split_role != "support_a":
            self._records.append(
                FeedbackCallRecord(call_index, handle.handle_id, prepared_sha, "denied", None)
            )
            raise MethodGateError("feedback is restricted to Support-A")
        if handle.channel != "closed_form_inner_val":
            self._records.append(
                FeedbackCallRecord(call_index, handle.handle_id, prepared_sha, "denied", None)
            )
            raise MethodGateError("feedback is restricted to the closed-form inner-val channel")
        if self.remaining_budget <= 0:
            self._records.append(
                FeedbackCallRecord(call_index, handle.handle_id, prepared_sha, "budget_exhausted", None)
            )
            raise FeedbackBudgetError("feedback budget exhausted")
        try:
            result = self._evaluator(handle, prepared)
        except Exception as exc:
            self._records.append(
                FeedbackCallRecord(
                    call_index,
                    handle.handle_id,
                    prepared_sha,
                    "evaluator_error",
                    None,
                )
            )
            raise MethodGateError("feedback evaluator failed") from exc
        if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(float(result)):
            self._records.append(
                FeedbackCallRecord(call_index, handle.handle_id, prepared_sha, "invalid_result", None)
            )
            raise MethodGateError("feedback evaluator returned a non-finite result")
        value = float(result)
        self._records.append(
            FeedbackCallRecord(call_index, handle.handle_id, prepared_sha, "complete", value)
        )
        return value


@dataclass(frozen=True)
class DryRunArtifact:
    method_id: str
    method_code_sha: str
    config_sha: str
    contract_output_sha: str
    artifact_sha: str


@dataclass(frozen=True)
class ConfirmationArtifact:
    method_id: str
    method_code_sha: str
    dry_run_sha: str
    contract_output_sha: str
    artifact_sha: str


def _canonical_json_sha(value: Any, name: str) -> str:
    try:
        payload = json.dumps(
            value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise MethodGateError(f"{name} must be canonical JSON data") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _run_contract(
    method: BenchmarkMethod,
    episodes: Sequence[PrivateSeriesEpisode],
    task_spec: TaskSpec,
    expected_role: str,
) -> str:
    if not isinstance(task_spec, TaskSpec):
        raise TypeError("task_spec must be TaskSpec")
    digests: list[str] = []
    for episode in episodes:
        if not isinstance(episode, PrivateSeriesEpisode):
            raise TypeError("episodes must contain PrivateSeriesEpisode values")
        if episode.split_role != expected_role:
            raise MethodGateError(f"contract run requires {expected_role}")
        view = MethodSeriesView.from_episode(episode)
        prepared = method.prepare(view, task_spec, MappingProxyType({}))
        verdict = validate_prepared(
            prepared, expected_length=len(episode.degraded_inner_train)
        )
        if prepared.series_uid != episode.series_uid:
            raise MethodGateError("prepared series_uid changed")
        if not verdict.valid:
            raise MethodGateError(f"method contract failed: {verdict.code}")
        digests.append(_prepared_sha(prepared))
    if not digests:
        raise MethodGateError("contract run requires at least one episode")
    return _canonical_json_sha(digests, "contract outputs")


def run_support_dry_run(
    method: BenchmarkMethod,
    episodes: Sequence[PrivateSeriesEpisode],
    task_spec: TaskSpec,
    *,
    method_code_sha: str,
    config: Mapping[str, Any],
) -> DryRunArtifact:
    method_id = _canonical_string(getattr(method, "method_id", None), "method_id")
    code_sha = _sha256_string(method_code_sha, "method_code_sha")
    config_sha = _canonical_json_sha(dict(config), "method config")
    output_sha = _run_contract(method, episodes, task_spec, "support_a")
    artifact_sha = _canonical_json_sha(
        [method_id, code_sha, config_sha, task_spec.to_dict(), output_sha],
        "dry-run artifact",
    )
    return DryRunArtifact(method_id, code_sha, config_sha, output_sha, artifact_sha)


def run_support_b_confirmation(
    method: BenchmarkMethod,
    episodes: Sequence[PrivateSeriesEpisode],
    task_spec: TaskSpec,
    *,
    method_code_sha: str,
    dry_run_sha: str,
    prior_artifact: ConfirmationArtifact | None = None,
) -> ConfirmationArtifact:
    if prior_artifact is not None:
        raise MethodGateError("Support-B confirmation is one-shot and cannot be overwritten")
    method_id = _canonical_string(getattr(method, "method_id", None), "method_id")
    code_sha = _sha256_string(method_code_sha, "method_code_sha")
    dry_sha = _sha256_string(dry_run_sha, "dry_run_sha")
    output_sha = _run_contract(method, episodes, task_spec, "support_b")
    artifact_sha = _canonical_json_sha(
        [method_id, code_sha, dry_sha, task_spec.to_dict(), output_sha],
        "confirmation artifact",
    )
    return ConfirmationArtifact(method_id, code_sha, dry_sha, output_sha, artifact_sha)


def require_final_eligibility(
    method_id: str,
    *,
    dry_run_sha: str | None,
    confirmation_sha: str | None,
) -> None:
    _canonical_string(method_id, "method_id")
    if dry_run_sha is None or confirmation_sha is None:
        raise MethodGateError("Final requires Support-A dry-run and Support-B confirmation")
    _sha256_string(dry_run_sha, "dry_run_sha")
    _sha256_string(confirmation_sha, "confirmation_sha")
