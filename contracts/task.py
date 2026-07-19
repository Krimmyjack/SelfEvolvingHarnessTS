"""policy/task_spec.py — TaskSpec / MetricSpec（P0 契约层，Final_Plan_CodeAgentFirst_2026-07-09 §P0）。

Readiness 是 task/metric/model 条件化的（STAGE1 C6 FAIL = 剂量增益模型类条件化；classify C1 =
平滑算子跨任务符号翻转）。本模块把任务契约显式化为一等公民：EvidencePacket、ProgramSpec v1
与后续 anomaly rig（P2）都以它为唯一任务真源，不再允许隐式 forecast。

字段语义：
  task_type                canonical 任务名（与 action_spec._TASK_ORDER / registry allowed_tasks 对齐）
  target_semantics         下游目标是什么——决定"什么算 readiness"（anomaly 下 spike 是信号非缺陷）
  label_availability       test-time 可用标签面（label-safe 协议：final-test 标签在任何任务下都不可用）
  metric                   判官口径（名字 + 方向；utility 公式按任务实例化，不做统一字面目标函数）
  horizon                  forecast 专用（其余任务必须 None；None = 打包时未知）
  downstream_model_class   下游模型/判官类——C6 证明增益是模型类条件化的，必须显式携带
  forbidden_modifications  实例级额外禁改集（registry allowed_tasks 之外的收紧；按 canonical 名判定）

身份：sha() = canonical JSON 的 sha256 前 16 位；字段序影响 forbidden_modifications 的列表序
（保持声明序），键序由 sort_keys 归一。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .canonical import canonical_sha256

TASK_TYPES = ("forecast", "classification", "anomaly_detection")
TARGET_SEMANTICS_BY_TASK = {
    "forecast": "future_values",
    "classification": "class_label",
    "anomaly_detection": "anomaly_events",
}
LABEL_AVAILABILITY = ("history_only", "train_labels", "unlabeled")
_METRIC_DIRECTIONS = ("lower_is_better", "higher_is_better")
_CANONICAL_ID = re.compile(r"^[a-z][a-z0-9._:-]*[a-z0-9]$")

QUALITY_OBJECTIVES = {
    "forecast": ("forecast_future_values",),
    "classification": ("preserve_class_evidence",),
    "anomaly_detection": ("preserve_anomaly_evidence",),
}
PRESERVATION_VOCABULARY = (
    "observed_values_outside_suspect_region",
    "temporal_order",
    "series_length",
    "forecast_relevant_structure",
    "class_relevant_structure",
    "anomaly_events",
)
HARM_VOCABULARY = (
    "unnecessary_modification",
    "global_over_smoothing",
    "future_information_use",
    "out_of_scope_change",
    "event_erasure",
)
EVIDENCE_VOCABULARY = (
    "public_observation",
    "fixed_public_probe_panel",
    "candidate_execution_receipt",
)
VERIFICATION_VOCABULARY = (
    "operator_legality",
    "execution_success",
    "finite_output",
    "shape_preservation",
    "effect_distinctness",
    "modification_scope",
)
ABSTENTION_VOCABULARY = (
    "insufficient_public_evidence",
    "no_effect_distinct_candidate",
    "risk_guard_failure",
    "capability_unavailable",
)


@dataclass(frozen=True)
class MetricSpec:
    """判官口径：指标名 + 优化方向。"""
    name: str
    direction: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"MetricSpec.name 须为非空字符串，得到 {self.name!r}")
        if self.direction not in _METRIC_DIRECTIONS:
            raise ValueError(
                f"MetricSpec.direction 须 ∈ {_METRIC_DIRECTIONS}，得到 {self.direction!r}")

    def to_dict(self) -> dict:
        return {"name": self.name, "direction": self.direction}


@dataclass(frozen=True)
class TaskSpec:
    task_type: str
    target_semantics: str
    label_availability: str
    metric: MetricSpec
    horizon: Optional[int]
    downstream_model_class: str
    forbidden_modifications: Tuple[str, ...] = ()

    def __post_init__(self):
        if self.task_type not in TASK_TYPES:
            raise ValueError(f"task_type 须 ∈ {TASK_TYPES}，得到 {self.task_type!r}")
        expected = TARGET_SEMANTICS_BY_TASK[self.task_type]
        if self.target_semantics != expected:
            raise ValueError(
                f"task_type={self.task_type!r} 的 target_semantics 须为 {expected!r}，"
                f"得到 {self.target_semantics!r}")
        if self.label_availability not in LABEL_AVAILABILITY:
            raise ValueError(
                f"label_availability 须 ∈ {LABEL_AVAILABILITY}，得到 {self.label_availability!r}")
        if self.horizon is not None:
            if self.task_type != "forecast":
                raise ValueError(f"horizon 仅 forecast 可用（task_type={self.task_type!r}）")
            if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon <= 0:
                raise ValueError(f"horizon 须为正整数或 None，得到 {self.horizon!r}")
        if not isinstance(self.downstream_model_class, str) or not self.downstream_model_class:
            raise ValueError("downstream_model_class 须为非空字符串（C6：模型类条件化必须显式）")
        if not isinstance(self.forbidden_modifications, tuple) or not all(
                isinstance(op, str) and op for op in self.forbidden_modifications):
            raise ValueError("forbidden_modifications 须为非空字符串组成的 tuple")

    def to_dict(self) -> dict:
        return {
            "task_type": self.task_type,
            "target_semantics": self.target_semantics,
            "label_availability": self.label_availability,
            "metric": self.metric.to_dict(),
            "horizon": self.horizon,
            "downstream_model_class": self.downstream_model_class,
            "forbidden_modifications": list(self.forbidden_modifications),
        }

    def to_packet_dict(self) -> dict:
        """EvidencePacket / conditioning 消费形状：保留 legacy `type` 键（下游按 task.type 读）。"""
        return {"type": self.task_type, **self.to_dict()}

    def sha(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def is_op_forbidden(self, op: str) -> bool:
        """实例级禁改判定（canonical 化后比较——旧 alias 如 fill_gaps≡impute_linear 同禁）。"""
        if not self.forbidden_modifications:
            return False
        from ..operators.registry import canonicalize
        return canonicalize(op) in {canonicalize(f) for f in self.forbidden_modifications}


def _canonical_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not _CANONICAL_ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a canonical identifier")
    return value


def _closed_values(
    values: Tuple[str, ...],
    vocabulary: tuple[str, ...],
    field_name: str,
) -> Tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    unknown = tuple(value for value in values if value not in vocabulary)
    if unknown:
        raise ValueError(f"{field_name} contains unknown vocabulary: {unknown!r}")
    return values


@dataclass(frozen=True)
class TaskQualityContract:
    """Immutable, deployment-safe semantics for what constitutes good preparation."""

    contract_id: str
    task_type: str
    objective: str
    preserve: Tuple[str, ...]
    harms: Tuple[str, ...]
    evidence_expectations: Tuple[str, ...]
    verification_dimensions: Tuple[str, ...]
    abstention_conditions: Tuple[str, ...]
    schema_version: str = "task-quality-contract/1"

    def __post_init__(self) -> None:
        if self.schema_version != "task-quality-contract/1":
            raise ValueError("unsupported TaskQualityContract revision")
        _canonical_id(self.contract_id, "contract_id")
        if self.task_type not in TASK_TYPES:
            raise ValueError("TaskQualityContract task_type is unsupported")
        if self.objective not in QUALITY_OBJECTIVES[self.task_type]:
            raise ValueError("TaskQualityContract objective does not match task_type")
        _closed_values(self.preserve, PRESERVATION_VOCABULARY, "preserve")
        _closed_values(self.harms, HARM_VOCABULARY, "harms")
        _closed_values(
            self.evidence_expectations,
            EVIDENCE_VOCABULARY,
            "evidence_expectations",
        )
        _closed_values(
            self.verification_dimensions,
            VERIFICATION_VOCABULARY,
            "verification_dimensions",
        )
        _closed_values(
            self.abstention_conditions,
            ABSTENTION_VOCABULARY,
            "abstention_conditions",
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "task_type": self.task_type,
            "objective": self.objective,
            "preserve": list(self.preserve),
            "harms": list(self.harms),
            "evidence_expectations": list(self.evidence_expectations),
            "verification_dimensions": list(self.verification_dimensions),
            "abstention_conditions": list(self.abstention_conditions),
        }

    def sha(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class DeploymentConstraintSpec:
    """Deployment limits kept separate from the task-quality objective."""

    constraint_id: str
    model_policy: str
    fixed_downstream_model_id: str
    maximum_candidates: int
    maximum_modified_fraction: float
    schema_version: str = "deployment-constraint/1"

    def __post_init__(self) -> None:
        if self.schema_version != "deployment-constraint/1":
            raise ValueError("unsupported DeploymentConstraintSpec revision")
        _canonical_id(self.constraint_id, "constraint_id")
        if self.model_policy != "fixed":
            raise ValueError("F1 supports fixed model policy only")
        _canonical_id(self.fixed_downstream_model_id, "fixed_downstream_model_id")
        if (
            isinstance(self.maximum_candidates, bool)
            or not isinstance(self.maximum_candidates, int)
            or self.maximum_candidates < 1
        ):
            raise ValueError("maximum_candidates must be a positive integer")
        if (
            isinstance(self.maximum_modified_fraction, bool)
            or not isinstance(self.maximum_modified_fraction, (int, float))
            or not math.isfinite(float(self.maximum_modified_fraction))
            or not 0.0 <= float(self.maximum_modified_fraction) <= 1.0
        ):
            raise ValueError("maximum_modified_fraction must be finite in [0, 1]")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "constraint_id": self.constraint_id,
            "model_policy": self.model_policy,
            "fixed_downstream_model_id": self.fixed_downstream_model_id,
            "maximum_candidates": self.maximum_candidates,
            "maximum_modified_fraction": float(self.maximum_modified_fraction),
        }

    def sha(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TaskContext:
    task_spec: TaskSpec
    quality_contract: TaskQualityContract
    deployment_constraints: DeploymentConstraintSpec
    schema_version: str = "task-context/1"

    def __post_init__(self) -> None:
        if self.schema_version != "task-context/1":
            raise ValueError("unsupported TaskContext revision")
        if not isinstance(self.task_spec, TaskSpec):
            raise TypeError("TaskContext.task_spec must be TaskSpec")
        if not isinstance(self.quality_contract, TaskQualityContract):
            raise TypeError("TaskContext.quality_contract must be TaskQualityContract")
        if not isinstance(self.deployment_constraints, DeploymentConstraintSpec):
            raise TypeError(
                "TaskContext.deployment_constraints must be DeploymentConstraintSpec"
            )
        if self.task_spec.task_type != self.quality_contract.task_type:
            raise ValueError("TaskSpec and TaskQualityContract task_type mismatch")

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "task_spec": self.task_spec.to_dict(),
            "quality_contract": self.quality_contract.to_dict(),
            "deployment_constraints": self.deployment_constraints.to_dict(),
        }

    def sha(self) -> str:
        return canonical_sha256(self.to_dict())


# ── 默认任务契约（v1；downstream_model_class 与既有判官/报告器口径对齐）────────────────

def forecast_task_spec_v1(
    *,
    horizon: Optional[int] = None,
    label_availability: str = "history_only",
    downstream_model_class: str = "dlinear_shared",
    forbidden_modifications: Tuple[str, ...] = (),
    metric: Optional[MetricSpec] = None,
) -> TaskSpec:
    """现任 forecast 口径：nRMSE(lower) + 域内共享训练 DLinear 报告器（协议 v2 勘误口径）。"""
    return TaskSpec(
        task_type="forecast",
        target_semantics="future_values",
        label_availability=label_availability,
        metric=metric or MetricSpec("nRMSE", "lower_is_better"),
        horizon=horizon,
        downstream_model_class=downstream_model_class,
        forbidden_modifications=forbidden_modifications,
    )


def forecast_task_quality_contract_v1() -> TaskQualityContract:
    return TaskQualityContract(
        contract_id="forecast-quality-v1",
        task_type="forecast",
        objective="forecast_future_values",
        preserve=(
            "observed_values_outside_suspect_region",
            "temporal_order",
            "series_length",
            "forecast_relevant_structure",
        ),
        harms=(
            "unnecessary_modification",
            "global_over_smoothing",
            "future_information_use",
            "out_of_scope_change",
        ),
        evidence_expectations=(
            "public_observation",
            "fixed_public_probe_panel",
            "candidate_execution_receipt",
        ),
        verification_dimensions=(
            "operator_legality",
            "execution_success",
            "finite_output",
            "shape_preservation",
            "effect_distinctness",
            "modification_scope",
        ),
        abstention_conditions=(
            "insufficient_public_evidence",
            "no_effect_distinct_candidate",
            "risk_guard_failure",
            "capability_unavailable",
        ),
    )


def forecast_neutral_task_quality_contract_v1() -> TaskQualityContract:
    """A deliberately sparse forecast contract for the F1 report-only contrast.

    It keeps deployment safety and mechanical verification explicit while
    withholding the task-specific preservation guidance carried by the primary
    forecast contract.  It is a diagnostic control, not a deployment default.
    """

    return TaskQualityContract(
        contract_id="forecast-neutral-v1",
        task_type="forecast",
        objective="forecast_future_values",
        preserve=("temporal_order", "series_length"),
        harms=("future_information_use", "out_of_scope_change"),
        evidence_expectations=(
            "public_observation",
            "candidate_execution_receipt",
        ),
        verification_dimensions=(
            "operator_legality",
            "execution_success",
            "finite_output",
            "shape_preservation",
            "effect_distinctness",
            "modification_scope",
        ),
        abstention_conditions=(
            "insufficient_public_evidence",
            "no_effect_distinct_candidate",
            "risk_guard_failure",
            "capability_unavailable",
        ),
    )


def deployment_constraints_v1(
    *,
    maximum_candidates: int = 3,
    maximum_modified_fraction: float = 0.35,
    fixed_downstream_model_id: str = "fixed:m0",
) -> DeploymentConstraintSpec:
    return DeploymentConstraintSpec(
        constraint_id="forecast-fixed-m0-v1",
        model_policy="fixed",
        fixed_downstream_model_id=fixed_downstream_model_id,
        maximum_candidates=maximum_candidates,
        maximum_modified_fraction=maximum_modified_fraction,
    )


def forecast_task_context_v1(
    *,
    task_spec: TaskSpec | None = None,
    quality_contract: TaskQualityContract | None = None,
    deployment_constraints: DeploymentConstraintSpec | None = None,
) -> TaskContext:
    resolved_task = task_spec or forecast_task_spec_v1()
    if resolved_task.task_type != "forecast":
        raise ValueError("forecast_task_context_v1 requires a forecast TaskSpec")
    resolved_quality = quality_contract or forecast_task_quality_contract_v1()
    if resolved_quality.task_type != "forecast":
        raise ValueError("forecast_task_context_v1 requires a forecast quality contract")
    return TaskContext(
        task_spec=resolved_task,
        quality_contract=resolved_quality,
        deployment_constraints=deployment_constraints or deployment_constraints_v1(),
    )


def classification_task_spec_v1(
    *,
    downstream_model_class: str = "rocket_ridge",
    forbidden_modifications: Tuple[str, ...] = (),
    metric: Optional[MetricSpec] = None,
) -> TaskSpec:
    """classify C1 rig 口径：accuracy(higher) + ROCKET 确定性判官。"""
    return TaskSpec(
        task_type="classification",
        target_semantics="class_label",
        label_availability="train_labels",
        metric=metric or MetricSpec("accuracy", "higher_is_better"),
        horizon=None,
        downstream_model_class=downstream_model_class,
        forbidden_modifications=forbidden_modifications,
    )


def anomaly_task_spec_v1(
    *,
    downstream_model_class: str = "residual_zscore_detector",
    forbidden_modifications: Tuple[str, ...] = (),
    metric: Optional[MetricSpec] = None,
) -> TaskSpec:
    """P2 最小 anomaly rig 的预留口径（固定检测器 + F1 判官；rig 落地时可改具体判官名）。"""
    return TaskSpec(
        task_type="anomaly_detection",
        target_semantics="anomaly_events",
        label_availability="unlabeled",
        metric=metric or MetricSpec("F1", "higher_is_better"),
        horizon=None,
        downstream_model_class=downstream_model_class,
        forbidden_modifications=forbidden_modifications,
    )


__all__ = [
    "ABSTENTION_VOCABULARY",
    "DeploymentConstraintSpec",
    "EVIDENCE_VOCABULARY",
    "HARM_VOCABULARY",
    "LABEL_AVAILABILITY",
    "PRESERVATION_VOCABULARY",
    "QUALITY_OBJECTIVES",
    "TARGET_SEMANTICS_BY_TASK",
    "TASK_TYPES",
    "TaskContext",
    "TaskQualityContract",
    "VERIFICATION_VOCABULARY",
    "MetricSpec",
    "TaskSpec",
    "anomaly_task_spec_v1",
    "classification_task_spec_v1",
    "deployment_constraints_v1",
    "forecast_task_context_v1",
    "forecast_neutral_task_quality_contract_v1",
    "forecast_task_quality_contract_v1",
    "forecast_task_spec_v1",
]
