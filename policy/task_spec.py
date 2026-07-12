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
from dataclasses import dataclass
from typing import Optional, Tuple

TASK_TYPES = ("forecast", "classification", "anomaly_detection")
TARGET_SEMANTICS_BY_TASK = {
    "forecast": "future_values",
    "classification": "class_label",
    "anomaly_detection": "anomaly_events",
}
LABEL_AVAILABILITY = ("history_only", "train_labels", "unlabeled")
_METRIC_DIRECTIONS = ("lower_is_better", "higher_is_better")


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
