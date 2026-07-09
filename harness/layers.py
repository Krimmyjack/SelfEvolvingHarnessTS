"""harness/layers.py — 四层可编辑 Harness 的数据结构 + 冷启动最小值。

来源：plan.md §2/§3 + Implementation_Design §3.1–3.4（已与 plan.md 同步：
task_templates 扁平为 Dict[name, PipelineTemplate]；L4 evaluator_registry 拆成
proxy_evaluators / grounded_evaluators；EvaluatorSpec 去掉 layer 字段）。

每层区分「可编辑面」与「只读基础设施」。精确权限/范围由 editable_surfaces.py 声明，
本文件只定义结构与冷启动最小值（minimal_l*()）。最小值用工厂函数返回**新实例**，
避免可变默认被多个 HarnessState 共享。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

TASK_TYPES = ("forecast", "anomaly_detection", "classification")


# ════════════════════════════ L1 Instructions ════════════════════════════
@dataclass
class L1Instructions:
    system_prompt: str = ""                                                  # ❌ 只读（冷启动一次性）
    constraints: List[str] = field(default_factory=list)                     # ✅ list_scalar
    recovery_rules: List[Dict[str, Any]] = field(default_factory=list)       # ✅ named_object (key=trigger)
    task_prompts: Dict[str, str] = field(default_factory=dict)               # ✅ leaf (key ∈ TASK_TYPES, value=str)
    task_sensitivity: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # ✅ leaf (value=dict)


def minimal_l1() -> L1Instructions:
    return L1Instructions(
        system_prompt=(
            "You are an agent responsible for preparing time-series data for downstream tasks. "
            "Your goal: produce a ready-to-use data artifact. "
            "Preserve the original sequence length. Avoid introducing NaN/Inf."
        ),
        constraints=[
            "Output length must equal input length",
            "Output must not contain NaN or Inf",
            "Output value range should be within [input_min - 3σ, input_max + 3σ]",
        ],
        recovery_rules=[
            {"trigger": "stl_decompose_failed", "fallback": "fft_lowpass"},
            {"trigger": "memory_error", "fallback": "chunked_processing"},
        ],
        task_prompts={
            "forecast": "You are preparing data for a forecasting model.",
            "anomaly_detection": "You are preparing data for an anomaly detection model.",
            "classification": "You are preparing data for a classification model.",
        },
        task_sensitivity={
            "forecast": {"preserve": ["trend", "seasonal_structure"], "suppress": ["high_freq_noise"]},
            "anomaly_detection": {"preserve": ["spikes", "changepoints"], "suppress": []},
            "classification": {"preserve": ["discriminative_shapes"], "suppress": ["within_class_noise"]},
        },
    )


# ════════════════════════════════ L2 Skills ════════════════════════════════
@dataclass
class OperatorConfig:
    """L2 operator_registry 的值 —— 只读配置面元数据（算子代码在 operators/ 基础设施里）。"""
    name: str
    category: str = ""        # impute | denoise | outlier | decompose | align | shape
    stage: str = ""           # s1 | s2 | s3
    tags: List[str] = field(default_factory=list)   # smoothing | destructive | scale_change ...
    applies_when: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageDef:
    stage: str
    preferred_ops: List[str] = field(default_factory=list)
    banned_ops: List[str] = field(default_factory=list)
    params_override: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StageDef":
        return cls(stage=str(d.get("stage", "")),
                   preferred_ops=list(d.get("preferred_ops", []) or []),
                   banned_ops=list(d.get("banned_ops", []) or []),
                   params_override=dict(d.get("params_override", {}) or {}))


@dataclass
class PipelineTemplate:
    name: str                                                # 全局唯一（= named_object 寻址 key）
    applies_to: Dict[str, Any] = field(default_factory=dict)  # {"task_type": str, "pattern_conditions": {...}|None}
    stages: List[StageDef] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineTemplate":
        """从 LLM JSON 重建（proposer 用）：stages 的 dict → StageDef。"""
        stages = [s if isinstance(s, StageDef) else StageDef.from_dict(s) for s in d.get("stages", []) or []]
        return cls(name=str(d.get("name", "")), applies_to=dict(d.get("applies_to", {}) or {}), stages=stages)


@dataclass
class MiddlewareDef:
    name: str
    composed_of: List[str] = field(default_factory=list)     # ⊆ operator_registry
    description: str = ""
    created_in_harness_version: int = 0


@dataclass
class L2Skills:
    operator_registry: Dict[str, OperatorConfig] = field(default_factory=dict)   # ❌ 只读基础设施
    active_operators: Dict[str, bool] = field(default_factory=dict)              # ✅ leaf (value=bool)
    operator_defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # ✅ leaf (value=dict)
    task_templates: Dict[str, PipelineTemplate] = field(default_factory=dict)    # ✅ named_object (key=name)
    middlewares: List[MiddlewareDef] = field(default_factory=list)               # ✅ named_object (key=.name)


def minimal_l2() -> L2Skills:
    # 算子 metadata 单一真源 = operators/registry.py（避免名字清单两处漂移）。
    # S0.7-6：harness 配置面只含 canonical 算子（alias 仅供旧 artifact 重放，执行层解析）。
    from ..operators.registry import OPERATOR_METADATA
    registry = {n: OperatorConfig(name=n, category=m["category"], stage=m["stage"], tags=list(m["tags"]))
                for n, m in OPERATOR_METADATA.items() if not m.get("is_alias")}
    return L2Skills(
        operator_registry=registry,
        active_operators={n: True for n in registry},        # 全算子激活：让 LLM 自由编排
        operator_defaults={
            "denoise_savgol": {"window": 11, "order": 3},    # [待校准]（order→scipy polyorder 由算子内部翻译）
            "denoise_median": {"window": 5},                 # [待校准]
            "smooth_ma": {"window": 5},                       # F0 剂量维默认（variant 用 params_override 覆盖 9/15/25）
            "stl_decompose": {"period": 0},                  # 0=自动猜测周期（非 "auto" 字符串：会被 STL 比较吞成 savgol 回退）
        },
        task_templates={},                                   # 最小 harness 无模板（LLM 完全自由编排）
        middlewares=[],
    )


# ════════════════════════════════ L3 Memory ════════════════════════════════
@dataclass
class RetrievalConfig:
    alpha: float = 0.5             # d_struct vs d_quality 权重 [待校准]
    min_similarity: float = 0.6    # 低于此值不返回 [待校准]
    max_prior_fragments: int = 5
    max_failure_warnings: int = 5
    injection_template: str = (
        "## Similar historical cases (advisory only):\n{prior_fragments}\n"
        "## Known failure patterns to avoid:\n{failure_warnings}\n"
    )


@dataclass
class FailureSignatureStats:
    """⚙️ 由 slow_path/mining 直接维护（非 EditPatch 面）。"""
    signature_id: str
    count: int = 0
    support: int = 0
    addressability: str = "unknown"      # high | low | unknown
    last_seen_version: int = 0


@dataclass
class StrengthSignatureStats:
    """🔒 受保护面（承重墙）：只许 consolidator 整固写。"""
    signature_id: str
    cell_id: str = ""
    win_margin: float = 0.0              # val_loss 相对 baseline 的优势
    support: int = 0
    must_preserve: bool = True
    promoted_in_version: int = 0


@dataclass
class L3Memory:
    retrieval_config: RetrievalConfig = field(default_factory=RetrievalConfig)            # ✅ leaf
    evidence_store: Optional[Any] = None                                                  # ❌ 后端（不进快照；外部注入）
    failure_signatures: Dict[str, FailureSignatureStats] = field(default_factory=dict)    # ⚙️ mining 维护
    strength_signatures: Dict[str, StrengthSignatureStats] = field(default_factory=dict)  # 🔒 protected


def minimal_l3() -> L3Memory:
    return L3Memory(retrieval_config=RetrievalConfig())


# ════════════════════════════ L4 Verification ════════════════════════════
@dataclass
class GateConfig:
    blowup_sigma: float = 10.0                    # 输出超 μ+Nσ → 爆炸 [待校准]
    constraint_violation_tolerance: float = 0.05  # violation 升高 ≤ 5pp 容忍 [待校准]
    sandbox_timeout_seconds: int = 60
    ast_check_enabled: bool = True


@dataclass
class EvaluatorSpec:
    """proxy/grounded 由所在 dict（proxy_evaluators / grounded_evaluators）表达，无 layer 字段。"""
    task_type: str
    eval_type: str = ""        # backtest | injection | train_classifier
    model: str = ""
    metric: str = ""           # rmse | recall | cross_entropy | silhouette
    params: Dict[str, Any] = field(default_factory=dict)
    epsilon: float = 0.03


@dataclass
class ShrinkageConfig:
    enabled: bool = True
    confidence_threshold: float = 0.5
    shrinkage_factor: float = 0.5
    frozen_domains: List[str] = field(default_factory=list)


@dataclass
class L4Verification:
    gate_config: GateConfig = field(default_factory=GateConfig)                  # ✅ leaf (step)
    proxy_evaluators: Dict[str, EvaluatorSpec] = field(default_factory=dict)     # ✅ named_object (step)；key=task_type
    grounded_evaluators: Dict[str, EvaluatorSpec] = field(default_factory=dict)  # 🔒 named_object (consolidator)；裁判
    shrinkage_config: ShrinkageConfig = field(default_factory=ShrinkageConfig)   # ✅ leaf (step)


def minimal_l4() -> L4Verification:
    return L4Verification(
        gate_config=GateConfig(),
        proxy_evaluators={  # Layer-1 预筛（轻量下游，参照 AegisTS §6.1.4）
            "forecast": EvaluatorSpec("forecast", "backtest", "dlinear", "rmse", {"horizon": 24}, 0.01),
            "anomaly_detection": EvaluatorSpec("anomaly_detection", "injection", "catch22_clusterer", "recall", {"contamination": 0.05}, 0.05),
            "classification": EvaluatorSpec("classification", "train_classifier", "minirocket", "cross_entropy", {"cv_folds": 5}, 0.02),
        },
        grounded_evaluators={  # 🔒 Layer-2 accept 裁判（强表征目标模型；默认套 frozen_probe 底座）
            "forecast": EvaluatorSpec("forecast", "backtest", "lstm_forecast", "rmse", {"horizon": 24, "epochs": 50}, 0.01),
            "anomaly_detection": EvaluatorSpec("anomaly_detection", "injection", "aedcnn_clusterer", "recall", {"contamination": 0.05}, 0.05),
            "classification": EvaluatorSpec("classification", "train_classifier", "inception_time", "cross_entropy", {"cv_folds": 5}, 0.02),
        },
        shrinkage_config=ShrinkageConfig(),
    )
