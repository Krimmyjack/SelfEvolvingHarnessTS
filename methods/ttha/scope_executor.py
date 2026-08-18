"""methods/ttha/scope_executor.py — 规范 Scope 执行器（training_windows_only）。

审查裁决（2026-08-08 十一）：把实验 Runner 中已验证的 scope executor/feedback
接入真实方法链。本组件是确定性 Runtime 的执行端：

  - **消费 Typed Workflow steps**（直接来自 Fast Agent 返回的
    trace.candidate_program_steps——不按算子名/参数重建，杜绝
    "Agent 返回 A、Runner 评估 B"）；
  - **同一组件完成**：窗口级 verifier（H0 max_modified_fraction，逐训练窗口
    独立 verify_candidate）→ 逐训练窗口执行（v6._evaluate 协议：对每个
    (train_series × anchor) 的 240 步窗口执行 run_pipeline）→ cohort Ridge
    Support receipt（gain / per_view_gain / behavior 计数）→ baseline 缓存；
  - 评估协议与 V1 主任务（v6._evaluate）逐位同构：同一窗口集合、同一
    anchor 过滤（anchor + HORIZON > origin 跳过）、同一 Ridge
    （_exact_weighted_ridge_prediction，alpha=1、unpenalized intercept）。

窗口 verifier 与执行器作用于**同一窗口集合**：verifier 用 verify_candidate
（内部 run_pipeline），评估用 _apply_program（内部同样 run_pipeline）——
Part A 已实证同一输入/参数/Program 下两者逐位一致。

与 fast_agent 的分工（审查裁决）：Fast Agent 生成 Typed Workflow（其内部
prefix 级验证保留为候选供给守卫，0.35 已生效）；本组件按规范 Scope 执行并
决定"该 Workflow 在当前决策点是否合法、增益多少"。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import CompiledWorkflow
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate

HORIZON = 48
CONTEXT_LENGTH = 192


@dataclass
class WindowVerification:
    """窗口级 verifier 结果（training_windows_only 语义）。

    E-1（rev4）新增字段只暴露 verify_candidate 的机械产物：
    - ``window_behavior_hashes``：每窗 prepared values 的 SHA-256（行为指纹，
      零 Outcome——不比较 downstream utility）；
    - ``window_modified_flags``：每窗是否至少修改一个已有观测；
    - ``window_identity_equivalent_flags``：每窗是否与 identity 字节等价。
    """
    passed: bool
    checked_windows: int
    rejected_windows: list[dict[str, Any]] = field(default_factory=list)
    window_behavior_hashes: tuple[str, ...] = ()
    window_modified_flags: tuple[bool, ...] = ()
    window_identity_equivalent_flags: tuple[bool, ...] = ()

    @property
    def modified_windows(self) -> int:
        return sum(1 for flag in self.window_modified_flags if flag)

    @property
    def identity_equivalent_windows(self) -> int:
        return sum(1 for flag in self.window_identity_equivalent_flags if flag)


@dataclass
class SupportReceipt:
    """一次决策点的 Support receipt（同一组件产出，供写 Episode 使用）。"""
    origin: int
    verification: WindowVerification
    gain: float | None          # None = 窗口 verifier 未通过或评估仪器失败
    per_view_gain: list[float] = field(default_factory=list)
    behavior_point_count: int = 0
    error: str | None = None


class ScopeExecutor:
    """确定性 Runtime：把 Typed Workflow steps 应用到 cohort 训练窗口并出 receipt。

    evaluate_fn 由调用方注入（实验层 v6._evaluate 即规范协议；方法层不反向
    依赖实验脚本）。签名：f(roster, values, compiled_or_None, config, *, origin)
    → {"mean_smase": float, "per_view_smase": [float], "behavior_point_count": int}。
    """

    def __init__(
        self,
        roster: Sequence[Mapping[str, object]],
        values: Mapping[str, Any],
        config: Mapping[str, object],
        *,
        evaluate_fn: Any | None = None,
        max_modified_fraction: float = 0.35,
        preserve_outside: bool = True,
    ) -> None:
        self.roster = list(roster)
        self.values = values
        self.config = dict(config)
        self._evaluate_impl = evaluate_fn
        self.max_modified_fraction = float(max_modified_fraction)
        self.preserve_outside = bool(preserve_outside)
        self._baseline_cache: dict[int, float] = {}
        self._per_view_cache: dict[int, list[float]] = {}

    def _evaluate(self, roster, values, compiled, config, *, origin: int) -> dict[str, Any]:
        if self._evaluate_impl is None:
            raise RuntimeError(
                "ScopeExecutor requires evaluate_fn (v6._evaluate) to be injected")
        return self._evaluate_impl(roster, values, compiled, config, origin=origin)

    # -- 窗口集合（与 v6._evaluate 完全同构）---------------------------------

    def training_windows(self, origin: int) -> list[tuple[str, int, np.ndarray]]:
        """(series_uid, anchor, window) 列表——v6._evaluate 同一过滤规则：
        train rows × config anchors，anchor + HORIZON > origin 跳过。"""
        windows: list[tuple[str, int, np.ndarray]] = []
        for row in self.roster:
            if str(row["role"]) != "train":
                continue
            uid = str(row["series_uid"])
            raw = np.asarray(self.values[uid], dtype=np.float64)
            for anchor in self.config["anchors"]:  # type: ignore[union-attr]
                anchor = int(anchor)
                if anchor + HORIZON > origin:
                    continue
                windows.append((uid, anchor, raw[anchor - CONTEXT_LENGTH: anchor + HORIZON]))
        return windows

    # -- 候选/编译（从 steps 构造，不重建参数）-------------------------------

    def _candidate(self, steps: Sequence[tuple[str, Mapping[str, object]]]) -> Candidate:
        program = Program.from_steps(list(steps), source="scope_executor")
        return Candidate(
            candidate_id="scope_executor",
            kind=CandidateKind.PROGRAM,
            program=program,
            source="scope_executor",
        )

    def _compiled(self, steps: Sequence[tuple[str, Mapping[str, object]]]) -> CompiledWorkflow:
        candidate = self._candidate(steps)
        assert candidate.program is not None
        return CompiledWorkflow(candidate, (), tuple(candidate.program.steps))

    @staticmethod
    def _operator_names(steps: Sequence[tuple[str, Mapping[str, object]]]) -> tuple[str, ...]:
        return tuple(op for op, _params in steps)

    # -- 窗口级 verifier（H0 约束，逐窗口独立验证）---------------------------

    def verify(self, steps: Sequence[tuple[str, Mapping[str, object]]],
               origin: int) -> WindowVerification:
        """对**实际将执行的每个训练窗口**独立 verify_candidate；保持 H0
        max_modified_fraction（0.35）。窗口即候选作用区域：inspected_regions
        覆盖整个窗口，窗口外修改不在此协议内（Workflow 只在窗口上执行）。"""
        candidate = self._candidate(steps)
        allowed = self._operator_names(steps)
        rejected: list[dict[str, Any]] = []
        checked = 0
        behavior_hashes: list[str] = []
        modified_flags: list[bool] = []
        identity_equivalent_flags: list[bool] = []
        for uid, anchor, window in self.training_windows(origin):
            checked += 1
            artifact = verify_candidate(
                candidate, window,
                allowed_operators=allowed,
                inspected_regions=((0, int(window.size)),),
                maximum_modified_fraction=self.max_modified_fraction,
                preserve_outside_inspected_region=self.preserve_outside,
                require_finite_output=False,
            )
            prepared = artifact.prepared_values
            behavior_hashes.append(
                hashlib.sha256(np.asarray(prepared).tobytes(order="C")).hexdigest()
                if prepared is not None else ""
            )
            modified_flags.append(bool(artifact.modified_indices))
            identity_equivalent_flags.append(
                artifact.receipt.effect_equivalent_to_identity
            )
            if not artifact.selectable:
                rejected.append({
                    "series_uid": uid,
                    "anchor": anchor,
                    "rejection_code": artifact.receipt.rejection_code,
                })
        return WindowVerification(
            passed=not rejected,
            checked_windows=checked,
            rejected_windows=rejected,
            window_behavior_hashes=tuple(behavior_hashes),
            window_modified_flags=tuple(modified_flags),
            window_identity_equivalent_flags=tuple(identity_equivalent_flags),
        )

    # -- 评估（v6._evaluate 协议：逐窗口执行 + cohort Ridge）------------------

    def _baseline(self, origin: int) -> dict[str, Any]:
        if origin not in self._baseline_cache:
            result = self._evaluate(self.roster, self.values, None, self.config,
                                    origin=origin)
            self._baseline_cache[origin] = float(result["mean_smase"])
            self._per_view_cache[origin] = [float(v) for v in result["per_view_smase"]]
        return {"mean_smase": self._baseline_cache[origin],
                "per_view_smase": self._per_view_cache[origin]}

    def evaluate(self, steps: Sequence[tuple[str, Mapping[str, object]]],
                 origin: int) -> SupportReceipt:
        """同一组件：窗口 verifier → 逐窗口执行 → cohort Support receipt。

        gain = baseline_mean_smase − candidate_mean_smase（处理后训练数据 →
        原始未来评价，与 V1 gain 语义一致）。
        """
        verification = self.verify(steps, origin)
        if not verification.passed:
            return SupportReceipt(
                origin=origin, verification=verification, gain=None,
                error=f"WINDOW_VERIFIER_REJECTED "
                      f"({len(verification.rejected_windows)} windows)")

        try:
            baseline = self._baseline(origin)
            candidate_result = self._evaluate(
                self.roster, self.values, self._compiled(steps), self.config,
                origin=origin)
        except Exception as exc:  # 仪器失败不伪装成负经验
            return SupportReceipt(
                origin=origin, verification=verification, gain=None,
                error=f"{type(exc).__name__}: {exc}")
        per_view = [float(ref - cand) for ref, cand in zip(
            baseline["per_view_smase"], candidate_result["per_view_smase"])]
        return SupportReceipt(
            origin=origin, verification=verification,
            gain=float(baseline["mean_smase"] - candidate_result["mean_smase"]),
            per_view_gain=per_view,
            behavior_point_count=int(candidate_result["behavior_point_count"]))
