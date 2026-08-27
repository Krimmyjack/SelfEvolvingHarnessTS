from __future__ import annotations
import dataclasses
import json

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.contracts.harness import HarnessSnapshot, SkillKind
from SelfEvolvingHarnessTS.contracts.method import (
    ExecutionReceipt,
    PreparationRequest,
    PreparationResult,
    PreparationStatus,
    PreparedSeries,
)
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.operators.registry import (
    OPERATOR_METADATA,
    OPERATOR_NAMES,
    operator_targeting_mode,
)
from SelfEvolvingHarnessTS.runtime.candidate_pool import (
    CandidatePool,
    ProtocolChoiceError,
)
from SelfEvolvingHarnessTS.runtime.candidate_verification import (
    CandidateExecutionArtifact,
    verify_candidate,
)
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

from .agent_core import (
    AgentProtocolError,
    AgentRole,
    AgentStageResult,
    StagePostValidationError,
    TTHAAgentCore,
)
from .public_tools import extract_public_features
from .retrieval import EffectiveHarnessView, resolve_harness_view


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def _cand_ops(c: object) -> tuple[str, ...]:
    """候选程序执行的算子序列（E2.5-A 双槽去重/探索检查用）。"""
    program = getattr(c, "program", None)
    if program is None:
        return ()
    return tuple(op for op, _p in program.execution_steps())


def _allowed_operators(request: PreparationRequest) -> tuple[str, ...]:
    allowed: list[str] = []
    for name in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[name]
        if request.task_spec.task_type not in metadata["allowed_tasks"]:
            continue
        if metadata.get("shape_changing"):
            continue
        if request.task_spec.is_op_forbidden(name):
            continue
        allowed.append(name)
    return tuple(allowed)


# 仅处理缺失/损坏数据的算子族（Operator 前提：需要缺失信号才有行为差异）。
_MISSING_ONLY_OPS = frozenset({
    "impute_ar", "impute_ema", "impute_fft", "impute_linear", "impute_ssm",
    "period_complete", "period_median_complete",
})


def _noop_ops_for_context(request: PreparationRequest) -> tuple[str, ...]:
    """依据公开 Context 与 Operator 前提的确定性 no-op 算子集合（不读取
    gain）。当前 Context 无缺失信号（recent.coverage == 1 且
    recent.maximum_missing_run_length == 0）时，缺失处理族是确定性 no-op
    ——供应前跳过（实验 1：Program Supply 前提修复）。
    Context 键缺失（未知缺失状态）时保守返回空（不误杀）。"""
    observed = dict(getattr(request, "observed_pattern_spec", {}) or {})
    coverage = observed.get("recent.coverage")
    max_run = observed.get("recent.maximum_missing_run_length")
    if coverage is None or max_run is None:
        return ()
    no_missing = float(coverage) >= 1.0 and int(max_run) == 0
    if not no_missing:
        return ()
    allowed = set(_allowed_operators(request))
    return tuple(sorted(_MISSING_ONLY_OPS & allowed))


def _dependency_available(dep: str) -> bool:
    """requires_dependency 的机械可得性判定（FULL_OPERATOR_SKILL_CAPABILITY
    2026-08-14）：依赖缺失时按 dependency_policy 决定保留/剔除——
    hard_fail → 剔除（执行会硬失败）；recorded_fallback → 保留
    （执行时记账回退，有定义行为）。"""
    import importlib.util as _ilu
    return _ilu.find_spec(dep) is not None


def _full_pool_operators(request: PreparationRequest) -> tuple[str, ...]:
    """FULL_OPERATOR_SKILL_CAPABILITY（2026-08-14，arm B 池）：26 canonical
    算子经四类机械过滤后的全部合法算子——

      1. Task：task_type ∈ allowed_tasks；
      2. 输入形状：shape_changing → 排除；
      3. TaskSpec 禁面；
      4. 依赖：requires_dependency 缺失且 dependency_policy=hard_fail → 排除；
      5. 执行前提：(a) 绑定声明（public_parameter_bindings）在当前 features
         不完整 → 排除（算子要求外部区域参数才能定向其效应）；
         (b) 确定性 no-op 前提（无缺失时缺失处理族）由调用方在
         propose_ops 构造处与 A 臂共用同一 _noop_ops_for_context 规则。

    **不含 verifier 可行动性探测（max_modified_fraction=0.35）**——
    B 只改暴露面；候选验证仍是运行时候选闸（deployment_constraints 经
    TaskContext 对 Agent 可见）。不根据 Outcome 人工排除算子；不含
    deprecated alias（OPERATOR_NAMES 只含 canonical）。
    """
    features = extract_public_features(
        np.asarray(request.values, dtype=float),
        task_kind=request.task_spec.task_type
        if request.task_spec is not None else "forecast")
    out: list[str] = []
    for name in OPERATOR_NAMES:
        metadata = OPERATOR_METADATA[name]
        if request.task_spec.task_type not in metadata["allowed_tasks"]:
            continue
        if metadata.get("shape_changing"):
            continue
        if request.task_spec.is_op_forbidden(name):
            continue
        dep = metadata.get("requires_dependency")
        if dep is not None and not _dependency_available(str(dep)):
            if metadata.get("dependency_policy") == "hard_fail":
                continue
        bindings = metadata.get("public_parameter_bindings") or {}
        if bindings:
            bound = {p: features[f]
                     for p, f in bindings.items() if f in features}
            if len(bound) != len(bindings):
                continue  # 执行前提未满足：绑定特征不完整
        out.append(name)
    return tuple(out)


def public_operator_contract(name: str) -> dict[str, object]:
    metadata = OPERATOR_METADATA[name]
    return {
        "name": name,
        "category": metadata["category"],
        "allowed_tasks": list(metadata["allowed_tasks"]),
        "destructive": metadata["destructive"],
        "preserves_observed": metadata["preserves_observed"],
        "changes_target_space": metadata["changes_target_space"],
        "requires_dependency": metadata["requires_dependency"],
        "dependency_policy": metadata["dependency_policy"],
        "public_parameter_bindings": dict(
            metadata.get("public_parameter_bindings", {})
        ),
        "public_parameter_schema": _plain(
            metadata.get("public_parameter_schema") or {"type": "object"}
        ),
        "targeting_mode": operator_targeting_mode(name),
    }


def _default_params_from_contract(name: str) -> dict[str, object]:
    """从 public_parameter_schema 构造最小合法参数（与候选供给同源）。

    审查裁决（2026-08-08 方案 2）：actionability 判定用同一参数构造路径，
    保证"能构造出通过 verifier 的候选"的判定与 Agent 实际提案一致。
    """
    metadata = OPERATOR_METADATA.get(name) or {}
    schema = metadata.get("public_parameter_schema")
    if not schema:
        return {}
    props = schema.get("properties") or {}
    params: dict[str, object] = {}
    for pname, spec in props.items():
        if "default" in spec:
            params[pname] = spec["default"]
        elif spec.get("type") == "integer":
            params[pname] = spec.get("minimum", 1)
        elif spec.get("type") == "number":
            params[pname] = spec.get("minimum", 1.0) or 1.0
    for req in schema.get("required") or []:
        if req not in params:
            spec = props.get(req, {})
            if spec.get("type") == "integer":
                params[req] = spec.get("minimum", 1)
            else:
                params[req] = 1
    return params


def _actionable_operators(
    request: PreparationRequest,
    values: np.ndarray,
    view: EffectiveHarnessView,
    allowed: Sequence[str],
) -> tuple[str, ...]:
    """候选供给与 verifier 对齐（审查裁决 2026-08-08 方案 2）。

    对每个 allowed 算子构造默认候选并实测当前 verifier（H0 部署约束：
    max_modified_fraction 等）——selectable 的才是可行动算子。全局平滑类
    算子（如 denoise_stl，修改分数 1.0 > 0.35）被排除：Memory 可保留其
    Episode，但不渲染为可执行 Reference，也不进入 propose 的 contracts。
    """
    maximum_modified_fraction, preserve_outside = _verification_limits(request, view)
    inspected = ((0, int(values.size)),)
    actionable: list[str] = []
    features_mapping: Mapping[str, object] | None = None
    for name in allowed:
        # CONTEXT_BOUND_PROGRAM_SUPPLY（2026-08-10，统一供应修复 a）：带公开
        # 参数绑定的算子（如 repair_level_shift）用部署可见 Context 特征值
        # 构造**绑定参数候选**实测 verifier——默认全局模式被 verifier 拒绝
        # （repair_level_shift 修改 45.7% > 0.35）不代表绑定模式不可行动。
        # 绑定不完整（特征缺失）→ 不可行动（不 fallback 到默认全局模式）。
        bindings = OPERATOR_METADATA[name].get("public_parameter_bindings") or {}
        if bindings:
            if features_mapping is None:
                features_mapping = dict(extract_public_features(
                    values,
                    task_kind=request.task_spec.task_type
                    if request.task_spec is not None else "forecast"))
            params = {p: features_mapping[f]
                      for p, f in bindings.items() if f in features_mapping}
            if len(params) != len(bindings):
                continue
        else:
            params = _default_params_from_contract(name)
        program = Program.from_steps([(name, params)], source="actionability_probe")
        candidate = Candidate.program_candidate(
            f"probe_{name}", program, source="actionability_probe")
        artifact = verify_candidate(
            candidate,
            values,
            allowed_operators=(name,),
            inspected_regions=inspected,
            maximum_modified_fraction=maximum_modified_fraction,
            preserve_outside_inspected_region=preserve_outside,
            require_finite_output=request.task_context is not None,
        )
        if artifact.selectable:
            actionable.append(name)
    return tuple(actionable)


def public_operator_contracts_for_task(task_kind: str) -> tuple[dict[str, object], ...]:
    """Return the deployment-safe operator menu from the registry single source."""

    return tuple(
        public_operator_contract(name)
        for name in OPERATOR_NAMES
        if task_kind in OPERATOR_METADATA[name]["allowed_tasks"]
        and not OPERATOR_METADATA[name].get("shape_changing")
    )


def _parse_frozen_steps(body: str) -> tuple[tuple[str, Mapping[str, object]], ...] | None:
    """P3 方法层（审查裁决 2026-08-09）：解析 CAPABILITY skill body 中
    "Frozen program steps:" marker 后的 JSON 数组 → Typed steps。

    解析失败（无 marker / JSON 非法 / 非列表 / 元素结构非法 / 空）→ None
    （ACTION_UNAVAILABLE 语义：不提供候选）。
    """
    marker = "Frozen program steps:"
    idx = body.find(marker)
    if idx < 0:
        return None
    seg = body[idx + len(marker):].strip()
    try:
        raw_steps = json.loads(seg)
    except json.JSONDecodeError:
        return None
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    steps: list[tuple[str, Mapping[str, object]]] = []
    for item in raw_steps:
        if not isinstance(item, Mapping) or not isinstance(item.get("op"), str):
            return None
        params = item.get("params")
        if not isinstance(params, Mapping):
            return None
        steps.append((item["op"], dict(params)))
    return tuple(steps) if steps else None


def _signed_reference_ops(instruction: str) -> tuple[set[str], set[str]]:
    """从渲染后的 instruction 解析 Reference 1（POSITIVE）与 2/3（CONFLICT/
    RISK）的算子集合（NORMAL_ENTRY_SIGNED_FEEDBACK_TO_SKILL_CONTROL 用）。

    render_signed_instruction 输出格式：'Reference N: candidate operators
    [...]'（Python repr）。解析失败返回空集（视为无反馈）。
    """
    import ast as _ast
    pos: set[str] = set()
    risk: set[str] = set()
    for num, target in ((1, pos), (2, risk), (3, risk)):
        match = re.search(rf"Reference {num}: candidate operators (\[[^\]]*\])",
                          instruction)
        if not match:
            continue
        try:
            parsed = _ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, list):
            target.update(str(op) for op in parsed)
    return pos, risk


def _skill_frozen_candidates(view: EffectiveHarnessView,
                             features: Mapping[str, object]
                             ) -> tuple[Candidate, ...]:
    """P3 方法层（外部审核裁决 2026-08-09，METHOD_LEVEL_CREDIT_TO_UPDATE_
    BINDING）：view.skills 中 capability skill 的冻结 Typed steps → PROGRAM
    Candidate（cand_skill_{skill_id}），与 Agent proposals 合并进入池。

    - view.skills 已经过 retrieval 的 applicability 过滤——**Context 不匹配
      不供应**（由检索层保证）；
    - 解析失败 → 不供应（ACTION_UNAVAILABLE）；
    - 不读取 future：候选只含冻结 steps；
    - 优先级由调用点（prepare 注入）按 signed 判定决定。
    """
    out: list[Candidate] = []
    for skill in view.skills:
        if skill.skill_kind is not SkillKind.CAPABILITY:
            continue
        steps = _parse_frozen_steps(skill.body)
        if steps is None:
            continue
        # CONTEXT_BOUND_SKILL_REBINDING（用户裁决 2026-08-12）：Skill 是
        # 可重用 Workflow 模板——每轮按当前公开 Context 通过 registry 的
        # public_parameter_bindings 重新绑定动态参数（body 旧参数是上一轮
        # 的绑定实例——不沿用）；绑定特征缺失/非法 → 安全拒绝该 Skill。
        rebound: list[tuple[str, Mapping[str, object]]] = []
        ok = True
        for op, params in steps:
            bindings = (OPERATOR_METADATA.get(op) or {}) \
                .get("public_parameter_bindings") or {}
            if bindings:
                bound = {name: features[f]
                         for name, f in bindings.items() if f in features}
                if len(bound) != len(bindings):
                    ok = False  # 绑定特征缺失 → 安全拒绝
                    break
                rebound.append((op, bound))
            else:
                rebound.append((op, dict(params)))
        if not ok or not rebound:
            continue
        program = Program.from_steps(
            rebound, source=f"skill:{skill.skill_id}")
        out.append(Candidate.program_candidate(
            f"cand_skill_{skill.skill_id}", program,
            source=f"skill:{skill.skill_id}"))
    return tuple(out)


# The permission rung a Skill entry was granted on the supply axis.  Who gets
# it is decided outside this package -- Slow consolidation, or the runner that
# compiles the card.  Nothing here grants, widens or audits the flag; this is
# only the reader that makes an already-granted flag effective.
_SUPPLY_AUTHORITY_KEY = "supplies_candidates"


def _supplies_candidates(skill: Any) -> bool:
    guards = getattr(skill, "risk_guards", None) or {}
    authority = guards.get("authority") or {}
    return authority.get(_SUPPLY_AUTHORITY_KEY) is True


def _supply_rung_candidates(view: EffectiveHarnessView,
                            features: Mapping[str, object]
                            ) -> tuple[Candidate, ...]:
    """The frozen programs the retrieved view is authorized to *supply*.

    A subset of :func:`_skill_frozen_candidates`, selected by the entry's own
    ``risk_guards.authority.supplies_candidates``.  Scope is not re-checked
    here: ``view.skills`` is already the applicability-filtered view, which is
    the layer that owns Scope.
    """
    supplying = {
        str(skill.skill_id) for skill in view.skills
        if _supplies_candidates(skill)
    }
    if not supplying:
        return ()
    out: list[Candidate] = []
    for candidate in _skill_frozen_candidates(view, features):
        source = str(getattr(candidate, "source", "") or "")
        if source.startswith("skill:") and source[len("skill:"):] in supplying:
            out.append(candidate)
    return tuple(out)


def _compile_candidates(
    payload: Mapping[str, object],
    request: PreparationRequest,
) -> tuple[tuple[Candidate, ...], Mapping[str, str]]:
    """编译候选并保留 candidate_id → addresses_hypothesis_id 映射
    （STRUCTURED_SKILL 引用链 2026-08-14：假设引用不得在编译处丢失——
    下游 select 载荷需携带；旧载荷无该字段 → 映射为空，兼容）。"""
    allowed = set(_allowed_operators(request))
    candidates: list[Candidate] = []
    hypothesis_map: dict[str, str] = {}
    for candidate_payload in payload["candidates"]:
        candidate_id = candidate_payload["candidate_id"]
        if candidate_id == "identity":
            raise AgentProtocolError("Agent cannot supply runtime identity")
        hypothesis_id = candidate_payload.get("addresses_hypothesis_id")
        if isinstance(hypothesis_id, str) and hypothesis_id:
            hypothesis_map[str(candidate_id)] = hypothesis_id
        steps: list[tuple[str, Mapping[str, object]]] = []
        for step in candidate_payload["steps"]:
            op = step["op"]
            params = step["params"]
            if op not in allowed:
                raise AgentProtocolError(f"operator is not allowed for task: {op}")
            canonical_json_bytes(params)
            steps.append((op, params))
        program = Program.from_steps(steps, source="agent")
        candidates.append(
            Candidate.program_candidate(candidate_id, program, source="agent")
        )
    return tuple(candidates), hypothesis_map


def _validate_inspect_hypotheses(
    payload: Mapping[str, object],
    features: Mapping[str, object],
) -> None:
    """STRUCTURED_SKILL 引用链（2026-08-14）：pattern_hypotheses 存在时，
    evidence_features 必须全部来自当前公开 features；hypothesis_id 不得
    重复。字段缺失（旧载荷/回放）→ 直接通过（兼容冻结证据）。"""
    hypotheses = payload.get("pattern_hypotheses") or ()
    if not isinstance(hypotheses, Sequence) or isinstance(
            hypotheses, (str, bytes, bytearray)) or not hypotheses:
        return
    seen: set[str] = set()
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, Mapping):
            continue
        hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
        if hypothesis_id and hypothesis_id in seen:
            raise StagePostValidationError(
                "HYPOTHESIS_ID_DUPLICATE",
                f"pattern_hypotheses hypothesis_id '{hypothesis_id}' appears "
                "more than once; use disjoint canonical ids.",
                retryable=True)
        if hypothesis_id:
            seen.add(hypothesis_id)
        evidence = hypothesis.get("evidence_features") or ()
        if not isinstance(evidence, Sequence) or isinstance(
                evidence, (str, bytes, bytearray)):
            continue
        for feature in evidence:
            if feature not in features:
                raise StagePostValidationError(
                    "HYPOTHESIS_EVIDENCE_UNGROUNDED",
                    f"evidence feature '{feature}' is not a deployment-visible "
                    "public feature; cite only features from the current "
                    "feature summary.",
                    retryable=True)


def _validate_hypothesis_references(
    payload: Mapping[str, object],
    inspection_payload: Mapping[str, object],
) -> None:
    """STRUCTURED_SKILL 引用链（2026-08-14）：候选的
    addresses_hypothesis_id 必须引用 inspect 阶段实际输出的 hypothesis。
    字段缺失（旧载荷/回放）→ 直接通过（兼容冻结证据）。"""
    candidates = payload.get("candidates") or ()
    if not isinstance(candidates, Sequence) or isinstance(
            candidates, (str, bytes, bytearray)):
        return
    hypotheses = inspection_payload.get("pattern_hypotheses") or ()
    valid_ids = {
        str(h.get("hypothesis_id"))
        for h in hypotheses
        if isinstance(h, Mapping) and h.get("hypothesis_id")
    } if isinstance(hypotheses, Sequence) and not isinstance(
        hypotheses, (str, bytes, bytearray)) else set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        hypothesis_id = candidate.get("addresses_hypothesis_id")
        if not isinstance(hypothesis_id, str) or not hypothesis_id:
            continue
        if not valid_ids or hypothesis_id not in valid_ids:
            # 反馈必须带上合法 id 集合（2026-08-19）：旧文案只说"引用一个
            # 已发出的 hypothesis"，却不告诉模型有哪些——重试因此无法收敛。
            # G2 回放实测：propose 阶段用 'extreme-deviation' 引用了 inspect
            # 未发出的 id，重试后仍失败，整个 Task 以 AGENT_PROTOCOL_ERROR
            # 结束。合法集合来自本轮 inspect 载荷，不是新信息。
            emitted = (
                ", ".join(f"'{value}'" for value in sorted(valid_ids))
                if valid_ids else "none"
            )
            raise StagePostValidationError(
                "HYPOTHESIS_REFERENCE_INVALID",
                f"addresses_hypothesis_id '{hypothesis_id}' does not reference "
                "a hypothesis_id emitted by the inspect stage. The ids emitted "
                f"this round are: {emitted}. Cite one of them exactly, or omit "
                "the field.",
                retryable=True)


def _validate_public_parameter_bindings(
    payload: Mapping[str, object],
    public_features: Mapping[str, object],
    fixed_probe_panel: Mapping[str, object] | None = None,
) -> None:
    fixed_step_signatures: set[bytes] = set()
    contracts = (fixed_probe_panel or {}).get("probe_contracts", {})
    probes = contracts.get("probes", {}) if isinstance(contracts, Mapping) else {}
    if isinstance(probes, Mapping):
        for probe in probes.values():
            if not isinstance(probe, Mapping):
                continue
            arms = probe.get("arms", ())
            if not isinstance(arms, Sequence) or isinstance(
                arms, (str, bytes, bytearray)
            ):
                continue
            for arm in arms:
                if not isinstance(arm, Mapping):
                    continue
                steps = arm.get("current_context_program_steps", ())
                if not isinstance(steps, Sequence) or isinstance(
                    steps, (str, bytes, bytearray)
                ):
                    continue
                for step in steps:
                    if isinstance(step, Mapping):
                        fixed_step_signatures.add(canonical_json_bytes(step))
    for candidate in payload["candidates"]:
        for step in candidate["steps"]:
            operator_name = step["op"]
            bindings = OPERATOR_METADATA[operator_name].get(
                "public_parameter_bindings", {}
            )
            if not bindings:
                continue
            if canonical_json_bytes(step) in fixed_step_signatures:
                continue
            params = step["params"]
            expected_keys = set(bindings)
            if set(params) != expected_keys:
                raise StagePostValidationError(
                    "PUBLIC_PARAMETER_BINDING_INVALID",
                    (
                        f"{operator_name} params must contain exactly the canonical "
                        f"keys {sorted(expected_keys)} declared in its public parameter "
                        "bindings."
                    ),
                    retryable=True,
                )
            mismatched = [
                parameter
                for parameter, feature in bindings.items()
                if feature not in public_features
                or params[parameter] != public_features[feature]
            ]
            if mismatched:
                raise StagePostValidationError(
                    "PUBLIC_PARAMETER_BINDING_INVALID",
                    (
                        f"{operator_name} bound parameter values must exactly equal "
                        "their deployment-visible feature values from the declared "
                        f"mapping; mismatched keys: {sorted(mismatched)}."
                    ),
                    retryable=True,
                )


def _regions_from_fractions(
    fractions: Sequence[Sequence[float]],
    length: int,
) -> tuple[tuple[int, int], ...]:
    regions: list[tuple[int, int]] = []
    for start_fraction, end_fraction in fractions:
        if float(end_fraction) <= float(start_fraction):
            raise AgentProtocolError("inspected region end must be after start")
        start = min(length - 1, max(0, int(math.floor(float(start_fraction) * length))))
        end = min(length, max(start + 1, int(math.ceil(float(end_fraction) * length))))
        regions.append((start, end))
    return tuple(regions)


def _verification_limits(
    request: PreparationRequest,
    view: EffectiveHarnessView,
) -> tuple[float, bool]:
    verification = view.controls.get("verification", {})
    if not isinstance(verification, Mapping):
        return 0.0, True
    maxima = [float(verification.get("max_modified_fraction", 1.0))]
    if request.task_context is not None:
        maxima.append(
            float(
                request.task_context.deployment_constraints.maximum_modified_fraction
            )
        )
    preserve_outside = verification.get("preserve_outside_candidate_region") is True
    for skill in view.skills:
        guards = skill.risk_guards
        if not isinstance(guards, Mapping):
            continue
        skill_maximum = guards.get("max_modified_fraction")
        if (
            isinstance(skill_maximum, (int, float))
            and not isinstance(skill_maximum, bool)
            and math.isfinite(float(skill_maximum))
        ):
            maxima.append(float(skill_maximum))
        preserve_outside = preserve_outside or (
            guards.get("preserve_outside_candidate_region") is True
        )
    return min(maxima), preserve_outside


def _task_binding(
    request: PreparationRequest, *, legacy_inspect_stage: bool = False
) -> dict[str, object]:
    if request.task_context is None:
        return {"task": request.task_spec.to_dict()} if legacy_inspect_stage else {}
    return {
        "task": request.task_spec.to_dict(),
        "task_context": request.task_context.to_dict(),
        "task_context_sha": request.task_context.sha(),
    }


class TTHAFastAgent:
    def __init__(self, core: TTHAAgentCore):
        self.core = core

    def _trace(
        self,
        *,
        request: PreparationRequest,
        view: EffectiveHarnessView,
        stages: Sequence[AgentStageResult],
        inspected_regions: tuple[tuple[int, int], ...],
        pool: CandidatePool | None,
        chosen_candidate_id: str,
        compilation_status: str,
        execution_status: str,
        modified_indices: tuple[int, ...],
        verification_actions: tuple[str, ...],
        identity_equivalent: bool,
        supplied_noop_candidate_ids: tuple[str, ...],
        candidate_artifacts: Mapping[str, CandidateExecutionArtifact],
        rejection_receipts: tuple[Mapping[str, object], ...],
        memory_resolution_status: str = "no_memory",
    ) -> DecisionTrace:
        tool_calls = tuple(
            {
                "tool_name": receipt.tool_name,
                "arguments": _plain(receipt.arguments),
                "public_result": _plain(receipt.public_result),
                "receipt_sha": receipt.receipt_sha,
            }
            for stage in stages
            for receipt in stage.tool_receipts
        )
        observation_ids = tuple(
            request_hash
            for stage in stages
            for request_hash in stage.request_hashes
        )
        candidates = pool.candidates if pool is not None else (Candidate.identity(),)
        candidate_program_steps = {
            candidate.candidate_id: tuple(
                (op, params) for op, params in candidate.program.execution_steps()
            )
            for candidate in candidates
            if candidate.program is not None
        }
        cache_hit_flags = tuple(
            bool(
                getattr(stage.response.cache_receipt, "hit", False)
                if stage.response.cache_receipt is not None
                else False
            )
            for stage in stages
            for _request_hash in stage.request_hashes
        )
        return DecisionTrace(
            case_id=request.series_uid,
            public_observation_ids=observation_ids,
            inspected_regions=inspected_regions,
            tool_calls=tool_calls,
            retrieved_skill_ids=view.skill_ids,
            retrieved_memory_ids=view.memory_ids,
            applicability_matches=tuple(
                entry_id for entry_id in (*view.skill_ids, *view.memory_ids)
            ),
            candidate_ids=tuple(candidate.candidate_id for candidate in candidates),
            candidate_program_shas=tuple(
                candidate.program.sha() if candidate.program is not None else None
                for candidate in candidates
            ),
            chosen_candidate_id=chosen_candidate_id,
            compilation_status=compilation_status,
            execution_status=execution_status,
            modified_indices=modified_indices,
            verification_actions=verification_actions,
            effect_equivalent_to_identity=identity_equivalent,
            series_length=request.values.size,
            supplied_noop_candidate_ids=supplied_noop_candidate_ids,
            candidate_program_steps=candidate_program_steps,
            agent_cache_hit_flags=cache_hit_flags,
            task_context_sha=(request.task_context.sha() if request.task_context else ""),
            run_context_sha=(
                request.run_dependency_binding.sha()
                if request.run_dependency_binding
                else ""
            ),
            memory_resolution_status=memory_resolution_status,
            selectable_candidate_ids=tuple(
                candidate.candidate_id for candidate in candidates
            ),
            candidate_receipt_shas={
                candidate_id: artifact.receipt.receipt_sha
                for candidate_id, artifact in candidate_artifacts.items()
                if request.task_context is not None
                and candidate_id in {candidate.candidate_id for candidate in candidates}
            },
            rejection_receipts=(
                rejection_receipts if request.task_context is not None else ()
            ),
        )

    def prepare(
        self,
        request: PreparationRequest,
        snapshot: HarnessSnapshot,
        *,
        fixed_probe_panel: Mapping[str, object] | None = None,
        experience_episodes: Sequence[object] = (),
        allowed_operators: Sequence[str] | None = None,
        calendar_period: int | None = None,
        runtime_prior_slot: bool = False,
        pool_mode: str = "actionable",
    ) -> tuple[PreparationResult, DecisionTrace]:
        # FULL_OPERATOR_SKILL_CAPABILITY（2026-08-14）：pool_mode 切换候选
        # 供给面过滤路径。"actionable"=当前管线（verifier 可行动性探测）；
        # "full"=机械过滤全池（不含 0.35 探测）。默认不变——生产路径零改动。
        if pool_mode not in ("actionable", "full"):
            raise ValueError(f"invalid pool_mode: {pool_mode!r}")
        verifier = getattr(self.core.tools, "verify_context", None)
        if verifier is not None and not verifier(
            request.values,
            task_kind=request.task_spec.task_type,
            fixed_probe_panel=fixed_probe_panel,
        ):
            raise ValueError("public tool context does not match preparation request")
        features = extract_public_features(
            request.values,
            task_kind=request.task_spec.task_type,
            fixed_probe_panel=fixed_probe_panel,
        )
        view = resolve_harness_view(snapshot, features, role="fast")
        # 候选供给与 verifier 对齐（审查裁决 2026-08-08 方案 2）：可行动算子集合
        # （构造默认候选实测当前 verifier）。用于：propose contracts 过滤 +
        # signed 渲染的可执行 Reference 过滤。Memory Episode 可保留非可执行算子，
        # 但不渲染为"建议优先探测"。
        actionable_ops: tuple[str, ...] = ()
        if request.task_spec is not None:
            actionable_ops = _actionable_operators(
                request, np.asarray(request.values, dtype=float), view,
                _allowed_operators(request))
        # FULL_OPERATOR_SKILL_CAPABILITY（2026-08-14）：propose 合约供给面
        # 来源切换。pool_mode=full 时用机械全池（不含 verifier 0.35 探测）；
        # 两臂共用下游 noop 前提过滤与绑定完整性排序。
        supply_ops: tuple[str, ...] = actionable_ops
        if pool_mode == "full" and request.task_spec is not None:
            supply_ops = _full_pool_operators(request)
        # 方法层接线（2026-08-08 审查修订）：经验对照包显式注入 LLM 可见 instruction。
        # 审查裁决：① 不隐式读全局 episodes.json（会污染 A3/H0 空 Memory）——
        #    episodes 由调用方显式传入（A3 传空、A5 传 Source Episode）；
        # ② task_consumer 从当前 TaskSpec 构造、allowed_operators 过滤；
        # ③ 检索用与 Episode 相同的特征键；无共同可识别特征时不注入；
        # ④ 该注入只改变 LLM 上下文（Memory 作为先验），Support 实测仍为最终确认。
        _memory_status = "no_memory"
        _signed: Any = None  # E2.5-A：Runtime-owned 双槽的 Slot P 来源
        if experience_episodes:
            try:
                from .experience_memory import (
                    render_experience_pack,
                    resolve_experience_contrast_pack,
                    task_consumer_key,
                )
                # 真实 TaskSpec 规范 key（收束裁决）：task_type|model_class|metric。
                # T4 (#40) A1：内联 f-string 收进 experience_memory.task_consumer_key，
                # 写入侧与检索侧从此共用同一处铸造（此前写入侧另有方言）。
                _task_key = task_consumer_key(request.task_spec)
                # allowed operators 方法内部自动取得（收束裁决）：TaskSpec 禁止面 +
                # 任务允许面 + 非 changes_target_space——不依赖调用方传入
                if allowed_operators is None and request.task_spec is not None:
                    from SelfEvolvingHarnessTS.operators.registry import (
                        OPERATOR_METADATA,
                        OPERATOR_NAMES,
                    )
                    allowed_operators = sorted(
                        name for name in OPERATOR_NAMES
                        if request.task_spec.task_type
                        in (OPERATOR_METADATA[name].get("allowed_tasks") or [])
                        and not request.task_spec.is_op_forbidden(name)
                        and not OPERATOR_METADATA[name].get("changes_target_space")
                    )
                # signed 半径接入（2026-08-08 审查批准）：Episode 带 window_context
                # 特征族（recent./change. 键）且调用方提供 calendar_period → 走 signed
                # 路径（部署可见 recent/change 半径 + signed 判定渲染）；否则回退原
                # ContrastPack 最近案例检索（向后兼容，现有调用方行为不变）。
                _has_signed_context = any(
                    isinstance(getattr(ep, "context_summary", None), Mapping)
                    and any(
                        str(k).startswith(("recent.", "change."))
                        for k in (ep.context_summary.get("local_pattern") or {})
                    )
                    for ep in experience_episodes
                )
                _observed = dict(getattr(request, "observed_pattern_spec", {}) or {})
                # calendar period 从同一公开请求上下文取得（裁决 ①-3）：
                # observed_pattern_spec.bound_period 优先，其次显式参数。
                _period = _observed.get("bound_period")
                if _period is None:
                    _period = calendar_period
                if _has_signed_context and _period is not None:
                    from .signed_radius import (
                        MATERIAL_THRESHOLD,
                        render_signed_instruction,
                        resolve_order,
                        window_context,
                    )
                    # query Context 口径（审查裁决 3' 方案）：
                    # 1) request.observed_pattern_spec 若含 recent/change 特征键
                    #    （上游从部署可见 cohort 历史计算）→ 优先使用（cohort 口径，
                    #    与记忆特征族一致）；仅取 recent./change. 键（bound_period
                    #    等辅助键不进入距离特征）；
                    # 2) 否则回退单序列 window_context（request.values 截断）。
                    if any(str(k).startswith(("recent.", "change.")) for k in _observed):
                        _query_context = {
                            k: float(v) for k, v in _observed.items()
                            if str(k).startswith(("recent.", "change."))
                            and isinstance(v, (int, float))
                        }
                    else:
                        _query_context = window_context(
                            {str(request.series_uid): request.values},
                            int(np.asarray(request.values).size),
                            int(_period),
                        )
                    _order, _signed = resolve_order(
                        query_context=_query_context,
                        episodes=experience_episodes,
                        operators=tuple(allowed_operators or ()),
                        material_threshold=MATERIAL_THRESHOLD,
                        # 审查修正（2026-08-08）：合法性过滤——Task/Consumer 不匹配、
                        # 仪器无效、非法 Operator 的 Episode 不得进入历史池/尺度/δ
                        task_consumer_key=_task_key,
                        allowed_operators=allowed_operators or (),
                    )
                    _rendered = render_signed_instruction(
                        _signed, _order, executable_ops=actionable_ops)
                    if _rendered:
                        view = dataclasses.replace(
                            view, instruction=_rendered + view.instruction
                        )
                    # E0：Memory 注入分辨率公开
                    _memory_status = ("rendered" if _rendered
                                      else "rendered_empty")
                else:
                    _rendered = None
                    _pack = resolve_experience_contrast_pack(
                        experience_episodes,
                        features,
                        _task_key,
                        allowed_operators=allowed_operators or (),
                    )
                    if _pack is not None:
                        _rendered = render_experience_pack(_pack.to_dict())
                        if _rendered:
                            view = dataclasses.replace(
                                view, instruction=_rendered + view.instruction
                            )
                    # E0（用户裁决 2026-08-12）：Memory 注入分辨率公开——
                    # A5 不得静默退化成 A3（注入失败必须可观测）。
                    _memory_status = (
                        "contrast_pack" if _rendered
                        else "contrast_pack_empty")
            except Exception:
                _memory_status = "injection_failed"
                pass  # 经验接线失败不阻塞 fast path（Support 兜底仍生效）
        else:
            _memory_status = "no_memory"
        stages: list[AgentStageResult] = []
        inspected_regions: tuple[tuple[int, int], ...] = ()
        pool: CandidatePool | None = None
        chosen_id = ""
        verification_actions: tuple[str, ...] = ()
        supplied_noop_candidate_ids: tuple[str, ...] = ()
        candidate_artifacts: dict[str, CandidateExecutionArtifact] = {}
        rejection_receipts: tuple[Mapping[str, object], ...] = ()
        chosen_artifact: CandidateExecutionArtifact | None = None
        compilation_status = "not_started"
        task_context_sha = request.task_context.sha() if request.task_context else ""
        run_context_sha = (
            request.run_dependency_binding.sha()
            if request.run_dependency_binding
            else ""
        )
        try:
            inspect = self.core.run_stage(
                role=AgentRole.FAST,
                stage="inspect",
                case_id=request.series_uid,
                public_input={
                    **_task_binding(request, legacy_inspect_stage=True),
                    "features": _plain(features),
                    "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                },
                harness_view=view,
                output_schema_name="fast_inspect_v1",
                output_schema=self.core.load_stage_schema("fast_inspect_v1"),
                source_snapshot_sha=snapshot.runtime_bundle_sha,
                task_context_sha=task_context_sha,
                run_context_sha=run_context_sha,
                validation_retries=1,
                post_validator=lambda payload: _validate_inspect_hypotheses(
                    payload, features
                ),
            )
            stages.append(inspect)
            inspected_regions = _regions_from_fractions(
                inspect.payload["inspected_region_fractions"], request.values.size
            )
            allowed = _allowed_operators(request)
            # 前提过滤（实验 1，2026-08-09）：无缺失 Context 时缺失处理族是
            # 确定性 no-op——从 propose contracts（Agent 可见动作空间）剔除，
            # 探测预算不浪费在 no-op 提案上（supply 层过滤作防御，见下）。
            noop_ops = _noop_ops_for_context(request)
            propose_ops = [name for name in supply_ops
                           if name not in noop_ops]
            # CONTEXT_BOUND_PROGRAM_SUPPLY（2026-08-10，统一修复配套）：
            # 绑定参数完整的算子前置（参数完整性是 context-relevant 相关性
            # 来源——一般规则，不硬编码算子名）；绑定不完整/无绑定保持原序。
            # 注：prepare 内存在条件性局部 import 遮蔽模块级 OPERATOR_METADATA
            # （空 memory 时不执行）——用独立别名避免 UnboundLocalError。
            from SelfEvolvingHarnessTS.operators.registry import (
                OPERATOR_METADATA as _OP_META)
            bound_ok = [n for n in propose_ops
                        if (_OP_META[n].get("public_parameter_bindings")
                            and all(f in features for f in _OP_META[n]
                                    ["public_parameter_bindings"].values()))]
            propose_ops = bound_ok + [n for n in propose_ops
                                      if n not in bound_ok]
            propose_contracts = [
                public_operator_contract(name) for name in propose_ops
            ]
            # W-1: a card that was granted the supply rung promises a
            # candidate to verify, and that promise cannot depend on how the
            # agent's own propose stage went.  PS-2 measured the coupling: in
            # 4 of 12 runs propose raised, the round fell to the outer
            # handler, and the frozen program the agent never had to author
            # died with it (trace pool degraded to identity via _trace's
            # pool-is-None fallback).  bootstrap 4b tells the agent the
            # runtime injects the Source prior; withdrawing it on an agent
            # protocol failure breaks that promise exactly when the agent
            # needed it most.
            supply_candidates = _supply_rung_candidates(view, features)
            try:
                propose = self.core.run_stage(
                    role=AgentRole.FAST,
                    stage="propose",
                    case_id=request.series_uid,
                    public_input={
                        **_task_binding(request),
                        "features": _plain(features),
                        "inspection": _plain(inspect.payload),
                        "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                        "allowed_operator_contracts": propose_contracts,
                    },
                    harness_view=view,
                    output_schema_name="fast_propose_v1",
                    output_schema=self.core.load_stage_schema("fast_propose_v1"),
                    source_snapshot_sha=snapshot.runtime_bundle_sha,
                    task_context_sha=task_context_sha,
                    run_context_sha=run_context_sha,
                    validation_retries=1,
                    post_validator=lambda payload: (
                        _validate_public_parameter_bindings(
                            payload, features, fixed_probe_panel
                        ),
                        _validate_hypothesis_references(payload, inspect.payload),
                    ),
                )
                stages.append(propose)
                supplied, hypothesis_map = _compile_candidates(
                    propose.payload, request)
            except (AgentProtocolError, ProtocolChoiceError, ValueError,
                    TypeError):
                # Same exception set the outer handler already catches, so
                # nothing newly survives a failure -- it is only redirected,
                # and only while there is something to supply.  With no
                # supply rung in view the round fails exactly as before.  The
                # failure stays visible: no propose entry reaches ``stages``,
                # so a round with a program pool and one stage is a round the
                # agent contributed nothing to.
                if not supply_candidates:
                    raise
                supplied, hypothesis_map = (), {}
            # 前提过滤（审核 2026-08-09 实验 1：Program Supply 前提修复）：
            # 依据部署可见 Context 与 Operator 前提，跳过确定性无行为的候选
            # （不读取 gain）——当前 Context 无缺失信号时，仅处理缺失数据的
            # 算子族（impute_*、period_complete、period_median_complete）是
            # 确定性 no-op，供应前过滤（不进池、不验证、不消耗反馈预算）。
            # 不改变 Memory/radius/Agent/反馈/预算。
            noop_ops = _noop_ops_for_context(request)
            if noop_ops:
                supplied = tuple(
                    c for c in supplied
                    if not any(op in noop_ops
                               for op, _ in c.program.execution_steps()))
            # P3 方法层（外部审核裁决 2026-08-09）：retrieved capability Skill
            # 的冻结 Typed steps 与 Agent proposals 合并进入 CandidatePool。
            #
            # **NORMAL_ENTRY_SIGNED_FEEDBACK_TO_SKILL_CONTROL**（外部审核
            # 第三轮）：Skill 优先级由当前 signed 判定（instruction 的
            # Reference 渲染）控制——
            #   - 无 CONFLICT/RISK（含 POSITIVE）：Skill 优先保留 slot
            #     （identity + 1 Skill + ≤1 Agent，Skill 在前——Agent 被
            #     预算截断时 Skill 仍保留；修正 total_k=2 边界）；
            #   - CONFLICT/RISK（Reference 2/3 含 Skill 的算子）：Skill 降级
            #     ——Agent 候选在前、Skill 排最后（预算截断 = 不硬删除）。
            # 反馈控制通过池内顺序表达；选择由 Agent/selector 按公开顺序。
            skill_candidates = _skill_frozen_candidates(view, features)
            if skill_candidates:
                pos_ops, risk_ops = _signed_reference_ops(view.instruction)
                # P1（用户裁决 2026-08-12）：Skill 执行权限——
                # risk_guards.requires_target_support=true 的 DRAFT Skill
                # 可进候选池但不得保留优先 slot、不得因历史 Reference 1
                # 自动获得当前执行权。合并顺序：ACTIVE → Agent → DRAFT
                # → signed-risk degraded（预算截断时 degraded 最后被截）。
                guards_by_id = {
                    skill.skill_id: dict(skill.risk_guards or {})
                    for skill in view.skills}

                def _skill_id(c: object) -> str | None:
                    src = str(getattr(c, "source", "") or "")
                    return src[len("skill:"):] if src.startswith("skill:") else None

                def _is_draft(c: object) -> bool:
                    g = guards_by_id.get(_skill_id(c)) or {}
                    return bool(g.get("requires_target_support") is True)

                def _degraded(c: object) -> bool:
                    return any(op in risk_ops
                               for op, _ in c.program.execution_steps())
                active = [c for c in skill_candidates
                          if not _is_draft(c) and not _degraded(c)]
                draft = [c for c in skill_candidates
                         if _is_draft(c) and not _degraded(c)]
                degraded = [c for c in skill_candidates if _degraded(c)]
                if degraded and not active and not draft:
                    # 全部降级：Agent 候选优先、Skill 排最后（预算截断语义）
                    supplied = (*supplied, *degraded)
                elif active:
                    # ACTIVE 在前保留 slot（原 NORMAL_ENTRY 语义）
                    supplied = (*active[:1], *supplied[:1],
                                *draft[:1], *degraded[:1])
                else:
                    # 无 ACTIVE：Agent 候选在前——DRAFT 不挤 Agent
                    supplied = (*supplied[:1], *draft[:1], *degraded[:1])
            # E2.5-A（用户裁决 2026-08-12）：Runtime-owned 双槽——
            # Slot P：从 signed Source Experience 的结构化结果（resolve_
            # order per_op verdict——不解析 Reference 文本）生成**最多一个**
            # 合法先验 Workflow；Slot E：真实 LLM 的当前 Context 探索候选。
            # Runtime 只保证 Source Memory 不能删除探索机会——LLM 提案
            # 无任何非 P 算子候选 → 协议失败（EXPLORATION_SLOT_EMPTY）。
            # 真实 LLM 负责最终排序（select 阶段）。
            if runtime_prior_slot and _signed is not None:
                prior_op = next(
                    (op for op, st in (_signed.get("per_op") or {}).items()
                     if (st or {}).get("verdict") == "POSITIVE_PRIOR"),
                    None)
                if prior_op is not None:
                    # CONTEXT_BOUND_REBINDING（用户裁决 2026-08-12）：Source
                    # prior 与 Skill 同一绑定逻辑——有 registry 绑定声明的
                    # 算子用当前 features 绑定；绑定特征缺失 → prior 不生成
                    # （安全——不注入失效先验）。
                    _pbind = (OPERATOR_METADATA.get(prior_op) or {}) \
                        .get("public_parameter_bindings") or {}
                    if _pbind:
                        _pparams = {
                            name: features[f]
                            for name, f in _pbind.items() if f in features}
                        if len(_pparams) != len(_pbind):
                            prior_op = None
                        else:
                            _prior_params: Mapping[str, object] = _pparams
                    else:
                        _prior_params = _default_params_from_contract(prior_op)
                if prior_op is not None:
                    prior_cand = Candidate.program_candidate(
                        f"cand_prior_{prior_op}",
                        Program.from_steps(
                            [(prior_op, dict(_prior_params))],
                            source="runtime_prior"),
                        source="runtime_prior")
                    # LLM 提案中同算子候选去重（P 保留一个 slot）
                    supplied = (
                        prior_cand,
                        *[c for c in supplied
                          if not (c.source == "agent"
                                  and _cand_ops(c) == (prior_op,))])
                    # 探索槽检查：agent 提案必须含非 P 算子候选
                    if not any(
                            c.source == "agent"
                            and set(_cand_ops(c)) != {prior_op}
                            for c in supplied):
                        raise AgentProtocolError(
                            "EXPLORATION_SLOT_EMPTY",
                            "LLM propose returned only the Source prior "
                            "operator — no current-context exploration "
                            "candidate (Slot E empty)")
            total_k = int(snapshot.candidate_policy["total_k"])
            if request.task_context is not None:
                total_k = min(
                    total_k,
                    request.task_context.deployment_constraints.maximum_candidates,
                )
            pool = CandidatePool.build(supplied, total_k=total_k)
            maximum_modified_fraction, preserve_outside = _verification_limits(
                request, view
            )
            def _regions_for(candidate: Candidate) -> tuple[tuple[int, int], ...]:
                """CONTEXT_BOUND_REBINDING（用户裁决 2026-08-12）：
                Skill/prior 候选的 verify 使用其**当前 Context 绑定后**的
                候选自身 Scope（region_start/end_fraction——绑定参数）；
                Agent 新提案继续使用 LLM inspect 区域。不允许退化全窗口。"""
                src = str(getattr(candidate, "source", "") or "")
                if src.startswith(("skill:", "runtime_prior")):
                    for _op, params in candidate.program.execution_steps():
                        if ("region_start_fraction" in params
                                and "region_end_fraction" in params):
                            size = int(np.asarray(request.values).size)
                            start = int(float(params["region_start_fraction"])
                                        * size)
                            end = int(float(params["region_end_fraction"])
                                      * size)
                            return ((max(start, 0), min(end, size)),)
                return inspected_regions

            verified = tuple(
                verify_candidate(
                    candidate,
                    request.values,
                    allowed_operators=allowed,
                    inspected_regions=_regions_for(candidate),
                    maximum_modified_fraction=maximum_modified_fraction,
                    preserve_outside_inspected_region=preserve_outside,
                    require_finite_output=request.task_context is not None,
                )
                for candidate in pool.candidates
            )
            candidate_artifacts = {
                artifact.candidate.candidate_id: artifact for artifact in verified
            }
            supplied_noop_candidate_ids = tuple(
                artifact.candidate.candidate_id
                for artifact in verified
                if artifact.candidate.kind is CandidateKind.PROGRAM
                and artifact.receipt.effect_equivalent_to_identity
            )
            rejection_receipts = tuple(
                artifact.receipt.to_dict()
                for artifact in verified
                if not artifact.selectable
            )
            pool = CandidatePool(
                tuple(
                    artifact.candidate for artifact in verified if artifact.selectable
                ),
                total_k,
            )
            public_candidates = []
            for candidate in pool.candidates:
                candidate_payload = {
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind.value,
                    "program_sha": candidate.program.sha() if candidate.program else None,
                    "steps": (
                        [
                            {"op": op, "params": params}
                            for op, params in candidate.program.execution_steps()
                        ]
                        if candidate.program
                        else []
                    ),
                }
                # STRUCTURED_SKILL 引用链（2026-08-14）：假设引用传递到
                # select——LLM 选择的理由可沿 hypothesis 链观察。
                hypothesis_id = hypothesis_map.get(candidate.candidate_id)
                if hypothesis_id:
                    candidate_payload["addresses_hypothesis_id"] = (
                        hypothesis_id
                    )
                if request.task_context is not None:
                    artifact = candidate_artifacts[candidate.candidate_id]
                    candidate_payload["verification_receipt"] = (
                        artifact.receipt.to_dict()
                    )
                    candidate_payload["verification_receipt_sha"] = (
                        artifact.receipt.receipt_sha
                    )
                public_candidates.append(candidate_payload)
            if not any(c.kind is CandidateKind.PROGRAM for c in pool.candidates):
                # 审查修复（2026-08-08）：无 PROGRAM 候选 = ABSTAIN 语义——
                # compilation_status = not_applicable（安全 abstention 不是编译故障）；
                # 跳过 select（省一次调用、避免 select 引用不存在候选导致
                # require_choice 抛错被误记为 failed）。
                compilation_status = "not_applicable"
                chosen_id = "identity"
                verification_actions = ("public_evidence_insufficient",)
                chosen = Candidate.identity()
                chosen_artifact = None
            else:
                compilation_status = "ok"
                select = self.core.run_stage(
                    role=AgentRole.FAST,
                    stage="select",
                    case_id=request.series_uid,
                    public_input={
                        **_task_binding(request),
                        "features": _plain(features),
                        "inspection": _plain(inspect.payload),
                        "fixed_probe_panel": _plain(fixed_probe_panel or {}),
                        "candidates": public_candidates,
                    },
                    harness_view=view,
                    output_schema_name="fast_select_v1",
                    output_schema=self.core.load_stage_schema("fast_select_v1"),
                    source_snapshot_sha=snapshot.runtime_bundle_sha,
                    task_context_sha=task_context_sha,
                    run_context_sha=run_context_sha,
                    validation_retries=1,
                )
                stages.append(select)
                chosen_id = select.payload["chosen_candidate_id"]
                verification_actions = tuple(select.payload["verification_actions"])
                chosen = pool.require_choice(chosen_id)
                chosen_artifact = candidate_artifacts[chosen.candidate_id]
        except (AgentProtocolError, ProtocolChoiceError, ValueError, TypeError) as exc:
            trace = self._trace(
                request=request,
                view=view,
                stages=stages,
                inspected_regions=inspected_regions,
                pool=pool,
                chosen_candidate_id=chosen_id,
                compilation_status="failed" if compilation_status != "ok" else compilation_status,
                execution_status="not_started",
    memory_resolution_status=_memory_status,
                modified_indices=(),
                verification_actions=verification_actions,
                identity_equivalent=False,
                supplied_noop_candidate_ids=supplied_noop_candidate_ids,
                candidate_artifacts=candidate_artifacts,
                rejection_receipts=rejection_receipts,
            )
            return (
                PreparationResult(
                    PreparationStatus.FAILED,
                    None,
                    None,
                    ExecutionReceipt(ok=False, error=f"AgentProtocolError: {exc}"),
                ),
                trace,
            )
        if chosen.kind is CandidateKind.IDENTITY:
            prepared_values = request.values.copy()
            prepared = PreparedSeries(
                request.series_uid, prepared_values, (), "original_units"
            )
            trace = self._trace(
                request=request,
                view=view,
                stages=stages,
                inspected_regions=inspected_regions,
                pool=pool,
                chosen_candidate_id=chosen_id,
                compilation_status=compilation_status,
                execution_status="ok",
    memory_resolution_status=_memory_status,
                modified_indices=(),
                verification_actions=verification_actions,
                identity_equivalent=True,
                supplied_noop_candidate_ids=supplied_noop_candidate_ids,
                candidate_artifacts=candidate_artifacts,
                rejection_receipts=rejection_receipts,
            )
            return (
                PreparationResult(
                    PreparationStatus.ABSTAINED,
                    prepared,
                    None,
                    ExecutionReceipt(ok=True),
                ),
                trace,
            )
        assert chosen.program is not None
        assert chosen_artifact is not None and chosen_artifact.prepared_values is not None
        receipt = ExecutionReceipt(
            ok=True,
            trace=tuple(dict(row) for row in chosen_artifact.execution_trace),
        )
        modified = chosen_artifact.modified_indices
        equivalent = chosen_artifact.receipt.effect_equivalent_to_identity
        prepared = PreparedSeries(
            request.series_uid,
            chosen_artifact.prepared_values,
            tuple(step.op for step in chosen.program.steps),
            "original_units",
        )
        trace = self._trace(
            request=request,
            view=view,
            stages=stages,
            inspected_regions=inspected_regions,
            pool=pool,
            chosen_candidate_id=chosen_id,
            compilation_status=compilation_status,
            execution_status="ok",
    memory_resolution_status=_memory_status,
            modified_indices=modified,
            verification_actions=verification_actions,
            identity_equivalent=equivalent,
            supplied_noop_candidate_ids=supplied_noop_candidate_ids,
            candidate_artifacts=candidate_artifacts,
            rejection_receipts=rejection_receipts,
        )
        return (
            PreparationResult(
                PreparationStatus.PREPARED, prepared, chosen.program, receipt
            ),
            trace,
        )


__all__ = [
    "TTHAFastAgent",
    "public_operator_contract",
    "public_operator_contracts_for_task",
]
