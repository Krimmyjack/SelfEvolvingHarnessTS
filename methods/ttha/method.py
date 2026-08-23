from __future__ import annotations

import re

from collections.abc import Callable, Mapping, Sequence

from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    HarnessSnapshot,
)
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest, PreparationResult
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

from .agent_core import AgentProtocolError
from .fast_agent import TTHAFastAgent
from .retrieval import evaluate_applicability
from .signed_radius import MATERIAL_THRESHOLD
from .experience_memory import classify_relation as _classify_relation


def _typed_patch_preflight(card: Mapping[str, object],
                           manifest: Any) -> None:
    """Typed Patch 契约强制（用户方案 C，2026-08-11）：card 提供
    typed_patch_options 时，manifest 必须提供 patch_id 且属于白名单；
    缺失或未知 → StagePostValidationError(retryable=True)（触发现有
    schema retry；第二次仍失败 → ACTION_UNAVAILABLE）。不改 Prompt/
    Schema。"""
    options = card.get("typed_patch_options") or []
    if not options:
        return
    from .agent_core import StagePostValidationError
    valid = [str(o.get("patch_id"))
             for o in options
             if isinstance(o, Mapping) and o.get("patch_id")]
    pid = getattr(manifest, "patch_id", None)
    if not pid:
        raise StagePostValidationError(
            "TYPED_PATCH_ID_MISSING",
            "patch_id is required when typed_patch_options are present. "
            f"Set edit_manifest.patch_id to exactly one of: {', '.join(valid)}.",
            retryable=True)
    if not any(str(o.get("patch_id")) == pid
               for o in options if isinstance(o, Mapping)):
        raise StagePostValidationError(
            "TYPED_PATCH_ID_UNKNOWN",
            f"patch_id '{pid}' is not in the typed_patch_options whitelist. "
            f"Valid values: {', '.join(valid)}.",
            retryable=True)


_GROUP_ADD_SURFACE_ID = "skill_library.entries/{skill_id}"


def _group_add_route_error(
    route_decision: Any,
    surface_catalog: Sequence[Mapping[str, object]],
) -> str | None:
    """E-2（rev4）：组路径只接受一条已经由 ProgramSupply 路由挣得的
    SKILL_LIBRARY_GAP / EDITABLE_M0 / capability-ADD 单 Surface 授权。

    返回 None = 可以构造 manifest；返回字符串 = 拒绝原因（调用方记
    ``route_not_add_only``，不得构造 manifest、不得调 Slow）。"""
    if route_decision is None:
        return "route_decision_missing"
    if getattr(route_decision, "cause_code", None) != "SKILL_LIBRARY_GAP":
        return f"cause_not_skill_library_gap:{getattr(route_decision, 'cause_code', None)}"
    if getattr(route_decision, "actionability", None) != "EDITABLE_M0":
        return f"actionability_not_editable:{getattr(route_decision, 'actionability', None)}"
    templates = tuple(getattr(route_decision, "surface_templates", ()) or ())
    if templates != (_GROUP_ADD_SURFACE_ID,):
        return f"surface_templates_not_add_capability:{templates!r}"
    entries = [dict(item) for item in surface_catalog
               if isinstance(item, Mapping)]
    if len(entries) != 1:
        return f"catalog_not_single_surface:{len(entries)}"
    entry = entries[0]
    if entry.get("operation") != "ADD":
        return f"catalog_operation_not_add:{entry.get('operation')!r}"
    if entry.get("target_class") != "capability":
        return f"catalog_target_not_capability:{entry.get('target_class')!r}"
    if entry.get("surface_id") != _GROUP_ADD_SURFACE_ID:
        return f"catalog_surface_not_add_template:{entry.get('surface_id')!r}"
    if entry.get("surface_precondition") != {"kind": "ABSENT"}:
        return f"catalog_precondition_not_absent:{entry.get('surface_precondition')!r}"
    return None


def _applicability_from_card(card: Mapping[str, object]) -> dict[str, object]:
    """P4.2-A（用户裁决 2026-08-11）：Applicability-to-Observation
    Binding——Skill 的 observable_applicability 由 Runtime 从 Failure
    Card 的 observable_signature（公开 Observation）机器生成；Slow Agent
    不得额外编造特征（P4.2 案例：LLM 编造 clipping_probe_direction==
    negative——评估装置从不填充该特征 → 下一轮检索门阻断，批准≠采用）。
    B2 同款 Runtime 所有权；signature 为空 → {"const": True}（无公开
    Observation 则不设 Scope 门）。"""
    sig = card.get("observable_signature") or {}
    leaves: list[dict[str, object]] = []
    if isinstance(sig, Mapping):
        for key, value in sig.items():
            if isinstance(value, (str, int, float, bool)):
                leaves.append({"feature": str(key), "op": "==", "value": value})
    if leaves:
        return {"all": leaves}
    return {"const": True}


def _applicability_is_wide(applicability: Mapping[str, object]) -> bool:
    """宽 Scope 判定（P1，用户裁决 2026-08-12）：const:true 或全部特征
    ∈ {task_kind}——不能区分效用翻转（P4.5：已暴露数据上无任何 Scope
    信号）→ 新 Skill 必须写入 requires_target_support=true（DRAFT——
    不自动优先，须当前 Target Support 确认）。"""
    leaves = _applicability_leaves(applicability)
    return not leaves or all(feature == "task_kind" for feature in leaves)


def _applicability_leaves(ast: object) -> list[str]:
    """收集 applicability AST 中的 feature 键（all/any/not 递归）。"""
    if not isinstance(ast, Mapping):
        return []
    if "feature" in ast and isinstance(ast.get("feature"), str):
        return [str(ast["feature"])]
    out: list[str] = []
    for key in ("all", "any", "not"):
        child = ast.get(key)
        if isinstance(child, Sequence) and not isinstance(child, (str, bytes)):
            for item in child:
                out.extend(_applicability_leaves(item))
        elif isinstance(child, Mapping):
            out.extend(_applicability_leaves(child))
    return out


def _guidance_body(snapshot: Any) -> str:
    """读取当前 snapshot 中 Workflow Construction Skill 的 body 文本。"""
    for skill in snapshot.skills:
        if skill.skill_id == "build_contrastive_candidates":
            return str(skill.body)
    return ""


def _parse_clause_payload(value: str) -> dict[str, str] | None:
    """P3 clause 级编辑载荷解析（用户裁决 2026-08-14）：minimal_patch.value
    必须是 REPLACE_CLAUSE 结构——Slow 只改一条带 ID 的 propose 规则；
    Runtime 绑定到正确阶段与规则位置。格式：

      REPLACE_CLAUSE
      target: propose.rule.<clause_id>
      new_clause: <新规则文本（单行、无数据集/系列/算子名/数值）>
      （可选 predicted_change: ... / falsification: ...——记录用）

    解析失败返回 None（调用方记 clause_payload_invalid）。"""
    if not isinstance(value, str) or not value.strip():
        return None
    lines = [line.rstrip("\r") for line in value.split("\n")]
    if not lines or lines[0].strip() != "REPLACE_CLAUSE":
        return None
    payload: dict[str, str] = {}
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        for key in ("target", "new_clause", "predicted_change", "falsification"):
            if line.startswith(key + ":"):
                payload[key] = line[len(key) + 1:].strip()
                break
        else:
            return None  # 未知行——拒绝
    target = payload.get("target") or ""
    if not target.startswith("propose.rule.") or len(target) <= len("propose.rule."):
        return None
    if not payload.get("new_clause"):
        return None
    if "\n" in payload["new_clause"]:
        return None  # 单行约束
    return payload


def _apply_clause_replacement(current_body: str,
                              payload: Mapping[str, str]) -> str:
    """把 REPLACE_CLAUSE 绑定到正确规则位置（P3 Runtime-owned binding）：
    只替换 target 指向的 propose.rule.<id> 行，其余内容逐字不变。
    unknown clause → ValueError。"""
    target = str(payload["target"])
    marker = target + ": "
    start = current_body.find(marker)
    if start < 0:
        raise ValueError(f"unknown clause target: {target}")
    content_start = start + len(marker)
    next_clause = current_body.find("\npropose.rule.", content_start)
    if next_clause < 0:
        next_clause = current_body.find("\n[select_guidance]", content_start)
    if next_clause < 0:
        raise ValueError("clause boundary not found")
    new_clause = str(payload["new_clause"])
    return current_body[:content_start] + new_clause + current_body[next_clause:]


def _op_of_episode(episode: object) -> str:
    """Episode 的算子名（Fast winner skill 命名用）。"""
    sig = str(getattr(episode, "workflow_signature", "") or "")
    if sig:
        return sig
    steps = ((getattr(episode, "context_summary", {}) or {})
             .get("program_geometry", {}) or {}).get("program_steps") or []
    if steps:
        return str(steps[0].get("op", "op"))
    return "op"


def _per_series_map(per_view_gain: Any,
                    series_uids: Any = None) -> dict[str, float] | None:
    """位置序列 → uid 映射（online_loop._per_series_gains 的同一形状转换）。
    读不到 per-view 读数就返回 None——"没读到"不得当成"读到 0 条有害"。"""
    if per_view_gain is None:
        return None
    values = [float(v) for v in per_view_gain]
    uids = [str(u) for u in (series_uids or ())]
    if len(uids) != len(values):
        uids = ["view_%d" % i for i in range(len(values))]
    return dict(zip(uids, values))


def _steps_are_identity(steps: Any) -> bool:
    ops = [str(op) for op, _p in (steps or ())]
    return not ops or ops == ["identity"]


def _task_scope_of_episode(episode: object) -> tuple[str, str, str]:
    """T5 #41 A5：Skill 命名用的任务范围三元组。

    取自 Episode 的任务硬键（task_type|downstream_model_class|metric.name）
    ——与 Memory 检索键同源，不另铸第四种方言。缺键时回落到历史默认，
    使旧 fixture 的命名保持可预测。"""
    key = str(getattr(episode, "task_consumer_key", "") or "")
    parts = key.split("|")
    if len(parts) == 3 and all(parts):
        return (parts[0], parts[1], parts[2])
    return ("forecast", "ridge", "sMASE")


def _series_uids_of_episode(episode: object) -> tuple[str, ...]:
    """Episode 里记下的逐 view 读数长度对应的 uid 序（无则空 tuple——
    _per_series_map 会自行退回 view_i 位置名，不伪造 uid）。"""
    summary = getattr(episode, "context_summary", {}) or {}
    uids = summary.get("series_uids")
    if isinstance(uids, (list, tuple)):
        return tuple(str(u) for u in uids)
    return ()


def _fast_winner_skill_id(episode: object) -> str:
    """无哈希、任务化的 Fast winner Skill ID。

    原名 fast_winner_{op} 只带算子：预测轮次学到的 fast_winner_winsorize
    与异常检测轮次学到的同名条目会在同一个 skill_library 里**撞名**——
    第二次 ADD 撞上 surface_precondition={"kind": "ABSENT"} 直接硬失败，
    真实双任务轨迹因此在第二轮就停住。ID 里带上任务范围即可分开。
    本轮只声称 task_kind 隔离（见 handle_fast_winner 的 applicability
    注记）；同任务跨 Consumer 的隔离不在本轮的声称范围内。"""
    task_type, model_class, metric = _task_scope_of_episode(episode)
    # EditManifest.edit_id 走 contracts.harness 的 canonical-id 语法
    # (^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$)：任务键里的 metric 名带大小写
    # （sMASE），直接拼进去会被 manifest 硬拒。逐段折成小写并把非法字符
    # 折成下划线——折叠是确定性的、无哈希的，且三处（edit_id /
    # target_surface_id / skill_id）共用这一个函数，不会再各写一遍。
    parts = [_canonical_fragment(x)
             for x in (task_type, model_class, metric,
                       _op_of_episode(episode))]
    return "fast_winner_" + "_".join(parts)


def _canonical_fragment(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    return text or "x"


def _applicability_reachable(
    card: Mapping[str, object],
    applicability: Mapping[str, object],
    fast_features: Mapping[str, object],
) -> tuple[bool, str]:
    """P4.2-B（用户裁决 2026-08-11）：批准前机械可达性检查——批准 Skill
    前必须满足：①applicability 每个特征来自 card 的公开 Observation
    （observable_signature）；②当前正常 Fast 入口的特征空间能产生该特征；
    ③新 Skill 在当前公开 Context 下至少能被检索（evaluate_applicability
    通过）。不满足 → 调用方记 ACTION_UNAVAILABLE（不写 active snapshot）。"""
    if not isinstance(fast_features, Mapping):
        return False, "fast_features_unavailable"
    sig = card.get("observable_signature")
    sig_keys = set(sig) if isinstance(sig, Mapping) else set()
    leaves = _applicability_leaves(applicability)
    missing = [key for key in leaves if key not in sig_keys]
    if missing:
        return False, f"feature_not_in_card_signature:{','.join(sorted(missing))}"
    absent = [key for key in leaves if key not in fast_features]
    if absent:
        return False, f"feature_not_in_fast_space:{','.join(sorted(absent))}"
    # 审查附注（2026-08-11）：applicability AST 非法（如 card 产出非
    # OBSERVABLE 键）时 evaluate_applicability 的 validate 会抛
    # ValueError——须干净拒绝（ACTION_UNAVAILABLE），不得让异常逃逸
    # 批准流程。
    try:
        matched, _ = evaluate_applicability(applicability, fast_features)
    except ValueError:
        return False, "invalid_applicability"
    if not matched:
        return False, "not_retrievable_in_current_context"
    return True, "ok"


class TTHAMethod:
    """正常方法入口（审查 2026-08-08 接线：显式持有 Episode 集合）。

    裁决约束：
      - 不隐式读取全局 episodes.json（避免污染 A3/H0 空 Memory）——Experience
        Episode 由调用方显式构造并注入；
      - 部署可见 cohort recent/change Context 由上游经 request.observed_pattern_spec
        提供（含 bound_period 键）；calendar period 从同一公开请求上下文取得；
      - A3（空集合）走同一入口且不注入。
    """

    method_id = "ttha_m0"

    def __init__(
        self,
        fast_agent: TTHAFastAgent,
        snapshot: HarnessSnapshot | Callable[[], HarnessSnapshot],
        experience_episodes: Sequence[object] = (),
    ) -> None:
        self.fast_agent = fast_agent
        self._snapshot = snapshot
        self._experience_episodes = tuple(experience_episodes)
        self.last_trace: DecisionTrace | None = None
        # P3.2 时间边界（用户裁决 2026-08-11）：method-local pending——
        # Support 后可生成/replay 候选但**不激活**；delayed 时间点到达后
        # 批准才更新 snapshot。普通属性，不建 Schema/Ledger/状态平台。
        self._pending_update: dict[str, Any] | None = None

    def _active_snapshot(self) -> HarnessSnapshot:
        return self._snapshot() if callable(self._snapshot) else self._snapshot

    def prepare(self, request: PreparationRequest,
                runtime_prior_slot: bool = False,
                pool_mode: str = "actionable") -> PreparationResult:
        result, trace = self.fast_agent.prepare(
            request,
            self._active_snapshot(),
            experience_episodes=self._experience_episodes,
            runtime_prior_slot=runtime_prior_slot,
            pool_mode=pool_mode,
        )
        self.last_trace = trace
        return result

    def bind_round_data(self, values: object, *, task_kind: str = "forecast") -> None:
        """同一实例的轮次数据切换（反馈生命周期 R1→R2 需要新决策点数据；
        verify_context 是 context_sha 全等比较，gateway 必须按该轮可见数据
        重建）。backend/快照/Memory 全部保留在同一实例。"""
        from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
            LocalPublicToolGateway,
        )

        core = self.fast_agent.core
        core.tools = LocalPublicToolGateway(values, task_kind=task_kind)

    @property
    def experience_episodes(self) -> tuple[object, ...]:
        """当前实例持有的 Episode（只读视图）。

        T4 (#40)：写入必须落 Runtime 才算写入，因此校验也必须从 Runtime 读回，
        而不是读 runner 手里那份构造列表。这里只暴露读，不暴露改——追加仍然
        只能走 append_experience_episode。"""
        return self._experience_episodes

    def append_experience_episode(self, episode: object) -> None:
        """反馈写回（裁决 2026-08-09 二十八）：R1 实测后立即追加当前臂
        Episode（最小接口；不建设 Memory Store/Schema/生命周期平台）。"""
        self._experience_episodes = (*self._experience_episodes, episode)

    def update_experience_episode(self, episode: object) -> None:
        """delayed 更新同一 Episode：按 episode_id 原位替换（不存在则追加）。
        用于"之后打开真实 delayed，更新同一 Episode"（裁决 2026-08-09
        二十八）。"""
        eps = list(self._experience_episodes)
        for i, e in enumerate(eps):
            if getattr(e, "episode_id", None) == getattr(episode,
                                                          "episode_id", None):
                eps[i] = episode
                break
        else:
            eps.append(episode)
        self._experience_episodes = tuple(eps)

    # ---- P3.1-A2（用户裁决 2026-08-11）：feedback 生命周期所有权移入方法层 ----
    # Runner 只提交 Episode/feedback（append/update + handle_feedback）；
    # relation 判定、Slow 触发、Controller、replay/delayed、active snapshot
    # 更新全部在方法层内完成。Runner 可提供 card_builder/evaluator 回调
    # （数据回调），但不得拥有触发与更新决策。

    def handle_feedback(
        self,
        episode: object,
        *,
        slow_agent: Any,
        controller: Any,
        store: Any,
        surface_catalog: Sequence[Mapping[str, object]],
        card_builder: Callable[[object], Mapping[str, object]],
        evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]], int],
                            Any],
        delayed_evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]],
                                    int], Any] | None = None,
        manifest_preflight: Callable[[Any], None] | None = None,
        confirmed_cause: str,
        fast_features: Mapping[str, object] | None = None,
        allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
        task_context: Any | None = None,
    ) -> dict[str, Any]:
        """方法层 feedback 生命周期：material NEGATIVE/CONFLICT 判定 →
        Slow Agent propose_edit → Controller.apply_to_fork → replay/delayed
        → 更新 active snapshot（下一轮 prepare 自动采用）。零 live LLM
        （Replay Backend 由调用方注入）。返回事件记录（Runner 仅读取报告）。

        P4.2-A/B（用户裁决 2026-08-11）：applicability 由 Runtime 从 card
        observable_signature 绑定（A）+ 批准前机械可达性检查（B）——
        fast_features 为当前正常 Fast 入口的公开特征（数据回调，调用方
        提供）；缺失 → 不批准（fail-safe）。
        E0（用户裁决 2026-08-12）：Slow 调用透传合法 Operator contracts
        与现有 TaskContext——不再空传（Slow Agent 必须能消费任务语义）。"""
        ev: dict[str, Any] = {"stage": "no_trigger"}
        sg = (getattr(episode, "support_response", {}) or {}).get("gain")
        relation = getattr(episode, "relation", None)
        material_negative = bool(
            isinstance(sg, (int, float)) and float(sg) < -MATERIAL_THRESHOLD)
        conflict = relation == "CONFLICT"
        if not (material_negative or conflict):
            return ev
        ev["triggered"] = True
        ev["material_negative"] = material_negative
        ev["relation"] = relation
        # ---- Slow Agent propose_edit（方法层调用）----
        card = card_builder(episode)
        ev["card_workflow"] = [s["op"] for s in
                               card.get("workflow", {}).get("steps", [])]
        # 方案 C（2026-08-11）：manifest_preflight=None 时用内置 Typed
        # Patch 契约（缺失/未知 patch_id → retryable → 现有 schema retry）
        if manifest_preflight is None:
            manifest_preflight = lambda m: _typed_patch_preflight(card, m)  # noqa: E731
        try:
            manifest = slow_agent.propose_edit(
                card, surface_catalog, self._active_snapshot(),
                manifest_preflight=manifest_preflight,
                allowed_operator_contracts=allowed_operator_contracts,
                task_context=task_context)
        except (RuntimeError, AgentProtocolError) as exc:  # noqa: BLE001
            # RuntimeError：LLM 调用预算超限（provider/预算故障）；
            # AgentProtocolError：preflight 重试后仍失败（契约强制——
            # 缺失/未知 patch_id）→ 调用方判 ACTION_UNAVAILABLE
            ev["stage"] = ("budget_exceeded" if isinstance(exc, RuntimeError)
                           else "manifest_preflight_failed")
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        if manifest is None:
            ev["stage"] = "abstained_by_agent"
            ev["no_proposal_reason"] = slow_agent.last_no_proposal_reason
            return ev
        ev["stage"] = "manifest_proposed"
        ev["edit_id"] = manifest.edit_id
        ev["target_surface_id"] = str(manifest.target_surface_id)
        ev["operation"] = manifest.operation.value
        # ---- B2 Typed Patch Binding：patch_id → card 白名单取冻结
        # steps；Runtime **机器生成** Skill body（Fast Agent 从 body 的
        # Frozen marker 解析——body 不再由 LLM 手写）。必须在 apply
        # **之前**完成（Controller 写入的是最终 body）。----
        import json as _json  # noqa: PLC0415
        import dataclasses as _dc  # noqa: PLC0415

        from .slow_agent import _steps_for_patch_id  # noqa: PLC0415

        steps = _steps_for_patch_id(card, manifest.patch_id)
        if not steps:
            ev["stage"] = "no_frozen_program"
            ev["patch_id"] = manifest.patch_id
            return ev
        ev["frozen_program"] = [{"op": o, "params": dict(p)} for o, p in steps]
        ev["patch_id"] = manifest.patch_id
        # P0（rev3）：capability Skill .body PATCH 的 minimal_patch.value
        # 由 Runtime 从 typed_patch_options 白名单覆写；Slow 文本一律忽略。
        from .slow_agent import (  # noqa: PLC0415
            bind_frozen_patch_program,
            is_capability_body_surface_id,
        )
        if (manifest.operation is EditOperation.PATCH
                and is_capability_body_surface_id(
                    str(manifest.target_surface_id))):
            manifest = bind_frozen_patch_program(manifest, steps)
            ev["patch_body_binding"] = "runtime_owned"
        nv = None
        if manifest.new_value is not None and "body" in manifest.new_value:
            nv = dict(manifest.new_value)
            nv["body"] = "Frozen program steps: " + _json.dumps(
                [{"op": o, "params": dict(p)} for o, p in steps])
            # B2：Runtime-owned 绑定覆盖 Skill 消费契约——Fast Agent 只
            # 供应 CAPABILITY skill（bootstrap 不入池）；skill_kind 不再
            # 由 LLM 决定
            nv["skill_kind"] = "capability"
            nv["allowed_tools"] = [o for o, _p in steps]
            # P4.2-A（用户裁决 2026-08-11）：Applicability-to-Observation
            # Binding——observable_applicability 由 Runtime 从 card 的
            # observable_signature（公开 Observation）机器生成（B2 同款
            # Runtime 所有权——LLM 不得编造运行时不可满足的 Scope 条件）。
            # manifest 级字段同步（controller 校验两者一致）。
            nv["observable_applicability"] = _applicability_from_card(card)
            # P1（用户裁决 2026-08-12）：宽 Scope（const:true / 仅
            # task_kind——不能区分效用翻转）的新 Skill 必须写入
            # requires_target_support=true——DRAFT 执行权限门（可进候选
            # 池但不自动优先，须当前 Target Support 确认）。
            if _applicability_is_wide(nv["observable_applicability"]):
                rg = dict(nv.get("risk_guards") or {})
                rg["requires_target_support"] = True
                nv["risk_guards"] = rg
            manifest = _dc.replace(
                manifest, new_value=nv,
                observable_applicability=nv["observable_applicability"])
        # P4.2-B（用户裁决）：批准前机械可达性检查——①特征来自 card 公开
        # Observation ②Fast 入口特征空间能产生 ③当前公开 Context 下可检索；
        # 不满足 → ACTION_UNAVAILABLE（不写 active snapshot）
        _final_app = (
            (nv.get("observable_applicability") if nv is not None else None)
            or manifest.observable_applicability or {})
        if _final_app:
            if fast_features is None:
                ev["stage"] = "applicability_uncheckable"
                ev["applicability_reason"] = "fast_features_unavailable"
                return ev
            _ok, _reason = _applicability_reachable(
                card, _final_app, fast_features)
            if not _ok:
                ev["stage"] = "applicability_unreachable"
                ev["applicability_reason"] = _reason
                ev["action"] = "ACTION_UNAVAILABLE"
                return ev
        # ---- Controller.apply_to_fork（Skill 写入正常 Harness；surface
        # 模板实例化 + dependency SHA 补全——P1 同款契约，方法层内完成）----
        from .slow_agent import _resolve_apply_manifest  # noqa: PLC0415
        from .slow_agent import verify_frozen_patch_program  # noqa: PLC0415

        try:
            manifest_applied = _resolve_apply_manifest(
                manifest, self._active_snapshot())
            receipt = controller.apply_to_fork(
                store.materialize(self._active_snapshot()), manifest_applied,
                confirmed_cause=confirmed_cause)
            if (manifest_applied.operation is EditOperation.PATCH
                    and is_capability_body_surface_id(
                        str(manifest_applied.target_surface_id))):
                # P0（rev3）：从 candidate snapshot 读回真实 body，解析后
                # 必须与 replay steps 逐元素相等；不等 -> apply_failed。
                verify_frozen_patch_program(
                    receipt.candidate_snapshot.snapshot,
                    target_surface_id=str(manifest_applied.target_surface_id),
                    replay_steps=steps)
                ev["patch_body_readback"] = "steps_match_replay"
        except Exception as exc:  # noqa: BLE001
            ev["stage"] = "apply_failed"
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        ev["stage"] = "applied"
        ev["candidate_snapshot_sha"] = (
            receipt.candidate_snapshot.snapshot.harness_content_sha)
        # ---- replay/delayed（evaluator 回调；沿 Typed Patch 白名单的
        # 冻结 Program）----
        support = evaluator(steps, 0)
        sg = (float(support.gain) if support.gain is not None else None)
        ev["support_gain"] = sg
        ev["support_passed"] = bool(support.verification.passed)
        # ---- 批准权（用户 2026-08-11 P0 缺口修复）：replay/delayed
        # 决定候选是否进入活跃 snapshot——LLM 不批准自己 ----
        approved = bool(support.verification.passed and sg is not None
                        and sg >= MATERIAL_THRESHOLD)
        if delayed_evaluator is not None:
            delayed = delayed_evaluator(steps, 1)
            dg = (float(delayed.gain) if delayed.gain is not None else None)
            dg_finite = dg is not None and bool(__import__("math").isfinite(dg))
            ev["delayed_gain"] = dg
            ev["delayed_ok"] = bool(
                delayed.verification.passed and dg_finite
                and dg >= -MATERIAL_THRESHOLD)
            # 复核 Blocker 3（2026-08-11）：delayed 必须 verifier 通过 +
            # gain 有限 + 不显著负向——None/NaN/verifier 失败均不批准
            if not (delayed.verification.passed and dg_finite
                    and dg >= -MATERIAL_THRESHOLD):
                approved = False
                ev["delayed_rejected"] = True
        if not approved:
            ev["stage"] = "replay_rejected"
            return ev  # snapshot 保持原版本（候选被 replay/delayed 否决）
        # ---- active snapshot 更新（下一轮 prepare 自动采用）----
        self._snapshot = receipt.candidate_snapshot.snapshot
        ev["snapshot_updated"] = True
        return ev

    # ---- P3.2 两阶段（时间边界）：Support 后 pending，delayed 到达后批准 ----

    def handle_fast_winner(
        self,
        episode: object,
        steps: Sequence[tuple[str, Mapping[str, object]]],
        *,
        controller: Any,
        store: Any,
        card: Mapping[str, object],
        evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]], int],
                            Any],
        surface_catalog: Sequence[Mapping[str, object]] | None = None,
        fast_features: Mapping[str, object] | None = None,
        # 计量修正（用户裁决 2026-08-12）：Fast winner 已在本轮探测获得
        # Support——传入该 gain 则**直接复用**（不再对相同 Context×Program
        # 重开 Support 评估——不产生不计预算的重复仪器评估）
        support_gain: float | None = None,
        # fault-routes/2 已注册 cause（授权 skill_library ADD）——Fast
        # winner 形成 Skill 的语义即"skill library gap 填充"。P2：必填，
        # 调用方必须显式给出 Runtime 授权的 cause。
        confirmed_cause: str,
    ) -> dict[str, Any]:
        """E2.5-B（用户裁决 2026-08-12）：Fast 正向 winner → Target-local
        Draft Skill 生命周期——**精确使用 trace.candidate_program_steps**
        构造 machine manifest（不调 Slow Agent——Runtime 生成）；宽 Scope
        → requires_target_support=true；Support replay + delayed 批准才写
        snapshot（同两阶段 pending——LLM 不批准自己）。"""
        import json as _json  # noqa: PLC0415
        import dataclasses as _dc  # noqa: PLC0415

        ev: dict[str, Any] = {"stage": "started"}
        applicability = _applicability_from_card(card)
        manifest = EditManifest(
            edit_id=_fast_winner_skill_id(episode),
            base_harness_sha=self._active_snapshot().harness_content_sha,
            target_pattern_id=str(card.get("pattern_id", "fast-winner")),
            target_surface_id="skill_library.entries/"
                              + _fast_winner_skill_id(episode),
            operation=EditOperation.ADD,
            surface_precondition={"kind": "ABSENT"},
            dependency_precondition_shas={},
            new_value={
                "schema_version": "skill-entry/1",
                "skill_id": _fast_winner_skill_id(episode),
                "skill_kind": "capability",
                "revision": 1,
                "body": "Frozen program steps: " + _json.dumps(
                    [{"op": o, "params": dict(p)} for o, p in steps]),
                "observable_applicability": dict(applicability),
                "allowed_tools": [o for o, _p in steps],
                "risk_guards": {"explicit_choice_required": True,
                                "observable_applicability_only": True,
                                "preserve_outside_candidate_region": True,
                                "single_surface_only": True},
            },
            observable_applicability=dict(applicability),
            patch_id=None,
            predicted_agent_behavior_change=(
                "retrieve_skill:" + _op_of_episode(episode),),
            predicted_data_effect=("local_improvement",),
            automatically_selected_risk_cases=(),
            falsification_condition=("no_improvement",),
        )
        # P1：宽 Scope → requires_target_support=true（Draft 门）
        if _applicability_is_wide(applicability):
            rg = dict(manifest.new_value.get("risk_guards") or {})
            rg["requires_target_support"] = True
            nv = dict(manifest.new_value)
            nv["risk_guards"] = rg
            manifest = _dc.replace(manifest, new_value=nv)
        # B 检查（P4.2-B）：applicability 可达
        _final_app = manifest.observable_applicability or {}
        if _final_app:
            if fast_features is None:
                ev["stage"] = "applicability_uncheckable"
                ev["applicability_reason"] = "fast_features_unavailable"
                return ev
            _ok, _reason = _applicability_reachable(
                card, _final_app, fast_features)
            if not _ok:
                ev["stage"] = "applicability_unreachable"
                ev["applicability_reason"] = _reason
                ev["action"] = "ACTION_UNAVAILABLE"
                return ev
        # apply（surface 模板实例化 + dependency SHA 补全）
        from .slow_agent import _resolve_apply_manifest  # noqa: PLC0415
        try:
            manifest_applied = _resolve_apply_manifest(
                manifest, self._active_snapshot())
            receipt = controller.apply_to_fork(
                store.materialize(self._active_snapshot()), manifest_applied,
                confirmed_cause=confirmed_cause)
        except Exception as exc:  # noqa: BLE001
            ev["stage"] = "apply_failed"
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        # Support 判定（计量修正 2026-08-12）：提供 support_gain（本轮
        # 探测已获——winner）则直接复用——不重开相同 Context×Program 的
        # Support 评估（不计预算的重复仪器评估）；否则重放确认。
        _uids = _series_uids_of_episode(episode)
        _consumer = _task_scope_of_episode(episode)[1]
        support_facts: Mapping[str, Any] | None = None
        if support_gain is not None:
            sg = float(support_gain)
            ev["support_gain"] = sg
            ev["support_passed"] = True
            ev["support_reused"] = True
            # 复用本轮探测的 Support 时，逐序列读数已由 online_loop 在写
            # Episode 时分类过——直接读回，不重开评估、也不第二次分类。
            recorded = dict(getattr(episode, "support_response", {}) or {})
            measured = recorded.get("measured_effect")
            if isinstance(measured, Mapping):
                support_facts = dict(measured)
        else:
            support = evaluator(steps, 0)
            sg = (float(support.gain) if support.gain is not None else None)
            ev["support_gain"] = sg
            ev["support_passed"] = bool(support.verification.passed)
            if sg is not None:
                support_facts = _classify_relation(
                    aggregate_gain=sg,
                    per_series_gains=_per_series_map(
                        getattr(support, "per_view_gain", None), _uids),
                    is_identity=_steps_are_identity(steps),
                    consumer_id=_consumer,
                )
        if sg is None:
            ev["stage"] = "support_rejected"
            return ev
        # T5 #41 A4：Support = POSITIVE 才形成 Draft。聚合过线但逐序列有害
        # （CONFLICT）与 NEGATIVE/NEUTRAL/ABSTAIN 只留 Episode，不扩执行权。
        # 读不到逐序列读数时 classify_relation 退化为纯聚合判定，与旧门
        # 在正向侧同结论（旧门另放行 NEUTRAL，本轮起不再放行）。
        if support_facts is None:
            support_facts = _classify_relation(
                aggregate_gain=sg, per_series_gains=None,
                is_identity=_steps_are_identity(steps),
                consumer_id=_consumer)
        ev["support_relation"] = support_facts["relation"]
        ev["support_evidence"] = dict(support_facts)
        if support_facts["relation"] != "POSITIVE":
            ev["stage"] = "support_rejected"
            ev["support_reject_reason"] = "relation_%s" % str(
                support_facts["relation"]).lower()
            return ev
        # 两阶段 pending（delayed 到达前不激活）
        self._pending_update = {
            "steps": tuple(steps),
            "manifest_applied": manifest_applied,
            "receipt": receipt,
            "episode_id": getattr(episode, "episode_id", None),
            "series_uids": tuple(_uids),
            "consumer_id": _consumer,
        }
        ev["stage"] = "pending"
        ev["edit_id"] = manifest.edit_id
        return ev

    def handle_group_feedback(
        self,
        group: Mapping[str, object],
        capsule: Mapping[str, object],
        *,
        slow_agent: Any,
        controller: Any,
        store: Any,
        card_builder: Callable[[Mapping[str, object], Mapping[str, object]],
                               Mapping[str, object]],
        evaluator_group: Callable[[Sequence[tuple[str, Mapping[str, object]]],
                                   Any], Any],
        holdout_evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]],
                                     int], Any] | None = None,
        fast_features: Mapping[str, object] | None = None,
        surface_catalog: Sequence[Mapping[str, object]],
        route_decision: Any,
        manifest_preflight: Callable[[Any], None] | None = None,
        allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
        task_context: Any | None = None,
        evidence_compiler: bool = False,
        runtime_selected_patch_id: str | None = None,
        verified_choice_offered: bool = False,
        verified_patch_ids: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """GROUP_FAULT（用户裁决 2026-08-12）：多轨迹共同错误归因的
        方法层入口——重复 first-fault 组（≥2 条失败 Episode）→ 整组
        Contrast Capsule → Slow Agent propose → **组内 replay（各组内
        Episode 的 Support 窗口全部 ≥M）** → **组外同域验证（holdout
        窗口不劣）** → pending → delayed 批准 → snapshot。

        把 Slow Path 从"单次失败→单 Card→≤2 Patch"推进为"多条失败
        轨迹→共同 first fault→共同修改→组外验证"。

        Wave 1 修复（2026-08-13）：card_builder(group, capsule)——
        Capsule 真正进入 Slow Agent 输入；manifest_preflight 默认 Typed
        Patch 契约（同单条路径——patch_id 必须属白名单）；Operator
        contracts 与 TaskContext 透传（E0 同款——不再空传）。

        P0 降级设计（2026-08-13）：evidence_compiler=True 时 Runtime
        依据 Batch Evidence 决策（LLM Batch Evidence 集成未建立——
        语义先验主导弃权）：
          - runtime_selected_patch_id=X → 白名单收敛到该 Patch——LLM
            只编译 Typed Patch（不再做选择）；
          - None → 确定性 abstain（零 LLM 调用——stage=evidence_abstain）。
        evidence_compiler=False（默认）→ 原 LLM 决策路径不变。

        E-2（rev4）：``route_decision`` 必须是现有 ProgramSupplyDecision，
        ``surface_catalog`` 必须是与其对应的单 Surface catalog；仅当
        cause=SKILL_LIBRARY_GAP、actionability=EDITABLE_M0、catalog 恰好
        是一个 capability ADD Surface 时才构造 manifest。调用方不得再
        硬编码 confirmed_cause。"""
        import json as _json  # noqa: PLC0415
        import dataclasses as _dc  # noqa: PLC0415

        ev: dict[str, Any] = {"stage": "started"}
        route_error = _group_add_route_error(route_decision, surface_catalog)
        if route_error is not None:
            ev["stage"] = "route_not_add_only"
            ev["reason"] = route_error
            return ev
        ev["route"] = {
            "case_id": getattr(route_decision, "case_id", None),
            "cause_code": getattr(route_decision, "cause_code", None),
            "actionability": getattr(route_decision, "actionability", None),
            "surface_templates": list(
                getattr(route_decision, "surface_templates", ())
            ),
        }
        workflow = str(group.get("workflow") or "?")
        card = card_builder(group, capsule)
        if verified_patch_ids is not None:
            verified_set = {str(value) for value in verified_patch_ids}
            card = dict(card)
            card["typed_patch_options"] = [
                option for option in (card.get("typed_patch_options") or [])
                if isinstance(option, Mapping)
                and str(option.get("patch_id") or "") in verified_set
            ]
            ev["verified_patch_ids"] = sorted(verified_set)
        typed_options = card.get("typed_patch_options") or []
        ev["typed_option_count"] = len(typed_options)
        if not typed_options:
            ev["stage"] = "no_verified_options"
            return ev
        # P4 strict semantics: choice_offered is only the E-1 verifier result.
        # No fallback to "count >= 2".
        ev["choice_offered"] = bool(
            not evidence_compiler
            and len(typed_options) >= 2
            and bool(verified_choice_offered)
        )
        if not ev["choice_offered"]:
            ev["no_choice_offered"] = True
        ev["verified_choice_offered"] = verified_choice_offered
        if evidence_compiler:
            if runtime_selected_patch_id is None:
                ev["stage"] = "evidence_abstain"
                ev["reason"] = "no_unique_common_positive"
                return ev
            options = card.get("typed_patch_options") or []
            filtered = [o for o in options
                        if isinstance(o, Mapping)
                        and str(o.get("patch_id"))
                        == runtime_selected_patch_id]
            if not filtered:
                ev["stage"] = "evidence_selection_unavailable"
                ev["patch_id"] = runtime_selected_patch_id
                return ev
            card = dict(card)
            card["typed_patch_options"] = filtered
        applicability = _applicability_from_card(card)
        manifest = EditManifest(
            edit_id=f"group_{workflow}_replacement",
            base_harness_sha=self._active_snapshot().harness_content_sha,
            target_pattern_id=str(card.get("pattern_id", "group-fault")),
            target_surface_id="skill_library.entries/"
                              f"group_{workflow}_replacement",
            operation=EditOperation.ADD,
            surface_precondition={"kind": "ABSENT"},
            dependency_precondition_shas={},
            new_value={
                "schema_version": "skill-entry/1",
                "skill_id": f"group_{workflow}_replacement",
                "skill_kind": "capability",
                "revision": 1,
                "body": "Group-fault replacement (proposed by Slow Agent "
                        "from Contrast Capsule)",
                "observable_applicability": dict(applicability),
                "allowed_tools": [],
                "risk_guards": {"explicit_choice_required": True,
                                "observable_applicability_only": True,
                                "preserve_outside_candidate_region": True,
                                "single_surface_only": True},
            },
            observable_applicability=dict(applicability),
            patch_id=None,
            predicted_agent_behavior_change=(
                "retrieve_skill:" + workflow,),
            predicted_data_effect=("group_fault_replacement",),
            automatically_selected_risk_cases=(),
            falsification_condition=("no_group_improvement",),
        )
        if _applicability_is_wide(applicability):
            rg = dict(manifest.new_value.get("risk_guards") or {})
            rg["requires_target_support"] = True
            nv = dict(manifest.new_value)
            nv["risk_guards"] = rg
            manifest = _dc.replace(manifest, new_value=nv)
        # Slow Agent 基于整组 Capsule propose（真实或 Replay——白名单约束）
        # Wave 1：默认 Typed Patch 契约（同单条路径——patch_id 白名单）；
        # contracts/TaskContext 透传（E0 同款）
        if manifest_preflight is None:
            manifest_preflight = lambda m: _typed_patch_preflight(card, m)  # noqa: E731
        try:
            proposed = slow_agent.propose_edit(
                card, surface_catalog,
                self._active_snapshot(),
                manifest_preflight=manifest_preflight,
                allowed_operator_contracts=allowed_operator_contracts,
                task_context=task_context)
        except (RuntimeError, AgentProtocolError) as exc:  # noqa: BLE001
            # Wave 2 接线修复（2026-08-13）：StagePostValidationError
            # （typed-patch 契约强制）如实命名——不再是误导性的
            # budget_exceeded（其基类也是 RuntimeError）
            from .agent_core import StagePostValidationError as _SPVE
            ev["stage"] = ("typed_patch_contract_failed"
                           if isinstance(exc, _SPVE)
                           else "budget_exceeded"
                           if isinstance(exc, RuntimeError)
                           else "manifest_preflight_failed")
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        if proposed is None:
            ev["stage"] = "abstained_by_agent"
            ev["no_proposal_reason"] = slow_agent.last_no_proposal_reason
            return ev
        if getattr(proposed, "operation", None) is not EditOperation.ADD:
            ev["stage"] = "proposal_not_add_only"
            ev["operation"] = getattr(proposed, "operation", None)
            return ev
        if verified_patch_ids is not None and str(
            getattr(proposed, "patch_id", None) or ""
        ) not in {str(value) for value in verified_patch_ids}:
            ev["stage"] = "verified_patch_binding_failed"
            ev["patch_id"] = getattr(proposed, "patch_id", None)
            ev["verified_patch_ids"] = sorted(
                {str(value) for value in verified_patch_ids}
            )
            return ev
        ev["stage"] = "manifest_proposed"
        ev["edit_id"] = proposed.edit_id
        ev["group_capsule"] = {
            "n_episodes": capsule.get("n_episodes"),
            "workflow": capsule.get("workflow"),
            "sign": capsule.get("sign"),
        }
        # 冻结步骤（B2：patch_id 白名单）
        from .slow_agent import _steps_for_patch_id  # noqa: PLC0415
        steps = _steps_for_patch_id(card, proposed.patch_id)
        if not steps:
            ev["stage"] = "no_frozen_program"
            ev["patch_id"] = proposed.patch_id
            return ev
        ev["frozen_program"] = [{"op": o, "params": dict(p)}
                                for o, p in steps]
        ev["patch_id"] = proposed.patch_id
        # Runtime-owned 绑定（B2 + P1——同 handle_feedback_support）
        if proposed.new_value is not None and "body" in proposed.new_value:
            nv = dict(proposed.new_value)
            nv["body"] = "Frozen program steps: " + _json.dumps(
                [{"op": o, "params": dict(p)} for o, p in steps])
            nv["skill_kind"] = "capability"
            nv["allowed_tools"] = [o for o, _p in steps]
            nv["observable_applicability"] = dict(applicability)
            if _applicability_is_wide(applicability):
                rg = dict(nv.get("risk_guards") or {})
                rg["requires_target_support"] = True
                nv["risk_guards"] = rg
            manifest = _dc.replace(
                manifest, new_value=nv,
                observable_applicability=dict(applicability))
        # B 检查（P4.2-B）
        _final_app = manifest.observable_applicability or {}
        if _final_app:
            if fast_features is None:
                ev["stage"] = "applicability_uncheckable"
                ev["applicability_reason"] = "fast_features_unavailable"
                return ev
            _ok, _reason = _applicability_reachable(
                card, _final_app, fast_features)
            if not _ok:
                ev["stage"] = "applicability_unreachable"
                ev["applicability_reason"] = _reason
                ev["action"] = "ACTION_UNAVAILABLE"
                return ev
        # apply
        from .slow_agent import _resolve_apply_manifest  # noqa: PLC0415
        try:
            manifest_applied = _resolve_apply_manifest(
                manifest, self._active_snapshot())
            receipt = controller.apply_to_fork(
                store.materialize(self._active_snapshot()), manifest_applied,
                confirmed_cause=str(route_decision.cause_code))
        except Exception as exc:  # noqa: BLE001
            ev["stage"] = "apply_failed"
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        # 组内 replay（各组内 Episode 的 Support 窗口——全部 ≥M）
        # Wave 4 修复（2026-08-13）：evaluator_group(steps, episode)——
        # 跨 series 组 origin 会碰撞（同 origin 多 series）——episode 级
        # 解析（episode_id → series → executor）
        group_gains: list[dict[str, Any]] = []
        group_ok = True
        for ep in (group.get("episodes") or []):
            origin = int(((getattr(ep, "context_summary", {}) or {})
                          .get("support_origin") or 0))
            rg_ = evaluator_group(steps, ep)
            gg = (float(rg_.gain) if rg_.gain is not None else None)
            group_gains.append({"origin": origin, "gain": gg})
            if gg is None or gg < MATERIAL_THRESHOLD:
                group_ok = False
        ev["group_replay"] = group_gains
        if not group_ok:
            ev["stage"] = "group_replay_rejected"
            return ev
        # 组外同域验证（holdout 窗口——不劣——≥ −M）
        if holdout_evaluator is not None:
            hg = (float(holdout_evaluator(steps, 0).gain)
                  if holdout_evaluator(steps, 0).gain is not None else None)
            ev["holdout_gain"] = hg
            if hg is None or hg < -MATERIAL_THRESHOLD:
                ev["stage"] = "holdout_rejected"
                return ev
        # 两阶段 pending（delayed 到达前不激活）
        self._pending_update = {
            "steps": tuple(steps),
            "manifest_applied": manifest_applied,
            "receipt": receipt,
            "episode_id": f"group:{workflow}:{capsule.get('n_episodes')}",
        }
        ev["stage"] = "pending"
        return ev

    def handle_feedback_support(
        self,
        episode: object,
        *,
        slow_agent: Any,
        controller: Any,
        store: Any,
        surface_catalog: Sequence[Mapping[str, object]],
        card_builder: Callable[[object], Mapping[str, object]],
        evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]], int],
                            Any],
        manifest_preflight: Callable[[Any], None] | None = None,
        confirmed_cause: str,
        fast_features: Mapping[str, object] | None = None,
        allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
        task_context: Any | None = None,
    ) -> dict[str, Any]:
        """Support 阶段：material 判定 → Slow Agent（≤1 次）→ patch_id →
        Support replay → **候选冻结（pending——不更新 snapshot）**。
        delayed 时间点到达前不激活（时间边界）。

        P4.2-A/B（用户裁决 2026-08-11）：applicability 由 Runtime 从 card
        observable_signature 绑定（A）+ 批准前机械可达性检查（B）——
        fast_features 为当前正常 Fast 入口的公开特征（数据回调，调用方
        提供）；缺失 → 不批准（fail-safe）。
        E0（用户裁决 2026-08-12）：Slow 调用透传合法 Operator contracts
        与现有 TaskContext。"""
        ev: dict[str, Any] = {"stage": "no_trigger"}
        sg0 = (getattr(episode, "support_response", {}) or {}).get("gain")
        relation = getattr(episode, "relation", None)
        material_negative = bool(
            isinstance(sg0, (int, float)) and float(sg0) < -MATERIAL_THRESHOLD)
        conflict = relation == "CONFLICT"
        if not (material_negative or conflict):
            return ev
        ev["triggered"] = True
        ev["material_negative"] = material_negative
        ev["relation"] = relation
        card = card_builder(episode)
        ev["card_workflow"] = [s["op"] for s in
                               card.get("workflow", {}).get("steps", [])]
        if manifest_preflight is None:
            manifest_preflight = lambda m: _typed_patch_preflight(card, m)  # noqa: E731
        try:
            manifest = slow_agent.propose_edit(
                card, surface_catalog, self._active_snapshot(),
                manifest_preflight=manifest_preflight,
                allowed_operator_contracts=allowed_operator_contracts,
                task_context=task_context)
        except (RuntimeError, AgentProtocolError) as exc:  # noqa: BLE001
            # Wave 2 接线修复（2026-08-13）：StagePostValidationError
            # （typed-patch 契约强制）如实命名——不再是误导性的
            # budget_exceeded（其基类也是 RuntimeError）
            from .agent_core import StagePostValidationError as _SPVE
            ev["stage"] = ("typed_patch_contract_failed"
                           if isinstance(exc, _SPVE)
                           else "budget_exceeded"
                           if isinstance(exc, RuntimeError)
                           else "manifest_preflight_failed")
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        if manifest is None:
            ev["stage"] = "abstained_by_agent"
            ev["no_proposal_reason"] = slow_agent.last_no_proposal_reason
            return ev
        ev["stage"] = "manifest_proposed"
        ev["edit_id"] = manifest.edit_id
        ev["target_surface_id"] = str(manifest.target_surface_id)
        ev["operation"] = manifest.operation.value
        import json as _json  # noqa: PLC0415
        import dataclasses as _dc  # noqa: PLC0415

        from .slow_agent import _steps_for_patch_id  # noqa: PLC0415

        steps = _steps_for_patch_id(card, manifest.patch_id)
        if not steps:
            ev["stage"] = "no_frozen_program"
            ev["patch_id"] = manifest.patch_id
            return ev
        ev["frozen_program"] = [{"op": o, "params": dict(p)} for o, p in steps]
        ev["patch_id"] = manifest.patch_id
        # P0（rev3）：capability Skill .body PATCH 由 Runtime 覆写 body。
        from .slow_agent import (  # noqa: PLC0415
            bind_frozen_patch_program,
            is_capability_body_surface_id,
        )
        if (manifest.operation is EditOperation.PATCH
                and is_capability_body_surface_id(
                    str(manifest.target_surface_id))):
            manifest = bind_frozen_patch_program(manifest, steps)
            ev["patch_body_binding"] = "runtime_owned"
        nv = None
        if manifest.new_value is not None and "body" in manifest.new_value:
            nv = dict(manifest.new_value)
            nv["body"] = "Frozen program steps: " + _json.dumps(
                [{"op": o, "params": dict(p)} for o, p in steps])
            nv["skill_kind"] = "capability"
            nv["allowed_tools"] = [o for o, _p in steps]
            # P4.2-A（用户裁决 2026-08-11）：Applicability-to-Observation
            # Binding——observable_applicability 由 Runtime 从 card 的
            # observable_signature（公开 Observation）机器生成（B2 同款
            # Runtime 所有权——LLM 不得编造运行时不可满足的 Scope 条件）。
            # manifest 级字段同步（controller 校验两者一致）。
            nv["observable_applicability"] = _applicability_from_card(card)
            # P1（用户裁决 2026-08-12）：宽 Scope（const:true / 仅
            # task_kind——不能区分效用翻转）的新 Skill 必须写入
            # requires_target_support=true——DRAFT 执行权限门（可进候选
            # 池但不自动优先，须当前 Target Support 确认）。
            if _applicability_is_wide(nv["observable_applicability"]):
                rg = dict(nv.get("risk_guards") or {})
                rg["requires_target_support"] = True
                nv["risk_guards"] = rg
            manifest = _dc.replace(
                manifest, new_value=nv,
                observable_applicability=nv["observable_applicability"])
        # P4.2-B（用户裁决）：批准前机械可达性检查——①特征来自 card 公开
        # Observation ②Fast 入口特征空间能产生 ③当前公开 Context 下可检索；
        # 不满足 → ACTION_UNAVAILABLE（不写 active snapshot）
        _final_app = (
            (nv.get("observable_applicability") if nv is not None else None)
            or manifest.observable_applicability or {})
        if _final_app:
            if fast_features is None:
                ev["stage"] = "applicability_uncheckable"
                ev["applicability_reason"] = "fast_features_unavailable"
                return ev
            _ok, _reason = _applicability_reachable(
                card, _final_app, fast_features)
            if not _ok:
                ev["stage"] = "applicability_unreachable"
                ev["applicability_reason"] = _reason
                ev["action"] = "ACTION_UNAVAILABLE"
                return ev
        from .slow_agent import _resolve_apply_manifest  # noqa: PLC0415
        from .slow_agent import verify_frozen_patch_program  # noqa: PLC0415

        try:
            manifest_applied = _resolve_apply_manifest(
                manifest, self._active_snapshot())
            receipt = controller.apply_to_fork(
                store.materialize(self._active_snapshot()), manifest_applied,
                confirmed_cause=confirmed_cause)
            if (manifest_applied.operation is EditOperation.PATCH
                    and is_capability_body_surface_id(
                        str(manifest_applied.target_surface_id))):
                # P0（rev3）：落盘后从 candidate snapshot 读回真实 body，
                # 与 replay steps 逐元素相等，否则 apply_failed。
                verify_frozen_patch_program(
                    receipt.candidate_snapshot.snapshot,
                    target_surface_id=str(manifest_applied.target_surface_id),
                    replay_steps=steps)
                ev["patch_body_readback"] = "steps_match_replay"
        except Exception as exc:  # noqa: BLE001
            ev["stage"] = "apply_failed"
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        # ---- Support replay（沿冻结 steps）----
        support = evaluator(steps, 0)
        sg = (float(support.gain) if support.gain is not None else None)
        ev["support_gain"] = sg
        ev["support_passed"] = bool(support.verification.passed)
        if not (support.verification.passed and sg is not None
                and sg >= MATERIAL_THRESHOLD):
            ev["stage"] = "support_rejected"
            return ev
        # ---- 候选冻结（pending——delayed 到达前不激活）----
        # T5 #41 A4：Slow 路径的 Support 门本轮不动（单假设纪律——本轮的
        # 行为机制只挂在 Fast winner 的 Draft 门与 delayed 门上）。逐序列
        # uid/consumer 仍随 pending 记下，供 delayed 侧分类使用。
        self._pending_update = {
            "steps": steps,
            "manifest_applied": manifest_applied,
            "receipt": receipt,
            "episode_id": getattr(episode, "episode_id", None),
            "series_uids": tuple(_series_uids_of_episode(episode)),
            "consumer_id": _task_scope_of_episode(episode)[1],
        }
        ev["stage"] = "pending"
        return ev

    # ---- WORKFLOW_GUIDANCE_GAP（用户任务书 2026-08-14）：组级 Slow 修改
    # Workflow Construction Skill（唯一变量 = build_contrastive_candidates
    # .body）——与 handle_group_feedback 的"组级 replacement capability"
    # 是两条不同路线：这里 Slow 提出的是 guidance 文本 PATCH，Runtime 只
    # 绑定 surface/precondition SHA（checker 裁决：Slow 无法计算 surface
    # sha——Runtime-owned 填充），不做程序 replay 门（guidance 无程序）；
    # 行为核销（G3 旧/新 snapshot replay）与 held-out Utility（G4）由
    # Runner 在 pending 后执行，activate_pending_guidance 才写入 active
    # snapshot（Slow 不批准自己，Runtime 也不批准自己——只有 replay 证据
    # 链批准）。

    def handle_group_guidance(
        self,
        group: Mapping[str, object],
        capsule: Mapping[str, object],
        *,
        slow_agent: Any,
        controller: Any,
        store: Any,
        card_builder: Callable[[Mapping[str, object], Mapping[str, object]],
                               Mapping[str, object]],
        surface_catalog: Sequence[Mapping[str, object]] | None = None,
        confirmed_cause: str = "WORKFLOW_GUIDANCE_GAP",
        manifest_preflight: Callable[[Any], None] | None = None,
        allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
        task_context: Any | None = None,
    ) -> dict[str, Any]:
        ev: dict[str, Any] = {"stage": "started"}
        card = card_builder(group, capsule)
        if surface_catalog is None:
            surface_catalog = [{
                "surface_id": "bootstrap_skills.entries/"
                              "build_contrastive_candidates.body",
                "operation": "PATCH",
                "surface_type": "skill_body",
                "allowed_operations": ["PATCH"],
            }]
        try:
            proposed = slow_agent.propose_edit(
                card, surface_catalog, self._active_snapshot(),
                manifest_preflight=manifest_preflight,
                allowed_operator_contracts=allowed_operator_contracts,
                task_context=task_context)
        except (RuntimeError, AgentProtocolError) as exc:  # noqa: BLE001
            from .agent_core import StagePostValidationError as _SPVE
            ev["stage"] = ("typed_patch_contract_failed"
                           if isinstance(exc, _SPVE)
                           else "budget_exceeded"
                           if isinstance(exc, RuntimeError)
                           else "manifest_preflight_failed")
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        if proposed is None:
            ev["stage"] = "no_manifest"
            ev["no_proposal_reason"] = slow_agent.last_no_proposal_reason
            return ev
        ev["stage"] = "manifest_proposed"
        ev["edit_id"] = proposed.edit_id
        # Runtime-owned surface 绑定：guidance 只许 PATCH 唯一 surface
        #（router 的 surface_ids 白名单在 authorize 时同样强制）。
        if str(proposed.target_surface_id) != (
                "bootstrap_skills.entries/build_contrastive_candidates.body"):
            ev["stage"] = "surface_mismatch_rejected"
            ev["target_surface_id"] = str(proposed.target_surface_id)
            return ev
        if proposed.operation is not EditOperation.PATCH:
            ev["stage"] = "operation_mismatch_rejected"
            ev["operation"] = proposed.operation.value
            return ev
        patch_value = (
            proposed.minimal_patch.get("value")
            if isinstance(proposed.minimal_patch, Mapping) else None)
        if not isinstance(patch_value, str) or not patch_value.strip():
            ev["stage"] = "empty_patch_rejected"
            return ev
        # P3 clause 级 Runtime-owned binding（用户裁决 2026-08-14 修订）：
        # Slow 的提案值 = REPLACE_CLAUSE（target + new_clause）——只改一条
        # propose.rule.<id>；Runtime 解析并绑定到正确规则位置，其余内容
        # （FIXED_CONTRACT / inspect_pattern_guidance / select_guidance /
        # 其他 clause）逐字不变——由确定性代码保证，不依赖模型复制。
        current_body = _guidance_body(self._active_snapshot())
        clause_payload = _parse_clause_payload(patch_value)
        if clause_payload is None:
            ev["stage"] = "clause_payload_invalid"
            return ev
        try:
            final_body = _apply_clause_replacement(
                current_body, clause_payload)
        except ValueError as exc:
            ev["stage"] = "clause_target_unknown"
            ev["error"] = str(exc)
            return ev
        ev["guidance_clause_proposed"] = dict(clause_payload)
        ev["guidance_editable_proposed"] = patch_value
        # apply（checker 裁决 2026-08-14：surface precondition SHA 无注入
        # 点——Runtime 在此填充，与 dependency shas 同构；Slow 的 sha 猜测
        # 一律忽略，避免 StaleEditError 假性拒绝）。
        import dataclasses as _dc2  # noqa: PLC0415
        from .slow_agent import _resolve_apply_manifest  # noqa: PLC0415
        parent = store.materialize(self._active_snapshot())
        resolved = _resolve_apply_manifest(proposed, self._active_snapshot())
        surface_sha = controller.surface_precondition_sha(
            parent, str(proposed.target_surface_id))
        manifest_applied = _dc2.replace(
            resolved,
            minimal_patch={"value": final_body},
            surface_precondition={"kind": "SHA", "sha": surface_sha})
        try:
            receipt = controller.apply_to_fork(
                parent, manifest_applied, confirmed_cause=confirmed_cause)
        except Exception as exc:  # noqa: BLE001
            ev["stage"] = "apply_failed"
            ev["error"] = f"{type(exc).__name__}: {exc}"
            return ev
        ev["stage"] = "applied"
        ev["candidate_snapshot_sha"] = (
            receipt.candidate_snapshot.snapshot.harness_content_sha)
        ev["guidance_body_old"] = _guidance_body(
            self._active_snapshot())
        ev["guidance_body_new"] = patch_value
        # pending：行为核销/held-out Utility 由 Runner 执行，激活只经
        # activate_pending_guidance（replay 证据链批准后调用）。
        self._pending_update = {
            "kind": "guidance",
            "receipt": receipt,
            "episode_id": f"group:{group.get('workflow')}:"
                          f"{capsule.get('n_episodes')}",
        }
        ev["stage"] = "pending"
        return ev

    def pending_guidance_snapshot(self) -> Any | None:
        pend = self._pending_update
        if pend and pend.get("kind") == "guidance":
            return pend["receipt"].candidate_snapshot.snapshot
        return None

    def activate_pending_guidance(
        self,
        *,
        g3_behavior_verified: bool = False,
        g4_support_passed: bool = False,
        delayed_ok: bool = False,
    ) -> bool:
        """只有 replay 证据链（G3 行为核销 + G4 held-out Support/delayed）
        全部通过时才激活——证据缺一即拒绝（用户裁决 2026-08-14：Runner
        不得持有裸激活权；Slow 不批准自己，方法层也不得无条件替换）。"""
        pend = self._pending_update
        if not pend or pend.get("kind") != "guidance":
            return False
        if not (g3_behavior_verified and g4_support_passed and delayed_ok):
            return False
        self._snapshot = pend["receipt"].candidate_snapshot.snapshot
        self._pending_update = None
        return True

    def adopt_guidance_candidate(
        self,
        candidate_root: Any,
        *,
        g3_behavior_verified: bool = False,
        g4_support_passed: bool = False,
        delayed_ok: bool = False,
        parent_snapshot: Any | None = None,
    ) -> dict[str, Any]:
        """跨进程正常反馈生命周期的 guidance 批准接口（G5 用）：证据链
        齐全 + 候选与当前 guidance 确有不同的前提下，把已编译候选写入
        active snapshot。返回事件记录；Runner 无直接写 snapshot 的权限。"""
        if not (g3_behavior_verified and g4_support_passed and delayed_ok):
            return {"adopted": False,
                    "reason": "evidence_chain_incomplete"}
        from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
            compile_snapshot,
        )
        try:
            candidate = compile_snapshot(candidate_root, verify_lock=False)
        except Exception as exc:  # noqa: BLE001
            return {"adopted": False,
                    "reason": f"compile_failed:{type(exc).__name__}"}
        parent = parent_snapshot if parent_snapshot is not None \
            else self._active_snapshot()
        if _guidance_body(candidate) == _guidance_body(parent):
            return {"adopted": False, "reason": "no_guidance_change"}
        self._snapshot = candidate
        self._pending_update = None
        return {"adopted": True,
                "harness_content_sha": candidate.harness_content_sha,
                "evidence": {
                    "g3_behavior_verified": bool(g3_behavior_verified),
                    "g4_support_passed": bool(g4_support_passed),
                    "delayed_ok": bool(delayed_ok),
                }}

    def discard_pending_guidance(self) -> None:
        if self._pending_update and self._pending_update.get("kind") == "guidance":
            self._pending_update = None

    def handle_feedback_delayed(
        self,
        delayed_evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]],
                                    int], Any],
        *,
        episode_id: str | None = None,
    ) -> dict[str, Any]:
        """delayed 阶段（时间点到达后调用）：delayed 验证 → 批准（更新
        active snapshot）/拒绝（丢弃 pending）。episode_id 提供时必须与
        pending 的触发 Episode 匹配（复核 Major：错误轮次的 delayed 不得
        批准当前 Patch）。"""
        if self._pending_update is None:
            return {"stage": "no_pending"}
        pend = self._pending_update
        # guidance pending 不经 delayed 程序评估批准（其批准链 = G3 行为
        # 核销 + G4 Support/delayed，由 Runner 调用
        # activate_pending_guidance）——防止 KeyError 与误批准。
        if pend.get("kind") == "guidance":
            return {"stage": "guidance_pending_not_program"}
        if episode_id is not None and pend.get("episode_id") != episode_id:
            return {"stage": "episode_mismatch"}
        delayed = delayed_evaluator(pend["steps"], 1)
        dg = (float(delayed.gain) if delayed.gain is not None else None)
        dg_finite = dg is not None and bool(__import__("math").isfinite(dg))
        ev: dict[str, Any] = {"stage": "pending", "delayed_gain": dg}
        # 复核 Blocker 3（2026-08-11）：delayed 必须 verifier 通过 + gain
        # 有限；None/NaN/verifier 失败 → 拒绝（丢弃 pending，snapshot 不变）
        if not (delayed.verification.passed and dg_finite):
            ev["stage"] = "delayed_rejected"
            ev["delayed_reject_reason"] = (
                "verifier_failed" if not delayed.verification.passed
                else "gain_unavailable")
            self._pending_update = None
            return ev
        # T5 #41 A4（生命周期风险门）：批准条件由 "dg >= -MATERIAL_THRESHOLD"
        # 改为 classify_relation(...) == POSITIVE，与 Memory 卡、online_loop
        # 的写回共用同一个分类器。两处实质变化：
        #   * 聚合过线但有逐序列伤害 → CONFLICT → 不批准（原门读不到逐序列
        #     读数，这类候选会被放进 active snapshot）；
        #   * NEUTRAL（|dg| < 阈值）不再扩权——原门 dg ≥ −0.005 即批准，
        #     零效果的候选也能进 snapshot。这是本轮授权的行为变化。
        # aggregate 与 per-series 原始读数无论批准与否都留在 evidence 里。
        facts = _classify_relation(
            aggregate_gain=dg,
            per_series_gains=_per_series_map(
                getattr(delayed, "per_view_gain", None),
                pend.get("series_uids")),
            is_identity=_steps_are_identity(pend["steps"]),
            consumer_id=pend.get("consumer_id"),
        )
        ev["delayed_relation"] = facts["relation"]
        ev["delayed_evidence"] = dict(facts)
        if facts["relation"] != "POSITIVE":
            ev["stage"] = "delayed_rejected"
            ev["delayed_reject_reason"] = "relation_%s" % str(
                facts["relation"]).lower()
            # CONFLICT/NEGATIVE：丢弃 pending。已部署 Skill 的限制由
            # online_loop 的 delayed 状态更新（RESTRICTED）与 revoke 路径
            # 处理——本方法不越权改别人的 snapshot 条目。
            self._pending_update = None
            return ev
        # episode_id 匹配检查（复核 Major）：pending 只应由其对应 Episode
        # 的 delayed 批准
        self._snapshot = pend["receipt"].candidate_snapshot.snapshot
        self._pending_update = None
        ev["stage"] = "approved"
        ev["snapshot_updated"] = True
        return ev


def fast_winner_skill_id(episode: object) -> str:
    """The Fast-winner Skill id for this Episode -- the public spelling.

    T5 (#41 A5) made this id task-scoped, and every caller that had its own
    ``f"fast_winner_{signature}"`` copy silently stopped agreeing with the
    manifest the method layer writes.  There is one rule and it lives here;
    call this instead of rebuilding the string.
    """
    return _fast_winner_skill_id(episode)


__all__ = ["TTHAMethod", "fast_winner_skill_id"]
