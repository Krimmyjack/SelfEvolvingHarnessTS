"""methods/ttha/online_loop.py——唯一薄在线入口（P2，用户裁决
2026-08-12）。

复用现有 TTHAMethod / ScopeExecutor / Episode（experience_memory.
build_episode）/ HarnessStore——不重建另一套 Agent/Controller。

三个函数：
  run_online_round(...) —— 一轮在线：prepare 一次 → chosen-first 探测
    → 每次合法 Action-Response 立即写 Episode → 第一个正向候选成为
    winner → 第一个 material failure 触发一次 Slow Path（replay 与
    Fast probe 共用同一 Target Support 预算）→ pending Patch（两阶段，
    不更新 snapshot）。
  open_delayed(...)     —— delayed 到达后：更新所有 Episode 的 delayed
    状态 + 用对应 episode_id 调用 handle_feedback_delayed（批准才更新
    snapshot；否则拒绝且 snapshot 不变）。
  current_status(...)   —— active snapshot / episodes 数 / pending /
    last_round / last_delayed / draft|active|restricted skills。

14 条固定语义（用户裁决）：
  1  prepare 一次；2 探测顺序 = Agent chosen（non-identity）→ 其余候选
    池顺序；3 Target Support 预算 ≤2；4 verifier 拒绝单独记 proposal，
    不计合法 Support receipt；5 每次合法 Action-Response 立即写 Episode；
    6 第一个正向候选成为 winner（停止探测）；7 第一个 material failure
    触发一次 Slow Path；8 Slow replay 消耗同一预算（不获额外反馈）；
    9 预算用尽 → SLOW_UPDATE_DEFERRED_NO_TARGET_BUDGET（不偷偷 replay）；
    10 Support 阶段只产生 pending——必须走 handle_feedback_support；
    11 delayed 用对应 episode_id 调 handle_feedback_delayed；
    12 delayed verifier 失败 / gain None/NaN / 显著负向 → 拒绝、snapshot
    不变；13 只有批准后的 snapshot 才经 HarnessStore.set_active() 更新
    正常入口（由调用方在批准后调用）；14 Runner 不自行调 Slow Agent/
    Controller。

统一指标（RoundResult——普通 dataclass，非 Ledger/Schema）：
  proposal_count / target_support_receipts_used /
  slow_replay_receipts_used / actual_probed_programs / winner_program /
  first_positive_support_receipt_index / harm_count / harm_magnitude /
  abstained / episode_ids / pending_patch_id / approved_skill_id /
  next_round_skill_retrieved / next_round_skill_chosen / delayed_utility
  —— delayed_utility 统计**实际 winner** 的 delayed gain（不再统计第一
  probe）。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .experience_memory import (
    MEASURED_EFFECT_KEY,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_DRAFT,
    build_episode,
    classify_relation,
    task_consumer_key,
    workflow_signature_of,
)
from .exploration_policy import (
    active_policy as _exploration_policy,
    is_supplied_candidate as _is_supplied,
)
from .fast_agent import public_operator_contract
from .signed_radius import MATERIAL_THRESHOLD, window_context

M = MATERIAL_THRESHOLD  # 0.005


@dataclass
class RoundResult:
    """一轮在线结果（普通数据对象——16 项统一指标 + 内部状态）。"""

    round_name: str
    origin: int
    proposal_count: int = 0
    target_support_receipts_used: int = 0
    slow_replay_receipts_used: int = 0
    actual_probed_programs: list = field(default_factory=list)
    winner_program: list | None = None
    first_positive_support_receipt_index: int | None = None
    harm_count: int = 0
    harm_magnitude: float = 0.0
    abstained: bool = True
    episode_ids: list = field(default_factory=list)
    pending_patch_id: str | None = None
    approved_skill_id: str | None = None
    next_round_skill_retrieved: bool | None = None
    next_round_skill_chosen: bool | None = None
    delayed_utility: float | None = None
    # P0（2026-08-15 评审裁定，UPDATE_POLICY_FAULT 修复）：winner 来自已批准
    # Skill（cand_skill_*）时记录部署，不再重复 ADD；该轮 delayed < −M 时
    # 撤销并记录 tombstone。
    deployed_skill_id: str | None = None
    revoked_skill_id: str | None = None
    revoked_runtime_bundle_sha: str | None = None
    # E0（用户裁决 2026-08-12）：chosen proposal 与 authorized deployment
    # 明确分开——chosen_proposal 只是 prepare 的选择（可能未获授权）；
    # authorized_deployment 必须是经当前 Target Support 探测确认的正向
    # 候选（= winner_program）。DRAFT 被 chosen 只表示申请一次 Support。
    chosen_proposal: str | None = None
    memory_resolution_status: str = "no_memory"
    # E1（2026-08-16）Runtime ordering consumer：Domain Ordering Card 只重排
    # Fast 已供应的候选——不注入候选、不改 Program Supply、不做 suppression。
    ordering_card_id: str | None = None
    probe_order_before_card: list = field(default_factory=list)
    probe_order_after_card: list = field(default_factory=list)
    # ---- 内部状态（open_delayed 消费；不对外）----
    _episodes: list = field(default_factory=list, repr=False)  # (episode, steps)
    _winner_candidate_id: str | None = field(default=None, repr=False)
    _winner_steps: tuple | None = field(default=None, repr=False)
    _slow_event: dict | None = field(default=None, repr=False)
    _trigger_episode_id: str | None = field(default=None, repr=False)
    _delayed_event: dict | None = field(default=None, repr=False)
    _method: Any = field(default=None, repr=False)
    _values: Mapping[str, Any] | None = field(default=None, repr=False)
    # T5 #41 A3：本轮的真实任务绑定——delayed 侧要用同一个键/同一批 uid
    # 分类，不得在 open_delayed 里第二次猜。
    _task_spec: Any = field(default=None, repr=False)
    _series_uids: tuple = field(default=(), repr=False)
    _consumer_id: str | None = field(default=None, repr=False)
    _period: int = field(default=24, repr=False)
    _domain: str = field(default="target", repr=False)
    _deferred_slow: str | None = field(default=None, repr=False)
    _fast_skill_event: dict | None = field(default=None, repr=False)
    _fast_skill_episode_id: str | None = field(default=None, repr=False)
    _group_slow_event: dict | None = field(default=None, repr=False)
    _group_slow_done: bool = field(default=False, repr=False)


def _per_series_gains(per_view_gain: Sequence[float] | None,
                      roster_uids: Sequence[str] | None = None
                      ) -> dict[str, float] | None:
    """per_view_gain 是位置序列（评估仪器逐 view 输出）；classify_relation
    要的是 uid → gain 映射。T5 #41 A3：只做形状转换，不做判断——没有
    per-view 读数就返回 None（"读不到"与"读到 0 条"不得混同）。"""
    if per_view_gain is None:
        return None
    values = [float(v) for v in per_view_gain]
    uids = list(roster_uids or ())
    if len(uids) != len(values):
        uids = ["view_%d" % i for i in range(len(values))]
    return {str(uid): value for uid, value in zip(uids, values)}


_SUPPLIED_CANDIDATE_PREFIX = "cand_skill_"


def source_skill_of_candidate(candidate_id: Any) -> str | None:
    """Which Skill card placed this candidate, or ``None`` if the agent did.

    SA-1 Part 0 (1).  ``fast_agent._skill_frozen_candidates`` mints the id and
    the ``Candidate.source`` from one string -- ``cand_skill_<skill_id>`` and
    ``skill:<skill_id>`` (``fast_agent.py:365-369``) -- so decoding the prefix
    recovers exactly what ``source`` carries.  The probe loop only keeps the
    id, and this is the one place that turns it back into a card id.  Pure
    reader: nothing here changes what is proposed, probed or written.
    """
    text = str(candidate_id or "")
    if text.startswith("skill:"):
        return text[len("skill:"):] or None
    if text.startswith(_SUPPLIED_CANDIDATE_PREFIX):
        return text[len(_SUPPLIED_CANDIDATE_PREFIX):] or None
    return None


def _write_target_episode(*, domain: str, op: str,
                          program_steps: Sequence[Mapping[str, object]],
                          support_gain: float, support_context: Mapping[str, float],
                          episode_id_suffix: str,
                          per_view_gain: Sequence[float] | None = None,
                          support_origin: int | None = None,
                          task_spec: Any = None,
                          series_uids: Sequence[str] | None = None,
                          consumer_id: str | None = None,
                          source_skill_id: str | None = None) -> Any:
    """与 run_v1_target_local_loop.write_target_episode 同构（生产路径
    直接复用 experience_memory.build_episode）。

    GROUP_FAULT（用户裁决 2026-08-12）：保留 per-view（per-series）
    Action–Response——多轨迹共同归因需要看到"哪些 series 改善/恶化"。
    per_view_gain 是评估仪器的细粒度输出（学习证据——非部署 Context——
    不渲染进 instruction——signed_radius 只取 recent./change. 键）。
    Wave 1（2026-08-13）：workflow_signature 用完整算子序列指纹
    （多步 workflow 不再压成首算子——分组键精度）。"""
    full_sig = workflow_signature_of(
        [{"op": s.get("op"), "params": dict(s.get("params") or {})}
         for s in program_steps]) if program_steps else op
    # T5 #41 A3（写回统一）：任务硬键从真实 request.task_spec 铸出——原先
    # 这里是一句 forecast|ridge|sMASE 字面量，任何非预测轮次写出的经验都
    # 会落在预测键下，AD 检索永远找不到。task_spec=None 时 helper 自己
    # 回落到历史默认，语义与旧字面量一致（legacy fixture 断言即此格）。
    key = task_consumer_key(task_spec)
    # 生命周期改读风险感知分类：relation 不再由 support_gain 的符号一句
    # 决定，而是与 Memory 卡、delayed 门共用同一个 classify_relation。
    # "聚合正、逐序列有害" 因此必然写成 CONFLICT，而不是 POSITIVE。
    facts = classify_relation(
        aggregate_gain=support_gain,
        per_series_gains=_per_series_gains(per_view_gain, series_uids),
        is_identity=(op == "identity"),
        consumer_id=consumer_id,
    )
    relation = str(facts["relation"])
    return build_episode(
        episode_id=f"{domain}_target_{op}{episode_id_suffix}",
        task_consumer_key=key,
        domain_namespace=domain,
        context_summary={
            # B 修 1：Episode 的证据计数单位。risk_skill._task_of 读的就是
            # 这个键，此前在线回路一处都不写，于是 _task_of 恒得空串，跨单元
            # 害证在 census 里塌成同一个 Task（分类线两单元 → count=1）。
            # 取 domain：它就是 domain_namespace 本身，由调用方从 cell /
            # cohort 标识机械拼出（分类线是 dataset/condition），在一个单元
            # 内跨探测、跨 r1/r2 恒定，换单元必变；且不含 Outcome、future
            # 或 delayed 读数——本 Episode 已经带着同一个串了。
            # 语义与预测线 e1._make_episode:937 同：一个 Task 单元一个串。
            "task_episode_id": str(domain),
            # SA-1 Part 0 (1): which card supplied the candidate this Episode
            # is a reading of, null when the agent proposed it itself.  Before
            # this field a card-supplied hampel Episode and a self-proposed
            # one were the same record, and the only join available was the
            # workflow signature -- which cannot tell the card's work from the
            # agent's luck when both name the same program.
            "source_skill_id": str(source_skill_id) if source_skill_id else None,
            "cohort": {"series_count": 1, "evaluation_series_count": 0},
            "local_pattern": {"support_gain": support_gain,
                              **(dict(support_context) if support_context else {})},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": list(program_steps)},
            "per_view_gain": list(per_view_gain)
            if per_view_gain is not None else [],
            # T5 #41 A3：逐 view 读数的 uid 序一并留痕——delayed 侧与
            # method 层要按同一批 uid 分类，不得各自猜。
            "series_uids": [str(u) for u in (series_uids or ())],
            "support_origin": support_origin,
        },
        workflow_signature=full_sig,
        support_response={"gain": support_gain,
                          "accepted": relation == "POSITIVE",
                          MEASURED_EFFECT_KEY: dict(facts)},
        delayed_response={"evaluated": False, "gain": None},
        relation=relation,
        evidence_level="SUPPORT",
        # Support = POSITIVE 才形成 Draft；CONFLICT/NEGATIVE/NEUTRAL/ABSTAIN
        # 只写 Episode，不扩执行权。
        local_status=STATUS_LOCAL_DRAFT if relation == "POSITIVE"
        else STATUS_EPISODE_ONLY,
        evidence_refs=["online_loop"],
    )


def _update_delayed_status(episode: Any, delayed_gain: float,
                           delayed_context: Mapping[str, float],
                           *,
                           per_view_gain: Sequence[float] | None = None,
                           series_uids: Sequence[str] | None = None,
                           consumer_id: str | None = None) -> Any:
    """T5 #41 A3（写回统一）：delayed 的 relation 也走 classify_relation。

    此前这里是第二套符号判断——四个状态由 (support 符号, delayed 符号)
    的组合硬写出来，既不看逐序列读数，也没有"聚合正、逐序列有害"这一格；
    真实回路因此**产生不出** CONFLICT，而 Memory 的卡正靠它。现在关系
    由同一个分类器给出，状态只作机械映射：

      POSITIVE                      -> LOCAL_ACTIVE（唯一扩权格）
      CONFLICT（含聚合正逐序列害）  -> RESTRICTED（已部署者受限，不撤证据）
      NEGATIVE / NEUTRAL / ABSTAIN  -> EPISODE_ONLY

    NEUTRAL 从此不再进 LOCAL_ACTIVE——这是本轮授权的行为变化，不是回归。
    aggregate 与 per-series 原始读数一并留在 delayed_response 里。"""
    facts = classify_relation(
        aggregate_gain=delayed_gain,
        per_series_gains=_per_series_gains(per_view_gain, series_uids),
        is_identity=(str(episode.workflow_signature) == "identity"),
        consumer_id=consumer_id,
    )
    relation = str(facts["relation"])
    if relation == "POSITIVE":
        status = "LOCAL_ACTIVE"
    elif relation == "CONFLICT":
        status = "RESTRICTED"
    else:
        status = "EPISODE_ONLY"
    return dataclasses.replace(
        episode,
        delayed_response={"evaluated": True, "gain": delayed_gain,
                          "per_view_gain": ([float(v) for v in per_view_gain]
                                            if per_view_gain is not None
                                            else None),
                          MEASURED_EFFECT_KEY: dict(facts)},
        relation=relation,
        evidence_level="DELAYED",
        local_status=status,
        context_summary={
            **dict(episode.context_summary),
            "delayed_pattern": dict(delayed_context),
        },
    )


def _select_ordering_card(method, public_features, scope_now):
    """E1（2026-08-16）：从 active snapshot 里挑一张适用的 Domain Ordering Card。

    两道门（缺一不可）——
      1. ``observable_applicability``：复用 retrieval 的同一个求值器（``task_kind == ...``）；
      2. ``card_scope_matches``：Runtime 机械精确匹配 domain / downstream_model_class /
         program_family——``resolve_harness_view`` **不读 risk_guards**，这一层它管不到。

    直接读 snapshot 而不走 ``resolve_harness_view``，是为了让排序卡不去和供应型
    capability skill 抢 retrieval 的 top_k 名额——候选供应链路完全不受本函数影响。
    """
    from SelfEvolvingHarnessTS.methods.ttha import ordering_card as oc  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: PLC0415
        evaluate_applicability)
    try:
        snap = method._active_snapshot()  # noqa: SLF001
    except Exception:  # noqa: BLE001 —— 无 active snapshot → 无卡，静默降级
        return None
    matches = []
    for skill in getattr(snap, "skills", ()):  # 确定性：按 skill_id 排序取首个
        if not oc.is_ordering_card(skill):
            continue
        if not evaluate_applicability(skill.observable_applicability, public_features)[0]:
            continue
        if not oc.card_scope_matches(skill, scope_now):
            continue
        matches.append(skill)
    matches.sort(key=lambda item: item.skill_id)
    return matches[0] if matches else None


def run_online_round(
    method: Any,
    executor: Any,
    request: Any,
    values: Mapping[str, Any],
    *,
    origin: int,
    slow_agent: Any,
    controller: Any,
    store: Any,
    card_builder: Callable[[object], Mapping[str, object]],
    round_name: str = "r1",
    budget: int = 2,
    allow_slow: bool = True,
    horizon: int = 48,
    period: int = 24,
    domain: str = "target",
    fast_features: Mapping[str, object] | None = None,
    surface_catalog: Sequence[Mapping[str, object]] | None = None,
    allow_fast_skill: bool = False,
    runtime_prior_slot: bool = False,
    allow_group_slow: bool = False,
    group_min: int = 2,
    group_card_builder: Callable[[Mapping[str, object]],
                                 Mapping[str, object]] | None = None,
    group_holdout_origin: int | None = None,
    ordering_program_family: str | None = None,
) -> RoundResult:
    """一轮在线（14 条固定语义——见模块 docstring）。
    E2.5-A/B（用户裁决 2026-08-12）：runtime_prior_slot=Runtime-owned
    双槽（prepare 透传）；allow_fast_skill=Fast winner → Draft Skill
    生命周期（方法层 handle_fast_winner——Runner 不手工 write_skill）。
    GROUP_FAULT（用户裁决 2026-08-12）：allow_group_slow=失败 Episode
    积累 → 轻量分组（≥group_min 同算子同 sign）→ 组级 Slow
    （method.handle_group_feedback——整组 Capsule → 组内/组外验证）——
    多轨迹共同归因接入主链（自动触发层）。group_card_builder 为组 Card
    数据回调（白名单需要 executor 验证——Runner 提供）；缺省无白名单
    → no_frozen_program（如实记录）。"""
    result = RoundResult(round_name=round_name, origin=origin)
    result._method = method
    result._values = values
    result._period = period
    result._domain = domain
    result._task_spec = getattr(request, "task_spec", None)
    result._series_uids = tuple(str(uid) for uid in values)
    result._consumer_id = str(getattr(request.task_spec, "downstream_model_class", ""))         if getattr(request, "task_spec", None) is not None else None
    series0 = np_values(request, values)
    # 1. prepare 一次（E2.5-A：runtime_prior_slot 透传——真实 LLM +
    # Runtime-owned 双槽）
    # #42k Part B2：直读 task_spec.task_type——PreparationRequest 的
    # __post_init__ 已保证 task_spec 是 TaskSpec，"forecast" 默认分支永远
    # 不会命中，只会在字段真的读不到时把异常检测轮次静默当预测轮跑。
    method.bind_round_data(series0[:origin],
                           task_kind=request.task_spec.task_type)
    method.prepare(request, runtime_prior_slot=runtime_prior_slot)
    trace = method.last_trace
    steps_map = dict(trace.candidate_program_steps or {})
    pool = [c for c in (trace.candidate_ids or ()) if c in steps_map]
    chosen = trace.chosen_candidate_id or ""
    result.chosen_proposal = chosen if chosen != "identity" else None
    # P1/P2（rev3）：Program Supply 在线归因只走公开纯路由；当前 view 用
    # 与 prepare 相同的公开特征重建（不用 fixture 默认值、不调 assess_case）。
    from .public_tools import extract_public_features  # noqa: PLC0415
    from .retrieval import resolve_harness_view  # noqa: PLC0415

    _route_features = extract_public_features(
        series0[:origin],
        task_kind=request.task_spec.task_type)
    _route_view = resolve_harness_view(
        method._active_snapshot(), _route_features, role="fast")
    result.memory_resolution_status = str(
        getattr(trace, "memory_resolution_status", "no_memory"))
    # 2. 探测顺序：Agent chosen（non-identity 且非空——空 chosen 表示
    #    select 输出异常——不入探测序）→ 其余候选池顺序
    # Stage 3 Part 0（2026-08-29）：探测序参数化。DEFAULT
    # （chosen_first_then_pool）= 参数化前行为；非 DEFAULT 序按供给/自主
    # 分组重排（identity 归入非供给组，保持池内相对序），不增不删。
    _policy = _exploration_policy()
    if _policy.probe_order_rule == "pool_as_built":
        probe_order = list(pool)
    elif _policy.probe_order_rule == "supply_first_then_agent":
        probe_order = ([c for c in pool if _is_supplied(c)]
                       + [c for c in pool if not _is_supplied(c)])
    elif _policy.probe_order_rule == "agent_first_then_supply":
        probe_order = ([c for c in pool if not _is_supplied(c)]
                       + [c for c in pool if _is_supplied(c)])
    else:  # chosen_first_then_pool（DEFAULT）
        probe_order = [
            c for c in ([chosen] if chosen and chosen != "identity" else [])
        ] + [c for c in pool if c != chosen]
    # 2b. E1（2026-08-16）Runtime ordering consumer——只重排，不增不删。
    #     无卡 / scope 不匹配 → probe_order 原样通过（默认行为不变）。
    result.probe_order_before_card = list(probe_order)
    _scope_now = {
        # #42k-b F4: same fix as Part B2 -- direct read, no "forecast" default.
        "task": str(request.task_spec.task_type),
        "domain": str(domain),
        "downstream_model_class": str(
            getattr(request.task_spec, "downstream_model_class", "")),
        "program_family": str(ordering_program_family or ""),
    }
    _card = _select_ordering_card(method, dict(fast_features or {}), _scope_now)
    if _card is not None:
        from SelfEvolvingHarnessTS.methods.ttha import (  # noqa: PLC0415
            ordering_card as _oc)
        probe_order = _oc.reorder_probe_order(probe_order, steps_map, _card)
        result.ordering_card_id = _card.skill_id
    result.probe_order_after_card = list(probe_order)
    result.proposal_count = len(pool)
    triggered = False
    # Stage 3 Part 0：supply_reserved_probe_slots（DEFAULT=0 时本守卫从不
    # 触发）——剩余合法 receipt ≤ 保留数且后方仍有未探供给候选时，跳过
    # 非供给候选（不评估、不耗 receipt），把预算位留给供给候选。
    # winner_compare_rule=max_support_gain_among_probed_positive 时在已探
    # POSITIVE 内按 displacement_margin/tie_break_rule 比较（_best_positive）；
    # DEFAULT=first_positive_in_probe_order 保持首正即胜。
    _best_positive: tuple[str, tuple, float] | None = None
    for _probe_idx, cand in enumerate(probe_order):
        if result.target_support_receipts_used >= budget:
            break
        if (_policy.supply_reserved_probe_slots > 0
                and not _is_supplied(cand)
                and (budget - result.target_support_receipts_used
                     <= _policy.supply_reserved_probe_slots)
                and any(_is_supplied(later)
                        for later in probe_order[_probe_idx + 1:])):
            result.actual_probed_programs.append({
                "candidate_id": cand, "kind": "skipped_reserved_for_supply",
                "gain": None, "passed": None})
            continue
        steps = steps_map[cand]
        rr = executor.evaluate(tuple(steps), origin)
        if not rr.verification.passed or rr.gain is None:
            # 4. verifier 拒绝/仪器失败：单独记 proposal，不计合法 receipt
            result.actual_probed_programs.append({
                "candidate_id": cand, "kind": "verifier_rejected",
                "gain": None, "passed": False})
            continue
        result.target_support_receipts_used += 1
        gain = float(rr.gain)
        result.actual_probed_programs.append({
            "candidate_id": cand, "kind": "probe", "gain": gain,
            "passed": True})
        # 5. 每次合法 Action-Response 立即写 Episode
        # GROUP_FAULT（用户裁决 2026-08-12）：保留 per-view（per-series）
        # gain——多轨迹共同归因的细粒度证据（学习证据——不进 instruction）
        ep = _write_target_episode(
            domain=result._domain,
            op=_op_of(cand, steps),
            program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
            support_gain=gain,
            support_context=dict(window_context(values, origin, period)),
            episode_id_suffix=f"_{round_name}_p{len(result._episodes) + 1}",
            per_view_gain=getattr(rr, "per_view_gain", None),
            support_origin=origin,
            task_spec=result._task_spec,
            series_uids=result._series_uids,
            consumer_id=result._consumer_id,
            source_skill_id=source_skill_of_candidate(cand))
        method.append_experience_episode(ep)
        result.episode_ids.append(ep.episode_id)
        result._episodes.append((ep, tuple(steps)))
        # 6. 第一个正向候选成为 winner（= authorized deployment——
        # 经当前 Target Support 探测确认；停止探测）。
        # E0：first-positive index 用合法 Support receipt 计数（不含
        # verifier_rejected 条目——原 len(actual_probed_programs) 会算入）。
        # T5 #41 A4：Support = POSITIVE 才形成 winner/Draft。聚合过线但
        # 逐序列有害（CONFLICT）不再取得部署权——证据照写，执行权不发。
        if str(ep.relation) == "POSITIVE":
            if result.first_positive_support_receipt_index is None:
                result.first_positive_support_receipt_index = (
                    result.target_support_receipts_used)
            if _policy.winner_compare_rule == (
                    "max_support_gain_among_probed_positive"):
                # 只比已探 POSITIVE；margin 域与 ±0.005 双门线分立。
                # 挑战者需超出 best 达 margin 以上才置换；|Δ|≤margin 视为
                # 打平，交 tie_break_rule（probe_order = 保持先到者）。
                if _best_positive is None:
                    _best_positive = (str(cand), tuple(steps), gain)
                else:
                    _delta = gain - _best_positive[2]
                    if _delta > _policy.displacement_margin:
                        _best_positive = (str(cand), tuple(steps), gain)
                    elif abs(_delta) <= _policy.displacement_margin:
                        _tie = _policy.tie_break_rule
                        _best_is_supplied = _is_supplied(_best_positive[0])
                        if (_tie == "prefer_self_proposed"
                                and _best_is_supplied
                                and not _is_supplied(cand)):
                            _best_positive = (str(cand), tuple(steps), gain)
                        elif (_tie == "prefer_supplied"
                                and not _best_is_supplied
                                and _is_supplied(cand)):
                            _best_positive = (str(cand), tuple(steps), gain)
            elif result.winner_program is None:
                result.winner_program = [
                    {"op": o, "params": dict(p)} for o, p in steps]
                result._winner_candidate_id = str(cand)
                result._winner_steps = tuple(steps)
            if _policy.first_positive_stop:
                break
            continue
        if gain < -M:
            result.harm_count += 1
            result.harm_magnitude += -gain
            # 7. 第一个 material failure 触发一次 Slow Path
            if not triggered and allow_slow:
                if result.target_support_receipts_used >= budget:
                    # 9. 预算已用尽——不偷偷 replay
                    result._deferred_slow = (
                        "SLOW_UPDATE_DEFERRED_NO_TARGET_BUDGET")
                else:
                    triggered = True
                    # P2（rev3）：归因 → 单一授权 Surface → Slow 或显式
                    # ABSTAIN。默认 catalog 不再由调用方先验写死为 ADD。
                    from .program_supply import (  # noqa: PLC0415
                        build_single_surface_catalog,
                        route_online_program_supply_fault,
                    )

                    _decision = route_online_program_supply_fault(
                        trace, ep, _route_view)
                    _retrieved_capability_ids = [
                        str(skill.skill_id)
                        for skill in _route_view.skills
                        if (getattr(skill, "skill_kind", None)
                            .value == "capability"
                            and skill.skill_id
                            in tuple(trace.retrieved_skill_ids or ()))
                    ]
                    _authorized_catalog = build_single_surface_catalog(
                        decision=_decision,
                        parent=store.materialize(
                            method._active_snapshot()),
                        controller=controller,
                        retrieved_capability_skill_ids=(
                            _retrieved_capability_ids))
                    if not _authorized_catalog:
                        # 空 surface 集 = 路由层 ABSTAIN：不调 Slow，不
                        # 计入 slow_replay_receipts_used，不算 protocol
                        # error。宁可 ABSTAIN，不可猜 cause。
                        # 注意：这里不创建 sev，也不进入下面的 Slow replay
                        # 核销块——否则会是确定性 UnboundLocalError。
                        result._slow_event = {
                            "stage": "abstained_by_route",
                            "case_id": _decision.case_id,
                            "cause_code": _decision.cause_code,
                            "actionability": _decision.actionability,
                            "surface_templates": list(
                                _decision.surface_templates),
                        }
                    else:
                        # E0：Slow 调用透传合法 Operator contracts 与现有
                        # TaskContext（不再空传）。
                        _contracts = tuple(
                            public_operator_contract(op)
                            for op in sorted({
                                op for steps in steps_map.values()
                                for op, _p in steps}))
                        sev = method.handle_feedback_support(
                            ep, slow_agent=slow_agent, controller=controller,
                            store=store,
                            surface_catalog=_authorized_catalog,
                            card_builder=card_builder,
                            evaluator=lambda s, _mode: executor.evaluate(
                                tuple(s), origin),
                            fast_features=fast_features,
                            allowed_operator_contracts=_contracts,
                            confirmed_cause=_decision.cause_code,
                            task_context=getattr(request, "task_context", None))
                        result._slow_event = sev
                        result._trigger_episode_id = ep.episode_id
                        if sev.get("stage") in ("pending", "support_rejected"):
                            # 8. Slow replay 消耗同一 Target Support 预算
                            result.slow_replay_receipts_used += 1
                            result.target_support_receipts_used += 1
                            # E0：Slow replay 进入 probe/harm 轨迹（实际
                            # 读 outcome 的合法评估——计入 harm）。
                            rg = sev.get("support_gain")
                            result.actual_probed_programs.append({
                                "candidate_id": "replay:" + str(
                                    sev.get("patch_id") or "?"),
                                "kind": "slow_replay",
                                "gain": (float(rg) if rg is not None else None),
                                "passed": bool(sev.get("support_passed"))})
                            if rg is not None and float(rg) < -M:
                                result.harm_count += 1
                                result.harm_magnitude += -float(rg)
                            if sev.get("stage") == "pending":
                                result.pending_patch_id = sev.get("patch_id")
                                # E2.5-B（用户裁决）：Slow replay 成为
                                # 第一个正向 Workflow → 立即成为本轮
                                # winner 并停止继续探测（预算不浪费在
                                # 非正向候选上）。
                                result.winner_program = [
                                    {"op": o, "params": dict(p)}
                                    for o, p in _steps_of_patch(sev)]
                                result._winner_steps = _steps_of_patch(sev)
                                break
    # Stage 3 Part 0：max-gain 比较规则的收尾——从已探 POSITIVE 的最优者
    # 铸 winner。Slow replay 若已按 E2.5-B 取得 winner（pending），不覆盖。
    if _best_positive is not None and result.winner_program is None:
        _bp_cand, _bp_steps, _bp_gain = _best_positive
        result.winner_program = [
            {"op": o, "params": dict(p)} for o, p in _bp_steps]
        result._winner_candidate_id = _bp_cand
        result._winner_steps = tuple(_bp_steps)
    result.abstained = result.winner_program is None
    # GROUP_FAULT 自动触发（用户裁决 2026-08-12）：失败 Episode 积累 →
    # 轻量分组（≥group_min 同算子同 sign）→ 组级 Slow（方法层
    # handle_group_feedback——整组 Capsule → 组内 replay 全 ≥M → 组外
    # holdout 不劣 → pending）。每实例最多触发一次。
    # Wave 1（2026-08-13）：分组键 = 完整 workflow 指纹；默认组 Card 嵌入
    # Capsule；Operator contracts 与 TaskContext 透传（E0 同款）。
    # P4 does NOT use this legacy auto trigger: its batch-end Runner must use
    # methods/ttha/p4_runner.run_p4_group_update with a verifier-earned
    # VerifiedProgramSupplyAssessment.
    if (allow_group_slow and not result._group_slow_done
            and slow_agent is not None and controller is not None
            and store is not None
            and getattr(method, "_experience_episodes", None)):
        from .group_fault import (  # noqa: PLC0415
            build_contrast_capsule,
            group_first_faults,
        )
        _groups = group_first_faults(
            method._experience_episodes, min_group=group_min)
        if _groups:
            _g = _groups[0]
            _capsule = build_contrast_capsule(
                _g,
                all_episodes=method._experience_episodes,
                target_domain_namespace=str(result._domain))
            from .program_supply import (  # noqa: PLC0415
                build_single_surface_catalog,
                route_online_program_supply_fault,
            )

            _trigger_ep = (_g.get("episodes") or [None])[0]
            _group_decision = route_online_program_supply_fault(
                trace, _trigger_ep, _route_view)
            _group_catalog = build_single_surface_catalog(
                decision=_group_decision,
                parent=store.materialize(method._active_snapshot()),
                controller=controller)
            if not _group_catalog:
                result._group_slow_event = {
                    "stage": "abstained_by_route",
                    "case_id": _group_decision.case_id,
                    "cause_code": _group_decision.cause_code,
                    "actionability": _group_decision.actionability,
                }
            else:
                _contracts = tuple(
                    public_operator_contract(op)
                    for op in sorted({
                        op for steps in steps_map.values()
                        for op, _p in steps}))
                # #42k Part B3：缺省组卡的 task_kind 取本轮 request 的
                # task_spec.task_type——硬编码 "forecast" 会让异常检测轮次
                # 的组级 Slow 卡带着预测的可观察签名落库。
                _group_task_kind = str(request.task_spec.task_type)
                _gb = (group_card_builder if group_card_builder is not None
                       else lambda g, cap: {"pattern_id": "group-fault",
                                            "failure_family":
                                                "workflow_component_negative",
                                            "observable_signature":
                                                {"task_kind":
                                                 _group_task_kind},
                                            "workflow": {"steps": [
                                                {"op": str(g.get("workflow")),
                                                 "params": {}}]},
                                            "facts": {
                                                "contrast_capsule": cap}})
                _gev = method.handle_group_feedback(
                    _g, _capsule, slow_agent=slow_agent,
                    controller=controller, store=store,
                    card_builder=_gb,
                    evaluator_group=lambda s, e: executor.evaluate(
                        tuple(s), int(((getattr(e, "context_summary", {})
                                        or {}).get("support_origin") or 0))),
                    holdout_evaluator=(
                        (lambda s, _m: executor.evaluate(
                            tuple(s), group_holdout_origin))
                        if group_holdout_origin is not None else None),
                    fast_features=fast_features,
                    surface_catalog=_group_catalog,
                    route_decision=_group_decision,
                    allowed_operator_contracts=_contracts,
                    task_context=getattr(request, "task_context", None))
                result._group_slow_event = _gev
            result._group_slow_done = True
    # E2.5-B：Fast winner → Target-local Draft Skill（方法层
    # handle_fast_winner——Runtime 机器生成 manifest——Runner 不手工
    # write_skill）。审查修正（2026-08-12）：门只对 **Fast 探测 winner**
    # 生效——Slow replay 正向 winner 是慢路径 pending（其 Skill 由
    # handle_feedback_support 管理——不重复形成）。
    _winner_is_slow_replay = bool(
        result._slow_event is not None
        and result._slow_event.get("stage") == "pending"
        and result._winner_steps is not None
        and result._winner_steps == _steps_of_patch(result._slow_event))
    # P0（2026-08-15 评审裁定，UPDATE_POLICY_FAULT 修复）：winner 若来自
    # 已批准 Skill（cand_skill_* 候选），不再走 handle_fast_winner 重复 ADD
    # （会撞 surface ABSENT 前置 → apply_failed，且本轮 delayed 判决无人
    # 接收）；改为记录 deployed_skill_id，delayed 判决由 open_delayed
    # 路由给原 Skill（delayed < −M → revoke_deployed_skill 撤销）。
    _wcid = str(getattr(result, "_winner_candidate_id", None) or "")
    if result.winner_program is not None and _wcid.startswith("cand_skill_"):
        result.deployed_skill_id = _wcid[len("cand_skill_"):]
        result._fast_skill_event = {
            "stage": "deployed_existing_skill",
            "skill_id": result.deployed_skill_id}
    if (allow_fast_skill and result.winner_program is not None
            and not _wcid.startswith("cand_skill_")
            and not _winner_is_slow_replay
            and controller is not None and store is not None
            and result._episodes and result._winner_steps is not None):
        _winner_ep = result._episodes[-1][0]
        # 计量修正（用户裁决 2026-08-12）：传本轮探测已获的 winner gain
        # ——handle_fast_winner 直接复用（不重开相同 Context×Program 的
        # Support——不计预算的重复仪器评估）
        # winner 是停止点——最后一个 probe（正向）即 winner 的 gain
        _winner_gain = next(
            (p.get("gain") for p in reversed(result.actual_probed_programs)
             if p.get("kind") == "probe" and p.get("gain") is not None),
            None)
        ev = method.handle_fast_winner(
            _winner_ep, result._winner_steps,
            controller=controller, store=store,
            card=card_builder(_winner_ep),
            evaluator=lambda s, _mode: executor.evaluate(tuple(s), origin),
            fast_features=fast_features,
            support_gain=_winner_gain,
            confirmed_cause="SKILL_LIBRARY_GAP")
        result._fast_skill_event = ev
        if ev.get("stage") == "pending":
            result._fast_skill_episode_id = getattr(
                _winner_ep, "episode_id", None)
    return result


def revoke_deployed_skill(result: RoundResult, store: Any | None) -> bool:
    """P0（2026-08-15 评审裁定，UPDATE_POLICY_FAULT 修复）：撤销本轮部署的
    已批准 Skill——delayed < −M 时调用。

    store 存在时走 tree 往返（fork → 删除 skills/** 中该 skill 的 json →
    compile_snapshot 重算 sha → materialize → set_active），method 的活动
    快照同步为撤销后快照；不新增 EditOperation，不改 Memory/Prompt/
    Observation。store 缺失时不改快照，仅记录 revocation_pending（调用方
    负责持久化——不得假装已撤销）。
    """
    sid = result.deployed_skill_id
    method = result._method
    if sid is None or method is None:
        return False
    snap = method._active_snapshot()  # noqa: SLF001
    if not any(s.skill_id == sid for s in snap.skills):
        return False
    if store is None:
        result.revoked_skill_id = None
        result.revoked_runtime_bundle_sha = None
        result._fast_skill_event = {
            "stage": "revocation_pending_no_store",
            "skill_id": sid,
            "delayed_gain": result.delayed_utility}
        return False
    import json as _json  # noqa: PLC0415
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: PLC0415
        compile_snapshot)
    parent = store.materialize(snap)
    fork = store.fork(parent, edit_id=f"revoke_{sid}")
    try:
        target = None
        for sub in ("learned", "bootstrap"):
            directory = fork / "skills" / sub
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                try:
                    doc = _json.loads(path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(doc, dict) and str(doc.get("skill_id")) == sid:
                    target = path
                    break
            if target is not None:
                break
        if target is None:
            return False
        target.unlink()
        new_snap = compile_snapshot(fork, verify_lock=False)
        store.materialize(new_snap, parent_sha=snap.runtime_bundle_sha)
        store.set_active(new_snap.runtime_bundle_sha)
        method._snapshot = new_snap  # noqa: SLF001
        result.revoked_skill_id = sid
        result.revoked_runtime_bundle_sha = new_snap.runtime_bundle_sha
        result._fast_skill_event = {
            "stage": "revoked_delayed_harm",
            "skill_id": sid,
            "delayed_gain": result.delayed_utility}
        return True
    finally:
        store.discard_fork(fork)


def open_delayed(result: RoundResult, executor: Any, *,
                 delayed_origin: int | None = None,
                 horizon: int = 48,
                 store: Any | None = None) -> RoundResult:
    """delayed 到达后（时间边界）：更新所有 Episode 的 delayed 状态 +
    匹配 episode_id 的 handle_feedback_delayed（批准 → snapshot 更新；
    拒绝 → snapshot 不变）。"""
    d_origin = delayed_origin if delayed_origin is not None \
        else result.origin + horizon
    method = result._method
    values = result._values
    # 11/12. 先更新本轮的 Episode delayed 状态
    # E0：delayed gain None（verifier 失败/仪器失败）不得转为 0.0——
    # 保持"未评估"状态（不掩盖协议失败）。
    # E2.5-B（用户裁决）：只有实际部署的 winner 获得 delayed——未部署
    # 候选不能获得反事实 delayed 并污染 Memory。
    # 修正（2026-08-12 用户承重 5）：**无 winner 时全部不开 delayed**
    # （原条件在 _winner_steps=None 时恒真——失败 probe 全被打开——
    # 违反"delayed 只评价最终部署 winner"）。
    _winner_delayed_status = ""
    for ep, steps in result._episodes:
        if result._winner_steps is None:
            continue  # 本轮无部署——不打开任何 delayed
        if tuple(steps) != tuple(result._winner_steps):
            continue  # 非 winner 探测——不部署——不获得 delayed
        rd = executor.evaluate(tuple(steps), d_origin)
        dg = (float(rd.gain) if rd.gain is not None else None)
        if dg is None:
            continue  # 未评估——不更新 delayed 状态
        upd = _update_delayed_status(
            ep, dg, delayed_context=dict(window_context(
                values, d_origin, result._period)),
            per_view_gain=getattr(rd, "per_view_gain", None),
            series_uids=result._series_uids,
            consumer_id=result._consumer_id)
        method.update_experience_episode(upd)
        _winner_delayed_status = str(getattr(upd, "local_status", "") or "")
    # 两阶段批准（10）：pending 必须经 handle_feedback_delayed
    if result._slow_event is not None \
            and result._slow_event.get("stage") == "pending":
        dev = method.handle_feedback_delayed(
            lambda s, _mode: executor.evaluate(tuple(s), d_origin),
            episode_id=result._trigger_episode_id)
        result._delayed_event = dev
        if dev.get("stage") == "approved":
            result.approved_skill_id = result._slow_event.get("edit_id")
    # E2.5-B：Fast winner 的 Draft Skill pending——同两阶段批准
    if result._fast_skill_event is not None \
            and result._fast_skill_event.get("stage") == "pending":
        dev = method.handle_feedback_delayed(
            lambda s, _mode: executor.evaluate(tuple(s), d_origin),
            episode_id=result._fast_skill_episode_id)
        result._delayed_event = dev
        if dev.get("stage") == "approved":
            result.approved_skill_id = (
                result._fast_skill_event.get("edit_id"))
    # GROUP_FAULT：组级 pending——同两阶段批准（delayed 窗口）
    if result._group_slow_event is not None \
            and result._group_slow_event.get("stage") == "pending":
        _gid = result._group_slow_event.get("episode_id")
        dev = method.handle_feedback_delayed(
            lambda s, _mode: executor.evaluate(tuple(s), d_origin),
            episode_id=_gid)
        result._delayed_event = dev
        if dev.get("stage") == "approved":
            result.approved_skill_id = (
                result._group_slow_event.get("edit_id"))
    # delayed_utility = 实际 winner 的 delayed gain
    if result._winner_steps is not None:
        wd = executor.evaluate(result._winner_steps, d_origin)
        result.delayed_utility = (
            float(wd.gain) if wd.gain is not None else None)
    # P0（2026-08-15 评审裁定）：部署已有 Skill 的轮次，delayed < −M 必须
    # 撤销该 Skill——下一正常入口不再供应（DELAYED_HARM_NOT_REVOKING_
    # RETRIEVED_SKILL 修复）。
    if (result.deployed_skill_id is not None
            and result.delayed_utility is not None
            and float(result.delayed_utility) < -M):
        revoke_deployed_skill(result, store)
    # W-1 同权：winner 来自已批准/已供给 Skill（cand_skill_*）时走上面的
    # deployed_existing_skill 路由——而该路由此前**没有 delayed 裁决出口**，
    # 于是 approved_skill_id 永远为 None，而每一条 ledger incumbent 规则
    # 都以它为准。结果是供给候选比 agent 自提 winner 权利**更少**：过了
    # Support、过了 delayed，仍然无法部署。PS-2 run9/run12 即此形：
    # Support +0.636/+0.600 POSITIVE、delayed +0.30、Episode LOCAL_ACTIVE，
    # 部署却回落 identity。
    #
    # 这里不新造判据：批准与否直接读 winner Episode 刚刚拿到的 delayed
    # 分级（_update_delayed_status 的既有三档映射，LOCAL_ACTIVE = delayed
    # 确认）。CONFLICT→RESTRICTED、NEGATIVE/NEUTRAL→EPISODE_ONLY 一律不
    # 批准，撤销过的更不批准。
    if (result.deployed_skill_id is not None
            and result.revoked_skill_id is None
            and result.approved_skill_id is None
            and _winner_delayed_status == "LOCAL_ACTIVE"):
        result.approved_skill_id = result.deployed_skill_id
        result._delayed_event = {
            "stage": "approved",
            "skill_id": result.deployed_skill_id,
            "route": "deployed_existing_skill",
            "delayed_gain": result.delayed_utility}
    return result


def activate_approved(result: RoundResult, store: Any) -> bool:
    """P2 语义 13：只有批准后的 snapshot 才经 HarnessStore.set_active()
    更新正常入口。返回是否激活。"""
    if result._delayed_event is None \
            or result._delayed_event.get("stage") != "approved":
        return False
    snap = result._method._active_snapshot()  # noqa: SLF001
    if getattr(snap, "runtime_bundle_sha", None) is None:
        return False
    store.set_active(snap.runtime_bundle_sha)
    return True


def current_status(store: Any, method: Any, *,
                   last_round: Any = None,
                   last_delayed: Any = None) -> dict[str, Any]:
    """P5：最小运行持久化视图——不新增数据库/Ledger。
    E0：状态分类修正——restricted 只含 SAFETY kind；bootstrap procedure
    单独列出（常驻引导，非"受限"）。"""
    snap = method._active_snapshot()  # noqa: SLF001
    skills = list(getattr(snap, "skills", ()) or ())
    active: list[str] = []
    draft: list[str] = []
    restricted: list[str] = []
    bootstrap: list[str] = []
    for s in skills:
        kind = str(getattr(getattr(s, "skill_kind", None), "value", ""))
        guards = dict(s.risk_guards or {})
        if kind == "capability":
            (draft if guards.get("requires_target_support") is True
             else active).append(s.skill_id)
        elif kind == "safety":
            restricted.append(s.skill_id)
        else:
            bootstrap.append(s.skill_id)
    pending = getattr(method, "_pending_update", None)
    return {
        "active_snapshot": {
            "runtime_bundle_sha": getattr(snap, "runtime_bundle_sha", None),
            "harness_content_sha": getattr(snap, "harness_content_sha", None),
        },
        "episodes_count": len(getattr(method, "_experience_episodes", ())),
        "pending_patch": bool(pending),
        "last_round": last_round,
        "last_delayed": last_delayed,
        "draft_skills": draft,
        "active_skills": active,
        "restricted_skills": restricted,
        "bootstrap_skills": bootstrap,
    }


def _op_of(cand: str, steps: Sequence[tuple[str, Mapping[str, object]]]
           ) -> str:
    if cand.startswith("cand_skill_"):
        return cand[len("cand_skill_"):]
    if steps:
        return str(steps[0][0])
    return cand


def _steps_of_patch(sev: Mapping[str, Any]) -> tuple[tuple[str, dict], ...]:
    """从 slow_event 的 frozen_program 提取可执行 steps（E2.5-B：Slow
    replay 正向成为本轮 winner 用——精确使用冻结程序）。"""
    frozen = sev.get("frozen_program") or []
    return tuple(
        (str(st["op"]), dict(st.get("params") or {}))
        for st in frozen if isinstance(st, Mapping) and st.get("op"))


def np_values(request: Any, values: Mapping[str, Any]):
    """轮次数据 = request.values（PreparationRequest 第二参——已截断的
    序列切片）；values 映射仅用于 window_context（部署可见 cohort 口径）。"""
    import numpy as np  # noqa: PLC0415
    return np.asarray(request.values, dtype=np.float64)


__all__ = ["RoundResult", "run_online_round", "open_delayed",
           "activate_approved", "current_status"]
