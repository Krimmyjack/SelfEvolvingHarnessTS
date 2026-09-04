"""Signed relation-aware radius resolver（方法层单一来源，2026-08-08）。

从 evaluation/functional/run_v1_signed_radius.py 迁移（实现单一来源），
并合并 window_context（部署可见 recent/change 特征，原 run_v1_target_local_radius_premise）。

冻结规则（用户裁决）：
  对每个 Workflow 分别计算 P = 半径内正向 Context 证据、R = 半径内负向/冲突 Context 证据：
    P 有、R 无 → POSITIVE_PRIOR，提前探测
    P 无、R 有 → RISK_PRIOR，降级但不硬排除
    P 有、R 有 → CONFLICT，放在未知候选之后
    P 无、R 无 → UNKNOWN，使用默认探索位置
  负向经验绝不从相似性计算中扔掉——否则 Memory 没有真正"吸取教训"。

WEAK_HISTORY（历史 Context < MIN，2026-08-08 审查修正）：
  按同一个 Episode 的成对证据判定，绝不把不同 Episode 的单侧证据拼成"双正"：
    双正 → POSITIVE_PRIOR；双负 → RISK_PRIOR；一正一负 → CONFLICT；
    一侧有效、另一侧中性或缺失 → UNKNOWN。

半径语义：
  - Context = 部署可见 recent/change 特征（window_context，compare_history_windows）
  - 历史 Context 池 = 当前记忆全部 Episode 的 support/delayed Context（去重）；
    query 不参与校准。
  - 尺度：历史 mean/std 冻结 z-score；常量特征不删除——查询偏离历史常量 → 半径外。
  - δ = 历史 Context 留一最近邻距离的固定分位数 q75（与查询距离同一标准化尺度）。

排序（确定性）：POSITIVE_PRIOR → UNKNOWN（字母序）→ CONFLICT（字母序）→ RISK_PRIOR（字母序）。
按 Workflow 聚合：同一 Workflow 每轮只出现一次。无适用证据 → 完整默认顺序（字母序）。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from SelfEvolvingHarnessTS.methods.ttha.experience_memory import VALIDITY_VALID
from SelfEvolvingHarnessTS.methods.ttha.public_tools import CohortHistoryPublicToolGateway

MATERIAL_THRESHOLD = 0.005  # 与 evaluation run_v1_fastpath.MATERIAL_THRESHOLD 一致
CONTEXT_PREFIXES = ("recent.", "change.")  # window_context 输出键前缀
DELTA_QUANTILE = 0.75
MIN_HISTORICAL_CONTEXTS = 3
WINDOW_LENGTH = 192  # 与 _evaluate eval context 一致

POSITIVE_PRIOR = "POSITIVE_PRIOR"
RISK_PRIOR = "RISK_PRIOR"
CONFLICT = "CONFLICT"
UNKNOWN = "UNKNOWN"


def window_context(values: Mapping[str, np.ndarray], origin: int, period: int) -> dict[str, float]:
    """部署可见 recent/change 特征：series 截断在 origin，recent = [origin-192, origin)。

    单一来源（迁移自 run_v1_target_local_radius_premise）。单序列可传
    {uid: array} 包装后调用（cohort of 1），键与 Cohort 版一致。
    """
    gate = CohortHistoryPublicToolGateway(
        [np.asarray(v)[:origin] for v in values.values()],
        calendar_period=period,
        window_length=WINDOW_LENGTH,
    )
    result = gate.call("compare_history_windows", {}).public_result
    recent = result["recent"]
    changes = result["early_to_recent_change"]
    vec: dict[str, float] = {}
    for key, val in recent.items():
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            vec[f"recent.{key}"] = float(val)
    for key, val in changes.items():
        if val is not None and isinstance(val, (int, float)) and math.isfinite(float(val)):
            vec[f"change.{key}"] = float(val)
    return vec


def episode_evidence(
    episode: Any,
    material_threshold: float,
) -> list[dict[str, Any]]:
    """单个 Episode → 证据对 [(context, gain, kind), ...]（support/delayed 各一）。

    0/0 与噪声（|gain| < threshold）不计入证据——中性不是成功也不是失败。
    context 仅取 window_context 特征（recent./change. 前缀），排除 support_gain 等辅助键。
    """
    out: list[dict[str, Any]] = []
    lp = episode.context_summary.get("local_pattern") or {}
    sg = episode.support_response.get("gain")
    if sg is not None and isinstance(sg, (int, float)):
        ctx = {k: float(v) for k, v in lp.items() if str(k).startswith(CONTEXT_PREFIXES)}
        if ctx:
            out.append({"context": ctx, "gain": float(sg), "kind": "support",
                        "episode_id": episode.episode_id})
    dp = episode.context_summary.get("delayed_pattern") or {}
    dg = episode.delayed_response.get("gain")
    if dg is not None and isinstance(dg, (int, float)):
        ctx = {k: float(v) for k, v in dp.items() if str(k).startswith(CONTEXT_PREFIXES)}
        if ctx:
            out.append({"context": ctx, "gain": float(dg), "kind": "delayed",
                        "episode_id": episode.episode_id})
    return out


def _scale(hist_vecs: Sequence[Mapping[str, float]]) -> tuple[list[str], dict[str, float], Any]:
    """历史冻结尺度。常量特征（历史 std≈0）保留为硬门：查询偏离 → 半径外。"""
    feats = sorted(set().union(*(set(h) for h in hist_vecs)))
    means = {f: float(np.mean([h[f] for h in hist_vecs])) for f in feats}
    stds = {f: float(np.std([h[f] for h in hist_vecs])) for f in feats}
    informative = [f for f in feats if stds[f] > 1e-12]
    const_vals = {f: means[f] for f in feats if stds[f] <= 1e-12}

    def z(vec: Mapping[str, float]) -> dict[str, float]:
        return {f: (float(vec[f]) - means[f]) / stds[f] for f in informative}

    return informative, const_vals, z


def _in_radius(
    query_context: Mapping[str, float],
    evidence: Sequence[dict[str, Any]],
    hist_vecs: Sequence[Mapping[str, float]],
    delta: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """证据 in/out 判定。delta=None（弱参考）→ 全部 in。

    返回 (in_evidence, meta)：meta 含 query 常量偏离与距离明细。
    """
    if delta is None:
        return list(evidence), {"query_const_deviations": {}, "distances": {}}
    informative, const_vals, z = _scale(hist_vecs)
    required = informative + list(const_vals)
    missing = [f for f in required if f not in query_context]
    if missing:
        return [], {"query_const_deviations": {}, "distances": {},
                    "missing_features": missing}
    deviations = {
        f: float(query_context[f] - const_vals[f])
        for f in const_vals
        if abs(float(query_context[f] - const_vals[f])) > 1e-12
    }
    if deviations:
        return [], {"query_const_deviations": deviations, "distances": {}}
    qz = z(query_context)
    hz = [z(h) for h in hist_vecs]
    distances: dict[str, float] = {}
    inside: list[dict[str, Any]] = []
    for ev in evidence:
        ez = z(ev["context"])
        d = float(sum(abs(qz[f] - ez[f]) for f in informative))
        distances[f"{ev['episode_id']}:{ev['kind']}"] = d  # support/delayed 不互相覆盖
        if d <= delta:
            inside.append(ev)
    return inside, {"query_const_deviations": {}, "distances": distances}


def _paired_weak_verdict(
    evs: Sequence[dict[str, Any]],
    material_threshold: float,
) -> tuple[str, list[dict[str, Any]]]:
    """WEAK_HISTORY（历史 Context < MIN）：按同一个 Episode 的成对证据判定。

    审查裁决（2026-08-08）：
      - 绝不能把不同 Episode 的单侧证据拼成"双正"——按内部记录索引（episode_idx）
        分组，episode_id 可能跨轮重复（如 {domain}_target_{op}），不能作分组键。
      - 聚合规则（审查修正）：
          存在 CONFLICT，或同时存在 POSITIVE 与 RISK      → CONFLICT
          只有 POSITIVE                                    → POSITIVE_PRIOR
          只有 RISK                                        → RISK_PRIOR
          其余                                             → UNKNOWN
      - 单 Episode 判定：双正 → POSITIVE_PRIOR；双负 → RISK_PRIOR；
          一正一负 → CONFLICT；一侧有效、另一侧中性或缺失 → UNKNOWN。
    """
    m = material_threshold
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for ev in evs:
        by_ep.setdefault(int(ev["episode_idx"]), []).append(ev)
    states: list[dict[str, Any]] = []
    for idx, ev_list in sorted(by_ep.items()):
        kinds = {ev["kind"] for ev in ev_list}
        if not ({"support", "delayed"} <= kinds):
            state = UNKNOWN  # 单侧有效、另一侧中性或缺失（不成对）
        else:
            g = {ev["kind"]: ev["gain"] for ev in ev_list}
            sg, dg = g["support"], g["delayed"]
            if sg >= m and dg >= m:
                state = POSITIVE_PRIOR
            elif sg <= -m and dg <= -m:
                state = RISK_PRIOR
            elif (sg >= m and dg <= -m) or (sg <= -m and dg >= m):
                state = CONFLICT
            else:
                state = UNKNOWN
        states.append({
            "episode_idx": idx,
            "episode_id": next((ev["episode_id"] for ev in ev_list), None),
            "verdict": state,
            "support_gain": next((ev["gain"] for ev in ev_list if ev["kind"] == "support"), None),
            "delayed_gain": next((ev["gain"] for ev in ev_list if ev["kind"] == "delayed"), None),
        })
    has_pos = any(s["verdict"] == POSITIVE_PRIOR for s in states)
    has_risk = any(s["verdict"] == RISK_PRIOR for s in states)
    has_conf = any(s["verdict"] == CONFLICT for s in states)
    if has_conf or (has_pos and has_risk):
        best = CONFLICT
    elif has_pos:
        best = POSITIVE_PRIOR
    elif has_risk:
        best = RISK_PRIOR
    else:
        best = UNKNOWN
    return best, states


def _legality_filter(
    episode: Any,
    *,
    task_consumer_key: str | None,
    allowed_operators: Sequence[str] | None,
    pattern_view: str | None,
) -> bool:
    """合法性过滤（对齐 SignedEpisodeRetriever._hard_filter，审查 2026-08-08）。

    不合法 Episode 绝不进入历史池（否则其 Context 会污染标准化尺度与 δ）。
    """
    if getattr(episode, "response_validity", None) != VALIDITY_VALID:
        return False  # 仪器故障（fit crash/compile failure/metric 无效）不参与检索
    if task_consumer_key is not None and getattr(episode, "task_consumer_key", None) != task_consumer_key:
        return False
    if pattern_view is not None and getattr(episode, "pattern_view", None) != pattern_view:
        return False
    if allowed_operators:
        ops = tuple(str(episode.workflow_signature).split("|"))
        informative = [op for op in ops if op and op not in ("identity", "unknown")]
        if not informative:
            return False
        if not any(op in allowed_operators for op in informative):
            return False
    return True


def resolve_order(
    *,
    query_context: Mapping[str, float],
    episodes: Sequence[Any],
    operators: Sequence[str],
    material_threshold: float,
    delta_quantile: float = DELTA_QUANTILE,
    min_historical: int = MIN_HISTORICAL_CONTEXTS,
    task_consumer_key: str | None = None,
    allowed_operators: Sequence[str] | None = None,
    pattern_view: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """signed relation-aware 半径排序。返回 (order, details)。

    details = {op: {"verdict", "pos_evidence"/"neg_evidence" 或 "episode_states",
                    "meta": {...}}}——radius mode 下证据含 episode_id / gain / kind / distance；
    WEAK_HISTORY 下为成对 episode_states。
    可选合法性过滤（task_consumer_key / allowed_operators / pattern_view）：
    不合法 Episode 排除（不进入历史池、不提供证据）。
    """
    # 0) 合法性过滤（审查 2026-08-08：非法 Episode 的 Context 不得污染历史池/尺度/δ）
    legal_episodes = [
        ep for ep in episodes
        if _legality_filter(ep, task_consumer_key=task_consumer_key,
                            allowed_operators=allowed_operators,
                            pattern_view=pattern_view)
    ]
    # 1) per-Workflow 证据收集（按 Workflow 聚合：每 op 一次；episode_idx = 内部
    #    记录索引——episode_id 可能跨轮重复，不能作分组键）
    evidence_by_op: dict[str, list[dict[str, Any]]] = {}
    for idx, ep in enumerate(legal_episodes):
        op = ep.workflow_signature
        if op == "identity":  # abstain/rejection episode 无信息量
            continue
        for ev in episode_evidence(ep, material_threshold):
            ev["episode_idx"] = idx
            evidence_by_op.setdefault(op, []).append(ev)

    # 2) 历史 Context 池（去重，全部可见 Context 参与尺度/δ 校准；query 不参与）
    hist_ctxs: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for evs in evidence_by_op.values():
        for ev in evs:
            key = tuple(sorted((k, round(float(v), 8)) for k, v in ev["context"].items()))
            if key not in seen:
                seen.add(key)
                hist_ctxs.append(ev["context"])
    n_hist = len(hist_ctxs)
    delta: float | None = None
    if n_hist >= min_historical:
        informative, _const, z = _scale(hist_ctxs)
        hz = [z(h) for h in hist_ctxs]
        loo = [
            min(
                sum(abs(a[f] - b[f]) for f in informative)
                for j, b in enumerate(hz) if j != i
            )
            for i, a in enumerate(hz)
        ]
        delta = float(np.quantile(loo, delta_quantile))

    # 3) signed 判定 per-op（WEAK_HISTORY 成对 / radius 证据级）
    details: dict[str, Any] = {}
    pos_prior: list[tuple[str, float]] = []
    for op in operators:
        evs = evidence_by_op.get(op, [])
        if delta is None:
            verdict, states = _paired_weak_verdict(evs, material_threshold)
            details[op] = {
                "verdict": verdict,
                "episode_states": states,
                "meta": {"radius_mode": "weak_reference"},
            }
            if verdict == POSITIVE_PRIOR:
                pp_keys = [
                    min(st["support_gain"], st["delayed_gain"])
                    for st in states if st["verdict"] == POSITIVE_PRIOR
                ]
                pos_prior.append((op, min(pp_keys)))
            continue
        inside, meta = _in_radius(query_context, evs, hist_ctxs, delta)
        pos_ev = [ev for ev in inside
                  if ev["gain"] >= material_threshold]
        neg_ev = [ev for ev in inside
                  if ev["gain"] <= -material_threshold]
        if pos_ev and not neg_ev:
            verdict = POSITIVE_PRIOR
            key = min(ev["gain"] for ev in pos_ev)
            pos_prior.append((op, key))
        elif not pos_ev and neg_ev:
            verdict = RISK_PRIOR
        elif pos_ev and neg_ev:
            verdict = CONFLICT
        else:
            verdict = UNKNOWN
        details[op] = {
            "verdict": verdict,
            "pos_evidence": [
                {"episode_id": ev["episode_id"], "gain": round(ev["gain"], 6),
                 "kind": ev["kind"], "distance": meta["distances"].get(f"{ev['episode_id']}:{ev['kind']}")}
                for ev in pos_ev
            ],
            "neg_evidence": [
                {"episode_id": ev["episode_id"], "gain": round(ev["gain"], 6),
                 "kind": ev["kind"], "distance": meta["distances"].get(f"{ev['episode_id']}:{ev['kind']}")}
                for ev in neg_ev
            ],
            "meta": meta,
        }

    # 4) 排序：POSITIVE_PRIOR（半径内正向证据最小 gain 降序）→ UNKNOWN → CONFLICT → RISK
    order = [op for op, _ in sorted(pos_prior, key=lambda t: -t[1])]
    order += [op for op in sorted(operators) if details[op]["verdict"] == UNKNOWN]
    order += [op for op in sorted(operators) if details[op]["verdict"] == CONFLICT]
    order += [op for op in sorted(operators) if details[op]["verdict"] == RISK_PRIOR]
    summary = {
        "n_historical_contexts": n_hist,
        "radius_mode": "radius" if delta is not None else "weak_reference",
        "delta_q75": delta,
        "verdict_counts": {v: sum(1 for d in details.values() if d["verdict"] == v)
                           for v in (POSITIVE_PRIOR, UNKNOWN, CONFLICT, RISK_PRIOR)},
    }
    return order, {"summary": summary, "per_op": details}


def render_signed_instruction(
    details: Mapping[str, Any],
    order: Sequence[str],
    executable_ops: Sequence[str] | None = None,
) -> str:
    """把 signed 判定渲染成 TIMECLAW 风格 fenced 前缀块（LLM 可见 instruction）。

    executable_ops（审查裁决 2026-08-08 方案 2）：候选供给与 verifier 对齐——
    非可执行算子（构造默认候选必然被 verifier 拒绝，如修改分数超限的全局算子）
    不渲染为"建议优先探测"的 Reference 1（Memory 可保留 Episode，但不作可执行
    参考；降级为非行动参考）。None = 不过滤（实验层语义）。

    对齐 render_experience_pack 的设计约束：fenced 前缀、任务指令之前、
    祈使指令、**不含 gain 数值**（TIMECLAW 消融：GT 答案导致 answer-anchoring）。

    审查修正（2026-08-08）：radius mode 的 POSITIVE_PRIOR 只保证"匹配 Context 内
    存在正向证据"——负向证据可能因超半径而不在匹配范围（如 denoise_savgol：
    Support 负、delayed 正，负 Context 在半径外）。渲染绝不笼统声称"Support and
    delayed segments both positive"（那只有 WEAK_HISTORY 成对判定才保证），也不
    声称固定 Support→delayed 翻转（radius 证据可能来自不同 Episode 的不同窗口）。
    """
    per = details["per_op"]
    executable = set(executable_ops) if executable_ops is not None else None
    prior = [op for op in order if per[op]["verdict"] == POSITIVE_PRIOR
             and (executable is None or op in executable)]
    conflict = [op for op in order if per[op]["verdict"] == CONFLICT]
    risk = [op for op in order if per[op]["verdict"] == RISK_PRIOR]

    def _mode() -> str:
        """全局 radius_mode：delta 是全局判定的（n_hist < 阈值 → weak_reference）。
        从任一已判定的 op 读取；无判定则默认 radius 措辞（历史行为）。"""
        for op in (*prior, *conflict, *risk):
            meta = per[op].get("meta") or {}
            mode = meta.get("radius_mode")
            if mode is not None:
                return str(mode)
        return "radius"

    weak = _mode() == "weak_reference"
    # 审查修正（2026-08-08 十四）：weak_reference 模式下（n_hist < min_historical、
    # delta=None）**没有发生 Context 匹配**——渲染不得声称 "similarity radius /
    # matched context"（误导性描述）。radius 模式保留原措辞。
    if weak:
        ref1_tail = (
            "carried paired positive Support/delayed evidence from the same "
            "historical episode (weak reference: context matching not yet "
            "calibrated, fewer than the minimum historical contexts). Probe "
            "them first, then confirm again on the current Support."
        )
        ref2_tail = (
            "showed mixed evidence in historical experience (both positive and "
            "negative Support/delayed windows within the same episode, or "
            "conflicting results across episodes; weak reference: no context "
            "matching). Treat as risk; confirm on the delayed segment before "
            "relying on them."
        )
        ref3_tail = (
            "carried only negative evidence in historical experience (harmful "
            "Support/delayed windows; weak reference: no context matching). "
            "Avoid them unless current evidence contradicts."
        )
    else:
        ref1_tail = (
            "carried positive evidence inside the matched context (positive "
            "Support/delayed windows within the similarity radius; known "
            "negative evidence, if any, fell outside the matched range). Probe "
            "them first, then confirm again on the current Support."
        )
        ref2_tail = (
            "showed mixed evidence inside the matched context (both positive "
            "and negative Support/delayed windows within the similarity radius, "
            "or conflicting results across episodes). Treat as risk; confirm on "
            "the delayed segment before relying on them."
        )
        ref3_tail = (
            "carried only negative evidence inside the matched context (harmful "
            "Support/delayed windows within the similarity radius). Avoid them "
            "unless current evidence contradicts."
        )

    entries: list[str] = []
    if prior:
        entries.append(f"Reference 1: candidate operators {prior} {ref1_tail}")
    if conflict:
        entries.append(f"Reference 2: candidate operators {conflict} {ref2_tail}")
    if risk:
        entries.append(f"Reference 3: candidate operators {risk} {ref3_tail}")
    if not entries:
        return ""
    body = "\n\n".join(entries)
    return (
        # 跨域渲染措辞（裁决 2026-08-08）：Source 经验可能来自其他数据集，
        # 不能错误描述为"target-local"（如 GEFCom Source → NOAA Target）。
        "The following references come from verified historical experience.\n"
        "Use them as a guide.\n\n" + body + "\n\n"
    )


__all__ = [
    "CONFLICT",
    "CONTEXT_PREFIXES",
    "DELTA_QUANTILE",
    "MATERIAL_THRESHOLD",
    "MIN_HISTORICAL_CONTEXTS",
    "POSITIVE_PRIOR",
    "RISK_PRIOR",
    "UNKNOWN",
    "WINDOW_LENGTH",
    "episode_evidence",
    "render_signed_instruction",
    "resolve_order",
    "window_context",
]
