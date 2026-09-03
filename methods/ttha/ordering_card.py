"""Domain Ordering Card —— Memory/Control 面的确定性派生视图（E1，2026-08-16）。

**这不是 Scope/Program Card。** 它只做一件事：把已发生的 Episode 计数聚合成一个
持久工件，用来**重排 Fast 已经供应的合法候选**。它：

- 不注入新候选（body 不含 ``Frozen program steps:`` marker → ``_parse_frozen_steps``
  返回 None → ``_skill_frozen_candidates`` 不会为它产生 Candidate）；
- 不改 Program Supply；
- 不做 suppression / 禁用（第一版只做排序，否则改进无法归因）；
- 不能直接执行——实际收益仍由 Target Support 判定，去留仍由 delayed 判定。

设计依据（2026-08-16 本地评审 + 代码核实）：

- ``retrieval.resolve_harness_view`` **不读 risk_guards**（risk_guards 只在序列化处出现），
  所以 scope 不能只写在 risk_guards 里就指望检索层过滤。本模块的做法是双层：
  1. ``observable_applicability`` 用合法叶子谓词 ``task_kind == <task>`` 门住任务（检索层生效）；
  2. Runtime 再用 :func:`card_scope_matches` 机械精确匹配 domain / downstream_model_class /
     program_family（检索层管不到的部分）。
- ``skill-entry/1`` 字段集封闭（8 个字段，``_require_exact_fields`` + ``_reject_forbidden_fields``），
  且 ``load_learned_skill_entry`` 强制 ``skill_kind=capability``——因此结构化内容只能进
  ``risk_guards``（自由 JSON）与 ``body``（人类可读）。
- ``downstream_model_class`` 第一版写**精确** consumer（如 ``ridge_alpha1``），
  **不**提前推广为 ``parametric_forecaster``。

计数口径（评审第 4 条）：

- ``legal_opportunities`` —— 只用于 coverage 展示，**不作分母**；
- ``evaluated_attempts`` —— E_gain / E_harm 的**唯一分母**；
- 合法但未探测 = UNKNOWN，**不按零收益处理**（否则排序会奖励"很少被试、恰好成功过一次"的算子）。

lambda 口径（评审第 5 条）：在 development/source 数据上**预先固定**，STATIC/A3/A5 三臂
完全相同；唯一变量是 Source Episode 是否用于初始化排序。本模块只消费传入的 lam，不拟合。
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

CARD_KIND = "ordering-control/1"
_MATERIAL_DEFAULT = 0.005
_SCOPE_KEYS = ("task", "domain", "downstream_model_class", "program_family")


def empty_evidence(operators: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        str(op): {
            "legal_opportunities": 0,
            "evaluated_attempts": 0,
            "positive": 0,
            "negative": 0,
            "conflict": 0,
            "gain_sum": 0.0,
            "harm_magnitude_sum": 0.0,
        }
        for op in operators
    }


def accumulate(
    evidence: dict[str, dict[str, float]],
    observations: Iterable[tuple[str, float | None]],
    *,
    material_threshold: float = _MATERIAL_DEFAULT,
) -> dict[str, dict[str, float]]:
    """把一批 (operator, support_gain) 记进 evidence。

    ``gain is None`` 表示该算子本轮**合法但未被探测**——只增 ``legal_opportunities``，
    不进任何均值分母（UNKNOWN，不是 0）。
    """
    for op, gain in observations:
        row = evidence.get(str(op))
        if row is None:
            continue
        row["legal_opportunities"] += 1
        if gain is None:
            continue
        row["evaluated_attempts"] += 1
        value = float(gain)
        row["gain_sum"] += value
        if value > 0:
            row["positive"] += 1
        elif value < -material_threshold:
            row["negative"] += 1
            row["harm_magnitude_sum"] += -value
        else:
            row["conflict"] += 1
    return evidence


def merge_evidence(
    source_evidence: Mapping[str, Mapping[str, float]] | None,
    target_evidence: Mapping[str, Mapping[str, float]],
    *,
    source_pseudo_count: int,
) -> dict[str, dict[str, float]]:
    """把 Source 与 Target 两块**分开保存**的证据合成排序用的打分证据。

    评审第 7 条（2026-08-16）：``Source 和 Target 计数必须分开，不能直接合池，
    否则 Source 样本量会长期压住 Target 反馈``。做法——Source 只以
    ``source_pseudo_count`` 条**伪观测**入场（保留其均值，丢弃其样本量），
    Target 以**真实计数**累加。于是 Target 每积累 1 条反馈就真实地稀释一次
    Source 先验；约 ``source_pseudo_count`` 条之后 Target 完全占优。

    两块原始计数不被本函数改写——调用方仍分别持有 source/target 两份，
    并原样写进卡的 ``risk_guards.evidence_blocks``（可审计）。
    """
    ops = sorted({*(source_evidence or {}), *target_evidence})
    out = empty_evidence(ops)
    cap = float(source_pseudo_count)
    for op in ops:
        row = out[op]
        src = (source_evidence or {}).get(op)
        if src is not None:
            n_src = float(src.get("evaluated_attempts", 0))
            if n_src > 0:
                row["evaluated_attempts"] += cap
                row["gain_sum"] += (src["gain_sum"] / n_src) * cap
                row["harm_magnitude_sum"] += (
                    src["harm_magnitude_sum"] / n_src) * cap
            row["legal_opportunities"] += float(
                src.get("legal_opportunities", 0))
        tgt = target_evidence.get(op)
        if tgt is not None:
            for key in ("legal_opportunities", "evaluated_attempts", "positive",
                        "negative", "conflict", "gain_sum",
                        "harm_magnitude_sum"):
                row[key] += float(tgt.get(key, 0))
    return out


def _stats(row: Mapping[str, float]) -> tuple[float, float] | None:
    n = float(row.get("evaluated_attempts", 0))
    if n <= 0:
        return None                     # UNKNOWN —— 不参与打分
    return row["gain_sum"] / n, row["harm_magnitude_sum"] / n


def rank_operators(
    evidence: Mapping[str, Mapping[str, float]],
    *,
    lam: float,
    tie_break: Sequence[str],
) -> list[str]:
    """已探测过的算子按 ``E_gain - lam * E_harm`` 降序；未探测过的（UNKNOWN）
    按 ``tie_break`` 顺序排在**已知算子之后**——不假设它们是 0 收益，也不假设它们更好。
    """
    order_hint = {op: i for i, op in enumerate(tie_break)}
    known: list[tuple[float, int, str]] = []
    unknown: list[tuple[int, str]] = []
    for op in tie_break:
        row = evidence.get(op)
        st = _stats(row) if row is not None else None
        if st is None:
            unknown.append((order_hint[op], op))
        else:
            e_gain, e_harm = st
            known.append((-(e_gain - lam * e_harm), order_hint[op], op))
    known.sort()
    unknown.sort()
    return [op for _, _, op in known] + [op for _, op in unknown]


def build_ordering_card(
    *,
    skill_id: str,
    scope: Mapping[str, str],
    evidence: Mapping[str, Mapping[str, float]],
    lam: float,
    tie_break: Sequence[str],
    revision: int = 1,
    evidence_blocks: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
) -> dict[str, Any]:
    """确定性生成一张 ``skill-entry/1``。同输入必得同输出，无 LLM 参与。"""
    missing = [k for k in _SCOPE_KEYS if k not in scope]
    if missing:
        raise ValueError(f"ordering card scope missing keys: {missing}")
    order = rank_operators(evidence, lam=lam, tie_break=tie_break)
    body = (
        "Domain ordering control. Probe the supplied legal candidates in this order: "
        + " > ".join(order)
        + ". This card reorders existing candidates only; it supplies no program, "
        "suppresses nothing, and grants no execution authority. Actual utility is "
        "decided by Target Support and retention by delayed feedback."
    )
    assert "Frozen program steps:" not in body, "ordering card must not supply a program"
    return {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": int(revision),
        "body": body,
        "observable_applicability": {
            "feature": "task_kind",
            "op": "==",
            "value": str(scope["task"]),
        },
        "allowed_tools": [],
        "risk_guards": {
            "card_kind": CARD_KIND,
            "scope": {k: str(scope[k]) for k in _SCOPE_KEYS},
            "ranking_key": {
                "formula": "E_gain - lambda * E_harm_magnitude",
                "lambda": float(lam),
                "denominator": "evaluated_attempts",
                "unknown_policy": "legal-but-unprobed is UNKNOWN, ranked after known "
                                  "operators; never treated as zero gain",
            },
            "order": list(order),
            "evidence": {op: dict(row) for op, row in sorted(evidence.items())},
            "evidence_blocks": {
                block: {op: dict(row) for op, row in sorted(rows.items())}
                for block, rows in sorted((evidence_blocks or {}).items())
            },
            "authority": {
                "reorders_supplied_candidates": True,
                "supplies_candidates": False,
                "suppresses_operators": False,
                "grants_execution": False,
            },
        },
    }


def is_ordering_card(skill: Any) -> bool:
    guards = getattr(skill, "risk_guards", None)
    if guards is None and isinstance(skill, Mapping):
        guards = skill.get("risk_guards")
    return bool(guards) and guards.get("card_kind") == CARD_KIND


def card_scope_matches(skill: Any, scope_now: Mapping[str, str]) -> bool:
    """Runtime 机械精确匹配。retrieval 只看 applicability，管不到这一层。"""
    guards = getattr(skill, "risk_guards", None)
    if guards is None and isinstance(skill, Mapping):
        guards = skill.get("risk_guards")
    if not guards or guards.get("card_kind") != CARD_KIND:
        return False
    scope = guards.get("scope") or {}
    return all(str(scope.get(k)) == str(scope_now.get(k)) for k in _SCOPE_KEYS)


def _leading_op(steps: Any) -> str | None:
    if isinstance(steps, str):
        try:
            steps = json.loads(steps)
        except json.JSONDecodeError:
            return None
    if isinstance(steps, Sequence) and steps:
        head = steps[0]
        if isinstance(head, Mapping) and isinstance(head.get("op"), str):
            return head["op"]
        if isinstance(head, Sequence) and head and isinstance(head[0], str):
            return head[0]
    return None


def reorder_probe_order(
    probe_order: Sequence[str],
    steps_map: Mapping[str, Any],
    card: Any,
) -> list[str]:
    """按卡的 ``order`` 重排**同一批**候选。

    严格保序不变量：返回值是 ``probe_order`` 的一个排列——**不增、不删、不替换**。
    候选的算子不在卡里（或无法解析）时，保持它们彼此的原相对次序并排在最后。
    """
    original = list(probe_order)
    guards = getattr(card, "risk_guards", None)
    if guards is None and isinstance(card, Mapping):
        guards = card.get("risk_guards")
    order = list((guards or {}).get("order") or ())
    if not order:
        return original
    rank = {op: i for i, op in enumerate(order)}
    big = len(order)
    decorated = []
    for position, cand in enumerate(original):
        op = _leading_op(steps_map.get(cand))
        decorated.append((rank.get(op, big), position, cand))
    decorated.sort()
    result = [cand for _, _, cand in decorated]
    assert sorted(result) == sorted(original), "reorder must be a permutation"
    return result


__all__ = [
    "CARD_KIND",
    "accumulate",
    "merge_evidence",
    "build_ordering_card",
    "card_scope_matches",
    "empty_evidence",
    "is_ordering_card",
    "rank_operators",
    "reorder_probe_order",
]
