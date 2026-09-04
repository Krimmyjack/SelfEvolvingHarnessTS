"""methods/ttha/group_fault.py——多轨迹共同错误归因（用户裁决
2026-08-12：第一缺口——不建复杂平台——轻量确定性分组 + Contrast
Capsule + 共同 replacement headroom）。

把 Slow Path 从"单次失败→单 Card→≤2 Patch"推进为：
  多条失败轨迹 → 按 (完整 Workflow 指纹, response sign) 轻量分组 →
  重复 first-fault 组 → Contrast Capsule（per-episode 对齐行 + view
  对齐 + 对比案例）→ 共同 replacement headroom 验证 → 组级 Patch
  （Slow Agent 基于整组 Capsule 提出）→ 组外同域验证。

Wave 1 修复（GROUP_FAULT 主链，2026-08-13）：
  - 分组键 = 完整 workflow 指纹（program_geometry.program_steps 的算子
    序列——不再压成首算子）；
  - Capsule 保留 per-episode 对齐行（不再跨 Episode flatten per-view）；
    view 身份由调用方经 view_keys 提供（executor roster 的 eval series
    序——与 per_view_gain 同序）；全部提供且一致才建立跨 Episode 对齐，
    否则如实降级（不猜身份）；
  - Capsule 含 matched positive/conflict 对比案例（同 workflow 指纹的
    对照组——来自全量 Episode 语料）；
  - find_common_headroom 改为 evaluator 注入（跨 series 组的 origin →
    executor 解析由调用方负责）。

只需要普通 Episode 列表 + 轻量分组——无向量数据库/Pattern Graph/
新 Ledger。per-view gain 与 support_origin 由 online_loop 写入
Episode（GROUP_FAULT 保留——学习证据，不渲染进 instruction）。
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .experience_memory import workflow_signature_of
from .signed_radius import MATERIAL_THRESHOLD

M = MATERIAL_THRESHOLD  # 0.005


def iter_failure_episodes(episodes: Sequence[Any]) -> list[Any]:
    """material failure Episode：support < −M 或 delayed < −M。"""
    out: list[Any] = []
    for ep in episodes:
        sg = (getattr(ep, "support_response", {}) or {}).get("gain")
        dg = (getattr(ep, "delayed_response", {}) or {}).get("gain")
        if (isinstance(sg, (int, float)) and float(sg) < -M) or (
                isinstance(dg, (int, float)) and float(dg) < -M):
            out.append(ep)
    return out


def _sign_of(ep: Any) -> str:
    """失败 Episode 的 response sign：support 负（NEGATIVE）/ 冲突
    （support 负 delayed 正——CONFLICT）。"""
    sg = (getattr(ep, "support_response", {}) or {}).get("gain")
    dg = (getattr(ep, "delayed_response", {}) or {}).get("gain")
    if (isinstance(sg, (int, float)) and float(sg) < -M
            and isinstance(dg, (int, float)) and float(dg) >= M):
        return "CONFLICT"
    return "NEGATIVE"


def _full_workflow_of(ep: Any) -> str:
    """Episode 的完整 workflow 指纹（算子序列——不含参数值）。
    优先 program_geometry.program_steps（online_loop 保留的完整步骤）；
    解析失败退化到 stored workflow_signature（历史 Episode 只有首算子）。"""
    steps = ((getattr(ep, "context_summary", {}) or {})
             .get("program_geometry", {}) or {}).get("program_steps") or []
    sig = workflow_signature_of(steps)
    if sig and sig != "unknown":
        return sig
    return str(getattr(ep, "workflow_signature", "?") or "?")


def group_first_faults(episodes: Sequence[Any],
                       min_group: int = 2) -> list[dict[str, Any]]:
    """轻量确定性分组：按 (完整 workflow 指纹, sign) 分组——≥ min_group
    的组 = 重复 first-fault 组（共同 first fault 的证据基础）。Wave 1：
    workflow 键不再压成首算子（多步 workflow 保留算子序列）。"""
    failures = iter_failure_episodes(episodes)
    buckets: dict[tuple[str, str], list[Any]] = {}
    for ep in failures:
        key = (_full_workflow_of(ep), _sign_of(ep))
        buckets.setdefault(key, []).append(ep)
    return [{"workflow": key[0], "sign": key[1], "episodes": eps}
            for key, eps in sorted(buckets.items())
            if len(eps) >= min_group]


def _per_view(ep: Any) -> list[float]:
    raw = ((getattr(ep, "context_summary", {}) or {})
           .get("per_view_gain") or [])
    return [float(v) for v in raw if isinstance(v, (int, float))]


def _origin_of(ep: Any) -> int:
    return int(((getattr(ep, "context_summary", {}) or {})
                .get("support_origin") or 0))


def _provenance_of(ep: Any, target_domain_namespace: str | None) -> str:
    """E-3/C-3：source/target provenance。

    第一版以 domain_namespace 为来源标记（不引入 Episode schema 字段）。
    ``target_domain_namespace`` 未提供时如实记 ``unknown``——不得猜。
    """
    if not target_domain_namespace:
        return "unknown"
    return (
        "target"
        if str(getattr(ep, "domain_namespace", "") or "")
        == target_domain_namespace
        else "source"
    )


def build_contrast_capsule(
    group: Mapping[str, Any],
    *,
    all_episodes: Sequence[Any] | None = None,
    view_keys: Mapping[str, Sequence[str]] | None = None,
    target_domain_namespace: str | None = None,
    candidate_workflows: Sequence[str] | None = None,
) -> dict[str, Any]:
    """组内 Episode 的 Contrast Capsule（确定性聚合——Wave 1 修复）：
    - per-episode 对齐行（episode_id / origin / 完整 program steps /
      support / delayed / per-view gain 列表）——不再跨 Episode flatten；
    - view 对齐：view_keys = {episode_id: [view_id...]}（与 per_view_gain
      同序——调用方从 executor roster 提供 eval series 身份）。全部
      Episode 提供且长度一致 → 建立公共 view 对齐行（view × 各 Episode
      gain）；否则 alignment=未建立（如实降级——不猜身份）；
    - 对比案例（同 workflow 指纹的 matched positive / negative / conflict
      ——来自全量 Episode 语料——共同归因需要的对照组），ref 带
      source/target/unknown provenance（`target_domain_namespace` 提供时）；
    - cohort gain 分布 + delayed sign 分布（不变）。"""
    eps = list(group.get("episodes") or [])
    origins = sorted({_origin_of(e) for e in eps})
    per_episode_rows: list[dict[str, Any]] = []
    cohort_gains: list[float] = []
    delayed_signs: Counter = Counter()
    for e in eps:
        cs = getattr(e, "context_summary", {}) or {}
        steps = (cs.get("program_geometry", {}) or {}).get("program_steps") or []
        sg = (getattr(e, "support_response", {}) or {}).get("gain")
        dr = getattr(e, "delayed_response", {}) or {}
        dg = dr.get("gain")
        delayed_evaluated = bool(dr.get("evaluated"))
        if isinstance(sg, (int, float)):
            cohort_gains.append(float(sg))
        delayed_signs[
            "negative" if (isinstance(dg, (int, float)) and float(dg) < -M)
            else "positive" if (isinstance(dg, (int, float)) and float(dg) >= M)
            else "unevaluated"] += 1
        per_episode_rows.append({
            "episode_id": str(getattr(e, "episode_id", "?")),
            "provenance": _provenance_of(e, target_domain_namespace),
            "origin": _origin_of(e),
            "workflow_signature": str(
                getattr(e, "workflow_signature", "?") or "?"),
            "program_steps": [
                {"op": str(s.get("op")), "params": dict(s.get("params") or {})}
                for s in steps if isinstance(s, Mapping) and s.get("op")],
            "support_gain": (float(sg) if isinstance(sg, (int, float))
                             else None),
            "delayed_gain": (float(dg) if delayed_evaluated
                             and isinstance(dg, (int, float)) else None),
            "per_view_gain": _per_view(e),
        })
    # view 对齐（view_keys 与 per_view_gain 同序；公共 view = 第一个
    # Episode 的 view 序 ∩ 其余全部）
    aligned_rows: list[dict[str, Any]] = []
    alignment_established = False
    if isinstance(view_keys, Mapping) and view_keys and per_episode_rows:
        keyed = {r["episode_id"]: tuple(
            str(x) for x in view_keys.get(r["episode_id"], ()))
            for r in per_episode_rows}
        lengths_ok = all(
            len(keyed.get(r["episode_id"], ())) == len(r["per_view_gain"])
            for r in per_episode_rows)
        if lengths_ok:
            first_ids = keyed[per_episode_rows[0]["episode_id"]]
            common = [v for v in first_ids
                      if all(v in keyed.get(r["episode_id"], ())
                             for r in per_episode_rows[1:])]
            for v in common:
                gains: list[float | None] = []
                for r in per_episode_rows:
                    vk = keyed.get(r["episode_id"], ())
                    i = vk.index(v)
                    gains.append(r["per_view_gain"][i]
                                 if i < len(r["per_view_gain"]) else None)
                aligned_rows.append({
                    "view_id": v,
                    "gains": gains,
                    "worsen_count": sum(
                        1 for g in gains
                        if isinstance(g, (int, float)) and float(g) < -M),
                    "improve_count": sum(
                        1 for g in gains
                        if isinstance(g, (int, float)) and float(g) >= M),
                })
            alignment_established = len(common) > 0
    # 对比案例（同 workflow 指纹——失败组的对照组）
    contrast: dict[str, list[dict[str, Any]]] = {
        "positive": [],
        "negative": [],
        "conflict": [],
    }
    source_episode_ids: list[str] = []
    referenced_source_episode_ids: list[str] = []
    filtered_source_episode_ids: list[str] = []
    if all_episodes:
        in_group = {r["episode_id"] for r in per_episode_rows}
        wf = str(group.get("workflow") or "?")
        matching_workflows = {wf}
        if candidate_workflows:
            matching_workflows.update(str(value) for value in candidate_workflows)
        for e in all_episodes:
            if str(getattr(e, "episode_id", "")) in in_group:
                continue
            provenance = _provenance_of(e, target_domain_namespace)
            if provenance == "source":
                source_episode_ids.append(str(getattr(e, "episode_id", "?")))
            if _full_workflow_of(e) not in matching_workflows:
                if provenance == "source":
                    filtered_source_episode_ids.append(
                        str(getattr(e, "episode_id", "?"))
                    )
                continue
            if provenance == "source":
                referenced_source_episode_ids.append(
                    str(getattr(e, "episode_id", "?"))
                )
            sg = (getattr(e, "support_response", {}) or {}).get("gain")
            dg = (getattr(e, "delayed_response", {}) or {}).get("gain")
            relation = str(getattr(e, "relation", "") or "")
            bucket = None
            if relation == "CONFLICT":
                bucket = "conflict"
            elif relation == "POSITIVE":
                bucket = "positive"
            elif relation == "NEGATIVE":
                bucket = "negative"
            else:
                # 数值回退只在缺 relation 时使用；relation 优先，避免把
                # support 正 / delayed 负 的常见翻转错放进 positive。
                if isinstance(sg, (int, float)) and float(sg) >= M:
                    bucket = "positive"
                elif (isinstance(sg, (int, float)) and float(sg) < -M
                      and isinstance(dg, (int, float)) and float(dg) >= M):
                    bucket = "conflict"
                elif isinstance(sg, (int, float)) and float(sg) < -M:
                    bucket = "negative"
            if bucket is None:
                continue
            ref = {"episode_id": str(getattr(e, "episode_id", "?")),
                   "provenance": provenance,
                   "origin": _origin_of(e),
                   "support_gain": (float(sg)
                                    if isinstance(sg, (int, float))
                                    else None)}
            contrast[bucket].append(ref)
    return {
        "workflow": group.get("workflow"),
        "sign": group.get("sign"),
        "n_episodes": len(eps),
        "origins": origins,
        "cohort_gain": {
            "min": round(min(cohort_gains), 4) if cohort_gains else None,
            "max": round(max(cohort_gains), 4) if cohort_gains else None,
            "mean": (round(sum(cohort_gains) / len(cohort_gains), 4)
                     if cohort_gains else None),
        },
        "delayed_signs": dict(delayed_signs),
        "per_episode_rows": per_episode_rows,
        "view_alignment": {
            "established": alignment_established,
            "common_view_ids": [r["view_id"] for r in aligned_rows],
            "aligned_rows": aligned_rows,
        },
        "contrast_cases": contrast,
        "retrieval_scope": {
            "incumbent_workflow": str(group.get("workflow") or "?"),
            "candidate_workflows": sorted(
                str(value) for value in (candidate_workflows or ())
            ),
        },
        "source_provenance": {
            "target_domain_namespace": target_domain_namespace,
            "source_episode_ids": sorted(set(source_episode_ids)),
            "referenced_source_episode_ids": sorted(
                set(referenced_source_episode_ids)
            ),
            "filtered_source_episode_ids": sorted(
                set(filtered_source_episode_ids)
            ),
            "filtered_source_episode_count": len(
                set(filtered_source_episode_ids)
            ),
        },
    }


def find_common_headroom(group: Mapping[str, Any],
                         evaluator_group: Any,
                         alternatives: Sequence[str],
                         steps_of: Any) -> dict[str, Any]:
    """共同 replacement headroom（Wave 1 修复——evaluator 注入）：对组内
    每个 Episode 的 Support 窗口 replay 替代候选。
    evaluator_group(steps, origin) → receipt——调用方按 series 解析
    executor（跨 series 组需要每 origin 对应其系列 executor；同系列组
    单一 executor 即可）。共同 headroom = 替代在**全部**组内 Episode 上
    support ≥ M。

    返回每个替代：{per_episode_gains, common_positive}。"""
    eps = list(group.get("episodes") or [])
    out: dict[str, Any] = {}
    for alt in alternatives:
        per_ep: list[dict[str, Any]] = []
        all_positive = True
        for e in eps:
            origin = _origin_of(e)
            try:
                rr = evaluator_group(tuple(steps_of(alt)), origin)
                g = (float(rr.gain) if rr.gain is not None else None)
                per_ep.append({"origin": origin, "gain": g,
                               "passed": bool(rr.verification.passed)})
            except Exception as exc:  # noqa: BLE001
                per_ep.append({"origin": origin, "gain": None,
                               "error": f"{type(exc).__name__}"})
                all_positive = False
                continue
            if g is None or g < M:
                all_positive = False
        out[alt] = {"per_episode_gains": per_ep,
                    "common_positive": all_positive}
    return out


def unique_common_positive(headroom: Mapping[str, Any],
                           alternatives: Sequence[str]) -> str | None:
    """确定性 Evidence Compiler（P0 降级设计，2026-08-13——LLM Batch
    Evidence 集成未建立后，Runtime 依据 Batch Evidence 决策）：唯一
    common_positive 替代 → 其名字；零或多个 → None（确定性 abstain）。
    同一 M 阈值同一门——不做选择推理，只做确定性判定。"""
    pos = [a for a in alternatives
           if (headroom.get(a) or {}).get("common_positive")]
    return pos[0] if len(pos) == 1 else None


__all__ = ["iter_failure_episodes", "group_first_faults",
           "build_contrast_capsule", "find_common_headroom",
           "unique_common_positive"]
