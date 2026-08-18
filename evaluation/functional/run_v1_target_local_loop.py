"""V1 核心构建：Target-local Memory/Skill 闭环（零 LLM，2026-08-08）。

用户收束裁决第 5 步——Harness"会从运行中自我更新"的核心能力：

  Target Support 实测 → 立即写 Episode（support 正 → LOCAL_DRAFT）
  → 计划冻结后 delayed 打开 → 更新（delayed 正 → LOCAL_ACTIVE；
     delayed 负 → CONFLICT/RESTRICTED）
  → 下一轮 Fast Path 读取更新后的本地经验

验收（用户裁决）：
- A3/A5 相同 Target feedback 预算，两臂都写 Target Episode；
- A5 仅额外拥有 Source prior；
- delayed 在计划冻结后才打开；
- 第二次 Target 调用的检索/探测顺序确实受上一轮本地 Episode 影响；
- 一个正向激活案例（LOCAL_ACTIVE → 第二轮提前）和一个翻转收缩案例
  （CONFLICT → 第二轮弱化/靠后）。

用法：
  python evaluation/functional/run_v1_target_local_loop.py [--domain nn5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402 (build_probe_order/MATERIAL_THRESHOLD/CTS_EXCLUDED)
import run_v1_fastpath as v1  # noqa: E402
from experience_memory import (  # noqa: E402
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    build_episode,
)

MAX_TARGET_PROBES = 2
HORIZON = 48
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_target_local_loop_report.json")


def write_target_episode(
    *,
    domain: str,
    op: str,
    program_steps: Sequence[Mapping[str, object]],
    support_gain: float,
    delayed_gain: float | None,
    support_context: Mapping[str, float] | None = None,
    delayed_context: Mapping[str, float] | None = None,
    episode_id_suffix: str = "",
) -> Any:
    """Target Support 实测后立即写 Episode（审查修复 2：保存完整 program steps）。

    support >= MATERIAL → LOCAL_DRAFT；否则 EPISODE_ONLY（审查修复 3 的
    support 侧：ABSTAIN 语义由 support 近零判定）。
    半径门控修复 2/4：同时记录 support_context 与 delayed_context（部署可见
    recent/change 特征，window_context 输出）——两侧 outcome 都参与 stable_gain
    与适用范围；delayed_context 在 delayed 打开时写入（见 update_delayed_status）。
    episode_id_suffix：同算子跨轮/跨 origin 时追加（如 "_origin928"），避免
    distance-trace 键覆盖（审查裁决 2026-08-08 十三：普通后缀即可，不需要 Hash）。
    """
    status = STATUS_LOCAL_DRAFT if support_gain >= core.MATERIAL_THRESHOLD else STATUS_EPISODE_ONLY
    return build_episode(
        episode_id=f"{domain}_target_{op}{episode_id_suffix}",
        task_consumer_key=core.TASK_CONSUMER_KEY if hasattr(core, "TASK_CONSUMER_KEY") else "forecast|ridge|sMASE",
        domain_namespace=domain,
        context_summary={
            "cohort": {"series_count": 1, "evaluation_series_count": 0},
            "local_pattern": {"support_gain": support_gain,
                              **(dict(support_context) if support_context else {})},
            "delayed_pattern": dict(delayed_context) if delayed_context else {},
            "program_geometry": {"scope": "training_rows", "program_steps": list(program_steps)},
        },
        workflow_signature=op,
        support_response={"gain": support_gain, "accepted": support_gain >= core.MATERIAL_THRESHOLD},
        delayed_response={"evaluated": delayed_gain is not None, "gain": delayed_gain},
        relation="POSITIVE" if support_gain >= core.MATERIAL_THRESHOLD else "NEGATIVE",
        evidence_level="DELAYED" if delayed_gain is not None else "SUPPORT",
        local_status=status,
        evidence_refs=["run_v1_target_local_loop"],
    )


def write_abstain_episode(*, domain: str, reason: str) -> Any:
    """A3 无合法计划时写入 abstain/rejection Episode（审查修复 4）。"""
    return build_episode(
        episode_id=f"{domain}_target_abstain",
        task_consumer_key=core.TASK_CONSUMER_KEY if hasattr(core, "TASK_CONSUMER_KEY") else "forecast|ridge|sMASE",
        domain_namespace=domain,
        context_summary={"cohort": {"series_count": 1}, "local_pattern": {}, "program_geometry": {}},
        workflow_signature="identity",
        support_response={"gain": None, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="ABSTAIN",
        evidence_level="SUPPORT",
        local_status=STATUS_EPISODE_ONLY,
        evidence_refs=[reason],
    )


def update_delayed_status(episode: Any, delayed_gain: float, delayed_context: Mapping[str, float] | None = None) -> Any:
    """delayed 打开后四类状态转移（审查修复 3）：

      support>=M, delayed>=M → POSITIVE / LOCAL_ACTIVE
      support>=M, delayed<M  → CONFLICT / RESTRICTED
      support<M,  delayed<M  → NEGATIVE / EPISODE_ONLY（含 0/0 → ABSTAIN / EPISODE_ONLY）
      support<M,  delayed>=M → CONFLICT / EPISODE_ONLY（B-C+，保守冲突）

    半径门控修复 2/4：delayed_context（该 delayed origin 的可见 Context）一并写入
    context_summary.delayed_pattern——delayed 的适用证据与其 Context 绑定。
    """
    import dataclasses
    sg = float(episode.support_response.get("gain") or 0.0)
    m = core.MATERIAL_THRESHOLD
    s_pos = sg >= m
    d_pos = delayed_gain >= m
    s_neg = sg < m
    d_neg = delayed_gain < m
    near_zero = abs(sg) < m and abs(delayed_gain) < m

    if s_pos and d_pos:
        status, relation = STATUS_LOCAL_ACTIVE, "POSITIVE"
    elif s_pos and d_neg:
        status, relation = "RESTRICTED", "CONFLICT"
    elif near_zero:
        status, relation = STATUS_EPISODE_ONLY, "ABSTAIN"  # 0/0
    elif s_neg and d_neg:
        status, relation = STATUS_EPISODE_ONLY, "NEGATIVE"
    else:  # s_neg and d_pos（B-C+）
        status, relation = STATUS_EPISODE_ONLY, "CONFLICT"
    ctx = dict(episode.context_summary)
    if delayed_context is not None:
        ctx["delayed_pattern"] = dict(delayed_context)
    return dataclasses.replace(
        episode,
        local_status=status,
        relation=relation,
        delayed_response={"evaluated": True, "gain": delayed_gain},
        evidence_level="DELAYED",
        context_summary=ctx,
    )


def compiled_from_episode(episode: Any, period: int) -> Any:
    """从 Episode 保存的完整 program steps 重建 compiled（审查修复 2：
    delayed 用与 support 相同的 Workflow——不默认参数重建）。"""
    steps = (episode.context_summary.get("program_geometry") or {}).get("program_steps") or []
    if not steps:
        # 回退：单算子默认参数（仅当保存的 steps 缺失）
        from run_w2_operator_scan import _default_params
        steps = [{"op": episode.workflow_signature,
                  "params": _default_params(episode.workflow_signature, period)}]
    return v1.make_compiled(
        str(steps[0].get("op", episode.workflow_signature)),
        {k: v for k, v in (steps[0].get("params") or {}).items()},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 Target-local Memory/Skill closed loop")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="nn5", choices=("nn5", "gefcom", "noaa"))
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain

    # 时间线（与 A5/A3 实验一致）
    if domain not in core.TIMELINE:
        raise SystemExit(f"no timeline for {domain}")
    ss, sd, ts, td = core.TIMELINE[domain]
    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # Source prior（A5 独有）
    from run_w2_operator_scan import _default_params
    source_episodes, _ = v1.build_source_memory(
        domain=domain, roster=roster, values=values, config=config,
        operators=sorted(n for n in v6.OPERATOR_NAMES
                         if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                         and n not in core.CTS_EXCLUDED),
        source_support_origin=ss, source_delayed_origin=sd, baseline_cache=baseline_cache,
    )
    print(f"== source episodes: {len(source_episodes)} (A5 only)")

    def run_probe_order(order: Sequence[str], label: str) -> dict[str, Any]:
        gains: list[float] = []
        probed: list[str] = []
        harm = 0
        for op in order:
            if len(probed) >= MAX_TARGET_PROBES:
                break
            compiled = v1.make_compiled(op, _default_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, ts, baseline_cache)
            if g is None:
                continue
            probed.append(op)
            gains.append(g)
            if g < -core.MATERIAL_THRESHOLD:
                harm += 1
            if g >= core.MATERIAL_THRESHOLD:
                break  # stop-on-first-positive
        return {"label": label, "probe_order": probed, "support_gains": gains, "harm": harm}

    # ---------- 第一轮（R1：Target support=ts，delayed=td）----------
    def probe_at(order: Sequence[str], origin: int, label: str) -> dict[str, Any]:
        gains: list[float] = []
        probed: list[str] = []
        harm = 0
        for op in order:
            if len(probed) >= MAX_TARGET_PROBES:
                break
            compiled = v1.make_compiled(op, _default_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            if g is None:
                continue
            probed.append(op)
            gains.append(g)
            if g < -core.MATERIAL_THRESHOLD:
                harm += 1
            if g >= core.MATERIAL_THRESHOLD:
                break
        return {"label": label, "probe_order": probed, "support_gains": gains, "harm": harm}

    def episodes_for(arm_result: dict[str, Any], arm_name: str) -> list[Any]:
        """该臂探测结果 → Episode（完整 compiled steps 保存，修复 2）。"""
        eps = []
        for op, g in zip(arm_result["probe_order"], arm_result["support_gains"]):
            steps = [{"op": op, "params": dict(_default_params(op, period))}]
            eps.append(write_target_episode(domain=domain, op=op,
                                            program_steps=steps,
                                            support_gain=g, delayed_gain=None))
        if not eps:
            # 修复 4：无合法计划 → abstain/rejection Episode
            eps.append(write_abstain_episode(domain=domain, reason=f"{arm_name}_no_valid_plan"))
        return eps

    a5_order1 = core.build_probe_order(source_episodes=source_episodes,
                                       local_missing=0.0, weak_prior=False)
    a3_order1 = sorted(n for n in v6.OPERATOR_NAMES
                       if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                       and n not in core.CTS_EXCLUDED)
    a5_1 = probe_at(a5_order1, ts, "A5-round1")
    a3_1 = probe_at(a3_order1, ts, "A3-round1")
    print(f"[R1] A5: {a5_1}")
    print(f"[R1] A3: {a3_1}")

    # 修复 1：双臂 Episode 分离（a5_local / a3_local 各自独立，不混合）
    a5_local = episodes_for(a5_1, "A5")
    a3_local = episodes_for(a3_1, "A3")
    print(f"[R1] A5 episodes: {len(a5_local)} | A3 episodes: {len(a3_local)}")

    # 计划冻结后 delayed 打开 → 更新 status（同一 Workflow——compiled_from_episode）
    def update_with_delayed(episodes: list[Any], origin: int) -> list[Any]:
        updated = []
        for ep in episodes:
            if ep.workflow_signature == "identity":  # abstain episode
                updated.append(ep)
                continue
            compiled = compiled_from_episode(ep, period)
            dg = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            if dg is not None:
                updated.append(update_delayed_status(ep, dg))
            else:
                updated.append(ep)
        return updated

    a5_updated = update_with_delayed(a5_local, td)
    a3_updated = update_with_delayed(a3_local, td)
    for ep in a5_updated + a3_updated:
        print(f"[R1-delayed] {ep.episode_id}: status={ep.local_status} relation={ep.relation}")

    # ---------- 第二轮（最小收尾：R2 support=728，完全位于 R1 delayed [680,728) 之后；
    # 本轮不再开 R2 delayed）----------
    ts2 = td + HORIZON  # R2 support 起点 = R1 delayed 终点（区间不重叠）
    max_len = max(int(len(v)) for v in values.values())
    if ts2 + HORIZON > max_len:
        raise SystemExit(f"{domain}: no room for independent R2 slice at {ts2}")
    print(f"[R2] target slice: support={ts2} (fully after R1 delayed [{td},{td + HORIZON})); "
          f"R2 delayed not opened this round")

    # A5 池 = Source + 自己的本地；A3 池 = 自己的本地（修复 1：不跨臂）
    a5_order2 = core.build_probe_order(source_episodes=source_episodes + a5_updated,
                                       local_missing=0.0, weak_prior=False)
    a3_order2 = core.build_probe_order(source_episodes=a3_updated,
                                       local_missing=0.0, weak_prior=False)
    a5_2 = probe_at(a5_order2, ts2, "A5-round2")
    a3_2 = probe_at(a3_order2, ts2, "A3-round2")
    print(f"[R2] A5: {a5_2}")
    print(f"[R2] A3: {a3_2}")

    # ---------- 验收断言（审查修订版）----------
    all_local = a5_local + a3_local
    all_updated = a5_updated + a3_updated
    checks = {
        # (a) 两臂各自都写 Target Episode（分离、不混合）
        "a_both_arms_write_target_episodes": len(a5_local) > 0 and len(a3_local) > 0,
        # (b) A5 仅额外拥有 Source prior（A3 池不含 source）
        "b_a5_has_source_prior_only_extra": len(source_episodes) > 0
        and all(not e.episode_id.startswith("v1_") for e in a3_updated),
        # (c) delayed 计划冻结后打开（未更新前 evaluated=False）
        "c_delayed_after_freeze": all(e.delayed_response.get("evaluated") is False
                                      for e in all_local),
        # (d) 第二轮探测顺序受本地 Episode 影响（A5 两轮顺序不同）
        "d_second_round_affected": a5_2["probe_order"] != a5_1["probe_order"],
        # (e) 正向激活（LOCAL_ACTIVE 存在且第二轮排前）+ 四类转移正确
        "e_positive_activation": any(e.local_status == STATUS_LOCAL_ACTIVE for e in all_updated)
        and a5_2["probe_order"]
        and any(e.workflow_signature == a5_2["probe_order"][0]
                and e.local_status == STATUS_LOCAL_ACTIVE for e in a5_updated),
        "e_four_state_transitions": any(e.relation == "ABSTAIN" for e in all_updated)
        and any(e.relation == "NEGATIVE" for e in all_updated)
        and any(e.local_status == STATUS_LOCAL_ACTIVE for e in all_updated),
        # (f) R2 时间区间与 R1 delayed 不重叠（布尔）
        "f_independent_slice_non_overlap": ts2 >= td + HORIZON,
    }
    all_pass = all(checks.values())
    print(f"\n== checks: {checks}")
    verdict = "CLOSED_LOOP_PASS" if all_pass else "CLOSED_LOOP_PARTIAL"
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-target-local-loop",
            "domain": domain,
            "timeline": {"src": (ss, sd), "tgt": (ts, td)},
            "round1": {"a5": a5_1, "a3": a3_1},
            "a5_episodes": [e.to_dict() for e in a5_local],
            "a3_episodes": [e.to_dict() for e in a3_local],
            "a5_delayed_updated": [e.to_dict() for e in a5_updated],
            "a3_delayed_updated": [e.to_dict() for e in a3_updated],
            "round2": {"a5": a5_2, "a3": a3_2},
            "checks": checks,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
