"""工作包 V1：排序键裁决（NN5 同一 Target 切片，零 LLM，2026-08-08）。

背景：`build_probe_order`（run_v1_a5_vs_a3.py）用 Source delayed_gain 降序定义
经验质量；`compile_experienced_order`（run_v1_fastpath.py）用 relation 分层 +
stable_gain=min(support,delayed) 定义。NN5 上两种排序曾给出相反结局
（阶段报告 §3-1），排序键=经验质量定义，未裁决。

本脚本：NN5 冻结时间线（Source 536/584 → Target 632/680），同一 Source 经验集、
同一候选集（23 个 forecast 算子）、同一 Target 切片、同一预算 B=2、
stop-on-first-positive，仅排序键不同。delayed 按 MED-2 修复语义：Target 计划
冻结后对两臂已探测算子并集统一打开（"未评估"不当"无 harm"）。

裁决（对称化 A5/A3 判定）：first_positive 小者优（abstain 为 None 最劣）；
平局比 harm_probe_count；再平局比 delayed_harm。均为单时间线单次运行，
不构成统计显著。

用法：
  python evaluation/functional/run_v1_sorting_key_verdict.py [--domain nn5]
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

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

HORIZON = 48
MAX_TARGET_PROBES = 2  # 等预算：Target Support probe 上限（与 A5/A3 一致）
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_sorting_key_verdict_report.json")

TIMELINE = {
    # NN5 max_len=791：Source 536/584 → Target 632/680（不重叠且 origin+48<=791）
    "nn5": (536, 584, 632, 680),
}


def build_order_delayed_gain(source_episodes: Sequence[Any]) -> list[str]:
    """排序键 A：Source delayed_gain 降序（run_v1_a5_vs_a3.build_probe_order 核心）。

    POSITIVE 按 Source delayed gain 降序 → CONFLICT → NEGATIVE（弱化，不排除）。
    """
    pos: list[tuple[str, float]] = []
    conf: list[str] = []
    neg: list[str] = []
    for ep in source_episodes:
        op = ep.workflow_signature
        if ep.relation == "POSITIVE":
            dg = ep.delayed_response.get("gain")
            pos.append((op, float(dg) if isinstance(dg, (int, float)) else 0.0))
        elif ep.relation == "CONFLICT":
            conf.append(op)
        else:
            neg.append(op)
    return [op for op, _ in sorted(pos, key=lambda t: -t[1])] + conf + neg


def build_order_stable_gain(
    source_episodes: Sequence[Any], operators: Sequence[str]
) -> list[str]:
    """排序键 B：relation 分层 + 层内 stable_gain=min(support,delayed) 降序。

    与 run_v1_fastpath.compile_experienced_order 的排序逻辑一致（直接按
    Episode 的 signed response 计算，不经过 Retriever 包装——排序键本身与
    episode 选择一一对应，等价于逐算子检索）。
    """
    relation_rank = {"POSITIVE": 0.0, "CONFLICT": 1.0, "NEGATIVE": 2.0}
    by_op = {ep.workflow_signature: ep for ep in source_episodes}
    rows: list[tuple[tuple[float, float, str], str]] = []
    for op in operators:
        ep = by_op.get(op)
        if ep is None:
            rows.append(((3.0, 0.0, op), op))  # 无经验的算子垫底（与 fastpath 一致）
            continue
        support_gain = float(ep.support_response["gain"])
        delayed_gain = float(ep.delayed_response["gain"])
        stable_gain = min(support_gain, delayed_gain)
        rows.append(((relation_rank[ep.relation], -stable_gain, op), op))
    rows.sort(key=lambda row: row[0])
    return [row[1] for row in rows]


def run_target_arm(
    *,
    order: Sequence[str],
    roster: list[dict[str, object]],
    values: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    origin: int,
    baseline_cache: dict[int, float],
    period: int,
) -> dict[str, Any]:
    """等预算 Target 探测：最多 MAX_TARGET_PROBES 次，首个正向即停，无改善 abstain。

    参数必须与 Source 经验构建一致（_default_params）。delayed 不在探测时评估
    （MED-2 修复：由 main 对两臂并集统一打开）。
    """
    gains: list[float] = []
    probed: list[str] = []
    first_positive: int | None = None
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
        if g < 0:
            harm += 1
        if g > 0:
            first_positive = len(probed)
            break  # stop-on-first-positive
    abstained = len(probed) == 0 or all(g <= 0 for g in gains)
    return {
        "probe_order": probed,
        "support_gains": [round(g, 4) for g in gains],
        "first_positive_probe": first_positive,
        "harm_probe_count": harm,
        "abstained": abstained,
    }


def verdict_between(
    key_a: dict[str, Any], key_b: dict[str, Any]
) -> tuple[str, str]:
    """对称化 A5/A3 判定：first_positive 小者优 → harm 少者优 → delayed_harm 无者优。"""

    def score(arm: dict[str, Any]) -> tuple[int, int, int]:
        fp = arm["first_positive_probe"]
        return (
            0 if fp is not None else 1,  # abstain 最劣
            fp if fp is not None else 10_000,
            -arm["harm_probe_count"],  # harm 少者优
            -(1 if arm["delayed_harm"] else 0),
        )

    sa, sb = score(key_a), score(key_b)
    if sa == sb:
        return "TIED", (
            f"两键首次正向均为 {key_a['first_positive_probe']}、harm 均为 "
            f"{key_a['harm_probe_count']}、delayed_harm 均为 {key_a['delayed_harm']}；"
            f"首个正向 delayed 均>0（delayed_gain 键 {key_a['delayed_gain']} vs "
            f"stable_gain 键 {key_b['delayed_gain']}），差异微小不构成裁决依据"
        )
    if sa > sb:
        return "ORDER_KEY_STABLE_GAIN_SUPERIOR", (
            f"stable_gain first={key_b['first_positive_probe']} harm="
            f"{key_b['harm_probe_count']} delayed_harm={key_b['delayed_harm']} 优于 "
            f"delayed_gain first={key_a['first_positive_probe']} harm="
            f"{key_a['harm_probe_count']} delayed_harm={key_a['delayed_harm']}"
        )
    return "ORDER_KEY_DELAYED_GAIN_SUPERIOR", (
        f"delayed_gain first={key_a['first_positive_probe']} harm="
        f"{key_a['harm_probe_count']} delayed_harm={key_a['delayed_harm']} 优于 "
        f"stable_gain first={key_b['first_positive_probe']} harm="
        f"{key_b['harm_probe_count']} delayed_harm={key_b['delayed_harm']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 sorting-key verdict (zero LLM)")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="nn5", choices=tuple(TIMELINE))
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain
    src_support, src_delayed, tgt_support, tgt_delayed = TIMELINE[domain]

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    max_len = max(int(len(v)) for v in values.values())
    if tgt_delayed + HORIZON > max_len:
        raise SystemExit(f"{domain}: target delayed segment exceeds max_len={max_len}")
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # 经验只来自 Source 切片（536/584），与 A5/A3 完全同源
    operators = sorted(
        name for name in v6.OPERATOR_NAMES
        if "forecast" in (v6.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
    )
    source_episodes, source_attempts = v1.build_source_memory(
        domain=domain,
        roster=roster,
        values=values,
        config=config,
        operators=operators,
        source_support_origin=src_support,
        source_delayed_origin=src_delayed,
        baseline_cache=baseline_cache,
    )
    print(f"== source episodes: {len(source_episodes)} "
          f"(from Source {src_support}/{src_delayed}, {len(operators)} candidates)")

    order_dg = build_order_delayed_gain(source_episodes)
    order_sg = build_order_stable_gain(source_episodes, operators)
    print(f"== ORDER_A delayed_gain 前 6: {order_dg[:6]}")
    print(f"== ORDER_B stable_gain   前 6: {order_sg[:6]}")

    arm_dg = run_target_arm(order=order_dg, roster=roster, values=values,
                            config=config, origin=tgt_support,
                            baseline_cache=baseline_cache, period=period)
    arm_sg = run_target_arm(order=order_sg, roster=roster, values=values,
                            config=config, origin=tgt_support,
                            baseline_cache=baseline_cache, period=period)

    # MED-2 修复：Target 计划冻结后，对两臂已探测算子并集打开 Target delayed
    union_ops = list(dict.fromkeys(arm_dg["probe_order"] + arm_sg["probe_order"]))
    delayed_by_op: dict[str, float] = {}
    for op in union_ops:
        compiled = v1.make_compiled(op, _default_params(op, period))
        dg = v1.gain_at(roster, values, config, compiled, tgt_delayed, baseline_cache)
        if dg is not None:
            delayed_by_op[op] = dg

    def attach_delayed(arm: dict[str, Any]) -> dict[str, Any]:
        arm = dict(arm)
        arm_delayed = {op: delayed_by_op[op] for op in arm["probe_order"] if op in delayed_by_op}
        arm["delayed_by_probed"] = {op: round(v, 4) for op, v in arm_delayed.items()}
        # delayed_harm：该臂任一已探测算子 delayed < 0（不再把"未评估"当无 harm）
        arm["delayed_harm"] = any(v < 0 for v in arm_delayed.values())
        fp = arm["first_positive_probe"]
        if fp is not None and fp - 1 < len(arm["probe_order"]):
            op = arm["probe_order"][fp - 1]
            arm["delayed_gain"] = round(delayed_by_op[op], 4) if op in delayed_by_op else None
        else:
            arm["delayed_gain"] = None
        return arm

    arm_dg = attach_delayed(arm_dg)
    arm_sg = attach_delayed(arm_sg)

    verdict, why = verdict_between(arm_dg, arm_sg)
    print("\n== 对比表（NN5 Target 632/680，B=2，stop-on-first-positive）")
    print(f"  {'指标':<22}{'排序键A delayed_gain':<28}{'排序键B stable_gain':<28}")
    print(f"  {'探测序列':<22}{str(arm_dg['probe_order']):<28}{str(arm_sg['probe_order']):<28}")
    print(f"  {'support gains':<22}{str(arm_dg['support_gains']):<28}{str(arm_sg['support_gains']):<28}")
    print(f"  {'首次正向 probe':<22}{str(arm_dg['first_positive_probe']):<28}{str(arm_sg['first_positive_probe']):<28}")
    print(f"  {'harm probe 数':<22}{arm_dg['harm_probe_count']:<28}{arm_sg['harm_probe_count']:<28}")
    print(f"  {'abstained':<22}{str(arm_dg['abstained']):<28}{str(arm_sg['abstained']):<28}")
    print(f"  {'delayed by probed':<22}{str(arm_dg['delayed_by_probed']):<28}{str(arm_sg['delayed_by_probed']):<28}")
    print(f"  {'delayed harm':<22}{str(arm_dg['delayed_harm']):<28}{str(arm_sg['delayed_harm']):<28}")
    print(f"  {'首个正向 delayed':<22}{str(arm_dg['delayed_gain']):<28}{str(arm_sg['delayed_gain']):<28}")
    print(f"\n== verdict: {verdict}")
    print(f"   {why}")
    print("   （单时间线单次运行，不构成统计显著）")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-sorting-key-verdict",
            "domain": domain,
            "timeline": {"src_support": src_support, "src_delayed": src_delayed,
                         "tgt_support": tgt_support, "tgt_delayed": tgt_delayed},
            "max_target_probes": MAX_TARGET_PROBES,
            "candidate_count": len(operators),
            "source_episode_count": len(source_episodes),
            "source_relation_counts": {
                r: sum(1 for ep in source_episodes if ep.relation == r)
                for r in ("POSITIVE", "CONFLICT", "NEGATIVE")
            },
            "order_key_delayed_gain": order_dg,
            "order_key_stable_gain": order_sg,
            "arm_delayed_gain_key": arm_dg,
            "arm_stable_gain_key": arm_sg,
            "target_delayed_gains_for_probed_union": {
                op: round(v, 4) for op, v in delayed_by_op.items()
            },
            "verdict": verdict,
            "verdict_reason": why,
            "claim_ceiling": (
                "Same slice, same budget, ordering-key only; single timeline single run, "
                "not statistically significant."
            ),
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
