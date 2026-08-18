"""工作包 V1 核心实验：A5 vs A3（deepseek 副本，2026-08-08）。

项目承重比较（AGENTS.md §2）：相同 Target downstream-feedback budget 下，
Source Memory 是否让首次正向 Workflow 的试错更少、且不增加 harm。

设计（用户裁决落地）：
- Source 切片产生经验（扫描 Action-Response，relation/status 自动决定）；
- Memory 只改变候选探测顺序（不硬排除任何 Workflow）；
- 弱经验规则（local_window_missing 作为可用 Pattern）：
  局部缺失较高 → impute_fft 值得优先 Probe（弱先验，非硬规则、不生成跨域 Skill）；
- Target Support 最终确认；无改善换候选或 abstain；
- 等预算：Target Support probe 上限 B=2，首个正向即停。

GEFCom 冻结时间线：Source support=832, Source delayed=880,
Target support=928, Target delayed=976。

用法：
  python evaluation/functional/run_v1_a5_vs_a3.py [--domain gefcom]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402

HORIZON = 48
MAX_TARGET_PROBES = 2  # 等预算：Target Support probe 上限
MATERIAL_THRESHOLD = 0.005  # 巡检裁决：material threshold，0/0 与数值噪声不计正向
# changes_target_space 算子与官方生成路径（v6）一致排除——候选池 21 个
CTS_EXCLUDED = tuple(
    n for n in v6.OPERATOR_NAMES
    if v6.OPERATOR_METADATA[n].get("changes_target_space")
)


def is_positive(gain: float | None) -> bool:
    """material 正向判定（>= MATERIAL_THRESHOLD），0/0 与噪声不计。"""
    return gain is not None and gain >= MATERIAL_THRESHOLD
WINDOW = 192
WEAK_PRIOR_OP = "impute_fft"  # 弱经验：局部缺失较高时优先 Probe（不排除其他）
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_a5_vs_a3_report.json")  # 实际输出带 domain 后缀

TIMELINE = {
    "gefcom": (832, 880, 928, 976),
    # NN5 max_len=791：Source 536/584 → Target 632/680（不重叠且 origin+48<=791）
    "nn5": (536, 584, 632, 680),
    # NOAA max_len=1024：与 GEFCom 同冻结时间线（重算合法：976+48=1024 <= 1024，
    # 且 source delayed 880+48=928 与 Target support 928 不重叠）
    "noaa": (832, 880, 928, 976),
}


def local_missing_median(values: Mapping[str, np.ndarray], origin: int) -> float:
    """局部窗口缺失（Pattern/Context）：每序列最近 WINDOW 步最大缺失 run 的 median。"""
    runs: list[int] = []
    for array in values.values():
        arr = np.asarray(array, dtype=np.float64)
        lo = max(0, origin - WINDOW)
        mask = ~np.isfinite(arr[lo:origin])
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        runs.append(best)
    return float(statistics.median(runs))


def build_probe_order(
    *,
    source_episodes: Sequence[Any],
    local_missing: float,
    weak_prior: bool,
    sort_key: str = "stable_gain",
) -> list[str]:
    """探测顺序：Source 切片实测的经验排序 + 可选弱先验提前。

    经验 = Source 切片实测的 relation + support/delayed gain：
    POSITIVE 段内按 sort_key 降序 → CONFLICT → NEGATIVE 最后（弱化，不排除）。
    sort_key（2026-08-08 排序键裁决）：
      "stable_gain"（默认）— min(support, delayed) 双稳定优先（GEFCom 实证防翻转反例）；
      "delayed_gain" — 单窗口 delayed 收益降序（历史变体，保留作对照）。
    Memory 只改顺序；weak_prior=True 且局部缺失较高时 impute_fft 提前（弱先验）。
    """
    pos: list[tuple[str, float]] = []
    conf: list[str] = []
    neg: list[str] = []
    for ep in source_episodes:
        op = ep.workflow_signature
        if ep.relation == "POSITIVE":
            sg = ep.support_response.get("gain")
            dg = ep.delayed_response.get("gain")
            sg = float(sg) if isinstance(sg, (int, float)) else 0.0
            dg = float(dg) if isinstance(dg, (int, float)) else 0.0
            key = min(sg, dg) if sort_key == "stable_gain" else dg
            pos.append((op, key))
        elif ep.relation == "CONFLICT":
            conf.append(op)
        else:
            neg.append(op)
    order = [op for op, _ in sorted(pos, key=lambda t: -t[1])] + conf + neg
    if weak_prior and WEAK_PRIOR_OP in order and local_missing > 1.5:
        order.remove(WEAK_PRIOR_OP)
        order.insert(1, WEAK_PRIOR_OP)  # 提前到第 2 位（弱先验，不排除其他）
    return order


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 core experiment: A5 vs A3")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom", choices=tuple(TIMELINE))
    parser.add_argument("--weak-prior", action="store_true",
                        help="enable impute_fft weak prior (local missing high)")
    parser.add_argument("--sort-key", choices=("stable_gain", "delayed_gain"), default="stable_gain",
                        help="experience quality sort key (default: stable_gain, 排序键裁决)")
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain
    if domain not in TIMELINE:
        raise SystemExit(f"no frozen timeline for {domain}")
    src_support, src_delayed, tgt_support, tgt_delayed = TIMELINE[domain]

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # Source 经验：在 Source 切片（832/880）上实测 26 算子（复用用户版 build_source_memory）
    # 注意：不能用 w2_operator_scan（其 B/C 切片 = Target 928/976，会造成经验泄漏）
    from run_w2_operator_scan import _default_params
    source_episodes, source_attempts = v1.build_source_memory(
        domain=domain,
        roster=roster,
        values=values,
        config=config,
        operators=sorted(
            name for name in v6.OPERATOR_NAMES
            if "forecast" in (v6.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
            and name not in CTS_EXCLUDED
        ),
        source_support_origin=src_support,
        source_delayed_origin=src_delayed,
        baseline_cache=baseline_cache,
    )
    print(f"== source episodes: {len(source_episodes)} (from Source {src_support}/{src_delayed})")

    # Target 局部缺失（Pattern/Context）
    local_missing = local_missing_median(values, tgt_support)
    print(f"== {domain}: Target support={tgt_support}, local_window_missing={local_missing:.1f}")

    # A5（有 Memory）：Source 经验排序（+ 可选弱先验）；A3（无 Memory）：固定顺序
    a5_order = build_probe_order(source_episodes=source_episodes,
                                 local_missing=local_missing,
                                 weak_prior=args.weak_prior,
                                 sort_key=args.sort_key)
    # A3 独立库存（巡检裁决）：固定 OPERATOR_NAMES 排序、排除 cts——不是空 Source
    # Memory 反推；与 Source 经验无任何关系
    a3_order = sorted(
        name for name in v6.OPERATOR_NAMES
        if "forecast" in (v6.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
        and name not in CTS_EXCLUDED
    )
    print(f"== A5 order (source memory): {a5_order}")
    print(f"== A3 order (fixed):         {a3_order}")

    def run_arm(order: Sequence[str], label: str) -> dict[str, Any]:
        """等预算 Target 探测：最多 MAX_TARGET_PROBES 次，首个正向即停，无改善 abstain。

        参数必须与 Source 经验构建一致（_default_params）。delayed 不在
        探测时评估（MED-2 修复：由 main 对两臂并集统一打开，避免"未评估当无 harm"）。
        """
        gains: list[float] = []
        probed: list[str] = []
        first_positive: int | None = None
        harm = 0
        for op in order:
            if len(probed) >= MAX_TARGET_PROBES:
                break
            compiled = v1.make_compiled(op, _default_params(op, period))
            g = v1.gain_at(roster, values, config, compiled, tgt_support, baseline_cache)
            if g is None:
                continue
            probed.append(op)
            gains.append(g)
            if g < -MATERIAL_THRESHOLD:
                harm += 1
            if is_positive(g):
                first_positive = len(probed)
                break  # stop-on-first-positive
        abstained = len(probed) == 0 or all(g <= 0 for g in gains)
        return {
            "label": label,
            "probe_order": probed,
            "support_gains": [round(g, 4) for g in gains],
            "first_positive_probe": first_positive,
            "harm_probe_count": harm,
            "abstained": abstained,
        }

    a5 = run_arm(a5_order, "A5")
    a3 = run_arm(a3_order, "A3")

    # MED-2 修复：计划冻结后，对两臂所有已探测算子的并集打开 Target delayed
    union_ops = list(dict.fromkeys(a5["probe_order"] + a3["probe_order"]))
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
        # 最终执行 = 首个正向候选（material 正向）；abstain = 无执行
        fp = arm["first_positive_probe"]
        if fp is not None and fp - 1 < len(arm["probe_order"]):
            op = arm["probe_order"][fp - 1]
            dg = delayed_by_op.get(op)
            arm["final_executed_workflow"] = op
            arm["delayed_gain"] = round(dg, 4) if dg is not None else None
            # 巡检裁决：delayed_harm 只基于最终执行的 Workflow（探测中有害候选
            # 是试错成本 harm_probe，不是执行 harm）
            arm["delayed_harm"] = bool(dg is not None and dg < -MATERIAL_THRESHOLD)
        else:
            arm["final_executed_workflow"] = None
            arm["delayed_gain"] = None
            arm["delayed_harm"] = False  # 无执行；失败由 abstained=True 表达
        return arm

    a5 = attach_delayed(a5)
    a3 = attach_delayed(a3)
    print(f"== A5: {a5}")
    print(f"== A3: {a3}")

    # 核心判定：A5 首次正向 probe <= A3 且 harm 不增（用户裁决指标）
    a5_first = a5["first_positive_probe"]
    a3_first = a3["first_positive_probe"]
    less_trial = (
        (a5_first is not None and a3_first is None)
        or (a5_first is not None and a3_first is not None and a5_first <= a3_first)
    )
    no_more_harm = a5["harm_probe_count"] <= a3["harm_probe_count"]
    no_delayed_harm_increase = (a5["delayed_harm"] or False) <= (a3["delayed_harm"] or False)
    verdict = (
        "A5_BETTER_OR_EQUAL_TRIALS_NO_MORE_HARM"
        if (less_trial and no_more_harm and no_delayed_harm_increase)
        else "A5_NOT_BETTER"
    )
    print(f"\n== verdict: {verdict} (first_positive A5={a5_first} A3={a3_first}, "
          f"harm A5={a5['harm_probe_count']} A3={a3['harm_probe_count']})")

    out = root / REPORT_OUT_REL.with_name(f"w1_a5_vs_a3_report_{domain}.json")  # A4：域后缀防覆盖
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-a5-vs-a3",
            "domain": domain,
            "timeline": {"src_support": src_support, "src_delayed": src_delayed,
                         "tgt_support": tgt_support, "tgt_delayed": tgt_delayed},
            "max_target_probes": MAX_TARGET_PROBES,
            "target_local_window_missing": local_missing,
            "weak_prior_enabled": args.weak_prior,
            "a5_order": a5_order,
            "a3_order": a3_order,
            "a5": a5,
            "a3": a3,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
