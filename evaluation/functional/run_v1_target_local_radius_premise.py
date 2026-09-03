"""V1 半径检索 premise（零 LLM，2026-08-08，修复版）。

用户裁决（修复 1/4）：历史常量特征不能删除——
  - 查询值 == 历史常量：距离贡献 0；
  - 查询值偏离历史常量：直接 OUTSIDE_OBSERVED_SUPPORT（观测支持域外），
    偏离量按自然单位报告（count/run 自然单位为 1，不调参）。
用户裁决（修复 4/4）：历史 Context 池 = Source/R1/R2 的 support+delayed Context
  （两侧 outcome 都参与 stable_gain 与适用范围；delayed 的可见 Context 属于正向适用证据）。

Context 向量 = CohortHistoryPublicToolGateway.compare_history_windows（public_tools.py）：
  recent 摘要（7 数值特征）+ early_to_recent_change（7 delta）；series 在 origin 截断
  （部署可见）；recent 窗口 = [origin-192, origin)，与 _evaluate 的 eval context 一致。

δ 校准（仅历史、去重 Context；query 完全不参与）：历史 Context 留一最近邻距离的
  固定分位数 q75，与查询距离同一 z-score 尺度（历史 mean/std 冻结）。
判定：常量偏离 → OUTSIDE_OBSERVED_SUPPORT；否则 d_min（与任一历史 Context 的最小
  标准化距离）> δ → 半径外。历史 Context 太少（n < 3）→ 不授予强匹配（弱参考/回退）。
对照：同一协议用累计前缀 extract_F 特征重算（与 W2 F_distance 同款 4 特征）。

决策树（用户裁决）：
  R3 半径外（含 OUTSIDE_OBSERVED_SUPPORT）→ RADIUS_GATE_APPLICABLE（批准实现距离门并重跑冻结链）
  R3 半径内 → OBSERVATION_BLIND（不调 δ；若仍盲，只补 missingness×seasonal-phase geometry
    Observation，不做 generic flatline/level-shift，不接现有三个素材）

用法：
  python evaluation/functional/run_v1_target_local_radius_premise.py [--domain gefcom]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
from public_tools import CohortHistoryPublicToolGateway  # noqa: E402

HORIZON = 48
WINDOW_LENGTH = 192  # 与 _evaluate eval context / a5_vs_a3 WINDOW 一致
DELTA_QUANTILE = 0.75
MIN_HISTORICAL_CONTEXTS = 3
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_target_local_radius_premise_report.json")

# (support_origin, delayed_origin|None)：源先验 + 轮次（与 3-rounds 脚本冻结链一致）
SLICES = {
    "gefcom": [(640, 688), (736, 784), (832, 880), (928, 976)],
    "nn5": [(536, 584), (632, 680), (728, None)],
}

domain = "gefcom"  # 运行时由 --domain 覆盖；evaluate_round 读取全局


def window_context(values: Mapping[str, np.ndarray], origin: int, period: int) -> dict[str, float]:
    """部署可见 recent/change 特征（单一来源：methods/ttha/signed_radius.py）。"""
    from signed_radius import window_context as _window_context
    return _window_context(values, origin, period)


def prefix_context(values: Mapping[str, np.ndarray], config: Mapping[str, object], origin: int) -> dict[str, float]:
    """对照：累计前缀 extract_F（W2 F_distance 同款 4 特征）。"""
    return v1.extract_F(values, config, origin)


def historical_origins(round_idx: int) -> list[int]:
    """当前轮之前全部已观测 Context：源 + 更早轮次的 support 与 delayed（修复 4/4）。"""
    origins: list[int] = []
    for i in range(0, round_idx):
        ts, td = SLICES[domain][i]
        origins.append(ts)
        if td is not None:
            origins.append(td)
    return origins


def _scale(hist_vecs: Sequence[dict[str, float]]) -> tuple[list[str], dict[str, float], Any]:
    """历史冻结尺度。常量特征（历史 std≈0）不删除：保留为硬门——查询偏离即半径外。

    返回 (informative, const_vals, z)：informative 参与标准化 L1；const_vals 由
    evaluate_round 检查查询是否偏离（自然单位：count/run 为 1，不调参）。
    """
    feats = sorted(set().union(*(set(h) for h in hist_vecs)))
    means = {f: float(np.mean([h[f] for h in hist_vecs])) for f in feats}
    stds = {f: float(np.std([h[f] for h in hist_vecs])) for f in feats}
    informative = [f for f in feats if stds[f] > 1e-12]
    const_vals = {f: means[f] for f in feats if stds[f] <= 1e-12}

    def z(vec: dict[str, float]) -> dict[str, float]:
        return {f: (vec[f] - means[f]) / stds[f] for f in informative}

    return informative, const_vals, z


def evaluate_round(round_idx: int, ctx_by_origin: Mapping[int, dict[str, float]]) -> dict[str, Any]:
    """逐轮判定：历史池（更早轮次 support+delayed，去重、不含本轮）→ 常量偏离门 →
    δ(q75 LOO) → min 距离 → in/out。"""
    support, delayed = SLICES[domain][round_idx]
    hist_origins = historical_origins(round_idx)
    hist_vecs = [ctx_by_origin[o] for o in hist_origins]
    n_hist = len(hist_vecs)
    sufficient = n_hist >= MIN_HISTORICAL_CONTEXTS
    delta: float | None = None
    d_min: float | None = None
    missing: list[str] = []
    deviations: dict[str, float] = {}
    nearest_origin: int | None = None
    per_origin_distances: dict[str, float] = {}
    if not sufficient:
        verdict = "WEAK_HISTORY"
    else:
        informative, const_vals, z = _scale(hist_vecs)
        required = informative + list(const_vals)
        query = ctx_by_origin[support]
        missing = [f for f in required if f not in query]
        if missing:
            verdict = "UNKNOWN_FEATURES"
        else:
            # 修复 1/4：历史常量特征的查询偏离 → 观测支持域外（状态从未在历史出现）
            deviations = {
                f: float(query[f] - const_vals[f])
                for f in const_vals
                if abs(float(query[f] - const_vals[f])) > 1e-12
            }
            if deviations:
                verdict = "OUTSIDE_OBSERVED_SUPPORT"
            else:
                hz = [z(h) for h in hist_vecs]
                loo = [
                    min(
                        sum(abs(a[f] - b[f]) for f in informative)
                        for j, b in enumerate(hz) if j != i
                    )
                    for i, a in enumerate(hz)
                ]
                delta = float(np.quantile(loo, DELTA_QUANTILE))
                qz = z(query)
                dists = [sum(abs(qz[f] - h[f]) for f in informative) for h in hz]
                per_origin_distances = {str(o): float(d) for o, d in zip(hist_origins, dists)}
                nearest_origin = int(hist_origins[int(np.argmin(dists))])
                d_min = float(min(dists))
                verdict = "OUTSIDE" if d_min > delta else "INSIDE"
    return {
        "round": f"R{round_idx}",
        "support": support,
        "delayed": delayed,
        "historical_origins": hist_origins,
        "n_historical": n_hist,
        "sufficient_history": sufficient,
        "delta_q75": delta,
        "d_min_standardized": d_min,
        "nearest_origin": nearest_origin,
        "per_origin_distances": per_origin_distances,
        "const_deviations": deviations,
        "missing_features": missing,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 radius-retrieval zero-LLM premise (fixed)")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom", choices=tuple(SLICES))
    args = parser.parse_args()
    root = args.root.resolve()
    global domain
    domain = args.domain
    slices = SLICES[domain]

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    max_len = max(int(len(v)) for v in values.values())

    # 所有相关 origin 的 Context（部署可见截断；绝不读取 target outcome）
    origins: list[int] = []
    for ts, td in slices:
        origins.append(ts)
        if td is not None:
            origins.append(td)
    contexts: dict[int, dict[str, float]] = {}
    prefix_ctx: dict[int, dict[str, float]] = {}
    for origin in origins:
        contexts[origin] = window_context(values, origin, period)
        prefix_ctx[origin] = prefix_context(values, config, origin)
    print(f"== {domain}: origins={origins}")

    rounds = [evaluate_round(i, contexts) for i in range(1, len(slices))]
    cumulative_rounds = [evaluate_round(i, prefix_ctx) for i in range(1, len(slices))]

    for r in rounds:
        print(f"  {r['round']}: hist={r['historical_origins']} n={r['n_historical']} "
              f"const_dev={r['const_deviations'] or '-'} "
              f"delta={r['delta_q75'] if r['delta_q75'] is not None else '-'} "
              f"d_min={r['d_min_standardized'] if r['d_min_standardized'] is not None else '-'} "
              f"(nearest={r['nearest_origin']}) -> {r['verdict']}")
    for r in cumulative_rounds:
        print(f"  [cumulative] {r['round']}: const_dev={r['const_deviations'] or '-'} "
              f"delta={r['delta_q75'] if r['delta_q75'] is not None else '-'} "
              f"d_min={r['d_min_standardized'] if r['d_min_standardized'] is not None else '-'} "
              f"(nearest={r['nearest_origin']}) -> {r['verdict']}")

    # 裁决（用户 premise 决策树，以最近一轮为准）
    last = rounds[-1]
    if last["verdict"] in ("OUTSIDE", "OUTSIDE_OBSERVED_SUPPORT"):
        verdict = "RADIUS_GATE_APPLICABLE"
    elif last["verdict"] == "INSIDE":
        verdict = "OBSERVATION_BLIND"
    else:
        verdict = last["verdict"]
    print(f"\n== verdict: {verdict}")

    out = root / REPORT_OUT_REL.with_name(f"{REPORT_OUT_REL.stem}_{domain}{REPORT_OUT_REL.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-target-local-radius-premise",
            "domain": domain,
            "slices": [{"support": ts, "delayed": td} for ts, td in slices],
            "max_len": max_len,
            "delta_quantile": DELTA_QUANTILE,
            "window_length": WINDOW_LENGTH,
            "contexts": {str(o): c for o, c in contexts.items()},
            "prefix_contexts": {str(o): c for o, c in prefix_ctx.items()},
            "rounds_recent_change": rounds,
            "rounds_cumulative_prefix": cumulative_rounds,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
