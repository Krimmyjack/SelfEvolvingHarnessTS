"""工作包 2 前置检验：Context-applicability premise（零 LLM，deepseek 副本，2026-08-07）。

用户裁决：暂停硬排除；first fault = 合法 TS Context 已算出但未绑定进 Episode/Query。

设计（用户裁决）：
- Episode = 部署时可见 Context F + 精确 Program P + Action-Response R（Support/delayed）。
  support_gain 只留 R，绝不参与 F 相似度。
- 第一轮只用三个与周期型 Program 机制直接相关的特征：
  maximum_missing_run_length / bound_period、median_acf_at_calendar_period、
  median_normalized_seasonal_residual（现有 _window_summary 已计算）+ Program geometry。
- 已暴露 development 数据上按固定时间顺序生成多个不重叠 block，零 LLM 重放同一 Program
  （period_median_complete——已知跨域翻转：NN5 B+C+、GEFCom B-，是测试 F 预测方向的最佳候选）。
  每 block 执行前算 F，执行后记 positive/negative/conflict。
- leave-one-block-out：只凭 F 检索最近 Episode，预测留出 block 的方向。
  记录 relation/sign 一致率、false veto、missed harm、与占位检索（常量 F）对比。
- 不先定距离阈值 θ（单点调阈值过拟合教训：CURRENT_STAGE.md:127）。

判定：
- 相近 F 下频繁翻转 → Context 不可识别，禁止硬排除（只能 Target Support/abstain/再找 Observation）；
- 留出 block 稳定区分方向且无 false veto → 冻结 Context 表达，下一轮实现
  "高匹配负经验→风险约束；弱匹配→仅作先验"。

用法：
  python evaluation/functional/run_w2_applicability_premise.py [--domain nn5] [--program period_median_complete]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402

HORIZON = 48
BLOCK_STEP = HORIZON  # 不重叠 block
MIN_ORIGIN = 200
REPORT_OUT_REL = Path("artifacts/functional/e2/w2_applicability_premise_report.json")


# ---------------------------------------------------------------------------
# 1. F 提取（三特征 + Program geometry；复用 _window_summary 逻辑）
# ---------------------------------------------------------------------------

def extract_F(values: Mapping[str, np.ndarray], config: Mapping[str, object], origin: int) -> dict[str, float]:
    """在 origin 之前的 context 上计算三个周期机制特征。

    复用 v6 的 _acf 与 robust center/scale 逻辑；特征全部来自部署时可见数据。
    """
    period = int(config.get("period", 1))
    windows = [
        np.asarray(values[str(row["series_uid"])][:origin], dtype=np.float64)
        for row in []  # roster 由调用方传入——此函数仅用 values
    ]
    # 直接实现（不依赖 roster）：对所有序列的 context 段
    all_run_lengths: list[int] = []
    all_acfs: list[float] = []
    all_seasonal: list[float] = []
    for uid, array in values.items():
        window = np.asarray(array[:origin], dtype=np.float64)
        mask = ~np.isfinite(window)
        # missing runs
        runs: list[tuple[int, int]] = []
        start = None
        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif not m and start is not None:
                runs.append((start, i))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        all_run_lengths.extend(stop - s for s, stop in runs)
        # acf at calendar period + seasonal residual
        left, right = _observed_lag_pairs(window, period)
        if left.size >= 3:
            lc = left - float(np.mean(left))
            rc = right - float(np.mean(right))
            denom = float(np.linalg.norm(lc) * np.linalg.norm(rc))
            if denom > 0.0:
                all_acfs.append(float(np.dot(lc, rc) / denom))
                observed = window[np.isfinite(window)]
                center = float(np.median(observed)) if observed.size else 0.0
                scale = float(1.4826 * np.median(np.abs(observed - center)))
                if not np.isfinite(scale) or scale <= 1e-12:
                    scale = float(np.std(observed)) if observed.size else 1.0
                if np.isfinite(scale) and scale > 1e-12:
                    all_seasonal.append(float(np.median(np.abs(right - left)) / scale))
    return {
        "maximum_missing_run_length": float(max(all_run_lengths)) if all_run_lengths else 0.0,
        "median_acf_at_calendar_period": float(statistics.median(all_acfs)) if all_acfs else 0.0,
        "median_normalized_seasonal_residual": float(statistics.median(all_seasonal)) if all_seasonal else 0.0,
        "bound_period": float(period),
    }


def _observed_lag_pairs(values: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= values.size:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    left = values[:-lag]
    right = values[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    return left[valid], right[valid]


def F_distance(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """特征距离（L1，仅数值特征；support_gain 绝不参与）。"""
    keys = ("maximum_missing_run_length", "median_acf_at_calendar_period",
            "median_normalized_seasonal_residual", "bound_period")
    return sum(abs(float(a.get(k, 0.0)) - float(b.get(k, 0.0))) for k in keys)


# ---------------------------------------------------------------------------
# 2. block 重放（零 LLM：同一 Program 在每个 block 上确定性评估）
# ---------------------------------------------------------------------------

def evaluate_program_at(
    root: Path,
    roster: list[dict[str, object]],
    values: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    compiled: Any,
    origin: int,
) -> float:
    """在指定 origin 上评估 Program 的 sMASE gain（相对 baseline）。"""
    baseline = v6._evaluate(roster, values, None, config, origin=origin)
    candidate = v6._evaluate(roster, values, compiled, config, origin=origin)
    return float(baseline["mean_smase"] - candidate["mean_smase"])


def main() -> int:
    parser = argparse.ArgumentParser(description="W2 zero-LLM Context-applicability premise")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="nn5", choices=("nn5", "gefcom"))
    parser.add_argument("--program", default="period_median_complete")
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    max_len = max(int(len(v)) for v in values.values())
    period = int(config.get("period", 1))

    # 构造 compiled Program（同一 Program 重放）
    from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
    from SelfEvolvingHarnessTS.contracts.program import Program
    from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import CompiledWorkflow
    params = {"period": period, "cycles": 3, "min_donors": 2}
    program = Program.from_steps([(args.program, dict(params))], source="applicability_premise_replay")
    candidate = Candidate(
        candidate_id=f"{args.program}_replay",
        kind=CandidateKind.PROGRAM,
        program=program,
        source="applicability_premise_replay",
    )
    compiled = CompiledWorkflow(candidate, (), tuple(program.steps))

    # 不重叠 block：origin 从 MIN_ORIGIN 按 HORIZON 步进，delayed 需要 origin+2*HORIZON <= max_len
    max_origin = max_len - 2 * HORIZON
    origins = list(range(MIN_ORIGIN, max_origin + 1, BLOCK_STEP))
    print(f"== {domain}: {len(origins)} non-overlapping blocks, origins={origins[:5]}...")

    blocks: list[dict[str, Any]] = []
    for origin in origins:
        F = extract_F(values, config, origin)
        support_gain = evaluate_program_at(root, roster, values, config, compiled, origin)
        delayed_gain = evaluate_program_at(root, roster, values, config, compiled, origin + HORIZON)
        if support_gain > 0 and delayed_gain > 0:
            relation = "POSITIVE"
        elif support_gain > 0 and delayed_gain <= 0:
            relation = "CONFLICT"
        elif support_gain <= 0:
            relation = "NEGATIVE"
        else:
            relation = "CONFLICT"
        blocks.append({
            "origin": origin,
            "F": F,
            "support_gain": support_gain,
            "delayed_gain": delayed_gain,
            "relation": relation,
        })
        print(f"  block origin={origin:4d} F={ {k: round(v, 3) for k, v in F.items()} } "
              f"support={support_gain:+.4f} delayed={delayed_gain:+.4f} -> {relation}")

    # ------------------------------------------------------------------
    # 3. leave-one-block-out：只凭 F 检索最近邻居预测方向
    # ------------------------------------------------------------------
    def _nearest(block_idx: int) -> dict[str, Any]:
        others = [b for i, b in enumerate(blocks) if i != block_idx]
        return min(others, key=lambda b: F_distance(b["F"], blocks[block_idx]["F"]))

    sign_consistent = 0
    false_veto = 0      # 检索为负，但实际 support 或 delayed 正
    missed_harm = 0     # 检索为正，但实际 support 或 delayed 有害
    total = len(blocks)
    predictions: list[dict[str, Any]] = []
    for i, b in enumerate(blocks):
        nb = _nearest(i)
        predicted = nb["relation"]
        actual = b["relation"]
        ok = (predicted == actual) or (
            predicted in ("POSITIVE", "CONFLICT") and actual in ("POSITIVE", "CONFLICT")
        ) or (predicted == "NEGATIVE" and actual == "NEGATIVE")
        if ok:
            sign_consistent += 1
        if predicted == "NEGATIVE" and (b["support_gain"] > 0 or b["delayed_gain"] > 0):
            false_veto += 1
        if predicted in ("POSITIVE", "CONFLICT") and (b["support_gain"] < 0 or b["delayed_gain"] < 0):
            missed_harm += 1
        predictions.append({
            "block_origin": b["origin"],
            "neighbor_origin": nb["origin"],
            "neighbor_F": {k: round(v, 3) for k, v in nb["F"].items()},
            "F_distance": round(F_distance(b["F"], nb["F"]), 4),
            "predicted": predicted,
            "actual": actual,
            "consistent": ok,
        })
        print(f"  LOO[{i}] nb={nb['origin']} d={F_distance(b['F'], nb['F']):.3f} "
              f"pred={predicted:9s} actual={actual:9s} {'OK' if ok else 'X'}")

    sign_rate = sign_consistent / total if total else 0.0
    print(f"\n== LOO: sign_consistency={sign_rate:.2f} ({sign_consistent}/{total}), "
          f"false_veto={false_veto}, missed_harm={missed_harm}")

    # 与占位检索（常量 F——所有 block 距离相等 → 取第一个）对比
    placeholder_consistent = sum(
        1 for i, b in enumerate(blocks)
        if blocks[0 if i != 0 else 1]["relation"] == b["relation"]
    )
    placeholder_rate = placeholder_consistent / total if total else 0.0
    print(f"== placeholder-retrieval (constant F) sign_consistency={placeholder_rate:.2f}")

    # 判定（用户裁决）
    if sign_rate < 0.6 and (false_veto > 0 or missed_harm > 0):
        verdict = "CONTEXT_NOT_IDENTIFIABLE_NO_HARD_EXCLUSION"
    elif sign_rate >= 0.6 and false_veto == 0 and missed_harm <= max(1, total // 5):
        verdict = "CONTEXT_APPLICABILITY_SUPPORTED_FREEZE_EXPRESSION"
    else:
        verdict = "PARTIAL_NEEDS_MORE_OBSERVATION"
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "w2-applicability-premise-zero-llm",
            "domain": domain,
            "program": args.program,
            "blocks": blocks,
            "loo_predictions": predictions,
            "sign_consistency_rate": sign_rate,
            "false_veto": false_veto,
            "missed_harm": missed_harm,
            "placeholder_sign_consistency_rate": placeholder_rate,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
