"""工作包 V1 阶段 C 前置：Program-specific Pattern 视角验证（零 LLM，2026-08-07）。

阶段 B 结论：单一 structural view 无法表达适用性——outlier_iqr 双稳定、
denoise_savgol 跨切片翻转，同一视角下检索无法区分。
用户裁决下一步：验证"Skill 选择 Program-specific Pattern 视角"。

本实验（零 LLM，用 GEFCom Source→Target 已冻结时间线）：
1. 对 26 个 forecast 算子：Source(832/880) 与 Target(928/976) 各测效果方向 + F。
2. 翻转算子（Source relation ≠ Target relation）vs 稳定算子：
   每个特征在 Source→Target 的变化量 ΔF 是否与翻转共现。
3. 若翻转算子存在"显著变化的特征"且稳定算子该特征变化小 →
   Program-specific 敏感特征有机制依据（skill = ⟨Program, 敏感特征⟩）；
   否则当前特征空间对 Program 适用性不可识别。

时间线（与阶段 B 一致）：
  Source support=832, Source delayed=880, Target support=928, Target delayed=976。

用法：
  python evaluation/functional/run_v1_program_pattern_view.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402  (用户版：extract_F/gain_at/relation_of/make_compiled)

SRC_SUPPORT = 832
SRC_DELAYED = 880
TGT_SUPPORT = 928
TGT_DELAYED = 976
FEATURES = ("maximum_missing_run_length", "median_acf_at_calendar_period",
            "median_normalized_seasonal_residual", "bound_period")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_program_pattern_view_report.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 Program-specific pattern view premise")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    domain = "gefcom"

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    F_src = v1.extract_F(values, config, SRC_SUPPORT)
    F_tgt = v1.extract_F(values, config, TGT_SUPPORT)
    print(f"== F_source(832)={ {k: round(v, 3) for k, v in F_src.items()} }")
    print(f"== F_target(928)={ {k: round(v, 3) for k, v in F_tgt.items()} }")
    delta_F = {k: F_tgt[k] - F_src[k] for k in FEATURES}
    print(f"== delta_F      ={ {k: round(v, 3) for k, v in delta_F.items()} }")

    operators = sorted(
        name for name in v6.OPERATOR_NAMES
        if "forecast" in (v6.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
    )

    rows: list[dict[str, Any]] = []
    for op in operators:
        params = v1._default_params(op, period) if hasattr(v1, "_default_params") else {
            "period": period, "cycles": 3, "min_donors": 2}
        compiled = v1.make_compiled(op, params)
        src_s = v1.gain_at(roster, values, config, compiled, SRC_SUPPORT, baseline_cache)
        src_d = v1.gain_at(roster, values, config, compiled, SRC_DELAYED, baseline_cache)
        tgt_s = v1.gain_at(roster, values, config, compiled, TGT_SUPPORT, baseline_cache)
        tgt_d = v1.gain_at(roster, values, config, compiled, TGT_DELAYED, baseline_cache)
        if None in (src_s, src_d, tgt_s, tgt_d):
            rows.append({"operator": op, "status": "INSTRUMENT_INVALID"})
            continue
        rel_src = v1.relation_of(src_s, src_d)
        rel_tgt = v1.relation_of(tgt_s, tgt_d)
        flip = rel_src != rel_tgt
        rows.append({
            "operator": op,
            "source_relation": rel_src,
            "target_relation": rel_tgt,
            "flip": flip,
            "source_gains": [round(src_s, 4), round(src_d, 4)],
            "target_gains": [round(tgt_s, 4), round(tgt_d, 4)],
        })
        print(f"  {op:26s} src={rel_src:9s} tgt={rel_tgt:9s} "
              f"src_g={src_s:+.4f}/{src_d:+.4f} tgt_g={tgt_s:+.4f}/{tgt_d:+.4f} "
              f"{'FLIP' if flip else 'ok'}")

    valid = [r for r in rows if r.get("status") != "INSTRUMENT_INVALID"]
    flips = [r for r in valid if r["flip"]]
    stable = [r for r in valid if not r["flip"]]
    print(f"\n== valid={len(valid)}, flip={len(flips)}, stable={len(stable)}")
    print(f"== flip operators: {[r['operator'] for r in flips]}")

    # 每个特征在 flip vs stable 上的 |ΔF| 对比——翻转是否由特征变化解释
    print("\n== per-feature |delta_F| by flip status ==")
    per_feature: dict[str, dict[str, float]] = {}
    for feat in FEATURES:
        flip_d = [abs(delta_F[feat]) for r in flips]
        stable_d = [abs(delta_F[feat]) for r in stable]
        flip_mean = sum(flip_d) / len(flip_d) if flip_d else 0.0
        stable_mean = sum(stable_d) / len(stable_d) if stable_d else 0.0
        per_feature[feat] = {"flip_mean_delta": round(flip_mean, 4),
                             "stable_mean_delta": round(stable_mean, 4),
                             "ratio": round(flip_mean / stable_mean, 2) if stable_mean > 0 else None}
        print(f"  {feat:38s} flip|dF|={flip_mean:.4f}  stable|dF|={stable_mean:.4f}  ratio={per_feature[feat]['ratio']}")

    # 判定：存在 flip 均值显著大于 stable 均值的特征 → Program-specific 敏感特征有依据
    candidates = [feat for feat, v in per_feature.items()
                  if v["ratio"] is not None and v["ratio"] >= 2.0 and v["flip_mean_delta"] > 0]
    if candidates and len(flips) >= 2:
        verdict = f"PROGRAM_SPECIFIC_VIEW_SUPPORTED_SENSITIVE_FEATURES={candidates}"
    elif not candidates and len(flips) == 0:
        verdict = "NO_FLIP_IN_SAMPLE_STABLE"
    else:
        verdict = "FLIP_NOT_EXPLAINED_BY_CURRENT_FEATURES"
    print(f"\n== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-program-pattern-view-premise",
            "domain": domain,
            "timeline": {"src_support": SRC_SUPPORT, "src_delayed": SRC_DELAYED,
                         "tgt_support": TGT_SUPPORT, "tgt_delayed": TGT_DELAYED},
            "F_source": {k: round(v, 3) for k, v in F_src.items()},
            "F_target": {k: round(v, 3) for k, v in F_tgt.items()},
            "delta_F": {k: round(v, 3) for k, v in delta_F.items()},
            "operator_rows": rows,
            "flip_operators": [r["operator"] for r in flips],
            "per_feature_delta_by_flip": per_feature,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
