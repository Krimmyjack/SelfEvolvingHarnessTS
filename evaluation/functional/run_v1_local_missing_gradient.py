"""工作包 V1：局部窗口缺失梯度检查（零 LLM，2026-08-08）。

审查裁决第 1 步：先检查已有自然数据在 series/interval/local-window 层
是否其实存在缺失梯度——而不是直接跳合成梯度。

- 对 GEFCom + NN5：每序列、每代表 origin，取 origin 前最近 W 步局部窗口，
  算窗口内最大缺失 run + 缺失率（取代 dataset-level cumulative max）。
- 对比：全局累计最大（现有 extract_F）vs 局部窗口的缺失水平多样性。
- 判定：局部窗口存在 >2 个缺失水平 → 梯度存在，Observation 应改用局部窗口特征
  （修正 extract_F，Program-specific 敏感性可重测）；
  局部窗口也只有 1-2 个水平 → 自然数据确实无梯度，才轮到合成受控梯度。

用法：
  python evaluation/functional/run_v1_local_missing_gradient.py
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

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402

WINDOW = 192  # 与 v6 context_length 一致的局部窗口
ORIGINS = (200, 400, 600, 832, 928)
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_local_missing_gradient_report.json")


def max_missing_run(mask: np.ndarray) -> int:
    """布尔 mask 内的最大连续 True 段长度。"""
    best = 0
    cur = 0
    for m in mask:
        if m:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 local-window missing gradient check")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()

    results: dict[str, Any] = {}
    for domain in ("gefcom", "nn5"):
        config = dict(v6.DATASET_CONFIGS[domain])
        roster, values = v6._fixed_roster(root, config)
        max_len = max(int(len(v)) for v in values.values())
        usable_origins = [o for o in ORIGINS if o <= max_len]  # MED-4：窗口在 origin 之前

        # 全局累计最大（现有 extract_F 语义）
        global_max_by_origin: dict[int, float] = {}
        # 局部窗口：每 (origin, 序列) 的最大缺失 run + 缺失率
        local_runs: dict[int, list[int]] = {o: [] for o in usable_origins}
        local_rates: dict[int, list[float]] = {o: [] for o in usable_origins}
        for origin in usable_origins:
            # 全局：累计前缀最大
            all_runs = []
            for array in values.values():
                mask = ~np.isfinite(np.asarray(array[:origin], dtype=np.float64))
                all_runs.append(max_missing_run(mask))
            global_max_by_origin[origin] = float(max(all_runs)) if all_runs else 0.0
            # 局部：最近 WINDOW 窗口
            for array in values.values():
                window = np.asarray(array[origin - WINDOW: origin], dtype=np.float64)
                mask = ~np.isfinite(window)
                local_runs[origin].append(max_missing_run(mask))
                local_rates[origin].append(float(mask.mean()))

        # 水平多样性
        global_levels = sorted({round(v, 1) for v in global_max_by_origin.values()})
        local_levels = sorted({float(r) for o in usable_origins for r in local_runs[o]})
        per_origin_local_levels = {
            o: sorted({float(r) for r in local_runs[o]}) for o in usable_origins
        }
        results[domain] = {
            "global_cumulative_max_by_origin": {
                str(o): global_max_by_origin[o] for o in usable_origins
            },
            "global_levels": global_levels,
            "local_window_levels": local_levels,
            "local_window_level_count": len(local_levels),
            "per_origin_local_levels": {str(o): v for o, v in per_origin_local_levels.items()},
            "local_rate_by_origin": {
                str(o): round(statistics.median(local_rates[o]), 4) for o in usable_origins
            },
            "series_count": len(values),
        }
        print(f"== {domain}: global levels={global_levels}, "
              f"local-window levels={local_levels} (n={len(local_levels)}), "
              f"per-origin={ {str(o): v for o, v in per_origin_local_levels.items()} }")

    # 判定
    verdicts = {}
    for domain, r in results.items():
        if r["local_window_level_count"] > 2:
            verdicts[domain] = "LOCAL_GRADIENT_EXISTS"
        elif r["local_window_level_count"] == 2:
            verdicts[domain] = "LOCAL_BINARY_ONLY"
        else:
            verdicts[domain] = "NO_LOCAL_GRADIENT"
    print(f"\n== verdicts: {verdicts}")
    overall = all(v == "LOCAL_GRADIENT_EXISTS" for v in verdicts.values())
    print(f"== overall: {'LOCAL_GRADIENT_EXISTS_BOTH' if overall else 'INSUFFICIENT_LOCAL_GRADIENT'}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-local-missing-gradient-check",
            "window": WINDOW,
            "origins": list(ORIGINS),
            "per_domain": results,
            "verdicts": verdicts,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
