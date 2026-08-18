"""工作包 V1：NN5 局部窗口敏感性重测（零 LLM，2026-08-08）。

审查裁决第 2 步（NN5 分支）：局部窗口梯度存在（6 水平），用修正后的
Observation（origin 前最近 192 步窗口的最大缺失 run，跨序列 median 保留
多样性——不用全局累计最大）重测算子敏感性。

算子：NN5 扫描 B+C+ 4 个（impute_ssm/outlier_iqr/period_median_complete/winsorize）
     + B+C- 翻转 3 个（impute_ar/impute_ema/impute_fft）。

产出：每个算子的"效果 vs 局部缺失水平"关系——若单调 → Observation
candidate（⟨算子, 敏感特征=局部缺失, 方向⟩）；否则该特征在 NN5 上不是
敏感特征。只产出 candidate，不生成 Skill（审查裁决）。

用法：
  python evaluation/functional/run_v1_nn5_local_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402

WINDOW = 192
OPERATORS = [
    "impute_ssm", "outlier_iqr", "period_median_complete", "winsorize",
    "impute_ar", "impute_ema", "impute_fft",
]
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_nn5_local_sensitivity_report.json")


def local_missing_feature(values: Mapping[str, np.ndarray], origin: int) -> dict[str, float]:
    """局部窗口特征：每序列 [origin-WINDOW, origin) 窗口的最大缺失 run。

    跨序列用 median（保留梯度多样性）+ max 辅助——不用全局累计最大。
    """
    per_series_runs: list[int] = []
    for array in values.values():
        arr = np.asarray(array, dtype=np.float64)
        lo = max(0, origin - WINDOW)
        window = arr[lo:origin]
        mask = ~np.isfinite(window)
        best = cur = 0
        for m in mask:
            cur = cur + 1 if m else 0
            best = max(best, cur)
        per_series_runs.append(best)
    return {
        "median_window_max_missing_run": float(statistics.median(per_series_runs)),
        "max_window_max_missing_run": float(max(per_series_runs)),
        "series_with_missing_fraction": float(
            sum(1 for r in per_series_runs if r > 0) / len(per_series_runs)
        ),
    }


def spearman(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 NN5 local-window sensitivity")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    domain = "nn5"

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # NN5 max_len ~791：origin 需 origin+96 <= 791 且 origin >= WINDOW
    max_len = max(int(len(v)) for v in values.values())
    origins = [o for o in range(200, max_len - 96 + 1, 48)]
    print(f"== nn5: origins={origins}, max_len={max_len}")

    per_op: dict[str, list[dict[str, Any]]] = {op: [] for op in OPERATORS}
    for origin in origins:
        F = local_missing_feature(values, origin)
        for op in OPERATORS:
            from run_w2_operator_scan import _default_params
            compiled = v1.make_compiled(op, _default_params(op, period))  # B2：与经验同源
            s = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            d = v1.gain_at(roster, values, config, compiled, origin + 48, baseline_cache)
            if s is None or d is None:
                continue
            per_op[op].append({"origin": origin, "F": F, "support_gain": s, "delayed_gain": d})

    results: dict[str, Any] = {}
    for op in OPERATORS:
        rows = per_op[op]
        if len(rows) < 3:
            results[op] = {"status": "INSUFFICIENT", "n": len(rows)}
            print(f"  {op:22s} INSUFFICIENT (n={len(rows)})")
            continue
        xs = [r["F"]["median_window_max_missing_run"] for r in rows]
        ys = [r["support_gain"] for r in rows]
        rho = spearman(xs, ys)
        levels = sorted({float(x) for x in xs})
        # 按 X 分组均值（保留梯度的真实分组）
        group_means: dict[float, float] = {}
        for x, y in zip(xs, ys):
            group_means.setdefault(float(x), []).append(y)  # type: ignore[union-attr]
        group_means = {k: sum(v) / len(v) for k, v in group_means.items()}
        results[op] = {
            "n": len(rows),
            "x_levels": levels,
            "level_count": len(levels),
            "spearman_rho": round(rho, 3),
            "group_means": {str(k): round(v, 4) for k, v in sorted(group_means.items())},
            "curve": [
                {"missing": round(r["F"]["median_window_max_missing_run"], 2),
                 "support": round(r["support_gain"], 4),
                 "delayed": round(r["delayed_gain"], 4)} for r in rows
            ],
        }
        print(f"  {op:22s} levels={levels} rho={rho:+.3f} "
              f"group_means={ {str(k): round(v, 3) for k, v in sorted(group_means.items())} }")

    # Observation candidate：|rho| >= 0.6 且 >= 3 个水平 → 敏感特征候选
    # B3：候选判定加 group_means 单调性检查（与 docstring 一致——单调才候选）
    def _is_monotonic(gm: dict[str, float]) -> bool:
        keys = sorted(gm)
        if len(keys) < 3:
            return False
        vals = [gm[k] for k in keys]
        inc = all(vals[i + 1] >= vals[i] for i in range(len(vals) - 1))
        dec = all(vals[i + 1] <= vals[i] for i in range(len(vals) - 1))
        return inc or dec

    candidates = [
        op for op, r in results.items()
        if isinstance(r, dict) and r.get("n", 0) >= 3
        and r["level_count"] >= 3 and abs(r["spearman_rho"]) >= 0.6
        and _is_monotonic({float(k): float(v) for k, v in r["group_means"].items()})
    ]
    verdict = (
        f"OBSERVATION_CANDIDATES={candidates}" if candidates
        else "NO_MONOTONIC_SENSITIVITY_TO_LOCAL_MISSING"
    )
    print(f"\n== verdict: {verdict}")
    print("   (candidate 仅是 Observation candidate，需自然 held-in + Target Support 再确认，不生成 Skill)")

    out = root / REPORT_OUT_REL.with_name("w1_nn5_local_sensitivity_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-nn5-local-sensitivity",
            "domain": domain,
            "window": WINDOW,
            "feature": "median_window_max_missing_run",
            "origins": origins,
            "per_operator": results,
            "observation_candidates": candidates,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
