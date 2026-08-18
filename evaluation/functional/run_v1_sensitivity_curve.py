"""工作包 V1：算子级敏感性曲线（零 LLM，2026-08-08）。

Program-specific pattern view 的机制验证：算子的效果是否随候选特征
（maximum_missing_run_length 为主）单调变化。

- GEFCom 多个 origin（200..928，步长 48，不重叠）：每 origin 的缺失水平天然不同。
- 9 个算子（8 个跨切片翻转 + outlier_iqr 稳定）：每 origin 测 support/delayed gain。
- 每算子：support gain vs 缺失长度的相关性（Spearman + 单调性检查）。
- 判定：翻转算子效果随缺失单调 → Program-specific skill 可自动生成
  （⟨算子, 敏感特征, 方向⟩）；否则该特征不是敏感特征。

用法：
  python evaluation/functional/run_v1_sensitivity_curve.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402

OPERATORS = [
    "denoise_savgol", "denoise_wavelet", "fft_decompose", "hampel_filter",
    "smooth_ema", "smooth_ma", "outlier_iqr",
]
ORIGINS = list(range(200, 929, 48))  # 200..928；delayed 需 origin+96 <= 1024
FEATURE = "maximum_missing_run_length"
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_sensitivity_curve_report.json")


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation（ties 用平均秩）。"""
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

    rx = ranks(xs)
    ry = ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def monotonic_ratio(xs: list[float], ys: list[float]) -> float:
    """单调性：符号一致的相邻差比例（y 随 x 同向/反向）。"""
    if len(xs) < 3:
        return 0.0
    agree = 0
    total = 0
    for i in range(len(xs) - 1):
        dx = xs[i + 1] - xs[i]
        dy = ys[i + 1] - ys[i]
        if abs(dx) < 1e-9 or abs(dy) < 1e-9:
            continue
        total += 1
        if (dx > 0) == (dy > 0):
            agree += 1
    return agree / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 operator sensitivity curve")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    domain = "gefcom"

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}

    # 每 origin 的缺失水平 + 每算子效果
    per_origin_F: dict[int, dict[str, float]] = {}
    per_operator: dict[str, list[dict[str, Any]]] = {op: [] for op in OPERATORS}
    for origin in ORIGINS:
        F = v1.extract_F(values, config, origin)
        per_origin_F[origin] = F
        for op in OPERATORS:
            params = {"period": period, "cycles": 3, "min_donors": 2}
            compiled = v1.make_compiled(op, params)
            s = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
            d = v1.gain_at(roster, values, config, compiled, origin + 48, baseline_cache)
            if s is None or d is None:
                continue
            per_operator[op].append({
                "origin": origin,
                "missing_run_length": F[FEATURE],
                "support_gain": s,
                "delayed_gain": d,
            })

    print(f"== origins={ORIGINS}, missing levels: "
          f"{sorted(set(round(per_origin_F[o][FEATURE], 1) for o in ORIGINS))}")

    results: dict[str, Any] = {}
    for op in OPERATORS:
        rows = per_operator[op]
        if len(rows) < 3:
            results[op] = {"status": "INSUFFICIENT_DATA", "n": len(rows)}
            print(f"  {op:20s} INSUFFICIENT (n={len(rows)})")
            continue
        xs = [r["missing_run_length"] for r in rows]
        ys = [r["support_gain"] for r in rows]
        rho = spearman(xs, ys)
        mono = monotonic_ratio(xs, ys)
        # 符号翻转区间：低缺失 vs 高缺失
        half = len(rows) // 2
        low_mean = sum(r["support_gain"] for r in rows[:half]) / half
        high_mean = sum(r["support_gain"] for r in rows[half:]) / (len(rows) - half)
        flip = (low_mean > 0) != (high_mean > 0)
        results[op] = {
            "n": len(rows),
            "spearman_rho": round(rho, 3),
            "monotonic_ratio": round(mono, 3),
            "low_missing_mean_gain": round(low_mean, 4),
            "high_missing_mean_gain": round(high_mean, 4),
            "sign_flip_across_missing": flip,
            "curve": [{"missing": round(r["missing_run_length"], 1),
                       "support": round(r["support_gain"], 4),
                       "delayed": round(r["delayed_gain"], 4)} for r in rows],
        }
        print(f"  {op:20s} rho={rho:+.3f} mono={mono:.2f} "
              f"low_missing={low_mean:+.4f} high_missing={high_mean:+.4f} "
              f"flip={flip}")

    # 判定：|rho| >= 0.6 且单调一致 → 该算子的敏感特征 = missing_run_length
    sensitive = [
        op for op, r in results.items()
        if isinstance(r, dict) and r.get("n", 0) >= 3
        and abs(r["spearman_rho"]) >= 0.6 and r["monotonic_ratio"] >= 0.7
    ]
    if sensitive:
        verdict = f"PROGRAM_SPECIFIC_SENSITIVITY_SUPPORTED={sensitive}"
    else:
        verdict = "NO_STRONG_SENSITIVITY_TO_MISSING"
    print(f"\n== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-sensitivity-curve",
            "domain": domain,
            "feature": FEATURE,
            "origins": ORIGINS,
            "per_operator": results,
            "sensitive_operators": sensitive,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
