"""FLIP_OBSERVATION_PROBE_DEV（Wave 4c-2，2026-08-13：正负翻转不可分
→ 两个预注册最小 TS Observation 探针诊断——确定性、零 LLM、零新
评估（纯特征计算）。纯诊断——不动生产特征提取；若分离成立，生产
接线（Observation 面编辑 + 最小 Scoped Skill）是下一分支。

预注册探针（部署可见——只用 origin 前的公开数据）：
  P1 outlier_density_recent_48 : 最近 48 点中 |x − median(168 窗口)| >
     3 × 1.4826 × MAD(168 窗口) 的比例（重尾质量）
  P2 calendar_phase : (origin mod 168) / 168（周内相位——确定性公开）

判定（预注册）：任一探针单特征零误差分开 material 正/负窗口 →
  FLIP_PROBE_SEPARABLE（报告探针/阈值/方向/margin）；
  两个都失败 → FLIP_PROBE_NOT_SEPARABLE → 按停止条件（连续两个最小
  Observation 无法区分 Utility flip）abstain 并转晨间报告。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import signed_radius as resolver  # noqa: E402

from run_v1_batch_census_dev import (  # noqa: E402
    DEV_ORIGINS,
    DEV_SERIES,
    _load_series,
)

M = resolver.MATERIAL_THRESHOLD
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_flip_observation_probe_dev_report.json"


def _outlier_density(series: np.ndarray, origin: int) -> float:
    """P1：最近 48 点重尾质量（部署可见——只用 origin 前数据）。"""
    window = series[max(0, origin - 168):origin]
    recent = series[origin - 48:origin]
    med = float(np.median(window))
    mad = float(np.median(np.abs(window - med))) * 1.4826
    if mad <= 0:
        return 0.0
    return float(np.mean(np.abs(recent - med) > 3.0 * mad))


def main() -> int:
    root = PROJECT_ROOT
    census = json.loads((root / "artifacts/functional/e2"
                         / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))
    windows: dict[tuple[str, int], float] = {}
    for sid, rounds in (census.get("development_rounds") or {}).items():
        for r in rounds:
            for cid, gain in r.get("probes") or []:
                if cid == "cand_winsorize" and gain is not None:
                    windows[(sid, r["origin"])] = float(gain)
    pos = {k: v for k, v in windows.items() if v >= M}
    neg = {k: v for k, v in windows.items() if v < -M}
    probes: dict[str, dict[tuple[str, int], float]] = {
        "outlier_density_recent_48": {},
        "calendar_phase": {},
    }
    for sid in DEV_SERIES:
        arr = _load_series(root, sid)
        for origin in DEV_ORIGINS:
            if (sid, origin) not in windows:
                continue
            probes["outlier_density_recent_48"][(sid, origin)] = \
                _outlier_density(arr, origin)
            probes["calendar_phase"][(sid, origin)] = \
                float((origin % 168) / 168.0)

    results: list[dict[str, Any]] = []
    for name, feat in probes.items():
        pvals = sorted(feat[k] for k in pos if k in feat)
        nvals = sorted(feat[k] for k in neg if k in feat)
        if not pvals or not nvals:
            continue
        entry: dict[str, Any] = {"probe": name,
                                 "pos_range": [min(pvals), max(pvals)],
                                 "neg_range": [min(nvals), max(nvals)],
                                 "values": {f"{k[0]}@{k[1]}": round(v, 6)
                                            for k, v in sorted(feat.items())}}
        if max(nvals) < min(pvals):
            entry["separable"] = True
            entry["threshold"] = (max(nvals) + min(pvals)) / 2.0
            entry["direction"] = "neg_below"
            entry["margin"] = min(pvals) - max(nvals)
        elif max(pvals) < min(nvals):
            entry["separable"] = True
            entry["threshold"] = (max(pvals) + min(nvals)) / 2.0
            entry["direction"] = "pos_below"
            entry["margin"] = min(nvals) - max(pvals)
        else:
            entry["separable"] = False
        results.append(entry)
    best = next((r for r in results if r.get("separable")), None)
    verdict = "FLIP_PROBE_SEPARABLE" if best else \
        "FLIP_PROBE_NOT_SEPARABLE"
    report = {
        "experiment_id": "v1-flip-observation-probe-dev",
        "note": "Wave 4c-2：两个预注册最小 Observation 探针的正负翻转"
                "可分性诊断（纯特征计算——零 LLM 零新评估——不动生产"
                "特征提取）",
        "probes": results,
        "best": best,
        "verdict": verdict,
    }
    for r in results:
        print("== probe " + r["probe"]
              + ": separable=" + str(r.get("separable", False))
              + " pos=" + str(r["pos_range"]) + " neg="
              + str(r["neg_range"]))
    print(f"== verdict: {verdict}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
