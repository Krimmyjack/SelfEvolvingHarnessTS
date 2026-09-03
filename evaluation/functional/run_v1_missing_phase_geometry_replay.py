"""V1 missingness × seasonal-phase geometry Observation 重放（零 LLM，2026-08-08）。

用户裁决（premise 确认 OBSERVATION_BLIND 后）：只实现这一个 regime Pattern——
missingness × seasonal-phase geometry，并对 denoise_stl 做零 LLM 重放。不接现有
三个素材（generic flatline / level-shift / _missing_window_context 粗粒度比例）。

特征（回答"这次缺失是否破坏 STL 所依赖的相位连续性"，比"缺失长度是不是 18"
更接近处理效果的原因）：
  1. last_gap_to_boundary    最近缺失段末端距 forecast boundary（=origin）的距离；
                             0 = 紧贴边界
  2. missing_phase           最近缺失段起/止位置相对 period 的 phase（绝对索引 mod period）
  3. series_fraction         同步受影响的 series 比例（recent 窗口内有缺失的序列占比）
  4. seasonal_pair_destroyed_fraction
                             recent 窗口内被缺失破坏的有效 seasonal-lag pair 比例
                             （pair (t, t+period) 中任一端缺失即破坏；窗口完整时全部有效）

块 = 冻结链 origin（W2 式不重叠块，支持+delayed 两处求值）：
  每块算 4 特征 + denoise_stl support/delayed gain（确定性重放，零 LLM）。
判定：flip 块（denoise_stl support<0）是否被 geometry 唯一区分于正向块。

用法：
  python evaluation/functional/run_v1_missing_phase_geometry_replay.py [--domain gefcom]
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
import run_v1_fastpath as v1  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

HORIZON = 48
WINDOW_LENGTH = 192  # recent 窗口 = [origin-192, origin)，与 eval context 一致
MATERIAL_THRESHOLD = v1.MATERIAL_THRESHOLD
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_missing_phase_geometry_replay_report.json")

# 冻结链 origin（与 3-rounds 脚本一致）：(origin, label)
BLOCKS = {
    "gefcom": [
        (640, "source_support"), (688, "source_delayed"),
        (736, "R1_support"), (784, "R1_delayed"),
        (832, "R2_support"), (880, "R2_delayed"),
        (928, "R3_support"), (976, "R3_delayed"),
    ],
    "nn5": [
        (536, "source_support"), (584, "source_delayed"),
        (632, "R1_support"), (680, "R1_delayed"),
        (728, "R2_support"),
    ],
}


def _missing_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """半开缺失段 [(start, stop), ...]（绝对索引）。"""
    padded = np.concatenate(([False], mask.astype(bool, copy=False), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(s), int(t)) for s, t in edges.reshape(-1, 2)]


def geometry_at(values: Mapping[str, np.ndarray], origin: int, period: int) -> dict[str, Any]:
    """recent 窗口 [origin-192, origin) 的缺失几何特征（部署可见）。"""
    gaps: list[int] = []
    phases_start: list[int] = []
    phases_stop: list[int] = []
    affected = 0
    destroyed_fractions: list[float] = []
    total_series = 0
    for array in values.values():
        total_series += 1
        raw = np.asarray(array, dtype=np.float64)
        lo = max(0, origin - WINDOW_LENGTH)
        window = raw[lo:origin]
        mask = ~np.isfinite(window)
        if mask.any():
            affected += 1
        runs = _missing_runs(mask)
        if runs:
            # 最近（末端最靠近 boundary）缺失段
            last = max(runs, key=lambda r: r[1])
            gaps.append(origin - (lo + last[1]))
            phases_start.append((lo + last[0]) % period)
            phases_stop.append((lo + last[1]) % period)
        # seasonal-lag pair 破坏比例：窗口内 (t, t+period) 对
        if window.size > period:
            left = window[: window.size - period]
            right = window[period:]
            total_pairs = int(left.size)
            destroyed = int(
                np.count_nonzero(~(np.isfinite(left) & np.isfinite(right)))
            )
            destroyed_fractions.append(destroyed / total_pairs if total_pairs else 0.0)
    return {
        "last_gap_to_boundary": int(np.median(gaps)) if gaps else None,
        "min_gap_to_boundary": int(min(gaps)) if gaps else None,
        "missing_phase_start_median": float(np.median(phases_start)) if phases_start else None,
        "missing_phase_stop_median": float(np.median(phases_stop)) if phases_stop else None,
        "series_fraction_affected": affected / total_series if total_series else 0.0,
        "seasonal_pair_destroyed_fraction": (
            float(np.median(destroyed_fractions)) if destroyed_fractions else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1 missingness x seasonal-phase geometry replay")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom", choices=tuple(BLOCKS))
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    period = int(config.get("period", 1))
    baseline_cache: dict[int, float] = {}
    compiled = v1.make_compiled("denoise_stl", _default_params("denoise_stl", period))

    blocks: list[dict[str, Any]] = []
    for origin, label in BLOCKS[domain]:
        geo = geometry_at(values, origin, period)
        support = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
        delayed = None
        if origin + HORIZON <= max(int(len(v)) for v in values.values()) - HORIZON:
            delayed = v1.gain_at(roster, values, config, compiled, origin + HORIZON, baseline_cache)
        relation = "POSITIVE" if (support is not None and support >= MATERIAL_THRESHOLD
                                  and delayed is not None and delayed >= MATERIAL_THRESHOLD) else (
            "NEGATIVE" if (support is not None and support < MATERIAL_THRESHOLD) else "CONFLICT")
        blocks.append({
            "origin": origin, "label": label,
            "geometry": geo,
            "denoise_stl_support_gain": support,
            "denoise_stl_delayed_gain": delayed,
            "relation": relation,
        })
        print(f"  {label:16s} origin={origin:4d} geo={geo} "
              f"support={support if support is None else round(support, 4)} "
              f"delayed={delayed if delayed is None else round(delayed, 4)} -> {relation}")

    # 判定：flip 块（support<0）是否被 geometry 唯一区分于正向块
    flip_blocks = [b for b in blocks if b["denoise_stl_support_gain"] is not None
                   and b["denoise_stl_support_gain"] < -MATERIAL_THRESHOLD]
    pos_blocks = [b for b in blocks if b["denoise_stl_support_gain"] is not None
                  and b["denoise_stl_support_gain"] >= MATERIAL_THRESHOLD]
    discriminates = True
    notes: list[str] = []
    for fb in flip_blocks:
        geo = fb["geometry"]
        markers = (
            geo["last_gap_to_boundary"] is not None and geo["last_gap_to_boundary"] <= WINDOW_LENGTH
            or geo["series_fraction_affected"] > 0.0
            or geo["seasonal_pair_destroyed_fraction"] > 0.0
        )
        if not markers:
            discriminates = False
            notes.append(f"flip block {fb['label']}@{fb['origin']} has NO missing-geometry marker")
    if pos_blocks:
        for fb in flip_blocks:
            same_as_positive = any(
                pb["geometry"]["series_fraction_affected"] == fb["geometry"]["series_fraction_affected"]
                and pb["geometry"]["last_gap_to_boundary"] == fb["geometry"]["last_gap_to_boundary"]
                for pb in pos_blocks
            )
            if same_as_positive:
                discriminates = False
                notes.append(
                    f"flip block {fb['label']}@{fb['origin']} shares missing-geometry "
                    f"with a positive block")
    if not flip_blocks:
        verdict = "NO_FLIP_BLOCK"
    elif not pos_blocks:
        # 无正向块可对照——geometry 恒在，无法判定判别性（退化 DISCRIMINATES 修正）
        verdict = "NO_POSITIVE_BASELINE"
    else:
        verdict = "GEOMETRY_DISCRIMINATES" if discriminates else "GEOMETRY_BLIND"
    print(f"\n== flip blocks: {[(b['label'], b['origin']) for b in flip_blocks]}"
          f" | positive blocks: {[(b['label'], b['origin']) for b in pos_blocks]}")
    for note in notes:
        print(f"   note: {note}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL.with_name(f"{REPORT_OUT_REL.stem}_{domain}{REPORT_OUT_REL.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-missing-phase-geometry-replay",
            "domain": domain,
            "window_length": WINDOW_LENGTH,
            "blocks": blocks,
            "flip_blocks": [b["label"] for b in flip_blocks],
            "positive_blocks": [b["label"] for b in pos_blocks],
            "notes": notes,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
