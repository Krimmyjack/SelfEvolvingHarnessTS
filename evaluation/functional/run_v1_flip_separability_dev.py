"""FLIP_SEPARABILITY_DEV（Wave 4c-1，2026-08-13：正负翻转 Context 可分性
检查——确定性、零 LLM、零新评估（特征提取 + census 读数）。

背景：winsorize 在 development block（T1/T10/T100/T101 × 600/792/888/
984）上翻转——8 个 material 正窗口 / 6 个 material 负窗口（census）。
P2 表分支：同 Program 正负翻转且 Context 可区分 → 最小 Scoped Skill；
不可区分 → 最小 Observation probe；仍不可分 → abstain。

本 runner 只做第一步（可分性检查）：
  特征 = window_context 的 recent./change. 键（部署可见 cohort 口径）+
  extract_public_features（公开特征）——每个 (series, origin) 窗口一组。
  判定（预注册）：存在单一特征阈值把 material 正/负窗口**零误差**分开
  → SEPARABLE（报告特征/阈值/margin）；否则 NOT_SEPARABLE。
  margin 并列时按预注册顺序取第一个（特征键名排序）。

verdict（预注册）：
  FLIP_CONTEXT_SEPARABLE / FLIP_CONTEXT_NOT_SEPARABLE / PROTOCOL_FAILURE
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
from run_v1_kdd2018_natural_slow_update import _request  # noqa: E402
from run_v1_batch_census_dev import (  # noqa: E402
    DEV_ORIGINS,
    DEV_SERIES,
    EVAL_SERIES,
    _dev_executor,
    _load_series,
)

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

M = resolver.MATERIAL_THRESHOLD
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_flip_separability_dev_report.json"


def main() -> int:
    root = PROJECT_ROOT
    census = json.loads((root / "artifacts/functional/e2"
                         / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))
    # 全部 winsorize 窗口读数（census development_rounds）
    windows: dict[tuple[str, int], float] = {}
    for sid, rounds in (census.get("development_rounds") or {}).items():
        for r in rounds:
            for cid, gain in r.get("probes") or []:
                if cid == "cand_winsorize" and gain is not None:
                    windows[(sid, r["origin"])] = float(gain)
    pos = {k: v for k, v in windows.items() if v >= M}
    neg = {k: v for k, v in windows.items() if v < -M}
    neutral = {k: v for k, v in windows.items()
               if -M <= v < M}
    # 特征提取（部署可见口径）
    feats: dict[tuple[str, int], dict[str, float]] = {}
    for sid in DEV_SERIES:
        ex, vals = _dev_executor(root, sid)
        series_arr = _load_series(root, sid)
        for origin in DEV_ORIGINS:
            if (sid, origin) not in windows:
                continue
            f: dict[str, float] = {}
            ctx = resolver.window_context(
                {sid: series_arr}, origin, 24)
            for k, v in ctx.items():
                if isinstance(v, (int, float)):
                    f[k] = float(v)
            try:
                pub = dict(extract_public_features(
                    series_arr[:origin], task_kind="forecast"))
            except Exception as exc:  # noqa: BLE001
                pub = {"error": str(exc)}
            for k, v in pub.items():
                if isinstance(v, (int, float)):
                    f[f"pub.{k}"] = float(v)
            feats[(sid, origin)] = f
    # 可分性：单一特征阈值零误差分开 pos/neg
    all_keys = sorted({k for f in feats.values() for k in f})
    separable: list[dict[str, Any]] = []
    for key in all_keys:
        pvals = sorted(feats[k][key] for k in pos if key in feats[k])
        nvals = sorted(feats[k][key] for k in neg if key in feats[k])
        if not pvals or not nvals:
            continue
        # 正窗口特征值全部 > 负窗口特征值（或全部 <）——零误差阈值存在
        if max(nvals) < min(pvals):
            threshold = (max(nvals) + min(pvals)) / 2.0
            direction = "neg_below"
            margin = min(pvals) - max(nvals)
            separable.append({"feature": key, "threshold": threshold,
                              "direction": direction, "margin": margin,
                              "pos_range": [min(pvals), max(pvals)],
                              "neg_range": [min(nvals), max(nvals)]})
        elif max(pvals) < min(nvals):
            threshold = (max(pvals) + min(nvals)) / 2.0
            direction = "pos_below"
            margin = min(nvals) - max(pvals)
            separable.append({"feature": key, "threshold": threshold,
                              "direction": direction, "margin": margin,
                              "pos_range": [min(pvals), max(pvals)],
                              "neg_range": [min(nvals), max(nvals)]})
    separable.sort(key=lambda s: (-s["margin"], s["feature"]))
    verdict = ("FLIP_CONTEXT_SEPARABLE" if separable
               else "FLIP_CONTEXT_NOT_SEPARABLE")
    report = {
        "experiment_id": "v1-flip-separability-dev",
        "note": "Wave 4c-1：winsorize 正负翻转的 Context 可分性检查"
                "（确定性零 LLM；特征=部署可见口径）",
        "windows": {"positive": {f"{k[0]}@{k[1]}": v for k, v in pos.items()},
                    "negative": {f"{k[0]}@{k[1]}": v for k, v in neg.items()},
                    "neutral": {f"{k[0]}@{k[1]}": v
                                for k, v in neutral.items()}},
        "n_features": len(all_keys),
        "separable_features": separable,
        "best": (separable[0] if separable else None),
        "verdict": verdict,
    }
    print("== positive windows: "
          + json.dumps({f"{k[0]}@{k[1]}": v for k, v in pos.items()}))
    print("== negative windows: "
          + json.dumps({f"{k[0]}@{k[1]}": v for k, v in neg.items()}))
    print("== separable features (top 5): "
          + json.dumps(separable[:5], ensure_ascii=False, default=str))
    print(f"== verdict: {verdict}")
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
