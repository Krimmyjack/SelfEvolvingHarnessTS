"""NOAA_IMPUTE_CENSUS（2026-08-13：NOAA block-missingness × impute_linear
作为潜在新 Source family——Batch Action–Response census。用户裁决
下一步：检查重复正向、合法执行、可观察 Context——非单窗口偶然。

装置（P2-v3 同构）：NOAA 20 series（12 train + 8 eval）+ train_series_
scope（per-source-series 干预、相同 eval 测下游）+ origins {600, 632,
664, 696, 728}（公开 Context missingness 窗口）+ Risk-contract 修复后
verifier（impute 可执行）。

检查（用户 3 项）：
  1. 重复正向：≥2 独立 series 的 impute_linear material positive
     （非单窗口偶然）；
  2. 合法执行：全部 probe 通过（无 verifier_rejected）；
  3. Context 方向性：正负（若存在）是否可被 6 公开特征区分（单特征
     规则留一——同 Wave 2 方法）；全正向则方向天然明确。

verdict（预注册）：
  NOAA_IMPUTE_FAMILY_CANDIDATE : 重复正向 + 合法执行 + 方向明确
    （全正或可分）→ 冻结 Source Experience
  NO_REPEATED_POSITIVE         : 无 ≥2 series 重复正向
  FLIP_UNSEPARABLE             : 正负翻转且当前特征不可分 → 关闭 family
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_noaa_impute_census.py
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

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_noaa_impute_fft_census_report.json"
DOMAIN = "noaa"
ORIGINS = (600, 632, 664, 696, 728)
M = 0.005
OP = "impute_fft"
FEATURES = ("missing_fraction", "longest_missing_run_fraction",
            "estimated_region_start_fraction",
            "estimated_region_end_fraction", "period_reliability",
            "period_change_score")
BASE_CACHE: dict[int, float] = {}


def _eval(sid, op, origin, roster_full, values, cfg):
    compiled = v1.make_compiled(op, _default_params(op, 7))
    try:
        if origin not in BASE_CACHE:
            base = v6._evaluate(roster_full, values, None, cfg,
                                origin=origin)
            BASE_CACHE[origin] = float(base["mean_smase"])
        cand = v6._evaluate(roster_full, values, compiled, cfg,
                            origin=origin,
                            train_series_scope=frozenset({sid}))
        gain = BASE_CACHE[origin] - float(cand["mean_smase"])
        return gain, int(cand.get("behavior_point_count") or 0)
    except Exception:
        return None, 0


def _loo_separable(units) -> bool:
    """6 公开特征单特征规则留一（同 Wave 2）——负向单元能否被
    held-out series 规则命中且正向不误标。全正 → 天然可分。"""
    negs = [u for u in units if u["cls"] == "NEGATIVE"]
    if not negs:
        return True  # 全正——方向明确
    series_ids = sorted({u["series"] for u in units})
    hits = 0
    pos_false = 0
    n_neg_series = 0
    for holdout in series_ids:
        train = [u for u in units if u["series"] != holdout]
        neg_train = [u for u in train if u["cls"] == "NEGATIVE"]
        pos_train = [u for u in train if u["cls"] != "NEGATIVE"]
        best_rule = None
        best_score = -1.0
        for f in FEATURES:
            vals = sorted({u[f] for u in train if u[f] is not None})
            for a, b in zip(vals, vals[1:]):
                thr = (a + b) / 2.0
                for op in ("lt", "gt"):
                    def _hit(u, ff=f, oo=op, tt=thr):
                        return (u[ff] < tt) if oo == "lt" else (u[ff] > tt)
                    rec = sum(1 for u in neg_train if _hit(u)) / len(neg_train)
                    fpr = (sum(1 for u in pos_train if _hit(u))
                           / len(pos_train)) if pos_train else 0.0
                    score = rec - 1.5 * fpr
                    if score > best_score:
                        best_score = score
                        best_rule = (f, op, thr)
        held = [u for u in units if u["series"] == holdout]
        n_neg_series += sum(1 for u in held if u["cls"] == "NEGATIVE") > 0
        if best_rule is None:
            continue
        f1, op1, t1 = best_rule
        for u in held:
            hit = (u[f1] < t1) if op1 == "lt" else (u[f1] > t1)
            if hit and u["cls"] == "NEGATIVE":
                hits += 1
            if hit and u["cls"] == "POSITIVE":
                pos_false += 1
    n_neg = sum(1 for u in units if u["cls"] == "NEGATIVE")
    n_pos = sum(1 for u in units if u["cls"] == "POSITIVE")
    return bool(hits >= 2 and hits == n_neg and pos_false <= n_pos / 2)


def main() -> int:
    root = PROJECT_ROOT
    cfg = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, cfg)
    series_ids = [r["series_uid"] for r in roster]
    roster_full = ([{"series_uid": s, "role": "train"}
                    for s in series_ids[:12]]
                   + [{"series_uid": s, "role": "eval"}
                      for s in series_ids[12:]])
    units = []
    unavailable = []
    for sid in series_ids[:12]:
        arr = values[sid]
        for origin in ORIGINS:
            if origin + 48 > len(arr):
                continue
            g, behavior = _eval(sid, OP, origin, roster_full, values, cfg)
            pub = extract_public_features(arr[:origin],
                                          task_kind="forecast")
            if g is None or behavior <= 0:
                unavailable.append(f"{sid[:8]}@{origin}")
                continue
            units.append({
                "series": sid, "origin": origin, "gain": g,
                "cls": ("POSITIVE" if g >= M
                        else "NEGATIVE" if g < -M else "NEUTRAL"),
                **{k: pub.get(k) for k in FEATURES}})
    report: dict[str, Any] = {
        "experiment_id": "v1-noaa-impute-census",
        "note": "NOAA block-missingness × impute_fft 候选 Source family "
                "census（Risk-contract 修复后——P2-v3 装置同构——"
                "development exposure——零新 Claim）",
        "apparatus": {"domain": DOMAIN, "op": OP,
                      "origins": list(ORIGINS),
                      "roster_split": {"n_train": 12, "n_eval": 8}},
        "n_unavailable": len(unavailable),
        "unavailable_sample": unavailable[:8],
        "distribution": {c: sum(1 for u in units if u["cls"] == c)
                         for c in ("POSITIVE", "NEGATIVE", "NEUTRAL")},
        "units": units,
    }
    pos_series = sorted({u["series"] for u in units
                         if u["cls"] == "POSITIVE"})
    neg_series = sorted({u["series"] for u in units
                         if u["cls"] == "NEGATIVE"})
    report["pos_series"] = [s[:12] for s in pos_series]
    report["neg_series"] = [s[:12] for s in neg_series]
    repeated_positive = len(pos_series) >= 2
    legal_exec = not unavailable
    separable = _loo_separable(units)
    report["checks"] = {"repeated_positive_series": len(pos_series),
                        "legal_execution": legal_exec,
                        "context_separable": separable}
    if not repeated_positive:
        verdict = "NO_REPEATED_POSITIVE"
    elif not legal_exec:
        verdict = "PROTOCOL_FAILURE"
    elif not separable:
        verdict = "FLIP_UNSEPARABLE"
    else:
        verdict = "NOAA_IMPUTE_FAMILY_CANDIDATE"
    report["verdict"] = verdict
    print("== distribution:", report["distribution"])
    print("== pos_series:", len(pos_series), "neg_series:", len(neg_series),
          "separable:", separable)
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
