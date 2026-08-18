"""P2_OBSERVATION_GAP（Wave 3-B，2026-08-13：现有 Context 不可区分
（EXISTING_OBSERVATION_UNIDENTIFIABLE——规则漏 1960d9bd 负向单元）
→ 检查正负对照差异是否明确指向一个缺失观察——只允许一个机制相关
候选 Observation，重复 Wave 2 留一一次。用户任务书）。

候选 Observation（机制相关——与 missingness 修复效果直接相关）：
  a) missing_block_phase    ：缺失段中心相对周期(7)的相位
  b) seasonal_peak_overlap  ：缺失段是否覆盖周内高值日（季节峰值）
  c) missing_block_count    ：连续缺失段数量
  d) changed_fraction       ：Program 真正修改的训练窗口覆盖率（已有
     behavior_count——检查其区分力）

流程：
  1. 对每个候选：在 6 公开特征 + 该候选的集合上重跑留一验证
     （同 Wave 2 方法——单/双特征规则）；
  2. 若某候选使留一全过（全部负向 series 命中 + 正向误标 < 半数）：
     该 Observation 机制相关且有区分力 → 报告该候选 → verdict
     OBSERVATION_CANDIDATE_SUPPORTED（供 Wave 3-A 决定是否加）；
  3. 若无一候选可区分 → 关闭 family：
     NN5_IMPUTE_FFT_UNIDENTIFIABLE_WITH_CURRENT_OBSERVATIONS
     （不得继续堆特征）。

用法：
  python evaluation/functional/run_v1_p2_observation_gap.py
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
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
from run_w2_operator_scan import _default_params  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_p2_observation_gap_report.json"
DOMAIN = "nn5"
ORIGINS = (600, 632, 680)
M = 0.005
OP = "impute_fft"
FEATURES = ("missing_fraction", "longest_missing_run_fraction",
            "estimated_region_start_fraction",
            "estimated_region_end_fraction", "period_reliability",
            "period_change_score")
BASE_CACHE: dict[int, float] = {}


def _gain_series(sid, op, origin, roster_full, values, cfg):
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


def _missing_blocks(arr: np.ndarray, origin: int) -> list[tuple[int, int]]:
    """连续 NaN 缺失段（[start, end) 半开）——机制相关 Observation 的
    原始计算。"""
    mask = np.isnan(arr[:origin])
    blocks = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            blocks.append((start, i))
            start = None
    if start is not None:
        blocks.append((start, origin))
    return blocks


def _observation_features(arr: np.ndarray, origin: int,
                          period: int) -> dict[str, float]:
    """候选 Observation 计算（纯公开数据——执行前窗口）。"""
    blocks = _missing_blocks(arr, origin)
    out: dict[str, float] = {}
    if not blocks:
        out["missing_block_count"] = 0.0
        out["missing_block_phase"] = -1.0
        out["seasonal_peak_overlap"] = 0.0
        return out
    out["missing_block_count"] = float(len(blocks))
    # 缺失段中心相对周期相位（0..1）
    centers = [(b[0] + b[1]) / 2.0 for b in blocks]
    out["missing_block_phase"] = float((sum(centers) / len(centers)
                                        % period) / period)
    # 季节峰值覆盖：周内均值最高日是否被任一缺失段覆盖
    window = arr[max(0, origin - 8 * period):origin]
    valid = window[~np.isnan(window)]
    if valid.size >= period:
        day_means = [np.nanmean(window[d::period])
                     for d in range(period)]
        peak_day = int(np.nanargmax(day_means))
        covered = any(any((b[0] - (origin - len(window))) % period
                          == peak_day for _ in range(1))
                      or (b[1] - 1 - (origin - len(window))) % period
                      == peak_day for b in blocks)
        out["seasonal_peak_overlap"] = 1.0 if covered else 0.0
    else:
        out["seasonal_peak_overlap"] = 0.0
    return out


def _rule_hits(rule, unit) -> bool:
    f1, op1, t1 = rule[0]
    hit = (unit[f1] < t1) if op1 == "lt" else (unit[f1] > t1)
    if len(rule) == 2:
        f2, op2, t2 = rule[1]
        hit = hit and ((unit[f2] < t2) if op2 == "lt"
                       else (unit[f2] > t2))
    return bool(hit)


def _fit_rule(train_units, feat_set):
    negs = [u for u in train_units if u["cls"] == "NEGATIVE"]
    poss = [u for u in train_units if u["cls"] != "NEGATIVE"]
    if not negs:
        return None
    best = None
    best_score = -1.0
    for f in feat_set:
        vals = sorted({u[f] for u in train_units
                       if u[f] is not None})
        for a, b in zip(vals, vals[1:]):
            thr = (a + b) / 2.0
            for op in ("lt", "gt"):
                def _hit(u, ff=f, oo=op, tt=thr):
                    return (u[ff] < tt) if oo == "lt" else (u[ff] > tt)
                hits_neg = sum(1 for u in negs if _hit(u))
                hits_pos = sum(1 for u in poss if _hit(u))
                score = hits_neg / len(negs) - 1.5 * hits_pos / len(poss)
                if score > best_score:
                    best_score = score
                    best = ((f, op, thr),)
    if best is None:
        return None
    rec = sum(1 for u in negs if _rule_hits(best, u)) / len(negs)
    if rec < 0.6 and len(negs) >= 3:
        inner = [u for u in negs if _rule_hits(best, u)]
        for f in feat_set:
            if f == best[0][0]:
                continue
            vals2 = sorted({u[f] for u in train_units
                            if u[f] is not None})
            for a, b in zip(vals2, vals2[1:]):
                thr2 = (a + b) / 2.0
                for op2 in ("lt", "gt"):
                    rule2 = best + ((f, op2, thr2),)
                    hits_neg = sum(1 for u in negs
                                   if _rule_hits(rule2, u))
                    hits_pos = sum(1 for u in poss
                                   if _rule_hits(rule2, u))
                    score = hits_neg / len(negs) - 1.5 * hits_pos / len(poss)
                    if score > best_score:
                        best_score = score
                        best = rule2
    return best


def _loo(units, series_ids, feat_set):
    """留一验证（Wave 2 同款）。返回 (ok, details)。"""
    loo = []
    for holdout in series_ids:
        train_units = [u for u in units if u["series"] != holdout]
        rule = _fit_rule(train_units, feat_set)
        held = [u for u in units if u["series"] == holdout]
        neg_hits = sum(1 for u in held if u["cls"] == "NEGATIVE"
                       and rule is not None and _rule_hits(rule, u))
        pos_marked = sum(1 for u in held if u["cls"] == "POSITIVE"
                         and rule is not None and _rule_hits(rule, u))
        loo.append({"holdout": holdout[:8],
                    "n_neg": sum(1 for u in held
                                 if u["cls"] == "NEGATIVE"),
                    "neg_hits": neg_hits, "pos_marked": pos_marked,
                    "rule": rule})
    n_neg_series = sum(1 for l in loo if l["n_neg"] > 0)
    hit_series = sum(1 for l in loo
                     if l["n_neg"] > 0 and l["neg_hits"] > 0)
    pos_false_series = sum(1 for l in loo if l["pos_marked"] > 0)
    ok = bool(hit_series >= 2 and hit_series == n_neg_series
              and pos_false_series < len(series_ids) * 0.5)
    return ok, {"hit_series": hit_series, "n_neg_series": n_neg_series,
                "pos_false_series": pos_false_series, "loo": loo}


def main() -> int:
    root = PROJECT_ROOT
    cfg = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, cfg)
    series_ids = [r["series_uid"] for r in roster]
    roster_full = ([{"series_uid": s, "role": "train"}
                    for s in series_ids[:12]]
                   + [{"series_uid": s, "role": "eval"}
                      for s in series_ids[12:]])
    period = int(cfg.get("period", 7))

    units = []
    for sid in series_ids[:12]:
        arr = values[sid]
        for origin in ORIGINS:
            g, behavior = _gain_series(sid, OP, origin, roster_full,
                                       values, cfg)
            pub = extract_public_features(arr[:origin],
                                          task_kind="forecast")
            obs = _observation_features(arr, origin, period)
            units.append({
                "series": sid, "origin": origin, "gain": g,
                "cls": ("NEGATIVE" if g is not None and g < -M
                        else "POSITIVE" if g is not None and g >= M
                        else "NEUTRAL"),
                **{k: pub.get(k) for k in FEATURES},
                **obs,
                "changed_fraction": (float(behavior) if behavior > 0
                                     else 0.0)})
    obs_candidates = ("missing_block_phase", "seasonal_peak_overlap",
                      "missing_block_count", "changed_fraction")
    results = {}
    for cand in obs_candidates:
        ok, det = _loo(units, sorted({u["series"] for u in units}),
                       FEATURES + (cand,))
        results[cand] = {"ok": ok, **det}

    supported = [c for c, r in results.items() if r["ok"]]
    verdict = ("OBSERVATION_CANDIDATE_SUPPORTED" if supported
               else "NN5_IMPUTE_FFT_UNIDENTIFIABLE_WITH_CURRENT_OBSERVATIONS")
    report = {
        "experiment_id": "v1-p2-observation-gap",
        "note": "Wave 3-B：现有 Context 不可区分后检查机制相关候选 "
                "Observation（仅一个——重复 Wave 2 留一一次；不得堆"
                "特征）——development exposure——零新 Claim",
        "candidates": results,
        "verdict": verdict,
    }
    print("== candidates: " + json.dumps(
        {k: {"ok": v["ok"], "hit": v["hit_series"],
             "n_neg": v["n_neg_series"], "pos_false": v["pos_false_series"]}
         for k, v in results.items()}, ensure_ascii=False))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
