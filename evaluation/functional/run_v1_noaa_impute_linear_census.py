"""NOAA_IMPUTE_LINEAR_CENSUS（2026-08-13：用户核查修正——此前
impute_linear 的"behavior=0 ≡ baseline"检查测的是 _default_params 的
strength=1.0（=baseline 完整线性插值）；而 NOAA A5/A3 实际产生 +0.267
的 Workflow 是 impute_linear(strength=0.5)（w1_noaa_a5_vs_a3_report.
json winner_program）。

本 census 以正确参数绑定（strength=0.5）重做 per-series Batch
census，并补测 full-cohort 与 delayed 稳定性（原报告 delayed_gain
0.0 同为错误参数绑定所测——不表征 strength=0.5 的 delayed utility）。

装置（P2-v3 同构）：NOAA 20 series（12 train + 8 eval）+
train_series_scope（per-source-series 干预、相同 eval 测下游）+
origins {600, 632, 664, 696, 728}（公开 Context missingness 窗口）+
Risk-contract 修复后 verifier（impute 可执行）。delayed offset=48
（与 A5/A3 run 同：Target @600 → delayed @648）。

检查（用户裁决 2026-08-13——不换数据、不新增框架、零 LLM）：
  1. per-series 正/负/中性分布（M=0.005）；
  2. 6 公开特征留一可分性（全正 → 天然可分）；
  3. full-cohort 稳定：≥1 origin 全 roster 干预 gain ≥ M 且无 ≤ -M
     （含 origin 600 原 +0.267 位点复现）；
  4. delayed 稳定：material 单元（|g|≥M）delayed 符号一致 ≥60%
     且 delayed full-cohort 无 ≤ -M。

verdict（预注册）：
  NOAA_IMPUTE_LINEAR_FAMILY_CANDIDATE : 正方向存在（重复正向或可分）
    + full-cohort 稳定 + delayed 稳定 → 方向性 Source family
  NO_REPEATED_POSITIVE : 无正向单元
  FLIP_UNSEPARABLE     : 正负翻转且当前特征不可分 → 正式关闭
    NOAA missingness family
  FULL_COHORT_UNSTABLE : full-cohort 无正或翻负
  DELAYED_UNSTABLE     : delayed 方向翻转
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_noaa_impute_linear_census.py
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

from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_noaa_impute_linear_census_report.json"
DOMAIN = "noaa"
ORIGINS = (600, 632, 664, 696, 728)
DELAYED_OFFSET = 48
M = 0.005
OP = "impute_linear"
# 参数绑定修复（用户核查 2026-08-13）：A5/A3 winner_program 的实际参数
OP_PARAMS: dict[str, Any] = {"strength": 0.5}
FEATURES = ("missing_fraction", "longest_missing_run_fraction",
            "estimated_region_start_fraction",
            "estimated_region_end_fraction", "period_reliability",
            "period_change_score")
BASE_CACHE: dict[int, float] = {}


def _eval(origin: int, roster_full, values, cfg,
          scope: frozenset | None) -> tuple[float | None, int]:
    """impute_linear(strength=0.5) 在 origin 的 gain（scope=None →
    full-cohort；否则 per-series 干预）。返回 (gain, behavior_point_
    count)——异常 → (None, 0)。"""
    compiled = v1.make_compiled(OP, OP_PARAMS)
    try:
        if origin not in BASE_CACHE:
            base = v6._evaluate(roster_full, values, None, cfg,
                                origin=origin)
            BASE_CACHE[origin] = float(base["mean_smase"])
        kwargs: dict[str, Any] = ({} if scope is None
                                  else {"train_series_scope": scope})
        cand = v6._evaluate(roster_full, values, compiled, cfg,
                            origin=origin, **kwargs)
        gain = BASE_CACHE[origin] - float(cand["mean_smase"])
        return gain, int(cand.get("behavior_point_count") or 0)
    except Exception:
        return None, 0


def _loo_separable(units) -> bool:
    """6 公开特征单特征规则留一（同 Wave 2 / impute_fft census）——
    负向单元能否被 held-out series 规则命中且正向不误标。全正 → 天然
    可分。"""
    negs = [u for u in units if u["cls"] == "NEGATIVE"]
    if not negs:
        return True  # 全正——方向明确
    series_ids = sorted({u["series"] for u in units})
    hits = 0
    pos_false = 0
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
    max_len = max(len(values[s]) for s in series_ids[:12])

    units = []
    unavailable = []
    for sid in series_ids[:12]:
        arr = values[sid]
        for origin in ORIGINS:
            if origin + 48 > len(arr):
                continue
            g, behavior = _eval(origin, roster_full, values, cfg,
                                frozenset({sid}))
            pub = extract_public_features(arr[:origin],
                                          task_kind="forecast")
            if g is None or behavior <= 0:
                unavailable.append(f"{sid[:8]}@{origin}")
                continue
            gd = None
            if origin + DELAYED_OFFSET + 48 <= len(arr):
                gd, _bd = _eval(origin + DELAYED_OFFSET, roster_full,
                                values, cfg, frozenset({sid}))
            units.append({
                "series": sid, "origin": origin, "gain": g,
                "delayed_gain": gd,
                "cls": ("POSITIVE" if g >= M
                        else "NEGATIVE" if g < -M else "NEUTRAL"),
                **{k: pub.get(k) for k in FEATURES}})
        print(f"== sid {sid[:8]}: {sum(1 for u in units if u['series'] == sid)}"
              f" units", flush=True)

    # full-cohort（无 scope——全部 12 train series 干预）+ delayed
    fc: dict[int, float | None] = {}
    fc_delayed: dict[int, float | None] = {}
    for origin in ORIGINS:
        g, _b = _eval(origin, roster_full, values, cfg, None)
        fc[origin] = g
        if origin + DELAYED_OFFSET + 48 <= max_len:
            gd, _bd = _eval(origin + DELAYED_OFFSET, roster_full,
                            values, cfg, None)
            fc_delayed[origin] = gd
        else:
            fc_delayed[origin] = None
        print(f"== full-cohort @{origin}: {g}"
              f"  delayed: {fc_delayed[origin]}", flush=True)

    pos_series = sorted({u["series"] for u in units
                         if u["cls"] == "POSITIVE"})
    neg_series = sorted({u["series"] for u in units
                         if u["cls"] == "NEGATIVE"})
    separable = _loo_separable(units)
    legal_exec = not unavailable

    # full-cohort 稳定：≥1 origin gain ≥ M 且无 ≤ -M
    fc_gains = [g for g in fc.values() if g is not None]
    fc_stable = bool(any(g >= M for g in fc_gains)
                     and not any(g <= -M for g in fc_gains))
    # delayed 稳定：material 单元（|g|≥M）delayed 符号一致 ≥60%
    # 且 delayed full-cohort 无 ≤ -M
    mat = [u for u in units if abs(u["gain"]) >= M
           and u.get("delayed_gain") is not None]
    agree = sum(1 for u in mat
                if (u["gain"] >= M) == (u["delayed_gain"] >= M))
    agree_frac = (agree / len(mat)) if mat else None
    fc_d_gains = [g for g in fc_delayed.values() if g is not None]
    delayed_stable = bool(mat and agree_frac is not None
                          and agree_frac >= 0.6
                          and not any(g <= -M for g in fc_d_gains))

    report: dict[str, Any] = {
        "experiment_id": "v1-noaa-impute-linear-census",
        "note": "NOAA block-missingness × impute_linear(strength=0.5) 候选 "
                "Source family census（用户核查修正——正确参数绑定——"
                "P2-v3 装置同构——development exposure——零新 Claim）",
        "apparatus": {"domain": DOMAIN, "op": OP, "op_params": OP_PARAMS,
                      "origins": list(ORIGINS),
                      "delayed_offset": DELAYED_OFFSET,
                      "roster_split": {"n_train": 12, "n_eval": 8}},
        "n_unavailable": len(unavailable),
        "unavailable_sample": unavailable[:8],
        "distribution": {c: sum(1 for u in units if u["cls"] == c)
                         for c in ("POSITIVE", "NEGATIVE", "NEUTRAL")},
        "units": units,
        "pos_series": [s[:12] for s in pos_series],
        "neg_series": [s[:12] for s in neg_series],
        "full_cohort": {"support": fc, "delayed": fc_delayed,
                        "original_site_600": fc.get(600),
                        "original_a5_a3_gain": 0.2668149728462872},
        "stability": {
            "full_cohort_stable": fc_stable,
            "full_cohort_gains": fc_gains,
            "delayed_stable": delayed_stable,
            "delayed_agreement": {"n_material": len(mat),
                                  "n_agree": agree,
                                  "fraction": agree_frac},
            "delayed_full_cohort_gains": fc_d_gains},
        "checks": {"n_pos_series": len(pos_series),
                   "n_neg_series": len(neg_series),
                   "legal_execution": legal_exec,
                   "context_separable": separable},
    }
    # verdict（用户裁决 2026-08-13：重复正向或可分 → 方向性 family；
    # 翻转不可分 → 正式关闭 NOAA missingness family）
    if not legal_exec:
        verdict = "PROTOCOL_FAILURE"
    elif not pos_series:
        verdict = "NO_REPEATED_POSITIVE"
    elif neg_series and not separable:
        verdict = "FLIP_UNSEPARABLE"
    elif not fc_stable:
        verdict = "FULL_COHORT_UNSTABLE"
    elif not delayed_stable:
        verdict = "DELAYED_UNSTABLE"
    else:
        verdict = "NOAA_IMPUTE_LINEAR_FAMILY_CANDIDATE"
    report["verdict"] = verdict
    print("== distribution:", report["distribution"])
    print("== pos_series:", len(pos_series), "neg_series:", len(neg_series),
          "separable:", separable)
    print("== full_cohort:", fc, "delayed:", fc_delayed,
          "stable:", fc_stable)
    print("== delayed agreement:", agree_frac, "stable:", delayed_stable)
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
