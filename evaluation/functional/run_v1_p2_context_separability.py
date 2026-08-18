"""P2_CONTEXT_SEPARABILITY（Wave 2，2026-08-13：现有公开 Pattern 是否
足以区分 impute_fft 的负向风险——轻量确定性检查，非聚类。用户任务书）。

方法：impute_fft 负向单元（NEGATIVE 8 个）vs 正向/中性对照；只允许：
  - 现有 numeric bins（6 公开特征）；
  - 单特征规则（阈值）；
  - 必要时最多两个特征的 conjunction（AND）；
  - 按 series 留一验证（规则在 N-1 series 形成 → held-out series 验证
    ——避免同一 series 的不同 origin 互相证明）。

检验目标：使用执行前 Context 检索时，能否在未参与规则形成的 series
上正确返回负向风险经验，同时不把所有正向对照都判成风险。

最小通过条件：
  - ≥2 个独立 held-out series 的负向单元检索到 RISK_PRIOR；
  - 不是所有正向对照都被标记为风险；
  - 规则只依赖部署时可见字段；
  - Scope 不是 dataset ID 或 series ID。

verdict（预注册）：
  EXISTING_CONTEXT_SUPPORTS_SIGNED_RETRIEVAL
  EXISTING_OBSERVATION_UNIDENTIFIABLE

用法：
  python evaluation/functional/run_v1_p2_context_separability.py
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
REPORT_REL = E2 / "w1_p2_context_separability_report.json"
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
        return BASE_CACHE[origin] - float(cand["mean_smase"])
    except Exception:
        return None


def _build_units(root) -> list[dict[str, Any]]:
    cfg = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, cfg)
    series_ids = [r["series_uid"] for r in roster]
    roster_full = ([{"series_uid": s, "role": "train"}
                    for s in series_ids[:12]]
                   + [{"series_uid": s, "role": "eval"}
                      for s in series_ids[12:]])
    units = []
    for sid in series_ids[:12]:
        arr = values[sid]
        for origin in ORIGINS:
            g = _gain_series(sid, OP, origin, roster_full, values, cfg)
            pub = extract_public_features(arr[:origin],
                                          task_kind="forecast")
            units.append({
                "series": sid, "origin": origin, "gain": g,
                "cls": ("NEGATIVE" if g is not None and g < -M
                        else "POSITIVE" if g is not None and g >= M
                        else "NEUTRAL"),
                **{k: pub.get(k) for k in FEATURES}})
    return units


def _rule_hits(rule, unit) -> bool:
    """规则命中：单特征 (feat, op, thr) 或双特征 AND。"""
    f1, op1, t1 = rule[0]
    hit = (unit[f1] < t1) if op1 == "lt" else (unit[f1] > t1)
    if len(rule) == 2:
        f2, op2, t2 = rule[1]
        hit = hit and ((unit[f2] < t2) if op2 == "lt"
                       else (unit[f2] > t2))
    return bool(hit)


def _fit_rule(train_units):
    """单特征阈值规则（网格）→ 若单特征 recall 不足再试双特征 AND。
    分数 = neg_recall − 1.5 × pos_false_rate（轻量确定性）。"""
    negs = [u for u in train_units if u["cls"] == "NEGATIVE"]
    poss = [u for u in train_units if u["cls"] != "NEGATIVE"]
    if not negs:
        return None, 0.0, 0.0
    best = None
    best_score = -1.0
    for f in FEATURES:
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
        return None, 0.0, 0.0
    rec = sum(1 for u in negs if _rule_hits(best, u)) / len(negs)
    fpr = sum(1 for u in poss if _rule_hits(best, u)) / len(poss)
    if rec < 0.6 and len(negs) >= 3:
        # 双特征 AND：第一规则命中集内再试第二特征
        inner = [u for u in negs if _rule_hits(best, u)]
        for f in FEATURES:
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
    if best is None:
        return None, 0.0, 0.0
    rec = sum(1 for u in negs if _rule_hits(best, u)) / len(negs)
    fpr = sum(1 for u in poss if _rule_hits(best, u)) / len(poss)
    return best, rec, fpr


def main() -> int:
    root = PROJECT_ROOT
    units = _build_units(root)
    series_ids = sorted({u["series"] for u in units})
    neg_series = sorted({u["series"] for u in units
                         if u["cls"] == "NEGATIVE"})

    # 留一验证：规则在 N-1 series 形成 → held-out series 验证
    loo = []
    for holdout in series_ids:
        train_units = [u for u in units if u["series"] != holdout]
        rule, rec, fpr = _fit_rule(train_units)
        held = [u for u in units if u["series"] == holdout]
        neg_hits = [u for u in held if u["cls"] == "NEGATIVE"
                    and rule is not None and _rule_hits(rule, u)]
        pos_marked = [u for u in held if u["cls"] == "POSITIVE"
                      and rule is not None and _rule_hits(rule, u)]
        loo.append({"holdout": holdout[:8],
                    "n_neg_units": sum(1 for u in held
                                       if u["cls"] == "NEGATIVE"),
                    "neg_risk_hits": len(neg_hits),
                    "pos_false_marked": len(pos_marked),
                    "rule": rule})
    # 通过条件（任务书最小）
    hits_series = sorted({l["holdout"] for l in loo
                          if l["n_neg_units"] > 0
                          and l["neg_risk_hits"] > 0})
    all_pos_false = [l for l in loo if l["pos_false_marked"] > 0]
    n_heldout_neg_series = sum(1 for l in loo if l["n_neg_units"] > 0)
    ok = bool(
        len(hits_series) >= 2
        and len(hits_series) == n_heldout_neg_series  # 全部负向 series 命中
        and len(all_pos_false) < len(loo) * 0.5  # 不是所有正向被误标
    )
    verdict = ("EXISTING_CONTEXT_SUPPORTS_SIGNED_RETRIEVAL" if ok
               else "EXISTING_OBSERVATION_UNIDENTIFIABLE")
    report = {
        "experiment_id": "v1-p2-context-separability",
        "note": "Wave 2：现有公开 Pattern 是否足以区分 impute_fft 负向"
                "风险（单/双特征规则 + 按 series 留一——轻量确定性，"
                "非聚类）——development exposure——零新 Claim",
        "apparatus": {"domain": DOMAIN, "op": OP, "origins": list(ORIGINS),
                      "features": list(FEATURES)},
        "units_summary": {"n": len(units),
                          "distribution": {c: sum(1 for u in units
                                                  if u["cls"] == c)
                                           for c in
                                           ("POSITIVE", "NEGATIVE",
                                            "NEUTRAL")}},
        "loo": loo,
        "checks": {"n_heldout_neg_series": n_heldout_neg_series,
                   "risk_hit_series": hits_series,
                   "pos_false_marked_series": [l["holdout"]
                                               for l in all_pos_false]},
        "verdict": verdict,
    }
    print("== loo: " + json.dumps(loo, ensure_ascii=False, default=str))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
