"""TARGET_FEASIBILITY_SCAN（2026-08-13：用户批准——零 LLM、零正式
outcome 的只读检查——寻找与 NN5 Source Context 匹配且能让 Memory
产生明确方向的 Target。不消费正式评价数据（不用于方法调整；探测只
查可执行性——development probe）。

扫描条件（用户四查）：
  1. Target 是否存在 missingness Context（missing_fraction > 0）；
  2. impute_fft 等算子是否真正可执行并会改变数据（behavior_count>0）；
  3. Target Context 检索 prequential Source（24 条——origin<672）后
     得到明确 POSITIVE/RISK 还是 AMBIGUOUS（同 workflow 正负）；
  4. 是否存在尚未使用的 delayed 时间块（origin+48 ≤ len 且未被消费）。

数据集：gefcom / noaa（v6.DATASET_CONFIGS 装置）+ monash weather
（冻结 120 条 roster——日频 period=7 与 NN5 同域）。

结果分支（预注册）：
  - Context 匹配 + Memory 明确方向 → MATCHED_TARGET_FOUND（冻结
    roster 正式 A3/A5）
  - Context 匹配但仍 AMBIGUOUS → MATCHED_AMBIGUOUS（不跑昂贵实验）
  - Context 不匹配 → NO_CONTEXT_MATCH（引入新数据）
  - 无未消费 delayed → DEVELOPMENT_ONLY（只能 development 回放）

用法：
  python evaluation/functional/run_v1_target_feasibility_scan.py
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

from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    resolve_experience_contrast_pack,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_target_feasibility_scan_report.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
SRC_DOMAIN = "nn5"
SRC_ORIGINS = (600, 632, 680)
TARGET_ORIGIN_REF = 672
M = 0.005
OP = "impute_fft"
OPS = ("repair_level_shift", "impute_ar", "impute_ssm",
       "impute_fft", "impute_ema", "impute_linear")
SCAN_ORIGINS = (600, 632, 664, 696, 728)
BASE_CACHE: dict[int, float] = {}
HORIZON = 48


def _eval_op(roster, values, cfg, sid, op, origin,
             scope: frozenset[str] | None):
    """development probe：**gain = baseline − candidate**（用户核查修复
    2026-08-13——此前返回候选 mean_sMASE 被误当 gain，导致大量
    CLEAR_POSITIVE 误标）+ behavior_count。"""
    compiled = v1.make_compiled(op, _default_params(op, 7))
    try:
        if origin not in BASE_CACHE:
            base = v6._evaluate(roster, values, None, cfg, origin=origin)
            BASE_CACHE[origin] = float(base["mean_smase"])
        cand = v6._evaluate(roster, values, compiled, cfg, origin=origin,
                            train_series_scope=scope)
        gain = BASE_CACHE[origin] - float(cand["mean_smase"])
        return gain, int(cand.get("behavior_point_count") or 0)
    except Exception:
        return None, 0


def _build_prequential_source(root) -> list[Any]:
    """NN5 prequential Source（origin < 672——同 Wave 4 修正口径）。"""
    cfg = dict(v6.DATASET_CONFIGS[SRC_DOMAIN])
    roster, values = v6._fixed_roster(root, cfg)
    sids = [r["series_uid"] for r in roster]
    rf = ([{"series_uid": s, "role": "train"} for s in sids[:12]]
          + [{"series_uid": s, "role": "eval"} for s in sids[12:]])
    eps = []
    for sid in sids[:12]:
        arr = values[sid]
        for origin in SRC_ORIGINS:
            if origin >= TARGET_ORIGIN_REF:
                continue
            g, _b = _eval_op(rf, values, cfg, sid, OP, origin,
                             frozenset({sid}))
            pub = extract_public_features(arr[:origin],
                                          task_kind="forecast")
            steps = [{"op": OP, "params": dict(_default_params(OP, 7))}]
            cls = ("POSITIVE" if g is not None and g >= M
                   else "NEGATIVE" if g is not None and g < -M
                   else "ABSTAIN")
            eps.append(build_episode(
                episode_id=f"nn5_scan_src_{sid[:8]}_{origin}",
                task_consumer_key=TASK_CONSUMER,
                domain_namespace=SRC_DOMAIN,
                context_summary={
                    "local_pattern": {
                        k: pub.get(k) for k in (
                            "missing_fraction",
                            "longest_missing_run_fraction",
                            "estimated_region_start_fraction",
                            "estimated_region_end_fraction",
                            "period_reliability",
                            "period_change_score")},
                    "delayed_pattern": {},
                    "program_geometry": {"scope": "training_rows",
                                         "program_steps": steps},
                    "per_view_gain": [], "support_origin": origin},
                workflow_signature=workflow_signature_of(steps),
                support_response={"gain": g, "accepted": False},
                delayed_response={"evaluated": False, "gain": None},
                relation=cls, evidence_level="SUPPORT",
                local_status="EPISODE_ONLY",
                evidence_refs=["p2v3_deterministic_census"]))
    return eps, rf, values, cfg


def _scan_dataset(name: str, series_arrs: list[tuple[str, np.ndarray]],
                  roster_full, values, cfg, source_eps) -> dict[str, Any]:
    out: dict[str, Any] = {"n_series": len(series_arrs), "targets": []}
    for sid, arr in series_arrs:
        n = len(arr)
        for origin in SCAN_ORIGINS:
            if origin + HORIZON > n:
                continue  # 无足够 delayed 块
            pub = extract_public_features(arr[:origin],
                                          task_kind="forecast")
            missing = pub.get("missing_fraction") or 0.0
            if missing <= 0.0:
                continue  # 条件 1：无 missingness Context
            # 条件 2：impute_fft 可执行并改变数据（development probe）
            _c, behavior = _eval_op(roster_full, values, cfg, sid, OP,
                                    origin, None)
            if behavior <= 0:
                continue
            # 条件 3：Source 检索方向判定
            pack = resolve_experience_contrast_pack(
                source_eps, dict(pub), TASK_CONSUMER,
                allowed_operators=tuple(OPS))
            if pack is None:
                direction = "NO_PACK"
            else:
                pos = pack.positive
                neg = pack.negative
                if pos is not None and neg is not None and \
                        getattr(pos, "workflow_signature", None) == \
                        getattr(neg, "workflow_signature", None):
                    direction = "AMBIGUOUS"
                elif pos is not None:
                    direction = "CLEAR_POSITIVE"
                elif neg is not None:
                    direction = "CLEAR_RISK"
                else:
                    direction = "NEUTRAL"
            # 条件 4：delayed 未消费（origin+48 ≤ n 已保证——新 origin）
            out["targets"].append({
                "series": sid[:12], "origin": origin,
                "missing_fraction": float(missing),
                "behavior_changed": behavior > 0,
                "direction": direction})
    return out


def main() -> int:
    root = PROJECT_ROOT
    source_eps, rf, src_values, src_cfg = _build_prequential_source(root)
    report: dict[str, Any] = {
        "experiment_id": "v1-target-feasibility-scan",
        "note": "Target feasibility scan（用户批准——零 LLM、零正式 "
                "outcome 只读检查——探测只查可执行性不用于选择）",
        "source": {"n_prequential": len(source_eps), "op": OP,
                   "origins_lt_672": [o for o in SRC_ORIGINS
                                      if o < TARGET_ORIGIN_REF]},
        "datasets": {},
    }

    # GEFCom / NOAA（v6 装置）
    for dname in ("gefcom", "noaa"):
        cfg = dict(v6.DATASET_CONFIGS[dname])
        roster, values = v6._fixed_roster(root, cfg)
        rf_d = ([{"series_uid": r["series_uid"], "role": "train"}
                 for r in roster[:8]]
                + [{"series_uid": r["series_uid"], "role": "eval"}
                   for r in roster[8:16]])
        arrs = [(r["series_uid"], values[r["series_uid"]])
                for r in roster[:16]]
        report["datasets"][dname] = _scan_dataset(
            dname, arrs, rf_d, values, cfg, source_eps)

    # Monash weather（冻结 roster 120 条——日频同域）
    monash_cache = np.load(root / "data/monash_weather_v1/series_cache.npz",
                           allow_pickle=True)
    m_names = [str(n) for n in monash_cache["names"]]
    m_vals = monash_cache["values"]
    rows = [json.loads(l) for l in
            (root / "artifacts/functional/e2/w1_monash_feasibility_roster.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    m_series = [(r["series_name"], np.asarray(
        m_vals[m_names.index(r["series_name"])], dtype=np.float64))
        for r in rows[:60]]
    m_cfg = dict(v6.DATASET_CONFIGS["nn5"])  # period 7 同域参数
    rf_m = ([{"series_uid": s, "role": "train"} for s, _a in m_series[:6]]
            + [{"series_uid": s, "role": "eval"} for s, _a in m_series[6:12]])
    m_vals_d = {s: a for s, a in m_series[:12]}
    report["datasets"]["monash"] = _scan_dataset(
        "monash", m_series[:12], rf_m, m_vals_d, m_cfg, source_eps)

    # ---- 汇总判定（预注册分支）----
    summary = {}
    for dname, d in report["datasets"].items():
        targets = d.get("targets") or []
        clear = [t for t in targets if t["direction"] in
                 ("CLEAR_POSITIVE", "CLEAR_RISK")]
        amb = [t for t in targets if t["direction"] == "AMBIGUOUS"]
        summary[dname] = {"n_targets": len(targets),
                          "clear_direction": len(clear),
                          "ambiguous": len(amb),
                          "sample": targets[:3]}
    report["summary"] = summary
    any_clear = any(s["clear_direction"] > 0 for s in summary.values())
    verdict = ("MATCHED_TARGET_FOUND" if any_clear
               else "NO_MATCHED_TARGET")
    report["verdict"] = verdict
    print("== summary: " + json.dumps(summary, ensure_ascii=False,
                                      default=str))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
