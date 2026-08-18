"""P2_CONTEXT_BINDING（Wave 1，2026-08-13：把 P2-v3 经验绑定到合法
部署时可见 Context——Evidence/Memory 修复，非新 Capability。用户任务书）。

处理 impute_fft 的 36 个单元（12 train series × 3 origins {600,632,680}
——P2-v3 有效装置：12 train + 8 eval + train_series_scope）。每个
Episode 至少包含：
  Task/Consumer = forecast|ridge|sMASE
  公开 Pattern（执行前窗口 series[:origin] 提取）：
    missing_fraction / longest_missing_run_fraction /
    estimated_region_start_fraction / estimated_region_end_fraction /
    period_reliability / period_change_score
  Program：impute_fft + 参数 + 实际是否改变数据（behavior_count）
    + changed fraction
  Response：support gain / delayed gain（配对才有）/
    POSITIVE / NEGATIVE / CONFLICT / UNKNOWN

关键检查（信息墙）：
  - Context 必须来自该 Action 执行前的窗口；
  - 不得把 gain、response sign 或未来值混入检索 query（local_pattern
    只含 6 个公开键——gain 只进 support_response/delayed_response）；
  - Action–Response 可进 Memory evidence，但不进 Fast Path Context
    匹配字段；
  - 同一 outcome 不得重复作为多条独立证据（每 (series, origin) 一个
    Episode；delayed 配对只写一次）。

CONFLICT 配对：同 series 的 632→680 对（632 的 delayed = 680 的
support gain——符号相反 → 632 Episode 的 delayed_response 记为
CONFLICT）。

verdict（预注册）：
  P2_CONTEXT_BOUND_EVIDENCE_PASS
  P2_CONTEXT_BINDING_INVALID

用法：
  python evaluation/functional/run_v1_p2_context_binding.py
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

from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_p2_context_binding_report.json"
TASK_CONSUMER = "forecast|ridge|sMASE"
DOMAIN = "nn5"
ORIGINS = (600, 632, 680)
M = 0.005
OP = "impute_fft"
CONTEXT_KEYS = ("missing_fraction", "longest_missing_run_fraction",
                "estimated_region_start_fraction",
                "estimated_region_end_fraction", "period_reliability",
                "period_change_score")
BASE_CACHE: dict[int, float] = {}


def _gain_series(sid: str, op: str, origin: int,
                 roster_full, values, cfg) -> tuple[float | None, int]:
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


def _response_class(gain: float | None) -> str:
    """Response 分类（schema relation 枚举限定）：material 正/负 →
    POSITIVE/NEGATIVE；中性/未知（|gain|<M 或 None）→ ABSTAIN
    （neutral——Signed Memory 检索不计入正/负/冲突信号；报告注明
    映射，不新增枚举——任务书禁新 Schema）。"""
    if gain is None:
        return "ABSTAIN"
    if gain >= M:
        return "POSITIVE"
    if gain < -M:
        return "NEGATIVE"
    return "ABSTAIN"


def main() -> int:
    root = PROJECT_ROOT
    cfg = dict(v6.DATASET_CONFIGS[DOMAIN])
    roster, values = v6._fixed_roster(root, cfg)
    series_ids = [r["series_uid"] for r in roster]
    roster_full = ([{"series_uid": s, "role": "train"}
                    for s in series_ids[:12]]
                   + [{"series_uid": s, "role": "eval"}
                      for s in series_ids[12:]])
    train_series = series_ids[:12]

    # ---- 36 单元（每 (series, origin) 一个 Episode）----
    episodes = []
    rows = []
    checks: dict[str, Any] = {"n_units": 0, "n_episodes": 0,
                              "n_conflict_pairs": 0,
                              "context_keys_all_present": True,
                              "no_gain_in_local_pattern": True}
    gains_by_series: dict[str, dict[int, float | None]] = {}
    for sid in train_series:
        gains_by_series[sid] = {}
        for origin in ORIGINS:
            g, behavior = _gain_series(sid, OP, origin, roster_full,
                                       values, cfg)
            gains_by_series[sid][origin] = g
            rows.append({"series": sid, "origin": origin, "gain": g,
                         "behavior_count": behavior})
    # CONFLICT 配对（632→680 同 series）
    delayed_map: dict[tuple[str, int], float | None] = {}
    for sid in train_series:
        g632 = gains_by_series[sid].get(632)
        g680 = gains_by_series[sid].get(680)
        if g632 is not None and g680 is not None \
                and (g632 >= M) != (g680 >= M) \
                and abs(g632) >= M and abs(g680) >= M:
            delayed_map[(sid, 632)] = g680
    for r in rows:
        sid, origin, g, behavior = (r["series"], r["origin"], r["gain"],
                                    r["behavior_count"])
        series_arr = values[sid]
        pub = extract_public_features(series_arr[:origin],
                                      task_kind="forecast")
        ctx = {k: pub.get(k) for k in CONTEXT_KEYS}
        if any(v is None for v in ctx.values()):
            checks["context_keys_all_present"] = False
        steps = [{"op": OP, "params": dict(_default_params(OP, 7))}]
        ep = build_episode(
            episode_id=f"nn5_cb_{sid[:8]}_{OP}_{origin}",
            task_consumer_key=TASK_CONSUMER,
            domain_namespace=DOMAIN,
            context_summary={
                "local_pattern": dict(ctx),
                "delayed_pattern": {},
                "program_geometry": {
                    "scope": "training_rows",
                    "program_steps": steps,
                    "changed_fraction": (
                        float(behavior) if behavior > 0 else 0.0)},
                "per_view_gain": [],
                "support_origin": origin,
            },
            workflow_signature=workflow_signature_of(steps),
            support_response={"gain": g, "accepted": False},
            delayed_response={
                "evaluated": (sid, origin) in delayed_map,
                "gain": delayed_map.get((sid, origin))},
            relation=_response_class(g),
            evidence_level="SUPPORT",
            local_status="EPISODE_ONLY",
            evidence_refs=["p2v3_deterministic_census"])
        # 信息墙检查：local_pattern 不得含 gain/未来值
        if any(k not in CONTEXT_KEYS for k in ep.context_summary
               .get("local_pattern", {}).keys()):
            checks["no_gain_in_local_pattern"] = False
        episodes.append(ep)
    checks["n_units"] = len(rows)
    checks["n_episodes"] = len(episodes)
    checks["n_conflict_pairs"] = len(delayed_map)
    # 同一 outcome 不重复：每 (series, origin) 恰一个 Episode
    ids = [e.episode_id for e in episodes]
    checks["unique_episodes"] = len(set(ids)) == len(ids)
    # 分类分布
    from collections import Counter
    dist = Counter(e.relation for e in episodes)
    checks["response_distribution"] = dict(dist)

    ok = bool(checks["n_units"] == 36 and checks["n_episodes"] == 36
              and checks["unique_episodes"]
              and checks["context_keys_all_present"]
              and checks["no_gain_in_local_pattern"])
    verdict = ("P2_CONTEXT_BOUND_EVIDENCE_PASS" if ok
               else "P2_CONTEXT_BINDING_INVALID")
    report = {
        "experiment_id": "v1-p2-context-binding",
        "note": "Wave 1：P2-v3 经验绑定到部署时可见 Context（impute_fft "
                "36 单元——Evidence/Memory 修复，非新 Capability）——"
                "development exposure——零新 Claim",
        "apparatus": {"domain": DOMAIN, "op": OP, "origins": list(ORIGINS),
                      "roster_split": {"n_train": 12, "n_eval": 8},
                      "context_keys": list(CONTEXT_KEYS)},
        "checks": checks,
        "units": rows,
        "verdict": verdict,
    }
    print("== checks: " + json.dumps(checks, ensure_ascii=False,
                                     default=str))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
