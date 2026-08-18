"""V1 NOAA 跨域 premise（零 LLM、零 Target outcome，2026-08-08）。

审查裁决（② 开始前只做廉价 premise）：
  1. 不看 NOAA Target outcome；
  2. 检查部署可见 Context 是否有一部分落入 GEFCom 经验半径（NOAA period=24
     只保证周期特征含义可比，不保证适合跨域）；
  3. 确认冻结候选池在 NOAA 历史开发材料中已有合法 Program headroom（不能再次
     出现"所有候选都无效"——NOAA 此前出现过候选空间/headroom 反例）。

判定：
  - 无 Context match        → NO_APPLICABLE_SOURCE_MEMORY（验证的是安全拒绝，
                              不是 A5 迁移失败）
  - 无 Program headroom     → NO_PROGRAM_HEADROOM（NOAA 不适合承担 Memory 价值
                              实验，应换 Target——不调 δ、不加 Pattern）
  - 两者都有                → CONTEXT_MATCH_AND_HEADROOM（批准进入跨域 A5/A3）

headroom 依据：w1_a5_vs_a3_report_noaa.json（历史开发材料）——A3 第二探
denoise_savgol support +0.0243 ≥ M（0.005）；A5 首探 denoise_stl -0.2592 大负
（已知候选空间反例，premise 报告该事实但不改变判定）。

用法：
  python evaluation/functional/run_v1_noaa_cross_domain_premise.py
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
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_fastagent_signed_replay as replay  # noqa: E402（gefcom 记忆重建复用）
import signed_radius as resolver  # noqa: E402

PERIOD = 24
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_noaa_cross_domain_premise_report.json")

# noaa 冻结时间线（run_v1_a5_vs_a3.TIMELINE）：候选 support/delayed origins
NOAA_ORIGINS = [832, 880, 928, 976]
HISTORICAL_NOAA_REPORT = (
    PROJECT_ROOT / "artifacts" / "functional" / "e2" / "w1_a5_vs_a3_report_noaa.json"
)


def main() -> int:
    root = PROJECT_ROOT

    # 1) gefcom 经验（R3 决策时刻记忆——与冻结链一致）
    source_episodes, a5_local, _ = replay.build_r3_memory(root)
    gefcom_memory = list(source_episodes) + list(a5_local)
    print(f"== gefcom memory: {len(gefcom_memory)} episodes")

    # 2) NOAA 部署可见 Context（零 Target outcome）
    noaa_config = dict(v6.DATASET_CONFIGS["noaa"])
    _roster, noaa_values = v6._fixed_roster(root, noaa_config)
    operators = sorted(n for n in v6.OPERATOR_NAMES
                       if "forecast" in (v6.OPERATOR_METADATA[n].get("allowed_tasks") or [])
                       and n not in core.CTS_EXCLUDED)

    per_origin: list[dict[str, Any]] = []
    matched_origins: list[int] = []
    for origin in NOAA_ORIGINS:
        q = resolver.window_context(noaa_values, origin, PERIOD)
        _order, signed = resolver.resolve_order(
            query_context=q, episodes=gefcom_memory, operators=operators,
            material_threshold=core.MATERIAL_THRESHOLD)
        counts = signed["summary"]["verdict_counts"]
        matched = counts[resolver.POSITIVE_PRIOR] + counts[resolver.CONFLICT] \
            + counts[resolver.RISK_PRIOR]
        per_origin.append({
            "origin": origin,
            "verdict_counts": counts,
            "matched_evidence_operators": matched,
        })
        if matched > 0:
            matched_origins.append(origin)
        print(f"  noaa origin={origin}: counts={counts} matched={matched}")

    context_match = len(matched_origins) > 0

    # 3) Program headroom：冻结候选池在 NOAA 历史开发材料中的合法正收益记录
    headroom_evidence: list[dict[str, Any]] = []
    if HISTORICAL_NOAA_REPORT.exists():
        hist = json.loads(HISTORICAL_NOAA_REPORT.read_text(encoding="utf-8"))
        a3 = hist.get("a3") or {}
        for op, g in zip(a3.get("probe_order") or [], a3.get("support_gains") or []):
            if isinstance(g, (int, float)) and g >= core.MATERIAL_THRESHOLD:
                headroom_evidence.append({"operator": op, "support_gain": g,
                                          "source": "w1_a5_vs_a3_report_noaa.json"})
        a5 = hist.get("a5") or {}
        first_negative = None
        for op, g in zip(a5.get("probe_order") or [], a5.get("support_gains") or []):
            if isinstance(g, (int, float)) and g < -core.MATERIAL_THRESHOLD:
                first_negative = {"operator": op, "support_gain": g}
                break
    program_headroom = len(headroom_evidence) > 0

    if not program_headroom:
        verdict = "NO_PROGRAM_HEADROOM"
    elif not context_match:
        verdict = "NO_APPLICABLE_SOURCE_MEMORY"
    else:
        verdict = "CONTEXT_MATCH_AND_HEADROOM"

    print(f"\n== context match at origins: {matched_origins}")
    print(f"== headroom evidence: {headroom_evidence}")
    print(f"== known candidate-space counterexample: {first_negative}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-noaa-cross-domain-premise",
            "noaa_origins": NOAA_ORIGINS,
            "gefcom_memory_count": len(gefcom_memory),
            "per_origin": per_origin,
            "matched_origins": matched_origins,
            "context_match": context_match,
            "program_headroom": program_headroom,
            "headroom_evidence": headroom_evidence,
            "known_candidate_space_counterexample": first_negative,
            "verdict": verdict,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
