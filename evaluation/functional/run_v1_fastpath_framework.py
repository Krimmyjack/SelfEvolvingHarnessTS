"""V1 Fast Path 最小框架：端到端组装（零新机制，deepseek 副本，2026-08-08）。

把已验证组件组装成一条可执行的 Fast Path 数据流（输入数据集配置，
输出结构化报告：探测顺序 / AUC / harm / Episode / 对照包 / Harness 视图）：

  Stage 1  FEATURE_EXTRACTION  extract_F（structural 视角，Source/Target 双点）
  Stage 2  SOURCE_EXPERIENCE   build_source_memory：Source 切片实测 26 forecast
                               算子 → ExperienceEpisode（F+P+R+pattern_view）
  Stage 3  PATTERN_MATCH        SignedEpisodeRetriever（pattern_view="structural"，
                               allowed_operators 逐算子检索）→ ContrastPack 对照包
                               + render_experience_pack 渲染 LLM 可见前缀
  Stage 4  HARNESS_STATE        CurrentHarnessState.apply_episode_status（当前视图）
  Stage 5  PROBING              compile_experienced_order（Memory 只改顺序，不硬排除）
                               + plan_target_support（Target Support 实测最终确认，
                               等预算 B=2，首个正向即停）；fixed 顺序对照
  Stage 6  EVALUATION           两个 Support 计划冻结后才打开 Target delayed outcome
                               → attach_delayed_outcomes → adaptation_auc / harm
  Stage 7  REPORT               结构化 JSON 报告

泄漏纪律（与 run_v1_fastpath / run_v1_a5_vs_a3 一致）：
- Episode 只来自 Source 切片；Target 探测结果只进评估，不写入 Memory；
- Target delayed 在计划冻结前不可读；
- 探测参数与经验构建参数必须一致（_default_params，参数失配是已修复的坑）。

用法：
  python evaluation/functional/run_v1_fastpath_framework.py --domain gefcom
  python evaluation/functional/run_v1_fastpath_framework.py --domain nn5
  python evaluation/functional/run_v1_fastpath_framework.py --domain noaa
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

import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_fastpath as v1  # noqa: E402
import SelfEvolvingHarnessTS.operators.registry as reg  # noqa: E402
from experience_memory import (  # noqa: E402
    CurrentHarnessState,
    SignedEpisodeRetriever,
    render_experience_pack,
)
from run_w2_operator_scan import _default_params  # noqa: E402

HORIZON = 48
# changes_target_space 算子与官方生成路径（v6）一致排除（外部巡检裁决）
CTS_EXCLUDED = tuple(
    n for n in v6.OPERATOR_NAMES
    if v6.OPERATOR_METADATA[n].get("changes_target_space")
)
MAX_TARGET_SUPPORT_EVALS = 2  # 等预算：Target Support probe 上限
PATTERN_VIEW = "structural"
TASK_CONSUMER_KEY = "forecast|ridge_smase"

# 冻结时间线（与 run_v1_a5_vs_a3 一致；noaa 与 gefcom 同为 max_len=1024，
# 复用同一合法时间线：976+48=1024 且 880+48=928 与 Target support 不重叠）
TIMELINE = {
    "gefcom": (832, 880, 928, 976),
    "nn5": (536, 584, 632, 680),
    "noaa": (832, 880, 928, 976),
}

REPORT_OUT_REL = Path("artifacts/functional/e2/v1_fastpath_framework_report.json")


# ---------------------------------------------------------------------------
# Stage 3：pattern_view 匹配 + 经验检索对照包（逐算子）
# ---------------------------------------------------------------------------

def build_contrast_packs(
    *,
    episodes: Sequence[Any],
    target_F: Mapping[str, float],
    roster: Sequence[Mapping[str, object]],
    config: Mapping[str, object],
    operators: Sequence[str],
    domain: str,
) -> list[dict[str, object]]:
    """对每个算子用现有 Retriever 检索对照包（positive/negative/conflict），
    并渲染成 LLM 可见前缀块（render_experience_pack 已验证格式）。"""
    period = int(config.get("period", 1))
    packs: list[dict[str, object]] = []
    for op in operators:
        params = _default_params(op, period)
        retriever = SignedEpisodeRetriever(
            episodes,
            task_consumer_key=TASK_CONSUMER_KEY,
            allowed_operators=(op,),
            pattern_view=PATTERN_VIEW,
        )
        pack = retriever.retrieve(
            v1.context_summary(target_F, roster, params),
            domain_namespace=domain,
        )
        rendered = render_experience_pack(pack.to_dict())
        packs.append(
            {
                "operator": op,
                "contrast_pack": pack.to_dict(),
                "rendered_reference_block": rendered,
            }
        )
    return packs


# ---------------------------------------------------------------------------
# Stage 4：Harness 当前视图
# ---------------------------------------------------------------------------

def build_harness_state(episodes: Sequence[Any]) -> dict[str, object]:
    """按 Episode 的 local_status 更新 CurrentHarnessState（Fast Path 唯一读取入口）。"""
    state = CurrentHarnessState()
    for episode in episodes:
        state.apply_episode_status(episode)
    return state.to_dict()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="V1 Fast Path minimal framework (assembled, zero new mechanisms)"
    )
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--domain", default="gefcom", choices=tuple(TIMELINE))
    args = parser.parse_args()
    root = args.root.resolve()
    domain = args.domain
    if domain not in TIMELINE:
        raise SystemExit(f"no frozen timeline for {domain}")
    src_support, src_delayed, tgt_support, tgt_delayed = TIMELINE[domain]

    config = dict(v6.DATASET_CONFIGS[domain])
    roster, values = v6._fixed_roster(root, config)
    max_len = max(int(len(array)) for array in values.values())
    period = int(config.get("period", 1))
    if tgt_delayed + HORIZON > max_len:
        raise SystemExit(f"{domain}: timeline exceeds max_len={max_len}")
    if src_delayed + HORIZON > tgt_support:
        raise SystemExit(f"{domain}: Source delayed overlaps Target Support")

    operators = sorted(
        name
        for name in reg.OPERATOR_NAMES
        if "forecast" in (reg.OPERATOR_METADATA.get(name, {}).get("allowed_tasks") or [])
        and name not in CTS_EXCLUDED
    )
    baseline_cache: dict[int, float] = {}

    # --- Stage 1: 特征提取 ---
    source_F = v1.extract_F(values, config, src_support)
    target_F = v1.extract_F(values, config, tgt_support)
    print(f"== [{domain}] Stage 1 FEATURE_EXTRACTION "
          f"(structural, source={src_support} target={tgt_support})")
    print(f"== source_F={ {k: round(v, 4) for k, v in source_F.items()} }")
    print(f"== target_F={ {k: round(v, 4) for k, v in target_F.items()} }")

    # --- Stage 2: Source 经验（只读 Source 切片，不读 Target future）---
    source_episodes, source_attempts = v1.build_source_memory(
        domain=domain,
        roster=roster,
        values=values,
        config=config,
        operators=operators,
        source_support_origin=src_support,
        source_delayed_origin=src_delayed,
        baseline_cache=baseline_cache,
    )
    print(f"== Stage 2 SOURCE_EXPERIENCE: {len(source_episodes)} episodes "
          f"(from Source [{src_support},{src_delayed + HORIZON}))")

    # --- Stage 3: pattern_view 匹配 + 经验检索对照包 ---
    contrast_packs = build_contrast_packs(
        episodes=source_episodes,
        target_F=target_F,
        roster=roster,
        config=config,
        operators=operators,
        domain=domain,
    )
    rendered_count = sum(
        1 for pack in contrast_packs if pack["rendered_reference_block"]
    )
    print(f"== Stage 3 PATTERN_MATCH/RETRIEVAL: {len(contrast_packs)} packs, "
          f"{rendered_count} rendered reference blocks")

    # --- Stage 4: Harness 当前视图 ---
    harness_state = build_harness_state(source_episodes)
    print(f"== Stage 4 HARNESS_STATE: {len(harness_state['local_skills'])} skills, "
          f"{len(harness_state['restrictions'])} restrictions, "
          f"{len(harness_state['rejected_bets'])} rejected bets")

    # --- Stage 5: 候选探测（Memory 只改顺序；Target Support 实测最终确认）---
    experienced_order, retrieval_trace = v1.compile_experienced_order(
        episodes=source_episodes,
        operators=operators,
        target_F=target_F,
        roster=roster,
        config=config,
        domain=domain,
    )
    fixed_order = list(operators)
    print(f"== Stage 5 PROBING: experienced[:5]={experienced_order[:5]} "
          f"fixed[:5]={fixed_order[:5]}")

    experienced_plan = v1.plan_target_support(
        order=experienced_order,
        roster=roster,
        values=values,
        config=config,
        target_support_origin=tgt_support,
        baseline_cache=baseline_cache,
    )
    fixed_plan = v1.plan_target_support(
        order=fixed_order,
        roster=roster,
        values=values,
        config=config,
        target_support_origin=tgt_support,
        baseline_cache=baseline_cache,
    )

    # --- Stage 6: 计划冻结后，才打开 Target delayed outcome（仅评估）---
    probed = sorted(
        set(experienced_plan["probed_workflows"])
        | set(fixed_plan["probed_workflows"])
    )
    delayed_gains: dict[str, float] = {}
    for op in probed:
        compiled = v1.make_compiled(op, _default_params(op, period))
        gain = v1.gain_at(
            roster, values, config, compiled, tgt_delayed, baseline_cache
        )
        if gain is None:
            raise RuntimeError(f"Target delayed evaluation failed for {op}")
        delayed_gains[op] = gain
    experienced_result = v1.attach_delayed_outcomes(experienced_plan, delayed_gains)
    fixed_result = v1.attach_delayed_outcomes(fixed_plan, delayed_gains)
    for plan, result in (
        (experienced_plan, experienced_result),
        (fixed_plan, fixed_result),
    ):
        result["first_positive_probe"] = plan["first_positive_probe"]
        result["harm_probe_count"] = plan["harm_probe_count"]
    print(f"== Stage 6 EVALUATION (delayed opened after plans frozen): "
          f"experienced probes={experienced_plan['probed_workflows']} "
          f"first={experienced_result['first_positive_probe']} "
          f"harm={experienced_result['harm_probe_count']} "
          f"auc={float(experienced_result['adaptation_auc']):+.6f}")

    # --- Stage 7: 结构化报告 ---
    relation_counts = {
        relation: sum(1 for ep in source_episodes if ep.relation == relation)
        for relation in ("POSITIVE", "NEGATIVE", "CONFLICT")
    }
    report = {
        "experiment_id": "v1-fastpath-framework-assembled",
        "evaluation_role": "EXPOSED_DEVELOPMENT_ZERO_LLM_MECHANISM",
        "domain": domain,
        "stages": [
            {"stage": "FEATURE_EXTRACTION", "component": "run_v1_fastpath.extract_F",
             "pattern_view": PATTERN_VIEW},
            {"stage": "SOURCE_EXPERIENCE", "component": "run_v1_fastpath.build_source_memory"},
            {"stage": "PATTERN_MATCH_RETRIEVAL", "component": "SignedEpisodeRetriever",
             "pattern_view": PATTERN_VIEW},
            {"stage": "HARNESS_STATE", "component": "CurrentHarnessState"},
            {"stage": "PROBING", "component": "run_v1_fastpath.compile_experienced_order + plan_target_support",
             "memory_role": "order_only_no_exclusion",
             "final_confirmation": "target_support_measured",
             "budget": MAX_TARGET_SUPPORT_EVALS,
             "stop": "first_positive"},
            {"stage": "EVALUATION", "component": "attach_delayed_outcomes",
             "future_read_during_target_planning": False},
        ],
        "timeline": {
            "source_support_origin": src_support,
            "source_delayed_origin": src_delayed,
            "target_support_origin": tgt_support,
            "target_delayed_origin": tgt_delayed,
        },
        "leakage_controls": {
            "episodes_derived_from": "source_slices_only",
            "target_outcome_in_memory": False,
            "target_delayed_opened_after_plans_frozen": True,
            "memory_excludes_workflows": False,
        },
        "source_F": {k: round(v, 6) for k, v in source_F.items()},
        "target_F": {k: round(v, 6) for k, v in target_F.items()},
        "common_candidate_inventory": operators,
        "source_episodes": [ep.to_dict() for ep in source_episodes],
        "source_relation_counts": relation_counts,
        "source_attempts": source_attempts,
        "contrast_packs": contrast_packs,
        "harness_state": harness_state,
        "retrieval_trace": retrieval_trace,
        "experienced_order": experienced_order,
        "fixed_order": fixed_order,
        "experienced": experienced_result,
        "fixed": fixed_result,
        "target_delayed_gains_for_probed_workflows": delayed_gains,
        "claim_ceiling": (
            "Same-domain exposed-development Memory probe-order mechanism only; "
            "not cross-domain transfer, not natural Agent generation, not fresh evidence; "
            "all components pre-verified (v1 fastpath / w2 scan / experience_memory)."
        ),
        "llm_api_call_count": 0,
    }
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"== Stage 7 REPORT -> {out.relative_to(root)}")
    print(f"== experienced: probes={experienced_result['probed_workflows']} "
          f"first={experienced_result['first_positive_probe']} "
          f"harm={experienced_result['harm_probe_count']} "
          f"auc={float(experienced_result['adaptation_auc']):+.6f}")
    print(f"== fixed:      probes={fixed_result['probed_workflows']} "
          f"first={fixed_result['first_positive_probe']} "
          f"harm={fixed_result['harm_probe_count']} "
          f"auc={float(fixed_result['adaptation_auc']):+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
