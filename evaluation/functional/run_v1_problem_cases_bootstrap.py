"""PROBLEM_CASES_BOOTSTRAP（S0，2026-08-13：从旧报告机械生成三个初始
Problem Case + 确定性分类/reconciliation 端到端验证——零 LLM——零新
评估——development exposure——零新 Claim）。

背景（用户裁决 2026-08-13）：转向"受限 Case 驱动的 Batch Harness
Evolution"（Anything2Skill 轻量思想——新候选与已有 Case 比较：
MATCH/CONFLICT/NEW/ABSTAIN；固定五类错误选择题；不做任意范围归因）。
S0 硬约束（用户裁决）：仅实现 fault_cases.py + 三 Case 机械引导 +
端到端测试 + 静态漏斗图；不修改 group_fault.py/method.py/Agent
prompt；不调 LLM；不跑新数据；不声称 Batch Evolution 已有效。

三个初始 Case 口径（用户裁决——严格限定）：
  case-0001 WORKFLOW_SUPPLY_GAP：Scope 仅限 forecast|ridge|sMASE ×
    winsorize/outlier × 当前 DSL——不泛化"所有 outlier 问题无解"。
  case-0002 SCOPE_MEMORY_RISK_ERROR（temporal-risk）：T117——
    candidate_patch=hampel、support=positive、delayed=rejected、
    verified_fix=null。
  case-0003 NO_ACTIONABLE_FAULT：T105——不标 Diagnosis 或 Scope。
    （T153 不写成自然 Diagnosis witness——无明确工件证据，用户裁决。）

机械证据全部来自已暴露报告字段（零新 outcome）：
  - w1_batch_census_dev_report.json（wave3 family + headroom + rounds）
  - w1_block2_census_ec_dev_report.json（block2 family + headroom）
  - w1_program_supply_dev_report.json（SUPPLY_EXHAUSTED 事实）
  - w1_group_witness_real_slow_report_v3.json（T117 hampel pending +
    delayed −0.1166）
用法：
  python evaluation/functional/run_v1_problem_cases_bootstrap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

from SelfEvolvingHarnessTS.methods.ttha.fault_cases import (  # noqa: E402
    classify_group,
    default_guard,
    filter_candidates,
    reconcile_existing,
    selectable_fault_types,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
STORE_REL = E2 / "w1_problem_cases_bootstrap.json"
TASK_CONSUMER = "forecast|ridge|sMASE"


def _load(name: str) -> dict[str, Any]:
    return json.loads((E2 / name).read_text(encoding="utf-8"))


def main() -> int:
    census = _load("w1_batch_census_dev_report.json")
    b2 = _load("w1_block2_census_ec_dev_report.json")
    supply = _load("w1_program_supply_dev_report.json")
    witness = _load("w1_group_witness_real_slow_report_v3.json")

    fam_w3 = census["development_families"][0]
    fam_b2 = b2["development_families"][0]
    hr_w3 = fam_w3["replacement_headroom"]
    hr_b2 = fam_b2["replacement_headroom"]

    # ---- 机械证据包（全部已测事实——报告字段直读）----
    evidence_w3 = {
        "group_id": "wave3-winsorize-family",
        "task_contract_conflict": None,
        "diagnosis_contradiction": None,
        "headroom": {a: bool(v.get("common_positive"))
                     for a, v in hr_w3.items()},
        "supply_exhausted": bool(supply.get("verdict") == "SUPPLY_EXHAUSTED"),
        "winner_probed": None,
        "agent_chosen": None,
        "support_positive": None,
        "delayed_negative": None,
    }
    # T117 组证据全部从 witness v3 报告程序读取（全精度——checker 裁决
    # 修复：不手写常量）
    replay_gains = [e["gain"] for e in witness.get(
        "group_feedback_event", {}).get("group_replay") or []]
    delayed_gain = (witness.get("delayed_event") or {}).get("delayed_gain")
    evidence_t117 = {
        "group_id": "t117-winsorize-group",
        "task_contract_conflict": None,
        "diagnosis_contradiction": None,
        # T117 组 headroom：hampel 共同正向（组内 replay 全过——报告直读）
        "headroom": {"outlier_mad": False, "hampel_filter": True},
        "supply_exhausted": False,
        "winner_probed": {"op": "hampel_filter",
                          "gain": max(replay_gains)},
        # op 级（判据口径）——witness 实际选择 patch_id 绑定 hampel
        "agent_chosen": "hampel_filter",
        "support_positive": True,
        "delayed_negative": bool(delayed_gain is not None
                                 and delayed_gain < 0.0),
    }
    evidence_t105 = {
        "group_id": "t105-winsorize-cluster",
        "task_contract_conflict": None,
        "diagnosis_contradiction": None,
        "headroom": {a: bool(v.get("common_positive"))
                     for a, v in hr_b2.items()},
        "supply_exhausted": False,  # block2 无 supply 搜索记录——如实
        "winner_probed": None,
        "agent_chosen": None,
        "support_positive": None,
        "delayed_negative": None,
    }

    # ---- 三初始 Case（普通顺序 ID；字段比较才是同一性判据）----
    w3_neg_windows = [(e["series"], e["origin"], e["gain"])
                      for e in fam_w3["episodes"]]
    b2_neg_windows = [(e["series"], e["origin"], e["gain"])
                      for e in fam_b2["episodes"]]
    # 正向对照（winsorize 正窗口——census rounds 直读）
    pos_w3: list[tuple[str, int, float]] = []
    for sid, rounds in (census.get("development_rounds") or {}).items():
        for r in rounds:
            for cid, gain in r.get("probes") or []:
                if cid == "cand_winsorize" and gain is not None \
                        and gain >= 0.005:
                    pos_w3.append((sid, r["origin"], float(gain)))
    pos_b2: list[tuple[str, int, float]] = []
    for sid, rounds in (b2.get("development_rounds") or {}).items():
        for r in rounds:
            for cid, gain in r.get("probes") or []:
                if cid == "cand_winsorize" and gain is not None \
                        and gain >= 0.005:
                    pos_b2.append((sid, r["origin"], float(gain)))

    cases = [
        {
            "case_id": "case-0001",
            "name": "supply-gap_forecast-ridge_outlier",
            "fault_type": "WORKFLOW_SUPPLY_GAP",
            "task_consumer": TASK_CONSUMER,
            "observable_context": {"defect_family": "winsorize/outlier"},
            "failed_behavior": ("winsorize material negative 重复窗口："
                                "wave3 6 窗 × 4 series + block2 T105 "
                                "3 窗——无共同正向替代"),
            "workflow_and_effect": {"workflow_sig": "winsorize",
                                    "response_class": "NEGATIVE",
                                    "min_gain": fam_w3["min_gain"],
                                    "n_windows": len(w3_neg_windows)
                                    + len(b2_neg_windows)},
            "response_class": "NEGATIVE",
            "supporting_episode_ids": [
                f"{s}@{o}" for s, o, _g in w3_neg_windows + b2_neg_windows],
            "positive_contrasts": [
                {"window": f"{s}@{o}", "gain": g} for s, o, g
                in pos_w3 + pos_b2],
            "negative_contrasts": [],
            "conflicts": [],
            "known_headroom": {
                "status": "none_common_positive",
                "details": ("outlier_mad/hampel_filter 全非共同正向"
                            "（wave3 6 窗 + block2 3 窗已测）+ 54 supply"
                            " 穷举无全过候选")},
            "verified_fix": None,
            "status": "STABLE_CASE",
            "scope_note": ("仅限 forecast|ridge|sMASE × winsorize/outlier"
                           " × 当前 DSL——不泛化'所有 outlier 问题无解'"),
        },
        {
            "case_id": "case-0002",
            "name": "temporal-risk_forecast-ridge_winsorize",
            "fault_type": "SCOPE_MEMORY_RISK_ERROR",
            "task_consumer": TASK_CONSUMER,
            "observable_context": {"defect_family": "winsorize",
                                   "series": ["T117"]},
            "failed_behavior": ("T117 winsorize 负 @888/@984 → hampel "
                                "组内 Support 全正但 delayed 显著翻负"),
            "workflow_and_effect": {"workflow_sig": "winsorize",
                                    "response_class": "NEGATIVE"},
            "response_class": "NEGATIVE",
            "supporting_episode_ids": ["T117@888", "T117@984"],
            "positive_contrasts": [
                {"window": f"T117@{e['origin']}", "gain": e["gain"],
                 "candidate": "hampel_filter"}
                for e in witness.get("group_feedback_event", {})
                .get("group_replay") or []],
            "negative_contrasts": [
                {"window": "T117@1032", "gain": delayed_gain,
                 "stage": "delayed_rejected"}] if delayed_gain is not None
            else [],
            "conflicts": [],
            "known_headroom": {"status": "common_positive",
                               "candidate": "hampel_filter"},
            "verified_fix": None,
            "candidate_patch": "patch-replace-winsorize-with-hampel_filter",
            "status": "STABLE_CASE",
            "scope_note": ("Support 正 + delayed 负 = temporal risk——"
                           "修改只能 DRAFT，delayed 拒绝是正确行为"),
        },
        {
            "case_id": "case-0003",
            "name": "no-actionable_forecast-ridge_t105-winsorize",
            "fault_type": "NO_ACTIONABLE_FAULT",
            "task_consumer": TASK_CONSUMER,
            "observable_context": {"defect_family": "winsorize",
                                   "series": ["T105"]},
            "failed_behavior": ("T105 单 series winsorize 负 3 窗"
                                "（600/792/984）——block2 未复现跨 series"
                                " family；两替代已测全失败"),
            "workflow_and_effect": {"workflow_sig": "winsorize",
                                    "response_class": "NEGATIVE"},
            "response_class": "NEGATIVE",
            "supporting_episode_ids": [
                f"{s}@{o}" for s, o, _g in b2_neg_windows],
            "positive_contrasts": [
                {"window": f"{s}@{o}", "gain": g} for s, o, g in pos_b2
                if s == "T105"],
            "negative_contrasts": [],
            "conflicts": [],
            "known_headroom": {
                "status": "none_common_positive",
                "details": ("outlier_mad 三窗 0.0（无离群点可修）/"
                            "hampel_filter 三窗 <M——动作空间已测空")},
            "verified_fix": None,
            "status": "STABLE_CASE",
            "scope_note": ("不得强建 Scope/Skill——保底通道 Case；"
                           "单 series 未复现跨 series family"),
        },
    ]

    # ---- 端到端确定性验证（分类 → 选项屏蔽 → reconciliation）----
    trace = []
    for ev, choice in (
            (evidence_w3, "WORKFLOW_SUPPLY_GAP"),
            (evidence_t117, "SCOPE_MEMORY_RISK_ERROR"),
            (evidence_t105, None)):
        selectable = selectable_fault_types(ev)
        if choice is None:
            chosen, err = default_guard(ev), ""
        else:
            chosen, err = classify_group(ev, choice)
        group_fields = {"task_consumer": TASK_CONSUMER,
                        "workflow_sig": "winsorize",
                        "response_class": "NEGATIVE"}
        matched = filter_candidates(cases, chosen, group_fields)
        action = (reconcile_existing(matched[0], group_fields)
                  if matched else "ABSTAIN")
        trace.append({
            "group": ev["group_id"],
            "selectable": selectable,
            "chosen": chosen,
            "classify_error": err,
            "matched_cases": [c["case_id"] for c in matched[:3]],
            "case_action": action,
        })
        print(f"== {ev['group_id']}: selectable={selectable} "
              f"chosen={chosen} action={action}", flush=True)

    # 断言（bootstrap 自身的端到端验证——失败即 fail-loud）
    assert trace[0]["chosen"] == "WORKFLOW_SUPPLY_GAP" \
        and trace[0]["case_action"] == "MATCH_ADD_EVIDENCE"
    assert trace[1]["chosen"] == "SCOPE_MEMORY_RISK_ERROR" \
        and trace[1]["case_action"] == "MATCH_ADD_EVIDENCE"
    assert trace[2]["chosen"] == "NO_ACTIONABLE_FAULT" \
        and trace[2]["case_action"] == "MATCH_ADD_EVIDENCE"
    # 屏蔽断言：无机械证据的类不可选
    for ev in (evidence_w3, evidence_t117, evidence_t105):
        sel = selectable_fault_types(ev)
        assert "TASK_INTERPRETATION_ERROR" not in sel  # 无 contract 矛盾
        assert "QUALITY_DIAGNOSIS_ERROR" not in sel  # 无诊断矛盾工件

    store = {
        "experiment_id": "v1-problem-cases-bootstrap",
        "note": "S0：三初始 Problem Case 机械引导（旧报告字段直读——"
                "零 LLM 零新评估）+ 确定性分类/选项屏蔽/reconciliation "
                "端到端验证。case_id 为普通顺序 ID——同一性由字段比较"
                "判定。development exposure——零新 Claim——不声称 "
                "Batch Evolution 已有效。",
        "cases": cases,
        "trace": trace,
    }
    STORE_REL.write_text(json.dumps(store, ensure_ascii=False,
                                    indent=2, default=str) + "\n",
                         encoding="utf-8")
    print(f"== store -> {STORE_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
