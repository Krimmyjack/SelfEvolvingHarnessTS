"""tests/methods/test_fault_cases.py — S0 端到端测试（2026-08-13）：
固定五类枚举封闭、选项屏蔽（无机械证据的类不可选）、确定性
reconciliation 四分支、三初始 Case 引导口径。零 LLM 零新评估。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

from SelfEvolvingHarnessTS.methods.ttha.fault_cases import (  # noqa: E402
    FAULT_TYPES,
    GUARDS,
    classify_group,
    default_guard,
    filter_candidates,
    reconcile_existing,
    selectable_fault_types,
)

E2 = PROJECT_ROOT / "artifacts/functional/e2"
STORE = json.loads((E2 / "w1_problem_cases_bootstrap.json")
                   .read_text(encoding="utf-8"))


def _empty_evidence(**over):
    ev = {
        "task_contract_conflict": None,
        "diagnosis_contradiction": None,
        "headroom": None,
        "supply_exhausted": False,
        "winner_probed": None,
        "agent_chosen": None,
        "support_positive": None,
        "delayed_negative": None,
    }
    ev.update(over)
    return ev


# ---- 枚举封闭 ----

def test_unknown_fault_type_falls_back_to_guard():
    out, err = classify_group(_empty_evidence(), "ARBITRARY_TYPE")
    assert out == "INSUFFICIENT_EVIDENCE" and err


def test_known_type_requires_mechanical_evidence():
    # SCOPE_MEMORY_RISK 无 support+/delayed− 证据 → 被屏蔽 → 回退 guard
    out, err = classify_group(_empty_evidence(), "SCOPE_MEMORY_RISK_ERROR")
    assert out == "INSUFFICIENT_EVIDENCE" and err


def test_guard_choice_allowed():
    out, err = classify_group(_empty_evidence(), "NO_ACTIONABLE_FAULT")
    assert out == "NO_ACTIONABLE_FAULT" and not err


# ---- 选项屏蔽 ----

def test_task_interpretation_masked_without_contract_conflict():
    assert "TASK_INTERPRETATION_ERROR" not in selectable_fault_types(
        _empty_evidence())


def test_task_interpretation_unmasked_with_contract_conflict():
    sel = selectable_fault_types(_empty_evidence(
        task_contract_conflict="wrong horizon"))
    assert "TASK_INTERPRETATION_ERROR" in sel


def test_diagnosis_masked_without_contradiction():
    assert "QUALITY_DIAGNOSIS_ERROR" not in selectable_fault_types(
        _empty_evidence())


def test_supply_gap_requires_exhausted_search():
    # headroom 全失败但无 supply 穷举记录 → 不构成 SUPPLY_GAP
    sel = selectable_fault_types(_empty_evidence(
        headroom={"a": False, "b": False}, supply_exhausted=False))
    assert "WORKFLOW_SUPPLY_GAP" not in sel
    sel = selectable_fault_types(_empty_evidence(
        headroom={"a": False, "b": False}, supply_exhausted=True))
    assert "WORKFLOW_SUPPLY_GAP" in sel


def test_decision_error_masked_when_agent_picked_winner():
    sel = selectable_fault_types(_empty_evidence(
        winner_probed={"op": "hampel_filter", "gain": 0.03},
        agent_chosen="hampel_filter"))
    assert "WORKFLOW_DECISION_ERROR" not in sel


def test_decision_error_unmasked_when_winner_missed():
    sel = selectable_fault_types(_empty_evidence(
        winner_probed={"op": "hampel_filter", "gain": 0.03},
        agent_chosen=None))
    assert "WORKFLOW_DECISION_ERROR" in sel


def test_scope_risk_requires_support_pos_delayed_neg():
    sel = selectable_fault_types(_empty_evidence(
        support_positive=True, delayed_negative=True))
    assert "SCOPE_MEMORY_RISK_ERROR" in sel


# ---- 保底通道 ----

def test_default_guard_no_actionable_when_headroom_measured_empty():
    assert default_guard(_empty_evidence(
        headroom={"a": False, "b": False})) == "NO_ACTIONABLE_FAULT"


def test_default_guard_insufficient_when_nothing_measured():
    assert default_guard(_empty_evidence()) == "INSUFFICIENT_EVIDENCE"


# ---- 确定性 reconciliation ----

CASE_BASE = {"case_id": "case-x", "fault_type": "WORKFLOW_SUPPLY_GAP",
             "task_consumer": "forecast|ridge|sMASE",
             "response_class": "NEGATIVE",
             "workflow_and_effect": {"workflow_sig": "winsorize"}}


def _gf(**over):
    g = {"task_consumer": "forecast|ridge|sMASE",
         "workflow_sig": "winsorize", "response_class": "NEGATIVE"}
    g.update(over)
    return g


def test_reconcile_match():
    assert reconcile_existing(CASE_BASE, _gf()) == "MATCH_ADD_EVIDENCE"


def test_reconcile_conflict_opposite_utility():
    assert reconcile_existing(CASE_BASE, _gf(response_class="CONFLICT")) \
        == "CONFLICT_WITH_EXISTING"


def test_reconcile_new_case_different_workflow():
    assert reconcile_existing(CASE_BASE, _gf(workflow_sig="outlier_mad")) \
        == "NEW_CASE"


def test_reconcile_abstain_missing_fields():
    assert reconcile_existing(CASE_BASE, {"task_consumer": "x"}) \
        == "ABSTAIN"


def test_filter_candidates_hard_filter():
    assert filter_candidates(STORE["cases"], "WORKFLOW_SUPPLY_GAP",
                             _gf()) == [STORE["cases"][0]]


# ---- 三初始 Case 口径（用户裁决硬约束）----

def test_bootstrap_three_cases_sequential_ids():
    ids = [c["case_id"] for c in STORE["cases"]]
    assert ids == ["case-0001", "case-0002", "case-0003"]


def test_case_0001_scope_is_bounded_not_generalized():
    c = STORE["cases"][0]
    assert c["fault_type"] == "WORKFLOW_SUPPLY_GAP"
    assert "不泛化" in c["scope_note"]
    assert "forecast|ridge|sMASE" in c["scope_note"]


def test_case_0002_t117_temporal_risk():
    c = STORE["cases"][1]
    assert c["fault_type"] == "SCOPE_MEMORY_RISK_ERROR"
    assert c["candidate_patch"] == "patch-replace-winsorize-with-hampel_filter"
    assert c["verified_fix"] is None
    delayed = c["negative_contrasts"][0]
    assert delayed["window"] == "T117@1032" and delayed["gain"] < 0


def test_case_0003_t105_no_actionable_not_scope():
    c = STORE["cases"][2]
    assert c["fault_type"] == "NO_ACTIONABLE_FAULT"
    assert c["fault_type"] not in ("QUALITY_DIAGNOSIS_ERROR",
                                   "SCOPE_MEMORY_RISK_ERROR")


def test_bootstrap_trace_all_match():
    for t in STORE["trace"]:
        assert t["case_action"] == "MATCH_ADD_EVIDENCE"


def test_store_has_no_identity_hash_fields():
    # 纪律：case_id 为普通顺序 ID——无 hash/sha 身份体系字段
    for c in STORE["cases"]:
        for k in c:
            assert "hash" not in k.lower() and "sha" not in k.lower()


def test_fault_types_enum_arity():
    assert len(FAULT_TYPES) == 5 and len(GUARDS) == 2
