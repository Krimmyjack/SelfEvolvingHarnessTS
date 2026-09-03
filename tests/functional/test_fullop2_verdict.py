"""FULLOP2（rev7 下完整算子池 v2）零 LLM 裁定函数测试——
2026-08-14 pre-run amendment 后口径：protocol > joint_failure > overload >
supply > phase2 效用；A 绝对阈值 veto 已取消（A 差 B 好 = 可解释方法效应）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

DEV = ("T1@888", "T100@600", "T10@600")
CTL = ("T101@792", "T101@600")
SYN = "T101@600+synmiss"
ALL = DEV + CTL + (SYN,)


def _m(chain=True, new_legal=(), new_rej=(), proposed=1, pool=1, rejected=0):
    return {"chain_complete": chain, "n_proposed": proposed, "n_pool": pool,
            "rejected_count": rejected, "abstention": not chain,
            "per_candidate": [], "param_binding_ok": None,
            "b_new_legal": list(new_legal),
            "b_new_proposed_rejected": list(new_rej)}


def _rows(a_chain=None, b_chain=None, b_new_legal=(), b_new_rej=(),
          b_rejected=0, perr_at=None):
    a_chain = a_chain or {}
    b_chain = b_chain or {}
    rows = []
    for key in ALL:
        for rep in (0, 1):
            perr = perr_at == (key, rep, "A")
            rows.append({"key": key, "rep": rep, "arm": "A",
                         "protocol_error": ("boom" if perr else None),
                         "metrics": _m(chain=a_chain.get(key, True))})
            perr = perr_at == (key, rep, "B")
            rows.append({"key": key, "rep": rep, "arm": "B",
                         "protocol_error": ("boom" if perr else None),
                         "metrics": _m(chain=b_chain.get(key, True),
                                       new_legal=b_new_legal,
                                       new_rej=b_new_rej,
                                       rejected=b_rejected)})
    return rows


def test_a_low_b_high_is_method_effect_not_blocked():
    # 承重回归（2026-08-14 amendment 动因）：A deviation 全灭但 B 全闭链
    # 且有新合法候选 → 可解释方法效应，不得被 A 健康门挡住
    rows = _rows(a_chain={k: False for k in DEV},  # A deviation 0/6
                 b_new_legal=("cand_new",))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "AWAITING_PHASE2"


def test_joint_failure_blocked_at_4():
    both_dead = {k: False for k in DEV + (SYN,)}  # 4 context 双臂零闭链
    rows = _rows(a_chain=both_dead, b_chain=dict(both_dead))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "PREPARE_STAGE_BLOCKED"
    assert len(v["joint_fail_contexts"]) == 4


def test_joint_failure_boundary_3_proceeds():
    both_dead = {k: False for k in DEV}  # 仅 3 context 共同失败
    rows = _rows(a_chain=both_dead, b_chain=dict(both_dead))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "FULL_POOL_NO_NEW_LEGAL_SUPPLY"
    assert len(v["joint_fail_contexts"]) == 3


def test_protocol_error_anywhere_inconclusive():
    rows = _rows(perr_at=("T1@888", 1, "B"))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "PROTOCOL_INCONCLUSIVE"


def test_precedence_protocol_over_joint_failure():
    both_dead = {k: False for k in DEV + (SYN,)}
    rows = _rows(a_chain=both_dead, b_chain=dict(both_dead),
                 perr_at=("T101@792", 0, "A"))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "PROTOCOL_INCONCLUSIVE"


def test_overload_chain_collapse():
    rows = _rows(b_chain={k: False for k in ALL})  # B 0/12 vs A 12/12
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "OPERATOR_SPACE_OVERLOAD"


def test_overload_rejected_branch():
    rows = _rows(b_rejected=2)  # B rejected_total=24 vs A 0
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "OPERATOR_SPACE_OVERLOAD"


def test_supply_none_clean():
    v = runner._fullop2_verdict({"rows": _rows()})
    assert v["verdict"] == "FULL_POOL_NO_NEW_LEGAL_SUPPLY"


def test_supply_indirect_effect_note():
    # B 闭链显著更高但没用新算子 → 仍 NO_NEW_LEGAL_SUPPLY，但须注明间接效应
    rows = _rows(a_chain={k: False for k in DEV + (SYN,)})
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "FULL_POOL_NO_NEW_LEGAL_SUPPLY"
    assert "间接效应" in v["note"]


def test_supply_none_but_rejected():
    rows = _rows(b_new_rej=("cand_x",))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "CONTEXT_OR_SELECTION_BLOCKER"


def test_supply_ok_awaiting_p2():
    rows = _rows(b_new_legal=("cand_new",))
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "AWAITING_PHASE2"


def test_phase2_pass_with_delayed_ok():
    rows = _rows(b_new_legal=("cand_new",))
    p2 = {"any_positive": True, "entries": [
        {"key": "T1@888", "winner_candidate_id": "cand_new",
         "positive_count": 1, "harm_count": 0,
         "delayed": {"gain": 0.01}}]}
    v = runner._fullop2_verdict({"rows": rows, "p2": p2})
    assert v["verdict"] == "FULL_OPERATOR_CAPABILITY_PASS"
    assert v["pass_contexts"] == ["T1@888"]


def test_phase2_delayed_regression_not_pass():
    rows = _rows(b_new_legal=("cand_new",))
    p2 = {"any_positive": True, "entries": [
        {"key": "T1@888", "winner_candidate_id": "cand_new",
         "positive_count": 1, "harm_count": 0,
         "delayed": {"gain": -9.0}}]}
    v = runner._fullop2_verdict({"rows": rows, "p2": p2})
    assert v["verdict"] == "LEGAL_BUT_NO_UTILITY"
    assert "delayed" in v["note"]


def test_phase2_no_positive():
    rows = _rows(b_new_legal=("cand_new",))
    p2 = {"any_positive": False, "entries": [
        {"key": "T1@888", "winner_candidate_id": None,
         "positive_count": 0, "harm_count": 1}]}
    v = runner._fullop2_verdict({"rows": rows, "p2": p2})
    assert v["verdict"] == "LEGAL_BUT_NO_UTILITY"
    assert v["harm_total"] == 1


def test_schedule_parity():
    s = runner._fullop2_schedule()
    assert len(s) == 24 and len(set(s)) == 24
    assert s[0] == ("T1@888", 0, "A") and s[1] == ("T1@888", 0, "B")
    assert s[2] == ("T100@600", 0, "B") and s[3] == ("T100@600", 0, "A")
    assert s[12] == ("T1@888", 1, "B") and s[13] == ("T1@888", 1, "A")

def test_precedence_joint_failure_over_overload_real_conflict():
    # 真冲突（checker MINOR-1 修正）：4 context 双臂共败（joint_fail=4）
    # 且 2 存活 context A 全闭链 B 全灭 → overload 条件同时成立；
    # joint_failure 必须先判 PREPARE_STAGE_BLOCKED
    dead = {k: False for k in DEV + (SYN,)}
    rows = _rows(a_chain=dead, b_chain={k: False for k in ALL})
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "PREPARE_STAGE_BLOCKED"


def test_overload_rejected_boundary():
    # B rejected = A+1 恰好不触发；超过才触发
    rows_ok = _rows()
    for r in rows_ok:
        r["metrics"]["rejected_count"] = 1  # A=12, B=12 → 12 > 13 False
    v = runner._fullop2_verdict({"rows": rows_ok})
    assert v["verdict"] == "FULL_POOL_NO_NEW_LEGAL_SUPPLY"
    rows_bad = _rows()
    for r in rows_bad:
        if r["arm"] == "A":
            r["metrics"]["rejected_count"] = 1   # A=12
        elif r["key"] in ("T1@888", "T100@600"):
            r["metrics"]["rejected_count"] = 7   # B=10*0+2*2*7-2*1... >13
    v2 = runner._fullop2_verdict({"rows": rows_bad})
    assert v2["verdict"] == "OPERATOR_SPACE_OVERLOAD"


def test_supply_note_chain_gap_boundary():
    # B chain > A+1 才注明间接效应
    rows = _rows(a_chain={k: False for k in ("T1@888", "T100@600")})
    v = runner._fullop2_verdict({"rows": rows})
    assert v["verdict"] == "FULL_POOL_NO_NEW_LEGAL_SUPPLY"
    assert "间接效应" in v["note"]
    rows1 = _rows(a_chain={"T1@888": False}, b_chain={"T1@888": False})
    for r in rows1:
        if r["key"] == "T1@888" and r["arm"] == "B" and r["rep"] == 0:
            r["metrics"]["chain_complete"] = True  # 差恰好 1
    v1 = runner._fullop2_verdict({"rows": rows1})
    assert v1["verdict"] == "FULL_POOL_NO_NEW_LEGAL_SUPPLY"
    assert "间接效应" not in v1["note"]


def test_row_metrics_rejected_candidate_ops_from_payload():
    # 口径修正回归（2026-08-14）：verifier 拒绝的候选不在 trace
    # candidate_steps 中——ops 必须从 propose 载荷回读，
    # 否则 b_new_proposed_rejected 漏记（fullop2 T10@600 B r0 实证）
    import json as _json
    insp = {"pattern_hypotheses": [
        {"hypothesis_id": "h1", "pattern_type": "regime_ambiguity"}]}
    prop = {"candidates": [
        {"candidate_id": "ext_rej", "addresses_hypothesis_id": "h1",
         "steps": [{"op": "repair_level_shift", "params": {}}]},
        {"candidate_id": "in_pool", "addresses_hypothesis_id": "h1",
         "steps": [{"op": "outlier_mad", "params": {}}]}]}
    row = {
        "prompt_calls": [
            {"stage": "inspect", "parse_status": "VALID_AGENT_ENVELOPE",
             "assistant_text": _json.dumps(
                 {"kind": "stage_result", "payload": insp})},
            {"stage": "propose", "parse_status": "VALID_AGENT_ENVELOPE",
             "assistant_text": _json.dumps(
                 {"kind": "stage_result", "payload": prop})}],
        "candidate_ids": ["identity", "in_pool"],
        "candidate_steps": {"in_pool": [{"op": "outlier_mad", "params": {}}]},
        "chosen_candidate_id": "in_pool",
        "rejection_receipts": [{"candidate_id": "ext_rej", "reason": None}],
        "compilation_status": "ok",
    }
    m = runner._fullop_row_metrics(row, ["outlier_mad"], {"x": 1.0})
    assert m["b_new_proposed_rejected"] == ["ext_rej"]
    assert m["b_new_legal"] == []


def test_missing_rows_refuse_verdict(monkeypatch):
    # checker MAJOR-2：行不完整 → SystemExit 拒裁（不产生裁定）
    import pytest
    fake = {"fullop2_protocol": {"state": "FROZEN_BEFORE_ANY_RUN"},
            "fullop2": {"rows": _rows()[:-1]}}
    monkeypatch.setattr(runner, "_load_report", lambda: fake)
    with pytest.raises(SystemExit):
        runner.phase_fullop2_verdict()

