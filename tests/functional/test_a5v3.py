"""A5v3 分维裁定与终态 reliable 语义的纯函数测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402


def _row(arm, entity="X", origin=600, receipts=1, probes=(), delayed=None,
         winner=True, created=None, approved=None, revoked=None,
         retrieved=(), skill_probes=(), mem="contrast_pack"):
    return {"arm": arm, "entity_id": entity, "origin": origin,
            "support_receipts": receipts, "probes": list(probes),
            "delayed_utility": delayed,
            "winner_program": [{"op": "x", "params": {}}] if winner else None,
            "skill_created_id": created, "approved_skill_id": approved,
            "revoked_skill_id": revoked,
            "retrieved_skill_ids": list(retrieved),
            "skill_probes": list(skill_probes), "abstained": not winner,
            "winner_gain": None, "memory_resolution_status": mem,
            "llm_calls": 4}


def test_confirmed_then_revoked_not_final_reliable():
    rows = [
        _row("A5", created="s1"),
        _row("A5", origin=792, approved="s1"),
        _row("A5", origin=888, retrieved=["s1"],
             skill_probes=[{"candidate_id": "cand_skill_s1", "gain": 0.01}]),
        _row("A5", origin=888, retrieved=["s1"], revoked="s1",
             delayed=-0.03,
             skill_probes=[{"candidate_id": "cand_skill_s1", "gain": 0.01}]),
    ]
    agg = runner._a5v3_aggregates(rows)
    assert agg["A5"]["n_confirmed"] == 1      # 历史事件保留
    assert agg["A5"]["n_removed"] == 1
    assert agg["A5"]["n_final_reliable"] == 0  # 终态被撤销
    assert agg["A5"]["feedback_to_first_confirmed"] == 3


def test_shared_harm_excluded_from_incremental():
    probe = {"candidate_id": "cand_repair", "kind": "probe", "gain": -0.0355}
    rows = [
        _row("A5", probes=[probe], winner=False),
        _row("A3", probes=[dict(probe)], winner=False),
        _row("A5", origin=792, delayed=-0.03),   # A5 独有 delayed harm
    ]
    agg = runner._a5v3_aggregates(rows)
    assert agg["A5"]["harm_events"] == 2
    assert agg["A5"]["incremental_harm"] == 1   # 共同 probe harm 剔除
    assert agg["A3"]["incremental_harm"] == 0


def test_verdict_pass():
    agg = {"A5": {"feedback_to_first_confirmed": 3, "incremental_harm": 0,
                  "n_final_reliable": 1, "n_removed": 0, "winners": 2,
                  "n_skills_created": 1},
           "A3": {"feedback_to_first_confirmed": None, "incremental_harm": 0,
                  "n_final_reliable": 0, "n_removed": 0, "winners": 0,
                  "n_skills_created": 0}}
    v = runner._a5v3_verdict(agg, [])
    assert v["verdict"] == "TRANSFER_CASE_PASS"
    assert v["dimensional"]["q1_actionability"]["signal"] \
        == "SOURCE_ACTIONABILITY_POSITIVE"


def test_verdict_safety_veto_even_with_skill():
    # A5 有终态 Skill 但增量 harm 更多 → NEGATIVE（安全否决优先）
    agg = {"A5": {"feedback_to_first_confirmed": 3, "incremental_harm": 2,
                  "n_final_reliable": 1, "n_removed": 0, "winners": 2,
                  "n_skills_created": 1},
           "A3": {"feedback_to_first_confirmed": None, "incremental_harm": 0,
                  "n_final_reliable": 0, "n_removed": 0, "winners": 0,
                  "n_skills_created": 0}}
    assert runner._a5v3_verdict(agg, [])["verdict"] == "NEGATIVE_TRANSFER"


def test_verdict_overlap_resolution_no_skills_but_more_harm():
    # 两臂皆无终态 Skill 且 A5 增量 harm 更多 → NEGATIVE（v2 重叠已解决）
    agg = {"A5": {"feedback_to_first_confirmed": None, "incremental_harm": 1,
                  "n_final_reliable": 0, "n_removed": 1, "winners": 1,
                  "n_skills_created": 1},
           "A3": {"feedback_to_first_confirmed": None, "incremental_harm": 0,
                  "n_final_reliable": 0, "n_removed": 0, "winners": 0,
                  "n_skills_created": 0}}
    assert runner._a5v3_verdict(agg, [])["verdict"] == "NEGATIVE_TRANSFER"
    # 两臂皆无终态且 harm 不增 → NO_SIGNAL
    agg["A5"]["incremental_harm"] = 0
    assert runner._a5v3_verdict(agg, [])["verdict"] == "NO_SIGNAL"


def test_verdict_gates():
    rows = [_row("A5", mem="rendered_empty")]
    agg = {"A5": {}, "A3": {}}
    assert runner._a5v3_verdict(agg, rows)["verdict"] == "CONTENT_INCONCLUSIVE"
    rows2 = [_row("A5")]
    rows2[0]["protocol_error"] = "boom"
    assert runner._a5v3_verdict(agg, rows2)["verdict"] \
        == "PROTOCOL_INCONCLUSIVE"
