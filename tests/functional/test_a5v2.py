"""A5v2 真 Skill 生命周期实验纯函数测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402


def _row(arm, receipts=1, harm=0, created=None, approved=None,
         retrieved=(), skill_probes=(), abstained=False, delayed=None,
         winner_gain=None, mem="contrast_pack"):
    return {"arm": arm, "entity_id": "X", "origin": 600,
            "support_receipts": receipts, "harm_probes": harm,
            "skill_created_id": created, "approved_skill_id": approved,
            "retrieved_skill_ids": list(retrieved),
            "skill_probes": list(skill_probes), "abstained": abstained,
            "delayed_utility": delayed, "winner_gain": winner_gain,
            "memory_resolution_status": mem, "llm_calls": 4}


def test_aggregates_skill_trajectory_reliable():
    rows = [
        _row("A5", created="fast_winner_outlier_mad"),          # r0 创建
        _row("A5", approved="fast_winner_outlier_mad"),         # r1 批准
        _row("A5", retrieved=["fast_winner_outlier_mad"],
             skill_probes=[{"candidate_id": "cand_skill_x", "gain": 0.01}]),
    ]
    agg = runner._a5v2_aggregates(rows)
    assert agg["A5"]["n_skills_created"] == 1
    assert agg["A5"]["n_approved"] == 1
    assert agg["A5"]["n_reliable"] == 1
    assert agg["A5"]["feedback_to_reliable_skill"] == 3  # 累计到确认轮


def test_aggregates_removal_blocks_reliable():
    rows = [
        _row("A5", created="s1"),
        _row("A5", approved="s1"),
        _row("A5", retrieved=["s1"],
             skill_probes=[{"candidate_id": "cand_skill_x", "gain": -0.02}]),
    ]
    agg = runner._a5v2_aggregates(rows)
    assert agg["A5"]["n_reliable"] == 0
    assert agg["A5"]["n_removed"] == 1
    assert agg["A5"]["feedback_to_reliable_skill"] is None


def test_aggregates_unapproved_never_reliable():
    rows = [_row("A5", created="s1"),
            _row("A5", retrieved=["s1"],
                 skill_probes=[{"candidate_id": "cand_skill_x", "gain": 0.5}])]
    agg = runner._a5v2_aggregates(rows)
    assert agg["A5"]["n_reliable"] == 0   # 未批准 → 检索+探测也不算 reliable


def test_verdict_pass_and_negative():
    agg = {"A5": {"feedback_to_reliable_skill": 2, "harm_events": 0,
                  "n_reliable": 1},
           "A3": {"feedback_to_reliable_skill": 5, "harm_events": 1,
                  "n_reliable": 1}}
    assert runner._a5v2_verdict(agg, [])["verdict"] == "TRANSFER_CASE_PASS"
    agg2 = {"A5": {"feedback_to_reliable_skill": 6, "harm_events": 2,
                   "n_reliable": 1},
            "A3": {"feedback_to_reliable_skill": 3, "harm_events": 0,
                   "n_reliable": 2}}
    assert runner._a5v2_verdict(agg2, [])["verdict"] == "NEGATIVE_TRANSFER"


def test_verdict_gates():
    rows = [{"arm": "A5", "protocol_error": "x",
             "memory_resolution_status": "contrast_pack"}]
    agg = {"A5": {"feedback_to_reliable_skill": 1, "harm_events": 0,
                  "n_reliable": 1},
           "A3": {"feedback_to_reliable_skill": 9, "harm_events": 0,
                  "n_reliable": 1}}
    assert runner._a5v2_verdict(agg, rows)["verdict"] == "PROTOCOL_INCONCLUSIVE"
    rows2 = [{"arm": "A5", "memory_resolution_status": "rendered_empty"}]
    assert runner._a5v2_verdict(agg, rows2)["verdict"] == "CONTENT_INCONCLUSIVE"
    zero = {"A5": {"feedback_to_reliable_skill": None, "harm_events": 0,
                   "n_reliable": 0},
            "A3": {"feedback_to_reliable_skill": None, "harm_events": 0,
                   "n_reliable": 0}}
    assert runner._a5v2_verdict(zero, [])["verdict"] == "NO_SIGNAL"
