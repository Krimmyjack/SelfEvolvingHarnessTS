"""LOOP1（首次全自主自进化闭环正向演示）测试。

只测纯函数 _loop1_judgment 与 _loop1_card——不触碰 LLM/数据文件/报告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402


def _arm(arm, rejections=0, nonid=0, program=0, compilation="ok",
         error=None):
    cids = ["identity"] + ["c%d" % i for i in range(nonid)]
    steps = {("c%d" % i): [{"op": "op", "params": {}}]
             for i in range(program)}
    return {"arm": arm, "candidate_ids": cids, "candidate_steps": steps,
            "compilation_status": compilation, "protocol_error": error,
            "rejection_receipts": [{"candidate_id": "r%d" % i}
                                   for i in range(rejections)]}


# ----------------------------------------------------------- 三判据组合
def test_positive_fork_demo():
    rows = [_arm("A", rejections=1, nonid=1, program=1),
            _arm("B", rejections=0, nonid=1, program=1, compilation="ok")]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_POSITIVE_FORK_DEMO"
    assert j["final"] is True
    assert j["criteria"]["contrast_reproduced"] is True
    assert j["criteria"]["fork_improved"] is True
    assert j["criteria"]["supply_preserved"] is True


def test_no_contrast_when_arm_a_clean():
    rows = [_arm("A", rejections=0, nonid=1, program=1),
            _arm("B", rejections=0, nonid=1, program=1)]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_NO_CONTRAST"
    assert j["criteria"]["contrast_reproduced"] is False


def test_no_improvement_arm_b_still_rejects():
    rows = [_arm("A", rejections=1, nonid=1, program=1),
            _arm("B", rejections=1, nonid=1, program=1)]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_NO_IMPROVEMENT"
    assert j["criteria"]["contrast_reproduced"] is True
    assert j["criteria"]["fork_improved"] is False


def test_no_improvement_supply_collapse():
    # B 自身闭链（fork_improved True）但供应坍缩（1 < 3-1=2）
    rows = [_arm("A", rejections=1, nonid=3, program=3),
            _arm("B", rejections=0, nonid=1, program=1)]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_NO_IMPROVEMENT"
    assert j["criteria"]["fork_improved"] is True
    assert j["criteria"]["supply_preserved"] is False


def test_no_improvement_arm_b_compilation_not_ok():
    rows = [_arm("A", rejections=1, nonid=1, program=1),
            _arm("B", rejections=0, nonid=1, program=1,
                 compilation="not_applicable")]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_NO_IMPROVEMENT"
    assert j["criteria"]["fork_improved"] is False


def test_no_improvement_arm_b_no_program_candidate():
    # B 零拒绝但非 identity 候选无 steps（不算 program 候选）
    rows = [_arm("A", rejections=1, nonid=1, program=1),
            _arm("B", rejections=0, nonid=1, program=0)]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_NO_IMPROVEMENT"
    assert j["criteria"]["fork_improved"] is False


# ----------------------------------------------------------- protocol error
def test_protocol_error_on_arm_b():
    rows = [_arm("A", rejections=1, nonid=1, program=1),
            _arm("B", error="boom")]
    j = runner._loop1_judgment(rows)
    assert j["verdict"] == "LOOP1_PROTOCOL_ERROR"
    assert j["reason"].startswith("protocol_error_arm_B")


def test_protocol_error_missing_arm():
    j = runner._loop1_judgment([_arm("A", rejections=1, nonid=1, program=1)])
    assert j["verdict"] == "LOOP1_PROTOCOL_ERROR"
    assert j["reason"] == "missing_arm_B"


def test_protocol_error_no_rows():
    j = runner._loop1_judgment([])
    assert j["verdict"] == "LOOP1_PROTOCOL_ERROR"
    assert j["reason"] == "no_rows"


def test_non_identity_counts_exclude_identity():
    rows = [_arm("A", rejections=1, nonid=2, program=2),
            _arm("B", rejections=0, nonid=2, program=2)]
    c = runner._loop1_judgment(rows)["criteria"]
    assert c["arm_a_non_identity"] == 2
    assert c["arm_b_non_identity"] == 2
    assert c["arm_a_rejections"] == 1
    assert c["arm_b_rejections"] == 0


# ------------------------------------------------------------- capsule 匿名
def test_loop1_card_anonymized_and_shape():
    body = """[propose_construction_guidance]
propose.rule.hypothesis_binding: text
[select_guidance]
5. select stage text
"""
    card = runner._loop1_card("sha123", body, [])
    facts = card["facts"]
    fr = facts["failure_receipt"]
    assert fr["modified_fraction"] == 0.952
    assert fr["cap"] == 0.35
    assert fr["candidate_family"] == "external_region repair"
    assert "T10" not in json.dumps(fr, ensure_ascii=False)
    assert "kdd" not in json.dumps(fr, ensure_ascii=False).lower()
    assert facts["guidance_surface_precondition_sha"] == "sha123"
    assert facts["current_guidance_body"] == body
    assert "hypothesis_binding" in facts["old_propose_rules"]
    assert "REPLACE_CLAUSE" in card["instruction"]
    assert "supply_effect_distinct" in card["instruction"]
