"""N3 自然 Source Episode Pack 纯函数测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402


def _ep(**kw):
    base = {"episode_id": "e1", "domain_namespace": "kdd2018_dev",
            "workflow_signature": "outlier_mad",
            "context_summary": {"local_pattern": {"x": 1.0},
                                "program_geometry": {
                                    "program_steps": [{"op": "outlier_mad",
                                                       "params": {}}]}},
            "support_response": {"gain": 0.1, "accepted": True},
            "delayed_response": {"evaluated": True, "gain": 0.05},
            "response_validity": "VALID"}
    base.update(kw)
    return base


def test_pack_membership_complete_no_cherry_pick():
    eps = [_ep(episode_id="a"),
           _ep(episode_id="b", relation="NEGATIVE"),
           _ep(episode_id="c", domain_namespace="nn5"),
           _ep(episode_id="d", workflow_signature="hampel_filter")]
    pack = runner._n3_pack_episodes(eps)
    assert {e["episode_id"] for e in pack} == {"a", "b"}


def test_content_check_ok():
    assert runner._n3_content_check(_ep()) == []


def test_content_check_missing_gain_and_validity():
    miss = runner._n3_content_check(
        _ep(support_response={"gain": None}, response_validity="INSTRUMENT_INVALID"))
    assert "support_response.gain" in miss
    assert "response_validity!=VALID" in miss


def test_content_check_unknown_signature():
    miss = runner._n3_content_check(_ep(workflow_signature="unknown"))
    assert "workflow_signature" in miss
