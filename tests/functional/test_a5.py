"""A5/A3 matched-budget 主实验纯函数测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

M = runner.M


def test_relation_labels():
    f = runner._a5_relation
    assert f(-0.05, {"evaluated": False, "gain": None}) == "NEGATIVE"
    assert f(0.001, {"evaluated": False, "gain": None}) is None      # 中性不成 Episode
    assert f(None, {"evaluated": False, "gain": None}) is None
    assert f(0.1, {"evaluated": False, "gain": None}) == "POSITIVE"  # 未评估 delayed
    assert f(0.1, {"evaluated": True, "gain": 0.05}) == "POSITIVE"
    assert f(0.1, {"evaluated": True, "gain": -0.02}) == "CONFLICT"  # 支持/延迟冲突


def _agg(f5, h5, n5e, f3, h3, n3e):
    return {"A5": {"feedback_to_first_effective": f5, "harm_events": h5,
                   "n_effective": n5e},
            "A3": {"feedback_to_first_effective": f3, "harm_events": h3,
                   "n_effective": n3e}}


def test_verdict_pass():
    v = runner._a5_verdict(_agg(3, 0, 1, 7, 1, 1), [])
    assert v["verdict"] == "TRANSFER_CASE_PASS"


def test_verdict_negative_slower_and_more_harm():
    assert runner._a5_verdict(_agg(9, 0, 1, 4, 0, 1), [])["verdict"] == "NEGATIVE_TRANSFER"
    assert runner._a5_verdict(_agg(3, 2, 1, 3, 0, 1), [])["verdict"] == "NEGATIVE_TRANSFER"


def test_verdict_no_signal_variants():
    assert runner._a5_verdict(_agg(None, 0, 0, None, 0, 0), [])["verdict"] == "NO_SIGNAL"
    assert runner._a5_verdict(_agg(5, 1, 1, 5, 1, 1), [])["verdict"] == "NO_SIGNAL"


def test_verdict_a5_effective_a3_none_is_pass_if_safer():
    v = runner._a5_verdict(_agg(4, 0, 1, None, 0, 0), [])
    assert v["verdict"] == "TRANSFER_CASE_PASS"   # A3 未形成 = fpe 无穷大


def test_verdict_a3_effective_a5_none_is_negative():
    v = runner._a5_verdict(_agg(None, 0, 0, 6, 0, 1), [])
    assert v["verdict"] == "NEGATIVE_TRANSFER"


def test_verdict_protocol_and_content_precedence():
    rows = [{"arm": "A5", "protocol_error": "x"},
            {"arm": "A5", "memory_resolution_status": "no_memory"}]
    v = runner._a5_verdict(_agg(1, 0, 1, 9, 0, 1), rows)
    assert v["verdict"] == "PROTOCOL_INCONCLUSIVE"   # 优先级最高
    rows2 = [{"arm": "A5", "memory_resolution_status": "contrast_pack_empty"}]
    v2 = runner._a5_verdict(_agg(1, 0, 1, 9, 0, 1), rows2)
    assert v2["verdict"] == "CONTENT_INCONCLUSIVE"
