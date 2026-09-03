"""N4 exact roster 资格审查纯函数测试。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402


def _elig(ds, uid, origin):
    return {"dataset": ds, "series_uid": uid, "origin": origin,
            "changed_total": 10, "outcome_exposure": "SEALED"}


def test_pick_roster_single_dominant():
    per = {"a_ds": {"eligible": [_elig("a_ds", f"u{i}", 600) for i in range(5)]},
           "b_ds": {"eligible": [_elig("b_ds", "u9", 600)]}}
    pick = runner._n4_pick_roster(per)
    assert pick["verdict"] == "N4_ROSTER_ELIGIBLE"
    assert pick["dataset"] == "a_ds"
    assert len(pick["roster"]) == 5  # 不足 K=6 取全部


def test_pick_roster_tie_breaks_by_dataset_id():
    per = {"z_ds": {"eligible": [_elig("z_ds", f"u{i}", 600) for i in range(4)]},
           "a_ds": {"eligible": [_elig("a_ds", f"v{i}", 600) for i in range(4)]}}
    pick = runner._n4_pick_roster(per)
    assert pick["dataset"] == "a_ds"  # 平手取字典序小者


def test_pick_roster_caps_at_k():
    per = {"a_ds": {"eligible": [_elig("a_ds", f"u{i:02d}", 600)
                                 for i in range(10)]}}
    pick = runner._n4_pick_roster(per)
    assert len(pick["roster"]) == runner.N4_ROSTER_K


def test_pick_roster_unavailable_when_too_few():
    per = {"a_ds": {"eligible": [_elig("a_ds", "u1", 600)]},
           "b_ds": {"eligible": []}}
    pick = runner._n4_pick_roster(per)
    assert pick["verdict"] == "FRESH_TARGET_CONTENT_UNAVAILABLE"


def test_roster_sorted_by_uid_origin():
    per = {"a_ds": {"eligible": [_elig("a_ds", "u2", 888),
                                 _elig("a_ds", "u1", 792),
                                 _elig("a_ds", "u1", 600),
                                 _elig("a_ds", "u2", 600)]}}
    pick = runner._n4_pick_roster(per)
    keys = [(e["series_uid"], e["origin"]) for e in pick["roster"]]
    assert keys == [("u1", 600), ("u1", 792), ("u2", 600), ("u2", 888)]
