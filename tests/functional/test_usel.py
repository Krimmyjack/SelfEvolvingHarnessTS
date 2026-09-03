"""USEL（候选效用与选择校验）零 LLM 机械测试——用户裁决 2026-08-14。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402

M = runner.M


def _row(chosen, steps=None, pool_cids=()):
    steps = steps or []
    return {"candidate_ids": ["identity"] + list(pool_cids),
            "candidate_steps": {str(c["candidate_id"]): c["steps"]
                                for c in steps},
            "chosen_candidate_id": chosen,
            "metrics": {"per_candidate": steps}}


def _cand(cid, ops_params):
    return {"candidate_id": cid,
            "steps": [{"op": op, "params": p} for op, p in ops_params],
            "in_pool": True}


def test_probe_plan_two_distinct_chosen():
    mad = _cand("c1", [("outlier_mad", {})])
    win = _cand("c2", [("winsorize", {})])
    rows = [
        _row("c1", [mad, win], pool_cids=("c1", "c2")),   # 行1 chosen=c1
        _row("c2", [mad, win], pool_cids=("c1", "c2")),   # 行2 chosen=c2
    ]
    plan = runner._usel_probe_plan(rows)
    assert plan["probe1"] == plan["evals"]["probe1"]["program_key"]
    assert plan["probe2"] == plan["evals"]["probe2"]["program_key"]
    assert plan["identity_rows"] == 0
    assert len(plan["evals"]) == 2


def test_probe_plan_fallback_alternative():
    mad = _cand("c1", [("outlier_mad", {})])
    win = _cand("c2", [("winsorize", {})])
    rows = [_row("c1", [mad, win], pool_cids=("c1", "c2")),
            _row("c1", [mad, win], pool_cids=("c1", "c2"))]
    plan = runner._usel_probe_plan(rows)
    assert plan["probe2"] == plan["evals"]["probe2"]["program_key"]
    # probe2 是 canonical 序中 ≠ probe1 的合法 program
    keys = sorted(runner._usel_prog_key(c["steps"]) for c in (mad, win))
    assert plan["probe1"] == keys[0] and plan["probe2"] == keys[1]


def test_probe_plan_all_identity():
    mad = _cand("c1", [("outlier_mad", {})])
    rows = [_row("identity", [mad], pool_cids=("c1",)),
            _row("identity", [mad], pool_cids=("c1",))]
    plan = runner._usel_probe_plan(rows)
    assert plan["probe1"] is None
    assert plan["probe2"] == plan["evals"]["probe2"]["program_key"]
    assert plan["identity_rows"] == 2


def _steps(op):
    return [{"op": op, "params": {}}]


def _entry(key, results, winner_key=None, error=None, probe1="p1", probe2="p2",
           identity_rows=0):
    return {"key": key, "probe1": probe1, "probe2": probe2,
            "identity_rows": identity_rows, "results": results,
            "winner_key": winner_key, "error": error}


def _res(steps, gain, delayed=None):
    r = {"steps": steps, "gain": gain, "passed": True}
    if delayed is not None:
        r["delayed"] = {"gain": delayed}
    return r


def test_utility_pass():
    entries = [
        _entry("T1@888", {"probe1": _res(_steps("outlier_mad"), 2 * M)},
               winner_key="probe1"),
    ]
    # winner delayed 需要在 phase 里评估；此处手工补 delayed 字段
    entries[0]["results"]["probe1"]["delayed"] = {"gain": 0.01}
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "CANDIDATE_UTILITY_PASS"


def test_utility_no_headroom():
    entries = [
        _entry("T1@888", {"probe1": _res(_steps("outlier_mad"), M / 2),
                          "probe2": _res(_steps("winsorize"), M / 4)}),
    ]
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "LEGAL_SUPPLY_NO_HEADROOM"


def test_utility_mixed_family_flip():
    entries = [
        _entry("T1@888", {"probe1": _res(_steps("outlier_mad"), 2 * M)},
               winner_key="probe1"),
        _entry("T10@600", {"probe1": _res(_steps("outlier_mad"), -2 * M)}),
    ]
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "MIXED_CONTEXT_UTILITY"
    assert v["flip_families"] == [["outlier"]]


def test_utility_protocol_error():
    entries = [_entry("T1@888", {}, error="T1@888 eval probe1 unreadable")]
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "PROTOCOL_INCONCLUSIVE"


def test_selection_conservatism():
    # identity 行组 + probe2 是 certified winner → CONSERVATISM
    entries = [
        _entry("T101@600+synmiss",
               {"probe1": _res(_steps("impute_linear"), M / 2),
                "probe2": _res(_steps("impute_ema"), 2 * M,
                               delayed=0.01)},
               winner_key="probe2", identity_rows=2),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECTOR_CONSERVATISM_CONFIRMED"


def test_selection_harm_avoidance():
    # 全 identity context：唯一 alternative 有害 → HARM_AVOIDANCE
    entries = [
        _entry("T101@600+synmiss",
               {"probe2": _res(_steps("impute_linear"), -2 * M)},
               probe1=None, identity_rows=2),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECTOR_HARM_AVOIDANCE_CONFIRMED"


def test_selection_aligned():
    entries = [
        _entry("T1@888",
               {"probe1": _res(_steps("outlier_mad"), 2 * M, delayed=0.01),
                "probe2": _res(_steps("winsorize"), M / 2)},
               winner_key="probe1"),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECT_ALIGNED"


def test_selection_misaligned():
    entries = [
        _entry("T10@600",
               {"probe1": _res(_steps("outlier_mad"), M / 2),
                "probe2": _res(_steps("winsorize"), 2 * M, delayed=0.01)},
               winner_key="probe2"),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECT_MISALIGNED"


def test_selection_unavailable():
    entries = [_entry("T101@792", {})]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECT_ALIGNED"  # 无可比情况 → 全局落回 ALIGNED
    assert v["per_context"][0]["labels"]["context"] == "SELECTION_UNAVAILABLE"


def test_selection_misaligned_harm_single_program():
    # checker major-1：probe2=None 的单程序 context，chosen 自身有害
    entries = [
        _entry("T1@888",
               {"probe1": _res(_steps("outlier_mad"), -2 * M)},
               probe2=None),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECT_MISALIGNED"


def test_selection_conservatism_winner_is_probe1():
    # checker major-2：混合 context，winner==probe1 且 identity 行组存在
    entries = [
        _entry("T101@600+synmiss",
               {"probe1": _res(_steps("impute_linear"), 2 * M, delayed=0.01),
                "probe2": _res(_steps("impute_ema"), M / 2)},
               winner_key="probe1", identity_rows=2),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECTOR_CONSERVATISM_CONFIRMED"


def test_utility_same_context_flip_not_mixed():
    # checker major-3：同 context probe1 正 + probe2 负 → 不得判 MIXED
    entries = [
        _entry("T10@600",
               {"probe1": _res(_steps("outlier_mad"), 2 * M),
                "probe2": _res(_steps("winsorize"), -2 * M)},
               winner_key="probe1"),
    ]
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "LEGAL_SUPPLY_NO_HEADROOM"  # 无 certified（无 delayed）


def test_utility_mixed_precedes_pass():
    # 跨 context 翻转 + certified winner 并存 → MIXED 优先
    entries = [
        _entry("T1@888",
               {"probe1": _res(_steps("outlier_mad"), 2 * M, delayed=0.01)},
               winner_key="probe1"),
        _entry("T10@600",
               {"probe1": _res(_steps("outlier_mad"), -2 * M)}),
    ]
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "MIXED_CONTEXT_UTILITY"


def test_utility_certified_boundary_at_neg_m():
    # certified 阈值：delayed == -M 恰好不 certified（> -M 才 certified）
    entries = [
        _entry("T1@888",
               {"probe1": _res(_steps("outlier_mad"), 2 * M,
                               delayed=-M)},
               winner_key="probe1"),
    ]
    v = runner._usel_verdict(entries)
    assert v["verdict"] == "LEGAL_SUPPLY_NO_HEADROOM"


def test_support_roster_dev_only():
    # 支持窗口纪律（2026-08-14）：roster 只用目标 dev series 自身；
    # census eval_series（T13/T128–T134 冻结窗口）禁止进入 Support 探测
    roster, vals = runner._support_roster("T10", {"T10": [1.0]})
    assert roster == [{"series_uid": "T10", "role": "train"},
                      {"series_uid": "T10", "role": "eval"}]
    assert set(vals) == {"T10"}
    forbidden = {"T128", "T129", "T13", "T130", "T131", "T132", "T133",
                 "T134"}
    assert not (set(vals) & forbidden)


def test_selection_global_precedence_conservatism_over_misaligned():
    entries = [
        _entry("T101@600+synmiss",
               {"probe1": _res(_steps("impute_linear"), M / 2),
                "probe2": _res(_steps("impute_ema"), 2 * M, delayed=0.01)},
               winner_key="probe2", identity_rows=2),
        _entry("T10@600",
               {"probe1": _res(_steps("outlier_mad"), M / 2),
                "probe2": _res(_steps("winsorize"), 2 * M, delayed=0.01)},
               winner_key="probe2"),
    ]
    v = runner._usel_selection(entries)
    assert v["verdict"] == "SELECTOR_CONSERVATISM_CONFIRMED"
