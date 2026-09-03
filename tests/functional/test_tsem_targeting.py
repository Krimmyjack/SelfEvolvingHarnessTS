"""TSEM（targeting semantics rev7 配对对照）零 LLM 机械测试——
用户裁决 2026-08-14：rev7 构造字节级断言、冻结交错 schedule、
四档裁定函数（含升级与 baseline-recovered 分支）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))

import run_v1_guidance_evolution as runner  # noqa: E402


def test_rev7_body_two_replacements_only():
    # 固化后（用户裁决 2026-08-14）：h0 = rev7；机械重建合成 rev6 校验构造器
    rev7 = runner._bootstrap_body(runner._h0_snapshot())
    rev6 = rev7.replace(runner.TSEM_NEW_4E, runner.TSEM_OLD_4E).replace(
        runner.TSEM_NEW_HB, runner.TSEM_OLD_HB)
    assert rev6 != rev7
    assert runner._tsem_rev7_body(rev6) == rev7  # 构造器输出=固化基线
    for marker in ("[FIXED_CONTRACT]", "[inspect_pattern_guidance]",
                   "[propose_construction_guidance]", "[select_guidance]"):
        assert marker in rev7
    for prefix in ("propose.rule.hypothesis_binding:",
                   "propose.rule.effect_distinct:",
                   "propose.rule.inert_and_order:",
                   "propose.rule.no_legal_binding:",
                   "propose.rule.exploration_supply:"):
        assert rev7.count(prefix) == 1


def test_rev7_body_anchor_missing_raises():
    try:
        runner._tsem_rev7_body("no anchors present")
    except ValueError:
        return
    raise AssertionError("expected ValueError on missing anchors")


def test_schedule_interleave_and_reversal():
    s1 = runner._tsem_schedule(1)
    assert len(s1) == 10 and len(set(s1)) == 10
    assert s1[0] == ("T1@888", 0, "rev6") and s1[1] == ("T1@888", 0, "rev7")
    assert s1[2] == ("T100@600", 0, "rev7") and s1[3] == ("T100@600", 0, "rev6")
    s2 = runner._tsem_schedule(2)
    assert len(s2) == 20 and len(set(s2)) == 20
    assert s2[10] == ("T1@888", 1, "rev7") and s2[11] == ("T1@888", 1, "rev6")
    assert s2[12] == ("T100@600", 1, "rev6") and s2[13] == ("T100@600", 1, "rev7")


def _row(key, rep, arm, chain, cats=("outlier",), rejected=0, perr=False):
    return {"key": key, "rep": rep, "arm": arm,
            "protocol_error": ("boom" if perr else None),
            "metrics": {"chain_complete": chain,
                        "chosen_categories": list(cats),
                        "rejected_count": rejected}}


def _rows(dev, ctl, rep_range=(0,)):
    rows = []
    for rep in rep_range:
        for (key, a_ok, b_ok) in dev:
            rows.append(_row(key, rep, "rev6", a_ok))
            rows.append(_row(key, rep, "rev7", b_ok))
        for (key, a_ok, b_ok) in ctl:
            rows.append(_row(key, rep, "rev6", a_ok, cats=("structural",)))
            rows.append(_row(key, rep, "rev7", b_ok, cats=("structural",)))
    return rows


CTL_OK = [("T101@792", True, True), ("T101@600", True, True)]


def test_verdict_causal_round1():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)], CTL_OK)
    assert runner._tsem_verdict(rows)["verdict"] == "TARGETING_SEMANTICS_CAUSAL_EFFECT"


def test_verdict_baseline_recovered_round1():
    rows = _rows([("T1@888", True, True), ("T100@600", True, True),
                  ("T10@600", True, True)], CTL_OK)
    assert runner._tsem_verdict(rows)["verdict"] == "BASELINE_RECOVERED_NO_INCREMENTAL_EFFECT"


def test_verdict_escalate_round1_mixed():
    rows = _rows([("T1@888", False, True), ("T100@600", False, False),
                  ("T10@600", False, False)], CTL_OK)
    assert runner._tsem_verdict(rows)["verdict"] == "ESCALATE_REP1"


def test_verdict_regressive_control_break():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)],
                 [("T101@792", True, False), ("T101@600", True, True)])
    assert runner._tsem_verdict(rows)["verdict"] == "REGRESSIVE"


def test_verdict_regressive_family_change():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)], CTL_OK)
    for r in rows:
        if r["key"] == "T101@792" and r["arm"] == "rev7":
            r["metrics"]["chosen_categories"] = ["impute"]
    assert runner._tsem_verdict(rows)["verdict"] == "REGRESSIVE"


def test_verdict_regressive_new_rejection():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)], CTL_OK)
    for r in rows:
        if r["key"] == "T101@600" and r["arm"] == "rev7":
            r["metrics"]["rejected_count"] = 1
    assert runner._tsem_verdict(rows)["verdict"] == "REGRESSIVE"


def test_verdict_protocol_inconclusive():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)], CTL_OK)
    for r in rows:
        if r["key"] == "T10@600" and r["arm"] == "rev7":
            r["protocol_error"] = "AgentProtocolError"
    assert runner._tsem_verdict(rows)["verdict"] == "PROTOCOL_INCONCLUSIVE"


def test_verdict_causal_round2():
    dev0 = [("T1@888", False, True), ("T100@600", False, True),
            ("T10@600", False, False)]
    dev1 = [("T1@888", False, True), ("T100@600", False, True),
            ("T10@600", False, True)]
    rows = _rows(dev0, CTL_OK, (0,)) + _rows(dev1, CTL_OK, (1,))
    v = runner._tsem_verdict(rows)
    assert v["verdict"] == "TARGETING_SEMANTICS_CAUSAL_EFFECT"
    assert v["deviation"]["b_chain"] == 5 and v["deviation"]["repairs"] == 5


def test_verdict_no_incremental_round2():
    dev0 = [("T1@888", False, True), ("T100@600", False, False),
            ("T10@600", False, False)]
    dev1 = [("T1@888", False, True), ("T100@600", False, False),
            ("T10@600", True, False)]
    rows = _rows(dev0, CTL_OK, (0,)) + _rows(dev1, CTL_OK, (1,))
    assert runner._tsem_verdict(rows)["verdict"] == "NO_INCREMENTAL_EFFECT"


def test_verdict_round1_b2_repairs1_escalates():
    # 边界：b_dev=2 达标但 repairs=1 —— repairs>=1 拼写错误会误判 CAUSAL
    rows = _rows([("T1@888", False, True), ("T100@600", True, True),
                  ("T10@600", False, False)], CTL_OK)
    v = runner._tsem_verdict(rows)
    assert v["deviation"]["b_chain"] == 2 and v["deviation"]["repairs"] == 1
    assert v["verdict"] == "ESCALATE_REP1"


def test_verdict_protocol_precedence_over_regressive():
    # 同时存在 protocol_error 与 P2 违例 → PROTOCOL_INCONCLUSIVE 优先
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)],
                 [("T101@792", True, False), ("T101@600", True, True)])
    for r in rows:
        if r["key"] == "T10@600" and r["arm"] == "rev6":
            r["protocol_error"] = "AgentProtocolError"
    assert runner._tsem_verdict(rows)["verdict"] == "PROTOCOL_INCONCLUSIVE"


def test_verdict_control_rev7_protocol_error_regressive():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)], CTL_OK)
    for r in rows:
        if r["key"] == "T101@792" and r["arm"] == "rev7":
            r["protocol_error"] = "AgentProtocolError"
    assert runner._tsem_verdict(rows)["verdict"] == "REGRESSIVE"


def test_verdict_missing_pair_member_inconclusive():
    rows = _rows([("T1@888", False, True), ("T100@600", False, True),
                  ("T10@600", True, True)], CTL_OK)
    rows = [r for r in rows
            if not (r["key"] == "T10@600" and r["arm"] == "rev7")]
    assert runner._tsem_verdict(rows)["verdict"] == "PROTOCOL_INCONCLUSIVE"


def test_verdict_round2_repairs2_boundary_no_incremental():
    # 边界：b_dev=5 达标但 repairs=2 —— round2 repairs>=2 拼写错误会误判
    dev0 = [("T1@888", True, True), ("T100@600", False, True),
            ("T10@600", False, False)]
    dev1 = [("T1@888", True, True), ("T100@600", False, True),
            ("T10@600", True, True)]
    rows = _rows(dev0, CTL_OK, (0,)) + _rows(dev1, CTL_OK, (1,))
    v = runner._tsem_verdict(rows)
    assert v["deviation"]["b_chain"] == 5 and v["deviation"]["repairs"] == 2
    assert v["verdict"] == "NO_INCREMENTAL_EFFECT"


def test_verdict_baseline_recovered_round2():
    dev = [("T1@888", True, True), ("T100@600", True, True),
           ("T10@600", True, True)]
    rows = _rows(dev, CTL_OK, (0,)) + _rows(dev, CTL_OK, (1,))
    assert runner._tsem_verdict(rows)["verdict"] == "BASELINE_RECOVERED_NO_INCREMENTAL_EFFECT"
