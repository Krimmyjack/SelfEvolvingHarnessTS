"""P4 契约测试：true 判官 held-in/held-out validator（P3 binding：proxy 不得进验收）。"""
import numpy as np

from SelfEvolvingHarnessTS.evaluators.anomaly_rig import make_anomaly_slice
from SelfEvolvingHarnessTS.policy.action_spec import action_menu_v1
from SelfEvolvingHarnessTS.policy.edits import AddRiskRule, bundle_v0
from SelfEvolvingHarnessTS.policy.risk_policy import RiskRule
from SelfEvolvingHarnessTS.slow_path.true_judge_validator import (
    SubstrateRouterPolicy,
    evaluate_bundle,
    validate_edit,
)

MENU = action_menu_v1()


def _rows():
    return make_anomaly_slice(8, seed=31)


def _split(rows):
    return rows[:4], rows[4:]


def _ban_rule(ban, repl, rid):
    return AddRiskRule(RiskRule(
        rule_id=rid,
        when={"cell": {"snr": "low"}, "base_action_in": [ban]},
        then={"op": "ban", "action": repl},
        scope="region:cell_snr=low",
        provenance={"source": "proposer:enum"},
    ))


def test_substrate_router_incumbent_semantics():
    router = SubstrateRouterPolicy()
    key_low = {"pattern": {"struct_feats": {}}, "task": {"type": "forecast"},
               "cell_id": "anomaly|snrLow|miss"}
    key_high = {"pattern": {"struct_feats": {}}, "task": {"type": "forecast"},
                "cell_id": "anomaly|snrHigh|full"}
    key_anom = {"pattern": {"struct_feats": {}}, "task": {"type": "anomaly_detection"},
                "cell_id": "anomaly|snrLow|miss"}
    assert router.predict(key_low, MENU).action_id == "f0_median_w25"     # F0 时代剂量启发式（v0 现任）
    assert router.predict(key_high, MENU).action_id == "f0_median_w9"
    assert router.predict(key_anom, MENU).action_id == "v_none"           # anomaly 只许插补基线


def test_evaluate_bundle_serves_and_scores_true_judge():
    rows = _rows()
    out = evaluate_bundle(bundle_v0(), rows, "forecast", MENU, SubstrateRouterPolicy())
    assert len(out) == len(rows)
    low = [r for r in out if "snrLow" in r["cell"]]
    assert all(r["action_id"] == "f0_median_w25" for r in low)
    assert all(isinstance(r["true_delta"], float) for r in out)


def test_validate_edit_accepts_good_rule_and_rejects_reverse():
    held_in, held_out = _split(_rows())
    good = _ban_rule("f0_median_w25", "f0_median_w9", "mined_ban_w25_snrLow")
    outcome = validate_edit(bundle_v0(), good, held_in, held_out, "forecast",
                            MENU, SubstrateRouterPolicy(),
                            min_heldout_gain=0.02, worst_cell_tol=0.05)
    assert outcome["accepted"], outcome["reasons"]
    assert outcome["held_out"]["mean_gain"] > 0.02
    assert outcome["non_targeted_identical"] is True          # 规则未触发处 serving bit 级不变

    bad = _ban_rule("f0_median_w9", "f0_median_w25", "reverse_ban_w9_snrLow")
    outcome_bad = validate_edit(bundle_v0(), bad, held_in, held_out, "forecast",
                                MENU, SubstrateRouterPolicy(),
                                min_heldout_gain=0.02, worst_cell_tol=0.05)
    # v0 在 snrLow 本就服务 w25，ban w9 永不触发 → 零效应 → 过不了 held-out 增益门
    assert not outcome_bad["accepted"]
    assert any("held_out" in r for r in outcome_bad["reasons"])


def test_validate_edit_rejects_harmful_force_rule():
    held_in, held_out = _split(_rows())
    harmful = AddRiskRule(RiskRule(
        rule_id="force_w25_high",
        when={"cell": {"snr": "high"}, "base_action_in": ["f0_median_w9"]},
        then={"op": "ban", "action": "f0_median_w25"},        # 把 snrHigh 的 w9 改成 w25 = 伤害
        scope="region:cell_snr=high",
        provenance={"source": "proposer:enum"},
    ))
    outcome = validate_edit(bundle_v0(), harmful, held_in, held_out, "forecast",
                            MENU, SubstrateRouterPolicy(),
                            min_heldout_gain=0.02, worst_cell_tol=0.05)
    assert not outcome["accepted"]
