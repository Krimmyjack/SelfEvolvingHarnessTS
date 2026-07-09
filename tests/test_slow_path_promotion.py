from SelfEvolvingHarnessTS.policy.edits import AddRiskRule, MemoryWrite
from SelfEvolvingHarnessTS.slow_path.promotion import PromotionGate, ProposalValidationOutcome
from SelfEvolvingHarnessTS.slow_path.proposal_schema import SlowProposal


def _memory_proposal(support_n=2):
    return SlowProposal(
        kind="MemoryWrite",
        scope="cell:forecast|snrLow|miss",
        payload={
            "pattern_region": "forecast|snrLow|miss",
            "action": "v_median",
            "grounded_utility": 0.2,
            "utility_ci": [0.1, 0.3],
            "scope": "cell:forecast|snrLow|miss",
            "source": "deployment_evidence_miner",
        },
        evidence_refs=("forecast|snrLow|miss:v_median:utility",),
        support={"n": support_n},
        provenance={"source": "deployment_evidence_miner"},
    )


def _risk_proposal(support_n=2):
    return SlowProposal(
        kind="ProposeRiskRule",
        scope="cell:forecast|snrLow|miss",
        payload={
            "rule_id": "auto_ban_forecast_snrLow_miss_v_median",
            "when": {"base_action_in": ["v_median"]},
            "then": {"op": "ban", "action": "v_none"},
            "scope": "cell:forecast|snrLow|miss",
        },
        evidence_refs=("forecast|snrLow|miss:v_median:harm",),
        support={"n": support_n},
        provenance={"source": "deployment_evidence_miner"},
    )


def test_promotion_gate_compiles_valid_memory_write_without_applying_bundle():
    outcome = PromotionGate(min_support=2).validate(_memory_proposal())

    assert isinstance(outcome, ProposalValidationOutcome)
    assert outcome.accepted is True
    assert outcome.reason is None
    assert isinstance(outcome.edit_op, MemoryWrite)
    assert outcome.edit_op.evidence["action"] == "v_median"


def test_promotion_gate_compiles_valid_risk_rule_without_applying_bundle():
    outcome = PromotionGate(min_support=2).validate(_risk_proposal())

    assert outcome.accepted is True
    assert outcome.reason is None
    assert isinstance(outcome.edit_op, AddRiskRule)
    assert outcome.edit_op.rule.rule_id == "auto_ban_forecast_snrLow_miss_v_median"


def test_promotion_gate_rejects_low_support_and_unsupported_kind():
    low = PromotionGate(min_support=3).validate(_memory_proposal(support_n=2))
    assert low.accepted is False
    assert low.reason == "insufficient support: 2 < 3"
    assert low.edit_op is None

    unsupported = SlowProposal(
        kind="ProposeSkillSpec",
        scope="cell:forecast|snrLow|miss",
        payload={"skill_id": "new_skill"},
        evidence_refs=("e1",),
        support={"n": 3},
    )
    outcome = PromotionGate(min_support=1).validate(unsupported)
    assert outcome.accepted is False
    assert outcome.reason == "unsupported proposal kind for promotion: ProposeSkillSpec"


def test_promotion_gate_prefers_independent_case_support_when_present():
    duplicate_only = _memory_proposal(support_n=5)
    duplicate_only.support["n_unique_cases"] = 1

    rejected = PromotionGate(min_support=2).validate(duplicate_only)

    assert rejected.accepted is False
    assert rejected.reason == "insufficient support: 1 < 2"

    independent = _memory_proposal(support_n=5)
    independent.support["n_unique_cases"] = 2

    accepted = PromotionGate(min_support=2).validate(independent)

    assert accepted.accepted is True
