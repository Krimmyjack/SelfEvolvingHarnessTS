from SelfEvolvingHarnessTS.memory import EvidenceRecord, EvidenceStore
from SelfEvolvingHarnessTS.slow_path.evidence_miner import (
    DeploymentEvidenceMiner,
    suggest_slow_path_proposals,
)
from SelfEvolvingHarnessTS.slow_path.proposal_schema import SlowProposal, validate_slow_proposal


def _record(
    *,
    cell="forecast|snrLow|miss",
    task="forecast",
    action="v_median",
    passed=True,
    utility=0.2,
    harm=0.0,
    failure=None,
    safety_reasons=(),
):
    return EvidenceRecord(
        conditioning_key={"task": {"type": task}, "cell_id": cell},
        cell_id=cell,
        harness_version=1,
        program={"source": "template", "steps": []},
        execution_trace=[],
        verification_result={
            "passed": passed,
            "failure_signature": failure,
            "output_status": "executed" if passed else "raw_fallback_not_compiled",
            "downstream": {
                "validator": "stub",
                "utility_delta_vs_raw": utility,
                "harm_delta_vs_raw": harm,
            },
        },
        routing={
            "selected_action": action,
            "candidate": {"skill_id": "median_smooth", "action_id": action},
            "safety": {"accepted": passed, "reasons": list(safety_reasons)},
        },
    )


def test_miner_summarizes_utility_harm_failures_and_safety_reasons():
    store = EvidenceStore()
    store.write(_record(action="v_median", utility=0.3, harm=0.0, passed=True))
    store.write(_record(action="v_median", utility=-0.1, harm=0.2, passed=False, failure="weak_support", safety_reasons=("weak_support",)))
    store.write(_record(action="v_none", utility=0.0, harm=0.0, passed=True))

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")

    assert summary.n_records == 3
    assert summary.n_passed == 2
    assert summary.action_stats["v_median"].n == 2
    assert summary.action_stats["v_median"].mean_utility_delta_vs_raw == 0.1
    assert summary.action_stats["v_median"].harm_count == 1
    assert summary.failure_signatures["weak_support"] == 1
    assert summary.safety_reasons["weak_support"] == 1


def test_suggest_proposals_emits_memory_write_for_clean_positive_utility():
    store = EvidenceStore()
    store.write(_record(action="v_median", utility=0.4, harm=0.0, passed=True))
    store.write(_record(action="v_median", utility=0.2, harm=0.0, passed=True))

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    proposals = suggest_slow_path_proposals(summary, min_support=1)

    kinds = {p.kind for p in proposals}
    assert "MemoryWrite" in kinds
    assert "ProposeRiskRule" not in kinds
    memory = next(p for p in proposals if p.kind == "MemoryWrite")
    assert memory.scope == "cell:forecast|snrLow|miss"
    assert memory.payload["action"] == "v_median"
    assert memory.payload["grounded_utility"] == 0.3


def test_suggest_proposals_emits_scoped_risk_rule_for_harm_conflict():
    store = EvidenceStore()
    store.write(_record(action="v_median", utility=0.4, harm=0.0, passed=True))
    store.write(_record(action="v_median", utility=-0.2, harm=0.3, passed=False, failure="harm", safety_reasons=("harm_rate_exceeds_policy",)))

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    proposals = suggest_slow_path_proposals(summary, min_support=1)

    kinds = {p.kind for p in proposals}
    assert "MemoryWrite" not in kinds
    assert "ProposeRiskRule" in kinds
    risk = next(p for p in proposals if p.kind == "ProposeRiskRule")
    assert risk.scope == "cell:forecast|snrLow|miss"
    assert risk.payload["then"]["op"] == "ban"
    assert risk.payload["when"]["base_action_in"] == ["v_median"]


def test_slow_proposal_validation_requires_kind_scope_and_evidence_refs():
    good = SlowProposal(
        kind="MemoryWrite",
        scope="cell:forecast|snrLow|miss",
        payload={
            "pattern_region": "forecast|snrLow|miss",
            "action": "v_median",
            "grounded_utility": 0.2,
            "utility_ci": [0.2, 0.2],
            "scope": "cell:forecast|snrLow|miss",
            "source": "deployment_evidence_miner",
        },
        evidence_refs=("forecast|snrLow|miss:v_median",),
        support={"n": 2},
    )
    assert validate_slow_proposal(good) is None

    bad = SlowProposal(kind="MemoryWrite", scope="", payload={}, evidence_refs=(), support={})
    assert validate_slow_proposal(bad) == "proposal requires non-empty scope"

def test_memory_write_proposal_uses_task_from_evidence_and_edit_schema():
    store = EvidenceStore()
    store.write(_record(cell="classify|snrLow|miss", task="classify", action="v_median", utility=0.4))

    summary = DeploymentEvidenceMiner(store).summarize_cell("classify|snrLow|miss")
    proposals = suggest_slow_path_proposals(summary, min_support=1)

    memory = next(p for p in proposals if p.kind == "MemoryWrite")
    assert memory.payload["task"] == "classify"
    assert memory.payload["pattern_region"] == "classify|snrLow|miss"
    assert memory.payload["action"] == "v_median"
    assert memory.payload["scope"] == memory.scope
    assert validate_slow_proposal(memory) is None


def test_slow_proposal_validation_rejects_invalid_deployment_payloads():
    bad_memory = SlowProposal(
        kind="MemoryWrite",
        scope="cell:forecast|snrLow|miss",
        payload={"action_id": "not_a_real_action"},
        evidence_refs=("e1",),
        support={"n": 2},
    )
    assert "memory write invalid" in validate_slow_proposal(bad_memory)

    bad_risk = SlowProposal(
        kind="ProposeRiskRule",
        scope="cell:forecast|snrLow|miss",
        payload={
            "rule_id": "bad",
            "when": {},
            "then": {"op": "ban", "action": "not_a_real_action"},
            "scope": "cell:forecast|snrLow|miss",
        },
        evidence_refs=("e2",),
        support={"n": 2},
    )
    assert "risk rule invalid" in validate_slow_proposal(bad_risk)

def test_suggest_proposals_does_not_promote_raw_action_memory_by_default():
    store = EvidenceStore()
    store.write(_record(action="v_none", utility=0.4, harm=0.0, passed=True))
    store.write(_record(action="v_none", utility=0.3, harm=0.0, passed=True))

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    proposals = suggest_slow_path_proposals(summary, min_support=1, raw_action="v_none")

    assert proposals == []



def test_miner_uses_independent_source_uid_for_support_threshold():
    store = EvidenceStore()
    first = _record(action="v_median", utility=0.4, harm=0.0, passed=True)
    duplicate = _record(action="v_median", utility=0.3, harm=0.0, passed=True)
    first.routing["source_uid"] = "case-1"
    duplicate.routing["source_uid"] = "case-1"
    store.write(first)
    store.write(duplicate)

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    support = summary.action_stats["v_median"].to_support()

    assert support["n"] == 2
    assert support["n_unique_cases"] == 1
    assert suggest_slow_path_proposals(summary, min_support=2) == []

    second = _record(action="v_median", utility=0.5, harm=0.0, passed=True)
    second.routing["source_uid"] = "case-2"
    store.write(second)

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    proposals = suggest_slow_path_proposals(summary, min_support=2)
    memory = next(p for p in proposals if p.kind == "MemoryWrite")

    assert memory.support["n_unique_cases"] == 2
    assert memory.support["utility_positive_case_count"] == 2


def test_miner_reports_case_averaged_utility_to_avoid_arm_weighting():
    store = EvidenceStore()
    a = _record(action="v_median", utility=1.0, harm=0.0, passed=True)
    b = _record(action="v_median", utility=1.0, harm=0.0, passed=True)
    c = _record(action="v_median", utility=0.5, harm=0.0, passed=True)
    a.routing["source_uid"] = "case-1"
    b.routing["source_uid"] = "case-1"
    c.routing["source_uid"] = "case-2"
    store.write(a)
    store.write(b)
    store.write(c)

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    support = summary.action_stats["v_median"].to_support()

    assert support["mean_utility_delta_vs_raw"] == 0.833333333333
    assert support["mean_case_utility_delta_vs_raw"] == 0.75
    assert support["mean_case_harm_delta_vs_raw"] == 0.0

    proposals = suggest_slow_path_proposals(summary, min_support=1)
    memory = next(p for p in proposals if p.kind == "MemoryWrite")

    assert memory.payload["grounded_utility"] == 0.75
    assert memory.payload["harm_delta_vs_raw"] == 0.0


def test_miner_does_not_memory_write_when_harm_cases_conflict():
    store = EvidenceStore()
    pos_a = _record(action="v_median", utility=0.4, harm=0.0, passed=True)
    pos_b = _record(action="v_median", utility=0.3, harm=0.0, passed=True)
    harm_a = _record(action="v_median", utility=-0.2, harm=0.2, passed=False)
    harm_b = _record(action="v_median", utility=-0.1, harm=0.1, passed=False)
    pos_a.routing["source_uid"] = "pos-a"
    pos_b.routing["source_uid"] = "pos-b"
    harm_a.routing["source_uid"] = "harm-a"
    harm_b.routing["source_uid"] = "harm-b"
    for row in (pos_a, pos_b, harm_a, harm_b):
        store.write(row)

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    proposals = suggest_slow_path_proposals(summary, min_support=2)

    assert not [p for p in proposals if p.kind == "MemoryWrite"]
    assert [p for p in proposals if p.kind == "ProposeRiskRule"]


def test_miner_counts_positive_utility_cases_by_case_mean():
    store = EvidenceStore()
    positive = _record(action="v_median", utility=1.0, harm=0.0, passed=True)
    offsetting = _record(action="v_median", utility=-1.0, harm=0.0, passed=False)
    weak_positive = _record(action="v_median", utility=0.2, harm=0.0, passed=True)
    positive.routing["source_uid"] = "case-1"
    offsetting.routing["source_uid"] = "case-1"
    weak_positive.routing["source_uid"] = "case-2"
    store.write(positive)
    store.write(offsetting)
    store.write(weak_positive)

    summary = DeploymentEvidenceMiner(store).summarize_cell("forecast|snrLow|miss")
    support = summary.action_stats["v_median"].to_support()

    assert support["utility_positive_count"] == 2
    assert support["utility_positive_case_count"] == 1
    assert suggest_slow_path_proposals(summary, min_support=2) == []


