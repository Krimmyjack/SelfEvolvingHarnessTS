from dataclasses import replace
from pathlib import Path

import pytest

from SelfEvolvingHarnessTS.evaluation.minipipe.config import load_m0_rules
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import (
    AssessmentStatus,
    CaseFacts,
    assess_case,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter


def _rules():
    root = Path(__file__).resolve().parents[2]
    return load_m0_rules(root / "evaluation/minipipe/config/m0_rules.json")


def _passing(**changes):
    return replace(CaseFacts.passing(case_id="m0-0001"), **changes)


def test_effective_candidate_present_but_unchosen_is_selection_miss():
    facts = _passing(
        candidate_utilities={"identity": -0.4, "agent-0": -0.2},
        effect_distinct_candidate_ids=("agent-0",),
        chosen_candidate_id="identity",
        chosen_gain=0.0,
    )
    result = assess_case(facts, rules=_rules())
    assert result.attribution.first_stage == "CANDIDATE_SELECTION"
    assert result.attribution.fault_code == "SELECTION_MISS"
    assert result.attribution.cause_code == "SELECTION_MISS"


def test_negative_matching_probe_and_selected_repair_routes_to_selection_control():
    facts = _passing(
        damage_d=-0.02,
        chosen_candidate_id="agent-0",
        chosen_probe_directions=("negative",),
        chosen_gain=-0.05,
    )
    result = assess_case(facts, rules=_rules())
    assert result.attribution.first_stage == "ELIGIBILITY"
    assert result.attribution.cause_code == "PROBE_SELECTION_CONTRADICTION"
    assert result.attribution.actionability == "EDITABLE_M0"
    assert result.attribution.suspect_surface_templates == (
        "candidate_policy.selection_guidance",
    )
    route = FaultRouter().authorize(
        "PROBE_SELECTION_CONTRADICTION",
        target_class="selection_control",
        operation="PATCH",
        target_surface_id="candidate_policy.selection_guidance",
    )
    assert route.actionability == "EDITABLE_M0"


def test_observable_witness_without_capability_skill_is_library_gap():
    facts = _passing(
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        chosen_candidate_id="identity",
        expressibility_status="PROVEN_EXPRESSIBLE",
        capability_skill_exists=False,
    )
    result = assess_case(facts, rules=_rules())
    assert result.attribution.first_stage == "CANDIDATE_SUPPLY"
    assert result.attribution.cause_code == "SKILL_LIBRARY_GAP"
    assert result.attribution.suspect_surface_templates == (
        "skill_library.entries/{skill_id}",
    )


def test_forced_existing_skill_success_is_retrieval_miss():
    facts = _passing(
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        capability_skill_exists=True,
        normal_retrieval=False,
        forced_skill_succeeds=True,
        skill_retrieved=False,
    )
    result = assess_case(facts, rules=_rules())
    assert result.attribution.first_stage == "RETRIEVAL_POLICY"
    assert result.attribution.cause_code == "RETRIEVAL_MISS"


def test_retrieved_skill_that_still_cannot_supply_is_content_gap():
    facts = _passing(
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        capability_skill_exists=True,
        normal_retrieval=True,
        skill_retrieved=True,
        forced_skill_succeeds=False,
        expressibility_status="PROVEN_EXPRESSIBLE",
    )
    result = assess_case(facts, rules=_rules())
    assert result.attribution.cause_code == "SKILL_CONTENT_GAP"


def test_period_unavailable_identity_is_agent_success_and_system_gap():
    facts = _passing(
        private_family="period_change",
        period_diagnostic_pass=True,
        public_probe_gains={},
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        chosen_candidate_id="identity",
        expressibility_status="PROVEN_UNAVAILABLE",
        required_transformation_class="period_correction",
    )
    result = assess_case(facts, rules=_rules())
    assert result.feedback.outcome.agent_decision_status == "CORRECT_IDENTITY"
    assert result.feedback.outcome.system_capability_status == "OPERATOR_GAP"
    assert result.attribution.actionability == "CAPABILITY_BACKLOG"


def test_uncertain_localization_stops_false_downstream_attribution():
    facts = _passing(
        localization_iou=0.20,
        candidate_utilities={"identity": -0.4, "agent-0": -0.2},
        effect_distinct_candidate_ids=("agent-0",),
        chosen_candidate_id="identity",
    )
    result = assess_case(facts, rules=_rules())
    assert result.assessments[2].status is AssessmentStatus.UNKNOWN
    assert result.attribution.fault_code == "LOCALIZATION_UNKNOWN"
    assert result.attribution.actionability == "EVIDENCE_BACKLOG"


def test_oracle_only_witness_cannot_sign_library_gap():
    facts = _passing(
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        expressibility_status="EXPRESSIBILITY_UNKNOWN",
        expressibility_cause="OBSERVABLE_DERIVATION_PROCEDURE_GAP",
        oracle_witness_succeeded=True,
        observable_witness_succeeded=False,
        capability_skill_exists=False,
    )
    result = assess_case(facts, rules=_rules())
    assert result.attribution.cause_code == "OBSERVABLE_DERIVATION_PROCEDURE_GAP"
    assert result.attribution.cause_code != "SKILL_LIBRARY_GAP"


def test_fault_router_enforces_skill_kind_and_noneditable_backlogs():
    router = FaultRouter()
    library = router.allowed_targets("SKILL_LIBRARY_GAP")
    assert library.allowed_skill_kinds == ("capability",)
    assert library.allowed_operations == ("ADD",)
    with pytest.raises(ValueError):
        router.authorize(
            "SKILL_LIBRARY_GAP",
            target_class="bootstrap_procedure",
            skill_kind="bootstrap_procedure",
            operation="ADD",
        )
    for cause in (
        "OPERATOR_GAP",
        "EXPRESSIBILITY_UNKNOWN",
        "OBSERVABLE_FEATURE_SCHEMA_GAP",
    ):
        assert router.allowed_targets(cause).allowed_operations == ()


def test_confirmed_surface_is_null_before_single_surface_replay():
    result = assess_case(_passing(), rules=_rules())
    assert result.feedback.update_attribution.confirmed_surface is None
    assert len(result.assessments) == 10
