"""P1/P2 tests: public Program Supply router, online adapter, and one-surface catalog."""

import inspect
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from SelfEvolvingHarnessTS.contracts.harness import (
    EditManifest,
    EditOperation,
    SkillKind,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import (
    PROGRAM_SUPPLY_ROUTE_FIELDS,
    CaseFacts,
    assess_case,
    route_program_supply_fault,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    ProgramSupplyDecision,
    build_program_supply_facts,
    build_single_surface_catalog,
    route_online_program_supply_fault,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
    _edit_rule_for_catalog,
    _resolve_apply_manifest,
)
from SelfEvolvingHarnessTS.runtime.decision_trace import DecisionTrace

ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"
RULES_ROOT = ROOT / "evaluation/minipipe/config/m0_rules.json"


def _route_kwargs(**changes):
    kwargs = {
        "expressibility_status": "PROVEN_EXPRESSIBLE",
        "expressibility_cause": None,
        "capability_skill_exists": True,
        "skill_retrieved": True,
        "constrained_proposal_succeeds": False,
    }
    kwargs.update(changes)
    return kwargs


def test_route_function_requires_every_program_supply_field():
    signature = inspect.signature(route_program_supply_fault)
    assert set(PROGRAM_SUPPLY_ROUTE_FIELDS).issubset(set(signature.parameters))
    for name in PROGRAM_SUPPLY_ROUTE_FIELDS:
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        route_program_supply_fault()  # type: ignore[call-arg]


def test_route_function_covers_each_program_supply_branch():
    assert route_program_supply_fault(
        expressibility_status="PROVEN_UNAVAILABLE",
        expressibility_cause=None,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    ) == ("OPERATOR_GAP", "CAPABILITY_BACKLOG", ())
    assert route_program_supply_fault(
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause="OBSERVABLE_DERIVATION_PROCEDURE_GAP",
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    ) == (
        "OBSERVABLE_DERIVATION_PROCEDURE_GAP",
        "EDITABLE_M0",
        ("bootstrap_skills.entries/inspect_and_localize.body",),
    )
    assert route_program_supply_fault(
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause="OBSERVABLE_FEATURE_SCHEMA_GAP",
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    ) == ("OBSERVABLE_FEATURE_SCHEMA_GAP", "OBSERVATION_CAPABILITY_BACKLOG", ())
    assert route_program_supply_fault(
        expressibility_status="EXPRESSIBILITY_UNKNOWN",
        expressibility_cause=None,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    ) == ("EXPRESSIBILITY_UNKNOWN", "EVIDENCE_BACKLOG", ())
    assert route_program_supply_fault(
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause=None,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    ) == ("SKILL_LIBRARY_GAP", "EDITABLE_M0", ("skill_library.entries/{skill_id}",))
    assert route_program_supply_fault(
        **_route_kwargs(constrained_proposal_succeeds=True)
    ) == (
        "PROPOSAL_CONTROL_GAP",
        "EDITABLE_M0",
        ("candidate_policy.proposal_guidance",),
    )
    assert route_program_supply_fault(
        **_route_kwargs(constrained_proposal_succeeds=False)
    ) == (
        "SKILL_CONTENT_GAP",
        "EDITABLE_M0",
        ("skill_library.entries/{skill_id}.body",),
    )
    # Unknown must ABSTAIN, never silently become False.
    assert route_program_supply_fault(
        **_route_kwargs(constrained_proposal_succeeds=None)
    ) == ("CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ())
    assert route_program_supply_fault(
        **_route_kwargs(skill_retrieved=False, constrained_proposal_succeeds=None)
    ) == ("CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ())
    with pytest.raises(ValueError):
        route_program_supply_fault(
            **_route_kwargs(constrained_proposal_succeeds="False")  # type: ignore[arg-type]
        )


def test_supply_failure_is_a_delegate_not_a_private_copy(monkeypatch):
    import SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault as ff

    captured = {}

    def fake_router(**kwargs):
        captured.update(kwargs)
        return ("SENTINEL_CAUSE", "EVIDENCE_BACKLOG", ())

    monkeypatch.setattr(ff, "route_program_supply_fault", fake_router)
    facts = CaseFacts.passing(case_id="m0-delegate")
    assert ff._supply_failure(facts) == ("SENTINEL_CAUSE", "EVIDENCE_BACKLOG", ())
    assert set(captured) == set(PROGRAM_SUPPLY_ROUTE_FIELDS)
    assert captured["constrained_proposal_succeeds"] is None


def test_assess_case_program_supply_branch_uses_the_same_public_router():
    facts = CaseFacts.passing(case_id="m0-0001")
    facts = replace(
        facts,
        candidate_utilities={"identity": -0.4},
        effect_distinct_candidate_ids=(),
        capability_skill_exists=True,
        normal_retrieval=True,
        skill_retrieved=True,
        forced_skill_succeeds=False,
        expressibility_status="PROVEN_EXPRESSIBLE",
        constrained_proposal_succeeds=None,
    )
    result = assess_case(facts, rules=__import__(
        "SelfEvolvingHarnessTS.evaluation.minipipe.config",
        fromlist=["load_m0_rules"]).load_m0_rules(RULES_ROOT))
    assert result.attribution.cause_code == "CANDIDATE_SUPPLY_UNKNOWN"
    assert result.attribution.actionability == "EVIDENCE_BACKLOG"

    fixed_false = replace(facts, constrained_proposal_succeeds=False)
    assert assess_case(fixed_false, rules=__import__(
        "SelfEvolvingHarnessTS.evaluation.minipipe.config",
        fromlist=["load_m0_rules"]).load_m0_rules(RULES_ROOT)
    ).attribution.cause_code == "SKILL_CONTENT_GAP"


def _trace(case_id="case-1"):
    return DecisionTrace(
        case_id=case_id,
        public_observation_ids=(),
        inspected_regions=(),
        tool_calls=(),
        retrieved_skill_ids=("cap-1", "build_contrastive_candidates"),
        retrieved_memory_ids=(),
        applicability_matches=(),
        candidate_ids=("identity",),
        candidate_program_shas=(None,),
        chosen_candidate_id="identity",
        compilation_status="OK",
        execution_status="OK",
        modified_indices=(),
        verification_actions=(),
        effect_equivalent_to_identity=True,
    )


def _view(*, with_capability=True):
    skills = [
        SimpleNamespace(
            skill_id="build_contrastive_candidates",
            skill_kind=SkillKind.BOOTSTRAP_PROCEDURE,
        )
    ]
    if with_capability:
        skills.append(SimpleNamespace(skill_id="cap-1", skill_kind=SkillKind.CAPABILITY))
    return SimpleNamespace(skills=tuple(skills))


def test_adapter_assigns_every_route_field_explicitly_and_abstains():
    facts = build_program_supply_facts(_trace(), object(), _view())
    # Enumerate the fields read by the router; the adapter must have assigned
    # each one explicitly (the dataclass itself has no defaults).
    for name in PROGRAM_SUPPLY_ROUTE_FIELDS:
        assert name in {item.name for item in fields(facts)}
    assert facts == build_program_supply_facts(_trace(), object(), _view())
    assert facts.case_id == "case-1"
    assert facts.expressibility_status == "EXPRESSIBILITY_UNKNOWN"
    assert facts.expressibility_cause is None
    assert facts.capability_skill_exists is True
    assert facts.skill_retrieved is True
    assert facts.constrained_proposal_succeeds is None

    decision = route_online_program_supply_fault(_trace(), object(), _view())
    assert decision.cause_code == "EXPRESSIBILITY_UNKNOWN"
    assert decision.actionability == "EVIDENCE_BACKLOG"
    assert decision.surface_templates == ()


def test_adapter_without_capability_skill_does_not_guess_cause():
    facts = build_program_supply_facts(_trace(), object(), _view(with_capability=False))
    assert facts.capability_skill_exists is False
    assert facts.skill_retrieved is False
    decision = route_online_program_supply_fault(
        _trace(), object(), _view(with_capability=False)
    )
    # Expressibility is unknown online, so this is an evidence backlog rather
    # than a guessed SKILL_LIBRARY_GAP.
    assert decision.cause_code == "EXPRESSIBILITY_UNKNOWN"
    assert decision.surface_templates == ()


@pytest.fixture
def h0_materialized(tmp_path):
    store = SnapshotStore(tmp_path / "store")
    return store.materialize(compile_snapshot(H0_ROOT, verify_lock=False))


@pytest.fixture
def controller(h0_materialized):
    return EditController(
        SnapshotStore(h0_materialized.root.parent),
        surfaces=SurfaceRegistry(),
        router=FaultRouter(),
    )


def _add_capability(parent, controller, *, skill_id="existing_skill"):
    manifest = EditManifest(
        edit_id=f"add-{skill_id}",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id="skill_library.entries/{skill_id}",
        operation=EditOperation.ADD,
        surface_precondition={"kind": "ABSENT"},
        dependency_precondition_shas={},
        new_value={
            "schema_version": "skill-entry/1",
            "skill_id": skill_id,
            "skill_kind": "capability",
            "revision": 1,
            "body": "OLD BODY",
            "observable_applicability": {"const": True},
            "allowed_tools": ["winsorize"],
            "risk_guards": {"explicit_choice_required": True},
        },
        observable_applicability={"const": True},
        predicted_agent_behavior_change=(f"retrieve_skill:{skill_id}",),
        predicted_data_effect=("local_improvement",),
        falsification_condition=("no_improvement",),
    )
    applied = _resolve_apply_manifest(manifest, parent.snapshot)
    receipt = controller.apply_to_fork(
        parent, applied, confirmed_cause="SKILL_LIBRARY_GAP"
    )
    return receipt.candidate_snapshot


def test_catalog_for_library_gap_is_single_authorized_add_surface(
    h0_materialized, controller
):
    decision = ProgramSupplyDecision(
        "case-1", "SKILL_LIBRARY_GAP", "EDITABLE_M0",
        ("skill_library.entries/{skill_id}",),
    )
    catalog = build_single_surface_catalog(
        decision=decision, parent=h0_materialized, controller=controller
    )
    assert len(catalog) == 1
    assert catalog[0]["surface_id"] == "skill_library.entries/{skill_id}"
    assert catalog[0]["operation"] == "ADD"
    assert catalog[0]["allowed_operations"] == ["ADD"]
    assert catalog[0]["surface_precondition"] == {"kind": "ABSENT"}


def test_catalog_for_content_gap_picks_one_existing_retrieved_skill(
    h0_materialized, controller
):
    parent = _add_capability(h0_materialized, controller, skill_id="picked_skill")
    decision = ProgramSupplyDecision(
        "case-1", "SKILL_CONTENT_GAP", "EDITABLE_M0",
        ("skill_library.entries/{skill_id}.body",),
    )
    catalog = build_single_surface_catalog(
        decision=decision,
        parent=parent,
        controller=controller,
        retrieved_capability_skill_ids=("picked_skill",),
    )
    assert len(catalog) == 1
    assert catalog[0]["surface_id"] == "skill_library.entries/picked_skill.body"
    assert catalog[0]["operation"] == "PATCH"
    assert catalog[0]["allowed_operations"] == ["PATCH"]
    assert catalog[0]["surface_precondition"]["kind"] == "SHA"
    assert len(catalog[0]["surface_precondition"]["sha"]) == 64


def test_catalog_for_unknown_route_is_empty_abstain(h0_materialized, controller):
    decision = ProgramSupplyDecision(
        "case-1", "CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ()
    )
    assert build_single_surface_catalog(
        decision=decision, parent=h0_materialized, controller=controller
    ) == ()


def test_slow_instruction_matches_the_single_authorized_surface():
    add_rule = _edit_rule_for_catalog([{
        "surface_id": "skill_library.entries/{skill_id}",
        "operation": "ADD",
        "allowed_operations": ["ADD"],
    }])
    assert "ADD" in add_rule
    assert "PATCH is not authorized" in add_rule
    patch_rule = _edit_rule_for_catalog([{
        "surface_id": "skill_library.entries/existing_skill.body",
        "operation": "PATCH",
        "allowed_operations": ["PATCH"],
    }])
    assert "PATCH exactly" in patch_rule
    assert "ADD is not authorized" in patch_rule
    abstain_rule = _edit_rule_for_catalog([])
    assert "No writable surface is authorized" in abstain_rule


def test_program_slow_entrypoints_require_routed_authorization():
    for name in (
        "handle_feedback",
        "handle_feedback_support",
        "handle_fast_winner",
    ):
        parameter = inspect.signature(getattr(TTHAMethod, name)).parameters[
            "confirmed_cause"
        ]
        assert parameter.default is inspect.Parameter.empty
    group_sig = inspect.signature(TTHAMethod.handle_group_feedback)
    assert "confirmed_cause" not in group_sig.parameters
    assert group_sig.parameters["route_decision"].default is inspect.Parameter.empty
    assert group_sig.parameters["surface_catalog"].default is inspect.Parameter.empty


def test_catalog_for_missing_patch_target_is_empty_abstain(h0_materialized, controller):
    decision = ProgramSupplyDecision(
        "case-1", "SKILL_CONTENT_GAP", "EDITABLE_M0",
        ("skill_library.entries/{skill_id}.body",),
    )
    assert build_single_surface_catalog(
        decision=decision, parent=h0_materialized, controller=controller
    ) == ()
