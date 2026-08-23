"""Focused G1 protocol test: General proposal-guidance repair.

No live LLM and no real Outcome: the Slow calls, the compiler and the probe
evaluator are stubbed.  The test covers exactly the four properties requested
in docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md rev2.2 section 10
item 6:

1. single-Surface permission -- cause PROPOSAL_CONTROL_GAP authorizes
   ``candidate_policy.proposal_guidance`` and nothing else, and an applied
   PATCH changes only that key;
2. the authorized guidance really reaches the E1 proposal payload and system
   instruction;
3. under ``post_shift_support_sufficient=false`` a guidance-following proposal
   stage stops leading with the known-harmful bare mechanism;
4. under ``post_shift_support_sufficient=true`` the same guidance leaves the
   mechanism's proposal eligibility intact, i.e. no global suppression.

Items 3 and 4 test the *plumbing* with a deterministic guidance-following
stub.  Whether the real Slow proposal model follows the patched guidance is a
behavioral question and is answered only by the live paired replay, never by
this test.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from evaluation.functional.task_episode_harness import e1, g1, runner
from SelfEvolvingHarnessTS.contracts.program_supply import (
    route_program_supply_fault,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditAuthorizationError,
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.contracts.candidate import Candidate
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import CompiledWorkflow
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore

PATCHED_GUIDANCE = (
    "Supply only minimal effect-distinct candidates justified by public "
    "evidence. When post_shift_support_sufficient is false, do not lead with a "
    "bare level-shift repair; prefer a different family or abstain. When it is "
    "true, a level-shift repair remains a legal first proposal."
)


def _public_context(observation_cutoff: int, *, sufficient: bool) -> dict:
    # The three public bindings repair_level_shift declares must be present,
    # otherwise the operator is UNAVAILABLE for a reason that has nothing to do
    # with the guidance under test.
    features = {
        "task_kind": "forecast",
        "estimated_region_start_fraction": "very_low",
        "estimated_region_end_fraction": 0.77 if sufficient else 0.99,
        "estimated_level_offset": 63.0,
        g1.G1_CONDITION_FEATURE: sufficient,
    }
    return {
        "task_kind": "forecast",
        "observation_cutoff": observation_cutoff,
        "scope_feature": "local_robust_z_peak",
        "scope_bin": "high",
        "projection_feature": "estimated_region_start_fraction",
        "scope_series_uids": ["T153", "T154"],
        "representative_uid": "T153",
        "representative_features": dict(features),
        "task_signature": {
            "task_kind": "forecast",
            "estimated_region_start_fraction": "very_low",
        },
        "task_fast_features": dict(features),
    }


def _task_spec(task_id: str, base: int) -> dict:
    return {
        "task_episode_id": task_id,
        "arm_order": "A3_A5",
        "horizon": 48,
        "support_origins": (base, base + 48, base + 96),
        "delayed_origins": (base + 144, base + 192, base + 240),
    }


def _probe_metrics(*args, **kwargs) -> dict:
    return {
        "macro_gain": 0.01,
        "se_block": 0.02,
        "gain_over_se": 0.5,
        "per_series_mean_gain": {},
        "per_origin_gain": {},
        "positive_series_count": 2,
        "negative_series_count": 0,
        "modified_point_count": 1,
        "program_steps": [],
    }


def _fake_compile(proposal, inventory, public_context, *, generation):
    op = proposal["steps"][0]["op"]
    return runner._compiled(op, name=f"test-{op}"), {
        "decision": "PROPOSE",
        "steps": proposal["steps"],
    }


class _GuidanceFollowingSlow:
    """Deterministic stub: proposes the mechanism unless guidance forbids it."""

    def __init__(self) -> None:
        self.proposal_systems: list[str] = []
        self.payloads: list[dict] = []

    def __call__(self, messages: list[dict]) -> dict:
        system = messages[0]["content"]
        if system == e1._DECISION_SYSTEM:
            return {"decision": "TRUST_DRAFT", "reason": "stub"}
        assert system in {
            e1._E1_PROPOSAL_SYSTEM,
            e1._E1_PROPOSAL_SYSTEM_WITH_GUIDANCE,
        }, system[:80]
        payload = json.loads(messages[1]["content"])
        self.proposal_systems.append(system)
        self.payloads.append(payload)
        guidance = str(payload.get("candidate_policy_proposal_guidance") or "")
        sufficient = bool(
            payload["target_public_context"]["representative_features"][
                g1.G1_CONDITION_FEATURE
            ]
        )
        op = g1.G1_MECHANISM_PROGRAM[0]
        if "do not lead with a bare level-shift repair" in guidance and not sufficient:
            op = "hampel_filter"
        return {
            "decision": "PROPOSE",
            "reason": "stub",
            "proposals": [
                {
                    "steps": [{"op": op, "params": {}, "bindings": {}}],
                    "requested_observations": [],
                    "fallback": "IDENTITY",
                    "experience_use": [],
                }
            ],
        }


def _arm_state(tmp_path: Path, arm: str, snapshot):
    store = SnapshotStore(tmp_path / arm / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    return e1._ArmState(
        arm=arm,
        memories=[],
        episodes=[],
        store=store,
        active_snapshot=snapshot,
        active_skill_ids=[],
    )


def _h0():
    return compile_snapshot(
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False
    )


# ------------------------------------------------------------------ item 1


def test_proposal_control_gap_is_reachable_only_with_the_runtime_predicate():
    """The narrow route repair opens one edge and leaves the table alone."""
    facts = {
        "expressibility_status": "PROVEN_EXPRESSIBLE",
        "expressibility_cause": None,
        "capability_skill_exists": True,
        "skill_retrieved": False,
        "constrained_proposal_succeeds": None,
    }
    # Pre-existing behavior: a Skill exists but was not retrieved -> not editable.
    assert route_program_supply_fault(**facts) == (
        "CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ()
    )
    # G1 edge: exactly one cause, one actionability, one surface.
    assert route_program_supply_fault(
        **facts, context_resolved_decision_fault=True
    ) == (
        "PROPOSAL_CONTROL_GAP",
        "EDITABLE_M0",
        ("candidate_policy.proposal_guidance",),
    )
    # The new predicate can never re-route a case an earlier branch owns.
    assert route_program_supply_fault(
        expressibility_status="PROVEN_UNAVAILABLE",
        expressibility_cause=None,
        capability_skill_exists=True,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
        context_resolved_decision_fault=True,
    ) == ("OPERATOR_GAP", "CAPABILITY_BACKLOG", ())
    assert route_program_supply_fault(
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause=None,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
        context_resolved_decision_fault=True,
    ) == (
        "SKILL_LIBRARY_GAP", "EDITABLE_M0", ("skill_library.entries/{skill_id}",)
    )


def test_single_surface_permission_and_diff(tmp_path):
    """PROPOSAL_CONTROL_GAP may patch one surface, and only that key moves."""
    router = FaultRouter()
    route = router.allowed_targets("PROPOSAL_CONTROL_GAP")
    assert route.actionability == "EDITABLE_M0"
    assert route.allowed_operations == ("PATCH",)
    assert route.allowed_surface_ids == ("candidate_policy.proposal_guidance",)

    h0 = _h0()
    store = SnapshotStore(tmp_path / "snapshots")
    store.materialize(h0)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    receipt = g1._apply_guidance_patch(controller, store, h0, PATCHED_GUIDANCE)
    patched = receipt.candidate_snapshot.snapshot

    assert receipt.confirmed_cause == "PROPOSAL_CONTROL_GAP"
    assert list(receipt.source_surfaces_changed) == [g1.G1_SURFACE]
    before, after = dict(h0.candidate_policy), dict(patched.candidate_policy)
    assert sorted(
        key for key in set(before) | set(after) if before.get(key) != after.get(key)
    ) == ["proposal_guidance"]
    assert after["proposal_guidance"] == PATCHED_GUIDANCE
    # Nothing else in the Harness moved.
    assert e1._skill_ids(h0) == e1._skill_ids(patched)
    assert dict(h0.retrieval) == dict(patched.retrieval)
    assert dict(h0.verification) == dict(patched.verification)

    # A second surface is not authorized under the same cause.
    with pytest.raises((EditAuthorizationError, ValueError)):
        router.authorize(
            "PROPOSAL_CONTROL_GAP",
            target_class="capability",
            operation="PATCH",
            skill_kind="capability",
            target_surface_id="skill_library.entries/{skill_id}.body",
        )


# ------------------------------------------------------------------ item 2


def test_guidance_from_the_arm_snapshot_reaches_the_proposal_payload(
    monkeypatch, tmp_path
):
    slow = _GuidanceFollowingSlow()
    monkeypatch.setattr(e1, "_e1_slow_call", slow)
    monkeypatch.setattr(e1, "_probe_compiled", _probe_metrics)
    monkeypatch.setattr(e1, "_compile_proposal", _fake_compile)

    h0 = _h0()
    state = _arm_state(tmp_path, "A3", h0)
    row = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=state,
        task_spec=_task_spec("g1_test_01", 100),
        public_context=_public_context(100, sufficient=True),
        source_prior=None,
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=[0],
        consume_proposal_guidance=True,
    )
    base_guidance = str(h0.candidate_policy["proposal_guidance"])
    assert row["proposal_guidance_consumed"] == base_guidance
    assert row["payload"]["candidate_policy_proposal_guidance"] == base_guidance
    assert slow.proposal_systems[0] == e1._E1_PROPOSAL_SYSTEM_WITH_GUIDANCE

    # Opting out keeps the frozen E1-v2 payload shape byte-for-byte.
    slow.proposal_systems.clear()
    frozen = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=_arm_state(tmp_path, "A3_frozen", _h0()),
        task_spec=_task_spec("g1_test_02", 700),
        public_context=_public_context(700, sufficient=True),
        source_prior=None,
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=[0],
    )
    assert "candidate_policy_proposal_guidance" not in frozen["payload"]
    assert frozen["proposal_guidance_consumed"] is None
    assert slow.proposal_systems[0] == e1._E1_PROPOSAL_SYSTEM


# ------------------------------------------------------------- items 3 and 4


@pytest.mark.parametrize("sufficient", [False, True])
def test_patched_guidance_is_context_conditional_not_a_global_ban(
    monkeypatch, tmp_path, sufficient
):
    """False Context loses the harmful lead; true Context keeps eligibility."""
    slow = _GuidanceFollowingSlow()
    monkeypatch.setattr(e1, "_e1_slow_call", slow)
    monkeypatch.setattr(e1, "_probe_compiled", _probe_metrics)
    monkeypatch.setattr(e1, "_compile_proposal", _fake_compile)

    h0 = _h0()
    store = SnapshotStore(tmp_path / "patched" / "snapshots")
    store.materialize(h0)
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    patched = g1._apply_guidance_patch(
        controller, store, h0, PATCHED_GUIDANCE
    ).candidate_snapshot.snapshot

    context = _public_context(100, sufficient=sufficient)
    spec = _task_spec("g1_test_03", 100)
    base_row = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=_arm_state(tmp_path, "base", h0),
        task_spec=spec,
        public_context=context,
        source_prior=None,
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=[0],
        consume_proposal_guidance=True,
    )
    patched_row = e1._run_arm(
        repo_root=PROJECT_ROOT,
        arm_state=_arm_state(tmp_path, "patched_arm", patched),
        task_spec=spec,
        public_context=context,
        source_prior=None,
        inventory=[],
        values={},
        mapped_roster=[],
        config={},
        eval_uids=[],
        llm_counter=[0],
        consume_proposal_guidance=True,
    )
    base_lead = g1._mechanism_first_probe(base_row)
    patched_lead = g1._mechanism_first_probe(patched_row)

    # The base arm always leads with the bare mechanism.
    assert base_lead["is_mechanism"] is True
    if sufficient:
        # True Context: eligibility survives -- this is not a global ban.
        assert patched_lead["is_mechanism"] is True
        assert (
            g1._arm_mechanism_stats(patched_row)["mechanism_probe_count"] > 0
        )
    else:
        # False Context: the known-harmful lead is gone, capability is not.
        assert patched_lead["is_mechanism"] is False
        assert g1._arm_mechanism_stats(patched_row)["mechanism_probe_count"] == 0
        # Capability was never removed: the operator is still executable in
        # this Context, the guidance only changed what is proposed first.
        executable = {
            str(row["name"]) for row in e1._inventory_rows(context)
            if row.get("availability") == "EXECUTABLE"
        }
        assert g1.G1_MECHANISM_PROGRAM[0] in executable


# --------------------------------------------------- Runtime attribution gate


def test_evidence_census_is_complete_deduplicated_and_unfiltered():
    """The load-bearing regression test for the 2026-08-18 Planner ruling.

    rev1 kept only the POSITIVE rows of mechanism-bearing alternative programs,
    so the Slow Agent never saw the same program's NEGATIVE rows in the same
    Context.  The census must now emit both relations of a program in one
    Context, must include programs that do not carry the mechanism operator,
    and must weigh evidence in distinct Task Episodes, not in attempts.
    """
    rows = [
        # one Task, both arms -> one distinct Task, two attempts
        {"task_episode_id": "t1", "arm": "A3", "program": ["repair_level_shift"],
         "gain_readable": True, "support_gain": -5.0,
         g1.G1_CONDITION_FEATURE: False},
        {"task_episode_id": "t1", "arm": "A5", "program": ["repair_level_shift"],
         "gain_readable": True, "support_gain": -5.0,
         g1.G1_CONDITION_FEATURE: False},
        # same program, same Context, opposite relation: must NOT be dropped
        {"task_episode_id": "t2", "arm": "A5",
         "program": ["outlier_mad", "repair_level_shift"],
         "gain_readable": True, "support_gain": 0.3,
         g1.G1_CONDITION_FEATURE: False},
        {"task_episode_id": "t3", "arm": "A5",
         "program": ["outlier_mad", "repair_level_shift"],
         "gain_readable": True, "support_gain": -9.0,
         g1.G1_CONDITION_FEATURE: False},
        # a program without the mechanism operator is still evidence
        {"task_episode_id": "t4", "arm": "A3", "program": ["hampel_filter"],
         "gain_readable": True, "support_gain": -0.2,
         g1.G1_CONDITION_FEATURE: False},
        # unreadable rows never enter the census
        {"task_episode_id": "t5", "arm": "A3", "program": ["outlier_mad"],
         "gain_readable": False, "support_gain": None,
         g1.G1_CONDITION_FEATURE: False},
    ]
    census = g1._program_evidence_census(rows)
    cells = {
        (tuple(c["canonical_program"]), c[g1.G1_CONDITION_FEATURE],
         c["support_relation"]): c
        for c in census
    }
    # de-duplication: two arm attempts on one Task are one Task of evidence
    bare = cells[(("repair_level_shift",), False, "NEGATIVE")]
    assert bare["distinct_task_count"] == 1
    assert bare["attempt_count"] == 2

    # both relations of the same program in the same Context survive
    combined = ("outlier_mad", "repair_level_shift")
    assert (combined, False, "POSITIVE") in cells
    assert (combined, False, "NEGATIVE") in cells
    assert cells[(combined, False, "POSITIVE")]["distinct_task_count"] == 1
    assert cells[(combined, False, "NEGATIVE")]["distinct_task_count"] == 1

    # non-mechanism programs are present and flagged
    assert (("hampel_filter",), False, "NEGATIVE") in cells
    assert cells[(("hampel_filter",), False, "NEGATIVE")][
        "contains_mechanism_operator"
    ] is False
    assert cells[(combined, False, "POSITIVE")][
        "contains_mechanism_operator"
    ] is True

    # unreadable rows are excluded
    assert not any(c["canonical_program"] == ["outlier_mad"] for c in census)


def test_slow_input_carries_the_census_and_the_per_clause_threshold():
    payload = g1._slow_guidance_payload(
        {
            "evidence_census": [
                {
                    "canonical_program": ["outlier_mad", "repair_level_shift"],
                    "contains_mechanism_operator": True,
                    g1.G1_CONDITION_FEATURE: False,
                    "support_relation": "POSITIVE",
                    "distinct_task_count": 1,
                    "distinct_task_episode_ids": ["t2"],
                    "attempt_count": 1,
                }
            ],
            "evidence_census_contract": {"unit_of_evidence": "distinct_task_count"},
        },
        "base guidance",
    )
    threshold = payload["active_clause_evidence_threshold"]
    assert threshold["unit"] == "distinct_task_count"
    assert threshold["minimum"] == g1.GENERAL_EVIDENCE_MIN_DISTINCT_TASKS == 2
    # The single-Task cell above is exactly the one that must not authorize an
    # active pairing clause.
    assert payload["evidence_census"][0]["distinct_task_count"] < threshold[
        "minimum"
    ]
    assert "distinct_task_count" in g1._G1_SLOW_SYSTEM
    assert "attempt_count is diagnostic" in g1._G1_SLOW_SYSTEM
    assert "never authorizes a new active recommendation" in g1._G1_SLOW_SYSTEM


def test_runtime_owns_the_cause_and_the_llm_never_reports_one():
    """The Slow contract exposes a Surface, never a Cause field."""
    assert "'decision':'PATCH'" in g1._G1_SLOW_SYSTEM
    assert "'decision':'ABSTAIN'" in g1._G1_SLOW_SYSTEM
    payload = g1._slow_guidance_payload(
        {"evidence_census": [], "evidence_census_contract": {}},
        "base guidance",
    )
    assert payload["attributed_cause"] == "PROPOSAL_CONTROL_GAP"
    assert [entry["surface_id"] for entry in payload["surface_catalog"]] == [
        g1.G1_SURFACE
    ]
    # The Slow input carries relations and Contexts, never raw trajectories or
    # utility numbers.
    serialized = json.dumps(payload)
    assert "support_gain" not in serialized
    assert "macro_gain" not in serialized
    assert "probes" not in serialized


# ------------------------------------------------- W1 parameterized Skill reuse


def _skill(skill_id: str, steps: list[dict]):
    """A minimal stand-in for a machine-added Target-local capability Skill."""
    from types import SimpleNamespace

    body = "Frozen program steps: " + json.dumps(steps)
    return SimpleNamespace(skill_id=skill_id, body=body)


def test_same_structure_and_binding_source_reuses_across_tasks():
    """W1: only the declared bound values differ, so it is the same Skill.

    These are the exact compiled steps of the three G1 fresh Tasks that raised
    AddTargetExistsError (22 / 23 / 25): identical operator structure, and the
    only differences are repair_level_shift's three declared public bindings.
    """
    task22 = [
        {"op": "outlier_mad", "params": {}},
        {"op": "repair_level_shift", "params": {
            "estimated_offset": 55.5,
            "region_end_fraction": 0.9972587719298246,
            "region_start_fraction": 0.0014254385964912282}},
    ]
    task23 = [
        {"op": "outlier_mad", "params": {}},
        {"op": "repair_level_shift", "params": {
            "estimated_offset": 55.5,
            "region_end_fraction": 0.9979804421768708,
            "region_start_fraction": 0.0013818027210884354}},
    ]
    snapshot = SimpleNamespace(
        skills=(_skill("fast_winner_forecast_ridge_smase_e1v2_outlier_mad_repair_level_shift", task22),)
    )
    current = [(s["op"], s["params"]) for s in task23]
    assert e1._existing_local_skill(snapshot, current) is not None

    # Identity ignores only the declared bindings, never the whole parameter set.
    assert e1._binding_free_signature(
        [(s["op"], s["params"]) for s in task22]
    ) == e1._binding_free_signature(current)


def test_different_structure_or_constant_is_never_merged():
    """W1: a different operator structure or a non-bound constant stays distinct."""
    stored = [
        {"op": "outlier_mad", "params": {}},
        {"op": "repair_level_shift", "params": {
            "estimated_offset": 55.5, "region_end_fraction": 0.99,
            "region_start_fraction": 0.001}},
    ]
    snapshot = SimpleNamespace(
        skills=(_skill("fast_winner_forecast_ridge_smase_e1v2_outlier_mad_repair_level_shift", stored),)
    )
    # different operator structure
    assert e1._existing_local_skill(
        snapshot, [("repair_level_shift", {
            "estimated_offset": 55.5, "region_end_fraction": 0.99,
            "region_start_fraction": 0.001})]
    ) is None
    # different order is a different structure
    assert e1._existing_local_skill(
        snapshot, [
            ("repair_level_shift", {
                "estimated_offset": 55.5, "region_end_fraction": 0.99,
                "region_start_fraction": 0.001}),
            ("outlier_mad", {}),
        ]
    ) is None
    # same structure but a different NON-bound constant is still distinct
    assert e1._existing_local_skill(
        snapshot, [
            ("outlier_mad", {"k": 4.0}),
            ("repair_level_shift", {
                "estimated_offset": 55.5, "region_end_fraction": 0.99,
                "region_start_fraction": 0.001}),
        ]
    ) is None


def test_next_task_reuses_instead_of_colliding(monkeypatch, tmp_path):
    """W1 end to end: the second Task deploys the existing Skill, no ADD collision.

    Before the repair this second Task produced
    method_event.stage == 'apply_failed' with AddTargetExistsError, the delayed
    window never opened, and the winner stayed LOCAL_DRAFT.
    """
    calls = {"n": 0}

    def _slow(messages):
        if messages[0]["content"] == e1._DECISION_SYSTEM:
            return {"decision": "TRUST_DRAFT", "reason": "stub"}
        calls["n"] += 1
        return {
            "decision": "PROPOSE", "reason": "stub",
            "proposals": [{"steps": [{"op": "repair_level_shift", "params": {},
                                      "bindings": {}}],
                           "requested_observations": [], "fallback": "IDENTITY",
                           "experience_use": []}],
        }

    def _compile_with_context(proposal, inventory, public_context, *, generation):
        # Mimic the real compiler: bound parameters come from the current
        # public Context, so they differ per Task.
        features = public_context["representative_features"]
        params = {
            "region_start_fraction": features["estimated_region_start_fraction"],
            "region_end_fraction": features["estimated_region_end_fraction"],
            "estimated_offset": features["estimated_level_offset"],
        }
        program = Program.from_steps(
            [("repair_level_shift", params)], source="w1-test"
        )
        candidate = Candidate.program_candidate("w1-test", program, source="w1-test")
        return CompiledWorkflow(candidate, (), ()), {"decision": "PROPOSE"}

    monkeypatch.setattr(e1, "_e1_slow_call", _slow)
    monkeypatch.setattr(e1, "_probe_compiled", _probe_metrics)
    monkeypatch.setattr(e1, "_compile_proposal", _compile_with_context)

    state = _arm_state(tmp_path, "A3", _h0())
    ctx1 = _public_context(100, sufficient=False)
    row1 = e1._run_arm(
        repo_root=PROJECT_ROOT, arm_state=state,
        task_spec=_task_spec("w1_task_01", 100), public_context=ctx1,
        source_prior=None, inventory=[], values={}, mapped_roster=[],
        config={}, eval_uids=[], llm_counter=[0],
    )
    assert row1["winner"]["local_status"] == "LOCAL_ACTIVE"
    assert row1["lifecycle"]["method_event"].get("stage") != "apply_failed"

    # Second Task: same structure, different Context -> different bound values.
    ctx2 = _public_context(700, sufficient=False)
    ctx2["representative_features"]["estimated_region_end_fraction"] = 0.9881
    ctx2["task_fast_features"]["estimated_region_end_fraction"] = 0.9881
    row2 = e1._run_arm(
        repo_root=PROJECT_ROOT, arm_state=state,
        task_spec=_task_spec("w1_task_02", 700), public_context=ctx2,
        source_prior=None, inventory=[], values={}, mapped_roster=[],
        config={}, eval_uids=[], llm_counter=[0],
    )
    event = row2["lifecycle"]["method_event"]
    assert event.get("stage") == "deployed_existing_skill", event
    assert row2["lifecycle"]["reused_existing_skill"] is True
    assert row2["winner"]["local_status"] == "LOCAL_ACTIVE"
    assert row2["winner"]["delayed_gain"] is not None
    # no duplicate Skill entry was created
    local = [s.skill_id for s in state.active_snapshot.skills
             if e1._is_local_skill_id(s.skill_id)]
    assert local == ["fast_winner_forecast_ridge_smase_e1v2_repair_level_shift"], local


# ------------------------------- Runtime-grounded clause view (autonomy test)


def test_clause_view_carries_evidence_but_never_a_verdict():
    """The property the autonomy experiment rests on.

    If the Runtime clause view ever gained a keep / revoke / downgrade field,
    or if the Slow system prompt named the target repair, the experiment would
    stop testing autonomy and start testing compliance.  This guards both.
    """
    view = g1.build_clause_evidence_view(
        {"e1_v2": {"rows": []}},
        "When post_shift_support_sufficient is false, do not make "
        "repair_level_shift the default.",
    )
    clause = view["clauses"][0]
    # the view describes the clause and its bar, and nothing about its fate
    assert clause["action_type"] == "DEPRIORITIZATION"
    assert clause["target_operators"] == ["repair_level_shift"]
    assert clause["observable_condition"] == {
        "feature": "post_shift_support_sufficient", "value": False
    }
    forbidden = {"verdict", "decision", "keep", "revoke", "downgrade",
                 "recommendation", "should_keep", "action"}
    assert not (set(clause) & forbidden), set(clause) & forbidden
    serialized = json.dumps(view).lower()
    for word in ("should be kept", "should be revoked", "should be downgraded",
                 "not sufficient on its own", "target support"):
        assert word not in serialized, word

    # an authorizing clause carries a strictly higher bar than a weakening one
    bars = g1._ACTION_EVIDENCE_BAR
    assert bars["ACTIVE_RECOMMENDATION"]["minimum"] >= bars["RESERVATION"]["minimum"]
    assert bars["ACTIVE_RECOMMENDATION"]["provenance_that_may_authorize"] == [
        g1.PROVENANCE_UNGUIDED
    ]


def test_autonomy_prompts_contain_no_target_repair():
    """Neither autonomy prompt may hint at which clause to change or how."""
    directive = (
        "not sufficient", "target support", "downgrade", "revoke", "keep the",
        "prohibition", "prioritize", "one cohort only",
        "consistent across every cohort", "repair_level_shift",
        "post_shift_support_sufficient",
    )
    for prompt in (g1._AUTONOMY_SLOW_SYSTEM, g1._CLAUSE_SLOW_SYSTEM):
        lowered = prompt.lower()
        leaked = [word for word in directive if word in lowered]
        assert not leaked, leaked
        # both must still offer ABSTAIN as a legitimate answer
        assert "abstain" in lowered
    # the Planner-specified arm is the control and IS allowed to be directive
    assert "not sufficient on its own" in g1._V2_SLOW_SYSTEM.lower()
