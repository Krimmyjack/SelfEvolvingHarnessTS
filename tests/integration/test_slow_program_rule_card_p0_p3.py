"""P0 + P3 controlled mechanical checks.

P0: a capability Skill ``.body`` PATCH is Runtime-owned: the Slow-authored
``minimal_patch.value`` is ignored, the candidate snapshot body is read back
after apply, and its Frozen Program must equal the replay steps element-wise.

P3: two controlled cases only (one ADD, one PATCH).  These are mechanism
checks; they do **not** establish natural Slow Evolution.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from SelfEvolvingHarnessTS.contracts.harness import EditManifest, EditOperation
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import (
    route_program_supply_fault,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
    _parse_frozen_steps,
    _skill_frozen_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    ProgramSupplyDecision,
    build_single_surface_catalog,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (
    FrozenProgramBindingError,
    _resolve_apply_manifest,
    verify_frozen_patch_program,
)

ROOT = Path(__file__).resolve().parents[2]
H0_ROOT = ROOT / "methods/ttha/harness/h0"


@pytest.fixture
def store(tmp_path):
    return SnapshotStore(tmp_path / "store")


@pytest.fixture
def h0_materialized(store):
    return store.materialize(compile_snapshot(H0_ROOT, verify_lock=False))


@pytest.fixture
def controller(store):
    return EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )


def _program_body(steps):
    return "Frozen program steps: " + json.dumps(
        [{"op": op, "params": dict(params)} for op, params in steps]
    )


class _Evaluator:
    def __init__(self, gain, *, passed=True):
        self.gain = gain
        self.verification = SimpleNamespace(passed=passed)

    def __call__(self, steps, mode):
        return self


class _SlowAgent:
    last_no_proposal_reason = None

    def __init__(self, manifest):
        self.manifest = manifest

    def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
        return self.manifest


def _episode(episode_id):
    return SimpleNamespace(
        episode_id=episode_id,
        relation="NEGATIVE",
        support_response={"gain": -0.02},
    )


def _method(parent):
    return TTHAMethod(object(), parent.snapshot)


def _card(steps, patch_id="p1"):
    return {
        "pattern_id": "pattern-a1b2c3d4e5f6",
        "observable_signature": {"task_kind": "forecast"},
        "typed_patch_options": [
            {
                "patch_id": patch_id,
                "program_steps": [
                    {"op": op, "params": dict(params)} for op, params in steps
                ],
            }
        ],
    }


def _add_manifest(parent, *, skill_id, body, allowed_tool):
    return EditManifest(
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
            "body": body,
            "observable_applicability": {"const": True},
            "allowed_tools": [allowed_tool],
            "risk_guards": {
                "explicit_choice_required": True,
                "frozen_plan": {
                    "program": allowed_tool,
                    "excluded_series": [],
                },
            },
        },
        observable_applicability={"const": True},
        predicted_agent_behavior_change=(f"retrieve_skill:{skill_id}",),
        predicted_data_effect=("local_improvement",),
        falsification_condition=("no_improvement",),
    )


def _add_capability(parent, controller, *, skill_id, body, allowed_tool="winsorize"):
    manifest = _add_manifest(
        parent, skill_id=skill_id, body=body, allowed_tool=allowed_tool
    )
    applied = _resolve_apply_manifest(manifest, parent.snapshot)
    receipt = controller.apply_to_fork(
        parent, applied, confirmed_cause="SKILL_LIBRARY_GAP"
    )
    return receipt.candidate_snapshot


def _catalog(controller, parent, *, capability_skill_exists, skill_retrieved,
             constrained_proposal_succeeds, retrieved_ids=()):
    cause, actionability, templates = route_program_supply_fault(
        expressibility_status="PROVEN_EXPRESSIBLE",
        expressibility_cause=None,
        capability_skill_exists=capability_skill_exists,
        skill_retrieved=skill_retrieved,
        constrained_proposal_succeeds=constrained_proposal_succeeds,
    )
    decision = ProgramSupplyDecision("p3-controlled-case", cause,
                                     actionability, templates)
    return cause, build_single_surface_catalog(
        decision=decision,
        parent=parent,
        controller=controller,
        retrieved_capability_skill_ids=retrieved_ids,
    )


def test_p3_add_case_lands_replays_pends_and_delayed_removal_restores(
    h0_materialized, controller, store
):
    steps = (("impute_linear", {}),)
    cause, catalog = _catalog(
        controller,
        h0_materialized,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    )
    assert cause == "SKILL_LIBRARY_GAP"
    assert len(catalog) == 1 and catalog[0]["operation"] == "ADD"

    method = _method(h0_materialized)
    before_sha = method._active_snapshot().harness_content_sha
    manifest = _add_manifest(
        h0_materialized, skill_id="p3_added_skill",
        body="SLOW TEXT IGNORED", allowed_tool="impute_linear"
    )
    manifest = __import__("dataclasses").replace(manifest, patch_id="p1")
    ev = method.handle_feedback_support(
        _episode("ep-add"),
        slow_agent=_SlowAgent(manifest),
        controller=controller,
        store=store,
        surface_catalog=catalog,
        card_builder=lambda episode: _card(steps),
        evaluator=_Evaluator(0.02),
        fast_features={"task_kind": "forecast"},
        confirmed_cause=cause,
    )
    assert ev["stage"] == "pending", ev
    pending = method._pending_update
    assert pending is not None
    candidate = pending["receipt"].candidate_snapshot.snapshot
    skill = next(s for s in candidate.skills if s.skill_id == "p3_added_skill")
    assert _parse_frozen_steps(skill.body) == steps
    assert "SLOW TEXT IGNORED" not in skill.body
    assert method._active_snapshot().harness_content_sha == before_sha

    delayed = method.handle_feedback_delayed(
        _Evaluator(-0.05), episode_id="ep-add"
    )
    assert delayed["stage"] == "delayed_rejected"
    assert method._pending_update is None
    assert method._active_snapshot().harness_content_sha == before_sha


def test_p3_patch_case_forces_whitelist_body_and_readback_matches(
    h0_materialized, controller, store
):
    steps = (("winsorize", {"lower": 0.05}),)
    parent = _add_capability(
        h0_materialized, controller,
        skill_id="p3_patched_skill", body="OLD SLOW BODY")
    cause, catalog = _catalog(
        controller,
        parent,
        capability_skill_exists=True,
        skill_retrieved=True,
        constrained_proposal_succeeds=False,
        retrieved_ids=("p3_patched_skill",),
    )
    assert cause == "SKILL_CONTENT_GAP"
    assert len(catalog) == 1 and catalog[0]["operation"] == "PATCH"

    method = _method(parent)
    before_sha = method._active_snapshot().harness_content_sha
    revision_before = next(
        skill.revision
        for skill in parent.snapshot.skills
        if skill.skill_id == "p3_patched_skill"
    )
    patch = EditManifest(
        edit_id="patch-p3-skill",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id=catalog[0]["surface_id"],
        operation=EditOperation.PATCH,
        surface_precondition=catalog[0]["surface_precondition"],
        dependency_precondition_shas=catalog[0]["dependency_precondition_shas"],
        minimal_patch={"value": "SLOW TEXT MUST BE IGNORED"},
        predicted_agent_behavior_change=("retrieve_skill:p3_patched_skill",),
        predicted_data_effect=("local_improvement",),
        falsification_condition=("no_improvement",),
        patch_id="p1",
    )
    ev = method.handle_feedback_support(
        _episode("ep-patch"),
        slow_agent=_SlowAgent(patch),
        controller=controller,
        store=store,
        surface_catalog=catalog,
        card_builder=lambda episode: _card(steps),
        evaluator=_Evaluator(0.02),
        confirmed_cause=cause,
    )
    assert ev["stage"] == "pending", ev
    assert ev["patch_body_binding"] == "runtime_owned"
    assert ev["patch_body_readback"] == "steps_match_replay"
    pending = method._pending_update
    candidate = pending["receipt"].candidate_snapshot.snapshot
    skill = next(s for s in candidate.skills if s.skill_id == "p3_patched_skill")
    assert skill.body == _program_body(steps)
    assert skill.revision == revision_before + 1
    assert _parse_frozen_steps(skill.body) == steps
    assert skill.allowed_tools == ("winsorize",)
    assert dict(skill.risk_guards)["frozen_plan"]["program"] == "winsorize"
    assert "SLOW TEXT MUST BE IGNORED" not in skill.body
    assert method._active_snapshot().harness_content_sha == before_sha

    delayed = method.handle_feedback_delayed(
        _Evaluator(-0.05), episode_id="ep-patch"
    )
    assert delayed["stage"] == "delayed_rejected"
    assert method._pending_update is None
    assert method._active_snapshot().harness_content_sha == before_sha
    active_skill = next(
        skill
        for skill in method._active_snapshot().skills
        if skill.skill_id == "p3_patched_skill"
    )
    assert active_skill.revision == revision_before


def test_p2_agent_abstain_is_explicit_and_never_pends(
    h0_materialized, controller, store
):
    steps = (("impute_linear", {}),)
    cause, catalog = _catalog(
        controller,
        h0_materialized,
        capability_skill_exists=False,
        skill_retrieved=False,
        constrained_proposal_succeeds=None,
    )
    assert cause == "SKILL_LIBRARY_GAP"

    class _NoProposalSlow:
        last_no_proposal_reason = "evidence_insufficient"

        def propose_edit(self, card, surface_catalog, snapshot, **kwargs):
            return None

    method = _method(h0_materialized)
    before_sha = method._active_snapshot().harness_content_sha
    ev = method.handle_feedback_support(
        _episode("ep-abstain"),
        slow_agent=_NoProposalSlow(),
        controller=controller,
        store=store,
        surface_catalog=catalog,
        card_builder=lambda episode: _card(steps),
        evaluator=_Evaluator(0.02),
        fast_features={"task_kind": "forecast"},
        confirmed_cause=cause,
    )
    assert ev["stage"] == "abstained_by_agent"
    assert ev["no_proposal_reason"] == "evidence_insufficient"
    assert method._pending_update is None
    assert method._active_snapshot().harness_content_sha == before_sha


def test_p3_patch_case_delayed_approval_activates_and_next_entry_supplies_program(
    h0_materialized, controller, store
):
    steps = (("winsorize", {"lower": 0.05}),)
    parent = _add_capability(
        h0_materialized, controller,
        skill_id="p3_approved_skill", body="OLD SLOW BODY")
    cause, catalog = _catalog(
        controller,
        parent,
        capability_skill_exists=True,
        skill_retrieved=True,
        constrained_proposal_succeeds=False,
        retrieved_ids=("p3_approved_skill",),
    )
    assert cause == "SKILL_CONTENT_GAP"

    method = _method(parent)
    before_sha = method._active_snapshot().harness_content_sha
    revision_before = next(
        skill.revision
        for skill in parent.snapshot.skills
        if skill.skill_id == "p3_approved_skill"
    )
    patch = EditManifest(
        edit_id="patch-p3-approve",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id=catalog[0]["surface_id"],
        operation=EditOperation.PATCH,
        surface_precondition=catalog[0]["surface_precondition"],
        dependency_precondition_shas=catalog[0]["dependency_precondition_shas"],
        minimal_patch={"value": "SLOW TEXT MUST BE IGNORED"},
        predicted_agent_behavior_change=("retrieve_skill:p3_approved_skill",),
        predicted_data_effect=("local_improvement",),
        falsification_condition=("no_improvement",),
        patch_id="p1",
    )
    ev = method.handle_feedback_support(
        _episode("ep-approve"),
        slow_agent=_SlowAgent(patch),
        controller=controller,
        store=store,
        surface_catalog=catalog,
        card_builder=lambda episode: _card(steps),
        evaluator=_Evaluator(0.02),
        confirmed_cause=cause,
    )
    assert ev["stage"] == "pending", ev
    pending = method._pending_update
    assert pending is not None
    pending_sha = pending["receipt"].candidate_snapshot.snapshot.harness_content_sha

    delayed = method.handle_feedback_delayed(
        _Evaluator(0.02), episode_id="ep-approve"
    )
    assert delayed["stage"] == "approved"
    assert method._pending_update is None
    active = method._active_snapshot()
    assert active.harness_content_sha != before_sha
    assert active.harness_content_sha == pending_sha

    # The modified Program is actually activated in the snapshot.
    skill = next(
        skill for skill in active.skills if skill.skill_id == "p3_approved_skill"
    )
    assert skill.body == _program_body(steps)
    assert skill.revision == revision_before + 1
    assert _parse_frozen_steps(skill.body) == steps

    # The next normal entry can retrieve and supply the modified Program.
    features = {"task_kind": "forecast"}
    view = resolve_harness_view(active, features, role="fast")
    assert "p3_approved_skill" in view.skill_ids
    supplied = _skill_frozen_candidates(view, features)
    candidate = next(
        candidate
        for candidate in supplied
        if candidate.candidate_id == "cand_skill_p3_approved_skill"
    )
    assert tuple(candidate.program.execution_steps()) == steps


def test_p0_broken_consistency_must_raise_and_stop_at_apply_failed(
    h0_materialized, controller, store, monkeypatch
):
    steps = (("winsorize", {}),)
    parent = _add_capability(
        h0_materialized, controller,
        skill_id="p0_broken_skill", body="OLD BODY")
    cause, catalog = _catalog(
        controller,
        parent,
        capability_skill_exists=True,
        skill_retrieved=True,
        constrained_proposal_succeeds=False,
        retrieved_ids=("p0_broken_skill",),
    )
    method = _method(parent)
    before_sha = method._active_snapshot().harness_content_sha

    # Direct readback assertion on a real inconsistent candidate.
    with pytest.raises(FrozenProgramBindingError):
        verify_frozen_patch_program(
            parent.snapshot,
            target_surface_id="skill_library.entries/p0_broken_skill.body",
            replay_steps=steps,
        )

    # Force the post-write assertion to fail and verify the method stops at
    # apply_failed without entering pending or mutating the active snapshot.
    def _broken_verify(*args, **kwargs):
        raise FrozenProgramBindingError("deliberately broken consistency")

    import SelfEvolvingHarnessTS.methods.ttha.slow_agent as slow_agent_module

    monkeypatch.setattr(slow_agent_module, "verify_frozen_patch_program",
                        _broken_verify)
    patch = EditManifest(
        edit_id="patch-p0-broken",
        base_harness_sha=parent.harness_content_sha,
        target_pattern_id="pattern-a1b2c3d4e5f6",
        target_surface_id=catalog[0]["surface_id"],
        operation=EditOperation.PATCH,
        surface_precondition=catalog[0]["surface_precondition"],
        dependency_precondition_shas=catalog[0]["dependency_precondition_shas"],
        minimal_patch={"value": "WHATEVER SLOW WROTE"},
        predicted_agent_behavior_change=("retrieve_skill:p0_broken_skill",),
        predicted_data_effect=("local_improvement",),
        falsification_condition=("no_improvement",),
        patch_id="p1",
    )
    ev = method.handle_feedback_support(
        _episode("ep-broken"),
        slow_agent=_SlowAgent(patch),
        controller=controller,
        store=store,
        surface_catalog=catalog,
        card_builder=lambda episode: _card(steps),
        evaluator=_Evaluator(0.02),
        confirmed_cause=cause,
    )
    assert ev["stage"] == "apply_failed"
    assert "deliberately broken consistency" in ev["error"]
    assert method._pending_update is None
    assert method._active_snapshot().harness_content_sha == before_sha
