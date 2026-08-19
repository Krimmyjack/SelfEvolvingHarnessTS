"""Negative Target evidence gets a carrier, and the carrier cannot propose.

The electricity development run showed the gap end to end: ``repair_level_shift``
was probed 14 times over 8 distinct Tasks, was negative every single time, and
both arms still led with it in 7 of 9 Tasks.  Inside a Task the Agent handles
the same failure correctly -- A5 on task_05 went -0.126 then pivoted to
``outlier_mad`` for +0.114 -- because a probe result is Target Support in that
trajectory.  Across Tasks nothing carried it, because only a *positive* probe
became a Target-local Skill and an Episode reaches Fast only as a Skill.

These are the zero-LLM checks for the carrier that closes it.  The two halves
that matter are the rule that decides when a claim is made, and the guarantee
that the claim can only ever lower a family's standing -- never raise one, and
never survive its own Domain contradicting it.
"""
from __future__ import annotations

import json
import shutil
import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_SUPPORT,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    STATUS_RESTRICTED,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _skill_frozen_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)
from evaluation.functional.task_episode_harness.agentic import risk_skill  # noqa: E402
from evaluation.functional.task_episode_harness.agentic.fast_path import (  # noqa: E402
    FastPathTrace,
    _deprioritized_probe_order,
)
from evaluation.functional.task_episode_harness.agentic import runner as g1r  # noqa: E402
from evaluation.functional.task_episode_harness.e1 import (  # noqa: E402
    E1_DOMAIN,
    MATERIAL_THRESHOLD,
    _ArmState,
    _update_delayed,
)
from evaluation.functional.task_episode_harness.t1 import (  # noqa: E402
    TASK_CONSUMER_KEY,
)

H0 = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"

# The Context the electricity failures actually shared.
SIGNATURE = {"task_kind": "forecast", "estimated_region_start_fraction": "zero"}
OTHER_CONTEXT = {"task_kind": "forecast", "estimated_region_start_fraction": "late"}


def _episode(task_id: str, ops: list[str], gain: float, signature=SIGNATURE):
    """Only the fields the census reads; no Support instrument is involved."""
    return types.SimpleNamespace(
        episode_id="ep_%s_%s" % (task_id, "_".join(ops)),
        context_summary={
            "task_episode_id": task_id,
            "task_signature": dict(signature),
            "program_geometry": {"program_steps": [{"op": op} for op in ops]},
        },
        support_response={"gain": gain},
    )


REPEATED_FAILURE = [
    _episode("task_01", ["repair_level_shift"], -0.0149),
    _episode("task_02", ["repair_level_shift"], -0.0548),
    _episode("task_02", ["outlier_mad"], -0.0289),
]


# --------------------------------------------------------------- the rule
def test_two_distinct_tasks_are_required_not_two_attempts():
    """One Task twice is still one Task: A3 and A5 probe the same Outcome cell."""
    one_task_twice = [
        _episode("task_01", ["repair_level_shift"], -0.015),
        _episode("task_01", ["repair_level_shift"], -0.019),
    ]
    assert risk_skill.risk_candidates(
        one_task_twice, threshold=MATERIAL_THRESHOLD) == []

    rows = risk_skill.risk_candidates(
        REPEATED_FAILURE, threshold=MATERIAL_THRESHOLD)
    assert [row["family"] for row in rows] == ["repair_level_shift"]
    assert rows[0]["distinct_negative_task_count"] == 2


def test_a_conflicted_family_produces_nothing():
    """Mixed evidence is a question for Slow, never a local deprioritization.

    This is the T233 shape: ``repair_level_shift`` there was 6 negative
    against 5 positive.  A rule that fired on it would be asserting a claim
    its own Domain half-refutes.
    """
    with_a_win = REPEATED_FAILURE + [
        _episode("task_03", ["repair_level_shift"], +0.20)]
    assert risk_skill.risk_candidates(
        with_a_win, threshold=MATERIAL_THRESHOLD) == []


def test_applicability_keeps_only_what_every_failure_shared():
    """A feature that varied across the failures is not evidence about Context."""
    mixed = [
        _episode("task_01", ["repair_level_shift"], -0.015,
                 {"task_kind": "forecast", "estimated_region_start_fraction": "zero"}),
        _episode("task_02", ["repair_level_shift"], -0.055,
                 {"task_kind": "forecast", "estimated_region_start_fraction": "late"}),
    ]
    row = risk_skill.risk_candidates(mixed, threshold=MATERIAL_THRESHOLD)[0]
    assert row["shared_task_signature"] == {"task_kind": "forecast"}
    assert row["observable_applicability"] == {
        "feature": "task_kind", "op": "==", "value": "forecast"}


# ------------------------------------------------------- the carrier itself
@pytest.fixture()
def arm(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = compile_snapshot(H0, verify_lock=False)
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    return _ArmState(
        arm="A3", memories=[], episodes=list(REPEATED_FAILURE), store=store,
        active_snapshot=snapshot, active_skill_ids=[],
    )


def test_the_skill_is_minted_into_the_learned_directory_and_compiles_back(arm):
    """SkillKind.SAFETY existed in the contract but had no working round trip.

    ``store`` used to file it under ``skills/bootstrap/``, whose ID set the
    compiler pins exactly, so a minted Skill could never be read back; the
    learned loader and the Slow-edit schema both forced ``capability``.
    """
    out = g1r.run_risk_skill_lifecycle(arm)
    assert [event["stage"] for event in out["events"]] == ["added"]
    skill_id = out["risk_skill_ids"][0]
    assert skill_id == "target_risk_repair_level_shift"

    root = arm.store.materialize(arm.active_snapshot).root
    entry = json.loads(
        (root / "skills" / "learned" / (skill_id + ".json")).read_text("utf-8"))
    assert entry["skill_kind"] == "safety"
    # Reads back through the real compiler, not just off disk.
    assert any(
        skill.skill_id == skill_id
        for skill in compile_snapshot(root, verify_lock=False).skills
    )


def test_repeating_the_same_failure_does_not_write_a_second_entry(arm):
    g1r.run_risk_skill_lifecycle(arm)
    arm.episodes.append(_episode("task_03", ["repair_level_shift"], -0.197))
    again = g1r.run_risk_skill_lifecycle(arm)
    assert [event["stage"] for event in again["events"]] == ["already_present"]
    assert len(again["risk_skill_ids"]) == 1


def test_it_reaches_fast_in_its_own_context_and_nowhere_else(arm):
    g1r.run_risk_skill_lifecycle(arm)
    skill_id = "target_risk_repair_level_shift"
    assert skill_id in resolve_harness_view(
        arm.active_snapshot, SIGNATURE, role="fast").skill_ids
    assert skill_id not in resolve_harness_view(
        arm.active_snapshot, OTHER_CONTEXT, role="fast").skill_ids


def test_it_can_never_become_an_executable_candidate(arm):
    """The point of SAFETY: it may lower a family, never put one on the table."""
    g1r.run_risk_skill_lifecycle(arm)
    view = resolve_harness_view(arm.active_snapshot, SIGNATURE, role="fast")
    skill = next(s for s in view.skills if s.skill_id.startswith("target_risk_"))
    assert skill.allowed_tools == ()
    assert "Frozen program steps:" not in skill.body
    assert tuple(_skill_frozen_candidates(view, SIGNATURE)) == ()


def test_the_body_says_deprioritize_and_refuses_to_say_forbid(arm):
    g1r.run_risk_skill_lifecycle(arm)
    view = resolve_harness_view(arm.active_snapshot, SIGNATURE, role="fast")
    body = next(s.body for s in view.skills if s.skill_id.startswith("target_risk_"))
    assert "not a prohibition" in body
    assert "overrides this note" in body
    # The evidence it rests on is stated, so the Agent can weigh it.
    assert "2 distinct Tasks" in body


# -------------------------------------------------- negative evidence retires
def test_a_material_positive_retires_the_note_its_domain_refuted(arm):
    g1r.run_risk_skill_lifecycle(arm)
    skill_id = "target_risk_repair_level_shift"
    arm.episodes.append(_episode("task_03", ["repair_level_shift"], +0.20))

    retire = g1r.run_risk_skill_lifecycle(arm)
    assert any(event["stage"] == "restricted" for event in retire["events"])
    assert skill_id not in resolve_harness_view(
        arm.active_snapshot, SIGNATURE, role="fast").skill_ids

    # Retired, not deleted: the claim and what refuted it stay auditable.
    root = arm.store.materialize(arm.active_snapshot).root
    entry = json.loads(
        (root / "skills" / "learned" / (skill_id + ".json")).read_text("utf-8"))
    assert entry["risk_guards"][risk_skill.RESTRICTED_GUARD] is True
    assert entry["risk_guards"]["restriction_reason"] == (
        "family_earned_a_material_positive_in_this_domain")


def test_a_restricted_skill_is_invisible_in_both_roles(arm):
    """Slow reads the library unfiltered by Context, but not by restriction."""
    g1r.run_risk_skill_lifecycle(arm)
    arm.episodes.append(_episode("task_03", ["repair_level_shift"], +0.20))
    g1r.run_risk_skill_lifecycle(arm)
    for role in ("fast", "slow"):
        ids = resolve_harness_view(arm.active_snapshot, SIGNATURE, role=role).skill_ids
        assert "target_risk_repair_level_shift" not in ids, role


# ------------------------------------------------- LOCAL_ACTIVE means active
def _graded(support_gain: float, delayed_gain: float) -> str:
    """Grade one real Episode through the shipped lifecycle transition."""
    episode = build_episode(
        episode_id="ep_grading",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace=E1_DOMAIN,
        context_summary={"task_episode_id": "task_01"},
        workflow_signature="outlier_mad",
        support_response={"gain": support_gain, "se_block": 0.01,
                          "gain_over_se": 1.0, "accepted": support_gain > 0,
                          "block_origins": [0]},
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation="POSITIVE" if support_gain > 0 else "NEGATIVE",
        evidence_level=EVIDENCE_SUPPORT,
        local_status=(
            STATUS_LOCAL_DRAFT if support_gain >= MATERIAL_THRESHOLD
            else "EPISODE_ONLY"),
        evidence_refs=["task_episode_harness_e1"],
    )
    updated = _update_delayed(
        episode,
        {"macro_gain": delayed_gain, "se_block": 0.01, "gain_over_se": 1.0},
        (0,),
    )
    return str(updated.local_status)


@pytest.mark.parametrize(
    "support_gain, delayed_gain, expected",
    [
        # Confirmed on both windows.
        (0.10, 0.10, STATUS_LOCAL_ACTIVE),
        # The old rule graded this ACTIVE, because it only asked that delayed
        # be no worse than -tau.  Every report then read "active" as "delayed
        # confirmed the gain", which this case never showed.
        (0.10, -0.004, STATUS_LOCAL_DRAFT),
        (0.10, 0.0, STATUS_LOCAL_DRAFT),
        # Delayed actively contradicts it.
        (0.10, -0.05, STATUS_RESTRICTED),
        # Never had a Support gain to begin with.
        (-0.01, 0.10, "EPISODE_ONLY"),
    ],
)
def test_delayed_grading_has_three_bands(support_gain, delayed_gain, expected):
    assert _graded(support_gain, delayed_gain) == expected


def test_a_skill_disconfirmed_by_its_delayed_window_stops_being_recalled(arm):
    """The reuse path used to keep an already-active Skill in the snapshot.

    ``existing_skill_revalidated`` recorded ``delayed_ok`` and moved on, so a
    Skill the Domain had just contradicted kept arriving on the next Task.
    """
    g1r.run_risk_skill_lifecycle(arm)
    skill_id = "target_risk_repair_level_shift"
    out = g1r.run_risk_skill_lifecycle(arm, disconfirmed_skill_ids=[skill_id])
    assert any(event["stage"] == "restricted" for event in out["events"])
    assert skill_id not in resolve_harness_view(
        arm.active_snapshot, SIGNATURE, role="fast").skill_ids


def test_restricting_something_absent_is_reported_not_raised(arm):
    out = g1r.run_risk_skill_lifecycle(arm, disconfirmed_skill_ids=["no_such_skill"])
    assert out["events"][0] == {"stage": "absent", "skill_id": "no_such_skill"}


def test_nothing_is_written_when_the_evidence_does_not_support_it(tmp_path):
    store = SnapshotStore(tmp_path / "snapshots")
    snapshot = compile_snapshot(H0, verify_lock=False)
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    quiet = _ArmState(
        arm="A3", memories=[],
        episodes=[_episode("task_01", ["repair_level_shift"], -0.015)],
        store=store, active_snapshot=snapshot, active_skill_ids=[],
    )
    out = g1r.run_risk_skill_lifecycle(quiet)
    assert out["events"] == []
    assert out["risk_skill_ids"] == []
    assert quiet.active_snapshot.runtime_bundle_sha == snapshot.runtime_bundle_sha

# ------------------------------------------------- the deprioritization acts
def _row(candidate_id: str, *ops: str) -> dict:
    return {"candidate_id": candidate_id,
            "steps": [(op, {}) for op in ops]}


class _View:
    def __init__(self, *skills):
        self.skills = skills


def _risk(skill_id: str):
    return types.SimpleNamespace(
        skill_id=skill_id,
        skill_kind=types.SimpleNamespace(value="safety"),
    )


def test_a_refuted_family_is_probed_last_not_dropped():
    """The first probe is spent before select runs, so order is the lever.

    The micro replay measured the alternative: the Skill reached the Fast
    prompt in both arms of Task 3, both still led with ``repair_level_shift``
    for -0.197, and both then chose identity -- the budget was gone before
    the Agent was asked anything.
    """
    trace = FastPathTrace()
    rows = [_row("c1", "repair_level_shift"), _row("c2", "hampel_filter")]
    ordered = _deprioritized_probe_order(
        rows, _View(_risk("target_risk_repair_level_shift")), trace)
    assert [row["candidate_id"] for row in ordered] == ["c2", "c1"]
    # Nothing is removed, and the move is on the record.
    assert len(ordered) == 2
    assert trace.probe_order_deprioritizations[0]["moved_behind"] == ["c1"]


def test_relative_order_inside_each_group_is_the_agents_own():
    trace = FastPathTrace()
    rows = [
        _row("c1", "repair_level_shift"),
        _row("c2", "hampel_filter"),
        _row("c3", "repair_level_shift"),
        _row("c4", "outlier_mad"),
    ]
    ordered = _deprioritized_probe_order(
        rows, _View(_risk("target_risk_repair_level_shift")), trace)
    assert [row["candidate_id"] for row in ordered] == ["c2", "c4", "c1", "c3"]


def test_nothing_moves_when_every_candidate_is_refuted():
    """Reshuffling equally-refuted candidates would invent a preference."""
    trace = FastPathTrace()
    rows = [_row("c1", "repair_level_shift"), _row("c2", "repair_level_shift")]
    ordered = _deprioritized_probe_order(
        rows, _View(_risk("target_risk_repair_level_shift")), trace)
    assert [row["candidate_id"] for row in ordered] == ["c1", "c2"]
    assert trace.probe_order_deprioritizations == []


def test_a_multi_step_program_only_matches_its_own_structure():
    trace = FastPathTrace()
    rows = [_row("c1", "outlier_mad", "repair_level_shift"),
            _row("c2", "repair_level_shift")]
    ordered = _deprioritized_probe_order(
        rows, _View(_risk("target_risk_repair_level_shift")), trace)
    assert [row["candidate_id"] for row in ordered] == ["c1", "c2"]


def test_a_view_without_a_risk_skill_changes_nothing():
    trace = FastPathTrace()
    rows = [_row("c1", "repair_level_shift"), _row("c2", "hampel_filter")]
    capability = types.SimpleNamespace(
        skill_id="fast_winner_e1v2_outlier_mad",
        skill_kind=types.SimpleNamespace(value="capability"))
    ordered = _deprioritized_probe_order(rows, _View(capability), trace)
    assert [row["candidate_id"] for row in ordered] == ["c1", "c2"]
    assert trace.probe_order_deprioritizations == []


def test_no_receipt_when_the_agent_already_ordered_it_correctly():
    """A5 on electricity Task 3 proposed the alternative first by itself.

    The reorder had nothing to do, and an earlier version still wrote a
    receipt claiming it had moved the refuted family -- which would inflate
    every later count of how often the deprioritization acted.
    """
    trace = FastPathTrace()
    rows = [_row("c1", "hampel_filter"), _row("c2", "repair_level_shift")]
    ordered = _deprioritized_probe_order(
        rows, _View(_risk("target_risk_repair_level_shift")), trace)
    assert [row["candidate_id"] for row in ordered] == ["c1", "c2"]
    assert trace.probe_order_deprioritizations == []
