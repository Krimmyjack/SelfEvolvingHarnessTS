"""A Skill's Scope must survive being written down.

The defect this locks
---------------------
``serving_scope`` was added to ``skill_entry_v1.json`` and to ``SkillEntry`` on
2026-09-01.  The parser was updated; ``skill_entry_to_dict`` -- the only writer,
used by ``SnapshotStore.materialize`` and by ``snapshot_to_dict`` -- was not.
So a Scope survived inside the round that created it and was gone from the
snapshot that round persisted.

That is the protocol's recurring defect class in its purest form: the thing
that was gated is not the thing that was stored.  The delayed gate and the
independent re-encounter both measured *program under predicate*; what
persisted was the program alone, and the next round retrieved it and deployed
it to every served series.  It is invisible from the outside -- the Skill is
there, it has the right program, and it is simply unscoped -- which is why it
survived two live Source runs (p4w2, p4w3) without anyone noticing.

Round-tripping is the only honest check: serialize, re-parse, compare.  Reading
the writer and agreeing it "looks right" is what let this through the first
time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "evaluation" / "functional"):
    sys.path.insert(0, str(_path))

import run_v1_guidance_evolution as runner  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    SkillEntry,
    SkillKind,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
    skill_entry_to_dict,
    snapshot_to_dict,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)

H0 = ROOT / "methods" / "ttha" / "harness" / "h0"

SCOPE = {
    "scope_type": "serving_series_predicate",
    "predicate": [
        {"feature": "local_robust_z_peak", "op": ">=", "threshold": 3.0},
        {"feature": "estimated_region_start_fraction", "op": "<=", "threshold": 0.5},
    ],
}


def _skill(serving_scope=None) -> SkillEntry:
    return SkillEntry(
        schema_version="skill-entry/1",
        skill_id="scoped_probe",
        skill_kind=SkillKind.CAPABILITY,
        revision=1,
        body='Frozen program steps: [{"op": "outlier_iqr", "params": {}}]',
        observable_applicability={
            "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]},
        allowed_tools=("outlier_iqr",),
        risk_guards={"requires_target_support": True},
        serving_scope=serving_scope,
    )


def test_a_scoped_skill_serializes_with_its_scope():
    payload = skill_entry_to_dict(_skill(SCOPE))
    assert "serving_scope" in payload, (
        "a Skill gated as 'this program on these series' must not be written "
        "down as 'this program'")
    assert payload["serving_scope"]["scope_type"] == "serving_series_predicate"
    assert len(payload["serving_scope"]["predicate"]) == 2


def test_the_scope_survives_a_full_write_and_read_back(tmp_path):
    """Serialize, persist, recompile: the predicate has to still be there."""
    snapshot = runner._h0_snapshot()
    store = SnapshotStore(tmp_path / "store")
    materialized = store.materialize(snapshot)
    written = Path(materialized.root) / "skills" / "learned" / "scoped_probe.json"
    written.parent.mkdir(parents=True, exist_ok=True)
    import json
    written.write_text(json.dumps(skill_entry_to_dict(_skill(SCOPE))),
                       encoding="utf-8")
    rebuilt = compile_snapshot(Path(materialized.root), verify_lock=False)
    entry = next(s for s in rebuilt.skills if s.skill_id == "scoped_probe")
    assert entry.serving_scope is not None
    clauses = [dict(c) for c in entry.serving_scope["predicate"]]
    assert clauses == SCOPE["predicate"]

    # ... and again through the store, which is where the live loss happened.
    second = store.materialize(rebuilt)
    again = compile_snapshot(Path(second.root), verify_lock=False)
    kept = next(s for s in again.skills if s.skill_id == "scoped_probe")
    assert kept.serving_scope is not None, (
        "SnapshotStore.materialize is the step that dropped it live")
    assert [dict(c) for c in kept.serving_scope["predicate"]] == SCOPE["predicate"]


def test_two_skills_differing_only_in_scope_no_longer_hash_alike():
    """The content SHA is computed over this dict, so a dropped field hid here."""
    unscoped = snapshot_to_dict(
        runner._h0_snapshot().__class__(
            **{**runner._h0_snapshot().__dict__, "skills": (_skill(None),)}))
    scoped = snapshot_to_dict(
        runner._h0_snapshot().__class__(
            **{**runner._h0_snapshot().__dict__, "skills": (_skill(SCOPE),)}))
    assert unscoped != scoped, (
        "before the fix these were byte-identical, so a Scope revision was "
        "indistinguishable from the policy it revised")


# ------------------------------------------------------ the no-drift half ---
#
# The fix must be invisible to everything that never carried a Scope, or it
# would rewrite the identity of every existing snapshot and artifact.

def test_an_unscoped_skill_is_serialized_exactly_as_before():
    payload = skill_entry_to_dict(_skill(None))
    assert "serving_scope" not in payload
    assert sorted(payload) == [
        "allowed_tools", "body", "observable_applicability",
        "revision", "risk_guards", "schema_version", "skill_id", "skill_kind"]


def test_h0_content_sha_does_not_move_because_of_this_fix():
    """The no-drift claim that matters, stated as a measurement.

    ``harness_content_sha`` is computed over ``snapshot_to_dict``, so widening
    the skill writer *could* have re-identified every existing snapshot and
    artifact.  It does not, because the key is omitted when there is no Scope:
    measured before and after the change, H0's content SHA is the same
    ``53b1c803…`` both times.

    ``runtime_bundle_sha`` does move, because it hashes the harness runtime
    code and this fix is a change to that code.  That is the intended
    behaviour -- it is what makes a snapshot say which runtime produced it --
    and it is why the fix lands in a new store directory rather than silently
    over the old one.

    H0's checked-in lock is *not* asserted here: it already fails to verify in
    this worktree for reasons that predate this change (measured by stashing
    the fix and re-running the same compile), so asserting it would be
    inheriting someone else's red rather than protecting this one.
    """
    first = compile_snapshot(H0, verify_lock=False)
    second = compile_snapshot(H0, verify_lock=False)
    assert first.harness_content_sha == second.harness_content_sha
    assert all(skill.serving_scope is None for skill in first.skills)
    # Every H0 skill round-trips to the same eight keys it always had, which is
    # what keeps the content SHA where it was.
    for skill in first.skills:
        assert sorted(skill_entry_to_dict(skill)) == [
            "allowed_tools", "body", "observable_applicability",
            "revision", "risk_guards", "schema_version", "skill_id",
            "skill_kind"]
