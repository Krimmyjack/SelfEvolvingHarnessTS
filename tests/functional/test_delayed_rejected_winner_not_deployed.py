"""A Support winner the delayed gate refused must not become the deployment.

Three-tier feedback doctrine: Support drafts, delayed approves, only then is a
Workflow Active.  The classification round body used to write
``state['incumbent']`` from the Support winner and never clear it, so
``_frozen_recall`` froze and deployed a program whose Draft had just been
rejected -- ``approved_skill_id`` was ``None`` and the arm deployed it anyway.
S1b's unit-1 smoke caught it live (winsorize, delayed relation NEGATIVE,
deploy source FROZEN_LEDGER_INCUMBENT).

Two seams are covered here, at zero LLM and zero Consumer fits:

* ``_incumbent_after_delayed`` -- what the ledger carries after the gate;
* ``_frozen_recall`` -- what a real h0-backed arm state then deploys.

Nothing about how Support or delayed are *judged* is exercised or changed:
these tests drive the decision with the two fields the lifecycle already
publishes on its RoundResult.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402

WINSORIZE = [{"op": "winsorize", "params": {}}]
HAMPEL = [{"op": "hampel_filter", "params": {}}]


def _result(*, winner, approved):
    """The two RoundResult fields the ledger decision reads."""
    return SimpleNamespace(winner_program=winner, approved_skill_id=approved)


# --------------------------------------------------------------------------- #
# seam 1: what the ledger carries
# --------------------------------------------------------------------------- #
def test_support_positive_delayed_rejected_does_not_adopt_the_winner():
    carried = cls._incumbent_after_delayed(
        _result(winner=WINSORIZE, approved=None), None)
    assert carried is None


def test_support_positive_delayed_approved_adopts_the_winner():
    carried = cls._incumbent_after_delayed(
        _result(winner=HAMPEL, approved="fast_winner_hampel_filter"), None)
    assert carried == HAMPEL


def test_a_round_without_a_winner_leaves_the_ledger_alone():
    carried = cls._incumbent_after_delayed(
        _result(winner=None, approved=None), HAMPEL)
    assert carried == HAMPEL


def test_a_refusal_does_not_discard_a_previously_approved_incumbent():
    """r1 approved hampel; r2 drafts winsorize and is refused.

    The refusal is about the new candidate.  The arm keeps standing on the
    Workflow that did pass both gates.
    """
    after_r1 = cls._incumbent_after_delayed(
        _result(winner=HAMPEL, approved="fast_winner_hampel_filter"), None)
    after_r2 = cls._incumbent_after_delayed(
        _result(winner=WINSORIZE, approved=None), after_r1)
    assert after_r1 == HAMPEL
    assert after_r2 == HAMPEL


def test_the_round_body_routes_the_ledger_through_the_rule():
    """Guard against the seam being re-inlined by a later edit."""
    import inspect

    for module, name in ((cls, "shared runner"),):
        source = inspect.getsource(module._run_round)
        assert "_incumbent_after_delayed(result" in source, name
        assert "state[\"incumbent\"] = _plain(result.winner_program)" \
            not in source, name

    import run_e2_s1_curriculum_four_arms as s1

    source = inspect.getsource(s1._run_round)
    assert "_incumbent_after_delayed(result" in source


# --------------------------------------------------------------------------- #
# seam 2: what the arm then deploys
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def h0():
    from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
        compile_snapshot,
    )
    return compile_snapshot(
        PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0",
        verify_lock=False)


def _arm_state(h0, tmp_path, *, tag):
    return cls._new_arm_state(snapshot=h0, agent=object(),
                              store_root=tmp_path, tag=tag)


FEATURES = {"task_kind": "classification"}


def test_a_cleared_ledger_deploys_identity(h0, tmp_path):
    state = _arm_state(h0, tmp_path, tag="cleared")
    state["incumbent"] = cls._incumbent_after_delayed(
        _result(winner=WINSORIZE, approved=None), None)

    decision = cls._frozen_recall(state, FEATURES)

    assert decision["applied_steps"] == []
    assert decision["source"] == cls.DEPLOY_SOURCE_IDENTITY
    assert decision["recall_hit"] is False


def test_an_approved_ledger_still_deploys_its_program(h0, tmp_path):
    state = _arm_state(h0, tmp_path, tag="approved")
    state["incumbent"] = cls._incumbent_after_delayed(
        _result(winner=HAMPEL, approved="fast_winner_hampel_filter"), None)

    decision = cls._frozen_recall(state, FEATURES)

    assert decision["applied_steps"] == HAMPEL
    assert decision["source"] == cls.DEPLOY_SOURCE_INCUMBENT


def test_the_two_round_sequence_deploys_the_approved_one(h0, tmp_path):
    state = _arm_state(h0, tmp_path, tag="sequence")
    state["incumbent"] = cls._incumbent_after_delayed(
        _result(winner=HAMPEL, approved="fast_winner_hampel_filter"),
        state["incumbent"])
    state["incumbent"] = cls._incumbent_after_delayed(
        _result(winner=WINSORIZE, approved=None), state["incumbent"])

    decision = cls._frozen_recall(state, FEATURES)

    assert decision["applied_steps"] == HAMPEL
    assert decision["source"] == cls.DEPLOY_SOURCE_INCUMBENT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
