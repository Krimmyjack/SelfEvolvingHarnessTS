"""Adding ``capability`` to RISK_GAP must open one door and no others.

The route table is the authorization boundary for every M0 edit, so widening a
cause's target classes is the kind of change that is easy to under-test: the
obvious assertion ("the new combination is now allowed") says nothing about the
hundreds of combinations that must still be refused.

So this enumerates the whole space against the pre-change table and asserts the
difference is exactly the intended family.  It also pins the fact that makes
``scope_narrowing_preflight`` necessary: RISK_GAP is not in the router's
narrow-direction map, and even the entry that is there is a target-class gate
that never inspects the proposed predicate.  If either changes, the preflight's
justification changed with it and this test says so.

0 LLM, no evaluation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SelfEvolvingHarnessTS.evaluation.minipipe.feedback import router as rt  # noqa: E402

LIVE = rt._ROUTES_PATH
ADDED_CLASS = "capability"

#: The family the ruling authorized, and only it: RISK_GAP may ADD a whole
#: capability Skill.  ``operations`` is declared per cause rather than per
#: target class, so the table alone would also have handed RISK_GAP PATCH over
#: every capability surface -- the Skill body included, which is the Program.
#: ``authorize`` pins it to ADD; that pin is what keeps this set a singleton.
INTENDED_NEW_AUTHORIZATIONS = {
    ("RISK_GAP", "capability", "ADD", "capability"),
}

#: Causes that legitimately PATCH a capability surface and must stay able to.
#: They are why the pin is cause-scoped instead of a blanket class rule.
OTHER_CAPABILITY_PATCHERS = (
    "SKILL_CONTENT_GAP", "MECHANISM_AMBIGUITY", "SCOPED_SELECTION_GAP",
    "OUTCOME_GAP",
)


def _table() -> dict:
    return json.loads(LIVE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def before_router(tmp_path_factory) -> rt.FaultRouter:
    """The table as it stood before the ruling."""
    table = _table()
    classes = table["routes"]["RISK_GAP"]["target_classes"]
    assert ADDED_CLASS in classes, "the live table no longer carries the change"
    table["routes"]["RISK_GAP"]["target_classes"] = [
        value for value in classes if value != ADDED_CLASS]
    path = tmp_path_factory.mktemp("routes") / "fault_routes.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return rt.FaultRouter(path)


def _outcome(router: rt.FaultRouter, cause, target_class, operation, skill_kind):
    try:
        router.authorize(cause, target_class=target_class,
                         operation=operation, skill_kind=skill_kind)
    except (ValueError, KeyError) as exc:
        return "refused: %s" % exc
    return "authorized"


def _space():
    table = _table()
    causes = sorted(table["routes"])
    classes = sorted({value for route in table["routes"].values()
                      for value in route.get("target_classes", ())})
    operations = sorted({value for route in table["routes"].values()
                         for value in route.get("operations", ())})
    kinds = sorted({value for route in table["routes"].values()
                    for value in route.get("skill_kinds", ())}) + [None]
    for cause in causes:
        for target_class in classes:
            for operation in operations:
                for kind in kinds:
                    yield cause, target_class, operation, kind


def _diff(before_router):
    """Split the change three ways, as the safety gate does.

    A refusal that starts citing a different rule is not a widening: the door
    is still shut.  Folding it in with the newly-authorized set would make the
    security assertion impossible to state, so the three are counted apart.
    """
    after = rt.FaultRouter()
    newly_authorized, newly_refused, reason_changed = set(), set(), set()
    total = 0
    for combination in _space():
        total += 1
        was = _outcome(before_router, *combination)
        now = _outcome(after, *combination)
        if was == now:
            continue
        if now == "authorized":
            newly_authorized.add(combination)
        elif was == "authorized":
            newly_refused.add(combination)
        else:
            reason_changed.add(combination)
    return total, newly_authorized, newly_refused, reason_changed


def test_the_extension_authorizes_exactly_the_intended_family(before_router):
    total, authorized, refused, _reasons = _diff(before_router)
    assert total > 500, "the enumerated space collapsed; the diff would be vacuous"
    assert authorized == INTENDED_NEW_AUTHORIZATIONS
    assert refused == set(), "an edit that used to be authorized is now refused"


def test_no_other_cause_or_class_is_disturbed_even_in_its_refusal(before_router):
    """Refusals may cite a new rule, but only inside the family that changed."""
    _total, _authorized, _refused, reason_changed = _diff(before_router)
    stray = {combination for combination in reason_changed
             if combination[0] != "RISK_GAP" or combination[1] != ADDED_CLASS}
    assert stray == set()
    # And every one of them is still a refusal, for the pair rule rather than
    # the class rule -- the skill kind now decides where the class used to.
    after = rt.FaultRouter()
    for combination in reason_changed:
        assert _outcome(after, *combination).startswith("refused")


def test_everything_newly_allowed_is_allowed_and_was_not_before(before_router):
    after = rt.FaultRouter()
    for combination in INTENDED_NEW_AUTHORIZATIONS:
        assert _outcome(after, *combination) == "authorized"
        assert _outcome(before_router, *combination).startswith("refused")


def test_risk_gap_may_not_patch_a_capability_surface():
    """A Scope fault must not become licence to rewrite the Program.

    The Skill body is a capability surface, so PATCH here would let a fault
    raised about *where* a program is applied edit *what* the program does.
    """
    with pytest.raises(ValueError, match="never edit one in place"):
        rt.FaultRouter().authorize(
            "RISK_GAP", target_class="capability",
            operation="PATCH", skill_kind="capability")


@pytest.mark.parametrize("cause", OTHER_CAPABILITY_PATCHERS)
def test_the_pin_is_cause_scoped_and_spares_the_legitimate_patchers(cause):
    """A blanket class rule would have broken these four."""
    assert rt.FaultRouter().authorize(
        cause, target_class="capability", operation="PATCH",
        skill_kind="capability")


def test_risk_gap_still_may_not_add_a_risk_guard():
    """The pre-existing guard on its original class is untouched."""
    with pytest.raises(ValueError, match="may only patch an existing"):
        rt.FaultRouter().authorize(
            "RISK_GAP", target_class="capability_risk_guard",
            operation="ADD", skill_kind="capability")


def test_risk_gap_still_requires_a_matching_skill_kind():
    with pytest.raises(ValueError, match="do not form an authorized pair"):
        rt.FaultRouter().authorize(
            "RISK_GAP", target_class="capability",
            operation="ADD", skill_kind="safety")


def test_the_router_still_cannot_tell_a_narrowing_from_a_widening():
    """Why scope_narrowing_preflight exists; if this changes, revisit it.

    RISK_GAP is absent from the narrow-direction map, so no direction rule
    applies to it at all -- and the one rule that does exist only compares the
    target class, never the predicate.
    """
    assert "RISK_GAP" not in rt._APPLICABILITY_DIRECTION
    assert rt._APPLICABILITY_DIRECTION == {
        "RETRIEVAL_MISS": "widen", "SCOPE_OVERREACH": "narrow"}
    # A SCOPE_OVERREACH edit on the applicability surface is authorized without
    # the router ever being shown what the new applicability says.
    assert rt.FaultRouter().authorize(
        "SCOPE_OVERREACH", target_class="applicability",
        operation="PATCH", skill_kind="capability")
