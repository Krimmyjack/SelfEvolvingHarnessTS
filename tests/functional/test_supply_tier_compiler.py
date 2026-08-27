"""P0: the supply tier exists on the *producing* side too.

W-1 wired the reader: ``fast_agent._supplies_candidates`` makes an already
granted ``authority.supplies_candidates`` effective, and the ps2p runs showed
a supplied candidate walking Support -> delayed -> deploy.  Nothing produced
such a card.  The live Source compilation path had one positive exit, the TRY
tier, whose leave-one-out floor means three unguided positive Tasks before a
card may name an operator -- so a course with two sources compiles either
nothing the reader can consume, or an old-style Active card with more
authority than the evidence bought.

This is the missing exit.  Two independent unguided positives of one Program
family, five-axis Scope non-empty, no opposing reading in the family, compiled
by mechanical template into ``supplies_candidates=true`` /
``grants_execution=false`` / ``requires_target_support=true``.

The two tiers share their clause vocabulary and differ in one parameter, the
count.  ``authorization_audit`` is not touched; test (d) is the proof.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import dataclasses  # noqa: E402

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    SkillKind,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    _supplies_candidates,
    _supply_rung_candidates,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    _is_inert_experience_card,
    evaluate_applicability,
    resolve_harness_view,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    source_skill as ss,
)

H0 = PROJECT_ROOT / "methods" / "ttha" / "harness" / "h0"
SKILL_ID = "p0_supply_tier_v1"
PROGRAM = "hampel_filter"

# A deployment-visible Pattern the two sources share, written in the binned
# vocabulary the contract compares against.
PATTERN_A = {
    "local_robust_z_peak": "high",
    "missing_fraction": "zero",
    "longest_missing_run_fraction": "zero",
    "period_reliability": "high",
    "estimated_level_offset": "low",
}
PATTERN_B = dict(PATTERN_A)


def _row(task_id, unit, *, relation="POSITIVE", conditioned=False,
         program=PROGRAM, pattern=None, support=0.40, delayed=0.40,
         consumer="ridge-raw-plus-difference-v1"):
    return {
        "task_episode_id": task_id,
        "unit_id": unit,
        "run_id": "run_%s" % task_id,
        "program": program,
        "relation": relation,
        "conditioned_snapshot": bool(conditioned),
        "task_kind": "classification",
        "consumer_id": consumer,
        "metric": "accuracy",
        "pattern": dict(pattern if pattern is not None else PATTERN_A),
        "support_gain": support,
        "delayed_gain": delayed,
    }


TWO_GOOD = [
    _row("GunPointAgeSpan__impulse_v2", "GunPointAgeSpan__impulse_v2"),
    _row("PowerCons__impulse_v2", "PowerCons__impulse_v2",
         support=0.0714, delayed=0.50, pattern=PATTERN_B),
]


def _compile(rows, **kwargs):
    return ss.compile_supply_tier(rows, skill_id=SKILL_ID, **kwargs)


# ============================================== (a) two qualifying Episodes
def test_a_two_independent_unguided_positives_compile_a_supply_card():
    out = _compile(TWO_GOOD)
    assert out["withheld_because"] is None
    card = out["card"]
    assert card is not None

    authority = card["risk_guards"]["authority"]
    assert authority == {
        "reorders_supplied_candidates": False,
        "supplies_candidates": True,
        "suppresses_operators": False,
        "grants_execution": False,
    }
    assert card["risk_guards"]["requires_target_support"] is True
    assert card["risk_guards"]["execution_right"] == (
        "withheld_supplies_candidate_only")
    # The frozen program is the shared one, with the evidence-side defaults.
    assert 'Frozen program steps: [{"op":"hampel_filter","params":{}}]' in \
        card["body"]
    assert card["allowed_tools"] == []
    # Scope is the five axes, and the Pattern axis is the intersection.
    scope = card["risk_guards"]["scope_v1"]
    assert scope["task_kind"] == "classification"
    assert scope["consumer_id"] == "ridge-raw-plus-difference-v1"
    assert scope["metric"] == "accuracy"
    assert scope["program_geometry"] == ["hampel_filter"]
    assert scope["pattern_intersection"] == PATTERN_A
    # Provenance is stated, and so is what n=2 is worth.
    evidence = card["risk_guards"]["evidence"]
    assert evidence["tier"] == "supply"
    assert evidence["source_count"] == 2
    assert "not a fact" in evidence["uncertainty"]


def test_a_the_template_is_deterministic_and_llm_free():
    first = _compile(TWO_GOOD)["card"]
    second = _compile(list(reversed(TWO_GOOD)))["card"]
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)


# ================================================== (b) one is not two
def test_b_a_single_positive_task_compiles_nothing():
    out = _compile(TWO_GOOD[:1])
    assert out["card"] is None
    assert out["withheld_because"] == (
        "fewer_than_2_distinct_unguided_positive_tasks")


def test_b_the_same_task_twice_is_still_one_task():
    twice = [TWO_GOOD[0], _row("GunPointAgeSpan__impulse_v2",
                               "GunPointAgeSpan__impulse_v2", support=0.31)]
    out = _compile(twice)
    assert out["card"] is None
    assert out["audit"][0]["unguided_positive"] == 1


# ============================================ (c) guided positives count zero
def test_c_a_conditioned_positive_does_not_count():
    rows = [TWO_GOOD[0],
            _row("PowerCons__impulse_v2", "PowerCons__impulse_v2",
                 conditioned=True)]
    out = _compile(rows)
    assert out["card"] is None
    assert out["audit"][0]["unguided_positive"] == 1
    assert out["audit"][0]["conditioned_positive"] == 1
    assert out["withheld_because"] == (
        "fewer_than_2_distinct_unguided_positive_tasks")


# ==================================== (d) the TRY tier is untouched by this
def test_d_the_try_tier_still_needs_three_and_leave_one_out():
    """Two tiers, one shared vocabulary, one differing parameter."""
    def probe(task_id, relation="POSITIVE"):
        return {"task_episode_id": task_id, "program": PROGRAM,
                "context_condition": False, "relation": relation,
                "conditioned_snapshot": False}

    two = [probe("t1"), probe("t2")]
    three = two + [probe("t3")]

    audit_two = ss.authorization_audit(two, min_distinct_tasks=2)
    audit_three = ss.authorization_audit(three, min_distinct_tasks=2)
    # Two unguided positives: leave one out and only one remains, so the TRY
    # tier withholds -- unchanged by anything in this book.
    assert audit_two[0]["active_try_authorized"] is False
    assert audit_two[0]["withheld_because"] == "does_not_survive_leave_one_out"
    assert audit_three[0]["active_try_authorized"] is True
    assert ss.authorized_try_operators(audit_two) == set()

    # The supply tier speaks at exactly the count the TRY tier does not.
    assert _compile(TWO_GOOD)["card"] is not None
    assert ss.SUPPLY_TIER_MIN_DISTINCT_TASKS == 2


def test_d_the_supply_exit_does_not_touch_the_try_payload():
    sections = {name: "%s text" % name for name in ss.SECTIONS}
    sections["TRY"] = ss.TRY_ABSTAIN
    payload = ss.build_skill_payload(sections)
    assert "authority" not in payload["risk_guards"]
    assert payload["risk_guards"]["carrier"] == "source_derived_general_skill"


# ==================== (e) the card is machine-judgeable and W-1 can read it
def _snapshot_with(card):
    h0 = compile_snapshot(H0, verify_lock=False)
    return dataclasses.replace(h0, skills=(*h0.skills, load_skill_entry(card)))


def _matching_features():
    return {"task_kind": "classification", **PATTERN_A}


def test_e_the_card_loads_is_scoped_and_the_w1_reader_consumes_it():
    card = _compile(TWO_GOOD, legal_features=None)["card"]
    entry = load_skill_entry(card)
    assert entry.skill_kind is SkillKind.CAPABILITY

    # The Scope AST is machine-judgeable in both directions.
    ast = card["observable_applicability"]
    assert evaluate_applicability(ast, _matching_features())[0] is True
    off_scope = {**_matching_features(), "local_robust_z_peak": "zero"}
    assert evaluate_applicability(ast, off_scope)[0] is False

    snapshot = _snapshot_with(card)
    view = resolve_harness_view(snapshot, _matching_features(), role="fast")
    assert SKILL_ID in view.skill_ids
    # Not an experience card, so the T1 inert predicate is a no-op on it.
    served = next(s for s in view.skills if s.skill_id == SKILL_ID)
    assert _is_inert_experience_card(served) is False
    # W-1's reader: the flag is seen and the frozen program materialises.
    assert _supplies_candidates(served) is True
    supplied = _supply_rung_candidates(view, _matching_features())
    assert [c.candidate_id for c in supplied] == ["cand_skill_%s" % SKILL_ID]

    # Out of Scope the card is not retrieved at all, so nothing is supplied.
    off_view = resolve_harness_view(snapshot, off_scope, role="fast")
    assert SKILL_ID not in off_view.skill_ids
    assert _supply_rung_candidates(off_view, off_scope) == ()


def test_e_uncontracted_pattern_leaves_are_dropped_and_reported():
    """PS-1 hit this: a leaf legal in the Python table but absent from
    observable_feature_v1.json fails edit-manifest shape validation."""
    rows = [dict(row) for row in TWO_GOOD]
    for row in rows:
        row["pattern"] = {**PATTERN_A, "level_region_fraction": "very_low"}
    legal = ss._edit_schema_features(PROJECT_ROOT)
    card = _compile(rows, legal_features=legal)["card"]
    dropped = card["risk_guards"][
        "pattern_leaves_dropped_as_uncontracted_for_edit_schema"]
    assert "level_region_fraction" in dropped
    leaves = {leaf["feature"]
              for leaf in card["observable_applicability"]["all"]}
    assert "level_region_fraction" not in leaves
    # Dropped from the machine AST only; the Scope record still carries it.
    assert "level_region_fraction" in (
        card["risk_guards"]["scope_v1"]["pattern_intersection"])


# ==================================== (f) an opposing reading blocks the tier
def test_f_an_unresolved_negative_in_the_same_family_blocks():
    rows = TWO_GOOD + [
        _row("Wine__impulse_v2", "Wine__impulse_v2", relation="NEGATIVE",
             support=-0.20, delayed=-0.20)]
    out = _compile(rows)
    assert out["card"] is None
    assert out["withheld_because"] == "opposing_evidence_in_the_same_family"
    assert out["audit"][0]["opposing_negative"] == 1


def test_f_a_negative_in_a_different_family_does_not_block():
    rows = TWO_GOOD + [
        _row("Wine__impulse_v2", "Wine__impulse_v2", program="outlier_mad",
             relation="NEGATIVE", support=-0.20, delayed=-0.20)]
    out = _compile(rows)
    assert out["card"] is not None
    assert out["card"]["risk_guards"]["scope_v1"]["program_geometry"] == [
        PROGRAM]


def test_f_disagreeing_identity_axes_produce_no_card():
    rows = [TWO_GOOD[0],
            _row("PowerCons__impulse_v2", "PowerCons__impulse_v2",
                 consumer="some-other-consumer-v1")]
    out = _compile(rows)
    assert out["card"] is None
    assert out["withheld_because"] == "identity_axes_disagree_across_sources"


def test_f_an_empty_pattern_intersection_produces_no_card():
    rows = [TWO_GOOD[0],
            _row("PowerCons__impulse_v2", "PowerCons__impulse_v2",
                 pattern={"local_robust_z_peak": "zero",
                          "missing_fraction": "high"})]
    out = _compile(rows)
    assert out["card"] is None
    assert out["withheld_because"] == "pattern_intersection_empty"
