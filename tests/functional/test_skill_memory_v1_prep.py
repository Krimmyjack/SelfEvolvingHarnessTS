"""Skill/Memory v1 freeze prep: one focused test per mechanism repair.

Q1 -- narrowing PATCH is authorized by SCOPE_OVERREACH, not RETRIEVAL_MISS.
Q7 -- uncontracted Pattern axes are declared on the card face.
Q11 -- n>=2 intersection cards carry one task_kind leaf.

Matching, thresholds, and the three write-back rules are untouched.
"""
from __future__ import annotations

import sys
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

from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import (  # noqa: E402
    FaultRouter,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    evaluate_applicability,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    skill_revision as rev,
    source_skill as ss,
)

SKILL_ID = "skill_memory_v1_prep"
PROGRAM = "hampel_filter"
PATTERN_A = {
    "local_robust_z_peak": "high",
    "missing_fraction": "zero",
    "longest_missing_run_fraction": "zero",
    "period_reliability": "high",
    "estimated_level_offset": "low",
}


def _row(task_id, unit, *, support=0.40, delayed=0.40, pattern=None):
    return {
        "task_episode_id": task_id,
        "unit_id": unit,
        "run_id": "run_%s" % task_id,
        "program": PROGRAM,
        "relation": "POSITIVE",
        "conditioned_snapshot": False,
        "task_kind": "classification",
        "consumer_id": "ridge-raw-plus-difference-v1",
        "metric": "accuracy",
        "pattern": dict(pattern if pattern is not None else PATTERN_A),
        "support_gain": support,
        "delayed_gain": delayed,
    }


TWO_GOOD = [
    _row("GunPointAgeSpan__impulse_v2", "GunPointAgeSpan__impulse_v2"),
    _row("PowerCons__impulse_v2", "PowerCons__impulse_v2",
         support=0.0714, delayed=0.50),
]


def _compile(rows, **kwargs):
    return ss.compile_supply_tier(rows, skill_id=SKILL_ID, **kwargs)


def _matching_features():
    return {"task_kind": "classification", **PATTERN_A}


def test_q1_narrowing_patch_uses_scope_overreach_not_retrieval_miss() -> None:
    """The SA-1 borrow point was APPLICABILITY_CAUSE = RETRIEVAL_MISS.

    That cause still means 'should have been retrieved and was not'
    (widening).  The new token authorizes applicability PATCH only.
    """
    assert rev.APPLICABILITY_CAUSE == "SCOPE_OVERREACH"
    assert rev.RISK_GUARD_CAUSE == "RISK_GAP"

    router = FaultRouter()
    narrow = router.authorize(
        "SCOPE_OVERREACH",
        target_class="applicability",
        operation="PATCH",
        skill_kind="capability",
        target_surface_id=(
            "skill_library.entries/sa1_supply_scope_v2.observable_applicability"
        ),
    )
    assert narrow.cause_code == "SCOPE_OVERREACH"
    assert narrow.target_classes == ("applicability",)
    assert "PATCH" in narrow.allowed_operations
    assert "retrieval" not in narrow.target_classes

    widen = router.allowed_targets("RETRIEVAL_MISS")
    assert "retrieval" in widen.target_classes
    assert "applicability" in widen.target_classes
    router.authorize(
        "RETRIEVAL_MISS",
        target_class="retrieval",
        operation="PATCH",
        skill_kind="capability",
    )
    router.authorize(
        "RETRIEVAL_MISS",
        target_class="applicability",
        operation="PATCH",
        skill_kind="capability",
    )

    with pytest.raises(ValueError, match="monotone Scope narrowing"):
        router.authorize(
            "SCOPE_OVERREACH",
            target_class="retrieval",
            operation="PATCH",
            skill_kind="capability",
        )


def test_q7_uncontracted_axes_are_declared_on_the_card_face() -> None:
    """n>=2 intersection path: dropped schema-illegal axes are named, and
    the machine AST / match verdict do not change."""
    rows = [dict(row) for row in TWO_GOOD]
    for row in rows:
        row["pattern"] = {**PATTERN_A, "level_region_fraction": "very_low"}
    legal = ss._edit_schema_features(PROJECT_ROOT)
    card = _compile(rows, legal_features=legal)["card"]
    assert card is not None
    assert len(rows) >= 2

    dropped = card["risk_guards"][
        "pattern_leaves_dropped_as_uncontracted_for_edit_schema"]
    declared = card["risk_guards"]["scope_unreachable_axes"]
    assert "level_region_fraction" in dropped
    assert declared == list(dropped)
    leaves = {leaf["feature"]
              for leaf in card["observable_applicability"]["all"]}
    assert "level_region_fraction" not in leaves
    assert "level_region_fraction" in (
        card["risk_guards"]["scope_v1"]["pattern_intersection"])

    # Matching is still the AST, not the declaration.
    features = _matching_features()
    assert evaluate_applicability(
        card["observable_applicability"], features)[0] is True
    off = {**features, "local_robust_z_peak": "zero"}
    assert evaluate_applicability(
        card["observable_applicability"], off)[0] is False

    # n=1 path declares the same field (empty when every leaf is contracted).
    single = _compile(TWO_GOOD[:1], legal_features=legal)["card"]
    assert "scope_unreachable_axes" in single["risk_guards"]
    assert single["risk_guards"]["scope_unreachable_axes"] == (
        single["risk_guards"][
            "pattern_leaves_dropped_as_uncontracted_for_edit_schema"])


def test_q11_n2_intersection_card_has_one_task_kind_leaf() -> None:
    """New cards only.  Historical cards are not rewritten."""
    rows = [
        _row("GunPointAgeSpan__impulse_v2", "GunPointAgeSpan__impulse_v2",
             pattern={**PATTERN_A, "task_kind": "classification"}),
        _row("PowerCons__impulse_v2", "PowerCons__impulse_v2",
             support=0.0714, delayed=0.50,
             pattern={**PATTERN_A, "task_kind": "classification"}),
    ]
    assert len(rows) >= 2
    card = _compile(rows)["card"]
    assert card is not None

    leaves = card["observable_applicability"]["all"]
    names = [leaf["feature"] for leaf in leaves]
    expected = 1 + len(PATTERN_A)
    assert names.count("task_kind") == 1
    assert len(leaves) == expected
    assert len(leaves) == len(set(names))
    assert card["risk_guards"]["evidence"]["source_count"] == 2
