"""CAP-1b's pre-declared dedup mark.

SA-1 r1's P4 found a third path nobody had written down.  The card matched
Herring, the card was in view, and still no ``cand_skill_`` entry reached the
pool: the Fast agent had already named the same frozen program and the pool
deduplicated the mechanical supply against it.  With no mark for that case,
"the card was not supplied" and "the card was refused" are the same row in the
ledger, and supply attribution on a fresh Target is unreadable.

CAP-1b therefore required the mark before the capstone.  It is a derivation
over fields the round record already carries -- nothing in ``methods/`` moves,
and nothing about what gets proposed or probed changes.
"""
from __future__ import annotations

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

import run_e2_capstone_epilepsy2 as cap  # noqa: E402

CARD = "sa1_supply_scope_v2"
OPS = ["hampel_filter"]


def _record(*, scope: bool, in_view: bool, pool, proposals) -> dict:
    return {
        "round": "r1",
        "pool": list(pool),
        "retrieved_skill_ids": ([CARD] if in_view else []) + ["bootstrap_x"],
        "scope_match_by_skill_id": {CARD: scope, "bootstrap_x": True},
        "proposals": list(proposals),
    }


def _proposal(candidate_id: str, operators):
    return {"candidate_id": candidate_id, "operators": list(operators)}


def test_a_the_dedup_case_is_marked() -> None:
    """Matched, in view, nothing supplied, and the agent named the same
    program: this is the r1 P4 path and it gets the mark."""
    mark = cap.dedup_swallowed(
        _record(scope=True, in_view=True,
                pool=["identity", "hampel_filter"],
                proposals=[_proposal("identity", []),
                           _proposal("hampel_filter", OPS)]),
        skill_id=CARD, card_operators=OPS)
    assert mark["dedup_swallowed"] is True
    assert mark["scope_match"] is True
    assert mark["card_in_view"] is True
    assert mark["supplied_in_pool"] is False
    assert [row["candidate_id"] for row in mark["self_proposed_same_program"]] \
        == ["hampel_filter"]


def test_b_a_supplied_candidate_is_not_the_dedup_case() -> None:
    """The supply reached the pool, so whatever happened next is a Target
    decision and not a deduplication."""
    mark = cap.dedup_swallowed(
        _record(scope=True, in_view=True,
                pool=["identity", "cand_skill_%s" % CARD],
                proposals=[_proposal("identity", []),
                           _proposal("cand_skill_%s" % CARD, OPS)]),
        skill_id=CARD, card_operators=OPS)
    assert mark["dedup_swallowed"] is False
    assert mark["supplied_in_pool"] is True


def test_c_a_scope_miss_is_not_the_dedup_case() -> None:
    """A card whose Scope does not admit the unit was never going to supply,
    so an absent candidate says nothing about deduplication."""
    mark = cap.dedup_swallowed(
        _record(scope=False, in_view=False,
                pool=["identity", "hampel_filter"],
                proposals=[_proposal("hampel_filter", OPS)]),
        skill_id=CARD, card_operators=OPS)
    assert mark["dedup_swallowed"] is False
    assert "Scope did not match" in mark["why"]


def test_d_a_different_program_is_not_the_dedup_case() -> None:
    """The agent proposed something else entirely; the supply's absence needs
    a different explanation and must not be attributed to dedup."""
    mark = cap.dedup_swallowed(
        _record(scope=True, in_view=True,
                pool=["identity", "outlier_iqr"],
                proposals=[_proposal("outlier_iqr", ["outlier_iqr"])]),
        skill_id=CARD, card_operators=OPS)
    assert mark["dedup_swallowed"] is False
    assert mark["self_proposed_same_program"] == []


def test_e_candidate_sources_split_supplied_from_self_proposed() -> None:
    result = {"rounds": [
        _record(scope=True, in_view=True,
                pool=["identity", "hampel_filter"],
                proposals=[_proposal("identity", []),
                           _proposal("hampel_filter", OPS)]),
    ]}
    split = cap.annotate_candidate_sources(
        result, skill_id=CARD, card_operators=OPS)
    assert split["supplied_total"] == 0
    assert split["self_proposed_total"] == 1
    assert split["dedup_swallowed_total"] == 1
    assert split["scope_matched_any_round"] is True


def test_f_the_mark_reads_only_recorded_fields() -> None:
    """A guard against the mark quietly becoming a second mechanism: it must
    be computable from an inert dict, with no snapshot, store or agent."""
    record = _record(scope=True, in_view=True, pool=[], proposals=[])
    before = dict(record)
    cap.dedup_swallowed(record, skill_id=CARD, card_operators=OPS)
    assert record == before
