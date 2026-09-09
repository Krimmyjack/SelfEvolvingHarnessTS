"""DEV-SEQ-4 step 3: what Slow is shown when it revises its own guidance.

DEV-SEQ-3 wrote a card, the card reached Fast, and Fast changed its selection.
The delivered program did not change.  Step 2 (``run_dev_seq4``) held the
pre-decision history fixed and repeated each cell, so the process differences
are now measured rather than inferred.

This module builds the input for a second Slow turn.  It adds three blocks to
the DEV-SEQ-2 evidence card:

1. ``previous_guidance`` -- the text that is in the snapshot now, the
   predictions it made about its own effect, and what was actually measured
   against each of them.  Slow is being asked to review its own work with the
   receipt in front of it.
2. ``how_a_program_is_actually_delivered`` -- a mechanical fact about the
   Harness that DEV-SEQ-3 established by reading ``online_loop``: the select
   stage orders probes, it does not authorise or refuse deployment.  This is
   input completion of the same kind as the feature semantics: withholding it
   guarantees another card whose stated goal the architecture cannot satisfy.
   It is **not** a suggested fix and names no surface.
3. ``what_the_same_knowledge_repeats_did`` -- the nearest-miss cells, where
   both arms held the same effective knowledge.  Slow needs to see how much
   the model varies between repeats before reading anything into a difference.

What is deliberately NOT here: any instruction to write a card, to keep the
ADD surface, to change a threshold in a particular direction, or to make Fast
deploy a different program.  Keeping the current guidance unchanged is a
first-class outcome, and the input says so.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from evaluation.main_protocol_p4 import dev_seq2_knowledge as know  # noqa: F401
from evaluation.main_protocol_p4 import run_dev_seq2 as seq2

#: Read out of ``methods/ttha/online_loop.py`` (probe loop and winner rule) and
#: confirmed against 120 recorded DEV-SEQ-3 decisions.  Stated as mechanism,
#: with the measured rates, and with the reason it is not simply a defect.
DELIVERY_MECHANISM: dict[str, Any] = {
    "what_the_select_stage_does": (
        "the candidate id Fast returns from select decides the ORDER probes "
        "are run in (chosen first, then the rest of the pool).  Choosing "
        "'identity' means only that no candidate is moved to the front; the "
        "pool is still probed"),
    "what_decides_the_delivered_program": (
        "the first probed candidate whose measured Support is admitted by the "
        "risk rule becomes the winner and is deployed.  Support is the "
        "authority here, not the selection"),
    "so_a_selection_of_identity_does_not_prevent_a_deployment": True,
    "measured_over_the_120_recorded_decisions": {
        "select_said_identity": 20,
        "of_those_the_runtime_deployed_anyway": 12,
        "select_named_a_candidate": 96,
        "of_those_the_runtime_abstained": 33,
        "of_those_the_runtime_deployed_a_different_candidate": 18,
        "selection_determined_the_delivered_program": "53 of 116",
    },
    "the_same_thing_under_control": (
        "step 2 repeated one matched situation three times with the "
        "pre-decision history held fixed.  Without your guidance the agent "
        "selected the outlier repair in 3 of 3 repeats; with it the agent "
        "selected identity in 3 of 3.  The delivered program was the outlier "
        "repair in all six, with the same delayed reading.  The selection "
        "moved the whole way and the delivery did not move at all"),
    "this_is_not_presented_to_you_as_a_defect": (
        "of the 12 deployments that overrode an identity selection, the "
        "delayed readings sum to +3.5842 (7 positive totalling +4.2745, 5 "
        "negative totalling -0.6903).  Overriding is not uniformly harmful "
        "and you are not being asked to argue that the runtime should change"),
    "why_you_are_told_this": (
        "your previous guidance predicted 'identity_retained'.  That "
        "prediction cannot be satisfied by changing what select returns.  "
        "Knowing this does not tell you what to write instead -- it only "
        "rules out one route.  Candidate supply is a different route: a "
        "candidate whose id begins 'cand_skill_' was placed in the pool by a "
        "Skill card, not proposed by the agent, and it is probed whatever the "
        "agent proposes"),
    "what_this_does_not_authorise": (
        "you may not edit candidate-supply authority, the risk rule, the "
        "verifier, or anything outside writable_surface_catalog.  This fact "
        "is context for judging your own prediction, not a new surface"),
}


def _predictions_vs_measurement(previous: Mapping[str, Any],
                                measured: Mapping[str, Any]
                                ) -> list[dict[str, Any]]:
    """Each prediction the previous edit made, against what was observed."""
    matched = (measured.get("by_situation_kind") or {}).get("matched") or {}
    arm_a = matched.get(seq2.ARM_A) or {}
    arm_b = matched.get(seq2.ARM_B) or {}
    rows: list[dict[str, Any]] = []
    for predicted in previous.get("predicted_agent_behavior_change") or ():
        text = str(predicted)
        if text.startswith("retrieve_skill:"):
            rows.append({
                "you_predicted": text,
                "observed": ("the card was retrieved in %s of %s decisions "
                             "where its condition held, and in %s of %s where "
                             "the card was absent from the snapshot"
                             % (arm_b.get("card_retrieved"),
                                arm_b.get("decisions"),
                                arm_a.get("card_retrieved"),
                                arm_a.get("decisions"))),
                "verdict": "HELD",
            })
        elif "identity" in text and "choose" in text:
            rows.append({
                "you_predicted": text,
                "observed": ("select returned identity in %s of %s decisions "
                             "with the card and %s of %s without it, on the "
                             "same fixed pre-decision history"
                             % (arm_b.get("select_identity"),
                                arm_b.get("decisions"),
                                arm_a.get("select_identity"),
                                arm_a.get("decisions"))),
                "verdict": "SEE_THE_NUMBERS",
            })
        else:
            rows.append({
                "you_predicted": text,
                "observed": ("the delivered program was identical in every "
                             "matched decision: %s deployments with the card, "
                             "%s without"
                             % (arm_b.get("deployed"), arm_a.get("deployed"))),
                "verdict": "DID_NOT_HOLD",
            })
    return rows


def revision_card(base_card: Mapping[str, Any], *,
                  pattern_id: str,
                  previous_proposal: Mapping[str, Any],
                  current_body: str,
                  current_applicability: Any,
                  contrast: Mapping[str, Any]) -> dict[str, Any]:
    """The DEV-SEQ-2 evidence card, plus the receipt for the previous edit."""
    measured = contrast.get("summary") or {}
    near = (measured.get("by_situation_kind") or {}).get("nearest_miss") or {}
    card = dict(base_card)
    card["pattern_id"] = pattern_id

    card["previous_guidance"] = {
        "this_text_is_already_in_the_snapshot": current_body,
        "its_applicability_condition": current_applicability,
        "you_wrote_it_at_the_previous_boundary": True,
        "what_you_predicted_and_what_was_measured":
            _predictions_vs_measurement(previous_proposal, measured),
        "your_own_falsification_conditions": list(
            previous_proposal.get("falsification_condition") or ()),
        "how_the_measurement_was_made": (
            "six sequence-window situations, two arms, three repeats each.  "
            "Both arms were handed the SAME pre-decision history for that "
            "sequence, so the only difference between them is whether this "
            "text is in the snapshot.  Three situations match your condition; "
            "three are the closest situations it does not match"),
        "the_question_put_to_you": (
            "what did this guidance lack?  Was it repeating a rule the "
            "Harness already applies, or did it fail to say what to check, in "
            "which situation, before choosing?  Answer from the evidence on "
            "this card, not from the outcome of any single sequence"),
        "a_later_loss_does_not_mean_it_should_have_been_foreseen": (
            "one matched decision deployed a program whose Support was "
            "positive and whose delayed reading was negative.  At decision "
            "time only the positive Support was visible.  Do not reason "
            "backwards from the delayed number to what the agent should have "
            "known"),
    }

    card["how_a_program_is_actually_delivered"] = DELIVERY_MECHANISM

    card["what_the_same_knowledge_repeats_did"] = {
        "these_cells_hold_the_same_effective_knowledge_in_both_arms": True,
        "per_arm": {name: value for name, value in near.items()
                    if isinstance(value, Mapping)},
        "why_it_is_here": (
            "the card is not retrievable in these situations, so every "
            "difference between the arms there is the model varying between "
            "runs.  Read the matched situations against this, not against "
            "zero"),
    }

    card["per_situation_detail"] = measured.get("per_situation") or []

    card["you_may_keep_the_current_guidance"] = (
        "revising is not required.  If the evidence here does not support a "
        "change, return the no_proposal envelope with "
        "insufficient_public_evidence.  That is a legitimate answer and it is "
        "not treated as a failure.  Do not invent a change to have one")

    card["what_you_may_write"] = (
        "Exactly one edit, on exactly one surface from "
        "writable_surface_catalog -- which now includes the body and the "
        "applicability condition of the guidance entry you wrote last time, "
        "so revising it in place is available and so is leaving it alone.  "
        "You are NOT required to add a new entry.  Say four things: (1) which "
        "facts on this card you are relying on; (2) what you now think the "
        "previous guidance got wrong or left out, as a hypothesis you accept "
        "may be wrong; (3) the one change; (4) what you expect to change in "
        "behaviour and effect, and what result would refute you.  Points (1) "
        "and (2) go in the text you write; (4) goes in "
        "predicted_agent_behavior_change, predicted_data_effect and "
        "falsification_condition.  Where the current behaviour works, say "
        "what should be kept.")

    card["where_this_will_be_checked"] = (
        "on later development windows that are not on this card.  A condition "
        "fitted to the exact windows shown here will not transfer, and "
        "tightening one to exclude a single sequence that lost is fitting, "
        "not learning")
    return card


def unchanged_treatment(reason: str) -> dict[str, Any]:
    """What the follow-up compares when Slow keeps the current guidance."""
    return {
        "outcome": "NO_REVISION_TREATMENT",
        "why": reason,
        "what_the_followup_then_compares": (
            "nothing new: with no second edit there is no revised knowledge "
            "to freeze, so the follow-up contrast is not run and no utility "
            "claim is made.  This is recorded as an answer, not as a fault"),
    }


def summarise_for_report(contrast: Mapping[str, Any]) -> dict[str, Any]:
    """The step-2 numbers a reader needs before any step-3 claim."""
    summary = contrast.get("summary") or {}
    by_kind = summary.get("by_situation_kind") or {}
    out: dict[str, Any] = {}
    for kind in ("matched", "nearest_miss"):
        block = by_kind.get(kind) or {}
        out[kind] = {
            arm: {k: v for k, v in (block.get(arm) or {}).items()
                  if k in ("decisions", "card_retrieved", "select_identity",
                           "deployed", "candidates_total", "tool_calls_total",
                           "delayed_mean")}
            for arm in (seq2.ARM_A, seq2.ARM_B)}
    out["reading_rule"] = (
        "a difference in 'matched' that 'nearest_miss' does not also show is "
        "the only thing here that can be the card; anything both blocks show "
        "is the model varying between runs")
    return out


__all__ = ["DELIVERY_MECHANISM", "revision_card", "unchanged_treatment",
           "summarise_for_report"]


def _selftest() -> None:  # pragma: no cover - exercised by the smoke
    assert "cand_skill_" in DELIVERY_MECHANISM["why_you_are_told_this"]
    assert json.dumps(DELIVERY_MECHANISM)
