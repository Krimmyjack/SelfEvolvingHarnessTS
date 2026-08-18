"""Program Supply fault routing contract (plan rev3, 2026-08-16).

The Program Supply slice has exactly five evidence inputs.  This module is
shared by the controlled minipipe fold and the online method layer so the two
callers cannot drift into separate copies of the routing logic.

G1 narrow repair (2026-08-18, docs/BOUNDED_SKILL_CARD_ATTRIBUTION_DESIGN_
2026-08-18.md rev2.0 §7 item 2): the five evidence inputs can only reach
``PROPOSAL_CONTROL_GAP`` through ``capability_skill_exists and
skill_retrieved``.  A repeated, Context-resolved harmful proposal that came
from the free proposal path with **no** matching Skill therefore fell through
to the non-editable ``CANDIDATE_SUPPLY_UNKNOWN`` / ``EVIDENCE_BACKLOG``.  One
optional Runtime-computed flag opens that single missing edge.  It defaults to
False, so every existing caller and every existing branch is unchanged, and
the 25-code fault table is not rewritten.
"""

from __future__ import annotations

PROGRAM_SUPPLY_ROUTE_FIELDS = (
    "expressibility_status",
    "expressibility_cause",
    "capability_skill_exists",
    "skill_retrieved",
    "constrained_proposal_succeeds",
)


def route_program_supply_fault(
    *,
    expressibility_status: str,
    expressibility_cause: str | None,
    capability_skill_exists: bool,
    skill_retrieved: bool,
    constrained_proposal_succeeds: bool | None,
    context_resolved_decision_fault: bool = False,
) -> tuple[str, str, tuple[str, ...]]:
    """Pure Program Supply router: ``(fault_family, actionability, surfaces)``.

    The five ``PROGRAM_SUPPLY_ROUTE_FIELDS`` parameters are keyword-only and
    required.  ``constrained_proposal_succeeds`` is tri-state: ``True`` ->
    PROPOSAL_CONTROL_GAP, ``False`` -> SKILL_CONTENT_GAP, ``None``/Unknown ->
    ABSTAIN.

    ``context_resolved_decision_fault`` is the optional G1 edge and is not a
    sixth evidence field: it is a Runtime-computed DECISION_GAP predicate over
    already-open Experience Episodes (repeated distinct Task Episodes, one
    Program mechanism, one already-expressible public Context condition, and
    an opposite-Context positive control).  It is only consulted after every
    pre-existing branch has declined, so it can never re-route an existing
    case.
    """
    if constrained_proposal_succeeds is not None and not isinstance(
        constrained_proposal_succeeds, bool
    ):
        raise ValueError(
            "constrained_proposal_succeeds must be True, False, or None"
        )
    if expressibility_status == "PROVEN_UNAVAILABLE":
        return "OPERATOR_GAP", "CAPABILITY_BACKLOG", ()
    if expressibility_cause == "OBSERVABLE_DERIVATION_PROCEDURE_GAP":
        return (
            "OBSERVABLE_DERIVATION_PROCEDURE_GAP",
            "EDITABLE_M0",
            ("bootstrap_skills.entries/inspect_and_localize.body",),
        )
    if expressibility_cause == "OBSERVABLE_FEATURE_SCHEMA_GAP":
        return "OBSERVABLE_FEATURE_SCHEMA_GAP", "OBSERVATION_CAPABILITY_BACKLOG", ()
    if expressibility_status == "EXPRESSIBILITY_UNKNOWN":
        return "EXPRESSIBILITY_UNKNOWN", "EVIDENCE_BACKLOG", ()
    if expressibility_status == "PROVEN_EXPRESSIBLE" and not capability_skill_exists:
        return (
            "SKILL_LIBRARY_GAP",
            "EDITABLE_M0",
            ("skill_library.entries/{skill_id}",),
        )
    if capability_skill_exists and skill_retrieved:
        if constrained_proposal_succeeds is True:
            return (
                "PROPOSAL_CONTROL_GAP",
                "EDITABLE_M0",
                ("candidate_policy.proposal_guidance",),
            )
        if constrained_proposal_succeeds is False:
            return (
                "SKILL_CONTENT_GAP",
                "EDITABLE_M0",
                ("skill_library.entries/{skill_id}.body",),
            )
        return "CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ()
    if context_resolved_decision_fault:
        return (
            "PROPOSAL_CONTROL_GAP",
            "EDITABLE_M0",
            ("candidate_policy.proposal_guidance",),
        )
    return "CANDIDATE_SUPPLY_UNKNOWN", "EVIDENCE_BACKLOG", ()


__all__ = [
    "PROGRAM_SUPPLY_ROUTE_FIELDS",
    "route_program_supply_fault",
]
