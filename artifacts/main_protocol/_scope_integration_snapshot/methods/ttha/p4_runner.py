"""P4 batch-end runner binding (rev4 final, 2026-08-16).

This is the only Slow-entry path allowed for the P4 natural slice.

Binding unit is ``(patch_id, exact ordered program_steps)``, not patch_id
alone.  The Card is built once, E-1 verification is computed from that same
Card, the filtered Card is then passed unchanged to Slow, and
``handle_group_feedback`` re-checks the selected patch_id against the
verified IDs.  A duplicate patch_id or a same-ID/different-steps mismatch is
rejected before Slow is invoked.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .program_supply import (
    bind_verified_program_options,
    build_single_surface_catalog,
    route_verified_program_supply_fault,
)


def run_p4_group_update(
    *,
    method: Any,
    group: Mapping[str, object],
    capsule: Mapping[str, object],
    trace: Any,
    episode: Any,
    view: Any,
    executor: Any,
    origin: int,
    slow_agent: Any,
    controller: Any,
    store: Any,
    card_builder: Callable[[Mapping[str, object], Mapping[str, object]],
                            Mapping[str, object]],
    evaluator_group: Callable[[Sequence[tuple[str, Mapping[str, object]]],
                               Any], Any],
    holdout_evaluator: Callable[[Sequence[tuple[str, Mapping[str, object]]],
                                 int], Any] | None = None,
    fast_features: Mapping[str, object] | None = None,
    allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
    task_context: Any | None = None,
    constrained_proposal_succeeds: bool | None = None,
) -> dict[str, Any]:
    """Batch-end P4 Slow update.

    The Card is built once; E-1 verification is run on that exact Card;
    then the exact-step-bound filtered Card is the only input Slow sees.
    """
    card = card_builder(group, capsule)
    if not isinstance(card, Mapping):
        return {"stage": "card_not_mapping"}
    assessment = route_verified_program_supply_fault(
        trace=trace,
        episode=episode,
        view=view,
        executor=executor,
        typed_patch_options=card.get("typed_patch_options") or [],
        origin=origin,
        constrained_proposal_succeeds=constrained_proposal_succeeds,
    )
    filtered_card, verified_ids, error = bind_verified_program_options(
        card, assessment
    )
    if error is not None:
        return error

    parent = store.materialize(method._active_snapshot())
    surface_catalog = build_single_surface_catalog(
        decision=assessment.decision,
        parent=parent,
        controller=controller,
    )
    if not surface_catalog:
        return {
            "stage": "route_not_add_only",
            "case_id": assessment.facts.case_id,
            "cause_code": assessment.decision.cause_code,
            "actionability": assessment.decision.actionability,
        }
    return method.handle_group_feedback(
        group,
        capsule,
        slow_agent=slow_agent,
        controller=controller,
        store=store,
        card_builder=lambda _group, _capsule: filtered_card,
        evaluator_group=evaluator_group,
        holdout_evaluator=holdout_evaluator,
        fast_features=fast_features,
        surface_catalog=surface_catalog,
        route_decision=assessment.decision,
        allowed_operator_contracts=allowed_operator_contracts,
        task_context=task_context,
        evidence_compiler=False,
        runtime_selected_patch_id=None,
        verified_choice_offered=bool(
            assessment.verification.choice_offered
        ),
        verified_patch_ids=verified_ids,
    )


__all__ = ["bind_verified_program_options", "run_p4_group_update"]
