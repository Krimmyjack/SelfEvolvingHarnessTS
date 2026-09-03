"""P4b held-out scoring: frozen state in, one reading out, no feedback back.

Held-out is where the primary endpoint is measured, and the only thing that
distinguishes the arms here is what each of them learned during held-in: the
admission gate needs a Support probe to fire, and held-out takes no Support at
all.  So an arm deploys whatever its frozen state recalls -- an ACTIVE Skill
carrying a Workflow, else the incumbent it froze on, else identity -- and the
reading is taken once and never returned to the arm.

This module therefore takes no backend and no agent: ``_frozen_recall`` is
deterministic retrieval (``resolve_harness_view`` plus a frozen-step parse),
so the held-out phase spends no LLM by construction, not by discipline.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import p4b_contract as contract
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features

# The roster split the held-out reading is taken on.  It is the support_a
# geometry only in the sense of train/eval roles; no Support is taken here and
# nothing flows back, so this is a deployment face, not a feedback face.
HELD_OUT_FACE = "support_a"
MATERIAL = 0.005


class _NoProviderAgent:
    """Stands in for the Fast agent on a face that must never call one.

    Held-out deployment is ``_frozen_recall``: deterministic retrieval over the
    frozen snapshot.  Giving the state a real agent would leave a live provider
    handle on the endpoint face; anything that tried to use it here would raise
    instead of quietly spending budget.
    """

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(
            "held-out scoring must not call the Fast agent (attempted %r)" % name
        )


def frozen_state(
    *, snapshot: Any, episodes: Any, ledger: Mapping[str, Any],
    store_root: Any, tag: str,
) -> dict[str, Any]:
    """An arm's frozen state, rebuilt for held-out with no provider attached."""
    from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms

    state = four_arms._new_state(
        snapshot=snapshot,
        agent=_NoProviderAgent(),
        store_root=store_root,
        tag=tag,
        episodes=tuple(episodes or ()),
    )
    state["incumbent"] = dict(ledger).get("incumbent")
    state["approved_skill_ids"] = list(dict(ledger).get("approved_skill_ids") or ())
    return state


def frozen_decision(state: Mapping[str, Any], cell: Any) -> dict[str, Any]:
    """What this frozen arm deploys here.  Reads no Outcome, spends no LLM."""
    features = dict(
        extract_public_features(cell.observation_block, task_kind=forecast_p4.TASK)
    )
    # _frozen_recall already applies the whole symmetric rule -- an applicable
    # ACTIVE Skill carrying a Workflow, else the frozen incumbent, else
    # identity -- and labels which of the three it took.  Adding a second
    # incumbent fallback here would only shadow its label.
    decision = shared_harness._frozen_recall(state, features)
    steps = list(decision.get("applied_steps") or [])
    return {
        "applied_steps": steps,
        "deploy_source": str(decision.get("source") or "identity"),
        "active_skill_id": decision.get("active_skill_id"),
        "recall_hit": bool(decision.get("recall_hit")),
        "why": decision.get("why"),
    }


def score(
    base_cell: Any, origin: int, applied_steps: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """One held-out reading of a frozen program against identity."""
    cell = forecast_p4._cell_at(base_cell, origin)
    identity = forecast_p4._reading(cell, HELD_OUT_FACE, (), origin=origin)
    if applied_steps:
        steps = tuple(
            (str(step["op"]), dict(step.get("params") or {}))
            for step in applied_steps
        )
        deployed = forecast_p4._reading(cell, HELD_OUT_FACE, steps, origin=origin)
    else:
        deployed = identity
    before = np.asarray(identity["per_series_smase"], dtype=np.float64)
    after = np.asarray(deployed["per_series_smase"], dtype=np.float64)
    gains = before - after  # positive = sMASE fell = this series improved
    harmed = gains < -MATERIAL
    lowest = float(gains.min()) if gains.size else 0.0
    aggregate = float(identity["smase"] - deployed["smase"])
    return {
        "origin": int(origin),
        "face": HELD_OUT_FACE,
        "identity_smase": float(identity["smase"]),
        "deployed_smase": float(deployed["smase"]),
        # The primary endpoint, on the same scale the old P4 reported.
        "delta_utility_vs_identity": aggregate,
        "material_harm_event": bool(aggregate < -MATERIAL),
        "series_count": int(gains.size),
        "harmed_count": int(harmed.sum()),
        "harmed_fraction": float(harmed.mean()) if gains.size else 0.0,
        "worst_single_series_harm": -lowest if lowest < 0.0 else 0.0,
        "per_series_gain": [float(value) for value in gains],
        "identity_per_series_smase": [float(value) for value in before],
        "consumer_fits": 1 if applied_steps else 0,
        "identity_reference_fits": 1,
        "llm_calls": 0,
    }


def row(
    *,
    arm: str,
    replica: str,
    origin: int,
    state: Mapping[str, Any],
    base_cell: Any,
) -> dict[str, Any]:
    """One held-out cell: recall under the frozen state, then score it once."""
    cell = forecast_p4._cell_at(base_cell, origin)
    decision = frozen_decision(state, cell)
    reading = score(base_cell, origin, decision["applied_steps"])
    return {
        "phase": "held_out",
        "arm": arm,
        "replica": replica,
        "held_out_role": contract.ARMS_BY_NAME[arm].held_out_role
        if arm in contract.ARMS_BY_NAME
        else "reference",
        **reading,
        "deploy": decision,
        "feedback_taken": False,
        "state_written": False,
    }


def identity_row(*, replica: str, origin: int, base_cell: Any) -> dict[str, Any]:
    """The Static reference: identity on every held-out origin, 0 LLM."""
    return {
        "phase": "held_out",
        "arm": "Static",
        "replica": replica,
        "held_out_role": "reference",
        **score(base_cell, origin, ()),
        "deploy": {
            "applied_steps": [],
            "deploy_source": "identity",
            "active_skill_id": None,
            "recall_hit": False,
            "why": "Static: identity frozen on every origin",
        },
        "feedback_taken": False,
        "state_written": False,
    }


def frozen_program_row(
    *,
    arm: str,
    replica: str,
    origin: int,
    base_cell: Any,
    applied_steps: Sequence[Mapping[str, Any]],
    selection_face: str,
) -> dict[str, Any]:
    """A reference that froze one program on held-in and redeploys it as-is.

    ``Parallel Best-of-N@8`` searches on held-in and must carry the winner over
    unchanged; selecting again on held-out would turn the endpoint into a
    selection face, so the frozen steps are passed in and the face they were
    chosen on is recorded for audit.
    """
    if selection_face != "held_in":
        raise ValueError(
            "a frozen comparator may only select on held-in, got %r" % selection_face
        )
    return {
        "phase": "held_out",
        "arm": arm,
        "replica": replica,
        "held_out_role": "reference",
        **score(base_cell, origin, applied_steps),
        "deploy": {
            "applied_steps": [dict(step) for step in applied_steps],
            "deploy_source": "frozen_held_in_selection",
            "active_skill_id": None,
            "recall_hit": False,
            "why": "program selected on held-in and redeployed unchanged",
        },
        "parallel_selection_face": selection_face,
        "feedback_taken": False,
        "state_written": False,
    }


def store_semantics(method: Any) -> tuple[Any, ...]:
    """The frozen Skill/Memory view K0's invariance check compares.

    Field by field -- skill id, revision, body, observable applicability, risk
    guards -- so a difference names what changed.  No SHA, no manifest, no
    ledger: K0's contract is that these fields are identical before and after
    held-in, and comparing the fields says that directly.
    """
    return tuple(forecast_p1._snapshot_state_view(method._active_snapshot()))
