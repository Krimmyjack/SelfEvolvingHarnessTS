"""P4b held-in phase: adapt under one arm's admission rule, then freeze.

Two things separate this from the old P4 adaptive row.

First, the arm's *state dict* survives the origin loop, not just its snapshot
and Episodes.  Held-out deployment reads ``incumbent`` and
``approved_skill_ids``, which is what the arm is actually standing on once both
gates have spoken; carrying only the snapshot would freeze an arm that has
nothing to deploy.  That is also what makes the canonical A3 possible (AGENTS
section 2.1: from the public h0, adapting on Target feedback) rather than the
old ``A3-reset``, which discarded state every unit.

Second, every probe's per-series split and admission verdict is written down.
The old P4 artifact kept only the aggregate ``support_gain``, which is why the
CONFLICT diagnosis had to recompute the per-series readings after the fact, and
why the conservative sensitivity point can only be re-read offline if the
readings are actually stored.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

MATERIAL = 0.005


def empty_ledger() -> dict[str, Any]:
    """What an arm is standing on before it has learned anything."""
    return {"incumbent": None, "approved_skill_ids": []}


def read_ledger(state: Mapping[str, Any]) -> dict[str, Any]:
    """The deployable part of a state: what held-out will actually recall."""
    return {
        "incumbent": state.get("incumbent"),
        "approved_skill_ids": list(state.get("approved_skill_ids") or ()),
    }


def new_state(*, snapshot: Any, cell: Any, backend: Any, store_root: Path,
              tag: str, episodes: Sequence[Any] = (),
              ledger: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One arm's live state for one origin.

    The state object is rebuilt each origin because the agent is bound to that
    origin's observation block and to a per-cell budget scope.  The *ledger* --
    the incumbent and the approved Skill ids -- is carried in explicitly, since
    that is what an arm deploys once feedback stops and rebuilding would silently
    reset it.
    """
    state = four_arms._new_state(
        snapshot=snapshot,
        agent=forecast_course._live_agent(cell.observation_block, backend),
        store_root=store_root,
        tag=tag,
        episodes=tuple(episodes),
    )
    carried = dict(ledger or empty_ledger())
    state["incumbent"] = carried.get("incumbent")
    state["approved_skill_ids"] = list(carried.get("approved_skill_ids") or ())
    return state


def _probe_rows(result: Any, identity_per_series: Sequence[float]) -> list[dict[str, Any]]:
    """Every probe, with the per-series split and the admission verdict.

    ``passed`` on a probe record means the probe executed legally, not that the
    candidate qualified -- the qualifying decision is the ``admission`` block.
    """
    rows: list[dict[str, Any]] = []
    for probe in result.actual_probed_programs:
        row = {
            "candidate_id": probe.get("candidate_id"),
            "kind": probe.get("kind"),
            "aggregate_gain": probe.get("gain"),
            "probe_executed_legally": probe.get("passed"),
            "admission": probe.get("admission"),
        }
        rows.append(row)
    return rows


def _episode_rows(method: Any, start: int) -> list[dict[str, Any]]:
    """Episodes written this cell, keeping the per-series readings."""
    rows: list[dict[str, Any]] = []
    for episode in method.experience_episodes[start:]:
        context = dict(getattr(episode, "context_summary", None) or {})
        support = dict(getattr(episode, "support_response", None) or {})
        delayed = dict(getattr(episode, "delayed_response", None) or {})
        rows.append(
            {
                "episode_id": str(getattr(episode, "episode_id", "")),
                "relation": str(getattr(episode, "relation", "")),
                "local_status": str(getattr(episode, "local_status", "")),
                "source_skill_id": context.get("source_skill_id"),
                "support_gain": support.get("gain"),
                "support_accepted": support.get("accepted"),
                # The reading the offline sensitivity re-read needs.
                "support_per_view_gain": list(context.get("per_view_gain") or ()),
                "delayed_gain": delayed.get("gain"),
                "delayed_per_view_gain": list(delayed.get("per_view_gain") or ()),
                "series_uids": list(context.get("series_uids") or ()),
            }
        )
    return rows


def run_cell(
    *,
    arm: contract.Arm,
    replica: str,
    origin: int,
    sequence_index: int,
    state: Mapping[str, Any],
    base_cell: Any,
    identity: Mapping[str, Mapping[str, Any]],
    spec: Any,
    context: Any,
) -> dict[str, Any]:
    """One held-in cell under this arm's admission rule.

    The caller owns ``state`` (and the backend behind its agent) and the
    admission scope; this runs the round, opens the delayed face, and reports
    what happened.
    """
    started = time.time()
    cell = forecast_p4._cell_at(base_cell, origin)
    method = state["method"]
    before_semantics = tuple(
        forecast_p1._snapshot_state_view(method._active_snapshot())
    )
    episode_start = len(method.experience_episodes)
    unit = {
        "replica": replica,
        "episode_id": "H%d" % sequence_index,
    }
    request, features = forecast_p4._request(
        unit=unit, cell=cell, origin=origin, spec=spec, context=context
    )
    config = forecast_p4._config(origin)
    evaluator = forecast_p4._CountingEval(cell, config, origin)
    support_token = int(origin)
    delayed_token = int(origin) + contract.HORIZON
    executors = {}
    for face, token in (("support_a", support_token), ("support_b", delayed_token)):
        executor = ScopeExecutor(
            cell.roster(face),
            cell.values,
            config,
            evaluate_fn=evaluator,
            max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
        )
        executor._baseline_cache[origin] = float(identity[face]["smase"])
        executor._per_view_cache[origin] = [
            float(value) for value in identity[face]["per_series_smase"]
        ]
        executors[face] = executor
    dispatcher = forecast_p4._OriginDispatcher(
        {
            support_token: ("support_a", executors["support_a"], origin),
            delayed_token: ("support_b", executors["support_b"], origin),
        }
    )

    result = run_online_round(
        method,
        dispatcher,
        request,
        cell.values,
        origin=support_token,
        slow_agent=None,
        controller=state["controller"],
        store=state["store"],
        card_builder=forecast_course._card_builder,
        round_name="%s_%s_%s" % (
            replica.lower(), unit["episode_id"].lower(),
            arm.name.lower().replace("-", "_"),
        ),
        budget=contract.MAX_SUPPORT_A,
        allow_slow=contract.ALLOW_SLOW,
        horizon=contract.HORIZON,
        period=contract.PERIOD,
        domain="forecast_p4b_%s" % unit["episode_id"],
        fast_features=features,
        allow_fast_skill=True,
        runtime_prior_slot=False,
        pool_mode="full",
    )
    open_delayed(result, dispatcher, delayed_origin=delayed_token,
                 store=state["store"])

    # Freeze bookkeeping: an approved Skill joins the ledger, and the incumbent
    # is the last Workflow that passed *both* gates -- a Support winner the
    # delayed face refused is not adopted.
    activated = False
    if result.approved_skill_id is not None:
        activated = activate_approved(result, state["store"])
        if activated:
            state["approved_skill_ids"].append(str(result.approved_skill_id))
    incumbent_before = state.get("incumbent")
    state["incumbent"] = shared_harness._incumbent_after_delayed(
        result, incumbent_before
    )

    after_semantics = tuple(
        forecast_p1._snapshot_state_view(method._active_snapshot())
    )
    identity_a = identity["support_a"]
    return {
        "phase": "held_in",
        "arm": arm.name,
        "replica": replica,
        "sequence_index": sequence_index,
        "origin": int(origin),
        "admission_rule": (
            contract.BOUNDED_POLICY if arm.bounded else contract.STRICT_POLICY
        ).to_dict(),
        # Diagnostics: an abstain with no probes at all means the round never
        # reached the admission gate, which is a different failure from a
        # candidate being refused by it.
        "retrieved_skill_ids": list(
            getattr(method.last_trace, "retrieved_skill_ids", None) or ()
        ),
        "candidate_ids": list(
            getattr(method.last_trace, "candidate_ids", None) or ()
        ),
        "chosen_candidate_id": str(
            getattr(method.last_trace, "chosen_candidate_id", "") or ""
        ),
        "abstain_reason": getattr(result, "abstain_reason", None),
        "winner_candidate_id": str(getattr(result, "_winner_candidate_id", "") or ""),
        "winner_program": (
            [dict(step) for step in result.winner_program]
            if result.winner_program is not None else []
        ),
        "abstained": bool(result.abstained),
        "support_receipts_used": int(result.target_support_receipts_used),
        "probes": _probe_rows(result, identity_a["per_series_smase"]),
        "episodes_written": _episode_rows(method, episode_start),
        "approved_skill_id": result.approved_skill_id,
        "activated": bool(activated),
        "incumbent_after": (
            [dict(step) for step in state["incumbent"]]
            if state.get("incumbent") else []
        ),
        "incumbent_changed": state.get("incumbent") != incumbent_before,
        "store_semantics_changed": before_semantics != after_semantics,
        "fast_skill_event": _plain_event(getattr(result, "_fast_skill_event", None)),
        "delayed_event": _plain_event(getattr(result, "_delayed_event", None)),
        "usage": {
            "support_a_full_evaluations": int(evaluator.fits_by_face["support_a"]),
            "support_b_full_evaluations": int(evaluator.fits_by_face["support_b"]),
            "cheap_probes": int(getattr(result, "cheap_probe_count", 0) or 0),
            "accepted_updates": 1 if activated else 0,
            "wall_seconds": round(time.time() - started, 3),
        },
        "boundary": {
            "held_out_origins_touched": 0,
            "natural_final_outcome_reads": 0,
        },
    }


def _plain_event(event: Any) -> dict[str, Any] | None:
    if event is None:
        return None
    return {
        str(key): value
        for key, value in dict(event).items()
        if isinstance(value, (str, int, float, bool, type(None), dict, list))
    }


def store_delta(before: tuple[Any, ...], after: tuple[Any, ...]) -> dict[str, Any]:
    """Field-by-field difference between two frozen Skill/Memory views."""
    return {
        "identical": before == after,
        "entries_before": len(before),
        "entries_after": len(after),
        "changed_entries": [
            {"before": list(b), "after": list(a)}
            for b, a in zip(before, after)
            if b != a
        ] if len(before) == len(after) else [],
    }


def gated_writeback_check(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The Skill store may only move through a gate that admitted something.

    This is the leakage guard that survives dropping the no-write-back arm.
    Both arms here write back, so "the store never changed" is not the property
    to check; the property is that every change was *paid for* -- an admitted
    probe, an approved Skill, or a new incumbent.  A Skill-level change on a
    cell where nothing was admitted means write-back found a path around the
    admission gate, which would void the strict/bounded contrast.
    """
    offenders = []
    for row in rows:
        if not row.get("store_semantics_changed"):
            continue
        admitted = any(
            (probe.get("admission") or {}).get("admitted")
            for probe in row.get("probes") or ()
        )
        if admitted or row.get("approved_skill_id") or row.get("incumbent_changed"):
            continue
        offenders.append(
            {
                "arm": row.get("arm"),
                "replica": row.get("replica"),
                "origin": row.get("origin"),
                "reading": "Skill store moved with no admission, approval, or "
                           "incumbent change on this cell",
            }
        )
    return {
        "identical": not offenders,
        "cells_checked": len(rows),
        "ungated_writes": offenders,
        "verdict": (
            "WRITEBACK_GATED" if not offenders else "LEAKAGE_SUSPECTED"
        ),
    }
