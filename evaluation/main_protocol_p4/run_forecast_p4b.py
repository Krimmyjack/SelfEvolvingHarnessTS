"""P4b runner: bounded-risk gate, held-in adaptation, held-out endpoint.

Phase order is the point of the design and is enforced here rather than
documented: every arm adapts on the held-in block under its own admission rule,
freezes, and only then is read once on a held-out block it never saw feedback
from.  The comparator freezes a program on held-in for the same reason.

Prospective risk-utility policy experiment.  It does not supersede
``p4_forecast_performance_b8_llm8_run2_20260830.json``; the strict-gate H1/H2/H3
results stand as collected.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import p4b_heldin as heldin
from evaluation.main_protocol_p4 import p4b_heldout as heldout
from evaluation.main_protocol_p4 import p4b_parallel as parallel
from evaluation.main_protocol_p4 import p4b_stats as stats
from evaluation.main_protocol_p4 import p4b_viability as viability
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability
from evaluation.functional.task_episode_harness.agentic.runner import live_transport

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4b_bounded_risk_forecast.json"
PREFLIGHT = PROJECT_ROOT / "artifacts/main_protocol/p4b_preflight.json"
# The live-backend smoke writes here, never to REPORT: it runs on the already
# exposed old P4 origins so it cannot spend any of the frozen blocks' novelty,
# and for the same reason its numbers are not a result.
SMOKE_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4b_live_gate_smoke.json"
# One shard per replica.  Sharding changes only which cells a process executes;
# the contract, origins, arms, budget, transport and thresholds are read from
# the same frozen module, and the merger refuses shards whose contracts differ.
SHARD = PROJECT_ROOT / "artifacts/main_protocol/p4b_shard_%s.json"


class P4bBlocked(RuntimeError):
    """A frozen contract was about to be broken."""


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def source_scope_census(base_cell: Any, origins: Sequence[int]) -> dict[str, Any]:
    """How often the audited Source card's Scope matches these origins.

    This is the check that should have run before the first P4: a Source card
    installed but never applicable makes the accumulation treatment empty, and
    every arm contrast that was supposed to express accumulation collapses to a
    comparison of a store entry that never fires.  It costs nothing -- public
    features and the same applicability evaluator retrieval uses, no Consumer
    fit, no Outcome, no LLM -- so it belongs in preflight, not in a post-hoc
    diagnosis.
    """
    card, _card_contract = forecast_p1._audited_forecast_supply_card()
    applicability = dict(card.get("observable_applicability") or {})
    matched: list[int] = []
    for origin in origins:
        cell = forecast_p4._cell_at(base_cell, int(origin))
        features = dict(
            extract_public_features(cell.observation_block, task_kind=forecast_p4.TASK)
        )
        if evaluate_applicability(applicability, features)[0]:
            matched.append(int(origin))
    return {
        "skill_id": str(card.get("skill_id")),
        "origins_checked": [int(origin) for origin in origins],
        "matched_origins": matched,
        "match_count": len(matched),
        "minimum_for_an_accumulation_study": contract.SOURCE_SCOPE_MATCH_MINIMUM,
        "accumulation_treatment_active": (
            len(matched) >= contract.SOURCE_SCOPE_MATCH_MINIMUM
        ),
        "consequence_if_zero": contract.NOT_TESTED_HERE,
        "consumer_fits": 0,
        "outcome_reads": 0,
        "llm_calls": 0,
    }


def preflight(base_cell: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the origin plan and refuse to run if the contract does not hold."""
    screen = viability.screen(base_cell)
    plan = contract.resolve_origins(screen)
    shortest = min(int(values.size) for values in base_cell.values.values())
    failures = contract.validate_geometry(plan, minimum_series_length=shortest)
    searched = [
        contract.HELD_IN_SEARCH_START + contract.ORIGIN_GRID_STEP * step
        for step in range(
            (plan["held_out_origins"][-1] - contract.HELD_IN_SEARCH_START)
            // contract.ORIGIN_GRID_STEP
            + 1
        )
    ]
    return {
        "stage": "P4B_PREFLIGHT",
        "status": "PASS" if not failures else "FAIL",
        "dataset": data.get("dataset"),
        "data_role": "EXPOSED_DEVELOPMENT",
        "shortest_series_length": shortest,
        "origin_plan": plan,
        "viability_census": viability.census(base_cell, searched),
        # Recorded, not enforced: this study has already been narrowed to the
        # gate question because the count is zero (contract.NOT_TESTED_HERE).
        # A study that does claim accumulation benefit must see a non-zero
        # count here before it may run.
        "source_scope_census": source_scope_census(
            base_cell,
            list(plan["held_in_origins"]) + list(plan["held_out_origins"])
            + list(contract.OLD_P4_ORIGINS),
        ),
        "geometry_failures": failures,
        "strict_equivalence_receipt": (
            "artifacts/main_protocol/p4b_preflight_strict_equivalence.json"
        ),
        "llm_calls": 0,
    }


def _held_in_replica(
    *,
    arm: contract.Arm,
    replica: str,
    origins: Sequence[int],
    base_cell: Any,
    initial_snapshot: Any,
    backend: Any,
    temp_root: Path,
    spec: Any,
    context: Any,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Walk one arm through one replica's held-in origins, then freeze it.

    The admission rule is installed for the whole walk and always removed
    afterwards, so an arm can never inherit the previous arm's gate.
    """
    snapshot = initial_snapshot
    episodes: tuple[Any, ...] = ()
    ledger = heldin.empty_ledger()
    opening_semantics: tuple[Any, ...] | None = None
    state: dict[str, Any] | None = None

    with contract.admission_scope(arm):
        for index, origin in enumerate(origins, start=1):
            cell = forecast_p4._cell_at(base_cell, int(origin))
            identity, _wall = forecast_p4._identity_reference(cell, int(origin))
            tag = "%s_%s_h%d" % (
                arm.name.lower().replace("-", "_"), replica.lower(), index
            )
            arm_backend = backend.new_arm_backend(
                scope_id="%s/%s/%s" % (replica, "H%d" % index, arm.name),
                maximum_calls=contract.MAX_LLM_CALLS,
            )
            state = heldin.new_state(
                snapshot=snapshot,
                cell=cell,
                backend=arm_backend,
                store_root=temp_root,
                tag=tag,
                episodes=episodes,
                ledger=ledger,
            )
            if opening_semantics is None:
                opening_semantics = heldout.store_semantics(state["method"])
            row = heldin.run_cell(
                arm=arm,
                replica=replica,
                origin=int(origin),
                sequence_index=index,
                state=state,
                base_cell=base_cell,
                identity=identity,
                spec=spec,
                context=context,
            )
            rows.append(row)
            ledger = heldin.read_ledger(state)
            if arm.carries_state:
                snapshot = state["method"]._active_snapshot()
                episodes = tuple(state["method"].experience_episodes)
            # An arm that did not carry state would restart from the same
            # audited snapshot every origin; both arms in the current table do
            # carry it, so the branch above always runs.

    if state is None:
        raise P4bBlocked("a held-in replica produced no cell")
    closing_semantics = heldout.store_semantics(state["method"])
    return {
        "arm": arm.name,
        "replica": replica,
        "frozen_snapshot": snapshot if arm.carries_state else initial_snapshot,
        "frozen_episodes": episodes if arm.carries_state else (),
        "ledger": ledger,
        "opening_store_semantics": opening_semantics,
        "closing_store_semantics": closing_semantics,
    }


def _held_out_for(
    *,
    frozen: Mapping[str, Any],
    origins: Sequence[int],
    base_cell: Any,
    temp_root: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Score one frozen arm on every held-out origin.  No feedback, no writes.

    No backend reaches this function: the frozen state is built with a stub
    agent that raises if anything tries to call it, so the 0-LLM property of the
    endpoint face is enforced rather than trusted.
    """
    arm = contract.ARMS_BY_NAME[str(frozen["arm"])]
    for index, origin in enumerate(origins, start=1):
        state = heldout.frozen_state(
            snapshot=frozen["frozen_snapshot"],
            episodes=frozen["frozen_episodes"],
            ledger=frozen["ledger"],
            store_root=temp_root,
            tag="heldout_%s_%s_o%d" % (
                arm.name.lower().replace("-", "_"), frozen["replica"].lower(), index
            ),
        )
        rows.append(
            heldout.row(
                arm=arm.name,
                replica=str(frozen["replica"]),
                origin=int(origin),
                state=state,
                base_cell=base_cell,
            )
        )


def _analysis(
    held_in_rows: Sequence[Mapping[str, Any]],
    held_out_rows: Sequence[Mapping[str, Any]],
    *,
    writeback: Mapping[str, Any],
    parallel_selection_face: str,
) -> dict[str, Any]:
    """Origin-level contrasts, the risk readouts, and the frozen verdict."""
    primary = stats.contrast(
        held_out_rows, left="A5-bounded", right="A5-strict",
        field="delta_utility_vs_identity",
    )
    risk = {
        field: stats.contrast(
            held_out_rows, left="A5-bounded", right="A5-strict", field=field
        )
        for field in ("harmed_fraction", "worst_single_series_harm")
    }
    bounded_rows = [
        row for row in held_out_rows if row.get("arm") == "A5-bounded"
    ]
    worst = max(
        (float(row["worst_single_series_harm"]) for row in bounded_rows),
        default=None,
    )
    harm_rate = (
        sum(1 for row in bounded_rows if row.get("material_harm_event"))
        / len(bounded_rows)
    ) if bounded_rows else None
    admitted = any(
        (probe.get("admission") or {}).get("admitted")
        for row in held_in_rows
        for probe in row.get("probes") or ()
    )
    reused = any(
        episode.get("source_skill_id")
        for row in held_in_rows
        if str(row.get("arm")) == "A5-bounded"
        for episode in row.get("episodes_written") or ()
    )
    # Support-A admission is provisional.  What counts as a deployable outcome
    # is a Skill that both faces approved and that was actually activated.
    active_skills = sum(1 for row in held_in_rows if row.get("activated"))
    return {
        "primary_utility": primary,
        # No arm contrast here can express cross-domain accumulation: the Source
        # card is inapplicable at every origin, so the accumulation treatment is
        # empty (contract.NOT_TESTED_HERE).  What remains is the comparison
        # against the two deterministic references.
        "secondary_cross_arm": {
            name: stats.contrast(
                held_out_rows, left="A5-bounded", right=name,
                field="delta_utility_vs_identity",
            )
            for name in ("Static", parallel.NAME)
        },
        "accumulation_contrast": {
            "reported": False,
            "reason": contract.NOT_TESTED_HERE,
        },
        "risk_contrasts": risk,
        "bounded_held_out_worst_single_series_harm": worst,
        "bounded_held_out_material_harm_rate": harm_rate,
        "any_admission_held_in": admitted,
        "support_a_admissions": sum(
            1
            for row in held_in_rows
            for probe in row.get("probes") or ()
            if (probe.get("admission") or {}).get("admitted")
        ),
        "active_skills_formed": active_skills,
        "causal_reuse_observed": reused,
        "gated_writeback": dict(writeback),
        "power": stats.power_note(len(primary["origins"])),
        "verdict": stats.primary_verdict(
            utility=primary,
            held_out_worst_single_series_harm=worst,
            held_out_harm_rate=harm_rate,
            max_single_series_harm=contract.BOUNDED_MAX_SINGLE_SERIES_HARM,
            max_harmed_fraction=contract.BOUNDED_MAX_HARMED_FRACTION,
            any_admission_held_in=admitted,
            active_skills_formed=active_skills,
            causal_reuse_observed=reused,
            writeback_gated=bool(writeback.get("identical")),
            parallel_selection_face=parallel_selection_face,
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("live", "scripted"), default="scripted")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--origins", type=int, default=None,
        help="use only the first N origins of each block (smoke runs only)",
    )
    parser.add_argument(
        "--replicas", type=int, default=None,
        help="use only the first N replica orders (smoke runs only)",
    )
    parser.add_argument(
        "--replica", default=None,
        help=(
            "execute only this replica's held-in cells and write its shard; "
            "held-out is not opened, because the endpoint may only be read once "
            "the merged held-in result justifies opening it"
        ),
    )
    parser.add_argument(
        "--old-origin-smoke", action="store_true",
        help=(
            "run held-in only, on the already-exposed old P4 origins, and write "
            "to the smoke artifact; the frozen held-in/held-out blocks are not "
            "touched and no result is produced"
        ),
    )
    args = parser.parse_args(argv)

    started = time.time()
    base_cell, _selection, data = forecast_p1._load_exposed_cells()
    check = preflight(base_cell, data)
    _write(PREFLIGHT, check)
    if check["status"] != "PASS":
        raise P4bBlocked("preflight failed: %s" % check["geometry_failures"])
    if args.preflight_only:
        print("preflight PASS -> %s" % PREFLIGHT.relative_to(PROJECT_ROOT).as_posix())
        return 0

    plan = check["origin_plan"]
    if args.old_origin_smoke:
        # Deliberately the exposed block: a live rehearsal of the gate must not
        # consume the freshness of an origin the endpoint depends on.
        held_in_origins = list(contract.OLD_P4_ORIGINS)[: args.origins]
        held_out_origins = []
    else:
        held_in_origins = list(plan["held_in_origins"])[: args.origins]
        held_out_origins = list(plan["held_out_origins"])[: args.origins]
    orders = contract.replica_orders(held_in_origins)
    replicas = list(orders)[: args.replicas]
    if args.replica:
        if args.replica not in orders:
            parser.error("--replica must name one of %s" % sorted(orders))
        replicas = [args.replica]
        # A shard stops after held-in by construction: its frozen arm states
        # live in this process only, and reading the endpoint per shard would
        # open it before the merged result says it should be opened.
        held_out_origins = []
    smoke = bool(
        args.origins or args.replicas or args.old_origin_smoke
        or args.backend != "live"
    )
    if args.old_origin_smoke:
        report_path = SMOKE_REPORT
    elif args.replica:
        report_path = Path(str(SHARD) % args.replica.lower())
    else:
        report_path = REPORT

    payload: dict[str, Any] = {
        "stage": "P4B_BOUNDED_RISK_FORECAST",
        "status": "RUNNING",
        "started_at": datetime.now().astimezone().isoformat(),
        "backend_mode": args.backend,
        # Which relay and model actually answered.  The old P4 ran on a
        # different transport, so a receipt that only says "live" would hide a
        # change of Fast agent between the two studies.
        "transport": (
            {
                key: value
                for key, value in live_transport().items()
                if key in ("base_url", "model", "source")
            }
            if args.backend == "live" else {"backend": "scripted", "model": None}
        ),
        "smoke": smoke,
        "evidence_grade": (
            "SMOKE_NOT_EVIDENCE" if smoke else "PROSPECTIVE_RISK_UTILITY_POLICY"
        ),
        "smoke_kind": (
            "LIVE_GATE_REHEARSAL_ON_EXPOSED_OLD_ORIGINS"
            if args.old_origin_smoke else None
        ),
        "smoke_checks": (
            [
                "a real agent proposes a candidate",
                "the bounded rule admits or refuses it",
                "an admitted winner opens the delayed face",
                "Support-B approval persists a Skill",
                "the persisted Skill is retrievable on the next origin",
            ]
            if args.old_origin_smoke else None
        ),
        "frozen_contract": contract.contract(plan),
        "preflight": check,
        "replicas": replicas,
        "held_in_origins": held_in_origins,
        "held_out_origins": held_out_origins,
        "held_in_rows": [],
        "held_out_rows": [],
        "frozen_arms": [],
        "releases": "NONE",
        "natural_final_outcome_reads": 0,
        "frozen_blocks_touched": not args.old_origin_smoke,
        "shard": (
            {
                "replica": args.replica,
                "phase": "held_in_only",
                "held_out_deferred_to_merge": True,
            }
            if args.replica else None
        ),
    }
    _write(report_path, payload)

    backend = (
        shared_harness._live_backend(contract.GLOBAL_LLM_CALL_CAP)
        if args.backend == "live"
        else shared_harness._scripted_backend(contract.GLOBAL_LLM_CALL_CAP)
    )
    spec, context = forecast_p1._task_contract(
        forecast_p1._eligible_programs(),
        maximum_candidates=forecast_p4.B_MAIN,
    )

    with tempfile.TemporaryDirectory(prefix="forecast_p4b_") as temp_name:
        temp_root = Path(temp_name)
        h0 = forecast_course._h0()
        card, card_contract = forecast_p1._audited_forecast_supply_card()
        shared_initial = forecast_course._install(
            h0, card, store_root=temp_root / "initial", tag="p4b_initial"
        )
        payload["initial_knowledge"] = {
            "source": card_contract,
            "historical_task_skill_ids": [str(card["skill_id"])],
            "both_arms_share_initial_knowledge": True,
            # Installed in both arms and applicable in neither: see
            # preflight["source_scope_census"].  The arms therefore differ only
            # by their admission rule, which is what this study measures.
            "source_scope_match_count": (
                check["source_scope_census"]["match_count"]
            ),
            "accumulation_treatment_active": (
                check["source_scope_census"]["accumulation_treatment_active"]
            ),
        }
        sources = {"h0": h0, "shared_initial": shared_initial}

        frozen_arms: list[dict[str, Any]] = []
        for replica in replicas:
            for arm in contract.ARMS:
                frozen = _held_in_replica(
                    arm=arm,
                    replica=replica,
                    origins=orders[replica][: args.origins] if args.origins
                    else orders[replica],
                    base_cell=base_cell,
                    initial_snapshot=sources[arm.snapshot_source],
                    backend=backend,
                    temp_root=temp_root,
                    spec=spec,
                    context=context,
                    rows=payload["held_in_rows"],
                )
                frozen_arms.append(frozen)
                _write(report_path, payload)

        # Both arms write back, so the guard is not "the store never moved" but
        # "every move was paid for by an admission".
        writeback = heldin.gated_writeback_check(payload["held_in_rows"])
        writeback["per_arm_store_delta"] = [
            {
                "arm": frozen["arm"],
                "replica": frozen["replica"],
                **heldin.store_delta(
                    frozen["opening_store_semantics"],
                    frozen["closing_store_semantics"],
                ),
            }
            for frozen in frozen_arms
        ]

        # The old-origin rehearsal has no held-out block by construction, so
        # the endpoint phase and its comparators simply do not run.
        selection = {"selection_face": "held_in", "skipped": True}
        if held_out_origins:
            for frozen in frozen_arms:
                _held_out_for(
                    frozen=frozen,
                    origins=held_out_origins,
                    base_cell=base_cell,
                    temp_root=temp_root,
                    rows=payload["held_out_rows"],
                )

            for replica in replicas:
                for origin in held_out_origins:
                    payload["held_out_rows"].append(
                        heldout.identity_row(
                            replica=replica, origin=int(origin), base_cell=base_cell
                        )
                    )
            selection = parallel.select_on_held_in(base_cell, held_in_origins)
            for replica in replicas:
                payload["held_out_rows"].extend(
                    parallel.held_out_rows(
                        base_cell=base_cell,
                        selection=selection,
                        held_out_origins=held_out_origins,
                        replica=replica,
                    )
                )
        payload["parallel_selection"] = selection

    payload["frozen_arms"] = [
        {
            "arm": frozen["arm"],
            "replica": frozen["replica"],
            "approved_skill_ids": list(frozen["ledger"]["approved_skill_ids"]),
            "incumbent": frozen["ledger"]["incumbent"],
        }
        for frozen in frozen_arms
    ]
    payload["analysis"] = _analysis(
        payload["held_in_rows"],
        payload["held_out_rows"],
        writeback=writeback,
        parallel_selection_face=str(selection["selection_face"]),
    )
    payload["status"] = "COMPLETE"
    payload["completed_at"] = datetime.now().astimezone().isoformat()
    payload["wall_seconds"] = round(time.time() - started, 3)
    _write(report_path, payload)

    verdict = payload["analysis"]["verdict"]
    print("held-in rows  : %d" % len(payload["held_in_rows"]))
    print("held-out rows : %d" % len(payload["held_out_rows"]))
    print("writeback     : %s" % writeback["verdict"])
    print("verdict       : %s -- %s" % (verdict["verdict"], verdict["reading"]))
    print("wrote %s" % report_path.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
