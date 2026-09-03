"""Fact correction to P4: the shared Source card was never retrieved.

The old P4 reported ``A5-online - K0-fixed = 0.0`` on all 24 units and read it
as the strict admission gate flattening every arm difference.  That reading was
incomplete.  The card the two arms share -- ``s2a_forecast_supply_v0``, the
audited Source supply card that *is* the accumulated-knowledge treatment --
carries an ``observable_applicability`` that matches no origin in the study.  It
sits in each arm's store and never enters a decision.

So the arms were never architecturally distinguishable on this data, gate or no
gate: an arm difference cannot be carried by a store entry that never fires.
The collected numbers are correct and are not restated here; what changes is
what may be *attributed* to them.

Nothing here re-runs the experiment.  It re-reads the receipt for retrieval
facts and recomputes the card's Scope match with the same evaluator retrieval
uses -- public features only, no Consumer fit, no Outcome, no LLM.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_forecast_p4b as p4b
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OLD_P4 = PROJECT_ROOT / (
    "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
)
REPORT = PROJECT_ROOT / (
    "artifacts/main_protocol/p4_source_treatment_empty_correction_20260831.json"
)


def retrieval_census(payload: dict[str, Any], card_id: str) -> list[dict[str, Any]]:
    """Per arm: how many units held the card, retrieved it, or used it."""
    by_arm: dict[str, dict[str, int]] = {}
    for row in payload.get("rows") or ():
        method = str(row.get("method"))
        details = dict(row.get("details") or {})
        if "state_skill_ids_before" not in details:
            continue  # Static and the deterministic comparators have no store
        bucket = by_arm.setdefault(
            method,
            {"units": 0, "card_in_store": 0, "card_retrieved": 0, "card_used": 0},
        )
        bucket["units"] += 1
        if card_id in (details.get("state_skill_ids_before") or ()):
            bucket["card_in_store"] += 1
        if card_id in (details.get("retrieved_skill_ids") or ()):
            bucket["card_retrieved"] += 1
        if str(details.get("winner_source_skill_id") or "") == card_id:
            bucket["card_used"] += 1
    return [{"arm": arm, **counts} for arm, counts in sorted(by_arm.items())]


def scope_census(card: dict[str, Any]) -> dict[str, Any]:
    """The card's Scope against the old P4 origins, recomputed from features."""
    base_cell, _selection, _data = forecast_p1._load_exposed_cells()
    applicability = dict(card.get("observable_applicability") or {})
    rows = []
    for origin in contract.OLD_P4_ORIGINS:
        cell = forecast_p4._cell_at(base_cell, int(origin))
        features = dict(
            extract_public_features(cell.observation_block, task_kind=forecast_p4.TASK)
        )
        applicable, why = evaluate_applicability(applicability, features)
        rows.append(
            {"origin": int(origin), "applicable": bool(applicable), "why": why}
        )
    return {
        "origins": rows,
        "match_count": sum(1 for row in rows if row["applicable"]),
        "observable_applicability": applicability,
    }


def main() -> int:
    payload = json.loads(OLD_P4.read_text(encoding="utf-8"))
    card, card_contract = forecast_p1._audited_forecast_supply_card()
    card_id = str(card["skill_id"])
    census = retrieval_census(payload, card_id)
    scope = scope_census(card)
    # Also state the match against the P4b blocks, so one receipt covers the
    # whole study rather than only its past.
    base_cell, _selection, _data = forecast_p1._load_exposed_cells()
    plan = json.loads(
        (PROJECT_ROOT / "artifacts/main_protocol/p4b_preflight.json").read_text(
            encoding="utf-8"
        )
    )["origin_plan"] if (
        PROJECT_ROOT / "artifacts/main_protocol/p4b_preflight.json"
    ).exists() else None
    p4b_match = (
        p4b.source_scope_census(
            base_cell,
            list(plan["held_in_origins"]) + list(plan["held_out_origins"]),
        )
        if plan else None
    )

    report = {
        "stage": "P4_SOURCE_TREATMENT_EMPTY_CORRECTION",
        "verdict": "SOURCE_TREATMENT_EMPTY",
        "written_at": datetime.now().astimezone().isoformat(),
        "corrects": OLD_P4.relative_to(PROJECT_ROOT).as_posix(),
        "supersedes_numbers": False,
        "reading": (
            "the audited Source card is present in every A5-online and K0-fixed "
            "unit's store and was retrieved in none of them; its Scope matches "
            "no origin in the study.  The collected H1/H2/H3 numbers stand as "
            "measured, but A5-K0 and A5-A3 may not be attributed to accumulated "
            "historical knowledge: on this data there was no accumulation "
            "treatment to attribute them to."
        ),
        "what_the_old_receipt_claimed": {
            "k0_a5_initial_semantics_equal": (
                payload.get("initial_knowledge", {}).get(
                    "k0_a5_initial_semantics_equal"
                )
            ),
            "correction": (
                "true but vacuous: the shared semantics are a card that never "
                "activates, so equality of starting knowledge implies equality of "
                "effective starting knowledge only in the trivial sense"
            ),
        },
        "card": {
            "skill_id": card_id,
            "contract": card_contract,
            "observable_applicability": scope["observable_applicability"],
        },
        "retrieval_census_old_p4": census,
        "scope_census_old_p4_origins": {
            "match_count": scope["match_count"],
            "origins": scope["origins"],
        },
        "scope_census_p4b_blocks": p4b_match,
        "consequences": [
            "old P4 H3 (A5 - K0) is uninformative about accumulation, "
            "independently of the CONFLICT gate",
            "P4b drops the A3 and K0 arms and does not claim accumulation "
            "benefit; see contract.NOT_TESTED_HERE",
            "a future accumulation study must pass the Scope-match gate: at "
            "least one pre-audited Source Skill applicable at a held-in origin, "
            "and widening an existing card's Scope to pass it is inadmissible",
        ],
        "consumer_fits": 0,
        "outcome_reads": 0,
        "llm_calls": 0,
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    for entry in census:
        print(
            "%-12s units %2d | in store %2d | retrieved %2d | used %2d"
            % (entry["arm"], entry["units"], entry["card_in_store"],
               entry["card_retrieved"], entry["card_used"])
        )
    print("scope match on old P4 origins : %d / %d"
          % (scope["match_count"], len(scope["origins"])))
    if p4b_match:
        print("scope match on P4b blocks     : %d / %d"
              % (p4b_match["match_count"], len(p4b_match["origins_checked"])))
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
