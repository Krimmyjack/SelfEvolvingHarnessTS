"""Merge P4b held-in shards into one receipt, refusing anything inconsistent.

Sharding is an execution split, not a design change, so the merge has to prove
that: the shards must carry a **field-by-field identical** frozen contract, and
their cells must tile ``(replica, arm, origin)`` exactly once with no gap.  A
shard that ran a different origin plan, a different arm table, a different
budget, or a different admission threshold is refused rather than averaged in.

The merge also refuses to open the endpoint.  Support-A admission is
provisional; only a Skill that both faces approved is deployable.  If no shard
formed one, there is nothing to deploy on held-out and the merged verdict is
``BOUNDED_GATE_STILL_BLOCKING`` with the blocking face named -- not a neutral
reading of a comparison that was never reached.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import p4b_contract as contract
from evaluation.main_protocol_p4 import p4b_heldin as heldin
from evaluation.main_protocol_p4 import run_forecast_p4b as p4b

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MERGED = PROJECT_ROOT / "artifacts/main_protocol/p4b_bounded_risk_forecast_merged.json"

# Contract fields that must match across shards.  Everything a shard could have
# silently changed is here: what was run, on what, under which rule, at what
# budget, and against which model.
CONTRACT_FIELDS = (
    "task", "consumer", "primary_metric", "experiment_label", "question",
    "not_tested_here", "source_treatment", "geometry", "arms",
    "deterministic_references", "bounded_budget", "offline_sensitivity_point",
    "per_cell_budget", "allow_slow", "statistics",
)


class ShardMismatch(RuntimeError):
    """Two shards disagree about what experiment they were running."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def contract_agreement(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Field-by-field contract comparison across shards."""
    disagreements = []
    for field in CONTRACT_FIELDS:
        seen = {
            _canonical((shard.get("frozen_contract") or {}).get(field)): shard["_name"]
            for shard in shards
        }
        if len(seen) > 1:
            disagreements.append(
                {"field": field, "distinct_values": len(seen),
                 "shards": sorted(seen.values())}
            )
    transports = {
        _canonical(shard.get("transport")): shard["_name"] for shard in shards
    }
    return {
        "fields_compared": list(CONTRACT_FIELDS),
        "disagreements": disagreements,
        "identical": not disagreements,
        "transport_distinct_values": len(transports),
        "transports": [json.loads(key) if key != "null" else None
                       for key in transports],
    }


def coverage(rows: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Every (replica, arm, origin) cell exactly once, and none missing."""
    origins = tuple(plan["held_in_origins"])
    expected = {
        (replica, arm.name, int(origin))
        for replica in contract.replica_orders(origins)
        for arm in contract.ARMS
        for origin in origins
    }
    seen: dict[tuple[str, str, int], int] = {}
    for row in rows:
        key = (str(row["replica"]), str(row["arm"]), int(row["origin"]))
        seen[key] = seen.get(key, 0) + 1
    duplicates = sorted(key for key, count in seen.items() if count > 1)
    missing = sorted(expected - set(seen))
    unexpected = sorted(set(seen) - expected)
    return {
        "expected_cells": len(expected),
        "collected_cells": len(rows),
        "distinct_cells": len(seen),
        "duplicates": [list(key) for key in duplicates],
        "missing": [list(key) for key in missing],
        "unexpected": [list(key) for key in unexpected],
        "complete": not duplicates and not missing and not unexpected
        and len(rows) == len(expected),
    }


def admission_ledger(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the two faces actually stopped things, per arm."""
    per_arm: dict[str, dict[str, Any]] = {}
    for row in rows:
        bucket = per_arm.setdefault(
            str(row["arm"]),
            {"cells": 0, "support_a_admitted": 0, "active_skills": 0,
             "incumbent_changes": 0, "support_b_refusals": {}},
        )
        bucket["cells"] += 1
        admitted = [
            probe for probe in row.get("probes") or ()
            if (probe.get("admission") or {}).get("admitted")
        ]
        bucket["support_a_admitted"] += len(admitted)
        if row.get("activated"):
            bucket["active_skills"] += 1
        if row.get("incumbent_changed"):
            bucket["incumbent_changes"] += 1
        event = row.get("delayed_event") or {}
        if admitted and str(event.get("stage")) == "delayed_rejected":
            reason = str(
                (event.get("delayed_admission") or {}).get("reason") or "unrecorded"
            )
            bucket["support_b_refusals"][reason] = (
                bucket["support_b_refusals"].get(reason, 0) + 1
            )
    return per_arm


def held_out_decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Whether the endpoint may be opened at all.

    With no approved Skill and no incumbent, every arm's frozen recall falls
    through to identity, so the held-out contrast is identically zero by
    construction -- it carries no information and opening it would only spend
    the endpoint's one reading on a foregone conclusion.
    """
    active = sum(1 for row in rows if row.get("activated"))
    incumbents = sum(1 for row in rows if row.get("incumbent_changed"))
    return {
        "active_skills_formed": active,
        "incumbent_changes": incumbents,
        "open_held_out": bool(active or incumbents),
        "reading": (
            "at least one arm froze something deployable; the endpoint phase is "
            "warranted"
            if (active or incumbents) else
            "no arm froze an Active Skill or an incumbent, so every held-out "
            "recall would return identity and the contrast would be zero by "
            "construction; the endpoint stays closed"
        ),
    }


def merge(shard_paths: Sequence[Path]) -> dict[str, Any]:
    shards = []
    for path in shard_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_name"] = path.name
        shards.append(payload)

    agreement = contract_agreement(shards)
    rows = [row for shard in shards for row in shard.get("held_in_rows") or ()]
    plan = shards[0]["frozen_contract"]["geometry"]
    cells = coverage(rows, plan)
    writeback = heldin.gated_writeback_check(rows)
    writeback["per_arm_store_delta"] = [
        entry
        for shard in shards
        for entry in ((shard.get("analysis") or {}).get("gated_writeback") or {}).get(
            "per_arm_store_delta", []
        )
    ]
    endpoint = held_out_decision(rows)
    analysis = p4b._analysis(
        rows, [],  # no held-out rows: the endpoint has not been opened
        writeback=writeback,
        parallel_selection_face="held_in",
    )
    return {
        "stage": "P4B_BOUNDED_RISK_FORECAST_MERGED",
        "status": "COMPLETE" if cells["complete"] and agreement["identical"]
        else "REFUSED",
        "merged_at": datetime.now().astimezone().isoformat(),
        "shards": [
            {
                "file": shard["_name"],
                "replicas": sorted({str(row["replica"])
                                    for row in shard.get("held_in_rows") or ()}),
                "held_in_rows": len(shard.get("held_in_rows") or ()),
                "shard_status": shard.get("status"),
                "transport": shard.get("transport"),
            }
            for shard in shards
        ],
        "contract_agreement": agreement,
        "cell_coverage": cells,
        "frozen_contract": shards[0]["frozen_contract"],
        "preflight": shards[0].get("preflight"),
        "initial_knowledge": shards[0].get("initial_knowledge"),
        "admission_ledger": admission_ledger(rows),
        "held_out_decision": endpoint,
        "held_in_rows": rows,
        "held_out_rows": [],
        "analysis": analysis,
        "evidence_grade": "PROSPECTIVE_RISK_UTILITY_POLICY",
        "does_not_supersede": (
            "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    args = parser.parse_args(argv)

    report = merge(args.shards)
    MERGED.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    cells = report["cell_coverage"]
    print("contract identical : %s" % report["contract_agreement"]["identical"])
    for entry in report["contract_agreement"]["disagreements"]:
        print("   DISAGREE %s across %s" % (entry["field"], entry["shards"]))
    print("cells              : %d / %d (dup %d, missing %d)" % (
        cells["collected_cells"], cells["expected_cells"],
        len(cells["duplicates"]), len(cells["missing"])))
    for arm, entry in sorted(report["admission_ledger"].items()):
        print("  %-11s cells %2d | Support-A admitted %d | Active Skills %d | "
              "B refusals %s" % (
                  arm, entry["cells"], entry["support_a_admitted"],
                  entry["active_skills"], entry["support_b_refusals"] or "-"))
    print("open held-out      : %s" % report["held_out_decision"]["open_held_out"])
    verdict = report["analysis"]["verdict"]
    print("verdict            : %s%s -- %s" % (
        verdict["verdict"],
        " [%s]" % verdict["blocking_face"] if verdict.get("blocking_face") else "",
        verdict["reading"]))
    print("status             : %s" % report["status"])
    print("wrote %s" % MERGED.relative_to(PROJECT_ROOT).as_posix())
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
