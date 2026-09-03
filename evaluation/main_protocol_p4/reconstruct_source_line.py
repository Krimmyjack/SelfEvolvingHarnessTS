"""Rebuild p4w from the audits that read it before a dry run destroyed it.

A ``--dry-run`` of ``run_source_line`` wrote its stub to the live artifact path
and overwrote the completed Source line.  ``.p4w_source_store`` is empty and the
artifact was never committed, so the file itself is unrecoverable.

Two audits had already read it and are unaffected: ``p4x_admission_regime``
carries every probe's aggregate gain, risk profile, treated/served counts and
admission verdict, and ``p4y_oracle_scope_bound`` carries the program, the
Scope predicate and the harmed counts for the seven risk-refused probes.  This
module joins them back into the p4w shape so downstream readers keep working,
and states plainly what no longer exists.

What is gone, and is not guessed here: the per-series gain vectors, the resolved
UID sets, the program step parameters, the episode ids, the snapshot shas, the
delayed events and the runtime status.  Their absence is why
``audit_oracle_scope_bound`` cannot be re-run against this file -- the oracle
bound it produced stands as a result, but its input no longer exists.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from evaluation.main_protocol_p4 import main_experiment_contract as contract
from evaluation.main_protocol_p4 import p4b_contract as bounded

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REGIME = PROJECT_ROOT / "artifacts/main_protocol/p4x_admission_regime.json"
ORACLE = PROJECT_ROOT / "artifacts/main_protocol/p4y_oracle_scope_bound.json"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4w_source_line.json"

LOST_FIELDS = (
    "probes[].per_series_gain",
    "probes[].resolved_serving_series",
    "probes[].program_steps[].params",
    "probes[].episode_id",
    "rounds[].snapshot_sha",
    "rounds[].delayed_event",
    "rounds[].delayed_utility",
    "rounds[].retrieved_skill_ids",
    "runtime_status",
    "wall_seconds",
)


def build() -> dict[str, Any]:
    regime = json.loads(REGIME.read_text(encoding="utf-8"))
    oracle = json.loads(ORACLE.read_text(encoding="utf-8"))
    by_key = {
        (int(entry["origin"]), str(entry["candidate_id"])): entry
        for entry in oracle["probes"]
    }

    rounds: dict[int, dict[str, Any]] = {}
    for row in regime["live_scoped_probes_on_source"]:
        origin = int(row["origin"])
        oracle_row = by_key.get((origin, str(row["policy"])))
        probe = {
            "candidate_id": row["policy"],
            "kind": "probe",
            "gain": row["aggregate_gain"],
            "passed": True,
            "program_steps": (
                [{"op": op} for op in oracle_row["program"]]
                if oracle_row else None),
            "serving_scope": oracle_row["original_scope"] if oracle_row else None,
            "resolved_serving_series": None,
            "per_series_gain": None,
            "risk_profile": {
                "series_read": row["series_served"],
                "harmed_count": (
                    oracle_row["as_deployed"]["harmed_count"]
                    if oracle_row else None),
                "harmed_fraction": row["harmed_fraction"],
                "max_single_series_harm": row["max_single_series_harm"],
            },
            "source_skill_id": None,
            "admission": {
                "admitted": row["admitted"],
                "rule": bounded.BOUNDED_POLICY.rule,
                "reason": row["refusal_reason"],
                "aggregate_gain": row["aggregate_gain"],
                "series_count": row["series_served"],
                "harmed_fraction": row["harmed_fraction"],
                "max_single_series_harm": row["max_single_series_harm"],
            },
        }
        entry = rounds.setdefault(origin, {
            "origin": origin,
            "served_count": row["series_served"],
            "winner_program": None,
            "winner_serving_scope": None,
            "winner_resolved_count": None,
            "probes": [],
            "harm_count": 0,
            "risk_refusal_count": 0,
            "approved_skill_id": None,
        })
        entry["probes"].append(probe)
        if row["series_treated"] is not None:
            probe["resolved_serving_series_count"] = row["series_treated"]

    for entry in rounds.values():
        entry["risk_refusal_count"] = sum(
            1 for probe in entry["probes"]
            if not probe["admission"]["admitted"]
            and probe["admission"]["reason"] in (
                "harmed_fraction_over_budget", "single_series_harm_over_budget"))

    return {
        "stage": "P4W_SOURCE_LINE",
        "status": "COMPLETE",
        "provenance": "RECONSTRUCTED_FROM_P4X_AND_P4Y_AFTER_A_DRY_RUN_OVERWROTE_IT",
        "written_at": datetime.now().astimezone().isoformat(),
        "how_it_was_lost": (
            "run_source_line --dry-run wrote its stub to this path and "
            "destroyed the completed run; the store was empty and the file was "
            "never committed.  The runner now writes dry runs to a separate "
            "path and refuses to overwrite an existing live artifact"
        ),
        "fields_that_no_longer_exist": list(LOST_FIELDS),
        "not_reconstructed_by_inference": (
            "every value below was read out of p4x or p4y, both of which were "
            "derived from the live run before the overwrite.  Nothing here was "
            "recomputed, estimated or filled in"
        ),
        "consequence": (
            "audit_oracle_scope_bound cannot be re-run against this file: its "
            "input was the per-series gain vectors.  Its verdict "
            "(FEASIBLE_SCOPE_EXISTS) stands as a recorded result, but is no "
            "longer reproducible from this artifact"
        ),
        "data_version": contract.DATA_VERSION,
        "cohort": "source readable[%d:%d]" % contract.SOURCE_SLICE,
        "face": "support_a",
        "origins": list(contract.HELD_IN_ORIGINS),
        "admission_policy_in_force": bounded.BOUNDED_POLICY.rule,
        "admission_policy": {
            "rule": bounded.BOUNDED_POLICY.rule,
            "max_harmed_fraction": bounded.BOUNDED_MAX_HARMED_FRACTION,
            "max_single_series_harm": bounded.BOUNDED_MAX_SINGLE_SERIES_HARM,
            "thresholds_changed": 0,
        },
        "rounds": [rounds[origin] for origin in sorted(rounds)],
        "approved_skill_ids": [],
        "source_skills_for_a5": [],
        "verdict": "A5_TREATMENT_EMPTY",
        "stopping_rule_reading": contract.STOPPING_RULES["A5_TREATMENT_EMPTY"],
        "boundary": {**contract.BOUNDARY, "held_out_reads": 0},
        "sources": [REGIME.name, ORACLE.name],
    }


def main() -> int:
    report = build()
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    total = sum(len(entry["probes"]) for entry in report["rounds"])
    print("rebuilt %d rounds, %d probes, %d risk refusals" % (
        len(report["rounds"]), total,
        sum(entry["risk_refusal_count"] for entry in report["rounds"])))
    print("lost and not guessed: %s" % ", ".join(LOST_FIELDS[:4]))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
