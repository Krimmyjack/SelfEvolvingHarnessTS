"""P4b preflight: the strict admission rule must reproduce the old P4 exactly.

The bounded gate is implemented by parameterising a branch that every arm walks
through (``online_loop`` grants deployment rights via
``admission_policy.decide`` instead of an inline ``relation == POSITIVE``
test).  ``A5-strict`` only remains the old policy if that parameterisation is
behaviour-preserving at its default, so this is a zero-tolerance gate: it fails,
nothing launches.

What is compared, and why it is not the artifact's ``relation`` field.  The
``relation`` recorded in ``episodes_written`` is the *final* one -- at origin
696 a probe was POSITIVE on Support-A, earned deployment, and was only
re-classified to CONFLICT/RESTRICTED afterwards by the delayed reading.  The
gate reads the Support-time relation, so that is what gets recomputed here.

0 LLM.  Reads only the exposed KDD development roster and the collected P4
artifact.  Writes one receipt, no SHA, no manifest, releases nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from SelfEvolvingHarnessTS.methods.ttha import admission_policy
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import classify_relation

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
)
REPORT = (
    PROJECT_ROOT / "artifacts/main_protocol/p4b_preflight_strict_equivalence.json"
)


def _probed_units(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        for episode in (row.get("details") or {}).get("episodes_written") or []:
            _head, _sep, tail = str(episode["episode_id"]).partition("_target_")
            operator = tail.rsplit("_", 5)[0] if tail else ""
            key = (int(row["origin"]), operator)
            seen.setdefault(
                key,
                {
                    "origin": int(row["origin"]),
                    "operator": operator,
                    "recorded_final_relation": str(episode["relation"]),
                    "recorded_support_gain": float(episode["support_gain"]),
                    "recorded_local_status": str(episode["local_status"]),
                },
            )
    return sorted(seen.values(), key=lambda unit: (unit["origin"], unit["operator"]))


def _deployed_origins(payload: dict[str, Any]) -> dict[str, list[int]]:
    deployed: dict[str, list[int]] = {}
    for row in payload["rows"]:
        details = row.get("details") or {}
        if details.get("episodes_written") and not details.get("abstained"):
            deployed.setdefault(str(row["method"]), []).append(int(row["origin"]))
    return {arm: sorted(origins) for arm, origins in deployed.items()}


def build_report() -> dict[str, Any]:
    payload = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
    base, _selection, _data = forecast_p1._load_exposed_cells()
    roster = [
        str(row["series_uid"])
        for row in base.roster("support_a")
        if row["role"] == "eval"
    ]

    checks: list[dict[str, Any]] = []
    for unit in _probed_units(payload):
        origin, operator = unit["origin"], unit["operator"]
        cell = forecast_p4._cell_at(base, origin)
        identity = forecast_p4._reading(cell, "support_a", (), origin=origin)
        candidate = forecast_p4._reading(
            cell, "support_a", forecast_p1._steps(operator), origin=origin
        )
        gains = np.asarray(identity["per_series_smase"], float) - np.asarray(
            candidate["per_series_smase"], float
        )
        aggregate = float(identity["smase"] - candidate["smase"])
        support_relation = str(
            classify_relation(
                aggregate_gain=aggregate,
                per_series_gains=dict(zip(roster, (float(g) for g in gains))),
            )["relation"]
        )
        verdict = admission_policy.decide(
            relation=support_relation,
            aggregate_gain=aggregate,
            per_series_gains=[float(g) for g in gains],
            policy=admission_policy.DEFAULT,
        )
        checks.append(
            {
                "origin": origin,
                "operator": operator,
                "recomputed_support_aggregate_gain": aggregate,
                "recorded_support_gain": unit["recorded_support_gain"],
                "aggregate_matches_artifact": bool(
                    abs(aggregate - unit["recorded_support_gain"]) < 5e-6
                ),
                "support_time_relation": support_relation,
                "recorded_final_relation": unit["recorded_final_relation"],
                "strict_admitted": bool(verdict.admitted),
                # The whole point of the default: admission must be exactly the
                # predicate the inline test used before parameterisation.
                "strict_equals_inline_predicate": bool(
                    verdict.admitted == (support_relation == "POSITIVE")
                ),
            }
        )

    admitted = sorted(
        {check["origin"] for check in checks if check["strict_admitted"]}
    )
    deployed = _deployed_origins(payload)
    failures: list[str] = []
    if not all(check["aggregate_matches_artifact"] for check in checks):
        failures.append("recomputed Support-A aggregate differs from the artifact")
    if not all(check["strict_equals_inline_predicate"] for check in checks):
        failures.append("strict admission differs from the inline POSITIVE test")
    for arm, origins in deployed.items():
        if sorted(set(origins)) != admitted:
            failures.append(
                "%s deployed at origins %s but strict admits %s"
                % (arm, sorted(set(origins)), admitted)
            )

    return {
        "stage": "P4B_PREFLIGHT_STRICT_EQUIVALENCE",
        "status": "COMPLETE",
        "evidence_grade": "PREFLIGHT_REGRESSION_GATE",
        "source_run": SOURCE_RUN.relative_to(PROJECT_ROOT).as_posix(),
        "llm_calls": 0,
        "boundary": {
            "natural_final_outcome_reads": 0,
            "query_evaluations": 0,
            "new_sha_added": False,
            "new_manifest_added": False,
            "live_provider_calls": 0,
        },
        "default_policy": admission_policy.DEFAULT.to_dict(),
        "units_checked": len(checks),
        "strict_admitted_origins": admitted,
        "deployed_origins_by_arm": deployed,
        "checks": checks,
        "failures": failures,
        "verdict": (
            "STRICT_EQUIVALENCE_PASS" if not failures else "STRICT_EQUIVALENCE_FAIL"
        ),
        "releases": "NONE",
    }


def main() -> int:
    report = build_report()
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("units checked      : %d" % report["units_checked"])
    print("strict admits at   : %s" % report["strict_admitted_origins"])
    print("deployed by arm    : %s" % report["deployed_origins_by_arm"])
    for failure in report["failures"]:
        print("FAILURE: %s" % failure)
    print("verdict: %s" % report["verdict"])
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0 if not report["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
