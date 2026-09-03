"""0-LLM calibration of a bounded-risk deployment gate (development only).

The strict gate in ``experience_memory.classify_relation`` grants deployment
rights only when every series is unharmed, and the per-series audit showed that
condition is met by none of the probed Forecast candidates -- so the harness
abstains, A5 collapses onto K0, and no Skill is ever exercised.  The per-series
and deployment-visible audits also closed the alternative of predicting which
series will be harmed, so the remaining move is to bound the harm rather than
forbid it.

The rule shape being calibrated here is:

    Support-A aggregate gain >= +t
    and Support-A local harm within a frozen risk budget (k, m)
    and Support-B aggregate gain >= +t          (independent confirmation)
    and Support-B local harm within the same budget
    -> Target-local deployment allowed

    Support-B sign reversal, or harm outside the budget -> reject / revoke

Support-A carries the budget test; Support-B is an independent face, so a
candidate that only looks safe on the face it was selected on is caught by the
confirmation step rather than by the budget.  Neither face is the deployment
face, so Support-B doubles as the honest estimate of what deployment would
look like -- that is what the sweep below reports as realised outcome.

Discipline.  No LLM.  No Final, Query, UCR TEST or sealed AD read.  No new SHA
or manifest.  This calibrates on the eight already-exposed development origins
of the collected P4 run and freezes nothing by itself: the frozen rule has to
be scored on origins that took no part in this calibration, which is a separate
step this script deliberately does not perform.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
)
REPORT = (
    PROJECT_ROOT
    / "artifacts/main_protocol/p4_bounded_risk_gate_calibration_20260831.json"
)

MATERIAL = 0.005
# Budget grid.  k = how many of the 20 series may be materially harmed;
# m = how much the worst single series may lose.  k=0 reproduces the current
# strict gate and is kept in the sweep as the safety baseline.
K_GRID = (0, 1, 2, 3, 4, 5, 6, 7)
M_GRID = (0.05, 0.10, 0.20, 0.30, 0.50, 0.75, float("inf"))


def _probed_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every unique (origin, operator) the arms actually probed, any relation.

    Calibrating on the CONFLICT units alone would only show what the rule
    admits, never what it rejects, so NEGATIVE probes are carried through too.
    """
    seen: dict[tuple[int, str], dict[str, Any]] = {}
    for row in payload["rows"]:
        for episode in (row.get("details") or {}).get("episodes_written") or []:
            _head, _sep, tail = str(episode["episode_id"]).partition("_target_")
            operator = tail.rsplit("_", 5)[0] if tail else ""
            key = (int(row["origin"]), operator)
            entry = seen.setdefault(
                key,
                {
                    "origin": int(row["origin"]),
                    "operator": operator,
                    "recorded_relation": str(episode["relation"]),
                    "recorded_support_gain": float(episode["support_gain"]),
                    "run_record_count": 0,
                },
            )
            entry["run_record_count"] += 1
    return sorted(seen.values(), key=lambda e: (e["origin"], e["operator"]))


def _face(cell: Any, face: str, operator: str, origin: int) -> dict[str, Any]:
    identity = forecast_p4._reading(cell, face, (), origin=origin)
    candidate = forecast_p4._reading(
        cell, face, forecast_p1._steps(operator), origin=origin
    )
    gains = np.asarray(identity["per_series_smase"], dtype=np.float64) - np.asarray(
        candidate["per_series_smase"], dtype=np.float64
    )
    harmed = gains < -MATERIAL
    return {
        "aggregate_gain": float(identity["smase"] - candidate["smase"]),
        "harmed_count": int(harmed.sum()),
        "harmed_fraction": float(harmed.mean()),
        "max_single_series_harm": float(-gains.min()) if gains.min() < 0 else 0.0,
        "series_count": int(gains.size),
    }


def _admits(candidate: dict[str, Any], k: int, m: float) -> bool:
    a, b = candidate["support_a"], candidate["support_b"]
    return bool(
        a["aggregate_gain"] >= MATERIAL
        and a["harmed_count"] <= k
        and a["max_single_series_harm"] <= m
        and b["aggregate_gain"] >= MATERIAL  # independent confirmation
        and b["harmed_count"] <= k
        and b["max_single_series_harm"] <= m
    )


def build_report() -> dict[str, Any]:
    payload = json.loads(SOURCE_RUN.read_text(encoding="utf-8"))
    base, _selection, data = forecast_p1._load_exposed_cells()

    candidates = []
    for entry in _probed_candidates(payload):
        origin = entry["origin"]
        cell = forecast_p4._cell_at(base, origin)
        candidates.append(
            {
                **entry,
                "support_a": _face(cell, "support_a", entry["operator"], origin),
                "support_b": _face(cell, "support_b", entry["operator"], origin),
            }
        )

    sweep = []
    for k in K_GRID:
        for m in M_GRID:
            admitted = [c for c in candidates if _admits(c, k, m)]
            realised = [c["support_b"] for c in admitted]
            sweep.append(
                {
                    "k_max_harmed_series": k,
                    "m_max_single_series_harm": None if m == float("inf") else m,
                    "admitted": len(admitted),
                    "admitted_of": len(candidates),
                    # Support-B is the face no selection happened on, so its
                    # readings are the honest stand-in for deployed outcome.
                    "realised_mean_aggregate_gain": (
                        float(np.mean([r["aggregate_gain"] for r in realised]))
                        if realised
                        else None
                    ),
                    "realised_min_aggregate_gain": (
                        float(np.min([r["aggregate_gain"] for r in realised]))
                        if realised
                        else None
                    ),
                    "realised_mean_harmed_fraction": (
                        float(np.mean([r["harmed_fraction"] for r in realised]))
                        if realised
                        else None
                    ),
                    "realised_worst_single_series_harm": (
                        float(np.max([r["max_single_series_harm"] for r in realised]))
                        if realised
                        else None
                    ),
                    "admitted_units": [
                        {"origin": c["origin"], "operator": c["operator"]}
                        for c in admitted
                    ],
                }
            )

    strict = next(row for row in sweep if row["k_max_harmed_series"] == 0
                  and row["m_max_single_series_harm"] == 0.05)
    return {
        "stage": "P4_BOUNDED_RISK_GATE_CALIBRATION",
        "status": "COMPLETE",
        "evidence_grade": "DEVELOPMENT_ONLY_CALIBRATION",
        "experiment_label": "PROSPECTIVE_RISK_UTILITY_POLICY_EXPERIMENT",
        "does_not_supersede": (
            "artifacts/main_protocol/p4_forecast_performance_b8_llm8_run2_20260830.json"
            " -- the strict-gate H1/H2/H3 results stand as collected"
        ),
        "source_run": SOURCE_RUN.relative_to(PROJECT_ROOT).as_posix(),
        "dataset": data.get("dataset"),
        "data_role": "EXPOSED_DEVELOPMENT",
        "llm_calls": 0,
        "rule_shape": {
            "admit": [
                "support_a.aggregate_gain >= +0.005",
                "support_a.harmed_count <= k",
                "support_a.max_single_series_harm <= m",
                "support_b.aggregate_gain >= +0.005",
                "support_b.harmed_count <= k",
                "support_b.max_single_series_harm <= m",
            ],
            "reject_or_revoke": [
                "support_b sign reversal",
                "harm outside the frozen budget on either face",
            ],
            "shared_scope": (
                "a positive carrying local conflict does not widen Shared Skill "
                "Scope; it stays Target-local"
            ),
            "safety_baseline_retained": "k=0 reproduces the current strict gate",
        },
        "boundary": {
            "natural_final_outcome_reads": 0,
            "query_evaluations": 0,
            "ucr_test_outcome_reads": 0,
            "sealed_ad_outcome_reads": 0,
            "new_sha_added": False,
            "new_manifest_added": False,
            "live_provider_calls": 0,
        },
        "calibration_origins": sorted({c["origin"] for c in candidates}),
        "candidates": candidates,
        "strict_gate_baseline": strict,
        "sweep": sweep,
        "frozen_rule": None,
        "note": (
            "the frozen (k, m) has to be scored on origins that took no part in "
            "this calibration; this script performs no such scoring and freezes "
            "nothing"
        ),
        "releases": "NONE",
    }


def main() -> int:
    report = build_report()
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("candidates = %d over origins %s"
          % (len(report["candidates"]), report["calibration_origins"]))
    header = "%3s %8s %6s %9s %9s %9s %9s" % (
        "k", "m", "admit", "meanB", "minB", "harmB", "worstB")
    print(header)
    print("-" * len(header))
    for row in report["sweep"]:
        m = row["m_max_single_series_harm"]
        def fmt(value: float | None, spec: str = "%9.4f") -> str:
            return "%9s" % "-" if value is None else spec % value
        print("%3d %8s %6d %s %s %s %s" % (
            row["k_max_harmed_series"],
            "inf" if m is None else "%.2f" % m,
            row["admitted"],
            fmt(row["realised_mean_aggregate_gain"]),
            fmt(row["realised_min_aggregate_gain"]),
            fmt(row["realised_mean_harmed_fraction"]),
            fmt(row["realised_worst_single_series_harm"]),
        ))
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
