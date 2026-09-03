"""Localize the exposed E1-TR source/target overlay failure without new judging.

This diagnostic replays only already-exposed E1-P/E1-T/E1-TR evidence.  It asks
whether target-local evidence that overrides a source Skill is actually closer
to the current Query Context, and whether that override improves the already
measured query loss.  It does not propose a new threshold or make a transfer
claim.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

SCHEMA_VERSION = "e1tr-overlay-failure-diagnostic/1"
NEAREST_K = 5
GLOBAL_FEATURE_NAMES = (
    "missing_fraction",
    "longest_missing_run_fraction",
    "local_robust_z_peak",
    "estimated_region_start_fraction",
    "estimated_region_end_fraction",
    "level_excursion_score",
    "estimated_level_offset",
    "period_change_score",
    "period_reliability",
    "pre_period",
    "post_period",
    "acf_spectral_consistency",
)
LOCAL_FEATURE_NAMES = (
    "observed_cycles",
    "phase_correlation",
    "amplitude_ratio",
    "amplitude_stability",
    "local_period_consistency",
    "boundary_proximity",
)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _squash_feature(name: str, value: float) -> float:
    if name in {
        "missing_fraction",
        "longest_missing_run_fraction",
        "estimated_region_start_fraction",
        "estimated_region_end_fraction",
        "period_reliability",
        "acf_spectral_consistency",
        "amplitude_stability",
        "local_period_consistency",
    }:
        return _clip(value, -1.0, 1.0)
    if name in {"pre_period", "post_period"}:
        return _clip(value / 48.0, 0.0, 1.5)
    if name == "observed_cycles":
        return _clip(value / 8.0, 0.0, 2.0)
    if name == "phase_correlation":
        return _clip(value, -1.0, 1.0)
    if name == "amplitude_ratio":
        return _clip(math.log(max(value, 1e-8)) / 3.0, -2.0, 2.0)
    return _clip(math.asinh(value) / 3.0, -2.0, 2.0)


def _feature_vector(row: Mapping[str, object]) -> tuple[float, ...]:
    global_features = row.get("global_features")
    local_features = row.get("local_features")
    if not isinstance(global_features, Mapping) or not isinstance(
        local_features, Mapping
    ):
        raise TypeError("selection row lacks public feature mappings")
    return tuple(
        [
            _squash_feature(name, float(global_features[name]))
            for name in GLOBAL_FEATURE_NAMES
        ]
        + [
            _squash_feature(name, float(local_features[name]))
            for name in LOCAL_FEATURE_NAMES
        ]
    )


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _nearest_distance_summary(
    evidence: Sequence[Mapping[str, object]], query: Mapping[str, object]
) -> dict[str, object]:
    query_vector = _feature_vector(query)
    ranked = sorted(
        (
            _distance(_feature_vector(row), query_vector),
            row,
        )
        for row in evidence
    )
    selected = ranked[: min(NEAREST_K, len(ranked))]
    selected_distances = [distance for distance, _ in selected]
    neighbor_rows: list[dict[str, object]] = []
    for distance, row in selected:
        arms = row.get("arms")
        action_gain: dict[str, float] = {}
        if isinstance(arms, Mapping) and isinstance(arms.get("identity"), Mapping):
            identity_loss = float(arms["identity"]["loss_j"])
            for action in ("linear", "seasonal"):
                arm = arms.get(action)
                if isinstance(arm, Mapping):
                    action_gain[action] = identity_loss - float(arm["loss_j"])
        neighbor_rows.append(
            {
                "case_id": row.get("case_id"),
                "distance": distance,
                "source_archetype_report_only": row.get("archetype"),
                "target_family_report_only": row.get("target_family_report_only"),
                "target_variant_report_only": row.get("target_variant_report_only"),
                "grader_winner": row.get("grader_winner"),
                "action_gain_vs_identity": action_gain,
            }
        )
    return {
        "available_evidence_count": len(ranked),
        "nearest_k": len(selected_distances),
        "minimum_distance": min(selected_distances),
        "mean_nearest_distance": sum(selected_distances) / len(selected_distances),
        "maximum_nearest_distance": max(selected_distances),
        "neighbors": neighbor_rows,
    }


def _action_loss(query_row: Mapping[str, object], action: str) -> float:
    arms = query_row.get("arms")
    if not isinstance(arms, Mapping) or action not in arms:
        raise KeyError(f"query row has no judged action {action!r}")
    arm = arms[action]
    if not isinstance(arm, Mapping):
        raise TypeError(f"query action {action!r} is not a mapping")
    return float(arm["loss_j"])


def run(
    *, source_report: Path, target_support_report: Path, repair_report: Path
) -> dict[str, object]:
    source_payload = json.loads(source_report.read_text(encoding="utf-8"))
    target_payload = json.loads(target_support_report.read_text(encoding="utf-8"))
    source_evidence = source_payload.get("cases")
    target_evidence = target_payload.get("support_cases")
    if not isinstance(source_evidence, list) or not isinstance(
        target_evidence, list
    ):
        raise TypeError("source or target support report lacks evidence rows")

    repair = json.loads(repair_report.read_text(encoding="utf-8"))
    query_rows = repair.get("query_cases")
    plan_by_budget = repair.get("query_action_plans_and_guard_receipts")
    if not isinstance(query_rows, list) or not isinstance(plan_by_budget, Mapping):
        raise TypeError("repair report lacks query rows or action plans")
    b8_plans = plan_by_budget.get("8")
    if not isinstance(b8_plans, Mapping):
        raise TypeError("repair report lacks B=8 action plans")

    cases: list[dict[str, object]] = []
    for raw_query in query_rows:
        if not isinstance(raw_query, Mapping):
            raise TypeError("query row is not a mapping")
        case_id = str(raw_query["case_id"])
        raw_plan = b8_plans.get(case_id)
        if not isinstance(raw_plan, Mapping):
            raise KeyError(f"missing B=8 plan for {case_id}")
        actions = raw_plan.get("actions")
        target_guard = raw_plan.get("target_guard")
        if not isinstance(actions, Mapping) or not isinstance(target_guard, Mapping):
            raise TypeError(f"incomplete B=8 plan for {case_id}")

        query = raw_query
        source_distance = _nearest_distance_summary(source_evidence, query)
        target_distance = _nearest_distance_summary(target_evidence, query)
        source_action = str(actions["a4_source_only_guarded"])
        overlay_action = str(actions["a5_guarded_overlay"])
        union_action = str(actions["a5_naive_union_ablation"])
        target_action = str(actions["a3_target_only_guarded"])
        source_loss = _action_loss(raw_query, source_action)
        overlay_loss = _action_loss(raw_query, overlay_action)
        union_loss = _action_loss(raw_query, union_action)

        source_mean = float(source_distance["mean_nearest_distance"])
        target_mean = float(target_distance["mean_nearest_distance"])
        override = overlay_action != source_action
        overlay_gain_vs_source = source_loss - overlay_loss
        cases.append(
            {
                "case_id": case_id,
                "report_only_family": raw_query.get("target_family_report_only"),
                "report_only_variant": raw_query.get("target_variant_report_only"),
                "grader_winner": raw_query.get("grader_winner"),
                "actions": {
                    "source_only": source_action,
                    "target_only": target_action,
                    "overlay": overlay_action,
                    "naive_union": union_action,
                },
                "target_resolution": target_guard.get("resolution"),
                "overlay_resolution_source": raw_plan.get(
                    "a5_overlay_resolution_source"
                ),
                "source_distance": source_distance,
                "target_distance": target_distance,
                "target_to_source_mean_distance_ratio": target_mean
                / max(source_mean, 1e-12),
                "overlay_changed_source_action": override,
                "overlay_gain_vs_source": overlay_gain_vs_source,
                "overlay_improved_source": override and overlay_gain_vs_source > 0.0,
                "overlay_harmed_source": override and overlay_gain_vs_source < 0.0,
                "union_gain_vs_overlay": overlay_loss - union_loss,
            }
        )

    overrides = [row for row in cases if row["overlay_changed_source_action"]]
    harmful = [row for row in overrides if row["overlay_harmed_source"]]
    beneficial = [row for row in overrides if row["overlay_improved_source"]]
    target_closer = [
        row
        for row in overrides
        if float(row["target_to_source_mean_distance_ratio"]) < 1.0
    ]
    target_farther = [
        row
        for row in overrides
        if float(row["target_to_source_mean_distance_ratio"]) >= 1.0
    ]
    union_differs = [
        row
        for row in cases
        if row["actions"]["naive_union"] != row["actions"]["overlay"]
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "zero_new_judge_exposed_first_fault_localization",
        "causal_question": (
            "Does unconditional target-over-source verdict precedence revise the "
            "Source Skill using evidence that is not more Context-local, thereby "
            "causing the observed E1-TR final-budget regression?"
        ),
        "claim_limit": (
            "All Query outcomes were previously exposed in E1-TR. This report may "
            "localize a Harness Update fault only; it is not fresh transfer evidence."
        ),
        "consumer_fit_count": 0,
        "target_query_opened_now": False,
        "configuration": {
            "nearest_k": NEAREST_K,
            "distance_features": (
                "frozen E1 public global/local Context vector; phase bridge omitted "
                "because the persisted source/support reports do not store it"
            ),
            "decision_threshold_tuned": False,
        },
        "overall": {
            "query_case_count": len(cases),
            "source_action_override_count": len(overrides),
            "beneficial_override_count": len(beneficial),
            "harmful_override_count": len(harmful),
            "target_closer_override_count": len(target_closer),
            "target_farther_or_equal_override_count": len(target_farther),
            "naive_union_differs_from_overlay_count": len(union_differs),
            "total_overlay_gain_vs_source": float(
                sum(float(row["overlay_gain_vs_source"]) for row in overrides)
            ),
            "total_union_gain_vs_overlay": float(
                sum(float(row["union_gain_vs_overlay"]) for row in cases)
            ),
        },
        "cases": cases,
        "next_step": (
            "If harmful overrides are not supported by more local target evidence, "
            "test one locality-qualified target revision rule on a fresh controlled "
            "Query. Otherwise the first fault remains Context representation."
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-report",
        type=Path,
        default=root / "artifacts/functional/e1p/periodic_missing_report.json",
    )
    parser.add_argument(
        "--target-support-report",
        type=Path,
        default=root / "artifacts/functional/e1t/source_target_transfer_report.json",
    )
    parser.add_argument(
        "--repair-report",
        type=Path,
        default=root / "artifacts/functional/e1tr/guarded_transfer_report.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "artifacts/functional/e1tr/overlay_failure_diagnostic_report.json",
    )
    args = parser.parse_args()
    payload = run(
        source_report=args.source_report,
        target_support_report=args.target_support_report,
        repair_report=args.repair_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
