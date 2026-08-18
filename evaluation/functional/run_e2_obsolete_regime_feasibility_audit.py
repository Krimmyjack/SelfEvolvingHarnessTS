"""Audit whether exposed Source data contains an obsolete-regime geometry.

This zero-fit W38 audit consumes the already exposed, outcome-free Program binding
from the query-context cohort-reweighting experiment.  It asks a narrower question
than the failed rank-weighting Program: whether one contiguous prefix of training
anchors is coherently farther from the visible evaluation-context centroid than the
remaining suffix.  Such a boundary is the minimum geometry required by a future
``exclude obsolete regime`` Program.

No Consumer is fit, no utility outcome is used, and no Target/UCI data is read.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-obsolete-regime-feasibility-audit/1"
SOURCE_REPORT_PATH = (
    "artifacts/functional/e2/source_query_context_cohort_reweighting_report.json"
)
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_obsolete_regime_feasibility_audit_report.json"
)
EXPECTED_ANCHORS = (240, 300, 360, 420, 480, 540)
MIN_ANCHORS_PER_SIDE = 2
MIN_RELATIVE_DISTANCE_IMPROVEMENT = 0.20
MIN_SERIES_SUPPORT_FRACTION = 0.75
REQUIRED_ELIGIBLE_DATASETS = 2


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _candidate_boundaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_series: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        uid = str(row["series_uid"])
        anchor = int(row["anchor"])
        distance = float(row["distance"])
        if anchor not in EXPECTED_ANCHORS or distance < 0.0:
            raise ValueError("invalid exposed train-window distance")
        if anchor in by_series[uid]:
            raise ValueError("duplicate series/anchor distance")
        by_series[uid][anchor] = distance
    if len(by_series) != 12:
        raise ValueError("expected twelve exposed training series")
    if any(tuple(sorted(values)) != EXPECTED_ANCHORS for values in by_series.values()):
        raise ValueError("training anchor geometry changed")

    candidates: list[dict[str, Any]] = []
    for split_index in range(
        MIN_ANCHORS_PER_SIDE, len(EXPECTED_ANCHORS) - MIN_ANCHORS_PER_SIDE + 1
    ):
        old_anchors = EXPECTED_ANCHORS[:split_index]
        recent_anchors = EXPECTED_ANCHORS[split_index:]
        if len(recent_anchors) < MIN_ANCHORS_PER_SIDE:
            continue
        old_values = [
            distance
            for values in by_series.values()
            for anchor, distance in values.items()
            if anchor in old_anchors
        ]
        recent_values = [
            distance
            for values in by_series.values()
            for anchor, distance in values.items()
            if anchor in recent_anchors
        ]
        old_mean = statistics.fmean(old_values)
        recent_mean = statistics.fmean(recent_values)
        relative_improvement = (
            (old_mean - recent_mean) / old_mean if old_mean > 0.0 else 0.0
        )
        per_series = []
        for uid, values in sorted(by_series.items()):
            series_old = statistics.fmean(values[anchor] for anchor in old_anchors)
            series_recent = statistics.fmean(
                values[anchor] for anchor in recent_anchors
            )
            per_series.append(
                {
                    "series_uid": uid,
                    "old_mean_distance": series_old,
                    "recent_mean_distance": series_recent,
                    "old_is_farther": series_old > series_recent,
                }
            )
        support_fraction = statistics.fmean(
            float(row["old_is_farther"]) for row in per_series
        )
        adjacent_jump = statistics.fmean(
            values[old_anchors[-1]] - values[recent_anchors[0]]
            for values in by_series.values()
        )
        eligible = (
            relative_improvement >= MIN_RELATIVE_DISTANCE_IMPROVEMENT
            and support_fraction >= MIN_SERIES_SUPPORT_FRACTION
            and adjacent_jump > 0.0
        )
        candidates.append(
            {
                "boundary_after_anchor": old_anchors[-1],
                "old_anchors": list(old_anchors),
                "recent_anchors": list(recent_anchors),
                "old_training_row_count": len(old_values),
                "recent_training_row_count": len(recent_values),
                "old_mean_query_context_distance": old_mean,
                "recent_mean_query_context_distance": recent_mean,
                "relative_distance_improvement": relative_improvement,
                "series_support_fraction": support_fraction,
                "adjacent_boundary_distance_drop": adjacent_jump,
                "eligible_obsolete_regime_geometry": eligible,
                "per_series": per_series,
            }
        )
    if len(candidates) != 3:
        raise ValueError("expected exactly three legal anchor boundaries")
    return candidates


def run(root: Path) -> dict[str, Any]:
    source = _read_object(root / SOURCE_REPORT_PATH)
    if source.get("target_query_opened") is not False:
        raise ValueError("source report did not preserve the Target boundary")
    if int(source.get("consumer_fit_count", -1)) != 8:
        raise ValueError("source report identity changed")
    datasets = source.get("dataset_evidence")
    if not isinstance(datasets, list) or len(datasets) != 4:
        raise ValueError("expected four exposed Source datasets")

    evidence: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_id = str(dataset["dataset_id"])
        if dataset_id.startswith("uci"):
            raise ValueError("UCI is forbidden in W38")
        rows = dataset["program_binding"]["train_window_weights"]
        candidates = _candidate_boundaries(rows)
        eligible = [
            row for row in candidates if row["eligible_obsolete_regime_geometry"]
        ]
        best = max(
            candidates,
            key=lambda row: (
                float(row["relative_distance_improvement"]),
                float(row["series_support_fraction"]),
                -int(row["boundary_after_anchor"]),
            ),
        )
        evidence.append(
            {
                "dataset_id": dataset_id,
                "candidate_boundaries": candidates,
                "eligible_boundary_count": len(eligible),
                "dataset_eligible": bool(eligible),
                "best_context_only_boundary": {
                    key: value for key, value in best.items() if key != "per_series"
                },
            }
        )

    eligible_count = sum(bool(row["dataset_eligible"]) for row in evidence)
    passed = eligible_count >= REQUIRED_ELIGIBLE_DATASETS
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "zero_fit_exposed_obsolete_regime_geometry_audit",
        "causal_hypothesis": (
            "At least two Source datasets contain one contiguous early-anchor prefix "
            "that is coherently farther from the visible evaluation-context centroid "
            "than the later-anchor suffix, making an obsolete-regime Program testable."
        ),
        "configuration": {
            "source_report": SOURCE_REPORT_PATH,
            "anchors": list(EXPECTED_ANCHORS),
            "minimum_anchors_per_side": MIN_ANCHORS_PER_SIDE,
            "minimum_relative_distance_improvement": (
                MIN_RELATIVE_DISTANCE_IMPROVEMENT
            ),
            "minimum_series_support_fraction": MIN_SERIES_SUPPORT_FRACTION,
            "required_eligible_dataset_count": REQUIRED_ELIGIBLE_DATASETS,
            "outcome_or_future_used_for_boundary": False,
            "dataset_identity_used_as_feature": False,
            "consumer_fit_count": 0,
            "target_query_opened": False,
        },
        "dataset_evidence": evidence,
        "overall": {
            "dataset_count": len(evidence),
            "eligible_dataset_count": eligible_count,
            "required_eligible_dataset_count": REQUIRED_ELIGIBLE_DATASETS,
            "feasibility_pass": passed,
        },
        "verdict": (
            "OBSOLETE_REGIME_GEOMETRY_FEASIBILITY_PASS"
            if passed
            else "OBSOLETE_REGIME_GEOMETRY_FEASIBILITY_FAIL"
        ),
        "next_step_if_pass": (
            "Run one exposed oracle headroom slice for excluding the context-bound old "
            "anchor prefix; do not design Witness or Memory yet."
        ),
        "next_step_if_fail": (
            "Close the obsolete-regime family under the current Source roster and "
            "fingerprint; do not fit a Consumer or add regime features."
        ),
        "consumer_fit_count": 0,
        "capability_claim": False,
        "memory_claim": False,
        "formal_transfer": False,
        "target_query_opened": False,
        "claim_limit": (
            "Exposed Source Context-only feasibility audit. PASS only establishes "
            "Program geometry; it is not headroom or Capability evidence."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(root)
    output = args.output or root / DEFAULT_REPORT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
