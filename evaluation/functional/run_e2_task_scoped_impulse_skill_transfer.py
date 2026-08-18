"""Run one controlled cross-base transfer of a task-scoped impulse Skill.

The frozen W45 Traffic/FRED evidence is compiled into a minimal dataset-agnostic
Skill.  On historically exposed but family-held-out NN5/METR natural backgrounds,
a cohort-level observer localizes class-conditioned impulse topology using only
training inputs and labels.  The Skill executes bound repair for global/coarse
label evidence and abstains for local-event evidence with event-erasure risk.

This is controlled cross-base mechanism evidence, not paper-fresh promotion or
formal A5-vs-A3 transfer evidence.  UCI and Target Query remain closed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-task-scoped-impulse-skill-transfer/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_task_scoped_impulse_skill_transfer_report.json"
)
SOURCE_AUDIT_PATH = (
    "artifacts/functional/e2/source_task_conditioned_tie_safe_readout_audit_report.json"
)
TRANSFER_BASES = ("legacy_monash:nn5_daily", "metr_la")
OBSERVER_MEDIAN_WIDTH = 9
OBSERVER_THRESHOLD_FRACTION = 0.25
OBSERVER_MIN_NODE_DISTANCE = 6


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rolling_median(np: Any, values: Any) -> Any:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or OBSERVER_MEDIAN_WIDTH % 2 != 1:
        raise ValueError("observer requires a one-dimensional odd-width median")
    radius = OBSERVER_MEDIAN_WIDTH // 2
    padded = np.pad(array, (radius, radius), mode="reflect")
    return np.asarray(
        [np.median(padded[index : index + OBSERVER_MEDIAN_WIDTH]) for index in range(array.size)],
        dtype=np.float64,
    )


def _observe_class_conditioned_impulse_topology(
    np: Any,
    train_inputs: Any,
    train_labels: Any,
) -> dict[str, object]:
    """Locate nodes without Dataset ID or private impulse-position access."""

    matrix = np.asarray(train_inputs, dtype=np.float64)
    labels = np.asarray(train_labels, dtype=np.int64)
    if matrix.ndim != 2 or labels.shape != (matrix.shape[0],):
        raise ValueError("observer requires aligned training inputs and labels")
    if set(int(label) for label in np.unique(labels)) != {0, 1}:
        raise ValueError("observer requires the frozen two-class task")
    class_zero = np.mean(matrix[labels == 0], axis=0)
    class_one = np.mean(matrix[labels == 1], axis=0)
    centroid_delta = class_one - class_zero
    low_frequency_baseline = _rolling_median(np, centroid_delta)
    residual = centroid_delta - low_frequency_baseline
    strength = np.abs(residual)
    maximum = float(np.max(strength))
    threshold = OBSERVER_THRESHOLD_FRACTION * maximum
    candidates = [
        index
        for index in range(strength.size)
        if strength[index] >= threshold
        and strength[index]
        >= float(np.max(strength[max(0, index - 1) : min(strength.size, index + 2)]))
    ]
    ranked = sorted(candidates, key=lambda index: (-float(strength[index]), index))
    selected: list[int] = []
    for index in ranked:
        if all(abs(index - prior) >= OBSERVER_MIN_NODE_DISTANCE for prior in selected):
            selected.append(index)
    nodes = sorted(selected)
    return {
        "observer_id": "class-conditioned-impulse-topology/1",
        "inputs_used": ["complete_training_inputs", "training_labels"],
        "dataset_id_used": False,
        "private_true_positions_used": False,
        "rolling_median_width": OBSERVER_MEDIAN_WIDTH,
        "threshold_fraction_of_global_max": OBSERVER_THRESHOLD_FRACTION,
        "minimum_node_distance": OBSERVER_MIN_NODE_DISTANCE,
        "nodes": nodes,
        "node_residual_strength": [float(strength[index]) for index in nodes],
        "global_max_residual_strength": maximum,
        "absolute_threshold": threshold,
        "node_count": len(nodes),
        "temporal_coverage_fraction": len(nodes) / float(matrix.shape[1]),
    }


def _localization_evaluation(
    observed_nodes: list[int], private_true_positions: tuple[int, ...]
) -> dict[str, object]:
    observed = set(int(node) for node in observed_nodes)
    truth = set(int(position) for position in private_true_positions)
    true_positive = len(observed & truth)
    precision = true_positive / float(len(observed)) if observed else 0.0
    recall = true_positive / float(len(truth)) if truth else 0.0
    return {
        "private_evaluator_only": True,
        "true_positions": sorted(truth),
        "true_positive_count": true_positive,
        "false_positive_count": len(observed - truth),
        "false_negative_count": len(truth - observed),
        "precision": precision,
        "recall": recall,
        "exact_precision_recall_pass": precision == 1.0 and recall == 1.0,
    }


def _task_scoped_skill_decision(task_context: dict[str, object]) -> str:
    """Dataset-agnostic execute/abstain decision from explicit task semantics."""

    evidence_scope = task_context.get("label_evidence_scope")
    event_erasure_risk = task_context.get("event_erasure_risk")
    if evidence_scope == "global_coarse" and event_erasure_risk is False:
        return "EXECUTE_BOUND_REPAIR"
    if evidence_scope == "local_event" and event_erasure_risk is True:
        return "ABSTAIN_KEEP_INCUMBENT"
    raise ValueError("task context is outside the compiled source Skill scope")


def run(root: Path) -> dict[str, object]:
    import numpy as np
    from sklearn.neighbors import NearestCentroid

    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
        read_registry_jsonl,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_bound_impulse_oracle import (
        _apply_bound_impulse_oracle,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_impulse_repair_control import (
        COARSE_SIGNAL_AMPLITUDE,
        MAX_EVENT_GAIN,
        MIN_COARSE_GAIN,
        SPIKE_AMPLITUDE,
        SPIKE_POSITIONS,
        WINDOW_LENGTH,
        _classifier_features,
        _normalize_background,
        _paired_world,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_task_conditioned_tie_safe_readout_audit import (
        _fit_nearest_centroid,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_flatline_actionability_credit import (
        _fresh_roster,
    )
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_training_target_reliability_weighting_headroom import (
        ANCHORS,
        FRESH_SPECS,
    )

    source_audit = _read_object(root / SOURCE_AUDIT_PATH)
    if (
        source_audit.get("verdict") != "TIE_SAFE_READOUT_AUDIT_PASS"
        or source_audit.get("protocol_decision")
        != "CONTINUE_FIXED_CLASSIFICATION_PROTOCOL"
        or source_audit.get("target_query_opened") is not False
    ):
        raise ValueError("frozen W45 Source evidence is not eligible for compilation")

    registry_rows = read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    )
    clean_root = root / "data/benchmark_v0_2/clean_base"
    roster: list[dict[str, object]] = []
    roster_audit: dict[str, object] = {}
    for dataset_id in TRANSFER_BASES:
        selected = _fresh_roster(
            np,
            root=root,
            registry_rows=registry_rows,
            dataset_id=dataset_id,
            spec=FRESH_SPECS[dataset_id],
        )
        roster.extend(selected)
        roster_audit[dataset_id] = {
            "selected_train_count": sum(row["cohort"] == "train" for row in selected),
            "selected_eval_count": sum(row["cohort"] == "eval" for row in selected),
            "selection_rule": "existing integrity-only _fresh_roster helper",
            "consumer_or_program_outcome_used": False,
        }
    if (
        {str(row["dataset_id"]) for row in roster} != set(TRANSFER_BASES)
        or len(roster) != 40
        or any(str(row["dataset_id"]).startswith("uci") for row in roster)
    ):
        raise ValueError("expected the controlled NN5/METR 12+8 rosters only")
    records = {row.series_uid: row for row in registry_rows}
    values = _load_values(
        [records[str(row["series_uid"])] for row in roster], clean_root
    )

    task_contexts = {
        "coarse_pattern": {
            "task_type": "classification",
            "label_evidence_scope": "global_coarse",
            "event_erasure_risk": False,
        },
        "event_evidence": {
            "task_type": "classification",
            "label_evidence_scope": "local_event",
            "event_erasure_risk": True,
        },
    }
    fit_count = 0
    dataset_evidence: list[dict[str, object]] = []
    for dataset_id in TRANSFER_BASES:
        train_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "train"
        ]
        eval_rows = [
            row
            for row in roster
            if row["dataset_id"] == dataset_id and row["cohort"] == "eval"
        ]
        if len(train_rows) != 12 or len(eval_rows) != 8:
            raise ValueError(f"controlled roster geometry changed: {dataset_id}")
        train_backgrounds: list[Any] = []
        for anchor in ANCHORS:
            for row in train_rows:
                uid = str(row["series_uid"])
                train_backgrounds.append(
                    _normalize_background(
                        np,
                        values[uid][anchor - WINDOW_LENGTH : anchor],
                        identity=f"{uid}@{anchor}",
                    )
                )
        train_stop = int(FRESH_SPECS[dataset_id]["train_stop"])
        eval_backgrounds = [
            _normalize_background(
                np,
                values[str(row["series_uid"])][
                    train_stop - WINDOW_LENGTH : train_stop
                ],
                identity=f"{row['series_uid']}@{train_stop}",
            )
            for row in eval_rows
        ]
        train_background_array = np.asarray(train_backgrounds, dtype=np.float64)
        eval_background_array = np.asarray(eval_backgrounds, dtype=np.float64)
        if train_background_array.shape != (72, WINDOW_LENGTH):
            raise AssertionError(f"training background geometry changed: {dataset_id}")
        if eval_background_array.shape != (8, WINDOW_LENGTH):
            raise AssertionError(f"evaluation background geometry changed: {dataset_id}")

        world_evidence: dict[str, dict[str, object]] = {}
        for world in ("coarse_pattern", "event_evidence"):
            train_inputs, train_labels = _paired_world(
                np, train_background_array, world=world, training=True
            )
            eval_inputs, eval_labels = _paired_world(
                np, eval_background_array, world=world, training=False
            )
            observation = _observe_class_conditioned_impulse_topology(
                np, train_inputs, train_labels
            )
            localization = _localization_evaluation(
                [int(node) for node in observation["nodes"]], SPIKE_POSITIONS
            )
            observed_nodes = tuple(int(node) for node in observation["nodes"])
            forced_program_inputs, modification = _apply_bound_impulse_oracle(
                np,
                train_inputs,
                positions=observed_nodes,
                window_length=WINDOW_LENGTH,
            )
            incumbent = _fit_nearest_centroid(
                np,
                NearestCentroid,
                _classifier_features,
                train_inputs,
                train_labels,
                eval_inputs,
                eval_labels,
            )
            forced_program = _fit_nearest_centroid(
                np,
                NearestCentroid,
                _classifier_features,
                forced_program_inputs,
                train_labels,
                eval_inputs,
                eval_labels,
            )
            fit_count += 2
            fits_succeeded = bool(
                incumbent["fit_success"] and forced_program["fit_success"]
            )
            incumbent_accuracy = (
                float(incumbent["accuracy"]) if incumbent["accuracy"] is not None else None
            )
            forced_accuracy = (
                float(forced_program["accuracy"])
                if forced_program["accuracy"] is not None
                else None
            )
            forced_gain = (
                forced_accuracy - incumbent_accuracy
                if forced_accuracy is not None and incumbent_accuracy is not None
                else None
            )
            task_context = task_contexts[world]
            policy_decision = _task_scoped_skill_decision(task_context)
            policy_accuracy = (
                forced_accuracy
                if policy_decision == "EXECUTE_BOUND_REPAIR"
                else incumbent_accuracy
            )
            policy_gain = (
                policy_accuracy - incumbent_accuracy
                if policy_accuracy is not None and incumbent_accuracy is not None
                else None
            )
            policy_harm = (
                max(0.0, incumbent_accuracy - policy_accuracy)
                if policy_accuracy is not None and incumbent_accuracy is not None
                else None
            )
            direction_pass = bool(
                fits_succeeded
                and forced_gain is not None
                and (
                    forced_gain >= MIN_COARSE_GAIN
                    if world == "coarse_pattern"
                    else forced_gain <= MAX_EVENT_GAIN
                )
            )
            policy_pass = bool(
                policy_decision
                == (
                    "EXECUTE_BOUND_REPAIR"
                    if world == "coarse_pattern"
                    else "ABSTAIN_KEEP_INCUMBENT"
                )
                and policy_harm == 0.0
            )
            world_evidence[world] = {
                "task_context": task_context,
                "task_context_core_schema_status": (
                    "semantic field available in this controlled runner; minimal formal "
                    "core TaskSpec binding remains future work"
                ),
                "observation": observation,
                "private_localization_evaluation": localization,
                "policy_decision": policy_decision,
                "incumbent_accuracy": incumbent_accuracy,
                "forced_program_accuracy": forced_accuracy,
                "forced_program_gain": forced_gain,
                "policy_accuracy": policy_accuracy,
                "policy_gain": policy_gain,
                "policy_harm": policy_harm,
                "direction_pass": direction_pass,
                "policy_pass": policy_pass,
                "both_fits_succeeded": fits_succeeded,
                "program_modification": modification,
                "evaluation_input_processed": False,
            }

        coarse = world_evidence["coarse_pattern"]
        event = world_evidence["event_evidence"]
        observer_pass = all(
            bool(
                world_evidence[world]["private_localization_evaluation"][
                    "exact_precision_recall_pass"
                ]
            )
            for world in world_evidence
        )
        dataset_pass = bool(
            observer_pass
            and coarse["direction_pass"]
            and event["direction_pass"]
            and coarse["policy_pass"]
            and event["policy_pass"]
        )
        b0_utility = (
            (float(coarse["incumbent_accuracy"]) + float(event["incumbent_accuracy"]))
            / 2.0
            if coarse["incumbent_accuracy"] is not None
            and event["incumbent_accuracy"] is not None
            else None
        )
        source_skill_utility = (
            (float(coarse["policy_accuracy"]) + float(event["policy_accuracy"])) / 2.0
            if coarse["policy_accuracy"] is not None
            and event["policy_accuracy"] is not None
            else None
        )
        utility_improvement = (
            source_skill_utility - b0_utility
            if source_skill_utility is not None and b0_utility is not None
            else None
        )
        dataset_evidence.append(
            {
                "dataset_id": dataset_id,
                "dataset_id_used_by_observer_or_decision": False,
                "historical_exposure_status": (
                    "held_out_from_W42_W45_family_but_historically_exposed_elsewhere"
                ),
                "task_worlds": world_evidence,
                "observer_both_worlds_exact_pass": observer_pass,
                "controlled_comparison": {
                    "A3_like_B0_target_only_no_learned_skill": {
                        "policy": "always_incumbent",
                        "mean_task_utility": b0_utility,
                    },
                    "A4_like_source_compiled_skill": {
                        "policy": "coarse_execute_event_abstain",
                        "mean_task_utility": source_skill_utility,
                    },
                    "utility_improvement": utility_improvement,
                    "event_policy_harm": event["policy_harm"],
                    "A5_vs_A3_claim": False,
                },
                "dataset_gate_pass": dataset_pass,
            }
        )

    expected_fits = len(TRANSFER_BASES) * 2 * 2
    if fit_count != expected_fits:
        raise RuntimeError(f"expected {expected_fits} fits, observed {fit_count}")
    passing_dataset_count = sum(
        bool(row["dataset_gate_pass"]) for row in dataset_evidence
    )
    overall_pass = passing_dataset_count == len(TRANSFER_BASES)
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "controlled_cross_base_task_scoped_skill_transfer",
        "causal_hypothesis": (
            "A dataset-agnostic class-conditioned impulse-topology Observation plus "
            "explicit Task Context can transfer bound repair across natural bases: "
            "execute for global/coarse evidence and abstain under event-erasure risk."
        ),
        "source_skill_compilation": {
            "source_bases": ["monash:traffic_hourly", "legacy_monash:fred_md"],
            "source_evidence_verdict": source_audit["verdict"],
            "skill_scope": {
                "observation": "class-conditioned impulse topology",
                "workflow": "bound-node center-excluded local median repair",
                "execute_context": "global_coarse label evidence",
                "abstain_context": "local_event label evidence with event_erasure risk",
            },
            "dataset_id_is_not_a_skill_input": True,
        },
        "configuration": {
            "transfer_bases": list(TRANSFER_BASES),
            "base_status": (
                "held out from W42-W45 family but historically exposed elsewhere; "
                "not paper-fresh"
            ),
            "roster_audit": roster_audit,
            "observer": {
                "rolling_median_width": OBSERVER_MEDIAN_WIDTH,
                "residual_threshold_fraction": OBSERVER_THRESHOLD_FRACTION,
                "minimum_node_distance": OBSERVER_MIN_NODE_DISTANCE,
                "parameter_search": False,
            },
            "data_program_consumer": (
                "unchanged W43-W45 paired worlds, bound median Program, raw+diff "
                "NearestCentroid readout"
            ),
            "frozen_gates": {
                "observer_precision": 1.0,
                "observer_recall": 1.0,
                "minimum_coarse_gain": MIN_COARSE_GAIN,
                "maximum_forced_event_gain": MAX_EVENT_GAIN,
                "event_policy_harm": 0.0,
                "all_datasets_required": True,
            },
        },
        "dataset_evidence": dataset_evidence,
        "overall": {
            "dataset_count": len(TRANSFER_BASES),
            "passing_dataset_count": passing_dataset_count,
            "all_dataset_controlled_transfer_pass": overall_pass,
        },
        "consumer_fit_count": fit_count,
        "verdict": (
            "CONTROLLED_CROSS_BASE_TASK_SCOPED_SKILL_TRANSFER_PASS"
            if overall_pass
            else "CONTROLLED_CROSS_BASE_TASK_SCOPED_SKILL_TRANSFER_FAIL"
        ),
        "next_step": (
            "Proceed only to fresh Source promotion and then a preregistered A5-vs-A3 "
            "comparison; do not claim it here."
            if overall_pass
            else "Stop this task-scoped impulse Skill transfer family."
        ),
        "target_query_opened": False,
        "uci_values_read": False,
        "capability_promotion": False,
        "formal_transfer": False,
        "A5_vs_A3_claim": False,
        "claim_limit": (
            "Controlled cross-base mechanism transfer to historically exposed NN5/METR "
            "natural backgrounds. It is not paper-fresh Capability promotion, formal "
            "transfer, or evidence that A5 exceeds A3."
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
