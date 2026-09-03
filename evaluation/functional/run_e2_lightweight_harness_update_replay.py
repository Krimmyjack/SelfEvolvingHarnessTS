"""Replay one failure-driven typed Harness update on exposed evidence.

This runner is deliberately development-only.  It compiles an Observation +
Scope patch from the already exposed W48 and W50b failures, then checks that
the frozen patch produced the recorded W52 behavior.  It does not fit a
Consumer, open new outcomes, claim fresh evidence, or promote a Capability.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-lightweight-harness-update-replay/1"
DEFAULT_REPORT_PATH = (
    "artifacts/functional/e2/source_lightweight_harness_update_replay_report.json"
)
W48_REPORT_PATH = (
    "artifacts/functional/e2/source_task_context_label_evidence_witness_report.json"
)
W50B_REPORT_PATH = (
    "artifacts/functional/e2/source_task_risk_confirmation_adaptation_report.json"
)
W52_REPORT_PATH = (
    "artifacts/functional/e2/source_integrated_context_harness_evolution_report.json"
)

ELIGIBLE = "ELIGIBLE_REQUEST_CONFIRMATION"
CONTRAINDICATED = "CONTRAINDICATED_ABSTAIN"
REQUEST = "REQUEST_FULL_CONFIRMATION"
ABSTAIN = "ABSTAIN_KEEP_INCUMBENT"


@dataclass(frozen=True)
class FailurePatternCardLite:
    card_id: str
    evidence_reports: tuple[str, ...]
    first_fault_surface: str
    observed_failures: tuple[str, ...]
    required_behavior: tuple[str, ...]
    w48_unscoped_event_harm_macro: float
    w48_unscoped_event_harmful_dataset_count: int
    w50b_negative_transfer_target_count: int


@dataclass(frozen=True)
class TypedHarnessPatch:
    operation: str
    target_surfaces: tuple[str, ...]
    added_observation: str
    eligible_predicate: str
    eligible_behavior: str
    contraindicated_predicate: str
    contraindicated_behavior: str
    program_change: str
    consumer_change: str
    metric_change: str
    memory_change: str
    patch_origin: str


@dataclass(frozen=True)
class HarnessStateLite:
    state_id: str
    observations: tuple[str, ...]
    scope_rules: tuple[str, ...]
    eligible_behavior: str
    contraindicated_behavior: str
    program: str
    consumer: str
    metric: str
    memory: str


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_original_uci_unopened(*reports: dict[str, Any]) -> None:
    if any(report.get("original_uci_target_query_opened") is not False for report in reports):
        raise ValueError("an input report does not preserve the original UCI boundary")


def build_failure_card(root: Path) -> FailurePatternCardLite:
    """Build the card only from exposed Source failures, before W52 is read."""
    w48 = _read_object(root / W48_REPORT_PATH)
    w50b = _read_object(root / W50B_REPORT_PATH)
    _require_original_uci_unopened(w48, w50b)
    if w48.get("verdict") != "CONTROLLED_NONORACLE_TASK_RISK_WITNESS_PASS":
        raise ValueError("W48 task-risk evidence is unavailable")
    if w50b.get("verdict") != "CONTROLLED_RISK_CONFIRMATION_A5_VS_A3_FAIL":
        raise ValueError("W50b unconfirmed-execution failure is unavailable")

    w48_overall = w48["overall"]
    w50b_overall = w50b["overall"]
    if int(w48_overall["unscoped_event_harmful_dataset_count"]) <= 0:
        raise ValueError("W48 does not exhibit the required unscoped event harm")
    if int(w50b_overall["negative_transfer_target_count"]) <= 0:
        raise ValueError("W50b does not exhibit unconfirmed negative transfer")

    return FailurePatternCardLite(
        card_id="w48-w50b-observation-scope-first-fault/1",
        evidence_reports=(W48_REPORT_PATH, W50B_REPORT_PATH),
        first_fault_surface="observation_and_scope",
        observed_failures=(
            "fit-local evidence alone cannot separate repairable artifacts from stable task events",
            "executing an eligible but unconfirmed action can cause negative transfer",
        ),
        required_behavior=(
            "observe support-versus-fit local label evidence",
            "abstain when local evidence repeats across cohorts",
            "request full confirmation when evidence is fit-only",
        ),
        w48_unscoped_event_harm_macro=float(
            w48_overall["unscoped_event_macro_query_harm"]
        ),
        w48_unscoped_event_harmful_dataset_count=int(
            w48_overall["unscoped_event_harmful_dataset_count"]
        ),
        w50b_negative_transfer_target_count=int(
            w50b_overall["negative_transfer_target_count"]
        ),
    )


def compile_patch(
    card: FailurePatternCardLite, h0: HarnessStateLite
) -> tuple[TypedHarnessPatch, HarnessStateLite]:
    """Compile the single allowed coupled Observation + Scope repair."""
    if card.first_fault_surface != "observation_and_scope":
        raise ValueError("the frozen compiler only accepts the W48/W50b first fault")
    if h0.state_id != "H0_fit_local_unscoped":
        raise ValueError("the frozen compiler only patches the declared H0")

    patch = TypedHarnessPatch(
        operation="ADD_OBSERVATION_AND_RESTRICT_SCOPE",
        target_surfaces=("observation", "applicability_scope"),
        added_observation="support_to_fit_node_strength_and_direction_alignment",
        eligible_predicate="localized_fit_evidence_absent_in_support",
        eligible_behavior=REQUEST,
        contraindicated_predicate="stable_local_evidence_repeated_in_support",
        contraindicated_behavior=ABSTAIN,
        program_change="UNCHANGED:bound_local_median_repair",
        consumer_change="UNCHANGED:ridge_raw_plus_difference",
        metric_change="UNCHANGED:accuracy",
        memory_change="UNCHANGED:none",
        patch_origin="deterministic_human_mechanistic_compile_from_exposed_source_failures",
    )
    h1 = HarnessStateLite(
        state_id="H1_cross_cohort_scoped_confirmation",
        observations=h0.observations + (patch.added_observation,),
        scope_rules=(
            f"{patch.eligible_predicate}->{ELIGIBLE}",
            f"{patch.contraindicated_predicate}->{CONTRAINDICATED}",
        ),
        eligible_behavior=patch.eligible_behavior,
        contraindicated_behavior=patch.contraindicated_behavior,
        program=h0.program,
        consumer=h0.consumer,
        metric=h0.metric,
        memory=h0.memory,
    )
    return patch, h1


def _dataset_replay(row: dict[str, Any]) -> dict[str, Any]:
    conditions = row["conditions"]
    artifact = conditions["fit_only_artifact"]
    event = conditions["stable_task_event"]
    a5_curve = row["A5_source_plus_target"]["mean_curve"]
    a5_b0_states = a5_curve[0]["states"]
    first_feedback = a5_curve[0].get("next_feedback", {})
    target_event_harm = max(float(point["event_harm"]) for point in a5_curve)
    checks = {
        "artifact_scope_is_confirmation": artifact["evolved_scope"] == ELIGIBLE,
        "event_scope_is_abstention": event["evolved_scope"] == CONTRAINDICATED,
        "A5_initial_artifact_requests_confirmation": (
            a5_b0_states["fit_only_artifact"] == REQUEST
        ),
        "A5_initial_event_abstains": a5_b0_states["stable_task_event"] == ABSTAIN,
        "A5_first_confirms_artifact": (
            first_feedback.get("condition") == "fit_only_artifact"
            and first_feedback.get("rule") == "full_support_confirmation"
        ),
        "A5_event_harm_is_zero": target_event_harm <= 1e-12,
    }
    return {
        "dataset": row["dataset"],
        "behavior_checks": checks,
        "behavior_realized": all(checks.values()),
        "H0_no_edit": {
            "policy": "unscoped_execute_both_conditions",
            "event_query_harm": max(0.0, -float(event["forced_query_gain"])),
            "adapt_auc": float(row["H0_unscoped"]["adapt_auc"]),
        },
        "H1_patched_A5": {
            "artifact_scope": artifact["evolved_scope"],
            "event_scope": event["evolved_scope"],
            "event_harm_max": target_event_harm,
            "adapt_auc": float(row["A5_source_plus_target"]["adapt_auc"]),
        },
        "A3_target_only_adapt_auc": float(row["A3_target_only"]["adapt_auc"]),
        "A5_minus_A3_adapt_auc": float(row["A5_minus_A3_adapt_auc"]),
    }


def replay_episode(
    root: Path,
    card: FailurePatternCardLite,
    patch: TypedHarnessPatch,
    h0: HarnessStateLite,
    h1: HarnessStateLite,
) -> dict[str, Any]:
    """Replay recorded W52 behavior without fitting or opening new outcomes."""
    w52 = _read_object(root / W52_REPORT_PATH)
    _require_original_uci_unopened(w52)
    if w52.get("verdict") != "CONTROLLED_INTEGRATED_HARNESS_EVOLUTION_FAIL":
        raise ValueError("expected the exposed W52 strict-gate failure")
    if w52.get("official_target_test_outcome_opened_once") is not True:
        raise ValueError("W52 exposure is not explicit")

    recorded_patch = w52["source_failure_pattern_card"]["source_patch"]
    compiled_patch_matches_record = bool(
        recorded_patch["operation"]
        == "ADD_CROSS_COHORT_OBSERVATION_AND_RESTRICT_SCOPE"
        and recorded_patch["target_surface"] == "observation_plus_scope"
        and recorded_patch["new_observation"] == patch.added_observation
        and recorded_patch["eligible_behavior"] == "request full Target confirmation"
        and patch.eligible_behavior == REQUEST
        and recorded_patch["contraindicated_behavior"] == "abstain"
        and patch.contraindicated_behavior == ABSTAIN
        and recorded_patch["program_changed"] is False
        and recorded_patch["consumer_changed"] is False
        and recorded_patch["proxy_used"] is False
        and patch.operation == "ADD_OBSERVATION_AND_RESTRICT_SCOPE"
    )

    targets = [_dataset_replay(row) for row in w52["dataset_evidence"]]
    overall = w52["overall"]
    behavior_realization_all = bool(
        compiled_patch_matches_record
        and targets
        and all(row["behavior_realized"] for row in targets)
        and overall["scope_update_realized_all_targets"] is True
    )
    average_adaptation_improved = bool(
        float(overall["A5_macro_adapt_auc"])
        > float(overall["A3_macro_adapt_auc"])
    )
    event_harm_eliminated = float(overall["A5_event_harm_max"]) <= 1e-12
    negative_target_count = int(overall["negative_transfer_target_count"])
    uniform_safety = negative_target_count == 0
    development_mechanism_realized = bool(
        behavior_realization_all
        and average_adaptation_improved
        and event_harm_eliminated
    )
    strict_gate = bool(development_mechanism_realized and uniform_safety)

    if not development_mechanism_realized or strict_gate:
        raise ValueError(
            "W52 no longer supports the expected partial development-only replay"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "development-only failure-driven typed Harness update replay",
        "development_only": True,
        "fresh_evidence": False,
        "consumer_fit_count": 0,
        "new_target_outcome_opened": False,
        "evidence_exposure": {
            "W48": "INSTANCE_AND_OUTCOME_EXPOSED",
            "W50b": "INSTANCE_AND_OUTCOME_EXPOSED",
            "W52_targets": "INSTANCE_AND_OUTCOME_EXPOSED",
            "original_UCI_target_query": "UNOPENED",
        },
        "failure_pattern_card": asdict(card),
        "compiled_typed_patch": asdict(patch),
        "harness_transition": {
            "before": asdict(h0),
            "after": asdict(h1),
            "compiled_patch_matches_recorded_W52": compiled_patch_matches_record,
        },
        "counterfactual_replay": {
            "target_count": len(targets),
            "targets": targets,
            "no_edit_H0": {
                "unscoped_event_harmful_target_count": int(
                    overall["H0_unscoped_event_harmful_target_count"]
                ),
                "unscoped_event_harm_macro": float(
                    overall["H0_unscoped_event_harm_macro"]
                ),
            },
            "patched_H1_A5": {
                "A3_macro_adapt_auc": float(overall["A3_macro_adapt_auc"]),
                "A5_macro_adapt_auc": float(overall["A5_macro_adapt_auc"]),
                "A5_minus_A3_macro_adapt_auc": float(
                    overall["A5_minus_A3_macro_adapt_auc"]
                ),
                "A5_event_harm_max": float(overall["A5_event_harm_max"]),
                "negative_transfer_target_count": negative_target_count,
            },
        },
        "acceptance": {
            "behavior_realization_all": behavior_realization_all,
            "average_A5_better_than_A3": average_adaptation_improved,
            "event_harm_eliminated": event_harm_eliminated,
            "development_mechanism_realized": development_mechanism_realized,
            "uniform_safety": uniform_safety,
            "strict_gate": strict_gate,
            "capability_promoted": False,
            "status": "PARTIAL_DEVELOPMENT_REPLAY_ONLY",
        },
        "verdict": "DEVELOPMENT_HARNESS_UPDATE_PARTIAL",
        "supported_claim": (
            "An exposed Source failure card can be deterministically compiled into one typed "
            "Observation+Scope patch whose recorded W52 replay changes the intended Harness "
            "behavior, improves macro AdaptAUC over A3, and avoids expressed event-erasure harm."
        ),
        "claim_ceiling": {
            "failure_driven_typed_patch_compiled_and_behavior_changed": True,
            "natural_capability_transfer_proven": False,
            "autonomous_LLM_patching_proven": False,
            "uniform_target_safety_proven": False,
            "fresh_promotion_evidence": False,
        },
        "remaining_counterexample": (
            "W52 contains one Target with negative A5-vs-A3 transfer; therefore the typed "
            "patch is not promoted and uniform safety remains unproven."
        ),
        "original_uci_target_query_opened": False,
        "persistent_memory_built": False,
    }


def run(root: Path) -> dict[str, Any]:
    h0 = HarnessStateLite(
        state_id="H0_fit_local_unscoped",
        observations=("fit_local_class_conditioned_nodes",),
        scope_rules=("all_localized_nodes->execute",),
        eligible_behavior="EXECUTE_WITHOUT_CONFIRMATION",
        contraindicated_behavior="UNAVAILABLE",
        program="bound_local_median_repair",
        consumer="ridge_raw_plus_difference",
        metric="accuracy",
        memory="none",
    )
    card = build_failure_card(root)
    patch, h1 = compile_patch(card, h0)
    return replay_episode(root, card, patch, h0, h1)


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
