"""Run the bounded v1.2.1 P3 three-task integration gate.

P3 is an integration stage, not the main Evolution experiment.  It reuses
the completed Forecast P2 result, exercises a TRAIN-only Classification
Macro-F1 scope-revision control, and connects the exposed Yahoo-24 roster to
the frozen AD Consumer in identity-only mode.  Official Classification TEST,
Yahoo held-out data, Natural Final, live providers, and P4 runners are outside
this module's capabilities.

Only this entry point writes one machine-readable P3 report.  Component state
is temporary and removed before the report is written.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
P0_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p0_readiness_20260830.json"
P1_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p1_core_baseline_smoke_20260830.json"
P2_REPORT = PROJECT_ROOT / "artifacts/main_protocol/p2_forecast_single_flow_pilot_20260830.json"
AD_CONTROL_REPORT = PROJECT_ROOT / "artifacts/functional/e2/t6_44a_r2_yahoo_positive_control.json"
OUT_JSON = PROJECT_ROOT / "artifacts/main_protocol/p3_unified_integration_gate_20260830.json"

PROTOCOL_VERSION = "v1.2.1-Core"
STAGE = "P3_CLASSIFICATION_AD_ROSTER_VERTICAL_INTEGRATION"
TASKS = ("forecast", "classification", "anomaly_detection")
ARMS = ("Static", "A3-reset", "K0-fixed", "A5-online")
MANDATORY_METHODS = (
    "Identity",
    "Best Fixed Per-task",
    "Fixed Linear-impute",
    "Fixed Hampel",
    "Fixed Winsor",
    "Fixed IQR",
    "Parallel Best-of-N@4",
    "Sequential Refinement@4",
    "Frozen H0",
    *ARMS,
)

B_MAIN = 4
MAX_SUPPORT_A = 3
MAX_SUPPORT_B = 1
MAX_CHEAP_PROBES = 12
MAX_LLM_CALLS = 4
MAX_TOKENS = 40_000
MAX_UPDATES = 1
MATERIAL = 0.005

CLASSIFICATION_SKILL_ID = "p3_cls_hampel_scope"
CLASSIFICATION_PROGRAM = "hampel_filter"
AD_ROUND = "p3_r1"
AD_SUPPORT_TOKEN = 3101
AD_DELAYED_TOKEN = 3102


class P3Blocked(RuntimeError):
    """A fail-closed P3 integration condition."""


class _RawFitCeiling:
    """Bound actual model fits without treating them as logical B=4 units."""

    def __init__(self, cap: int) -> None:
        self.cap = int(cap)
        self.used = 0

    def spend(self, count: int = 1) -> None:
        count = int(count)
        if count < 0 or self.used + count > self.cap:
            raise P3Blocked("raw Consumer fit ceiling exceeded")
        self.used += count


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise P3Blocked("expected a JSON object: %s" % path)
    return payload


def _zero_counters(value: Mapping[str, Any] | None) -> bool:
    return all(int(item or 0) == 0 for item in dict(value or {}).values())


def _sealed_roster_contract() -> tuple[list[str], dict[str, Any]]:
    payload = _read_object(P0_REPORT)
    final_pool = payload.get("final_pool") or {}
    invariants = payload.get("sealed_read_invariants") or {}
    classification = list(final_pool.get("classification") or ())
    anomaly = list(final_pool.get("anomaly_detection") or ())
    failures: list[str] = []
    if classification != ["Adiac", "ArrowHead"]:
        failures.append("P0 Classification sealed roster changed")
    if anomaly != ["Yahoo S5 sealed 41"]:
        failures.append("P0 AD sealed roster changed")
    for key in (
        "ucr_test_member_bytes",
        "yahoo_sealed_41_csv_bytes",
        "solar_numeric_bytes_by_this_runner",
    ):
        if int(invariants.get(key) or 0) != 0:
            failures.append("P0 sealed boundary is non-zero: %s" % key)
    return failures, {
        "source": P0_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "classification": {
            "exposed_train_development": ["Epilepsy2", "GunPoint", "PowerCons"],
            "sealed_final": classification,
            "sealed_test_member_bytes_read": int(
                invariants.get("ucr_test_member_bytes") or 0
            ),
        },
        "anomaly_detection": {
            "structural_roster_count": 65,
            "exposed_development_count": 24,
            "sealed_final_count": 41,
            "sealed_final_label": anomaly,
            "sealed_csv_bytes_read": int(
                invariants.get("yahoo_sealed_41_csv_bytes") or 0
            ),
        },
        "forecast": {
            "sealed_final": list(final_pool.get("forecast") or ()),
            "opened_by_p3": False,
        },
    }


def _validate_p1(payload: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("P1 protocol version mismatch")
    if not payload.get("overall_p1_complete") or not payload.get("release_p2"):
        failures.append("P1 did not complete and release P2")
    if payload.get("live_outcome_release"):
        failures.append("P1 unexpectedly released live outcomes")
    components = payload.get("components") or {}
    if set(components) != set(TASKS):
        failures.append("P1 task roster is not exactly the three protocol tasks")
    method_sets: dict[str, list[str]] = {}
    for task in TASKS:
        component = components.get(task) or {}
        methods = [str(row.get("method")) for row in component.get("methods") or ()]
        method_sets[task] = methods
        if methods != list(MANDATORY_METHODS):
            failures.append("P1 %s method roster changed" % task)
        if not component.get("reported_component_pass"):
            failures.append("P1 %s component did not pass" % task)
        if (component.get("common_dsl_contract") or {}).get("status") != "PASS":
            failures.append("P1 %s Common DSL did not pass" % task)
        if component.get("blocking_failures"):
            failures.append("P1 %s has blocking failures" % task)
        if not _zero_counters(component.get("protocol_errors")):
            failures.append("P1 %s has non-zero protocol errors" % task)
    classification = components.get("classification") or {}
    if (classification.get("consumer") or {}).get("primary_metric") != "Macro-F1":
        failures.append("P1 Classification primary metric is not Macro-F1")
    history = (classification.get("backend") or {}).get("history_contract") or {}
    if not history.get("accuracy_metric_fail_closed"):
        failures.append("P1 Accuracy history is not fail-closed")
    return failures, {
        "source": P1_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "overall_p1_complete": bool(payload.get("overall_p1_complete")),
        "three_task_roster_exact": set(components) == set(TASKS),
        "thirteen_methods_per_task": all(
            method_sets.get(task) == list(MANDATORY_METHODS) for task in TASKS
        ),
        "classification_metric": (classification.get("consumer") or {}).get(
            "primary_metric"
        ),
        "classification_production_lifecycle": bool(
            (classification.get("backend") or {}).get(
                "production_lifecycle_exercised"
            )
        ),
        "ad_method_gate": _plain(payload.get("ad_method_gate") or {}),
    }


def _forecast_p2_facts(payload: Mapping[str, Any]) -> dict[str, Any]:
    runs = list(payload.get("runs") or ())

    def row(decision: int, arm: str) -> Mapping[str, Any]:
        return next(
            (
                item
                for item in runs
                if int(item.get("decision_index") or 0) == decision
                and item.get("arm") == arm
            ),
            {},
        )

    first = row(1, "A5-online")
    later_k0 = row(2, "K0-fixed")
    later_a5 = row(2, "A5-online")
    first_trace = first.get("trace") or {}
    first_update = first.get("update") or {}
    k0_trace = later_k0.get("trace") or {}
    a5_trace = later_a5.get("trace") or {}
    return {
        "positive_support_before_update": bool(
            first_trace.get("controlled_probe_count")
            and any(
                float(value) >= MATERIAL
                for value in first_trace.get("controlled_probe_gains") or ()
            )
        ),
        "legal_update": bool(
            first_update.get("accepted")
            and first_update.get("production_revocation")
        ),
        "retained_across_unit": bool(first_update.get("retained_at_unit_boundary")),
        "later_reencounter": bool(later_k0 and later_a5),
        "later_behavior_influenced": bool(
            int(k0_trace.get("controlled_supply_count") or 0) > 0
            and int(a5_trace.get("controlled_supply_count") or 0) == 0
            and bool(a5_trace.get("abstained"))
        ),
        "surviving_usable_skill": False,
        "revalidated_after_revision": False,
        "update_kind": "REVOKE",
    }


def _validate_p2(payload: Mapping[str, Any]) -> tuple[list[str], dict[str, Any], dict[str, Any]]:
    failures: list[str] = []
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("P2 protocol version mismatch")
    if not payload.get("p2_complete") or not payload.get("release_p3"):
        failures.append("P2 did not complete and release P3")
    if payload.get("release_p4") or payload.get("live_outcome_release"):
        failures.append("P2 crossed its release boundary")
    if (payload.get("budget_gate") or {}).get("status") != "PASS":
        failures.append("P2 budget gate did not pass")
    if not _zero_counters(payload.get("protocol_errors")):
        failures.append("P2 has non-zero protocol errors")
    boundary = payload.get("boundaries") or {}
    for key in (
        "development_query_evaluations",
        "natural_final_outcome_reads",
        "traffic_loader_invocations",
        "solar_loader_invocations",
        "query_feedback_events",
    ):
        if int(boundary.get(key) or 0) != 0:
            failures.append("P2 boundary counter is non-zero: %s" % key)
    facts = _forecast_p2_facts(payload)
    if not all(
        facts[key]
        for key in (
            "positive_support_before_update",
            "legal_update",
            "retained_across_unit",
            "later_reencounter",
            "later_behavior_influenced",
        )
    ):
        failures.append("P2 ordered Forecast treatment facts do not recompute")
    summary = {
        "source": P2_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "executed_by_p3": False,
        "p2_complete": bool(payload.get("p2_complete")),
        "treatment_recomputed": not any(
            failure.startswith("P2 ordered") for failure in failures
        ),
        "evidence_grade": payload.get("evidence_grade"),
        "claim_ceiling": "RISK_CONTROL_ONLY",
    }
    return failures, summary, facts


def _relation(delta: float) -> str:
    if float(delta) >= MATERIAL:
        return "POSITIVE"
    if float(delta) <= -MATERIAL:
        return "NEGATIVE"
    return "IMMATERIAL"


def _usage(
    *,
    support_a: int,
    support_b: int,
    raw_a: int,
    raw_b: int,
    cache_a: int = 0,
    cache_b: int = 0,
    cheap_probes: int = 0,
    accepted_updates: int = 0,
    wall_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "full_support_evaluations": {
            "support_a": int(support_a),
            "support_b": int(support_b),
        },
        "raw_consumer_fits": {
            "support_a": int(raw_a),
            "support_b": int(raw_b),
        },
        "cache_hits": {
            "support_a": int(cache_a),
            "support_b": int(cache_b),
        },
        "cheap_probes": int(cheap_probes),
        "llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "tokens": 0,
        "accepted_updates": int(accepted_updates),
        "wall_seconds": round(float(wall_seconds), 3),
    }


def budget_failures(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Validate B=4 per task x arm x target; raw fits are descriptive."""
    failures: list[str] = []
    for index, row in enumerate(rows):
        usage = row.get("usage") or {}
        full = usage.get("full_support_evaluations") or {}
        raw = usage.get("raw_consumer_fits") or {}
        cache = usage.get("cache_hits") or {}
        a = int(full.get("support_a") or 0)
        b = int(full.get("support_b") or 0)
        values = [
            a,
            b,
            int(raw.get("support_a") or 0),
            int(raw.get("support_b") or 0),
            int(cache.get("support_a") or 0),
            int(cache.get("support_b") or 0),
            int(usage.get("cheap_probes") or 0),
            int(usage.get("llm_calls") or 0),
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            int(usage.get("tokens") or 0),
            int(usage.get("accepted_updates") or 0),
        ]
        label = "%s/%s/%s" % (
            row.get("task", "unknown"),
            row.get("arm", "unknown"),
            row.get("target", index),
        )
        if any(value < 0 for value in values):
            failures.append("negative cost in %s" % label)
        if a > MAX_SUPPORT_A or b > MAX_SUPPORT_B or a + b > B_MAIN:
            failures.append("logical evaluation cap exceeded in %s" % label)
        if int(usage.get("cheap_probes") or 0) > MAX_CHEAP_PROBES:
            failures.append("cheap probe cap exceeded in %s" % label)
        if int(usage.get("llm_calls") or 0) > MAX_LLM_CALLS:
            failures.append("LLM call cap exceeded in %s" % label)
        token_sum = int(usage.get("input_tokens") or 0) + int(
            usage.get("output_tokens") or 0
        )
        if token_sum != int(usage.get("tokens") or 0):
            failures.append("token arithmetic mismatch in %s" % label)
        if token_sum > MAX_TOKENS:
            failures.append("token cap exceeded in %s" % label)
        if int(usage.get("accepted_updates") or 0) > MAX_UPDATES:
            failures.append("accepted update cap exceeded in %s" % label)
    return failures


def derive_treatment_state(facts: Mapping[str, Any]) -> dict[str, Any]:
    """Separate no-op, generation, terminal control, and nonterminal revision."""
    if not facts.get("legal_update"):
        state = "NO_TREATMENT"
        ceiling = "NO_ADAPTATION_CLAIM"
    elif not all(
        facts.get(key)
        for key in ("retained_across_unit", "later_reencounter", "later_behavior_influenced")
    ):
        state = "GENERATION_ONLY"
        ceiling = "NO_ADAPTATION_CLAIM"
    elif not facts.get("surviving_usable_skill"):
        state = "TERMINAL_RISK_CONTROL"
        ceiling = "RISK_CONTROL_ONLY"
    elif facts.get("revalidated_after_revision"):
        state = "NONTERMINAL_REVISION"
        ceiling = "CONTROLLED_REVISION_MECHANISM"
    else:
        state = "GENERATION_ONLY"
        ceiling = "NO_ADAPTATION_CLAIM"
    return {"state": state, "claim_ceiling": ceiling}


def _classification_readings(cell: Any, faces: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from evaluation.main_protocol_p1 import classification_component as classification

    started = time.time()
    adapter = classification.MacroF1ConsumerAdapter(
        cell=cell,
        budget=classification.FitBudget(cap=2 * len(tuple(faces))),
    )
    readings: dict[str, Any] = {}
    for face in faces:
        identity = adapter.evaluate((), face)
        candidate = adapter.evaluate(classification._steps(CLASSIFICATION_PROGRAM), face)
        delta = float(candidate["macro_f1"]) - float(identity["macro_f1"])
        readings[str(face)] = {
            "identity_macro_f1": float(identity["macro_f1"]),
            "candidate_macro_f1": float(candidate["macro_f1"]),
            "delta_u_vs_identity": delta,
            "relation": _relation(delta),
            "identity_accuracy": float(identity["accuracy"]),
            "candidate_accuracy": float(candidate["accuracy"]),
            "identity_worst_class_recall": float(identity["worst_class_recall"]),
            "candidate_worst_class_recall": float(candidate["worst_class_recall"]),
            "identity_recall_by_class": _plain(identity["per_class_recall"]),
            "candidate_recall_by_class": _plain(candidate["per_class_recall"]),
            "candidate_behavior_point_count": int(candidate["behavior_point_count"]),
        }
    return readings, _usage(
        support_a=int("support_a" in readings),
        support_b=int("support_b" in readings),
        raw_a=2 * int("support_a" in readings),
        raw_b=2 * int("support_b" in readings),
        wall_seconds=time.time() - started,
    )


def _skill_projection(skill: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "skill_id": str(skill["skill_id"]),
        "revision": int(skill["revision"]),
        "skill_kind": str(skill["skill_kind"]),
        "body": str(skill["body"]),
        "observable_applicability": _plain(skill["observable_applicability"]),
        "allowed_tools": list(skill["allowed_tools"]),
        "risk_guards": _plain(skill["risk_guards"]),
    }


def _supply_projection(skill: Mapping[str, Any], features: Mapping[str, Any]) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _parse_frozen_steps
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability

    matched, _score = evaluate_applicability(
        skill["observable_applicability"], features
    )
    steps = _parse_frozen_steps(str(skill["body"])) if matched else None
    return {
        "retrieved": bool(matched),
        "supplied": steps is not None,
        "program_steps": [] if steps is None else _plain(steps),
    }


def _classification_component() -> dict[str, Any]:
    from evaluation.functional.task_episode_harness.agentic import skill_revision
    from evaluation.main_protocol_p1 import classification_component as classification
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features

    started = time.time()
    target, selection, data_boundary = classification._load_exposed_cells()
    cells = {cell.fixture_id: cell for cell in (target, *selection)}
    positive_cell = cells["Epilepsy2"]
    refusing_cell = cells["PowerCons"]

    qualification, qualification_usage = _classification_readings(
        positive_cell, ("support_a", "support_b")
    )
    conflict, conflict_usage = _classification_readings(
        refusing_cell, ("support_a",)
    )

    card = {
        "schema_version": "skill-entry/1",
        "skill_id": CLASSIFICATION_SKILL_ID,
        "skill_kind": "capability",
        "revision": 1,
        "body": "Frozen program steps: "
        + json.dumps(
            [{"op": CLASSIFICATION_PROGRAM, "params": {}}],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "observable_applicability": {
            "feature": "task_kind",
            "op": "==",
            "value": "classification",
        },
        "allowed_tools": [CLASSIFICATION_PROGRAM],
        "risk_guards": {
            "requires_target_support": True,
            "grants_execution": False,
            "controlled_mechanism_card": True,
        },
    }
    positive_features = dict(
        extract_public_features(
            positive_cell.observation_block, task_kind="classification"
        )
    )
    refusing_features = dict(
        extract_public_features(
            refusing_cell.observation_block, task_kind="classification"
        )
    )
    initial_positive_supply = _supply_projection(card, positive_features)
    initial_refusing_supply = _supply_projection(card, refusing_features)
    exclusion = skill_revision.compile_exclusion(
        refusing_view=refusing_features,
        source_views=[positive_features],
        axes=skill_revision.contracted_axes(task_kind="classification"),
    )
    leaves = list(exclusion.get("leaves") or ())
    narrowed = skill_revision.narrow_applicability(
        card["observable_applicability"], leaves
    )

    # This is deliberately a semantic policy replay, not a production
    # promotion.  The retained-context readings are checked before the
    # narrowed projection is used, and no Harness state is written.
    revalidation, revalidation_usage = _classification_readings(
        positive_cell, ("support_a", "support_b")
    )
    updated = dict(card)
    updated["observable_applicability"] = narrowed
    initial_projection = _skill_projection(card)
    updated_projection = _skill_projection(updated)
    changed_fields = [
        key
        for key in initial_projection
        if initial_projection[key] != updated_projection[key]
    ]
    updated_positive_supply = _supply_projection(updated, positive_features)
    updated_refusing_supply = _supply_projection(updated, refusing_features)
    k0_reencounter, k0_usage = _classification_readings(
        refusing_cell, ("support_a",)
    )
    identity_adapter = classification.MacroF1ConsumerAdapter(
        cell=refusing_cell, budget=classification.FitBudget(cap=1)
    )
    a5_identity = identity_adapter.evaluate((), "support_a")
    a5_usage = _usage(
        support_a=1,
        support_b=0,
        raw_a=1,
        raw_b=0,
        cheap_probes=1,
    )

    conflict_usage["cheap_probes"] = 1
    k0_old = k0_reencounter["support_a"]
    a5_gain = float(a5_identity["macro_f1"]) - float(k0_old["candidate_macro_f1"])
    expected_steps = [[CLASSIFICATION_PROGRAM, {}]]
    gate_checks = {
        "initial_support_a_positive": qualification["support_a"]["relation"] == "POSITIVE",
        "initial_support_b_positive": qualification["support_b"]["relation"] == "POSITIVE",
        "later_material_conflict": conflict["support_a"]["relation"] == "NEGATIVE",
        "initial_card_retrieved_and_supplied_on_both_contexts": bool(
            initial_positive_supply["retrieved"]
            and initial_positive_supply["supplied"]
            and initial_refusing_supply["retrieved"]
            and initial_refusing_supply["supplied"]
        ),
        "only_scope_changed": changed_fields == ["observable_applicability"],
        "same_skill_and_program_survived": bool(
            initial_projection["skill_id"] == updated_projection["skill_id"]
            and initial_projection["body"] == updated_projection["body"]
            and updated_positive_supply["program_steps"] == expected_steps
        ),
        "revised_scope_retains_positive_context": bool(
            updated_positive_supply["retrieved"]
            and updated_positive_supply["supplied"]
        ),
        "revised_scope_withholds_refusing_context": bool(
            not updated_refusing_supply["retrieved"]
            and not updated_refusing_supply["supplied"]
        ),
        "revalidation_support_a_positive": revalidation["support_a"]["relation"] == "POSITIVE",
        "revalidation_support_b_positive": revalidation["support_b"]["relation"] == "POSITIVE",
        "k0_replays_old_card_on_same_surface": bool(
            initial_refusing_supply["supplied"]
            and k0_old["relation"] == "NEGATIVE"
        ),
        "controlled_policy_changes_later_same_surface_supply": bool(
            not updated_refusing_supply["supplied"] and a5_gain >= MATERIAL
        ),
        "one_semantic_scope_edit": True,
    }
    policy_replay_pass = all(gate_checks.values())
    facts = {
        "legal_update": False,
        "controlled_semantic_edit": bool(
            gate_checks["later_material_conflict"]
            and gate_checks["only_scope_changed"]
        ),
        "retained_across_unit": gate_checks["same_skill_and_program_survived"],
        "later_reencounter": False,
        "later_same_surface_replay": True,
        "later_behavior_influenced": gate_checks[
            "controlled_policy_changes_later_same_surface_supply"
        ],
        "surviving_usable_skill": gate_checks[
            "revised_scope_retains_positive_context"
        ],
        "revalidated_after_revision": False,
        "retained_context_replay_positive": bool(
            gate_checks["revalidation_support_a_positive"]
            and gate_checks["revalidation_support_b_positive"]
        ),
        "update_kind": "SEMANTIC_SCOPE_NARROW_REPLAY",
    }
    cost_rows = [
        {
            "task": "classification",
            "arm": "K0-fixed/controlled-policy initial qualification",
            "target": "Epilepsy2",
            "usage": qualification_usage,
        },
        {
            "task": "classification",
            "arm": "controlled scope-edit policy replay",
            "target": "PowerCons",
            "usage": conflict_usage,
        },
        {
            "task": "classification",
            "arm": "controlled retained-context policy replay",
            "target": "Epilepsy2",
            "usage": revalidation_usage,
        },
        {
            "task": "classification",
            "arm": "K0-fixed same-surface replay",
            "target": "PowerCons",
            "usage": k0_usage,
        },
        {
            "task": "classification",
            "arm": "controlled narrowed-policy same-surface replay",
            "target": "PowerCons",
            "usage": a5_usage,
        },
    ]
    boundary_ok = bool(
        int(data_boundary.get("test_member_bytes_read") or 0) == 0
        and int(data_boundary.get("held_out_requests") or 0) == 0
        and int(data_boundary.get("development_query_evaluations") or 0) == 0
        and int(data_boundary.get("natural_final_outcome_reads") or 0) == 0
    )
    integration_pass = (
        policy_replay_pass and boundary_ok and not budget_failures(cost_rows)
    )
    return {
        "task": "classification",
        "integration_status": "PASS" if integration_pass else "FAIL",
        "evidence_grade": "CONTROLLED_REAL_CONSUMER_POLICY_REPLAY",
        "consumer": {
            "id": classification.CONSUMER_ID,
            "primary_metric": "Macro-F1",
            "secondary_metrics": ["Accuracy", "per-class recall", "worst-class recall"],
        },
        "data": {
            "role": "EXPOSED_TRAIN_ONLY",
            "datasets": [target.fixture_id, *[cell.fixture_id for cell in selection]],
            "sealed_final_roster": ["Adiac", "ArrowHead"],
            "sealed_final_opened_by_p3": False,
            "qualification_context": positive_cell.fixture_id,
            "refusing_and_reencounter_context": refusing_cell.fixture_id,
            "surface_counts": data_boundary.get("surface_counts"),
            "fit_support_a_support_b_disjoint": True,
            "official_test_member_bytes_read": int(
                data_boundary.get("test_member_bytes_read") or 0
            ),
            "held_out_requests": int(data_boundary.get("held_out_requests") or 0),
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
        },
        "controlled_scope_edit_replay": {
            "status": (
                "CONTROLLED_SCOPE_POLICY_REPLAY_PASS"
                if policy_replay_pass
                else "NOT_EXERCISED"
            ),
            "skill_id": CLASSIFICATION_SKILL_ID,
            "program": CLASSIFICATION_PROGRAM,
            "card_role": "CONTROLLED_SUPPLY_ONLY_MECHANISM_CARD",
            "source_learned": False,
            "autonomous_failure_diagnosis": False,
            "post_hoc_capability_claim": False,
            "evidence_independence": {
                "qualification_and_revalidation_reuse_exposed_surfaces": True,
                "conflict_and_reencounter_reuse_exposed_surface": True,
                "fresh_replica": False,
                "generalization_claim": False,
            },
            "qualification": qualification,
            "conflict": conflict,
            "revision": {
                "kind": "SEMANTIC_SCOPE_NARROW",
                "changed_fields": changed_fields,
                "entry_revision_counter_changed": (
                    initial_projection["revision"] != updated_projection["revision"]
                ),
                "body_unchanged": initial_projection["body"] == updated_projection["body"],
                "skill_id_unchanged": (
                    initial_projection["skill_id"] == updated_projection["skill_id"]
                ),
                "exclusion_leaf_count": len(leaves),
                "exclusion_axes": sorted(str(leaf.get("feature")) for leaf in leaves),
                "semantic_scope_edits": 1,
                "production_update_executed": False,
                "accepted_updates": 0,
                "pending_before_delayed": False,
                "independent_delayed_approval": False,
            },
            "revalidation": revalidation,
            "later_policy_replay": {
                "k0_supply": initial_refusing_supply,
                "a5_supply": updated_refusing_supply,
                "k0_candidate_macro_f1": float(k0_old["candidate_macro_f1"]),
                "a5_identity_macro_f1": float(a5_identity["macro_f1"]),
                "a5_minus_k0_utility": a5_gain,
                "a5_action": "WITHHOLD_CONTROLLED_CARD_AND_USE_IDENTITY",
                "autonomous_abstention_claim": False,
                "same_surface_replay": True,
                "independent_reencounter": False,
            },
            "gate_checks": gate_checks,
            "claim_ceiling": "MECHANICAL_SCOPE_NARROWING_REPLAY",
        },
        "treatment_facts": facts,
        "cost_rows": cost_rows,
        "boundaries": {
            "official_test_member_bytes_read": int(
                data_boundary.get("test_member_bytes_read") or 0
            ),
            "held_out_requests": int(data_boundary.get("held_out_requests") or 0),
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "live_provider_calls": 0,
        },
        "rq3_status": "NOT_EXERCISED",
        "natural_performance_claim": False,
        "source_transfer_claim": False,
        "wall_seconds": round(time.time() - started, 3),
    }


def _ad_control_status() -> dict[str, Any]:
    payload = _read_object(AD_CONTROL_REPORT)
    verdict = payload.get("verdict") or {}
    primary = str(verdict.get("verdict") or "UNKNOWN")
    secondary = str(verdict.get("secondary") or "")
    b1 = payload.get("b1") or {}
    passed_rates = [
        str(name)
        for name, row in b1.items()
        if isinstance(row, Mapping) and row.get("verdict") == "EFFECT_CONFIRMED"
    ]
    b2 = payload.get("b2") or {}
    passed = bool(passed_rates and b2 and payload.get("signals_predictive"))
    return {
        "source": AD_CONTROL_REPORT.relative_to(PROJECT_ROOT).as_posix(),
        "status": "PASSED" if passed else "NOT_PASSED",
        "verdict": primary,
        "secondary": secondary or None,
        "b1_effect_confirmed_rates": passed_rates,
        "b2_executed": bool(b2),
        "ad_evolution_allowed": passed,
    }


def _anomaly_component() -> dict[str, Any]:
    import numpy as np

    from evaluation.functional import run_e2_t6_natural_a5_a3 as yahoo
    from evaluation.functional.consumers import aegists_iforest_v1 as consumer
    from evaluation.functional.consumers.p0b_scope_adapters import (
        TrainingBlockScopeExecutor,
        WindowedIForestAdapter,
    )

    started = time.time()
    pack = yahoo._load_yahoo_l1_roster()
    base_rows = pack["rows"]
    order = [str(uid) for uid in pack["order"]]
    rows: dict[str, Any] = {}
    for uid in order:
        source = base_rows[uid]
        rows[uid] = {
            "values": np.asarray(source["values"], dtype=np.float64),
            "windows": {
                AD_ROUND: {
                    "train": list(source["windows"]["r1"]["train"]),
                    "support_a": list(source["windows"]["r1"]["support"]),
                    "support_b": list(source["windows"]["r1"]["delayed"]),
                }
            },
        }
    roster = [{"series_uid": uid, "role": "train"} for uid in sorted(rows)]
    values = {uid: rows[uid]["values"] for uid in rows}
    wall = yahoo.YahooHeldInWall(base_rows)
    fit_ceiling = _RawFitCeiling(len(rows))
    adapter = WindowedIForestAdapter(
        consumer=consumer,
        rows=rows,
        round_name=AD_ROUND,
        event_reader=wall.events_for,
        fit_budget=fit_ceiling,
        phase_by_origin={
            AD_SUPPORT_TOKEN: "support_a",
            AD_DELAYED_TOKEN: "support_b",
        },
    )
    geometry = TrainingBlockScopeExecutor(
        rows=rows,
        round_name=AD_ROUND,
        evaluate_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("geometry-only executor must not evaluate")
        ),
        max_modified_fraction=0.20,
    )
    training_windows = geometry.training_windows(AD_SUPPORT_TOKEN)
    support_a = adapter(roster, values, None, {}, origin=AD_SUPPORT_TOKEN)
    support_b = adapter(roster, values, None, {}, origin=AD_DELAYED_TOKEN)
    calls = list(adapter.calls)
    if len(calls) != 2:
        raise P3Blocked("AD adapter did not produce exactly two face calls")

    heldout_safe = True
    for request in wall.requests:
        rec = base_rows[str(request["uid"])]
        heldout_safe = heldout_safe and bool(
            request.get("granted")
            and int(request["hi"]) <= int(rec["windows"]["heldout"][0])
        )
    window_lengths = [int(np.asarray(row[2]).size) for row in training_windows]
    geometry_pass = bool(
        len(order) == 24
        and order == sorted(order)
        and len(training_windows) == 24
        and all(length >= int(consumer.WINDOW) for length in window_lengths)
        and all(
            int(rows[uid]["windows"][AD_ROUND]["train"][1])
            <= int(rows[uid]["windows"][AD_ROUND]["support_a"][0])
            < int(rows[uid]["windows"][AD_ROUND]["support_a"][1])
            <= int(rows[uid]["windows"][AD_ROUND]["support_b"][0])
            < int(rows[uid]["windows"][AD_ROUND]["support_b"][1])
            <= int(base_rows[uid]["windows"]["heldout"][0])
            for uid in order
        )
    )
    macro_a = float(support_a["ad_macro_f1"])
    macro_b = float(support_b["ad_macro_f1"])
    metric_pass = math.isfinite(macro_a) and math.isfinite(macro_b)
    cache_pass = bool(
        int(calls[0]["raw_consumer_fits"]) == 24
        and int(calls[0]["model_cache_hits"]) == 0
        and int(calls[1]["raw_consumer_fits"]) == 0
        and int(calls[1]["model_cache_hits"]) == 24
        and fit_ceiling.used == 24
    )
    control = _ad_control_status()
    usage = _usage(
        support_a=1,
        support_b=1,
        raw_a=int(calls[0]["raw_consumer_fits"]),
        raw_b=int(calls[1]["raw_consumer_fits"]),
        cache_a=int(calls[0]["model_cache_hits"]),
        cache_b=int(calls[1]["model_cache_hits"]),
        wall_seconds=time.time() - started,
    )
    cost_rows = [
        {
            "task": "anomaly_detection",
            "arm": "method-gate identity diagnostic",
            "target": "Yahoo exposed first-24",
            "usage": usage,
        }
    ]
    integration_pass = bool(
        geometry_pass
        and metric_pass
        and cache_pass
        and heldout_safe
        and wall.held_out_requests == 0
        and not budget_failures(cost_rows)
    )
    per_a = [-float(value) for value in support_a["per_view_smase"]]
    per_b = [-float(value) for value in support_b["per_view_smase"]]
    return {
        "task": "anomaly_detection",
        "integration_status": "PASS" if integration_pass else "FAIL",
        "diagnostic_verdict": (
            "AD_P3_TRAIN_ONLY_ROSTER_ADAPTER_PASS__METHOD_GATE_CLOSED"
            if integration_pass
            else "AD_P3_INTEGRATION_BLOCKED"
        ),
        "evidence_grade": "INSTRUMENT",
        "consumer": {
            "id": consumer.CONSUMER_ID,
            "primary_metric": "macro Event-F1",
            "program": "identity",
        },
        "data": {
            "role": "EXPOSED_DEVELOPMENT_TRAIN_ONLY",
            "dataset": "Yahoo S5 A1",
            "roster_count": len(order),
            "roster": order,
            "full_structural_roster_count": 65,
            "sealed_final_count": 41,
            "sealed_final_opened_by_p3": False,
            "window_plan": "r1: train [0,.30), Support-A [.30,.40), Support-B [.40,.50)",
            "training_window_count": len(training_windows),
            "training_window_length_min": min(window_lengths),
            "training_window_length_max": max(window_lengths),
            "geometry_pass": geometry_pass,
        },
        "readings": {
            "support_a": {
                "macro_event_f1": macro_a,
                "pooled_event_f1": float(support_a["ad_pooled_f1"]),
                "per_series_event_f1": dict(zip(sorted(rows), per_a, strict=True)),
            },
            "support_b": {
                "macro_event_f1": macro_b,
                "pooled_event_f1": float(support_b["ad_pooled_f1"]),
                "per_series_event_f1": dict(zip(sorted(rows), per_b, strict=True)),
            },
        },
        "behavior": {
            "status": "NOT_EXERCISED",
            "deployed_program": "identity",
            "production_agent_executed_in_p3": False,
            "episodes_written": 0,
            "skills_written": 0,
            "store_updates": 0,
            "accepted_updates": 0,
            "forced_identity_is_not_autonomous_abstention": True,
        },
        "method_gate": control,
        "treatment_facts": {
            "legal_update": False,
            "retained_across_unit": False,
            "later_reencounter": False,
            "later_behavior_influenced": False,
            "surviving_usable_skill": False,
            "revalidated_after_revision": False,
            "update_kind": "NONE",
        },
        "cost_rows": cost_rows,
        "boundaries": {
            "held_in_label_requests": len(wall.requests),
            "held_out_requests": int(wall.held_out_requests),
            "all_label_requests_before_heldout": heldout_safe,
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
            "production_agent_calls": 0,
            "episode_writes": 0,
            "skill_writes": 0,
            "store_updates": 0,
        },
        "rq3_status": "NOT_EXERCISED",
        "release_p4_ad": False,
        "natural_performance_claim": False,
        "wall_seconds": round(time.time() - started, 3),
    }


def derive_p4_gate(
    *,
    p3_integration_complete: bool,
    forecast_state: Mapping[str, Any],
    classification_state: Mapping[str, Any],
    anomaly_state: Mapping[str, Any],
    ad_method_gate_passed: bool,
) -> dict[str, Any]:
    by_task = {
        "forecast": {
            "treatment_state": forecast_state.get("state"),
            "method_gate_passed": forecast_state.get("state")
            == "TERMINAL_RISK_CONTROL",
            "rq3_exercisable": False,
            "claim_ceiling": forecast_state.get("claim_ceiling"),
        },
        "classification": {
            "treatment_state": classification_state.get("state"),
            "method_gate_passed": classification_state.get("state")
            == "NONTERMINAL_REVISION",
            "production_revision_gate_passed": classification_state.get("state")
            == "NONTERMINAL_REVISION",
            "rq3_exercisable": False,
            "claim_ceiling": classification_state.get("claim_ceiling"),
        },
        "anomaly_detection": {
            "treatment_state": anomaly_state.get("state"),
            "method_gate_passed": bool(ad_method_gate_passed),
            "rq3_exercisable": False,
            "claim_ceiling": anomaly_state.get("claim_ceiling"),
        },
    }
    release = bool(
        p3_integration_complete
        and all(row["method_gate_passed"] for row in by_task.values())
        and anomaly_state.get("state") != "NO_TREATMENT"
    )
    blockers: list[str] = []
    if not p3_integration_complete:
        blockers.append("P3 integration is incomplete")
    if not ad_method_gate_passed:
        blockers.append("AD #44a positive control is not passed")
    if anomaly_state.get("state") == "NO_TREATMENT":
        blockers.append("AD Treatment reachability is not formed")
    if classification_state.get("state") != "NONTERMINAL_REVISION":
        blockers.append("Classification production revision reachability is not formed")
    return {
        "release_p4": release,
        "release_scope": "P4 three-task Evolution only" if release else "NONE",
        "by_task": by_task,
        "blocking_failures": blockers,
        "p4_executed": False,
        "natural_final_release": False,
    }


def _failure_component(task: str, exc: Exception) -> dict[str, Any]:
    return {
        "task": task,
        "integration_status": "FAIL",
        "runtime_failure_type": type(exc).__name__,
        "runtime_failure": str(exc),
        "treatment_facts": {
            "legal_update": False,
            "retained_across_unit": False,
            "later_reencounter": False,
            "later_behavior_influenced": False,
            "surviving_usable_skill": False,
            "revalidated_after_revision": False,
            "update_kind": "NONE",
        },
        "cost_rows": [],
        "boundaries": {
            "development_query_evaluations": 0,
            "natural_final_outcome_reads": 0,
        },
    }


def build_report() -> dict[str, Any]:
    started = time.time()
    upstream_failures: list[str] = []
    try:
        roster_failures, data_roster = _sealed_roster_contract()
        upstream_failures.extend(roster_failures)
    except Exception as exc:  # noqa: BLE001 - one bounded gate result
        data_roster = {"source": P0_REPORT.relative_to(PROJECT_ROOT).as_posix()}
        upstream_failures.append("P0 roster report unreadable: %s" % type(exc).__name__)
    try:
        p1_raw = _read_object(P1_REPORT)
        p1_failures, p1_summary = _validate_p1(p1_raw)
        upstream_failures.extend(p1_failures)
    except Exception as exc:  # noqa: BLE001 - one bounded gate result
        p1_summary = {"source": P1_REPORT.relative_to(PROJECT_ROOT).as_posix()}
        upstream_failures.append("P1 report unreadable: %s" % type(exc).__name__)
    try:
        p2_raw = _read_object(P2_REPORT)
        p2_failures, p2_summary, forecast_facts = _validate_p2(p2_raw)
        upstream_failures.extend(p2_failures)
    except Exception as exc:  # noqa: BLE001 - one bounded gate result
        p2_summary = {"source": P2_REPORT.relative_to(PROJECT_ROOT).as_posix()}
        forecast_facts = {"legal_update": False}
        upstream_failures.append("P2 report unreadable: %s" % type(exc).__name__)

    if upstream_failures:
        classification = _failure_component(
            "classification", P3Blocked("upstream gate did not release P3 execution")
        )
        anomaly = _failure_component(
            "anomaly_detection", P3Blocked("upstream gate did not release P3 execution")
        )
    else:
        try:
            classification = _classification_component()
        except Exception as exc:  # noqa: BLE001 - preserve AD diagnostics
            classification = _failure_component("classification", exc)
        try:
            anomaly = _anomaly_component()
        except Exception as exc:  # noqa: BLE001 - preserve Classification diagnostics
            anomaly = _failure_component("anomaly_detection", exc)

    forecast_state = derive_treatment_state(forecast_facts)
    classification_state = derive_treatment_state(
        classification.get("treatment_facts") or {}
    )
    anomaly_state = derive_treatment_state(anomaly.get("treatment_facts") or {})
    cost_rows = [
        *list(classification.get("cost_rows") or ()),
        *list(anomaly.get("cost_rows") or ()),
    ]
    costs_failed = budget_failures(cost_rows)
    p3_failures = [*upstream_failures, *costs_failed]
    if classification.get("integration_status") != "PASS":
        p3_failures.append("Classification P3 vertical slice did not pass")
    if anomaly.get("integration_status") != "PASS":
        p3_failures.append("AD P3 identity-only vertical slice did not pass")
    if not bool(
        (classification.get("treatment_facts") or {}).get(
            "controlled_semantic_edit"
        )
    ):
        p3_failures.append("controlled Classification scope-policy replay did not pass")
    p3_failures = list(dict.fromkeys(p3_failures))
    p3_complete = not p3_failures
    ad_gate_passed = bool(
        (anomaly.get("method_gate") or {}).get("ad_evolution_allowed")
    )
    p4_gate = derive_p4_gate(
        p3_integration_complete=p3_complete,
        forecast_state=forecast_state,
        classification_state=classification_state,
        anomaly_state=anomaly_state,
        ad_method_gate_passed=ad_gate_passed,
    )
    protocol_errors = {
        "forecast_runner_invocations": 0,
        "classification_official_test_member_bytes_read": int(
            (classification.get("boundaries") or {}).get(
                "official_test_member_bytes_read", 0
            )
            or 0
        ),
        "classification_held_out_requests": int(
            (classification.get("boundaries") or {}).get("held_out_requests", 0)
            or 0
        ),
        "ad_held_out_requests": int(
            (anomaly.get("boundaries") or {}).get("held_out_requests", 0) or 0
        ),
        "development_query_evaluations": int(
            (classification.get("boundaries") or {}).get(
                "development_query_evaluations", 0
            )
            or 0
        )
        + int(
            (anomaly.get("boundaries") or {}).get(
                "development_query_evaluations", 0
            )
            or 0
        ),
        "natural_final_outcome_reads": int(
            (classification.get("boundaries") or {}).get(
                "natural_final_outcome_reads", 0
            )
            or 0
        )
        + int(
            (anomaly.get("boundaries") or {}).get(
                "natural_final_outcome_reads", 0
            )
            or 0
        ),
        "live_provider_calls": 0,
        "ad_agent_calls": int(
            (anomaly.get("boundaries") or {}).get("production_agent_calls", 0)
            or 0
        ),
        "ad_episode_writes": int(
            (anomaly.get("boundaries") or {}).get("episode_writes", 0) or 0
        ),
        "ad_skill_writes": int(
            (anomaly.get("boundaries") or {}).get("skill_writes", 0) or 0
        ),
        "p4_runner_invocations": 0,
    }
    nonzero_errors = [
        key for key, value in protocol_errors.items() if int(value) != 0
    ]
    if nonzero_errors:
        p3_complete = False
        p3_failures.extend("protocol_error:" + key for key in nonzero_errors)
        p3_failures = list(dict.fromkeys(p3_failures))
        p4_gate = derive_p4_gate(
            p3_integration_complete=False,
            forecast_state=forecast_state,
            classification_state=classification_state,
            anomaly_state=anomaly_state,
            ad_method_gate_passed=ad_gate_passed,
        )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "stage": STAGE,
        "evidence_grade": "INTEGRATION_WITH_CONTROLLED_POLICY_REPLAY",
        "verdict": (
            "P3_UNIFIED_VERTICAL_INTEGRATION_PASS__P4_HELD"
            if p3_complete and not p4_gate["release_p4"]
            else "P3_UNIFIED_VERTICAL_INTEGRATION_PASS__P4_RELEASED"
            if p3_complete
            else "P3_UNIFIED_VERTICAL_INTEGRATION_BLOCKED"
        ),
        "p3_integration_complete": p3_complete,
        "release_p4": bool(p4_gate["release_p4"]),
        "p4_executed": False,
        "live_outcome_release": False,
        "release_scope": p4_gate["release_scope"],
        "upstream": {
            "p0_roster": data_roster,
            "p1": p1_summary,
            "p2_forecast": p2_summary,
        },
        "roster_contract": {
            "tasks": list(TASKS),
            "arms": list(ARMS),
            "methods": list(MANDATORY_METHODS),
            "data": data_roster,
            "forecast_mode": "READ_ONLY_REUSE_P2",
            "classification_mode": "EXPOSED_TRAIN_MACRO_F1_CONTROLLED_SCOPE_POLICY_REPLAY",
            "anomaly_detection_mode": "EXPOSED_YAHOO24_IDENTITY_ONLY_METHOD_GATE",
        },
        "budget_caps": {
            "full_support_evaluations": B_MAIN,
            "support_a_full_evaluations": MAX_SUPPORT_A,
            "support_b_full_evaluations": MAX_SUPPORT_B,
            "cheap_probes": MAX_CHEAP_PROBES,
            "llm_calls": MAX_LLM_CALLS,
            "tokens": MAX_TOKENS,
            "accepted_updates": MAX_UPDATES,
            "raw_consumer_fits": "REPORTED_SEPARATELY_NOT_A_B4_GATE",
        },
        "task_components": {
            "forecast": {
                "mode": "READ_ONLY_REUSE_P2",
                "treatment_facts": forecast_facts,
                "treatment_state": forecast_state,
                "claim_ceiling": "RISK_CONTROL_ONLY",
                "p3_incremental_cost": _usage(
                    support_a=0, support_b=0, raw_a=0, raw_b=0
                ),
                "boundaries": {
                    "forecast_runner_invocations": 0,
                    "natural_final_outcome_reads": 0,
                },
            },
            "classification": classification,
            "anomaly_detection": anomaly,
        },
        "derived_treatment_state": {
            "forecast": forecast_state,
            "classification": classification_state,
            "anomaly_detection": anomaly_state,
        },
        "revision_concern_gate": {
            "status": "PARTIAL_MECHANICAL_SCOPE_REPLAY__PRODUCTION_REVISION_PENDING",
            "revoke_only_is_sufficient": False,
            "required_chain": [
                "positive qualification",
                "later material conflict",
                "bounded production nonterminal revision",
                "independent delayed revalidation and promotion",
                "later independent reencounter behavior change",
                "surviving useful context",
            ],
            "observed_semantic_edit": "SCOPE_NARROW",
            "production_update_observed": False,
            "revision_counter_increment_observed": False,
            "independent_delayed_approval_observed": False,
            "independent_reencounter_observed": False,
            "same_surface_policy_replay_observed": (
                bool(
                    (classification.get("treatment_facts") or {}).get(
                        "controlled_semantic_edit"
                    )
                )
            ),
            "controlled_mechanism_only": True,
            "natural_skill_evolution_claim": False,
            "remaining_requirements": [
                "production pending update",
                "independent Support-B approval",
                "promoted revision with incremented version",
                "independent later-context survival or benefit",
            ],
            "p4_failure_rule": (
                "if formal Evolution only generates and revokes, classify it as "
                "risk control rather than sustained Skill evolution"
            ),
        },
        "unified_cost_rows": cost_rows,
        "budget_gate": {
            "status": "PASS" if not costs_failed else "FAIL",
            "failures": costs_failed,
            "forecast_p2_cost_reused_not_recharged": True,
        },
        "protocol_errors": protocol_errors,
        "blocking_failures": p3_failures,
        "p4_gate": p4_gate,
        "claim_boundaries": {
            "p3_integration_claim": p3_complete,
            "classification_scope_policy_replay": (
                bool(
                    (classification.get("treatment_facts") or {}).get(
                        "controlled_semantic_edit"
                    )
                )
            ),
            "classification_nonterminal_production_revision_claim": False,
            "classification_autonomous_diagnosis_claim": False,
            "natural_data_performance_improvement_claim": False,
            "source_experience_learning_claim": False,
            "cross_dataset_transfer_claim": False,
            "ad_safe_behavior_claim": False,
            "p4_evolution_release": bool(p4_gate["release_p4"]),
            "natural_final_release": False,
        },
        "wall_seconds": round(time.time() - started, 3),
    }


def run() -> dict[str, Any]:
    payload = build_report()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect-integration-pass",
        action="store_true",
        help="return non-zero unless the P3 integration gate passes",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = run()
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "p3_integration_complete": payload["p3_integration_complete"],
                "release_p4": payload["release_p4"],
                "p4_blockers": payload["p4_gate"]["blocking_failures"],
                "output": OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return int(
        bool(args.expect_integration_pass and not payload["p3_integration_complete"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
