"""Plan and evaluate the lean E2 natural policy-transfer pilot.

This runner is deliberately split in two.  ``--phase plan`` may inspect only a
promoted source gate, the already-exposed UCI Support-B development cache, and
Dev-Query contexts.  It freezes every policy and per-query action without
reading a Query target.  ``--phase evaluate`` consumes that plan, validates it,
then performs one complete three-program pass over the frozen Dev-Query roster.

Support-B is reused only as an exposed development feedback cache.  It is not a
fresh confirmation cohort.  Final-Query is not selectable by this module.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _slot
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.split import (
    SplitAssignment,
    SplitManifest,
    SplitRole,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e1p_periodic_missing import (
    PROGRAM_IDS,
    _execute_program,
    _global_features,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.valuation.chronos import (
    FrozenChronosValuator,
)


DATASET_ID = "uci_electricity_load_diagrams"
BUDGETS = (0, 2, 4, 8)
ARMS = ("a3_target_only", "a4_source_only", "a5_source_plus_target")
BUCKETS = ("short", "daily", "long")
QUERY_CONTEXT = (784, 976)
QUERY_TARGET = (976, 1024)
GAP_BOUNDS = (156, 180)
QUERY_STRATA = {
    "seasonal_high": 8,
    "structured_mixed": 4,
    "trend_high": 3,
    "low_structure": 1,
}
GAIN_MIN = 0.005
HARM_MARGIN = 0.005
HARM_RATE_MAX = 0.25
TARGET_MIN_AFFECTED = 2
PLAN_SCHEMA = "e2-natural-policy-plan/1"
REPORT_SCHEMA = "e2-natural-policy-transfer/1"
SOURCE_GATE_SCHEMA = "e2-natural-source-promotion/1"
SUPPORT_CACHE_SCHEMA = "e2-natural-periodic-missing-headroom/1"


class _Receipt(Protocol):
    loss_j: float


class _Valuator(Protocol):
    def evaluate(
        self,
        prepared_context: np.ndarray,
        clean_future: np.ndarray,
        *,
        scale_context: np.ndarray,
    ) -> _Receipt: ...


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _bucket(observed_period: int) -> str:
    period = int(observed_period)
    if period < 20:
        return "short"
    if period < 25:
        return "daily"
    return "long"


def _require_policy(value: object, *, name: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(BUCKETS):
        raise ValueError(f"{name} must map exactly the three frozen period buckets")
    policy = {bucket: str(value[bucket]) for bucket in BUCKETS}
    if any(action not in PROGRAM_IDS for action in policy.values()):
        raise ValueError(f"{name} contains an action outside the fixed program menu")
    return policy


def _load_source_gate(payload: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    if payload.get("schema_version") != SOURCE_GATE_SCHEMA:
        raise ValueError("source gate has an unsupported schema")
    if payload.get("policy_status") != "PROMOTED":
        raise ValueError("source gate must be PROMOTED before policy planning")
    baseline = _require_policy(
        payload.get("baseline_policy"), name="source baseline_policy"
    )
    policy = _require_policy(
        payload.get("candidate_policy"), name="source candidate_policy"
    )
    if baseline != {bucket: "identity" for bucket in BUCKETS}:
        raise ValueError("source gate baseline must be the frozen identity policy")
    expected = {"short": "identity", "daily": "identity", "long": "seasonal"}
    if policy != expected:
        raise ValueError("source gate does not bind the frozen long-period capability")
    return policy, {
        "schema_version": SOURCE_GATE_SCHEMA,
        "policy_status": "PROMOTED",
        "baseline_policy": baseline,
        "source_policy": policy,
        "capability_id": str(
            payload.get("capability_id", "long_period_seasonal_v1")
        ),
    }


def _load_support_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != SUPPORT_CACHE_SCHEMA:
        raise ValueError("target support cache has an unsupported schema")
    if payload.get("all_gates_pass") is not True:
        raise ValueError("target support positive-control gates did not pass")
    configuration = payload.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("target support cache lacks configuration")
    if (
        configuration.get("dataset_id") != DATASET_ID
        or configuration.get("split") != SplitRole.SUPPORT_B.value
    ):
        raise ValueError("target support cache is not the exposed UCI Support-B cache")
    raw_rows = payload.get("cases")
    if not isinstance(raw_rows, list) or len(raw_rows) != 8:
        raise ValueError("target support cache must contain exactly eight cases")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise TypeError("target support cache contains a non-object case")
        uid = str(raw.get("series_uid", ""))
        public = raw.get("public_features")
        losses = raw.get("loss_by_action")
        if not uid or uid in seen:
            raise ValueError("target support cache has a missing or duplicate UID")
        if not isinstance(public, Mapping) or "observed_period" not in public:
            raise ValueError(f"support case lacks observed period: {uid}")
        if not isinstance(losses, Mapping) or set(losses) != set(PROGRAM_IDS):
            raise ValueError(f"support case lacks the complete program menu: {uid}")
        loss_map = {action: float(losses[action]) for action in PROGRAM_IDS}
        if not all(np.isfinite(value) for value in loss_map.values()):
            raise ValueError(f"support case contains non-finite feedback: {uid}")
        period = int(public["observed_period"])
        rows.append(
            {
                "series_uid": uid,
                "observed_period": period,
                "bucket": _bucket(period),
                "loss_by_action": loss_map,
            }
        )
        seen.add(uid)
    return rows


def _compile_target_policy(
    base_policy: Mapping[str, str],
    support_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Fit disjoint bucket patches and credit each against the complete base policy."""

    base = _require_policy(base_policy, name="base_policy")
    policy = dict(base)
    receipts: list[dict[str, Any]] = []
    for bucket in BUCKETS:
        affected = [row for row in support_rows if row["bucket"] == bucket]
        baseline_action = base[bucket]
        estimates: dict[str, float] = {}
        for action in PROGRAM_IDS:
            estimates[action] = (
                0.0
                if not affected
                else statistics.fmean(
                    float(row["loss_by_action"][baseline_action])
                    - float(row["loss_by_action"][action])
                    for row in affected
                )
            )
        candidate = max(PROGRAM_IDS, key=lambda action: (estimates[action], -PROGRAM_IDS.index(action)))
        gains = [
            float(row["loss_by_action"][baseline_action])
            - float(row["loss_by_action"][candidate])
            for row in affected
        ]
        affected_gain = statistics.fmean(gains) if gains else 0.0
        harm_rate = (
            sum(gain < -HARM_MARGIN for gain in gains) / len(gains) if gains else 0.0
        )
        full_gains = []
        for row in support_rows:
            old_action = base[str(row["bucket"])]
            new_action = candidate if row["bucket"] == bucket else old_action
            full_gains.append(
                float(row["loss_by_action"][old_action])
                - float(row["loss_by_action"][new_action])
            )
        full_gain = statistics.fmean(full_gains) if full_gains else 0.0
        admitted = (
            candidate != baseline_action
            and len(affected) >= TARGET_MIN_AFFECTED
            and affected_gain >= GAIN_MIN
            and harm_rate <= HARM_RATE_MAX
            and full_gain >= 0.0
        )
        if admitted:
            policy[bucket] = candidate
        receipts.append(
            {
                "bucket": bucket,
                "baseline_action": baseline_action,
                "candidate_action": candidate,
                "affected_count": len(affected),
                "affected_mean_gain": float(affected_gain),
                "harm_rate_at_margin": float(harm_rate),
                "full_prefix_mean_gain": float(full_gain),
                "admitted": admitted,
                "verdict": "TARGET_LOCAL_ADMITTED" if admitted else "NOT_ADMITTED",
                "evidence_scope": "exposed_target_support_development_cache",
                "transferable_promotion": False,
            }
        )
    return policy, receipts


def _select_query_roster(
    registry_path: Path,
    split_path: Path,
) -> tuple[list[tuple[SeriesRecord, SplitAssignment]], dict[str, Any]]:
    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    candidates: list[tuple[SeriesRecord, SplitAssignment]] = []
    for assignment in manifest.assignments:
        if assignment.dataset_id != DATASET_ID or assignment.role is not SplitRole.DEV_QUERY:
            continue
        record = records.get(assignment.series_uid)
        if record is None:
            raise ValueError(f"Dev-Query UID is absent from registry: {assignment.series_uid}")
        if record.dataset_id != assignment.dataset_id or record.regime_tag != assignment.regime_tag:
            raise ValueError(f"registry/split mismatch: {assignment.series_uid}")
        if record.admission_reasons != ():
            raise ValueError(f"ineligible Dev-Query record: {assignment.series_uid}")
        if SplitRole.DEV_QUERY.value not in record.roles_allowed:
            raise ValueError(f"record disallows Dev-Query: {assignment.series_uid}")
        candidates.append((record, assignment))
    candidates.sort(key=lambda pair: pair[0].series_uid)

    selected: list[tuple[SeriesRecord, SplitAssignment]] = []
    selected_uids: set[str] = set()
    deficits: dict[str, int] = {}
    for stratum, desired in QUERY_STRATA.items():
        matches = [
            pair
            for pair in candidates
            if pair[1].regime_tag == stratum and pair[0].series_uid not in selected_uids
        ]
        chosen = matches[:desired]
        selected.extend(chosen)
        selected_uids.update(pair[0].series_uid for pair in chosen)
        if len(chosen) < desired:
            deficits[stratum] = desired - len(chosen)
    fallback_needed = sum(QUERY_STRATA.values()) - len(selected)
    if fallback_needed:
        fallback = [pair for pair in candidates if pair[0].series_uid not in selected_uids][
            :fallback_needed
        ]
        selected.extend(fallback)
    if len(selected) != 16:
        raise ValueError("fewer than sixteen eligible UCI Dev-Query records")
    actual = Counter(pair[1].regime_tag for pair in selected)
    return selected, {
        "method": "frozen_stratum_quota_then_series_uid_ascending",
        "desired_strata": dict(QUERY_STRATA),
        "actual_strata": dict(sorted(actual.items())),
        "candidate_count": len(candidates),
        "fallback_used": bool(deficits),
        "fallback_deficits": deficits,
    }


def _validate_boundaries(assignment: SplitAssignment) -> None:
    bounds = assignment.chronological_boundaries
    if bounds is None:
        raise ValueError(f"missing chronological boundaries: {assignment.series_uid}")
    expected = {"train": (0, 928), "validation": (928, 976), "test": (976, 1024)}
    if any(tuple(bounds.get(name, ())) != interval for name, interval in expected.items()):
        raise ValueError(f"unexpected chronological boundaries: {assignment.series_uid}")


def _context_descriptor(
    record: SeriesRecord,
    assignment: SplitAssignment,
    clean_root: Path,
) -> dict[str, Any]:
    _validate_boundaries(assignment)
    path = _slot(record, clean_root) / "values.npy"
    mapped = np.load(path, allow_pickle=False, mmap_mode="r")
    if mapped.ndim != 1 or mapped.shape[0] < QUERY_CONTEXT[1]:
        raise ValueError(f"invalid Query value array: {record.series_uid}")
    clean_context = np.asarray(mapped[slice(*QUERY_CONTEXT)], dtype=np.float64).copy()
    del mapped
    if clean_context.shape != (192,) or not np.isfinite(clean_context).all():
        raise ValueError(f"invalid Query context: {record.series_uid}")
    corrupt = clean_context.copy()
    corrupt[slice(*GAP_BOUNDS)] = np.nan
    _, observed_period = _global_features(corrupt)
    return {
        "series_uid": record.series_uid,
        "regime_tag_report_only": assignment.regime_tag,
        "observed_period": observed_period,
        "bucket": _bucket(observed_period),
    }


def _assert_plan_has_no_outcome_fields(value: object, *, path: str = "plan") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if "loss" in lowered or "future" in lowered:
                raise ValueError(f"forbidden Query-plan field: {path}.{key}")
            _assert_plan_has_no_outcome_fields(nested, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _assert_plan_has_no_outcome_fields(nested, path=f"{path}[{index}]")


def build_policy_plan_from_payloads(
    source_gate: Mapping[str, Any],
    support_cache: Mapping[str, Any],
    *,
    registry_path: Path,
    split_path: Path,
    clean_root: Path,
) -> dict[str, Any]:
    source_policy, source_snapshot = _load_source_gate(source_gate)
    support_rows = _load_support_rows(support_cache)
    roster, roster_receipt = _select_query_roster(registry_path, split_path)
    query = [
        _context_descriptor(record, assignment, clean_root)
        for record, assignment in roster
    ]

    policies: dict[str, Any] = {}
    support_receipts: dict[str, Any] = {}
    query_actions: dict[str, Any] = {}
    support_prefixes: dict[str, list[str]] = {}
    identity = {bucket: "identity" for bucket in BUCKETS}
    for budget in BUDGETS:
        prefix = support_rows[:budget]
        a3, a3_receipts = _compile_target_policy(identity, prefix)
        a5, a5_receipts = _compile_target_policy(source_policy, prefix)
        arm_policies = {
            "a3_target_only": a3,
            "a4_source_only": dict(source_policy),
            "a5_source_plus_target": a5,
        }
        policies[str(budget)] = arm_policies
        support_receipts[str(budget)] = {
            "a3_target_only": a3_receipts,
            "a5_source_plus_target": a5_receipts,
        }
        support_prefixes[str(budget)] = [str(row["series_uid"]) for row in prefix]
        query_actions[str(budget)] = {
            arm: {
                str(row["series_uid"]): policy[str(row["bucket"])]
                for row in query
            }
            for arm, policy in arm_policies.items()
        }

    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "phase": "PLAN_FROZEN",
        "scientific_role": "development_policy_transfer_plan",
        "configuration": {
            "dataset_id": DATASET_ID,
            "target_feedback_role": SplitRole.SUPPORT_B.value,
            "query_role": SplitRole.DEV_QUERY.value,
            "final_query_access": "FORBIDDEN_AND_UNREFERENCED",
            "budgets": list(BUDGETS),
            "programs": list(PROGRAM_IDS),
            "period_buckets": {
                "short": "observed_period < 20",
                "daily": "20 <= observed_period < 25",
                "long": "observed_period >= 25",
            },
            "context_interval": list(QUERY_CONTEXT),
            "target_interval_opened_only_by_evaluate": list(QUERY_TARGET),
            "gap_relative_to_context": list(GAP_BOUNDS),
            "target_patch_gate": {
                "minimum_affected": TARGET_MIN_AFFECTED,
                "affected_mean_gain_min": GAIN_MIN,
                "harm_margin": HARM_MARGIN,
                "harm_rate_max": HARM_RATE_MAX,
                "full_prefix_mean_gain_min": 0.0,
            },
        },
        "source_gate": source_snapshot,
        "target_feedback_receipt": {
            "status": "EXPOSED_DEVELOPMENT_CACHE",
            "fresh_confirmation": False,
            "case_count": len(support_rows),
            "fixed_reveal_order": [str(row["series_uid"]) for row in support_rows],
            "bucket_counts": dict(sorted(Counter(row["bucket"] for row in support_rows).items())),
        },
        "query_roster_receipt": roster_receipt,
        "query_context_descriptors": query,
        "support_prefix_uids_by_budget": support_prefixes,
        "policy_snapshots_by_budget": policies,
        "target_local_admission_receipts": support_receipts,
        "query_actions_by_budget": query_actions,
        "information_wall": {
            "query_context_read": True,
            "query_target_read": False,
            "query_outcome_read": False,
            "plans_frozen_before_evaluation": True,
        },
        "claim_limit": (
            "Frozen development plan only. Target-local admitted patches are not "
            "transferable promotions, and exposed Support-B is not fresh confirmation."
        ),
    }
    _assert_plan_has_no_outcome_fields(plan)
    return plan


def build_policy_plan(
    *,
    source_gate_path: Path,
    support_cache_path: Path,
    registry_path: Path,
    split_path: Path,
    clean_root: Path,
) -> dict[str, Any]:
    return build_policy_plan_from_payloads(
        _read_json(source_gate_path),
        _read_json(support_cache_path),
        registry_path=registry_path,
        split_path=split_path,
        clean_root=clean_root,
    )


def _validate_plan(
    plan: Mapping[str, Any],
    *,
    registry_path: Path,
    split_path: Path,
) -> tuple[list[tuple[SeriesRecord, SplitAssignment]], list[Mapping[str, Any]]]:
    _assert_plan_has_no_outcome_fields(plan)
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("phase") != "PLAN_FROZEN":
        raise ValueError("evaluate requires a frozen E2 policy plan")
    source = plan.get("source_gate")
    if not isinstance(source, Mapping) or source.get("policy_status") != "PROMOTED":
        raise ValueError("plan does not contain a promoted source gate")
    source_policy = _require_policy(source.get("source_policy"), name="planned source policy")
    if source_policy != {"short": "identity", "daily": "identity", "long": "seasonal"}:
        raise ValueError("planned source policy differs from the frozen capability")

    descriptors = plan.get("query_context_descriptors")
    if not isinstance(descriptors, list) or len(descriptors) != 16:
        raise ValueError("plan must contain sixteen Query context descriptors")
    descriptor_by_uid = {str(row.get("series_uid", "")): row for row in descriptors if isinstance(row, Mapping)}
    if len(descriptor_by_uid) != 16:
        raise ValueError("planned Query descriptor UIDs are missing or duplicated")

    expected_roster, expected_receipt = _select_query_roster(registry_path, split_path)
    expected_uids = [record.series_uid for record, _ in expected_roster]
    planned_uids = [str(row["series_uid"]) for row in descriptors]
    if planned_uids != expected_uids:
        raise ValueError("planned Query roster differs from deterministic selection")
    if plan.get("query_roster_receipt") != expected_receipt:
        raise ValueError("planned Query roster receipt differs from frozen metadata")

    records = {row.series_uid: row for row in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    assignments = {row.series_uid: row for row in manifest.assignments}
    roster: list[tuple[SeriesRecord, SplitAssignment]] = []
    for raw in descriptors:
        uid = str(raw["series_uid"])
        record = records.get(uid)
        assignment = assignments.get(uid)
        if record is None or assignment is None:
            raise ValueError(f"planned Query UID is absent from frozen inputs: {uid}")
        if assignment.role is not SplitRole.DEV_QUERY or assignment.dataset_id != DATASET_ID:
            raise ValueError(f"planned Query UID has the wrong role: {uid}")
        if raw.get("regime_tag_report_only") != assignment.regime_tag:
            raise ValueError(f"planned Query stratum differs from split: {uid}")
        period = int(raw.get("observed_period", -1))
        if raw.get("bucket") != _bucket(period):
            raise ValueError(f"planned Query bucket is malformed: {uid}")
        _validate_boundaries(assignment)
        roster.append((record, assignment))

    policies = plan.get("policy_snapshots_by_budget")
    actions = plan.get("query_actions_by_budget")
    if not isinstance(policies, Mapping) or not isinstance(actions, Mapping):
        raise ValueError("plan lacks policy snapshots or Query actions")
    for budget in BUDGETS:
        key = str(budget)
        if key not in policies or key not in actions:
            raise ValueError(f"plan lacks budget {budget}")
        if not isinstance(policies[key], Mapping) or not isinstance(actions[key], Mapping):
            raise ValueError(f"malformed policy budget {budget}")
        for arm in ARMS:
            policy = _require_policy(policies[key].get(arm), name=f"{arm}@{budget}")
            arm_actions = actions[key].get(arm)
            if not isinstance(arm_actions, Mapping) or set(arm_actions) != set(descriptor_by_uid):
                raise ValueError(f"{arm}@{budget} does not bind the complete Query roster")
            for uid, descriptor in descriptor_by_uid.items():
                expected = policy[str(descriptor["bucket"])]
                if arm_actions[uid] != expected:
                    raise ValueError(f"Query action differs from policy snapshot: {arm}/{budget}/{uid}")
    return roster, descriptors


def _curve_auc(values: Mapping[int, float]) -> float:
    area = 0.0
    for left, right in zip(BUDGETS, BUDGETS[1:]):
        area += 0.5 * (values[left] + values[right]) * (right - left)
    return area / float(BUDGETS[-1] - BUDGETS[0])


def _harm_summary(candidate: Sequence[float], baseline: Sequence[float]) -> dict[str, float]:
    deltas = [cand - base for cand, base in zip(candidate, baseline)]
    return {
        "mean_delta": float(statistics.fmean(deltas)),
        "harm_rate_at_margin": sum(delta > HARM_MARGIN for delta in deltas) / len(deltas),
    }


def evaluate_policy_plan(
    plan: Mapping[str, Any],
    valuator: _Valuator,
    *,
    registry_path: Path,
    split_path: Path,
    clean_root: Path,
) -> dict[str, Any]:
    roster, descriptors = _validate_plan(
        plan, registry_path=registry_path, split_path=split_path
    )

    # Validate every context-derived binding before opening any Query target.
    contexts: list[tuple[SeriesRecord, Mapping[str, Any], np.ndarray, np.ndarray]] = []
    for (record, _), descriptor in zip(roster, descriptors):
        mapped = np.load(
            _slot(record, clean_root) / "values.npy", allow_pickle=False, mmap_mode="r"
        )
        clean_context = np.asarray(mapped[slice(*QUERY_CONTEXT)], dtype=np.float64).copy()
        del mapped
        if clean_context.shape != (192,) or not np.isfinite(clean_context).all():
            raise ValueError(f"invalid Query context: {record.series_uid}")
        corrupt = clean_context.copy()
        corrupt[slice(*GAP_BOUNDS)] = np.nan
        _, observed_period = _global_features(corrupt)
        if observed_period != int(descriptor["observed_period"]):
            raise ValueError(f"Query context changed after plan freeze: {record.series_uid}")
        contexts.append((record, descriptor, clean_context, corrupt))

    rows: list[dict[str, Any]] = []
    judge_calls = 0
    for record, descriptor, clean_context, corrupt in contexts:
        mapped = np.load(
            _slot(record, clean_root) / "values.npy", allow_pickle=False, mmap_mode="r"
        )
        clean_target = np.asarray(mapped[slice(*QUERY_TARGET)], dtype=np.float64).copy()
        del mapped
        if clean_target.shape != (48,) or not np.isfinite(clean_target).all():
            raise ValueError(f"invalid Query evaluation window: {record.series_uid}")
        observed_period = int(descriptor["observed_period"])
        losses: dict[str, float] = {}
        for action in PROGRAM_IDS:
            prepared = _execute_program(action, corrupt, observed_period=observed_period)
            receipt = valuator.evaluate(
                prepared, clean_target, scale_context=clean_context
            )
            judge_calls += 1
            value = float(receipt.loss_j)
            if not np.isfinite(value):
                raise ValueError(f"non-finite Query result: {record.series_uid}/{action}")
            losses[action] = value
        rows.append(
            {
                "series_uid": record.series_uid,
                "regime_tag_report_only": descriptor["regime_tag_report_only"],
                "observed_period": observed_period,
                "bucket": descriptor["bucket"],
                "loss_by_action": losses,
                "menu_oracle_loss": min(losses.values()),
            }
        )
    if judge_calls != 48:
        raise AssertionError(f"expected exactly 48 Query Judge calls, observed {judge_calls}")

    fixed_means = {
        action: statistics.fmean(row["loss_by_action"][action] for row in rows)
        for action in PROGRAM_IDS
    }
    best_fixed = min(PROGRAM_IDS, key=lambda action: (fixed_means[action], action))
    oracle_mean = statistics.fmean(row["menu_oracle_loss"] for row in rows)
    plan_actions = plan["query_actions_by_budget"]
    by_budget: dict[str, Any] = {}
    regret_curves: dict[str, dict[int, float]] = {arm: {} for arm in ARMS}
    selected_by_budget: dict[tuple[int, str], list[float]] = {}
    for budget in BUDGETS:
        arm_results: dict[str, Any] = {}
        for arm in ARMS:
            choices = plan_actions[str(budget)][arm]
            selected = [row["loss_by_action"][choices[row["series_uid"]]] for row in rows]
            selected_by_budget[(budget, arm)] = selected
            mean_value = statistics.fmean(selected)
            regret = mean_value - oracle_mean
            regret_curves[arm][budget] = regret
            arm_results[arm] = {
                "mean_query_loss": float(mean_value),
                "mean_oracle_regret": float(regret),
                "action_distribution": dict(sorted(Counter(choices.values()).items())),
            }
        by_budget[str(budget)] = {"arms": arm_results}

    auc = {arm: _curve_auc(regret_curves[arm]) for arm in ARMS}
    identity_losses = [row["loss_by_action"]["identity"] for row in rows]
    a4_zero = selected_by_budget[(0, "a4_source_only")]
    zero_gain = statistics.fmean(
        raw - selected for raw, selected in zip(identity_losses, a4_zero)
    )
    zero_harm = _harm_summary(a4_zero, identity_losses)
    a3_b8 = selected_by_budget[(8, "a3_target_only")]
    a4_b8 = selected_by_budget[(8, "a4_source_only")]
    a5_b8 = selected_by_budget[(8, "a5_source_plus_target")]
    a5_gain_over_a4 = statistics.fmean(base - cand for base, cand in zip(a4_b8, a5_b8))
    a5_gain_over_a3 = statistics.fmean(base - cand for base, cand in zip(a3_b8, a5_b8))
    a5_harm_vs_a4 = _harm_summary(a5_b8, a4_b8)
    a5_harm_vs_a3 = _harm_summary(a5_b8, a3_b8)
    auc_gain = auc["a3_target_only"] - auc["a5_source_plus_target"]

    receipts = plan["target_local_admission_receipts"]
    b8_a5_receipts = receipts["8"]["a5_source_plus_target"]
    b8_patches = [receipt for receipt in b8_a5_receipts if receipt["admitted"]]
    all_patch_count = sum(
        receipt["admitted"]
        for budget in map(str, BUDGETS)
        for receipt in receipts[budget]["a5_source_plus_target"]
    )
    source_transfer_supported = (
        zero_gain >= GAIN_MIN
        and zero_harm["harm_rate_at_margin"] <= HARM_RATE_MAX
        and auc_gain >= GAIN_MIN
    )
    target_adaptation_supported = (
        bool(b8_patches)
        and a5_gain_over_a4 >= GAIN_MIN
        and a5_harm_vs_a4["harm_rate_at_margin"] <= HARM_RATE_MAX
        and a5_harm_vs_a3["mean_delta"] <= GAIN_MIN
        and a5_harm_vs_a3["harm_rate_at_margin"] <= HARM_RATE_MAX
    )

    return {
        "schema_version": REPORT_SCHEMA,
        "scientific_role": "one_pass_dev_query_policy_transfer_pilot",
        "plan_schema_version": plan["schema_version"],
        "configuration": {
            "dataset_id": DATASET_ID,
            "support_role": "EXPOSED_SUPPORT_B_DEVELOPMENT_CACHE",
            "query_role": SplitRole.DEV_QUERY.value,
            "final_query_access": "NONE",
            "budgets": list(BUDGETS),
            "programs": list(PROGRAM_IDS),
            "context": list(QUERY_CONTEXT),
            "target": list(QUERY_TARGET),
            "gap_relative_to_context": list(GAP_BOUNDS),
        },
        "judge_call_count": judge_calls,
        "query_cases": rows,
        "diagnostics": {
            "fixed_mean_loss_by_action": fixed_means,
            "best_fixed_action": best_fixed,
            "best_fixed_mean_loss": fixed_means[best_fixed],
            "menu_oracle_mean_loss": oracle_mean,
            "best_fixed_minus_menu_oracle": fixed_means[best_fixed] - oracle_mean,
        },
        "budget_results": by_budget,
        "oracle_regret_auc": auc,
        "comparisons": {
            "zero_shot_a4_vs_identity": {
                "mean_gain": float(zero_gain),
                **zero_harm,
            },
            "a3_minus_a5_regret_auc": float(auc_gain),
            "a5_b8_vs_a4": {
                "mean_gain": float(a5_gain_over_a4),
                **a5_harm_vs_a4,
            },
            "a5_b8_vs_a3": {
                "mean_gain": float(a5_gain_over_a3),
                **a5_harm_vs_a3,
            },
        },
        "target_patch_summary": {
            "admitted_patch_count_across_budget_snapshots": int(all_patch_count),
            "b8_admitted_patch_count": len(b8_patches),
            "b8_admitted_patches": b8_patches,
            "b8_query_value_over_source_policy": float(a5_gain_over_a4),
        },
        "policy_snapshots_by_budget": plan["policy_snapshots_by_budget"],
        "target_local_admission_receipts": plan[
            "target_local_admission_receipts"
        ],
        "source_transfer_supported": source_transfer_supported,
        "target_adaptation_supported": target_adaptation_supported,
        "stop_conditions": {
            "source_transfer_not_supported": not source_transfer_supported,
            "target_adaptation_not_supported": not target_adaptation_supported,
            "no_b8_target_patch": not bool(b8_patches),
            "a5_regret_auc_not_better_than_a3_by_0_005": auc_gain < GAIN_MIN,
            "a5_b8_not_better_than_a4_by_0_005": a5_gain_over_a4 < GAIN_MIN,
            "a5_b8_harm_rate_vs_a3_exceeds_0_25": (
                a5_harm_vs_a3["harm_rate_at_margin"] > HARM_RATE_MAX
            ),
        },
        "information_wall": {
            "plan_was_outcome_free": True,
            "query_actions_frozen_before_target_read": True,
            "query_outcomes_written_back_to_plan": False,
            "single_complete_query_pass": True,
        },
        "claim_limit": (
            "Development evidence only. The three-bucket compiler is a deterministic, "
            "router-like mechanism probe; target-local admitted patches are not "
            "transferable capabilities, Support-B is already exposed, and Final-Query "
            "remains sealed."
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("plan", "evaluate"), required=True)
    parser.add_argument(
        "--source-gate",
        type=Path,
        default=project_root
        / "artifacts/functional/e2/natural_source_promotion_report.json",
    )
    parser.add_argument(
        "--support-cache",
        type=Path,
        default=project_root / "artifacts/functional/e2/natural_periodic_missing_headroom_report.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/series_registry.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=project_root / "artifacts/frozen/benchmark_v02/split_manifest.json",
    )
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=project_root / "data/benchmark_v0_2/clean_base",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=project_root / "artifacts/functional/e2/natural_policy_plan.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "artifacts/functional/e2/natural_policy_transfer_report.json",
    )
    args = parser.parse_args()

    if args.phase == "plan":
        plan = build_policy_plan(
            source_gate_path=args.source_gate,
            support_cache_path=args.support_cache,
            registry_path=args.registry,
            split_path=args.split,
            clean_root=args.clean_root,
        )
        args.plan.parent.mkdir(parents=True, exist_ok=True)
        args.plan.write_bytes(canonical_json_bytes(plan) + b"\n")
        print(f"plan={args.plan.resolve()}")
        print("query_target_read=False")
        return 0

    plan = _read_json(args.plan)
    report = evaluate_policy_plan(
        plan,
        FrozenChronosValuator(),
        registry_path=args.registry,
        split_path=args.split,
        clean_root=args.clean_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(f"report={args.output.resolve()}")
    print(f"source_transfer_supported={report['source_transfer_supported']}")
    print(f"target_adaptation_supported={report['target_adaptation_supported']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_policy_plan",
    "build_policy_plan_from_payloads",
    "evaluate_policy_plan",
]
