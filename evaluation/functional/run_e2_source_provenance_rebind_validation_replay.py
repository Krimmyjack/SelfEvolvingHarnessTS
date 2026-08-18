"""Validate key-only target rebind on Support-A validation Traffic and METR-LA.

The roster is fixed from frozen metadata before selected numeric values are loaded:
within each dataset, sort eligible ``support_a_validation`` members by series UID,
take the first twelve for training and the next eight for evaluation.  This is a
held-out intervention-family replay, not fresh-dataset evidence, formal promotion,
or a Target/Query evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

from SelfEvolvingHarnessTS.contracts.canonical import canonical_json_bytes
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import (
    SeriesRecord,
    read_registry_jsonl,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.split import (
    SplitAssignment,
    SplitManifest,
    SplitRole,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_coherent_missingness_positive_control import (
    CONTEXT_LENGTH,
    EVAL_SERIES_PER_DATASET,
    HORIZON,
    RECENT_V2,
    SUPPORT_A_SUBSPLIT_SCHEMA,
    TRAIN_SERIES_PER_DATASET,
    DatasetSpec,
    RosterItem,
    _evaluation_matrices,
)
from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_provenance_rebind_source_replay import (
    HARM_RATE_MAX as SOURCE_REPLAY_HARM_RATE_MAX,
    HARM_THRESHOLD as SOURCE_REPLAY_HARM_THRESHOLD,
    MEAN_GAIN_MIN as SOURCE_REPLAY_MEAN_GAIN_MIN,
    MEDIAN_GAIN_MIN_EXCLUSIVE as SOURCE_REPLAY_MEDIAN_GAIN_MIN_EXCLUSIVE,
    POLICIES,
    POSITIVE_GAIN_MIN_COUNT as SOURCE_REPLAY_POSITIVE_GAIN_MIN_COUNT,
    TrainingBundle,
    _evidence as _source_replay_evidence,
    _training_bundle,
)


SCHEMA_VERSION = "e2-source-provenance-rebind-validation-replay/1"
VALIDATION_SUBSPLIT = "support_a_validation"
DATASET_SPECS = (
    DatasetSpec(
        dataset_id="monash:traffic_hourly",
        period=24,
        frequency="hourly",
        train_stop=928,
        validation_bounds=(928, 976),
    ),
    DatasetSpec(
        dataset_id="metr_la",
        period=24,
        frequency="hourly",
        train_stop=928,
        validation_bounds=(928, 976),
    ),
)
DATASET_IDS = tuple(spec.dataset_id for spec in DATASET_SPECS)

# These are decision-bearing dependencies, not floating "latest" reports.
CONTRACT_REPORT_SHA256 = (
    "eecd7427748c2be26e8e18c78290b86170c4fea9a4d4477d52080e4c27419f92"
)
PRIOR_REPLAY_REPORT_SHA256 = (
    "b38b25e64013591d7a15bb779459f7f47f802e1bc4568906d20be333d2619cbd"
)

# Frozen unchanged from the FRED/NN5 source replay.
TRAIN_ANCHORS = (240, 300, 360, 420, 480, 540)
MEAN_GAIN_MIN = 0.005
MEDIAN_GAIN_MIN_EXCLUSIVE = 0.0
POSITIVE_GAIN_MIN_COUNT = 5
HARM_THRESHOLD = -0.005
HARM_RATE_MAX = 0.25
EXPECTED_FIT_COUNT = len(DATASET_SPECS) * len(POLICIES)

OUTPUT_RELATIVE_PATH = (
    "artifacts/functional/e2/"
    "source_provenance_rebind_validation_replay_report.json"
)

if tuple(RECENT_V2.anchors) != TRAIN_ANCHORS:
    raise RuntimeError("source replay train anchors changed")
if TRAIN_SERIES_PER_DATASET != 12 or EVAL_SERIES_PER_DATASET != 8:
    raise RuntimeError("source replay roster geometry changed")
if POLICIES != ("positional_incumbent", "key_rebind_repaired"):
    raise RuntimeError("source replay policy pair changed")
if (
    SOURCE_REPLAY_MEAN_GAIN_MIN,
    SOURCE_REPLAY_MEDIAN_GAIN_MIN_EXCLUSIVE,
    SOURCE_REPLAY_POSITIVE_GAIN_MIN_COUNT,
    SOURCE_REPLAY_HARM_THRESHOLD,
    SOURCE_REPLAY_HARM_RATE_MAX,
) != (
    MEAN_GAIN_MIN,
    MEDIAN_GAIN_MIN_EXCLUSIVE,
    POSITIVE_GAIN_MIN_COUNT,
    HARM_THRESHOLD,
    HARM_RATE_MAX,
):
    raise RuntimeError("source replay gates changed")
if EXPECTED_FIT_COUNT != 4:
    raise AssertionError("validation replay must contain exactly four consumer fits")


def _read_validation_uids(path: Path) -> set[str]:
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("schema_version") != SUPPORT_A_SUBSPLIT_SCHEMA:
        raise ValueError("unsupported Support-A subsplit schema")
    members = payload.get("members")
    counts = payload.get("counts")
    if not isinstance(members, dict) or not isinstance(counts, dict):
        raise ValueError("Support-A subsplit members/counts must be objects")
    raw = members.get(VALIDATION_SUBSPLIT)
    if not isinstance(raw, list) or not all(isinstance(uid, str) and uid for uid in raw):
        raise ValueError("invalid support_a_validation member list")
    if len(raw) != len(set(raw)):
        raise ValueError("duplicate support_a_validation UID")
    if counts.get(VALIDATION_SUBSPLIT) != len(raw):
        raise ValueError("support_a_validation count disagrees with frozen metadata")
    return set(raw)


def _validate_candidate(
    record: SeriesRecord, assignment: SplitAssignment, spec: DatasetSpec
) -> None:
    uid = record.series_uid
    if record.dataset_id != spec.dataset_id or assignment.dataset_id != spec.dataset_id:
        raise ValueError(f"candidate dataset mismatch: {uid}")
    if record.regime_tag != assignment.regime_tag:
        raise ValueError(f"registry/split regime mismatch: {uid}")
    if record.admission_reasons != () or record.natural_missing_count != 0:
        raise ValueError(f"ineligible validation record: {uid}")
    if record.frequency != spec.frequency:
        raise ValueError(f"unexpected validation frequency: {uid}")
    if SplitRole.SUPPORT_A.value not in record.roles_allowed:
        raise ValueError(f"record disallows Support-A: {uid}")
    if assignment.role is not SplitRole.SUPPORT_A:
        raise ValueError(f"validation member is not Support-A: {uid}")
    boundaries = assignment.chronological_boundaries
    if boundaries is None:
        raise ValueError(f"missing chronological boundaries: {uid}")
    if tuple(boundaries.get("train", ())) != (0, spec.train_stop):
        raise ValueError(f"unexpected train boundary: {uid}")
    if tuple(boundaries.get("validation", ())) != spec.validation_bounds:
        raise ValueError(f"unexpected validation boundary: {uid}")
    expected_test = (spec.validation_bounds[1], spec.validation_bounds[1] + HORIZON)
    if tuple(boundaries.get("test", ())) != expected_test:
        raise ValueError(f"unexpected test boundary: {uid}")


def select_validation_roster(
    *, registry_path: Path, split_path: Path, support_a_subsplit_path: Path
) -> tuple[list[RosterItem], dict[str, object]]:
    """Fix first-12/next-8 validation UIDs using metadata only."""

    records = {record.series_uid: record for record in read_registry_jsonl(registry_path)}
    manifest = SplitManifest.from_dict(json.loads(split_path.read_text("utf-8")))
    assignments = {row.series_uid: row for row in manifest.assignments}
    validation_uids = _read_validation_uids(support_a_subsplit_path)

    roster: list[RosterItem] = []
    selected_by_dataset: dict[str, dict[str, list[str]]] = {}
    available_by_dataset: dict[str, int] = {}
    overlap_audit: dict[str, dict[str, object]] = {}
    required = TRAIN_SERIES_PER_DATASET + EVAL_SERIES_PER_DATASET
    for spec in DATASET_SPECS:
        candidates: list[tuple[SeriesRecord, SplitAssignment]] = []
        for uid in validation_uids:
            assignment = assignments.get(uid)
            if assignment is None:
                raise ValueError(f"validation UID absent from split manifest: {uid}")
            if assignment.dataset_id != spec.dataset_id:
                continue
            record = records.get(uid)
            if record is None:
                raise ValueError(f"validation UID absent from registry: {uid}")
            _validate_candidate(record, assignment, spec)
            candidates.append((record, assignment))
        candidates.sort(key=lambda pair: pair[0].series_uid)
        available_by_dataset[spec.dataset_id] = len(candidates)
        if len(candidates) < required:
            raise ValueError(
                f"fewer than {required} eligible validation series: {spec.dataset_id}"
            )
        train = candidates[:TRAIN_SERIES_PER_DATASET]
        evaluate = candidates[TRAIN_SERIES_PER_DATASET:required]
        roster.extend(RosterItem(record, assignment, "train") for record, assignment in train)
        roster.extend(RosterItem(record, assignment, "eval") for record, assignment in evaluate)
        train_uids = [record.series_uid for record, _ in train]
        eval_uids = [record.series_uid for record, _ in evaluate]
        selected_by_dataset[spec.dataset_id] = {"train": train_uids, "eval": eval_uids}
        train_groups = {record.overlap_group for record, _ in train}
        eval_groups = {record.overlap_group for record, _ in evaluate}
        shared_groups = sorted(train_groups & eval_groups)
        overlap_audit[spec.dataset_id] = {
            "train_overlap_group_count": len(train_groups),
            "eval_overlap_group_count": len(eval_groups),
            "shared_overlap_groups": shared_groups,
            "overlap_group_disjoint": not shared_groups,
        }

    train_uid_set = {item.record.series_uid for item in roster if item.cohort == "train"}
    eval_uid_set = {item.record.series_uid for item in roster if item.cohort == "eval"}
    if len(train_uid_set) != len(DATASET_SPECS) * TRAIN_SERIES_PER_DATASET:
        raise AssertionError("training roster contains duplicate series")
    if len(eval_uid_set) != len(DATASET_SPECS) * EVAL_SERIES_PER_DATASET:
        raise AssertionError("evaluation roster contains duplicate series")
    if train_uid_set & eval_uid_set:
        raise AssertionError("training and evaluation rosters overlap")
    selected_uid_set = train_uid_set | eval_uid_set
    if not selected_uid_set <= validation_uids:
        raise AssertionError("selected roster escaped support_a_validation")
    return roster, {
        "fixed_before_selected_numeric_value_loading": True,
        "selection_rule": (
            "per dataset, filter frozen support_a_validation to eligible records; "
            "sort by series_uid ascending; first 12 train, next 8 eval"
        ),
        "selection_features": [
            "frozen_support_a_validation_membership",
            "frozen_registry_eligibility",
            "dataset_id",
            "series_uid",
        ],
        "numeric_value_future_or_outcome_used_for_selection": False,
        "available_by_dataset": available_by_dataset,
        "selected_by_dataset": selected_by_dataset,
        "train_eval_series_disjoint": True,
        "support_a_validation_membership": True,
        "overlap_group_audit_by_dataset": overlap_audit,
    }


def _load_dependency(
    path: Path, *, expected_sha256: str, expected_verdict: str
) -> tuple[dict[str, object], dict[str, object]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    actual_sha = hashlib.sha256(raw).hexdigest()
    hash_matches = actual_sha == expected_sha256
    verdict_matches = payload.get("verdict") == expected_verdict
    return payload, {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha,
        "hash_matches": hash_matches,
        "expected_verdict": expected_verdict,
        "actual_verdict": payload.get("verdict"),
        "verdict_matches": verdict_matches,
        "pass": hash_matches and verdict_matches,
    }


def _dependency_uid_strings(payload: object, selected_uids: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, str):
        if payload in selected_uids:
            found.add(payload)
    elif isinstance(payload, list):
        for item in payload:
            found.update(_dependency_uid_strings(item, selected_uids))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            found.update(_dependency_uid_strings(key, selected_uids))
            found.update(_dependency_uid_strings(value, selected_uids))
    return found


def _evidence(
    spec: DatasetSpec,
    losses: dict[str, list[float]],
    uids: list[str],
    train_diagnostics: dict[str, object],
    eval_diagnostics: dict[str, object],
) -> dict[str, object]:
    row = _source_replay_evidence(
        spec, losses, uids, train_diagnostics, eval_diagnostics
    )
    row["scientific_unit"] = "dataset_level_support_a_validation_structural_replay"
    row["train_cohort"]["subsplit"] = VALIDATION_SUBSPLIT
    row["eval_cohort"]["subsplit"] = VALIDATION_SUBSPLIT
    return row


def run_replay(
    registry: Path,
    split: Path,
    subsplit: Path,
    clean_root: Path,
    contract_report_path: Path,
    prior_replay_report_path: Path,
) -> dict[str, object]:
    contract, contract_dependency = _load_dependency(
        contract_report_path,
        expected_sha256=CONTRACT_REPORT_SHA256,
        expected_verdict="PROVENANCE_REBIND_CONTRACT_PASS",
    )
    prior_replay, prior_replay_dependency = _load_dependency(
        prior_replay_report_path,
        expected_sha256=PRIOR_REPLAY_REPORT_SHA256,
        expected_verdict="STRUCTURAL_PROVENANCE_REBIND_SOURCE_REPLAY_PASS",
    )
    prior_gate = prior_replay.get("p1_replay_gate")
    prior_gate_pass = isinstance(prior_gate, dict) and prior_gate.get("pass") is True
    prior_fit_count_matches = prior_replay.get("consumer_fit_count") == 4
    prior_replay_dependency["prior_p1_replay_gate_pass"] = prior_gate_pass
    prior_replay_dependency["prior_consumer_fit_count_is_four"] = prior_fit_count_matches
    prior_replay_dependency["pass"] = (
        bool(prior_replay_dependency["pass"])
        and prior_gate_pass
        and prior_fit_count_matches
    )
    dependency_pass = bool(contract_dependency["pass"]) and bool(
        prior_replay_dependency["pass"]
    )

    # Roster selection is complete before this function performs its only value load.
    roster, selection = select_validation_roster(
        registry_path=registry,
        split_path=split,
        support_a_subsplit_path=subsplit,
    )
    selected_uids = {item.record.series_uid for item in roster}
    prior_family_uid_hits = sorted(
        _dependency_uid_strings(contract, selected_uids)
        | _dependency_uid_strings(prior_replay, selected_uids)
    )
    intervention_family_fresh = not prior_family_uid_hits
    values = _load_values([item.record for item in roster], clean_root)

    bundles: dict[str, TrainingBundle] = {
        spec.dataset_id: _training_bundle(
            spec,
            [
                item
                for item in roster
                if item.record.dataset_id == spec.dataset_id and item.cohort == "train"
            ],
            values,
        )
        for spec in DATASET_SPECS
    }
    p0_by_dataset = {dataset_id: bundle.p0 for dataset_id, bundle in bundles.items()}
    dataset_p0_pass = all(bool(row["pass"]) for row in p0_by_dataset.values())
    p0_pass = dependency_pass and intervention_family_fresh and dataset_p0_pass

    evidence_rows: list[dict[str, object]] = []
    fit_count = 0
    if p0_pass:
        for spec in DATASET_SPECS:
            bundle = bundles[spec.dataset_id]
            eval_items = [
                item
                for item in roster
                if item.record.dataset_id == spec.dataset_id and item.cohort == "eval"
            ]
            x_eval, y_eval, eval_uids, eval_diagnostics = _evaluation_matrices(
                spec=spec, eval_items=eval_items, values_by_uid=values
            )
            losses: dict[str, list[float]] = {}
            targets = {
                POLICIES[0]: bundle.incumbent_y,
                POLICIES[1]: bundle.repaired_y,
            }
            for policy in POLICIES:
                model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
                model.fit(bundle.x_train, targets[policy])
                fit_count += 1
                prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
                if prediction.shape != y_eval.shape or not np.isfinite(prediction).all():
                    raise RuntimeError(f"invalid Ridge prediction: {spec.dataset_id}/{policy}")
                losses[policy] = [
                    float(value)
                    for value in np.mean(np.abs(prediction - y_eval), axis=1)
                ]
            evidence_rows.append(
                _evidence(
                    spec,
                    losses,
                    eval_uids,
                    bundle.diagnostics,
                    eval_diagnostics,
                )
            )
    if p0_pass and fit_count != EXPECTED_FIT_COUNT:
        raise AssertionError("successful exact P0 must lead to exactly four Ridge fits")
    if not p0_pass and fit_count != 0:
        raise AssertionError("failed exact P0 must stop before every consumer fit")

    harms = [
        bool(row["harmed"])
        for evidence in evidence_rows
        for row in evidence["paired_eval_rows"]
    ]
    harm_rate = sum(harms) / len(harms) if harms else None
    gate_pass = (
        p0_pass
        and fit_count == EXPECTED_FIT_COUNT
        and len(evidence_rows) == len(DATASET_SPECS)
        and all(bool(row["dataset_gate"]["pass"]) for row in evidence_rows)
        and harm_rate is not None
        and harm_rate <= HARM_RATE_MAX
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "support_a_validation_intervention_family_structural_replay",
        "configuration": {
            "datasets": list(DATASET_IDS),
            "split": "support_a",
            "subsplit": VALIDATION_SUBSPLIT,
            "policies": list(POLICIES),
            "geometry_id": RECENT_V2.geometry_id,
            "train_anchors": list(TRAIN_ANCHORS),
            "train_series_per_dataset": TRAIN_SERIES_PER_DATASET,
            "eval_series_per_dataset": EVAL_SERIES_PER_DATASET,
            "context_length": CONTEXT_LENGTH,
            "horizon": HORIZON,
            "training_features": "clean normalized 192 values plus 192 zero-mask values",
            "target_binding": "whole TargetRow fixed derangement, then key-only rebind",
            "trusted_unique_provenance_key": [
                "dataset_sha",
                "series_uid",
                "anchor",
                "horizon",
            ],
            "consumer": {
                "class": "sklearn.linear_model.Ridge",
                "alpha": 1.0,
                "fit_intercept": True,
                "solver": "svd",
            },
            "metric": "per-series normalized MAE over the 48-point validation future",
            "expected_consumer_fit_count": EXPECTED_FIT_COUNT,
        },
        "dependencies": {
            "all_required": True,
            "contract_report": contract_dependency,
            "prior_fred_nn5_source_replay_report": prior_replay_dependency,
            "pass": dependency_pass,
        },
        "roster": {
            "selection": selection,
            "selected_numeric_value_series_count": len(roster),
        },
        "intervention_family_uid_exposure_audit": {
            "audited_pinned_reports": [
                str(contract_report_path),
                str(prior_replay_report_path),
            ],
            "selected_uid_count": len(selected_uids),
            "selected_uid_hits_in_prior_family_reports": prior_family_uid_hits,
            "intervention_family_fresh_on_selected_uids": intervention_family_fresh,
        },
        "p0_pre_fit_exact_gate": {
            "dependency_reports_required": True,
            "intervention_family_uid_freshness_required": True,
            "dataset_results": p0_by_dataset,
            "conjunction_across_both_datasets": True,
            "checked_before_any_consumer_fit": True,
            "consumer_fit_count_at_check": 0,
            "failure_action": "zero-fit stop",
            "pass": p0_pass,
        },
        "policy_intervention_evidence": evidence_rows,
        "validation_gate": {
            "frozen_equal_to_fred_nn5_source_replay": True,
            "dataset_gates_conjunctive": True,
            "pooled_harm": {
                "definition": "gain < -0.005",
                "count": sum(harms),
                "total": len(harms),
                "rate": harm_rate,
                "rate_max": HARM_RATE_MAX,
                "pass": harm_rate is not None and harm_rate <= HARM_RATE_MAX,
            },
            "all_conditions_conjunctive": True,
            "pass": gate_pass,
        },
        "information_wall": {
            "roster_fixed_before_selected_numeric_value_loading": True,
            "support_a_validation_only": True,
            "support_a_validation_membership": True,
            "support_a_discovery_values_read": False,
            "support_b_values_read": False,
            "uci_records_selected": False,
            "uci_values_read": False,
            "target_values_read": False,
            "query_values_read": False,
            "target_closed": True,
            "query_closed": True,
            "target_query_opened": False,
        },
        "consumer_fit_count": fit_count,
        "chronos_judge_call_count": 0,
        "intervention_family_fresh_on_selected_uids": intervention_family_fresh,
        "support_a_validation_membership": True,
        "possible_exposure_to_other_intervention_diagnostics": True,
        "other_intervention_exposure_note": (
            "No global diagnostic-exposure exclusion was applied; these Source datasets "
            "and some selected UIDs may have appeared in unrelated development diagnostics."
        ),
        "formal_dataset_fresh": False,
        "formal_dataset_fresh_reason": (
            "Traffic and METR-LA are previously used Source datasets."
        ),
        "formal_promotion_eligible": False,
        "target_query_status": {"target": "closed", "query": "closed"},
        "target_query_opened": False,
        "agent_enabled": False,
        "memory_enabled": False,
        "formal_transfer": False,
        "verdict": (
            "STRUCTURAL_PROVENANCE_REBIND_VALIDATION_PASS"
            if gate_pass
            else "STRUCTURAL_PROVENANCE_REBIND_VALIDATION_FAIL"
        ),
        "claim_limit": (
            "Support-A validation intervention-family replay on previously used Source "
            "datasets only; not formal dataset freshness, formal promotion, natural-defect, "
            "Target, Query, Memory, Agent, or transfer evidence."
        ),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=root / "artifacts/frozen/benchmark_v02/series_registry.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=root / "artifacts/frozen/benchmark_v02/split_manifest.json",
    )
    parser.add_argument(
        "--support-a-subsplit",
        type=Path,
        default=root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json",
    )
    parser.add_argument(
        "--clean-root", type=Path, default=root / "data/benchmark_v0_2/clean_base"
    )
    parser.add_argument(
        "--contract-report",
        type=Path,
        default=root
        / "artifacts/functional/e2/source_provenance_rebind_contract_report.json",
    )
    parser.add_argument(
        "--prior-replay-report",
        type=Path,
        default=root
        / "artifacts/functional/e2/source_provenance_rebind_source_replay_report.json",
    )
    parser.add_argument("--output", type=Path, default=root / OUTPUT_RELATIVE_PATH)
    args = parser.parse_args()
    report = run_replay(
        args.registry,
        args.split,
        args.support_a_subsplit,
        args.clean_root,
        args.contract_report,
        args.prior_replay_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    print(args.output)
    print(report["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
