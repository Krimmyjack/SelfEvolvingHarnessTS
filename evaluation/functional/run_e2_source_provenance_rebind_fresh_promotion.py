"""Plan or evaluate fresh-UID Source promotion for structural target rebind."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "e2-source-provenance-rebind-fresh-promotion/1"
PLAN_NAME = "source_provenance_rebind_fresh_promotion_plan.json"
REPORT_NAME = "source_provenance_rebind_fresh_promotion_report.json"
PLAN_RELATIVE_PATH = f"artifacts/functional/e2/{PLAN_NAME}"
REPORT_RELATIVE_PATH = f"artifacts/functional/e2/{REPORT_NAME}"
SUBSPLIT_SCHEMA = "benchmark-support-a-subsplit/2"
POLICIES = ("positional_incumbent", "key_rebind_repaired")
HORIZON = 48
MEAN_GAIN_MIN = 0.005
MEDIAN_GAIN_MIN_EXCLUSIVE = 0.0
HARM_THRESHOLD = -0.005
HARM_RATE_MAX = 0.25
EXPECTED_FITS = 4
DEPENDENCIES = (
    ("source_provenance_rebind_contract_report.json",
     "eecd7427748c2be26e8e18c78290b86170c4fea9a4d4477d52080e4c27419f92",
     "PROVENANCE_REBIND_CONTRACT_PASS"),
    ("source_provenance_rebind_source_replay_report.json",
     "b38b25e64013591d7a15bb779459f7f47f802e1bc4568906d20be333d2619cbd",
     "STRUCTURAL_PROVENANCE_REBIND_SOURCE_REPLAY_PASS"),
    ("source_provenance_rebind_validation_replay_report.json",
     "0c874fa28bb6fec3c25732f75cac0727dcc8927d697b860ea6324338dc6cc586",
     "STRUCTURAL_PROVENANCE_REBIND_VALIDATION_PASS"),
)


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    subsplit: str
    train_count: int
    eval_count: int
    frequency: str
    train_stop: int
    validation_bounds: tuple[int, int]
    context_length: int
    anchors: tuple[int, ...]
    positive_min_count: int


SPECS = (
    DatasetSpec("monash:traffic_hourly", "support_a_validation", 6, 6,
                "hourly", 928, (928, 976), 192,
                (240, 300, 360, 420, 480, 540), 4),
    DatasetSpec("monash:covid_deaths", "support_a_validation", 6, 6,
                "daily", 116, (116, 164), 64, (64, 68), 4),
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scan_e2(root: Path, selected_uids: set[str]) -> tuple[list[dict[str, object]], list[str]]:
    directory = root / "artifacts/functional/e2"
    excluded = {PLAN_NAME, REPORT_NAME}
    rows: list[dict[str, object]] = []
    hits: set[str] = set()
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        if path.name in excluded:
            continue
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        found = sorted(uid for uid in selected_uids if uid in text)
        hits.update(found)
        rows.append({"path": path.relative_to(root).as_posix(),
                     "sha256": hashlib.sha256(raw).hexdigest(),
                     "selected_uid_hits": found})
    return rows, sorted(hits)


def _read_frozen(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]],
                                      dict[str, Any], dict[str, str]]:
    registry_path = root / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
    split_path = root / "artifacts/frozen/benchmark_v02/split_manifest.json"
    subsplit_path = root / "artifacts/frozen/benchmark_v02/support_a_subsplit.json"
    registry = {row["series_uid"]: row for row in
                (json.loads(line) for line in registry_path.read_text("utf-8").splitlines())}
    split = json.loads(split_path.read_text("utf-8"))
    assignments = {row["series_uid"]: row for row in split["assignments"]}
    subsplit = json.loads(subsplit_path.read_text("utf-8"))
    hashes = {path.relative_to(root).as_posix(): _sha(path)
              for path in (registry_path, split_path, subsplit_path)}
    return registry, assignments, subsplit, hashes


def _select_plan_roster(root: Path) -> tuple[list[dict[str, object]], dict[str, object],
                                              dict[str, str]]:
    registry, assignments, subsplit, frozen_hashes = _read_frozen(root)
    if subsplit.get("schema_version") != SUBSPLIT_SCHEMA:
        raise ValueError("unsupported Support-A subsplit schema")
    members = subsplit.get("members")
    if not isinstance(members, dict):
        raise ValueError("Support-A members missing")
    roster: list[dict[str, object]] = []
    audit: dict[str, object] = {}
    source_shas: list[str] = []
    for spec in SPECS:
        raw_uids = members.get(spec.subsplit)
        if not isinstance(raw_uids, list) or len(raw_uids) != len(set(raw_uids)):
            raise ValueError(f"invalid subsplit members: {spec.subsplit}")
        candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for uid in raw_uids:
            record, assignment = registry.get(uid), assignments.get(uid)
            if not record or not assignment or record.get("dataset_id") != spec.dataset_id:
                continue
            if assignment.get("dataset_id") != spec.dataset_id:
                raise ValueError(f"registry/split dataset mismatch: {uid}")
            clean = (record.get("admission_reasons") == []
                     and record.get("natural_missing_count") == 0
                     and record.get("irregular_interval_count") == 0
                     and record.get("exposure_class") == "certified_virgin"
                     and record.get("frequency") == spec.frequency
                     and int(record.get("length", 0)) >= spec.validation_bounds[1]
                     and "support_a" in record.get("roles_allowed", [])
                     and assignment.get("role") == "support_a"
                     and record.get("regime_tag") == assignment.get("regime_tag")
                     and assignment.get("chronological_boundaries") == {
                         "train": [0, spec.train_stop],
                         "validation": list(spec.validation_bounds),
                         "test": [spec.validation_bounds[1],
                                  spec.validation_bounds[1] + HORIZON],
                     })
            if not clean:
                continue
            candidates.append((record, assignment))
        available_clean_certified = len(candidates)
        _, exposed_candidate_uids = _scan_e2(
            root, {str(row[0]["series_uid"]) for row in candidates}
        )
        exposed_candidate_uid_set = set(exposed_candidate_uids)
        candidates = [row for row in candidates
                      if row[0]["series_uid"] not in exposed_candidate_uid_set]
        candidates.sort(key=lambda pair: pair[0]["entity_id"])
        required = spec.train_count + spec.eval_count
        if len(candidates) < required:
            raise ValueError(
                f"fewer than {required} fresh clean same-boundary candidates: "
                f"{spec.dataset_id}; observed {len(candidates)}"
            )
        if len({row[0]["entity_id"] for row in candidates}) != len(candidates):
            raise ValueError(f"duplicate entity_id: {spec.dataset_id}")
        selected = candidates[:required]
        train, evaluate = selected[:spec.train_count], selected[spec.train_count:]
        train_uids = {row[0]["series_uid"] for row in train}
        eval_uids = {row[0]["series_uid"] for row in evaluate}
        train_groups = {row[0]["overlap_group"] for row in train}
        eval_groups = {row[0]["overlap_group"] for row in evaluate}
        if train_uids & eval_uids or train_groups & eval_groups:
            raise ValueError(f"train/eval overlap: {spec.dataset_id}")
        dataset_shas = {row[0]["source_asset_sha256"] for row in selected}
        if len(dataset_shas) != 1:
            raise ValueError(f"non-unique deployment asset: {spec.dataset_id}")
        source_shas.extend(dataset_shas)
        for cohort, rows in (("train", train), ("eval", evaluate)):
            for record, _ in rows:
                roster.append({"dataset_id": spec.dataset_id, "subsplit": spec.subsplit,
                               "cohort": cohort, "entity_id": record["entity_id"],
                               "series_uid": record["series_uid"],
                               "overlap_group": record["overlap_group"],
                               "source_asset_sha256": record["source_asset_sha256"]})
        audit[spec.dataset_id] = {
            "available_clean_certified_count": available_clean_certified,
            "excluded_prior_report_uid_count": len(exposed_candidate_uids),
            "available_after_exposure_count": len(candidates),
            "selection_rule": "entity_id ascending; first train_count, next eval_count",
            "entity_id_unique": True, "train_eval_uid_disjoint": True,
            "train_eval_overlap_group_disjoint": True,
            "selected_train_count": len(train), "selected_eval_count": len(evaluate),
            "source_asset_sha256": next(iter(dataset_shas)),
        }
    if len(set(source_shas)) != 2:
        raise ValueError("the two datasets must have different source_asset_sha256 values")
    if any(row["dataset_id"] == "uci_electricity_load_diagrams" for row in roster):
        raise AssertionError("UCI is forbidden")
    return roster, audit, frozen_hashes


def build_plan(root: Path) -> dict[str, object]:
    roster, audit, frozen_hashes = _select_plan_roster(root)
    selected_uids = {str(row["series_uid"]) for row in roster}
    scanned, hits = _scan_e2(root, selected_uids)
    if hits:
        raise ValueError("selected UIDs already occur in existing E2 reports")
    return {
        "schema_version": SCHEMA_VERSION, "phase": "plan", "plan_status": "READY",
        "scientific_role": "fresh_uid_structural_source_promotion_plan",
        "configuration": {spec.dataset_id: {
            "subsplit": spec.subsplit, "train_count": spec.train_count,
            "eval_count": spec.eval_count, "train_stop": spec.train_stop,
            "validation_bounds": list(spec.validation_bounds),
            "context_length": spec.context_length, "anchors": list(spec.anchors),
            "horizon": HORIZON, "positive_gain_min_count": spec.positive_min_count,
        } for spec in SPECS},
        "frozen_metadata_sha256": frozen_hashes, "roster": roster,
        "roster_audit": audit,
        "exposure_scan": {"all_existing_e2_json_scanned": True,
                          "excluded_output_names": [PLAN_NAME, REPORT_NAME],
                          "reports": scanned, "selected_uid_hits": [], "pass": True},
        "information_wall": {"series_values_loaded": False, "uci_selected": False,
                             "target_closed": True, "query_closed": True,
                             "target_query_opened": False},
        "target_query_opened": False,
        "claim_limit": "Metadata-only fresh-UID plan; no fit, outcome, promotion, Target, or Query claim.",
    }


def _verify_plan(root: Path, path: Path, expected_sha: str) -> tuple[dict[str, Any], dict[str, object]]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha:
        raise ValueError("plan SHA256 mismatch")
    plan = json.loads(raw)
    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("plan_status") != "READY":
        raise ValueError("plan is not ready")
    _, _, _, frozen_hashes = _read_frozen(root)
    metadata_match = frozen_hashes == plan.get("frozen_metadata_sha256")
    selected = {str(row["series_uid"]) for row in plan["roster"]}
    scan, hits = _scan_e2(root, selected)
    scan_match = scan == plan.get("exposure_scan", {}).get("reports") and not hits
    plan_exposure_pass = (plan.get("exposure_scan", {}).get("pass") is True
                          and plan.get("exposure_scan", {}).get("selected_uid_hits") == [])
    return plan, {"path": str(path), "expected_sha256": expected_sha,
                  "actual_sha256": actual, "hash_matches": True,
                  "frozen_metadata_matches": metadata_match,
                  "exposure_snapshot_matches": scan_match,
                  "plan_exposure_pass": plan_exposure_pass,
                  "pass": metadata_match and scan_match and plan_exposure_pass}


def _dependencies(root: Path) -> tuple[list[dict[str, object]], bool]:
    rows = []
    for name, expected_sha, expected_verdict in DEPENDENCIES:
        path = root / "artifacts/functional/e2" / name
        raw = path.read_bytes()
        payload = json.loads(raw)
        actual = hashlib.sha256(raw).hexdigest()
        passed = actual == expected_sha and payload.get("verdict") == expected_verdict
        rows.append({"path": path.relative_to(root).as_posix(),
                     "expected_sha256": expected_sha, "actual_sha256": actual,
                     "hash_matches": actual == expected_sha,
                     "expected_verdict": expected_verdict,
                     "actual_verdict": payload.get("verdict"), "pass": passed})
    return rows, all(bool(row["pass"]) for row in rows)


def _evaluate(root: Path, plan: dict[str, Any], preflight: dict[str, object],
              dependencies: list[dict[str, object]], dependency_pass: bool) -> dict[str, object]:
    base: dict[str, object] = {
        "schema_version": SCHEMA_VERSION, "phase": "evaluate",
        "scientific_role": "formal_source_promotion_for_injected_structural_capability",
        "configuration": {"datasets": [spec.dataset_id for spec in SPECS],
            "policies": list(POLICIES),
            "target_binding": "whole TargetRow derangement followed by key-only rebind",
            "trusted_unique_provenance_key": ["dataset_sha", "series_uid", "anchor", "horizon"],
            "consumer": {"class": "sklearn.linear_model.Ridge", "alpha": 1.0,
                         "fit_intercept": True, "solver": "svd"},
            "metric": "per-series normalized MAE"},
        "plan_dependency": preflight,
        "dependencies": {"reports": dependencies, "pass": dependency_pass},
        "consumer_fit_count": 0, "policy_intervention_evidence": [],
    }
    if not bool(preflight["pass"]) or not dependency_pass:
        return _finish_report(base, False, [], None)

    import numpy as np
    from sklearn.linear_model import Ridge
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.dev_eval import _load_values
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.registry import read_registry_jsonl
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_cohort_policy_premise import _center_scale
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_label_binding_positive_control import _row_multiset
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_provenance_rebind_contract import (
        InputRow, RowKey, TargetRow, _dataset_sha, _rebind_supervised_targets_v0)
    from SelfEvolvingHarnessTS.evaluation.functional.run_e2_source_provenance_rebind_source_replay import _derange_present_anchors

    records = {row.series_uid: row for row in read_registry_jsonl(
        root / "artifacts/frozen/benchmark_v02/series_registry.jsonl")}
    selected = plan["roster"]
    selected_records = [records[str(row["series_uid"])] for row in selected]
    values = _load_values(selected_records, root / "data/benchmark_v0_2/clean_base")
    bundles: dict[str, dict[str, Any]] = {}
    p0_rows: dict[str, object] = {}
    for spec in SPECS:
        train_meta = [row for row in selected if row["dataset_id"] == spec.dataset_id
                      and row["cohort"] == "train"]
        items = [type("Item", (), {"record": records[str(row["series_uid"])]})()
                 for row in train_meta]
        dataset_sha, deployment = _dataset_sha(items)
        inputs, clean_targets = [], []
        for anchor in spec.anchors:
            for row in train_meta:
                uid = str(row["series_uid"])
                raw = values[uid]
                context = np.asarray(raw[anchor - spec.context_length:anchor], dtype=np.float64)
                target = np.asarray(raw[anchor:anchor + HORIZON], dtype=np.float64)
                if context.shape != (spec.context_length,) or target.shape != (HORIZON,):
                    raise ValueError(f"insufficient training row: {uid}/{anchor}")
                if not np.isfinite(context).all() or not np.isfinite(target).all():
                    raise ValueError(f"non-finite training row: {uid}/{anchor}")
                center, scale, _ = _center_scale(context)
                key = RowKey(dataset_sha, uid, anchor, HORIZON)
                inputs.append(InputRow(key, np.concatenate(((context-center)/scale,
                                                            np.zeros(spec.context_length)))))
                clean_targets.append(TargetRow(key, (target-center)/scale))
        clean_y = np.asarray([row.payload for row in clean_targets])
        x_train = np.asarray([row.payload for row in inputs])
        deranged = _derange_present_anchors(spec.dataset_id, inputs, clean_targets)
        status, action, repaired, reasons = _rebind_supervised_targets_v0(inputs, deranged)
        incumbent_y = np.asarray([row.payload for row in deranged])
        repaired_y = np.asarray([row.payload for row in repaired]) if repaired else np.empty((0, HORIZON))
        fixed = sum(left.key == right.key for left, right in zip(inputs, deranged))
        exact_error = float(np.max(np.abs(repaired_y-clean_y)))
        clean_by_key = {row.key: row for row in clean_targets}
        keys_travelled = all(np.array_equal(row.payload, clean_by_key[row.key].payload)
                             for row in deranged)
        p0_pass = (status == "ELIGIBLE" and action == "REPAIRED" and fixed == 0
                   and len(inputs) == len({row.key for row in inputs})
                   and len(deranged) == len({row.key for row in deranged})
                   and {row.key for row in inputs} == {row.key for row in deranged}
                   and _row_multiset(clean_y) == _row_multiset(incumbent_y)
                   and keys_travelled
                   and exact_error <= 1e-12)
        p0_rows[spec.dataset_id] = {"checked_before_any_consumer_fit": True,
                                    "consumer_fit_count_at_check": 0,
                                    "status": status, "action": action,
                                    "reason_codes": reasons, "fixed_point_count": fixed,
                                    "target_key_travelled_with_payload": keys_travelled,
                                    "repaired_target_max_abs_error_to_clean": exact_error,
                                    "deployment_metadata": deployment, "pass": p0_pass}
        bundles[spec.dataset_id] = {"x": x_train, "targets": {
            POLICIES[0]: incumbent_y, POLICIES[1]: repaired_y}}
    p0_pass = all(bool(row["pass"]) for row in p0_rows.values())
    evidence, fit_count = [], 0
    if p0_pass:
        for spec in SPECS:
            eval_meta = [row for row in selected if row["dataset_id"] == spec.dataset_id
                         and row["cohort"] == "eval"]
            x_eval, y_eval, uids = [], [], []
            for row in eval_meta:
                uid, raw = str(row["series_uid"]), values[str(row["series_uid"])]
                context = np.asarray(raw[spec.train_stop-spec.context_length:spec.train_stop])
                future = np.asarray(raw[slice(*spec.validation_bounds)])
                if context.shape != (spec.context_length,) or future.shape != (HORIZON,):
                    raise ValueError(f"insufficient evaluation row: {uid}")
                if not np.isfinite(context).all() or not np.isfinite(future).all():
                    raise ValueError(f"non-finite evaluation row: {uid}")
                center, scale, _ = _center_scale(context)
                x_eval.append(np.concatenate(((context-center)/scale,
                                              np.zeros(spec.context_length))))
                y_eval.append((future-center)/scale); uids.append(uid)
            x_matrix, y_matrix = np.asarray(x_eval), np.asarray(y_eval)
            losses: dict[str, list[float]] = {}
            for policy in POLICIES:
                model = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
                model.fit(bundles[spec.dataset_id]["x"],
                          bundles[spec.dataset_id]["targets"][policy]); fit_count += 1
                prediction = np.asarray(model.predict(x_matrix))
                if prediction.shape != y_matrix.shape or not np.isfinite(prediction).all():
                    raise RuntimeError(f"invalid Ridge prediction: {spec.dataset_id}/{policy}")
                losses[policy] = [float(v) for v in np.mean(abs(prediction-y_matrix), axis=1)]
            paired = [{"series_uid": uid,
                       "positional_incumbent_normalized_mae": incumbent,
                       "key_rebind_repaired_normalized_mae": repaired,
                       "gain_incumbent_minus_repaired": incumbent-repaired,
                       "positive_gain": incumbent-repaired > 0,
                       "harmed": incumbent-repaired < HARM_THRESHOLD}
                      for uid, incumbent, repaired in zip(uids, losses[POLICIES[0]], losses[POLICIES[1]])]
            gains = [float(row["gain_incumbent_minus_repaired"]) for row in paired]
            gate = (statistics.fmean(gains) >= MEAN_GAIN_MIN
                    and statistics.median(gains) > MEDIAN_GAIN_MIN_EXCLUSIVE
                    and sum(bool(row["positive_gain"]) for row in paired) >= spec.positive_min_count)
            evidence.append({"dataset_id": spec.dataset_id, "paired_eval_rows": paired,
                             "mean_gain_incumbent_minus_repaired": statistics.fmean(gains),
                             "median_gain_incumbent_minus_repaired": statistics.median(gains),
                             "positive_gain_count": sum(bool(row["positive_gain"]) for row in paired),
                             "dataset_gate": {"mean_gain_min": MEAN_GAIN_MIN,
                                "median_gain_must_exceed": MEDIAN_GAIN_MIN_EXCLUSIVE,
                                "positive_gain_min_count": spec.positive_min_count, "pass": gate}})
    if p0_pass and fit_count != EXPECTED_FITS:
        raise AssertionError("exact P0 must lead to exactly four Ridge fits")
    harms = [bool(row["harmed"]) for item in evidence for row in item["paired_eval_rows"]]
    harm_rate = sum(harms)/len(harms) if harms else None
    passed = (p0_pass and fit_count == EXPECTED_FITS and len(evidence) == 2
              and all(bool(row["dataset_gate"]["pass"]) for row in evidence)
              and harm_rate is not None and harm_rate <= HARM_RATE_MAX)
    base.update({"p0_pre_fit_exact_gate": {"dataset_results": p0_rows,
                                           "failure_action": "zero-fit stop", "pass": p0_pass},
                 "consumer_fit_count": fit_count, "policy_intervention_evidence": evidence})
    return _finish_report(base, passed, harms, harm_rate)


def _finish_report(base: dict[str, object], passed: bool, harms: list[bool],
                   harm_rate: float | None) -> dict[str, object]:
    plan_dependency = base.get("plan_dependency")
    fresh_selected_uids = (
        isinstance(plan_dependency, dict)
        and plan_dependency.get("exposure_snapshot_matches") is True
        and plan_dependency.get("plan_exposure_pass") is True
    )
    base.update({
        "promotion_gate": {"dataset_gates_conjunctive": True,
            "pooled_harm": {"definition": "gain < -0.005", "count": sum(harms),
                            "total": len(harms), "rate": harm_rate,
                            "rate_max": HARM_RATE_MAX,
                            "pass": harm_rate is not None and harm_rate <= HARM_RATE_MAX},
            "pass": passed},
        "structural_source_promotion_eligible": passed,
        "promotion_scope": "formal Source promotion for injected structural capability only",
        "natural_defect_claim": False, "numeric_pattern_claim": False,
        "broad_capability_claim": False, "formal_dataset_fresh": False,
        "intervention_family_fresh_on_selected_uids": fresh_selected_uids,
        "freshness_scope": "selected UIDs have zero hits in the frozen pre-plan E2 report set",
        "fresh_cohort_evidence": passed,
        "target_query_opened": False,
        "information_wall": {"selected_plan_values_only": True,
                             "uci_values_read": False, "target_closed": True,
                             "query_closed": True, "target_query_opened": False},
        "verdict": ("STRUCTURAL_PROVENANCE_REBIND_FRESH_PROMOTION_PASS" if passed
                    else "STRUCTURAL_PROVENANCE_REBIND_FRESH_PROMOTION_FAIL"),
        "claim_limit": "Formal Source promotion is limited to the injected structural rebind capability; no natural-defect, numeric-Pattern, broad capability, Target, Query, Memory, Agent, or transfer claim.",
    })
    return base


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("plan", "evaluate"))
    parser.add_argument("--plan-report", type=Path)
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "plan":
        report = build_plan(root)
        output = args.output or root / PLAN_RELATIVE_PATH
    else:
        if args.plan_report is None or args.expected_plan_sha256 is None:
            parser.error("evaluate requires --plan-report and --expected-plan-sha256")
        plan, preflight = _verify_plan(root, args.plan_report, args.expected_plan_sha256)
        dependencies, dependency_pass = _dependencies(root)
        report = _evaluate(root, plan, preflight, dependencies, dependency_pass)
        output = args.output or root / REPORT_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_json_bytes(report) + b"\n")
    print(output)
    print(report.get("verdict", report.get("plan_status")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
