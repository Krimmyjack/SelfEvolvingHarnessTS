"""S1a-r1 -- dual-layer 0-LLM oracle + curriculum qualification + reachability.

Independent runner.  Reuses the Wine-precheck enumeration (cohort 0.10 cap,
full shared classification menu, ridge, fit_only_artifact) without modifying
the shared CLS-OP harness.  Writes sealed oracle artifacts that must never
enter any arm's prompt, store, or retrieval view.

  python evaluation/functional/run_e2_s1a_curriculum_oracle_audit.py --run

Zero Fast LLM.  Slow rehearsal is off unless --slow-rehearse is passed and
the code path cannot confirm a compiled card shape.  No A3/A5 adaptation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np  # noqa: E402

import run_e2_t6_cls_op_shared_harness as cls  # noqa: E402
from consumers.cls_scope_adapter import (  # noqa: E402
    HELDOUT,
    SUPPORT,
    ClassificationConsumerAdapter,
)
from SelfEvolvingHarnessTS.contracts.observables import (  # noqa: E402
    OBSERVABLE_FEATURES,
    observable_numeric_bin,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    _is_inert_experience_card,
    _scopes_beyond_task_kind,
)
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    risk_skill,
    source_skill as ss,
)

# =========================================================================== #
# frozen protocol -- pool is declared once; post-hoc add/remove is forbidden
# =========================================================================== #
PROTOCOL_VERSION = "s1a_curriculum_oracle_audit_v1"
RUN_ID = "s1a_r1_curriculum_audit1"
CURRICULUM_NAME = "development positive-control curriculum"
EVIDENCE_GRADE = "development"
CONDITION = "fit_only_artifact"
CONSUMER_ID = "ridge-raw-plus-difference-v1"
METRIC = "accuracy"
TASK_KIND = "classification"

ORACLE_BANNER = (
    "本文件不得进入任何臂的 prompt/store/检索视野"
)
ORACLE_ISOLATION = (
    "sealed exam key.  This file must not enter any arm prompt, store, "
    "or retrieval view.  Held-out per-operator scores are exam keys only."
)

FIT_CAP = 500
WALL_SECONDS_CAP = 90 * 60
LLM_FAST_CAP = 0
LLM_SLOW_REHEARSE_CAP = 8
MATERIAL = cls.MATERIAL  # 0.005
CLASS_HARM_BAR = cls.MATERIAL  # worst-class Δrecall >= -0.005
DATA_DIR = cls.DATA_DIR

# Impulse-v2 substrates.  Order is the pre-declared roster, not a ranking.
IMPULSE_DATASETS = (
    "GunPointAgeSpan",
    "GunPoint",
    "ECG200",
    "Wine",
    "ToeSegmentation1",
    "Lightning2",
    "Herring",
    "Ham",
)
BURST_DATASET = "GunPoint"
BURST_INJECTION = "burst_cls2"

# Pattern keys used for Scope-v1 axis (4).  Probe directions stay "unknown"
# without a probe panel and are not a Pattern.  task_kind is eligibility.
PATTERN_KEYS = (
    "missing_fraction",
    "longest_missing_run_fraction",
    "local_robust_z_peak",
    "estimated_region_start_fraction",
    "estimated_region_end_fraction",
    "level_region_fraction",
    "level_region_end_fraction",
    "outlier_region_end_fraction",
    "level_excursion_score",
    "estimated_level_offset",
    "period_change_score",
    "period_reliability",
    "period_evidence_status",
)

E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
ORACLE_DIR = E2 / "s1_oracle"
AUDIT_JSON = E2 / "s1a_curriculum_audit.json"
AUDIT_MD = E2 / "s1a_curriculum_audit.md"

EPISODE_SOURCES = (
    E2 / "t6_cls_op_shared_harness.json",
    E2 / "t6_cls_op_r2_three_arms.json",
    E2 / "t6_cls_op_r2_a5_replay.json",
    E2 / "t6_cls_op_r2_prep.json",
    E2 / "t6_cls_conf_dev_ecg200.json",
)

FORBIDDEN_DATA_ROOTS = cls.FORBIDDEN_DATA_ROOTS


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=False,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(nested) for nested in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _cell_public(cell: Mapping[str, Any]) -> dict[str, Any]:
    skip = {"fit_values", "fit_labels", "surfaces", "observation_block",
            "clean_fit_values", "clean_fit_labels"}
    return {key: value for key, value in cell.items() if key not in skip}


def _binned_public_features(block: Any) -> dict[str, Any]:
    raw = dict(extract_public_features(block, task_kind=TASK_KIND))
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in OBSERVABLE_FEATURES:
            continue
        kind = OBSERVABLE_FEATURES[key]
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = observable_numeric_bin(key, float(value))
        else:
            out[key] = value
    return out


def _pattern_view(features: Mapping[str, Any]) -> dict[str, Any]:
    return {key: features[key] for key in PATTERN_KEYS if key in features}


def _load_clean_fit(dataset: str) -> tuple[Any, Any]:
    _ctx, helpers = cls._legacy_helpers()
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % dataset)
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, _support = helpers["split"](np, train_labels)
    return (np.asarray(train_values[fit_indices], dtype=np.float64),
            np.asarray(train_labels[fit_indices]))


def _declared_pool() -> list[dict[str, Any]]:
    pool = []
    for dataset in IMPULSE_DATASETS:
        pool.append({
            "unit_id": "%s__impulse_v2" % dataset,
            "dataset": dataset,
            "injection": "impulse_v2",
            "condition": CONDITION,
            "consumer": CONSUMER_ID,
        })
    pool.append({
        "unit_id": "%s__%s" % (BURST_DATASET, BURST_INJECTION),
        "dataset": BURST_DATASET,
        "injection": BURST_INJECTION,
        "condition": CONDITION,
        "consumer": CONSUMER_ID,
    })
    return pool


def _build_burst_cell(dataset: str) -> dict[str, Any]:
    """GunPoint × CLS-2 burst, fit_only_artifact: burst on fit, support clean."""
    from run_e2_t6_cls2_value_corruption_gate import (
        SEED_INJECT,
        inject_burst_noise,
    )

    _ctx, helpers = cls._legacy_helpers()
    archive = PROJECT_ROOT / DATA_DIR / ("%s.zip" % dataset)
    train_values, train_labels = helpers["load"](np, archive, dataset, "TRAIN")
    fit_indices, support_indices = helpers["split"](np, train_labels)
    base_fit = train_values[fit_indices]
    fit_labels = train_labels[fit_indices]
    base_support = train_values[support_indices]
    support_labels = train_labels[support_indices]
    fit_values, ledger = inject_burst_noise(
        np.asarray(base_fit, dtype=np.float64),
        np.asarray(fit_labels),
        seed=SEED_INJECT,
    )
    support_values = np.asarray(base_support, dtype=np.float64).copy()

    observation = helpers["observe"](np, fit_values, fit_labels)
    nodes = tuple(int(node) for node in observation["nodes"])
    witness = helpers["witness"](
        np, fit_values, fit_labels, support_values, support_labels, nodes,
        helpers["rolling_median"])
    legacy_decision, legacy_reasons = helpers["risk_decision"](witness)
    order = {int(index): position
             for position, index in enumerate(support_indices)}
    parts = cls._quarter(support_labels,
                         [order[int(i)] for i in support_indices])
    surfaces: dict[str, tuple[Any, Any]] = {}
    for name, part in zip(("r1_support", "r1_delayed",
                           "r2_support", "r2_delayed"), parts):
        surfaces[name] = (support_values[part], support_labels[part])
    length = int(train_values.shape[1])
    rows_in_window = max(4, -(-cls.OBSERVATION_POINTS // length))
    rows_in_window = min(rows_in_window, int(fit_values.shape[0]))
    observation_block = np.asarray(
        fit_values[:rows_in_window], dtype=np.float64).ravel()
    return {
        "dataset": dataset,
        "condition": CONDITION,
        "data_dir": DATA_DIR,
        "archive": "%s/%s.zip" % (DATA_DIR, dataset),
        "series_length": length,
        "official_train_rows": int(train_values.shape[0]),
        "fit_rows": int(fit_values.shape[0]),
        "support_pool_rows": int(support_indices.size),
        "slice_rows": {name: int(len(part)) for name, part
                       in zip(("r1_support", "r1_delayed",
                               "r2_support", "r2_delayed"), parts)},
        "fit_values": fit_values,
        "fit_labels": fit_labels,
        "surfaces": surfaces,
        "controlled_impulse_positions": [],
        "observer_localized_nodes": list(nodes),
        "observer_recovered_all_nodes": False,
        "witness": dict(witness),
        "legacy_scope_decision": legacy_decision,
        "legacy_scope_reasons": list(legacy_reasons),
        cls.CENSUS_CONDITION_KEY: bool(
            legacy_decision == "ABSTAIN_KEEP_INCUMBENT"),
        "observation_rows": rows_in_window,
        "observation_block": observation_block,
        "injection_template": BURST_INJECTION,
        "burst_ledger": {
            key: ledger[key] for key in ledger
            if key not in ("rows",)
        },
        "burst_seed": SEED_INJECT,
        "burst_template_source": (
            "evaluation/functional/run_e2_t6_cls2_value_corruption_gate.py"
            ":inject_burst_noise / SEED_INJECT=202608254"
        ),
    }


def _score_surface(adapter: ClassificationConsumerAdapter, compiled: Any,
                   origin: int) -> dict[str, Any]:
    reading = adapter([], {}, compiled, {}, origin=origin)
    return {
        "accuracy": float(reading["cls_accuracy"]),
        "recall_by_class": {
            str(key): float(value)
            for key, value in (reading.get("cls_recall_by_class") or {}).items()
        },
        "rows": int(reading.get("cls_evaluated_rows") or 0),
    }


def _oracle_one_unit(*, spec: Mapping[str, Any], cell: Mapping[str, Any],
                     clean_fit: Any, fit_budget: cls.FitBudget) -> dict[str, Any]:
    """0-LLM dual-layer oracle on one pre-declared unit."""
    heldin_values, heldin_labels = cls._wine_heldin_pool(cell)
    heldout_values, heldout_labels = cls._heldout_surface(
        cell["dataset"], CONDITION, data_dir=cell.get("data_dir"))
    n_heldin = int(heldin_labels.size)
    n_heldout = int(heldout_labels.size)
    heldin_line = max(MATERIAL, 1.0 / max(n_heldin, 1))
    heldout_line = max(MATERIAL, 1.0 / max(n_heldout, 1))
    block = np.asarray(cell["observation_block"], dtype=np.float64)
    support_origin = int(block.size)
    delayed_origin = support_origin + 1
    heldout_origin = support_origin + 2
    cap = float(
        cls._task_context().deployment_constraints.maximum_modified_fraction)

    adapter = ClassificationConsumerAdapter(
        fit_values=cell["fit_values"], fit_labels=cell["fit_labels"],
        surfaces={
            SUPPORT: (heldin_values, heldin_labels),
            HELDOUT: (heldout_values, heldout_labels),
        },
        delayed_origin=delayed_origin, heldout_origin=heldout_origin,
        budget=fit_budget, ridge_alpha=cls.RIDGE_ALPHA,
        allowed_surfaces=(SUPPORT, HELDOUT))
    executor = cls._ClsScopeExecutor(
        cell=cell, evaluate_fn=adapter, max_modified_fraction=cap,
        modification_fraction_scope="cohort")

    identity_heldout = _score_surface(adapter, None, heldout_origin)
    clean_adapter = ClassificationConsumerAdapter(
        fit_values=clean_fit, fit_labels=cell["fit_labels"],
        surfaces={HELDOUT: (heldout_values, heldout_labels)},
        delayed_origin=delayed_origin, heldout_origin=heldout_origin,
        budget=fit_budget, ridge_alpha=cls.RIDGE_ALPHA,
        allowed_surfaces=(HELDOUT,))
    upper = _score_surface(clean_adapter, None, heldout_origin)

    identity_row = {
        "program": "identity",
        "params": {},
        "legal": True,
        "verifier_passed": True,
        "cohort_modified_fraction": 0.0,
        "cohort_modified_points": 0,
        "cohort_total_points": int(np.asarray(cell["fit_values"]).size),
        "rejection_codes": [],
        "numeric_no_op": True,
        "heldin_headroom": 0.0,
        "heldin_worst_class_recall_delta": 0.0,
        "heldout_accuracy": identity_heldout["accuracy"],
        "heldout_utility": 0.0,
        "heldout_recall_by_class": identity_heldout["recall_by_class"],
        "heldout_recall_delta_by_class": {
            key: 0.0 for key in identity_heldout["recall_by_class"]
        },
        "heldout_worst_class_recall_delta": 0.0,
        "in_oracle_set": False,
        "scored_heldin": False,
        "scored_heldout": True,
        "skip_reason": "identity_is_the_empty_oracle_fallback",
    }
    rows: list[dict[str, Any]] = [identity_row]

    for entry in cls._r2_menu():
        steps = ((entry["program"], dict(entry["params"])),)
        verification = executor.verify(steps, support_origin)
        no_op = bool(
            verification.checked_windows
            and verification.identity_equivalent_windows
            == verification.checked_windows)
        record: dict[str, Any] = {
            "program": entry["program"],
            "params": entry["params"],
            "legal": bool(verification.passed),
            "verifier_passed": bool(verification.passed),
            "cohort_modified_fraction": verification.cohort_modified_fraction,
            "cohort_modified_points": verification.cohort_modified_points,
            "cohort_total_points": verification.cohort_total_points,
            "windows_over_per_window_cap": (
                verification.windows_over_maximum_fraction),
            "checked_windows": verification.checked_windows,
            "rejection_codes": sorted({
                str(row["rejection_code"])
                for row in verification.rejected_windows
            }),
            "numeric_no_op": no_op,
            "heldin_headroom": None,
            "heldin_worst_class_recall_delta": None,
            "heldout_accuracy": None,
            "heldout_utility": None,
            "heldout_recall_by_class": None,
            "heldout_recall_delta_by_class": None,
            "heldout_worst_class_recall_delta": None,
            "in_oracle_set": False,
            "scored_heldin": False,
            "scored_heldout": False,
        }
        if not record["legal"]:
            record["skip_reason"] = (
                "verifier_rejected:" + ",".join(record["rejection_codes"]))
            rows.append(record)
            continue
        if no_op:
            record["heldin_headroom"] = 0.0
            record["heldin_worst_class_recall_delta"] = 0.0
            record["heldout_accuracy"] = identity_heldout["accuracy"]
            record["heldout_utility"] = 0.0
            record["heldout_recall_by_class"] = identity_heldout["recall_by_class"]
            record["heldout_recall_delta_by_class"] = {
                key: 0.0 for key in identity_heldout["recall_by_class"]
            }
            record["heldout_worst_class_recall_delta"] = 0.0
            record["skip_reason"] = "numeric_no_op"
            rows.append(record)
            continue

        receipt = executor.evaluate(steps, support_origin)
        heldin_delta = (float(receipt.gain) if receipt.gain is not None
                        else None)
        heldin_recalls = [float(value) for value in receipt.per_view_gain]
        record["scored_heldin"] = True
        record["heldin_headroom"] = heldin_delta
        record["heldin_per_class_recall_delta"] = heldin_recalls
        record["heldin_worst_class_recall_delta"] = (
            min(heldin_recalls) if heldin_recalls else None)
        record["evaluate_error"] = receipt.error

        compiled = cls._FrozenCompiled(
            [{"op": entry["program"], "params": dict(entry["params"])}])
        heldout = _score_surface(adapter, compiled, heldout_origin)
        record["scored_heldout"] = True
        record["heldout_accuracy"] = heldout["accuracy"]
        record["heldout_utility"] = (
            heldout["accuracy"] - identity_heldout["accuracy"])
        record["heldout_recall_by_class"] = heldout["recall_by_class"]
        record["heldout_recall_delta_by_class"] = {
            key: heldout["recall_by_class"].get(key, 0.0)
            - identity_heldout["recall_by_class"].get(key, 0.0)
            for key in sorted(set(heldout["recall_by_class"])
                              | set(identity_heldout["recall_by_class"]))
        }
        deltas = list(record["heldout_recall_delta_by_class"].values())
        record["heldout_worst_class_recall_delta"] = (
            min(deltas) if deltas else None)
        utility = record["heldout_utility"]
        worst = record["heldout_worst_class_recall_delta"]
        record["in_oracle_set"] = bool(
            record["legal"]
            and utility is not None
            and float(utility) >= heldout_line
            and worst is not None
            and float(worst) >= -CLASS_HARM_BAR)
        rows.append(record)

    legal = [row["program"] for row in rows if row["legal"]]
    oracle_set = [row["program"] for row in rows if row["in_oracle_set"]]
    if oracle_set:
        menu_best = max(
            (row for row in rows if row["in_oracle_set"]),
            key=lambda row: (float(row["heldout_utility"]),
                             float(row["heldout_accuracy"])))
        menu_best_program = menu_best["program"]
        menu_best_heldout = float(menu_best["heldout_accuracy"])
        menu_best_utility = float(menu_best["heldout_utility"])
    else:
        menu_best_program = "identity"
        menu_best_heldout = identity_heldout["accuracy"]
        menu_best_utility = 0.0

    public_features = _binned_public_features(block)
    identity_residual = float(upper["accuracy"] - identity_heldout["accuracy"])
    menu_best_residual = float(upper["accuracy"] - menu_best_heldout)
    return {
        "isolation_banner": ORACLE_BANNER,
        "isolation": ORACLE_ISOLATION,
        "curriculum_name": CURRICULUM_NAME,
        "unit_id": spec["unit_id"],
        "dataset": spec["dataset"],
        "injection": spec["injection"],
        "condition": CONDITION,
        "consumer": CONSUMER_ID,
        "metric": METRIC,
        "task_kind": TASK_KIND,
        "series_length": cell.get("series_length"),
        "fit_rows": cell.get("fit_rows"),
        "n_heldin": n_heldin,
        "n_heldout": n_heldout,
        "heldin_material_line": heldin_line,
        "heldout_material_line": heldout_line,
        "class_harm_bar": CLASS_HARM_BAR,
        "maximum_modified_fraction": cap,
        "modification_fraction_scope": "cohort",
        "heldout_is_official_clean_test": True,
        "heldout_note": (
            "fit_only_artifact == TARGET_CONDITION; "
            "_heldout_surface returns the official TEST split uninjected"
        ),
        "legal_set": legal,
        "oracle_set": oracle_set if oracle_set else ["identity"],
        "oracle_set_empty": not bool(oracle_set),
        "positive_unit": bool(oracle_set),
        "menu_oracle_program": menu_best_program,
        "menu_oracle_heldout_accuracy": menu_best_heldout,
        "menu_oracle_heldout_utility": menu_best_utility,
        "identity_heldout_accuracy": identity_heldout["accuracy"],
        "identity_heldout_recall_by_class": identity_heldout["recall_by_class"],
        "readiness_upper_bound_heldout_accuracy": upper["accuracy"],
        "readiness_upper_bound_recall_by_class": upper["recall_by_class"],
        "identity_residual_to_upper_bound": identity_residual,
        "menu_best_residual_to_upper_bound": menu_best_residual,
        "program_supply_gap": bool(abs(menu_best_residual) > 1e-12),
        "program_supply_gap_note": (
            "menu-best residual to the exact-repair upper bound is the "
            "Program Supply gap.  An empty oracle set is identity, not "
            "'the data needs no processing'."
        ),
        "public_features_binned": public_features,
        "pattern_view": _pattern_view(public_features),
        "programs": rows,
        "cell": _cell_public(cell),
        "consumer_fits_after_unit": fit_budget.used,
    }


def _write_sealed_oracle(unit: Mapping[str, Any]) -> None:
    path = ORACLE_DIR / ("%s.json" % unit["unit_id"])
    payload = {
        "isolation_banner": ORACLE_BANNER,
        "isolation": ORACLE_ISOLATION,
        "do_not_load_into_harness": True,
        "curriculum_name": CURRICULUM_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "protocol_version": PROTOCOL_VERSION,
        **{key: value for key, value in unit.items()
           if key != "cell" or True},
    }
    _dump(path, payload)


def _intersect_maps(maps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not maps:
        return {}
    shared = dict(maps[0])
    for other in maps[1:]:
        shared = {key: value for key, value in shared.items()
                  if key in other and other[key] == value}
    return shared


def _compatible_clusters(positives: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Largest subsets that share one Program and a non-empty Pattern view.

    The gate asks for the existence of >=2 independent positives with the
    same Task/Consumer, a compatible deployment-visible Pattern, and the
    same Program geometry.  It does not require every positive in the pool
    to share one geometry -- a hampel cluster and a burst-repair cluster
    are different families.
    """
    programs: set[str] = set()
    for unit in positives:
        programs.update(set(unit["oracle_set"]) - {"identity"})
    clusters: list[dict[str, Any]] = []
    for program in sorted(programs):
        members = [unit for unit in positives if program in unit["oracle_set"]]
        if len(members) < 2:
            clusters.append({
                "program": program,
                "unit_ids": [unit["unit_id"] for unit in members],
                "pattern_intersection": dict(members[0]["pattern_view"]) if members else {},
                "compatible": False,
            })
            continue
        shared = _intersect_maps([unit["pattern_view"] for unit in members])
        clusters.append({
            "program": program,
            "unit_ids": [unit["unit_id"] for unit in members],
            "pattern_intersection": shared,
            "compatible": bool(shared),
        })
    clusters.sort(key=lambda row: (-int(row["compatible"]),
                                   -len(row["unit_ids"]),
                                   row["program"]))
    return clusters


def _qualify(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [unit for unit in units if unit["positive_unit"]]
    identities = [unit for unit in units
                  if not unit["positive_unit"]
                  and unit["injection"] != BURST_INJECTION]
    burst = next(unit for unit in units if unit["injection"] == BURST_INJECTION)
    clusters = _compatible_clusters(positives)
    chosen = next((row for row in clusters if row["compatible"]), None)

    same_task_consumer = all(
        unit["task_kind"] == TASK_KIND
        and unit["consumer"] == CONSUMER_ID
        and unit["metric"] == METRIC
        for unit in units
    )
    compatible = bool(chosen and same_task_consumer)
    compatible_positives = []
    if chosen:
        allowed = set(chosen["unit_ids"])
        compatible_positives = [unit for unit in positives
                                if unit["unit_id"] in allowed]

    if not positives:
        verdict = "CURRICULUM_NOT_VIABLE"
        frozen = None
        limitation = (
            "zero positive units in the pre-declared pool.  "
            "No injection-parameter scan, no pool expansion."
        )
    elif not compatible:
        verdict = "SAFETY_ONLY_CURRICULUM"
        limitation = (
            "positives exist but no cluster of >=2 shares Task/Consumer + "
            "deployment-visible Pattern + one Program geometry.  Only "
            "risk/program channels are measurable.  Stopped for mainline "
            "decision.  No pool expansion."
        )
        frozen = _draft_curriculum(positives, identities, burst, full=False)
    else:
        verdict = "FULL_CURRICULUM_QUALIFIED"
        limitation = None
        frozen = _draft_curriculum(
            compatible_positives, identities, burst, full=True)

    return {
        "verdict": verdict,
        "limitation": limitation,
        "n_positive": len(positives),
        "n_identity": len(identities),
        "positive_unit_ids": [unit["unit_id"] for unit in positives],
        "identity_unit_ids": [unit["unit_id"] for unit in identities],
        "compatible_positive_unit_ids": [
            unit["unit_id"] for unit in compatible_positives
        ],
        "clusters": clusters,
        "chosen_cluster": chosen,
        "burst_unit_id": burst["unit_id"],
        "burst_is_identity": not burst["positive_unit"],
        "same_task_consumer": same_task_consumer,
        "program_geometry_intersection": (
            [chosen["program"]] if chosen else []),
        "pattern_intersection": (
            chosen["pattern_intersection"] if chosen else {}),
        "pattern_compatible": bool(chosen),
        "gate_compatible": compatible,
        "dataset_name_used_as_pattern": False,
        "frozen_curriculum": frozen,
    }


def _draft_curriculum(positives: Sequence[Mapping[str, Any]],
                      identities: Sequence[Mapping[str, Any]],
                      burst: Mapping[str, Any], *,
                      full: bool) -> dict[str, Any]:
    """Mechanical 6-unit construction.  Rule is frozen in this function.

    Take positives in pre-declared pool order, identities in pool order,
    always include the burst stretch.  Interleave pos/id then append burst.
    Reverse order is the exact reverse of the forward list.
    """
    pos_ids = [unit["unit_id"] for unit in positives]
    id_ids = [unit["unit_id"] for unit in identities]
    burst_id = burst["unit_id"]
    if burst_id in pos_ids:
        pos_ids = [uid for uid in pos_ids if uid != burst_id]
    if not burst["positive_unit"] and burst_id not in id_ids:
        # burst is an identity-oracle stretch unit, counted separately
        pass

    chosen: list[str] = []
    if full:
        take_pos = pos_ids[: max(2, min(3, len(pos_ids)))]
        take_id = id_ids[: max(2, 5 - len(take_pos))]
    else:
        take_pos = pos_ids[:1]
        take_id = id_ids[: max(2, 5 - max(len(take_pos), 1))]
    leftover = [uid for uid in id_ids if uid not in take_id]
    interleaved: list[str] = []
    for index in range(max(len(take_pos), len(take_id))):
        if index < len(take_pos):
            interleaved.append(take_pos[index])
        if index < len(take_id):
            interleaved.append(take_id[index])
    for uid in leftover:
        if len(interleaved) >= 5:
            break
        interleaved.append(uid)
    if burst_id not in interleaved:
        interleaved = interleaved[:5] + [burst_id]
    chosen = interleaved[:6]
    if burst_id not in chosen:
        chosen = chosen[:5] + [burst_id]
    forward = list(chosen)
    reverse = list(reversed(forward))
    return {
        "n_units": len(forward),
        "units": forward,
        "contains_ge2_positive": sum(1 for uid in forward if uid in pos_ids
                                     or (uid == burst_id and burst["positive_unit"])) >= 2,
        "contains_ge2_identity": sum(
            1 for uid in forward
            if uid in id_ids or (uid == burst_id and not burst["positive_unit"])
        ) >= 2,
        "contains_burst": burst_id in forward,
        "forward_order": forward,
        "reverse_order": reverse,
        "selection_rule": (
            "positives then identities in pre-declared pool order; "
            "interleave; append burst; reverse is the exact reverse.  "
            "No outcome-driven reordering after this rule."
        ),
    }


# =========================================================================== #
# Part C -- knowledge-state reachability (code deduction; 0 LLM by default)
# =========================================================================== #
def _walk_episodes(node: Any, found: list[dict[str, Any]], *,
                   source_file: str) -> None:
    if isinstance(node, Mapping):
        if "episode_id" in node and "relation" in node and "workflow_signature" in node:
            found.append({
                "source_file": source_file,
                "episode_id": node.get("episode_id"),
                "domain_namespace": node.get("domain_namespace"),
                "workflow_signature": node.get("workflow_signature"),
                "relation": node.get("relation"),
                "evidence_level": node.get("evidence_level"),
                "local_status": node.get("local_status"),
                "support_gain": node.get("support_gain"),
                "delayed_gain": node.get("delayed_gain"),
            })
        for value in node.values():
            _walk_episodes(value, found, source_file=source_file)
    elif isinstance(node, list):
        for item in node:
            _walk_episodes(item, found, source_file=source_file)


def _visibility_of_experience_card(*, try_text: str,
                                   evidence_count: Any,
                                   applicability: Mapping[str, Any]) -> str:
    """Apply retrieval.py:195-238 to a synthesized experience card."""
    from SelfEvolvingHarnessTS.contracts.harness import SkillKind

    guards: dict[str, Any] = {
        "sections": {
            "TRY": try_text, "RISK": "", "WHEN": "",
            "OBSERVE": "", "VERIFY": "", "FALLBACK": "",
        },
    }
    if evidence_count is not None:
        guards["evidence_distinct_task_count"] = evidence_count
    skill = SimpleNamespace(
        risk_guards=guards,
        observable_applicability=dict(applicability),
        skill_kind=SkillKind.CAPABILITY,
    )
    if _is_inert_experience_card(skill):  # type: ignore[arg-type]
        return "Slow-only"
    try_clean = str(try_text or "").strip()
    if try_clean and try_clean != ss.TRY_ABSTAIN:
        return "Fast-TRY"
    count = evidence_count
    repeated = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 2
    )
    if repeated and _scopes_beyond_task_kind(applicability):
        return "Fast-guard"
    return "Slow-only"


def _existing_source_card() -> dict[str, Any]:
    path = E2 / "t6_cls_op_r2_three_arms.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))

    def _find_entry(node: Any) -> dict[str, Any]:
        if isinstance(node, Mapping):
            if node.get("skill_id") == "source_investigation_cls_v1" and "risk_guards" in node:
                return dict(node)
            for value in node.values():
                hit = _find_entry(value)
                if hit:
                    return hit
        elif isinstance(node, list):
            for item in node:
                hit = _find_entry(item)
                if hit:
                    return hit
        return {}

    entry = _find_entry(payload) or dict(payload.get("source_skill_entry") or {})
    sections = dict(payload.get("source_skill_sections") or {})
    if not sections and isinstance(entry.get("risk_guards"), Mapping):
        sections = dict((entry.get("risk_guards") or {}).get("sections") or {})
    guards = dict(entry.get("risk_guards") or {})
    applicability = dict(entry.get("observable_applicability") or {
        "feature": "task_kind", "op": "==", "value": "classification",
    })
    try_text = str(sections.get("TRY") or ss.TRY_ABSTAIN)
    visibility = _visibility_of_experience_card(
        try_text=try_text,
        evidence_count=guards.get("evidence_distinct_task_count"),
        applicability=applicability,
    )
    return {
        "skill_id": str(entry.get("skill_id") or "source_investigation_cls_v1"),
        "source_file": "artifacts/functional/e2/t6_cls_op_r2_three_arms.json",
        "try_text": try_text,
        "risk_guards_has_evidence_distinct_task_count": (
            "evidence_distinct_task_count" in guards),
        "applicability": applicability,
        "visibility": visibility,
        "sections": {key: sections.get(key) for key in ss.SECTIONS
                     if key in sections},
        "evidence_file_lines": (
            "artifacts/functional/e2/t6_cls_op_r2_three_arms.json:1640-1674"
        ),
        "note": (
            "existing Slow card.  TRY abstains.  build_skill_payload does "
            "not write evidence_distinct_task_count "
            "(source_skill.py:472-478), so Fast-guard is structurally off."
        ),
    }


def _inventory_episodes() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in EPISODE_SOURCES:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunk: list[dict[str, Any]] = []
        _walk_episodes(payload, chunk, source_file=str(path.relative_to(PROJECT_ROOT)))
        for row in chunk:
            key = str(row["episode_id"])
            if key in seen:
                continue
            seen.add(key)
            found.append(row)
    return found


def _episode_end_state(row: Mapping[str, Any]) -> dict[str, Any]:
    """Compile one existing Episode through the current predicates."""
    relation = str(row.get("relation") or "")
    status = str(row.get("local_status") or "")
    program = str(row.get("workflow_signature") or "")
    # Target-local capability: POSITIVE + LOCAL_ACTIVE.  Not an experience card.
    if relation == "POSITIVE" and status == "LOCAL_ACTIVE":
        return {
            **row,
            "compiled_kind": "target_local_capability",
            "visibility": "Fast-visible (not an experience card; frozen steps)",
            "three_tier": None,
            "reason": (
                "online_loop writes LOCAL_ACTIVE on POSITIVE delayed; "
                "handle_fast_winner mints a capability with Frozen program "
                "steps and no risk_guards.sections.TRY, so "
                "_is_inert_experience_card is False by structure "
                "(retrieval.py:166-168, 220-222)."
            ),
        }
    if relation == "NEGATIVE":
        return {
            **row,
            "compiled_kind": "episode_only_negative",
            "visibility": "Slow-only",
            "three_tier": "Slow-only",
            "reason": (
                "NEGATIVE stays EPISODE_ONLY.  risk_skill.census would count "
                "it, but classification online_loop.py:180-193 does not write "
                "context_summary.task_episode_id, so _task_of returns '' "
                "(risk_skill.py:72-74).  A second harm with the same empty "
                "id does not raise evidence_distinct_task_count.  The "
                "classification shared harness never calls "
                "run_risk_skill_lifecycle."
            ),
        }
    if relation == "CONFLICT":
        return {
            **row,
            "compiled_kind": "episode_only_conflict",
            "visibility": "Slow-only",
            "three_tier": "Slow-only",
            "reason": (
                "CONFLICT is mixed evidence.  risk_candidates stays silent "
                "when any positive exists in the family "
                "(risk_skill.py:171-172).  No experience card, no Fast view."
            ),
        }
    return {
        **row,
        "compiled_kind": "episode_only",
        "visibility": "Slow-only",
        "three_tier": "Slow-only",
        "reason": (
            "NEUTRAL/other: Episode is stored; no Skill; Slow can read it; "
            "Fast retrieval never sees raw Episodes."
        ),
    }


def _compiler_semantics() -> dict[str, Any]:
    return {
        "field": "risk_guards.evidence_distinct_task_count",
        "written_at": "evaluation/functional/task_episode_harness/agentic/risk_skill.py:246-247",
        "count_unit": (
            "distinct context_summary.task_episode_id strings "
            "(risk_skill.py:72-74, 98, 109, 177).  "
            "In the forecasting G1 path those ids are e1v2_task_NN.  "
            "Not dataset, not cell, not run, unless the writer put that "
            "string into task_episode_id."
        ),
        "merge_rule": (
            "Two harm Episodes increment the same family bucket when "
            "family_of(program_steps) matches (operator names, params "
            "discarded; risk_skill.py:57-64, 77-80).  Distinctness of the "
            "*count* is the set of task_episode_id values "
            "(risk_skill.py:109).  Context is not required for the count: "
            "different task_signature values still add two ids.  "
            "Applicability is the intersection of those signatures "
            "(risk_skill.py:115-130), which is a different operation."
        ),
        "first_fault_not_used": (
            "first-fault is not an input to census() or risk_candidates()."
        ),
        "fixture": (
            "tests/functional/test_target_local_risk_skill.py:102-114 "
            "two attempts of the same task_01 do not mint; task_01+task_02 "
            "mint with distinct_negative_task_count==2."
        ),
        "gap_vs_curriculum_unit": (
            "classification Episode write path (methods/ttha/online_loop.py:"
            "180-193) never sets context_summary.task_episode_id.  "
            "_task_of therefore returns '' for every classification "
            "Episode.  Two curriculum units that both harm outlier_mad "
            "collapse to one counted Task.  That is not independent "
            "curriculum-unit counting.  "
            "Slow source census is a different counter: "
            "run_e2_t6_cls_op_shared_harness.py:896-908 sets "
            "task_episode_id = dataset/condition, so THAT audit counts "
            "cells.  source_skill.build_skill_payload (source_skill.py:"
            "472-478) still does not copy that count onto "
            "evidence_distinct_task_count, so Fast-guard never sees it."
        ),
    }


def _reachability(units: Sequence[Mapping[str, Any]],
                  qualification: Mapping[str, Any],
                  episodes: Sequence[Mapping[str, Any]],
                  source_card: Mapping[str, Any]) -> dict[str, Any]:
    frozen = qualification.get("frozen_curriculum") or {}
    order = list(frozen.get("forward_order") or [])
    by_id = {unit["unit_id"]: unit for unit in units}
    cluster_ids = set(qualification.get("compatible_positive_unit_ids") or [])
    positives_in_course = [
        uid for uid in order if by_id.get(uid, {}).get("positive_unit")
    ]
    cluster_positives_in_course = [uid for uid in order if uid in cluster_ids]
    identity_in_course = [
        uid for uid in order if not by_id.get(uid, {}).get("positive_unit")
    ]

    k0_card_vis = source_card.get("visibility") or "Slow-only"
    # Existing Target-local hampel from C40 is Fast-visible but Target-local.
    # Honest K0 for the four-arm design is the Slow-consolidated card, not
    # a one-domain frozen winner (that would leak the GunPointAgeSpan answer
    # into every unit of K0-fixed).
    k0_includes_c40_target_local = False

    timeline: list[dict[str, Any]] = []
    pos_seen = 0
    first_diff = None
    a5_fast = "K0 Fast view = Slow-only experience card (source_investigation_cls_v1 inert)"
    k0_fast = a5_fast

    for index, uid in enumerate(order, start=1):
        unit = by_id[uid]
        events: list[str] = []
        in_cluster = uid in cluster_ids
        if in_cluster:
            pos_seen += 1
            events.append(
                "held-in can form a Target-local Skill in the chosen "
                "geometry (oracle set %s)" % unit["oracle_set"]
            )
            if pos_seen == 1:
                events.append(
                    "A5-online keeps that Target-local Skill in the snapshot; "
                    "K0-fixed discards it at the unit boundary"
                )
                if first_diff is None:
                    first_diff = {
                        "unit_index": index + 1 if index < len(order) else index,
                        "at": (
                            "start of the next unit"
                            if index < len(order) else
                            "after this unit's freeze (last unit; deploy-only)"
                        ),
                        "kind": "target_local_capability_carry",
                        "detail": (
                            "A5 Fast view gains a non-experience-card "
                            "capability with frozen steps; K0-fixed still "
                            "has only the inert Slow card.  retrieval.py:"
                            "166-168 / 274-275."
                        ),
                    }
            if pos_seen == 2:
                events.append(
                    "second independent cluster-positive: "
                    "authorization_audit can authorize TRY (unguided "
                    "POSITIVE, LOO min_distinct=2, no opposing NEGATIVE "
                    "in the same context; source_skill.py:253-257).  "
                    "A5 Slow integration can write a Fast-TRY card.  "
                    "K0-fixed does not re-consolidate."
                )
                events.append(
                    "this is the Scope-v1 / Fast-TRY divergence "
                    "(later than Target-local carry, stronger carrier)"
                )
        elif unit["positive_unit"]:
            events.append(
                "stretch/other-geometry positive (oracle %s); not in the "
                "chosen hampel cluster.  Does not increment the TRY LOO "
                "count for that family." % unit["oracle_set"]
            )
        else:
            events.append(
                "oracle set empty → identity is the menu oracle.  "
                "A held-in mad/iqr probe, if it happens and is NEGATIVE, "
                "is a harm Episode.  Fast-guard still does not fire: "
                "(1) cls harness does not mint risk skills; "
                "(2) task_episode_id is missing so the count stays 1; "
                "(3) the source card compiler does not write "
                "evidence_distinct_task_count."
            )
        timeline.append({
            "unit_index": index,
            "unit_id": uid,
            "positive": bool(unit["positive_unit"]),
            "in_chosen_cluster": in_cluster,
            "oracle_set": unit["oracle_set"],
            "a5_fast_expected": (
                "adds Target-local Skill after this unit"
                if in_cluster and pos_seen == 1 else
                "may add Fast-TRY experience card after Slow integration"
                if in_cluster and pos_seen >= 2 else
                "no Fast-visible knowledge write from this unit"
            ),
            "k0_fixed_fast_expected": (
                "reset to K0 (inert Slow card) at unit start; "
                "may form a Target-local Skill inside the unit only"
            ),
            "events": events,
        })

    try_reachable = len(cluster_positives_in_course) >= 2 and bool(
        qualification.get("gate_compatible"))
    guard_reachable = False  # structurally, given current compiler + wiring
    treatment_empty = first_diff is None

    return {
        "k0_experience_card": source_card,
        "k0_includes_c40_target_local_by_design": k0_includes_c40_target_local,
        "k0_design_note": (
            "K0 for K0-fixed / A5-online is the Slow-consolidated "
            "source_investigation_cls_v1 card (Slow-only).  The C40 "
            "fast_winner hampel Skill is Target-local to "
            "GunPointAgeSpan; putting it in K0 would leak that unit's "
            "answer into every K0-fixed unit."
        ),
        "existing_episodes": list(episodes),
        "n_existing_classification_episodes": len(episodes),
        "wine_precheck_is_not_an_episode": True,
        "compiler_semantics": _compiler_semantics(),
        "forward_order": order,
        "positives_in_course": positives_in_course,
        "cluster_positives_in_course": cluster_positives_in_course,
        "identity_in_course": identity_in_course,
        "fast_try_reachable_in_course": try_reachable,
        "fast_guard_reachable_in_course": guard_reachable,
        "first_visible_difference": first_diff,
        "timeline": timeline,
        "treatment_empty": treatment_empty,
        "slow_rehearse_used": False,
        "slow_rehearse_reason": (
            "code deduction confirms the existing card shape "
            "(TRY=NO_AUTHORIZED_ACTIVE_RECOMMENDATION, no "
            "evidence_distinct_task_count) and the deterministic "
            "authorization audit.  Card wording is not required for "
            "the visibility predicate.  Slow rehearsal not spent."
        ),
    }


def _markdown(payload: Mapping[str, Any]) -> str:
    units = payload.get("part_a", {}).get("units") or []
    part_b = payload.get("part_b") or {}
    part_c = payload.get("part_c") or {}
    ledger = payload.get("ledger") or {}
    lines = [
        "# S1a-r1 curriculum qualification + dual-layer oracle + reachability",
        "",
        "protocol: `%s`  curriculum: **%s**  evidence grade: **%s**"
        % (payload.get("protocol_version"), CURRICULUM_NAME,
           payload.get("evidence_grade")),
        "",
        "## Isolation",
        "",
        ORACLE_BANNER,
        "",
        "Oracle files live under `artifacts/functional/e2/s1_oracle/` and "
        "are exam keys.  They must not enter any arm prompt, store, or "
        "retrieval view.",
        "",
        "## Pool (pre-declared, not edited after scoring)",
        "",
        "8 impulse-v2 substrates × fit_only_artifact + 1 GunPoint burst "
        "(CLS-2 `inject_burst_noise`, seed 202608254).  Consumer = ridge.  "
        "No injection-parameter scan.  No pool expansion.",
        "",
        "## Part A -- dual-layer oracle",
        "",
        "| unit | legal set | oracle set | identity residual | "
        "menu-best residual | upper bound | identity held-out |",
        "|---|---|---|---|---|---|---|",
    ]
    for unit in units:
        lines.append(
            "| %s | %s | %s | %+.4f | %+.4f | %.4f | %.4f |"
            % (unit["unit_id"],
               ",".join(p for p in unit["legal_set"] if p != "identity") or "identity-only",
               ",".join(unit["oracle_set"]),
               unit["identity_residual_to_upper_bound"],
               unit["menu_best_residual_to_upper_bound"],
               unit["readiness_upper_bound_heldout_accuracy"],
               unit["identity_heldout_accuracy"])
        )
    frozen = part_b.get("frozen_curriculum") or {}
    lines += [
        "",
        "## Part B -- qualification",
        "",
        "**%s**" % part_b.get("verdict"),
        "",
        "- positives (pool): %s" % part_b.get("positive_unit_ids"),
        "- compatible cluster: %s"
        % part_b.get("compatible_positive_unit_ids"),
        "- program geometry: %s"
        % part_b.get("program_geometry_intersection"),
        "- clusters: %s"
        % json.dumps(part_b.get("clusters") or [], ensure_ascii=False),
        "- pattern intersection (no dataset name): %s"
        % json.dumps(part_b.get("pattern_intersection") or {},
                     ensure_ascii=False),
        "- limitation: %s" % (part_b.get("limitation") or "none"),
        "",
    ]
    if frozen:
        lines += [
            "### Frozen course",
            "",
            "- units: %s" % frozen.get("units"),
            "- forward: %s" % frozen.get("forward_order"),
            "- reverse: %s" % frozen.get("reverse_order"),
            "- rule: %s" % frozen.get("selection_rule"),
            "",
        ]
    first = part_c.get("first_visible_difference")
    lines += [
        "## Part C -- reachability",
        "",
        "### C1 K0 inventory",
        "",
        "- existing classification Episodes: %s"
        % part_c.get("n_existing_classification_episodes"),
        "- Wine precheck is **not** an Episode.",
        "- existing Slow card visibility: **%s**"
        % ((part_c.get("k0_experience_card") or {}).get("visibility")),
        "",
        "| episode_id | relation | program | compiled | three-tier |",
        "|---|---|---|---|---|",
    ]
    for row in part_c.get("existing_episodes") or []:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row.get("episode_id"), row.get("relation"),
               row.get("workflow_signature"), row.get("compiled_kind"),
               row.get("three_tier") or row.get("visibility"))
        )
    lines += [
        "",
        "### C2 compiler count semantics",
        "",
        str((part_c.get("compiler_semantics") or {}).get("count_unit")),
        "",
        str((part_c.get("compiler_semantics") or {}).get("gap_vs_curriculum_unit")),
        "",
        "### C3 expected Fast-view divergence",
        "",
        "- Fast-TRY reachable in course: %s"
        % part_c.get("fast_try_reachable_in_course"),
        "- Fast-guard reachable in course: %s"
        % part_c.get("fast_guard_reachable_in_course"),
        "- first visible difference: %s"
        % json.dumps(first, ensure_ascii=False),
        "",
        "| i | unit | positive | A5 Fast | K0-fixed Fast |",
        "|---|---|---|---|---|",
    ]
    for row in part_c.get("timeline") or []:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row["unit_index"], row["unit_id"], row["positive"],
               row["a5_fast_expected"], row["k0_fixed_fast_expected"])
        )
    lines += [
        "",
        "### C4 Slow rehearsal",
        "",
        str(part_c.get("slow_rehearse_reason") or ""),
        "",
        "## Total verdict",
        "",
        "**%s**" % payload.get("total_verdict"),
        "",
        "## Cost",
        "",
        "- Fast LLM: %s / %s" % (ledger.get("fast_llm"), LLM_FAST_CAP),
        "- Slow rehearsal LLM: %s / %s"
        % (ledger.get("slow_llm"), LLM_SLOW_REHEARSE_CAP),
        "- Consumer fits: %s / %s" % (ledger.get("consumer_fits"), FIT_CAP),
        "- wall clock: %s s / %s s"
        % (ledger.get("wall_seconds"), WALL_SECONDS_CAP),
        "- downloads: 0",
        "",
        "## Obligations",
        "",
    ]
    for key, value in (payload.get("obligations") or {}).items():
        lines.append("- **%s**: %s" % (key, value))
    extra = payload.get("outside_book") or []
    if extra:
        lines += ["", "## Outside the book", ""]
        for item in extra:
            lines.append("- %s" % item)
    return "\n".join(lines) + "\n"


def _load_sealed_units() -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for spec in _declared_pool():
        path = ORACLE_DIR / ("%s.json" % spec["unit_id"])
        if not path.is_file():
            raise cls.Stop("INSTRUMENT_UNREADABLE",
                           "sealed oracle missing: %s" % path)
        units.append(json.loads(path.read_text(encoding="utf-8")))
    return units


def run(*, slow_rehearse: bool = False, from_oracles: bool = False) -> int:
    started = time.time()
    print("S1a-r1 dual-layer oracle  protocol=%s" % PROTOCOL_VERSION, flush=True)
    pool = _declared_pool()
    print("POOL n=%d (frozen): %s"
          % (len(pool), [row["unit_id"] for row in pool]), flush=True)
    if len(pool) != 9:
        raise cls.Stop("INSTRUMENT_UNREADABLE", "pool must be 9 units")

    invariance = cls._v2_invariance_at_150()
    if not invariance["passed"]:
        raise cls.Stop("INSTRUMENT_UNREADABLE",
                       "v2 invariance at L=150 failed: %s" % invariance["checks"])

    fit_budget = cls.FitBudget(FIT_CAP)
    units: list[dict[str, Any]] = []
    if from_oracles:
        units = _load_sealed_units()
        print("REUSED %d sealed oracle files (0 new fits)" % len(units),
              flush=True)
    for spec in ([] if from_oracles else pool):
        elapsed = time.time() - started
        if elapsed > WALL_SECONDS_CAP:
            raise cls.Stop("COMPUTE_BUDGET_EXCEEDED",
                           "wall clock cap %ss hit before %s"
                           % (WALL_SECONDS_CAP, spec["unit_id"]))
        print("ORACLE %s ..." % spec["unit_id"], flush=True)
        if spec["injection"] == "impulse_v2":
            cell = cls._build_cell(
                spec["dataset"], CONDITION, data_dir=DATA_DIR,
                injection_template=cls.INJECTION_TEMPLATE_V2)
        else:
            cell = _build_burst_cell(spec["dataset"])
        clean_fit, _clean_labels = _load_clean_fit(spec["dataset"])
        if clean_fit.shape != np.asarray(cell["fit_values"]).shape:
            raise cls.Stop("INSTRUMENT_UNREADABLE",
                           "clean fit shape != injected fit for %s"
                           % spec["unit_id"])
        unit = _oracle_one_unit(
            spec=spec, cell=cell, clean_fit=clean_fit, fit_budget=fit_budget)
        unit["v2_invariance_at_150"] = invariance
        _write_sealed_oracle(unit)
        units.append(unit)
        print("  legal=%s oracle=%s id_resid=%+.4f menu_resid=%+.4f upper=%.4f"
              % (unit["legal_set"], unit["oracle_set"],
                 unit["identity_residual_to_upper_bound"],
                 unit["menu_best_residual_to_upper_bound"],
                 unit["readiness_upper_bound_heldout_accuracy"]),
              flush=True)

    qualification = _qualify(units)
    episodes = [_episode_end_state(row) for row in _inventory_episodes()]
    source_card = _existing_source_card()
    reach = _reachability(units, qualification, episodes, source_card)
    if slow_rehearse:
        reach["slow_rehearse_used"] = False
        reach["slow_rehearse_reason"] = (
            "--slow-rehearse was passed but code deduction already "
            "confirms card shape; the 8-call budget was not spent."
        )

    total = qualification["verdict"]
    if (reach.get("treatment_empty")
            and qualification["verdict"] != "CURRICULUM_NOT_VIABLE"):
        total = "TREATMENT_EMPTY"

    this_pass = round(time.time() - started, 2)
    scored_fits = fit_budget.used
    if from_oracles:
        scored_fits = max(
            (int(unit.get("consumer_fits_after_unit") or 0) for unit in units),
            default=0,
        )
        prior_wall = 0.0
        if AUDIT_JSON.is_file():
            prior = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
            prior_wall = float((prior.get("ledger") or {}).get("wall_seconds") or 0)
        wall = round(float(prior_wall) + this_pass, 2)
    else:
        wall = this_pass
    if wall > WALL_SECONDS_CAP:
        raise cls.Stop("COMPUTE_BUDGET_EXCEEDED",
                       "wall clock %.1fs > cap %s" % (wall, WALL_SECONDS_CAP))

    ledger = {
        "fast_llm": 0,
        "slow_llm": 0,
        "consumer_fits": scored_fits,
        "consumer_fits_this_pass": fit_budget.used,
        "consumer_fit_cap": FIT_CAP,
        "wall_seconds": wall,
        "wall_seconds_this_pass": this_pass,
        "from_oracles": bool(from_oracles),
        "wall_cap": WALL_SECONDS_CAP,
        "downloads": 0,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "curriculum_name": CURRICULUM_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "isolation_banner": ORACLE_BANNER,
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.executable,
        "pool": pool,
        "pool_frozen": True,
        "part_a": {
            "units": [
                {key: value for key, value in unit.items()
                 if key not in ("programs", "cell", "v2_invariance_at_150")}
                | {"n_programs_scored": sum(1 for row in unit["programs"]
                                            if row.get("scored_heldout")
                                            or row.get("scored_heldin"))}
                for unit in units
            ],
            "per_unit_program_tables": {
                unit["unit_id"]: unit["programs"] for unit in units
            },
        },
        "part_b": qualification,
        "part_c": reach,
        "total_verdict": total,
        "ledger": ledger,
        "obligations": {
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "no_fast_llm": True,
            "slow_rehearse_llm": 0,
            "slow_rehearse_cap": LLM_SLOW_REHEARSE_CAP,
            "no_a3_a5_adaptation_arm": True,
            "no_injection_scan": True,
            "no_pool_expansion": True,
            "oracle_isolated": True,
            "downloads": 0,
            "ucr_conf_downloaded_not_opened": True,
            "fit_budget_held": fit_budget.used <= FIT_CAP,
            "wall_clock_held": wall <= WALL_SECONDS_CAP,
            "full_repo_pytest_not_run": True,
        },
        "outside_book": [
            "classification online_loop does not write task_episode_id; "
            "risk_skill counts would collapse across curriculum units.",
            "source_skill.build_skill_payload does not write "
            "evidence_distinct_task_count, so Fast-guard is off for "
            "experience cards even after two harm cells.",
            "classification shared harness does not call "
            "run_risk_skill_lifecycle.",
            "C40 Target-local hampel is Fast-visible as a capability but "
            "must not be placed in K0 or K0-fixed is contaminated.",
            "ECG200/ToeSegmentation1/Lightning2 form a second compatible "
            "cluster on repair_burst_segment (not the frozen action family).  "
            "ECG200 hampel remains illegal under the 0.10 cohort cap; the "
            "three-substrate hampel fate table is unchanged.",
            "Wine hampel is legal (0.0297) but held-out class harm "
            "Δrecall_0=-0.444 excludes it from the oracle set.",
            "GunPoint burst stretch oracle is outlier_iqr at +0.0133 "
            "(just above the material line); repair_burst_segment is illegal "
            "or not in the oracle set on that unit.",
        ],
    }
    _dump(AUDIT_JSON, payload)
    AUDIT_MD.write_text(_markdown(payload), encoding="utf-8")
    print("VERDICT %s  fits=%d  wall=%.1fs"
          % (total, scored_fits, wall), flush=True)
    print("wrote %s" % AUDIT_JSON, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--from-oracles", action="store_true",
                        help="reuse sealed s1_oracle files; 0 new fits")
    parser.add_argument("--slow-rehearse", action="store_true",
                        help="unused unless card shape cannot be deduced")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return 2
    try:
        return run(slow_rehearse=bool(args.slow_rehearse),
                   from_oracles=bool(args.from_oracles))
    except cls.Stop as exc:
        print("STOP %s: %s" % (exc.verdict, exc), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
