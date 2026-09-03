"""S1a curriculum oracle + r2 legal reaggregation + r3 pool census.

Independent runner.  Reuses the Wine-precheck enumeration (cohort 0.10 cap,
full shared classification menu, ridge, fit_only_artifact) without modifying
the shared CLS-OP harness.  Writes sealed oracle artifacts that must never
enter any arm's prompt, store, or retrieval view.

  python evaluation/functional/run_e2_s1a_curriculum_oracle_audit.py --run
  python evaluation/functional/run_e2_s1a_curriculum_oracle_audit.py --legal-r2
  python evaluation/functional/run_e2_s1a_curriculum_oracle_audit.py --census-r3

``--legal-r2`` is 0-LLM / 0-fit: it only re-aggregates sealed ``s1_oracle``
JSON and walks live approval / retrieval code.  It writes
``s1a_r2_legal_treatment_audit.json/.md`` and never overwrites r1 artifacts
or the sealed oracles.

``--census-r3`` is the one-shot remaining-pool census (0 LLM).  It
pre-declares every remaining local binary substrate × {impulse-v2, burst},
scores the dual-layer oracle once, and exits POOL_EXHAUSTED or
LEGAL_CURRICULUM_CONSTRUCTIBLE.  There is no r4.

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
from io import BytesIO
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

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
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    CLASSIFICATION_MATERIAL_THRESHOLD,
    classify_relation,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    _is_inert_experience_card,
    _scopes_beyond_task_kind,
)
from SelfEvolvingHarnessTS.methods.ttha.signed_radius import (  # noqa: E402
    MATERIAL_THRESHOLD,
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
R2_PROTOCOL_VERSION = "s1a_r2_legal_treatment_audit_v1"
R2_RUN_ID = "s1a_r2_legal_treatment_audit1"
R2_JSON = E2 / "s1a_r2_legal_treatment_audit.json"
R2_MD = E2 / "s1a_r2_legal_treatment_audit.md"
R3_PROTOCOL_VERSION = "s1a_r3_pool_census_v1"
R3_RUN_ID = "s1a_r3_pool_census1"
R3_JSON = E2 / "s1a_r3_pool_census.json"
R3_MD = E2 / "s1a_r3_pool_census.md"
R3_FIT_CAP = 600
R3_TRAIN_POINT_CAP = 100000
R1_COMMIT = "837b537"
R2_COMMIT = "e74c021"
MIN_DISTINCT_TASKS = 2  # cls harness :168; source_skill LOO floor
GUNPOINT_FAMILY = ("GunPoint", "GunPointAgeSpan")
R1_TESTED_DATASETS = frozenset(IMPULSE_DATASETS)
R1_TESTED_UNIT_IDS = frozenset(
    ["%s__impulse_v2" % name for name in IMPULSE_DATASETS]
    + ["%s__%s" % (BURST_DATASET, BURST_INJECTION)]
)
# Longest-prefix first.  Phalanx/Phalanges is an infix rule, not a prefix.
R3_FAMILY_PREFIXES = (
    ("GunPoint", "GunPointFamily"),
    ("Freezer", "FreezerFamily"),
    ("SemgHand", "SemgHandFamily"),
    ("DodgerLoop", "DodgerLoopFamily"),
    ("SonyAIBO", "SonyAIBOFamily"),
    ("ToeSegmentation", "ToeSegmentationFamily"),
    ("Ford", "FordFamily"),
    ("ECG", "ECGFamily"),
)

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


def _write_sealed_oracle(unit: Mapping[str, Any],
                         extra: Mapping[str, Any] | None = None) -> None:
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
    if extra:
        payload.update(dict(extra))
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


# =========================================================================== #
# S1a-r2 -- legal treatment reaggregation (0 LLM / 0 fit)
# =========================================================================== #

def _family_key(dataset: str) -> str:
    if str(dataset).startswith("GunPoint"):
        return "GunPointFamily"
    return str(dataset)


def _heldin_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    """Apply live classify_relation to the sealed held-in oracle reading.

    The oracle scored ``_wine_heldin_pool`` (r1_support + r1_delayed +
    r2_support + r2_delayed) as one SUPPORT surface
    (run_e2_t6_cls_op_shared_harness.py:5539-5547; adapter allowed_surfaces
    = SUPPORT+HELDOUT only).  There is no separate delayed slice in the
    sealed JSON.  The same POSITIVE classifier is the Support Draft gate
    (method.py:742-757) and the delayed approve gate (method.py:1466-1492).
    Threshold is the live constant, not the oracle heldin_material_line.
    """
    program = str(row.get("program") or "")
    pcs = row.get("heldin_per_class_recall_delta")
    per_series = None
    if isinstance(pcs, Sequence) and not isinstance(pcs, (str, bytes)):
        per_series = {str(index): float(value)
                      for index, value in enumerate(pcs)}
    facts = classify_relation(
        aggregate_gain=row.get("heldin_headroom"),
        per_series_gains=per_series,
        is_identity=(program == "identity"),
        consumer_id=CONSUMER_ID,
    )
    return {
        "heldin_headroom": row.get("heldin_headroom"),
        "heldin_worst_class_recall_delta": row.get(
            "heldin_worst_class_recall_delta"),
        "heldin_per_class_recall_delta": list(pcs) if pcs else None,
        "heldout_utility": row.get("heldout_utility"),
        "heldout_worst_class_recall_delta": row.get(
            "heldout_worst_class_recall_delta"),
        "in_oracle_set": bool(row.get("in_oracle_set")),
        "legal": bool(row.get("legal")),
        "relation": facts["relation"],
        "classification_basis": facts["classification_basis"],
        "material_threshold": facts["material_threshold"],
        "would_pass_support_draft": facts["relation"] == "POSITIVE",
        "would_pass_delayed_approve": facts["relation"] == "POSITIVE",
        "instrument_note": (
            "single combined held-in pool; Support vs delayed were not "
            "scored separately; both live gates use classify_relation "
            "== POSITIVE (method.py:742-757 / 1466-1492; "
            "experience_memory.py:434-439; threshold "
            "CLASSIFICATION_MATERIAL_THRESHOLD=%s == MATERIAL_THRESHOLD=%s)"
            % (CLASSIFICATION_MATERIAL_THRESHOLD, MATERIAL_THRESHOLD)
        ),
    }


def _learnability_label(program: str, facts: Mapping[str, Any],
                        oracle_set: Sequence[str]) -> str:
    if program == "identity" or program not in oracle_set:
        return "N/A"
    if facts.get("would_pass_support_draft") and facts.get(
            "would_pass_delayed_approve"):
        return "LEARNABLE"
    return "HELDOUT_ONLY"


def _program_row(unit: Mapping[str, Any], program: str) -> dict[str, Any]:
    for row in unit.get("programs") or []:
        if row.get("program") == program:
            return dict(row)
    return {"program": program, "heldin_headroom": None}


def _r2_learnability(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    table: list[dict[str, Any]] = []
    for unit in units:
        oracle_set = [str(p) for p in (unit.get("oracle_set") or [])]
        judged = [p for p in oracle_set if p != "identity"]
        if not judged:
            judged = ["identity"] if "identity" in oracle_set else []
        if not judged:
            judged = ["identity"]
        rows = []
        labels = []
        for program in judged:
            raw = _program_row(unit, program)
            facts = _heldin_facts(raw)
            label = _learnability_label(program, facts, oracle_set)
            labels.append(label)
            rows.append({
                "program": program,
                "learnability": label,
                **facts,
                "oracle_json": (
                    "artifacts/functional/e2/s1_oracle/%s.json" % unit["unit_id"]
                ),
            })
        primary = rows[0] if rows else {}
        table.append({
            "unit_id": unit["unit_id"],
            "dataset": unit.get("dataset"),
            "family_key": _family_key(str(unit.get("dataset") or "")),
            "injection": unit.get("injection"),
            "n_heldin": unit.get("n_heldin"),
            "heldin_material_line": unit.get("heldin_material_line"),
            "heldin_material_line_is_not_the_approval_threshold": True,
            "approval_threshold": CLASSIFICATION_MATERIAL_THRESHOLD,
            "oracle_set": oracle_set,
            "positive_unit_r1": bool(unit.get("positive_unit")),
            "pattern_view": dict(unit.get("pattern_view") or {}),
            "learnability": primary.get("learnability") or "N/A",
            "oracle_program": primary.get("program"),
            "heldin_headroom": primary.get("heldin_headroom"),
            "heldin_relation": primary.get("relation"),
            "heldout_utility": primary.get("heldout_utility"),
            "operators": rows,
        })
    hampel_ids = [
        "GunPointAgeSpan__impulse_v2",
        "GunPoint__impulse_v2",
        "Herring__impulse_v2",
    ]
    burst_ids = [
        "ECG200__impulse_v2",
        "ToeSegmentation1__impulse_v2",
        "Lightning2__impulse_v2",
    ]
    by_id = {row["unit_id"]: row for row in table}

    def _cluster(name: str, program: str, ids: Sequence[str]) -> dict[str, Any]:
        members = []
        for uid in ids:
            row = by_id[uid]
            op = next((item for item in row["operators"]
                       if item["program"] == program), None)
            members.append({
                "unit_id": uid,
                "dataset": row["dataset"],
                "family_key": row["family_key"],
                "learnability": (op or {}).get("learnability") or "N/A",
                "heldin_headroom": (op or {}).get("heldin_headroom"),
                "heldin_relation": (op or {}).get("relation"),
                "heldout_utility": (op or {}).get("heldout_utility"),
                "independence_note": (
                    "GunPoint family; Scope uses features not names "
                    "(formally legal) but Source independence is weakened"
                    if row["family_key"] == "GunPointFamily" else None
                ),
            })
        learnable = [m for m in members if m["learnability"] == "LEARNABLE"]
        independent = sorted({m["family_key"] for m in learnable})
        return {
            "cluster": name,
            "program": program,
            "n_members": len(members),
            "n_learnable": len(learnable),
            "n_heldout_only": sum(
                1 for m in members if m["learnability"] == "HELDOUT_ONLY"),
            "n_independent_learnable_families": len(independent),
            "learnable_unit_ids": [m["unit_id"] for m in learnable],
            "heldout_only_unit_ids": [
                m["unit_id"] for m in members
                if m["learnability"] == "HELDOUT_ONLY"
            ],
            "independent_families": independent,
            "members": members,
        }

    return {
        "threshold_citations": {
            "classify_relation": (
                "methods/ttha/experience_memory.py:411-451 "
                "(agg >= +t and min per-view >= -t -> POSITIVE; "
                "t = CLASSIFICATION_MATERIAL_THRESHOLD = 0.005)"
            ),
            "support_draft": (
                "methods/ttha/method.py:742-757 handle_fast_winner; "
                "methods/ttha/online_loop.py:201-204 "
                "Support POSITIVE only forms Draft"
            ),
            "delayed_approve": (
                "methods/ttha/method.py:1466-1492 "
                "handle_feedback_delayed: classify_relation == POSITIVE; "
                "NEUTRAL no longer expands rights (T5 #41 A4)"
            ),
            "material_constants_equal": (
                CLASSIFICATION_MATERIAL_THRESHOLD == MATERIAL_THRESHOLD
            ),
            "not_used": (
                "oracle heldin_material_line = max(0.005, 1/n_heldin) is "
                "an instrument resolution line, not the approval gate"
            ),
        },
        "units": table,
        "hampel_cluster": _cluster(
            "hampel", "hampel_filter", hampel_ids),
        "repair_burst_cluster": _cluster(
            "repair_burst_segment", "repair_burst_segment", burst_ids),
        "gunpoint_family_note": (
            "GunPointAgeSpan and GunPoint share family_key=GunPointFamily "
            "and have identical pattern_view (byte-equal).  Scope v1 uses "
            "features not dataset names, so both may formally count as "
            "Source evidence; independence is weakened and is reported "
            "separately as n_independent_learnable_families."
        ),
    }


def _r2_rules() -> list[dict[str, Any]]:
    return [
        {
            "id": "a_no_target_local_carry",
            "rule": (
                "Target-local Skill 禁止跨单元携带进下一单元 Fast 视图。"
            ),
            "canon": [
                "AGENTS.md:174-175 Target-local Skill 在当前 Domain "
                "held-in Support 上形成，由同域 delayed 更新；冻结后仅同域 "
                "held-out 使用",
                "AGENTS.md:76-81 合法通路 = Source Episode → census → "
                "Slow consolidation → audited Source-derived Skill → Fast",
                "AGENTS.md:184-191 Fast 禁止读取未匹配当前 Domain 的 "
                "Source Target-local Card",
            ],
            "live_code_gap": (
                "现行 Target-local 卡由 "
                "run_e2_t6_cls_op_shared_harness.py:610-613 "
                "_card_builder 写入 observable_signature = "
                "{task_kind: classification}；method.py:89-105 "
                "_applicability_from_card 因此只编译 task_kind 叶；"
                "retrieval.py:278-282 evaluate_applicability 对所有 "
                "classification 单元为真。Target-local 卡不是经验卡"
                "（retrieval.py:158-164），T1 惰性闸口（retrieval.py:274）"
                "拦不住它。照跑会测到宽 Scope bug，不是合法演化。"
            ),
            "audit_applies": (
                "本审计按正典拦截跨单元 Target-local 携带，不按现行 "
                "task_kind-only 匹配放行。"
            ),
        },
        {
            "id": "b_heldin_positive_authorizes",
            "rule": (
                "单元计入可授权 Source 证据，须 held-in Support 与 delayed "
                "均被现役生命周期判为 POSITIVE（材料级正向）。禁止自造阈值。"
            ),
            "canon": [
                "AGENTS.md:174-175 / 139-146 Support 与 delayed 仅 held-in",
                "AGENTS.md:172-173 Episode 不自动获执行权",
            ],
            "live_code": [
                "experience_memory.py:398-451 classify_relation: "
                "agg >= +0.005 且逐 view >= -0.005 → POSITIVE；"
                "cls_scope_adapter.py:31-36 分类 view = 逐类 recall",
                "method.py:742-757 handle_fast_winner Support != POSITIVE "
                "→ support_rejected，不形成 Draft",
                "online_loop.py:201-204 Support = POSITIVE 才 LOCAL_DRAFT",
                "method.py:1466-1492 handle_feedback_delayed 改为 "
                "classify_relation == POSITIVE 才 approved；NEUTRAL / "
                "CONFLICT / NEGATIVE 丢弃 pending（旧门 dg >= -0.005 已废）",
                "signed_radius.py:40 MATERIAL_THRESHOLD = 0.005",
            ],
            "oracle_proxy": (
                "密封 oracle 只评了拼接 held-in 池一次"
                "（_wine_heldin_pool）。本审计把该读数送入同一个 "
                "classify_relation，作为 Support 与 delayed 两道门的代理；"
                "不发明四分切片，也不改用 heldin_material_line。"
            ),
        },
        {
            "id": "c_unguided_authorizes_try",
            "rule": (
                "仅未受旧 Skill 引导的正例可授权新 Shared TRY。"
                "未引导 = 该单元 Fast 视图不存在指向同 Program 族的 "
                "TRY / capability 卡。"
            ),
            "canon": [
                "AGENTS.md:176-177 Shared Capability 需多 Domain 重复正向",
            ],
            "live_code": [
                "source_skill.py:217-257 authorization_audit: 仅 UNGUIDED "
                "POSITIVE 可授权新 TRY；conditioned 只可确认/反驳/撤回",
                "source_skill.py:249-256 LOO: 去掉任一 Task 后 UNGUIDED "
                "POSITIVE 仍须 >= min_distinct_tasks（cls harness :168 "
                "MIN_DISTINCT_TASKS=2）→ 2 个正例 loo_minimum=1，"
                "TRY 不授权（does_not_survive_leave_one_out）",
                "retrieval.py:195-238 / 274 T1: 无授权 TRY 且无重复 "
                "scoped RISK 的经验卡 Fast 不可见",
            ],
        },
    ]


def _pattern_matches(unit_pattern: Mapping[str, Any],
                     scope: Mapping[str, Any]) -> bool:
    if not scope:
        return False
    return all(unit_pattern.get(key) == value for key, value in scope.items())


def _try_audit(program: str, unguided_ids: Sequence[str]) -> dict[str, Any]:
    probes = [{
        "program": program,
        "context_condition": True,
        "task_episode_id": uid,
        "relation": "POSITIVE",
        "conditioned_snapshot": False,
    } for uid in unguided_ids]
    rows = ss.authorization_audit(
        probes, min_distinct_tasks=MIN_DISTINCT_TASKS)
    hit = next((row for row in rows if row["program"] == program), None)
    return hit or {
        "active_try_authorized": False,
        "leave_one_out_minimum_positive": 0,
        "unguided_positive": len(unguided_ids),
        "withheld_because": "no_unguided_positive",
    }


def _legal_timeline(
    order: Sequence[str],
    learn: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    by_id = {row["unit_id"]: row for row in learn["units"]}
    unguided: dict[str, list[str]] = {}
    source_cards: dict[str, dict[str, Any]] = {}
    timeline: list[dict[str, Any]] = []
    first_fast_diff = None

    for index, uid in enumerate(order, start=1):
        unit = by_id[uid]
        program = unit["oracle_program"]
        learnability = unit["learnability"]
        legality: list[str] = []
        events: list[str] = []

        matching_cards = []
        for prog, card in source_cards.items():
            if card.get("fast_visible") and _pattern_matches(
                    unit["pattern_view"], card.get("scope") or {}):
                matching_cards.append(prog)
        guided = program in matching_cards and program != "identity"
        a5_fast = (
            "Source-derived TRY Fast-visible for %s (Scope v1 match)"
            % ",".join(matching_cards)
            if matching_cards else
            "K0 inert Slow card only; no Target-local carry"
        )
        k0_fast = "K0 inert Slow card; in-unit Target-local only, discarded at boundary"
        if matching_cards and first_fast_diff is None:
            first_fast_diff = {
                "unit_index": index,
                "unit_id": uid,
                "kind": "source_derived_try_fast_visible",
                "programs": list(matching_cards),
                "approvable": bool(
                    learnability == "LEARNABLE"
                    and program in matching_cards
                ),
            }

        episode_kind = "none"
        if program == "identity" or learnability == "N/A":
            episode_kind = "identity_or_empty"
            events.append(
                "oracle set is identity/empty.  Episode if any is ABSTAIN "
                "(classify_relation is_identity; experience_memory.py:428-430)."
            )
            legality.append(
                "L-EP AGENTS.md:172-173 Episode 可记，不获执行权"
            )
        elif learnability == "LEARNABLE":
            episode_kind = "positive_unguided" if not guided else "positive_guided"
            events.append(
                "held-in classify_relation=POSITIVE (headroom=%s).  "
                "Support Draft + delayed approve would both pass "
                "(method.py:742-757 / 1466-1492)."
                % unit["heldin_headroom"]
            )
            events.append(
                "Target-local Skill may form in-domain; it is NOT carried "
                "into the next unit Fast view."
            )
            legality.append(
                "L-TL-FORM AGENTS.md:174-175 + method.py:742-757 + "
                "online_loop.py:201-204 + method.py:1466-1492"
            )
            legality.append(
                "L-TL-NOCARRY AGENTS.md:174-175,184-191; 不按 "
                "task_kind-only 宽 Scope 放行"
            )
            if guided:
                events.append(
                    "Fast already has a same-family TRY/capability card; "
                    "this POSITIVE is conditioned and cannot authorize a "
                    "new Shared TRY (source_skill.py:217-221)."
                )
                legality.append("L-UNGUIDED source_skill.py:217-221")
            else:
                unguided.setdefault(program, []).append(uid)
                legality.append("L-UNGUIDED 本单元 Fast 无同族 TRY 卡")
        else:
            episode_kind = "heldout_only_not_authorizing"
            events.append(
                "oracle-set program is HELDOUT_ONLY "
                "(held-in relation=%s, headroom=%s, held-out utility=%s).  "
                "Target feedback would not approve; not Source evidence."
                % (unit["heldin_relation"], unit["heldin_headroom"],
                   unit["heldout_utility"])
            )
            legality.append(
                "L-APPROVE method.py:1466-1492 relation != POSITIVE → "
                "delayed_rejected; experience_memory.py:449-451 NEUTRAL "
                "is |agg| < 0.005"
            )

        formed = None
        if program not in (None, "identity") and learnability == "LEARNABLE":
            ids = list(unguided.get(program) or [])
            members = [by_id[item] for item in ids]
            scope = _intersect_maps([m["pattern_view"] for m in members]) if len(
                members) >= 2 else {}
            independent = sorted({m["family_key"] for m in members})
            audit = _try_audit(program, ids)
            scope_v1_ok = len(members) >= 2 and bool(scope)
            independent_ok = len(independent) >= 2 and bool(scope)
            fast_visible = bool(audit.get("active_try_authorized"))
            later = []
            for later_uid in order[index:]:
                later_unit = by_id[later_uid]
                match = _pattern_matches(later_unit["pattern_view"], scope)
                later.append({
                    "unit_id": later_uid,
                    "pattern_match": match,
                    "learnability": later_unit["learnability"],
                    "same_program": later_unit["oracle_program"] == program,
                    "approvable_field": bool(
                        match
                        and later_unit["learnability"] == "LEARNABLE"
                        and later_unit["oracle_program"] == program
                    ),
                })
            formed = {
                "program": program,
                "unguided_learnable_ids": ids,
                "n_unguided": len(ids),
                "n_independent_families": len(independent),
                "independent_families": independent,
                "scope_v1_can_form_candidate": scope_v1_ok,
                "independent_enough_for_scope_v1": independent_ok,
                "scope": scope,
                "authorization_audit": audit,
                "fast_visible_try": fast_visible,
                "later_units": later,
            }
            source_cards[program] = {
                "scope": scope,
                "fast_visible": fast_visible,
                "formed_after_unit": uid,
            }
            if scope_v1_ok:
                events.append(
                    "Slow may write a Source-derived candidate "
                    "(Scope v1: n=%d formal learnable, intersection %s)."
                    % (len(ids), "non-empty" if scope else "empty")
                )
                legality.append(
                    "L-SLOW AGENTS.md:76-81; L-SCOPE Scope v1 五轴 "
                    "(STAGE_REPORT 2026-08-25 20:1x)"
                )
            else:
                events.append(
                    "Slow cannot form Source-derived Skill yet "
                    "(unguided learnable=%d, independent_families=%d, "
                    "intersection=%s)."
                    % (len(ids), len(independent),
                       "non-empty" if scope else "empty")
                )
                legality.append("L-SCOPE 未满 ≥2 独立可学正例或交为空")
            if len(independent) < len(ids):
                events.append(
                    "independence weakened: formal %d / independent %d "
                    "(GunPoint family; identical pattern_view)."
                    % (len(ids), len(independent))
                )
            events.append(
                "authorization_audit TRY authorized=%s loo_min=%s "
                "withheld=%s (source_skill.py:249-256; "
                "MIN_DISTINCT_TASKS=%d)."
                % (audit.get("active_try_authorized"),
                   audit.get("leave_one_out_minimum_positive"),
                   audit.get("withheld_because"),
                   MIN_DISTINCT_TASKS)
            )
            legality.append("L-LOO source_skill.py:249-256")
            if fast_visible:
                events.append(
                    "TRY authorized → experience card Fast-visible "
                    "on subsequent Scope-matching units."
                )
                legality.append("L-T1 retrieval.py:274 非 inert，可进 Fast")
            else:
                events.append(
                    "TRY not authorized → T1 inert experience card "
                    "withheld from Fast (retrieval.py:274).  A5 Fast "
                    "still equals K0 on this surface."
                )
                legality.append("L-T1 retrieval.py:274 inert → Fast 不可见")

        timeline.append({
            "unit_index": index,
            "unit_id": uid,
            "oracle_program": program,
            "learnability": learnability,
            "heldin_headroom": unit["heldin_headroom"],
            "heldin_relation": unit["heldin_relation"],
            "episode": episode_kind,
            "guided": guided,
            "a5_fast": a5_fast,
            "k0_fast": k0_fast,
            "slow_integration": formed,
            "events": events,
            "legality": legality,
        })

    approvable = []
    for row in timeline:
        formed = row.get("slow_integration") or {}
        for later in formed.get("later_units") or []:
            if later.get("approvable_field") and formed.get("fast_visible_try"):
                approvable.append({
                    "after_unit": row["unit_id"],
                    "field": later["unit_id"],
                    "program": formed.get("program"),
                })

    return {
        "label": label,
        "order": list(order),
        "timeline": timeline,
        "first_legal_fast_visible_difference": first_fast_diff,
        "approvable_transfer_channels": approvable,
        "has_legal_approvable_channel": bool(approvable),
    }


def _search_reorganization(learn: Mapping[str, Any]) -> dict[str, Any]:
    """Mechanical 9-unit search: >=2 learnable sources before >=1 field.

    Official learnable = oracle-set program is LEARNABLE.  Scope = pattern
    intersection of the chosen sources (Scope v1 axis 4).  A field must
    match that Scope, share the program, and be LEARNABLE.
    """
    by_id = {row["unit_id"]: row for row in learn["units"]}
    pool = list(by_id)
    programs: dict[str, list[str]] = {}
    for row in learn["units"]:
        if row["learnability"] == "LEARNABLE" and row["oracle_program"] not in (
                None, "identity"):
            programs.setdefault(row["oracle_program"], []).append(row["unit_id"])

    trials: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for program, sources in sorted(programs.items()):
        scope_all = _intersect_maps(
            [by_id[uid]["pattern_view"] for uid in sources])
        fields = [
            uid for uid in pool
            if by_id[uid]["oracle_program"] == program
            and by_id[uid]["learnability"] == "LEARNABLE"
            and _pattern_matches(by_id[uid]["pattern_view"], scope_all)
        ]
        independent = sorted({by_id[uid]["family_key"] for uid in sources})
        trial = {
            "program": program,
            "learnable_source_ids": sources,
            "n_learnable": len(sources),
            "n_independent_families": len(independent),
            "independent_families": independent,
            "scope_of_all_learnable": scope_all,
            "matching_learnable_field_ids": fields,
            "n_matching_learnable_fields": len(fields),
            "can_place_2_before_1_after": len(fields) >= 3 or (
                len(sources) >= 2 and any(
                    uid not in sources[:2] for uid in fields
                ) and len(set(sources) | set(fields)) >= 3
            ),
            "blocked_because": [],
        }
        if len(sources) < 2:
            trial["blocked_because"].append("fewer_than_2_learnable_sources")
        if len(independent) < 2:
            trial["blocked_because"].append(
                "fewer_than_2_independent_families")
        if not scope_all:
            trial["blocked_because"].append("empty_pattern_intersection")
        if len(set(fields)) < 3 and set(fields) <= set(sources) and len(
                sources) < 3:
            trial["blocked_because"].append(
                "no_external_or_third_learnable_matching_field"
            )
        # 2-before-1-after exists iff at least 3 distinct LEARNABLE
        # matching units for this program+scope, so two can sit in front
        # and one behind.
        if len(set(fields) | set(sources)) >= 3 and len(sources) >= 2 and scope_all:
            # still require the field to match the scope of the *front* pair
            pair_ok = False
            for front in combinations(sources, 2):
                front_scope = _intersect_maps(
                    [by_id[uid]["pattern_view"] for uid in front])
                if not front_scope:
                    continue
                for field in set(fields) | set(sources):
                    if field in front:
                        continue
                    if by_id[field]["learnability"] != "LEARNABLE":
                        continue
                    if by_id[field]["oracle_program"] != program:
                        continue
                    if not _pattern_matches(by_id[field]["pattern_view"],
                                            front_scope):
                        continue
                    pair_ok = True
                    fillers = [
                        uid for uid in pool
                        if uid not in front and uid != field
                    ]
                    course = list(front) + [field] + fillers
                    course = course[:6]
                    if len(course) < 6:
                        continue
                    candidates.append({
                        "status": "pending_arbitration",
                        "program": program,
                        "forward_order": course,
                        "reverse_order": list(reversed(course)),
                        "sources_in_front": list(front),
                        "field": field,
                        "front_scope": front_scope,
                        "note": (
                            "mechanical 2+1 hit under Scope v1 / task-book "
                            "search.  Not an approved curriculum."
                        ),
                    })
            trial["can_place_2_before_1_after"] = pair_ok
            if not pair_ok and "no_external_or_third_learnable_matching_field" not in trial["blocked_because"]:
                trial["blocked_because"].append(
                    "no_front_pair_whose_scope_matches_a_later_learnable_field"
                )
        trials.append(trial)

    return {
        "search_rule": (
            "over the frozen 9-unit pool, no expansion, no rescoring; "
            "a 6-unit course is a hit iff some Program has >=2 LEARNABLE "
            "oracle-set sources whose Scope-v1 pattern intersection is "
            "non-empty, and a later unit is a LEARNABLE matching field "
            "for that same Program.  GunPoint family independence is "
            "reported, not used as a silent veto on the mechanical hit "
            "test (a hit still carries the independence note)."
        ),
        "trials": trials,
        "candidates_pending_arbitration": candidates,
        "n_candidates": len(candidates),
    }


def _s1b_spec() -> dict[str, Any]:
    return {
        "title": "S1b runner-layer domain binding (text spec, no code this book)",
        "why_protocol_not_rewriting_the_exam": (
            "The exam asks whether legal evolution changes Fast behaviour.  "
            "Current Target-local cards match every classification unit "
            "(task_kind-only applicability).  Running S1b against that "
            "matcher measures the wide-Scope bug (copying a frozen winner "
            "across domains), not Harness evolution.  A runner-layer "
            "filter implements AGENTS.md:174-191 as already written.  "
            "It does not change held-in budgets, menus, Consumer, splits, "
            "or oracle keys."
        ),
        "minimal_runner_hooks": [
            {
                "when": "cell / unit construction",
                "do": (
                    "stamp every newly minted Target-local Skill with "
                    "domain_namespace = current unit dataset (already on "
                    "the Episode; copy it onto the Skill entry or a "
                    "runner-owned side table).  Do not put dataset name "
                    "into observable_applicability."
                ),
            },
            {
                "when": "cross-unit snapshot carry into the next Fast view",
                "do": (
                    "drop any Target-local capability (frozen program "
                    "steps; not an experience card — retrieval.py:158-164) "
                    "whose domain_namespace != current unit.  This is the "
                    "AGENTS.md:184-191 wall."
                ),
            },
            {
                "when": "Source-derived experience cards",
                "do": (
                    "admit them to Fast only when Scope v1 matches: "
                    "task_kind × consumer_id × metric × pattern_view "
                    "intersection × Program geometry.  Dataset name is "
                    "not an axis.  If methods-layer Scope compile step ③ "
                    "is not yet live, the runner evaluates this 5-axis "
                    "predicate as an exam-wall before retrieve."
                ),
            },
        ],
        "relation_to_methods_step3": (
            "四步修复序第③步（STAGE_REPORT 2026-08-25 17:3x / 17:35）"
            "才是 methods 层 Scope 编译：Target-local 限本域；跨域绑 "
            "Task×Consumer×Metric+部署可见 Pattern+Program 几何。"
            "本书不改 methods/。理由：单假设纪律；③ 是行为机制变更，"
            "需要自己的切片、锁与测试；S1b 只需要考试墙与正典对齐，"
            "避免把宽 Scope bug 当成处理组。③ 落地后删除 runner 墙，"
            "不得长期叠两道门。"
        ),
        "do_not": [
            "do not patch methods/ttha/method.py _applicability_from_card",
            "do not invent Pattern thresholds from seen outcomes",
            "do not put C40 Target-local hampel into K0",
            "do not treat task_kind-only match as Scope v1",
        ],
    }


def _r2_markdown(payload: Mapping[str, Any]) -> str:
    learn = payload.get("learnability") or {}
    rules = payload.get("rules") or []
    fwd = payload.get("forward_timeline") or {}
    rev = payload.get("reverse_timeline") or {}
    search = payload.get("reorganization_search") or {}
    ledger = payload.get("ledger") or {}
    lines = [
        "# S1a-r2 legal evolution treatment audit",
        "",
        "protocol: `%s`  parent r1: `%s`  evidence grade: **development**"
        % (payload.get("protocol_version"),
           (payload.get("parent_r1") or {}).get("commit")),
        "",
        "0 LLM / 0 fit.  Sealed oracles reused, not rescored.  "
        "r1 artifacts not overwritten.",
        "",
        "## 1. Three legality rules",
        "",
    ]
    for rule in rules:
        lines += [
            "### %s" % rule["id"],
            "",
            rule["rule"],
            "",
        ]
        for cite in rule.get("canon") or []:
            lines.append("- canon: %s" % cite)
        for cite in rule.get("live_code") or []:
            lines.append("- live: %s" % cite)
        if rule.get("live_code_gap"):
            lines.append("- gap: %s" % rule["live_code_gap"])
        if rule.get("oracle_proxy"):
            lines.append("- proxy: %s" % rule["oracle_proxy"])
        if rule.get("audit_applies"):
            lines.append("- audit: %s" % rule["audit_applies"])
        lines.append("")
    lines += [
        "## 2. Learnability (oracle-set operators only)",
        "",
        "Approval proxy = `classify_relation` on the sealed combined "
        "held-in reading.  Threshold = 0.005 "
        "(experience_memory.py:408 / signed_radius.py:40).  "
        "The oracle `heldin_material_line` is **not** used.",
        "",
        "| unit | oracle program | held-in | relation | held-out Δacc | label |",
        "|---|---|---|---|---|---|",
    ]
    for row in learn.get("units") or []:
        lines.append(
            "| %s | %s | %s | %s | %s | **%s** |"
            % (row["unit_id"], row.get("oracle_program"),
               row.get("heldin_headroom"), row.get("heldin_relation"),
               row.get("heldout_utility"), row.get("learnability"))
        )
    hampel = learn.get("hampel_cluster") or {}
    burst = learn.get("repair_burst_cluster") or {}
    lines += [
        "",
        "### Cluster learnable counts",
        "",
        "- hampel (GPA / GunPoint / Herring): LEARNABLE **%s**/3 "
        "(%s); HELDOUT_ONLY %s; independent families **%s** (%s)."
        % (hampel.get("n_learnable"), hampel.get("learnable_unit_ids"),
           hampel.get("heldout_only_unit_ids"),
           hampel.get("n_independent_learnable_families"),
           hampel.get("independent_families")),
        "- repair_burst_segment (ECG200 / Toe / Lightning2): LEARNABLE "
        "**%s**/3 (%s); HELDOUT_ONLY %s; independent families **%s**."
        % (burst.get("n_learnable"), burst.get("learnable_unit_ids"),
           burst.get("heldout_only_unit_ids"),
           burst.get("n_independent_learnable_families")),
        "- GunPoint↔GPA: %s" % learn.get("gunpoint_family_note"),
        "",
        "## 3. Legal timelines on the r1-frozen 6-unit course",
        "",
    ]

    def _emit_timeline(block: Mapping[str, Any], title: str) -> None:
        lines.append("### %s" % title)
        lines.append("")
        lines.append("order: `%s`" % block.get("order"))
        lines.append("")
        lines.append(
            "first legal Fast-visible difference: **%s**"
            % json.dumps(block.get("first_legal_fast_visible_difference"),
                         ensure_ascii=False)
        )
        lines.append("")
        lines.append(
            "approvable transfer channels: **%s**"
            % json.dumps(block.get("approvable_transfer_channels"),
                         ensure_ascii=False)
        )
        lines.append("")
        lines.append(
            "| i | unit | learnability | episode | A5 Fast | legality |"
        )
        lines.append("|---|---|---|---|---|---|")
        for row in block.get("timeline") or []:
            lines.append(
                "| %s | %s | %s | %s | %s | %s |"
                % (row["unit_index"], row["unit_id"], row["learnability"],
                   row["episode"], row["a5_fast"],
                   "; ".join(row.get("legality") or []))
            )
        lines.append("")
        for row in block.get("timeline") or []:
            lines.append("#### unit %s %s" % (row["unit_index"], row["unit_id"]))
            lines.append("")
            for event in row.get("events") or []:
                lines.append("- %s" % event)
            formed = row.get("slow_integration")
            if formed:
                lines.append(
                    "- Slow: candidate=%s independent=%s TRY=%s loo=%s "
                    "withheld=%s"
                    % (formed.get("scope_v1_can_form_candidate"),
                       formed.get("n_independent_families"),
                       (formed.get("authorization_audit") or {}).get(
                           "active_try_authorized"),
                       (formed.get("authorization_audit") or {}).get(
                           "leave_one_out_minimum_positive"),
                       (formed.get("authorization_audit") or {}).get(
                           "withheld_because"))
                )
            lines.append("")

    _emit_timeline(fwd, "Forward")
    _emit_timeline(rev, "Reverse")
    lines += [
        "## 4. Verdict",
        "",
        "**%s**" % payload.get("verdict"),
        "",
        str(payload.get("verdict_reason") or ""),
        "",
        "### Reorganization search (9-unit pool, no expansion)",
        "",
        str(search.get("search_rule") or ""),
        "",
        "candidates pending arbitration: **%s**"
        % search.get("n_candidates"),
        "",
    ]
    for trial in search.get("trials") or []:
        lines.append(
            "- program `%s`: learnable=%s independent_families=%s "
            "matching_fields=%s blocked=%s"
            % (trial.get("program"), trial.get("learnable_source_ids"),
               trial.get("independent_families"),
               trial.get("matching_learnable_field_ids"),
               trial.get("blocked_because"))
        )
    if search.get("candidates_pending_arbitration"):
        lines += ["", "Candidate courses:", ""]
        for cand in search["candidates_pending_arbitration"]:
            lines.append("- %s" % json.dumps(cand, ensure_ascii=False))
    spec = payload.get("s1b_domain_binding_spec") or {}
    lines += [
        "",
        "## 5. S1b domain-binding spec",
        "",
        spec.get("why_protocol_not_rewriting_the_exam") or "",
        "",
    ]
    for hook in spec.get("minimal_runner_hooks") or []:
        lines.append("- **%s**: %s" % (hook.get("when"), hook.get("do")))
    lines += [
        "",
        spec.get("relation_to_methods_step3") or "",
        "",
        "## Cost",
        "",
        "- Fast LLM: %s" % ledger.get("fast_llm"),
        "- Slow LLM: %s" % ledger.get("slow_llm"),
        "- Consumer fits: %s (this pass %s)"
        % (ledger.get("consumer_fits"), ledger.get("consumer_fits_this_pass")),
        "- wall clock: %s s / %s s"
        % (ledger.get("wall_seconds"), ledger.get("wall_cap")),
        "- downloads: 0",
        "- oracle rescoring: 0",
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


def run_legal_r2() -> int:
    started = time.time()
    print("S1a-r2 legal treatment reaggregation  protocol=%s"
          % R2_PROTOCOL_VERSION, flush=True)
    units = _load_sealed_units()
    print("REUSED %d sealed oracle files (0 new fits)" % len(units),
          flush=True)
    if len(units) != 9:
        raise cls.Stop("INSTRUMENT_UNREADABLE",
                       "r2 pool must be the 9 sealed units")

    learn = _r2_learnability(units)
    parent = {}
    if AUDIT_JSON.is_file():
        parent = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    frozen = ((parent.get("part_b") or {}).get("frozen_curriculum") or {})
    forward = list(frozen.get("forward_order") or [])
    reverse = list(frozen.get("reverse_order") or [])
    if not forward:
        raise cls.Stop("INSTRUMENT_UNREADABLE",
                       "r1 frozen curriculum missing; will not invent an order")

    fwd = _legal_timeline(forward, learn, label="forward")
    rev = _legal_timeline(reverse, learn, label="reverse")
    search = _search_reorganization(learn)

    if fwd["has_legal_approvable_channel"] or rev["has_legal_approvable_channel"]:
        verdict = "LEGAL_EVOLUTION_TREATMENT_QUALIFIED"
        reason = (
            "at least one frozen order has a Fast-visible authorized TRY "
            "and a later LEARNABLE matching field that Target can approve."
        )
    else:
        verdict = "HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH"
        reason = (
            "the frozen 6-unit course still has LEARNABLE oracle-set "
            "operators (hampel on GPA and GunPoint; repair_burst on Toe "
            "and Lightning2), but neither frozen order has a legal, "
            "unguided, subsequently-approvable transfer channel.  "
            "Target-local carry is forbidden.  Shared TRY is not "
            "authorized (LOO needs 3 unguided positives; each cluster "
            "has only 2 LEARNABLE members; hampel independence is 1 "
            "family).  T1 therefore withholds any Slow candidate from "
            "Fast.  Mechanical search over the same 9-unit pool found "
            "%d pending-arbitration 2+1 rearrangement(s)."
            % search["n_candidates"]
        )

    wall = round(time.time() - started, 2)
    payload = {
        "protocol_version": R2_PROTOCOL_VERSION,
        "run_id": R2_RUN_ID,
        "curriculum_name": CURRICULUM_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "isolation_banner": ORACLE_BANNER,
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.executable,
        "parent_r1": {
            "commit": R1_COMMIT,
            "artifact": "artifacts/functional/e2/s1a_curriculum_audit.json",
            "r1_verdict": parent.get("total_verdict"),
            "r1_verdict_narrowed_by_arbitration": True,
            "frozen_curriculum": frozen,
        },
        "rules": _r2_rules(),
        "learnability": learn,
        "forward_timeline": fwd,
        "reverse_timeline": rev,
        "verdict": verdict,
        "verdict_reason": reason,
        "reorganization_search": search,
        "s1b_domain_binding_spec": _s1b_spec(),
        "ledger": {
            "fast_llm": 0,
            "slow_llm": 0,
            "consumer_fits": 0,
            "consumer_fits_this_pass": 0,
            "consumer_fit_cap": FIT_CAP,
            "wall_seconds": wall,
            "wall_seconds_this_pass": wall,
            "from_oracles": True,
            "oracle_rescored": False,
            "wall_cap": WALL_SECONDS_CAP,
            "downloads": 0,
        },
        "obligations": {
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "no_fast_llm": True,
            "no_slow_llm": True,
            "no_a3_a5_adaptation_arm": True,
            "no_oracle_rescore": True,
            "no_injection_scan": True,
            "no_pool_expansion": True,
            "r1_artifacts_not_overwritten": True,
            "sealed_oracles_not_rewritten": True,
            "downloads": 0,
            "this_book_ran_no_adaptation_arm_and_did_not_recompute_oracle_numbers": True,
            "wall_clock_held": wall <= WALL_SECONDS_CAP,
            "full_repo_pytest_not_run": True,
        },
        "outside_book": [
            "authorization_audit LOO with min_distinct_tasks=2 requires "
            "3 unguided positives (loo_minimum after dropping one is "
            "n-1).  r1 treated 2 cluster positives as enough for TRY.  "
            "Live code: source_skill.py:249-256.",
            "ECG200 repair_burst_segment is HELDOUT_ONLY: held-in "
            "headroom=0 / held-out +0.04 "
            "(s1_oracle/ECG200__impulse_v2.json programs row).  "
            "Same failure mode as Herring hampel.  The arbitration "
            "preview that the burst-repair cluster might host a legal "
            "course is not supported by the sealed held-in numbers.",
            "GunPoint burst outlier_iqr is HELDOUT_ONLY (held-in=0 / "
            "held-out +0.0133).",
            "ToeSegmentation1 hampel_filter held-in is POSITIVE "
            "(+0.0833, one row at n=12) but is not in the oracle set "
            "(held-out utility 0).  Official table does not count it.  "
            "Even as an extra hampel source it disagrees with GPA/GP "
            "on period_change_score, so it does not match the GPA∩GP "
            "Scope; including it as a third source still yields no "
            "LEARNABLE matching field.",
            "GPA and GunPoint pattern_view are byte-equal.  A Scope "
            "built from only those two is the full pattern; no other "
            "pool unit matches it.",
            "classification online_loop still does not write "
            "task_episode_id; source_skill.build_skill_payload still "
            "does not write evidence_distinct_task_count; Fast-guard "
            "stays off.  Unchanged from r1 outside-book.",
        ],
    }
    _dump(R2_JSON, payload)
    R2_MD.write_text(_r2_markdown(payload), encoding="utf-8")
    print("VERDICT %s  fits=0  wall=%.1fs" % (verdict, wall), flush=True)
    print("wrote %s" % R2_JSON, flush=True)
    return 0


# =========================================================================== #
# S1a-r3 -- one-shot remaining-pool census (0 LLM)
# =========================================================================== #

def _r3_name_family(dataset: str) -> str:
    """Mechanical name-prefix family.  Citations: task book prefix list.

    Phalanx/Phalanges is an infix/stem rule (Distal/Middle/ProximalPhalanx*,
    Phalanges*).  Remaining names are their own family until pattern_view
    byte-equality merges them.
    """
    name = str(dataset)
    if "Phalanx" in name or name.startswith("Phalanges"):
        return "PhalanxFamily"
    for prefix, key in R3_FAMILY_PREFIXES:
        if name.startswith(prefix):
            return key
    return name


def _r3_pattern_fingerprint(pattern: Mapping[str, Any]) -> str:
    return json.dumps(dict(pattern or {}), sort_keys=True, ensure_ascii=True)


def _r3_union_find(keys: Sequence[str]):
    parent: dict[str, str] = {key: key for key in keys}

    def _find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def _union(left: str, right: str) -> None:
        if left not in parent or right not in parent:
            return
        root_l, root_r = _find(left), _find(right)
        if root_l == root_r:
            return
        keep, drop = ((root_l, root_r) if root_l <= root_r
                      else (root_r, root_l))
        parent[drop] = keep

    return parent, _find, _union


def _r3_independence_keys(
    members: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Independent-family keys among THESE members only.

    Same name-prefix family, or byte-equal pattern_view.  Identity /
    construction-failed units must not participate -- otherwise an
    identity BeetleFly row can glue GunPointFamily to PowerCons through
    a shared impulse fingerprint.
    """
    ids = [str(row["unit_id"]) for row in members]
    if not ids:
        return {}
    parent, find, union = _r3_union_find(ids)
    by_name: dict[str, list[str]] = {}
    by_fp: dict[str, list[str]] = {}
    for row in members:
        uid = str(row["unit_id"])
        by_name.setdefault(str(row.get("name_family") or uid), []).append(uid)
        pattern = row.get("pattern_view") or {}
        if pattern:
            by_fp.setdefault(_r3_pattern_fingerprint(pattern), []).append(uid)
    for group in list(by_name.values()) + list(by_fp.values()):
        first = group[0]
        for other in group[1:]:
            union(first, other)
    root_label: dict[str, str] = {}
    for row in members:
        uid = str(row["unit_id"])
        root = find(uid)
        label = str(row.get("name_family") or uid)
        prev = root_label.get(root)
        if prev is None or label < prev:
            root_label[root] = label
    return {uid: root_label[find(uid)] for uid in ids}


def _r3_raw_train_meta(path: Path, name: str) -> dict[str, Any]:
    """Official TRAIN shape without the binary-only loader side effects."""
    with ZipFile(path) as archive:
        members = list(archive.namelist())
        key = "%s_TRAIN.txt" % name
        if key not in members:
            return {"loadable": False, "zip_members": members,
                    "excluded_reason": "no_TRAIN_member"}
        table = np.loadtxt(BytesIO(archive.read(key)), dtype=np.float64)
    if table.ndim != 2:
        return {"loadable": False, "excluded_reason": "not_2d_table"}
    rows = int(table.shape[0])
    length = int(table.shape[1] - 1)
    finite = bool(np.isfinite(table).all())
    classes = int(len({float(value) for value in np.unique(table[:, 0])}))
    return {
        "loadable": True,
        "train_rows": rows,
        "series_length": length,
        "train_points": int(rows * length),
        "finite": finite,
        "class_count": classes,
        "zip_members": members,
    }


def _r3_enumerate_substrates() -> dict[str, Any]:
    """Freeze the remaining-substrate list before any oracle is scored."""
    data_dir = PROJECT_ROOT / DATA_DIR
    paths = sorted(data_dir.glob("*.zip"))
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for path in paths:
        name = path.stem
        row: dict[str, Any] = {
            "dataset": name,
            "zip_bytes": int(path.stat().st_size),
            "archive": "%s/%s.zip" % (DATA_DIR, name),
        }
        reasons: list[str] = []
        try:
            meta = _r3_raw_train_meta(path, name)
        except Exception as exc:  # noqa: BLE001
            meta = {"loadable": False,
                    "load_error": "%s: %s" % (type(exc).__name__, exc),
                    "excluded_reason": "not_loadable"}
        row.update({key: value for key, value in meta.items()
                    if key != "excluded_reason"})
        if not meta.get("loadable"):
            reasons.append(str(meta.get("excluded_reason") or "not_loadable"))
        else:
            if not meta.get("finite", True):
                reasons.append("not_finite")
            if int(meta.get("class_count") or 0) != 2:
                reasons.append("not_binary")
            if int(meta.get("train_points") or 0) > R3_TRAIN_POINT_CAP:
                reasons.append("train_points_over_%d" % R3_TRAIN_POINT_CAP)
            if int(meta.get("series_length") or 0) < 7:
                reasons.append("series_too_short_for_ucr_loader")
        if name in R1_TESTED_DATASETS:
            reasons.append("in_r1_tested_units")
        row["excluded_because"] = reasons
        row["eligible"] = not reasons
        row["name_family"] = _r3_name_family(name)
        (included if row["eligible"] else excluded).append(row)
    units: list[dict[str, Any]] = []
    for substrate in included:
        for injection, suffix in (("impulse_v2", "impulse_v2"),
                                  (BURST_INJECTION, BURST_INJECTION)):
            unit_id = "%s__%s" % (substrate["dataset"], suffix)
            units.append({
                "unit_id": unit_id,
                "dataset": substrate["dataset"],
                "injection": injection,
                "condition": CONDITION,
                "consumer": CONSUMER_ID,
                "name_family": substrate["name_family"],
                "train_rows": substrate.get("train_rows"),
                "series_length": substrate.get("series_length"),
                "train_points": substrate.get("train_points"),
            })
    colliding = [row["unit_id"] for row in units
                 if row["unit_id"] in R1_TESTED_UNIT_IDS]
    if colliding:
        raise cls.Stop("INSTRUMENT_UNREADABLE",
                       "r3 unit ids collide with r1: %s" % colliding)
    return {
        "rule": (
            "every zip in data/ucr_task_context; keep iff binary AND "
            "loadable AND official TRAIN rows*length <= %d AND dataset "
            "not among the 8 substrates that appear in the r1 9 units.  "
            "Each kept substrate x {impulse_v2, burst_cls2}.  Roster is "
            "frozen before the first oracle score."
            % R3_TRAIN_POINT_CAP
        ),
        "zip_count": len(paths),
        "included_substrates": included,
        "excluded_substrates": excluded,
        "n_included": len(included),
        "n_excluded": len(excluded),
        "declared_units": units,
        "n_declared_units": len(units),
        "r1_excluded_datasets": sorted(R1_TESTED_DATASETS),
        "r1_excluded_unit_ids": sorted(R1_TESTED_UNIT_IDS),
        "pool_frozen": True,
    }


def _r3_v2_construction_reason(length: int) -> str | None:
    """Why the frozen v2 template cannot be applied.  No param change."""
    segment = int(round(1.0 / 150.0 * int(length)))
    if segment <= 0:
        return "v2_segment_length_zero_at_L=%d" % length
    try:
        positions = tuple(int(p) for p in
                          cls._legacy_helpers()[1]["positions"](length))
    except Exception as exc:  # noqa: BLE001
        return "v2_positions_rejected:%s" % exc
    for position in positions:
        if int(position) + segment > int(length):
            return "v2_segment_overflow_at_pos_%d_L=%d" % (position, length)
    return None


def _r3_burst_construction_reason(length: int) -> str | None:
    from run_e2_t6_cls2_value_corruption_gate import (
        SEG_FRAC_MAX,
        SEG_FRAC_MIN,
    )
    length_lo = int(np.ceil(SEG_FRAC_MIN * length))
    length_hi = int(np.floor(SEG_FRAC_MAX * length))
    if length_lo < 1 or length_hi < length_lo or length_hi >= length:
        return "burst_cannot_host_15_20pct_at_L=%d" % length
    return None


def _r3_build_cell(spec: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    dataset = str(spec["dataset"])
    injection = str(spec["injection"])
    length = spec.get("series_length")
    if injection == "impulse_v2":
        if length is not None:
            reason = _r3_v2_construction_reason(int(length))
            if reason:
                return None, reason
        try:
            cell = cls._build_cell(
                dataset, CONDITION, data_dir=DATA_DIR,
                injection_template=cls.INJECTION_TEMPLATE_V2)
        except Exception as exc:  # noqa: BLE001
            return None, "impulse_v2_build_failed:%s: %s" % (
                type(exc).__name__, exc)
        return cell, ""
    if length is not None:
        reason = _r3_burst_construction_reason(int(length))
        if reason:
            return None, reason
    try:
        cell = _build_burst_cell(dataset)
    except Exception as exc:  # noqa: BLE001
        return None, "burst_build_failed:%s: %s" % (type(exc).__name__, exc)
    return cell, ""


def _r3_operator_table(unit: Mapping[str, Any]) -> list[dict[str, Any]]:
    oracle_set = [str(item) for item in (unit.get("oracle_set") or [])]
    judged = [item for item in oracle_set if item != "identity"]
    if not judged:
        judged = ["identity"]
    rows: list[dict[str, Any]] = []
    for program in judged:
        raw = _program_row(unit, program)
        facts = _heldin_facts(raw)
        label = _learnability_label(program, facts, oracle_set)
        rows.append({
            "program": program,
            "learnability": label,
            **facts,
            "oracle_json": (
                "artifacts/functional/e2/s1_oracle/%s.json" % unit["unit_id"]
            ),
        })
    return rows


def _r3_summarize_unit(unit: Mapping[str, Any], *,
                       name_family: str,
                       source: str,
                       construction_error: str = "") -> dict[str, Any]:
    operators = _r3_operator_table(unit) if unit.get("programs") else []
    primary = operators[0] if operators else {}
    return {
        "unit_id": unit.get("unit_id"),
        "dataset": unit.get("dataset"),
        "injection": unit.get("injection"),
        "source": source,
        "name_family": name_family,
        "family_key": name_family,
        "family_merged_from_pattern_view": False,
        "construction_error": construction_error or None,
        "scored": bool(unit.get("programs")),
        "n_heldin": unit.get("n_heldin"),
        "n_heldout": unit.get("n_heldout"),
        "oracle_set": list(unit.get("oracle_set") or []),
        "oracle_set_empty": bool(unit.get("oracle_set_empty")),
        "positive_unit": bool(unit.get("positive_unit")),
        "menu_oracle_program": unit.get("menu_oracle_program"),
        "menu_oracle_heldout_utility": unit.get("menu_oracle_heldout_utility"),
        "identity_residual_to_upper_bound": unit.get(
            "identity_residual_to_upper_bound"),
        "menu_best_residual_to_upper_bound": unit.get(
            "menu_best_residual_to_upper_bound"),
        "pattern_view": dict(unit.get("pattern_view") or {}),
        "learnability": primary.get("learnability") or "N/A",
        "oracle_program": primary.get("program"),
        "heldin_headroom": primary.get("heldin_headroom"),
        "heldin_relation": primary.get("relation"),
        "heldout_utility": primary.get("heldout_utility"),
        "operators": operators,
        "approval_threshold": CLASSIFICATION_MATERIAL_THRESHOLD,
        "learnability_citations": {
            "classify_relation": (
                "methods/ttha/experience_memory.py:411-451 "
                "(agg >= +t and min per-view >= -t -> POSITIVE; "
                "t = CLASSIFICATION_MATERIAL_THRESHOLD = 0.005)"
            ),
            "support_draft": "methods/ttha/method.py:742-757",
            "delayed_approve": "methods/ttha/method.py:1466-1492",
            "same_as_r2": True,
        },
    }


def _r3_program_members(rows: Sequence[Mapping[str, Any]],
                        program: str) -> list[dict[str, Any]]:
    members: list[dict[str, Any]] = []
    for row in rows:
        op = next((item for item in (row.get("operators") or [])
                   if item.get("program") == program), None)
        if op is None:
            continue
        members.append({
            "unit_id": row["unit_id"],
            "dataset": row["dataset"],
            "injection": row.get("injection"),
            "source": row.get("source"),
            "name_family": row["name_family"],
            "family_key": row["family_key"],
            "learnability": op.get("learnability") or "N/A",
            "heldin_headroom": op.get("heldin_headroom"),
            "heldin_relation": op.get("relation"),
            "heldout_utility": op.get("heldout_utility"),
            "pattern_view": dict(row.get("pattern_view") or {}),
        })
    return members


def _r3_clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    programs: set[str] = set()
    for row in rows:
        programs.update(p for p in (row.get("oracle_set") or [])
                        if p != "identity")
    clusters: list[dict[str, Any]] = []
    for program in sorted(programs):
        members = _r3_program_members(rows, program)
        learnable = [m for m in members if m["learnability"] == "LEARNABLE"]
        heldout_only = [m for m in members
                        if m["learnability"] == "HELDOUT_ONLY"]
        indep_map = _r3_independence_keys(learnable)
        for member in learnable:
            member["independence_key"] = indep_map.get(
                member["unit_id"], member["name_family"])
        independent = sorted(set(indep_map.values()))
        name_families = sorted({m["name_family"] for m in learnable})
        scope_all = _intersect_maps([m["pattern_view"] for m in learnable])
        clusters.append({
            "program": program,
            "n_oracle_members": len(members),
            "n_learnable": len(learnable),
            "n_heldout_only": len(heldout_only),
            "n_independent_learnable_families": len(independent),
            "independent_families": independent,
            "n_learnable_name_families": len(name_families),
            "learnable_name_families": name_families,
            "learnable_unit_ids": [m["unit_id"] for m in learnable],
            "heldout_only_unit_ids": [m["unit_id"] for m in heldout_only],
            "scope_of_all_learnable": scope_all,
            "scope_of_all_learnable_nonempty": bool(scope_all),
            "members": members,
        })
    clusters.sort(key=lambda row: (-int(row["n_learnable"]),
                                   -int(row["n_independent_learnable_families"]),
                                   row["program"]))
    return clusters


def _r3_search_constructible(
    rows: Sequence[Mapping[str, Any]],
    clusters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """3 independent-family LEARNABLE + 1 extra LEARNABLE matching field."""
    by_id = {row["unit_id"]: row for row in rows}
    identities = [
        row for row in rows
        if row.get("scored")
        and (row.get("learnability") == "N/A"
             or row.get("oracle_program") in (None, "identity")
             or not row.get("positive_unit"))
    ]
    bursts = [row for row in rows
              if row.get("injection") == BURST_INJECTION and row.get("scored")]
    hits: list[dict[str, Any]] = []
    closest: list[dict[str, Any]] = []

    for cluster in clusters:
        program = str(cluster["program"])
        learnable = [m for m in cluster["members"]
                     if m["learnability"] == "LEARNABLE"]
        indep_map = _r3_independence_keys(learnable)
        by_family: dict[str, list[dict[str, Any]]] = {}
        for member in learnable:
            key = indep_map.get(member["unit_id"], member["name_family"])
            member["independence_key"] = key
            by_family.setdefault(key, []).append(member)
        families = sorted(by_family)
        trial = {
            "program": program,
            "n_learnable": len(learnable),
            "n_independent_learnable_families": len(families),
            "independent_families": families,
            "n_identities_available": len(identities),
            "blocked_because": [],
            "closest_note": "",
        }
        if len(learnable) < 4:
            trial["blocked_because"].append(
                "fewer_than_4_learnable_units_need_3_plus_exam_field")
        if len(families) < 3:
            trial["blocked_because"].append(
                "fewer_than_3_independent_learnable_families")
            trial["closest_note"] = (
                "independent LEARNABLE families=%d (need 3); "
                "LEARNABLE units=%d (need >=4).  Short by %d famil%s "
                "and/or %d learnable unit(s)."
                % (len(families), len(learnable),
                   max(0, 3 - len(families)),
                   "y" if (3 - len(families)) == 1 else "ies",
                   max(0, 4 - len(learnable)))
            )

        trio_hits: list[dict[str, Any]] = []
        if len(families) >= 3:
            family_picks = []
            for fam in families:
                family_picks.append(by_family[fam])
            for fam_triple in combinations(families, 3):
                lists = [by_family[fam] for fam in fam_triple]
                for a in lists[0]:
                    for b in lists[1]:
                        for c in lists[2]:
                            trio = (a, b, c)
                            scope = _intersect_maps(
                                [m["pattern_view"] for m in trio])
                            if not scope:
                                continue
                            used = {m["unit_id"] for m in trio}
                            fields = [
                                m for m in learnable
                                if m["unit_id"] not in used
                                and _pattern_matches(m["pattern_view"], scope)
                            ]
                            if not fields:
                                continue
                            trio_hits.append({
                                "sources": [m["unit_id"] for m in trio],
                                "source_families": list(fam_triple),
                                "scope": scope,
                                "field_ids": [m["unit_id"] for m in fields],
                            })
        trial["n_valid_3plus1"] = len(trio_hits)
        if families and len(families) >= 3 and not trio_hits:
            trial["blocked_because"].append(
                "no_3_independent_families_with_nonempty_scope_and_exam_field"
            )
            if not trial["closest_note"]:
                trial["closest_note"] = (
                    "have %d independent LEARNABLE families but no trio "
                    "has a non-empty Scope-v1 pattern intersection plus a "
                    "fourth LEARNABLE matching field."
                    % len(families)
                )
        if trio_hits and len(identities) < 2:
            trial["blocked_because"].append(
                "fewer_than_2_identity_units_for_6_to_8_draft")
        if trio_hits:
            chosen = trio_hits[0]
            drafts = _r3_draft_courses(
                chosen, identities, bursts, by_id, program)
            hits.append({
                "status": "pending_arbitration",
                "program": program,
                "sources": chosen["sources"],
                "source_families": chosen["source_families"],
                "field_ids": chosen["field_ids"],
                "front_scope": chosen["scope"],
                "drafts": drafts,
                "n_alternative_trios": len(trio_hits),
                "note": (
                    "mechanical 3+1 hit under Scope v1 / r3 census.  "
                    "Not an approved curriculum.  待仲裁批准."
                ),
            })
            trial["blocked_because"] = []
            trial["closest_note"] = "constructible"
        closest.append(trial)

    closest.sort(key=lambda row: (
        0 if row.get("n_valid_3plus1") else 1,
        -int(row["n_independent_learnable_families"]),
        -int(row["n_learnable"]),
        row["program"],
    ))
    return {
        "search_rule": (
            "combined r1 9 units + r3 declared units.  A Program cluster "
            "is constructible iff some 3 LEARNABLE oracle-set members from "
            "3 distinct families have a non-empty Scope-v1 pattern "
            "intersection and a fourth LEARNABLE unit matches that "
            "intersection.  Family keys = name prefix plus pattern_view "
            "byte-equality merge.  Learnability is the r2 predicate "
            "(classify_relation == POSITIVE on the sealed held-in pool)."
        ),
        "n_hits": len(hits),
        "hits": hits,
        "closest_per_program": closest,
    }


def _r3_draft_courses(
    hit: Mapping[str, Any],
    identities: Sequence[Mapping[str, Any]],
    bursts: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    program: str,
) -> dict[str, Any]:
    sources = list(hit["sources"])
    field = str(hit["field_ids"][0])
    used = set(sources) | {field}
    id_ids = [row["unit_id"] for row in identities
              if row["unit_id"] not in used][:4]
    burst_id = next((row["unit_id"] for row in bursts
                     if row["unit_id"] not in used), None)
    # 3 authorizing positives first, then identities, exam field, optional burst
    forward = list(sources)
    if len(id_ids) >= 2:
        forward.append(id_ids[0])
        forward.append(field)
        forward.append(id_ids[1])
    else:
        forward.append(field)
        forward.extend(id_ids)
    if burst_id and len(forward) < 8:
        forward.append(burst_id)
    extra = [uid for uid in id_ids[2:] if uid not in forward]
    while extra and len(forward) < 8:
        forward.append(extra.pop(0))
    if len(forward) > 8:
        forward = forward[:8]
    reverse = list(reversed(forward))
    return {
        "forward_order": forward,
        "reverse_order": reverse,
        "n_units": len(forward),
        "authorizing_positives": sources,
        "exam_field": field,
        "identity_unit_ids": [uid for uid in forward if uid in id_ids],
        "burst_unit_id": burst_id if burst_id in forward else None,
        "program": program,
        "status": "pending_arbitration",
        "label": "待仲裁批准",
        "assembly_rule": (
            "3 unguided LEARNABLE positives first; >=1 LEARNABLE exam "
            "field after; >=2 identity fillers; optional burst stretch.  "
            "Reverse is the exact reverse.  6-8 units."
        ),
    }


def _r3_markdown(payload: Mapping[str, Any]) -> str:
    census = payload.get("pool_census") or {}
    units = payload.get("units") or []
    clusters = payload.get("clusters") or []
    search = payload.get("constructible_search") or {}
    ledger = payload.get("ledger") or {}
    lines = [
        "# S1a-r3 remaining-pool census",
        "",
        "protocol: `%s`  parent r1: `%s`  parent r2: `%s`  "
        "evidence grade: **development**"
        % (payload.get("protocol_version"), R1_COMMIT, R2_COMMIT),
        "",
        "0 LLM.  One-shot take-what-comes.  No r4.  Sealed oracles for "
        "new units only; r1/r2 artifacts not overwritten.",
        "",
        "## Isolation",
        "",
        ORACLE_BANNER,
        "",
        "## 1. Pool enumeration (frozen before scoring)",
        "",
        census.get("rule") or "",
        "",
        "zip count=%s; included substrates=%s; excluded=%s; "
        "declared units=%s"
        % (census.get("zip_count"), census.get("n_included"),
           census.get("n_excluded"), census.get("n_declared_units")),
        "",
        "### Included substrates",
        "",
        "| dataset | family | TRAIN rows | L | points |",
        "|---|---|---|---|---|",
    ]
    for row in census.get("included_substrates") or []:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row["dataset"], row.get("name_family"),
               row.get("train_rows"), row.get("series_length"),
               row.get("train_points"))
        )
    lines += [
        "",
        "### Excluded substrates",
        "",
        "| dataset | reason | rows | L | points |",
        "|---|---|---|---|---|",
    ]
    for row in census.get("excluded_substrates") or []:
        lines.append(
            "| %s | %s | %s | %s | %s |"
            % (row["dataset"], ",".join(row.get("excluded_because") or []),
               row.get("train_rows"), row.get("series_length"),
               row.get("train_points"))
        )
    lines += [
        "",
        "## 2. Unit oracle + learnability + family",
        "",
        "Learnability = r2 predicate on the sealed held-in pool "
        "(`classify_relation == POSITIVE`; "
        "experience_memory.py:411-451; method.py:742-757 / 1466-1492).  "
        "Family = name prefix + pattern_view byte-equality merge.",
        "",
        "| unit | src | family | oracle set | learnability | held-in | "
        "held-out | construction |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in units:
        lines.append(
            "| %s | %s | %s | %s | **%s** | %s | %s | %s |"
            % (row.get("unit_id"), row.get("source"),
               row.get("name_family"),
               ",".join(row.get("oracle_set") or []) or "—",
               row.get("learnability"),
               row.get("heldin_headroom"),
               row.get("heldout_utility"),
               row.get("construction_error") or "")
        )
    lines += [
        "",
        "## 3. Program clusters (oracle operator + Scope v1)",
        "",
    ]
    for cluster in clusters:
        lines += [
            "### `%s`" % cluster["program"],
            "",
            "- LEARNABLE **%s** / oracle-members %s; HELDOUT_ONLY %s; "
            "independent families **%s** (%s); name-prefix families %s (%s)"
            % (cluster["n_learnable"], cluster["n_oracle_members"],
               cluster["n_heldout_only"],
               cluster["n_independent_learnable_families"],
               ", ".join(cluster.get("independent_families") or []) or "—",
               cluster.get("n_learnable_name_families"),
               ", ".join(cluster.get("learnable_name_families") or []) or "—"),
            "- all-learnable Scope-v1 intersection nonempty: **%s**"
            % cluster.get("scope_of_all_learnable_nonempty"),
            "- learnable: %s"
            % (", ".join(cluster.get("learnable_unit_ids") or []) or "—"),
            "- held-out only: %s"
            % (", ".join(cluster.get("heldout_only_unit_ids") or []) or "—"),
            "",
        ]
    lines += [
        "## 4. Verdict",
        "",
        "**%s**" % payload.get("verdict"),
        "",
        payload.get("verdict_reason") or "",
        "",
        "### Closest miss per program",
        "",
        "| program | LEARNABLE | families | 3+1 hits | blocked | note |",
        "|---|---|---|---|---|---|",
    ]
    for trial in search.get("closest_per_program") or []:
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (trial.get("program"), trial.get("n_learnable"),
               trial.get("n_independent_learnable_families"),
               trial.get("n_valid_3plus1"),
               ";".join(trial.get("blocked_because") or []) or "—",
               trial.get("closest_note") or "")
        )
    lines.append("")
    hits = search.get("hits") or []
    if hits:
        lines += ["", "### Candidate drafts (待仲裁批准)", ""]
        for hit in hits:
            drafts = hit.get("drafts") or {}
            lines += [
                "- program `%s` sources %s field %s"
                % (hit.get("program"), hit.get("sources"),
                   (hit.get("field_ids") or [None])[0]),
                "- forward: %s" % drafts.get("forward_order"),
                "- reverse: %s" % drafts.get("reverse_order"),
                "",
            ]
    lines += [
        "## Cost",
        "",
        "- Fast LLM: %s" % ledger.get("fast_llm"),
        "- Slow LLM: %s" % ledger.get("slow_llm"),
        "- Consumer fits: %s / %s (this pass %s)"
        % (ledger.get("consumer_fits"), ledger.get("consumer_fit_cap"),
           ledger.get("consumer_fits_this_pass")),
        "- wall clock: %s s / %s s"
        % (ledger.get("wall_seconds"), ledger.get("wall_cap")),
        "- downloads: 0",
        "- units scored / declared: %s / %s"
        % (ledger.get("units_scored"), ledger.get("units_declared")),
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


def _r3_finalize(*, census: Mapping[str, Any],
                 rows: Sequence[dict[str, Any]],
                 started: float, fit_budget: cls.FitBudget,
                 stop_verdict: str | None,
                 stop_detail: str,
                 invariance: Mapping[str, Any],
                 extra_notes: Sequence[str]) -> int:
    clusters = _r3_clusters(rows)
    search = _r3_search_constructible(rows, clusters)
    wall = round(time.time() - started, 2)
    if stop_verdict:
        verdict = stop_verdict
        reason = stop_detail
    elif search["n_hits"]:
        verdict = "LEGAL_CURRICULUM_CONSTRUCTIBLE"
        programs = sorted({hit["program"] for hit in search["hits"]})
        reason = (
            "at least one Program cluster has >=3 independent-family "
            "LEARNABLE positives plus a later LEARNABLE matching field "
            "under Scope v1.  Programs: %s.  Drafts are 待仲裁批准."
            % programs
        )
    else:
        verdict = "POOL_EXHAUSTED_FOR_TRY_CHANNEL"
        closest = (search.get("closest_per_program") or [{}])[0]
        reason = (
            "the pre-declared local pool is exhausted.  No Program "
            "cluster has 3 independent-family LEARNABLE positives plus "
            "a fourth LEARNABLE matching field.  Closest: program=%s "
            "LEARNABLE=%s families=%s.  %s"
            % (closest.get("program"), closest.get("n_learnable"),
               closest.get("n_independent_learnable_families"),
               closest.get("closest_note") or "")
        )
    scored = sum(1 for row in rows if row.get("scored")
                 and row.get("source") != "r1_sealed")
    r1_reused = sum(1 for row in rows if row.get("source") == "r1_sealed")
    failed = [row["unit_id"] for row in rows if row.get("construction_error")]
    prior_fits = 0
    prior_wall = 0.0
    if R3_JSON.is_file():
        try:
            prior = json.loads(R3_JSON.read_text(encoding="utf-8"))
            prior_fits = int((prior.get("ledger") or {}).get("consumer_fits") or 0)
            prior_wall = float((prior.get("ledger") or {}).get("wall_seconds") or 0)
        except Exception:  # noqa: BLE001
            prior_fits = 0
            prior_wall = 0.0
    fits_total = fit_budget.used if fit_budget.used else prior_fits
    wall_total = wall if fit_budget.used else round(prior_wall + wall, 2)
    payload = {
        "protocol_version": R3_PROTOCOL_VERSION,
        "run_id": R3_RUN_ID,
        "curriculum_name": CURRICULUM_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "isolation_banner": ORACLE_BANNER,
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.executable,
        "parent_r1": {"commit": R1_COMMIT},
        "parent_r2": {"commit": R2_COMMIT,
                      "verdict": "HEADROOM_WITHOUT_LEGAL_TRANSFER_PATH"},
        "v2_invariance_at_150": invariance,
        "pool_census": census,
        "units": rows,
        "clusters": [
            {key: value for key, value in cluster.items()
             if key != "members"} | {
                "members": [
                    {k: v for k, v in member.items() if k != "pattern_view"}
                    | {"pattern_view": member.get("pattern_view")}
                    for member in cluster.get("members") or []
                ]
            }
            for cluster in clusters
        ],
        "constructible_search": search,
        "verdict": verdict,
        "verdict_reason": reason,
        "no_r4": True,
        "ledger": {
            "fast_llm": 0,
            "slow_llm": 0,
            "consumer_fits": fits_total,
            "consumer_fits_this_pass": fit_budget.used,
            "consumer_fit_cap": R3_FIT_CAP,
            "wall_seconds": wall_total,
            "wall_seconds_this_pass": wall,
            "wall_cap": WALL_SECONDS_CAP,
            "downloads": 0,
            "units_declared": census.get("n_declared_units"),
            "units_scored_r3": scored,
            "units_scored": scored,
            "r1_units_reused": r1_reused,
            "units_construction_failed": len(failed),
            "from_oracles": bool(fit_budget.used == 0 and prior_fits),
        },
        "obligations": {
            "methods_package_unmodified": True,
            "runtime_contracts_operators_unmodified": True,
            "no_fast_llm": True,
            "no_slow_llm": True,
            "no_a3_a5_adaptation_arm": True,
            "no_injection_scan": True,
            "no_pool_edit_after_declaration": True,
            "no_r4": True,
            "r1_artifacts_not_overwritten": True,
            "r2_artifacts_not_overwritten": True,
            "r1_sealed_oracles_not_rewritten": True,
            "oracle_isolated": True,
            "downloads": 0,
            "ucr_conf_downloaded_not_opened": True,
            "fit_budget_held": (
                fits_total <= R3_FIT_CAP
                and verdict != "COMPUTE_BUDGET_EXCEEDED"
            ),
            "wall_clock_held": wall <= WALL_SECONDS_CAP,
            "full_repo_pytest_not_run": True,
            "learnability_reuses_r2_predicate": True,
        },
        "outside_book": list(extra_notes) + [
            "SonyAIBO L=65/70 makes v2 segment=round(L/150)=0; those "
            "impulse units are construction failures, not silent drops.",
            "Independence keys are union-find over LEARNABLE members "
            "only (name prefix OR byte-equal pattern_view).  A first "
            "draft that unioned every name-family sharing any unit's "
            "pattern_view collapsed GunPoint/PowerCons/ECG into "
            "BeetleFly via identity rows; that merge was rejected "
            "before the verdict was filed.  Sealed oracle numbers "
            "were not rescored.",
            "ECG200 (r1) and ECGFiveDays share name prefix ECG → "
            "ECGFamily; TwoLeadECG does not.",
            "Phalanx OutlineCorrect trio is one family; Freezer* is one "
            "family; GunPoint MaleVersusFemale/OldVersusYoung join the "
            "existing GunPointFamily and cannot add independence.",
            "classification online_loop still does not write "
            "task_episode_id; Fast-guard stays off.  Unchanged from r1/r2.",
        ],
    }
    _dump(R3_JSON, payload)
    R3_MD.write_text(_r3_markdown(payload), encoding="utf-8")
    print("VERDICT %s  scored=%d/%d  fits=%d  wall=%.1fs"
          % (verdict, scored, census.get("n_declared_units") or 0,
             fit_budget.used, wall), flush=True)
    print("wrote %s" % R3_JSON, flush=True)
    return 0 if verdict != "COMPUTE_BUDGET_EXCEEDED" else 1


def run_census_r3(*, from_oracles: bool = False) -> int:
    started = time.time()
    print("S1a-r3 pool census  protocol=%s" % R3_PROTOCOL_VERSION, flush=True)
    census = _r3_enumerate_substrates()
    declared = list(census["declared_units"])
    print("POOL frozen n_substrates=%d n_units=%d"
          % (census["n_included"], census["n_declared_units"]), flush=True)
    print("INCLUDED %s"
          % [row["dataset"] for row in census["included_substrates"]],
          flush=True)
    print("UNITS %s" % [row["unit_id"] for row in declared], flush=True)

    invariance = cls._v2_invariance_at_150()
    if not invariance["passed"]:
        raise cls.Stop("INSTRUMENT_UNREADABLE",
                       "v2 invariance at L=150 failed: %s"
                       % invariance["checks"])

    r1_units = _load_sealed_units()
    if len(r1_units) != 9:
        raise cls.Stop("INSTRUMENT_UNREADABLE",
                       "r1 sealed pool must remain 9 units")
    rows: list[dict[str, Any]] = []
    for unit in r1_units:
        rows.append(_r3_summarize_unit(
            unit, name_family=_r3_name_family(str(unit.get("dataset") or "")),
            source="r1_sealed"))

    fit_budget = cls.FitBudget(R3_FIT_CAP)
    extra_notes: list[str] = []
    for spec in declared:
        elapsed = time.time() - started
        if elapsed > WALL_SECONDS_CAP:
            extra_notes.append(
                "wall cap hit before %s; remaining units not scored"
                % spec["unit_id"])
            return _r3_finalize(
                census=census, rows=rows, started=started,
                fit_budget=fit_budget,
                stop_verdict="COMPUTE_BUDGET_EXCEEDED",
                stop_detail=(
                    "wall clock cap %ss hit before %s; reporting completed "
                    "units only.  Pool roster was not edited."
                    % (WALL_SECONDS_CAP, spec["unit_id"])
                ),
                invariance=invariance, extra_notes=extra_notes)
        path = ORACLE_DIR / ("%s.json" % spec["unit_id"])
        if from_oracles or path.is_file():
            if path.is_file():
                sealed = json.loads(path.read_text(encoding="utf-8"))
                rows.append(_r3_summarize_unit(
                    sealed, name_family=spec["name_family"],
                    source="r3_sealed_reused"))
                print("REUSE %s" % spec["unit_id"], flush=True)
                continue
        print("ORACLE %s ..." % spec["unit_id"], flush=True)
        cell, error = _r3_build_cell(spec)
        if cell is None:
            rows.append({
                "unit_id": spec["unit_id"],
                "dataset": spec["dataset"],
                "injection": spec["injection"],
                "source": "r3_construction_failed",
                "name_family": spec["name_family"],
                "family_key": spec["name_family"],
                "construction_error": error,
                "scored": False,
                "oracle_set": [],
                "positive_unit": False,
                "pattern_view": {},
                "learnability": "N/A",
                "oracle_program": None,
                "heldin_headroom": None,
                "heldin_relation": None,
                "heldout_utility": None,
                "operators": [],
            })
            print("  CONSTRUCTION_FAIL %s" % error, flush=True)
            continue
        try:
            clean_fit, _labels = _load_clean_fit(spec["dataset"])
            if clean_fit.shape != np.asarray(cell["fit_values"]).shape:
                raise cls.Stop(
                    "INSTRUMENT_UNREADABLE",
                    "clean fit shape != injected fit for %s" % spec["unit_id"])
            unit = _oracle_one_unit(
                spec=spec, cell=cell, clean_fit=clean_fit,
                fit_budget=fit_budget)
            unit["v2_invariance_at_150"] = invariance
            _write_sealed_oracle(unit, extra={"census_round": "s1a-r3"})
            rows.append(_r3_summarize_unit(
                unit, name_family=spec["name_family"], source="r3_scored"))
            print("  legal=%s oracle=%s learnability pending-aggregate"
                  % (unit["legal_set"], unit["oracle_set"]), flush=True)
        except cls.Stop as exc:
            if "BUDGET" in str(exc.verdict):
                extra_notes.append(
                    "fit/wall budget stop at %s: %s" % (spec["unit_id"], exc))
                return _r3_finalize(
                    census=census, rows=rows, started=started,
                    fit_budget=fit_budget,
                    stop_verdict="COMPUTE_BUDGET_EXCEEDED",
                    stop_detail=(
                        "budget stop at %s (%s); reporting completed "
                        "units only.  Pool roster was not edited."
                        % (spec["unit_id"], exc.verdict)
                    ),
                    invariance=invariance, extra_notes=extra_notes)
            rows.append({
                "unit_id": spec["unit_id"],
                "dataset": spec["dataset"],
                "injection": spec["injection"],
                "source": "r3_stop",
                "name_family": spec["name_family"],
                "family_key": spec["name_family"],
                "construction_error": "%s: %s" % (exc.verdict, exc),
                "scored": False,
                "oracle_set": [],
                "positive_unit": False,
                "pattern_view": {},
                "learnability": "N/A",
                "operators": [],
            })
            print("  STOP %s: %s" % (exc.verdict, exc), flush=True)
    return _r3_finalize(
        census=census, rows=rows, started=started, fit_budget=fit_budget,
        stop_verdict=None, stop_detail="",
        invariance=invariance, extra_notes=extra_notes)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--from-oracles", action="store_true",
                        help="reuse sealed s1_oracle files; 0 new fits")
    parser.add_argument("--legal-r2", action="store_true",
                        help="r2 legal-treatment reaggregation; 0 fits; "
                             "does not overwrite r1")
    parser.add_argument("--census-r3", action="store_true",
                        help="r3 one-shot remaining-pool census")
    parser.add_argument("--slow-rehearse", action="store_true",
                        help="unused unless card shape cannot be deduced")
    args = parser.parse_args()
    if args.legal_r2:
        try:
            return run_legal_r2()
        except cls.Stop as exc:
            print("STOP %s: %s" % (exc.verdict, exc), flush=True)
            return 1
    if args.census_r3:
        try:
            return run_census_r3(from_oracles=bool(args.from_oracles))
        except cls.Stop as exc:
            print("STOP %s: %s" % (exc.verdict, exc), flush=True)
            return 1
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
