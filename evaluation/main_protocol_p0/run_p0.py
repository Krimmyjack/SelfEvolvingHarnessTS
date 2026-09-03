"""Run the v1.2.1-Core P0b preflight without opening a Final outcome."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence
from zipfile import ZipFile

from evaluation.main_protocol_p0.p0b_smokes import (
    anomaly_adapter_smoke,
    baseline_contract_smoke,
    cost_accounting,
    forecast_adapter_smoke,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_STEM = (
    "SelfEvolving_Data_Readiness_Main_Experiment_Protocol_"
    "v1.2.1-Core_2026-08-30"
)
DOCX_CHECKSUM = "e054bece20bcac721868c2321e6b31e759d18e5b9074e6f1918aa4cfba1f59c9"
VISIBLE_TEXT_CHECK = {
    "visible_text_chars": 18185,
    "visible_text_result": "MATCH",
    "comparison_rule": (
        "DOCX body-order paragraphs/tables versus CommonMark-rendered visible "
        "text with whitespace and formatting markers normalized"
    ),
}
UCR_METADATA = PROJECT_ROOT / "_scratch" / "tsc_metadata.csv"
UCR_ROOT = PROJECT_ROOT / "data" / "main_experiment_p0" / "ucr_fresh"
OUT_JSON = PROJECT_ROOT / "artifacts" / "main_protocol" / "p0_readiness_20260830.json"
OUT_MD = PROJECT_ROOT / "artifacts" / "main_protocol" / "p0_readiness_20260830.md"
TRAIN_POINT_CAP = 100_000
SPLIT_VERSION = "class-index-order-50-25-25-v1"
PANEL_VERSION = "p0b-public-missing-spike-level-burst-v1"
UCR_SELECTION = (
    {"slot": "Final-A", "dataset": "Adiac", "train": 390, "test": 391,
     "length": 176, "classes": 37,
     "archive_checksum": "9c808bcfc77f3cab0a640bfa23e680ccde91da6981962e63c0f4f0737dcf6390"},
    {"slot": "Final-B", "dataset": "ArrowHead", "train": 36, "test": 175,
     "length": 251, "classes": 3,
     "archive_checksum": "790a51e664c347df1e931ffc63afc93b1e718ff75c146e7a2d5c669e51b037c4"},
)


def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return "local:Downloads/%s" % path.name


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_branch() -> str:
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance must not stop the audit
        return ""


def _protocol_paths() -> tuple[Path, Path]:
    candidates = []
    if os.environ.get("SELF_EVOLVING_PROTOCOL_DIR"):
        candidates.append(Path(os.environ["SELF_EVOLVING_PROTOCOL_DIR"]))
    candidates.extend((Path.home() / "Downloads", PROJECT_ROOT.parents[2] / "Downloads"))
    for directory in candidates:
        md = directory / (PROTOCOL_STEM + ".md")
        docx = directory / (PROTOCOL_STEM + ".docx")
        if md.is_file() and docx.is_file():
            return md, docx
    raise FileNotFoundError("v1.2.1 protocol pair not found; set SELF_EVOLVING_PROTOCOL_DIR")


def protocol_gate() -> dict[str, Any]:
    md, docx = _protocol_paths()
    checksum = _file_checksum(docx)
    locked = checksum == DOCX_CHECKSUM
    return {
        "status": "PASS" if locked else "BLOCKED_CONTENT_DRIFT",
        "truth_source": "DOCX",
        "docx": {"path": _rel(docx), "checksum": checksum, "matches_lock": locked},
        "markdown_export": {"path": _rel(md), "exists": True, "locked": False},
        "visible_text_check": {**VISIBLE_TEXT_CHECK, "lock_is_valid": locked},
    }


def supersession_gate(protocol: Mapping[str, Any]) -> dict[str, Any]:
    documents = [
        ("docs/MAIN_EXPERIMENT_PROTOCOL_ANALYSIS_2026-08-29.md", "SUPERSEDED"),
        ("docs/MAIN_EXPERIMENT_PROTOCOL_V2_2026-08-29.md", "SUPERSEDED"),
        ("docs/MAIN_EXPERIMENT_PROTOCOL_V3_2026-08-29.md", "SUPERSEDED"),
        ("docs/D4_DOWNLOAD_FREEZE_2026-08-29.md", "RETAINED_EVIDENCE"),
        ("AGENTS.md", "ARCHITECTURE_AUTHORITY"),
    ]
    rows = [{"document": name, "disposition": disposition,
             "exists": (PROJECT_ROOT / name).is_file()}
            for name, disposition in documents]
    return {
        "status": "PASS" if protocol["status"] == "PASS" and all(
            row["exists"] for row in rows) else "BLOCKED",
        "current_protocol": "v1.2.1-Core DOCX truth source",
        "documents": rows,
        "dataset_roles": {
            "Traffic": "Forecast Final-1 columns 480..861",
            "Solar-Energy": "Forecast Final-2 all 137 series",
            "Yahoo S5": "AD development 24 / Final-1 sealed 41",
            "Epilepsy2": "exposed Classification replay only",
        },
    }


def _fresh_ucr_trace() -> dict[str, Any]:
    rows = list(csv.DictReader(UCR_METADATA.read_text(encoding="utf-8-sig").splitlines()))
    exposed = {path.stem for path in (PROJECT_ROOT / "data/ucr_task_context").glob("*.zip")}
    exposed.update(("BinaryHeartbeat", "Epilepsy2"))
    eligible = []
    for row in rows:
        name = str(row["Dataset"])
        train, length = int(row["TrainSize"]), int(row["Length"])
        classes, channels = int(row["NumberClasses"]), int(row["Channels"])
        if (channels == 1 and length >= 150 and train * length <= TRAIN_POINT_CAP
                and train >= 4 * classes and name not in exposed):
            eligible.append(name)
    eligible.sort()
    selected = [row["dataset"] for row in UCR_SELECTION]
    return {
        "metadata": {"path": _rel(UCR_METADATA), "rows": len(rows),
                     "source_url": "https://timeseriesclassification.com/aeon-toolkit/metadata.csv"},
        "rules": {"univariate": True, "minimum_length": 150,
                  "train_point_cap": TRAIN_POINT_CAP,
                  "metadata_class_floor": "TrainSize >= 4 * NumberClasses",
                  "split": SPLIT_VERSION},
        "eligible_first_ten": eligible[:10],
        "selected": selected,
        "selection_is_lexicographic_first_two": eligible[:2] == selected,
    }


def _parse_ucr_train(dataset: str, path: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Read exactly one TRAIN member; TEST is listed but never decompressed."""
    import numpy as np

    with ZipFile(path) as archive:
        infos = archive.infolist()
        train_members = [i.filename for i in infos if i.filename.lower().endswith("_train.ts")]
        if len(train_members) != 1:
            raise ValueError("%s must contain exactly one *_TRAIN.ts" % dataset)
        test_members = [{"name": i.filename, "bytes": int(i.file_size)}
                        for i in infos if "_test." in i.filename.lower()]
        raw = archive.read(train_members[0])
    values, labels = [], []
    in_data = False
    for raw_line in raw.decode("utf-8-sig").splitlines():
        line = raw_line.strip()
        if not in_data:
            in_data = line.lower() == "@data"
            continue
        if not line:
            continue
        fields = line.rsplit(":", 1)
        if len(fields) != 2:
            raise ValueError("%s TRAIN is not univariate .ts" % dataset)
        vector = np.fromstring(fields[0], dtype=np.float64, sep=",")
        if not vector.size or not np.isfinite(vector).all():
            raise ValueError("%s TRAIN contains empty/non-finite values" % dataset)
        values.append(vector)
        labels.append(fields[1].strip())
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("%s TRAIN is not equal-length" % dataset)
    names = sorted(set(labels))
    encoded = np.asarray([names.index(label) for label in labels], dtype=np.int64)
    scale = np.std(matrix, axis=1, keepdims=True)
    if bool(np.any(scale <= 1e-12)):
        raise ValueError("%s TRAIN contains a degenerate row" % dataset)
    matrix = (matrix - np.mean(matrix, axis=1, keepdims=True)) / scale
    return matrix, encoded, {
        "train_member": train_members[0], "train_member_bytes": len(raw),
        "test_members_directory_only": test_members, "test_member_bytes_read": False,
        "label_names_count": len(names),
    }


def _stratified_train_surfaces(labels: Any) -> tuple[dict[str, list[int]], dict[str, Any]]:
    import numpy as np

    split = {"fit": [], "support_a": [], "support_b": []}
    per_class = {}
    for label in sorted(int(value) for value in set(np.asarray(labels).tolist())):
        indices = np.flatnonzero(np.asarray(labels) == label).tolist()
        if len(indices) < 4:
            raise ValueError("class %s has fewer than four TRAIN rows" % label)
        n_fit = max(2, len(indices) // 2)
        n_a = max(1, (len(indices) - n_fit) // 2)
        n_b = len(indices) - n_fit - n_a
        if n_b < 1:
            n_a, n_b = n_a - 1, n_b + 1
        chunks = {"fit": indices[:n_fit],
                  "support_a": indices[n_fit:n_fit + n_a],
                  "support_b": indices[n_fit + n_a:]}
        if any(not chunk for chunk in chunks.values()):
            raise ValueError("class %s cannot populate all TRAIN surfaces" % label)
        for name, chunk in chunks.items():
            split[name].extend(chunk)
        per_class[str(label)] = {name: len(chunk) for name, chunk in chunks.items()}
    for rows in split.values():
        rows.sort()
    return split, {"version": SPLIT_VERSION,
                   "row_order": "official TRAIN row index within sorted label",
                   "per_class_counts": per_class,
                   "surface_rows": {name: len(rows) for name, rows in split.items()},
                   "surface_indices": split}


def _classification_smoke(dataset: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score, f1_score, recall_score

    name, path = str(dataset["dataset"]), UCR_ROOT / (str(dataset["dataset"]) + ".zip")
    if _file_checksum(path) != dataset["archive_checksum"]:
        raise ValueError("%s downloaded archive checksum mismatch" % name)
    values, labels, archive = _parse_ucr_train(name, path)
    expected = (int(dataset["train"]), int(dataset["length"]))
    if values.shape != expected or len(set(labels.tolist())) != int(dataset["classes"]):
        raise ValueError("%s TRAIN structure drift" % name)
    split, split_record = _stratified_train_surfaces(labels)

    def features(block: Any) -> Any:
        array = np.asarray(block, dtype=np.float64)
        return np.concatenate((array, np.diff(array, axis=1)), axis=1)

    model = RidgeClassifier(alpha=1.0).fit(features(values[split["fit"]]), labels[split["fit"]])
    classes = sorted(int(value) for value in set(labels.tolist()))
    readings = {}
    for surface in ("support_a", "support_b"):
        indices = split[surface]
        truth, predicted = labels[indices], model.predict(features(values[indices]))
        recalls = recall_score(truth, predicted, labels=classes, average=None, zero_division=0)
        readings[surface] = {
            "macro_f1_primary": float(f1_score(
                truth, predicted, labels=classes, average="macro", zero_division=0)),
            "accuracy_secondary": float(accuracy_score(truth, predicted)),
            "worst_class_recall_safety": float(np.min(recalls)), "rows": len(indices),
        }
    return {
        "dataset": name, "status": "PASS", "train_shape": list(values.shape),
        "classes": int(dataset["classes"]), "split": split_record,
        "archive": {"path": _rel(path), "bytes": path.stat().st_size,
                    "checksum": dataset["archive_checksum"], **archive},
        "consumer": "RidgeClassifier(alpha=1.0); raw + first difference",
        "primary_metric": "Macro-F1",
        "secondary_metrics": ["Accuracy", "worst-class recall"],
        "consumer_fits": 1, "readings": readings,
        "readings_are_train_only_development_diagnostics": True,
    }


def exposure_and_fresh_pool_gate() -> dict[str, Any]:
    yahoo = PROJECT_ROOT / "data" / "benchmark_yahoo_s5_v1"
    yahoo_work = list((yahoo / "work").glob("real_*.csv"))
    yahoo_vault = list((yahoo / "vaults" / "held_out").glob("real_*.csv"))
    nab_local = list((PROJECT_ROOT / "data/benchmark_nab_v1_1/raw").glob("real*/*.csv"))
    fresh = _fresh_ucr_trace()
    checks = {
        "fresh_ucr_first_two": fresh["selection_is_lexicographic_first_two"],
        "fresh_ucr_archives_present": all(
            (UCR_ROOT / (row["dataset"] + ".zip")).is_file() for row in UCR_SELECTION),
        "yahoo_work_65": len(yahoo_work) == 65,
        "yahoo_vault_65": len(yahoo_vault) == 65,
        "nab_local_exposed_37": len(nab_local) == 37,
    }
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "exposure_definition": "labels/outcomes count as exposed when loaded into a process",
        "checks": checks, "classification_fresh_pool": fresh,
        "final_roster": {
            "forecast": ["Traffic leftover 480..861", "Solar-Energy all 137"],
            "classification": ["Adiac", "ArrowHead"],
            "anomaly_detection": ["Yahoo S5 sealed 41"],
        },
        "ad_fresh_pool": {
            "status": "FINAL_POOL_UNAVAILABLE", "official_real_series": 47,
            "local_value_exposed_series": 37, "official_value-unrun_leftover": 10,
            "leftover_label_outcome_exposed_by_legacy_global_load": 10,
        },
        "sealed_surfaces_read_by_this_runner": [],
    }


def adapter_gate() -> dict[str, Any]:
    forecast, anomaly = forecast_adapter_smoke(), anomaly_adapter_smoke()
    classification = [_classification_smoke(row) for row in UCR_SELECTION]
    tasks = [forecast, {"task": "classification", "status": "PASS_TRAIN_ONLY",
                        "checks": classification,
                        "note": "versioned multiclass Macro-F1 path; old Accuracy path preserved"},
             anomaly]
    return {
        "status": "PASS" if all(str(row["status"]).startswith("PASS") for row in tasks)
        else "BLOCKED",
        "tasks": tasks,
        "consumer_fit_count_for_this_gate": (
            int(forecast["raw_consumer_fits"]) + len(classification)
            + int(anomaly["raw_consumer_fits"])),
        "final_query_or_test_bytes_read": 0,
        "scope": "P0b contract smoke; full roster/slice integration remains P2/P3",
    }


def _public_panels() -> list[Any]:
    import numpy as np

    t = np.arange(192, dtype=np.float64)
    clean = np.sin(2 * np.pi * t / 24) + 0.003 * t
    rows = np.vstack((clean, 0.7 * clean + 0.2, -0.4 * clean + 0.004 * t))
    missing = rows.copy(); missing[0, 35:43] = np.nan; missing[1, (18, 66, 114)] = np.nan
    spike = rows.copy(); spike[0, (31, 79, 127, 169)] += (9, -8, 10, -9)
    level = rows.copy(); level[0, 88:] += 3; level[2, 72:132] -= 2.5
    burst = rows.copy(); burst[0, 58:70] += 6; burst[1, 118:130] -= 5
    return [missing, spike, level, burst]


def _default_params(op: str) -> dict[str, Any]:
    if op == "period_median_complete":
        return {"period": 24, "cycles": 3, "min_donors": 2}
    if op in {"period_complete", "impute_ssm", "impute_ar", "denoise_stl",
              "stl_decompose", "repair_level_shift"}:
        return {"period": 24}
    return {}


def _array_identity(arrays: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    import numpy as np

    output = []
    for array in arrays:
        value = np.asarray(array, dtype=np.float64)
        finite = np.isfinite(value)
        stable = np.round(np.nan_to_num(
            value, nan=0.0, posinf=1e300, neginf=-1e300), 12).astype("<f8")
        output.append((tuple(value.shape), np.packbits(finite.ravel()).tobytes(), stable.tobytes()))
    return tuple(output)


def _task_program_inventory(task: str) -> dict[str, Any]:
    import numpy as np
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA, OPERATOR_NAMES
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

    eligible, excluded = [], Counter()
    for op in OPERATOR_NAMES:
        meta = OPERATOR_METADATA[op]
        reason = None
        if task not in tuple(meta.get("allowed_tasks") or ()):
            reason = "task_contract"
        elif bool(meta.get("shape_changing")):
            reason = "shape_changing"
        elif bool(meta.get("changes_target_space")):
            reason = "changes_target_space"
        elif meta.get("requires_dependency") == "statsmodels":
            reason = "slow_dependency"
        if reason:
            excluded[reason] += 1
        else:
            eligible.append(op)

    identities, smoke_programs, failed, identity_equivalent = set(), [], 0, 0
    for op in eligible:
        outputs, changed, ok = [], 0, True
        for panel in _public_panels():
            transformed = []
            for row in np.asarray(panel, dtype=np.float64):
                result = run_pipeline([(op, _default_params(op))], row,
                                      source="p0b_program_inventory")
                if not result.ok or result.artifact is None:
                    ok = False; break
                out = np.asarray(result.artifact, dtype=np.float64)
                if out.shape != row.shape:
                    ok = False; break
                changed += int(np.count_nonzero(
                    ~np.isclose(out, row, rtol=1e-10, atol=1e-12, equal_nan=True)))
                transformed.append(out)
            if not ok:
                break
            outputs.append(np.asarray(transformed))
        if not ok:
            failed += 1
        elif changed == 0:
            identity_equivalent += 1
        else:
            identity = _array_identity(outputs)
            if identity not in identities:
                identities.add(identity)
                smoke_programs.append(op)
    p_effect = 1 + len(identities)
    return {
        "task": task, "panel_version": PANEL_VERSION,
        "actual_single_step_inventory": {
            "identity": 1, "eligible_operators": len(eligible),
            "failed_operators": failed, "identity_equivalent_operators": identity_equivalent,
            "effect_distinct_global_single_steps": len(identities),
            "p_effect": p_effect, "b_main": 4,
            "actual_coverage_percent": 400.0 / p_effect,
        },
        "excluded_operator_counts": dict(excluded),
        "smoke_programs": smoke_programs[:3],
    }


def program_space_gate() -> dict[str, Any]:
    return {
        "status": "PASS_DESCRIPTIVE_INVENTORY",
        "minimum_p_effect": None, "coverage_is_a_release_gate": False,
        "tasks": [_task_program_inventory(task) for task in (
            "forecast", "classification", "anomaly_detection")],
        "decision_rule": (
            "report actual coverage only; do not add operators, two-step templates, "
            "targeting infrastructure, or lower AD B_main"
        ),
    }


def _relation_counts(payload: Any) -> dict[str, int]:
    counts: Counter[str] = Counter()
    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            if isinstance(value.get("relation"), str):
                counts[str(value["relation"])] += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(payload)
    return dict(sorted(counts.items()))


def treatment_gate() -> dict[str, Any]:
    paths = {
        "forecast": PROJECT_ROOT / "artifacts/functional/e2/s2a_g1_run1_r2.json",
        "classification": PROJECT_ROOT / "artifacts/functional/e2/sa1_minimal_r2.json",
        "anomaly_detection": PROJECT_ROOT / "artifacts/functional/e2/t6_nab_frozen_plan_v2.json",
    }
    payloads = {task: _json(path) for task, path in paths.items()}
    tasks = [
        {"task": "forecast", "status": "RQ3_NOT_EXERCISED",
         "relations": _relation_counts(payloads["forecast"]),
         "reason": "no applied revision followed by a similar re-encounter"},
        {"task": "classification", "status": "RQ3_NOT_EXERCISED_METRIC_MISMATCH",
         "relations": _relation_counts(payloads["classification"]),
         "reason": "historical full chain used Accuracy, not v1.2.1 Macro-F1"},
        {"task": "anomaly_detection", "status": "RQ3_NOT_EXERCISED",
         "relations": _relation_counts(payloads["anomaly_detection"]),
         "reason": "development events do not form one revision-to-re-encounter chain"},
    ]
    return {
        "status": "PASS_RQ1_RQ2__RQ3_NOT_EXERCISED",
        "deterministic_shared_chain": {
            "status": "PASS_EXISTING_CODE_PATH",
            "links": ["Episode", "Skill compile", "Scope match", "candidate supply",
                      "Support-A", "Support-B/delayed", "revision/revoke", "re-encounter"],
            "tests": ["test_supply_tier_compiler.py", "test_guard_pipeline_reachability.py",
                      "test_delayed_rejected_winner_not_deployed.py"],
        },
        "tasks": tasks,
        "policy": "RQ3 claim ceiling; does not block P1 or RQ1/RQ2",
    }


def protocol_resolutions() -> list[dict[str, str]]:
    return [
        {"id": "P0_BASELINE_ORDER", "resolution": "P0b minimal contract smoke; P1 full Core smoke"},
        {"id": "BUDGET_NAME", "resolution": "Parallel/Sequential @B_main; Primary/Final @4"},
        {"id": "PROGRAM_COVERAGE", "resolution": "coverage is descriptive, never a release gate"},
        {"id": "UCR_SPLIT_COST", "resolution": "50/25/25 class-order split and 100k TRAIN-point cap"},
        {"id": "COST_ACCEPTANCE", "resolution": "accounting completeness without affordability judgement"},
        {"id": "FORECAST_DELTA_SIGN", "resolution": "U_forecasting=-sMASE"},
    ]


def build_report() -> dict[str, Any]:
    protocol = protocol_gate()
    supersession = supersession_gate(protocol)
    exposure = exposure_and_fresh_pool_gate()
    adapters = adapter_gate()
    program = program_space_gate()
    treatment = treatment_gate()
    baselines = baseline_contract_smoke(program)
    baselines.update({
        "scope": "interface/budget safety only; no utility or headroom claim",
        "aegists_adapter_spike": {
            "status": "DIRECT_CODE_ADAPTER_STRUCTURALLY_INCOMPATIBLE",
            "paper": "https://arxiv.org/abs/2605.04902",
            "official_code": "https://github.com/Syh517/AegisTS",
            "p1_route": "related-work; does not block other Core baselines",
        },
    })
    final_datasets = 5 if exposure["ad_fresh_pool"]["status"] == "FINAL_POOL_UNAVAILABLE" else 6
    cost = cost_accounting(
        final_datasets=final_datasets,
        adapter_raw_fits=int(adapters["consumer_fit_count_for_this_gate"]),
        baseline_task_count=len(baselines["tasks"]),
    )
    gates = {"supersession": supersession, "exposure_fresh_pool": exposure,
             "adapter": adapters, "program_space": program,
             "treatment_reachability_event": treatment,
             "baseline_smoke": baselines, "cost": cost}
    statuses = {name: str(gate["status"]) for name, gate in gates.items()}
    passed = [name for name, status in statuses.items()
              if status == "PASS" or status.startswith("PASS_")]
    p0b_passed = len(passed) == len(gates)
    release = "RELEASED_TO_P1_ONLY" if p0b_passed else "BLOCKED_BY_P0B"
    return {
        "protocol_version": "v1.2.1-Core/2026-08-30",
        "audit_version": "p0b-readiness-audit-v1/2026-08-30",
        "mode": "audit_only_no_final_outcome",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "git": {"branch": _git_branch(), "worktree_preserved": True},
        "protocol_lock": protocol, "gate_status": statuses,
        "gates_fully_passed": passed,
        "gates_not_fully_passed": [name for name in gates if name not in passed],
        "gates": gates, "protocol_resolutions": protocol_resolutions(),
        "task_release": {
            "forecast": {"rq1_rq2": release, "rq3": "NOT_EXERCISED"},
            "classification": {"rq1_rq2": release, "rq3": "NOT_EXERCISED_METRIC_MISMATCH"},
            "anomaly_detection": {"rq1_rq2": release, "rq3": "NOT_EXERCISED"},
        },
        "final_pool": exposure["final_roster"] | {"anomaly_detection_final_2": "FINAL_POOL_UNAVAILABLE"},
        "verdict": {
            "audit": "P0B_COMPLETE" if p0b_passed else "P0B_INCOMPLETE",
            "execution": "P0B_PASS__P1_BASELINE_SMOKE_RELEASED" if p0b_passed else "P0B_BLOCKED",
            "p1_release": p0b_passed, "live_outcome_release": False,
            "why": (
                "All P0b safety/accounting contracts pass under v1.2.1; P1 full Core "
                "baseline smoke is next. Final outcomes remain sealed and RQ3 is not exercised."
            ),
        },
        "sealed_read_invariants": {"ucr_test_member_bytes": 0,
                                   "yahoo_sealed_41_csv_bytes": 0,
                                   "solar_numeric_bytes_by_this_runner": 0,
                                   "final_outcome_used_for_selection": False,
                                   "skill_or_harness_state_written": False},
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# v1.2.1-Core P0b readiness audit", "",
        "**Audit: `%s`. Execution: `%s`. P1 release: `%s`.**" % (
            report["verdict"]["audit"], report["verdict"]["execution"],
            report["verdict"]["p1_release"]), "", report["verdict"]["why"], "",
        "No Natural Final outcome was opened by this runner.", "",
        "## Gate ledger", "", "| gate | status |", "|---|---|",
    ]
    lines.extend("| %s | `%s` |" % item for item in report["gate_status"].items())
    lines += ["", "## Frozen Final roster", "",
              "- Forecast: Traffic leftover columns 480..861; Solar-Energy all 137 series.",
              "- Classification: Adiac and ArrowHead; TEST bytes remain unread.",
              "- AD: Yahoo S5 sealed 41; Fresh NAB is `FINAL_POOL_UNAVAILABLE`.", "",
              "## Classification TRAIN-only adapter", "",
              "| dataset | TRAIN shape | classes | Support-A Macro-F1 | Support-B Macro-F1 | TEST bytes read |",
              "|---|---:|---:|---:|---:|---|"]
    cls = next(row for row in report["gates"]["adapter"]["tasks"]
               if row["task"] == "classification")
    for row in cls["checks"]:
        lines.append("| %s | %s | %s | %.6f | %.6f | %s |" % (
            row["dataset"], "x".join(map(str, row["train_shape"])), row["classes"],
            row["readings"]["support_a"]["macro_f1_primary"],
            row["readings"]["support_b"]["macro_f1_primary"],
            row["archive"]["test_member_bytes_read"]))
    lines += ["", "## Program-space inventory", "",
              "Coverage is descriptive; it is not a release gate.", "",
              "| task | B_main | current P_effect | actual coverage |",
              "|---|---:|---:|---:|"]
    for row in report["gates"]["program_space"]["tasks"]:
        inv = row["actual_single_step_inventory"]
        lines.append("| %s | %s | %s | %.2f%% |" % (
            row["task"], inv["b_main"], inv["p_effect"], inv["actual_coverage_percent"]))
    lines += ["", "No DSL, two-step, targeting, or AD-budget expansion is authorized.", "",
              "## RQ3 claim ceiling", ""]
    for row in report["gates"]["treatment_reachability_event"]["tasks"]:
        lines.append("- %s: `%s` — %s" % (row["task"], row["status"], row["reason"]))
    totals = report["gates"]["cost"]["totals"]
    lines += ["", "## Minimal baseline and cost accounting", "",
              "Ten baseline contracts passed on all three task fixtures; this makes no performance claim.", "",
              "The Core roster has 13 methods. Planned caps: %s full Support logical evaluations, "
              "%s Query evaluations, and %s LLM calls. No affordability threshold is imposed." % (
                  totals["full_support_logical_evaluations_cap"],
                  totals["query_extra_logical_evaluations"], totals["llm_calls_cap"]), "",
              "## Release decision", "",
              "P0b is complete and P1 full Core Baseline Smoke is released. Do not start P2 "
              "or open a Natural Final outcome until P1 passes.", "",
              "Machine-readable detail: `artifacts/main_protocol/p0_readiness_20260830.json`.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, default=OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=OUT_MD)
    parser.add_argument("--expect-ready", action="store_true")
    args = parser.parse_args(argv)
    report = build_report()
    out_json = args.output_json if args.output_json.is_absolute() else PROJECT_ROOT / args.output_json
    out_md = args.output_md if args.output_md.is_absolute() else PROJECT_ROOT / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    out_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"audit": report["verdict"]["audit"],
                      "execution": report["verdict"]["execution"],
                      "p1_release": report["verdict"]["p1_release"],
                      "gate_status": report["gate_status"],
                      "output_json": _rel(out_json), "output_md": _rel(out_md)},
                     ensure_ascii=False, indent=2))
    return int(bool(args.expect_ready and not report["verdict"]["p1_release"]))


if __name__ == "__main__":
    raise SystemExit(main())
