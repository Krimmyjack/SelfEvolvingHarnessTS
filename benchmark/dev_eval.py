"""Repeatable Dev-Query closed-form baseline evaluation; never reads Final-Query."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..p6.fast_path import prepared_artifact, run_fast_path
from ..p6.harness_state import default_state
from . import BENCHMARK_VERSION, HEADLINE_HORIZON
from .baselines import (
    ProgramLoss,
    oracle_insample,
    oracle_transfer,
    select_best_fixed,
)
from .corruption import CORRUPTION_GRID, apply_corruption, corruption_seed
from .ingestion import canonical_ingest
from .metrics import seasonal_scale, smase
from .registry import SeriesRecord, read_registry_jsonl
from .report import DevDiscriminationRow, build_dev_discrimination_report
from .split import SplitManifest, SplitRole
from .trainers import NormalizationState, build_windows, fit_closed_form

FIXED_PROGRAMS = ("raw", "forward_fill", "seasonal_fill")
PROGRAM_IDS = FIXED_PROGRAMS + ("h_ref",)


def aggregate_per_dose(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (
            str(row["program_id"]),
            str(row["cell_id"]),
            str(row["scenario"]),
            float(row["dose"]),
        )
        grouped[key].append(float(row["loss"]))
    return [
        {
            "program_id": program,
            "cell_id": cell,
            "scenario": scenario,
            "dose": dose,
            "mean_smase": float(np.mean(losses)),
            "n_measurements": len(losses),
        }
        for (program, cell, scenario, dose), losses in sorted(grouped.items())
    ]


def canonical_evaluation_context(
    prepared_history: np.ndarray, *, lookback: int
) -> tuple[np.ndarray, float]:
    if isinstance(lookback, bool) or not isinstance(lookback, int) or lookback < 1:
        raise ValueError("evaluation lookback must be positive")
    ingested = canonical_ingest(np.asarray(prepared_history))
    if len(ingested.values) < lookback:
        raise ValueError("prepared history is shorter than evaluation lookback")
    return ingested.values[-lookback:].copy(), ingested.fill_rate


def oracle_transfer_with_coverage(
    support_rows: list[ProgramLoss], query_rows: list[ProgramLoss]
) -> tuple[list[ProgramLoss], tuple[str, ...]]:
    support_cells = {row.cell_id for row in support_rows}
    query_cells = {row.cell_id for row in query_rows}
    missing = tuple(sorted(query_cells - support_cells))
    comparable = [row for row in query_rows if row.cell_id in support_cells]
    selected = oracle_transfer(support_rows, comparable) if comparable else []
    return selected, missing


def bind_dev_report_to_manifest(
    manifest_path: Path | str,
    report_bytes: bytes,
    timeout_values: dict[str, float],
) -> str:
    path = Path(manifest_path)
    payload = json.loads(path.read_text("utf-8"))
    if payload.get("final_query_state") != "sealed":
        raise RuntimeError("Dev report can bind only while Final-Query remains sealed")
    expected = {"prepare_p95_x2", "trainer_p95_x2"}
    if set(timeout_values) != expected or not all(
        np.isfinite(value) and value > 0 for value in timeout_values.values()
    ):
        raise ValueError("Dev timeout binding requires positive prepare/trainer p95x2 values")
    digest = hashlib.sha256(report_bytes).hexdigest()
    payload["dev_discrimination_report_sha256"] = digest
    payload["dev_evaluation_state"] = "frozen"
    payload["timeouts_seconds"] = {
        "prepare": float(timeout_values["prepare_p95_x2"]),
        "trainer": float(timeout_values["trainer_p95_x2"]),
        "rule": "same_hardware_dev_p95_x2",
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return digest


def apply_fixed_program(program_id: str, values: np.ndarray, *, period: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size == 0 or np.isinf(source).any():
        raise ValueError("program input must be a non-empty finite-or-NaN vector")
    result = source.copy()
    if program_id == "raw":
        return result
    if program_id == "forward_fill":
        finite = np.flatnonzero(np.isfinite(result))
        if not finite.size:
            raise ValueError("forward_fill has no finite anchor")
        first = int(finite[0])
        result[:first] = result[first]
        for index in range(first + 1, len(result)):
            if not np.isfinite(result[index]):
                result[index] = result[index - 1]
        return result
    if program_id == "seasonal_fill":
        if isinstance(period, bool) or not isinstance(period, int) or period < 1:
            raise ValueError("seasonal_fill period must be positive")
        for index in range(len(result)):
            if not np.isfinite(result[index]) and index >= period and np.isfinite(result[index - period]):
                result[index] = result[index - period]
        return canonical_ingest(result).values.copy()
    raise ValueError(f"unknown fixed program: {program_id!r}")


def _slot(record: SeriesRecord, clean_root: Path) -> Path:
    key = hashlib.sha256(
        json.dumps(
            [record.source_id, record.dataset_id, record.entity_id],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return clean_root / key


def _load_values(records: list[SeriesRecord], clean_root: Path) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for record in records:
        array = np.load(_slot(record, clean_root) / "values.npy", allow_pickle=False)
        record.verify_values(array, timestamps=None if record.timestamps_sha is None else np.load(
            _slot(record, clean_root) / "timestamps.npy", allow_pickle=False
        ))
        values[record.series_uid] = np.asarray(array, dtype=np.float64)
    return values


def _period(record: SeriesRecord) -> int:
    return {"hourly": 24, "daily": 7, "monthly": 12}[record.frequency]


def _corrupt_history(record: SeriesRecord, values: np.ndarray, scenario: str, dose: float, replicate: int) -> np.ndarray:
    history = values[:-HEADLINE_HORIZON]
    seed = corruption_seed(BENCHMARK_VERSION, record.content_sha, scenario, dose, replicate)
    return apply_corruption(history, scenario=scenario, dose=dose, seed=seed).copy()


def _program_values(
    program_id: str,
    histories: dict[str, np.ndarray],
    inner: dict[str, np.ndarray],
    records: dict[str, SeriesRecord],
    h_ref_choices: object,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    prepared_history: dict[str, np.ndarray] = {}
    prepared_inner: dict[str, np.ndarray] = {}
    for uid in histories:
        period = _period(records[uid])
        if program_id == "h_ref":
            choice = h_ref_choices.get(uid)
            full_artifact = prepared_artifact(choice, histories[uid])
            train_artifact = prepared_artifact(choice, inner[uid])
            if full_artifact is None or train_artifact is None:
                raise RuntimeError(f"H_ref execution failed for {uid}")
            prepared_history[uid] = np.asarray(full_artifact, dtype=np.float64)
            prepared_inner[uid] = np.asarray(train_artifact, dtype=np.float64)
        else:
            prepared_history[uid] = apply_fixed_program(program_id, histories[uid], period=period)
            prepared_inner[uid] = apply_fixed_program(program_id, inner[uid], period=period)
    return prepared_history, prepared_inner


def _evaluate_role(
    role: SplitRole,
    assignments: list[object],
    records_by_uid: dict[str, SeriesRecord],
    values_by_uid: dict[str, np.ndarray],
) -> tuple[
    list[ProgramLoss],
    dict[tuple[str, str], list[float]],
    list[float],
    list[float],
    list[dict[str, object]],
]:
    by_cell: dict[str, list[str]] = defaultdict(list)
    for assignment in assignments:
        if assignment.role is role:
            by_cell[f"{assignment.dataset_id}|{assignment.regime_tag}"].append(assignment.series_uid)
    raw_losses: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    fill_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    prepare_times: list[float] = []
    trainer_times: list[float] = []
    repeat_rows: list[dict[str, object]] = []
    for scenario, dose in CORRUPTION_GRID:
        for replicate in (0, 1):
            for cell_id, uids in sorted(by_cell.items()):
                histories = {
                    uid: _corrupt_history(records_by_uid[uid], values_by_uid[uid], scenario, dose, replicate)
                    for uid in sorted(uids)
                }
                inner = {
                    uid: history[: len(values_by_uid[uid]) - 2 * HEADLINE_HORIZON]
                    for uid, history in histories.items()
                }
                state = default_state()
                started = time.perf_counter()
                h_ref_choices = run_fast_path(inner, state, state.sampler.expected_total)
                prepare_times.append(time.perf_counter() - started)
                for program_id in PROGRAM_IDS:
                    started = time.perf_counter()
                    prepared_history, prepared_inner = _program_values(
                        program_id, histories, inner, records_by_uid, h_ref_choices
                    )
                    normalization = {
                        uid: NormalizationState.fit(inner[uid]) for uid in inner
                    }
                    prepare_times.append(time.perf_counter() - started)
                    started = time.perf_counter()
                    batch = build_windows(prepared_inner, normalization)
                    model = fit_closed_form(batch)
                    trainer_times.append(time.perf_counter() - started)
                    for uid in sorted(uids):
                        context, fill_rate = canonical_evaluation_context(
                            prepared_history[uid], lookback=48
                        )
                        normalized = normalization[uid].normalize(context)
                        prediction = normalization[uid].denormalize(
                            model.predict(normalized[None, :])[0]
                        )
                        clean = values_by_uid[uid]
                        train = clean[: len(clean) - 2 * HEADLINE_HORIZON]
                        scale = seasonal_scale(
                            train,
                            np.isfinite(train),
                            period=_period(records_by_uid[uid]),
                            min_pairs=32,
                        )
                        loss = smase(clean[-HEADLINE_HORIZON:], prediction, scale=scale)
                        raw_losses[(program_id, cell_id, uid)].append(loss)
                        fill_rates[(program_id, cell_id)].append(fill_rate)
                        repeat_rows.append(
                            {
                                "split_role": role.value,
                                "program_id": program_id,
                                "cell_id": cell_id,
                                "uid": uid,
                                "scenario": scenario,
                                "dose": dose,
                                "corruption_replicate": replicate,
                                "loss": loss,
                            }
                        )
    rows: list[ProgramLoss] = []
    for (program_id, cell_id, uid), losses in sorted(raw_losses.items()):
        if len(losses) != len(CORRUPTION_GRID) * 2:
            raise RuntimeError("Dev loss folding lacks a corruption scenario or replicate")
        rows.append(
            ProgramLoss(role.value, cell_id, program_id, uid, float(np.mean(losses)))
        )
    return rows, fill_rates, prepare_times, trainer_times, repeat_rows


def run_dev_evaluation(root: Path | str, out: Path | str) -> dict[str, object]:
    """Evaluate the full public baseline pool on repeatable Dev-Query only."""

    data_root, output = Path(root), Path(out)
    records = read_registry_jsonl(output / "series_registry.jsonl")
    records_by_uid = {row.series_uid: row for row in records}
    split = SplitManifest.from_dict(json.loads((output / "split_manifest.json").read_text("utf-8")))
    selected_uids = {
        row.series_uid for row in split.assignments
        if row.role in {SplitRole.SUPPORT_A, SplitRole.DEV_QUERY}
    }
    selected_records = [records_by_uid[uid] for uid in sorted(selected_uids)]
    values = _load_values(selected_records, data_root / "clean_base")
    support, support_fill, support_prepare, support_train, support_repeats = _evaluate_role(
        SplitRole.SUPPORT_A, list(split.assignments), records_by_uid, values
    )
    dev, dev_fill, dev_prepare, dev_train, dev_repeats = _evaluate_role(
        SplitRole.DEV_QUERY, list(split.assignments), records_by_uid, values
    )
    if not dev:
        raise RuntimeError("frozen split has no Dev-Query rows")
    best = select_best_fixed(support)
    best_rows = [row for row in dev if row.program_id == best.program_id]
    h_ref_rows = [row for row in dev if row.program_id == "h_ref"]
    insample_rows = oracle_insample(dev)
    transfer_rows, transfer_missing_cells = oracle_transfer_with_coverage(support, dev)
    h_ref_by_uid = {row.uid: row for row in h_ref_rows}
    insample_by_uid = {row.uid: row for row in insample_rows}
    discrimination_rows = [
        DevDiscriminationRow(
            split_role="dev_query",
            dataset_id=records_by_uid[uid].dataset_id,
            regime=records_by_uid[uid].regime_tag or "unassigned",
            uid=uid,
            h_ref_loss=h_ref_by_uid[uid].loss,
            oracle_insample_loss=insample_by_uid[uid].loss,
        )
        for uid in sorted(h_ref_by_uid)
    ]
    report = build_dev_discrimination_report(discrimination_rows)
    fill_disclosure = {
        f"{program}|{cell}": float(np.mean(values))
        for (program, cell), values in sorted(dev_fill.items())
    }
    report["ingestion_fill_rate_by_method_cell"] = fill_disclosure
    report["baseline_mean_smase"] = {
        "raw": float(np.mean([row.loss for row in dev if row.program_id == "raw"])),
        "best_fixed": float(np.mean([row.loss for row in best_rows])),
        "h_ref": float(np.mean([row.loss for row in h_ref_rows])),
        "oracle_transfer": (
            float(np.mean([row.loss for row in transfer_rows])) if transfer_rows else None
        ),
        "oracle_insample": float(np.mean([row.loss for row in insample_rows])),
    }
    report["best_fixed_program"] = best.program_id
    report["oracle_transfer_missing_support_cells"] = list(transfer_missing_cells)
    report["closed_form_model_seed_semantics"] = "deterministic_not_applicable"
    report["timeout_calibration_seconds"] = {
        "prepare_p95_x2": 2.0 * float(np.quantile(support_prepare + dev_prepare, 0.95)),
        "trainer_p95_x2": 2.0 * float(np.quantile(support_train + dev_train, 0.95)),
    }
    report_bytes = (
        json.dumps(report, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    ).encode("utf-8")
    (output / "dev_discrimination_report.json").write_bytes(report_bytes)
    bind_dev_report_to_manifest(
        output / "benchmark_manifest_v0.yaml",
        report_bytes,
        report["timeout_calibration_seconds"],
    )
    with (output / "dev_program_losses.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in support + dev:
            handle.write(json.dumps(row.__dict__, sort_keys=True, ensure_ascii=True) + "\n")
    (output / "dev_per_dose_report.json").write_text(
        json.dumps(
            {
                "split_role": "dev_query",
                "folding": "corruption replicate then uid/cell mean; scenario and dose remain separate",
                "rows": aggregate_per_dose(dev_repeats),
            },
            sort_keys=True,
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    means = report["baseline_mean_smase"]
    (output / "baseline_report.md").write_text(
        "# Benchmark v0 Dev-Query baseline report\n\n"
        "Final-Query was not read. All values below are repeatable Dev-Query sMASE.\n\n"
        + "\n".join(f"- {name}: {value:.6f}" for name, value in means.items())
        + f"\n\nBest-fixed program selected on Support-A: `{best.program_id}`.\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "FIXED_PROGRAMS",
    "PROGRAM_IDS",
    "aggregate_per_dose",
    "apply_fixed_program",
    "bind_dev_report_to_manifest",
    "canonical_evaluation_context",
    "oracle_transfer_with_coverage",
    "run_dev_evaluation",
]
