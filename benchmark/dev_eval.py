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
from .corruption import (
    CORRUPTION_GRID,
    apply_corruption,
    corruption_seed,
    replicates_for,
)
from .ingestion import canonical_ingest
from .metrics import seasonal_scale, smase
from .registry import SeriesRecord, read_registry_jsonl
from .report import DevDiscriminationRow, build_dev_discrimination_report
from .split import SplitManifest, SplitRole
from .trainers import NormalizationState, build_windows, fit_closed_form

FIXED_PROGRAMS = ("raw", "forward_fill", "seasonal_fill")
PROGRAM_IDS = FIXED_PROGRAMS + ("h_ref",)


def fold_to_headline(rows: list[ProgramLoss]) -> dict[str, object]:
    """Apply the frozen folding ladder to already-per-uid rows.

    Amendment-1 order, resumed from the point where one row per uid already exists:

        cell (dataset x regime): series-equal mean
        -> regime: dataset macro mean
        -> overall: equal mean over regimes

    The plain mean over every uid -- which is what a naive `np.mean(losses)` computes --
    is NOT this number.  It is a series-micro mean, so it silently weights each cell by
    how many series happen to be in it, letting the largest dataset (862 traffic sensors)
    dominate the headline over the smallest (20 GEFCom zones).  The micro mean is still
    reported, labelled, as a descriptive figure.
    """
    by_cell: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_cell[row.cell_id].append(row.loss)
    cell_means = {cell: float(np.mean(losses)) for cell, losses in sorted(by_cell.items())}

    by_regime: dict[str, list[float]] = defaultdict(list)
    for cell, mean in cell_means.items():
        # cell_id is "<dataset_id>|<regime>"; the macro step averages datasets within regime.
        regime = cell.rsplit("|", 1)[-1]
        by_regime[regime].append(mean)
    regime_means = {
        regime: float(np.mean(values)) for regime, values in sorted(by_regime.items())
    }

    return {
        "overall": float(np.mean([regime_means[key] for key in sorted(regime_means)])),
        "by_regime_dataset_macro": regime_means,
        "by_cell_series_equal": cell_means,
        "series_micro_descriptive": float(np.mean([row.loss for row in rows])),
        "n_uid": len(rows),
    }


def audit_h_ref_behaviour(
    raw_rows: list[ProgramLoss],
    h_ref_rows: list[ProgramLoss],
    *,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Ask what H_ref actually DOES, rather than assuming the ladder is doing work.

    A reference pipeline that is a no-op on most series and actively harmful on the rest
    is not a baseline worth beating -- and a benchmark whose "headroom" is really the
    oracle undoing that harm is measuring the wrong thing.  So this is reported up front,
    not derived later by whoever reads the numbers carefully enough.
    """
    raw_by_uid = {row.uid: row.loss for row in raw_rows}
    shared = [row for row in h_ref_rows if row.uid in raw_by_uid]
    if not shared:
        raise RuntimeError("H_ref audit needs Raw and H_ref rows over the same uids")

    no_op = 0
    helped = 0
    harmed = 0
    harm_total = 0.0
    help_total = 0.0
    for row in shared:
        raw_loss = raw_by_uid[row.uid]
        delta = row.loss - raw_loss
        if abs(delta) <= tolerance * max(1.0, abs(raw_loss)):
            no_op += 1
        elif delta > 0:
            harmed += 1
            harm_total += delta
        else:
            helped += 1
            help_total += -delta

    n = len(shared)
    return {
        "n_uid": n,
        "indistinguishable_from_raw": no_op,
        "indistinguishable_from_raw_fraction": no_op / n,
        "better_than_raw": helped,
        "worse_than_raw": harmed,
        "mean_improvement_where_better": (help_total / helped) if helped else 0.0,
        "mean_damage_where_worse": (harm_total / harmed) if harmed else 0.0,
        "net_vs_raw_series_micro": float(
            np.mean([row.loss - raw_by_uid[row.uid] for row in shared])
        ),
        "reading": (
            "indistinguishable_from_raw_fraction near 1.0 means H_ref is mostly a no-op, "
            "so 'H_ref' and 'Raw' are not two independent baselines. net_vs_raw > 0 means "
            "the reference ladder is, on net, worse than doing nothing."
        ),
    }


def dual_headroom(
    raw_rows: list[ProgramLoss],
    h_ref_rows: list[ProgramLoss],
    insample_rows: list[ProgramLoss],
) -> dict[str, object]:
    """Report headroom above BOTH floors, and flag where the oracle is only undoing harm.

    Measuring headroom only as `H_ref - oracle` is a trap: if H_ref hurt a cell, the
    oracle simply picks Raw back and the resulting "gain" is exactly the harm H_ref did.
    That is not repair space a method could win -- it is H_ref's own damage, refunded.
    Every such cell is flagged, and the honest number (`gain_over_raw`, the gain above the
    no-op floor) is reported beside it.
    """
    raw_by_cell: dict[str, list[float]] = defaultdict(list)
    h_ref_by_cell: dict[str, list[float]] = defaultdict(list)
    oracle_by_cell: dict[str, list[float]] = defaultdict(list)
    oracle_program_of_cell: dict[str, str] = {}

    for row in raw_rows:
        raw_by_cell[row.cell_id].append(row.loss)
    for row in h_ref_rows:
        h_ref_by_cell[row.cell_id].append(row.loss)
    for row in insample_rows:
        oracle_by_cell[row.cell_id].append(row.loss)
        oracle_program_of_cell[row.cell_id] = row.program_id

    cells: dict[str, dict[str, object]] = {}
    for cell in sorted(raw_by_cell):
        raw_mean = float(np.mean(raw_by_cell[cell]))
        h_ref_mean = float(np.mean(h_ref_by_cell[cell]))
        oracle_mean = float(np.mean(oracle_by_cell[cell]))
        selected = oracle_program_of_cell.get(cell, "unknown")
        cells[cell] = {
            "oracle_selected_program": selected,
            "raw_mean": raw_mean,
            "h_ref_mean": h_ref_mean,
            "oracle_insample_mean": oracle_mean,
            "gain_over_raw": raw_mean - oracle_mean,
            "gain_over_h_ref": h_ref_mean - oracle_mean,
            "h_ref_self_harm": h_ref_mean - raw_mean,
            "oracle_reverts_to_raw": selected == "raw",
        }

    reverting = sorted(cell for cell, row in cells.items() if row["oracle_reverts_to_raw"])
    return {
        "definition": {
            "gain_over_raw": (
                "pool-best minus the No-op floor. This is the honest headroom: space a "
                "method could actually win."
            ),
            "gain_over_h_ref": (
                "pool-best minus the current reference ladder. Inflated wherever H_ref "
                "hurt, because the oracle just picks Raw back."
            ),
            "h_ref_self_harm": "positive means H_ref is worse than doing nothing on that cell",
        },
        "cells": cells,
        "cells_where_oracle_reverts_to_raw": reverting,
        "n_cells_reverting_to_raw": len(reverting),
    }


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


def _expected_measurements() -> int:
    return sum(len(replicates_for(scenario)) for scenario, _ in CORRUPTION_GRID)


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
    """Evaluate every program on one role.

    The downstream model is trained ONCE per (program, scenario, dose, replicate) on the
    whole role's pooled inner-train, with series-equal weighting.  It is emphatically NOT
    trained per dataset x regime cell: `regime_tag` is a benchmark-private diagnostic
    label, and slicing the training pool by it would (a) hand the model a private label no
    method is allowed to see and (b) train a different, smaller model for every cell, so
    "the shared model" would not be shared at all.  `cell_id` survives only as a reporting
    key.
    """
    cell_of_uid: dict[str, str] = {}
    for assignment in assignments:
        if assignment.role is role:
            cell_of_uid[assignment.series_uid] = (
                f"{assignment.dataset_id}|{assignment.regime_tag}"
            )
    uids = sorted(cell_of_uid)
    if not uids:
        return [], {}, [], [], []

    # Keyed by (program, cell, uid, scenario, dose) -> one loss per replicate, so the
    # replicate fold happens BEFORE the scenario x dose fold. A flat mean over all
    # measurements would silently weight the one-replicate Natural lane at half the
    # weight of every two-replicate stochastic lane.
    raw_losses: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    fill_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    prepare_times: list[float] = []
    trainer_times: list[float] = []
    repeat_rows: list[dict[str, object]] = []

    for scenario, dose in CORRUPTION_GRID:
        for replicate in replicates_for(scenario):
            histories = {
                uid: _corrupt_history(
                    records_by_uid[uid], values_by_uid[uid], scenario, dose, replicate
                )
                for uid in uids
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
                normalization = {uid: NormalizationState.fit(inner[uid]) for uid in inner}
                prepare_times.append(time.perf_counter() - started)

                started = time.perf_counter()
                batch = build_windows(prepared_inner, normalization)
                model = fit_closed_form(batch)
                trainer_times.append(time.perf_counter() - started)

                for uid in uids:
                    cell_id = cell_of_uid[uid]
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
                    raw_losses[(program_id, cell_id, uid, scenario, dose)].append(loss)
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

    # Fold 1: average corruption replicates within uid x scenario x dose.
    folded: dict[tuple[str, str, str], dict[tuple[str, float], float]] = defaultdict(dict)
    for (program_id, cell_id, uid, scenario, dose), losses in raw_losses.items():
        if len(losses) != len(replicates_for(scenario)):
            raise RuntimeError(
                f"uid {uid} is missing a corruption replicate for {scenario}/{dose}"
            )
        folded[(program_id, cell_id, uid)][(scenario, dose)] = float(np.mean(losses))

    # Fold 2: equal-average every frozen scenario x dose value -> exactly one row per uid.
    grid = set(CORRUPTION_GRID)
    rows: list[ProgramLoss] = []
    for (program_id, cell_id, uid), by_cell in sorted(folded.items()):
        if set(by_cell) != grid:
            raise RuntimeError(
                f"uid {uid} is missing a corruption scenario/dose under {program_id}"
            )
        mean_loss = float(np.mean([by_cell[key] for key in sorted(by_cell)]))
        rows.append(ProgramLoss(role.value, cell_id, program_id, uid, mean_loss))
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

    # The sMASE denominator, carried into the report so a cell whose losses look enormous
    # can be read as "tiny seasonal scale" rather than "catastrophic forecast".
    scales_by_uid = {
        uid: float(
            seasonal_scale(
                values[uid][: len(values[uid]) - 2 * HEADLINE_HORIZON],
                np.isfinite(values[uid][: len(values[uid]) - 2 * HEADLINE_HORIZON]),
                period=_period(records_by_uid[uid]),
                min_pairs=32,
            )
        )
        for uid in sorted(h_ref_by_uid)
    }
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
    report = build_dev_discrimination_report(
        discrimination_rows,
        raw_loss_by_uid={row.uid: row.loss for row in dev if row.program_id == "raw"},
        seasonal_scale_by_uid=scales_by_uid,
    )
    fill_disclosure = {
        f"{program}|{cell}": float(np.mean(values))
        for (program, cell), values in sorted(dev_fill.items())
    }
    report["ingestion_fill_rate_by_method_cell"] = fill_disclosure

    raw_rows = [row for row in dev if row.program_id == "raw"]
    report["baseline_smase"] = {
        "raw": fold_to_headline(raw_rows),
        "best_fixed": fold_to_headline(best_rows),
        "h_ref": fold_to_headline(h_ref_rows),
        "oracle_transfer": fold_to_headline(transfer_rows) if transfer_rows else None,
        "oracle_insample": fold_to_headline(insample_rows),
    }
    report["aggregation_note"] = (
        "'overall' follows the frozen ladder (cell series-equal -> regime dataset-macro "
        "-> mean over regimes). 'series_micro_descriptive' is the plain per-uid mean and "
        "is descriptive only -- it lets the biggest dataset dominate."
    )
    report["best_fixed_program"] = best.program_id
    report["oracle_transfer_missing_support_cells"] = list(transfer_missing_cells)
    report["closed_form_model_seed_semantics"] = "deterministic_not_applicable"
    report["h_ref_behaviour_audit"] = audit_h_ref_behaviour(raw_rows, h_ref_rows)
    report["headroom"] = dual_headroom(raw_rows, h_ref_rows, insample_rows)
    # Numbers only: bind_dev_report_to_manifest validates every value here as a positive
    # finite float. Provenance lives in its own key.
    report["timeout_calibration_seconds"] = {
        "prepare_p95_x2": 2.0 * float(np.quantile(support_prepare + dev_prepare, 0.95)),
        "trainer_p95_x2": 2.0 * float(np.quantile(support_train + dev_train, 0.95)),
    }
    report["timeout_calibration_provenance"] = {
        "measured_on": (
            "the corrected per-config pooled training path; the pre-fix per-cell numbers "
            "were measured on much smaller training pools and do not transfer"
        ),
        "trainer_scope": "closed_form_only -- Adam and LSTM must be calibrated separately",
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
    (output / "baseline_report.md").write_text(
        _baseline_report_markdown(report, best.program_id),
        encoding="utf-8",
    )
    return report


def _baseline_report_markdown(report: dict[str, object], best_program: str) -> str:
    baselines = report["baseline_smase"]
    audit = report["h_ref_behaviour_audit"]
    headroom = report["headroom"]

    lines = [
        f"# Benchmark {BENCHMARK_VERSION} Dev-Query baseline report",
        "",
        "Final-Query was not read. Every value below is repeatable Dev-Query sMASE.",
        "",
        "## Baselines (frozen folding ladder)",
        "",
        "| baseline | overall (macro) | series-micro (descriptive) |",
        "| --- | --- | --- |",
    ]
    for name in ("raw", "best_fixed", "h_ref", "oracle_transfer", "oracle_insample"):
        fold = baselines.get(name)
        if fold is None:
            lines.append(f"| {name} | n/a | n/a |")
            continue
        lines.append(
            f"| {name} | {fold['overall']:.6f} | {fold['series_micro_descriptive']:.6f} |"
        )
    lines += [
        "",
        f"Best-fixed program selected on Support-A: `{best_program}`.",
        "",
        "## What H_ref actually does",
        "",
        f"- Indistinguishable from Raw on {audit['indistinguishable_from_raw']}"
        f"/{audit['n_uid']} Dev series "
        f"({audit['indistinguishable_from_raw_fraction']:.1%}).",
        f"- Better than Raw on {audit['better_than_raw']}; worse on {audit['worse_than_raw']}.",
        f"- Net vs Raw (series-micro): {audit['net_vs_raw_series_micro']:+.6f} "
        "(positive = worse than doing nothing).",
        "",
        "## Headroom, measured from both floors",
        "",
        f"The oracle reverts to Raw in {headroom['n_cells_reverting_to_raw']} cell(s): "
        f"`{headroom['cells_where_oracle_reverts_to_raw']}`. In those cells the apparent "
        "gain over H_ref is H_ref's own damage refunded, not repair space.",
        "",
        "| cell | oracle pick | gain over Raw | gain over H_ref | H_ref self-harm |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cell, row in headroom["cells"].items():
        flag = " *(reverts)*" if row["oracle_reverts_to_raw"] else ""
        lines.append(
            f"| `{cell}` | {row['oracle_selected_program']}{flag} "
            f"| {row['gain_over_raw']:+.4f} | {row['gain_over_h_ref']:+.4f} "
            f"| {row['h_ref_self_harm']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "FIXED_PROGRAMS",
    "PROGRAM_IDS",
    "aggregate_per_dose",
    "apply_fixed_program",
    "audit_h_ref_behaviour",
    "bind_dev_report_to_manifest",
    "canonical_evaluation_context",
    "dual_headroom",
    "fold_to_headline",
    "oracle_transfer_with_coverage",
    "run_dev_evaluation",
]
