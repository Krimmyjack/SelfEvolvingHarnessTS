"""Repeatable Dev-Query closed-form baseline evaluation; never reads Final-Query."""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from ._frozen_reference.fast_path import prepared_artifact
from .method_compat import _run_frozen_reference_batch
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
from .materialize import write_text_lf
from .metrics import seasonal_scale, smase
from .power import power_panel
from .programs import (
    PROGRAM_IDS,
    RUNNER_EXECUTED,
    apply_program,
    mechanism_of,
    pool_manifest,
)
from .registry import SeriesRecord, read_registry_jsonl
from .report import DevDiscriminationRow, build_dev_discrimination_report
from .split import SplitManifest, SplitRole
from .trainers import NormalizationState, build_windows, fit_closed_form

# The two retrained oracles. They are not programs a method could run -- they are runner
# privileges -- but they are evaluated through the identical path as every program, which
# is the whole point of retraining them.
ORACLE_TRANSFER_RETRAINED = "oracle_transfer_retrained"
ORACLE_INSAMPLE_RETRAINED = "oracle_insample_retrained"


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


def mechanism_panel(repeat_rows: list[dict[str, object]]) -> dict[str, object]:
    """The second reporting axis: dataset x scenario x dose, never folded into the first.

    The headline aggregation is dataset x regime, and it has to stay that way -- it is the
    frozen ladder.  But a defect-mechanism question ("can anything in this pool touch a
    level shift?") is invisible there, because the fold averages every scenario together.
    So the mechanism axis gets its own table.  The `programs_indistinguishable` flag is the
    one that mattered in v0.1: when every program's mean agrees to four decimals, the pool
    has no action for that defect, and any "saturation" reading is a fact about the pool,
    not about the data.
    """
    grouped: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in repeat_rows:
        dataset = str(row["cell_id"]).rsplit("|", 1)[0]
        key = (dataset, str(row["scenario"]), float(row["dose"]))
        grouped[key][str(row["program_id"])].append(float(row["loss"]))

    panel: list[dict[str, object]] = []
    for (dataset, scenario, dose), by_program in sorted(grouped.items()):
        means = {
            program: float(np.mean(losses)) for program, losses in sorted(by_program.items())
        }
        pool_means = {
            program: value
            for program, value in means.items()
            if program not in RUNNER_EXECUTED
            and program not in {ORACLE_TRANSFER_RETRAINED, ORACLE_INSAMPLE_RETRAINED}
        }
        spread = (max(pool_means.values()) - min(pool_means.values())) if pool_means else 0.0
        best_program = min(pool_means, key=lambda key: (pool_means[key], key))
        panel.append(
            {
                "dataset_id": dataset,
                "scenario": scenario,
                "dose": dose,
                "mechanism_of_best_program": mechanism_of(best_program),
                "best_pool_program": best_program,
                "program_mean_smase": means,
                "pool_spread": spread,
                "programs_indistinguishable": bool(spread < 1e-4),
                "n_series": len(next(iter(by_program.values()))) if by_program else 0,
            }
        )
    dead = [
        f"{row['dataset_id']}|{row['scenario']}|{row['dose']}"
        for row in panel
        if row["programs_indistinguishable"]
    ]
    return {
        "reading": (
            "programs_indistinguishable=true means no program in the frozen pool can act on "
            "that defect. That is a capability gap in the operator library, reported as "
            "such; it is not evidence that the data has nothing to gain."
        ),
        "rows": panel,
        "cells_where_pool_cannot_act": dead,
        "n_cells_where_pool_cannot_act": len(dead),
    }


def dual_headroom(
    raw_rows: list[ProgramLoss],
    h_ref_rows: list[ProgramLoss],
    insample_rows: list[ProgramLoss],
    *,
    selection_by_cell: dict[str, list[str]] | None = None,
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
        # A retrained oracle carries the oracle's own id on every row, because its picks
        # vary by (scenario, dose) within the cell. In that case the caller hands over the
        # actual picks, and "reverts to Raw" becomes a majority question rather than a
        # single label.
        if selection_by_cell is not None:
            picks = sorted(selection_by_cell.get(cell, []))
            raw_share = (
                sum(1 for pick in picks if pick == "raw") / len(picks) if picks else 0.0
            )
            selected: object = picks
            reverts = raw_share >= 0.5
        else:
            single = oracle_program_of_cell.get(cell, "unknown")
            selected = single
            raw_share = 1.0 if single == "raw" else 0.0
            reverts = single == "raw"
        cells[cell] = {
            "oracle_selected_program": selected,
            "oracle_raw_pick_share": raw_share,
            "raw_mean": raw_mean,
            "h_ref_mean": h_ref_mean,
            "oracle_insample_mean": oracle_mean,
            "gain_over_raw": raw_mean - oracle_mean,
            "gain_over_h_ref": h_ref_mean - oracle_mean,
            "h_ref_self_harm": h_ref_mean - raw_mean,
            "oracle_reverts_to_raw": reverts,
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
    write_text_lf(
        path, json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2) + "\n"
    )
    return digest


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
        if program_id in RUNNER_EXECUTED:
            choice = h_ref_choices.get(uid)
            full_artifact = prepared_artifact(choice, histories[uid])
            train_artifact = prepared_artifact(choice, inner[uid])
            if full_artifact is None or train_artifact is None:
                raise RuntimeError(f"H_ref execution failed for {uid}")
            prepared_history[uid] = np.asarray(full_artifact, dtype=np.float64)
            prepared_inner[uid] = np.asarray(train_artifact, dtype=np.float64)
        else:
            prepared_history[uid] = apply_program(program_id, histories[uid], period=period)
            prepared_inner[uid] = apply_program(program_id, inner[uid], period=period)
    return prepared_history, prepared_inner


def _mixed_values(
    policy: dict[str, str],
    prepared_history_by_program: dict[str, dict[str, np.ndarray]],
    prepared_inner_by_program: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Assemble the corpus a per-series policy actually produces.

    Every program has already been run over every series for this corruption realization,
    so composing the mixture costs nothing but a dictionary lookup.  What it buys is the
    thing the old oracle never had: a corpus that a model can then be trained *on*.
    """
    history = {uid: prepared_history_by_program[program][uid] for uid, program in policy.items()}
    inner = {uid: prepared_inner_by_program[program][uid] for uid, program in policy.items()}
    return history, inner


def _fit_and_score(
    uids_by_dataset: dict[str, list[str]],
    prepared_inner: dict[str, np.ndarray],
    prepared_history: dict[str, np.ndarray],
    normalization: dict[str, NormalizationState],
    records_by_uid: dict[str, SeriesRecord],
    scale_by_uid: dict[str, float],
    clean_by_uid: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, float], float]:
    """Train one closed-form model per dataset and score that dataset's series with it.

    The training unit is `(program, scenario, dose, replicate, dataset)`.  Under the v0.1
    role-pooled unit, a program applied to COVID moved the shared weights that scored
    traffic, so "the effect of this program on this dataset" was never isolated -- and the
    per-cell oracle described a mixed corpus that no model had been fitted to.  Slicing by
    `dataset_id` leaks nothing (it is public metadata, unlike `regime_tag`) and makes a
    dataset-conditioned choice a world that was actually trained.
    """
    losses: dict[str, float] = {}
    fills: dict[str, float] = {}
    trainer_seconds = 0.0
    for dataset in sorted(uids_by_dataset):
        ds_uids = uids_by_dataset[dataset]
        started = time.perf_counter()
        batch = build_windows(
            {uid: prepared_inner[uid] for uid in ds_uids},
            {uid: normalization[uid] for uid in ds_uids},
        )
        model = fit_closed_form(batch)
        trainer_seconds += time.perf_counter() - started

        for uid in ds_uids:
            context, fill_rate = canonical_evaluation_context(
                prepared_history[uid], lookback=48
            )
            normalized = normalization[uid].normalize(context)
            prediction = normalization[uid].denormalize(model.predict(normalized[None, :])[0])
            losses[uid] = smase(
                clean_by_uid[uid][-HEADLINE_HORIZON:], prediction, scale=scale_by_uid[uid]
            )
            fills[uid] = fill_rate
    return losses, fills, trainer_seconds


def _policy_from_cell_means(
    cell_means: dict[tuple[str, str], float],
    cells: set[str],
    programs: tuple[str, ...],
    fallback: str,
) -> tuple[dict[str, str], list[str]]:
    """Pick the best program per cell; report the cells with no evidence to pick from."""
    policy: dict[str, str] = {}
    uncovered: list[str] = []
    for cell in sorted(cells):
        scored = [
            (cell_means[(cell, program)], program)
            for program in programs
            if (cell, program) in cell_means
        ]
        if not scored:
            policy[cell] = fallback
            uncovered.append(cell)
            continue
        policy[cell] = min(scored)[1]
    return policy, uncovered


def _expected_measurements() -> int:
    return sum(len(replicates_for(scenario)) for scenario, _ in CORRUPTION_GRID)


def _evaluate_role(
    role: SplitRole,
    assignments: list[object],
    records_by_uid: dict[str, SeriesRecord],
    values_by_uid: dict[str, np.ndarray],
    *,
    transfer_policy: dict[tuple[str, str, float], dict[str, str]] | None = None,
    best_fixed_program: str = "raw",
) -> tuple[
    list[ProgramLoss],
    dict[tuple[str, str], list[float]],
    list[float],
    list[float],
    list[dict[str, object]],
    dict[str, object],
]:
    """Evaluate every pool program, plus both retrained oracles, on one role.

    Two things here are the v0.2 protocol, and both are deliberate.

    **The training unit is per dataset.**  One closed-form model is fitted for every
    `(program, scenario, dose, replicate, dataset)`.  `regime_tag` never touches the
    training pool -- it is a benchmark-private label and slicing by it would hand the model
    something no method may see -- so `cell_id` survives only as a reporting key.

    **The oracles are retrained.**  Selecting a different program for each cell describes a
    corpus that mixes programs.  Reading that oracle's loss off the per-program models --
    which is what v0/v0.1 did -- reports a world in which no model was ever fitted to the
    corpus the policy actually produces.  So once a policy is chosen, its corpus is
    assembled and a model is trained *on it*, through the identical path a Method takes.
    The floor (`best_fixed`), the ceiling (the oracles), and any method are then the same
    kind of measurement, and the gap between them is a gap a method could actually close.
    """
    cell_of_uid: dict[str, str] = {}
    dataset_of_uid: dict[str, str] = {}
    for assignment in assignments:
        if assignment.role is role:
            cell_of_uid[assignment.series_uid] = (
                f"{assignment.dataset_id}|{assignment.regime_tag}"
            )
            dataset_of_uid[assignment.series_uid] = assignment.dataset_id
    uids = sorted(cell_of_uid)
    if not uids:
        return [], {}, [], [], [], {}, {}

    uids_by_dataset: dict[str, list[str]] = defaultdict(list)
    for uid in uids:
        uids_by_dataset[dataset_of_uid[uid]].append(uid)
    uids_by_dataset = {key: sorted(value) for key, value in sorted(uids_by_dataset.items())}
    cells = {cell_of_uid[uid] for uid in uids}

    # The sMASE denominator and the clean future are properties of the clean series, so
    # they are constant across programs and corruption realizations. Hoisted out of the
    # innermost loop, where the pool's ninefold widening would otherwise recompute them.
    clean_by_uid = {uid: values_by_uid[uid] for uid in uids}
    scale_by_uid = {
        uid: seasonal_scale(
            clean_by_uid[uid][: len(clean_by_uid[uid]) - 2 * HEADLINE_HORIZON],
            np.isfinite(clean_by_uid[uid][: len(clean_by_uid[uid]) - 2 * HEADLINE_HORIZON]),
            period=_period(records_by_uid[uid]),
            min_pairs=32,
        )
        for uid in uids
    }

    # Keyed by (program, cell, uid, scenario, dose) -> one loss per replicate, so the
    # replicate fold happens BEFORE the scenario x dose fold. A flat mean over all
    # measurements would silently weight the one-replicate Natural lane at half the
    # weight of every two-replicate stochastic lane.
    raw_losses: dict[tuple[str, str, str, str, float], list[float]] = defaultdict(list)
    fill_rates: dict[tuple[str, str], list[float]] = defaultdict(list)
    prepare_times: list[float] = []
    trainer_times: list[float] = []
    repeat_rows: list[dict[str, object]] = []
    # (cell, program) -> per-(scenario,dose) mean, the evidence a transfer policy is built
    # from. Recorded per scenario/dose because degradation-conditioned choice is exactly
    # what C1 is about.
    cell_loss_by_grid: dict[tuple[str, float], dict[tuple[str, str], list[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    oracle_selections: dict[str, list[dict[str, object]]] = {}
    oracle_picks: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    uncovered_transfer_cells: set[str] = set()

    all_programs = tuple(PROGRAM_IDS)
    # H_ref is the incumbent being measured, not a tool the oracle may reach for. Letting
    # the ceiling pick it would make "headroom" partly a statement about H_ref's own
    # quality rather than about the pool's reachable space.
    selectable = tuple(p for p in all_programs if p not in RUNNER_EXECUTED)

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
            # Normalization is fitted on the PRE-method degraded inner-train and is shared
            # by every program, so it is computed once here rather than once per program.
            normalization = {uid: NormalizationState.fit(inner[uid]) for uid in inner}

            started = time.perf_counter()
            h_ref_choices = _run_frozen_reference_batch(inner)
            prepare_times.append(time.perf_counter() - started)

            prepared_history_by_program: dict[str, dict[str, np.ndarray]] = {}
            prepared_inner_by_program: dict[str, dict[str, np.ndarray]] = {}
            losses_by_program: dict[str, dict[str, float]] = {}

            for program_id in all_programs:
                started = time.perf_counter()
                prepared_history, prepared_inner = _program_values(
                    program_id, histories, inner, records_by_uid, h_ref_choices
                )
                prepare_times.append(time.perf_counter() - started)
                prepared_history_by_program[program_id] = prepared_history
                prepared_inner_by_program[program_id] = prepared_inner

                losses, fills, trainer_seconds = _fit_and_score(
                    uids_by_dataset,
                    prepared_inner,
                    prepared_history,
                    normalization,
                    records_by_uid,
                    scale_by_uid,
                    clean_by_uid,
                )
                trainer_times.append(trainer_seconds)
                losses_by_program[program_id] = losses

                for uid in uids:
                    cell_id = cell_of_uid[uid]
                    raw_losses[(program_id, cell_id, uid, scenario, dose)].append(losses[uid])
                    fill_rates[(program_id, cell_id)].append(fills[uid])
                    repeat_rows.append(
                        {
                            "split_role": role.value,
                            "program_id": program_id,
                            "cell_id": cell_id,
                            "uid": uid,
                            "scenario": scenario,
                            "dose": dose,
                            "corruption_replicate": replicate,
                            "loss": losses[uid],
                        }
                    )

            # Per-(cell, program) losses for THIS (scenario, dose), pooled over replicates.
            # This is the evidence another role's transfer policy reads.
            grid_key = (scenario, dose)
            for program_id in selectable:
                for uid in uids:
                    cell_loss_by_grid[grid_key][(cell_of_uid[uid], program_id)].append(
                        losses_by_program[program_id][uid]
                    )

            # --- the two retrained oracles -------------------------------------------
            insample_cell_means = {
                (cell_of_uid_key, program_id): float(
                    np.mean(
                        [
                            losses_by_program[program_id][uid]
                            for uid in uids
                            if cell_of_uid[uid] == cell_of_uid_key
                        ]
                    )
                )
                for cell_of_uid_key in cells
                for program_id in selectable
            }
            insample_policy_by_cell, _ = _policy_from_cell_means(
                insample_cell_means, cells, selectable, best_fixed_program
            )
            policies: dict[str, dict[str, str]] = {
                ORACLE_INSAMPLE_RETRAINED: insample_policy_by_cell
            }
            if transfer_policy is not None:
                by_cell_transfer = transfer_policy.get(grid_key, {})
                resolved = {
                    cell: by_cell_transfer.get(cell, best_fixed_program) for cell in cells
                }
                uncovered_transfer_cells.update(
                    cell for cell in cells if cell not in by_cell_transfer
                )
                policies[ORACLE_TRANSFER_RETRAINED] = resolved

            for oracle_id, cell_policy in policies.items():
                uid_policy = {uid: cell_policy[cell_of_uid[uid]] for uid in uids}
                for cell, program in cell_policy.items():
                    oracle_picks[oracle_id][cell].append(program)
                    oracle_selections.setdefault(oracle_id, []).append(
                        {
                            "cell_id": cell,
                            "scenario": scenario,
                            "dose": dose,
                            "replicate": replicate,
                            "program_id": program,
                        }
                    )
                mixed_history, mixed_inner = _mixed_values(
                    uid_policy, prepared_history_by_program, prepared_inner_by_program
                )
                losses, fills, trainer_seconds = _fit_and_score(
                    uids_by_dataset,
                    mixed_inner,
                    mixed_history,
                    normalization,
                    records_by_uid,
                    scale_by_uid,
                    clean_by_uid,
                )
                trainer_times.append(trainer_seconds)
                for uid in uids:
                    cell_id = cell_of_uid[uid]
                    raw_losses[(oracle_id, cell_id, uid, scenario, dose)].append(losses[uid])
                    fill_rates[(oracle_id, cell_id)].append(fills[uid])
                    repeat_rows.append(
                        {
                            "split_role": role.value,
                            "program_id": oracle_id,
                            "cell_id": cell_id,
                            "uid": uid,
                            "scenario": scenario,
                            "dose": dose,
                            "corruption_replicate": replicate,
                            "loss": losses[uid],
                        }
                    )

    # Fold 1: average corruption replicates within uid x scenario x dose.
    folded: dict[tuple[str, str, str], dict[tuple[str, float], float]] = defaultdict(dict)
    for (program_id, cell_id, uid, scenario, dose), losses_list in raw_losses.items():
        if len(losses_list) != len(replicates_for(scenario)):
            raise RuntimeError(
                f"uid {uid} is missing a corruption replicate for {scenario}/{dose}"
            )
        folded[(program_id, cell_id, uid)][(scenario, dose)] = float(np.mean(losses_list))

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

    cell_means_by_grid: dict[tuple[str, float], dict[tuple[str, str], float]] = {
        grid_key: {key: float(np.mean(values)) for key, values in bucket.items()}
        for grid_key, bucket in cell_loss_by_grid.items()
    }
    diagnostics: dict[str, object] = {
        "training_unit": "(program, scenario, dose, replicate, dataset)",
        "n_datasets": len(uids_by_dataset),
        "series_per_dataset": {key: len(value) for key, value in uids_by_dataset.items()},
        "selectable_programs": list(selectable),
        "oracle_selections": {
            key: sorted(
                value, key=lambda row: (row["cell_id"], row["scenario"], row["dose"], row["replicate"])
            )
            for key, value in sorted(oracle_selections.items())
        },
        "oracle_picks_by_cell": {
            oracle_id: {cell: sorted(picks) for cell, picks in sorted(by_cell.items())}
            for oracle_id, by_cell in sorted(oracle_picks.items())
        },
        "transfer_cells_without_support_evidence": sorted(uncovered_transfer_cells),
    }
    return (
        rows,
        fill_rates,
        prepare_times,
        trainer_times,
        repeat_rows,
        diagnostics,
        cell_means_by_grid,
    )


def _selectable_rows(rows: list[ProgramLoss]) -> list[ProgramLoss]:
    """Keep only rows a floor/oracle may choose between.

    H_ref is the incumbent under test and the two retrained oracles are runner privileges.
    Leaving any of them in the candidate set would let `best_fixed` "select" a ceiling, or
    let an oracle select another oracle, and the resulting number would be meaningless.
    """
    excluded = set(RUNNER_EXECUTED) | {ORACLE_TRANSFER_RETRAINED, ORACLE_INSAMPLE_RETRAINED}
    return [row for row in rows if row.program_id not in excluded]


def _transfer_policy_from_support(
    support_cell_means: dict[tuple[str, float], dict[tuple[str, str], float]],
) -> dict[tuple[str, float], dict[str, str]]:
    """Best program per (cell, scenario, dose), chosen on Support-A and never on the query.

    This is the policy the Gate-bearing ceiling executes.  It is conditioned on both axes
    C1 names -- the pattern proxy (`regime` inside `cell_id`) and the degradation actually
    encountered (`scenario`, `dose`) -- and it is selected on a disjoint set of series, so
    the ceiling it produces is one a real method could aim at rather than a winner's-curse
    artifact of the query set.
    """
    policy: dict[tuple[str, float], dict[str, str]] = {}
    for grid_key, bucket in support_cell_means.items():
        by_cell: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for (cell, program), mean in bucket.items():
            by_cell[cell].append((mean, program))
        policy[grid_key] = {cell: min(scored)[1] for cell, scored in by_cell.items()}
    return policy


def run_dev_evaluation(root: Path | str, out: Path | str) -> dict[str, object]:
    """Evaluate the full public baseline pool on repeatable Dev-Query only."""

    data_root, output = Path(root), Path(out)
    pool = pool_manifest()
    records = read_registry_jsonl(output / "series_registry.jsonl")
    records_by_uid = {row.series_uid: row for row in records}
    split = SplitManifest.from_dict(json.loads((output / "split_manifest.json").read_text("utf-8")))
    selected_uids = {
        row.series_uid for row in split.assignments
        if row.role in {SplitRole.SUPPORT_A, SplitRole.DEV_QUERY}
    }
    selected_records = [records_by_uid[uid] for uid in sorted(selected_uids)]
    values = _load_values(selected_records, data_root / "clean_base")

    # Pass 1: Support-A. No transfer policy exists yet -- that is what this pass produces.
    (
        support,
        support_fill,
        support_prepare,
        support_train,
        support_repeats,
        support_diag,
        support_cell_means,
    ) = _evaluate_role(SplitRole.SUPPORT_A, list(split.assignments), records_by_uid, values)

    best = select_best_fixed(_selectable_rows(support))
    transfer_policy = _transfer_policy_from_support(support_cell_means)

    # Pass 2: Dev-Query, with the ceiling's policy already fixed on disjoint series.
    (
        dev,
        dev_fill,
        dev_prepare,
        dev_train,
        dev_repeats,
        dev_diag,
        _dev_cell_means,
    ) = _evaluate_role(
        SplitRole.DEV_QUERY,
        list(split.assignments),
        records_by_uid,
        values,
        transfer_policy=transfer_policy,
        best_fixed_program=best.program_id,
    )
    if not dev:
        raise RuntimeError("frozen split has no Dev-Query rows")

    dev_selectable = _selectable_rows(dev)
    best_rows = [row for row in dev if row.program_id == best.program_id]
    h_ref_rows = [row for row in dev if row.program_id == "h_ref"]
    retrained_transfer_rows = [
        row for row in dev if row.program_id == ORACLE_TRANSFER_RETRAINED
    ]
    retrained_insample_rows = [
        row for row in dev if row.program_id == ORACLE_INSAMPLE_RETRAINED
    ]
    if not retrained_transfer_rows:
        raise RuntimeError("the Gate-bearing retrained transfer oracle produced no rows")

    # The legacy cell-level oracles are kept for continuity with the v0/v0.1 reports and
    # are labelled for what they are: a program picked per cell, but scored under models
    # that were each trained on a corpus prepared with ONE program throughout. No model was
    # ever fitted to the mixed corpus those picks describe, so the number is not a value
    # any method could reach. It does not enter the Gate.
    insample_rows = oracle_insample(dev_selectable)
    transfer_rows, transfer_missing_cells = oracle_transfer_with_coverage(
        _selectable_rows(support), dev_selectable
    )
    h_ref_by_uid = {row.uid: row for row in h_ref_rows}
    insample_by_uid = {row.uid: row for row in retrained_insample_rows}

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
        ORACLE_TRANSFER_RETRAINED: fold_to_headline(retrained_transfer_rows),
        ORACLE_INSAMPLE_RETRAINED: fold_to_headline(retrained_insample_rows),
        "oracle_transfer_untrained_counterfactual": (
            fold_to_headline(transfer_rows) if transfer_rows else None
        ),
        "oracle_insample_untrained_counterfactual": fold_to_headline(insample_rows),
    }
    report["per_program_smase"] = {
        program: fold_to_headline([row for row in dev if row.program_id == program])
        for program in PROGRAM_IDS
    }
    report["oracle_semantics"] = {
        ORACLE_TRANSFER_RETRAINED: (
            "GATE-BEARING CEILING. The best program per (cell, scenario, dose) is chosen on "
            "Support-A; Dev-Query is then prepared with those choices, a model is TRAINED on "
            "the resulting mixed corpus (per dataset), and that model is scored. Same path a "
            "Method takes, so floor, ceiling, and method are one kind of measurement."
        ),
        ORACLE_INSAMPLE_RETRAINED: (
            "INFLATION ENVELOPE. Same retraining, but the policy is chosen on Dev-Query "
            "itself. Reports how much of any apparent ceiling is winner's curse."
        ),
        "oracle_transfer_untrained_counterfactual": (
            "DESCRIPTIVE ONLY, NOT A GATE INPUT. The v0/v0.1 oracle: a program picked per "
            "cell, but each cell's loss read off a model trained on a corpus prepared with "
            "ONE program throughout. No model was ever fitted to the mixed corpus these "
            "picks describe, so this is not a value any method could reach. Kept only so "
            "the v0.1 report remains comparable."
        ),
        "oracle_insample_untrained_counterfactual": (
            "DESCRIPTIVE ONLY, NOT A GATE INPUT. As above, selected on Dev-Query."
        ),
        "h_ref_and_oracles_are_not_selectable": (
            "The oracles choose only among pool programs. H_ref is the incumbent under "
            "test; letting the ceiling pick it would make headroom partly a statement "
            "about H_ref's quality instead of the pool's reachable space."
        ),
    }
    report["aggregation_note"] = (
        "'overall' follows the frozen ladder (cell series-equal -> regime dataset-macro "
        "-> mean over regimes). 'series_micro_descriptive' is the plain per-uid mean and "
        "is descriptive only -- it lets the biggest dataset dominate."
    )
    report["best_fixed_program"] = best.program_id
    report["program_pool"] = pool
    report["training_unit"] = {
        "unit": "(program, scenario, dose, replicate, dataset)",
        "support_a": support_diag,
        "dev_query": dev_diag,
        "spec_gap_note": (
            "The frozen spec fixes trainer internals but never fixed the training pool's "
            "scope. v0 sliced by dataset x regime (leaking the private regime tag); v0.1 "
            "pooled the whole role (coupling datasets through shared weights). v0.2 fixes "
            "the scope explicitly at dataset, which is public metadata."
        ),
    }
    report["oracle_transfer_missing_support_cells"] = list(transfer_missing_cells)
    report["closed_form_model_seed_semantics"] = "deterministic_not_applicable"
    report["h_ref_behaviour_audit"] = audit_h_ref_behaviour(raw_rows, h_ref_rows)
    # The Gate. Headroom is measured from the No-op floor to the RETRAINED transfer ceiling.
    picks_by_cell = dev_diag.get("oracle_picks_by_cell", {})
    report["headroom"] = dual_headroom(
        raw_rows,
        h_ref_rows,
        retrained_transfer_rows,
        selection_by_cell=picks_by_cell.get(ORACLE_TRANSFER_RETRAINED, {}),
    )
    report["headroom_envelope_insample_retrained"] = dual_headroom(
        raw_rows,
        h_ref_rows,
        retrained_insample_rows,
        selection_by_cell=picks_by_cell.get(ORACLE_INSAMPLE_RETRAINED, {}),
    )
    report["headroom_untrained_counterfactual_descriptive"] = dual_headroom(
        raw_rows, h_ref_rows, insample_rows
    )
    mechanism = mechanism_panel(dev_repeats)
    report["mechanism_diagnostics"] = mechanism

    # Detectability. Cells are resampled by overlap group, not by series: METR-LA's sensors
    # come in spatial blocks and are not independent draws. A cell whose mde_80 exceeds the
    # material threshold cannot support a saturation claim, and says so.
    cluster_of_uid = {
        row.series_uid: (row.overlap_group or row.series_uid) for row in split.assignments
    }
    raw_by_uid = {row.uid: row.loss for row in raw_rows}
    gate_by_uid = {row.uid: row.loss for row in retrained_transfer_rows}
    paired_by_cell: dict[str, dict[str, float]] = defaultdict(dict)
    for row in retrained_transfer_rows:
        paired_by_cell[row.cell_id][row.uid] = raw_by_uid[row.uid] - gate_by_uid[row.uid]
    scale_warned = [
        cell_id
        for cell_id, cell in report.get("cells", {}).items()
        if cell.get("scale_warning")
    ]
    report["power_panel_cells"] = power_panel(
        paired_by_cell,
        cluster_of_uid,
        scale_warning_keys=scale_warned,
    )
    report["power_panel_note"] = (
        "Effect = Raw minus the Gate-bearing retrained transfer oracle, paired per uid "
        "under CRN. A cell carrying diagnostic_unavailable contributes nothing to a "
        "saturation claim."
    )
    # Numbers only: bind_dev_report_to_manifest validates every value here as a positive
    # finite float. Provenance lives in its own key.
    report["timeout_calibration_seconds"] = {
        "prepare_p95_x2": 2.0 * float(np.quantile(support_prepare + dev_prepare, 0.95)),
        "trainer_p95_x2": 2.0 * float(np.quantile(support_train + dev_train, 0.95)),
    }
    report["timeout_calibration_provenance"] = {
        "measured_on": (
            "the v0.2 per-dataset training path with the widened pool. Neither the v0 "
            "per-cell numbers nor the v0.1 role-pooled numbers transfer: the training pool "
            "changed size in both directions and the pool went from four programs to nine "
            "plus two retrained oracles."
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
    # The unfolded measurements. Everything above is a fold of these, and a fold cannot be
    # inverted: without them, any later per-uid x scenario question -- why does H_ref hurt
    # on block 0.24? does the pool separate anything on the spike lane? -- would need
    # another full re-run of the arena to answer.
    with (output / "dev_repeat_losses.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in support_repeats + dev_repeats:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    write_text_lf(
        output / "dev_per_dose_report.json",
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
    )
    write_text_lf(
        output / "baseline_report.md",
        _baseline_report_markdown(report, best.program_id),
    )
    return report


def _baseline_report_markdown(report: dict[str, object], best_program: str) -> str:
    baselines = report["baseline_smase"]
    audit = report["h_ref_behaviour_audit"]
    headroom = report["headroom"]
    mechanism = report["mechanism_diagnostics"]
    per_program = report["per_program_smase"]

    lines = [
        f"# Benchmark {BENCHMARK_VERSION} Dev-Query baseline report",
        "",
        "Final-Query was not read. Every value below is repeatable Dev-Query sMASE.",
        "",
        "One closed-form model is trained per `(program, scenario, dose, replicate, "
        "dataset)`. The two `*_retrained` oracles pick a program per `(cell, scenario, "
        "dose)` and are then **trained on the corpus those picks produce** -- the same path "
        "a Method takes. The `*_untrained_counterfactual` rows are the v0/v0.1 oracle, kept "
        "for continuity and excluded from the Gate: they read each cell's loss off a model "
        "trained on a single-program corpus, so they describe a world no model was fitted "
        "to.",
        "",
        "## The three-point C1 comparison",
        "",
        "| role | baseline | overall (macro) | series-micro (descriptive) |",
        "| --- | --- | --- | --- |",
    ]
    ladder = (
        ("floor (no-op)", "raw"),
        ("floor (best single program)", "best_fixed"),
        ("incumbent", "h_ref"),
        ("CEILING -- gate", ORACLE_TRANSFER_RETRAINED),
        ("envelope (winner's curse)", ORACLE_INSAMPLE_RETRAINED),
        ("descriptive only", "oracle_transfer_untrained_counterfactual"),
        ("descriptive only", "oracle_insample_untrained_counterfactual"),
    )
    for role_label, name in ladder:
        fold = baselines.get(name)
        if fold is None:
            lines.append(f"| {role_label} | {name} | n/a | n/a |")
            continue
        lines.append(
            f"| {role_label} | `{name}` | {fold['overall']:.6f} "
            f"| {fold['series_micro_descriptive']:.6f} |"
        )

    lines += [
        "",
        "## Every program in the frozen pool",
        "",
        "| program | mechanism | overall (macro) |",
        "| --- | --- | --- |",
    ]
    for program in PROGRAM_IDS:
        fold = per_program.get(program)
        if fold is None:
            continue
        lines.append(
            f"| `{program}` | {mechanism_of(program)} | {fold['overall']:.6f} |"
        )

    lines += [
        "",
        "## Where the pool has no action at all",
        "",
        f"The pool cannot act on {mechanism['n_cells_where_pool_cannot_act']} "
        "dataset/scenario/dose cell(s) -- every program scores identically there. That is a "
        "capability gap in the operator library, not evidence the data has nothing to gain.",
        "",
    ]
    for cell in mechanism["cells_where_pool_cannot_act"]:
        lines.append(f"- `{cell}`")

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
        picks = row["oracle_selected_program"]
        rendered = ", ".join(sorted(set(picks))) if isinstance(picks, list) else str(picks)
        lines.append(
            f"| `{cell}` | {rendered}{flag} "
            f"| {row['gain_over_raw']:+.4f} | {row['gain_over_h_ref']:+.4f} "
            f"| {row['h_ref_self_harm']:+.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ORACLE_INSAMPLE_RETRAINED",
    "ORACLE_TRANSFER_RETRAINED",
    "PROGRAM_IDS",
    "aggregate_per_dose",
    "audit_h_ref_behaviour",
    "bind_dev_report_to_manifest",
    "canonical_evaluation_context",
    "dual_headroom",
    "fold_to_headline",
    "mechanism_panel",
    "oracle_transfer_with_coverage",
    "run_dev_evaluation",
]
