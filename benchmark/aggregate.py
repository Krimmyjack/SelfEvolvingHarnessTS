"""One-way repeat folding and series/dataset-equal benchmark aggregation."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from . import BOOTSTRAP_B, BOOTSTRAP_MASTER_SEED, CORRUPTION_REPLICATES, MODEL_SEEDS
from .corruption import CORRUPTION_GRID
from .metrics import gain

FOLD_AXES = ("model_seed", "corruption_replicate", "scenario_dose", "uid")


class AggregationContractError(ValueError):
    """Rows have not satisfied the frozen paired-repeat contract."""


@dataclass(frozen=True)
class LossRow:
    uid: str
    dataset_id: str
    regime: str
    method_id: str
    scenario: str
    dose: float
    corruption_replicate: int
    model_seed: int
    reference_loss: float
    method_loss: float

    def __post_init__(self) -> None:
        strings = (self.uid, self.dataset_id, self.regime, self.method_id, self.scenario)
        if any(not isinstance(item, str) or not item or item != item.strip() for item in strings):
            raise ValueError("loss row identifiers must be canonical non-empty strings")
        numeric = np.asarray([self.dose, self.reference_loss, self.method_loss], dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError("loss row numeric fields must be finite")

    @property
    def absolute_gain(self) -> float:
        return gain(self.reference_loss, self.method_loss)

    @property
    def coordinate(self) -> tuple[str, float, int, int]:
        return (
            self.scenario,
            float(self.dose),
            int(self.corruption_replicate),
            int(self.model_seed),
        )


@dataclass(frozen=True)
class ScenarioDoseGain:
    scenario: str
    dose: float
    gain: float


@dataclass(frozen=True)
class UidGain:
    uid: str
    dataset_id: str
    regime: str
    method_id: str
    gain: float
    per_scenario_dose: tuple[ScenarioDoseGain, ...]


@dataclass(frozen=True)
class CellGain:
    dataset_id: str
    regime: str
    method_id: str
    mean_gain: float
    n_uid: int


@dataclass(frozen=True)
class RegimeGain:
    regime: str
    method_id: str
    mean_gain: float
    n_datasets: int


@dataclass(frozen=True)
class AggregateReport:
    cells: tuple[CellGain, ...]
    regimes: tuple[RegimeGain, ...]


def _validate_common_crn(rows: Sequence[LossRow]) -> None:
    by_uid_method: dict[tuple[str, str, str], set[tuple[str, float, int, int]]] = defaultdict(set)
    seen: set[tuple[str, str, str, str, float, int, int]] = set()
    for row in rows:
        duplicate_key = (row.uid, row.method_id, *row.coordinate)
        if duplicate_key in seen:
            raise AggregationContractError("duplicate repeat coordinate")
        seen.add(duplicate_key)
        by_uid_method[(row.uid, row.dataset_id, row.method_id)].add(row.coordinate)
    by_uid: dict[tuple[str, str], list[set[tuple[str, float, int, int]]]] = defaultdict(list)
    for (uid, dataset_id, _), coordinates in by_uid_method.items():
        by_uid[(uid, dataset_id)].append(coordinates)
    for coordinate_sets in by_uid.values():
        if len(coordinate_sets) > 1 and any(item != coordinate_sets[0] for item in coordinate_sets[1:]):
            raise AggregationContractError("all methods must use identical model seeds and CRN coordinates")


def collapse_uid_gains(
    rows: Sequence[LossRow],
    *,
    expected_model_seeds: Sequence[int] = MODEL_SEEDS,
    expected_replicates: Sequence[int] = CORRUPTION_REPLICATES,
    expected_scenario_doses: Sequence[tuple[str, float]] = CORRUPTION_GRID,
) -> list[UidGain]:
    """Fold model seed, replicate, and frozen scenario/dose into one uid row."""
    rows = list(rows)
    if not rows:
        raise AggregationContractError("cannot aggregate empty loss rows")
    _validate_common_crn(rows)
    expected_seeds = tuple(int(seed) for seed in expected_model_seeds)
    expected_reps = tuple(int(rep) for rep in expected_replicates)
    expected_pairs = tuple((str(scenario), float(dose)) for scenario, dose in expected_scenario_doses)
    groups: dict[tuple[str, str, str, str], list[LossRow]] = defaultdict(list)
    for row in rows:
        groups[(row.uid, row.dataset_id, row.regime, row.method_id)].append(row)
    result: list[UidGain] = []
    for (uid, dataset_id, regime, method_id), group in sorted(groups.items()):
        present_pairs = {(row.scenario, float(row.dose)) for row in group}
        if present_pairs != set(expected_pairs):
            raise AggregationContractError("scenario/dose axis is incomplete or unexpected")
        pair_values: list[ScenarioDoseGain] = []
        for scenario, dose in expected_pairs:
            pair_rows = [
                row for row in group
                if row.scenario == scenario and float(row.dose) == dose
            ]
            present_reps = {row.corruption_replicate for row in pair_rows}
            if present_reps != set(expected_reps):
                raise AggregationContractError("corruption replicates are incomplete or unexpected")
            replicate_values: list[float] = []
            for replicate in expected_reps:
                repeat_rows = [row for row in pair_rows if row.corruption_replicate == replicate]
                seeds = [row.model_seed for row in repeat_rows]
                if len(seeds) != len(set(seeds)) or set(seeds) != set(expected_seeds):
                    raise AggregationContractError("model seeds are incomplete or unexpected")
                by_seed = {row.model_seed: row.absolute_gain for row in repeat_rows}
                replicate_values.append(float(np.mean([by_seed[seed] for seed in expected_seeds])))
            pair_values.append(
                ScenarioDoseGain(scenario, dose, float(np.mean(replicate_values)))
            )
        result.append(
            UidGain(
                uid=uid,
                dataset_id=dataset_id,
                regime=regime,
                method_id=method_id,
                gain=float(np.mean([item.gain for item in pair_values])),
                per_scenario_dose=tuple(pair_values),
            )
        )
    return result


def aggregate_cells(rows: Sequence[UidGain]) -> AggregateReport:
    """Equal-weight uid within cells, then macro-average datasets within regime."""
    rows = list(rows)
    if not rows:
        raise AggregationContractError("cannot aggregate empty uid rows")
    seen: set[tuple[str, str]] = set()
    cells_raw: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        identity = (row.uid, row.method_id)
        if identity in seen:
            raise AggregationContractError("uid rows must be unique per method")
        seen.add(identity)
        if not np.isfinite(row.gain):
            raise AggregationContractError("uid gains must be finite")
        cells_raw[(row.dataset_id, row.regime, row.method_id)].append(row.gain)
    cells = tuple(
        CellGain(dataset, regime, method, float(np.mean(values)), len(values))
        for (dataset, regime, method), values in sorted(cells_raw.items())
    )
    regime_raw: dict[tuple[str, str], list[float]] = defaultdict(list)
    for cell in cells:
        regime_raw[(cell.regime, cell.method_id)].append(cell.mean_gain)
    regimes = tuple(
        RegimeGain(regime, method, float(np.mean(values)), len(values))
        for (regime, method), values in sorted(regime_raw.items())
    )
    return AggregateReport(cells=cells, regimes=regimes)


def bootstrap_subseed(master_seed: int, *coordinates: str) -> int:
    payload = json.dumps(
        [int(master_seed), *[str(item) for item in coordinates]],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def bootstrap_ci90(
    uid_gain: Mapping[str, float] | Sequence[tuple[str, float]],
    *,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_MASTER_SEED,
) -> tuple[float, float]:
    if isinstance(uid_gain, Mapping):
        items = list(uid_gain.items())
    else:
        items = list(uid_gain)
    uids = [str(uid) for uid, _ in items]
    if not items or len(uids) != len(set(uids)):
        raise AggregationContractError("bootstrap requires exactly one row per uid")
    if not isinstance(b, int) or isinstance(b, bool) or b < 1:
        raise ValueError("bootstrap B must be a positive integer")
    values = np.asarray([value for _, value in items], dtype=np.float64)
    if not np.isfinite(values).all():
        raise AggregationContractError("bootstrap uid gains must be finite")
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(values), size=(b, len(values)))
    means = values[draw].mean(axis=1)
    low, high = np.quantile(means, [0.05, 0.95], method="linear")
    return float(low), float(high)


__all__ = [
    "AggregateReport",
    "AggregationContractError",
    "CellGain",
    "FOLD_AXES",
    "LossRow",
    "RegimeGain",
    "ScenarioDoseGain",
    "UidGain",
    "aggregate_cells",
    "bootstrap_ci90",
    "bootstrap_subseed",
    "collapse_uid_gains",
]
