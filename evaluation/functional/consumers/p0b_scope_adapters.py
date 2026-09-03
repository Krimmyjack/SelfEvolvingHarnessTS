"""P0b's two missing task adapters, using the existing ScopeExecutor shape.

There is deliberately no new adapter base class, loader, registry, or path
handling here.  The runner owns data loading and label walls.  These callables
only bridge already-in-memory episode surfaces to the existing runtime shape::

    (roster, values, compiled, config, *, origin)
        -> {mean_smase, per_view_smase, behavior_point_count}

Unknown origins fail before an anomaly-label callback can be reached.  Natural
Final paths and outcomes are therefore outside this module's capabilities.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.runtime.executor import run_pipeline
from evaluation.functional.consumers.ad_scope_adapter import compiled_steps


def _reading(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate the three fields ScopeExecutor consumes."""
    mean = float(raw["mean_smase"])
    views = [float(value) for value in raw["per_view_smase"]]
    behavior = int(raw.get("behavior_point_count") or 0)
    if not np.isfinite(mean) or not views or not np.isfinite(views).all():
        raise RuntimeError("adapter returned an empty or non-finite reading")
    if behavior < 0:
        raise RuntimeError("behavior_point_count must be non-negative")
    return {
        "mean_smase": mean,
        "per_view_smase": views,
        "behavior_point_count": behavior,
    }


class ForecastScopeAdapter:
    """Budgeted fail-closed wrapper over the frozen pooled-Ridge evaluator."""

    def __init__(
        self,
        *,
        frozen_evaluate: Callable[..., Mapping[str, Any]],
        fit_budget: Any,
        phase_by_origin: Mapping[int, str],
    ) -> None:
        self._evaluate = frozen_evaluate
        self._budget = fit_budget
        self._phase_by_origin = {
            int(origin): str(phase) for origin, phase in phase_by_origin.items()
        }
        if not self._phase_by_origin:
            raise ValueError("phase_by_origin must not be empty")
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        roster: Sequence[Mapping[str, object]],
        values: Mapping[str, Any],
        compiled: Any,
        config: Mapping[str, object],
        *,
        origin: int,
    ) -> dict[str, Any]:
        token = int(origin)
        if token not in self._phase_by_origin:
            raise RuntimeError("forecast origin is not an open P0b surface: %s" % token)
        self._budget.spend(1)
        raw = self._evaluate(
            roster, values, compiled, config, origin=token,
        )
        reading = _reading(raw)
        self.calls.append({
            "phase": self._phase_by_origin[token],
            "origin": token,
            "logical_evaluations": 1,
            "raw_consumer_fits": 1,
        })
        return reading


class TrainingBlockScopeExecutor(ScopeExecutor):
    """Use each AD series' frozen training block as its verification window."""

    def __init__(
        self,
        *,
        rows: Mapping[str, Mapping[str, Any]],
        round_name: str,
        evaluate_fn: Callable[..., Mapping[str, Any]],
        max_modified_fraction: float = 0.35,
    ) -> None:
        if not rows:
            raise ValueError("rows must not be empty")
        roster = [{"series_uid": uid, "role": "train"} for uid in sorted(rows)]
        values = {
            uid: np.asarray(rows[uid]["values"], dtype=np.float64)
            for uid in rows
        }
        super().__init__(
            roster,
            values,
            {"anchors": []},
            evaluate_fn=evaluate_fn,
            max_modified_fraction=float(max_modified_fraction),
        )
        self._rows = dict(rows)
        self._round = str(round_name)

    def training_windows(self, origin: int):  # noqa: D401 - runtime signature
        windows = []
        for uid in sorted(self._rows):
            lo, hi = self._rows[uid]["windows"][self._round]["train"]
            block = np.asarray(self._rows[uid]["values"], dtype=np.float64)[
                int(lo):int(hi)
            ]
            windows.append((uid, int(lo), block))
        return windows


def _steps_key(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> tuple[Any, ...]:
    """Structured cache identity with parameters; no digest or persisted lock."""
    return tuple(
        (str(op), json.dumps(dict(params), sort_keys=True, separators=(",", ":")))
        for op, params in steps
    )


class WindowedIForestAdapter:
    """One IForest fit per AD series, scored on pre-opened Support-A/B only."""

    def __init__(
        self,
        *,
        consumer: Any,
        rows: Mapping[str, Mapping[str, Any]],
        round_name: str,
        event_reader: Callable[[str, int, int], Sequence[Sequence[int]]],
        fit_budget: Any,
        phase_by_origin: Mapping[int, str],
    ) -> None:
        if not rows:
            raise ValueError("rows must not be empty")
        self._consumer = consumer
        self._rows = dict(rows)
        self._round = str(round_name)
        self._event_reader = event_reader
        self._budget = fit_budget
        self._phase_by_origin = {
            int(origin): str(phase) for origin, phase in phase_by_origin.items()
        }
        if not self._phase_by_origin:
            raise ValueError("phase_by_origin must not be empty")
        allowed = {"support_a", "support_b"}
        unknown = set(self._phase_by_origin.values()) - allowed
        if unknown:
            raise ValueError("AD phases must be support_a/support_b: %s" % sorted(unknown))
        for uid, row in self._rows.items():
            values = np.asarray(row["values"], dtype=np.float64)
            if values.ndim != 1 or not values.size:
                raise ValueError("AD row %s must contain one non-empty series" % uid)
            windows = row["windows"][self._round]
            for phase in ("train", *sorted(set(self._phase_by_origin.values()))):
                lo, hi = (int(value) for value in windows[phase])
                if lo < 0 or hi <= lo or hi > values.size:
                    raise ValueError("invalid %s window for %s" % (phase, uid))
                if not np.isfinite(values[lo:hi]).all():
                    raise ValueError("non-finite %s window for %s" % (phase, uid))
        self._models: dict[tuple[Any, ...], tuple[Any, int]] = {}
        self.calls: list[dict[str, Any]] = []

    def _model(
        self,
        uid: str,
        values: np.ndarray,
        steps: Sequence[tuple[str, Mapping[str, Any]]],
    ) -> tuple[Any, int, bool]:
        lo, hi = (
            int(value)
            for value in self._rows[uid]["windows"][self._round]["train"]
        )
        key = (uid, lo, hi, _steps_key(steps))
        if key in self._models:
            model, behavior = self._models[key]
            return model, behavior, True
        raw = np.asarray(values[lo:hi], dtype=np.float64)
        prepared = raw
        if steps:
            result = run_pipeline(list(steps), raw, source="p0b_exposed_adapter_smoke")
            if not result.ok or result.artifact is None:
                raise RuntimeError("AD program failed: %s" % result.error)
            prepared = np.asarray(result.artifact, dtype=np.float64).ravel()
            if prepared.shape != raw.shape or not np.isfinite(prepared).all():
                raise RuntimeError("AD program changed shape or produced non-finite values")
        behavior = int(np.count_nonzero(
            ~np.isclose(prepared, raw, equal_nan=True)
        ))
        self._budget.spend(1)
        model = self._consumer.fit_series(prepared)
        self._models[key] = (model, behavior)
        return model, behavior, False

    def __call__(
        self,
        roster: Sequence[Mapping[str, object]],
        values: Mapping[str, Any],
        compiled: Any,
        config: Mapping[str, object],
        *,
        origin: int,
    ) -> dict[str, Any]:
        token = int(origin)
        if token not in self._phase_by_origin:
            raise RuntimeError("AD origin is not an open P0b surface: %s" % token)
        roster_uids = sorted(
            str(row["series_uid"]) for row in roster if row.get("role") == "train"
        )
        if roster_uids != sorted(self._rows) or sorted(values) != sorted(self._rows):
            raise RuntimeError("AD roster/values do not match the frozen adapter rows")

        phase = self._phase_by_origin[token]
        steps = compiled_steps(compiled)
        per_series: dict[str, Mapping[str, Any]] = {}
        behavior = 0
        raw_fits = 0
        cache_hits = 0
        for uid in sorted(self._rows):
            array = np.asarray(values[uid], dtype=np.float64)
            model, changed, cached = self._model(uid, array, steps)
            behavior += changed
            raw_fits += int(not cached)
            cache_hits += int(cached)
            lo, hi = (
                int(value)
                for value in self._rows[uid]["windows"][self._round][phase]
            )
            truth = self._event_reader(uid, lo, hi)
            per_series[uid] = self._consumer.score_series(
                model, array, (lo, hi), truth,
            )
        macro = self._consumer.macro_f1(per_series)
        if macro is None:
            raise RuntimeError("AD consumer produced no Event-F1 reading")
        reading = _reading({
            "mean_smase": -float(macro),
            "per_view_smase": [
                -float(per_series[uid]["f1"]) for uid in sorted(per_series)
            ],
            "behavior_point_count": behavior,
        })
        reading.update({
            "ad_macro_f1": float(macro),
            "ad_pooled_f1": self._consumer.pooled_f1(per_series),
            "ad_phase": phase,
        })
        self.calls.append({
            "phase": phase,
            "origin": token,
            "logical_evaluations": 1,
            "raw_consumer_fits": raw_fits,
            "model_cache_hits": cache_hits,
        })
        return reading


__all__ = [
    "ForecastScopeAdapter",
    "TrainingBlockScopeExecutor",
    "WindowedIForestAdapter",
]
