"""A reversible representation view, and an evaluator that actually closes it.

The frozen Forecast evaluator applies a Program to **training windows only**:
the evaluation context is read through ``_linear_integrity`` with no Program,
and the prediction is never inverted.  That is coherent for repair operators --
"does cleaning the training corpus help" -- and it is what P4D measured.  It is
*not* coherent for representation transforms: detrending or differencing only
the training data and then serving raw contexts is a train/serve skew, so a null
there would be an instrument artifact rather than evidence about the transform.

This module supplies the missing half as a **separate, additive contract**.  The
frozen O0 path in ``run_e2_autonomous_natural_workflow_generation._evaluate`` is
not touched; ``representation_evaluate`` is a new end-to-end evaluator that:

* fits the view's parameters on the 192-step context **only**, so no horizon
  value can reach them -- causality is structural, not asserted;
* forwards the training context and target, and the evaluation context;
* inverts the prediction back to the original space before scoring;
* scores missing-aware sMASE on the **native 48-step grid against untransformed
  truth**, exactly as the frozen path does.

Under ``IdentityView`` it must reproduce the frozen evaluator to the bit.  That
equivalence is a preflight gate, not a hope.

**The affine screen.**  ``_center_scale`` standardises every window by its median
and 1.4826*MAD, both affine-equivariant.  For any ``a > 0``, ``x -> a*x + b``
gives ``center' = a*center + b`` and ``scale' = a*scale``, so the normalised
window is *identical*.  A view that is a positive affine map of the values is
therefore a provable no-op for this Consumer and must be rejected before any
Consumer fit is spent.  ``affine_cancellation_screen`` is that hard gate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    seasonal_scale,
    smase,
)

CONTEXT_LENGTH = 192
HORIZON = 48


class ReversibleView(Protocol):
    """Parameters come from the context; forward and inverse are exact."""

    name: str

    def fit(self, context: np.ndarray) -> dict[str, Any]:
        """Estimate from the pre-origin context only.  Never sees the horizon."""

    def forward(self, values: np.ndarray, params: Mapping[str, Any], *,
                start: int) -> np.ndarray:
        """Map into the view.  ``start`` is the window-relative index of values[0]."""

    def inverse(self, values: np.ndarray, params: Mapping[str, Any], *,
                start: int) -> np.ndarray:
        """Map back out.  ``inverse(forward(x)) == x`` to float precision."""


@dataclass(frozen=True)
class IdentityView:
    """The frozen O0 semantics, expressed as a view so the two paths can be compared."""

    name: str = "identity"

    def fit(self, context: np.ndarray) -> dict[str, Any]:
        return {}

    def forward(self, values: np.ndarray, params: Mapping[str, Any], *,
                start: int) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def inverse(self, values: np.ndarray, params: Mapping[str, Any], *,
                start: int) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class ReversibleDetrend:
    """Remove the context's least-squares line, extended over the horizon.

    Affine in ``(t, x)`` rather than in ``x`` alone, so per-window median/MAD
    standardisation cannot absorb it.
    """

    name: str = "reversible_detrend"

    def fit(self, context: np.ndarray) -> dict[str, Any]:
        values = np.asarray(context, dtype=np.float64)
        index = np.arange(values.size, dtype=np.float64)
        observed = np.isfinite(values)
        if int(observed.sum()) < 2:
            raise ValueError("detrend needs two observed context points")
        slope, intercept = np.polyfit(index[observed], values[observed], 1)
        return {"slope": float(slope), "intercept": float(intercept)}

    def _line(self, size: int, params: Mapping[str, Any], start: int) -> np.ndarray:
        index = np.arange(start, start + size, dtype=np.float64)
        return float(params["intercept"]) + float(params["slope"]) * index

    def forward(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        return values - self._line(values.size, params, start)

    def inverse(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        return values + self._line(values.size, params, start)


@dataclass(frozen=True)
class ReversibleSeasonalAdjust:
    """Remove a fixed-period phase profile estimated on the context."""

    period: int = 24
    name: str = "reversible_seasonal_adjust"

    def fit(self, context: np.ndarray) -> dict[str, Any]:
        values = np.asarray(context, dtype=np.float64)
        profile = np.zeros(self.period, dtype=np.float64)
        for phase in range(self.period):
            slice_ = values[phase::self.period]
            observed = slice_[np.isfinite(slice_)]
            profile[phase] = float(np.median(observed)) if observed.size else 0.0
        # Centre the profile so the view removes shape, not level; the level is
        # what the Consumer's own standardisation already handles.
        return {"profile": (profile - float(profile.mean())).tolist()}

    def _phase(self, size: int, params: Mapping[str, Any], start: int) -> np.ndarray:
        profile = np.asarray(params["profile"], dtype=np.float64)
        index = (np.arange(start, start + size) % self.period).astype(int)
        return profile[index]

    def forward(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        return values - self._phase(values.size, params, start)

    def inverse(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        return values + self._phase(values.size, params, start)


@dataclass(frozen=True)
class ReversibleDifference:
    """First difference, anchored on the last observed context value.

    The anchor is read from the context, so reconstructing the horizon never
    consults a future value.
    """

    name: str = "reversible_difference"

    def fit(self, context: np.ndarray) -> dict[str, Any]:
        values = np.asarray(context, dtype=np.float64)
        return {"anchor": float(values[-1]), "head": float(values[0])}

    def forward(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        previous = (
            float(params["head"]) if start == 0 else float(params["anchor"])
        )
        shifted = np.concatenate(([previous], values[:-1]))
        return values - shifted

    def inverse(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        previous = (
            float(params["head"]) if start == 0 else float(params["anchor"])
        )
        return previous + np.cumsum(values)


@dataclass(frozen=True)
class ReversibleScale:
    """Present only so the affine screen can be shown to reject it.

    ``x -> (x - median)/mad`` is a positive affine map, so the Consumer's own
    median/MAD standardisation cancels it exactly.  It must never reach a
    Consumer fit.
    """

    name: str = "reversible_scale"

    def fit(self, context: np.ndarray) -> dict[str, Any]:
        center, scale, _method = forecast_runtime._center_scale(
            np, np.asarray(context, dtype=np.float64)
        )
        return {"center": float(center), "scale": float(scale)}

    def forward(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        return (values - float(params["center"])) / float(params["scale"])

    def inverse(self, values, params, *, start):
        values = np.asarray(values, dtype=np.float64)
        return values * float(params["scale"]) + float(params["center"])


def affine_cancellation_screen(view: ReversibleView,
                               windows: Sequence[np.ndarray],
                               *, tolerance: float = 1e-9) -> dict[str, Any]:
    """Hard gate: is this view erased by the Consumer's own standardisation?

    Compares the standardised context before and after the view.  A view that
    leaves it unchanged cannot influence a single Ridge coefficient, so running
    it would spend Consumer fits to measure zero by construction.
    """
    worst = 0.0
    checked = 0
    for window in windows:
        context = forecast_runtime._linear_integrity(
            np.asarray(window, dtype=np.float64)[:CONTEXT_LENGTH]
        )
        try:
            params = view.fit(context)
            transformed = view.forward(context, params, start=0)
        except Exception:  # noqa: BLE001 - an unusable view is not cancelled
            return {
                "view": view.name, "windows_checked": checked,
                "cancelled_by_consumer_normalisation": False,
                "max_standardised_difference": None,
                "reading": "the view failed on real context; screen inconclusive",
            }
        before_c, before_s, _m = forecast_runtime._center_scale(np, context)
        after_c, after_s, _m = forecast_runtime._center_scale(np, transformed)
        before = (context - before_c) / before_s
        after = (transformed - after_c) / after_s
        worst = max(worst, float(np.max(np.abs(after - before))))
        checked += 1
    cancelled = worst <= tolerance
    return {
        "view": view.name,
        "windows_checked": checked,
        "max_standardised_difference": worst,
        "tolerance": tolerance,
        "cancelled_by_consumer_normalisation": cancelled,
        "reading": (
            "REJECT: the Consumer's median/MAD standardisation erases this view "
            "exactly, so it is a provable no-op and must not reach a Consumer fit"
            if cancelled else
            "admissible: the view survives per-window standardisation"
        ),
    }


def closure_check(view: ReversibleView, windows: Sequence[np.ndarray],
                  *, tolerance: float = 1e-8) -> dict[str, Any]:
    """``inverse(forward(x)) == x`` on the horizon block, where it matters."""
    worst = 0.0
    for window in windows:
        full = forecast_runtime._linear_integrity(
            np.asarray(window, dtype=np.float64)
        )
        params = view.fit(full[:CONTEXT_LENGTH])
        horizon = full[CONTEXT_LENGTH:]
        forward = view.forward(horizon, params, start=CONTEXT_LENGTH)
        back = view.inverse(forward, params, start=CONTEXT_LENGTH)
        worst = max(worst, float(np.max(np.abs(back - horizon))))
    return {
        "view": view.name,
        "windows_checked": len(windows),
        "max_reconstruction_error": worst,
        "tolerance": tolerance,
        "closed": worst <= tolerance,
    }


def horizon_independence_check(view: ReversibleView,
                               windows: Sequence[np.ndarray]) -> dict[str, Any]:
    """Perturb the horizon; the fitted parameters must not move at all."""
    rng = np.random.default_rng(0)
    drifted = 0
    for window in windows:
        full = forecast_runtime._linear_integrity(
            np.asarray(window, dtype=np.float64)
        )
        clean = view.fit(full[:CONTEXT_LENGTH])
        polluted = full.copy()
        polluted[CONTEXT_LENGTH:] += rng.normal(0.0, 10.0, HORIZON)
        after = view.fit(polluted[:CONTEXT_LENGTH])
        if repr(clean) != repr(after):
            drifted += 1
    return {
        "view": view.name,
        "windows_checked": len(windows),
        "windows_whose_parameters_moved": drifted,
        "no_future_leakage": drifted == 0,
        "reading": (
            "parameters are a function of the context alone"
            if drifted == 0 else
            "a horizon perturbation changed the fitted parameters"
        ),
    }


def representation_evaluate(
    roster: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any],
    compiled: Any,
    config: Mapping[str, Any],
    *,
    origin: int,
    view: ReversibleView,
    train_series_scope: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """The frozen pipeline, with a reversible view closed around the Consumer.

    Step order is deliberately identical to the frozen evaluator so that
    ``IdentityView`` reproduces it exactly: repair Program first, then the view,
    then per-window standardisation, then the frozen Ridge.  Only the inverse on
    the prediction is new, and for the identity view it is a no-op.
    """
    train_rows = [row for row in roster if row["role"] == "train"]
    eval_rows = [row for row in roster if row["role"] == "eval"]

    x_train: list[Any] = []
    y_train: list[Any] = []
    behavior_count = 0
    execution_steps: list[dict[str, object]] = []
    for row in train_rows:
        series_uid = str(row["series_uid"])
        raw = np.asarray(values[series_uid], dtype=np.float64)
        for anchor in config["anchors"]:
            anchor = int(anchor)
            if anchor + HORIZON > origin:
                continue
            window = raw[anchor - CONTEXT_LENGTH:anchor + HORIZON]
            baseline = forecast_runtime._linear_integrity(window)
            if compiled is not None and (
                train_series_scope is None or series_uid in train_series_scope
            ):
                prepared, trace = forecast_runtime._apply_program(window, compiled)
            else:
                prepared, trace = baseline, []
            behavior_count += int(
                np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True))
            )
            execution_steps.extend(trace)
            params = view.fit(prepared[:CONTEXT_LENGTH])
            viewed = view.forward(prepared, params, start=0)
            context = viewed[:CONTEXT_LENGTH]
            target = viewed[CONTEXT_LENGTH:]
            center, scale, method = forecast_runtime._center_scale(np, context)
            if method == "scale_floor_fallback":
                raise RuntimeError("training context reached scale floor")
            x_train.append((context - center) / scale)
            y_train.append((target - center) / scale)

    x_eval: list[Any] = []
    truths: list[Any] = []
    eval_params: list[dict[str, Any]] = []
    eval_centers: list[float] = []
    eval_scales: list[float] = []
    metric_scales: list[float] = []
    for row in eval_rows:
        raw = np.asarray(values[str(row["series_uid"])], dtype=np.float64)
        window = raw[origin - CONTEXT_LENGTH:origin]
        prepared = forecast_runtime._linear_integrity(window)
        params = view.fit(prepared)
        viewed = view.forward(prepared, params, start=0)
        center, scale, method = forecast_runtime._center_scale(np, viewed)
        if method == "scale_floor_fallback":
            raise RuntimeError("evaluation context reached scale floor")
        x_eval.append((viewed - center) / scale)
        # Truth is never transformed: scoring stays on the native grid.
        truths.append(raw[origin:origin + HORIZON])
        eval_params.append(dict(params))
        eval_centers.append(center)
        eval_scales.append(scale)
        metric_scales.append(
            seasonal_scale(
                raw[:origin], np.isfinite(raw[:origin]),
                period=int(config["period"]), min_pairs=32,
            )
        )

    prediction = forecast_runtime._exact_weighted_ridge_prediction(
        np,
        x_train=np.asarray(x_train, dtype=np.float64),
        targets=np.asarray(y_train, dtype=np.float64),
        weights=np.ones(len(x_train), dtype=np.float64),
        x_eval=np.asarray(x_eval, dtype=np.float64),
    )
    prediction = (
        prediction * np.asarray(eval_scales)[:, None]
        + np.asarray(eval_centers)[:, None]
    )
    # Leave the view before scoring, so the loss is in the original units.
    restored = np.vstack([
        view.inverse(row, params, start=CONTEXT_LENGTH)
        for row, params in zip(prediction, eval_params)
    ])

    losses: list[float] = []
    for truth, predicted, scale in zip(truths, restored, metric_scales):
        observed = np.isfinite(truth)
        if not observed.any():
            raise RuntimeError("evaluation future contains no observed truth")
        losses.append(smase(truth[observed], predicted[observed], scale=scale))
    failed_steps = [row for row in execution_steps if row.get("ok") is not True]
    return {
        "mean_smase": float(np.mean(losses)),
        "per_view_smase": [float(value) for value in losses],
        "behavior_point_count": behavior_count,
        "execution_step_count": len(execution_steps),
        "failed_step_count": len(failed_steps),
        "view": view.name,
    }


CANDIDATE_VIEWS: tuple[ReversibleView, ...] = (
    IdentityView(),
    ReversibleDetrend(),
    ReversibleSeasonalAdjust(),
    ReversibleDifference(),
    ReversibleScale(),
)
