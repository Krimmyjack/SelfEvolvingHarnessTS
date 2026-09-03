"""Two pipelines, one scope: the served series is finally the thing being prepared.

Every reading in the main protocol so far came from an evaluator that applies a
Program to the training corpus and serves the raw context (``p4n`` demonstrates
it mechanically).  A scope over training rows cannot express "do not treat this
served series", because the served series was never treated.  This module adds
the missing structure.

    Raw pipeline      raw train      -> raw model      -> raw serve context
    Program pipeline  prepared train -> program model  -> prepared serve context
    Scope             selected series take the Program pipeline
                      everyone else takes the Raw pipeline

The second pipeline is what makes the fallback real.  Serving a raw context out
of a model fitted on prepared data is not ``raw``: the preparation still reaches
the series through the coefficients.  Only a separately fitted raw model gives a
series that the Harness declined to treat a prediction bit-identical to Static.
That costs a second Consumer fit, and the cost is returned so it can be billed.

Three surfaces, and only one of them may ever be touched:

* ``train_context`` + ``train_target`` -- prepared, as the frozen path already did;
* ``serve_context`` -- prepared **causally**, from ``raw[origin-192:origin]`` only;
* ``evaluation_truth`` -- always raw, never transformed, scored missing-aware on
  the native 48-step grid.

``serving_mode="train_only"`` reproduces the frozen semantics exactly, so P4D and
P4M remain reproducible from this module and the two lines can be compared.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from evaluation.main_protocol_p4 import representation_view as views
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    seasonal_scale,
    smase,
)

CONTEXT_LENGTH = views.CONTEXT_LENGTH
HORIZON = views.HORIZON

#: ``train_only`` keeps the frozen semantics; ``scoped`` prepares what is served.
SERVING_MODES = ("train_only", "scoped")


class ServingContextDegenerate(RuntimeError):
    """Preparing a served context flattened it; the Scope cannot be honoured."""


def _training_windows(roster: Sequence[Mapping[str, Any]],
                      values: Mapping[str, Any], config: Mapping[str, Any],
                      origin: int) -> list[np.ndarray]:
    windows = []
    for row in roster:
        if row["role"] != "train":
            continue
        raw = np.asarray(values[str(row["series_uid"])], dtype=np.float64)
        for anchor in config["anchors"]:
            anchor = int(anchor)
            if anchor + HORIZON > origin:
                continue
            windows.append(raw[anchor - CONTEXT_LENGTH:anchor + HORIZON])
    return windows


def _prepare(window: np.ndarray, compiled: Any) -> tuple[np.ndarray, int, list]:
    """One window through the Program, or through linear integrity alone."""
    baseline = forecast_runtime._linear_integrity(window)
    if compiled is None:
        return baseline, 0, []
    prepared, trace = forecast_runtime._apply_program(window, compiled)
    moved = int(np.count_nonzero(
        ~np.isclose(prepared, baseline, equal_nan=True)))
    return prepared, moved, list(trace)


def _design(windows: Sequence[np.ndarray], view: views.ReversibleView
            ) -> tuple[np.ndarray, np.ndarray]:
    x_train, y_train = [], []
    for prepared in windows:
        params = view.fit(prepared[:CONTEXT_LENGTH])
        viewed = view.forward(prepared, params, start=0)
        context, target = viewed[:CONTEXT_LENGTH], viewed[CONTEXT_LENGTH:]
        center, scale, method = forecast_runtime._center_scale(np, context)
        if method == "scale_floor_fallback":
            raise RuntimeError("training context reached scale floor")
        x_train.append((context - center) / scale)
        y_train.append((target - center) / scale)
    return (np.asarray(x_train, dtype=np.float64),
            np.asarray(y_train, dtype=np.float64))


def _serve(contexts: Sequence[np.ndarray], view: views.ReversibleView,
           x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """One Consumer fit, then predictions back in the original units."""
    x_eval, centers, scales, params_list = [], [], [], []
    for prepared in contexts:
        params = view.fit(prepared)
        viewed = view.forward(prepared, params, start=0)
        center, scale, method = forecast_runtime._center_scale(np, viewed)
        if method == "scale_floor_fallback":
            raise RuntimeError("evaluation context reached scale floor")
        x_eval.append((viewed - center) / scale)
        centers.append(center)
        scales.append(scale)
        params_list.append(params)
    prediction = forecast_runtime._exact_weighted_ridge_prediction(
        np,
        x_train=x_train, targets=y_train,
        weights=np.ones(x_train.shape[0], dtype=np.float64),
        x_eval=np.asarray(x_eval, dtype=np.float64),
    )
    prediction = (prediction * np.asarray(scales)[:, None]
                  + np.asarray(centers)[:, None])
    return np.vstack([
        view.inverse(row, params, start=CONTEXT_LENGTH)
        for row, params in zip(prediction, params_list)
    ])


def scoped_evaluate(
    roster: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any],
    compiled: Any,
    config: Mapping[str, Any],
    *,
    origin: int,
    scope: frozenset[str] | set[str] | None = None,
    view: views.ReversibleView | None = None,
    serving_mode: str = "scoped",
) -> dict[str, Any]:
    """Evaluate a Program under a serving-series scope.

    ``scope`` names the **served** series that take the Program pipeline.
    ``None`` means every served series takes it; an empty set means none does,
    which must reproduce Static exactly.
    """
    if serving_mode not in SERVING_MODES:
        raise ValueError("serving_mode must be one of %s" % (SERVING_MODES,))
    view = view or views.IdentityView()
    eval_rows = [row for row in roster if row["role"] == "eval"]
    eval_uids = [str(row["series_uid"]) for row in eval_rows]
    selected = (
        set(eval_uids) if scope is None else {str(uid) for uid in scope}
    )
    in_scope = np.array([uid in selected for uid in eval_uids], dtype=bool)

    raw_windows = _training_windows(roster, values, config, int(origin))
    behavior, steps_run = 0, []
    prepared_windows = []
    for window in raw_windows:
        prepared, moved, trace = _prepare(window, compiled)
        behavior += moved
        steps_run.extend(trace)
        prepared_windows.append(prepared)
    raw_only = [forecast_runtime._linear_integrity(w) for w in raw_windows]

    raw_contexts, prepared_contexts, truths, metric_scales = [], [], [], []
    degenerate: list[str] = []
    for uid in eval_uids:
        raw = np.asarray(values[uid], dtype=np.float64)
        window = raw[int(origin) - CONTEXT_LENGTH:int(origin)]
        raw_context = forecast_runtime._linear_integrity(window)
        raw_contexts.append(raw_context)
        if serving_mode == "train_only" or compiled is None:
            prepared_contexts.append(raw_context)
        else:
            # Causal by construction: only the pre-origin window is touched.
            served, _moved, _trace = _prepare(window, compiled)
            # Preparing a served context can flatten it, which the frozen path
            # never risked because it never prepared one.  Falling back to raw
            # here would make the Scope mean something other than what it
            # declared, so a degenerate series makes the (program, scope) pair
            # illegal and the caller is told exactly which series did it.
            if uid in selected:
                _c, _s, method = forecast_runtime._center_scale(np, served)
                if method == "scale_floor_fallback":
                    degenerate.append(uid)
            prepared_contexts.append(served)
        truths.append(raw[int(origin):int(origin) + HORIZON])
        metric_scales.append(
            seasonal_scale(raw[:int(origin)], np.isfinite(raw[:int(origin)]),
                           period=int(config["period"]), min_pairs=32)
        )

    if degenerate:
        raise ServingContextDegenerate(
            "preparing the served context flattened %d scoped series (%s); the "
            "(program, scope) pair is illegal here rather than silently "
            "falling back to raw"
            % (len(degenerate), ", ".join(degenerate[:6])))

    fits = 0
    raw_design = _design(raw_only, view)
    raw_prediction = _serve(raw_contexts, view, *raw_design)
    fits += 1
    if compiled is None or not in_scope.any():
        prediction = raw_prediction
        program_prediction = None
    else:
        program_design = _design(prepared_windows, view)
        program_prediction = _serve(prepared_contexts, view, *program_design)
        fits += 1
        prediction = np.where(
            in_scope[:, None], program_prediction, raw_prediction)

    losses, raw_losses = [], []
    for index, (truth, scale) in enumerate(zip(truths, metric_scales)):
        observed = np.isfinite(truth)
        if not observed.any():
            raise RuntimeError("evaluation future contains no observed truth")
        losses.append(
            smase(truth[observed], prediction[index][observed], scale=scale))
        raw_losses.append(
            smase(truth[observed], raw_prediction[index][observed], scale=scale))
    failed = [row for row in steps_run if row.get("ok") is not True]
    return {
        "mean_smase": float(np.mean(losses)),
        "per_view_smase": [float(value) for value in losses],
        "static_per_view_smase": [float(value) for value in raw_losses],
        "behavior_point_count": behavior,
        "execution_step_count": len(steps_run),
        "failed_step_count": len(failed),
        "serving_mode": serving_mode,
        "view": view.name,
        "scope_size": int(in_scope.sum()),
        "served_series": len(eval_uids),
        "series_in_scope": [
            uid for uid, flag in zip(eval_uids, in_scope) if flag
        ],
        "consumer_fits": fits,
        "program_pipeline_used": program_prediction is not None,
    }
