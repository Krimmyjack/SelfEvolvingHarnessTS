"""DEV-TRAIN-1 numeric kernel: heterogeneous train assignment → one pooled Ridge.

Fit and predict are separate.  Every legal training window of every train
series is stacked into **one** design matrix and one Ridge is solved.  Rows
are never taken from per-program models, and there is no per-channel average.

Identity is ``None`` only when that series uid is present as a key in the
assignment mapping.  A missing key is an error, not identity.  A failed
program is an error, not identity.

Training designs
----------------
``whole_window``
    W receives the raw 240-point window ``[X, y]`` and the design uses the
    split ``(Xp, yp)``.  All-identity reproduces the historical pooled
    ``_evaluate_assignment`` / v6 ``_evaluate`` path.
``input_only``
    W receives **only** the 192-point X.  The target is the common baseline
    ``y0 = B([X, y])[192:]``.  This is the true no-y-access contrast.
    ``B(X) = linear_integrity(X)`` may differ from ``B([X, y])[:192]`` when
    missing values sit near the X/y cut; they are not forced equal.
``recombination``
    Compute ``B([X, y]) = (X0, y0)`` and ``W([X, y]) = (Xp, yp)``, then pick
    one of ``X0_y0`` / ``Xp_y0`` / ``X0_yp`` / ``Xp_yp``.  W **does** see y;
    this is not no-y-access.  ``Xp_yp`` matches ``whole_window`` numerically;
    ``Xp_y0`` does not match ``input_only``.

The pooled solve is Ridge(alpha=1) with an unpenalized intercept on
per-window median/MAD-standardized rows (same geometry as
``_exact_weighted_ridge_prediction`` with unit weights):

    Z = [X_norm | 1],  coef = solve(Z.T @ Z + diag([1]*192 + [0]), Z.T @ Y)

Predict prepares each eval series' pre-origin 192-point input, standardizes
it the same way, and does ``Z_eval @ coef`` then inverts the eval window's
center/scale.  Predict never slices ``raw[origin:]``.  Evaluation truth is
read only by ``score_predictions``.

Runner wiring::

    design = build_training_design(roster, values, train_w, config, origin=origin)
    model = fit_assignment(design)
    pred = predict_assignment(model, roster, values, eval_v, config, origin=origin)
    scores = score_predictions(pred, roster, values, config, origin=origin)
    payload = model.to_dict()          # JSON-compatible
    model = FrozenRidge.from_dict(payload)
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.functional import (
    run_e2_autonomous_natural_workflow_generation as forecast_runtime,
)
from SelfEvolvingHarnessTS.contracts.candidate import Candidate, CandidateKind
from SelfEvolvingHarnessTS.contracts.program import Program
from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
    seasonal_scale,
    smase,
)
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import CompiledWorkflow

CONTEXT_LENGTH = int(forecast_runtime.CONTEXT_LENGTH)
HORIZON = int(forecast_runtime.HORIZON)
RIDGE_ALPHA = 1.0

TRAINING_DESIGNS = ("whole_window", "input_only", "recombination")
RECOMBINATIONS = ("X0_y0", "Xp_y0", "X0_yp", "Xp_yp")

CONSUMER_STRUCTURE = {
    "family": "linear_ridge",
    "shared_across_series": True,
    "input_length": CONTEXT_LENGTH,
    "horizon": HORIZON,
    "regularization": "ridge alpha=1, unpenalized intercept",
    "normalization": "per-window center/scale",
    "training_loss": "squared error on normalized windows",
    "metric": "sMASE",
}

_NORMALIZATION = (
    "per-window median/MAD center-scale on the 192-point context; "
    "applied to context and target before the pooled solve; "
    "not an inside-Ridge normalize flag"
)


class MissingAssignmentError(ValueError):
    """A train or eval uid is missing from the assignment, or extras are present."""

    def __init__(self, *, role: str, missing: Sequence[str], extra: Sequence[str]):
        self.role = str(role)
        self.missing = [str(uid) for uid in missing]
        self.extra = [str(uid) for uid in extra]
        super().__init__(
            "%s assignment missing=%s extra=%s (missing keys are not identity)"
            % (self.role, self.missing, self.extra)
        )


class PreparationFailed(RuntimeError):
    """Program execution, shape, or window geometry failed.  Not identity."""


class DegenerateContext(RuntimeError):
    """A prepared context hit the scale floor.  Not a raw fallback."""


def compile_steps(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> CompiledWorkflow:
    """Compile ``((op, params), ...)`` the same way ScopeExecutor builds a candidate."""
    program = Program.from_steps(list(steps), source="dev_train1")
    candidate = Candidate.program_candidate(
        "dev_train1", program, source="dev_train1"
    )
    return CompiledWorkflow(candidate, (), tuple(candidate.program.steps))


def _train_rows(roster: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in roster if str(row["role"]) == "train"]


def _eval_rows(roster: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in roster if str(row["role"]) == "eval"]


def _legal_anchors(config: Mapping[str, Any], origin: int) -> list[int]:
    origin = int(origin)
    return [
        int(anchor)
        for anchor in config["anchors"]
        if int(anchor) + HORIZON <= origin
    ]


def _require_assignment(
    rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, Any],
    *,
    role: str,
) -> None:
    if not isinstance(assignment, Mapping):
        raise MissingAssignmentError(role=role, missing=["<assignment is not a mapping>"], extra=[])
    needed = [str(row["series_uid"]) for row in rows]
    needed_set = set(needed)
    if len(needed) != len(needed_set):
        raise PreparationFailed("duplicate %s UID in roster" % role)
    got = {str(key) for key in assignment}
    missing = [uid for uid in needed if uid not in assignment]
    extra = sorted(got - needed_set)
    if missing or extra:
        raise MissingAssignmentError(role=role, missing=missing, extra=extra)


def _as_compiled(value: Any) -> CompiledWorkflow | None:
    """``None`` is identity.  Anything else must be executable, not a silent fallback."""
    if value is None:
        return None
    if isinstance(value, CompiledWorkflow):
        return value
    if isinstance(value, Candidate) and value.kind is CandidateKind.IDENTITY:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        steps: list[tuple[str, dict[str, Any]]] = []
        for item in value:
            if not (isinstance(item, Sequence) and len(item) == 2 and isinstance(item[0], str)):
                raise PreparationFailed(
                    "assignment value is not compiled or explicit identity"
                )
            op, params = item[0], item[1]
            steps.append((op, dict(params or {})))
        if not steps:
            raise PreparationFailed(
                "empty steps are not explicit identity; use None with the uid present"
            )
        return compile_steps(steps)
    raise PreparationFailed("assignment value is not compiled or explicit identity")


def _series(values: Mapping[str, Any], uid: str) -> np.ndarray:
    raw = np.asarray(values[uid], dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0:
        raise PreparationFailed("series %s is not a non-empty 1-d array" % uid)
    return raw


def _apply(raw: np.ndarray, compiled: CompiledWorkflow | None) -> tuple[np.ndarray, list]:
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise PreparationFailed("program input must be a non-empty 1-d array")
    try:
        prepared, trace = forecast_runtime._apply_program(array, compiled)
    except Exception as exc:
        raise PreparationFailed(str(exc) or "program execution failed") from exc
    prepared = np.asarray(prepared, dtype=np.float64).ravel()
    if prepared.shape != array.shape:
        raise PreparationFailed("program changed series shape")
    if not np.isfinite(prepared).all():
        raise PreparationFailed("program produced non-finite values after integrity")
    return prepared, list(trace)


def _normalize_pair(
    context: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    center, scale, method = forecast_runtime._center_scale(np, context)
    if method == "scale_floor_fallback":
        raise DegenerateContext("training context reached scale floor")
    return (
        (context - center) / scale,
        (target - center) / scale,
        float(center),
        float(scale),
        str(method),
    )


def _train_window(raw: np.ndarray, anchor: int) -> np.ndarray:
    start = int(anchor) - CONTEXT_LENGTH
    stop = int(anchor) + HORIZON
    if start < 0 or stop > raw.size:
        raise PreparationFailed(
            "training window [%d:%d] is out of range for length %d" % (start, stop, raw.size)
        )
    window = np.asarray(raw[start:stop], dtype=np.float64).ravel()
    if window.shape != (CONTEXT_LENGTH + HORIZON,):
        raise PreparationFailed("training window is not length %d" % (CONTEXT_LENGTH + HORIZON))
    return window


def _eval_window(raw: np.ndarray, origin: int) -> np.ndarray:
    origin = int(origin)
    start = origin - CONTEXT_LENGTH
    if start < 0 or origin > raw.size:
        raise PreparationFailed(
            "eval context [%d:%d] is out of range for length %d" % (start, origin, raw.size)
        )
    window = np.asarray(raw[start:origin], dtype=np.float64).ravel()
    if window.shape != (CONTEXT_LENGTH,):
        raise PreparationFailed("eval context is not length %d" % CONTEXT_LENGTH)
    return window


def _split_recombination(label: str) -> tuple[str, str]:
    for x_name in ("Xp", "X0"):
        prefix = x_name + "_"
        if label.startswith(prefix):
            return x_name, label[len(prefix):]
    raise ValueError("unknown recombination %r" % (label,))


def _prepare_train_window(
    window: np.ndarray,
    compiled: CompiledWorkflow | None,
    *,
    training_design: str,
    recombination: str | None,
) -> dict[str, Any]:
    baseline = forecast_runtime._linear_integrity(window)
    x0 = np.asarray(baseline[:CONTEXT_LENGTH], dtype=np.float64).copy()
    y0 = np.asarray(baseline[CONTEXT_LENGTH:], dtype=np.float64).copy()
    # B(X) can be undefined even when B([X,y]) is valid.  It is a
    # diagnostic in the full-window arm, not an additional admission rule.
    bx = None
    if np.count_nonzero(np.isfinite(window[:CONTEXT_LENGTH])) >= 2:
        bx = np.asarray(
            forecast_runtime._linear_integrity(window[:CONTEXT_LENGTH]),
            dtype=np.float64,
        )

    if training_design == "input_only":
        if bx is None:
            raise PreparationFailed("input-only integrity requires two observed X values")
        xp, trace = _apply(window[:CONTEXT_LENGTH], compiled)
        yp = y0.copy()
        moved = int(np.count_nonzero(~np.isclose(xp, bx, equal_nan=True)))
        y_access = False
    else:
        prepared, trace = _apply(window, compiled)
        xp = np.asarray(prepared[:CONTEXT_LENGTH], dtype=np.float64).copy()
        yp = np.asarray(prepared[CONTEXT_LENGTH:], dtype=np.float64).copy()
        moved = int(np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True)))
        y_access = True
        if training_design == "recombination":
            x_name, y_name = _split_recombination(str(recombination))
            xp = {"X0": x0, "Xp": xp}[x_name].copy()
            yp = {"y0": y0, "yp": yp}[y_name].copy()
        elif training_design != "whole_window":
            raise ValueError("unknown training_design %r" % (training_design,))

    source_moved = moved
    if training_design == "recombination":
        # Count the retained design, not discarded components of W's output.
        moved = int(np.count_nonzero(~np.isclose(xp, x0, equal_nan=True)))
        moved += int(np.count_nonzero(~np.isclose(yp, y0, equal_nan=True)))
    x_norm, y_norm, center, scale, method = _normalize_pair(xp, yp)
    return {
        "context": xp,
        "target": yp,
        "x_norm": x_norm,
        "y_norm": y_norm,
        "baseline_full_context": x0,
        "baseline_full_target": y0,
        "input_baseline_context": bx,
        "moved_points": moved,
        "source_workflow_moved_points": source_moved,
        "center": center,
        "scale": scale,
        "scale_method": method,
        "y_access": y_access,
        "trace": trace,
    }


@dataclass
class FrozenRidge:
    """Saved pooled Ridge coefficients.  Freeze is this array, not an object id."""

    coefficients: Any
    n_train_rows: int
    n_features: int
    n_targets: int
    ridge_alpha: float
    training_design: str
    recombination: str | None
    origin: int
    consumer_fits: int
    intercept_unpenalized: bool = True
    normalization: str = _NORMALIZATION

    def to_dict(self) -> dict[str, Any]:
        coef = np.asarray(self.coefficients, dtype=np.float64)
        return {
            "coefficients": coef.tolist(),
            "coefficient_shape": [int(coef.shape[0]), int(coef.shape[1])],
            "n_train_rows": int(self.n_train_rows),
            "n_features": int(self.n_features),
            "n_targets": int(self.n_targets),
            "ridge_alpha": float(self.ridge_alpha),
            "intercept_unpenalized": bool(self.intercept_unpenalized),
            "normalization": str(self.normalization),
            "training_design": str(self.training_design),
            "recombination": None if self.recombination is None else str(self.recombination),
            "origin": int(self.origin),
            "consumer_fits": int(self.consumer_fits),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FrozenRidge":
        shape = tuple(int(v) for v in payload["coefficient_shape"])
        coef = np.asarray(payload["coefficients"], dtype=np.float64)
        if coef.shape != shape:
            raise ValueError("coefficient_shape does not match coefficients")
        if shape != (CONTEXT_LENGTH + 1, HORIZON):
            raise ValueError("unexpected coefficient shape %s" % (shape,))
        if int(payload["n_features"]) != CONTEXT_LENGTH or int(payload["n_targets"]) != HORIZON:
            raise ValueError("frozen n_features/n_targets do not match 192/48")
        if float(payload["ridge_alpha"]) != RIDGE_ALPHA:
            raise ValueError("frozen ridge_alpha is not 1.0")
        return cls(
            coefficients=coef,
            n_train_rows=int(payload["n_train_rows"]),
            n_features=int(payload["n_features"]),
            n_targets=int(payload["n_targets"]),
            ridge_alpha=float(payload["ridge_alpha"]),
            training_design=str(payload["training_design"]),
            recombination=(
                None if payload.get("recombination") is None else str(payload["recombination"])
            ),
            origin=int(payload["origin"]),
            consumer_fits=int(payload.get("consumer_fits", 1)),
            intercept_unpenalized=bool(payload.get("intercept_unpenalized", True)),
            normalization=str(payload.get("normalization", _NORMALIZATION)),
        )


def build_training_design(
    roster: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any],
    train_assignment: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    origin: int,
    training_design: str = "whole_window",
    recombination: str | None = None,
) -> dict[str, Any]:
    """Assemble the pooled training matrix for one origin.  Does not fit.

    Parameters
    ----------
    roster:
        Rows with ``series_uid`` and ``role`` (``train`` / ``eval``).  Only
        ``role=="train"`` rows enter the design.  Each train uid contributes
        every legal anchor in ``config["anchors"]`` with ``anchor + 48 <= origin``.
    values:
        ``{uid: 1-d float array}``.  This function does not load data.
    train_assignment:
        ``{train_uid: compiled_or_None}``.  Every train uid must be a key.
        ``None`` (key present) is identity: linear integrity only.  A missing
        key raises ``MissingAssignmentError``.  ``compiled`` is a
        ``CompiledWorkflow`` or a non-empty ``((op, params), ...)`` step tuple.
    config:
        Must contain ``anchors``.  ``period`` is not used here.
    origin:
        Exclusive end of visible history.  Training targets never extend past
        this index.
    training_design:
        ``whole_window``, ``input_only``, or ``recombination``.
    recombination:
        Required for ``recombination``: ``X0_y0``, ``Xp_y0``, ``X0_yp``, or
        ``Xp_yp``.  Must be ``None`` for the other designs.

    Returns
    -------
    dict
        ``x_train`` / ``y_train`` (normalized float64), ``windows`` (roster
        then anchor order), ``behavior_point_count``, ``y_access``,
        ``training_design``, ``recombination``, ``origin``.
    """
    if training_design not in TRAINING_DESIGNS:
        raise ValueError("training_design must be one of %s" % (TRAINING_DESIGNS,))
    if training_design == "recombination":
        if recombination not in RECOMBINATIONS:
            raise ValueError("recombination must be one of %s" % (RECOMBINATIONS,))
    elif recombination is not None:
        raise ValueError("recombination is only used with training_design='recombination'")

    origin = int(origin)
    train_rows = _train_rows(roster)
    _require_assignment(train_rows, train_assignment, role="train")
    anchors = _legal_anchors(config, origin)

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    windows: list[dict[str, Any]] = []
    behavior = 0
    y_access = training_design != "input_only"
    for row in train_rows:
        uid = str(row["series_uid"])
        compiled = _as_compiled(train_assignment[uid])
        raw = _series(values, uid)
        for anchor in anchors:
            prepared = _prepare_train_window(
                _train_window(raw, anchor),
                compiled,
                training_design=training_design,
                recombination=recombination,
            )
            record = {
                "series_uid": uid,
                "anchor": int(anchor),
                "context": prepared["context"],
                "target": prepared["target"],
                "baseline_full_context": prepared["baseline_full_context"],
                "baseline_full_target": prepared["baseline_full_target"],
                "input_baseline_context": prepared["input_baseline_context"],
                "moved_points": prepared["moved_points"],
                "source_workflow_moved_points": prepared["source_workflow_moved_points"],
                "center": prepared["center"],
                "scale": prepared["scale"],
                "scale_method": prepared["scale_method"],
                "y_access": prepared["y_access"],
            }
            windows.append(record)
            x_rows.append(prepared["x_norm"])
            y_rows.append(prepared["y_norm"])
            behavior += int(prepared["moved_points"])

    x_train = (
        np.asarray(x_rows, dtype=np.float64)
        if x_rows
        else np.zeros((0, CONTEXT_LENGTH), dtype=np.float64)
    )
    y_train = (
        np.asarray(y_rows, dtype=np.float64)
        if y_rows
        else np.zeros((0, HORIZON), dtype=np.float64)
    )
    return {
        "x_train": x_train,
        "y_train": y_train,
        "windows": windows,
        "behavior_point_count": int(behavior),
        "y_access": bool(y_access),
        "training_design": training_design,
        "recombination": recombination,
        "origin": origin,
        "n_windows": int(x_train.shape[0]),
        "n_train_series": len(train_rows),
        "anchors": list(anchors),
    }


def fit_assignment(design: Mapping[str, Any]) -> FrozenRidge:
    """Fit one pooled Ridge on a design from ``build_training_design``.

    One Consumer fit.  Does not predict and does not read eval series.
    Does not fit a model per program.
    """
    x = np.asarray(design["x_train"], dtype=np.float64)
    y = np.asarray(design["y_train"], dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[0] != y.shape[0]
        or x.shape[1] != CONTEXT_LENGTH
        or y.shape[1] != HORIZON
        or x.shape[0] == 0
    ):
        raise ValueError("invalid pooled training design geometry")
    n_rows, n_features = x.shape
    z = np.column_stack((x, np.ones(n_rows, dtype=np.float64)))
    penalty = np.diag(np.array([RIDGE_ALPHA] * n_features + [0.0], dtype=np.float64))
    coefficients = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("pooled Ridge coefficients are non-finite")
    return FrozenRidge(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        n_train_rows=int(n_rows),
        n_features=int(n_features),
        n_targets=int(y.shape[1]),
        ridge_alpha=float(RIDGE_ALPHA),
        training_design=str(design["training_design"]),
        recombination=(
            None if design.get("recombination") is None else str(design["recombination"])
        ),
        origin=int(design["origin"]),
        consumer_fits=1,
    )


def predict_assignment(
    model: FrozenRidge,
    roster: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any],
    eval_assignment: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    origin: int,
) -> dict[str, Any]:
    """Prepare each eval series' pre-origin 192-point input and predict 48 steps.

    Parameters
    ----------
    model:
        Frozen coefficients from ``fit_assignment``.  Not refit.
    roster / values:
        Same contract as ``build_training_design``.  Only ``role=="eval"`` rows
        are predicted, in roster order.
    eval_assignment:
        ``{eval_uid: compiled_or_None}``.  Every eval uid must be a key.
        ``None`` (key present) is identity.  A missing key is an error.
    config:
        Unused for the numeric path (accepted so the runner can pass one config).
    origin:
        Eval context is ``raw[origin-192:origin]`` only.  ``raw[origin:]`` is
        not read.

    Returns
    -------
    dict
        ``eval_uids``, ``predictions`` (original units, shape ``(n_eval, 48)``),
        ``centers``, ``scales``, ``behavior_point_count``, ``consumer_fits=0``,
        ``origin``.  Does not include evaluation truth.
    """
    del config
    origin = int(origin)
    eval_rows = _eval_rows(roster)
    _require_assignment(eval_rows, eval_assignment, role="eval")
    coef = np.asarray(model.coefficients, dtype=np.float64)
    if coef.shape != (CONTEXT_LENGTH + 1, HORIZON):
        raise ValueError("frozen coefficients have unexpected shape")

    uids: list[str] = []
    x_rows: list[np.ndarray] = []
    centers: list[float] = []
    scales: list[float] = []
    moved_total = 0
    for row in eval_rows:
        uid = str(row["series_uid"])
        compiled = _as_compiled(eval_assignment[uid])
        raw = _series(values, uid)
        window = _eval_window(raw, origin)
        baseline = forecast_runtime._linear_integrity(window)
        prepared, _trace = _apply(window, compiled)
        moved_total += int(np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True)))
        center, scale, method = forecast_runtime._center_scale(np, prepared)
        if method == "scale_floor_fallback":
            raise DegenerateContext(
                "evaluation context reached scale floor on %s; not falling back to raw" % uid
            )
        x_rows.append((prepared - center) / scale)
        centers.append(float(center))
        scales.append(float(scale))
        uids.append(uid)

    if not x_rows:
        predictions = np.zeros((0, HORIZON), dtype=np.float64)
        center_arr = np.zeros((0,), dtype=np.float64)
        scale_arr = np.zeros((0,), dtype=np.float64)
    else:
        x_eval = np.asarray(x_rows, dtype=np.float64)
        z_eval = np.column_stack((x_eval, np.ones(x_eval.shape[0], dtype=np.float64)))
        norm_pred = z_eval @ coef
        if not np.isfinite(norm_pred).all():
            raise RuntimeError("pooled Ridge prediction is non-finite")
        center_arr = np.asarray(centers, dtype=np.float64)
        scale_arr = np.asarray(scales, dtype=np.float64)
        predictions = norm_pred * scale_arr[:, None] + center_arr[:, None]
    return {
        "eval_uids": uids,
        "predictions": np.asarray(predictions, dtype=np.float64),
        "centers": center_arr,
        "scales": scale_arr,
        "behavior_point_count": int(moved_total),
        "consumer_fits": 0,
        "origin": origin,
    }


def score_predictions(
    predictions: Mapping[str, Any],
    roster: Sequence[Mapping[str, Any]],
    values: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    origin: int,
) -> dict[str, Any]:
    """Score frozen predictions against raw evaluation truth.

    Truth is ``raw[origin:origin+48]`` and is never written into the model.
    ``config["period"]`` is the sMASE seasonal period; the scale is computed
    on ``raw[:origin]`` with ``min_pairs=32``, matching the scoped / v6 path.

    If any eval series cannot be scored, ``mean_smase`` is ``None``: the
    population is incomplete and the overall mean is not readable.  Individual
    readable losses stay in ``per_view_smase`` with ``None`` at failed rows.
    """
    import statistics

    origin = int(origin)
    period = int(config["period"])
    eval_rows = _eval_rows(roster)
    pred_uids = [str(uid) for uid in predictions["eval_uids"]]
    pred = np.asarray(predictions["predictions"], dtype=np.float64)
    if pred_uids != [str(row["series_uid"]) for row in eval_rows]:
        raise ValueError("prediction eval_uids do not match the eval roster order")
    if pred.shape != (len(eval_rows), HORIZON):
        raise ValueError("prediction array does not match eval roster × 48")

    losses: list[float | None] = []
    metric_scales: list[float | None] = []
    complete = True
    for index, row in enumerate(eval_rows):
        uid = str(row["series_uid"])
        raw = _series(values, uid)
        stop = origin + HORIZON
        if stop > raw.size:
            losses.append(None)
            metric_scales.append(None)
            complete = False
            continue
        truth = np.asarray(raw[origin:stop], dtype=np.float64).ravel()
        observed = np.isfinite(truth)
        if not observed.any():
            losses.append(None)
            metric_scales.append(None)
            complete = False
            continue
        try:
            scale = seasonal_scale(
                raw[:origin],
                np.isfinite(raw[:origin]),
                period=period,
                min_pairs=32,
            )
            loss = smase(truth[observed], pred[index][observed], scale=scale)
        except Exception:
            losses.append(None)
            metric_scales.append(None)
            complete = False
            continue
        losses.append(float(loss))
        metric_scales.append(float(scale))

    readable = [float(v) for v in losses if v is not None]
    mean = float(statistics.fmean(readable)) if complete and readable else None
    return {
        "per_view_smase": losses,
        "mean_smase": mean,
        "metric_scales": metric_scales,
        "complete": bool(complete),
        "origin": origin,
        "eval_uids": pred_uids,
    }
