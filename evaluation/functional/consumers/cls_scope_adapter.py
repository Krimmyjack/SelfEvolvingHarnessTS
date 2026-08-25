"""CLS-OP -- the classification Consumer adapter.

What this file is: one ``evaluate_fn`` in the shape ``ScopeExecutor`` already
injects, so the *same* ``run_online_round`` / ``open_delayed`` entry point can
be handed a classification TaskSpec and the C38 family's Consumer without a
second Harness existing anywhere.  It is the classification twin of
``consumers/ad_scope_adapter.py`` and follows it line for line where the two
Consumers agree.

    f(roster, values, compiled, config, *, origin)
        -> {"mean_smase": float, "per_view_smase": [float],
            "behavior_point_count": int}

What this file deliberately is NOT: it never decides a ``relation``, never
picks a winner, never reads or writes a Skill, and never applies a risk
threshold.  Those belong to ``classify_relation`` and ``method.py``.  It also
never touches the window verifier: the guard stays where it is and this
adapter only supplies the reading that comes after the guard passed.

Three conventions, all load-bearing:

1. **Sign.**  The executor computes ``gain = baseline - candidate`` over the
   ``mean_smase`` field, i.e. it assumes a loss (lower is better).  The
   classification Consumer's primary reading is accuracy, where *higher* is
   better, so the adapter reports ``-accuracy`` and the executor's own
   arithmetic yields ``candidate_accuracy - baseline_accuracy`` with no change
   to the executor.  ``per_view_smase`` carries the negated per-class recalls
   for the same reason.

2. **What a "view" is.**  The forecasting and AD sides read one value per
   series.  A classification cohort has no per-series utility: accuracy exists
   only over a set of rows.  The per-view axis is therefore the *class* axis --
   one recall per label -- which is exactly the granularity the harm question
   is asked at ("did a class get worse?").  ``classify_relation`` then writes
   CONFLICT when aggregate accuracy rises while a class recall falls, which is
   the reading the book wants and the one a single scalar cannot express.

3. **Where "origin" lands.**  The forecasting side uses ``origin`` as a time
   boundary.  Classification rows are not a time-ordered stream, so origin is
   used only to select which frozen evaluation surface is read:
   ``origin < delayed_origin`` -> the held-in Support rows, ``origin <
   heldout_origin`` -> the held-in delayed rows, otherwise the official TEST
   rows.  The caller owns those three integers; the adapter only compares.

Budget: a memo keyed by (program signature, surface) makes a repeated reading
of the same program on the same surface free.  The fit and the scoring are
deterministic and closed-form, so a re-run returns identical numbers by
construction.  Only cache misses are counted as Consumer fits.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from consumers.ad_scope_adapter import compiled_steps

SUPPORT = "support"
DELAYED = "delayed"
HELDOUT = "heldout"
SURFACES = (SUPPORT, DELAYED, HELDOUT)


def _steps_signature(compiled: Any) -> str:
    steps = compiled_steps(compiled)
    if not steps:
        return "identity"
    return "|".join(
        "%s(%s)" % (op, json.dumps(params, sort_keys=True))
        for op, params in steps)


def raw_plus_difference(matrix: Any) -> np.ndarray:
    """The C38 Consumer's feature map, unchanged: [raw | first difference].

    Kept here rather than imported from the legacy runner so the shared-Harness
    path has one readable definition; a byte-equality check against the legacy
    ``_features`` is part of the runner's preflight.
    """
    block = np.asarray(matrix, dtype=np.float64)
    return np.concatenate((block, np.diff(block, axis=1)), axis=1)


class ClassificationConsumerAdapter:
    """steps -> ridge-raw-plus-difference readings, in the executor's shape.

    ``fit_values`` are the rows the Workflow is allowed to act on (the family's
    fit cohort, with the controlled impulse already injected by the runner).
    ``surfaces`` maps each surface name to the (values, labels) pair that
    surface scores on.  Nothing else is visible to this object.
    """

    def __init__(
        self,
        *,
        fit_values: Any,
        fit_labels: Any,
        surfaces: Mapping[str, tuple[Any, Any]],
        delayed_origin: int,
        heldout_origin: int,
        budget: Any,
        ridge_alpha: float = 1.0,
        allowed_surfaces: Sequence[str] = (SUPPORT, DELAYED),
    ) -> None:
        self._fit_values = np.asarray(fit_values, dtype=np.float64)
        self._fit_labels = np.asarray(fit_labels)
        self._surfaces = {
            str(name): (np.asarray(values, dtype=np.float64),
                        np.asarray(labels))
            for name, (values, labels) in surfaces.items()
        }
        missing = [name for name in allowed_surfaces
                   if name not in self._surfaces]
        if missing:
            raise ValueError("adapter is missing surfaces: %s" % missing)
        self._delayed_origin = int(delayed_origin)
        self._heldout_origin = int(heldout_origin)
        self._budget = budget
        self._alpha = float(ridge_alpha)
        self._allowed = tuple(str(name) for name in allowed_surfaces)
        self._classes = tuple(int(label)
                              for label in sorted(set(self._fit_labels.tolist())))
        self._memo: dict[tuple[str, str], dict[str, Any]] = {}
        self.calls: list[dict[str, Any]] = []

    # -- surface selection ---------------------------------------------------
    def surface_for(self, origin: int) -> str:
        if int(origin) < self._delayed_origin:
            return SUPPORT
        if int(origin) < self._heldout_origin:
            return DELAYED
        return HELDOUT

    # -- the fit block, with the program applied -----------------------------
    def _prepared_fit(self, compiled: Any) -> tuple[np.ndarray, int]:
        """Row-wise application of the compiled Workflow to the fit cohort.

        Identity (no steps) returns the untouched block.  A row-wise pipeline
        failure raises: an unreadable instrument must not be laundered into a
        negative reading.
        """
        from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

        steps = compiled_steps(compiled)
        if not steps:
            return self._fit_values, 0
        rows: list[np.ndarray] = []
        behavior = 0
        for row in self._fit_values:
            result = run_pipeline(list(steps), row)
            if not result.ok or result.artifact is None:
                raise RuntimeError(
                    "classification Workflow failed on a fit row: %s"
                    % result.error)
            out = np.asarray(result.artifact, dtype=np.float64).ravel()
            if out.shape != row.shape:
                raise RuntimeError(
                    "classification Workflow changed the row shape: %s -> %s"
                    % (row.shape, out.shape))
            behavior += int(np.count_nonzero(~np.isclose(out, row)))
            rows.append(out)
        return np.asarray(rows, dtype=np.float64), behavior

    # -- the reading ---------------------------------------------------------
    def __call__(
        self,
        roster: Sequence[Mapping[str, object]],
        values: Mapping[str, Any],
        compiled: Any,
        config: Mapping[str, object],
        *,
        origin: int,
    ) -> dict[str, object]:
        from sklearn.linear_model import RidgeClassifier

        signature = _steps_signature(compiled)
        surface = self.surface_for(int(origin))
        if surface not in self._allowed:
            raise RuntimeError(
                "surface %r is not open to this executor (allowed: %s)"
                % (surface, list(self._allowed)))
        memo_key = (signature, surface)
        if memo_key in self._memo:
            self.calls.append({"signature": signature, "surface": surface,
                               "origin": int(origin), "cache": "hit",
                               "consumer_fits": 0})
            return dict(self._memo[memo_key])

        prepared, behavior = self._prepared_fit(compiled)
        if self._budget is not None:
            self._budget.spend(1)
        model = RidgeClassifier(alpha=self._alpha)
        model.fit(raw_plus_difference(prepared), self._fit_labels)
        eval_values, eval_labels = self._surfaces[surface]
        predicted = model.predict(raw_plus_difference(eval_values))
        accuracy = float(np.mean(predicted == eval_labels))
        recalls: list[float] = []
        recall_by_class: dict[str, float] = {}
        for label in self._classes:
            mask = eval_labels == label
            value = (float(np.mean(predicted[mask] == label))
                     if bool(np.any(mask)) else float(accuracy))
            recalls.append(value)
            recall_by_class[str(label)] = value
        reading = {
            # negated: the executor reads these fields as a loss
            "mean_smase": -accuracy,
            "per_view_smase": [-value for value in recalls],
            "behavior_point_count": int(behavior),
            "cls_accuracy": accuracy,
            "cls_recall_by_class": recall_by_class,
            "cls_surface": surface,
            "cls_evaluated_rows": int(eval_labels.size),
        }
        self._memo[memo_key] = dict(reading)
        self.calls.append({"signature": signature, "surface": surface,
                           "origin": int(origin), "cache": "miss",
                           "consumer_fits": 1, "accuracy": accuracy})
        return dict(reading)


__all__ = [
    "ClassificationConsumerAdapter",
    "raw_plus_difference",
    "SUPPORT",
    "DELAYED",
    "HELDOUT",
    "SURFACES",
]
