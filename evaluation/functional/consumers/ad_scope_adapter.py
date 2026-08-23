"""T5 #41 A2 -- the anomaly-detection Consumer adapter.

What this file is: one ``evaluate_fn`` in the shape ``ScopeExecutor`` already
injects, so the *same* ``run_online_round`` / ``open_delayed`` entry point can
be handed an anomaly-detection TaskSpec and a different Consumer without a
second Harness existing anywhere.

    f(roster, values, compiled, config, *, origin)
        -> {"mean_smase": float, "per_view_smase": [float],
            "behavior_point_count": int}

What this file deliberately is NOT: it never decides a ``relation``, never
picks a winner, never reads or writes a Skill, and never applies a risk
threshold.  Those all belong to the one lifecycle that ``classify_relation``
and ``method.py`` own; an adapter that made any of those calls would be the
second Harness this round exists to prove unnecessary.  It also does not
touch the window verifier: the guard syntax stays exactly where it is, and
this adapter only supplies the reading that comes after the guard passed.

Two conventions to be explicit about, because both are load-bearing:

1. **Sign.**  The executor computes ``gain = baseline - candidate`` over the
   ``mean_smase`` field, i.e. it assumes the reading is a loss (lower is
   better).  The AD Consumer's primary reading is macro-averaged per-series
   event F1, where *higher* is better.  The adapter therefore reports
   ``-macro_f1``, so the executor's own arithmetic yields
   ``candidate_F1 - baseline_F1`` with no change to the executor.

2. **Where "origin" lands.**  The forecasting side uses ``origin`` as a time
   boundary.  Here it selects which frozen Query region is read:
   ``origin < delayed_boundary`` -> Qcal (the development Support surface),
   otherwise Qf (the delayed surface).  Both surfaces are already-exposed
   positive-control material; nothing sealed is opened here.

Budget: a memo keyed by (program signature, region) makes a repeated reading
of the same program on the same region free.  This is not a shortcut around
the instrument -- the fit and the scoring are deterministic and closed-form,
so a re-run returns the identical numbers by construction.  Only cache
misses are counted as AD evaluations.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONTEXT_LENGTH = 192
HORIZON = 48


def compiled_steps(compiled: Any) -> tuple[tuple[str, dict], ...]:
    """The (op, params) sequence a CompiledWorkflow will actually execute.

    CompiledWorkflow has no ``steps`` attribute -- it carries ``candidate``
    and ``template_steps`` -- so reading ``compiled.steps`` silently yields
    nothing and every candidate looks like identity.  Go through the
    candidate's Program, which is where execution_steps() lives.
    """
    if compiled is None:
        return ()
    candidate = getattr(compiled, "candidate", None)
    program = getattr(candidate, "program", None)
    if program is not None:
        return tuple((str(op), dict(params))
                     for op, params in program.execution_steps())
    out: list[tuple[str, dict]] = []
    for step in (getattr(compiled, "template_steps", None) or ()):
        if isinstance(step, Mapping) and step.get("op"):
            out.append((str(step["op"]), dict(step.get("params") or {})))
    return tuple(out)


def _steps_signature(compiled: Any) -> str:
    steps = compiled_steps(compiled)
    if not steps:
        return "identity"
    return "|".join(
        "%s(%s)" % (op, json.dumps(params, sort_keys=True))
        for op, params in steps)


class ADConsumerAdapter:
    """Presents the AD Consumer in the executor's reading shape.

    ``train_uids`` are the twelve injected stations; ``block`` is the region
    of each series the program is allowed to act on, and it is exactly the
    union of the windows the executor's verifier checks (so what gets fitted
    is what got verified, with no second notion of an action region).
    """

    def __init__(
        self,
        *,
        adt: Any,
        apply_program: Any,
        train_uids: Sequence[str],
        block: tuple[int, int],
        anchors: Sequence[int],
        query_root: Path,
        train_ledger_path: Path,
        qcal_region: tuple[int, int],
        qf_region: tuple[int, int],
        delayed_boundary: int,
        evaluation_budget: int,
    ) -> None:
        self._adt = adt
        self._apply_program = apply_program
        self.train_uids = [str(uid) for uid in train_uids]
        self.block = (int(block[0]), int(block[1]))
        self.anchors = tuple(int(a) for a in anchors)
        self.qcal_region = (int(qcal_region[0]), int(qcal_region[1]))
        self.qf_region = (int(qf_region[0]), int(qf_region[1]))
        self.delayed_boundary = int(delayed_boundary)
        self.evaluation_budget = int(evaluation_budget)
        self.evaluations_used = 0
        self.calls: list[dict[str, Any]] = []
        self._memo: dict[tuple[str, str], dict[str, Any]] = {}
        self._query: dict[str, dict[str, np.ndarray]] = {}
        self._ledger: dict[str, dict[str, list]] = {}
        for tag, sub in (("qcal", "qcal"), ("qf", "qf")):
            root = query_root / sub
            self._query[tag] = {
                uid: np.asarray(np.load(root / ("%s.npy" % uid)),
                                dtype=np.float64)
                for uid in self.train_uids
            }
            rows = json.loads((root / "ledger.json").read_text(encoding="utf-8"))
            self._ledger[tag] = {uid: list(rows.get(uid, []))
                                 for uid in self.train_uids}
        # the training labels come from the *training block* ledger, not from
        # either Query ledger: the Query ledgers site their events at 2100+
        # and would label the block all-negative.
        _train_rows = json.loads(
            train_ledger_path.read_text(encoding="utf-8"))
        self._ledger["train"] = {uid: list(_train_rows.get(uid, []))
                                 for uid in self.train_uids}

    # -- region selection ---------------------------------------------------
    def region_for(self, origin: int) -> tuple[str, tuple[int, int]]:
        if int(origin) < self.delayed_boundary:
            return "qcal", self.qcal_region
        return "qf", self.qf_region

    # -- the training block, with the program applied -----------------------
    def _prepared_block(self, values: Mapping[str, Any],
                        compiled: Any) -> tuple[dict[str, np.ndarray], int]:
        """Rebuild each station's P(B) by concatenating the very windows the
        verifier checked, in anchor order.  Identity (compiled is None) gives
        the untouched block."""
        prepared: dict[str, np.ndarray] = {}
        behavior = 0
        for uid in self.train_uids:
            raw = np.asarray(values[uid], dtype=np.float64)
            pieces: list[np.ndarray] = []
            for anchor in self.anchors:
                window = raw[anchor - CONTEXT_LENGTH: anchor + HORIZON]
                if compiled is None:
                    piece = np.asarray(window, dtype=np.float64)
                else:
                    piece, _trace = self._apply_program(window, compiled)
                    piece = np.asarray(piece, dtype=np.float64)
                    behavior += int(np.count_nonzero(
                        ~np.isclose(piece, window, equal_nan=True)))
                pieces.append(piece)
            prepared[uid] = np.concatenate(pieces)
        return prepared, behavior

    # -- the reading --------------------------------------------------------
    def __call__(
        self,
        roster: Sequence[Mapping[str, object]],
        values: Mapping[str, Any],
        compiled: Any,
        config: Mapping[str, object],
        *,
        origin: int,
    ) -> dict[str, object]:
        signature = _steps_signature(compiled)
        tag, region = self.region_for(int(origin))
        memo_key = (signature, tag)
        if memo_key in self._memo:
            cached = dict(self._memo[memo_key])
            self.calls.append({"signature": signature, "region": tag,
                               "origin": int(origin), "cache": "hit",
                               "ad_evaluations": 0})
            return cached
        if self.evaluations_used + len(self.train_uids) > self.evaluation_budget:
            raise RuntimeError(
                "AD evaluation budget exceeded: %d used, %d requested, cap %d"
                % (self.evaluations_used, len(self.train_uids),
                   self.evaluation_budget))

        prepared, behavior = self._prepared_block(values, compiled)
        adt = self._adt
        # -- fit on the processed training block ---------------------------
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        constants: dict[str, Any] = {}
        for uid in self.train_uids:
            buffer = prepared[uid]
            feats = adt.block_features(buffer)
            z = np.asarray(feats["z"], dtype=np.float64)
            constants[uid] = {"median": None, "scale": None,
                              "source": "none_v3_z_feature"}
            events = {
                int(point)
                for row in self._ledger["train"].get(uid, [])
                for point in range(int(row["index"]),
                                   int(row["index"]) + int(row["points"]))
            }
            # the block's own absolute indices: the concatenated windows start
            # at the first anchor's window start
            absolute = np.arange(buffer.size, dtype=np.int64) + self.block[0]
            y_all = np.array([1.0 if int(t) in events else 0.0
                              for t in absolute], dtype=np.float64)
            finite = np.isfinite(z)
            xs.append(z[finite, None])
            ys.append(y_all[finite])
        model = adt.fit(np.concatenate(xs, axis=0), np.concatenate(ys, axis=0))
        model["constants"] = constants

        # -- score the selected Query region -------------------------------
        per_series: dict[str, Any] = {}
        for uid in self.train_uids:
            per_series[uid] = adt.score_query_series(
                model, self._query[tag][uid], region,
                self._ledger[tag].get(uid, []),
                window=int(adt.FEATURE_WINDOW), median=None, scale=None,
            )
        self.evaluations_used += len(self.train_uids)
        macro = adt.macro_f1(per_series)
        per_view = [float(per_series[uid]["f1"] or 0.0)
                    for uid in self.train_uids]
        reading = {
            # negated: the executor reads this field as a loss
            "mean_smase": -float(macro if macro is not None else 0.0),
            "per_view_smase": [-value for value in per_view],
            "behavior_point_count": int(behavior),
            "ad_macro_f1": float(macro) if macro is not None else None,
            "ad_pooled_f1": adt.pooled_f1(per_series),
            "ad_f1_by_series": {uid: per_series[uid]["f1"]
                                for uid in self.train_uids},
            "ad_region": tag,
        }
        self._memo[memo_key] = dict(reading)
        self.calls.append({"signature": signature, "region": tag,
                           "origin": int(origin), "cache": "miss",
                           "ad_evaluations": len(self.train_uids),
                           "macro_f1": reading["ad_macro_f1"]})
        return dict(reading)


__all__ = ["ADConsumerAdapter", "compiled_steps"]
