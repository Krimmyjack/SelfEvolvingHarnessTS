"""Which origins the Consumer can be read at, decided without reading Outcome.

Some KDD context windows are flat enough that the forecast runtime's robust
scale collapses to its floor, and it refuses the window rather than divide by
it (``evaluation context reached scale floor``).  An origin plan that ignores
this aborts mid-run, so the plan is screened first.

The screen is structural: it recomputes the same centre/scale the runtime would
on ``[origin - CONTEXT_LENGTH, origin)``, which is observed history the Fast
Path may see anyway.  It never fits a Consumer, never reads the evaluation
horizon, and never looks at an Outcome -- it is the same kind of act as
checking that the roster is long enough.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from evaluation.functional import run_e2_autonomous_natural_workflow_generation as runtime
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

CONTEXT_LENGTH = runtime.CONTEXT_LENGTH
FACES = ("support_a", "support_b")


def origin_is_viable(base_cell: Any, origin: int) -> bool:
    """Can every evaluation series be read at this origin?"""
    if origin - CONTEXT_LENGTH < 0:
        return False
    try:
        cell = forecast_p4._cell_at(base_cell, int(origin))
    except Exception:  # noqa: BLE001 - a short roster is simply not viable here
        return False
    for face in FACES:
        for row in cell.roster(face):
            if str(row.get("role")) != "eval":
                continue
            raw = np.asarray(cell.values[str(row["series_uid"])], dtype=np.float64)
            window = runtime._linear_integrity(raw[origin - CONTEXT_LENGTH : origin])
            _centre, _scale, method = runtime._center_scale(np, window)
            if method == "scale_floor_fallback":
                return False
    return True


def screen(base_cell: Any) -> Callable[[int], bool]:
    """A memoised viability predicate for one roster."""
    cache: dict[int, bool] = {}

    def is_viable(origin: int) -> bool:
        key = int(origin)
        if key not in cache:
            cache[key] = origin_is_viable(base_cell, key)
        return cache[key]

    return is_viable


def census(base_cell: Any, origins: list[int]) -> dict[str, Any]:
    """What the screen found, for the preflight receipt."""
    is_viable = screen(base_cell)
    viable = [origin for origin in origins if is_viable(origin)]
    rejected = [origin for origin in origins if origin not in set(viable)]
    return {
        "screened": len(origins),
        "viable": viable,
        "rejected": rejected,
        "rejection_reason": "context window robust scale collapsed to its floor",
        "outcome_reads": 0,
        "consumer_fits": 0,
        "llm_calls": 0,
    }
