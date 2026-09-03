"""Parallel Best-of-N@8: search on held-in, freeze, redeploy unchanged.

The old P4 comparator selected and scored inside the same origin.  Here the
endpoint lives on a separate origin block, so the search has to finish on
held-in and hand over one frozen program; searching again on held-out would
turn the endpoint into a selection face and there would be nothing left to
compare the arms against.

Deterministic and 0 LLM: eight fixed candidates, one Consumer reading each per
held-in origin, and a name tie-break so the choice does not depend on
dictionary order.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import p4b_heldout as heldout
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4

NAME = forecast_p4.PARALLEL_COMPARATOR
IDENTITY = "identity"
# The frozen menu, unchanged from P4: seven programs plus identity is the N=8.
PROGRAMS: tuple[str, ...] = (IDENTITY,) + tuple(forecast_p4.PARALLEL_PROGRAMS)
SELECTION_FACE = "held_in"


def _steps(program: str) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    return () if program == IDENTITY else forecast_p1._steps(program)


def select_on_held_in(
    base_cell: Any, held_in_origins: Sequence[int]
) -> dict[str, Any]:
    """Pick one program on held-in and freeze it.

    Scored by mean sMASE over the held-in origins on the Support-A face, so no
    single origin decides, and tie-broken by name so the result is stable.
    """
    per_program: dict[str, list[float]] = {program: [] for program in PROGRAMS}
    for origin in held_in_origins:
        cell = forecast_p4._cell_at(base_cell, origin)
        for program in PROGRAMS:
            reading = forecast_p4._reading(
                cell, "support_a", _steps(program), origin=origin
            )
            per_program[program].append(float(reading["smase"]))
    mean_smase = {
        program: float(np.mean(values)) for program, values in per_program.items()
    }
    selected = min(PROGRAMS, key=lambda program: (mean_smase[program], program))
    return {
        "comparator": NAME,
        "selection_face": SELECTION_FACE,
        "selection_origins": [int(origin) for origin in held_in_origins],
        "candidates": list(PROGRAMS),
        "candidate_count": len(PROGRAMS),
        "mean_support_a_smase": mean_smase,
        "selected_program": selected,
        "frozen_steps": (
            [] if selected == IDENTITY else [{"op": selected, "params": {}}]
        ),
        "selection_rule": (
            "minimum mean Support-A sMASE across the held-in origins, "
            "ties broken by program name"
        ),
        "held_out_origins_read_during_selection": 0,
        "consumer_fits": len(PROGRAMS) * len(held_in_origins),
        "llm_calls": 0,
    }


def held_out_rows(
    *,
    base_cell: Any,
    selection: Mapping[str, Any],
    held_out_origins: Sequence[int],
    replica: str,
) -> list[dict[str, Any]]:
    """Redeploy the frozen program on every held-out origin, unchanged."""
    if selection.get("selection_face") != SELECTION_FACE:
        raise ValueError(
            "Parallel selection must come from held-in, got %r"
            % (selection.get("selection_face"),)
        )
    steps = [dict(step) for step in selection.get("frozen_steps") or ()]
    return [
        {
            **heldout.frozen_program_row(
                arm=NAME,
                replica=replica,
                origin=int(origin),
                base_cell=base_cell,
                applied_steps=steps,
                selection_face=SELECTION_FACE,
            ),
            "selected_program": selection.get("selected_program"),
        }
        for origin in held_out_origins
    ]
