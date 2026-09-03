"""batch-recipe across development windows: three windows per cell, and the
first out-of-selection check the v2 adoption rule has ever had.

``batch_recipe_v2_all_cells_v1`` produced one adopted plan per cell on one
window each.  Two things are missing from that.  First, the (window x cell x
adopted plan) sample is one window deep, which is not enough to say anything
about reuse across windows.  Second -- and this is the load-bearing gap -- the
v2 rule adopts a masked plan only if it clears a delayed bar, so the delayed
window is *inside* its selection.  No adopted plan has ever been read on a
window that did not participate in adopting it.

This runner does both at once.  For each of the six cells (electricity / T233 /
traffic x pooled / per_channel) it runs the same frozen v2 recipe on two more
development windows, and then evaluates **window 1's already-adopted plan** on
those two new windows, where that plan's own adoption rule never looked.

Origins come from already-exposed records only:

* electricity and T233 use the frozen e1v2 Task roster verbatim --
  ``e1v2_task_01`` (window 1, the existing artifact), ``e1v2_task_02`` and
  ``e1v2_task_03``.  Those Tasks are inside the rosters the g1 pipeline and the
  M0a census already ran (electricity 01..09, T233 01..19);
* traffic has no roster origins.  Its screening record declares exactly one
  triple of development origins (1104 / 1368 / 1800, window 1) plus
  ``sealed_from_index=3072``.  Windows 2 and 3 keep window 1's shape and
  spacing and are shifted forward inside that same pre-sealed development
  region, so the farthest index any of them reads is 2808, well before 3072.
  This is the one place where an origin is *chosen* rather than quoted, and it
  is flagged as such in the artifact.

Discipline: 0 LLM calls, deterministic, no Skill, no Episode, no Fast or Slow
path.  It writes one new stem and overwrites nothing: every existing artifact,
including ``batch_recipe_v2_all_cells_v1``, is read-only input.  The recipe
module is imported, not modified; the traffic window is selected by rebinding
its module-level development-origin constant for the duration of one call and
restoring it afterwards, with the module's own sealed-boundary guard left
active.

Pre-stated before the numbers were seen:

* out-of-selection check, per cell: window 1's adopted plan passes iff its
  delayed aggregate gain is ``>= 0.0`` on **both** new windows.  The v2 rule's
  own bar is ``max(best full-batch delayed, 0)``, i.e. identity is the
  incumbent, so zero is the honest bar here too.  A second column reports the
  stricter ``> MATERIAL_THRESHOLD`` reading for information only;
* stability, per cell: ``program_stable`` iff all three windows adopt the same
  program, ``mask_stable`` iff all three adopt the same excluded set.

Run:

    python evaluation/functional/run_e2_batch_recipe_windows.py

Writes ``artifacts/functional/e2/batch_recipe_windows_v1.json`` and ``.md``.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import run_batch_composition_headroom as bch  # noqa: E402
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

from evaluation.functional.task_episode_harness.e1 import (  # noqa: E402
    _frozen_task_roster,
)
from evaluation.functional.task_episode_harness.runner import (  # noqa: E402
    _compiled,
)

PROTOCOL_VERSION = "batch_recipe_windows_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "batch_recipe_windows_v1.json"
OUT_MD = E2 / "batch_recipe_windows_v1.md"
WINDOW_1_ARTIFACT = E2 / "batch_recipe_v2_all_cells_v1.json"

ADOPTION_RULE_VERSION = "v2"
MATERIAL_THRESHOLD = float(bch.MATERIAL_THRESHOLD)
HORIZON = int(bch.v6.HORIZON)
TRAFFIC_SEALED_FROM_INDEX = int(bch._TRAFFIC_SEALED_FROM_INDEX)

# electricity / T233: roster Task indices, quoted not chosen.
ROSTER_WINDOW_TASK_INDEX = {"W1": 0, "W2": 1, "W3": 2}
ROSTER_PROVENANCE = (
    "frozen e1v2 Task roster (task_episode_harness.e1._frozen_task_roster), "
    "Task %s, support and delayed origins verbatim; this Task is inside the "
    "roster the g1 agentic pipeline and the M0a census already ran on this "
    "cohort (electricity e1v2_task_01..09, T233 e1v2_task_01..19)"
)

# traffic: window 1 is the screening record; windows 2 and 3 keep its shape
# (two Support origins, one delayed) and its spacing (+264, then +432) and are
# shifted forward by +480 and +960 inside the same pre-sealed region.
TRAFFIC_WINDOWS: dict[str, tuple[int, int, int]] = {
    "W1": (1104, 1368, 1800),
    "W2": (1584, 1848, 2280),
    "W3": (2064, 2328, 2760),
}
TRAFFIC_PROVENANCE = {
    "W1": (
        "artifacts/functional/e2/g3_candidate_screening_v2.json, "
        "criteria.development_origins = [1104, 1368, 1800]; the same triple "
        "the batch recipe and the M0a traffic census already ran on"
    ),
    "W2": (
        "chosen, not quoted: window 1's shape and spacing shifted by +480 "
        "inside the same pre-sealed development region declared by "
        "g3_candidate_screening_v2.json (sealed_from_index = 3072)"
    ),
    "W3": (
        "chosen, not quoted: window 1's shape and spacing shifted by +960 "
        "inside the same pre-sealed development region declared by "
        "g3_candidate_screening_v2.json (sealed_from_index = 3072)"
    ),
}

PRE_REGISTERED = {
    "out_of_selection_check": (
        "per cell, window 1's adopted plan passes iff its delayed aggregate "
        "gain is >= 0.0 on both new windows; the v2 rule's own bar is "
        "max(best full-batch delayed, 0), so identity at zero is the honest "
        "bar. A second column reports the stricter > MATERIAL_THRESHOLD=%.3f "
        "reading for information only." % MATERIAL_THRESHOLD
    ),
    "per_cell_verdicts": [
        "W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION: delayed >= 0 on both "
        "new windows",
        "W1_PLAN_DELAYED_NEGATIVE_ON_SOME_WINDOW: otherwise",
    ],
    "stability": (
        "program_stable iff all three windows adopt the same program; "
        "mask_stable iff all three adopt the same excluded set; kind_stable "
        "iff all three adopt the same plan kind"
    ),
    "fixed_before_the_numbers_were_seen": True,
}


# ------------------------------------------------------------------- windows
def _window_specs(cohort: str) -> list[dict[str, Any]]:
    """The three windows for one cohort, each with where its origins came from."""
    specs: list[dict[str, Any]] = []
    if cohort == "traffic":
        for window_id, origins in TRAFFIC_WINDOWS.items():
            support = [int(origins[0]), int(origins[1])]
            delayed = [int(origins[2])]
            farthest = max(support + delayed) + HORIZON
            if farthest > TRAFFIC_SEALED_FROM_INDEX:
                raise RuntimeError(
                    "traffic window %s would read index %d at or past "
                    "sealed_from_index=%d"
                    % (window_id, farthest, TRAFFIC_SEALED_FROM_INDEX)
                )
            specs.append({
                "window_id": window_id,
                "task_index": 0,
                "task_episode_id": "e1v2_task_01",
                "support_origins": support,
                "delayed_origins": delayed,
                "farthest_index_read": farthest,
                "origin_source": (
                    "quoted from the screening record" if window_id == "W1"
                    else "chosen inside the declared pre-sealed region"
                ),
                "origin_provenance": TRAFFIC_PROVENANCE[window_id],
                "sealed_from_index": TRAFFIC_SEALED_FROM_INDEX,
                "traffic_origin_override": list(origins),
            })
        return specs
    roster = _frozen_task_roster()
    for window_id, task_index in ROSTER_WINDOW_TASK_INDEX.items():
        spec = roster[task_index]
        support = [int(origin) for origin in spec["support_origins"]]
        delayed = [int(origin) for origin in spec["delayed_origins"]]
        specs.append({
            "window_id": window_id,
            "task_index": int(task_index),
            "task_episode_id": str(spec["task_episode_id"]),
            "support_origins": support,
            "delayed_origins": delayed,
            "farthest_index_read": max(support + delayed) + HORIZON,
            "origin_source": "quoted from the frozen roster",
            "origin_provenance": ROSTER_PROVENANCE % spec["task_episode_id"],
            "traffic_origin_override": None,
        })
    return specs


@contextlib.contextmanager
def _traffic_origins(override: Sequence[int] | None):
    """Rebind the recipe module's traffic development origins for one call.

    ``_task_windows`` reads ``_TRAFFIC_DEVELOPMENT_ORIGINS`` from the module
    namespace at call time, and there is no argument for it.  Rebinding it here
    -- and restoring it in ``finally`` -- selects the window without editing the
    recipe module.  ``_TRAFFIC_SEALED_FROM_INDEX`` is deliberately **not**
    touched, so the module's own boundary guard stays live over the override.
    """
    if override is None:
        yield
        return
    saved = bch._TRAFFIC_DEVELOPMENT_ORIGINS
    bch._TRAFFIC_DEVELOPMENT_ORIGINS = tuple(int(item) for item in override)
    try:
        yield
    finally:
        bch._TRAFFIC_DEVELOPMENT_ORIGINS = saved


class WindowEvaluator:
    """Score one plan on one window, on the recipe's own executor.

    Identity baselines are cached per (cohort, consumer_variant, origins) so a
    window is only ever baselined once.  Nothing here re-implements the gain
    definition: ``_evaluate_variant`` / ``_evaluate_assignment`` / ``_gain_rows``
    are the recipe module's.
    """

    def __init__(self) -> None:
        self._cells: dict[tuple[str, str], dict[str, Any]] = {}
        self._identity: dict[tuple[str, str, tuple[int, ...]], Any] = {}

    def cell(self, cohort: str, consumer_variant: str) -> dict[str, Any]:
        key = (cohort, consumer_variant)
        cached = self._cells.get(key)
        if cached is not None:
            return cached
        loaded = bch.load_cohort(PROJECT_ROOT, cohort)
        cell = {
            "config": dict(_config()),
            "roster": loaded["mapped_roster"],
            "values": loaded["values"],
            "train_uids": [str(uid) for uid in loaded["train_uids"]],
            "eval_uids": [str(uid) for uid in loaded["eval_uids"]],
            "exposure": loaded["exposure"],
            "compiled": {},
        }
        self._cells[key] = cell
        return cell

    def _identity_rows(
        self, cohort: str, consumer_variant: str, origins: tuple[int, ...]
    ) -> Any:
        key = (cohort, consumer_variant, origins)
        cached = self._identity.get(key)
        if cached is None:
            cell = self.cell(cohort, consumer_variant)
            cached = bch._evaluate_variant(
                cell["roster"], cell["values"], None, cell["config"], origins,
                None, consumer_variant,
            )
            self._identity[key] = cached
        return cached

    def evaluate(
        self,
        *,
        cohort: str,
        consumer_variant: str,
        program: str,
        excluded_series: Sequence[str],
        origins: Sequence[int],
    ) -> dict[str, Any]:
        cell = self.cell(cohort, consumer_variant)
        origins = tuple(int(origin) for origin in origins)
        excluded = {str(uid) for uid in excluded_series}
        if program == bch.IDENTITY:
            assignment = {uid: None for uid in cell["train_uids"]}
        else:
            compiled = cell["compiled"].get(program)
            if compiled is None:
                compiled = _compiled(program, name="brw_%s" % program)
                cell["compiled"][program] = compiled
            assignment = {
                uid: (None if uid in excluded else compiled)
                for uid in cell["train_uids"]
            }
        rows = [
            bch._evaluate_assignment(
                cell["roster"], cell["values"], assignment, cell["config"],
                origin=origin, consumer_variant=consumer_variant,
            )
            for origin in origins
        ]
        gains = bch._gain_rows(
            self._identity_rows(cohort, consumer_variant, origins),
            rows,
            cell["eval_uids"],
        )
        return {
            "origins": list(origins),
            "aggregate_gain": float(gains["aggregate_gain"]),
            "harmed_eval_series_count": int(gains["harmed_eval_series_count"]),
            "harmed_eval_series_total_harm": float(
                gains["harmed_eval_series_total_harm"]
            ),
            "harmed_eval_series": list(gains["harmed_eval_series"]),
        }


# ------------------------------------------------------------- window 1 input
def _window_1_cells() -> dict[str, Any]:
    """The existing all-cells artifact, read verbatim.  Nothing is written."""
    data = json.loads(WINDOW_1_ARTIFACT.read_text(encoding="utf-8"))
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in data["cells"]:
        recipe = cell.get("recipe")
        if recipe is None:
            continue
        plan = recipe["adopted_plan"]
        cells[(str(cell["cohort"]), str(cell["consumer_variant"]))] = {
            "adopted_plan": {
                "kind": str(plan["kind"]),
                "program": str(plan["program"]),
                "excluded_series": sorted(
                    str(uid) for uid in plan["excluded_series"]
                ),
                "treated_series_count": int(plan["treated_series_count"]),
            },
            "support_aggregate_gain": float(
                recipe["comparison"]["support"]["adopted"]
            ),
            "delayed_aggregate_gain": float(
                recipe["comparison"]["delayed"]["adopted"]
            ),
            "harmed_eval_series_count": int(
                recipe["harm_account"]["adopted"]["harmed_eval_series_count"]
            ),
            "harmed_eval_series_total_harm": float(
                recipe["harm_account"]["adopted"][
                    "harmed_eval_series_total_harm"
                ]
            ),
            "adoption_path": str(recipe["adoption_path"]),
            "support_origins": [int(o) for o in recipe["support_origins"]],
            "delayed_origins": [int(o) for o in recipe["delayed_origins"]],
            "adoption_rule_version": str(recipe["adoption_rule_version"]),
        }
    return {
        "artifact": WINDOW_1_ARTIFACT.relative_to(PROJECT_ROOT).as_posix(),
        "protocol_version": str(data["protocol_version"]),
        "adoption_rule_version": str(data["adoption_rule_version"]),
        "cells": cells,
    }


def _compact_recipe(recipe: Mapping[str, Any]) -> dict[str, Any]:
    plan = recipe["adopted_plan"]
    harm = recipe["harm_account"]["adopted"]
    return {
        "adopted_plan": {
            "kind": str(plan["kind"]),
            "program": str(plan["program"]),
            "excluded_series": sorted(
                str(uid) for uid in plan["excluded_series"]
            ),
            "treated_series_count": int(plan["treated_series_count"]),
        },
        "support_aggregate_gain": float(recipe["comparison"]["support"]["adopted"]),
        "delayed_aggregate_gain": float(recipe["comparison"]["delayed"]["adopted"]),
        "harmed_eval_series_count": int(harm["harmed_eval_series_count"]),
        "harmed_eval_series_total_harm": float(
            harm["harmed_eval_series_total_harm"]
        ),
        "adoption_path": str(recipe["adoption_path"]),
        "delayed_stability_bar": recipe.get("delayed_stability_bar"),
        "best_full_batch_program": str(
            recipe["comparison"]["best_full_batch_program"]
        ),
        "best_full_batch_support": float(
            recipe["comparison"]["support"]["best_full_batch"]
        ),
        "best_full_batch_delayed": float(
            recipe["comparison"]["delayed"]["best_full_batch"]
        ),
        "programs_searched": [str(p) for p in recipe["programs_searched"]],
        "adoption_trace": [
            {
                "program": str(row["program"]),
                "excluded_series": sorted(
                    str(uid) for uid in row["excluded_series"]
                ),
                "support_aggregate_gain": float(row["support_aggregate_gain"]),
                "delayed_aggregate_gain": float(row["delayed_aggregate_gain"]),
                "delayed_bar": row.get("delayed_bar"),
                "stability_check": str(row["stability_check"]),
            }
            for row in recipe["adoption_trace"]
        ],
        "wall_seconds": float(recipe["wall_seconds"]),
    }


def _run_window(
    cohort: str, consumer_variant: str, window: Mapping[str, Any]
) -> dict[str, Any]:
    """One frozen v2 recipe run on one window.  Writes no file."""
    override = window["traffic_origin_override"]
    with _traffic_origins(override):
        _spec, support, delayed = bch._task_windows(
            cohort, int(window["task_index"])
        )
        if (
            [int(o) for o in support] != list(window["support_origins"])
            or [int(o) for o in delayed] != list(window["delayed_origins"])
        ):
            raise RuntimeError(
                "window %s for %s resolved to %s / %s, expected %s / %s"
                % (
                    window["window_id"], cohort, list(support), list(delayed),
                    window["support_origins"], window["delayed_origins"],
                )
            )
        recipe = bch.make_batch_recipe(
            cohort,
            task_index=int(window["task_index"]),
            consumer_variant=consumer_variant,
            adoption_rule_version=ADOPTION_RULE_VERSION,
        )
    if (
        [int(o) for o in recipe["support_origins"]]
        != list(window["support_origins"])
    ):
        raise RuntimeError("recipe reported a different Support window")
    return _compact_recipe(recipe)


# ------------------------------------------------------------------ per cell
def _run_cell(
    cohort: str,
    consumer_variant: str,
    windows: Sequence[Mapping[str, Any]],
    window_1: Mapping[str, Any],
    evaluator: WindowEvaluator,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for window in windows:
        base = {
            "window_id": window["window_id"],
            "task_episode_id": window["task_episode_id"],
            "support_origins": window["support_origins"],
            "delayed_origins": window["delayed_origins"],
            "farthest_index_read": window["farthest_index_read"],
            "origin_source": window["origin_source"],
        }
        if window["window_id"] == "W1":
            rows.append({
                **base,
                "source": "read from %s, not re-run" % (
                    WINDOW_1_ARTIFACT.name
                ),
                "adopted_plan": window_1["adopted_plan"],
                "support_aggregate_gain": window_1["support_aggregate_gain"],
                "delayed_aggregate_gain": window_1["delayed_aggregate_gain"],
                "harmed_eval_series_count": window_1[
                    "harmed_eval_series_count"],
                "harmed_eval_series_total_harm": window_1[
                    "harmed_eval_series_total_harm"],
                "adoption_path": window_1["adoption_path"],
            })
            continue
        print(
            "WINDOWS %s %s %s: recipe on %s / %s"
            % (cohort, consumer_variant, window["window_id"],
               window["support_origins"], window["delayed_origins"]),
            flush=True,
        )
        rows.append({
            **base,
            "source": "this run, frozen v2 recipe",
            **_run_window(cohort, consumer_variant, window),
        })

    # ---- the out-of-selection check --------------------------------------
    plan = window_1["adopted_plan"]
    out_of_selection: list[dict[str, Any]] = []
    for window in windows:
        if window["window_id"] == "W1":
            continue
        support = evaluator.evaluate(
            cohort=cohort, consumer_variant=consumer_variant,
            program=plan["program"], excluded_series=plan["excluded_series"],
            origins=window["support_origins"],
        )
        delayed = evaluator.evaluate(
            cohort=cohort, consumer_variant=consumer_variant,
            program=plan["program"], excluded_series=plan["excluded_series"],
            origins=window["delayed_origins"],
        )
        out_of_selection.append({
            "window_id": window["window_id"],
            "plan": dict(plan),
            "support": support,
            "delayed": delayed,
            "delayed_non_negative": bool(delayed["aggregate_gain"] >= 0.0),
            "delayed_above_material_threshold": bool(
                delayed["aggregate_gain"] > MATERIAL_THRESHOLD
            ),
            "reading": (
                "the plan adopted on W1 by the v2 rule, evaluated on a window "
                "that took no part in adopting it"
            ),
        })
        print(
            "WINDOWS %s %s W1-plan on %s: support %+.6f delayed %+.6f (>=0 %s)"
            % (cohort, consumer_variant, window["window_id"],
               support["aggregate_gain"], delayed["aggregate_gain"],
               delayed["aggregate_gain"] >= 0.0),
            flush=True,
        )

    passed = all(row["delayed_non_negative"] for row in out_of_selection)
    programs = [row["adopted_plan"]["program"] for row in rows]
    masks = [tuple(row["adopted_plan"]["excluded_series"]) for row in rows]
    kinds = [row["adopted_plan"]["kind"] for row in rows]
    return {
        "cohort": cohort,
        "consumer_variant": consumer_variant,
        "windows": rows,
        "window1_plan_out_of_selection": out_of_selection,
        "out_of_selection_verdict": (
            "W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION" if passed
            else "W1_PLAN_DELAYED_NEGATIVE_ON_SOME_WINDOW"
        ),
        "stability": {
            "programs": programs,
            "excluded_series_per_window": [list(mask) for mask in masks],
            "kinds": kinds,
            "program_stable": len(set(programs)) == 1,
            "mask_stable": len(set(masks)) == 1,
            "kind_stable": len(set(kinds)) == 1,
        },
        "wall_seconds": time.perf_counter() - started,
    }


# --------------------------------------------------------------- orchestration
def run(*, cohorts: Sequence[str] | None = None) -> int:
    started = time.perf_counter()
    if OUT_JSON.exists() or OUT_MD.exists():
        print(
            "note: %s already exists and will be replaced by this run"
            % OUT_JSON.name,
            flush=True,
        )
    window_1 = _window_1_cells()
    if window_1["adoption_rule_version"] != ADOPTION_RULE_VERSION:
        raise RuntimeError(
            "window 1 artifact is rule %s, expected %s"
            % (window_1["adoption_rule_version"], ADOPTION_RULE_VERSION)
        )
    names = list(cohorts) if cohorts else list(bch.RECIPE_V2_COHORTS)
    evaluator = WindowEvaluator()
    window_specs = {name: _window_specs(name) for name in names}
    cells: list[dict[str, Any]] = []
    for cohort in names:
        for consumer_variant in bch.CONSUMER_VARIANTS:
            reference = window_1["cells"].get((cohort, consumer_variant))
            if reference is None:
                raise RuntimeError(
                    "window 1 artifact has no cell %s x %s"
                    % (cohort, consumer_variant)
                )
            specs = window_specs[cohort]
            first = next(row for row in specs if row["window_id"] == "W1")
            if (
                first["support_origins"] != reference["support_origins"]
                or first["delayed_origins"] != reference["delayed_origins"]
            ):
                raise RuntimeError(
                    "W1 origins %s / %s do not match the artifact's %s / %s"
                    % (
                        first["support_origins"], first["delayed_origins"],
                        reference["support_origins"],
                        reference["delayed_origins"],
                    )
                )
            cells.append(
                _run_cell(cohort, consumer_variant, specs, reference, evaluator)
            )

    passing = [
        row for row in cells
        if row["out_of_selection_verdict"]
        == "W1_PLAN_DELAYED_NON_NEGATIVE_OUT_OF_SELECTION"
    ]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "two extra development windows per cell for the frozen v2 batch "
            "recipe, and the first out-of-selection reading of the plans the "
            "v2 rule adopted on window 1"
        ),
        "not_authorization_evidence": (
            "engineering effect measurement; no Skill is written, no Episode "
            "is formed, no Fast or Slow path is entered, and no execution "
            "right is granted or implied"
        ),
        "llm_api_call_count": 0,
        "deterministic": True,
        "adoption_rule_version": ADOPTION_RULE_VERSION,
        "adoption_rule": bch.ADOPTION_RULE_V2,
        "pre_registered": PRE_REGISTERED,
        "window_1_source": {
            "artifact": window_1["artifact"],
            "protocol_version": window_1["protocol_version"],
            "re_run": False,
            "note": (
                "read verbatim; window 1 was not re-run and no existing "
                "artifact is written by this run"
            ),
        },
        "window_definitions": window_specs,
        "traffic_window_note": (
            "traffic has exactly one triple of development origins on record "
            "(1104/1368/1800). Windows 2 and 3 keep its shape and spacing and "
            "are shifted forward inside the same pre-sealed region; they are "
            "chosen rather than quoted, they are the only chosen origins in "
            "this run, and the farthest index any of them reads is %d against "
            "sealed_from_index=%d."
            % (
                max(
                    row["farthest_index_read"]
                    for row in window_specs.get("traffic", [])
                ) if "traffic" in window_specs else 0,
                TRAFFIC_SEALED_FROM_INDEX,
            )
        ),
        "recipe_module_note": (
            "run_batch_composition_headroom is imported and not modified; the "
            "traffic window is selected by rebinding its module-level "
            "_TRAFFIC_DEVELOPMENT_ORIGINS for the duration of one call and "
            "restoring it afterwards, with _TRAFFIC_SEALED_FROM_INDEX left "
            "untouched so the module's own boundary guard stays live"
        ),
        "summary": {
            "cells": len(cells),
            "windows_per_cell": 3,
            "new_recipe_runs": sum(
                1 for cell in cells for row in cell["windows"]
                if row["window_id"] != "W1"
            ),
            "out_of_selection_pass_count": len(passing),
            "out_of_selection_pass_cells": [
                "%s x %s" % (row["cohort"], row["consumer_variant"])
                for row in passing
            ],
            "out_of_selection_fail_cells": [
                "%s x %s" % (row["cohort"], row["consumer_variant"])
                for row in cells if row not in passing
            ],
            "program_stable_cells": [
                "%s x %s" % (row["cohort"], row["consumer_variant"])
                for row in cells if row["stability"]["program_stable"]
            ],
            "mask_stable_cells": [
                "%s x %s" % (row["cohort"], row["consumer_variant"])
                for row in cells if row["stability"]["mask_stable"]
            ],
        },
        "cells": cells,
        "wall_seconds": time.perf_counter() - started,
    }
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print(
        "out_of_selection_pass %d/%d"
        % (len(passing), len(cells)), flush=True,
    )
    return 0


# --------------------------------------------------------------------- report
def _gain(value: Any) -> str:
    return "n/a" if value is None else "%+.6f" % float(value)


def _mask(values: Sequence[str]) -> str:
    return ", ".join(str(item) for item in values) or "none"


def _markdown_head(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines: list[str] = [
        "# batch recipe across development windows v1",
        "",
        "Two extra development windows per cell for the frozen v2 batch "
        "recipe, and the **first out-of-selection reading** of the plans the "
        "v2 rule adopted on window 1.",
        "",
        "**Engineering effect measurement, not authorization evidence.** No "
        "Skill is written, no Episode is formed, no Fast or Slow path is "
        "entered, and no execution right is granted or implied. 0 LLM calls, "
        "deterministic.",
        "",
        "Window 1 is `%s` (`%s`), **read verbatim and not re-run**. This run "
        "writes one new stem and overwrites nothing."
        % (
            payload["window_1_source"]["artifact"],
            payload["window_1_source"]["protocol_version"],
        ),
        "",
        "Headline: window 1's plan holds its delayed gain at or above zero on "
        "**%d of %d cells** across both new windows; the adopted program is "
        "stable across all three windows on %d cells and the exclusion mask "
        "on %d."
        % (
            summary["out_of_selection_pass_count"], summary["cells"],
            len(summary["program_stable_cells"]),
            len(summary["mask_stable_cells"]),
        ),
        "",
        "## 0. Why this is the first honest reading",
        "",
        "The v2 rule adopts a masked plan only if its delayed aggregate gain "
        "clears `max(best full-batch delayed, 0)`. That puts the delayed "
        "window **inside** the selection, which the recipe artifacts have said "
        "all along. Until now no adopted plan had been read on a window that "
        "took no part in adopting it. Section 2 is that reading.",
        "",
        "## 1. Windows and where their origins come from",
        "",
        "| cohort | window | Task | support origins | delayed origins | "
        "farthest read | origin source |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for cohort, windows in payload["window_definitions"].items():
        for window in windows:
            lines.append(
                "| %s | %s | `%s` | %s | %s | %d | %s |"
                % (
                    cohort, window["window_id"], window["task_episode_id"],
                    window["support_origins"], window["delayed_origins"],
                    window["farthest_index_read"], window["origin_source"],
                )
            )
    lines += ["", "Provenance, verbatim:", ""]
    for cohort, windows in payload["window_definitions"].items():
        for window in windows:
            lines.append(
                "- `%s` %s: %s"
                % (cohort, window["window_id"], window["origin_provenance"])
            )
    lines += [
        "",
        "> %s" % payload["traffic_window_note"],
        "",
        "> %s" % payload["recipe_module_note"],
        "",
    ]
    return lines

def _markdown_body(payload: Mapping[str, Any]) -> list[str]:
    lines: list[str] = [
        "## 2. Window 1's adopted plan, read out of selection",
        "",
        "Pre-stated: %s" % payload["pre_registered"]["out_of_selection_check"],
        "",
        "| cell | W1 plan | W1 delayed (in selection) | W2 support | "
        "W2 delayed | W3 support | W3 delayed | delayed >= 0 on both | verdict |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cell in payload["cells"]:
        rows = {row["window_id"]: row for row in cell["windows"]}
        out = {
            row["window_id"]: row
            for row in cell["window1_plan_out_of_selection"]
        }
        plan = rows["W1"]["adopted_plan"]
        w2, w3 = out.get("W2"), out.get("W3")
        lines.append(
            "| %s x %s | `%s` minus %s | %s | %s | %s | %s | %s | %s | `%s` |"
            % (
                cell["cohort"], cell["consumer_variant"],
                plan["program"], _mask(plan["excluded_series"]),
                _gain(rows["W1"]["delayed_aggregate_gain"]),
                _gain(w2["support"]["aggregate_gain"] if w2 else None),
                _gain(w2["delayed"]["aggregate_gain"] if w2 else None),
                _gain(w3["support"]["aggregate_gain"] if w3 else None),
                _gain(w3["delayed"]["aggregate_gain"] if w3 else None),
                all(row["delayed_non_negative"] for row in out.values()),
                cell["out_of_selection_verdict"],
            )
        )
    strict = "; ".join(
        "%s x %s %s"
        % (
            cell["cohort"], cell["consumer_variant"],
            " ".join(
                "%s=%s" % (
                    row["window_id"], row["delayed_above_material_threshold"],
                )
                for row in cell["window1_plan_out_of_selection"]
            ),
        )
        for cell in payload["cells"]
    )
    lines += [
        "",
        "The stricter `delayed > %.3f` reading, for information only: %s."
        % (float(bch.MATERIAL_THRESHOLD), strict),
        "",
        "## 3. What each window adopted",
        "",
        "| cell | window | kind | program | mask | support | delayed | "
        "harmed eval | adoption path |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for cell in payload["cells"]:
        for row in cell["windows"]:
            plan = row["adopted_plan"]
            lines.append(
                "| %s x %s | %s | %s | `%s` | %s | %s | %s | %d | %s |"
                % (
                    cell["cohort"], cell["consumer_variant"], row["window_id"],
                    plan["kind"], plan["program"],
                    _mask(plan["excluded_series"]),
                    _gain(row["support_aggregate_gain"]),
                    _gain(row["delayed_aggregate_gain"]),
                    row["harmed_eval_series_count"],
                    row["adoption_path"].split(";")[0][:58],
                )
            )
    lines += [
        "",
        "## 4. Stability across windows",
        "",
        "| cell | programs W1/W2/W3 | masks W1/W2/W3 | program stable | "
        "mask stable | kind stable |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in payload["cells"]:
        stability = cell["stability"]
        lines.append(
            "| %s x %s | %s | %s | %s | %s | %s |"
            % (
                cell["cohort"], cell["consumer_variant"],
                " / ".join("`%s`" % item for item in stability["programs"]),
                " / ".join(
                    "[%s]" % _mask(mask)
                    for mask in stability["excluded_series_per_window"]
                ),
                stability["program_stable"], stability["mask_stable"],
                stability["kind_stable"],
            )
        )
    lines.append("")
    return lines

def _markdown(payload: Mapping[str, Any]) -> str:
    lines = _markdown_head(payload) + _markdown_body(payload)
    lines += [
        "## 5. What this does not say",
        "",
        "- It does not authorize anything and it does not promote the v2 "
        "rule. It is one more reading of the same engineering measurement.",
        "- Three windows on one Task family per cohort is a small sample. A "
        "per-cell pass is two numbers, not a rate.",
        "- The new windows are development windows: quoted roster origins for "
        "electricity and T233, and for traffic origins inside an "
        "already-declared pre-sealed development region. Nothing sealed was "
        "opened and no claim here is a fresh-window claim.",
        "- Window 1's plan being non-negative out of selection is not the same "
        "as it being the best plan for the new window. Sections 3 and 4 show "
        "where the recipe itself would have chosen differently.",
        "",
        "## Provenance",
        "",
        "- recipe: `run_batch_composition_headroom.make_batch_recipe`, "
        "adoption_rule_version `%s`, imported and not modified"
        % payload["adoption_rule_version"],
        "- scoring: the same `_evaluate_assignment` + `_gain_rows`, identity "
        "baseline recomputed once per (cell, window)",
        "- new recipe runs in this artifact: %d"
        % payload["summary"]["new_recipe_runs"],
        "- LLM calls: %d" % payload["llm_api_call_count"],
        "- wall seconds: %.1f" % payload["wall_seconds"],
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohorts", nargs="+", default=None,
        choices=list(bch.RECIPE_V2_COHORTS),
        help="restrict the run to these cohorts (default: all three)",
    )
    args = parser.parse_args(argv)
    return run(cohorts=args.cohorts)


if __name__ == "__main__":
    raise SystemExit(main())
