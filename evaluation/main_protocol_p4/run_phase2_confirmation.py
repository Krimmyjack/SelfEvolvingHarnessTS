"""Phase 2: freeze on cohort 1, then read cohort 2 exactly once.

The module is deliberately split in two, and the split is the whole point.

``freeze_on_cohort_1`` reads **only** the cohort-1 tensor.  It builds the O0 and
O1 menus, picks Best-Fixed inside each, and trains the four Targeters.  Its
output is a plain dictionary of decisions, persisted before anything on cohort 2
is evaluated.

``confirm_on_cohort_2`` takes that frozen dictionary and evaluates it.  It never
ranks, never retrains and never re-selects: the only cohort-2 quantity it
computes is the gain of an already-chosen program on an already-chosen series.
If that discipline were relaxed, cohort 2 would become a second tuning set and
the comparison would mean nothing.

Seven arms, all scored against the same reference (empty program, identity view):

    raw, best_fixed_O0, best_fixed_O1,
    targeter_X0_O0, targeter_X1_O0, targeter_X0_O1, targeter_X1_O1

Cohort 2 is ``DEVELOPMENT_CONFIRMATION`` -- an already-exposed pool re-read under
a different data version, never fresh and never held-out.  Per-series oracle is
reported as an upper bound and is never an outcome.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import audit_cross_fitted_targeting as targeting
from evaluation.main_protocol_p4 import audit_program_repairability as p4c
from evaluation.main_protocol_p4 import natural_structure_features as x1
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import phase2_contract as contract
from evaluation.main_protocol_p4 import preflight_natural_gap_variant as preflight
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_phase2_development as dev

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEV_TENSOR = PROJECT_ROOT / "artifacts/main_protocol/p4k_phase2_development_gain.npz"
FROZEN = PROJECT_ROOT / "artifacts/main_protocol/p4l_phase2_frozen_decisions.json"
REPORT = PROJECT_ROOT / "artifacts/main_protocol/p4m_phase2_confirmation.json"
CONFIRM_TENSOR = PROJECT_ROOT / "artifacts/main_protocol/p4m_phase2_confirmation_gain.npz"
COHORT_AUDIT = PROJECT_ROOT / "artifacts/main_protocol/p4j_candidate_cohort.json"

FACES = contract.FACES
RAW = dev.EMPTY
MATERIAL = 0.005


# ------------------------------------------------------- cohort 1 (freeze) ---

def _entry(program_id: str, view: str) -> str:
    return "%s@%s" % (program_id, view)


def freeze_on_cohort_1() -> dict[str, Any]:
    """Menus, Best-Fixed and Targeters -- derived from cohort 1 alone."""
    tensor = np.load(DEV_TENSOR, allow_pickle=True)
    gain = np.asarray(tensor["gain"], dtype=np.float64)
    programs = [str(value) for value in tensor["program_ids"]]
    view_names = [str(value) for value in tensor["views"]]
    origins = [int(value) for value in tensor["origins"]]
    support = {
        "support_a": [str(v) for v in tensor["support_a"]],
        "support_b": [str(v) for v in tensor["support_b"]],
    }
    identity = view_names.index("identity")

    # program x view flattened, so a menu entry is one (program, view) pair.
    pairs = [(p, v) for p in range(len(programs)) for v in range(len(view_names))]
    flat = gain.reshape(len(pairs), *gain.shape[2:])
    mean_gain = np.nanmean(flat, axis=(1, 2, 3))

    def _menu(candidates: Sequence[int]) -> list[dict[str, Any]]:
        ranked = sorted(candidates, key=lambda r: -mean_gain[r])
        chosen: list[dict[str, Any]] = []
        for row in ranked:
            program, view = pairs[row]
            if programs[program] == RAW:
                continue
            chosen.append({
                "entry": _entry(programs[program], view_names[view]),
                "program_id": programs[program],
                "view": view_names[view],
                "row": int(row),
                "cohort_1_mean_gain": round(float(mean_gain[row]), 6),
            })
            if len(chosen) == contract.TARGETER["menu_size"]:
                break
        raw_row = pairs.index((programs.index(RAW), identity))
        chosen.append({
            "entry": _entry(RAW, "identity"), "program_id": RAW,
            "view": "identity", "row": int(raw_row), "cohort_1_mean_gain": 0.0,
        })
        return chosen

    o0_rows = [r for r, (_p, v) in enumerate(pairs) if v == identity]
    menus = {"O0": _menu(o0_rows), "O1": _menu(range(len(pairs)))}

    # Features on cohort 1, in the frozen column order.
    variant = preflight.load_variant()
    features = _feature_blocks(variant, support, origins)

    frozen: dict[str, Any] = {
        "cohort": "cohort_1",
        "origins": origins,
        "menus": menus,
        "best_fixed": {},
        "targeters": {},
    }
    for space, menu in menus.items():
        rows = [item["row"] for item in menu]
        block = flat[rows]                       # menu x origin x face x series
        per_entry = np.nanmean(block, axis=(1, 2, 3))
        best = int(np.argmax(per_entry))
        frozen["best_fixed"][space] = {
            **menu[best],
            "chosen_because": "highest mean per-series gain on cohort 1",
        }
        labels = np.argmax(
            block.reshape(len(rows), -1), axis=0
        )                                        # one label per cohort-1 cell
        for feature_set in ("X0", "X1"):
            design = features[feature_set]
            model = _fit_targeter(design, labels)
            frozen["targeters"]["%s_%s" % (feature_set, space)] = {
                "feature_set": feature_set,
                "space": space,
                "trained_on_cells": int(design.shape[0]),
                "distinct_labels": int(len(set(labels.tolist()))),
                "structure": dict(contract.TARGETER),
                "label_histogram": {
                    menu[index]["entry"]: int((labels == index).sum())
                    for index in sorted(set(labels.tolist()))
                },
                "_model": model,
            }
    frozen["feature_names"] = features["names"]
    return frozen


def _feature_blocks(variant: Mapping[str, np.ndarray],
                    support: Mapping[str, Sequence[str]],
                    origins: Sequence[int]) -> dict[str, Any]:
    """X0 and X1 design matrices, cells ordered origin-major then face."""
    x0_rows, x1_rows, names0 = [], [], None
    for origin in origins:
        for face in FACES:
            uids = list(support[face])
            block, names0 = targeting.series_features(variant, uids, int(origin))
            x0_rows.append(block)
            contexts = [
                np.asarray(
                    variant[uid][int(origin) - contract.CONTEXT:int(origin)],
                    dtype=np.float64,
                )
                for uid in uids
            ]
            extra, _names = x1.matrix(contexts, period=forecast_p1.PERIOD)
            x1_rows.append(np.hstack([block, extra]))
    return {
        "X0": np.vstack(x0_rows),
        "X1": np.vstack(x1_rows),
        "names": {
            "X0": list(names0),
            "X1": list(names0) + list(x1.FEATURE_NAMES),
        },
    }


def _fit_targeter(design: np.ndarray, labels: np.ndarray) -> Any:
    if len(set(labels.tolist())) < 2:
        return None
    return DecisionTreeClassifier(
        max_depth=contract.TARGETER["max_depth"],
        min_samples_leaf=contract.TARGETER["min_samples_leaf"],
        random_state=contract.TARGETER["random_state"],
    ).fit(design, labels)


# --------------------------------------------------- cohort 2 (confirm once) ---

def _cohort_2_cell(variant: Mapping[str, np.ndarray]) -> Any:
    payload = json.loads(COHORT_AUDIT.read_text(encoding="utf-8"))
    selection = payload["cohort_2_selection_cell"]
    support_a = tuple(selection["support_a"])
    support_b = tuple(selection["support_b"])
    values = {uid: variant[uid] for uid in (*support_a, *support_b)}
    return forecast_p1.ForecastCell(
        values=values, support_a=support_a, support_b=support_b,
        observation_block=np.asarray(
            values[support_a[0]][:forecast_p1.ORIGIN], dtype=np.float64
        ),
    ), support_a, support_b


def confirm_on_cohort_2(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen decisions.  No ranking, no retraining, no reselection."""
    started = time.time()
    variant = preflight.load_variant()
    cell, support_a, support_b = _cohort_2_cell(variant)
    origins = list(contract.ORIGINS)

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for menu in frozen["menus"].values():
        for item in menu:
            if item["entry"] not in seen:
                seen.add(item["entry"])
                entries.append(item)

    by_id = {p["program_id"]: p for p in dev.frozen_program_space()}
    gains = np.full(
        (len(entries), len(origins), len(FACES), len(support_a)),
        np.nan, dtype=np.float64,
    )
    fits = 0
    for o_index, origin in enumerate(origins):
        at = forecast_p4._cell_at(cell, int(origin))
        config = forecast_p4._config(int(origin))
        for f_index, face in enumerate(FACES):
            roster = at.roster(face)
            reference = views.representation_evaluate(
                roster, at.values, None, config,
                origin=int(origin), view=views.IdentityView(),
            )
            fits += 1
            base = np.asarray(reference["per_view_smase"], dtype=np.float64)
            executor = p4c.ScopeExecutor(
                roster, at.values, config,
                evaluate_fn=views.forecast_runtime._evaluate,
                max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION,
            )
            for e_index, item in enumerate(entries):
                steps = tuple(
                    (str(s["op"]), dict(s["params"]))
                    for s in by_id[item["program_id"]]["steps"]
                )
                compiled = None
                if steps:
                    if not executor.verify(steps, int(origin)).passed:
                        continue
                    compiled = executor._compiled(steps)
                try:
                    reading = views.representation_evaluate(
                        roster, at.values, compiled, config,
                        origin=int(origin),
                        view=dev.VIEW_BY_NAME[item["view"]],
                    )
                except Exception:  # noqa: BLE001 - a refusal stays NaN
                    continue
                fits += 1
                gains[e_index, o_index, f_index, :] = (
                    base - np.asarray(reading["per_view_smase"], dtype=np.float64)
                )

    support = {"support_a": list(support_a), "support_b": list(support_b)}
    features = _feature_blocks(variant, support, origins)
    index_of = {item["entry"]: index for index, item in enumerate(entries)}

    blocks: dict[str, np.ndarray] = {"raw": np.zeros_like(gains[0])}
    arms: dict[str, Any] = {
        "raw": _score(blocks["raw"], origins=origins)
    }
    for space in ("O0", "O1"):
        best = frozen["best_fixed"][space]
        blocks["best_fixed_%s" % space] = gains[index_of[best["entry"]]]
        arms["best_fixed_%s" % space] = {
            **_score(blocks["best_fixed_%s" % space], origins=origins),
            "choice": best["entry"],
        }
    for name, targeter in frozen["targeters"].items():
        feature_set, space = targeter["feature_set"], targeter["space"]
        menu = frozen["menus"][space]
        model = targeter["_model"]
        design = features[feature_set]
        if model is None:
            picks = np.zeros(design.shape[0], dtype=int)
        else:
            picks = model.predict(design).astype(int)
        selected = np.array([index_of[menu[int(p)]["entry"]] for p in picks])
        shaped = selected.reshape(len(origins), len(FACES), len(support_a))
        chosen = np.take_along_axis(gains, shaped[None, :, :, :], axis=0)[0]
        coverage = float(np.mean([
            menu[int(p)]["program_id"] != RAW for p in picks
        ]))
        blocks["targeter_%s_%s" % (feature_set, space)] = chosen
        arms["targeter_%s_%s" % (feature_set, space)] = {
            **_score(chosen, origins=origins),
            "deployment_coverage": round(coverage, 4),
            "pick_histogram": {
                menu[int(p)]["entry"]: int((picks == p).sum())
                for p in sorted(set(picks.tolist()))
            },
        }
    blocks["per_series_oracle"] = np.nanmax(gains, axis=0)
    oracle = _score(blocks["per_series_oracle"], origins=origins)
    np.savez_compressed(
        CONFIRM_TENSOR,
        menu_gain=gains,
        entries=np.array([item["entry"] for item in entries], dtype=object),
        origins=np.array(origins, dtype=np.int64),
        faces=np.array(list(FACES), dtype=object),
        support_a=np.array(support_a, dtype=object),
        support_b=np.array(support_b, dtype=object),
        **{"arm__%s" % name: block for name, block in blocks.items()},
    )
    return {
        "arms": arms,
        "per_series_oracle_upper_bound": oracle,
        "consumer_fits": fits,
        "entries_evaluated": [item["entry"] for item in entries],
        "wall_seconds": round(time.time() - started, 1),
    }


def _budget(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean_gain": None, "clears_bounded_budget": False,
                "cells_scored": 0}
    harmed = float((finite < -MATERIAL).mean())
    worst = max(0.0, float(-finite.min()))
    mean = float(finite.mean())
    return {
        "mean_gain": round(mean, 6),
        "harmed_fraction": round(harmed, 4),
        "max_single_series_harm": round(worst, 6),
        "cells_scored": int(finite.size),
        "clears_bounded_budget": bool(
            mean >= MATERIAL
            and harmed <= bounded.BOUNDED_MAX_HARMED_FRACTION
            and worst <= bounded.BOUNDED_MAX_SINGLE_SERIES_HARM
        ),
    }


def _score(block: np.ndarray, *,
           origins: Sequence[int] | None = None) -> dict[str, Any]:
    """Two granularities, because they are different estimands.

    ``pooled`` treats all origins x faces x series as one deployment population.
    ``per_cell`` applies the budget inside each (origin, face) exactly as P4B and
    P4D did, so the two lines can be read against each other.  A pooled failure
    can hide per-cell successes and vice versa; reporting only one of them was
    the reason an earlier reading looked inconsistent with P4D.
    """
    values = np.asarray(block, dtype=np.float64)
    summary = _budget(values)
    if origins is None or values.ndim != 3:
        return summary
    cells = []
    for o_index, origin in enumerate(origins):
        for f_index, face in enumerate(FACES):
            cells.append({
                "origin": int(origin), "face": face,
                **_budget(values[o_index, f_index, :]),
            })
    summary["per_cell"] = cells
    summary["cells_clearing_budget"] = sum(
        1 for cell in cells if cell["clears_bounded_budget"]
    )
    summary["cells_total"] = len(cells)
    return summary


def _plain(frozen: Mapping[str, Any]) -> dict[str, Any]:
    """The frozen decisions without the fitted estimators, for the receipt."""
    return {
        key: (
            {
                name: {k: v for k, v in entry.items() if k != "_model"}
                for name, entry in value.items()
            }
            if key == "targeters" else value
        )
        for key, value in frozen.items()
    }


def main() -> int:
    state = contract.assert_frozen()
    if not state["frozen"]:
        raise RuntimeError("Phase-2 contract drifted: %s" % state["failures"])
    frozen = freeze_on_cohort_1()
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(
        json.dumps({
            "stage": "P4L_PHASE2_FROZEN_DECISIONS",
            "written_at": datetime.now().astimezone().isoformat(),
            "derived_from": "cohort 1 only",
            "cohort_2_read_by_this_step": False,
            **_plain(frozen),
        }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("frozen menus written to %s" % FROZEN.relative_to(PROJECT_ROOT).as_posix())
    for space, menu in frozen["menus"].items():
        print("  %s menu: %s" % (space, [item["entry"] for item in menu]))
        print("  %s best-fixed: %s" % (
            space, frozen["best_fixed"][space]["entry"]))

    result = confirm_on_cohort_2(frozen)
    report = {
        "stage": "P4M_PHASE2_CONFIRMATION",
        "status": "COMPLETE",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DEVELOPMENT_CONFIRMATION_COHORT_HOLDOUT",
        "data_version": contract.DATA_VERSION,
        "cohort": "cohort_2",
        "cohort_role": contract.COHORTS["cohort_2"]["role"],
        "not_fresh_because": contract.COHORTS["cohort_2"]["not_fresh_because"],
        "statistical_unit": contract.STATISTICAL_UNIT,
        "frozen_decisions": FROZEN.relative_to(PROJECT_ROOT).as_posix(),
        "selection_used_cohort_2_outcome": False,
        "frozen_contract": state,
        "gain_reference": contract.GAIN_REFERENCE,
        "boundary": {**contract.BOUNDARY, "consumer_fits": result["consumer_fits"]},
        "arms": result["arms"],
        "per_series_oracle_upper_bound": result["per_series_oracle_upper_bound"],
        "oracle_note": contract.ORACLE_IS_NOT_A_RESULT,
        "entries_evaluated": result["entries_evaluated"],
        "wall_seconds": result["wall_seconds"],
        "releases": "NONE",
    }
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print()
    print("pooled over %d cells; per-cell applies the budget inside each "
          "(origin, face) as P4B/P4D did" % 10)
    print("%-22s %10s %9s %9s %9s %8s %9s" % (
        "arm", "mean gain", "harmed", "max harm", "coverage", "pooled",
        "per-cell"))
    for name in contract.ARMS:
        arm = report["arms"].get(name)
        if arm is None:
            continue
        print("%-22s %+10.6f %9.2f %9.4f %9s %8s %9s" % (
            name, arm["mean_gain"], arm["harmed_fraction"],
            arm["max_single_series_harm"],
            arm.get("deployment_coverage", "-"),
            arm["clears_bounded_budget"],
            "%d/%d" % (arm.get("cells_clearing_budget", 0),
                       arm.get("cells_total", 0))))
    oracle = report["per_series_oracle_upper_bound"]
    print("%-22s %+10.6f  (upper bound, not a result)" % (
        "per-series oracle", oracle["mean_gain"]))
    print("consumer fits : %d in %.1f min" % (
        result["consumer_fits"], result["wall_seconds"] / 60))
    print("wrote %s" % REPORT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
