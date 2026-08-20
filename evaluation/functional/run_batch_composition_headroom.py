"""batch-composition-headroom: how much is left on the table by treating a
whole batch with one program.

Zero LLM, fully deterministic.  The project's primary readout has been
simplified to a batch question: after the Harness processes a batch, is the
downstream effect better than (a) doing nothing at all and (b) the best single
program applied to everything?  This runner measures the *headroom* of
per-series selective treatment.  If the same program helps some series and
hurts others, then picking the best program per series -- identity included --
should beat any uniform full-batch treatment on the aggregate readout.

This is an engineering effect measurement.  It is not authorization evidence:
nothing here writes a Skill, forms an Episode, or touches the Fast/Slow path.
The per-series argmax uses the same Support outcomes it is scored on, so the
Support column is an upper bound on selective headroom, not a deployable
policy.  The delayed column is the honest out-of-selection readout.

Everything measurable is reused verbatim:

* cohorts            -- ``agentic.runner.load_cohort`` (exposed development
                        cohorts only; nothing sealed is reachable);
* Consumer + Judge   -- ``run_e2_autonomous_natural_workflow_generation._evaluate``
                        through ``task_episode_harness.runner._evaluate_origins``;
* windows            -- the frozen Task roster's Task 01 Support and delayed
                        origins;
* compile path       -- ``task_episode_harness.runner._compiled``.

The one thing that did not already exist is a single retrain under a
*per-series* program assignment: ``_evaluate`` takes one compiled workflow and
a scope set, which cannot express "series A gets winsorize, series B gets
identity".  :func:`_evaluate_assignment` is that generalization, built out of
the same v6 primitives, and every run checks it reproduces ``_evaluate``
exactly on the uniform assignments.

``--mode masked`` runs the follow-up experiment, masked-single-program, on the
same executor: one program applied to the whole batch, then a greedy
harm-ordered exclusion mask where every single revert is validated by a real
retrain.  It exists because the composition verdict was
``COMPOSITION_NO_HEADROOM`` for a specific reason -- the per-series argmax
trusted an additive credit signal the pooled Consumer does not honour -- and a
mask search never adds anything up.  Selection reads the Support window only;
the delayed window is evaluated for each accepted mask and reported.

``--mode recipe`` is the productised entry, :func:`make_batch_recipe`.  It runs
the menu scan and the mask search in order and then applies one frozen adoption
rule -- a masked plan is adopted only if it also holds up on the delayed window
-- to emit a single adopted plan for the batch.  It is a capability entry, not
a new experiment: the rule was read off the masked run and has no tunable
threshold.  Its cost is stated in every artifact it writes, because the delayed
window now participates in selection.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import run_e2_autonomous_natural_workflow_generation as v6
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.agentic.runner import (
    load_cohort as _load_exposed_cohort,
)
from evaluation.functional.task_episode_harness.e1 import (
    MATERIAL_THRESHOLD,
    _frozen_task_roster,
)
from evaluation.functional.task_episode_harness.runner import (
    _compiled,
    _evaluate_origins,
)

PROTOCOL_VERSION = "batch_composition_headroom_v1"
REPORT_JSON = (
    PROJECT_ROOT / "artifacts/functional/e2" / "batch_composition_headroom_v1.json"
)
REPORT_MD = (
    PROJECT_ROOT / "artifacts/functional/e2" / "batch_composition_headroom_v1.md"
)

# masked-single-program: the low-dimensional variant of the same question.
MASKED_PROTOCOL_VERSION = "masked_single_program_v1"
MASKED_REPORT_JSON = (
    PROJECT_ROOT / "artifacts/functional/e2" / "masked_single_program_v1.json"
)
MASKED_REPORT_MD = (
    PROJECT_ROOT / "artifacts/functional/e2" / "masked_single_program_v1.md"
)
M0A_CENSUS = (
    PROJECT_ROOT / "artifacts/functional/e2" / "m0a_mask_geometry_census_v1.json"
)
# How many programs get a mask search: the cohort's top two by full-batch
# Support aggregate gain.
MASKED_PROGRAM_COUNT = 2
# One revert per step, at most one pass over the batch.
MASKED_MAX_STEPS = 12
# The frozen M0a geometry fields quoted in the descriptive contrast.  Read from
# the census artifact, never recomputed here.
M0A_GEOMETRY_FIELDS: tuple[str, ...] = (
    "outlier_region_fraction",
    "level_region_fraction",
    "outlier_region_end_fraction",
    "level_region_end_fraction",
    "union_region_fraction",
    "union_region_end_fraction",
    "outlier_point_fraction",
    "local_robust_z_peak",
    "level_excursion_score",
)

# The menu.  Eight entries, identity plus seven zero-parameter operators that
# already exist in the registry and already compile on the current path.  It is
# written down here rather than derived so the run cannot quietly widen.
IDENTITY = "identity"
PROGRAM_MENU: tuple[str, ...] = (
    IDENTITY,
    "repair_level_shift",
    "hampel_filter",
    "outlier_iqr",
    "outlier_mad",
    "winsorize",
    "denoise_median",
    "smooth_ma",
)
TREATMENTS: tuple[str, ...] = tuple(op for op in PROGRAM_MENU if op != IDENTITY)

COHORTS: tuple[str, ...] = ("electricity", "T233")
CONSUMER_POOLED = "pooled"
CONSUMER_PER_CHANNEL = "per_channel"
CONSUMER_VARIANTS: tuple[str, ...] = (CONSUMER_POOLED, CONSUMER_PER_CHANNEL)
CCR_JSON = (
    PROJECT_ROOT / "artifacts/functional/e2" / "consumer_conditioned_recipe_v1.json"
)
CCR_MD = (
    PROJECT_ROOT / "artifacts/functional/e2" / "consumer_conditioned_recipe_v1.md"
)
RECIPE_V2_COHORTS: tuple[str, ...] = ("electricity", "T233", "traffic")
RECIPE_V2_ALL_CELLS_JSON = (
    PROJECT_ROOT / "artifacts/functional/e2" / "batch_recipe_v2_all_cells_v1.json"
)
RECIPE_V2_ALL_CELLS_MD = (
    PROJECT_ROOT / "artifacts/functional/e2" / "batch_recipe_v2_all_cells_v1.md"
)

# traffic is structurally accepted (screening v2) but not in
# agentic.runner.load_cohort.  Roster is the frozen screening 12/8 split;
# windows stay on development origins and never cross sealed_from_index.
_TRAFFIC_TRAIN: tuple[str, ...] = tuple(str(i) for i in range(12))
_TRAFFIC_EVAL: tuple[str, ...] = tuple(str(i) for i in range(12, 20))
_TRAFFIC_DEVELOPMENT_ORIGINS: tuple[int, ...] = (1104, 1368, 1800)
_TRAFFIC_SEALED_FROM_INDEX = 3072


def _traffic_csv_path() -> Path:
    candidates = (
        Path(r"C:/Users/辉/desktop/agent/shared_tsq_datasets")
        / "traffic/traffic.csv",
        Path("/mnt/c/Users/辉/desktop/agent/shared_tsq_datasets/traffic/traffic.csv"),
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "traffic.csv not found under shared_tsq_datasets"
    )


def load_cohort(repo_root: Path, name: str) -> dict[str, Any]:
    """Recipe-local loader.  Delegates to agentic.runner except for traffic."""
    if name != "traffic":
        return _load_exposed_cohort(repo_root, name)
    from evaluation.functional.task_episode_harness.agentic import g3_sourcing
    from evaluation.functional.task_episode_harness.runner import _mapped_roster

    all_names, values = g3_sourcing.load_csv_columns(_traffic_csv_path())
    train = list(_TRAFFIC_TRAIN)
    evaluation = list(_TRAFFIC_EVAL)
    needed = train + evaluation
    missing = [uid for uid in needed if uid not in values]
    if missing:
        raise RuntimeError(
            f"traffic screening roster missing from CSV columns: {missing}"
        )
    values = {uid: values[uid] for uid in needed}
    roster = (
        [{"series_uid": uid, "role": "train"} for uid in train]
        + [{"series_uid": uid, "role": "eval"} for uid in evaluation]
    )
    return {
        "name": name,
        "roster": roster,
        "mapped_roster": _mapped_roster(roster),
        "values": values,
        "train_uids": train,
        "eval_uids": evaluation,
        "exposure": (
            "STRUCTURALLY_ACCEPTED_BUT_SOURCE_FAMILY_EXPOSURE_UNRESOLVED: "
            "PeMS SF Bay Area / monash:traffic_hourly family has unresolved "
            "prior exposure; this run uses development origins "
            "(1104/1368/1800) only and does not open a sealed Outcome"
        ),
    }


def _task_windows(
    cohort_name: str, task_index: int
) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    spec = _frozen_task_roster()[task_index]
    if cohort_name == "traffic":
        support = _TRAFFIC_DEVELOPMENT_ORIGINS[:2]
        delayed = _TRAFFIC_DEVELOPMENT_ORIGINS[2:]
        farthest = max(support + delayed) + int(v6.HORIZON)
        if farthest > _TRAFFIC_SEALED_FROM_INDEX:
            raise RuntimeError(
                "traffic recipe would read at or past sealed_from_index="
                f"{_TRAFFIC_SEALED_FROM_INDEX} (farthest={farthest})"
            )
        return spec, support, delayed
    return (
        spec,
        tuple(int(o) for o in spec["support_origins"]),
        tuple(int(o) for o in spec["delayed_origins"]),
    )


# --------------------------------------------------------------- assignment
def _evaluate_assignment(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    assignment: Mapping[str, Any],
    config: Mapping[str, object],
    *,
    origin: int,
    consumer_variant: str = CONSUMER_POOLED,
) -> dict[str, object]:
    """One retrain under a per-training-series program assignment.

    Line-for-line the body of ``v6._evaluate`` with a single change: the
    compiled workflow is looked up per series instead of being one workflow
    plus a scope set.  ``assignment[uid] is None`` (or a missing key) means the
    series is left at the identity baseline, which is exactly what an
    out-of-scope series gets in ``_evaluate``.

    ``consumer_variant`` is experiment-local.  ``pooled`` is the frozen
    Consumer: one ridge on the stacked training-channel windows.
    ``per_channel`` keeps the same window features and the same ridge
    (alpha=1, unpenalized intercept) and only removes that stacking: each
    training channel fits its own ridge, and each eval channel is predicted
    by the equal-weight mean of those channel-wise ridges.  Nothing outside
    this runner is changed.
    """
    from SelfEvolvingHarnessTS.evaluation.benchmark_v02.metrics import (
        seasonal_scale,
        smase,
    )

    if consumer_variant not in CONSUMER_VARIANTS:
        raise ValueError(f"unknown consumer_variant: {consumer_variant!r}")

    train_rows = [row for row in roster if row["role"] == "train"]
    eval_rows = [row for row in roster if row["role"] == "eval"]
    per_train_x: list[list[Any]] = []
    per_train_y: list[list[Any]] = []
    behavior_count = 0
    for row in train_rows:
        series_uid = str(row["series_uid"])
        compiled = assignment.get(series_uid)
        raw = np.asarray(values[series_uid], dtype=np.float64)
        series_x: list[Any] = []
        series_y: list[Any] = []
        for anchor in config["anchors"]:  # type: ignore[union-attr]
            anchor = int(anchor)
            if anchor + v6.HORIZON > origin:
                continue
            window = raw[anchor - v6.CONTEXT_LENGTH : anchor + v6.HORIZON]
            baseline = v6._linear_integrity(window)
            if compiled is not None:
                prepared, _trace = v6._apply_program(window, compiled)
            else:
                prepared = baseline
            behavior_count += int(
                np.count_nonzero(~np.isclose(prepared, baseline, equal_nan=True))
            )
            context = prepared[: v6.CONTEXT_LENGTH]
            target = prepared[v6.CONTEXT_LENGTH :]
            center, scale, method = v6._center_scale(np, context)
            if method == "scale_floor_fallback":
                raise RuntimeError("training context reached scale floor")
            series_x.append((context - center) / scale)
            series_y.append((target - center) / scale)
        per_train_x.append(series_x)
        per_train_y.append(series_y)

    x_eval: list[Any] = []
    truths: list[Any] = []
    eval_centers: list[float] = []
    eval_scales: list[float] = []
    metric_scales: list[float] = []
    for row in eval_rows:
        raw = np.asarray(values[str(row["series_uid"])], dtype=np.float64)
        window = raw[origin - v6.CONTEXT_LENGTH : origin]
        prepared = v6._linear_integrity(window)
        center, scale, method = v6._center_scale(np, prepared)
        if method == "scale_floor_fallback":
            raise RuntimeError("evaluation context reached scale floor")
        x_eval.append((prepared - center) / scale)
        truths.append(raw[origin : origin + v6.HORIZON])
        eval_centers.append(center)
        eval_scales.append(scale)
        metric_scales.append(
            seasonal_scale(
                raw[:origin],
                np.isfinite(raw[:origin]),
                period=int(config["period"]),
                min_pairs=32,
            )
        )

    x_eval_arr = np.asarray(x_eval, dtype=np.float64)
    if consumer_variant == CONSUMER_POOLED:
        x_train = np.asarray(
            [row for series in per_train_x for row in series], dtype=np.float64
        )
        y_train = np.asarray(
            [row for series in per_train_y for row in series], dtype=np.float64
        )
        prediction = v6._exact_weighted_ridge_prediction(
            np,
            x_train=x_train,
            targets=y_train,
            weights=np.ones(len(x_train), dtype=np.float64),
            x_eval=x_eval_arr,
        )
    else:
        channel_preds = [
            v6._exact_weighted_ridge_prediction(
                np,
                x_train=np.asarray(series_x, dtype=np.float64),
                targets=np.asarray(series_y, dtype=np.float64),
                weights=np.ones(len(series_x), dtype=np.float64),
                x_eval=x_eval_arr,
            )
            for series_x, series_y in zip(per_train_x, per_train_y)
        ]
        prediction = np.mean(np.stack(channel_preds, axis=0), axis=0)
    prediction = (
        prediction * np.asarray(eval_scales)[:, None]
        + np.asarray(eval_centers)[:, None]
    )
    losses: list[float] = []
    for truth, predicted, scale in zip(truths, prediction, metric_scales):
        observed = np.isfinite(truth)
        if not observed.any():
            raise RuntimeError("evaluation future contains no observed truth")
        losses.append(smase(truth[observed], predicted[observed], scale=scale))
    import statistics

    return {
        "mean_smase": float(statistics.fmean(losses)),
        "per_view_smase": [float(value) for value in losses],
        "behavior_point_count": behavior_count,
        "consumer_variant": consumer_variant,
    }


def _evaluate_variant(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    compiled: Any,
    config: Mapping[str, object],
    origins: tuple[int, ...],
    scope: set[str] | None,
    consumer_variant: str,
) -> list[dict[str, object]]:
    """Uniform-program evaluation under the chosen Consumer.

    ``pooled`` is the frozen path (``_evaluate_origins`` / v6 ``_evaluate``).
    ``per_channel`` stays inside this runner.
    """
    if consumer_variant == CONSUMER_POOLED:
        return _evaluate_origins(
            roster, values, compiled, config, origins, scope,
        )
    train_uids = [
        str(row["series_uid"]) for row in roster if row["role"] == "train"
    ]
    assignment = {
        uid: (compiled if (scope is None or uid in scope) else None)
        for uid in train_uids
    }
    return [
        _evaluate_assignment(
            roster, values, assignment, config, origin=origin,
            consumer_variant=consumer_variant,
        )
        for origin in origins
    ]


# ------------------------------------------------------------------ readout
def _gain_rows(
    identity_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    eval_uids: Sequence[str],
) -> dict[str, Any]:
    """Aggregate and per-eval-series gain, in the ``_arm_metrics`` idiom.

    Aggregate gain is the mean over origins of the mean over eval views of
    ``identity_sMASE - candidate_sMASE``; per-eval-series gain is that
    difference averaged over the origins.  Positive is better.
    """
    per_origin: list[float] = []
    per_series: dict[str, list[float]] = {uid: [] for uid in eval_uids}
    for base, candidate in zip(identity_rows, candidate_rows):
        gains = [
            float(reference - method)
            for reference, method in zip(
                base["per_view_smase"], candidate["per_view_smase"]
            )
        ]
        per_origin.append(float(np.mean(gains)))
        for uid, gain in zip(eval_uids, gains):
            per_series[uid].append(gain)
    per_series_mean = {
        uid: float(np.mean(rows)) for uid, rows in per_series.items()
    }
    harmed = {
        uid: value
        for uid, value in per_series_mean.items()
        if value < -MATERIAL_THRESHOLD
    }
    return {
        "aggregate_gain": float(np.mean(per_origin)),
        "per_origin_gain": per_origin,
        "per_eval_series_gain": per_series_mean,
        "harmed_eval_series_count": len(harmed),
        "harmed_eval_series_total_harm": float(-sum(harmed.values())),
        "harmed_eval_series": sorted(harmed),
    }


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _identity_absolute_loss(rows: Sequence[Mapping[str, Any]]) -> float:
    """Mean identity sMASE over origins then eval views.  Recorded, never a rule."""
    return float(np.mean([
        float(np.mean(row["per_view_smase"])) for row in rows
    ]))


# --------------------------------------------------------------- one cohort
def run_cohort(
    cohort_name: str,
    *,
    task_index: int = 0,
    consumer_variant: str = CONSUMER_POOLED,
) -> dict[str, Any]:
    started = time.perf_counter()
    if consumer_variant not in CONSUMER_VARIANTS:
        raise ValueError(f"unknown consumer_variant: {consumer_variant!r}")
    config = dict(_config())
    cohort = load_cohort(PROJECT_ROOT, cohort_name)
    roster = cohort["mapped_roster"]
    values = cohort["values"]
    train_uids = [str(uid) for uid in cohort["train_uids"]]
    eval_uids = [str(uid) for uid in cohort["eval_uids"]]

    spec, support, delayed = _task_windows(cohort_name, task_index)

    compiled = {
        op: _compiled(op, name=f"bch_{op}") for op in TREATMENTS
    }

    def evaluate(scope: set[str] | None, program: str, origins: tuple[int, ...]):
        return _evaluate_variant(
            roster, values,
            None if program == IDENTITY else compiled[program],
            config, origins, scope, consumer_variant,
        )

    print(f"BCH {cohort_name}: identity baseline", flush=True)
    identity_support = evaluate(None, IDENTITY, support)
    identity_delayed = evaluate(None, IDENTITY, delayed)

    # ---- 1. single-program full-batch baselines --------------------------
    full_batch: dict[str, Any] = {}
    executability: list[dict[str, Any]] = []
    for program in TREATMENTS:
        try:
            rows = evaluate(None, program, support)
        except Exception as exc:  # noqa: BLE001
            executability.append({
                "program": program, "compiles_and_runs": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        executability.append({
            "program": program, "compiles_and_runs": True,
            "modified_point_count": int(
                sum(int(row.get("behavior_point_count") or 0) for row in rows)
            ),
        })
        full_batch[program] = _gain_rows(identity_support, rows, eval_uids)
        print(
            "BCH %s full %-19s gain=%+.6f harmed_eval=%d"
            % (cohort_name, program, full_batch[program]["aggregate_gain"],
               full_batch[program]["harmed_eval_series_count"]),
            flush=True,
        )
    runnable = sorted(full_batch)
    if not runnable:
        raise RuntimeError("no menu program executed on this cohort")

    # ---- per-series response, one series treated at a time ---------------
    per_series: dict[str, dict[str, float]] = {}
    for uid in train_uids:
        row: dict[str, float] = {IDENTITY: 0.0}
        for program in runnable:
            rows = evaluate({uid}, program, support)
            row[program] = _gain_rows(
                identity_support, rows, eval_uids
            )["aggregate_gain"]
        per_series[uid] = row
        best = max(row, key=lambda op: (row[op], -PROGRAM_MENU.index(op)))
        print(
            "BCH %s series %-14s best=%-19s gain=%+.6f"
            % (cohort_name, uid, best, row[best]),
            flush=True,
        )

    # ---- 2. composition: one argmax per series, nothing else -------------
    assignment: dict[str, str] = {}
    for uid in train_uids:
        row = per_series[uid]
        # Deterministic tie-break: earlier in the frozen menu wins, so a tie
        # with identity resolves to identity and never to a treatment.
        assignment[uid] = min(
            row, key=lambda op: (-row[op], PROGRAM_MENU.index(op))
        )
    compiled_assignment = {
        uid: (None if op == IDENTITY else compiled[op])
        for uid, op in assignment.items()
    }

    # ---- 3. composition validation: one retrain under the assignment -----
    composition_support_rows = [
        _evaluate_assignment(
            roster, values, compiled_assignment, config, origin=origin,
            consumer_variant=consumer_variant,
        )
        for origin in support
    ]
    composition_support = _gain_rows(
        identity_support, composition_support_rows, eval_uids
    )

    # The generalization is only trustworthy if it collapses to the frozen
    # readout on the uniform case.  Checked against every runnable program at
    # the first Support origin, which also re-confirms the compile path.
    equivalence: list[dict[str, Any]] = []
    for program in runnable:
        mirrored = _evaluate_assignment(
            roster, values,
            {uid: compiled[program] for uid in train_uids},
            config, origin=support[0],
            consumer_variant=consumer_variant,
        )
        frozen = evaluate(None, program, (support[0],))[0]
        equivalence.append({
            "program": program,
            "exact_match": (
                mirrored["per_view_smase"] == frozen["per_view_smase"]
            ),
            "max_abs_difference": float(
                np.max(np.abs(
                    np.asarray(mirrored["per_view_smase"])
                    - np.asarray(frozen["per_view_smase"])
                ))
            ),
        })

    # ---- 4. robustness beyond Support ------------------------------------
    best_single = max(
        runnable,
        key=lambda op: (full_batch[op]["aggregate_gain"], -PROGRAM_MENU.index(op)),
    )
    best_single_delayed = _gain_rows(
        identity_delayed, evaluate(None, best_single, delayed), eval_uids
    )
    composition_delayed = _gain_rows(
        identity_delayed,
        [
            _evaluate_assignment(
                roster, values, compiled_assignment, config, origin=origin,
                consumer_variant=consumer_variant,
            )
            for origin in delayed
        ],
        eval_uids,
    )

    # ---- 5. composition interaction, reported rather than assumed --------
    sum_of_parts = float(
        sum(per_series[uid][assignment[uid]] for uid in train_uids)
    )
    interaction = composition_support["aggregate_gain"] - sum_of_parts

    # ---- harm accounting --------------------------------------------------
    def train_side_harm(plan: Mapping[str, str]) -> dict[str, Any]:
        harmed = {
            uid: per_series[uid][op]
            for uid, op in plan.items()
            if per_series[uid][op] < -MATERIAL_THRESHOLD
        }
        return {
            "harmed_training_series_count": len(harmed),
            "harmed_training_series_total_harm": float(-sum(harmed.values())),
            "harmed_training_series": sorted(harmed),
        }

    harm_table = {
        IDENTITY: {
            "plan": "no series treated",
            "aggregate_support_gain": 0.0,
            "harmed_eval_series_count": 0,
            "harmed_eval_series_total_harm": 0.0,
            **train_side_harm({uid: IDENTITY for uid in train_uids}),
        },
    }
    for program in runnable:
        harm_table[program] = {
            "plan": f"every training series treated with {program}",
            "aggregate_support_gain": full_batch[program]["aggregate_gain"],
            "harmed_eval_series_count": full_batch[program][
                "harmed_eval_series_count"],
            "harmed_eval_series_total_harm": full_batch[program][
                "harmed_eval_series_total_harm"],
            **train_side_harm({uid: program for uid in train_uids}),
        }
    harm_table["COMPOSITION"] = {
        "plan": "per-series argmax over the menu, identity included",
        "aggregate_support_gain": composition_support["aggregate_gain"],
        "harmed_eval_series_count": composition_support[
            "harmed_eval_series_count"],
        "harmed_eval_series_total_harm": composition_support[
            "harmed_eval_series_total_harm"],
        **train_side_harm(assignment),
    }

    # ---- response divergence ----------------------------------------------
    divergent: list[dict[str, Any]] = []
    for program in runnable:
        gains = {uid: per_series[uid][program] for uid in train_uids}
        positive = {u: g for u, g in gains.items() if g > MATERIAL_THRESHOLD}
        negative = {u: g for u, g in gains.items() if g < -MATERIAL_THRESHOLD}
        if not (positive and negative):
            continue
        best_uid = max(positive, key=lambda u: positive[u])
        worst_uid = min(negative, key=lambda u: negative[u])
        divergent.append({
            "program": program,
            "positive_series_count": len(positive),
            "negative_series_count": len(negative),
            "best_series": best_uid,
            "best_series_gain": positive[best_uid],
            "worst_series": worst_uid,
            "worst_series_gain": negative[worst_uid],
            "spread": positive[best_uid] - negative[worst_uid],
        })
    divergent.sort(key=lambda row: -row["spread"])

    # ---- verdict ----------------------------------------------------------
    beats_best_single = (
        composition_support["aggregate_gain"]
        - full_batch[best_single]["aggregate_gain"]
    )
    harm_not_worse = (
        composition_support["harmed_eval_series_count"]
        <= full_batch[best_single]["harmed_eval_series_count"]
    )
    if not divergent:
        verdict = "RESPONSES_HOMOGENEOUS"
    elif beats_best_single > MATERIAL_THRESHOLD and harm_not_worse:
        verdict = "SELECTIVE_COMPOSITION_HEADROOM_PRESENT"
    else:
        verdict = "COMPOSITION_NO_HEADROOM"

    return {
        "cohort": cohort_name,
        "consumer_variant": consumer_variant,
        "exposure": cohort["exposure"],
        "verdict": verdict,
        "task_episode_id": str(spec["task_episode_id"]),
        "support_origins": list(support),
        "delayed_origins": list(delayed),
        "train_series": train_uids,
        "eval_series": eval_uids,
        "program_menu": list(PROGRAM_MENU),
        "program_executability": executability,
        "readout_equivalence_check": {
            "claim": (
                "_evaluate_assignment reproduces the frozen _evaluate readout "
                "exactly on every uniform assignment"
                if consumer_variant == CONSUMER_POOLED
                else (
                    "_evaluate_assignment reproduces _evaluate_variant "
                    "exactly on every uniform assignment under per_channel"
                )
            ),
            "all_exact": all(row["exact_match"] for row in equivalence),
            "rows": equivalence,
        },
        "single_program_full_batch": {
            program: full_batch[program] for program in runnable
        },
        "best_single_program": best_single,
        "per_training_series_response": per_series,
        "assignment": assignment,
        "assignment_table": [
            {
                "series_uid": uid,
                "chosen_program": assignment[uid],
                "chosen_per_series_gain": per_series[uid][assignment[uid]],
                "identity_gain": 0.0,
                "best_single_program_gain_on_this_series": per_series[uid][
                    best_single],
                "all_program_gains": per_series[uid],
            }
            for uid in train_uids
        ],
        "headline": {
            "support": {
                "identity": 0.0,
                "best_single_program": full_batch[best_single][
                    "aggregate_gain"],
                "composition": composition_support["aggregate_gain"],
                "composition_minus_best_single": beats_best_single,
            },
            "delayed": {
                "identity": 0.0,
                "best_single_program": best_single_delayed["aggregate_gain"],
                "composition": composition_delayed["aggregate_gain"],
                "composition_minus_best_single": (
                    composition_delayed["aggregate_gain"]
                    - best_single_delayed["aggregate_gain"]
                ),
                "ordering_preserved": (
                    composition_delayed["aggregate_gain"]
                    > best_single_delayed["aggregate_gain"] > 0.0
                ),
                "window_role": (
                    "the delayed origins of the same frozen Task; already "
                    "exposed development data, no sealed Outcome opened"
                ),
                "selection_independence": (
                    "the per-series argmax was fixed on Support alone and is "
                    "not re-fit here, so this column is out-of-selection"
                ),
            },
        },
        "composition_detail": {
            "support": composition_support,
            "delayed": composition_delayed,
            "best_single_delayed": best_single_delayed,
        },
        "composition_interaction": {
            "sum_of_chosen_per_series_gains": sum_of_parts,
            "validated_composition_gain": composition_support["aggregate_gain"],
            "interaction": interaction,
            "note": (
                "The two differ because the Consumer is retrained once on the "
                "whole treated batch; per-series gains are not additive and "
                "were never summed to produce the validated number."
            ),
        },
        "harm_account": harm_table,
        "response_divergence": divergent,
        "strongest_divergence": divergent[0] if divergent else None,
        "wall_seconds": time.perf_counter() - started,
    }


# ------------------------------------------- masked single program (follow-up)
def _m0a_rows(cohort_name: str, task_episode_id: str) -> dict[str, dict[str, Any]]:
    """The frozen M0a census rows for this cohort and Task, keyed by series.

    Read-only.  Nothing is recomputed and no threshold is fitted; the geometry
    fields exist here only to describe which series the mask search removed.
    """
    census = json.loads(M0A_CENSUS.read_text(encoding="utf-8"))
    return {
        str(row["series_uid"]): row
        for row in census["rows"]
        if str(row["cohort"]) == cohort_name
        and str(row["task_episode_id"]) == task_episode_id
    }


def _geometry_contrast(
    census: Mapping[str, Mapping[str, Any]],
    excluded: Sequence[str],
    retained: Sequence[str],
) -> dict[str, Any]:
    """Descriptive only.  What the removed series look like, geometrically.

    No verdict is derived from this and no threshold is fitted.  It exists so a
    later reader can see whether the series a real-retrain search chose to drop
    have anything observable in common.
    """
    def group(uids: Sequence[str]) -> dict[str, Any]:
        rows = [census[uid] for uid in uids if uid in census]
        classes: dict[str, int] = {}
        for row in rows:
            key = str(row.get("mask_class"))
            classes[key] = classes.get(key, 0) + 1
        fields: dict[str, Any] = {}
        for field in M0A_GEOMETRY_FIELDS:
            observed = [
                float(row[field]) for row in rows
                if isinstance(row.get(field), (int, float))
            ]
            fields[field] = {
                "mean": float(np.mean(observed)) if observed else None,
                "min": float(np.min(observed)) if observed else None,
                "max": float(np.max(observed)) if observed else None,
            }
        return {
            "series": list(uids),
            "series_count": len(uids),
            "census_rows_found": len(rows),
            "mask_class_counts": classes,
            "post_shift_support_sufficient_true_count": sum(
                1 for row in rows if bool(row.get("union_pss"))
            ),
            "fields": fields,
        }

    excluded_group = group(excluded)
    retained_group = group(retained)
    separation: list[dict[str, Any]] = []
    for field in M0A_GEOMETRY_FIELDS:
        left = excluded_group["fields"][field]
        right = retained_group["fields"][field]
        if left["mean"] is None or right["mean"] is None:
            continue
        overlaps = not (
            left["min"] > right["max"] or left["max"] < right["min"]
        )
        separation.append({
            "field": field,
            "excluded_mean": left["mean"],
            "retained_mean": right["mean"],
            "excluded_range": [left["min"], left["max"]],
            "retained_range": [right["min"], right["max"]],
            "observed_ranges_overlap": overlaps,
            "direction": (
                "excluded_higher" if left["mean"] > right["mean"]
                else "excluded_lower"
            ),
        })
    return {
        "reading": (
            "descriptive contrast only; no threshold is fitted, no Observation "
            "is wired, and nothing here contributes to the verdict"
        ),
        "provenance": (
            "frozen M0a census artifact "
            "artifacts/functional/e2/m0a_mask_geometry_census_v1.json, "
            "read verbatim"
        ),
        "excluded": excluded_group,
        "retained": retained_group,
        "fields_with_non_overlapping_observed_ranges": [
            row["field"] for row in separation
            if not row["observed_ranges_overlap"]
        ],
        "per_field": separation,
    }


def run_masked_cohort(
    cohort_name: str,
    *,
    task_index: int = 0,
    consumer_variant: str = CONSUMER_POOLED,
) -> dict[str, Any]:
    """Greedy harm-driven exclusion masks over one single program at a time.

    The composition experiment failed because the per-series argmax trusted an
    additive credit signal that the pooled Consumer does not honour.  This
    variant never sums anything: it starts from one program applied to the whole
    batch, reverts one treated series to identity at a time, and re-reads the
    Support aggregate with a real retrain after every single revert.  A step is
    kept only if the measured aggregate improved.

    Pre-declared before any number was read, and honoured in the code below:
    selection looks at the Support window only.  The delayed window is
    evaluated for every accepted mask and reported, and it never steers the
    search.
    """
    started = time.perf_counter()
    if consumer_variant not in CONSUMER_VARIANTS:
        raise ValueError(f"unknown consumer_variant: {consumer_variant!r}")
    config = dict(_config())
    cohort = load_cohort(PROJECT_ROOT, cohort_name)
    roster = cohort["mapped_roster"]
    values = cohort["values"]
    train_uids = [str(uid) for uid in cohort["train_uids"]]
    eval_uids = [str(uid) for uid in cohort["eval_uids"]]

    spec, support, delayed = _task_windows(cohort_name, task_index)
    task_id = str(spec["task_episode_id"])

    compiled = {op: _compiled(op, name=f"bch_{op}") for op in TREATMENTS}

    def frozen(program: str, origins: tuple[int, ...], scope: set[str] | None = None):
        return _evaluate_variant(
            roster, values,
            None if program == IDENTITY else compiled[program],
            config, origins, scope, consumer_variant,
        )

    def masked(program: str, excluded: set[str], origins: tuple[int, ...]):
        assignment = {
            uid: (None if uid in excluded else compiled[program])
            for uid in train_uids
        }
        return [
            _evaluate_assignment(
                roster, values, assignment, config, origin=origin,
                consumer_variant=consumer_variant,
            )
            for origin in origins
        ]

    identity_support = frozen(IDENTITY, support)
    identity_delayed = frozen(IDENTITY, delayed)

    # ---- full-batch baselines, and the top-two programs by Support ---------
    full_support: dict[str, Any] = {}
    for program in TREATMENTS:
        try:
            full_support[program] = _gain_rows(
                identity_support, frozen(program, support), eval_uids
            )
        except Exception as exc:  # noqa: BLE001
            print(f"BCH-M {cohort_name} {program} unavailable: {exc}", flush=True)
    ranked = sorted(
        full_support,
        key=lambda op: (-full_support[op]["aggregate_gain"], PROGRAM_MENU.index(op)),
    )
    searched = ranked[:MASKED_PROGRAM_COUNT]
    best_full_program = ranked[0]
    full_delayed: dict[str, Any] = {}
    for program in TREATMENTS:
        if program not in full_support:
            continue
        full_delayed[program] = _gain_rows(
            identity_delayed, frozen(program, delayed), eval_uids
        )

    # ---- per-series harm under full treatment: the revert ordering only ----
    per_series: dict[str, dict[str, float]] = {}
    for program in searched:
        row: dict[str, float] = {}
        for uid in train_uids:
            row[uid] = _gain_rows(
                identity_support, frozen(program, support, {uid}), eval_uids
            )["aggregate_gain"]
        per_series[program] = row

    # ---- greedy exclusion, every step validated by a real retrain ----------
    searches: dict[str, Any] = {}
    for program in searched:
        order = sorted(train_uids, key=lambda uid: per_series[program][uid])
        excluded: set[str] = set()
        current = full_support[program]
        steps: list[dict[str, Any]] = []
        accepted_masks: list[dict[str, Any]] = [{
            "step": 0,
            "excluded": [],
            "support": full_support[program],
            "delayed": full_delayed[program],
            "origin_of_this_row": "full-batch start, no series reverted",
        }]
        for step, uid in enumerate(order[:MASKED_MAX_STEPS], start=1):
            trial = excluded | {uid}
            trial_support = _gain_rows(
                identity_support, masked(program, trial, support), eval_uids
            )
            improved = (
                trial_support["aggregate_gain"] > current["aggregate_gain"]
            )
            steps.append({
                "step": step,
                "reverted_series": uid,
                "its_singleton_gain_used_only_for_ordering": per_series[
                    program][uid],
                "excluded_after_this_step": sorted(trial),
                "support_aggregate_gain": trial_support["aggregate_gain"],
                "previous_support_aggregate_gain": current["aggregate_gain"],
                "delta": (
                    trial_support["aggregate_gain"] - current["aggregate_gain"]
                ),
                "decision": "ACCEPTED" if improved else "REJECTED_AND_STOPPED",
            })
            print(
                "BCH-M %s %-12s step%02d revert %-8s support=%+.6f (%+.6f) %s"
                % (cohort_name, program, step, uid,
                   trial_support["aggregate_gain"],
                   trial_support["aggregate_gain"] - current["aggregate_gain"],
                   "accept" if improved else "reject/stop"),
                flush=True,
            )
            if not improved:
                break
            excluded = trial
            current = trial_support
            # Delayed is read for every accepted mask and never fed back.
            accepted_masks.append({
                "step": step,
                "excluded": sorted(excluded),
                "support": trial_support,
                "delayed": _gain_rows(
                    identity_delayed, masked(program, excluded, delayed),
                    eval_uids,
                ),
                "origin_of_this_row": "accepted mask",
            })
        searches[program] = {
            "revert_order": order,
            "revert_order_basis": (
                "ascending singleton per-series gain under this program; a "
                "heuristic ordering only -- it decides what to try next and "
                "never what to keep"
            ),
            "steps": steps,
            "accepted_step_count": sum(
                1 for row in steps if row["decision"] == "ACCEPTED"
            ),
            "accepted_masks": accepted_masks,
            "final_excluded": sorted(excluded),
            "final_support": current,
            "final_delayed": accepted_masks[-1]["delayed"],
        }

    # ---- the cohort's best masked plan ------------------------------------
    best_program = max(
        searches,
        key=lambda op: (
            searches[op]["final_support"]["aggregate_gain"],
            -PROGRAM_MENU.index(op),
        ),
    )
    best = searches[best_program]
    best_support = best["final_support"]
    best_delayed = best["final_delayed"]
    bar_support = full_support[best_full_program]
    bar_delayed = full_delayed[best_full_program]

    any_accepted = any(
        searches[op]["accepted_step_count"] > 0 for op in searches
    )
    beats_support = (
        best_support["aggregate_gain"] - bar_support["aggregate_gain"]
    )
    beats_delayed = (
        best_delayed["aggregate_gain"] - bar_delayed["aggregate_gain"]
    )
    harm_smaller = (
        best_support["harmed_eval_series_count"]
        <= bar_support["harmed_eval_series_count"]
        and best_support["harmed_eval_series_total_harm"]
        < bar_support["harmed_eval_series_total_harm"]
    )
    if not any_accepted:
        verdict = "FULL_BATCH_IS_CEILING"
    elif (
        beats_support > MATERIAL_THRESHOLD
        and beats_delayed >= 0.0
        and harm_smaller
    ):
        verdict = "MASKED_SINGLE_PROGRAM_IMPROVES"
    elif beats_support > 0.0:
        verdict = "MASKED_IMPROVES_SUPPORT_ONLY"
    else:
        verdict = "FULL_BATCH_IS_CEILING"

    census = _m0a_rows(cohort_name, task_id)
    retained = [uid for uid in train_uids if uid not in best["final_excluded"]]

    # One equivalence re-check, so this artifact stands on its own: the masked
    # executor must still reproduce the frozen readout on the empty mask.
    equivalence = []
    for program in searched:
        mirrored = masked(program, set(), (support[0],))[0]
        reference = frozen(program, (support[0],))[0]
        equivalence.append({
            "program": program,
            "exact_match": (
                mirrored["per_view_smase"] == reference["per_view_smase"]
            ),
        })

    return {
        "cohort": cohort_name,
        "consumer_variant": consumer_variant,
        "exposure": cohort["exposure"],
        "verdict": verdict,
        "task_episode_id": task_id,
        "support_origins": list(support),
        "delayed_origins": list(delayed),
        "train_series": train_uids,
        "eval_series": eval_uids,
        "programs_searched": searched,
        "program_selection_rule": (
            "the two programs with the highest full-batch Support aggregate "
            "gain on this cohort"
        ),
        "best_full_batch_single_program": best_full_program,
        "full_batch_support": {
            program: full_support[program] for program in full_support
        },
        "full_batch_delayed": full_delayed,
        "identity_absolute_loss": {
            "support": _identity_absolute_loss(identity_support),
            "delayed": _identity_absolute_loss(identity_delayed),
            "unit": "mean sMASE over origins then eval views",
            "feeds_adoption_rule": False,
        },
        "per_series_singleton_gain": per_series,
        "searches": searches,
        "best_masked_plan": {
            "program": best_program,
            "excluded_series": best["final_excluded"],
            "excluded_series_count": len(best["final_excluded"]),
            "treated_series_count": len(retained),
            "support": best_support,
            "delayed": best_delayed,
        },
        "headline": {
            "support": {
                "identity": 0.0,
                "best_full_batch_single_program": bar_support["aggregate_gain"],
                "masked": best_support["aggregate_gain"],
                "masked_minus_best_full_batch": beats_support,
                "masked_minus_own_program_full_batch": (
                    best_support["aggregate_gain"]
                    - full_support[best_program]["aggregate_gain"]
                ),
            },
            "delayed": {
                "identity": 0.0,
                "best_full_batch_single_program": bar_delayed["aggregate_gain"],
                "masked": best_delayed["aggregate_gain"],
                "masked_minus_best_full_batch": beats_delayed,
                "masked_minus_own_program_full_batch": (
                    best_delayed["aggregate_gain"]
                    - full_delayed[best_program]["aggregate_gain"]
                ),
            },
            "harm": {
                "identity_harmed_eval_series_count": 0,
                "best_full_batch_harmed_eval_series_count": bar_support[
                    "harmed_eval_series_count"],
                "best_full_batch_harmed_eval_total_harm": bar_support[
                    "harmed_eval_series_total_harm"],
                "masked_harmed_eval_series_count": best_support[
                    "harmed_eval_series_count"],
                "masked_harmed_eval_total_harm": best_support[
                    "harmed_eval_series_total_harm"],
                "harm_smaller_than_full_batch": harm_smaller,
            },
        },
        "selection_discipline": {
            "pre_declared": True,
            "rule": (
                "every accept/reject decision reads the Support aggregate "
                "only; the delayed window is evaluated for each accepted mask "
                "and reported, and never enters the search"
            ),
            "delayed_is_reported_not_selected_on": True,
            "no_additive_credit_used": (
                "the singleton per-series gain orders the revert queue and is "
                "never added up or used to accept a mask"
            ),
        },
        "excluded_series_geometry": _geometry_contrast(
            census, best["final_excluded"], retained
        ),
        "readout_equivalence_check": {
            "claim": (
                "the masked executor reproduces the frozen _evaluate readout "
                "exactly on the empty mask"
            ),
            "all_exact": all(row["exact_match"] for row in equivalence),
            "rows": equivalence,
        },
        "wall_seconds": time.perf_counter() - started,
    }


def _masked_markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# masked-single-program v1")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "The companion composition experiment returned "
        "`COMPOSITION_NO_HEADROOM` on both cohorts: per-series responses are "
        "strongly heterogeneous, but a per-series argmax loses essentially the "
        "whole summed gain to a cross-series retraining interaction, because "
        "the singleton-scope gain is not the credit a series contributes once "
        "the whole batch is treated.  This run tests the low-dimensional "
        "variant that does not need an additive credit signal at all: **one "
        "program, plus a harm-driven exclusion mask**.  It starts from that "
        "program applied to the whole batch, reverts one treated series to "
        "identity at a time, and re-measures the aggregate with a real retrain "
        "after every single revert.  A revert is kept only if the measured "
        "aggregate improved; the first revert that does not improve is rolled "
        "back and ends the search."
    )
    lines.append("")
    lines.append(
        "**This is an engineering effect measurement, not authorization "
        "evidence.**  No Skill is written, no Episode is formed, no Fast or "
        "Slow path runs, and no execution right is granted or implied.  Data is "
        "already-exposed development data only; nothing sealed is reachable."
    )
    lines.append("")
    lines.append(
        "**Selection happens on the Support window only.**  This was declared "
        "before any number was read and is enforced in the code: every "
        "accept/reject decision reads the Support aggregate, and the delayed "
        "window is evaluated once for each accepted mask purely so it can be "
        "reported honestly.  The delayed column never steers the search, and it "
        "is the only column here that is out of selection."
    )
    lines.append("")
    lines.append(
        "The revert queue is ordered by each series' singleton per-series gain "
        "under the program, most harmful first.  That ordering is a heuristic "
        "that decides what to *try* next; it never decides what is *kept*.  No "
        "per-series gain is summed anywhere in this run."
    )
    lines.append("")

    for cohort in result["cohorts"]:
        if "error" in cohort:
            lines.append(f"## Cohort `{cohort['cohort']}` -- FAILED")
            lines.append("")
            lines.append(f"```\n{cohort['error']}\n```")
            lines.append("")
            continue
        head = cohort["headline"]
        plan = cohort["best_masked_plan"]
        lines.append(f"## Cohort `{cohort['cohort']}`")
        lines.append("")
        lines.append(f"**Verdict: `{cohort['verdict']}`**")
        lines.append("")
        lines.append(
            f"Task `{cohort['task_episode_id']}`, Support origins "
            f"{cohort['support_origins']}, delayed origins "
            f"{cohort['delayed_origins']}; "
            f"{len(cohort['train_series'])} training series, "
            f"{len(cohort['eval_series'])} evaluation series.  Programs "
            f"searched: {', '.join('`%s`' % p for p in cohort['programs_searched'])} "
            f"(top two by full-batch Support aggregate gain)."
        )
        lines.append("")

        lines.append("### Headline")
        lines.append("")
        lines.append("| plan | support | delayed |")
        lines.append("| --- | ---: | ---: |")
        lines.append("| identity | 0.000000 | 0.000000 |")
        lines.append(
            f"| best full batch single program "
            f"(`{cohort['best_full_batch_single_program']}`) | "
            f"{head['support']['best_full_batch_single_program']:+.6f} | "
            f"{head['delayed']['best_full_batch_single_program']:+.6f} |"
        )
        lines.append(
            f"| **best masked plan** (`{plan['program']}`, "
            f"{plan['excluded_series_count']} reverted) | "
            f"{head['support']['masked']:+.6f} | "
            f"{head['delayed']['masked']:+.6f} |"
        )
        lines.append("")
        lines.append(
            f"Masked minus best full batch: support "
            f"{head['support']['masked_minus_best_full_batch']:+.6f}, delayed "
            f"{head['delayed']['masked_minus_best_full_batch']:+.6f}.  Masked "
            f"minus its own program's full batch: support "
            f"{head['support']['masked_minus_own_program_full_batch']:+.6f}, "
            f"delayed "
            f"{head['delayed']['masked_minus_own_program_full_batch']:+.6f}."
        )
        lines.append("")
        harm = head["harm"]
        lines.append(
            f"Harm account on the evaluation side: identity 0 series; best "
            f"full batch {harm['best_full_batch_harmed_eval_series_count']} "
            f"series / {harm['best_full_batch_harmed_eval_total_harm']:.6f} "
            f"total; best masked plan "
            f"{harm['masked_harmed_eval_series_count']} series / "
            f"{harm['masked_harmed_eval_total_harm']:.6f} total.  Harm smaller "
            f"than full batch: `{harm['harm_smaller_than_full_batch']}`."
        )
        lines.append("")

        lines.append("### Greedy exclusion trace")
        lines.append("")
        for program in cohort["programs_searched"]:
            search = cohort["searches"][program]
            lines.append(
                f"`{program}` -- full batch "
                f"{cohort['full_batch_support'][program]['aggregate_gain']:+.6f}"
                f", {search['accepted_step_count']} revert(s) accepted, final "
                f"Support {search['final_support']['aggregate_gain']:+.6f}, "
                f"final delayed "
                f"{search['final_delayed']['aggregate_gain']:+.6f}."
            )
            lines.append("")
            lines.append(
                "| step | reverted series | support after | delta | decision |"
            )
            lines.append("| ---: | --- | ---: | ---: | --- |")
            for step in search["steps"]:
                lines.append(
                    f"| {step['step']} | {step['reverted_series']} | "
                    f"{step['support_aggregate_gain']:+.6f} | "
                    f"{step['delta']:+.6f} | {step['decision']} |"
                )
            lines.append("")
            if search["accepted_step_count"]:
                lines.append(
                    "| accepted mask | reverted so far | support | delayed | "
                    "harmed eval series | total eval harm |"
                )
                lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
                for row in search["accepted_masks"]:
                    label = ", ".join(row["excluded"]) or "(none, full batch)"
                    lines.append(
                        f"| {row['step']} | {label} | "
                        f"{row['support']['aggregate_gain']:+.6f} | "
                        f"{row['delayed']['aggregate_gain']:+.6f} | "
                        f"{row['support']['harmed_eval_series_count']} | "
                        f"{row['support']['harmed_eval_series_total_harm']:.6f} |"
                    )
                lines.append("")

        geometry = cohort["excluded_series_geometry"]
        lines.append("### Reverted-series geometry (descriptive only)")
        lines.append("")
        if not plan["excluded_series"]:
            lines.append(
                "No series was reverted on this cohort, so there is nothing to "
                "contrast."
            )
            lines.append("")
        else:
            lines.append(
                f"Reverted: {', '.join('`%s`' % u for u in plan['excluded_series'])} "
                f"({geometry['excluded']['mask_class_counts']}).  Retained: "
                f"{geometry['retained']['series_count']} series "
                f"({geometry['retained']['mask_class_counts']}).  Fields from "
                f"the frozen M0a census, read verbatim; no threshold is fitted "
                f"and nothing here contributes to the verdict."
            )
            lines.append("")
            lines.append(
                "| field | reverted mean | retained mean | reverted range | "
                "retained range | ranges overlap |"
            )
            lines.append("| --- | ---: | ---: | --- | --- | --- |")
            for row in geometry["per_field"]:
                lines.append(
                    f"| `{row['field']}` | {row['excluded_mean']:.6f} | "
                    f"{row['retained_mean']:.6f} | "
                    f"[{row['excluded_range'][0]:.6f}, "
                    f"{row['excluded_range'][1]:.6f}] | "
                    f"[{row['retained_range'][0]:.6f}, "
                    f"{row['retained_range'][1]:.6f}] | "
                    f"{row['observed_ranges_overlap']} |"
                )
            lines.append("")
            separated = geometry["fields_with_non_overlapping_observed_ranges"]
            lines.append(
                "Fields whose observed ranges do not overlap between the two "
                "groups: "
                + (", ".join(f"`{f}`" for f in separated) if separated
                   else "none")
                + "."
            )
            lines.append("")

        check = cohort["readout_equivalence_check"]
        lines.append(
            f"Readout equivalence check (masked executor reproduces the frozen "
            f"`_evaluate` readout exactly on the empty mask): "
            f"`{check['all_exact']}`."
        )
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "The low-dimensional variant does find Support headroom where the "
        "per-series argmax composition found none, and the difference is "
        "exactly the thing that broke the composition: nothing is added up "
        "here, so no step depends on the additive credit the pooled Consumer "
        "does not honour.  On both cohorts the best masked plan beats the best "
        "full-batch single program on Support, and the evaluation-side harm "
        "drops on both -- electricity from 4 harmed series / 0.0936 to 1 / "
        "0.0108, T233 from 2 / 0.6786 to 1 / 0.1738.  Reverting a handful of "
        "series is also enough to change which program wins: `outlier_iqr` is "
        "*below identity* on electricity at full batch (-0.002188) and becomes "
        "the cohort's best plan once three series are reverted (+0.034643), so "
        "the full-batch ranking is not the post-mask ranking."
    )
    lines.append("")
    lines.append(
        "The two cohorts then split on the one column the search never saw.  "
        "On electricity the delayed aggregate rises monotonically with every "
        "accepted revert (-0.001864, +0.001344, +0.004548, +0.016343), so the "
        "Support-only search happened to track the honest window and the "
        "verdict is unqualified.  On T233 it does not: the first accepted "
        "revert raises Support from +0.072156 to +0.086841 while cutting the "
        "delayed gain from +0.116627 to +0.016079, the final mask recovers only "
        "part of that, and the second searched program behaves the same way "
        "(`outlier_iqr` delayed +0.155763 at full batch, +0.047465 masked).  On "
        "that cohort the full batch is close to the delayed ceiling and the "
        "Support-only greedy step is buying Support at the delayed window's "
        "expense, which is why the verdict is "
        "`MASKED_IMPROVES_SUPPORT_ONLY` and not an improvement claim."
    )
    lines.append("")
    lines.append(
        "So the mechanism is real and cheap, and Support-gain-driven exclusion "
        "is not by itself safe to read as a downstream improvement.  What "
        "separates the two cohorts is not visible in the Support column, which "
        "is the part a deployable version of this would have to solve."
    )
    lines.append("")
    lines.append("## Verdict summary")
    lines.append("")
    lines.append(
        "| cohort | verdict | best masked plan | support | delayed |"
    )
    lines.append("| --- | --- | --- | ---: | ---: |")
    for cohort in result["cohorts"]:
        if "error" in cohort:
            lines.append(f"| {cohort['cohort']} | RUN_FAILED | | | |")
            continue
        plan = cohort["best_masked_plan"]
        head = cohort["headline"]
        label = (
            f"`{plan['program']}` minus "
            f"{plan['excluded_series_count']} series"
            if plan["excluded_series"] else "no revert accepted"
        )
        lines.append(
            f"| {cohort['cohort']} | `{cohort['verdict']}` | {label} | "
            f"{head['support']['masked']:+.6f} | "
            f"{head['delayed']['masked']:+.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------- batch recipe entry
RECIPE_PROTOCOL_VERSION = "batch_recipe_v1"


def _recipe_paths(cohort_name: str) -> tuple[Path, Path]:
    stem = f"batch_recipe_{cohort_name}_v1"
    root = PROJECT_ROOT / "artifacts/functional/e2"
    return root / f"{stem}.json", root / f"{stem}.md"


# The adoption rule, learned directly from masked-single-program and frozen.
# There is deliberately no threshold argument anywhere in this module for it.
ADOPTION_RULE_V1 = (
    "Candidates are the masked plans, ordered by Support aggregate gain, "
    "highest first.  A masked plan is adopted only if it also clears the "
    "delayed stability check: its delayed aggregate gain must be at least the "
    "delayed aggregate gain of the best full-batch plan.  The first candidate "
    "that clears it is adopted.  If none clears it, fall back to the best "
    "full-batch plan.  If the best full-batch plan's delayed aggregate gain is "
    "not positive, fall back to identity."
)
ADOPTION_RULE = ADOPTION_RULE_V1
ADOPTION_RULE_V2 = (
    "adoption_rule_version: v2.  Candidates are the masked plans, ordered by "
    "Support aggregate gain, highest first.  A masked plan is adopted only if "
    "it also clears the delayed stability check: its delayed aggregate gain "
    "must be at least max(best full-batch delayed aggregate gain, 0) -- "
    "identity is an incumbent on the delayed window, not only the best "
    "full-batch plan.  The first candidate that clears it is adopted.  If none "
    "clears it, fall back to the best full-batch plan only if that plan's "
    "delayed aggregate gain is positive; otherwise fall back to identity.  "
    "v1 failure case: T233@per_channel in artifacts/functional/e2/"
    "consumer_conditioned_recipe_v1.json adopted MASKED_PLAN smooth_ma at "
    "delayed -0.076 because v1 compared the mask only against the best "
    "full-batch delayed gain (-0.146) and did not treat identity (delayed 0) "
    "as the incumbent."
)
RECIPE_PROTOCOL_VERSION_V2 = "batch_recipe_v2"


def make_batch_recipe(
    cohort_name: str,
    *,
    task_index: int = 0,
    search: Mapping[str, Any] | None = None,
    consumer_variant: str = CONSUMER_POOLED,
    adoption_rule_version: str = "v1",
) -> dict[str, Any]:
    """Menu scan -> mask search -> stability gate -> one adopted batch plan.

    This is the productised form of the two experiments, not a new one.  It
    runs the existing machinery in order and then applies :data:`ADOPTION_RULE`,
    which is frozen: the rule has no tunable threshold and this function takes
    no knob that could become one.

    The honest cost of the rule is stated in the artifact it writes.  The
    delayed window is what stops a Support-overfitted mask from being adopted,
    which means the delayed window is now part of the selection and has stopped
    being an out-of-selection readout for this recipe.  Any external claim
    needs a fresh window.
    """
    started = time.perf_counter()
    if consumer_variant not in CONSUMER_VARIANTS:
        raise ValueError(f"unknown consumer_variant: {consumer_variant!r}")
    if adoption_rule_version not in ("v1", "v2"):
        raise ValueError(
            f"unknown adoption_rule_version: {adoption_rule_version!r}"
        )
    search = search or run_masked_cohort(
        cohort_name, task_index=task_index, consumer_variant=consumer_variant,
    )
    best_full = str(search["best_full_batch_single_program"])
    bar_support = search["full_batch_support"][best_full]
    bar_delayed = search["full_batch_delayed"][best_full]
    delayed_bar = (
        max(float(bar_delayed["aggregate_gain"]), 0.0)
        if adoption_rule_version == "v2"
        else float(bar_delayed["aggregate_gain"])
    )
    rule_text = (
        ADOPTION_RULE_V2 if adoption_rule_version == "v2" else ADOPTION_RULE_V1
    )

    # Only real masked plans are candidates.  A search that accepted no revert
    # produced its own full batch back, which is the fallback, not a candidate.
    candidates = [
        {
            "program": program,
            "excluded_series": list(
                search["searches"][program]["final_excluded"]
            ),
            "support": search["searches"][program]["final_support"],
            "delayed": search["searches"][program]["final_delayed"],
        }
        for program in search["programs_searched"]
        if search["searches"][program]["final_excluded"]
    ]
    candidates.sort(key=lambda row: -row["support"]["aggregate_gain"])

    trace: list[dict[str, Any]] = []
    adopted_candidate: dict[str, Any] | None = None
    for candidate in candidates:
        if adopted_candidate is not None:
            trace.append({
                "program": candidate["program"],
                "excluded_series": candidate["excluded_series"],
                "support_aggregate_gain": candidate["support"]["aggregate_gain"],
                "delayed_aggregate_gain": candidate["delayed"]["aggregate_gain"],
                "stability_check": "NOT_REACHED",
            })
            continue
        passes = (
            candidate["delayed"]["aggregate_gain"] >= delayed_bar
        )
        trace.append({
            "program": candidate["program"],
            "excluded_series": candidate["excluded_series"],
            "support_aggregate_gain": candidate["support"]["aggregate_gain"],
            "delayed_aggregate_gain": candidate["delayed"]["aggregate_gain"],
            "delayed_bar": delayed_bar,
            "best_full_batch_delayed": bar_delayed["aggregate_gain"],
            "identity_delayed": 0.0,
            "delayed_margin": (
                candidate["delayed"]["aggregate_gain"] - delayed_bar
            ),
            "stability_check": "PASS" if passes else "FAIL",
        })
        if passes:
            adopted_candidate = candidate

    if adopted_candidate is not None:
        adopted = {
            "kind": "MASKED_PLAN",
            "program": adopted_candidate["program"],
            "excluded_series": adopted_candidate["excluded_series"],
            "support": adopted_candidate["support"],
            "delayed": adopted_candidate["delayed"],
        }
        path = (
            "masked plan cleared the delayed stability check "
            f"(bar=max(best_full_batch_delayed, 0)={delayed_bar:+.6f})"
            if adoption_rule_version == "v2"
            else "masked plan cleared the delayed stability check"
        )
    elif bar_delayed["aggregate_gain"] > 0.0:
        adopted = {
            "kind": "BEST_FULL_BATCH",
            "program": best_full,
            "excluded_series": [],
            "support": bar_support,
            "delayed": bar_delayed,
        }
        path = (
            "no masked plan cleared the delayed stability check; fell back to "
            "the best full-batch plan, whose delayed gain is positive"
        )
    else:
        adopted = {
            "kind": "IDENTITY",
            "program": IDENTITY,
            "excluded_series": [],
            "support": {
                "aggregate_gain": 0.0, "harmed_eval_series_count": 0,
                "harmed_eval_series_total_harm": 0.0, "harmed_eval_series": [],
            },
            "delayed": {
                "aggregate_gain": 0.0, "harmed_eval_series_count": 0,
                "harmed_eval_series_total_harm": 0.0, "harmed_eval_series": [],
            },
        }
        path = (
            "no masked plan cleared the delayed stability check and the best "
            "full-batch plan's delayed gain is not positive; fell back to "
            "identity"
        )

    train_uids = [str(uid) for uid in search["train_series"]]
    census = _m0a_rows(cohort_name, str(search["task_episode_id"]))
    adopted_excluded = list(adopted["excluded_series"])
    adopted_geometry = _geometry_contrast(
        census, adopted_excluded,
        [uid for uid in train_uids if uid not in adopted_excluded],
    )
    rejected_geometry = None
    rejected = next(
        (row for row in trace if row["stability_check"] == "FAIL"), None
    )
    if rejected is not None:
        rejected_geometry = _geometry_contrast(
            census, rejected["excluded_series"],
            [uid for uid in train_uids
             if uid not in rejected["excluded_series"]],
        )

    delayed_ranked = sorted(
        search["full_batch_delayed"],
        key=lambda op: (
            -search["full_batch_delayed"][op]["aggregate_gain"],
            PROGRAM_MENU.index(op),
        ),
    )
    delayed_champion = delayed_ranked[0] if delayed_ranked else best_full
    menu_scan = {
        program: {
            "support": search["full_batch_support"][program],
            "delayed": search["full_batch_delayed"][program],
            "aggregate_gain": search["full_batch_support"][program][
                "aggregate_gain"],
            "delayed_aggregate_gain": search["full_batch_delayed"][program][
                "aggregate_gain"],
        }
        for program in search["full_batch_support"]
        if program in search["full_batch_delayed"]
    }
    identity_loss = search.get("identity_absolute_loss") or {
        "support": None, "delayed": None,
        "unit": "mean sMASE over origins then eval views",
        "feeds_adoption_rule": False,
    }

    return {
        "protocol_version": (
            RECIPE_PROTOCOL_VERSION_V2 if adoption_rule_version == "v2"
            else RECIPE_PROTOCOL_VERSION
        ),
        "role": (
            "batch recipe: one adopted data-processing plan for one already-"
            "exposed batch, produced by menu scan -> mask search -> frozen "
            "delayed stability gate"
        ),
        "llm_api_call_count": 0,
        "deterministic": True,
        "cohort": cohort_name,
        "consumer_variant": consumer_variant,
        "consumer_variant_scope": (
            "experiment-local fork inside this runner; the frozen pooled "
            "Consumer used by v6._evaluate is unchanged"
        ),
        "exposure": search["exposure"],
        "task_episode_id": search["task_episode_id"],
        "support_origins": search["support_origins"],
        "delayed_origins": search["delayed_origins"],
        "train_series": train_uids,
        "eval_series": search["eval_series"],
        "adoption_rule_version": adoption_rule_version,
        "adoption_rule": rule_text,
        "adoption_rule_is_frozen": (
            "no threshold argument and no tuning knob is exposed for this rule "
            "anywhere in this module"
        ),
        "delayed_stability_bar": delayed_bar,
        "identity_absolute_loss": identity_loss,
        "adopted_plan": {
            "kind": adopted["kind"],
            "program": adopted["program"],
            "excluded_series": adopted_excluded,
            "excluded_series_count": len(adopted_excluded),
            "treated_series_count": len(train_uids) - len(adopted_excluded),
            "how_to_apply": (
                f"apply `{adopted['program']}` to every training series except "
                f"{adopted_excluded}, then retrain the Consumer once"
                if adopted["kind"] == "MASKED_PLAN"
                else f"apply `{adopted['program']}` to every training series, "
                     f"then retrain the Consumer once"
                if adopted["kind"] == "BEST_FULL_BATCH"
                else "treat nothing; the batch is left at identity"
            ),
        },
        "adoption_path": path,
        "adoption_trace": trace,
        "comparison": {
            "support": {
                "adopted": adopted["support"]["aggregate_gain"],
                "best_full_batch": bar_support["aggregate_gain"],
                "identity": 0.0,
            },
            "delayed": {
                "adopted": adopted["delayed"]["aggregate_gain"],
                "best_full_batch": bar_delayed["aggregate_gain"],
                "identity": 0.0,
            },
            "best_full_batch_program": best_full,
            "best_full_batch_delayed_program": delayed_champion,
            "support_champion_equals_delayed_champion": (
                delayed_champion == best_full
            ),
        },
        "harm_account": {
            "adopted": {
                "harmed_eval_series_count": adopted["support"][
                    "harmed_eval_series_count"],
                "harmed_eval_series_total_harm": adopted["support"][
                    "harmed_eval_series_total_harm"],
                "harmed_eval_series": adopted["support"].get(
                    "harmed_eval_series", []),
            },
            "best_full_batch": {
                "harmed_eval_series_count": bar_support[
                    "harmed_eval_series_count"],
                "harmed_eval_series_total_harm": bar_support[
                    "harmed_eval_series_total_harm"],
                "harmed_eval_series": bar_support.get("harmed_eval_series", []),
            },
            "identity": {
                "harmed_eval_series_count": 0,
                "harmed_eval_series_total_harm": 0.0,
                "harmed_eval_series": [],
            },
        },
        "excluded_series_geometry": adopted_geometry,
        "stability_rejected_candidate_geometry": rejected_geometry,
        "menu_scan": menu_scan,
        "programs_searched": search["programs_searched"],
        "caveat": (
            "The delayed window participated in the adoption decision, so for "
            "this recipe it is no longer an out-of-selection readout.  Both "
            "columns reported here are now in-selection.  Any external claim "
            "about this recipe needs a fresh window."
        ),
        "not_authorization_evidence": (
            "engineering effect measurement only; no Skill is written, no "
            "Episode is formed, no Fast or Slow path is entered, and no "
            "execution right is granted or implied"
        ),
        "wall_seconds": time.perf_counter() - started,
    }


def _recipe_markdown(recipe: Mapping[str, Any]) -> str:
    plan = recipe["adopted_plan"]
    comparison = recipe["comparison"]
    lines: list[str] = []
    lines.append(f"# batch recipe -- `{recipe['cohort']}` v1")
    lines.append("")
    lines.append(
        "One adopted data-processing plan for one already-exposed batch, "
        "produced by the frozen three-step recipe: scan the program menu at "
        "full batch, run a greedy harm-ordered exclusion mask search on the two "
        "best programs, then apply the delayed stability gate."
    )
    lines.append("")
    lines.append(
        "**Engineering effect measurement, not authorization evidence.**  No "
        "Skill is written, no Episode is formed, no Fast or Slow path is "
        "entered, and no execution right is granted or implied.  0 LLM calls. "
        "Already-exposed development data only."
    )
    lines.append("")
    lines.append(
        f"> **Caveat.** {recipe['caveat']}"
    )
    lines.append("")

    lines.append("## Adopted plan")
    lines.append("")
    lines.append(f"- kind: `{plan['kind']}`")
    lines.append(f"- program: `{plan['program']}`")
    lines.append(
        f"- reverted to identity: "
        + (", ".join(f"`{u}`" for u in plan["excluded_series"])
           if plan["excluded_series"] else "none")
    )
    lines.append(
        f"- treated series: {plan['treated_series_count']} of "
        f"{len(recipe['train_series'])}"
    )
    lines.append(f"- how to apply: {plan['how_to_apply']}")
    lines.append("")
    lines.append(f"Adoption path: {recipe['adoption_path']}.")
    lines.append("")

    lines.append("## Comparison")
    lines.append("")
    lines.append("| plan | support | delayed |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| **adopted** (`{plan['program']}`"
        + (f", {plan['excluded_series_count']} reverted" if plan[
            "excluded_series"] else ", full batch")
        + f") | {comparison['support']['adopted']:+.6f} | "
        f"{comparison['delayed']['adopted']:+.6f} |"
    )
    lines.append(
        f"| best full batch (`{comparison['best_full_batch_program']}`) | "
        f"{comparison['support']['best_full_batch']:+.6f} | "
        f"{comparison['delayed']['best_full_batch']:+.6f} |"
    )
    lines.append("| identity | 0.000000 | 0.000000 |")
    lines.append("")

    harm = recipe["harm_account"]
    lines.append("## Harm account (evaluation series worse than identity)")
    lines.append("")
    lines.append("| plan | harmed series | total harm |")
    lines.append("| --- | ---: | ---: |")
    for label in ("adopted", "best_full_batch", "identity"):
        lines.append(
            f"| {label} | {harm[label]['harmed_eval_series_count']} | "
            f"{harm[label]['harmed_eval_series_total_harm']:.6f} |"
        )
    lines.append("")

    lines.append("## Adoption trace")
    lines.append("")
    lines.append(
        "Rule (frozen, no tunable threshold): " + recipe["adoption_rule"]
    )
    lines.append("")
    if recipe["adoption_trace"]:
        lines.append(
            "| candidate | reverted | support | delayed | bar | check |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | --- |")
        for row in recipe["adoption_trace"]:
            bar = row.get("delayed_bar")
            lines.append(
                f"| `{row['program']}` | "
                f"{', '.join(row['excluded_series']) or 'none'} | "
                f"{row['support_aggregate_gain']:+.6f} | "
                f"{row['delayed_aggregate_gain']:+.6f} | "
                + (f"{bar:+.6f}" if bar is not None else "--")
                + f" | {row['stability_check']} |"
            )
    else:
        lines.append(
            "No masked candidate was produced: neither searched program "
            "accepted a revert, so the mask search returned its own full batch "
            "and there was nothing for the stability gate to judge."
        )
    lines.append("")

    lines.append("## Menu scan (full batch, Support aggregate gain)")
    lines.append("")
    lines.append("| program | support aggregate gain |")
    lines.append("| --- | ---: |")
    scan = recipe["menu_scan"]
    for program in sorted(scan, key=lambda op: -scan[op]["aggregate_gain"]):
        mark = " (searched)" if program in recipe["programs_searched"] else ""
        lines.append(
            f"| `{program}`{mark} | {scan[program]['aggregate_gain']:+.6f} |"
        )
    lines.append("")

    lines.append("## Reverted-series geometry (descriptive only)")
    lines.append("")
    geometry = recipe["excluded_series_geometry"]
    if not plan["excluded_series"]:
        lines.append(
            "The adopted plan reverts nothing, so there is no reverted-series "
            "geometry to describe."
        )
        lines.append("")
        rejected = recipe.get("stability_rejected_candidate_geometry")
        if rejected:
            lines.append(
                "For reference, the masked candidate the stability gate turned "
                "down would have reverted "
                + ", ".join(f"`{u}`" for u in rejected["excluded"]["series"])
                + f" ({rejected['excluded']['mask_class_counts']})."
            )
            lines.append("")
    else:
        lines.append(
            "Fields from the frozen M0a census, read verbatim.  No threshold is "
            "fitted here and nothing in this section feeds the adoption rule."
        )
        lines.append("")
        lines.append(
            f"Reverted {geometry['excluded']['series_count']} series "
            f"({geometry['excluded']['mask_class_counts']}); retained "
            f"{geometry['retained']['series_count']} "
            f"({geometry['retained']['mask_class_counts']})."
        )
        lines.append("")
        lines.append(
            "| field | reverted mean | retained mean | reverted range | "
            "retained range | ranges overlap |"
        )
        lines.append("| --- | ---: | ---: | --- | --- | --- |")
        for row in geometry["per_field"]:
            lines.append(
                f"| `{row['field']}` | {row['excluded_mean']:.6f} | "
                f"{row['retained_mean']:.6f} | "
                f"[{row['excluded_range'][0]:.6f}, "
                f"{row['excluded_range'][1]:.6f}] | "
                f"[{row['retained_range'][0]:.6f}, "
                f"{row['retained_range'][1]:.6f}] | "
                f"{row['observed_ranges_overlap']} |"
            )
        lines.append("")
        separated = geometry["fields_with_non_overlapping_observed_ranges"]
        lines.append(
            "Fields whose observed ranges do not overlap between reverted and "
            "retained: "
            + (", ".join(f"`{f}`" for f in separated) if separated else "none")
            + "."
        )
        lines.append("")
    return "\n".join(lines)


def _run_recipe(
    cohort_names: Sequence[str],
    task_index: int,
    consumer_variant: str = CONSUMER_POOLED,
) -> int:
    for name in cohort_names:
        recipe = make_batch_recipe(
            name, task_index=task_index, consumer_variant=consumer_variant,
            adoption_rule_version="v1",
        )
        if consumer_variant != CONSUMER_POOLED:
            plan = recipe["adopted_plan"]
            comparison = recipe["comparison"]
            print(
                "RECIPE %s consumer=%s adopted=%s program=%s excluded=%s "
                "support(adopted=%+.6f full=%+.6f) delayed(adopted=%+.6f "
                "full=%+.6f) (not writing pooled recipe artifacts)"
                % (name, consumer_variant, plan["kind"], plan["program"],
                   plan["excluded_series"],
                   comparison["support"]["adopted"],
                   comparison["support"]["best_full_batch"],
                   comparison["delayed"]["adopted"],
                   comparison["delayed"]["best_full_batch"]),
                flush=True,
            )
            continue
        json_path, md_path = _recipe_paths(name)
        if json_path.exists() or md_path.exists():
            print(
                "RECIPE %s skip write: v1 artifact already exists at %s"
                % (name, json_path),
                flush=True,
            )
            continue
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(recipe, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_recipe_markdown(recipe) + "\n", encoding="utf-8")
        plan = recipe["adopted_plan"]
        comparison = recipe["comparison"]
        print(
            "RECIPE %s adopted=%s program=%s excluded=%s support(adopted="
            "%+.6f full=%+.6f) delayed(adopted=%+.6f full=%+.6f)"
            % (name, plan["kind"], plan["program"], plan["excluded_series"],
               comparison["support"]["adopted"],
               comparison["support"]["best_full_batch"],
               comparison["delayed"]["adopted"],
               comparison["delayed"]["best_full_batch"]),
            flush=True,
        )
    return 0


def _v1_cell_signature(plan: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(plan["kind"]),
        "program": str(plan["program"]),
        "excluded_series": [str(uid) for uid in plan["excluded_series"]],
        "support": float(comparison["support"]["adopted"]),
        "delayed": float(comparison["delayed"]["adopted"]),
    }


def _load_v1_cell(cohort_name: str, consumer_variant: str) -> dict[str, Any] | None:
    if consumer_variant == CONSUMER_POOLED:
        recipe = _load_pooled_recipe(cohort_name)
        if recipe is None:
            return None
        return _v1_cell_signature(recipe["adopted_plan"], recipe["comparison"])
    path = CCR_JSON
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("cohorts") or []:
        if str(row.get("cohort")) != cohort_name or "error" in row:
            continue
        rec = row["recipe"][CONSUMER_PER_CHANNEL]
        return _v1_cell_signature(rec["adopted_plan"], rec["comparison"])
    return None


def _cell_plan_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["kind"] == right["kind"]
        and left["program"] == right["program"]
        and sorted(left["excluded_series"]) == sorted(right["excluded_series"])
        and abs(float(left["support"]) - float(right["support"])) < 1e-12
        and abs(float(left["delayed"]) - float(right["delayed"])) < 1e-12
    )


def _recipe_v2_all_cells_markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# batch recipe v2 -- all cells")
    lines.append("")
    lines.append(
        "**Engineering effect measurement, not authorization evidence.**  "
        "adoption_rule_version v2.  Consumer variants live only inside this "
        "experiment runner.  0 LLM.  v1 recipe artifacts were not overwritten."
    )
    lines.append("")
    lines.append(
        "v2 delayed gate: a masked plan is adopted only if its delayed "
        "aggregate gain >= max(best full-batch delayed, 0).  Identity is an "
        "incumbent on the delayed window.  v1 failure case: T233@per_channel "
        "in consumer_conditioned_recipe_v1.json."
    )
    lines.append("")
    unexpected = result["unexpected_cell_changes"]
    if unexpected:
        lines.append("**UNEXPECTED_CELL_CHANGE**")
        lines.append("")
        for row in unexpected:
            lines.append(
                "- `" + row["cohort"] + "` `" + row["consumer_variant"]
                + "`: v1 " + str(row["v1"]) + " -> v2 " + str(row["v2"])
            )
        lines.append("")
    lines.append(
        "| cell | kind | program | reverted | support | delayed | vs v1 |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | --- |")
    for cell in result["cells"]:
        if "error" in cell:
            lines.append(
                "| `" + cell["cohort"] + "` `" + cell["consumer_variant"]
                + "` | RUN_FAILED | | | | | |"
            )
            continue
        plan = cell["recipe"]["adopted_plan"]
        cmp_ = cell["recipe"]["comparison"]
        reverted = (
            ", ".join("`" + u + "`" for u in plan["excluded_series"])
            if plan["excluded_series"] else "none"
        )
        lines.append(
            "| `" + cell["cohort"] + "` `" + cell["consumer_variant"]
            + "` | `" + plan["kind"] + "` | `" + plan["program"]
            + "` | " + reverted + " | "
            + f"{cmp_['support']['adopted']:+.6f} | "
            + f"{cmp_['delayed']['adopted']:+.6f} | " + cell["vs_v1"] + " |"
        )
    lines.append("")
    for cell in result["cells"]:
        if "error" in cell:
            lines.append(
                "## `" + cell["cohort"] + "` `" + cell["consumer_variant"]
                + "` -- FAILED"
            )
            lines.append("")
            lines.append("```")
            lines.append(str(cell["error"]))
            lines.append("```")
            lines.append("")
            continue
        rec = cell["recipe"]
        plan = rec["adopted_plan"]
        cmp_ = rec["comparison"]
        loss = rec["identity_absolute_loss"]
        lines.append(
            "## `" + cell["cohort"] + "` `" + cell["consumer_variant"] + "`"
        )
        lines.append("")
        lines.append("**vs v1: " + cell["vs_v1"] + "**")
        lines.append("")
        lines.append(
            "Windows: Support " + str(rec["support_origins"]) + ", delayed "
            + str(rec["delayed_origins"]) + "."
        )
        lines.append("")
        reverted = (
            ", ".join("`" + u + "`" for u in plan["excluded_series"])
            if plan["excluded_series"] else "none"
        )
        lines.append(
            "Adopted: `" + plan["kind"] + "` `" + plan["program"]
            + "` reverted " + reverted + "."
        )
        lines.append("")
        lines.append(
            "Gain: support " + f"{cmp_['support']['adopted']:+.6f}"
            + ", delayed " + f"{cmp_['delayed']['adopted']:+.6f}."
        )
        lines.append("")
        lines.append(
            "Identity absolute loss (sMASE, recorded, not a rule): support "
            + f"{loss['support']:.6f}" + ", delayed "
            + f"{loss['delayed']:.6f}."
        )
        lines.append("")
        same = cmp_["support_champion_equals_delayed_champion"]
        if same:
            lines.append(
                "Full-batch delayed champion `"
                + cmp_["best_full_batch_delayed_program"]
                + "` equals support champion `"
                + cmp_["best_full_batch_program"] + "`."
            )
        else:
            lines.append(
                "Full-batch delayed champion `"
                + cmp_["best_full_batch_delayed_program"]
                + "` DIFFERS from support champion `"
                + cmp_["best_full_batch_program"] + "`."
            )
        lines.append("")
        lines.append("| program | support | delayed |")
        lines.append("| --- | ---: | ---: |")
        scan = rec["menu_scan"]
        def _delayed_key(op: str) -> float:
            row = scan[op]
            if "delayed_aggregate_gain" in row:
                return -float(row["delayed_aggregate_gain"])
            if "delayed" in row:
                return -float(row["delayed"]["aggregate_gain"])
            return -float(row["aggregate_gain"])
        for program in sorted(scan, key=_delayed_key):
            row = scan[program]
            delayed = row.get("delayed_aggregate_gain")
            if delayed is None and "delayed" in row:
                delayed = row["delayed"]["aggregate_gain"]
            lines.append(
                "| `" + program + "` | "
                + f"{row['aggregate_gain']:+.6f} | "
                + f"{float(delayed):+.6f} |"
            )
        lines.append("")
        if rec["adoption_trace"]:
            lines.append(
                "| candidate | reverted | support | delayed | bar | check |"
            )
            lines.append("| --- | --- | ---: | ---: | ---: | --- |")
            for row in rec["adoption_trace"]:
                bar = row.get("delayed_bar")
                lines.append(
                    "| `" + row["program"] + "` | "
                    + (", ".join(row["excluded_series"]) or "none") + " | "
                    + f"{row['support_aggregate_gain']:+.6f} | "
                    + f"{row['delayed_aggregate_gain']:+.6f} | "
                    + (f"{bar:+.6f}" if bar is not None else "--")
                    + " | " + row["stability_check"] + " |"
                )
            lines.append("")
    return "\n".join(lines)


def _run_recipe_v2_all_cells(task_index: int) -> int:
    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []
    for name in RECIPE_V2_COHORTS:
        for variant in CONSUMER_VARIANTS:
            print("RECIPE-V2 %s %s: start" % (name, variant), flush=True)
            try:
                recipe = make_batch_recipe(
                    name, task_index=task_index, consumer_variant=variant,
                    adoption_rule_version="v2",
                )
            except Exception as exc:  # noqa: BLE001
                import traceback

                cells.append({
                    "cohort": name,
                    "consumer_variant": variant,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                    "traceback": traceback.format_exc(),
                })
                print(
                    "RECIPE-V2 %s %s FAILED: %s: %s"
                    % (name, variant, type(exc).__name__, exc),
                    flush=True,
                )
                continue
            v2_sig = _v1_cell_signature(
                recipe["adopted_plan"], recipe["comparison"]
            )
            v1_sig = _load_v1_cell(name, variant)
            expected_change = (
                name == "T233" and variant == CONSUMER_PER_CHANNEL
            )
            if v1_sig is None:
                vs = "NO_V1_BASELINE"
            elif _cell_plan_equal(v1_sig, v2_sig):
                vs = "UNCHANGED"
            elif expected_change:
                vs = "EXPECTED_V2_CORRECTION"
            else:
                vs = "UNEXPECTED_CELL_CHANGE"
                unexpected.append({
                    "cohort": name,
                    "consumer_variant": variant,
                    "v1": v1_sig,
                    "v2": v2_sig,
                })
            cells.append({
                "cohort": name,
                "consumer_variant": variant,
                "vs_v1": vs,
                "v1": v1_sig,
                "recipe": recipe,
            })
            plan = recipe["adopted_plan"]
            print(
                "RECIPE-V2 %s %s %s adopted=%s program=%s excluded=%s "
                "support=%+.6f delayed=%+.6f"
                % (name, variant, vs, plan["kind"], plan["program"],
                   plan["excluded_series"],
                   recipe["comparison"]["support"]["adopted"],
                   recipe["comparison"]["delayed"]["adopted"]),
                flush=True,
            )
    result = {
        "protocol_version": "batch_recipe_v2_all_cells_v1",
        "adoption_rule_version": "v2",
        "adoption_rule": ADOPTION_RULE_V2,
        "llm_api_call_count": 0,
        "deterministic": True,
        "role": (
            "engineering effect measurement of the v2 delayed-gate correction; "
            "not authorization evidence"
        ),
        "v1_artifacts_preserved": True,
        "unexpected_cell_changes": unexpected,
        "cells": cells,
        "wall_seconds": time.perf_counter() - started,
    }
    RECIPE_V2_ALL_CELLS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RECIPE_V2_ALL_CELLS_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    RECIPE_V2_ALL_CELLS_MD.write_text(
        _recipe_v2_all_cells_markdown(result) + "\n", encoding="utf-8"
    )
    return 0


# --------------------------------------------------------------------- report
def _markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# batch-composition-headroom v1")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "The project's primary readout is now a batch question: for one batch "
        "of data, is the downstream effect after Harness processing better "
        "than (a) doing nothing (identity) and (b) the best single program "
        "applied to the whole batch?  This run measures the *headroom* of "
        "selective per-series treatment.  If one program's per-series response "
        "points in different directions on different series, then choosing the "
        "best program per series -- identity included -- should beat any "
        "uniform full-batch treatment on the aggregate readout."
    )
    lines.append("")
    lines.append(
        "**This is an engineering effect measurement, not authorization "
        "evidence.**  No Skill is written, no Episode is formed, no Fast or "
        "Slow path runs, and no execution right is granted or implied.  The "
        "per-series argmax is fitted on the same Support outcomes it is then "
        "scored on, so the Support column is an upper bound on selective "
        "headroom rather than a deployable policy; the delayed column is the "
        "out-of-selection readout.  Data is already-exposed development data "
        "only."
    )
    lines.append("")
    lines.append(
        "Protocol reuse: cohorts from `agentic.runner.load_cohort`, Consumer "
        "and Judge from `_evaluate_origins` / v6 `_evaluate` (ridge, sMASE, "
        "macro gain over identity), windows from the frozen Task roster, "
        "compile path from `task_episode_harness.runner._compiled`."
    )
    lines.append("")

    for cohort in result["cohorts"]:
        if "error" in cohort:
            lines.append(f"## Cohort `{cohort['cohort']}` -- FAILED")
            lines.append("")
            lines.append(f"```\n{cohort['error']}\n```")
            lines.append("")
            continue
        head = cohort["headline"]
        lines.append(f"## Cohort `{cohort['cohort']}`")
        lines.append("")
        lines.append(f"**Verdict: `{cohort['verdict']}`**")
        lines.append("")
        lines.append(
            f"Task `{cohort['task_episode_id']}`, Support origins "
            f"{cohort['support_origins']}, delayed origins "
            f"{cohort['delayed_origins']}; "
            f"{len(cohort['train_series'])} training series, "
            f"{len(cohort['eval_series'])} evaluation series."
        )
        lines.append("")
        lines.append("### Aggregate gain over identity (higher is better)")
        lines.append("")
        lines.append("| plan | support | delayed |")
        lines.append("| --- | ---: | ---: |")
        lines.append("| identity (baseline) | 0.000000 | 0.000000 |")
        for program in sorted(cohort["single_program_full_batch"]):
            row = cohort["single_program_full_batch"][program]
            delayed_cell = (
                f"{head['delayed']['best_single_program']:+.6f}"
                if program == cohort["best_single_program"] else "not evaluated"
            )
            mark = " **(best single)**" if program == cohort[
                "best_single_program"] else ""
            lines.append(
                f"| full batch: {program}{mark} | {row['aggregate_gain']:+.6f} "
                f"| {delayed_cell} |"
            )
        lines.append(
            f"| **composition (per-series argmax)** | "
            f"{head['support']['composition']:+.6f} | "
            f"{head['delayed']['composition']:+.6f} |"
        )
        lines.append("")
        lines.append(
            f"Composition minus best single program: "
            f"support {head['support']['composition_minus_best_single']:+.6f}, "
            f"delayed {head['delayed']['composition_minus_best_single']:+.6f}. "
            f"Ordering preserved on the delayed window: "
            f"`{head['delayed']['ordering_preserved']}`."
        )
        lines.append("")

        lines.append("### Per-series assignment")
        lines.append("")
        lines.append(
            "| training series | chosen program | its per-series gain | "
            "best-single-program gain on this series |"
        )
        lines.append("| --- | --- | ---: | ---: |")
        for row in cohort["assignment_table"]:
            lines.append(
                f"| {row['series_uid']} | {row['chosen_program']} | "
                f"{row['chosen_per_series_gain']:+.6f} | "
                f"{row['best_single_program_gain_on_this_series']:+.6f} |"
            )
        lines.append("")
        lines.append(
            "A per-series gain is the aggregate batch gain when that one "
            "series is treated and everything else is left alone -- the same "
            "singleton-scope readout the repository already uses.  Identity is "
            "always available at exactly 0."
        )
        lines.append("")

        lines.append("### Harm account")
        lines.append("")
        lines.append(
            "| plan | aggregate support gain | harmed eval series | "
            "total eval harm | harmed training series | total training harm |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for name, row in cohort["harm_account"].items():
            lines.append(
                f"| {name} | {row['aggregate_support_gain']:+.6f} | "
                f"{row['harmed_eval_series_count']} | "
                f"{row['harmed_eval_series_total_harm']:.6f} | "
                f"{row['harmed_training_series_count']} | "
                f"{row['harmed_training_series_total_harm']:.6f} |"
            )
        lines.append("")
        lines.append(
            "Two harm columns, because they answer different questions.  The "
            "eval-side count is how many downstream evaluation series ended up "
            "worse than identity under that plan; it is the one that can "
            "falsify the composition.  The training-side count is how many "
            "treated training series carry a negative singleton response; the "
            "composition drives it to zero **by construction** (identity is in "
            "the argmax), so it is a consistency check, not a finding."
        )
        lines.append("")

        interaction = cohort["composition_interaction"]
        lines.append("### Composition interaction (reported, not assumed)")
        lines.append("")
        lines.append(
            f"Sum of the chosen per-series gains: "
            f"{interaction['sum_of_chosen_per_series_gains']:+.6f}.  "
            f"Validated composition gain from a single retrain under the "
            f"assignment: {interaction['validated_composition_gain']:+.6f}.  "
            f"Cross-series retraining interaction: "
            f"{interaction['interaction']:+.6f}."
        )
        lines.append("")

        lines.append("### Response divergence")
        lines.append("")
        if not cohort["response_divergence"]:
            lines.append(
                "No program in the menu produced both a materially positive "
                "and a materially negative training series.  Per-series "
                "responses are homogeneous in sign, so there is nothing for "
                "selective treatment to exploit."
            )
        else:
            strongest = cohort["strongest_divergence"]
            lines.append(
                f"Strongest divergence: `{strongest['program']}` gains "
                f"{strongest['best_series_gain']:+.6f} on "
                f"`{strongest['best_series']}` and loses "
                f"{strongest['worst_series_gain']:+.6f} on "
                f"`{strongest['worst_series']}` -- a spread of "
                f"{strongest['spread']:.6f} for the same program on the same "
                f"batch.  {strongest['positive_series_count']} series positive, "
                f"{strongest['negative_series_count']} negative."
            )
            lines.append("")
            lines.append(
                "| program | positive series | negative series | best | "
                "worst | spread |"
            )
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for row in cohort["response_divergence"]:
                lines.append(
                    f"| {row['program']} | {row['positive_series_count']} | "
                    f"{row['negative_series_count']} | "
                    f"{row['best_series']} {row['best_series_gain']:+.6f} | "
                    f"{row['worst_series']} {row['worst_series_gain']:+.6f} | "
                    f"{row['spread']:.6f} |"
                )
        lines.append("")

        check = cohort["readout_equivalence_check"]
        lines.append(
            f"Readout equivalence check (per-series-assignment retrain "
            f"reproduces the frozen `_evaluate` readout exactly on every "
            f"uniform assignment): `{check['all_exact']}`."
        )
        lines.append("")

    lines.append("## Reading")
    lines.append("")
    lines.append(
        "Per-series responses are strongly heterogeneous -- every treatment in "
        "the menu has both materially helped and materially hurt series inside "
        "the same batch -- so `RESPONSES_HOMOGENEOUS` is ruled out on both "
        "cohorts.  The heterogeneity is real; the naive way of cashing it in is "
        "not.  Choosing each series' own argmax and applying all of those "
        "choices at once loses roughly the entire summed per-series gain to a "
        "cross-series retraining interaction, and lands below the best single "
        "full-batch program on Support and below identity on the delayed "
        "window."
    )
    lines.append("")
    lines.append(
        "The mechanism is visible in the numbers rather than inferred: the "
        "Consumer is one ridge model pooled over the whole treated batch, so a "
        "singleton-scope gain measures the marginal effect of treating one "
        "series *while the other eleven stay untreated*, and that quantity is "
        "not the credit each series contributes once they are all treated "
        "together.  Selective composition on this instrument therefore needs a "
        "credit signal that is defined jointly, or a composition rule that is "
        "validated rather than assembled.  Nothing here says selective "
        "treatment cannot pay; it says the additive singleton proxy is not the "
        "way to find out."
    )
    lines.append("")
    lines.append("## Verdict summary")
    lines.append("")
    lines.append("| cohort | verdict | identity | best single | composition |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for cohort in result["cohorts"]:
        if "error" in cohort:
            lines.append(f"| {cohort['cohort']} | RUN_FAILED | | | |")
            continue
        head = cohort["headline"]["support"]
        lines.append(
            f"| {cohort['cohort']} | `{cohort['verdict']}` | 0.000000 | "
            f"{head['best_single_program']:+.6f} "
            f"({cohort['best_single_program']}) | "
            f"{head['composition']:+.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_pooled_composition(cohort_name: str) -> dict[str, Any] | None:
    path = REPORT_JSON
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("cohorts") or []:
        if str(row.get("cohort")) == cohort_name and "error" not in row:
            return row
    return None


def _load_pooled_recipe(cohort_name: str) -> dict[str, Any] | None:
    path, _md = _recipe_paths(cohort_name)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _plan_key(recipe: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...]]:
    plan = recipe["adopted_plan"]
    return (
        str(plan["kind"]),
        str(plan["program"]),
        tuple(str(uid) for uid in plan["excluded_series"]),
    )


def _ccr_verdict(
    pooled_comp: Mapping[str, Any],
    per_channel_comp: Mapping[str, Any],
    pooled_recipe: Mapping[str, Any],
    per_channel_recipe: Mapping[str, Any],
) -> str:
    if _plan_key(pooled_recipe) != _plan_key(per_channel_recipe):
        return "CONSUMER_STRUCTURE_CHANGES_RECIPE"
    if per_channel_comp["verdict"] != "SELECTIVE_COMPOSITION_HEADROOM_PRESENT":
        return "COMPOSITION_STILL_NO_HEADROOM_UNDER_PER_CHANNEL"
    return "RECIPE_INVARIANT_TO_CONSUMER"


def _ccr_markdown(result: Mapping[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# consumer-conditioned recipe v1")
    lines.append("")
    lines.append(
        "**Engineering effect measurement, not authorization evidence.**  "
        "The Consumer variant lives only inside this experiment runner; the "
        "frozen pooled Consumer used by v6 `_evaluate` is unchanged.  No Skill "
        "is written, no Episode is formed, no Fast or Slow path is entered, "
        "and no execution right is granted or implied.  0 LLM calls.  "
        "Already-exposed development data only."
    )
    lines.append("")
    lines.append(
        "Question: the pooled ridge Consumer ate the additive per-series "
        "composition.  Does a per-channel ridge -- same features, no "
        "cross-channel pooling -- unlock that headroom, and does the adopted "
        "batch recipe change with the Consumer?"
    )
    lines.append("")
    lines.append("> **Caveat.** " + str(result["caveat"]))
    lines.append("")
    lines.append(
        "| cohort | judgment | pooled interaction | per_channel "
        "interaction | composition flip | recipe changed |"
    )
    lines.append("| --- | --- | ---: | ---: | --- | --- |")
    for row in result["cohorts"]:
        if "error" in row:
            lines.append("| " + row["cohort"] + " | RUN_FAILED | | | | |")
            continue
        lines.append(
            "| `" + row["cohort"] + "` | `" + row["judgment"] + "` | "
            + f"{row['interaction']['pooled']:+.6f} | "
            + f"{row['interaction']['per_channel']:+.6f} | "
            + str(row["composition_flip"]) + " | " + str(row["recipe_changed"]) + " |"
        )
    lines.append("")
    for row in result["cohorts"]:
        if "error" in row:
            lines.append("## Cohort `" + row["cohort"] + "` -- FAILED")
            lines.append("")
            lines.append("```")
            lines.append(str(row["error"]))
            lines.append("```")
            lines.append("")
            continue
        lines.append("## Cohort `" + row["cohort"] + "`")
        lines.append("")
        lines.append("**Judgment: `" + row["judgment"] + "`**")
        lines.append("")
        lines.append(
            "Windows: Support " + str(row["support_origins"]) + ", delayed "
            + str(row["delayed_origins"]) + "."
        )
        lines.append("")
        lines.append(
            "### Interaction (validated composition - additive expectation)"
        )
        lines.append("")
        lines.append(
            "| Consumer | additive expected | validated composition | "
            "interaction |"
        )
        lines.append("| --- | ---: | ---: | ---: |")
        for variant in (CONSUMER_POOLED, CONSUMER_PER_CHANNEL):
            block = row["composition"][variant]
            inter = block["interaction"]
            lines.append(
                "| `" + variant + "` | "
                + f"{inter['sum_of_chosen_per_series_gains']:+.6f} | "
                + f"{inter['validated_composition_gain']:+.6f} | "
                + f"{inter['interaction']:+.6f} |"
            )
        lines.append("")
        lines.append("### Composition vs best single program (Support)")
        lines.append("")
        lines.append(
            "| Consumer | verdict | best single | composition | "
            "composition - best |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for variant in (CONSUMER_POOLED, CONSUMER_PER_CHANNEL):
            block = row["composition"][variant]
            head = block["headline"]["support"]
            lines.append(
                "| `" + variant + "` | `" + block["verdict"] + "` | "
                + f"{head['best_single_program']:+.6f} "
                + "(" + str(block["best_single_program"]) + ") | "
                + f"{head['composition']:+.6f} | "
                + f"{head['composition_minus_best_single']:+.6f} |"
            )
        lines.append("")
        lines.append(
            "Composition flip under per_channel: "
            + ("yes" if row["composition_flip"] else "no")
            + "."
        )
        lines.append("")
        lines.append("### Adopted recipe")
        lines.append("")
        lines.append(
            "| Consumer | kind | program | reverted | support | delayed |"
        )
        lines.append("| --- | --- | --- | --- | ---: | ---: |")
        for variant in (CONSUMER_POOLED, CONSUMER_PER_CHANNEL):
            rec = row["recipe"][variant]
            plan = rec["adopted_plan"]
            cmp_ = rec["comparison"]
            reverted = (
                ", ".join("`" + u + "`" for u in plan["excluded_series"])
                if plan["excluded_series"] else "none"
            )
            lines.append(
                "| `" + variant + "` | `" + plan["kind"] + "` | `"
                + plan["program"] + "` | "
                + reverted + " | "
                + f"{cmp_['support']['adopted']:+.6f} | "
                + f"{cmp_['delayed']['adopted']:+.6f} |"
            )
        lines.append("")
        lines.append(
            "### per_channel menu scan (full batch, Support aggregate gain)"
        )
        lines.append("")
        lines.append("| program | support aggregate gain |")
        lines.append("| --- | ---: |")
        scan = row["per_channel_menu_scan"]
        searched = row["recipe"][CONSUMER_PER_CHANNEL]["programs_searched"]
        for program in sorted(scan, key=lambda op: -scan[op]):
            mark = " (searched)" if program in searched else ""
            lines.append(
                "| `" + program + "`" + mark + " | "
                + f"{scan[program]:+.6f} |"
            )
        lines.append("")
    return "\n".join(lines)


def _run_consumer_conditioned(
    cohort_names: Sequence[str], task_index: int,
) -> int:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for name in cohort_names:
        print("CCR %s: start" % name, flush=True)
        try:
            pooled_comp_loaded = _load_pooled_composition(name)
            pooled_comp = pooled_comp_loaded
            if pooled_comp is None:
                print("CCR %s: running pooled composition" % name, flush=True)
                pooled_comp = run_cohort(
                    name, task_index=task_index,
                    consumer_variant=CONSUMER_POOLED,
                )
            pooled_recipe_loaded = _load_pooled_recipe(name)
            pooled_recipe = pooled_recipe_loaded
            if pooled_recipe is None:
                print("CCR %s: running pooled recipe" % name, flush=True)
                pooled_recipe = make_batch_recipe(
                    name, task_index=task_index,
                    consumer_variant=CONSUMER_POOLED,
                    adoption_rule_version="v1",
                )
            print("CCR %s: running per_channel composition" % name, flush=True)
            pc_comp = run_cohort(
                name, task_index=task_index,
                consumer_variant=CONSUMER_PER_CHANNEL,
            )
            print("CCR %s: running per_channel recipe" % name, flush=True)
            pc_recipe = make_batch_recipe(
                name, task_index=task_index,
                consumer_variant=CONSUMER_PER_CHANNEL,
                adoption_rule_version="v1",
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            rows.append({
                "cohort": name,
                "error": "%s: %s" % (type(exc).__name__, exc),
                "traceback": traceback.format_exc(),
            })
            print("CCR %s FAILED: %s: %s" % (name, type(exc).__name__, exc), flush=True)
            continue
        judgment = _ccr_verdict(pooled_comp, pc_comp, pooled_recipe, pc_recipe)
        recipe_changed = _plan_key(pooled_recipe) != _plan_key(pc_recipe)
        composition_flip = (
            pooled_comp["verdict"] == "COMPOSITION_NO_HEADROOM"
            and pc_comp["verdict"] == "SELECTIVE_COMPOSITION_HEADROOM_PRESENT"
        )
        menu = {
            program: float(gain_row["aggregate_gain"])
            for program, gain_row in pc_comp["single_program_full_batch"].items()
        }
        row = {
            "cohort": name,
            "exposure": pc_comp["exposure"],
            "support_origins": pc_comp["support_origins"],
            "delayed_origins": pc_comp["delayed_origins"],
            "judgment": judgment,
            "composition_flip": composition_flip,
            "recipe_changed": recipe_changed,
            "interaction": {
                "pooled": pooled_comp["composition_interaction"]["interaction"],
                "per_channel": pc_comp["composition_interaction"]["interaction"],
            },
            "composition": {
                CONSUMER_POOLED: {
                    "verdict": pooled_comp["verdict"],
                    "best_single_program": pooled_comp["best_single_program"],
                    "headline": pooled_comp["headline"],
                    "interaction": pooled_comp["composition_interaction"],
                    "source": (
                        "loaded from artifacts/functional/e2/"
                        "batch_composition_headroom_v1.json"
                        if pooled_comp_loaded is not None
                        else "run in this session under consumer_variant=pooled"
                    ),
                },
                CONSUMER_PER_CHANNEL: {
                    "verdict": pc_comp["verdict"],
                    "best_single_program": pc_comp["best_single_program"],
                    "headline": pc_comp["headline"],
                    "interaction": pc_comp["composition_interaction"],
                    "source": (
                        "run in this session under consumer_variant=per_channel"
                    ),
                },
            },
            "recipe": {
                CONSUMER_POOLED: {
                    "adopted_plan": pooled_recipe["adopted_plan"],
                    "comparison": pooled_recipe["comparison"],
                    "adoption_path": pooled_recipe["adoption_path"],
                    "programs_searched": pooled_recipe["programs_searched"],
                    "harm_account": pooled_recipe["harm_account"],
                },
                CONSUMER_PER_CHANNEL: {
                    "adopted_plan": pc_recipe["adopted_plan"],
                    "comparison": pc_recipe["comparison"],
                    "adoption_path": pc_recipe["adoption_path"],
                    "programs_searched": pc_recipe["programs_searched"],
                    "harm_account": pc_recipe["harm_account"],
                },
            },
            "per_channel_menu_scan": menu,
        }
        rows.append(row)
        print(
            "CCR RESULT %s %s pooled_inter=%+.6f per_channel_inter=%+.6f "
            "flip=%s recipe_changed=%s pc_recipe=%s %s %s"
            % (name, judgment,
               row["interaction"]["pooled"], row["interaction"]["per_channel"],
               composition_flip, recipe_changed,
               pc_recipe["adopted_plan"]["kind"],
               pc_recipe["adopted_plan"]["program"],
               pc_recipe["adopted_plan"]["excluded_series"]),
            flush=True,
        )
    result = {
        "protocol_version": "consumer_conditioned_recipe_v1",
        "llm_api_call_count": 0,
        "deterministic": True,
        "role": (
            "engineering effect measurement of whether the batch recipe is "
            "Consumer-structure-conditioned; not authorization evidence, no "
            "Skill written, no Fast/Slow path entered"
        ),
        "consumer_variant_scope": (
            "per_channel exists only inside this experiment runner; the frozen "
            "pooled Consumer (v6._evaluate, one ridge on stacked training "
            "windows) is unchanged"
        ),
        "per_channel_definition": (
            "same window construction, same ridge (alpha=1, unpenalized "
            "intercept); each training channel fits its own ridge on its own "
            "windows; each eval channel is predicted by the equal-weight mean "
            "of those channel-wise ridges"
        ),
        "data_exposure": (
            "already-exposed development cohorts only; traffic uses "
            "development origins 1104/1368/1800 and does not cross "
            "sealed_from_index=3072; no NOAA, no KDD W3, no "
            "g3_final_query_outcome"
        ),
        "caveat": (
            "The delayed window participates in recipe adoption, so both "
            "columns of a recipe are in-selection.  Pooled composition numbers "
            "for electricity and T233 are loaded from the already-accepted "
            "headroom artifact when present; traffic pooled composition is run "
            "here if that artifact has no traffic row.  per_channel numbers "
            "are produced in this session."
        ),
        "cohorts": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    CCR_JSON.parent.mkdir(parents=True, exist_ok=True)
    CCR_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    CCR_MD.write_text(_ccr_markdown(result) + "\n", encoding="utf-8")
    return 0


def _run_masked(cohort_names: Sequence[str], task_index: int) -> int:
    started = time.perf_counter()
    cohorts: list[dict[str, Any]] = []
    for name in cohort_names:
        try:
            cohorts.append(run_masked_cohort(name, task_index=task_index))
        except Exception as exc:  # noqa: BLE001
            import traceback

            cohorts.append({
                "cohort": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            print(f"BCH-M {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
    result = {
        "protocol_version": MASKED_PROTOCOL_VERSION,
        "llm_api_call_count": 0,
        "deterministic": True,
        "role": (
            "engineering effect measurement of single-program-plus-exclusion-"
            "mask headroom; not authorization evidence, no Skill written, no "
            "Fast/Slow path entered"
        ),
        "selection_discipline": (
            "pre-declared: every accept/reject reads the Support aggregate "
            "only; the delayed window is evaluated for each accepted mask and "
            "reported, never selected on"
        ),
        "data_exposure": (
            "already-exposed development cohorts only; no NOAA, no KDD W3, no "
            "g3_final_query_outcome"
        ),
        "consumer": "ridge_alpha_1_with_intercept, sMASE, macro gain over identity",
        "follows_up_on": PROTOCOL_VERSION,
        "cohorts": cohorts,
        "wall_seconds": time.perf_counter() - started,
    }
    MASKED_REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    MASKED_REPORT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    MASKED_REPORT_MD.write_text(_masked_markdown(result) + "\n", encoding="utf-8")
    for cohort in cohorts:
        if "error" in cohort:
            print(f"BCH-M RESULT {cohort['cohort']}: RUN_FAILED", flush=True)
            continue
        head = cohort["headline"]
        plan = cohort["best_masked_plan"]
        print(
            "BCH-M RESULT %s %s best=%s excluded=%s support(masked=%+.6f "
            "full=%+.6f) delayed(masked=%+.6f full=%+.6f)"
            % (cohort["cohort"], cohort["verdict"], plan["program"],
               plan["excluded_series"], head["support"]["masked"],
               head["support"]["best_full_batch_single_program"],
               head["delayed"]["masked"],
               head["delayed"]["best_full_batch_single_program"]),
            flush=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--mode", default="composition",
                        choices=("composition", "masked", "recipe",
                                 "consumer_conditioned",
                                 "recipe_v2_all_cells"))
    parser.add_argument(
        "--consumer-variant", default=CONSUMER_POOLED,
        choices=CONSUMER_VARIANTS,
        help="experiment-local Consumer; pooled is the frozen default",
    )
    args = parser.parse_args(argv)

    if args.mode == "masked":
        return _run_masked(args.cohorts, args.task_index)
    if args.mode == "recipe":
        return _run_recipe(
            args.cohorts, args.task_index, args.consumer_variant,
        )
    if args.mode == "consumer_conditioned":
        return _run_consumer_conditioned(args.cohorts, args.task_index)
    if args.mode == "recipe_v2_all_cells":
        return _run_recipe_v2_all_cells(args.task_index)

    if args.consumer_variant != CONSUMER_POOLED:
        cohorts: list[dict[str, Any]] = []
        for name in args.cohorts:
            try:
                cohorts.append(run_cohort(
                    name, task_index=args.task_index,
                    consumer_variant=args.consumer_variant,
                ))
            except Exception as exc:  # noqa: BLE001
                import traceback

                cohorts.append({
                    "cohort": name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                })
                print(f"BCH {name} FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
        for cohort in cohorts:
            if "error" in cohort:
                print(f"BCH RESULT {cohort['cohort']}: RUN_FAILED", flush=True)
                continue
            head = cohort["headline"]["support"]
            print(
                "BCH RESULT %s consumer=%s %s identity=0 best_single=%s "
                "%+.6f composition=%+.6f (not writing pooled composition "
                "artifacts)"
                % (cohort["cohort"], args.consumer_variant, cohort["verdict"],
                   cohort["best_single_program"], head["best_single_program"],
                   head["composition"]),
                flush=True,
            )
        return 0

    started = time.perf_counter()
    cohorts: list[dict[str, Any]] = []
    for name in args.cohorts:
        try:
            cohorts.append(run_cohort(name, task_index=args.task_index))
        except Exception as exc:  # noqa: BLE001
            import traceback

            cohorts.append({
                "cohort": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            })
            print(f"BCH {name} FAILED: {type(exc).__name__}: {exc}", flush=True)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "llm_api_call_count": 0,
        "deterministic": True,
        "role": (
            "engineering effect measurement of selective-treatment headroom; "
            "not authorization evidence, no Skill written, no Fast/Slow path "
            "entered"
        ),
        "data_exposure": (
            "already-exposed development cohorts only; no NOAA, no KDD W3, no "
            "g3_final_query_outcome"
        ),
        "consumer": "ridge_alpha_1_with_intercept, sMASE, macro gain over identity",
        "cohorts": cohorts,
        "wall_seconds": time.perf_counter() - started,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    REPORT_MD.write_text(_markdown(result) + "\n", encoding="utf-8")
    for cohort in cohorts:
        if "error" in cohort:
            print(f"BCH RESULT {cohort['cohort']}: RUN_FAILED", flush=True)
            continue
        head = cohort["headline"]["support"]
        print(
            "BCH RESULT %s %s identity=0 best_single=%s %+.6f composition=%+.6f"
            % (cohort["cohort"], cohort["verdict"],
               cohort["best_single_program"], head["best_single_program"],
               head["composition"]),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
