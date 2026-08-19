"""G3 candidate screening.  Outcome-blind, and the final query stays sealed.

The criteria below are fixed before any candidate is opened, because the point
of screening is to reject a dataset the Judge cannot read -- not to search for
one whose numbers flatter the method.  The Weather lesson is the reason this
file exists: there, ``macro_gain`` was an unweighted mean over eval channels
whose identity losses spanned 24.4x, so three unforecastable solar channels
carried a result the two well-forecast channels contradicted.  T233, which
reads cleanly, spans 2.0x.

Information wall
----------------
Everything here reads either a public prefix ``raw[:origin]`` or a development
time block strictly earlier than the G3 roster's first Support origin.  The
substrate guards are outcome-blind by construction -- they look at
``raw[origin-192:origin]`` and ``raw[anchor-192:anchor+48]`` at anchors far
below the roster.  Judge readability is measured only on the development
block.  The G3 roster's own Outcomes are never evaluated here; opening them is
the single one-shot event the G3 run performs.

Selecting on the final query loss is what this file is built to prevent, so it
has no code path that can reach one.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ---- pre-registered screening criteria (fixed before any candidate opened) --
MIN_TRAIN_SERIES = 12
MIN_EVAL_SERIES = 8
# 9 Task Episodes at the frozen roster stride reach index 5376 + 5*48 = 5616,
# plus one horizon of truth.
MIN_SERIES_LENGTH = 5760
# Judge readability, measured on the development block only.  Weather failed
# at 24.4x and T233 passes at 2.0x; 5x sits between them and is set here, once.
MAX_EVAL_LOSS_SPREAD = 5.0
# No single eval series may carry the aggregate.  With 8 eval series an even
# split is 12.5%; 40% is three times that and still well short of dominance.
MAX_SINGLE_SERIES_LOSS_SHARE = 0.40
# The Operator DSL must have something publicly visible to act on.
MIN_SERIES_WITH_PUBLIC_PHENOMENON = 4
# Development origins.  Strictly below the roster's first Support origin, so
# nothing measured here overlaps anything the G3 run will open.
DEVELOPMENT_ORIGINS = (1104, 1368, 1800)
SEALED_FROM_INDEX = 3072

CRITERIA = {
    "min_train_series": MIN_TRAIN_SERIES,
    "min_eval_series": MIN_EVAL_SERIES,
    "min_series_length": MIN_SERIES_LENGTH,
    "max_eval_loss_spread": MAX_EVAL_LOSS_SPREAD,
    "max_single_series_loss_share": MAX_SINGLE_SERIES_LOSS_SHARE,
    "min_series_with_public_phenomenon": MIN_SERIES_WITH_PUBLIC_PHENOMENON,
    "development_origins": list(DEVELOPMENT_ORIGINS),
    "sealed_from_index": SEALED_FROM_INDEX,
    "fixed_before_any_candidate_was_opened": True,
    "rationale": (
        "Weather's Judge was unreadable at a 24.4x eval-loss spread and T233 "
        "reads at 2.0x; the bar is set once, between them, and is never moved "
        "to make a candidate pass"
    ),
}


def load_csv_columns(
    path: Path,
    *,
    max_columns: int = 400,
    max_rows: int = 20000,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Numeric columns of a wide CSV, truncated to a bounded prefix.

    Only the first ``max_rows`` rows are read.  That is far more than the
    roster needs and keeps the reader from touching the far tail of a large
    file for no reason.
    """
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows: list[list[str]] = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append(row)
    names: list[str] = []
    values: dict[str, np.ndarray] = {}
    for column, name in enumerate(header[:max_columns]):
        raw = np.array(
            [row[column] if column < len(row) else "" for row in rows],
            dtype=object,
        )
        try:
            series = np.array(
                [float(item) if item not in ("", None) else np.nan
                 for item in raw],
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            continue
        if not np.isfinite(series).any():
            continue
        names.append(str(name))
        values[str(name)] = series
    return names, values


def public_phenomenon_census(
    values: Mapping[str, np.ndarray],
    uids: Sequence[str],
    cutoff: int,
) -> dict[str, Any]:
    """Does the Operator DSL have anything publicly visible to act on?

    Reads the public prefix only, through the same extractor the Workspace
    tools use, so this sees exactly what the Agent would be able to see.
    """
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
        extract_public_features,
    )

    per_series: dict[str, Any] = {}
    for uid in uids:
        prefix = np.asarray(values[str(uid)], dtype=np.float64)[:cutoff]
        features = dict(extract_public_features(prefix, task_kind="forecast"))
        missing = float(features.get("missing_fraction", 0.0))
        outlier = float(features.get("outlier_fraction", 0.0) or 0.0)
        peak = float(features.get("local_robust_z_peak", 0.0) or 0.0)
        region = (
            float(features.get("estimated_region_end_fraction", 0.0))
            - float(features.get("estimated_region_start_fraction", 0.0))
        )
        per_series[str(uid)] = {
            "missing_fraction": missing,
            "outlier_fraction": outlier,
            "local_robust_z_peak": peak,
            "estimated_region_width": region,
            "has_public_phenomenon": bool(
                missing > 0.0 or outlier > 0.0 or peak >= 4.0
            ),
        }
    count = sum(1 for row in per_series.values() if row["has_public_phenomenon"])
    return {
        "series_with_public_phenomenon": count,
        "pass": count >= MIN_SERIES_WITH_PUBLIC_PHENOMENON,
        "per_series": per_series,
    }


def development_judge_readability(
    roster: Sequence[Mapping[str, Any]],
    values: Mapping[str, np.ndarray],
    eval_uids: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Identity loss per eval series, on the development block only.

    No candidate Program runs here and no roster Outcome is opened: this is
    the identity baseline at development origins strictly below
    ``SEALED_FROM_INDEX``.  It answers one question -- can the aggregate the
    Judge reports be read at all, or is it a few broken channels in a trench
    coat.
    """
    from evaluation.functional.task_episode_harness.runner import (
        _evaluate_origins,
    )

    assert max(DEVELOPMENT_ORIGINS) < SEALED_FROM_INDEX
    rows = _evaluate_origins(
        list(roster), dict(values), None, dict(config), DEVELOPMENT_ORIGINS, None
    )
    losses = [
        float(np.mean([row["per_view_smase"][index] for row in rows]))
        for index in range(len(eval_uids))
    ]
    total = float(sum(losses))
    spread = (max(losses) / min(losses)) if min(losses) > 0 else float("inf")
    share = (max(losses) / total) if total > 0 else 1.0
    return {
        "development_origins": list(DEVELOPMENT_ORIGINS),
        "sealed_from_index": SEALED_FROM_INDEX,
        "per_series_identity_smase": {
            str(uid): value for uid, value in zip(eval_uids, losses)
        },
        "min": min(losses), "max": max(losses),
        "eval_loss_spread": spread,
        "largest_single_series_loss_share": share,
        "pass": bool(
            spread <= MAX_EVAL_LOSS_SPREAD
            and share <= MAX_SINGLE_SERIES_LOSS_SHARE
        ),
        "reason": (
            "" if spread <= MAX_EVAL_LOSS_SPREAD
            else f"eval loss spread {spread:.1f}x exceeds "
                 f"{MAX_EVAL_LOSS_SPREAD}x; the aggregate would be carried by "
                 "series the forecaster cannot read"
        ),
    }


def screen_candidate(
    name: str,
    values: Mapping[str, np.ndarray],
    names: Sequence[str],
    repo_root: Path,
) -> dict[str, Any]:
    """One candidate, all criteria, in the order that fails cheapest first."""
    from evaluation.functional.task_episode_harness import g1
    from evaluation.functional.task_episode_harness.e1 import (
        _frozen_task_roster,
    )
    from run_v1_kdd2018_natural_slow_update import _config

    config = dict(_config())
    anchors = [int(a) for a in config["anchors"]]
    specs = list(_frozen_task_roster()[:9])
    result: dict[str, Any] = {"candidate": name, "criteria": CRITERIA}

    long_enough = [
        uid for uid in names
        if np.asarray(values[uid]).size >= MIN_SERIES_LENGTH
    ]
    result["structure"] = {
        "numeric_column_count": len(names),
        "columns_long_enough": len(long_enough),
        "series_length": (
            int(np.asarray(values[names[0]]).size) if names else 0
        ),
        "pass": len(long_enough) >= MIN_TRAIN_SERIES + MIN_EVAL_SERIES,
    }
    if not result["structure"]["pass"]:
        result["verdict"] = "REJECTED_STRUCTURE"
        return result

    train_pf = g1.train_substrate_preflight(values, long_enough, anchors)
    eval_pf = g1.eval_substrate_preflight(values, long_enough, specs)
    clean = [
        uid for uid in long_enough
        if train_pf["per_series"][uid]["clean"]
        and eval_pf["per_series"][uid]["clean"]
    ]
    result["substrate"] = {
        "columns_clean_under_both_guards": len(clean),
        "pass": len(clean) >= MIN_TRAIN_SERIES + MIN_EVAL_SERIES,
        "zero_new_outcome": True,
    }
    if not result["substrate"]["pass"]:
        result["verdict"] = "REJECTED_SUBSTRATE"
        return result

    train = clean[:MIN_TRAIN_SERIES]
    ev = clean[MIN_TRAIN_SERIES:MIN_TRAIN_SERIES + MIN_EVAL_SERIES]
    roster = (
        [{"series_uid": uid, "role": "train"} for uid in train]
        + [{"series_uid": uid, "role": "eval"} for uid in ev]
    )
    result["roster"] = {"train": train, "eval": ev}

    phenomena = public_phenomenon_census(
        values, train, int(specs[0]["support_origins"][0])
    )
    result["public_phenomena"] = phenomena
    if not phenomena["pass"]:
        result["verdict"] = "REJECTED_NO_PUBLIC_PHENOMENON"
        return result

    readability = development_judge_readability(roster, values, ev, config)
    result["judge_readability"] = readability
    if not readability["pass"]:
        result["verdict"] = "REJECTED_JUDGE_UNREADABLE"
        return result

    result["verdict"] = "ACCEPTED"
    return result


__all__ = [
    "CRITERIA",
    "development_judge_readability",
    "load_csv_columns",
    "public_phenomenon_census",
    "screen_candidate",
]
