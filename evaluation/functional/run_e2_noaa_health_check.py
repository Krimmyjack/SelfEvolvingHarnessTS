"""Is the NOAA reserve worth burning?  An instrument check before the spend.

``e1.SEALED_CONFIRMATION_DATASET`` names ``noaa_global_hourly`` as the last
sealed confirmation set.  Before any of it is spent, one question has to be
answered that cannot be answered afterwards: is the instrument readable on
this cohort at all?  Weather was rejected at a 24.4x eval-loss spread -- its
aggregate was carried by channels the forecaster could not read -- and that is
exactly the failure that looks like a null result if you only discover it
after opening the Outcome.

This check touches the development region only.  Every series it loads is
truncated at ``g3_sourcing.SEALED_FROM_INDEX`` before anything sees it, and
the truncation is asserted, so no index at or past the boundary can be read by
construction rather than by care.

0 LLM calls.  The frozen screening criteria are read from ``g3_sourcing``, not
restated here, and no threshold is moved to make anything pass.

Writes ``artifacts/functional/e2/noaa_health_check_v1.json`` and ``.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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

import numpy as np  # noqa: E402

import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402

from evaluation.functional.task_episode_harness import e1 as e1mod  # noqa: E402
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    g3_sourcing,
)
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

PROTOCOL_VERSION = "noaa_health_check_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "noaa_health_check_v1.json"
OUT_MD = E2 / "noaa_health_check_v1.md"

DATASET_ID = str(e1mod.SEALED_CONFIRMATION_DATASET)
REGISTRY = PROJECT_ROOT / "artifacts/frozen/benchmark_v02/series_registry.jsonl"
MANIFEST = PROJECT_ROOT / "artifacts/frozen/benchmark_v02/dataset_manifest.json"
CLEAN_BASE = PROJECT_ROOT / "data/benchmark_v0_2/clean_base"
RAW_DIR = PROJECT_ROOT / "data/benchmark_v0/raw" / DATASET_ID

# Every constant below is read from the frozen module, never restated.
SEALED_FROM_INDEX = int(g3_sourcing.SEALED_FROM_INDEX)
DEVELOPMENT_ORIGINS = tuple(int(o) for o in g3_sourcing.DEVELOPMENT_ORIGINS)
CRITERIA = dict(g3_sourcing.CRITERIA)
MIN_TRAIN_SERIES = int(g3_sourcing.MIN_TRAIN_SERIES)
MIN_EVAL_SERIES = int(g3_sourcing.MIN_EVAL_SERIES)
MIN_SERIES_LENGTH = int(g3_sourcing.MIN_SERIES_LENGTH)
MAX_EVAL_LOSS_SPREAD = float(g3_sourcing.MAX_EVAL_LOSS_SPREAD)
MAX_SINGLE_SERIES_LOSS_SHARE = float(g3_sourcing.MAX_SINGLE_SERIES_LOSS_SHARE)
HORIZON = int(v6.HORIZON)
CONTEXT_LENGTH = int(v6.CONTEXT_LENGTH)
CONSUMER_POOLED = bch.CONSUMER_POOLED
CONSUMER_PER_CHANNEL = bch.CONSUMER_PER_CHANNEL

RETRAIN_BUDGET = 12
MIN_CHANNELS_FOR_PER_CHANNEL = 2

PRE_REGISTERED: dict[str, Any] = {
    "fixed_before_the_run": True,
    "zero_llm": True,
    "why_before_the_spend": (
        "unreadability cannot be diagnosed after the Outcome is opened: it "
        "looks like a null result.  Weather failed at a 24.4x eval-loss "
        "spread and T233 reads at 2.0x"
    ),
    "sealed_rule": (
        "no index at or past g3_sourcing.SEALED_FROM_INDEX=%d is read.  Every "
        "series is truncated to that prefix at load time and the truncation "
        "is asserted, so the constraint holds by construction" % SEALED_FROM_INDEX
    ),
    "step_0_in_place": (
        "the data has to exist and be readable, and its path, series count, "
        "channel structure and shortest length are quoted verbatim from the "
        "frozen registry rather than recalled"
    ),
    "step_1_outcome_blind": (
        "the frozen screening criteria in g3_sourcing, applied through its own "
        "functions: structure (min_train_series, min_eval_series, "
        "min_series_length), the substrate double guard, and the public "
        "phenomenon census.  Any one failing is NOAA_STRUCTURE_FAIL and step 2 "
        "does not run"
    ),
    "step_2_readability_probe": (
        "identity baseline only, at the frozen development origins, read for "
        "eval-loss spread and largest single-series loss share against the "
        "frozen bars; pooled always, per_channel only when the channel "
        "structure supports it"
    ),
    "criteria_are_read_not_restated": True,
    "criteria": CRITERIA,
    "per_channel_rule": (
        "per_channel is measured only when the cohort has at least %d "
        "training channels" % MIN_CHANNELS_FOR_PER_CHANNEL
    ),
    "delayed_reading_rule": (
        "the last development origin is read as the delayed block; its "
        "readings must be finite, strictly positive and not degenerate to a "
        "single value"
    ),
    "retrain_budget": RETRAIN_BUDGET,
    "verdicts": [
        "NOAA_HEALTH_PROCEED: every measured Consumer variant clears the "
        "frozen bars",
        "PROCEED_POOLED_ONLY: pooled clears them and per_channel does not",
        "NOAA_JUDGE_UNREADABLE: the readability probe ran and failed",
        "NOAA_STRUCTURE_FAIL: an outcome-blind criterion failed, so the probe "
        "was not run",
        "NOAA_DATA_MISSING: the data is not there or cannot be read",
    ],
    "verdicts_are_reported_side_by_side": True,
    "no_threshold_is_moved": True,
}


# ------------------------------------------------------------------ step 0
def _sealed_semantics() -> dict[str, Any]:
    return {
        "kind": "index_sealed_boundary",
        "sealed_from_index": SEALED_FROM_INDEX,
        "constant_source": (
            "evaluation/functional/task_episode_harness/agentic/"
            "g3_sourcing.py::SEALED_FROM_INDEX"
        ),
        "cross_check_recipe_module": int(bch._TRAFFIC_SEALED_FROM_INDEX),
        "development_origins": list(DEVELOPMENT_ORIGINS),
        "horizon": HORIZON,
        "farthest_index_the_probe_would_read": max(DEVELOPMENT_ORIGINS) + HORIZON,
        "probe_inside_boundary": (
            max(DEVELOPMENT_ORIGINS) + HORIZON <= SEALED_FROM_INDEX
        ),
        "enforcement": (
            "every loaded series is sliced to [:%d] before any consumer sees "
            "it, and an assertion refuses to continue if any array is longer"
            % SEALED_FROM_INDEX
        ),
    }


def _registry_rows() -> list[dict[str, Any]]:
    if not REGISTRY.is_file():
        return []
    return [
        json.loads(line)
        for line in REGISTRY.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and json.loads(line).get("dataset_id") == DATASET_ID
    ]


def _manifest_entry() -> dict[str, Any] | None:
    if not MANIFEST.is_file():
        return None
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    datasets = payload.get("datasets")
    if isinstance(datasets, Mapping):
        entry = datasets.get(DATASET_ID)
        if isinstance(entry, Mapping):
            return dict(entry)
    return None


def _materialized(rows: Sequence[Mapping[str, Any]]) -> dict[str, Path]:
    wanted = {str(row["series_uid"]) for row in rows}
    found: dict[str, Path] = {}
    if not CLEAN_BASE.is_dir():
        return found
    for record_path in sorted(CLEAN_BASE.glob("*/record.json")):
        record = json.loads(record_path.read_text(encoding="utf-8"))
        uid = str(record.get("series_uid", ""))
        if uid in wanted and (record_path.parent / "values.npy").is_file():
            found[uid] = record_path.parent
    return found


def _load_truncated(paths: Mapping[str, Path]) -> tuple[
    dict[str, np.ndarray], dict[str, Any]
]:
    """Load every series, truncated at the sealed boundary, and prove it."""
    values: dict[str, np.ndarray] = {}
    truncated: list[str] = []
    for uid, directory in sorted(paths.items()):
        handle = np.load(directory / "values.npy", mmap_mode="r")
        if int(handle.shape[0]) > SEALED_FROM_INDEX:
            truncated.append(uid)
        series = np.array(
            handle[:SEALED_FROM_INDEX], dtype=np.float64, copy=True
        )
        del handle
        values[uid] = series
    sizes = [int(series.size) for series in values.values()]
    if sizes and max(sizes) > SEALED_FROM_INDEX:
        raise SystemExit(
            "a loaded series is longer than the sealed boundary; refusing to "
            "continue"
        )
    return values, {
        "loaded_series": len(values),
        "series_truncated_at_the_boundary": len(truncated),
        "truncated_examples": truncated[:8],
        "longest_loaded_length": max(sizes) if sizes else 0,
        "shortest_loaded_length": min(sizes) if sizes else 0,
        "no_index_at_or_past_boundary_was_read": True,
    }


def step_0() -> dict[str, Any]:
    rows = _registry_rows()
    result: dict[str, Any] = {
        "dataset_id": DATASET_ID,
        "designated_by": (
            "evaluation/functional/task_episode_harness/e1.py::"
            "SEALED_CONFIRMATION_DATASET"
        ),
        "registry_path": REGISTRY.relative_to(PROJECT_ROOT).as_posix(),
        "registry_exists": REGISTRY.is_file(),
        "clean_base_path": CLEAN_BASE.relative_to(PROJECT_ROOT).as_posix(),
        "sealed_semantics": _sealed_semantics(),
        "manifest_entry": _manifest_entry(),
    }
    if not rows:
        result["status"] = "NOAA_DATA_MISSING"
        result["reason"] = (
            "no %s row in %s" % (DATASET_ID, result["registry_path"])
        )
        return result
    paths = _materialized(rows)
    if not paths:
        result["status"] = "NOAA_DATA_MISSING"
        result["reason"] = (
            "%d registry rows but no materialized record under %s"
            % (len(rows), result["clean_base_path"])
        )
        result["registry_series_count"] = len(rows)
        return result
    values, load_report = _load_truncated(paths)
    lengths = Counter(int(row.get("length", 0)) for row in rows)
    result.update({
        "status": "READABLE",
        "registry_series_count": len(rows),
        "materialized_series_count": len(paths),
        "unmaterialized_series_count": len(rows) - len(paths),
        "registry_lengths": {
            str(key): value for key, value in sorted(lengths.items())
        },
        "shortest_registry_length": min(lengths),
        "longest_registry_length": max(lengths),
        "frequency": sorted({str(row.get("frequency")) for row in rows}),
        "exposure_class": sorted(
            {str(row.get("exposure_class")) for row in rows}
        ),
        "overlap_family": sorted({str(row.get("overlap_family")) for row in rows}),
        "overlap_status": sorted({str(row.get("overlap_status")) for row in rows}),
        "roles_allowed": sorted(
            {json.dumps(row.get("roles_allowed"), sort_keys=True) for row in rows}
        ),
        "channel_structure": {
            "shape": "one univariate series per record",
            "series_are_channels": True,
            "distinct_entities": len({str(row.get("entity_id")) for row in rows}),
            "entity_field": "entity_id, the NOAA ISD station identifier",
            "array_shape_example": list(
                np.asarray(next(iter(values.values()))).shape
            ),
            "note": (
                "there is no multi-channel matrix here: each record is one "
                "station's hourly series, so a per_channel Consumer treats "
                "each training series as its own channel"
            ),
        },
        "example_record_dir": (
            sorted(paths.values())[0].relative_to(PROJECT_ROOT).as_posix()
        ),
        "raw_source_dir": RAW_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "raw_source_dir_exists": RAW_DIR.is_dir(),
        "raw_station_files": (
            sum(
                1 for path in RAW_DIR.rglob("*.csv")
                if not path.name.endswith(".asset.json")
            ) if RAW_DIR.is_dir() else 0
        ),
        "raw_note": (
            "listed by filesystem metadata only; not one byte of raw series "
            "content is read by this check"
        ),
        "load_report": load_report,
    })
    result["_values"] = values
    result["_uids"] = sorted(values)
    return result


# ------------------------------------------------------------------ step 1
def _frozen_screen_reach() -> dict[str, Any]:
    """How far into the index axis the frozen screen's own guards reach.

    Recorded because it is a property of the screen, not of NOAA: the eval
    substrate guard validates the nine frozen roster windows, and those live
    at or past the sealed boundary.  On any cohort, that part of the screen
    cannot be completed under a zero-read rule.
    """
    specs = list(e1mod._frozen_task_roster()[:9])
    anchors = [int(a) for a in dict(_config())["anchors"]]
    roster_indices = [
        int(origin)
        for spec in specs
        for role in ("support_origins", "delayed_origins")
        for origin in spec[role]
    ]
    train_guard_far = max(anchors) + HORIZON
    eval_guard_far = max(roster_indices)
    return {
        "public_phenomenon_cutoff": int(specs[0]["support_origins"][0]),
        "train_guard_anchors": anchors,
        "train_guard_farthest_index": train_guard_far,
        "train_guard_inside_boundary": train_guard_far <= SEALED_FROM_INDEX,
        "eval_guard_roster_tasks": [
            str(spec["task_episode_id"]) for spec in specs
        ],
        "eval_guard_farthest_index": eval_guard_far,
        "eval_guard_inside_boundary": eval_guard_far <= SEALED_FROM_INDEX,
        "note": (
            "the frozen roster's first Support origin is the sealed boundary "
            "itself, so the eval substrate guard necessarily reads into the "
            "sealed region.  Under this check's zero-read rule that guard "
            "cannot be completed on any cohort, NOAA included"
        ),
    }


def step_1(
    values: Mapping[str, np.ndarray],
    uids: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """The outcome-blind screen, through g3_sourcing's own criteria."""
    materialized = set(str(uid) for uid in uids)
    by_uid = {str(row["series_uid"]): row for row in rows}
    registry_long_enough = sorted(
        uid for uid in materialized
        if int(by_uid[uid].get("length", 0)) >= MIN_SERIES_LENGTH
    )
    needed = MIN_TRAIN_SERIES + MIN_EVAL_SERIES
    structure = {
        "criterion": (
            "at least %d train plus %d eval series, each at least %d long"
            % (MIN_TRAIN_SERIES, MIN_EVAL_SERIES, MIN_SERIES_LENGTH)
        ),
        "series_needed": needed,
        "registry_series": len(rows),
        "materialized_series": len(materialized),
        "series_at_or_over_min_length": len(registry_long_enough),
        "length_read_from": (
            "the frozen registry's length field, so no series value is read "
            "to decide this"
        ),
        "shortest_length": min(
            int(by_uid[uid].get("length", 0)) for uid in materialized
        ) if materialized else 0,
        "pass": len(registry_long_enough) >= needed,
    }
    structure["reason"] = (
        "" if structure["pass"] else
        "%d of %d materialized series reach the %d-point minimum, and %d are "
        "needed" % (
            len(registry_long_enough), len(materialized), MIN_SERIES_LENGTH,
            needed,
        )
    )

    # The frozen function's own verdict on the same inputs, as corroboration.
    frozen: dict[str, Any]
    try:
        frozen = dict(g3_sourcing.screen_candidate(
            DATASET_ID, dict(values), list(uids), PROJECT_ROOT,
        ))
    except Exception as exc:  # noqa: BLE001
        frozen = {
            "verdict": "SCREEN_RAISED",
            "error": "%s: %s" % (type(exc).__name__, exc),
        }
    frozen_verdict = str(frozen.get("verdict"))

    reached_substrate = "substrate" in frozen
    reached_phenomena = "public_phenomena" in frozen
    substrate = {
        "criterion": (
            "both the train and the eval substrate guard call the series "
            "clean, for at least %d of them" % needed
        ),
        "reached": reached_substrate,
        "result": frozen.get("substrate"),
        "not_reached_because": (
            None if reached_substrate else
            "the screen returned %s before the guards ran" % frozen_verdict
        ),
    }
    phenomena = {
        "criterion": (
            "at least %d training series carry a publicly visible phenomenon"
            % g3_sourcing.MIN_SERIES_WITH_PUBLIC_PHENOMENON
        ),
        "reached": reached_phenomena,
        "result": (
            None if not reached_phenomena else {
                key: value
                for key, value in frozen["public_phenomena"].items()
                if key != "per_series"
            }
        ),
        "not_reached_because": (
            None if reached_phenomena else
            "the screen returned %s before the census ran" % frozen_verdict
        ),
    }
    passed = bool(
        structure["pass"] and substrate.get("result", {}).get("pass")
        and phenomena.get("result", {}).get("pass")
    )
    return {
        "pass": passed,
        "structure": structure,
        "substrate_double_guard": substrate,
        "public_phenomenon": phenomena,
        "frozen_screen_verdict": frozen_verdict,
        "frozen_screen_result": {
            key: value for key, value in frozen.items()
            if key not in ("criteria", "public_phenomena")
        },
        "frozen_screen_reach": _frozen_screen_reach(),
        "zero_outcome_opened": True,
    }


# ------------------------------------------------------------------ step 2
def _readability_from_rows(
    rows: Sequence[Mapping[str, Any]],
    eval_uids: Sequence[str],
    origins: Sequence[int],
) -> dict[str, Any]:
    """Spread and share, by the same formulas development_judge_readability uses."""
    losses = [
        float(np.mean([row["per_view_smase"][index] for row in rows]))
        for index in range(len(eval_uids))
    ]
    total = float(sum(losses))
    spread = (max(losses) / min(losses)) if min(losses) > 0 else float("inf")
    share = (max(losses) / total) if total > 0 else 1.0
    last = [
        float(rows[-1]["per_view_smase"][index])
        for index in range(len(eval_uids))
    ]
    finite = bool(np.all(np.isfinite(last)))
    positive = bool(all(value > 0.0 for value in last))
    degenerate = bool(len({round(value, 12) for value in last}) <= 1)
    return {
        "development_origins": list(origins),
        "per_series_identity_smase": {
            str(uid): value for uid, value in zip(eval_uids, losses)
        },
        "min": min(losses),
        "max": max(losses),
        "eval_loss_spread": spread,
        "largest_single_series_loss_share": share,
        "pass": bool(
            spread <= MAX_EVAL_LOSS_SPREAD
            and share <= MAX_SINGLE_SERIES_LOSS_SHARE
        ),
        "delayed_block": {
            "origin": int(origins[-1]),
            "per_series_identity_smase": {
                str(uid): value for uid, value in zip(eval_uids, last)
            },
            "finite": finite,
            "strictly_positive": positive,
            "degenerate_single_value": degenerate,
            "usable": bool(finite and positive and not degenerate),
        },
        "reason": (
            "" if spread <= MAX_EVAL_LOSS_SPREAD
            else "eval loss spread %.1fx exceeds %.1fx; the aggregate would "
                 "be carried by series the forecaster cannot read"
                 % (spread, MAX_EVAL_LOSS_SPREAD)
        ),
    }


def step_2(
    values: Mapping[str, np.ndarray],
    roster_uids: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Identity baseline at the development origins, per Consumer variant."""
    assert max(DEVELOPMENT_ORIGINS) + HORIZON <= SEALED_FROM_INDEX, (
        "the readability probe would cross the sealed boundary"
    )
    from evaluation.functional.task_episode_harness.runner import (
        _mapped_roster,
    )

    train = [str(uid) for uid in roster_uids["train"]]
    ev = [str(uid) for uid in roster_uids["eval"]]
    roster = (
        [{"series_uid": uid, "role": "train"} for uid in train]
        + [{"series_uid": uid, "role": "eval"} for uid in ev]
    )
    mapped = _mapped_roster(roster)
    config = dict(_config())
    variants = [CONSUMER_POOLED]
    per_channel_note = None
    if len(train) >= MIN_CHANNELS_FOR_PER_CHANNEL:
        variants.append(CONSUMER_PER_CHANNEL)
    else:
        per_channel_note = (
            "only %d training channels; the frozen rule needs at least %d"
            % (len(train), MIN_CHANNELS_FOR_PER_CHANNEL)
        )
    measured: dict[str, Any] = {}
    retrains = 0
    for variant in variants:
        if retrains + len(DEVELOPMENT_ORIGINS) > RETRAIN_BUDGET:
            measured[variant] = {
                "measured": False,
                "reason": (
                    "the retrain budget of %d would be exceeded" % RETRAIN_BUDGET
                ),
            }
            continue
        rows = bch._evaluate_variant(
            mapped, dict(values), None, config, tuple(DEVELOPMENT_ORIGINS),
            None, variant,
        )
        retrains += len(DEVELOPMENT_ORIGINS)
        reading = _readability_from_rows(rows, ev, DEVELOPMENT_ORIGINS)
        reading["measured"] = True
        reading["consumer_retrains"] = len(DEVELOPMENT_ORIGINS)
        measured[variant] = reading
        print(
            "NOAA %-12s spread %.2fx share %.3f -> %s"
            % (
                variant, reading["eval_loss_spread"],
                reading["largest_single_series_loss_share"],
                "PASS" if reading["pass"] else "FAIL",
            ),
            flush=True,
        )
    return {
        "ran": True,
        "roster": {"train": train, "eval": ev},
        "variants_measured": list(measured),
        "per_channel_note": per_channel_note,
        "readings": measured,
        "consumer_retrains": retrains,
        "retrain_budget": RETRAIN_BUDGET,
        "bars": {
            "max_eval_loss_spread": MAX_EVAL_LOSS_SPREAD,
            "max_single_series_loss_share": MAX_SINGLE_SERIES_LOSS_SHARE,
            "reference_weather_spread": 24.4,
            "reference_t233_spread": 2.0,
        },
    }


# ----------------------------------------------------------------- verdict
def _verdict(zero: Mapping[str, Any], one: Mapping[str, Any] | None,
             two: Mapping[str, Any] | None) -> dict[str, Any]:
    if str(zero.get("status")) == "NOAA_DATA_MISSING":
        return {
            "verdict": "NOAA_DATA_MISSING",
            "reason": str(zero.get("reason")),
        }
    if one is None or not one["pass"]:
        failed: list[str] = []
        not_reached: list[str] = []
        if one is not None:
            if not one["structure"]["pass"]:
                failed.append("structure")
            for name, key in (
                ("substrate double guard", "substrate_double_guard"),
                ("public phenomenon", "public_phenomenon"),
            ):
                row = one[key]
                if not row["reached"]:
                    not_reached.append(name)
                elif not (row.get("result") or {}).get("pass"):
                    failed.append(name)
        return {
            "verdict": "NOAA_STRUCTURE_FAIL",
            "failed_criteria": failed,
            "criteria_not_reached": not_reached,
            "reason": (
                "%s failed (%s); %s the frozen screen itself returned %s, so "
                "the readability probe was not run and no Outcome was opened"
                % (
                    ", ".join(failed) or "an outcome-blind criterion",
                    (one or {}).get("structure", {}).get("reason") or "",
                    (
                        "%s were never reached, so they are neither passed "
                        "nor failed; " % ", ".join(not_reached)
                    ) if not_reached else "",
                    (one or {}).get("frozen_screen_verdict"),
                )
            ),
        }
    pooled = (two or {}).get("readings", {}).get(CONSUMER_POOLED) or {}
    per_channel = (two or {}).get("readings", {}).get(CONSUMER_PER_CHANNEL)
    if not pooled.get("pass"):
        return {
            "verdict": "NOAA_JUDGE_UNREADABLE",
            "reason": (
                "pooled read a %.1fx eval-loss spread and a %.3f largest "
                "single-series share against bars of %.1fx and %.2f; %s"
                % (
                    pooled.get("eval_loss_spread", float("inf")),
                    pooled.get("largest_single_series_loss_share", 1.0),
                    MAX_EVAL_LOSS_SPREAD, MAX_SINGLE_SERIES_LOSS_SHARE,
                    pooled.get("reason") or "",
                )
            ),
        }
    if per_channel is not None and not per_channel.get("pass"):
        return {
            "verdict": "PROCEED_POOLED_ONLY",
            "reason": (
                "pooled clears the bars at %.2fx spread; per_channel does not "
                "(%.2fx spread, %.3f share)"
                % (
                    pooled["eval_loss_spread"],
                    per_channel.get("eval_loss_spread", float("inf")),
                    per_channel.get("largest_single_series_loss_share", 1.0),
                )
            ),
        }
    return {
        "verdict": "NOAA_HEALTH_PROCEED",
        "reason": (
            "every measured Consumer variant clears the frozen bars (%s)"
            % ", ".join(
                "%s %.2fx" % (name, row["eval_loss_spread"])
                for name, row in (two or {}).get("readings", {}).items()
                if row.get("measured")
            )
        ),
    }


def _exposure_ledger(
    zero: Mapping[str, Any], one: Mapping[str, Any] | None,
    two: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """What this check actually spent, partition by partition."""
    values_loaded = str(zero.get("status")) == "READABLE"
    outcome_opened = bool(two and two.get("ran") and two.get("consumer_retrains"))
    prior = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in E2.glob("*noaa*.json")
    ) + sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in E2.glob("*_noaa_*.json")
        if "noaa" not in path.name.split("_")[0]
    )
    prior = sorted(set(prior))
    return {
        "dataset_id": DATASET_ID,
        "context_partition": {
            "state_after_this_check": (
                "INSTANCE_SEEN" if values_loaded else "UNTOUCHED"
            ),
            "what_was_seen": (
                "series values on the prefix below index %d, plus registry "
                "metadata" % SEALED_FROM_INDEX
            ) if values_loaded else "nothing",
        },
        "development_outcome_partition": {
            "state_after_this_check": "EXPOSED" if outcome_opened else "UNTOUCHED",
            "consumer_retrains": int((two or {}).get("consumer_retrains") or 0),
            "why": (
                "an identity baseline was fitted and evaluated at the "
                "development origins"
                if outcome_opened else
                "the outcome-blind screen failed, so no Consumer was fitted "
                "and no Judge reading was taken by this check"
            ),
        },
        "sealed_partition": {
            "state_after_this_check": "SEALED",
            "boundary": SEALED_FROM_INDEX,
            "indices_read_at_or_past_boundary": 0,
            "enforcement": zero["sealed_semantics"]["enforcement"],
            "note": (
                "no NOAA series in the frozen materialization even reaches "
                "index %d, so the boundary was never approached"
                % SEALED_FROM_INDEX
            ) if values_loaded else "",
        },
        "prior_exposure_on_the_record": {
            "claim_under_test": (
                "this dataset was described as the last sealed reserve, never "
                "touched"
            ),
            "artifacts_already_on_disk": prior,
            "artifact_count": len(prior),
            "reading": (
                "these are this project's own NOAA reports; a reserve with "
                "outcome reports already written is not untouched.  Reported "
                "as found, not resolved here"
            ) if prior else "no prior NOAA artifact was found",
        },
        "registry_says": {
            "exposure_class": zero.get("exposure_class"),
            "overlap_status": zero.get("overlap_status"),
            "overlap_family": zero.get("overlap_family"),
        },
        "g3_exposed_families_says": {
            "weather": list(g3_sourcing.EXPOSED_FAMILIES.get("weather", ())),
            "note": (
                "g3_sourcing.EXPOSED_FAMILIES lists %s under the exposed "
                "weather family, while the frozen registry marks every one of "
                "its series certified_virgin.  Both readings are recorded; "
                "this check does not adjudicate between them" % DATASET_ID
            ),
        },
    }


# --------------------------------------------------------------------- run
def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    zero = step_0()
    values = dict(zero.pop("_values", {}) or {})
    uids = list(zero.pop("_uids", []) or [])
    rows = _registry_rows()
    one: dict[str, Any] | None = None
    two: dict[str, Any] | None = None
    if str(zero.get("status")) == "READABLE":
        one = step_1(values, uids, rows)
        if one["pass"]:
            roster = (one["frozen_screen_result"].get("roster") or {})
            two = step_2(values, roster)
        else:
            two = {
                "ran": False,
                "reason": (
                    "the outcome-blind screen did not pass, so the "
                    "readability probe was not run and no Outcome was opened"
                ),
                "consumer_retrains": 0,
                "retrain_budget": RETRAIN_BUDGET,
            }
    verdict = _verdict(zero, one, two)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "instrument check on the sealed confirmation set before any of it "
            "is spent: can the Judge be read on this cohort at all"
        ),
        "not_authorization_evidence": (
            "no Skill, no Episode, no Harness change and no claim about NOAA "
            "utility. Development region only, and the sealed region is not "
            "read"
        ),
        "overall_verdict": verdict["verdict"],
        "overall_verdict_reason": verdict["reason"],
        "pre_registered": PRE_REGISTERED,
        "step_0_in_place": zero,
        "step_1_outcome_blind_screen": one,
        "step_2_readability_probe": two,
        "exposure_ledger": _exposure_ledger(zero, one, two),
        "llm_calls": 0,
        "consumer_retrains": int((two or {}).get("consumer_retrains") or 0),
        "retrain_budget": RETRAIN_BUDGET,
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(
            {
                "verdict": payload["overall_verdict"],
                "reason": payload["overall_verdict_reason"],
                "retrains": payload["consumer_retrains"],
            },
            indent=2, ensure_ascii=False, default=str,
        ))
        return 0
    E2.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8", newline="\n",
    )
    OUT_MD.write_text(_markdown(payload), encoding="utf-8", newline="\n")
    print("wrote", OUT_JSON, flush=True)
    print("wrote", OUT_MD, flush=True)
    print("verdict", payload["overall_verdict"], flush=True)
    print(
        "cost: %d Consumer retrains of a budget of %d, 0 LLM"
        % (payload["consumer_retrains"], RETRAIN_BUDGET),
        flush=True,
    )
    return 0


# ------------------------------------------------------------------ report
def _markdown(payload: Mapping[str, Any]) -> str:
    zero = payload["step_0_in_place"]
    one = payload["step_1_outcome_blind_screen"]
    two = payload["step_2_readability_probe"]
    ledger = payload["exposure_ledger"]
    lines = [
        "# NOAA health check: is the reserve worth burning?",
        "",
        "**Overall: `%s`** -- %s."
        % (payload["overall_verdict"], payload["overall_verdict_reason"]),
        "",
        "`%s` is the dataset `e1.SEALED_CONFIRMATION_DATASET` names as the "
        "last sealed confirmation set.  Before any of it is spent, one "
        "question has to be answered that cannot be answered afterwards: is "
        "the instrument readable on this cohort?  Weather was rejected at a "
        "24.4x eval-loss spread, and that failure is indistinguishable from a "
        "null result once the Outcome is open."
        % zero["dataset_id"],
        "",
        "**Development region only.**  Every series is truncated to "
        "`[:%d]` at load time and the truncation is asserted, so no index at "
        "or past the boundary can be read by construction.  0 LLM calls."
        % zero["sealed_semantics"]["sealed_from_index"],
        "",
        "## Step 0 -- in place",
        "",
    ]
    if str(zero.get("status")) != "READABLE":
        lines += [
            "`%s`: %s" % (zero.get("status"), zero.get("reason")), "",
        ]
    else:
        channels = zero["channel_structure"]
        lines += [
            "| field | value |",
            "| --- | --- |",
            "| registry | `%s` |" % zero["registry_path"],
            "| series in registry | %d |" % zero["registry_series_count"],
            "| materialized under `%s` | %d |"
            % (zero["clean_base_path"], zero["materialized_series_count"]),
            "| not materialized | %d |" % zero["unmaterialized_series_count"],
            "| channel structure | %s; %d distinct %s |"
            % (
                channels["shape"], channels["distinct_entities"],
                channels["entity_field"],
            ),
            "| array shape | `%s` |" % channels["array_shape_example"],
            "| registry lengths | %s |"
            % json.dumps(zero["registry_lengths"], sort_keys=True),
            "| shortest length | **%d** |" % zero["shortest_registry_length"],
            "| frequency | %s |" % ", ".join(zero["frequency"]),
            "| exposure_class | %s |" % ", ".join(zero["exposure_class"]),
            "| overlap family / status | %s / %s |"
            % (
                ", ".join(zero["overlap_family"]),
                ", ".join(zero["overlap_status"]),
            ),
            "| raw source dir | `%s` (%d station files, listed by filesystem "
            "metadata only) |"
            % (zero["raw_source_dir"], zero["raw_station_files"]),
            "",
            "Sealed semantics: `kind = %s`, boundary %d from `%s`.  The "
            "readability probe's farthest index would be %d, %s the boundary. "
            " %s"
            % (
                zero["sealed_semantics"]["kind"],
                zero["sealed_semantics"]["sealed_from_index"],
                zero["sealed_semantics"]["constant_source"],
                zero["sealed_semantics"]["farthest_index_the_probe_would_read"],
                "inside"
                if zero["sealed_semantics"]["probe_inside_boundary"]
                else "**past**",
                zero["sealed_semantics"]["enforcement"],
            ),
            "",
            "Load report: %s."
            % json.dumps(zero["load_report"], sort_keys=True),
            "",
        ]
        entry = zero.get("manifest_entry")
        if entry:
            lines += [
                "The frozen manifest records it as: %s"
                % json.dumps(entry, ensure_ascii=False, sort_keys=True),
                "",
            ]
    if one is not None:
        structure = one["structure"]
        lines += [
            "## Step 1 -- outcome-blind screen",
            "",
            "Criteria are read from `g3_sourcing.CRITERIA`, never restated "
            "here, and no bar is moved.",
            "",
            "| criterion | required | measured | verdict |",
            "| --- | --- | --- | --- |",
            "| structure | %s | %d of %d materialized series reach the "
            "minimum length; shortest is %d | **%s** |"
            % (
                structure["criterion"], structure["series_at_or_over_min_length"],
                structure["materialized_series"], structure["shortest_length"],
                "PASS" if structure["pass"] else "FAIL",
            ),
        ]
        for key, label in (
            ("substrate_double_guard", "substrate double guard"),
            ("public_phenomenon", "public phenomenon"),
        ):
            row = one[key]
            if row["reached"]:
                lines.append(
                    "| %s | %s | %s | **%s** |"
                    % (
                        label, row["criterion"],
                        json.dumps(row["result"], sort_keys=True),
                        "PASS" if (row["result"] or {}).get("pass") else "FAIL",
                    )
                )
            else:
                lines.append(
                    "| %s | %s | not reached -- %s | `NOT_REACHED` |"
                    % (label, row["criterion"], row["not_reached_because"])
                )
        reach = one["frozen_screen_reach"]
        lines += [
            "",
            "The frozen screen's own verdict on the same inputs: "
            "`%s`." % one["frozen_screen_verdict"],
            "",
            "How far the screen's guards reach on the index axis: the train "
            "guard stops at %d (%s the boundary); the eval guard validates "
            "the nine frozen roster windows and reaches %d (%s the boundary). "
            " %s"
            % (
                reach["train_guard_farthest_index"],
                "inside" if reach["train_guard_inside_boundary"] else "**past**",
                reach["eval_guard_farthest_index"],
                "inside" if reach["eval_guard_inside_boundary"] else "**past**",
                reach["note"],
            ),
            "",
        ]
    if two is not None and two.get("ran"):
        lines += [
            "## Step 2 -- readability probe",
            "",
            "Identity baseline only, at the frozen development origins %s.  "
            "Bars: spread <= %.1fx, largest single-series share <= %.2f.  "
            "Weather failed at 24.4x; T233 reads at 2.0x."
            % (
                list(DEVELOPMENT_ORIGINS), MAX_EVAL_LOSS_SPREAD,
                MAX_SINGLE_SERIES_LOSS_SHARE,
            ),
            "",
            "| Consumer variant | eval-loss spread | largest share | delayed "
            "block finite / positive / non-degenerate | retrains | verdict |",
            "| --- | ---: | ---: | --- | ---: | --- |",
        ]
        for name, row in two["readings"].items():
            if not row.get("measured"):
                lines.append(
                    "| `%s` | -- | -- | -- | 0 | `NOT_MEASURED` (%s) |"
                    % (name, row.get("reason"))
                )
                continue
            block = row["delayed_block"]
            lines.append(
                "| `%s` | %.2fx | %.3f | %s / %s / %s | %d | **%s** |"
                % (
                    name, row["eval_loss_spread"],
                    row["largest_single_series_loss_share"],
                    block["finite"], block["strictly_positive"],
                    not block["degenerate_single_value"],
                    row["consumer_retrains"],
                    "PASS" if row["pass"] else "FAIL",
                )
            )
        if two.get("per_channel_note"):
            lines += ["", "per_channel was not measured: %s."
                      % two["per_channel_note"]]
        lines.append("")
    elif two is not None:
        lines += [
            "## Step 2 -- readability probe",
            "",
            "**Not run.**  %s" % two.get("reason"),
            "",
        ]
    lines += [
        "## Exposure ledger",
        "",
        "| partition | state after this check | detail |",
        "| --- | --- | --- |",
        "| Context | `%s` | %s |"
        % (
            ledger["context_partition"]["state_after_this_check"],
            ledger["context_partition"]["what_was_seen"],
        ),
        "| development Outcome | `%s` | %d Consumer retrains -- %s |"
        % (
            ledger["development_outcome_partition"]["state_after_this_check"],
            ledger["development_outcome_partition"]["consumer_retrains"],
            ledger["development_outcome_partition"]["why"],
        ),
        "| index >= %d | `%s` | %d indices read at or past the boundary. %s |"
        % (
            ledger["sealed_partition"]["boundary"],
            ledger["sealed_partition"]["state_after_this_check"],
            ledger["sealed_partition"]["indices_read_at_or_past_boundary"],
            ledger["sealed_partition"]["note"],
        ),
        "",
        "### The reserve was not untouched",
        "",
        "%s  Artifacts already on disk (%d):"
        % (
            ledger["prior_exposure_on_the_record"]["reading"],
            ledger["prior_exposure_on_the_record"]["artifact_count"],
        ),
        "",
    ]
    for item in ledger["prior_exposure_on_the_record"]["artifacts_already_on_disk"]:
        lines.append("- `%s`" % item)
    lines += [
        "",
        "The two records also disagree with each other: the frozen registry "
        "marks every series `%s` with overlap status `%s`, while "
        "`g3_sourcing.EXPOSED_FAMILIES` lists `%s` under the exposed weather "
        "family (%s).  %s"
        % (
            ", ".join(zero.get("exposure_class") or ["--"]),
            ", ".join(zero.get("overlap_status") or ["--"]),
            zero["dataset_id"],
            ", ".join(ledger["g3_exposed_families_says"]["weather"]),
            ledger["g3_exposed_families_says"]["note"],
        ),
        "",
        "## Cost",
        "",
        "%d Consumer retrains of a budget of %d; 0 LLM calls."
        % (payload["consumer_retrains"], payload["retrain_budget"]),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in-place-only", action="store_true",
        help="run step 0 and stop (0 retrains)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run everything but print the verdict instead of writing",
    )
    args = parser.parse_args(argv)
    if args.in_place_only:
        zero = step_0()
        zero.pop("_values", None)
        zero.pop("_uids", None)
        print(json.dumps(zero, indent=2, ensure_ascii=False, default=str))
        return 0
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
