"""Close the runtime loop: a later-window delayed probe drives the transition.

The integration slice formed five ``LOCAL_DRAFT`` Episodes on the real
channel and then stopped, honestly: its delayed reading had set the adoption
bar, so it was in-selection and could not serve as promotion evidence.  This
slice supplies what was missing -- a delayed probe on a window that took part
in no selection -- and feeds it to the existing update path.

It is a mechanism demonstration and presumes no direction.  ``e1._update_delayed``
grades three bands: a probe at or above the material threshold promotes to
``LOCAL_ACTIVE``, a probe at or below its negative restricts to
``RESTRICTED``, and anything between leaves the Draft standing with delayed
evidence recorded.  A restriction closes the loop exactly as well as a
promotion.

0 LLM calls.  The probe reading is never used to choose anything: no plan is
re-proposed, no shortlist is re-run, no threshold is touched.  State changes
land only under ``_scratch/skill_store``; ``methods/ttha/harness/h0``, the
frozen artifacts and every committed file are read-only.

Writes ``artifacts/functional/e2/lifecycle_probe_v1.json`` and ``.md``.
"""
from __future__ import annotations

import argparse
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
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from evaluation.functional.task_episode_harness import e1 as e1mod  # noqa: E402
from evaluation.functional.task_episode_harness.agentic import (  # noqa: E402
    g3_sourcing,
)
from evaluation.functional.task_episode_harness.runner import (  # noqa: E402
    _arm_metrics,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_DELAYED,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    STATUS_RESTRICTED,
    episode_from_dict,
)

PROTOCOL_VERSION = "lifecycle_probe_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "lifecycle_probe_v1.json"
OUT_MD = E2 / "lifecycle_probe_v1.md"
SOURCE_JSON = E2 / "skill_store_integration_v1.json"
SCREENING = {
    "g3_candidate_screening_v2": E2 / "g3_candidate_screening_v2.json",
    "g3_candidate_screening_v3": E2 / "g3_candidate_screening_v3.json",
}
# State changes land here and nowhere else.
STATE_ROOT = PROJECT_ROOT / "_scratch" / "skill_store" / PROTOCOL_VERSION

HORIZON = int(bch.v6.HORIZON)
IDENTITY = bch.IDENTITY
MATERIAL_THRESHOLD = float(bch.MATERIAL_THRESHOLD)
DECLARED_SHIFT = 480
EVALUATION_CALL_BUDGET = 8

# Frozen before the run.
PROBE_WINDOW_RULE: dict[str, Any] = {
    "one_probe_window_per_draft": True,
    "first_choice": (
        "when the Draft's window is a task of the frozen roster and the next "
        "task exists in it, the probe origins are that next task's "
        "delayed_origins, quoted verbatim from "
        "task_episode_harness.e1._frozen_task_roster().  The delayed role is "
        "kept so the window shape is the one the Draft was graded on, and the "
        "next task's block sits entirely beyond everything the Draft's own "
        "window read"
    ),
    "fallback": (
        "otherwise a declared shift of +%d on the Draft's own delayed "
        "origins, same count and same spacing, labelled origin_source="
        "\"chosen\"" % DECLARED_SHIFT
    ),
    "legality": (
        "the probe's farthest read, max(origins) + horizon, must not pass the "
        "cohort's frozen sealed boundary.  Over the boundary the Draft is "
        "recorded PROBE_WINDOW_UNAVAILABLE: the window is not moved, not "
        "shortened and not replaced by a nearer one"
    ),
    "traffic_boundary": (
        "g3_sourcing.SEALED_FROM_INDEX cross-checked against the "
        "sealed_from_index recorded in the two screening artifacts; the "
        "tightest of them is used"
    ),
    "roster_boundary": (
        "the KDD roster cohorts have no index-sealed ceiling -- e1's sealed "
        "confirmation set is a different dataset (%s) and is not touched "
        "here.  What bounds them is the roster itself: the probe task index "
        "must be inside AVAILABLE_TASK_COUNT, and the farthest read must be "
        "inside the series" % e1mod.SEALED_CONFIRMATION_DATASET
    ),
    "horizon": HORIZON,
    "declared_shift": DECLARED_SHIFT,
}

PRE_REGISTERED: dict[str, Any] = {
    "fixed_before_the_run": True,
    "zero_llm": True,
    "what_this_slice_does": (
        "supplies each surviving LOCAL_DRAFT with one delayed probe on a "
        "window that took part in no selection, and lets the existing update "
        "path decide the transition"
    ),
    "direction_is_not_presumed": (
        "a probe that restricts the Draft closes the loop exactly as well as "
        "one that promotes it; the run is a mechanism demonstration"
    ),
    "probe_window": PROBE_WINDOW_RULE,
    "probe_is_never_used_to_choose": (
        "the reading grades an already-adopted plan.  No plan is re-proposed, "
        "no shortlist re-run, no mask re-searched and no threshold touched"
    ),
    "update_path": (
        "the probe is built by task_episode_harness.runner._arm_metrics, the "
        "same cluster-unit metric the real path uses, and handed to "
        "task_episode_harness.e1._update_delayed unchanged.  The status is "
        "whatever that function returns; nothing is set by hand"
    ),
    "state_is_scoped": (
        "updated Episodes are written under _scratch/skill_store/%s and "
        "nowhere else" % PROTOCOL_VERSION
    ),
    "budget": {
        "evaluation_calls": EVALUATION_CALL_BUDGET,
        "probe_evaluations_per_draft": 1,
        "note": (
            "one evaluation call is one call into the recipe evaluator for "
            "one plan over one window.  A probe needs two -- the identity "
            "baseline the gain is measured against, and the adopted plan -- "
            "and the baseline is shared by Drafts on the same cohort, "
            "Consumer structure and window.  Consumer retrains are counted "
            "separately at one per origin per call, the convention the "
            "bridge and integration slices used"
        ),
    },
    "verdicts": [
        "LIFECYCLE_CLOSES: at least one Draft changed state through the real "
        "path, in either direction",
        "PROBE_WINDOWS_UNAVAILABLE: no Draft has a legal probe window",
        "UPDATE_PATH_BLOCKED: a legal probe exists but the real path will not "
        "accept it; the refusing interface is named",
        "NO_TRANSITION_NEUTRAL_PROBE: added before the run because the three "
        "above are not exhaustive -- the path can accept a probe, record "
        "delayed evidence, and still leave the Draft standing when the "
        "reading lands inside the neutral band",
    ],
    "verdicts_are_reported_side_by_side": True,
}


# --------------------------------------------------------------- the windows
def _roster_index(window_id: str) -> int | None:
    """The frozen roster index a window id names, or None."""
    prefix = "e1v2_task_"
    if not window_id.startswith(prefix):
        return None
    tail = window_id[len(prefix):]
    if not tail.isdigit():
        return None
    return int(tail) - 1


def _traffic_boundary() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for name, path in SCREENING.items():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
        found = _find_key(payload, "sealed_from_index")
        if found is not None:
            artifacts[name] = int(found)
    boundaries = [int(g3_sourcing.SEALED_FROM_INDEX)] + list(artifacts.values())
    boundaries.append(int(bch._TRAFFIC_SEALED_FROM_INDEX))
    return {
        "code_boundary": int(g3_sourcing.SEALED_FROM_INDEX),
        "code_boundary_source": (
            "evaluation/functional/task_episode_harness/agentic/"
            "g3_sourcing.py::SEALED_FROM_INDEX"
        ),
        "recipe_module_boundary": int(bch._TRAFFIC_SEALED_FROM_INDEX),
        "artifact_boundaries": artifacts,
        "tightest_boundary": min(boundaries),
    }


def _find_key(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            found = _find_key(value, key)
            if found is not None:
                return found
    return None


def _probe_window(draft: Mapping[str, Any]) -> dict[str, Any]:
    """One probe window for one Draft, or the reason there is none."""
    cohort = str(draft["cohort"])
    window_id = str(draft["window_id"])
    own_delayed = [int(origin) for origin in draft["delayed_origins"]]
    index = _roster_index(window_id)
    if index is not None and index + 1 < int(e1mod.AVAILABLE_TASK_COUNT):
        spec = e1mod._frozen_task_roster()[index + 1]
        origins = [int(origin) for origin in spec["delayed_origins"]]
        window = {
            "probe_window_id": str(spec["task_episode_id"]),
            "probe_origins": origins,
            "origin_source": "quoted from the frozen roster",
            "origin_provenance": (
                "task_episode_harness.e1._frozen_task_roster()[%d], %s, "
                "delayed_origins verbatim; the Draft was graded on %s"
                % (index + 1, spec["task_episode_id"], window_id)
            ),
        }
    else:
        origins = [origin + DECLARED_SHIFT for origin in own_delayed]
        window = {
            "probe_window_id": "%s_probe_shift_%d" % (window_id, DECLARED_SHIFT),
            "probe_origins": origins,
            "origin_source": "chosen",
            "origin_provenance": (
                "no next task exists in the frozen roster for this window, so "
                "the rule's fallback applies: the Draft's own delayed origins "
                "%s shifted by +%d, same count and spacing"
                % (own_delayed, DECLARED_SHIFT)
            ),
        }
    farthest = max(window["probe_origins"]) + HORIZON
    window["horizon"] = HORIZON
    window["farthest_index_read"] = farthest
    window["own_delayed_origins"] = own_delayed
    window["overlaps_own_window"] = bool(
        set(window["probe_origins"]) & set(own_delayed)
    )

    if cohort == "traffic":
        boundary = _traffic_boundary()
        inside = farthest <= int(boundary["tightest_boundary"])
        window["sealed_check"] = {
            **boundary,
            "farthest_index_read": farthest,
            "inside": inside,
            "kind": "index_sealed_boundary",
        }
        window["available"] = inside
        if not inside:
            window["unavailable_reason"] = (
                "the probe would read to index %d, past the tightest frozen "
                "sealed boundary %d for this cohort; the window is left where "
                "the rule put it and not shortened"
                % (farthest, int(boundary["tightest_boundary"]))
            )
        return window

    loaded = _cohort(cohort)
    lengths = {
        len(loaded["values"][uid]) for uid in loaded["train_uids"]
    } | {
        len(loaded["values"][uid]) for uid in loaded["eval_uids"]
    }
    shortest = min(lengths)
    in_roster = index is not None and index + 1 < int(
        e1mod.AVAILABLE_TASK_COUNT
    )
    inside = bool(in_roster and farthest <= shortest)
    window["sealed_check"] = {
        "kind": "roster_bound",
        "why_no_index_boundary": (
            "e1's sealed confirmation set is a different dataset (%s), which "
            "this run does not touch; the ceiling on this cohort is the "
            "frozen roster itself"
            % e1mod.SEALED_CONFIRMATION_DATASET
        ),
        "probe_task_index": None if index is None else index + 1,
        "available_task_count": int(e1mod.AVAILABLE_TASK_COUNT),
        "probe_task_inside_roster": in_roster,
        "shortest_series_length": int(shortest),
        "farthest_index_read": farthest,
        "inside": inside,
        "sealed_confirmation_dataset": e1mod.SEALED_CONFIRMATION_DATASET,
        "sealed_confirmation_dataset_touched": False,
    }
    window["available"] = inside
    if not inside:
        window["unavailable_reason"] = (
            "the probe task is outside the frozen roster or would read past "
            "the end of the series (farthest %d, shortest series %d)"
            % (farthest, shortest)
        )
    return window


# ------------------------------------------------------------- the evaluator
_COHORTS: dict[str, Any] = {}


def _cohort(name: str) -> dict[str, Any]:
    loaded = _COHORTS.get(name)
    if loaded is None:
        loaded = bch.load_cohort(PROJECT_ROOT, name)
        _COHORTS[name] = loaded
    return loaded


class ProbeEvaluator:
    """One delayed probe, on the recipe evaluator, counted honestly.

    Nothing here is new machinery: ``_evaluate_variant`` and
    ``_evaluate_assignment`` are the recipe module's, ``_arm_metrics`` is the
    real path's cluster-unit metric, and the identity baseline is the same one
    every gain in this line is measured against.
    """

    def __init__(self) -> None:
        self.config = dict(wvc._config())
        self.evaluation_calls = 0
        self.consumer_retrains = 0
        self.log: list[dict[str, Any]] = []
        self._identity: dict[tuple[str, str, tuple[int, ...]], Any] = {}
        self._programs: dict[str, Any] = {}

    def _program(self, program: str) -> Any:
        compiled = self._programs.get(program)
        if compiled is None:
            compiled = wvc._compiled(program, name="probe_%s" % program)
            self._programs[program] = compiled
        return compiled

    def _rows(
        self, *, cohort: str, variant: str, program: str | None,
        excluded: Sequence[str], origins: Sequence[int], role: str,
    ) -> list[Any]:
        if self.evaluation_calls >= EVALUATION_CALL_BUDGET:
            raise SystemExit(
                "the evaluation-call budget of %d is spent; refusing to "
                "measure more" % EVALUATION_CALL_BUDGET
            )
        loaded = _cohort(cohort)
        roster = loaded["mapped_roster"]
        values = loaded["values"]
        origins = tuple(int(origin) for origin in origins)
        self.evaluation_calls += 1
        self.consumer_retrains += len(origins)
        self.log.append({
            "role": role,
            "cohort": cohort,
            "consumer_variant": variant,
            "program": program,
            "excluded_series": list(excluded),
            "origins": list(origins),
            "consumer_retrains": len(origins),
        })
        if program is None:
            return bch._evaluate_variant(
                roster, values, None, self.config, origins, None, variant,
            )
        compiled = self._program(program)
        excluded_set = {str(uid) for uid in excluded}
        if not excluded_set:
            return bch._evaluate_variant(
                roster, values, compiled, self.config, origins, None, variant,
            )
        train_uids = [str(uid) for uid in loaded["train_uids"]]
        assignment = {
            uid: (None if uid in excluded_set else compiled)
            for uid in train_uids
        }
        return [
            bch._evaluate_assignment(
                roster, values, assignment, self.config, origin=origin,
                consumer_variant=variant,
            )
            for origin in origins
        ]

    def identity_rows(
        self, *, cohort: str, variant: str, origins: Sequence[int],
    ) -> tuple[list[Any], bool]:
        key = (cohort, variant, tuple(int(o) for o in origins))
        cached = self._identity.get(key)
        if cached is not None:
            return cached, True
        rows = self._rows(
            cohort=cohort, variant=variant, program=None, excluded=(),
            origins=origins, role="identity_baseline",
        )
        self._identity[key] = rows
        return rows, False

    def probe(self, draft: Mapping[str, Any], window: Mapping[str, Any]):
        cohort = str(draft["cohort"])
        variant = str(draft["consumer_variant"])
        plan = draft["final_plan"]
        origins = [int(origin) for origin in window["probe_origins"]]
        identity_rows, reused = self.identity_rows(
            cohort=cohort, variant=variant, origins=origins,
        )
        program = str(plan["program"])
        if program == IDENTITY:
            candidate_rows = identity_rows
        else:
            candidate_rows = self._rows(
                cohort=cohort, variant=variant, program=program,
                excluded=[str(uid) for uid in plan["excluded_series"]],
                origins=origins, role="adopted_plan",
            )
        eval_uids = [str(uid) for uid in _cohort(cohort)["eval_uids"]]
        metrics = _arm_metrics(
            identity_rows, candidate_rows, tuple(origins), eval_uids,
        )
        return metrics, {
            "identity_baseline_reused": reused,
            "evaluation_calls_so_far": self.evaluation_calls,
            "consumer_retrains_so_far": self.consumer_retrains,
            "metric_function": (
                "evaluation/functional/task_episode_harness/runner.py::"
                "_arm_metrics"
            ),
            "evaluator": (
                "evaluation/functional/run_batch_composition_headroom.py::"
                "_evaluate_variant / _evaluate_assignment"
            ),
        }


# ------------------------------------------------------------- the transition
UPDATE_PATH = {
    "callable": (
        "evaluation/functional/task_episode_harness/e1.py::_update_delayed"
    ),
    "signature": "_update_delayed(episode, delayed_probe, delayed_origins)",
    "reads": (
        "episode.support_response['gain'] and delayed_probe['macro_gain'], "
        "'se_block', 'gain_over_se'"
    ),
    "bands": (
        "Support below the material threshold grades EPISODE_ONLY/NEGATIVE; "
        "otherwise delayed at or above +threshold grades "
        "LOCAL_ACTIVE/POSITIVE, at or below -threshold grades "
        "RESTRICTED/CONFLICT, and anything between grades "
        "LOCAL_DRAFT/ABSTAIN with evidence_level raised to DELAYED"
    ),
    "material_threshold": MATERIAL_THRESHOLD,
    "reachable_targets": {
        "promote": STATUS_LOCAL_ACTIVE,
        "restrict": STATUS_RESTRICTED,
        "stand": STATUS_LOCAL_DRAFT,
        "evidence_level_after": EVIDENCE_DELAYED,
    },
    "nothing_is_set_by_hand": True,
}


def _episode_view(episode: Any) -> dict[str, Any]:
    delayed = dict(episode.delayed_response)
    return {
        "local_status": str(episode.local_status),
        "relation": str(episode.relation),
        "evidence_level": str(episode.evidence_level),
        "response_validity": str(episode.response_validity),
        "support_gain": (
            None if episode.support_response.get("gain") is None
            else float(episode.support_response["gain"])
        ),
        "delayed_evaluated": bool(delayed.get("evaluated")),
        "delayed_gain": (
            None if delayed.get("gain") is None else float(delayed["gain"])
        ),
        "delayed_se_block": delayed.get("se_block"),
        "delayed_gain_over_se": delayed.get("gain_over_se"),
        "delayed_block_origins": list(delayed.get("block_origins") or ()),
        "delayed_took_part_in_selection": delayed.get(
            "took_part_in_selection"
        ),
    }


def _transition(
    episode_payload: Mapping[str, Any], probe: Mapping[str, Any],
    origins: Sequence[int],
) -> dict[str, Any]:
    """Hand the probe to the real path and record what it did."""
    try:
        episode = episode_from_dict(dict(episode_payload))
    except Exception as exc:  # noqa: BLE001
        return {
            "accepted": False,
            "blocked_at_interface": (
                "SelfEvolvingHarnessTS.methods.ttha.experience_memory."
                "episode_from_dict"
            ),
            "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
        }
    before = _episode_view(episode)
    try:
        updated = e1mod._update_delayed(
            episode, dict(probe), tuple(int(o) for o in origins),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "accepted": False,
            "before": before,
            "blocked_at_interface": UPDATE_PATH["callable"],
            "blocked_reason": "%s: %s" % (type(exc).__name__, exc),
        }
    after = _episode_view(updated)
    changed = before["local_status"] != after["local_status"]
    if changed:
        why = (
            "the probe read %+.6f against a material threshold of %.3f, so "
            "the path moved the Episode from %s to %s"
            % (
                after["delayed_gain"], MATERIAL_THRESHOLD,
                before["local_status"], after["local_status"],
            )
        )
    else:
        why = (
            "the probe read %+.6f, inside the neutral band of +/-%.3f, so the "
            "path recorded delayed evidence (evidence_level %s -> %s) and "
            "left the Episode at %s"
            % (
                after["delayed_gain"], MATERIAL_THRESHOLD,
                before["evidence_level"], after["evidence_level"],
                after["local_status"],
            )
        )
    return {
        "accepted": True,
        "before": before,
        "after": after,
        "status_changed": changed,
        "evidence_level_changed": (
            before["evidence_level"] != after["evidence_level"]
        ),
        "relation_changed": before["relation"] != after["relation"],
        "why": why,
        "updated_episode": updated.to_dict(),
    }


# --------------------------------------------------------------------- run
def _drafts() -> list[dict[str, Any]]:
    """The surviving Drafts of the integration slice, read-only."""
    if not SOURCE_JSON.is_file():
        raise SystemExit(
            "this slice needs %s, which is not there"
            % SOURCE_JSON.relative_to(PROJECT_ROOT).as_posix()
        )
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    if str(payload.get("protocol_version")) != "skill_store_integration_v1":
        raise SystemExit(
            "unexpected protocol_version %r in %s"
            % (payload.get("protocol_version"), SOURCE_JSON.name)
        )
    by_id = {str(row["episode_id"]): row for row in payload["arm_targets"]}
    drafts: list[dict[str, Any]] = []
    for episode_id, record in payload["lifecycle_records"].items():
        if str(record.get("status")) != STATUS_LOCAL_DRAFT:
            continue
        arm_target = by_id[str(episode_id)]
        drafts.append({
            "draft_id": str(episode_id),
            "episode_id": str(record["episode"]["episode_id"]),
            "target_id": str(arm_target["target_id"]),
            "arm": str(arm_target["arm"]),
            "cohort": str(arm_target["cohort"]),
            "consumer_variant": str(arm_target["consumer_variant"]),
            "window_id": str(arm_target["window_id"]),
            "support_origins": list(arm_target["support_origins"]),
            "delayed_origins": list(arm_target["delayed_origins"]),
            "final_plan": dict(arm_target["final_plan"]),
            "support_aggregate_gain": float(
                arm_target["support"]["aggregate_gain"]
            ),
            "in_selection_delayed_aggregate_gain": float(
                arm_target["delayed"]["aggregate_gain"]
            ),
            "episode": dict(record["episode"]),
        })
    drafts.sort(key=lambda row: row["draft_id"])
    return drafts


def _verdict(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unavailable = [
        str(row["draft_id"]) for row in rows
        if row["outcome"] == "PROBE_WINDOW_UNAVAILABLE"
    ]
    blocked = [
        str(row["draft_id"]) for row in rows
        if row["outcome"] == "UPDATE_PATH_BLOCKED"
    ]
    transitioned = [
        str(row["draft_id"]) for row in rows if row["outcome"] == "TRANSITIONED"
    ]
    neutral = [
        str(row["draft_id"]) for row in rows
        if row["outcome"] == "NO_TRANSITION_NEUTRAL_PROBE"
    ]
    labels: list[str] = []
    if transitioned:
        labels.append("LIFECYCLE_CLOSES")
    if unavailable and len(unavailable) == len(rows):
        labels.append("PROBE_WINDOWS_UNAVAILABLE")
    if blocked:
        labels.append("UPDATE_PATH_BLOCKED")
    if neutral:
        labels.append("NO_TRANSITION_NEUTRAL_PROBE")
    if not labels:
        labels.append("PROBE_WINDOWS_UNAVAILABLE")
    return {
        "verdict": " + ".join(labels),
        "labels": labels,
        "transitioned": transitioned,
        "neutral": neutral,
        "probe_window_unavailable": unavailable,
        "update_path_blocked": blocked,
        "reason": "; ".join(
            part for part in (
                (
                    "%d of %d Drafts changed state through the real path: %s"
                    % (
                        len(transitioned), len(rows),
                        ", ".join(
                            "%s -> %s" % (
                                row["draft_id"],
                                row["transition"]["after"]["local_status"],
                            )
                            for row in rows if row["outcome"] == "TRANSITIONED"
                        ),
                    )
                ) if transitioned else "",
                (
                    "%s had no legal probe window" % ", ".join(unavailable)
                ) if unavailable else "",
                (
                    "%s produced a neutral probe and stayed a Draft"
                    % ", ".join(neutral)
                ) if neutral else "",
                (
                    "the update path refused %s" % ", ".join(blocked)
                ) if blocked else "",
            ) if part
        ),
    }


def run(*, dry_run: bool = False) -> int:
    started = time.perf_counter()
    drafts = _drafts()
    evaluator = ProbeEvaluator()
    rows: list[dict[str, Any]] = []
    for draft in drafts:
        window = _probe_window(draft)
        row: dict[str, Any] = {
            "draft_id": draft["draft_id"],
            "episode_id": draft["episode_id"],
            "target_id": draft["target_id"],
            "arm": draft["arm"],
            "cohort": draft["cohort"],
            "consumer_variant": draft["consumer_variant"],
            "graded_window_id": draft["window_id"],
            "adopted_plan": draft["final_plan"],
            "support_aggregate_gain": draft["support_aggregate_gain"],
            "in_selection_delayed_aggregate_gain": draft[
                "in_selection_delayed_aggregate_gain"
            ],
            "probe_window": window,
        }
        if not window.get("available"):
            row["outcome"] = "PROBE_WINDOW_UNAVAILABLE"
            row["probe"] = None
            row["transition"] = None
            rows.append(row)
            print(
                "LP %-6s PROBE_WINDOW_UNAVAILABLE %s"
                % (draft["draft_id"], window.get("unavailable_reason")),
                flush=True,
            )
            continue
        metrics, accounting = evaluator.probe(draft, window)
        row["probe"] = {
            "origins": list(window["probe_origins"]),
            "macro_gain": float(metrics["macro_gain"]),
            "se_block": float(metrics["se_block"]),
            "gain_over_se": metrics["gain_over_se"],
            "per_origin_gain": dict(metrics["per_origin_gain"]),
            "per_series_mean_gain": dict(metrics["per_series_mean_gain"]),
            "positive_series_count": int(metrics["positive_series_count"]),
            "negative_series_count": int(metrics["negative_series_count"]),
            "took_part_in_any_selection": False,
            **accounting,
        }
        transition = _transition(
            draft["episode"], metrics, window["probe_origins"],
        )
        row["transition"] = transition
        if not transition.get("accepted"):
            row["outcome"] = "UPDATE_PATH_BLOCKED"
        elif transition["status_changed"]:
            row["outcome"] = "TRANSITIONED"
        else:
            row["outcome"] = "NO_TRANSITION_NEUTRAL_PROBE"
        rows.append(row)
        print(
            "LP %-6s probe %s %+.6f -> %s (%s)"
            % (
                draft["draft_id"], window["probe_window_id"],
                float(metrics["macro_gain"]), row["outcome"],
                (transition.get("after") or {}).get("local_status")
                or transition.get("blocked_at_interface"),
            ),
            flush=True,
        )

    verdict = _verdict(rows)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "close the runtime loop: one out-of-selection delayed probe per "
            "surviving LOCAL_DRAFT, driven through the existing update path"
        ),
        "not_authorization_evidence": (
            "a mechanism demonstration with no presumed direction. No Skill "
            "is promoted into any Harness snapshot, no TRY right is granted, "
            "and every state change lands under _scratch/skill_store"
        ),
        "overall_verdict": verdict["verdict"],
        "overall_verdict_reason": verdict["reason"],
        "labels": verdict["labels"],
        "outcome_counts": {
            name: sum(1 for row in rows if row["outcome"] == name)
            for name in (
                "TRANSITIONED", "NO_TRANSITION_NEUTRAL_PROBE",
                "PROBE_WINDOW_UNAVAILABLE", "UPDATE_PATH_BLOCKED",
            )
        },
        "pre_registered": PRE_REGISTERED,
        "update_path": UPDATE_PATH,
        "probe_window_rule": PROBE_WINDOW_RULE,
        "source_artifact": SOURCE_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "source_artifact_read_only": True,
        "drafts": rows,
        "cost": {
            "llm_calls": 0,
            "evaluation_calls": evaluator.evaluation_calls,
            "evaluation_call_budget": EVALUATION_CALL_BUDGET,
            "consumer_retrains": evaluator.consumer_retrains,
            "retrain_convention": (
                "one retrain per origin per evaluation call, the convention "
                "the bridge and integration slices used"
            ),
            "log": evaluator.log,
        },
        "state_root": STATE_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "wall_seconds": time.perf_counter() - started,
    }
    if dry_run:
        print(json.dumps(
            {
                "verdict": payload["overall_verdict"],
                "outcomes": {
                    row["draft_id"]: row["outcome"] for row in rows
                },
                "cost": {
                    "evaluation_calls": evaluator.evaluation_calls,
                    "consumer_retrains": evaluator.consumer_retrains,
                },
            },
            indent=2, ensure_ascii=False, default=str,
        ))
        return 0

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    (STATE_ROOT / "episodes_before.jsonl").write_text(
        "".join(
            json.dumps(draft["episode"], ensure_ascii=False, default=str) + "\n"
            for draft in drafts
        ),
        encoding="utf-8", newline="\n",
    )
    (STATE_ROOT / "episodes_after.jsonl").write_text(
        "".join(
            json.dumps(
                (row["transition"] or {}).get("updated_episode")
                or next(
                    d["episode"] for d in drafts
                    if d["draft_id"] == row["draft_id"]
                ),
                ensure_ascii=False, default=str,
            ) + "\n"
            for row in rows
        ),
        encoding="utf-8", newline="\n",
    )
    (STATE_ROOT / "transitions.json").write_text(
        json.dumps(
            {
                row["draft_id"]: {
                    "outcome": row["outcome"],
                    "probe_window": row["probe_window"].get(
                        "probe_window_id"
                    ),
                    "transition": row["transition"],
                }
                for row in rows
            },
            indent=2, ensure_ascii=False, default=str,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    payload["state_files"] = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in STATE_ROOT.iterdir() if path.is_file()
    )
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
        "cost: %d evaluation calls, %d Consumer retrains, 0 LLM"
        % (evaluator.evaluation_calls, evaluator.consumer_retrains),
        flush=True,
    )
    return 0


# ------------------------------------------------------------------- report
def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return "--"
    excluded = [str(uid) for uid in (plan.get("excluded_series") or [])]
    if not excluded:
        return "`%s` full batch" % plan["program"]
    return "`%s` minus %s" % (plan["program"], ", ".join(sorted(excluded)))


def _markdown(payload: Mapping[str, Any]) -> str:
    rows = payload["drafts"]
    cost = payload["cost"]
    lines = [
        "# closing the runtime loop: an out-of-selection delayed probe",
        "",
        "**Overall: `%s`** -- %s."
        % (payload["overall_verdict"], payload["overall_verdict_reason"]),
        "",
        "The integration slice formed five `LOCAL_DRAFT` Episodes on the real "
        "channel and stopped there, honestly: its delayed reading had set the "
        "adoption bar, so it was in-selection and could not serve as "
        "promotion evidence.  This slice supplies what was missing -- one "
        "delayed probe per Draft on a window that took part in no selection "
        "-- and hands it to the existing update path.",
        "",
        "**No direction is presumed.**  `_update_delayed` grades three bands; "
        "a probe that restricts a Draft closes the loop exactly as well as "
        "one that promotes it.",
        "",
        "0 LLM calls.  The probe grades an already-adopted plan and chooses "
        "nothing: no plan re-proposed, no shortlist re-run, no mask "
        "re-searched, no threshold touched.  State changes land under "
        "`%s` only." % payload["state_root"],
        "",
        "## Per Draft",
        "",
        "| draft | plan | probe window | origins | source | probe delayed | "
        "status before -> after | outcome |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        window = row["probe_window"]
        probe = row["probe"]
        transition = row["transition"] or {}
        before = (transition.get("before") or {}).get("local_status") or "--"
        after = (transition.get("after") or {}).get("local_status") or "--"
        lines.append(
            "| `%s` | %s | `%s` | %s | %s | %s | `%s` -> `%s` | `%s` |"
            % (
                row["draft_id"], _plan_label(row["adopted_plan"]),
                window.get("probe_window_id"),
                ", ".join(str(o) for o in window.get("probe_origins") or []),
                window.get("origin_source"),
                "--" if probe is None else "%+.6f" % probe["macro_gain"],
                before, after, row["outcome"],
            )
        )
    lines += [
        "",
        "## Evidence fields, before and after",
        "",
        "| draft | evidence level | relation | delayed evaluated | delayed "
        "gain | se_block | gain/se | block origins |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        transition = row["transition"] or {}
        for phase in ("before", "after"):
            view = transition.get(phase)
            if view is None:
                continue
            lines.append(
                "| `%s` %s | `%s` | `%s` | %s | %s | %s | %s | %s |"
                % (
                    row["draft_id"], phase, view["evidence_level"],
                    view["relation"], view["delayed_evaluated"],
                    "--" if view["delayed_gain"] is None
                    else "%+.6f" % view["delayed_gain"],
                    "--" if view["delayed_se_block"] is None
                    else "%.6f" % float(view["delayed_se_block"]),
                    "--" if view["delayed_gain_over_se"] is None
                    else "%.3f" % float(view["delayed_gain_over_se"]),
                    ", ".join(str(o) for o in view["delayed_block_origins"])
                    or "--",
                )
            )
    lines += [
        "",
        "## The update path",
        "",
        "- call: `%s`" % payload["update_path"]["callable"],
        "- signature: `%s`" % payload["update_path"]["signature"],
        "- it reads: %s" % payload["update_path"]["reads"],
        "- the bands: %s" % payload["update_path"]["bands"],
        "- the probe is built by `%s`, the same cluster-unit metric the real "
        "path uses, over rows from `%s`."
        % (
            (rows[0]["probe"] or {}).get("metric_function", "--")
            if rows and rows[0].get("probe") else "--",
            (rows[0]["probe"] or {}).get("evaluator", "--")
            if rows and rows[0].get("probe") else "--",
        ),
        "- nothing is set by hand: the status is whatever that function "
        "returns.",
        "",
        "## Probe windows",
        "",
    ]
    for row in rows:
        window = row["probe_window"]
        lines.append(
            "- **%s** (%s, %s): `%s`, origins %s, %s.  %s"
            % (
                row["draft_id"], row["cohort"], row["consumer_variant"],
                window.get("probe_window_id"),
                window.get("probe_origins"),
                window.get("origin_provenance"),
                (
                    "Farthest read %d; %s."
                    % (
                        window["farthest_index_read"],
                        "inside the boundary"
                        if window.get("available")
                        else "**%s**" % window.get("unavailable_reason"),
                    )
                ),
            )
        )
    unavailable = [
        row for row in rows if row["outcome"] == "PROBE_WINDOW_UNAVAILABLE"
    ]
    blocked = [row for row in rows if row["outcome"] == "UPDATE_PATH_BLOCKED"]
    if unavailable or blocked:
        lines += ["", "### Where it stopped", ""]
        for row in unavailable:
            lines.append(
                "- **%s** -- probe window layer: %s"
                % (row["draft_id"], row["probe_window"]["unavailable_reason"])
            )
        for row in blocked:
            lines.append(
                "- **%s** -- update path layer: `%s` refused (%s)"
                % (
                    row["draft_id"],
                    row["transition"].get("blocked_at_interface"),
                    row["transition"].get("blocked_reason"),
                )
            )
    lines += [
        "",
        "## Cost",
        "",
        "%d evaluation calls of a budget of %d, %d Consumer retrains (%s), "
        "0 LLM calls."
        % (
            cost["evaluation_calls"], cost["evaluation_call_budget"],
            cost["consumer_retrains"], cost["retrain_convention"],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--windows-only", action="store_true",
        help="resolve and legality-check the probe windows, then stop",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="run everything but print the verdict instead of writing",
    )
    args = parser.parse_args(argv)
    if args.windows_only:
        for draft in _drafts():
            window = _probe_window(draft)
            print("=== %s (%s, %s)" % (
                draft["draft_id"], draft["cohort"], draft["consumer_variant"]
            ))
            print(json.dumps(window, indent=2, ensure_ascii=False, default=str))
        return 0
    return run(dry_run=bool(args.dry_run))


if __name__ == "__main__":
    raise SystemExit(main())
