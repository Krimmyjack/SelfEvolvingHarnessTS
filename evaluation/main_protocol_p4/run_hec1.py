"""HEC-1 runner: arms, orderings, units, faces, budgets, faults, records.

A measurement runner, not a gate
--------------------------------
The old shape stopped at the first fault and wrote the verdict at the top of the
artifact.  A twenty-six unit course cannot be run that way -- one bad candidate
would end the curve -- and a run that stops early has no verdict to write.  So
faults are split in two, and the split is the whole difference:

* ``UnitFault`` -- a candidate fails, the window verifier refuses, the served
  context degenerates, an LLM cell runs out.  The unit abstains to identity, the
  reason is recorded, and the course continues.
* ``RunFault`` -- the backend is gone past the retry policy, a wall leaked, a
  global cap blew, the data is wrong, something read held-out, or (in a
  scientific run) the relay returned a transport failure such as HTTP 403.
  The whole run stops and records ``RUN_BLOCKED_NO_VERDICT``, which is *not* a
  scientific verdict and must never be written as one.  A scientific course
  must not convert that into identity and keep scoring.

The verdict is read at the end of the course, by the pre-registered tests in the
contract, and nowhere else.

One authoritative gate
----------------------
Source-v3's round 2856 recorded ``delayed_gate.passes=False`` on the coverage
floor and an ``online_loop`` delayed event of ``approved`` at the same time.
Two gates with different calibres cannot both decide who becomes Active, so here
the P4 ``_gate`` -- coverage floor included -- is the only authority.  The
``online_loop`` approval is recorded, ``gate_disagreement`` is written on every
unit, and the Active set is asserted to grow only through the authority.

What is 0 LLM and what is not
-----------------------------
Everything the ``--smoke`` gate exercises is 0 LLM: the contract check, the
three orderings, identity deployment and evaluation-face scoring, the W3 replay
of three real Source-v3 windows, one empty and one synthetic outer step, and a
threshold calibration.  The live arm path is guarded by
``hec1_contract.assert_launchable``, which refuses while sol has not ratified the
draft and the user has not released the budget, so a forgotten check cannot spend
a call nobody authorised.
"""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import p4b_contract as bounded
from evaluation.main_protocol_p4 import representation_view as views
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_forecast_p4_performance as forecast_p4
from evaluation.main_protocol_p4 import run_main_baselines as baselines
from evaluation.main_protocol_p4 import run_source_line as v1runner
from evaluation.main_protocol_p4 import run_source_line_v3 as v3runner
from evaluation.main_protocol_p4 import scope_repair_distance as distance
from evaluation.main_protocol_p4 import scope_threshold_tool as threshold_tool
from evaluation.main_protocol_p4 import scoped_serving_evaluator as scoped
from evaluation.main_protocol_p4 import smoke_live_scope as live_smoke
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = PROJECT_ROOT / "artifacts/main_protocol"
SUPPLY = ARTIFACTS / "p4s_main_experiment_supply.json"
SOURCE_V3 = ARTIFACTS / "p4w3b_source_line_v3_clean_post_fix_replicate_1.json"
SOURCE_V3_DRY_RUN = ARTIFACTS / "p4w3_source_line_v3.dry_run.json"
SMOKE_OUT = ARTIFACTS / "hec1_smoke.json"

FACE = "support_a"
DELAYED_OFFSET = 48
EVALUATION_OFFSET = contract.EVALUATION_FACE["offset"]

#: The one Task/Consumer HEC-1 runs.  Named once so the census key, the bank
#: rows and the lineage map cannot disagree about what the Task half of the key
#: is.
TASK_CONSUMER_KEY = "forecast|pooled-ridge-a1|sMASE"

#: A well-formed origin that resolves to nothing.  ``TTHAAgentCore`` validates
#: the URL shape before the first stage runs, so the offline path cannot simply
#: pass a word; and ``.invalid`` is reserved by RFC 2606, so if a request ever
#: escaped the scripted backend it would fail to connect rather than reach a real
#: relay.
OFFLINE_BASE_URL = "https://offline.invalid/v1"

#: Instrument failures that already produced a scientific-looking label.
#: A new course must not reuse these run-labels or their checkpoints.
ARCHIVED_SCIENTIFIC_LABELS = {
    "v11fix_forward_live": "RUN_BLOCKED_NO_VERDICT__TRANSPORT_QUOTA",
    "v11fix_reverse_live": "RUN_BLOCKED_NO_VERDICT__TRANSPORT_QUOTA",
    "v11fix_interleaved_live": "RUN_BLOCKED_NO_VERDICT__TRANSPORT_QUOTA",
    "v11live_forward_live": "RUN_BLOCKED_NO_VERDICT__OUTER_BACKEND_BUDGET_LEAK",
    "v11live_reverse_live": "RUN_BLOCKED_NO_VERDICT__OUTER_BACKEND_BUDGET_LEAK",
    "v11live_interleaved_live": "RUN_BLOCKED_NO_VERDICT__OUTER_BACKEND_BUDGET_LEAK",
}

#: Consumer fits one scored face costs when a policy is deployed: the Static
#: reference, plus the scoped evaluator's own raw and program models.  Measured
#: from ``scoped_evaluate``'s ``consumer_fits``, not assumed.
FITS_PER_SCORED_FACE = 3

#: Faces scored per unit-arm: the delayed gate and the evaluation face.
SCORED_FACES_PER_CELL = 2

#: What one cell costs a replay screen **through the cache**: a single
#: ``scoped_evaluate`` call, which fits the raw and the program model and
#: returns both prediction sets.  Measured from ``consumer_fits``, not assumed,
#: and lower than ``FITS_PER_SCORED_FACE`` because the cached path needs no
#: separate Static reference -- the raw vector is already in the entry.
CACHE_FITS_PER_CELL = 2

#: The classification every Fast call gets, so a zero-candidate round can say
#: which of four very different things happened.  "Abstained with a reason" is
#: correct behaviour; "empty output" and "malformed" are instrument faults; and
#: before this field existed the artifact could not tell them apart.
FAST_DECISIONS = tuple(contract.READOUTS["fast_raw_decision"])


class RunFault(RuntimeError):
    """The run stops.  No scientific verdict may be written from it."""


class UnitFault(RuntimeError):
    """This unit abstains to identity.  The course continues."""


# ---------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------

@dataclass
class Ledgers:
    """Every cost, listed apart.  A single total could hide any of them."""

    llm_fast: int = 0
    llm_outer: int = 0
    replay_fits: int = 0
    shadow_fits: int = 0
    course_fits: int = 0
    baseline_fits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    wall_seconds: float = 0.0

    def llm_total(self) -> int:
        return int(self.llm_fast + self.llm_outer)

    def to_dict(self) -> dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "llm_fast": self.llm_fast,
            "llm_outer": self.llm_outer,
            "llm_total": self.llm_total(),
            "replay_fits": self.replay_fits,
            "shadow_fits": self.shadow_fits,
            "course_fits": self.course_fits,
            "baseline_fits": self.baseline_fits,
            # These count the **replay prediction cache** (arm x cell x face x
            # Consumer x Program), aggregated over the online arms.  The
            # shakedown reported 0/0 because nothing ever incremented them,
            # which reads identically to "enabled and never hit" -- so the LLM
            # prompt cache's state is stated separately rather than inferred
            # from a zero.
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (round(self.cache_hits / total, 4)
                               if total else None),
            "cache_counts": "the replay prediction cache",
            "llm_prompt_cache_enabled": False,
            "why_no_llm_prompt_cache": (
                "wiring CachedAgentBackend would change what the arms are "
                "charged for and how their proposals diverge; it is not "
                "enabled for HEC-1 and its absence is recorded rather than "
                "left as a zero that looks like a miss rate"),
            "wall_seconds": round(self.wall_seconds, 1),
        }


@dataclass
class BudgetGuard:
    """A hard stop before the backend, never a check after the invoice.

    P4's third B=4 attempt failed because the budget was checked after the call
    had already been paid for.  ``reserve`` is therefore called *before* a
    request is built: the N+1 call is refused and costs nothing.
    """

    ordering_cap: int
    per_unit_arm_cap: int
    ledgers: Ledgers
    spent_this_cell: int = 0
    scripted: int = 0
    blocked: list[dict[str, Any]] = field(default_factory=list)

    def open_cell(self) -> None:
        self.spent_this_cell = 0

    def reserve(self, *, kind: str, where: Mapping[str, Any]) -> None:
        if self.ledgers.llm_total() + 1 > self.ordering_cap:
            self.blocked.append({"kind": kind, "where": dict(where),
                                 "reason": "ORDERING_LLM_CAP"})
            raise RunFault(
                "the ordering's LLM cap of %d would be exceeded; refused "
                "before the backend and not billed" % self.ordering_cap)
        if kind == "fast" and self.spent_this_cell + 1 > self.per_unit_arm_cap:
            self.blocked.append({"kind": kind, "where": dict(where),
                                 "reason": "LLM_CELL_BUDGET_EXHAUSTED"})
            raise UnitFault(
                "this unit-arm's LLM budget of %d is spent; the cell abstains "
                "to identity and the course continues" % self.per_unit_arm_cap)

    def spend(self, *, kind: str, calls: int = 1, billable: bool = True) -> None:
        """Bill a reserved call.

        ``billable=False`` is the offline path: the scripted backend answers
        without a relay, so the cap is still exercised through ``reserve`` while
        the LLM ledger stays honestly at zero.  A scripted call that billed would
        make "0 LLM" a claim the artifact contradicts.
        """
        if not billable:
            self.scripted += int(calls)
            if kind == "fast":
                self.spent_this_cell += int(calls)
            return
        if kind == "fast":
            self.ledgers.llm_fast += int(calls)
            self.spent_this_cell += int(calls)
        else:
            self.ledgers.llm_outer += int(calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordering_cap": self.ordering_cap,
            "per_unit_arm_cap": self.per_unit_arm_cap,
            "scripted_calls_not_billed": self.scripted,
            "blocked_before_backend": list(self.blocked),
            "why_before": (
                "a post-hoc budget check pays for the call it refuses; this "
                "one refuses first and bills nothing"
            ),
        }


class _OuterLlmBudgetSpent(RuntimeError):
    """This outer step's two physical requests are used.  Not a crash."""


class _MeteredOuterBackend:
    """Physical-request meter for one arm × one outer step.

    Isolated from the inner cell's ``BudgetedAgentBackend(maximum_calls=5)``.
    Schema-correction retries go through ``complete`` and are billed here.
    """

    def __init__(self, inner: Any, *, guard: "BudgetGuard",
                 billable: bool) -> None:
        self.inner = inner
        self.guard = guard
        self.billable = bool(billable)
        self.requests: list[Any] = []

    @property
    def calls(self) -> int:
        return int(getattr(self.inner, "calls", 0) or 0)

    @property
    def maximum_calls(self) -> int:
        return int(getattr(self.inner, "maximum_calls",
                           contract.OUTER_LLM_PER_STEP) or
                   contract.OUTER_LLM_PER_STEP)

    def complete(self, request: Any) -> Any:
        if self.calls >= self.maximum_calls:
            raise _OuterLlmBudgetSpent(
                "OUTER_LLM_BUDGET_SPENT at %d physical requests"
                % self.maximum_calls)
        self.guard.reserve(kind="outer",
                           where={"physical_index": self.calls + 1})
        before = self.calls
        try:
            response = self.inner.complete(request)
        except Exception:
            made = self.calls - before
            if made:
                self.guard.spend(kind="outer", calls=made,
                                 billable=self.billable)
            raise
        made = max(1, self.calls - before)
        self.guard.spend(kind="outer", calls=made, billable=self.billable)
        self.requests.append(request)
        return response


# ---------------------------------------------------------------------------
# units and faces
# ---------------------------------------------------------------------------

def readable_uids() -> list[str]:
    return list(json.loads(SUPPLY.read_text(encoding="utf-8"))["readable_uids"])


def block_uids(span: Sequence[int]) -> list[str]:
    return readable_uids()[int(span[0]):int(span[1])]


def _executor(roster: Any, values: Any, config: Mapping[str, Any]) -> Any:
    """The same executor the frozen line uses, with the same verifier bound."""
    return ScopeExecutor(
        roster, values, config,
        evaluate_fn=views.forecast_runtime._evaluate,
        max_modified_fraction=forecast_p4.MAX_MODIFIED_FRACTION)


class UnitContext:
    """Everything one (block, origin) unit needs, built the same way each time.

    Built per unit rather than cached across units on purpose: the predicate is
    re-resolved at every decision point precisely because the structure it names
    changes, and a resolver carried over from another origin would silently
    answer with the wrong window.
    """

    def __init__(self, unit: Mapping[str, Any]) -> None:
        self.unit = {"block": str(unit["block"]),
                     "span": [int(unit["span"][0]), int(unit["span"][1])],
                     "origin": int(unit["origin"]),
                     "exposure": list(unit.get("exposure") or ())}
        self.origin = int(unit["origin"])
        uids = block_uids(self.unit["span"])
        self.cell, self.variant = baselines._cell(uids)
        self.config = forecast_p4._config(self.origin)
        self.at = forecast_p4._cell_at(self.cell, self.origin)
        self.roster = self.at.roster(FACE)
        self.eval_uids = [str(row["series_uid"]) for row in self.roster
                          if row["role"] == "eval"]
        self.executor = _executor(self.roster, self.at.values, self.config)
        self.resolve = v1runner._resolver(self.variant, self.eval_uids)
        self.features = live_smoke._feature_cards(
            self.variant, self.eval_uids, self.origin)
        self.available = sorted({name for uid in self.eval_uids
                                 for name in (self.features.get(uid) or {})})

    def face_origin(self, offset: int) -> int:
        return self.origin + int(offset)

    def cell_at(self, origin: int) -> Any:
        return forecast_p4._cell_at(self.cell, int(origin))


def _static_reading(ctx: UnitContext, origin: int) -> dict[str, Any]:
    """Static on one face: raw model, raw context, nothing prepared."""
    at = ctx.cell_at(origin)
    return scoped.scoped_evaluate(
        at.roster(FACE), at.values, None, forecast_p4._config(origin),
        origin=int(origin))


def _policy_reading(ctx: UnitContext, origin: int,
                    steps: Sequence[tuple] | None,
                    resolved: frozenset[str] | None) -> dict[str, Any]:
    """One deployed policy on one face, plus Static, as gains per series.

    ``steps=None`` or an empty resolved set is identity: the reading is Static
    by construction, and the gains are exactly zero rather than nearly zero,
    which is what makes "the arm abstained" distinguishable from "the arm
    deployed something that did nothing".
    """
    at = ctx.cell_at(origin)
    config = forecast_p4._config(origin)
    roster = at.roster(FACE)
    static = _guarded(lambda: scoped.scoped_evaluate(
        roster, at.values, None, config, origin=int(origin)))
    base = np.asarray(static["per_view_smase"], dtype=np.float64)
    fits = int(static["consumer_fits"])
    if not steps or resolved is None or not resolved:
        return {
            "identity": True,
            "treated": 0,
            "served": len(ctx.eval_uids),
            "per_series_gain": {uid: 0.0 for uid in ctx.eval_uids},
            "aggregate_gain": 0.0,
            "harmed_fraction": 0.0,
            "max_single_series_harm": 0.0,
            "consumer_fits": fits,
            "mean_smase": float(static["mean_smase"]),
            "static_mean_smase": float(static["mean_smase"]),
        }
    executor = _executor(roster, at.values, config)
    if not executor.verify(tuple(steps), int(origin)).passed:
        raise UnitFault("WINDOW_VERIFIER_REJECTED at origin %d" % origin)
    reading = _guarded(lambda: scoped.scoped_evaluate(
        roster, at.values, executor._compiled(tuple(steps)), config,
        origin=int(origin), scope=frozenset(resolved)))
    fits += int(reading["consumer_fits"])
    gains = base - np.asarray(reading["per_view_smase"], dtype=np.float64)
    material = contract.RISK["material"]
    return {
        "identity": False,
        "treated": int(reading["scope_size"]),
        "served": len(ctx.eval_uids),
        "per_series_gain": {uid: round(float(value), 6)
                            for uid, value in zip(ctx.eval_uids, gains)},
        "aggregate_gain": round(float(gains.mean()), 6),
        "harmed_fraction": round(float((gains < -material).mean()), 4),
        "max_single_series_harm": round(max(0.0, float(-gains.min())), 6),
        "consumer_fits": fits,
        "mean_smase": float(reading["mean_smase"]),
        "static_mean_smase": float(static["mean_smase"]),
    }


class ReplayPredictionCache:
    """Fit each (cell, face, Consumer, Program) once; let Scope only re-mask.

    Why this is exact rather than an approximation
    ----------------------------------------------
    In ``scoped_evaluate`` neither model depends on the Scope.  The raw model is
    fitted on the raw training rows and the program model on the *prepared*
    training rows -- all of them, whatever the Scope selects -- and both predict
    every served series.  The Scope enters at one line:

        prediction = where(in_scope, program_prediction, raw_prediction)

    and each series' loss is computed from its own prediction row alone.  So a
    series' scored value under any Scope is one of exactly two numbers that were
    already computed, and selecting between them reproduces the full evaluation
    bit for bit.  That is what makes a cache legitimate here instead of a
    shortcut: nothing is re-derived, only re-selected.

    The one Scope-dependent thing is legality.  Preparing a served context can
    flatten it, and ``scoped_evaluate`` raises only when such a series is *in*
    the Scope.  The degenerate set is therefore recorded with the entry and the
    original rule is re-applied on every query, so a Scope that reaches a
    degenerate series is refused exactly as it would have been.

    What the key does not contain
    -----------------------------
    The Scope, deliberately -- re-masking is the whole point.  The arm **is** in
    the key: a cache shared between arms would let one arm's fits pay for
    another's reading, and the arms are the contrast.  So is the face origin, so
    is the Consumer configuration, and so is the typed Program with its
    parameters.

    Three ledgers, because they answer three different questions: how much
    Consumer time was actually spent (``physical_fits``), how many readings the
    protocol took (``logical_evaluations``), and how much the cache saved
    (``cache_hits``).
    """

    def __init__(self, arm_name: str) -> None:
        self.arm = str(arm_name)
        self._entries: dict[tuple, dict[str, Any]] = {}
        self.physical_fits = 0
        self.logical_evaluations = 0
        self.cache_hits = 0

    @staticmethod
    def _consumer_signature(config: Mapping[str, Any]) -> str:
        return json.dumps(
            {key: config.get(key) for key in
             ("dataset_id", "period", "support_origin", "selection_origin")},
            sort_keys=True, default=str)

    def key(self, unit: Mapping[str, Any], face_origin: int,
            config: Mapping[str, Any], steps: Sequence[tuple]) -> tuple:
        return (
            self.arm,
            json.dumps(unit, sort_keys=True, default=str),
            int(face_origin),
            self._consumer_signature(config),
            outer_loop._program_signature(
                [{"op": op, "params": dict(params)} for op, params in steps]),
        )

    def _build(self, ctx: "UnitContext", origin: int,
               steps: Sequence[tuple]) -> dict[str, Any]:
        """Two fits, once: the raw model and the program model on this cell."""
        at = ctx.cell_at(origin)
        config = forecast_p4._config(origin)
        roster = at.roster(FACE)
        executor = _executor(roster, at.values, config)
        verified = executor.verify(tuple(steps), int(origin)).passed
        if not verified:
            return {"verifier_passed": False, "eval_uids": list(ctx.eval_uids)}

        compiled = executor._compiled(tuple(steps))
        # Which served series the program would flatten.  Computed the same way
        # scoped_evaluate computes it, so the refusal rule is identical.
        degenerate = []
        for uid in ctx.eval_uids:
            raw = np.asarray(at.values[uid], dtype=np.float64)
            window = raw[int(origin) - scoped.CONTEXT_LENGTH:int(origin)]
            served, _moved, _trace = scoped._prepare(window, compiled)
            _c, _s, method = scoped.forecast_runtime._center_scale(np, served)
            if method == "scale_floor_fallback":
                degenerate.append(uid)
        legal = frozenset(uid for uid in ctx.eval_uids if uid not in degenerate)
        reading = _guarded(lambda: scoped.scoped_evaluate(
            roster, at.values, compiled, config, origin=int(origin),
            scope=legal))
        self.physical_fits += int(reading["consumer_fits"])
        return {
            "verifier_passed": True,
            "eval_uids": list(ctx.eval_uids),
            "raw_per_view": [float(v) for v in reading["static_per_view_smase"]],
            # Program values are defined only for the series the program may
            # legally treat; a degenerate one can never be in a legal Scope.
            "program_per_view": [float(v) for v in reading["per_view_smase"]],
            "degenerate_uids": list(degenerate),
            "consumer_fits": int(reading["consumer_fits"]),
        }

    def reading(self, ctx: "UnitContext", origin: int,
                steps: Sequence[tuple] | None,
                resolved: frozenset[str] | None) -> dict[str, Any]:
        """A ``_policy_reading``-shaped result, re-masked from cached models."""
        self.logical_evaluations += 1
        key = self.key(ctx.unit, origin, forecast_p4._config(origin),
                       steps or ())
        entry = self._entries.get(key)
        if entry is None:
            entry = self._build(ctx, origin, steps or ())
            self._entries[key] = entry
        else:
            self.cache_hits += 1
        if not entry.get("verifier_passed"):
            raise UnitFault("WINDOW_VERIFIER_REJECTED at origin %d" % origin)

        selected = set(resolved or ())
        reached = sorted(selected & set(entry["degenerate_uids"]))
        if reached:
            raise UnitFault(
                "SERVING_CONTEXT_DEGENERATE: preparing the served context "
                "flattened %d scoped series (%s)"
                % (len(reached), ", ".join(reached[:6])))

        uids = entry["eval_uids"]
        raw = np.asarray(entry["raw_per_view"], dtype=np.float64)
        program = np.asarray(entry["program_per_view"], dtype=np.float64)
        mask = np.array([uid in selected for uid in uids], dtype=bool)
        per_view = np.where(mask, program, raw)
        gains = raw - per_view
        material = contract.RISK["material"]
        return {
            "identity": not selected,
            "treated": int(mask.sum()),
            "served": len(uids),
            "per_series_gain": {uid: round(float(value), 6)
                                for uid, value in zip(uids, gains)},
            "aggregate_gain": round(float(gains.mean()), 6),
            "harmed_fraction": round(float((gains < -material).mean()), 4),
            "max_single_series_harm": round(max(0.0, float(-gains.min())), 6),
            "consumer_fits": 0,   # billed once, at build time
            "mean_smase": float(per_view.mean()),
            "static_mean_smase": float(raw.mean()),
        }

    def to_dict(self) -> dict[str, Any]:
        total = self.logical_evaluations
        return {
            "arm": self.arm,
            "entries": len(self._entries),
            "physical_fits": self.physical_fits,
            "logical_evaluations": total,
            "cache_hits": self.cache_hits,
            "hit_rate": (round(self.cache_hits / total, 4) if total else None),
            "key": ["arm", "unit", "face_origin", "consumer_config",
                    "typed_program"],
            "scope_is_not_in_the_key": (
                "re-masking is the point; both models are Scope-independent and "
                "each series' loss depends only on its own prediction row"
            ),
            "never_shared": ["across arms", "across Consumers", "with future "
                             "cells"],
        }


class FaceNotEvaluable(UnitFault):
    """This window has no observed truth to score against, for any arm.

    D2 screened whether the missing-aware sMASE is defined **at the origin**;
    the +144 evaluation face is 144 steps further on and can run past the end of
    a series' observed data.  That is a property of the data, not of a policy, so
    it hits every arm on that unit identically: the unit simply contributes no
    point to the curve, and the count of such units is reported rather than
    quietly reducing the denominator.

    Deliberately a ``UnitFault`` subclass: the course continues.
    """


def _guarded(read: Any) -> dict[str, Any]:
    """Run one reading, turning an unevaluable window into a stated fault."""
    try:
        return read()
    except scoped.ServingContextDegenerate as exc:
        raise UnitFault("SERVING_CONTEXT_DEGENERATE: %s" % str(exc)[:160])
    except RuntimeError as exc:
        if "no observed truth" in str(exc) or "scale floor" in str(exc):
            raise FaceNotEvaluable(str(exc)[:200])
        raise


def authoritative_gate(reading: Mapping[str, Any]) -> dict[str, Any]:
    """The P4 four lines, coverage floor included.  The only activation gate."""
    material = contract.RISK["material"]
    lines = {
        "coverage_floor": int(reading["treated"]) >= contract.RISK["min_treated"],
        "aggregate": float(reading["aggregate_gain"]) >= material,
        "harmed_fraction": (float(reading["harmed_fraction"])
                            <= contract.RISK["max_harmed_fraction"]),
        "single_series_harm": (float(reading["max_single_series_harm"])
                              <= contract.RISK["max_single_series_harm"]),
    }
    return {
        "lines": lines,
        "passes": all(lines.values()),
        "failed_lines": [name for name, ok in lines.items() if not ok],
        "authority": "p4_gate",
        "thresholds": {
            "min_treated": contract.RISK["min_treated"],
            "min_aggregate": material,
            "max_harmed_fraction": contract.RISK["max_harmed_fraction"],
            "max_single_series_harm": contract.RISK["max_single_series_harm"],
        },
    }


#: The two ways the authoritative gate and the lifecycle event can differ, which
#: are not the same event and must not be counted as one.
#:
#: ``AUTHORITY_UPHELD`` -- the P4 gate refused and ``online_loop`` approved, and
#: nothing activated.  This is **structural**: ``online_loop``'s delayed
#: admission has no coverage floor and the P4 gate does, so the two disagree
#: whenever a winner treats fewer than ``MIN_TREATED`` series at the delayed
#: window.  The resolution worked; the record exists so the calibre difference
#: stays visible.  The shakedown produced three of these in 26 units.
#:
#: ``AUTHORITY_BYPASSED`` -- something activated that the authority refused, or
#: the Active set grew without the authority approving.  This is a real fault and
#: it is what "the Active set only ever grows through the authoritative gate"
#: exists to catch.
AUTHORITY_UPHELD = "AUTHORITY_UPHELD"
AUTHORITY_BYPASSED = "AUTHORITY_BYPASSED"
LOST_ACTIVATION = "LOST_ACTIVATION"

#: A disagreement on the coverage floor alone is the structural calibre
#: difference.  A disagreement that touches a **risk** line is not: it means the
#: two gates read the harm differently, and that is a demotion.
COVERAGE_ONLY_LINE = "coverage_floor"
RISK_LINES = ("aggregate", "harmed_fraction", "single_series_harm")


def _store_active_sha(store: Any) -> str | None:
    """The Store's active pointer, or None when it has never been set."""
    path = getattr(store, "active_path", None)
    if path is None or not Path(path).is_file():
        return None
    try:
        return str(json.loads(Path(path).read_text(encoding="utf-8")).get(
            "runtime_bundle_sha") or "") or None
    except (OSError, ValueError):
        return None


def resolve_gate_disagreement(p4_gate: Mapping[str, Any],
                              online_loop_event: Mapping[str, Any] | None,
                              state: Mapping[str, Any] | None = None,
                              ) -> dict[str, Any]:
    """Record both gates, say which one decided, and classify the difference.

    Recording the disagreement rather than reconciling it is deliberate: the two
    calibres differ on the coverage floor, and quietly picking whichever agreed
    with the run would delete the evidence that they differ.

    The classification matters because the two directions are different events.
    ``online_loop`` approving what the P4 gate refused is the *structural*
    calibre difference and the authority holding is the system working.  The P4
    gate approving what ``online_loop`` did not is a ``lost_activation``: a Skill
    that earned its rights and did not get them.  Neither is the fault the
    contract is guarding against, which is the Active set growing through
    anything other than the authority.
    """
    online = str((online_loop_event or {}).get("stage") or "none")
    authoritative = bool(p4_gate.get("passes"))
    online_approved = online == "approved"
    failed = [str(name) for name in (p4_gate.get("failed_lines") or ())]
    risk_lines = [name for name in failed if name in RISK_LINES]
    leaked = state is not None and not state.get("unchanged", True)

    kind = None
    if online_approved and not authoritative:
        # sol's classification, in order of severity.  State first: a leak is a
        # leak whatever line it happened on.
        if leaked:
            kind = AUTHORITY_BYPASSED
        elif risk_lines:
            # The two gates read the *harm* differently.  That is not the
            # coverage-floor calibre difference and it is not disclosable-only.
            kind = AUTHORITY_BYPASSED
        elif failed == [COVERAGE_ONLY_LINE]:
            kind = AUTHORITY_UPHELD
        else:
            kind = AUTHORITY_BYPASSED
    elif authoritative and not online_approved:
        kind = LOST_ACTIVATION
    elif leaked:
        # Agreement is no defence: if state moved while the authority refused,
        # something committed outside the gate.
        kind = AUTHORITY_BYPASSED if not authoritative else None

    return {
        "p4_gate": {"passes": authoritative, "failed_lines": failed},
        "online_loop_event": online,
        "resolved_by": "p4_gate",
        "disagree": online_approved != authoritative,
        "kind": kind,
        "failed_risk_lines": risk_lines,
        "state": dict(state) if state else None,
        "state_unchanged": (None if state is None
                            else bool(state.get("unchanged"))),
        "may_activate": authoritative,
        "demotes_the_ordering": kind == AUTHORITY_BYPASSED,
        "note": (
            "an online_loop approval never triggers activate_approved on its "
            "own, and the P4 gate is now consulted *before* the snapshot is "
            "committed; a coverage-floor-only disagreement with no state "
            "movement is the structural calibre difference and is disclosed, "
            "while a risk-line disagreement or any state movement demotes"
        ),
    }


def classify_authority_breach(cell: Mapping[str, Any]) -> str | None:
    """Did anything escape the authority on this cell?

    Three ways, and the first is the one the P0 leak took: **state moved while
    the gate refused**.  An activation the gate refused, or an activation with
    no authority record at all, are the other two.  ``AUTHORITY_UPHELD`` means
    the guard fired and nothing moved.
    """
    disagreement = cell.get("gate_disagreement")
    state = cell.get("authority_state")
    if state is not None and not state.get("unchanged", True):
        may = (disagreement or {}).get("may_activate")
        if not may:
            return AUTHORITY_BYPASSED
    if disagreement is not None and disagreement.get(
            "kind") == AUTHORITY_BYPASSED:
        return AUTHORITY_BYPASSED
    if not cell.get("activated"):
        return None
    if disagreement is None:
        return AUTHORITY_BYPASSED  # activated with no authority record at all
    if not disagreement.get("may_activate"):
        return AUTHORITY_BYPASSED
    return None


def h_readings(*, window: int, treated_prev: Sequence[str] | None,
               treated_now: Sequence[str] | None,
               per_series_gain: Mapping[str, float] | None,
               features_prev: Mapping[str, Mapping[str, float]] | None = None,
               features_now: Mapping[str, Mapping[str, float]] | None = None,
               predicate: Sequence[Mapping[str, Any]] | None = None,
               ) -> dict[str, Any]:
    """H1 / H2 / H3, mechanically, at one verification face.

    H1 membership turnover, H2 effect non-stationarity, H3 coverage as pattern
    prevalence.  Every number is a count over sets the reading already produced;
    nothing is fitted and no LLM is involved.
    """
    material = contract.RISK["material"]
    facts = drafts.attribute(treated_prev, treated_now, per_series_gain,
                             material)
    gains = {str(uid): float(value)
             for uid, value in dict(per_series_gain or {}).items()}
    harmed_new = [uid for uid in facts["new_entrant"]
                  if gains.get(uid, 0.0) < -material]
    harmed_continuing = [uid for uid in facts["continuing"]
                         if gains.get(uid, 0.0) < -material]
    left_by_feature = []
    for uid in facts["left"]:
        card = dict((features_now or {}).get(uid) or {})
        if not card or not predicate:
            continue
        if any(_clause_fails(card, clause) for clause in predicate):
            left_by_feature.append(uid)
    return {
        "window": int(window),
        "attribution": facts,
        "H1_new_entrant_share_of_harm": (
            round(len(harmed_new) / len(facts["harmed"]), 4)
            if facts["harmed"] else None),
        "H1_continuing_harm_rate": (
            round(len(harmed_continuing) / len(facts["continuing"]), 4)
            if facts["continuing"] else None),
        "H2_continuing_sign_flip_rate": (
            round(len(harmed_continuing) / len(facts["continuing"]), 4)
            if facts["continuing"] else None),
        "H3_treated": len(facts["treated_now"]),
        "H3_left": len(facts["left"]),
        "H3_left_because_a_feature_exited_predicate": len(left_by_feature),
        "H3_left_share_explained": (
            round(len(left_by_feature) / len(facts["left"]), 4)
            if facts["left"] else None),
    }


def _clause_fails(card: Mapping[str, float],
                  clause: Mapping[str, Any]) -> bool:
    name = str(clause.get("feature"))
    if name not in card:
        return True
    value, threshold = float(card[name]), float(clause.get("threshold"))
    op = str(clause.get("op"))
    if op == "<=":
        return not value <= threshold
    if op == ">=":
        return not value >= threshold
    if op == "<":
        return not value < threshold
    return not value > threshold


def classify_fast_decision(payload: Mapping[str, Any] | None,
                           candidate_ids: Sequence[str],
                           no_proposal_reason: str | None = None,
                           ) -> dict[str, Any]:
    """Which of four very different things a Fast call did.

    Source-v3 recorded two zero-candidate rounds and the artifact could not say
    whether the agent had abstained with a reason -- correct behaviour -- or
    returned nothing at all, which is an instrument fault.  The two mean
    opposite things and are counted apart from here on.  The stored text is a
    de-identified summary, never the raw transcript.
    """
    if candidate_ids:
        decision = "PROPOSED"
    elif no_proposal_reason:
        decision = "ABSTAINED_WITH_REASON"
    elif payload is None:
        decision = "EMPTY_OUTPUT"
    else:
        decision = "MALFORMED"
    summary = ""
    if payload is not None:
        summary = json.dumps(payload, sort_keys=True, default=str)[:400]
    return {
        "decision": decision,
        "candidate_count": len(list(candidate_ids)),
        "no_proposal_reason": no_proposal_reason,
        "payload_summary": summary,
        "vocabulary": list(FAST_DECISIONS),
    }


# ---------------------------------------------------------------------------
# the replay screen the outer loop is handed
# ---------------------------------------------------------------------------

def reserve_for_future_steps(k_index: int, total_steps: int, *,
                             period: int,
                             fits_per_cell: int = CACHE_FITS_PER_CELL) -> int:
    """Hold back one screen's worth of **fits** for every outer step to come.

    sol v1.1 C.  A screen at step *j* re-scores ``j x period`` already-processed
    cells, so the conservative cost is that many first sightings; over a
    five-step course the cell counts are 5 + 10 + 15 + 20 + 25 = 75.  The
    reservation is that sum for the steps *after* this one, converted to fits at
    the cache's measured build cost.

    The unit conversion is the whole point of this function.  sol states the
    reservation in cells and the budget is denominated in fits, and subtracting
    one from the other reserves a third of what was intended -- which lets an
    early step spend most of the allowance and starves exactly the later steps
    the reservation exists to protect.

    The arithmetic closes: at ``fits_per_cell = 2`` a 26-unit course reserves
    140 of its 156 fits before step 1, leaving 16 -- eight cells, against the
    five that step 1's screen needs -- and the whole course's five screens cost
    150 of 156.

    What it guarantees is narrow and worth stating exactly: **if a later step
    has a candidate, the budget will not block its first one.**  It does not
    promise every candidate is screened, and it adds no per-step candidate cap
    and no new ordering -- the frozen deterministic order is untouched, and a
    candidate the budget could not reach is recorded rather than dropped.
    """
    if total_steps <= 0:
        return 0
    return sum(step * int(period) * int(fits_per_cell)
               for step in range(int(k_index) + 1, int(total_steps) + 1))


def replay_screen_for(contexts: Sequence[UnitContext], ledgers: Ledgers,
                      cache: ReplayPredictionCache | None = None):
    """Re-score a candidate on cells this arm has already processed.

    Only those cells.  A screen that reached a unit the arm has not run yet
    would be reading the future, and one that reached another arm's cells would
    make the arms share evidence they are supposed to be contrasted on.
    """

    def replay(*, steps, scope):
        cells: list[dict[str, Any]] = []
        spent_before = cache.physical_fits if cache is not None else 0
        fits = 0
        for ctx in contexts:
            resolved = (ctx.resolve(scope, ctx.origin)
                        if scope else frozenset(ctx.eval_uids))
            try:
                if cache is not None:
                    reading = cache.reading(ctx, ctx.origin, steps, resolved)
                else:
                    reading = _policy_reading(ctx, ctx.origin, steps, resolved)
            except UnitFault as exc:
                cells.append({"unit": ctx.unit, "unusable": str(exc)[:160],
                              "aggregate_gain": None})
                continue
            fits += int(reading["consumer_fits"])
            cells.append({
                "unit": ctx.unit,
                "treated": reading["treated"],
                "aggregate_gain": reading["aggregate_gain"],
                "harmed_fraction": reading["harmed_fraction"],
                "max_single_series_harm": reading["max_single_series_harm"],
            })
        if cache is not None:
            fits = cache.physical_fits - spent_before
        ledgers.replay_fits += fits
        return {"cells": cells, "fits": fits}

    # What one screen costs, published so ``consolidate`` can refuse *before*
    # spending it.  This is the **worst case** -- every cell a first sight of
    # this (program, face) -- and a screen that hits the cache costs nothing.
    # Reserving the worst case is deliberate: a reservation that assumed hits
    # would let the first genuinely new program overrun the budget.  The per
    # cell figure differs by path: the cached build is one scoped_evaluate call,
    # the uncached reading additionally re-derives its Static reference.
    per_cell = CACHE_FITS_PER_CELL if cache is not None else FITS_PER_SCORED_FACE
    replay.estimated_fits_per_candidate = per_cell * len(list(contexts))
    replay.fits_per_cell = per_cell
    replay.cells = len(list(contexts))
    return replay


# ---------------------------------------------------------------------------
# the smoke gate
# ---------------------------------------------------------------------------

def _smoke_contract() -> dict[str, Any]:
    state = contract.assert_frozen()
    return {
        "check": "contract",
        "passed": bool(state["frozen"]),
        "mechanical_drift": state["failures"],
        "is_ratified": state["is_ratified"],
        "launchable": {
            phase: contract.assert_launchable(phase)["launchable"]
            for phase in ("phase_s", "phase_t_forward", "phase_f")},
        "note": (
            "a clean drift check is not a ratification; every phase is "
            "correctly not launchable while sol and the user have not answered"
        ),
    }


def _smoke_orderings() -> dict[str, Any]:
    units = contract.phase_t_units()
    expected = sorted((row["block"], row["origin"]) for row in units)
    rows = {}
    for name in contract.ORDERINGS:
        sequence = contract.ordering(name)
        rows[name] = {
            "length": len(sequence),
            "is_permutation": sorted(
                (row["block"], row["origin"]) for row in sequence) == expected,
            "first": sequence[0] if sequence else None,
            "last": sequence[-1] if sequence else None,
        }
    supply = json.loads(
        (ARTIFACTS / "p4ac_hec1_course_supply.json").read_text(
            encoding="utf-8"))
    intersection = supply["exposure_cross_check"]["held_out_intersection"]
    return {
        "check": "orderings_and_exposure",
        "passed": (all(row["is_permutation"] and row["length"] == len(units)
                       for row in rows.values())
                   and not intersection),
        "phase_t_units": len(units),
        "orderings": rows,
        "held_out_intersection": len(intersection),
    }


def _smoke_units(limit: int = 2) -> dict[str, Any]:
    """Two units, identity deployment, evaluation-face scoring.  0 LLM.

    The arms exercised are the two that need no Skill: Static, and the empty-K0
    frozen arm, which starts from h0 with nothing to recall and therefore
    deploys identity.  Both must land on Static exactly -- if a declining arm
    were not bit-identical to Static, every gain in the course would be measured
    against a moving reference.
    """
    rows: list[dict[str, Any]] = []
    fits = 0
    for unit in contract.ordering("forward")[:int(limit)]:
        started = time.time()
        ctx = UnitContext(unit)
        evaluation_origin = ctx.face_origin(EVALUATION_OFFSET)
        static = _static_reading(ctx, evaluation_origin)
        fits += int(static["consumer_fits"])
        identity = _policy_reading(ctx, evaluation_origin, None, None)
        fits += int(identity["consumer_fits"])
        gate = authoritative_gate(identity)
        disagreement = resolve_gate_disagreement(
            gate, {"stage": "approved"})
        rows.append({
            "unit": ctx.unit,
            "served": len(ctx.eval_uids),
            "evaluation_origin": evaluation_origin,
            "static_mean_smase": round(float(static["mean_smase"]), 6),
            "arms": {
                "Static": {"treated": 0, "aggregate_gain": 0.0},
                "A3-frozen(empty K0)": {
                    "treated": identity["treated"],
                    "aggregate_gain": identity["aggregate_gain"],
                    "identity": identity["identity"]},
            },
            "identity_is_exactly_static": (
                identity["aggregate_gain"] == 0.0
                and abs(identity["mean_smase"]
                        - float(static["mean_smase"])) < 1e-12),
            "gate": gate,
            "gate_disagreement": disagreement,
            "wall_seconds": round(time.time() - started, 2),
        })
    return {
        "check": "two_units_identity_and_scoring",
        "passed": all(row["identity_is_exactly_static"] for row in rows),
        "units": rows,
        "consumer_fits": fits,
        "llm_calls": 0,
        "gate_authority_holds": all(
            not row["gate"]["passes"]
            and row["gate_disagreement"]["resolved_by"] == "p4_gate"
            and row["gate_disagreement"]["disagree"]
            and not row["gate_disagreement"]["may_activate"]
            for row in rows),
    }


def _source_v3_replay_cases() -> list[dict[str, Any]]:
    """The three real re-encounter windows, rebuilt from the frozen artifact.

    Per-series vectors in that artifact are positional over face A in
    lexicographic order, and the mainline's hand attribution was checked against
    the executor's own ``delayed_serving_series`` before being used.  Rebuilding
    the sets from those positions is therefore reading the artifact, not
    re-deriving it.
    """
    face = block_uids([160, 200])[:20]
    payload = json.loads(SOURCE_V3.read_text(encoding="utf-8"))
    cases = []
    for round_ in payload["rounds"]:
        reencounter = round_.get("re_encounter_gate") or {}
        if not reencounter or reencounter.get("passes") is not False:
            continue
        gains = list(reencounter.get("per_series_gain") or ())
        treated_now = [face[index] for index, value in enumerate(gains)
                       if value != 0]
        cases.append({
            "restricted_at_origin": int(round_["origin"]),
            "window": int(reencounter["read_origin"]),
            "failed_lines": list(reencounter.get("failed_lines") or ()),
            "treated_prev": list(round_.get("delayed_serving_series") or ()),
            "treated_now": treated_now,
            "per_series_gain": {face[index]: float(value)
                               for index, value in enumerate(gains)
                               if value != 0},
        })
    return cases


#: What the three Source-v3 windows have to classify as.  sol's own reading of
#: the three mechanisms, and the acceptance criterion D4 states.
EXPECTED_STATES = {1896: drafts.FLAGGED, 2376: drafts.WAITING,
                   2616: drafts.REVISABLE}


def _smoke_three_state() -> dict[str, Any]:
    material = contract.RISK["material"]
    rows: list[dict[str, Any]] = []
    for case in _source_v3_replay_cases():
        ledger = drafts.DraftLedger()
        draft = ledger.restrict(
            program_steps=(("outlier_mad", {}),),
            root_scope={"scope_type": "serving_series_predicate",
                        "predicate": [{"feature": "local_robust_z_peak",
                                       "op": ">=", "threshold": 3.0}]},
            current_scope={"scope_type": "serving_series_predicate",
                           "predicate": [{"feature": "local_robust_z_peak",
                                          "op": ">=", "threshold": 3.0}]},
            origin=case["restricted_at_origin"],
            delayed_reading={"delayed_origin": case["window"], "lines": {}},
        )
        entry = ledger.record_verification(
            draft, window=case["window"],
            failed_lines=case["failed_lines"],
            per_series_gain=case["per_series_gain"],
            treated_prev=case["treated_prev"],
            treated_now=case["treated_now"],
            material=material)
        expected = EXPECTED_STATES.get(case["restricted_at_origin"])
        refused_clause = None
        if draft.state == drafts.FLAGGED:
            try:
                ledger.record_revision(
                    draft, origin=case["window"],
                    new_scope=dict(draft.current_scope), preflight=None,
                    support=None)
                refused_clause = False
            except ValueError:
                refused_clause = True
        rows.append({
            "restricted_at_origin": case["restricted_at_origin"],
            "window": case["window"],
            "failed_lines": case["failed_lines"],
            "turnover": "%d -> %d" % (len(case["treated_prev"]),
                                     len(case["treated_now"])),
            "attribution": {
                key: entry["attribution"][key] for key in (
                    "harm_from_new_entrant", "harm_from_continuing",
                    "dominant")},
            "continuing": len(entry["attribution"]["continuing"]),
            "new_entrant": len(entry["attribution"]["new_entrant"]),
            "left": len(entry["attribution"]["left"]),
            "state": draft.state,
            "expected": expected,
            "matches": draft.state == expected,
            "verification_attempts": draft.verification_attempts,
            "clause_request_refused_when_flagged": refused_clause,
            "may_add_clause": draft.may_add_clause(),
        })
    # The WAITING path must not spend a revision on a free re-verification.
    ledger = drafts.DraftLedger()
    waiting = ledger.restrict(
        program_steps=(("outlier_mad", {}),), root_scope={},
        current_scope={}, origin=2376,
        delayed_reading={"delayed_origin": 2424, "lines": {}})
    ledger.record_verification(
        waiting, window=2616, failed_lines=["coverage_floor"],
        per_series_gain={}, treated_prev=[], treated_now=[],
        material=material, consumes_attempt=False)
    free_reverification = (waiting.state == drafts.WAITING
                          and waiting.verification_attempts == 0
                          and waiting.revisions == 1)
    return {
        "check": "three_state_replay",
        "passed": (len(rows) == 3
                   and all(row["matches"] for row in rows)
                   and all(row["clause_request_refused_when_flagged"] is not False
                           for row in rows)
                   and free_reverification),
        "cases": rows,
        "waiting_reverification_is_free": free_reverification,
        "llm_calls": 0,
        "consumer_fits": 0,
    }


def _smoke_outer_loop() -> dict[str, Any]:
    """One empty step and one synthetic step, both 0 LLM.

    The synthetic bank carries a candidate that breaches the single-series line
    on a cell the arm has already processed.  It has to be eliminated, and
    nothing may reach the Active set from here at all.
    """
    empty_ledger = drafts.DraftLedger()
    empty = outer_loop.consolidate(bank=[], ledger=empty_ledger, k_index=0)

    def bank_row(origin: int, gains: Mapping[str, float]) -> dict[str, Any]:
        return {
            "unit": {"block": "[0:40]", "origin": origin},
            "task_consumer_key": TASK_CONSUMER_KEY,
            "program_steps": [{"op": "outlier_mad", "params": {}}],
            "features": {uid: {"local_robust_z_peak": 9.0} for uid in gains},
            "per_series_gain": dict(gains),
        }

    bank = [bank_row(1176, {"a": 0.4, "b": 0.3, "c": 0.2, "d": 0.1, "e": 0.1}),
            bank_row(1896, {"a": 0.4, "b": 0.3, "c": 0.25, "d": 0.1, "e": 0.1})]

    def unsafe_replay(*, steps, scope):
        return {"cells": [{"unit": {"block": "[0:40]", "origin": 1176},
                           "treated": 5, "aggregate_gain": 0.22,
                           "harmed_fraction": 0.0,
                           "max_single_series_harm": 0.91}],
                "fits": 2}

    def safe_replay(*, steps, scope):
        return {"cells": [{"unit": {"block": "[0:40]", "origin": 1176},
                           "treated": 5, "aggregate_gain": 0.22,
                           "harmed_fraction": 0.0,
                           "max_single_series_harm": 0.02}],
                "fits": 2}

    rejected_ledger = drafts.DraftLedger()
    rejected = outer_loop.consolidate(
        bank=bank, ledger=rejected_ledger, k_index=1, replay=unsafe_replay)
    accepted_ledger = drafts.DraftLedger()
    accepted = outer_loop.consolidate(
        bank=bank, ledger=accepted_ledger, k_index=1, replay=safe_replay)
    return {
        "check": "outer_loop",
        "passed": (
            empty.slow_calls == 0 and empty.replay_fits == 0
            and empty.empty_reason == "the bank is empty"
            and not empty.drafts_opened
            and len(rejected.rejected) == 1
            and not rejected.drafts_opened
            and not rejected_ledger.resupplied_programs()
            and len(accepted.drafts_opened) == 1
            and list(accepted_ledger.resupplied_programs())
                == accepted.drafts_opened
            and accepted.slow_calls == 0
            and not accepted.to_dict()["wrote_active"]),
        "empty_step": {"slow_calls": empty.slow_calls,
                       "replay_fits": empty.replay_fits,
                       "empty_reason": empty.empty_reason},
        "unsafe_candidate": {
            "outcome": [row["outcome"] for row in rejected.candidates],
            "violated": [item["violated"]
                         for row in rejected.rejected
                         for item in row["violations"]],
            "drafts_opened": rejected.drafts_opened,
        },
        "safe_candidate": {
            "drafts_opened": accepted.drafts_opened,
            "resupplied": list(accepted_ledger.resupplied_programs()),
            "aliases_collapsed": [row["aliases"] for row in accepted.groups],
        },
        "wrote_active": False,
        "llm_calls": 0,
    }


def _smoke_threshold_tool() -> dict[str, Any]:
    """Synthetic rows: the widest feasible edge, the tie-break, the refusal.

    Two calibrations, because they exercise different halves of the rule.  With
    the harmful series well below every candidate edge, ``>= 3`` and ``>= 6``
    select the same five rows, and the tie has to resolve to the coarser box --
    the lower edge -- since that is the one the bins actually describe on the
    next window.  Adding a harmful series at ``z = 4`` makes ``>= 3`` breach the
    single-series line, and the widest feasible edge becomes a unique ``>= 6``.
    """
    z_bins = threshold_tool.frozen_bin_edges("local_robust_z_peak")
    rows = [
        {"features": {"local_robust_z_peak": 9.0, "missing_fraction": 0.30},
         "gain": 0.50},
        {"features": {"local_robust_z_peak": 8.0, "missing_fraction": 0.25},
         "gain": 0.40},
        {"features": {"local_robust_z_peak": 7.0, "missing_fraction": 0.22},
         "gain": 0.30},
        {"features": {"local_robust_z_peak": 6.5, "missing_fraction": 0.21},
         "gain": 0.20},
        {"features": {"local_robust_z_peak": 6.2, "missing_fraction": 0.21},
         "gain": 0.10},
        {"features": {"local_robust_z_peak": 2.0, "missing_fraction": 0.00},
         "gain": -0.90},
        {"features": {"local_robust_z_peak": 1.0, "missing_fraction": 0.00},
         "gain": -0.80},
    ]
    tied = threshold_tool.calibrate(
        feature="local_robust_z_peak", direction=">=", rows=rows)
    unique = threshold_tool.calibrate(
        feature="local_robust_z_peak", direction=">=",
        rows=rows + [{"features": {"local_robust_z_peak": 4.0,
                                   "missing_fraction": 0.10},
                      "gain": -0.90}])
    refused = None
    try:
        threshold_tool.calibrate(
            feature="period_reliability", direction="<=",
            rows=[{"features": {"period_reliability": 0.9}, "gain": -1.0}])
    except threshold_tool.NoFeasibleThreshold as exc:
        refused = exc.to_dict()
    unknown = None
    try:
        threshold_tool.calibrate(feature="series_uid", direction=">=", rows=rows)
    except Exception as exc:  # noqa: BLE001 - the refusal is the reading
        unknown = type(exc).__name__
    ignored = threshold_tool.clause_from_slow(
        {"scope_clause": {"feature": "local_robust_z_peak", "op": ">=",
                          "threshold": 4.25},
         "rationale": "spiky series are the ones the filter helps"},
        rows=rows)
    shadow = threshold_tool.best_stump(rows=rows)
    consistent = all(
        (row["feasible"] is (row["threshold"] in {
            item["threshold"] for item in tied["candidates_tried"]
            if item["feasible"]}))
        for row in tied["candidates_tried"])
    return {
        "check": "threshold_tool",
        "passed": (
            tied["threshold"] == 3.0 and tied["treated"] == 5
            and tied["tie_break"] == "coarser_box"
            and unique["threshold"] == 6.0
            and unique["tie_break"] == "unique_widest"
            and refused is not None
            and refused["outcome"] == "NO_FEASIBLE_THRESHOLD"
            and unknown == "ScopeError"
            and threshold_tool.LLM_THRESHOLD_IGNORED in ignored["notes"]
            and ignored["threshold"] == tied["threshold"]
            and shadow["outcome"] == "BEST_STUMP"
            and shadow["deployable"] is False
            and consistent),
        "bin_edges": list(z_bins),
        "tied_case": {"threshold": tied["threshold"],
                      "treated": tied["treated"],
                      "tie_break": tied["tie_break"]},
        "unique_case": {"threshold": unique["threshold"],
                        "treated": unique["treated"],
                        "tie_break": unique["tie_break"]},
        "no_feasible_threshold": bool(refused),
        "feature_outside_vocabulary_refused": unknown,
        "llm_threshold_ignored": {
            "notes": ignored["notes"],
            "slow_returned": ignored["slow_threshold_as_returned"],
            "tool_used": ignored["threshold"]},
        "shadow": {"feature": shadow.get("feature"),
                   "direction": shadow.get("direction"),
                   "threshold": shadow.get("threshold"),
                   "deployable": shadow.get("deployable"),
                   "agrees_with_slow": ignored.get("slow_and_shadow_agree")},
        "feasibility_agrees_between_calibrate_and_stump": consistent,
        "llm_calls": 0,
    }


#: Fields the current Source-v3 runner emits that the reference dry-run artifact
#: predates.  Both were added by the v3b replicate ledger before any of this
#: work, and the reference artifact is history and is not rewritten -- so the
#: delta is declared here rather than allowed to look like a fresh regression.
KNOWN_DRY_RUN_FIELD_DELTA = ("run_label", "run_ledger")


def _smoke_source_v3_regression() -> dict[str, Any]:
    """Source-v3's dry run must still produce the fields it produced before.

    The comparison is the field set and the embedded contract receipt, not the
    status: ``status`` depends on whether transport happens to be configured on
    the machine running the smoke, and comparing it would make the regression
    fail for a reason that has nothing to do with the code.
    """
    from evaluation.main_protocol_p4 import run_source_line_v3 as v3runner

    reference = json.loads(SOURCE_V3_DRY_RUN.read_text(encoding="utf-8"))
    report = v3runner.run(dry_run=True)
    missing = sorted(set(reference) - set(report))
    added = sorted(set(report) - set(reference))
    unexplained = sorted(set(added) - set(KNOWN_DRY_RUN_FIELD_DELTA))
    contract_same = (json.dumps(report.get("contract_v3"), sort_keys=True,
                                default=str)
                     == json.dumps(reference.get("contract_v3"), sort_keys=True,
                                   default=str))
    frozen_same = (
        report.get("contract_frozen", {}).get("v3", {}).get("frozen")
        == reference.get("contract_frozen", {}).get("v3", {}).get("frozen"))
    return {
        "check": "source_v3_dry_run_regression",
        "passed": (not missing and not unexplained and contract_same
                   and frozen_same
                   and int(report.get("llm_calls") or 0) == 0),
        "fields_missing": missing,
        "fields_added": added,
        "fields_added_unexplained": unexplained,
        "known_field_delta": list(KNOWN_DRY_RUN_FIELD_DELTA),
        "contract_receipt_identical": contract_same,
        "contract_frozen_identical": frozen_same,
        "llm_calls": int(report.get("llm_calls") or 0),
        "status": report.get("status"),
        "reference": SOURCE_V3_DRY_RUN.name,
        "note": (
            "status and written_at are run-dependent and are not compared; "
            "run_label and run_ledger post-date the reference artifact and are "
            "declared as a known delta rather than treated as a regression"
        ),
    }


SMOKE_CHECKS = (
    ("contract", _smoke_contract),
    ("orderings_and_exposure", _smoke_orderings),
    ("two_units_identity_and_scoring", _smoke_units),
    ("three_state_replay", _smoke_three_state),
    ("outer_loop", _smoke_outer_loop),
    ("threshold_tool", _smoke_threshold_tool),
    ("source_v3_dry_run_regression", _smoke_source_v3_regression),
)


def smoke() -> dict[str, Any]:
    started = time.time()
    results: list[dict[str, Any]] = []
    for name, check in SMOKE_CHECKS:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001 - a failed check is a reading
            results.append({"check": name, "passed": False,
                            "error": "%s: %s" % (type(exc).__name__,
                                                 str(exc)[:400])})
    fits = sum(int(row.get("consumer_fits") or 0) for row in results)
    return {
        "stage": "HEC1_SMOKE",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "data_version": contract.DATA_VERSION,
        "checks": results,
        "passed": all(row.get("passed") for row in results),
        "checks_passed": sum(1 for row in results if row.get("passed")),
        "checks_total": len(results),
        "boundary": {
            "llm_calls": 0,
            "consumer_fits": fits,
            "held_out_reads": 0,
            "thresholds_changed": 0,
            "methods_ttha_files_changed": 0,
        },
        "wall_seconds": round(time.time() - started, 1),
        "releases": (
            "nothing; the smoke is a delivery gate and the contract still needs "
            "sol's ratification and the user's budget release before an arm may "
            "spend an LLM call"
        ),
    }


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArmSpec:
    """One arm's four properties.  Everything else about it is shared."""

    name: str
    start: str          # "static" | "k0" | "h0"
    write_back: bool
    outer: bool
    llm: bool

    @property
    def slug(self) -> str:
        """The arm name as a canonical identifier.

        ``EditManifest`` requires ``^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$`` of every
        pattern id, so ``A5-frozen`` is refused and the whole round dies on it
        before a single probe -- which is exactly how the first offline run
        failed on four of six units.
        """
        return "hec1-%s" % self.name.lower().replace("_", "-")


def arm_specs(*, k0_empty: bool) -> tuple[ArmSpec, ...]:
    """The arm set, which depends only on whether Phase S left a survivor.

    When K0 is empty, A5-online would be bit-identical to A3-online, so it is
    not run: paying for an equivalent arm buys no contrast.  What *is* still run
    is the frozen contrast, because "starts at h0, adapts inside each unit, takes
    nothing away" is equivalent to no other arm and is the only control
    criterion 1 has.
    """
    if k0_empty:
        return (
            ArmSpec("Static", "static", False, False, False),
            ArmSpec("A3-frozen", "h0", False, False, True),
            ArmSpec("A3-online", "h0", True, True, True),
        )
    return (
        ArmSpec("Static", "static", False, False, False),
        ArmSpec("A5-frozen", "k0", False, False, True),
        ArmSpec("A5-online", "k0", True, True, True),
        ArmSpec("A3-online", "h0", True, True, True),
    )


class OuterSlowAgent:
    """The outer loop's Slow call: semantics only, never a number.

    Asked for ``{feature, direction, rationale}`` against the frozen
    ``slow_scope_clause_v1`` schema.  The schema still has a ``threshold`` field
    and the prompt tells the model not to rely on it; whatever comes back there
    is dropped by ``scope_threshold_tool.clause_from_slow`` and recorded as
    ``LLM_THRESHOLD_IGNORED``.  Removing the field would rotate the snapshot
    lock for no method gain, so the enforcement lives in the tool.

    ``snapshot`` is the arm's active snapshot, and it is required rather than
    optional: ``core.run_stage`` reads ``harness_view.instruction`` to build the
    system message, so a plain ``{}`` raises ``AttributeError`` on the first
    call that actually needs Slow.  Phase S never reached this path (both of its
    outer steps found no candidate needing Slow, so ``llm_outer`` was 0) and the
    end-to-end tests scripted the outer Slow, so nothing exercised it until the
    first Forward attempt -- which is why that attempt is an instrument record
    (``RUN_BLOCKED_NO_VERDICT``) and not a reading.  The view is resolved the
    same way ``scope_clause_agent`` resolves it for the Source line: Slow's role
    with empty public features, so no Target observation reaches the Slow view
    through this door.
    """

    def __init__(self, core: Any, *, vocabulary: Sequence[str],
                 guard: "BudgetGuard", snapshot: Any) -> None:
        self.core = core
        self.vocabulary = [str(name) for name in vocabulary]
        self.guard = guard
        self.snapshot = snapshot
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *, candidate: Mapping[str, Any],
                 rejected: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        rows = list(candidate.get("rows") or ())
        public_input = {
            "task": (
                "A program helps on average across the served series of several "
                "already-processed units and damages a few of them past the "
                "deployment's risk budget.  Name ONE deployment-visible feature "
                "and a direction whose conjunction with the current scope would "
                "leave the damaged series outside it.  Do NOT choose a "
                "threshold: the runtime calibrates it on frozen bins."
            ),
            "allowed_features": self.vocabulary,
            "current_scope": candidate.get("base_scope"),
            "program": candidate.get("program_steps"),
            "evidence_rows": [
                {"features": row.get("features"), "gain": row.get("gain")}
                for row in rows[:60]
            ],
            "already_refused": [dict(row) for row in (rejected or ())],
            "output_contract": (
                "Return one scope_clause object: {feature, op, threshold}.  The "
                "threshold is ignored and recorded as ignored; only feature and "
                "op are read."
            ),
        }
        from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: PLC0415
            AgentRole,
        )
        from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: PLC0415
            resolve_harness_view,
        )

        view = resolve_harness_view(self.snapshot, {}, role="slow")
        backend = getattr(self.core, "backend", None)
        metered = hasattr(backend, "calls") and hasattr(backend, "maximum_calls")
        if metered and int(backend.calls) >= int(backend.maximum_calls):
            self.calls.append({"candidate": candidate.get("kind"),
                               "outcome": "OUTER_LLM_BUDGET_SPENT"})
            return {"outcome": "OUTER_LLM_BUDGET_SPENT"}
        # Unmetered test doubles have no per-complete hook: one reserve/spend
        # pair still exercises the ordering cap the way the original tests do.
        if not metered:
            self.guard.reserve(kind="outer",
                               where={"candidate": candidate.get("kind")})
        try:
            stage = self.core.run_stage(
                role=AgentRole.SLOW,
                stage="edit",
                case_id="hec1-outer-%s" % candidate.get("kind", "step"),
                public_input=public_input,
                harness_view=view,
                output_schema_name="slow_scope_clause_v1",
                output_schema=self.core.load_stage_schema(
                    "slow_scope_clause_v1"),
                source_snapshot_sha=self.snapshot.runtime_bundle_sha,
                task_context_sha="",
                validation_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 - outer Slow must not crash the course
            if _is_transport_failure(exc):
                raise RunFault("TRANSPORT_FAILED: %s: %s"
                               % (type(exc).__name__, str(exc)[:240]))
            if type(exc).__name__ == "AgentCallBudgetExceeded" or isinstance(
                    exc, _OuterLlmBudgetSpent):
                self.calls.append({"candidate": candidate.get("kind"),
                                   "outcome": "OUTER_LLM_BUDGET_SPENT"})
                return {"outcome": "OUTER_LLM_BUDGET_SPENT"}
            raise
        finally:
            if not metered:
                self.guard.spend(kind="outer")
        payload = dict(stage.payload or {}) if stage is not None else {}
        self.calls.append({
            "candidate": candidate.get("kind"),
            "no_proposal_reason": getattr(stage, "no_proposal_reason", None),
            "returned": payload.get("scope_clause"),
        })
        return payload or None


class Arm:
    """One arm's state across one ordering, and the reset rule that defines it.

    The frozen arms are not implemented as "the online arm with write-back
    switched off".  They get a **fresh store and a fresh method rebuilt from the
    start snapshot at every unit**, because a flag can be forgotten in one code
    path and a rebuild cannot: after the rebuild there is physically nothing to
    carry.  The cost is one snapshot compile per unit, which is milliseconds.

    The Draft ledger is part of what a frozen arm must not carry.  A restricted
    Draft is opened after a unit's delayed gate refuses, and from the next unit
    on it is candidate supply (``resupplied_programs``) and a scope source for
    the Fast Path -- which is memory across units in everything but name.  The
    first offline course showed exactly that leak: the frozen arm's store was
    rebuilt every unit while ``resupplied_draft_1`` from unit 1896 kept being
    supplied to units 2136 through 2856.  So the rebuild drops the ledger too,
    and reports how many Drafts it dropped.
    """

    def __init__(self, spec: ArmSpec, *, root: Path, machinery: Mapping[str, Any],
                 start_snapshot: Any, ledgers: Ledgers, guard: "BudgetGuard",
                 backend_factory: Any, outer_slow_factory: Any | None,
                 offline: bool) -> None:
        self.spec = spec
        self.root = Path(root)
        self.m = machinery
        self.start_snapshot = start_snapshot
        self.ledgers = ledgers
        self.guard = guard
        self.backend_factory = backend_factory
        self.outer_slow_factory = outer_slow_factory
        self.offline = bool(offline)
        self.draft_ledger = drafts.DraftLedger()
        self.bank: list[dict[str, Any]] = []
        self.processed: list[UnitContext] = []
        self.outer_steps: list[dict[str, Any]] = []
        #: This arm's own replay spend.  The course-wide ledger still counts
        #: every replay fit, but the allowance is per arm: two online arms
        #: drawing on one shared remainder would hand the arm that happens to
        #: run its outer step first the other arm's screening budget, and the
        #: A5-online minus A3-online contrast would then partly measure arm
        #: order.
        self.replay_fits_spent = 0
        self.active_skill_ids: list[str] = []
        #: program signature -> skill id, for every Skill this arm holds as
        #: Active: K0's cards (from the receipt) plus cards activated in this
        #: course.  ``deployed_via`` reads it so that a unit that re-proposes
        #: an already-Active program from scratch is not booked as a search.
        self.active_program_signatures: dict[str, str] = {}
        #: The same cards under the **full** census key -- Task x Consumer x
        #: typed Program x root Scope (sol v1.1 A).  Kept beside the signature
        #: map rather than replacing it because the two answer different
        #: questions: "is this program already deployed" (attribution) and "does
        #: this key already have a lineage" (deduplication).  The same program
        #: under a different root Scope is a different lineage.
        self.active_lineage_keys: set[str] = set()
        #: One cache per arm.  Sharing it would let one arm's fits pay for
        #: another's reading, and the arms are the contrast.
        self.replay_cache = ReplayPredictionCache(spec.name)
        #: Skill ids the arm's snapshot held when it was last (re)built, so a
        #: newly minted card is found by difference and K0's cards are never
        #: booked as minted in this course.
        self.known_skill_ids: set[str] = set()
        self.unit_index = 0
        self._method: Any = None
        self._store: Any = None
        self._controller: Any = None
        self._backend: Any = None
        self._core: Any = None

    # ---- lifecycle -------------------------------------------------------

    def _build(self, tag: str) -> None:
        store_root = self.root / self.spec.name / ("store_%s" % tag)
        self._store = self.m["SnapshotStore"](store_root)
        self._controller = self.m["EditController"](
            self._store, surfaces=self.m["SurfaceRegistry"](),
            router=self.m["FaultRouter"]())
        self._backend = self.backend_factory()
        # The core stamps model and base_url onto every request and validates the
        # URL shape before the first stage, so the offline path needs a
        # well-formed origin it can never reach rather than a placeholder word.
        target = (self.m["agentic"].live_transport() if not self.offline
                  else {"model": "offline-scripted",
                        "base_url": OFFLINE_BASE_URL})
        self._core = self.m["TTHAAgentCore"](
            self._backend, self.m["LocalPublicToolGateway"](
                np.zeros(8, dtype=np.float64), task_kind="forecast"),
            model=target["model"], base_url=target["base_url"])
        self._method = self.m["TTHAMethod"](
            self.m["TTHAFastAgent"](self._core), self.start_snapshot, ())
        self.known_skill_ids = set(self.snapshot_skill_ids())

    def seed_active_programs(self, signatures: Mapping[str, str],
                             lineage_keys: Sequence[str] = ()) -> None:
        """K0's program -> skill map and lineage keys, from the Phase S receipt.

        K0's keys are seeded as held so the outer loop cannot propose an ADD for
        a card the arm already starts with -- which would open a second lineage
        for one card and give it fresh revision counters.
        """
        for signature, skill_id in dict(signatures or {}).items():
            self.active_program_signatures.setdefault(str(signature),
                                                      str(skill_id))
        self.active_lineage_keys.update(str(key) for key in (lineage_keys or ()))

    def begin_unit(self, position: int) -> dict[str, Any]:
        """Build once for an online arm; rebuild every unit for a frozen one."""
        if self.spec.start == "static":
            return {"reset": False, "reason": "Static carries no Harness"}
        if self._method is None:
            self._build("online" if self.spec.write_back else "u%03d" % position)
            return {"reset": False, "reason": "first unit"}
        if self.spec.write_back:
            # A fresh backend per unit is what gives the per-unit-arm LLM cap a
            # hard stop at the transport rather than a count we check afterwards.
            self._backend = self.backend_factory()
            self._core.backend = self._backend
            return {"reset": False, "reason": "online arm carries its store"}
        dropped_drafts = len(self.draft_ledger.drafts)
        self.draft_ledger = drafts.DraftLedger()
        self._build("u%03d" % position)
        return {"reset": True,
                "reason": "frozen arm rebuilt from its start snapshot",
                "carried_skills": 0,
                "carried_drafts": 0,
                "dropped_drafts": dropped_drafts}

    def active_snapshot(self) -> Any:
        """What this arm would ask Slow with.

        A resumed course replays its cells from checkpoints and never calls
        ``begin_unit``, so ``_method`` can still be ``None`` when an outer step
        runs.  The start snapshot is then the honest answer -- the arm has
        learned nothing this process -- rather than a ``None`` that would only
        fail once Slow was actually asked.
        """
        if self._method is None:
            return self.start_snapshot
        return self._method._active_snapshot()

    def snapshot_skill_ids(self) -> list[str]:
        if self._method is None:
            return []
        return [row["skill_id"]
                for row in v1runner._skill_rows(self._method._active_snapshot())]

    def backend_calls(self) -> int:
        """Billable relay calls.  The scripted backend has none, by construction.

        ``SealedProbeBackend`` answers stages without a relay and carries no
        ``calls`` counter, so the offline path bills nothing -- which is correct
        rather than convenient.  Its stage count is reported separately as
        ``scripted_stage_calls`` so "0 LLM" is never confused with "nothing ran".
        """
        return int(getattr(self._backend, "calls", 0) or 0)

    def scripted_stage_calls(self) -> int:
        return len(list(getattr(self._backend, "requests", ()) or ()))

    # ---- the outer loop --------------------------------------------------

    def outer_step(self, k_index: int, *, replay_fit_allowance: int,
                   total_steps: int = 0) -> dict[str, Any] | None:
        if not self.spec.outer:
            return None
        # A fresh Slow core per arm × outer step: the inner cell backend is
        # capped at 5 and may already be spent, and Slow's two physical
        # requests are a different budget.  Passing self._core here is the
        # leak that killed v11live_ after unit 5.
        slow = (self.outer_slow_factory(self.guard, self.active_snapshot())
                if self.outer_slow_factory is not None else None)
        screen = replay_screen_for(self.processed, self.ledgers,
                                   self.replay_cache)
        reserved = reserve_for_future_steps(
            k_index, total_steps,
            period=int(contract.OUTER_LOOP["period_k_units"]))
        remaining = (int(replay_fit_allowance) - self.replay_fits_spent
                     - reserved)
        budget = outer_loop.OuterBudget(
            outer_llm_per_step=contract.OUTER_LLM_PER_STEP,
            # Against this arm's projected course fits, minus one screen held
            # back for each outer step still to come (sol v1.1 C).  Without the
            # reservation an early step with several candidates can spend the
            # whole allowance and every later step is budget-blocked before it
            # sees its first candidate -- which is what the shakedown's
            # allowance arithmetic would have produced.
            replay_fits_remaining=max(0, remaining),
        )
        record = outer_loop.consolidate(
            bank=self.bank, ledger=self.draft_ledger, k_index=k_index,
            slow=slow, replay=screen,
            budget=budget,
            # The lineage this arm actually holds, not an inference from which
            # bank rows happen to carry a source_skill_id: a card can hold a key
            # without any row in this window naming it (sol v1.1 A).
            held_lineage_keys=sorted(self.active_lineage_keys),
            bank_boundary={
                "arm": self.spec.name,
                "units": [ctx.unit for ctx in self.processed],
                "excludes": ["the evaluation face (+144)", "future units",
                             "other arms", "held-out"],
            },
        )
        self.replay_fits_spent += int(record.replay_fits)
        payload = record.to_dict()
        payload["arm"] = self.spec.name
        payload["replay_fits_spent_by_arm"] = self.replay_fits_spent
        payload["replay_fit_allowance_per_arm"] = int(replay_fit_allowance)
        self.outer_steps.append(payload)
        return payload


# ---------------------------------------------------------------------------
# one unit, one arm
# ---------------------------------------------------------------------------

def _bank_rows_from_round(ctx: UnitContext, result: Any) -> list[dict[str, Any]]:
    """Support-A evidence for the bank: per-series gains, features, scope.

    Probe rows only.  The delayed reading is a gate, not evidence to select on,
    and the evaluation face never enters the bank at all -- which is the property
    that keeps the curve readable on a face no arm learned from.
    """
    rows = []
    for probe in result.actual_probed_programs:
        if probe.get("kind") != "probe":
            continue
        gains = list(probe.get("per_series_gain") or ())
        if len(gains) != len(ctx.eval_uids):
            continue
        rows.append({
            "unit": ctx.unit,
            "task_consumer_key": TASK_CONSUMER_KEY,
            "candidate_id": probe.get("candidate_id"),
            "program_steps": probe.get("program_steps"),
            "serving_scope": probe.get("serving_scope"),
            "resolved_serving_series": probe.get("resolved_serving_series"),
            "source_skill_id": probe.get("source_skill_id"),
            "relation": (probe.get("admission") or {}).get("relation"),
            "admission": (probe.get("admission") or {}).get("reason"),
            "features": {uid: dict(ctx.features.get(uid) or {})
                        for uid in ctx.eval_uids},
            "per_series_gain": {uid: float(value)
                               for uid, value in zip(ctx.eval_uids, gains)},
        })
    return rows


class _VerifiableLedgerView:
    """The ledger as ``_MergedScopes`` and the preflight read it, HEC-1 style.

    Resupply is keyed on ``may_verify`` rather than ``may_revise`` (see
    ``DraftLedger.verifiable_drafts``); the root lookup is unchanged.  A view
    rather than a subclass so the v3 runner keeps its own ledger semantics.
    """

    def __init__(self, ledger: drafts.DraftLedger) -> None:
        self._ledger = ledger

    def resupplied_scopes(self) -> dict[str, dict[str, Any]]:
        return self._ledger.resupplied_scopes_for_verification()

    def root_for_scope(self, scope: Mapping[str, Any] | None):
        return self._ledger.root_for_scope(scope)


def run_unit_arm(arm: Arm, ctx: UnitContext, *, position: int,
                 ledgers: Ledgers, guard: BudgetGuard,
                 machinery: Mapping[str, Any],
                 scientific: bool = False) -> dict[str, Any]:
    """One (unit, arm) cell: probe, gate, score, and only then write back."""
    started = time.time()
    reset = arm.begin_unit(position)
    record: dict[str, Any] = {
        "unit": ctx.unit,
        "position": int(position),
        "arm": arm.spec.name,
        "served": len(ctx.eval_uids),
        "served_denominator_source": "roster eval rows",
        "reset": reset,
        "faults": [],
    }
    evaluation_origin = ctx.face_origin(EVALUATION_OFFSET)

    if arm.spec.start == "static":
        record.update({"deployed": None, "identity": True,
                       **_evaluate_face(ctx, evaluation_origin, None, None,
                                        ledgers, record),
                       "wall_seconds": round(time.time() - started, 2)})
        return record

    guard.open_cell()
    # What the arm's snapshot holds as it enters the unit.  For a frozen arm
    # this must equal its start snapshot (K0, or h0's empty library) on every
    # unit, and the instrument audit asserts exactly that rather than trusting
    # the reset flag.
    record["snapshot_skill_ids_at_start"] = arm.snapshot_skill_ids()
    record["active_program_signatures_at_start"] = dict(
        arm.active_program_signatures)
    calls_before = arm.backend_calls()
    scripted_before = arm.scripted_stage_calls()
    ledger_view = _VerifiableLedgerView(arm.draft_ledger)
    resupplied = arm.draft_ledger.resupplied_programs_for_verification()
    try:
        guard.reserve(kind="fast", where={"unit": ctx.unit, "arm": arm.spec.name})
        result = machinery["online_loop"].run_online_round(
            arm._method, ctx.executor,
            machinery["runner"]._a5_request(
                ctx.at.observation_block, ctx.at.values, ctx.origin,
                arm.spec.slug),
            ctx.at.values,
            origin=ctx.origin,
            slow_agent=None, controller=arm._controller, store=arm._store,
            card_builder=lambda _episode: {
                "pattern_id": arm.spec.slug,
                "observable_signature": {"task_kind": "forecast"}},
            round_name="hec1_%s_u%03d" % (arm.spec.slug, position),
            budget=int(contract.PER_UNIT_ARM_BUDGET["probes"]),
            # sol's confirmed default: every Slow call belongs to the outer
            # loop, so an edit is never proposed and judged on one unit.
            allow_slow=False, allow_group_slow=False,
            domain=arm.spec.slug,
            period=int(ctx.config["period"]),
            fast_features=dict(machinery["extract_public_features"](
                ctx.at.observation_block[:ctx.origin], task_kind="forecast")),
            allow_fast_skill=True,
            candidate_scopes=v3runner._MergedScopes(
                v1runner.InitializerScopes(arm._method), ledger_view),
            scope_resolver=ctx.resolve,
            scope_revision_preflight=v3runner._preflight(
                ctx.features, ctx.available, ledger_view),
            program_supply_verifier=ctx.executor,
            resupplied_programs=resupplied,
            risk_refusal_selector=distance.selector,
        )
    except UnitFault as exc:
        record["faults"].append({"kind": type(exc).__name__,
                                 "why": str(exc)[:240]})
        record.update({"deployed": None, "identity": True,
                       **_evaluate_face(ctx, evaluation_origin, None, None,
                                        ledgers, record),
                       "wall_seconds": round(time.time() - started, 2)})
        return record
    except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
        if _is_run_fault(exc, scientific=scientific):
            if _is_transport_failure(exc):
                raise RunFault("TRANSPORT_FAILED: %s: %s"
                               % (type(exc).__name__, str(exc)[:240]))
            raise RunFault("%s: %s" % (type(exc).__name__, str(exc)[:240]))
        record["faults"].append({"kind": "UnitFault",
                                 "why": "%s: %s" % (type(exc).__name__,
                                                    str(exc)[:240])})
        record.update({"deployed": None, "identity": True,
                       **_evaluate_face(ctx, evaluation_origin, None, None,
                                        ledgers, record),
                       "wall_seconds": round(time.time() - started, 2)})
        return record

    spent = arm.backend_calls() - calls_before
    guard.spend(kind="fast", calls=max(0, spent - 1))  # the reserve took one
    trace = getattr(arm._method, "last_trace", None)
    candidate_ids = list(getattr(trace, "candidate_program_steps", {}) or {})
    record["fast_decision"] = classify_fast_decision(
        {"candidates": candidate_ids}, candidate_ids,
        no_proposal_reason=getattr(trace, "memory_resolution_status", None)
        if not candidate_ids else None)
    record["probes"] = [dict(row) for row in result.actual_probed_programs]
    record["risk_refusals"] = len(result.risk_refusals)
    record["retrieved_skill_ids"] = list(
        getattr(trace, "retrieved_skill_ids", ()) or ())
    record["resupplied_candidate_ids"] = list(result._resupplied_candidate_ids)
    record["llm_calls_this_cell"] = spent
    record["scripted_stage_calls"] = arm.scripted_stage_calls() - scripted_before

    steps, scope = result._winner_steps, result._winner_serving_scope
    record["deployed"] = result.winner_program
    record["deployed_serving_scope"] = drafts._plain(scope)
    # sol final ruling §4: both the candidate's source and whether the program
    # was already in this arm's Active set when the unit began.  The two can
    # disagree -- Fast may re-propose an Active program under a fresh candidate
    # id -- and the three-way attribution needs both facts, not a guess.
    winner_id = str(result._winner_candidate_id or "")
    from_skill_card = bool(
        machinery["online_loop"].source_skill_of_candidate(winner_id))
    from_resupplied_draft = winner_id.startswith(drafts.RESUPPLY_PREFIX)
    signature = outer_loop._program_signature(steps) if steps else ""
    active_at_start = dict(record.get("active_program_signatures_at_start") or {})
    in_active_set = bool(signature) and signature in active_at_start
    record["winner_candidate_id"] = winner_id or None
    record["winner_from_skill_candidate"] = from_skill_card
    record["winner_from_resupplied_draft"] = from_resupplied_draft
    record["program_in_active_set_at_start"] = in_active_set
    record["deployed_via"] = (
        "identity" if not steps else
        "recalled_skill" if from_skill_card else
        "resupplied_draft" if from_resupplied_draft else
        "searched_active_program" if in_active_set else
        "searched_this_unit")

    # ---- the delayed face, and the one gate that decides ----
    delayed_origin = ctx.face_origin(DELAYED_OFFSET)
    if steps:
        resolved = (ctx.resolve(scope, delayed_origin) if scope
                    else frozenset(ctx.eval_uids))
        try:
            delayed = _policy_reading(ctx, delayed_origin, steps, resolved)
        except UnitFault as exc:
            record["faults"].append({"kind": type(exc).__name__,
                                     "why": str(exc)[:240]})
            delayed = None
        if delayed is None:
            # No delayed reading means no authority to activate on.  The unit is
            # recorded and nothing is promoted -- an unreadable gate is never a
            # pass.
            record["delayed"] = None
            record["gate_disagreement"] = None
            record["activated"] = False
            record.update(_evaluate_face(ctx, evaluation_origin, None, None,
                                         ledgers, record))
            record["wall_seconds"] = round(time.time() - started, 2)
            return record
        ledgers.course_fits += int(delayed["consumer_fits"])
        gate = authoritative_gate(delayed)
        # The state the authority is protecting, sampled before the lifecycle
        # runs, so a leak is detected rather than assumed absent.
        snapshot_before = sorted(arm.snapshot_skill_ids())
        active_before = _store_active_sha(arm._store)
        try:
            # The P4 gate goes in **as the authorizer**, not as an afterwards
            # check.  handle_feedback_delayed used to commit the new snapshot
            # the moment its own admission passed, and _active_snapshot() is
            # what the Fast Path retrieves from -- so a Skill the P4 gate
            # refused was still visible, retrievable and deployable on the next
            # unit.  Declining to call activate_approved only held the Store's
            # active pointer; it never held the in-memory snapshot.
            machinery["online_loop"].open_delayed(
                result, ctx.executor, delayed_origin=delayed_origin,
                store=arm._store, scope_resolver=ctx.resolve,
                delayed_authorizer=lambda _evidence: bool(gate["passes"]))
        except Exception as exc:  # noqa: BLE001 - the lifecycle event is a reading
            record["faults"].append({
                "kind": "UnitFault",
                "why": "open_delayed: %s" % str(exc)[:200]})
        # Did any protected state move while the authority was refusing?  This
        # is measured, not argued: the snapshot's Skill set and the Store's
        # active pointer, before and after.
        state = {
            "snapshot_before": snapshot_before,
            "snapshot_after": sorted(arm.snapshot_skill_ids()),
            "store_active_before": active_before,
            "store_active_after": _store_active_sha(arm._store),
        }
        state["snapshot_unchanged"] = (
            state["snapshot_before"] == state["snapshot_after"])
        state["store_active_unchanged"] = (
            state["store_active_before"] == state["store_active_after"])
        state["unchanged"] = (state["snapshot_unchanged"]
                              and state["store_active_unchanged"])
        disagreement = resolve_gate_disagreement(
            gate, result._delayed_event, state=state)
        record["delayed"] = {**_scored(delayed, delayed_origin), "gate": gate}
        record["gate_disagreement"] = disagreement
        record["authority_state"] = state

        activated = False
        if disagreement["may_activate"] and arm.spec.write_back:
            activated = bool(machinery["online_loop"].activate_approved(
                result, arm._store))
        record["activated"] = activated
        # sol final ruling §4: the authoritative gate passed but the lifecycle
        # carried no approved event to activate on (already Active, or the
        # online_loop read the face differently).  Nothing is activated and the
        # fact is recorded, so a would-be card lost this way is counted rather
        # than silently absent.
        record["lost_activation"] = bool(
            disagreement["may_activate"] and arm.spec.write_back
            and not activated and not in_active_set)
        if record["lost_activation"]:
            record["lost_activation_why"] = (
                "P4 gate passed; online_loop event was %r; not activated"
                % disagreement["online_loop_event"])
        if activated:
            new_ids = [skill_id for skill_id in arm.snapshot_skill_ids()
                       if skill_id not in arm.known_skill_ids]
            for skill_id in new_ids:
                arm.known_skill_ids.add(skill_id)
                if skill_id not in arm.active_skill_ids:
                    arm.active_skill_ids.append(skill_id)
            # The card that was just minted is the winner's program; with one
            # new id the map is exact, with more than one it is left unmapped
            # rather than guessed.
            if signature and len(new_ids) == 1:
                arm.active_program_signatures.setdefault(signature, new_ids[0])
            record["skills_minted_this_unit"] = new_ids
            # The lineage this card now holds, under the full census key.  The
            # root Scope is the predicate it was *initialised* with, not the one
            # it reached: a narrowed revision is the same lineage.
            if steps:
                key = outer_loop.census_key(
                    TASK_CONSUMER_KEY, steps,
                    record.get("deployed_root_scope") or scope)
                arm.active_lineage_keys.add(key)
                record["lineage_key"] = key
        record["h_readings"] = h_readings(
            window=delayed_origin,
            treated_prev=sorted(result._winner_resolved_series or ()),
            treated_now=sorted(resolved),
            per_series_gain=delayed["per_series_gain"],
            features_now=ctx.features,
            predicate=(scope or {}).get("predicate"))
        if not gate["passes"] and scope:
            # ``revisions=0``: the inner Slow is closed in HEC-1, so a Draft the
            # runner restricts here still carries the initialiser's predicate
            # unrevised.  v3's default of 1 counts the Support-window clause its
            # own Slow wrote, which does not exist in this protocol.
            entry = arm.draft_ledger.record_verification(
                arm.draft_ledger.by_scope(scope)
                or arm.draft_ledger.restrict(
                    program_steps=steps, root_scope=scope, current_scope=scope,
                    origin=ctx.origin,
                    delayed_reading={"delayed_origin": delayed_origin,
                                     "lines": gate["lines"]},
                    revisions=0),
                window=delayed_origin, failed_lines=gate["failed_lines"],
                per_series_gain=delayed["per_series_gain"],
                treated_prev=sorted(result._winner_resolved_series or ()),
                treated_now=sorted(resolved),
                material=contract.RISK["material"], reading=gate)
            record["restricted_state"] = entry.get("state_after")
    else:
        record["delayed"] = None
        record["gate_disagreement"] = None
        record["activated"] = False

    # ---- the evaluation face: scored, never fed back ----
    eval_resolved = (ctx.resolve(scope, evaluation_origin) if (steps and scope)
                     else (frozenset(ctx.eval_uids) if steps else None))
    record.update(_evaluate_face(ctx, evaluation_origin, steps, eval_resolved,
                                 ledgers, record))

    # ---- write back, and only now ----
    if arm.spec.write_back:
        arm.bank.extend(_bank_rows_from_round(ctx, result))
        if ctx not in arm.processed:
            arm.processed.append(ctx)
    record["bank_rows_after"] = len(arm.bank)
    record["active_skill_ids"] = list(arm.active_skill_ids)
    record["wall_seconds"] = round(time.time() - started, 2)
    return record


def _evaluate_face(ctx: UnitContext, origin: int, steps: Any, resolved: Any,
                   ledgers: Ledgers, record: dict[str, Any]) -> dict[str, Any]:
    """Score the evaluation face, or state why this unit has no curve point.

    A window with no observed truth is a property of the data and hits every arm
    on that unit identically, so it is recorded once per cell and the unit drops
    out of the curve for all arms rather than for some.
    """
    try:
        reading = _policy_reading(ctx, origin, steps, resolved)
    except FaceNotEvaluable as exc:
        record["faults"].append({"kind": "FaceNotEvaluable",
                                 "why": str(exc)[:240]})
        return {"evaluation": None,
                "evaluation_unreadable": str(exc)[:240],
                "evaluation_face_enters_bank": False}
    except UnitFault as exc:
        record["faults"].append({"kind": "UnitFault", "why": str(exc)[:240]})
        try:
            reading = _policy_reading(ctx, origin, None, None)
        except FaceNotEvaluable as inner:
            return {"evaluation": None,
                    "evaluation_unreadable": str(inner)[:240],
                    "evaluation_face_enters_bank": False}
    ledgers.course_fits += int(reading["consumer_fits"])
    return {"evaluation": _scored(reading, origin),
            "evaluation_face_enters_bank": False}


def _scored(reading: Mapping[str, Any], origin: int) -> dict[str, Any]:
    """What goes in the scoring ledger: the reading, never the raw predictions."""
    return {
        "origin": int(origin),
        "treated": int(reading["treated"]),
        "served": int(reading["served"]),
        "coverage": (round(int(reading["treated"]) / int(reading["served"]), 4)
                     if int(reading["served"]) else 0.0),
        "aggregate_gain": float(reading["aggregate_gain"]),
        "harmed_fraction": float(reading["harmed_fraction"]),
        "max_single_series_harm": float(reading["max_single_series_harm"]),
        "identity": bool(reading["identity"]),
        "mean_smase": float(reading["mean_smase"]),
        "static_mean_smase": float(reading["static_mean_smase"]),
    }


#: Exception names that are instrument or protocol failures rather than one
#: unit's bad luck.  Everything else abstains the unit and the course continues.
_RUN_FAULT_NAMES = (
    "AgentTransportError", "InfrastructureError", "BACKEND_UNAVAILABLE",
    "ReplayTapeMiss", "OSError", "MemoryError",
)

#: Relay/auth/quota failures.  In a scientific run these are RunFault -- the
#: v11fix_ course converted HTTP 403 into identity and the instrument gate
#: still passed.  The expected per-cell 5-call cap is *not* in this set.
_TRANSPORT_FAULT_NAMES = (
    "AgentTransportError", "InfrastructureError", "BACKEND_UNAVAILABLE",
    "PermissionDeniedError", "AuthenticationError", "APIConnectionError",
    "APITimeoutError", "RateLimitError",
)


def _is_transport_failure(exc: BaseException) -> bool:
    if type(exc).__name__ == "AgentCallBudgetExceeded":
        return False
    name = type(exc).__name__
    if name in _TRANSPORT_FAULT_NAMES or name == "STOP_TRANSPORT":
        return True
    text = str(exc)
    if "STOP_TRANSPORT" in text:
        return True
    if "insufficient_quota" in text or "Error code: 403" in text:
        return True
    return False


def _is_run_fault(exc: BaseException, *, scientific: bool = False) -> bool:
    if isinstance(exc, RunFault):
        return True
    name = type(exc).__name__
    if name == "AgentCallBudgetExceeded":
        return False  # a cell budget: the unit abstains, the course continues
    if scientific and _is_transport_failure(exc):
        return True
    return name in _RUN_FAULT_NAMES


# ---------------------------------------------------------------------------
# a course
# ---------------------------------------------------------------------------

def _run_root(run_label: str) -> Path:
    return PROJECT_ROOT / ".hec1_runs" / str(run_label)


def _checkpoint_key(ordering_name: str, position: int, arm: str) -> str:
    return "%s__%03d__%s" % (ordering_name, position, arm)


def _assert_checkpoint_mode(checkpoints: Path, ordering_name: str,
                            mode: str) -> None:
    """Refuse to mix readings of different kinds in one checkpoint directory.

    Three modes: ``offline`` (scripted), ``shakedown`` (relay, instrument only)
    and ``scientific`` (relay, enters the curve).  A scientific run may resume
    only checkpoints stamped ``scientific``: the v1 Forward shakedown wrote its
    cells under the label ``forward_live`` before checkpoints carried a mode at
    all, and a scientific Forward under the same label would otherwise have
    resumed them as its own.
    """
    for path in sorted(checkpoints.glob("%s__*.json" % ordering_name)):
        try:
            existing = json.loads(path.read_text(encoding="utf-8")).get("mode")
        except (OSError, ValueError):
            continue
        if existing is None and mode == "scientific":
            raise RunFault(
                "checkpoint %s carries no mode stamp (pre-v1.1 shakedown) and "
                "this run is scientific; it must not resume those cells.  Use "
                "a different --run-label." % path.name)
        if existing is not None and str(existing) != mode:
            raise RunFault(
                "checkpoint %s was written in %s mode and this run is %s; a "
                "resume would replay one kind of reading as the other.  Use a "
                "different --run-label."
                % (path.name, existing, mode))


def _heartbeat(root: Path, payload: Mapping[str, Any]) -> None:
    """A file a monitor can poll and a dead session cannot silence."""
    import os

    path = root / "heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "pid": os.getpid(),
        "updated_at": datetime.now().astimezone().isoformat(),
        **dict(payload),
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def code_state() -> dict[str, Any]:
    """HEAD and the dirtiness of the HEC-1 runner files, as git reports them.

    sol v1.1 R-B: a scientific ordering runs from one commit with the runner
    files clean, so that the bytes that produced a curve can be recovered.
    Git's own commit id is the record; nothing else is hashed.  When git is not
    available the state is ``None`` and the readout treats it as unknown.
    """
    import subprocess  # noqa: PLC0415

    def _git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=str(PROJECT_ROOT), capture_output=True,
                text=True, timeout=30, check=False).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", *contract.CODE_FREEZE["runner_files"])
    dirty = sorted(line[3:].strip() for line in (status or "").splitlines()
                   if line.strip())
    return {"code_commit": head or None,
            "runner_files_dirty": dirty,
            "runner_files_clean": (status is not None and not dirty),
            "git_available": head is not None}


def run_course(*, phase: str, ordering_name: str, units: Sequence[Mapping[str, Any]],
               run_label: str, offline: bool = False, resume: bool = False,
               limit: int | None = None,
               k0: Mapping[str, Any] | None = None,
               verdict: str | None = None,
               seal_released: bool = False,
               shakedown: bool = False) -> dict[str, Any]:
    """One phase over one unit sequence.  Refuses unless the phase is released.

    ``offline=True`` swaps the transport for the sealed probe backend and the
    outer Slow for a scripted one.  Everything else -- the method chain, the
    online loop, the gates, the ledgers, the checkpoints -- is the same code the
    live run takes, which is the only way the 0-LLM test is worth anything.

    ``shakedown=True`` marks a live run as instrument-only: it is exempt from
    the clean-runner-files requirement and the readout excludes it from the
    curve.  A live scientific run on dirty runner files is refused before any
    LLM call (sol v1.1 R-B).
    """
    started = time.time()
    launchable = contract.assert_launchable(
        phase, verdict=verdict, seal_released=seal_released)
    root = _run_root(run_label)
    state = code_state()
    scientific = not bool(offline) and not bool(shakedown)
    report: dict[str, Any] = {
        "stage": "HEC1_COURSE",
        "written_at": datetime.now().astimezone().isoformat(),
        "contract_version": contract.VERSION,
        "contract_frozen": contract.assert_frozen()["frozen"],
        "phase": phase,
        "ordering": ordering_name,
        "run_label": run_label,
        "offline": bool(offline),
        "resume": bool(resume),
        "mode": ("offline" if offline else
                 "shakedown" if shakedown else "scientific"),
        "code_state": state,
        "assert_launchable": launchable,
        "run_root": root.relative_to(PROJECT_ROOT).as_posix(),
    }
    archived = ARCHIVED_SCIENTIFIC_LABELS.get(str(run_label))
    if archived:
        report.update({
            "status": "BLOCKED",
            "verdict": "RUN_BLOCKED_NO_VERDICT",
            "run_fault": "ARCHIVED_INSTRUMENT_RUN: %s is %s; start a new "
                         "prefix from unit 0, never resume it"
                         % (run_label, archived),
            "llm_calls": 0, "consumer_fits": 0,
        })
        return report
    if not launchable["launchable"]:
        report.update({"status": "BLOCKED_ON_CONTRACT",
                       "verdict": "BLOCKED_ON_CONTRACT",
                       "why": launchable["blockers"],
                       "llm_calls": 0, "consumer_fits": 0})
        return report
    if not offline and not shakedown and not state["runner_files_clean"]:
        report.update({
            "status": "BLOCKED_ON_CODE_FREEZE",
            "verdict": "BLOCKED_ON_CODE_FREEZE",
            "why": [
                "a scientific ordering runs from one commit with clean runner "
                "files (sol v1.1 R-B); dirty or unknown: %s"
                % (state["runner_files_dirty"] or "git state unavailable"),
                "commit the HEC-1 files first, or pass --shakedown for an "
                "instrument-only run that enters no curve",
            ],
            "llm_calls": 0, "consumer_fits": 0})
        return report

    machinery = v1runner._machinery()
    machinery["admission_policy"].install_policy(bounded.BOUNDED_POLICY)
    ledgers = Ledgers()
    guard = BudgetGuard(
        ordering_cap=int(launchable["llm_cap"]),
        per_unit_arm_cap=int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]),
        ledgers=ledgers)

    k0 = dict(k0 or {})
    k0_empty = not (k0.get("active_skill_ids") or ())
    if phase == "phase_s":
        specs = (ArmSpec("A5-online", "h0", True, True, True),)
    else:
        specs = arm_specs(k0_empty=k0_empty)

    h0 = machinery["compile_snapshot"](
        PROJECT_ROOT / "methods/ttha/harness/h0", verify_lock=False)
    k0_snapshot = h0
    if not k0_empty:
        # Fail closed.  A K0 that names Active Skills but cannot be compiled
        # would otherwise fall back to h0 silently, and both A5 arms would run
        # from h0 under A5 labels: "A5 minus A3 shows no accumulation" would
        # then be an artifact of this branch and not a reading.
        store_root, sha = k0.get("store_root"), k0.get("runtime_bundle_sha")
        snapshot_dir = (PROJECT_ROOT / str(store_root) / str(sha)
                        if store_root and sha else None)
        if snapshot_dir is None or not snapshot_dir.is_dir():
            report.update({
                "status": "BLOCKED",
                "verdict": "RUN_BLOCKED_NO_VERDICT",
                "run_fault": (
                    "K0_SNAPSHOT_UNRESOLVED: the K0 receipt lists %d Active "
                    "Skill(s) but store_root=%r runtime_bundle_sha=%r does not "
                    "name a compiled snapshot; refusing to start the A5 arms "
                    "from h0 under a K0 label" % (
                        len(k0.get("active_skill_ids") or ()), store_root, sha)),
                "llm_calls": 0, "consumer_fits": 0,
            })
            return report
        k0_snapshot = machinery["compile_snapshot"](snapshot_dir,
                                                   verify_lock=False)
        k0_skill_ids = {row["skill_id"] for row in v1runner._skill_rows(
            k0_snapshot)}
        missing = sorted(set(k0.get("active_skill_ids") or ()) - k0_skill_ids)
        if missing:
            report.update({
                "status": "BLOCKED",
                "verdict": "RUN_BLOCKED_NO_VERDICT",
                "run_fault": (
                    "K0_SNAPSHOT_MISMATCH: the receipt names Active Skills the "
                    "compiled K0 snapshot does not carry: %s" % missing),
                "llm_calls": 0, "consumer_fits": 0,
            })
            return report
        report["k0_snapshot"] = {"store_root": str(store_root),
                                 "runtime_bundle_sha": str(sha),
                                 "skill_ids": sorted(k0_skill_ids)}

    def backend_factory():
        if offline:
            import run_v1_sealed_a5_a3 as sealed  # noqa: PLC0415

            return sealed.SealedProbeBackend(
                explore=True, operators=("outlier_mad", "winsorize"),
                force_pool=True)
        return machinery["agentic"]._default_backend_factory(
            int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]))

    def outer_backend_factory():
        if offline:
            return backend_factory()
        return machinery["agentic"]._default_backend_factory(
            int(contract.OUTER_LLM_PER_STEP))

    def _core_for(backend):
        target = (machinery["agentic"].live_transport() if not offline
                  else {"model": "offline-scripted",
                        "base_url": OFFLINE_BASE_URL})
        return machinery["TTHAAgentCore"](
            backend,
            machinery["LocalPublicToolGateway"](
                np.zeros(8, dtype=np.float64), task_kind="forecast"),
            model=target["model"], base_url=target["base_url"])

    def outer_slow_factory(guard_, snapshot):
        if offline:
            def scripted(*, candidate, rejected):
                guard_.reserve(kind="outer", where={"offline": True})
                guard_.spend(kind="outer", billable=False)
                # A numeric threshold on purpose: the offline course then also
                # proves the Runtime drops it and records LLM_THRESHOLD_IGNORED.
                return {"scope_clause": {"feature": "local_robust_z_peak",
                                         "op": ">=", "threshold": 4.25},
                        "rationale": "scripted: spiky series only"}
            return scripted
        inner = outer_backend_factory()
        metered = _MeteredOuterBackend(inner, guard=guard_, billable=True)
        return OuterSlowAgent(
            _core_for(metered),
            vocabulary=contract.SCOPE_CLASS["vocabulary"],
            guard=guard_, snapshot=snapshot)

    arms = [Arm(spec, root=root, machinery=machinery,
                start_snapshot=(k0_snapshot if spec.start == "k0" else h0),
                ledgers=ledgers, guard=guard, backend_factory=backend_factory,
                outer_slow_factory=(outer_slow_factory if spec.llm else None),
                offline=offline)
            for spec in specs]
    for arm in arms:
        if arm.spec.start == "k0":
            arm.seed_active_programs(k0.get("program_signatures") or {},
                                     k0.get("lineage_keys") or ())

    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    # An offline run and a live run must never share a checkpoint directory.  A
    # 1-unit offline chain test once wrote into the labels a live chain would
    # want, and ``--resume`` would then have replayed scripted readings as if
    # they were relay readings.  The mode is stamped on every checkpoint and a
    # mismatch is a RunFault, not a merge.
    mode = str(report["mode"])
    _assert_checkpoint_mode(checkpoints, ordering_name, mode)
    done: dict[str, dict[str, Any]] = {}
    if resume:
        for path in sorted(checkpoints.glob("%s__*.json" % ordering_name)):
            done[path.stem] = json.loads(path.read_text(encoding="utf-8"))

    sequence = list(units)[:limit] if limit else list(units)
    # The replay allowance is a share of the projected course fits, so it is
    # knowable on the first outer step rather than only at the end.  It is
    # granted *per online arm* from that arm's own projected fits: the share is
    # the contract's, the split is what keeps two online arms from competing
    # for one remainder in arm order.
    projected_fits_per_llm_arm = (
        FITS_PER_SCORED_FACE * SCORED_FACES_PER_CELL * len(sequence))
    projected_course_fits = (projected_fits_per_llm_arm
                             * max(1, sum(1 for spec in specs if spec.llm)))
    online_arms = max(1, sum(1 for spec in specs if spec.outer))
    # sol v1.1 ruling 2: each online arm's cap is REPLAY_FITS_SHARE (1.0) of
    # that arm's OWN projected course fits -- not a slice of a pool that
    # includes the frozen arm's fits, and never one remainder two arms draw on.
    # (The v1 Forward shakedown ran under 0.25 of all LLM arms' fits.)
    replay_fit_allowance = int(contract.REPLAY_FITS_SHARE
                               * projected_fits_per_llm_arm)
    replay_fit_allowance_total = replay_fit_allowance * online_arms
    # How many outer steps this course will run, so each step can hold back a
    # screen for the ones after it (sol v1.1 C).
    total_outer_steps = len(sequence) // int(
        contract.OUTER_LOOP["period_k_units"])
    rows: list[dict[str, Any]] = []
    run_fault: str | None = None
    for position, unit in enumerate(sequence):
        ctx: UnitContext | None = None
        for arm in arms:
            key = _checkpoint_key(ordering_name, position, arm.spec.name)
            if key in done:
                rows.append({**done[key], "resumed": True})
                continue
            if ctx is None:
                ctx = UnitContext(unit)
            try:
                row = run_unit_arm(arm, ctx, position=position, ledgers=ledgers,
                                   guard=guard, machinery=machinery,
                                   scientific=scientific)
            except RunFault as exc:
                run_fault = str(exc)[:400]
                break
            rows.append(row)
            (checkpoints / ("%s.json" % key)).write_text(
                json.dumps({**row, "mode": mode}, ensure_ascii=False, indent=2,
                           default=str) + "\n",
                encoding="utf-8")
            _heartbeat(root, {
                "phase": phase, "ordering": ordering_name,
                "position": position, "of": len(sequence),
                "arm": arm.spec.name, "llm_total": ledgers.llm_total(),
                "course_fits": ledgers.course_fits,
                "wall_seconds": round(time.time() - started, 1),
            })
        if run_fault:
            break
        if (position + 1) % int(contract.OUTER_LOOP["period_k_units"]) == 0:
            for arm in arms:
                try:
                    step = arm.outer_step(
                        k_index=(position + 1)
                        // int(contract.OUTER_LOOP["period_k_units"]),
                        replay_fit_allowance=replay_fit_allowance,
                        total_steps=total_outer_steps)
                except RunFault as exc:
                    run_fault = str(exc)[:400]
                    break
                if step is not None:
                    _heartbeat(root, {
                        "phase": phase, "ordering": ordering_name,
                        "outer_step": step["k_index"], "arm": arm.spec.name,
                        "drafts_opened": step["drafts_opened"],
                        "llm_total": ledgers.llm_total(),
                    })
            if run_fault:
                break

    # End of course: a Draft still open in an online arm never met its pattern
    # again (WAITING), or ran out of units before its last clause was read.
    # Recorded by the ledger's own reasons so the lifecycle table does not
    # report an archived Draft as a live one.  Frozen arms carry no ledger
    # across units, so there is nothing of theirs to close.
    course_end_closures = {
        arm.spec.name: arm.draft_ledger.close_unreencountered()
        for arm in arms if arm.spec.write_back and not run_fault
    }

    # The replay caches' hit/miss counts, aggregated onto the run ledger so the
    # course artifact carries them instead of a pair of zeros.
    for arm in arms:
        if not arm.spec.outer:
            continue
        cache = arm.replay_cache
        ledgers.cache_hits += cache.cache_hits
        ledgers.cache_misses += max(
            0, cache.logical_evaluations - cache.cache_hits)

    by_unit: dict[str, dict[str, Any]] = {}
    for row in rows:
        token = json.dumps(row["unit"], sort_keys=True, default=str)
        entry = by_unit.setdefault(token, {"unit": row["unit"],
                                          "position": row["position"],
                                          "arms": {}})
        entry["arms"][row["arm"]] = row.get("evaluation") or {}
    ledgers.wall_seconds = time.time() - started

    report.update({
        "status": "BLOCKED" if run_fault else "COMPLETE",
        "verdict": "RUN_BLOCKED_NO_VERDICT" if run_fault else None,
        "run_fault": run_fault,
        "arms": [spec.name for spec in specs],
        "k0": {"empty": k0_empty,
               "active_skill_ids": list(k0.get("active_skill_ids") or ())},
        "units_planned": len(sequence),
        "units_completed": len(by_unit),
        "cells": rows,
        "units": [by_unit[key] for key in sorted(by_unit)],
        "outer_steps": [step for arm in arms for step in arm.outer_steps],
        "lifecycle": {arm.spec.name: arm.draft_ledger.to_dict() for arm in arms},
        "course_end_closures": course_end_closures,
        "active_skill_ids": {arm.spec.name: list(arm.active_skill_ids)
                            for arm in arms},
        "h_readings": [row["h_readings"] for row in rows
                       if row.get("h_readings")],
        "shadow_records": [record for arm in arms for step in arm.outer_steps
                           for record in step.get("shadow_records") or ()],
        "fast_decisions": [row["fast_decision"] for row in rows
                           if row.get("fast_decision")],
        "ledgers": ledgers.to_dict(),
        "replay_cache": {arm.spec.name: arm.replay_cache.to_dict()
                         for arm in arms if arm.spec.outer},
        "future_step_reservation": {
            "rule": "one screen per outer step still to come",
            "total_outer_steps": total_outer_steps,
            "period_units": int(contract.OUTER_LOOP["period_k_units"]),
            "fits_per_cell": CACHE_FITS_PER_CELL,
            "guarantees": (
                "if a later step has a candidate, the budget does not block "
                "its first one"),
            "does_not_guarantee": "that every candidate is screened",
        },
        "replay_fit_allowance": {
            "share": contract.REPLAY_FITS_SHARE,
            "projected_course_fits": projected_course_fits,
            "projected_fits_per_llm_arm": projected_fits_per_llm_arm,
            "allowance_per_online_arm": replay_fit_allowance,
            "allowance": replay_fit_allowance_total,
            "spent": ledgers.replay_fits,
            "spent_by_arm": {arm.spec.name: arm.replay_fits_spent
                             for arm in arms if arm.spec.outer},
            "within": (ledgers.replay_fits <= replay_fit_allowance_total
                       and all(arm.replay_fits_spent <= replay_fit_allowance
                               for arm in arms if arm.spec.outer)),
            "cost_note": (
                "one screen re-scores every processed cell at %d fits each, so "
                "its cost grows with the course (15k fits at outer step k); the "
                "cap is %.0f%% of the online arm's own projected course fits, "
                "per arm (sol v1.1 ruling 2); replay fits are their own ledger "
                "line and are never folded into the frozen arm's cost"
                % (FITS_PER_SCORED_FACE, 100 * contract.REPLAY_FITS_SHARE)),
            "record": dict(contract.REPLAY_SHARE_RECORD),
        },
        "budget_guard": guard.to_dict(),
        "transport_failed": bool(
            run_fault and str(run_fault).startswith("TRANSPORT_FAILED")),
        "boundary": {"held_out_reads": 0, "thresholds_changed": 0,
                     "evaluation_face_in_bank": 0},
    })
    return report


PHASE_S_ARM = "A5-online"


#: The chain sol authorised: three orderings, then one frozen readout, then stop
#: before Phase F.  Written as data so the driver cannot be talked into a fourth.
CHAIN = (
    ("phase_t_forward", "forward"),
    ("phase_t_reverse", "reverse"),
    ("phase_t_interleaved", "interleaved"),
)


def run_chain(*, run_label_prefix: str = "", k0: Mapping[str, Any] | None = None,
              resume: bool = True, offline: bool = False,
              limit: int | None = None, shakedown: bool = False
              ) -> dict[str, Any]:
    """Forward, then Reverse, then Interleaved -- gated on instruments only.

    The gate between orderings is ``audit_hec1_instrument``: mechanical
    assertions over counts, ledgers and set intersections.  It is imported here
    rather than reimplemented, and it is handed the finished course artifact, so
    the thing that decides whether to continue **cannot see the curve**.  That is
    the whole point of sol's release rule: instrument completeness only, never
    the effect's sign.

    Stops on the first ordering whose audit fails, and stops before Phase F
    unconditionally -- the readout is run once by ``audit_hec1_readout`` and the
    seal is a human's.
    """
    from evaluation.main_protocol_p4 import (  # noqa: PLC0415
        audit_hec1_instrument as instrument,
    )

    started = time.time()
    steps: list[dict[str, Any]] = []
    for phase, ordering_name in CHAIN:
        label = "%s%s_live" % (run_label_prefix, ordering_name)
        report = run_course(
            phase=phase, ordering_name=ordering_name,
            units=contract.ordering(ordering_name), run_label=label,
            offline=offline, resume=resume, limit=limit, k0=k0,
            shakedown=shakedown)
        out = ARTIFACTS / ("hec1_course_%s.json" % label)
        if out.exists():
            # Never silently drop a finished course because a file of that
            # name exists: a resumed course is written beside the original.
            out = out.with_suffix(".resumed.json")
            if out.exists():
                out = out.with_name("%s.%s.json" % (
                    out.stem, datetime.now().strftime("%Y%m%dT%H%M%S")))
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2,
                       default=str) + "\n", encoding="utf-8")
        verdict = instrument.audit(report)
        steps.append({
            "phase": phase,
            "ordering": ordering_name,
            "run_label": label,
            "course_status": report.get("status"),
            "units_completed": report.get("units_completed"),
            "llm_total": (report.get("ledgers") or {}).get("llm_total"),
            "instrument_passed": verdict["passed"],
            "instrument_checks": {row["check"]: row.get("passed")
                                  for row in verdict["checks"]},
            "may_continue": verdict["may_continue"],
        })
        if not verdict["may_continue"]:
            return {
                "stage": "HEC1_CHAIN",
                "written_at": datetime.now().astimezone().isoformat(),
                "steps": steps,
                "status": "STOPPED_ON_INSTRUMENT",
                "stopped_after": ordering_name,
                "next_action": (
                    "repair the instrument and resume from the checkpoint; "
                    "never re-throw the science"
                ),
                "wall_seconds": round(time.time() - started, 1),
            }
    return {
        "stage": "HEC1_CHAIN",
        "written_at": datetime.now().astimezone().isoformat(),
        "steps": steps,
        "status": "ALL_ORDERINGS_COMPLETE",
        "next_action": (
            "run audit_hec1_readout once, then stop; Phase F needs a supported "
            "verdict and a human seal release"
        ),
        "phase_f": "not opened, and not openable from here",
        "wall_seconds": round(time.time() - started, 1),
    }


def phase_s_k0(report: Mapping[str, Any]) -> dict[str, Any]:
    """What Phase S hands Phase T.  An empty K0 is a legal freeze.

    A non-empty K0 is a *snapshot*, not a list of names: ``run_course`` compiles
    the A5 arms' start state from ``store_root / runtime_bundle_sha``, so the
    receipt must carry both or Phase T cannot start the A5 arms from K0.  The
    first version of this receipt carried only the ids, and ``run_course`` then
    fell back to h0 -- the receipt now names the snapshot and the runner refuses
    a K0 it cannot compile.
    """
    active = sorted({
        skill_id
        for ids in (report.get("active_skill_ids") or {}).values()
        for skill_id in ids})
    # program signature -> skill id for every card minted in Phase S, read off
    # the cells that activated: the winner's program is the card's program.
    program_signatures: dict[str, str] = {}
    lineage_keys: set[str] = set()
    for cell in report.get("cells") or ():
        minted = list(cell.get("skills_minted_this_unit") or ())
        steps = cell.get("deployed")
        if cell.get("activated") and len(minted) == 1 and steps:
            program_signatures.setdefault(
                outer_loop._program_signature(steps), str(minted[0]))
        # The full census key of every card K0 carries, so Phase T can seed it
        # as held and the outer loop cannot open a second lineage for a card the
        # A5 arms already start with (sol v1.1 A).
        if cell.get("activated") and steps:
            key = cell.get("lineage_key") or outer_loop.census_key(
                TASK_CONSUMER_KEY, steps,
                cell.get("deployed_root_scope")
                or cell.get("deployed_serving_scope"))
            lineage_keys.add(str(key))
    run_root = PROJECT_ROOT / str(report.get("run_root") or "")
    store_root = run_root / PHASE_S_ARM / "store_online"
    active_pointer = store_root.parent / "active.json"
    sha: str | None = None
    if active and active_pointer.is_file():
        sha = str(json.loads(active_pointer.read_text(
            encoding="utf-8")).get("runtime_bundle_sha") or "") or None
    receipt = {
        "active_skill_ids": active,
        "empty": not active,
        "why_empty_is_legal": (
            "recorded as A5_TREATMENT_EMPTY; the arm set contracts and criterion "
            "3 is not scored, and nothing is re-run to manufacture a treatment"
        ),
        "run_label": report.get("run_label"),
        "units_completed": report.get("units_completed"),
        "store_root": (store_root.relative_to(PROJECT_ROOT).as_posix()
                       if active and sha else None),
        "runtime_bundle_sha": sha if active else None,
        "snapshot_resolved": bool(
            active and sha and (store_root / str(sha)).is_dir()),
        "program_signatures": program_signatures,
        "lineage_keys": sorted(lineage_keys),
        "unmapped_active_skill_ids": sorted(
            set(active) - set(program_signatures.values())),
    }
    if active and not receipt["snapshot_resolved"]:
        receipt["k0_fault"] = (
            "K0 names Active Skills but no compiled snapshot was found under "
            "%s; Phase T will refuse this receipt" % store_root)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                       help="the 0-LLM delivery gate: seven checks")
    parser.add_argument("--phase", default="phase_t_forward",
                       choices=list(contract.PHASE_RELEASE_FIELD))
    parser.add_argument("--ordering", choices=list(contract.ORDERINGS),
                       default="forward")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--offline", action="store_true",
                       help="0 LLM: sealed probe backend and scripted outer Slow")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--units", type=int, default=None,
                       help="cap the sequence; for tests only")
    parser.add_argument("--k0", default=None,
                       help="path to the Phase S K0 receipt")
    parser.add_argument("--chain", action="store_true",
                       help="Forward, Reverse, Interleaved, gated on the "
                            "mechanical instrument checks between them")
    parser.add_argument("--run-label-prefix", default="",
                       help="prefix for the chain's labels (e.g. v11_), so a "
                            "scientific chain never shares a label, a run root "
                            "or a checkpoint directory with the shakedown")
    parser.add_argument("--shakedown", action="store_true",
                       help="instrument-only live run: exempt from the clean-"
                            "runner-files requirement and excluded from the curve")
    args = parser.parse_args(argv)

    if args.smoke:
        report = smoke()
        SMOKE_OUT.parent.mkdir(parents=True, exist_ok=True)
        SMOKE_OUT.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        for row in report["checks"]:
            print("  %-34s %s%s" % (
                row["check"], "PASS" if row.get("passed") else "FAIL",
                "" if row.get("passed") else "  <- %s" % (
                    row.get("error") or row.get("mechanical_drift") or "")))
        print("checks          : %d/%d" % (report["checks_passed"],
                                          report["checks_total"]))
        print("llm calls       : %d" % report["boundary"]["llm_calls"])
        print("consumer fits   : %d" % report["boundary"]["consumer_fits"])
        print("wrote %s" % SMOKE_OUT.relative_to(PROJECT_ROOT).as_posix())
        return 0 if report["passed"] else 1

    k0_for_chain = (json.loads(Path(args.k0).read_text(encoding="utf-8"))
                    if args.k0 else None)
    if args.chain:
        report = run_chain(run_label_prefix=args.run_label_prefix,
                          k0=k0_for_chain, resume=args.resume,
                          offline=args.offline, limit=args.units,
                          shakedown=args.shakedown)
        out = ARTIFACTS / "hec1_chain.json"
        if out.exists():
            out = out.with_name("hec1_chain_%s.json" % datetime.now().strftime(
                "%Y%m%dT%H%M%S"))
        out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        for step in report["steps"]:
            print("  %-12s units=%-3s llm=%-4s instruments=%s" % (
                step["ordering"], step["units_completed"], step["llm_total"],
                "PASS" if step["instrument_passed"] else "FAIL"))
        print("status      : %s" % report["status"])
        print("next action : %s" % report["next_action"])
        print("wrote %s" % out.relative_to(PROJECT_ROOT).as_posix())
        return 0 if report["status"] == "ALL_ORDERINGS_COMPLETE" else 1

    units = (contract.phase_s_units() if args.phase == "phase_s"
             else contract.ordering(args.ordering))
    ordering_name = "phase_s" if args.phase == "phase_s" else args.ordering
    # The default live label is exactly what audit_hec1_readout whitelists
    # (``forward_live`` ...).  The earlier ``hec1_<ordering>_live`` default
    # produced ``hec1_course_hec1_forward_live.json``, which the readout would
    # have rejected as "not one of the three live ordering labels" -- a live
    # ordering run to completion and then invisible to its own readout.
    run_label = args.run_label or ("%s_%s" % (
        ordering_name, "offline" if args.offline else "live"))
    k0 = (json.loads(Path(args.k0).read_text(encoding="utf-8"))
          if args.k0 else None)

    out = ARTIFACTS / ("hec1_course_%s.json" % run_label)
    if args.phase == "phase_s":
        out = ARTIFACTS / ("hec1_phase_s_%s.json" % run_label)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.resume:
        # Checked before the run, not after: a fresh run under a used label
        # would otherwise spend its budget and then overwrite the old artifact.
        print("refusing to overwrite %s; use a new --run-label (or --resume)"
              % out.relative_to(PROJECT_ROOT).as_posix())
        return 2
    if out.exists():
        out = out.with_suffix(".resumed.json")

    report = run_course(
        phase=args.phase, ordering_name=ordering_name, units=units,
        run_label=run_label, offline=args.offline, resume=args.resume,
        limit=args.units, k0=k0, shakedown=args.shakedown)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    print("status          : %s" % report.get("status"))
    for blocker in report.get("why") or ():
        print("  - %s" % blocker)
    if report.get("status") in ("BLOCKED_ON_CONTRACT", "BLOCKED_ON_CODE_FREEZE"):
        return 2
    print("mode            : %s  (code %s, runner files %s)" % (
        report.get("mode"),
        (report.get("code_state") or {}).get("code_commit"),
        "clean" if (report.get("code_state") or {}).get("runner_files_clean")
        else "DIRTY"))
    print("units completed : %s / %s" % (report.get("units_completed"),
                                        report.get("units_planned")))
    print("llm calls       : %s" % (report.get("ledgers") or {}).get("llm_total"))
    print("course fits     : %s" % (report.get("ledgers") or {}).get(
        "course_fits"))
    if args.phase == "phase_s":
        receipt = phase_s_k0(report)
        k0_path = ARTIFACTS / ("hec1_k0_%s.json" % run_label)
        if k0_path.exists() and not args.resume:
            k0_path = k0_path.with_suffix(".rerun.json")
        k0_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8")
        print("K0 empty        : %s" % receipt["empty"])
        print("wrote %s" % k0_path.relative_to(PROJECT_ROOT).as_posix())
    print("wrote %s" % out.relative_to(PROJECT_ROOT).as_posix())
    return 0 if report.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
