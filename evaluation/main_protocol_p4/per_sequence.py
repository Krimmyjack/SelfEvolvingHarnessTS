"""The decision unit is one sequence and its own current legal window.

What this module changes, and why it is not a refactor
------------------------------------------------------
Every reading the main protocol has taken so far was produced like this: one
``PreparationRequest`` per unit, built from ``cell.observation_block`` -- which
is ``values[support_a[0]]``, the block's **first** eval series -- and one
compiled program broadcast to the whole served cohort.  The Fast Path saw one
representative sequence and answered for twenty.  ``run_dev_auto1_skill_
revision.run_cell`` is that shape, and DEV-AUTO-1/2/3 and DEV-KNOW-1 all
inherit it.

Here every sequence in the decision population gets

    its own UID, its own values, its own deployment-visible observed pattern,
    its own public features, its own history, its own Fast session, its own
    program, and its own Support and delayed reading.

and the executed policy for a unit is the **assignment** ``uid -> program``,
not one program with a scope predicate around it.

Why the frozen Consumer contract already carries this
-----------------------------------------------------
``scoped_serving_evaluator.scoped_evaluate`` fits two models per program -- a
raw one on the raw training rows and a program one on the *prepared* training
rows -- and neither depends on the Scope.  The Scope enters at one line,

    prediction = where(in_scope, program_prediction, raw_prediction)

and each served series' loss is computed from its own prediction row alone.
So a series' score under program P is **the same number whether that series is
treated alone or alongside nineteen others**, and a heterogeneous assignment is
scored by reading each series' own row out of its own program's entry.  That is
the property ``run_hec1.ReplayPredictionCache`` already documents and relies
on, and it is why this module can be exact rather than approximate: nothing is
re-derived, only re-selected.  The smoke executes that claim instead of
asserting it.

The one honest caveat, stated rather than buried: the **training corpus is
shared**.  A sequence choosing program P is served by a model fitted on the
train-role series prepared with P, so different sequences choosing different
programs are served by different fitted models.  That is not new modelling
introduced here -- it is the two-pipeline structure the frozen serving-side
evaluator already has, with K+1 pipelines instead of 2 -- but it does mean a
per-sequence reading is "what this preparation strategy did for this evaluated
sequence", not "the independent causal effect of that sequence processing
itself".  The train and eval rosters are disjoint, so no sequence's own
program touches another sequence's served context.

What is shared and what is not
------------------------------
Shared: the Agent, the tools, the typed operator DSL, the knowledge (Skill)
version, the Consumer and the evaluation semantics.  Not shared: the request,
the candidate pool, the local state, the program.  A sequence's decision may
not read another sequence's raw Episodes, so each sequence is built with its
own history only.

Nothing here writes knowledge.  ``allow_fast_skill`` is off and ``store`` is
None, so a per-sequence round cannot mint, patch or revoke a card: the batch
runs on one frozen knowledge version and only the batch boundary (Slow) may
change it.  A round that *would* have revoked a card records
``revocation_pending_no_store`` as evidence for that boundary instead.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import run_hec1 as runner
from SelfEvolvingHarnessTS.methods.ttha import signed_radius
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import SupportReceipt

MATERIAL = float(contract.RISK["material"])
DELAYED = int(runner.DELAYED_OFFSET)
PROBE_RECEIPTS_PER_SEQUENCE = int(contract.PER_UNIT_ARM_BUDGET["probes"])

#: The decision population, registered before the run: the first N eval series
#: in roster order.  A rule, not a selection -- no gain, feature or outcome is
#: consulted, and the same rule picks the same series in every unit and arm.
DECISION_POPULATION_SIZE = 10


def decision_uids(ctx: Any, size: int | None = None) -> list[str]:
    """The registered population, read at call time so a run-level override
    of ``DECISION_POPULATION_SIZE`` (a smaller dry run) actually applies."""
    limit = DECISION_POPULATION_SIZE if size is None else int(size)
    return [str(uid) for uid in list(ctx.eval_uids)[: int(limit)]]


def program_signature(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    return outer_loop._program_signature(
        [{"op": str(op), "params": dict(params)} for op, params in steps])


def program_label(steps: Sequence[tuple[str, Mapping[str, Any]]] | None) -> str:
    if not steps:
        return "identity"
    return " -> ".join(str(op) for op, _params in steps)


def program_label_with_params(
        steps: Sequence[tuple[str, Mapping[str, Any]]] | None) -> str:
    """The label, plus the parameters when there are any.

    ``program_label`` names operators only, which is what a reader wants in a
    table -- but two sequences can legitimately choose the same operator with
    different parameters, and DEV-SEQ-1 run2 did (u8: ``outlier_mad`` bare on
    five sequences and ``z_threshold=3.5`` on a sixth).  Keying a per-unit
    summary by the bare label collapses those two into one and understates
    what the batch actually did, so every keyed summary uses this instead.
    """
    if not steps:
        return "identity"
    parts = []
    for op, params in steps:
        params = dict(params or {})
        parts.append(str(op) if not params
                     else "%s(%s)" % (op, ", ".join(
                         "%s=%s" % (key, params[key])
                         for key in sorted(params))))
    return " -> ".join(parts)


def normalise_steps(value: Any) -> tuple[tuple[str, dict], ...]:
    """Typed steps from either shape the machinery hands back."""
    out: list[tuple[str, dict]] = []
    for item in value or ():
        if isinstance(item, Mapping):
            if not item.get("op"):
                continue
            out.append((str(item["op"]), dict(item.get("params") or {})))
        else:
            op, params = item
            out.append((str(op), dict(params or {})))
    return tuple(out)


# ---------------------------------------------------------------------------
# the request: one sequence, its own context
# ---------------------------------------------------------------------------

def sequence_request(machinery: Mapping[str, Any], ctx: Any, uid: str, *,
                     origin: int, domain: str) -> Any:
    """The frozen TaskSpec/TaskContext, rebound to one sequence.

    Built by replacing three fields of the line's own request rather than by
    constructing a new one, so the Task, the Consumer class, the metric and
    the deployment constraints are byte-identical to what every previous
    package used.  What changes is exactly the three things that make the
    decision this sequence's: the UID, the values, and the observed pattern.
    """
    values = np.asarray(ctx.at.values[str(uid)], dtype=np.float64)
    period = int(ctx.config["period"])
    observed = dict(signed_radius.window_context({str(uid): values},
                                                 int(origin), period))
    observed["bound_period"] = float(period)
    base = machinery["runner"]._a5_request(
        ctx.at.observation_block, ctx.at.values, int(origin), domain)
    return dataclasses.replace(base, series_uid=str(uid),
                               values=values[: int(origin)],
                               observed_pattern_spec=observed)


def sequence_features(machinery: Mapping[str, Any], ctx: Any, uid: str, *,
                      origin: int) -> dict[str, Any]:
    values = np.asarray(ctx.at.values[str(uid)], dtype=np.float64)
    return dict(machinery["extract_public_features"](
        values[: int(origin)], task_kind="forecast"))


# ---------------------------------------------------------------------------
# readings: one build per (arm, unit, face, program), re-selected per sequence
# ---------------------------------------------------------------------------

class UnitReadings:
    """Every reading one (arm, unit) needs, billed once per distinct program.

    Two memos, kept apart because they cost different things.  The window
    verifier is deterministic CPU and no Consumer fits, so it is memoised by
    (origin, program) and re-read freely.  The Consumer reading costs two
    physical fits per (unit, face, program) and is delegated to the arm's own
    ``ReplayPredictionCache``, the instrument the frozen line already bills
    through.  Ten sequences probing the same program pay for it once -- that is
    the physical cost, not a discount -- and each sequence still reads only its
    own row.
    """

    def __init__(self, *, cache: Any, ctx: Any, executor: Any,
                 guard: Any = None) -> None:
        self.cache = cache
        self.ctx = ctx
        self.executor = executor
        #: Called with the fits a cache miss would cost, before the spend.  It
        #: raises to refuse; a cache hit never reaches it, because a hit costs
        #: nothing and refusing it would only hide a reading already paid for.
        self.guard = guard
        self._verifications: dict[tuple[int, str], Any] = {}
        self.verifier_calls = 0

    def verification(self, steps: Sequence[tuple], origin: int) -> Any:
        key = (int(origin), program_signature(steps))
        if key not in self._verifications:
            self.verifier_calls += 1
            self._verifications[key] = self.executor.verify(tuple(steps),
                                                            int(origin))
        return self._verifications[key]

    def reading(self, steps: Sequence[tuple], origin: int,
                uids: Sequence[str]) -> tuple[dict[str, Any], int]:
        """The cache's re-masked reading, and the fits this call really spent."""
        if self.guard is not None:
            key = self.cache.key(self.ctx.unit, int(origin),
                                 runner.forecast_p4._config(int(origin)),
                                 tuple(steps) or ())
            if key not in self.cache._entries:
                self.guard(runner.CACHE_FITS_PER_CELL)
        before = int(self.cache.physical_fits)
        out = self.cache.reading(self.ctx, int(origin), tuple(steps),
                                 frozenset(str(uid) for uid in uids))
        return out, int(self.cache.physical_fits) - before

    def compose(self, assignment: Mapping[str, Any], origin: int,
                population: Sequence[str]) -> tuple[dict[str, Any], int]:
        """The unit's executed policy: every sequence under its own program.

        ``assignment`` maps a UID to its typed steps, or to None/() for a
        sequence the Harness declined to treat.  A declined sequence takes the
        raw pipeline, which ``scoped_evaluate`` fits separately and which is
        bit-identical to Static -- so its 0.0 is a measured consequence of the
        path that actually ran, not a filler.
        """
        population = [str(uid) for uid in population]
        by_program: dict[str, tuple[tuple, list[str]]] = {}
        per_series: dict[str, float] = {}
        for uid in population:
            steps = assignment.get(uid)
            if not steps:
                per_series[uid] = 0.0
                continue
            key = program_signature(steps)
            slot = by_program.setdefault(key, (normalise_steps(steps), []))
            slot[1].append(uid)
        fits = 0
        unreadable: dict[str, str] = {}
        for _key, (steps, uids) in sorted(by_program.items()):
            try:
                reading, spent = self.reading(steps, origin, uids)
            except runner.UnitFault as exc:
                for uid in uids:
                    unreadable[uid] = str(exc)[:160]
                continue
            fits += spent
            for uid in uids:
                per_series[uid] = float(reading["per_series_gain"][uid])
        treated = sum(1 for uid in population
                      if assignment.get(uid) and uid not in unreadable)
        scored = [per_series[uid] for uid in population if uid in per_series]
        gains = np.asarray(scored, dtype=np.float64) if scored else np.zeros(0)
        return ({
            "identity": treated == 0,
            "treated": int(treated),
            "served": len(population),
            "scored": int(gains.size),
            "unreadable_series": unreadable,
            "per_series_gain": {uid: round(float(per_series[uid]), 6)
                                for uid in population if uid in per_series},
            "aggregate_gain": (round(float(gains.mean()), 6)
                               if gains.size else None),
            "harmed_series": int((gains < -MATERIAL).sum()) if gains.size else 0,
            "harmed_fraction": (round(float((gains < -MATERIAL).mean()), 4)
                                if gains.size else 0.0),
            "max_single_series_harm": (round(max(0.0, float(-gains.min())), 6)
                                       if gains.size else 0.0),
            "programs_used": {program_label_with_params(steps): sorted(uids)
                              for steps, uids in by_program.values()},
            "distinct_programs": len(by_program),
        }, fits)


class SequenceView:
    """The executor as one sequence sees it: every probe runs on that sequence.

    ``serving_scope`` is accepted and deliberately ignored.  In this design the
    execution boundary is not a predicate the candidate carries; it is the
    sequence the decision belongs to.  A Scope predicate still exists in the
    system and still means something -- it is the *applicability condition* on
    a knowledge card, deciding which sequences a piece of guidance is offered
    to -- but it no longer stands in for generating a program per sequence.

    ``gain`` is this sequence's own gain, not the cohort mean with one series
    treated.  That matters at the gate: with a twenty-series denominator a real
    +0.10 for one sequence arrives as +0.005 and lands on the material line by
    arithmetic rather than by evidence.  ``per_view_gain`` is the one-element
    list that goes with it, so ``classify_relation`` and the admission rule
    read a population of one -- under which strict and bounded_risk_v1 coincide
    exactly (a single series cannot be both aggregate-positive and harmed).
    """

    def __init__(self, readings: UnitReadings, uid: str) -> None:
        self.readings = readings
        self.uid = str(uid)
        self.fits_spent = 0
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, steps: Sequence[tuple], origin: int,
                 serving_scope: Any = None) -> SupportReceipt:
        verification = self.readings.verification(steps, origin)
        if not verification.passed:
            return SupportReceipt(
                origin=int(origin), verification=verification, gain=None,
                error="WINDOW_VERIFIER_REJECTED (%d windows)"
                      % len(verification.rejected_windows))
        try:
            reading, spent = self.readings.reading(steps, origin, (self.uid,))
        except runner.UnitFault as exc:
            return SupportReceipt(origin=int(origin),
                                  verification=verification, gain=None,
                                  error=str(exc)[:200])
        self.fits_spent += int(spent)
        gain = float(reading["per_series_gain"][self.uid])
        self.calls.append({"origin": int(origin),
                           "program": program_label(steps),
                           "gain": round(gain, 6), "fits": int(spent)})
        return SupportReceipt(
            origin=int(origin), verification=verification, gain=gain,
            per_view_gain=[gain],
            behavior_point_count=int(verification.cohort_modified_points),
            consumer_fits=int(spent))


__all__ = [
    "DECISION_POPULATION_SIZE",
    "DELAYED",
    "MATERIAL",
    "PROBE_RECEIPTS_PER_SEQUENCE",
    "SequenceView",
    "UnitReadings",
    "decision_uids",
    "normalise_steps",
    "program_label",
    "program_label_with_params",
    "program_signature",
    "sequence_features",
    "sequence_request",
]
