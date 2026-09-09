"""DEV-AUTO-1: the relaxed revision proposer, its evaluator, and the two ways a
revision is written back onto an existing Skill.

What this module is
-------------------
The library half of one development package.  It answers the plan's 4.6 design
turn -- *propose wide, accept strictly, stop simply* -- by replacing four things
about the outer Slow call and nothing else:

1. **Role.**  ``REVISION_ROLE_INSTRUCTION`` states the outer revision role in
   its own register instead of reusing the inner-loop preparation instruction
   (which forbids inferring candidate utility, asks for an inspect stage this
   call does not have, and closes with "abstain when public evidence does not
   justify a repair").  Abstention stays legal and reachable; it is no longer
   the path of least resistance.
2. **Exit.**  A proposal may be a Scope clause **or** a Workflow program inside
   the frozen typed DSL.  The production outer loop can only narrow a Scope,
   which is the one exit M-R0k enumerated and judged negative on this course.
3. **Input.**  The whole bank, with unit and series identity, both faces, the
   series that gained and the series that lost, and -- for every earlier
   attempt -- what the tool actually measured, not a single ``REJECTED``.
4. **Budget.**  Package-level, not two physical calls per outer step.  A single
   abstention or a failed candidate does not end the course.

What it does **not** change
---------------------------
The accepting side.  The full served population stays the denominator, the four
risk lines keep their frozen numbers, the frozen bin edges are untouched, the
comparison target is the policy the rights snapshot made executable, a revision
still has to pass an independent later unit before it can be Active, and the
LLM never approves its own patch: every acceptance in this module is decided by
``scope_threshold_tool``, ``scope_narrowing_preflight``, the replay screen and
``EditController``.

Where a revision lands
----------------------
* **Workflow** -> ``EditController.apply_to_fork`` PATCHes
  ``skill_library.entries/<skill_id>.body`` under the ``OUTCOME_GAP`` route,
  which is the existing authorised path for "the deployed capability's outcome
  is bad".  That is a real compile + validate + materialize, and the Skill the
  next unit's Fast retrieves is the compiled one.
* **Scope** -> the production narrowing path: a calibrated clause on the
  lineage's ``RestrictedDraft`` via ``record_revision``, resupplied to later
  units through ``DraftLedger.resupplied_scopes``.  The card's ``serving_scope``
  has no declared edit surface and none is added here.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
from typing import Any, Callable, Mapping, Sequence

from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import scope_narrowing_preflight as preflight
from evaluation.main_protocol_p4 import scope_threshold_tool as tool
from evaluation.main_protocol_p4 import smoke_m_r0k_scope_workflow as m_r0k_smoke

MATERIAL = float(contract.RISK["material"])
MAX_HARMED = float(contract.RISK["max_harmed_fraction"])
MAX_HARM = float(contract.RISK["max_single_series_harm"])
FLOOR = int(contract.RISK["min_treated"])

#: The cause that authorises a PATCH of a capability Skill's body.  Declared in
#: ``fault_routes.json`` as ``EDITABLE_M0`` over target class ``capability``
#: with operation ``PATCH``; nothing here widens it.  ``RISK_GAP`` is
#: deliberately not used: the router pins it to ADD for this class precisely so
#: a risk fault cannot become licence to rewrite the program.
WORKFLOW_CAUSE = "OUTCOME_GAP"

#: The maximum number of proposals one call may return.  Plan 4.6 #2.
MAX_PROPOSALS_PER_CALL = 2

REVISION_ROLE_INSTRUCTION = (
    "You are the revision agent for a piece of data-preparation knowledge that "
    "is already deployed.\n"
    "A Skill card holds two things: a frozen preparation program, and a "
    "serving-scope predicate that decides which of the served series that "
    "program is applied to. Series outside the scope keep the raw pipeline and "
    "score exactly zero.\n"
    "The public evidence you are given is that Skill's own record: for every "
    "unit it has already been through, which series it helped, which it "
    "damaged, which it left unchanged, what the deployment gate said, and what "
    "every earlier revision attempt was measured to do.\n"
    "Your job is to propose concrete modifications to that Skill. A "
    "modification is either a scope clause (one deployment-visible feature and "
    "a direction, whose conjunction with the current predicate changes which "
    "series are treated) or a workflow program (a new sequence of at most two "
    "typed operators from the allowed list, replacing the frozen program).\n"
    "Propose by default. Return up to two modifications, each with a short "
    "statement of what it changes and which series you expect it to change. "
    "You do not have to prove a unique cause of failure, you do not have to "
    "rank the modifications, and you do not choose numeric thresholds: the "
    "runtime calibrates every threshold on frozen bins and ignores whatever "
    "you put there.\n"
    "You do not decide what is accepted. Every proposal is replayed "
    "deterministically against the units already processed, on the full served "
    "population, and is kept only if it beats the policy that is executable "
    "today. A modification that is refused comes back to you with the exact "
    "consequence it had, so a second attempt can differ from the first.\n"
    "Declining is legal and is recorded, but it is an answer that needs a "
    "reason from the evidence: if the record genuinely supports no "
    "modification, return the no_proposal envelope and say why."
)

#: The stage schema, passed inline.  It is deliberately **not** added to
#: ``methods/ttha/schemas``: every file there is a dependency SHA of every
#: snapshot lock, so adding one would rotate K0's runtime bundle and this
#: package would no longer be starting from the K0 the course started from.
PROPOSAL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "dev-auto1-revision-proposal/1",
    "type": "object",
    "additionalProperties": False,
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "minItems": 0,
            "maxItems": MAX_PROPOSALS_PER_CALL,
            # Two branches, each carrying the payload its own kind needs.  A
            # single permissive object let a proposal name a kind and supply no
            # content: the schema accepted it, the evaluator could only record
            # MALFORMED, and the attempt was spent measuring the schema rather
            # than the proposer.  Under ``oneOf`` the missing field is a
            # validation error, so the model is told what is missing and gets
            # its retry.
            "items": {"oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "scope_clause", "what_changes",
                                 "expected_change"],
                    "properties": {
                        "kind": {"const": "scope_clause"},
                        "scope_clause": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["feature", "op"],
                            "properties": {
                                "feature": {"type": "string"},
                                "op": {"enum": ["<=", ">="]},
                                "threshold": {"type": "number"},
                            },
                        },
                        "what_changes": {"type": "string", "minLength": 1,
                                         "maxLength": 500},
                        "expected_change": {"type": "string", "minLength": 1,
                                            "maxLength": 500},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["kind", "workflow_program", "what_changes",
                                 "expected_change"],
                    "properties": {
                        "kind": {"const": "workflow_program"},
                        "workflow_program": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 2,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["op"],
                                "properties": {
                                    "op": {"type": "string"},
                                    "params": {"type": "object"},
                                },
                            },
                        },
                        "what_changes": {"type": "string", "minLength": 1,
                                         "maxLength": 500},
                        "expected_change": {"type": "string", "minLength": 1,
                                            "maxLength": 500},
                    },
                },
            ]},
        },
        "no_modification_reason": {"type": "string", "maxLength": 500},
    },
}


# ---------------------------------------------------------------------------
# the operator space the proposer is allowed to name
# ---------------------------------------------------------------------------

def legal_operator_table() -> list[dict[str, Any]]:
    """Every registry operator this task may run, with its public parameters.

    Read off the operator registry, not restated.  Nothing is added: the frozen
    program space allows compositions of length <= 2 over exactly these.
    """
    from SelfEvolvingHarnessTS.operators import registry  # noqa: PLC0415

    rows = []
    for op, meta in sorted(registry.OPERATOR_METADATA.items()):
        if "forecast" not in tuple(meta.get("allowed_tasks") or ()):
            continue
        schema = meta.get("public_parameter_schema") or {}
        properties = dict(schema.get("properties") or {})
        rows.append({
            "op": str(op),
            "public_parameters": sorted(properties),
            "required_parameters": sorted(schema.get("required") or ()),
        })
    return rows


def _steps_of(program: Sequence[Mapping[str, Any]]
              ) -> tuple[tuple[str, dict], ...]:
    return tuple((str(step["op"]), dict(step.get("params") or {}))
                 for step in program)


def program_label(steps: Sequence[tuple[str, Mapping[str, Any]]]) -> str:
    return outer_loop._program_signature(
        [{"op": op, "params": dict(params)} for op, params in steps])


# ---------------------------------------------------------------------------
# the feedback the proposer reads
# ---------------------------------------------------------------------------

def _face_row(reading: Mapping[str, Any] | None,
              gate: Mapping[str, Any] | None) -> dict[str, Any]:
    if not reading:
        return {"readable": False}
    gains = dict(reading.get("per_series_gain") or {})
    helped = sorted(uid for uid, value in gains.items() if value > MATERIAL)
    harmed = sorted(uid for uid, value in gains.items() if value < -MATERIAL)
    unchanged = sorted(uid for uid, value in gains.items()
                       if abs(value) <= MATERIAL)
    return {
        "readable": True,
        "treated": int(reading.get("treated") or 0),
        "served": int(reading.get("served") or 0),
        "aggregate_gain_over_the_served_population": reading.get(
            "aggregate_gain"),
        "series_that_gained": helped,
        "series_that_lost": harmed,
        "series_unchanged": len(unchanged),
        "worst_single_series_harm": reading.get("max_single_series_harm"),
        "gate_passes": (bool(gate.get("passes")) if gate else None),
        "gate_failed_lines": (list(gate.get("failed_lines") or ())
                              if gate else None),
    }


def build_feedback(*, skill_id: str,
                   program_steps: Sequence[tuple[str, Mapping[str, Any]]],
                   serving_scope: Mapping[str, Any] | None,
                   history: Sequence[Mapping[str, Any]],
                   attempts: Sequence[Mapping[str, Any]],
                   vocabulary: Sequence[str]) -> dict[str, Any]:
    """The whole record, compactly, with nothing silently dropped.

    ``history`` is one row per unit already processed by this arm, each already
    carrying its Support and delayed readings.  Every row goes in: M-R0e showed
    the production prompt sent ``rows[:60]`` while the shadow search read all
    100, and no "LLM versus deterministic" reading survives that.  The number of
    rows actually sent is reported next to the number available so a future
    truncation is visible rather than silent.
    """
    units = []
    for row in history:
        units.append({
            "unit": {"block": row["unit"]["block"],
                     "origin": int(row["unit"]["origin"])},
            "position_in_the_course": int(row["position"]),
            "program_that_ran": row.get("deployed_label"),
            "serving_scope_that_ran": row.get("deployed_scope"),
            "support_face": _face_row(row.get("support_reading"), None),
            "delayed_face": _face_row(row.get("delayed_reading"),
                                      row.get("delayed_gate")),
        })
    return {
        "skill_under_revision": {
            "skill_id": skill_id,
            "frozen_program": [{"op": op, "params": dict(params)}
                               for op, params in program_steps],
            "serving_scope_predicate": (dict(serving_scope)
                                        if serving_scope else None),
        },
        "record_of_this_skill": units,
        "units_available": len(history),
        "units_sent": len(units),
        "nothing_was_truncated": len(units) == len(history),
        "earlier_attempts_and_what_they_did": [dict(row) for row in attempts],
        "what_a_scope_clause_may_name": list(vocabulary),
        "what_a_workflow_program_may_use": legal_operator_table(),
        "composition_limit": "at most two operators",
        "how_a_proposal_is_judged": {
            "denominator": ("the full served population of every unit; a "
                            "series outside the scope carries the raw "
                            "prediction and scores exactly 0"),
            "comparison_target": ("the policy that is executable today -- the "
                                  "parent version, which holds deployment "
                                  "rights"),
            "accepted_only_at": "parent aggregate gain + %g" % MATERIAL,
            "risk_lines": {"max_harmed_fraction": MAX_HARMED,
                           "max_single_series_harm": MAX_HARM,
                           "coverage_floor_treated_series": FLOOR},
            "thresholds": ("calibrated by the runtime on frozen bin edges; a "
                           "threshold you return is dropped and recorded as "
                           "dropped"),
            "after_acceptance": ("the revision holds no deployment right until "
                                 "a later independent unit passes the "
                                 "authoritative delayed gate"),
        },
    }


# ---------------------------------------------------------------------------
# the proposer
# ---------------------------------------------------------------------------

class _InstructionOverrideCore:
    """Send the revision role instead of the snapshot's own instruction.

    The snapshot is not modified and no SHA is minted: only the resolved view's
    ``instruction`` field is replaced on the way into ``run_stage``, exactly as
    M-R0f did for its one-factor contrast, and both texts are recorded.
    """

    def __init__(self, inner: Any, instruction: str) -> None:
        self.inner = inner
        self.instruction = str(instruction)
        self.seen: list[dict[str, Any]] = []

    def run_stage(self, **kwargs: Any) -> Any:
        view = kwargs.get("harness_view")
        if view is not None:
            kwargs["harness_view"] = dataclasses.replace(
                view, instruction=self.instruction)
        self.seen.append({
            "stage": kwargs.get("stage"),
            "role": str(kwargs.get("role")),
            "instruction_replaced_at_the_call_seam": True,
            "snapshot_modified": False,
        })
        return self.inner.run_stage(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class RevisionProposer:
    """One physical Slow request; up to two proposals; abstention legal."""

    def __init__(self, core: Any, *, snapshot: Any,
                 vocabulary: Sequence[str]) -> None:
        self.core = _InstructionOverrideCore(core, REVISION_ROLE_INSTRUCTION)
        self.snapshot = snapshot
        self.vocabulary = [str(name) for name in vocabulary]
        self.calls: list[dict[str, Any]] = []

    def __call__(self, feedback: Mapping[str, Any], *,
                 case_id: str) -> dict[str, Any]:
        from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: PLC0415
            AgentRole,
        )
        from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: PLC0415
            resolve_harness_view,
        )

        view = resolve_harness_view(self.snapshot, {}, role="slow")
        public_input = {
            "task": ("Propose up to two concrete modifications to the Skill "
                     "below, from its own success and failure record."),
            **dict(feedback),
        }
        try:
            stage = self.core.run_stage(
                role=AgentRole.SLOW,
                stage="edit",
                case_id=case_id,
                public_input=public_input,
                harness_view=view,
                output_schema_name="dev_auto1_revision_proposal_v1",
                output_schema=PROPOSAL_SCHEMA,
                source_snapshot_sha=self.snapshot.runtime_bundle_sha,
                task_context_sha="",
                validation_retries=1,
            )
        except Exception as exc:  # noqa: BLE001 - recorded, never hidden
            record = {"outcome": "PROPOSER_FAULT",
                      "fault": "%s: %s" % (type(exc).__name__, str(exc)[:300]),
                      "last_assistant_text": getattr(
                          exc, "last_assistant_text", None),
                      "proposals": []}
            self.calls.append(record)
            return record
        payload = dict(stage.payload or {}) if stage is not None else {}
        reason = getattr(stage, "no_proposal_reason", None)
        proposals = [dict(row) for row in (payload.get("proposals") or ())]
        record = {
            "outcome": ("ABSTAINED" if (reason or not proposals)
                        else "PROPOSED"),
            "no_proposal_reason": reason,
            "no_modification_reason": payload.get("no_modification_reason"),
            "proposals": proposals,
        }
        self.calls.append(record)
        return record


# ---------------------------------------------------------------------------
# the deterministic evaluator
# ---------------------------------------------------------------------------

def _screen_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The production screen's arithmetic, plus the served-population utility.

    Two different populations of cells, kept apart on purpose:

    * the **risk screen** runs over the cells ``outer_loop._applicable`` calls
      applicable -- a cell whose predicate resolves below the coverage floor is
      Static by construction and is not evidence for or against a candidate,
      which is the production rule and the reason a narrowing is not eliminated
      on every window where its pattern is rare;
    * the **utility** is the mean over every readable cell, because a cell where
      the policy treats nothing still delivers its (zero) gain to the served
      population, and dropping those cells would score a narrowing only where it
      acts.
    """
    usable = [c for c in cells if c.get("aggregate_gain") is not None]
    applicable = [c for c in usable
                  if outer_loop._applicable(c, min_treated=FLOOR)]
    lines: dict[str, list[Any]] = {}
    for cell in applicable:
        for name in outer_loop._violations(cell, material=MATERIAL,
                                           max_harmed=MAX_HARMED,
                                           max_harm=MAX_HARM):
            lines.setdefault(name, []).append(cell.get("unit"))
    return {
        "cells": len(cells),
        "cells_readable": len(usable),
        "cells_applicable": len(applicable),
        "unusable": [c.get("unusable") for c in cells if c.get("unusable")],
        "mean_aggregate_gain_over_the_served_population": (
            round(statistics.fmean([float(c["aggregate_gain"])
                                    for c in usable]), 6) if usable else None),
        "cells_at_or_above_the_coverage_floor": len(applicable),
        "worst_single_series_harm": (
            round(max(float(c.get("max_single_series_harm") or 0.0)
                      for c in applicable), 6) if applicable else None),
        "worst_harmed_fraction": (
            round(max(float(c.get("harmed_fraction") or 0.0)
                      for c in applicable), 6) if applicable else None),
        "production_screen_violations": {name: len(units)
                                         for name, units in lines.items()},
        "screen_rule": ("outer_loop.screen: any single applicable cell that "
                        "breaches a line eliminates the candidate; there is no "
                        "averaging across cells"),
    }


def _risk_lines(summary: Mapping[str, Any]) -> list[str]:
    """Exactly what ``outer_loop.screen`` would refuse this candidate for."""
    if not summary["cells_applicable"]:
        return ["no_applicable_cell"]
    return sorted(summary["production_screen_violations"])


def _relative_to_the_parent(summary: Mapping[str, Any],
                            parent: Mapping[str, Any]) -> dict[str, Any]:
    """How the candidate's screen compares with the **parent's own** screen.

    An independent analysis column.  It changes no rule, grants nothing and
    does not enter the verdict: acceptance stays with the production screen
    above.  It exists because the production screen is absolute, and a card
    that already holds deployment rights is not required to pass it -- so
    without this column a candidate can be eliminated by a bar the incumbent
    also fails, and the receipt would not say so.
    """
    def worse(field: str) -> bool | None:
        mine, theirs = summary.get(field), parent.get(field)
        if mine is None or theirs is None:
            return None
        return float(mine) > float(theirs) + 1e-12

    utility = summary["mean_aggregate_gain_over_the_served_population"]
    parent_utility = parent.get(
        "mean_aggregate_gain_over_the_served_population")
    beats = (utility is not None and parent_utility is not None
             and float(utility) >= float(parent_utility) + MATERIAL)
    harm_worse = worse("worst_single_series_harm")
    fraction_worse = worse("worst_harmed_fraction")
    return {
        "status": "INDEPENDENT ANALYSIS COLUMN ONLY -- no rule, threshold or "
                  "gate is changed by it and it grants nothing",
        "parent_passes_the_production_screen": not _risk_lines(parent),
        "parent_screen_violations": _risk_lines(parent),
        "candidate_worst_single_series_harm_is_worse_than_the_parents":
            harm_worse,
        "candidate_worst_harmed_fraction_is_worse_than_the_parents":
            fraction_worse,
        "candidate_beats_the_parent_utility_by_material": beats,
        "would_be_kept_under_4_5_c_with_the_parent_as_the_bar": bool(
            beats and harm_worse is False and fraction_worse is False),
    }


def evaluate_proposal(proposal: Mapping[str, Any], *,
                      parent_steps: Sequence[tuple[str, Mapping[str, Any]]],
                      parent_scope: Mapping[str, Any] | None,
                      root_scope: Mapping[str, Any] | None,
                      bank_rows: Sequence[Mapping[str, Any]],
                      replay: Callable[..., Mapping[str, Any]],
                      parent_summary: Mapping[str, Any],
                      ) -> dict[str, Any]:
    """Legality, then calibration, then a paid replay, then the comparison.

    Returns the consequence in full: what the screen measured cell by cell, how
    it compares with the parent on the served denominator, and -- when it is
    refused -- exactly which line refused it.  This dictionary is what goes
    back to the proposer, in place of the single ``REJECTED`` the production
    path returns.
    """
    kind = str(proposal.get("kind") or "")
    out: dict[str, Any] = {"kind": kind,
                           "what_changes": proposal.get("what_changes"),
                           "expected_change": proposal.get("expected_change")}

    if kind == "scope_clause":
        clause_in = dict(proposal.get("scope_clause") or {})
        if not clause_in:
            out.update({"outcome": "MALFORMED",
                        "why": "kind is scope_clause but no clause was given"})
            return out
        existing = [dict(c) for c in
                    ((parent_scope or {}).get("predicate") or ())]
        calibrated = tool.clause_from_slow({"scope_clause": clause_in},
                                           rows=bank_rows,
                                           existing_clauses=existing)
        out["calibration"] = calibrated
        if calibrated.get("outcome") != "CALIBRATED":
            outcome = str(calibrated.get("outcome"))
            why = (calibrated.get("why") if outcome == "SLOW_CLAUSE_UNUSABLE"
                   else ("no frozen bin edge of %r in direction %r clears the "
                         "four lines on the bank evidence"
                         % (clause_in.get("feature"), clause_in.get("op"))))
            out.update({
                "outcome": outcome,
                "why": why,
                "edges_tried": [row.get("threshold") for row in
                                (calibrated.get("candidates_tried") or ())],
            })
            return out
        scope = {"scope_type": "serving_series_predicate",
                 "predicate": existing + [dict(calibrated["clause"])]}
        verdict = preflight.validate_narrowing(parent_scope, scope,
                                               root=root_scope)
        accepted = bool(getattr(verdict, "accepted", False))
        out["narrowing_preflight"] = (verdict.to_dict()
                                      if hasattr(verdict, "to_dict")
                                      else str(verdict))
        if not accepted:
            out.update({"outcome": "NARROWING_REFUSED",
                        "why": "validate_narrowing refused the revised scope"})
            return out
        steps, new_scope = tuple(parent_steps), scope
    elif kind == "workflow_program":
        program = list(proposal.get("workflow_program") or ())
        if not program:
            out.update({"outcome": "MALFORMED",
                        "why": "kind is workflow_program but no program given"})
            return out
        steps = _steps_of(program)
        legality = m_r0k_smoke._legality(program_label(steps), steps)
        out["legality"] = legality
        if not legality["legal"]:
            out.update({"outcome": "ILLEGAL_PROGRAM",
                        "why": "; ".join(legality["reasons"])[:400]})
            return out
        if program_label(steps) == program_label(parent_steps):
            out.update({"outcome": "NO_CHANGE",
                        "why": "the proposed program is the parent program"})
            return out
        new_scope = dict(parent_scope) if parent_scope else None
    else:
        out.update({"outcome": "UNKNOWN_KIND", "why": "kind %r" % kind})
        return out

    out["program_replayed"] = program_label(steps)
    out["scope_replayed"] = new_scope
    out["steps_replayed"] = [{"op": op, "params": dict(params)}
                             for op, params in steps]
    screen = replay(steps=steps, scope=new_scope)
    summary = _screen_summary(list(screen.get("cells") or ()))
    out["screen"] = {"per_cell": list(screen.get("cells") or ()),
                     "summary": summary,
                     "consumer_fits_spent": int(screen.get("fits") or 0)}
    failed = _risk_lines(summary)
    parent_value = parent_summary.get(
        "mean_aggregate_gain_over_the_served_population")
    value = summary["mean_aggregate_gain_over_the_served_population"]
    out["relative_to_the_parent_screen"] = _relative_to_the_parent(
        summary, parent_summary)
    out["comparison_with_the_parent"] = {
        "parent_mean_aggregate_gain": parent_value,
        "candidate_mean_aggregate_gain": value,
        "difference": (round(float(value) - float(parent_value), 6)
                       if value is not None and parent_value is not None
                       else None),
        "material": MATERIAL,
        "denominator": "the full served population",
    }
    if value is None:
        out.update({"outcome": "NOT_READABLE",
                    "why": "no processed cell produced a reading"})
        return out
    if failed:
        out.update({"outcome": "RISK_LINE_FAILED", "failed_lines": failed,
                    "why": ("the replay over the processed units fails %s"
                            % ", ".join(failed))})
        return out
    if parent_value is not None \
            and float(value) < float(parent_value) + MATERIAL:
        out.update({
            "outcome": "NOT_BETTER_THAN_THE_PARENT",
            "why": ("on the served denominator it delivers %.6f against the "
                    "parent's %.6f, which is short of parent + %g"
                    % (value, parent_value, MATERIAL))})
        return out
    out["outcome"] = "ACCEPTED_FOR_VERIFICATION"
    out["why"] = ("it clears every risk line on the processed units and beats "
                  "the executable parent on the served denominator")
    return out


# ---------------------------------------------------------------------------
# writing the revision onto the Skill
# ---------------------------------------------------------------------------

def apply_workflow_revision(*, controller: Any, store: Any, snapshot: Any,
                            skill_id: str,
                            steps: Sequence[tuple[str, Mapping[str, Any]]],
                            edit_id: str) -> dict[str, Any]:
    """PATCH the Skill's frozen body through the real compile/validate path.

    Nothing here is hand-written into the store: ``apply_to_fork`` validates the
    manifest shape against ``slow_edit_v1``, checks the surface precondition and
    the route authorisation, forks the snapshot, applies the patch, **compiles**
    the result and materialises it.  A patch that leaves the Harness
    semantically unchanged is refused by the compiler, not by this function.
    """
    from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: PLC0415
        EditManifest,
        EditOperation,
    )
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: PLC0415
        _resolve_apply_manifest,
        verify_frozen_patch_program,
    )

    surface_id = "skill_library.entries/%s.body" % skill_id
    body = "Frozen program steps: " + json.dumps(
        [{"op": op, "params": dict(params)} for op, params in steps])
    # DEV-AUTO-2 fixed two shapes here that no run had ever reached, because
    # DEV-AUTO-1 promoted nothing.  Both are the contract's, copied from the
    # production writer in ``method.py``: the precondition is
    # ``{"kind": "SHA", "sha": ...}`` (``contracts/harness.py`` refuses
    # anything else for a PATCH), and ``minimal_patch`` is ``{"value": body}``,
    # which is what ``EditController._apply`` reads.  Building the manifest is
    # inside the guarded block now as well: a promotion that cannot be written
    # is a recorded refusal, not the end of the course.
    try:
        parent = store.materialize(snapshot)
        precondition = controller.surface_precondition_sha(parent, surface_id)
        manifest = EditManifest(
            edit_id=edit_id,
            base_harness_sha=parent.harness_content_sha,
            target_pattern_id="dev-auto1-revision",
            target_surface_id=surface_id,
            operation=EditOperation.PATCH,
            surface_precondition={"kind": "SHA", "sha": precondition},
            dependency_precondition_shas={},
            minimal_patch={"value": body},
            patch_id="dev-auto1-workflow-revision",
            predicted_agent_behavior_change=("retrieve_skill:%s" % skill_id,),
            predicted_data_effect=("local_improvement",),
            automatically_selected_risk_cases=(),
            falsification_condition=("no_improvement",),
        )
    except Exception as exc:  # noqa: BLE001
        return {"applied": False, "stage": "precondition",
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:240])}
    try:
        resolved = _resolve_apply_manifest(manifest, snapshot)
        receipt = controller.apply_to_fork(parent, resolved,
                                           confirmed_cause=WORKFLOW_CAUSE)
        # The Fast consumer's own parser, run over the compiled candidate: the
        # steps the next unit will actually retrieve have to be the steps the
        # screen replayed, or the whole chain is measuring two programs.
        verify_frozen_patch_program(receipt.candidate_snapshot.snapshot,
                                    target_surface_id=surface_id,
                                    replay_steps=tuple(steps))
    except Exception as exc:  # noqa: BLE001 - recorded, never relaxed
        return {"applied": False, "stage": "apply_to_fork",
                "surface": surface_id, "cause": WORKFLOW_CAUSE,
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
    return {
        "applied": True,
        "surface": surface_id,
        "cause": WORKFLOW_CAUSE,
        "edit_id": edit_id,
        "parent_runtime_bundle_sha": receipt.parent_runtime_bundle_sha,
        "candidate_runtime_bundle_sha": receipt.candidate_runtime_bundle_sha,
        "new_body": body,
        "compiled_and_materialized": True,
        "body_readback": "the compiled card parses to the replayed steps",
        "snapshot": receipt.candidate_snapshot.snapshot,
        "materialized": receipt.candidate_snapshot,
    }


def apply_scope_revision(*, ledger: drafts.DraftLedger,
                         draft: Any,
                         new_scope: Mapping[str, Any],
                         narrowing_preflight: Mapping[str, Any] | None,
                         support: Mapping[str, Any] | None,
                         origin: int) -> dict[str, Any]:
    """Add the calibrated clause to the lineage's Draft, the production way.

    Plan 4.6 #4 in force: a ``FLAGGED`` Draft still refuses a further clause --
    the three-state machine is untouched -- but the refusal is recorded as an
    *ordering* fact (`try a program difference first`) rather than as the end
    of the course, and the Workflow branch stays open on the same step.
    """
    if draft is None:
        return {"applied": False, "reason": "NO_SCOPE_TARGET",
                "why": "the lineage holds no open Draft"}
    if draft.state == drafts.FLAGGED:
        return {"applied": False, "reason": "NO_SCOPE_TARGET_FLAGGED",
                "why": ("the Draft is FLAGGED: the damage was dominated by "
                        "series it had already treated, so narrowing is not an "
                        "available action on this lineage"),
                "suggested_order": "try a program difference on this step"}
    if not draft.may_add_clause():
        return {"applied": False, "reason": "NO_SCOPE_TARGET_BUDGET",
                "why": ("the Draft is %s with %d revisions; it may not take a "
                        "further clause" % (draft.state, draft.revisions))}
    try:
        ledger.record_revision(draft, origin=int(origin),
                               new_scope=dict(new_scope),
                               preflight=(dict(narrowing_preflight)
                                          if narrowing_preflight else None),
                               support=(dict(support) if support else None))
    except Exception as exc:  # noqa: BLE001
        return {"applied": False, "reason": "RECORD_REVISION_REFUSED",
                "why": "%s: %s" % (type(exc).__name__, str(exc)[:240])}
    return {"applied": True, "draft_id": draft.draft_id,
            "revisions": draft.revisions,
            "current_scope": dict(draft.current_scope),
            "revision_history": [dict(row) for row in draft.revision_history],
            "resupplied_to_later_units": True}


# ---------------------------------------------------------------------------
# DEV-AUTO-1R: one comparison function
#
# DEV-AUTO-1 compared a candidate's mean over *its own* readable cells with the
# parent's mean over *the parent's* readable cells.  Where a candidate was
# refused on eight of fifteen cells that is a mean over 7 against a mean over
# 15 -- two different denominators wearing one name.  Everything below exists
# to make both sides name the same unit set, and to keep three states apart
# that DEV-AUTO-1 collapsed into two:
#
#   READ          a real reading exists; use it, over the full served
#                 population, exactly as the instrument returned it.
#   RAW_FALLBACK  the pair is confirmed not executable here and the policy's
#                 recorded legal fallback is raw.  ``run_main_baselines``
#                 already scores this case Static-equal ("a refusal returns
#                 Static-equal gains so the arm is scored on what it could
#                 actually deploy"), so the unit stays in the denominator at
#                 exactly 0.0 and treats nothing.
#   UNKNOWN       not measured, no cached entry, the read failed, or the state
#                 is undetermined.  It is **not** zero and it is **not**
#                 dropped quietly: it suppresses the complete verdict and is
#                 listed by name.
#
# Coverage that is merely low is not zero gain, and a unit no one measured is
# not a unit that delivered nothing.
# ---------------------------------------------------------------------------

READ = "READ"
RAW_FALLBACK = "RAW_FALLBACK"
UNKNOWN = "UNKNOWN"

#: The two statuses the harness itself already treats as "cannot be executed
#: here, fall back to raw".  Kept as a tuple rather than an ``in`` test on
#: substrings so a new fault class lands in UNKNOWN by default -- the safe
#: side, because UNKNOWN withholds a verdict and RAW_FALLBACK grants a number.
CONFIRMED_NOT_EXECUTABLE = ("WINDOW_VERIFIER_REJECTED",
                            "SERVING_CONTEXT_DEGENERATE")

_RAW_FALLBACK_AUTHORITY = (
    "run_main_baselines._one_pair: an illegal (program, scope) pair returns "
    "Static-equal gains 'so the arm is scored on what it could actually "
    "deploy, and the reason is recorded rather than averaged away'"
)


def classify_reading(reading: Mapping[str, Any] | None) -> dict[str, Any]:
    """One cell of one policy, in the three states above.

    Accepts either shape this line produces: a ``replay_screen_for`` cell
    (``aggregate_gain`` present, or ``unusable`` carrying the fault text) or a
    ``_reading_from_entry`` result (``status`` present).  Anything else, and
    anything missing, is UNKNOWN.
    """
    if not reading:
        return {"state": UNKNOWN, "status": "NO_READING",
                "why": "no reading of any kind was recorded for this cell",
                "aggregate_gain": None, "harmed_fraction": None,
                "max_single_series_harm": None, "treated": None,
                "served": None}

    status = reading.get("status")
    if status is None:
        # a replay cell
        if reading.get("aggregate_gain") is not None:
            status = READ
        else:
            text = str(reading.get("unusable") or "").strip()
            status = next((name for name in CONFIRMED_NOT_EXECUTABLE
                           if text.startswith(name)), None) or (
                              text.split(":")[0] or "NO_READING")

    if status == READ:
        return {"state": READ, "status": READ, "why": None,
                "aggregate_gain": float(reading["aggregate_gain"]),
                "harmed_fraction": (None
                                    if reading.get("harmed_fraction") is None
                                    else float(reading["harmed_fraction"])),
                "max_single_series_harm": (
                    None if reading.get("max_single_series_harm") is None
                    else float(reading["max_single_series_harm"])),
                "treated": (None if reading.get("treated") is None
                            else int(reading["treated"])),
                "served": (None if reading.get("served") is None
                           else int(reading["served"]))}

    if status in CONFIRMED_NOT_EXECUTABLE:
        return {"state": RAW_FALLBACK, "status": status,
                "why": str(reading.get("why") or reading.get("unusable")
                           or status)[:200],
                "scored_as": _RAW_FALLBACK_AUTHORITY,
                "aggregate_gain": 0.0, "harmed_fraction": 0.0,
                "max_single_series_harm": 0.0, "treated": 0,
                "served": (None if reading.get("served") is None
                           else int(reading["served"]))}

    return {"state": UNKNOWN, "status": str(status),
            "why": str(reading.get("why") or reading.get("unusable")
                       or status)[:200],
            "aggregate_gain": None, "harmed_fraction": None,
            "max_single_series_harm": None, "treated": None, "served": None}


def _key(unit: Mapping[str, Any] | None, position: Any = None) -> str:
    if position is not None:
        return "pos%s" % position
    unit = unit or {}
    return "%s@%s" % (unit.get("block"), unit.get("origin"))


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def compare_on_a_common_denominator(
        *, units: Sequence[Mapping[str, Any]],
        candidate: Mapping[str, Mapping[str, Any]],
        reference: Mapping[str, Mapping[str, Any]],
        face: str, candidate_label: str, reference_label: str,
        comparison_kind: str) -> dict[str, Any]:
    """Score two policies over one declared unit set, one served population.

    ``units`` is the unit set both sides are asked about -- not the set either
    side happens to be readable on.  ``candidate`` and ``reference`` map a unit
    key to a ``classify_reading`` row.  A unit either side calls UNKNOWN leaves
    the mean and is named; with any UNKNOWN present the complete verdict is
    withheld rather than approximated.

    ``comparison_kind`` is carried through untouched because it is the thing
    that must not be mixed: a replay of the parent *program* diagnoses the
    program, and it is not the arm's actually executed policy.
    """
    rows: list[dict[str, Any]] = []
    unknown_candidate: list[str] = []
    unknown_reference: list[str] = []
    for unit in units:
        key = _key(unit.get("unit"), unit.get("position"))
        mine = candidate.get(key) or classify_reading(None)
        theirs = reference.get(key) or classify_reading(None)
        if mine["state"] == UNKNOWN:
            unknown_candidate.append(key)
        if theirs["state"] == UNKNOWN:
            unknown_reference.append(key)
        determined = (mine["state"] != UNKNOWN and theirs["state"] != UNKNOWN)
        row = {
            "key": key,
            "position": unit.get("position"),
            "origin": (unit.get("unit") or {}).get("origin"),
            "both_determined": determined,
            "candidate": {k: mine[k] for k in
                          ("state", "status", "aggregate_gain",
                           "harmed_fraction", "max_single_series_harm",
                           "treated")},
            "reference": {k: theirs[k] for k in
                          ("state", "status", "aggregate_gain",
                           "harmed_fraction", "max_single_series_harm",
                           "treated")},
        }
        if determined:
            row["difference"] = round(mine["aggregate_gain"]
                                      - theirs["aggregate_gain"], 6)
            for name, field in (("single_series_harm",
                                 "max_single_series_harm"),
                                ("harmed_fraction", "harmed_fraction")):
                a, b = mine[field], theirs[field]
                row["candidate_worse_on_" + name] = (
                    None if a is None or b is None else bool(a > b + 1e-12))
        rows.append(row)

    both = [r for r in rows if r["both_determined"]]
    mine_gains = [r["candidate"]["aggregate_gain"] for r in both]
    their_gains = [r["reference"]["aggregate_gain"] for r in both]
    mine_mean, their_mean = _mean(mine_gains), _mean(their_gains)

    def worst(rows_: Sequence[Mapping[str, Any]], side: str,
              field: str) -> float | None:
        values = [r[side][field] for r in rows_ if r[side][field] is not None]
        return round(max(values), 6) if values else None

    worst_pairs = {}
    for name, field in (("single_series_harm", "max_single_series_harm"),
                        ("harmed_fraction", "harmed_fraction")):
        a = worst(both, "candidate", field)
        b = worst(both, "reference", field)
        worst_pairs[name] = {
            "candidate": a, "reference": b,
            "candidate_worse": (None if a is None or b is None
                                else bool(a > b + 1e-12)),
        }

    per_unit_worse = {
        name: [r["key"] for r in both if r.get("candidate_worse_on_" + name)]
        for name in ("single_series_harm", "harmed_fraction")}

    withheld: list[str] = []
    if unknown_candidate:
        withheld.append("%s has %d UNKNOWN cell(s): %s"
                        % (candidate_label, len(unknown_candidate),
                           ", ".join(unknown_candidate)))
    if unknown_reference:
        withheld.append("%s has %d UNKNOWN cell(s): %s"
                        % (reference_label, len(unknown_reference),
                           ", ".join(unknown_reference)))

    return {
        "comparison_kind": comparison_kind,
        "face": face,
        "candidate": candidate_label,
        "reference": reference_label,
        "denominator_rule": (
            "one declared unit set for both sides; every unit keeps the full "
            "served population; a confirmed-not-executable cell scores its raw "
            "fallback at 0.0 and stays in the denominator; an UNKNOWN cell is "
            "neither zeroed nor dropped quietly"),
        "units_declared": len(units),
        "units_scored_on_both_sides": len(both),
        "candidate_cell_states": _state_counts(rows, "candidate"),
        "reference_cell_states": _state_counts(rows, "reference"),
        "unknown_cells": {"candidate": unknown_candidate,
                          "reference": unknown_reference},
        "mean_aggregate_gain": {
            "candidate": mine_mean, "reference": their_mean,
            "difference": (None if mine_mean is None or their_mean is None
                           else round(mine_mean - their_mean, 6)),
            "material": MATERIAL,
            "candidate_beats_the_reference_by_material": (
                None if mine_mean is None or their_mean is None
                else bool(mine_mean >= their_mean + MATERIAL)),
            "counted_over": len(both),
        },
        "cross_unit_worst": worst_pairs,
        "per_unit_risk": {
            "units_where_the_candidate_is_worse": per_unit_worse,
            "cross_unit_worst_hides_a_per_unit_regression": {
                name: bool(not worst_pairs[name]["candidate_worse"]
                           and per_unit_worse[name])
                for name in per_unit_worse},
            "why_both_are_shown": (
                "a candidate can hold the cross-unit worst while being worse "
                "than the reference on some individual unit; the two are "
                "different statements and neither implies the other"),
        },
        "per_unit": rows,
        "verdict_is_complete": not withheld,
        "why_the_verdict_is_withheld": withheld,
    }


def _state_counts(rows: Sequence[Mapping[str, Any]],
                  side: str) -> dict[str, int]:
    out = {READ: 0, RAW_FALLBACK: 0, UNKNOWN: 0}
    for row in rows:
        out[row[side]["state"]] += 1
    return out


def frame_b_admits(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Frame B: reference + material on the common population, neither risk
    line worse in the cross-unit summary.  The historical absolute screen is
    not consulted; every other check (legality, Scope calibration, the same
    readings) is.

    The reference is whatever the caller declared *before* the numbers were
    computed.  This function cannot choose one, which is the point.
    """
    gain = comparison["mean_aggregate_gain"]
    worst = comparison["cross_unit_worst"]
    reasons: list[str] = []
    if not comparison["verdict_is_complete"]:
        return {"admits": None, "why": ["UNDETERMINED: "
                                        + "; ".join(comparison[
                                            "why_the_verdict_is_withheld"])]}
    if not gain["candidate_beats_the_reference_by_material"]:
        reasons.append("utility_not_material_over_the_reference")
    for name in ("single_series_harm", "harmed_fraction"):
        if worst[name]["candidate_worse"]:
            reasons.append("summary_%s_worse_than_the_reference" % name)
    return {"admits": not reasons, "why": reasons,
            "rule": ("common-population utility >= reference + material, and "
                     "neither summarised risk line worse than the reference"),
            "note_on_the_summary": (
                "this rule reads the cross-unit summary, so a per-unit "
                "regression can pass it; the per-unit list is reported beside "
                "it and is not silently folded in")}

__all__ = [
    "CONFIRMED_NOT_EXECUTABLE",
    "MAX_PROPOSALS_PER_CALL",
    "PROPOSAL_SCHEMA",
    "RAW_FALLBACK",
    "READ",
    "REVISION_ROLE_INSTRUCTION",
    "RevisionProposer",
    "UNKNOWN",
    "WORKFLOW_CAUSE",
    "apply_scope_revision",
    "apply_workflow_revision",
    "build_feedback",
    "classify_reading",
    "compare_on_a_common_denominator",
    "evaluate_proposal",
    "frame_b_admits",
    "legal_operator_table",
    "program_label",
]
