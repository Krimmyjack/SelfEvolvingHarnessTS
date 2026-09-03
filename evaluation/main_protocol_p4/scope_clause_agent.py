"""Slow writes one Scope clause; the Runtime writes the manifest around it.

Why this exists
---------------
In the v2 live run, three of the five rounds died before any Scope judgement was
reached, and none of them died on the question under test:

* origin 1896 -- ``payload has unexpected fields``: the model put
  ``observable_applicability``, ``falsification_condition`` and friends at the
  payload's top level instead of inside ``edit_manifest``;
* origin 2616 -- ``dependency_precondition_shas.observable_contract does not
  match pattern``: a 64-hex SHA the model had no way to know and had to copy;
* an earlier attempt -- ``missing fields: ['observable_applicability']``, twice,
  burning that round's whole retry budget.

Every one of those fields is Runtime-owned or Runtime-derivable.  ``edit_id``,
``base_harness_sha``, the surface id, the precondition kind, the dependency
SHAs, the frozen program body, the applicability and the patch id are all
determined before Slow is called -- and ``handle_feedback_support`` already
overwrites the body and the applicability after the fact, so asking Slow to
write them was never even load-bearing.  The single field that is on trial is
the added Scope clause.  So that is the only field Slow is asked for.

What is given up, and what is not
---------------------------------
``predicted_agent_behavior_change``, ``predicted_data_effect`` and
``falsification_condition`` were the Slow agent's own commitments, and here the
Runtime writes them.  They gate nothing in this protocol -- the gates are the
narrowing preflight, the Support re-verification and the delayed four lines --
but the loss is real and is recorded in ``RUNTIME_AUTHORED_FIELDS`` rather than
left to be discovered in a diff.

What is *not* given up: the clause itself.  The Runtime never proposes a
feature, a direction or a threshold, never repairs a clause it dislikes, and
refuses rather than substitutes when the clause is unusable.  A run in which
the Runtime picked the predicate would measure nothing at all.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from SelfEvolvingHarnessTS.contracts.harness import EditManifest, HarnessSnapshot
from SelfEvolvingHarnessTS.methods.ttha.agent_core import AgentRole
from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (
    load_stage_schema,
    validate_local_schema,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

#: Fields the Runtime now authors that the v2 Slow call authored itself.  Kept
#: as data so the contract can assert the list has not quietly grown.
RUNTIME_AUTHORED_FIELDS = (
    "edit_id",
    "base_harness_sha",
    "target_pattern_id",
    "target_surface_id",
    "operation",
    "surface_precondition",
    "dependency_precondition_shas",
    "new_value.skill_id",
    "new_value.skill_kind",
    "new_value.revision",
    "new_value.body",
    "new_value.allowed_tools",
    "new_value.risk_guards",
    "new_value.observable_applicability",
    "observable_applicability",
    "predicted_agent_behavior_change",
    "predicted_data_effect",
    "falsification_condition",
    "patch_id",
)

#: The one field Slow authors.
SLOW_AUTHORED_FIELDS = ("new_value.serving_scope.predicate[-1]",)


class ScopeClauseError(ValueError):
    """The clause Slow returned cannot be turned into a legal revision."""


def _skill_id(pattern_id: str, clauses: Sequence[Mapping[str, Any]]) -> str:
    """A deterministic id for the Draft, derived from what it actually is.

    Deterministic so a re-run of the same readings produces the same entry id,
    and derived from the clause set so two different revisions cannot collide
    on one id.
    """
    digest = hashlib.sha256(
        json.dumps([dict(c) for c in clauses], sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]
    return "scope_narrowed_%s" % digest


def _extend(scope: Mapping[str, Any],
            clause: Mapping[str, Any]) -> dict[str, Any]:
    """The original predicate with one clause appended -- never rewritten.

    Appending rather than merging is what makes the preflight's structural
    check meaningful: the revised clause set is a superset of the original by
    construction here, and the preflight then verifies it independently rather
    than trusting this function.
    """
    predicate = [dict(item) for item in (scope.get("predicate") or ())]
    predicate.append({
        "feature": str(clause["feature"]),
        "op": str(clause["op"]),
        "threshold": float(clause["threshold"]),
    })
    return {"scope_type": "serving_series_predicate", "predicate": predicate}


class ScopeClauseSlowAgent:
    """Drop-in for ``TTHASlowAgent`` on the risk-refusal path only.

    Same ``propose_edit`` signature, so ``handle_feedback_support`` needs no
    branch: it calls a slow agent and receives an ``EditManifest``.  The
    aggregate-negative path keeps the real ``TTHASlowAgent`` and is untouched.
    """

    def __init__(self, core: Any) -> None:
        self.core = core
        self.last_no_proposal_reason: str | None = None
        self.last_stage_result: Any | None = None
        #: Every clause proposed this session, whether or not it survived.
        self.proposals: list[dict[str, Any]] = []

    # ---- the parts that are the Runtime's ---------------------------------

    @staticmethod
    def _behaviour_predictions(skill_id: str) -> list[str]:
        """Legal predicates that are also true of a narrowed scoped ADD."""
        return [
            "retrieve_skill:%s" % skill_id,
            "effective_view_unchanged_out_of_scope",
        ]

    def _assemble(self, *, card: Mapping[str, Any],
                  entry: Mapping[str, Any],
                  snapshot: HarnessSnapshot,
                  clause: Mapping[str, Any]) -> tuple[EditManifest, dict]:
        refusal = dict(card.get("risk_refusal") or {})
        original = refusal.get("serving_scope")
        if not isinstance(original, Mapping) or not original:
            raise ScopeClauseError(
                "the card carries no Scope to narrow, so a clause cannot be "
                "appended to anything")
        available = [str(name)
                     for name in (card.get("deployment_visible_features") or ())]
        if available and str(clause["feature"]) not in available:
            raise ScopeClauseError(
                "the clause names %r, which is not a deployment-visible "
                "feature; the deployment could not read it at serving time"
                % str(clause["feature"]))

        revised = _extend(original, clause)
        skill_id = _skill_id(str(card.get("pattern_id") or "p"),
                             revised["predicate"])
        options = [dict(option) for option in (card.get("typed_patch_options") or ())
                   if isinstance(option, Mapping)]
        if not options or not options[0].get("patch_id"):
            raise ScopeClauseError(
                "no Runtime typed-patch option to bind the frozen program to")
        steps = [dict(step) for step in (options[0].get("program_steps") or ())]
        if not steps:
            raise ScopeClauseError("the typed-patch option carries no program")

        payload = {"edit_manifest": {
            "edit_id": "scope-narrowing-%s" % skill_id.rsplit("_", 1)[-1],
            "base_harness_sha": snapshot.harness_content_sha,
            "target_pattern_id": str(card.get("pattern_id")),
            "target_surface_id": str(entry["surface_id"]),
            "operation": "ADD",
            "surface_precondition": dict(entry["surface_precondition"]),
            "dependency_precondition_shas": dict(
                entry.get("dependency_precondition_shas") or {}),
            "new_value": {
                "schema_version": "skill-entry/1",
                "skill_id": skill_id,
                "skill_kind": "capability",
                "revision": 1,
                # Overwritten verbatim by handle_feedback_support from the
                # Runtime whitelist; written here so the manifest is already
                # legal on its own rather than legal only after repair.
                "body": "Frozen program steps: " + json.dumps(
                    [{"op": str(s.get("op")), "params": dict(s.get("params") or {})}
                     for s in steps]),
                "observable_applicability": dict(
                    card.get("observable_applicability") or {"const": True}),
                "allowed_tools": sorted({str(s.get("op")) for s in steps}),
                "risk_guards": {"requires_target_support": True},
                "serving_scope": revised,
            },
            "observable_applicability": dict(
                card.get("observable_applicability") or {"const": True}),
            "predicted_agent_behavior_change": self._behaviour_predictions(
                skill_id),
            "predicted_data_effect": [
                "series outside the revised serving scope are left bit-identical "
                "to the Static baseline"],
            "falsification_condition": [
                "at the delayed origin the revised scope still breaches the "
                "harmed-fraction or single-series harm budget"],
            "patch_id": str(options[0]["patch_id"]),
        }}
        # The skeleton is held to the very schema Slow used to have to satisfy,
        # so "the Runtime writes it" cannot become "the Runtime is exempt".
        validate_local_schema(payload, load_stage_schema("slow_edit_v1"))
        return TTHASlowAgent._manifest_from_payload(payload), payload

    # ---- the part that is Slow's ------------------------------------------

    def _ask(self, card: Mapping[str, Any], snapshot: HarnessSnapshot,
             task_context: Any | None) -> Mapping[str, Any] | None:
        refusal = dict(card.get("risk_refusal") or {})
        public_input: dict[str, Any] = {
            "failure_pattern_card": json.loads(json.dumps(card, default=str)),
            "task": (
                "A program helps on average across the served series and "
                "damages a few of them past the deployment's risk budget, so "
                "it cannot be deployed.  Its serving scope is a conjunction of "
                "clauses over deployment-visible features.  Add exactly ONE "
                "more clause so the damaged series fall outside the scope "
                "while enough helped series stay inside it."
            ),
            "hard_constraints": [
                "the feature must be one of "
                "failure_pattern_card.deployment_visible_features",
                "you add one clause; the existing clauses are kept by the "
                "runtime and cannot be edited or removed",
                "the revised scope must still select at least %d of the served "
                "series, or the revision is discarded as an abstention"
                % int(card.get("budget", {}).get("min_treated", 5) or 5),
                "no series identity is available to you and none is needed: "
                "per_series_features rows are anonymous and positional",
            ],
            "evidence_layout": (
                "failure_pattern_card.per_series_features[i] and "
                "failure_pattern_card.risk_refusal.per_series_gain[i] describe "
                "the same served series"
            ),
            "current_scope": refusal.get("serving_scope"),
            "budget": card.get("budget"),
            "output_contract": (
                "Return one scope_clause object: {feature, op, threshold}. "
                "Do not return an edit manifest; the runtime builds it."
            ),
        }
        if task_context is not None:
            public_input["task_context"] = task_context.to_dict()
            public_input["task_context_sha"] = task_context.sha()

        view = resolve_harness_view(snapshot, {}, role="slow")
        stage = self.core.run_stage(
            role=AgentRole.SLOW,
            stage="edit",
            case_id=str(card.get("pattern_id") or "pattern-unknown"),
            public_input=public_input,
            harness_view=view,
            output_schema_name="slow_scope_clause_v1",
            output_schema=self.core.load_stage_schema("slow_scope_clause_v1"),
            source_snapshot_sha=snapshot.runtime_bundle_sha,
            task_context_sha=(
                task_context.sha() if task_context is not None else ""),
            validation_retries=1,
        )
        self.last_stage_result = stage
        if stage.no_proposal_reason is not None:
            self.last_no_proposal_reason = stage.no_proposal_reason
            return None
        clause = (stage.payload or {}).get("scope_clause")
        return dict(clause) if isinstance(clause, Mapping) else None

    def propose_edit(
        self,
        card: Mapping[str, object],
        surface_catalog: Mapping[str, object] | Sequence[Mapping[str, object]],
        snapshot: HarnessSnapshot,
        *,
        manifest_preflight: Any = None,
        allowed_operator_contracts: Sequence[Mapping[str, object]] = (),
        task_context: Any | None = None,
        **_ignored: Any,
    ) -> EditManifest | None:
        self.last_no_proposal_reason = None
        self.last_stage_result = None
        entries = ([dict(surface_catalog)] if isinstance(surface_catalog, Mapping)
                   else [dict(item) for item in surface_catalog
                         if isinstance(item, Mapping)])
        if not entries:
            self.last_no_proposal_reason = "no_authorized_surface"
            return None
        entry = entries[0]
        if str(entry.get("operation")) != "ADD":
            # The route pin already refuses this upstream; refusing again here
            # means the assembler cannot become a way around it.
            self.last_no_proposal_reason = "authorized_operation_is_not_add"
            return None

        clause = self._ask(card, snapshot, task_context)
        record: dict[str, Any] = {
            "clause": dict(clause) if clause else None,
            "no_proposal_reason": self.last_no_proposal_reason,
        }
        if clause is None:
            self.proposals.append({**record, "outcome": "abstained"})
            return None
        try:
            manifest, payload = self._assemble(
                card=card, entry=entry, snapshot=snapshot, clause=clause)
        except (ScopeClauseError, KeyError, TypeError, ValueError) as exc:
            # A clause that cannot be assembled is a refusal of *that clause*,
            # not a licence to invent a different one.
            self.last_no_proposal_reason = "clause_unusable: %s" % exc
            self.proposals.append({**record, "outcome": "clause_unusable",
                                   "error": str(exc)})
            return None
        record["assembled_serving_scope"] = payload[
            "edit_manifest"]["new_value"]["serving_scope"]
        record["skill_id"] = payload["edit_manifest"]["new_value"]["skill_id"]
        record["outcome"] = "assembled"
        self.proposals.append(record)
        if manifest_preflight is not None:
            manifest_preflight(manifest)
        return manifest
