"""B: the trigger is wired -- and wiring it exposed the next link.

``RISK_REFUSAL_INVISIBLE_TO_FAULT_ROUTER`` is fixed: a materially positive
candidate refused on the tail budget now increments its own counter, carries the
material a Patch would act on, and enters the Slow Path.  Six behaviour locks
hold the old paths in place, including the one that matters most -- under the
library default policy the new branch is unreachable, because a strict refusal
reports ``relation_not_positive`` and never a budget reason.

Wiring it made the *next* fault observable, and this module proves it by
enumeration rather than asserting it in prose:

* ``SCOPE_OVERREACH`` (monotone Scope narrowing) and ``RISK_GAP`` (risk guards)
  are declared in ``fault_routes.json``, authorized ``EDITABLE_M0``, enforced
  monotone by the router, and implemented in ``skill_revision``;
* but ``route_program_supply_fault`` -- the one attribution function the online
  loop calls -- cannot return either, for **any** input.  Its five evidence
  fields are all about program *supply*: whether the DSL can express the
  program, whether a capability Skill exists, whether it was retrieved, whether
  a constrained proposal succeeds.  None of them can say "the program works and
  its serving scope is too wide";
* and both causes PATCH a **capability Skill**, while every Source-line probe
  was a fresh Fast proposal carrying no Skill at all.

So the refusal now reaches the router, and the router has no cause it can emit
for it.  Fixing that means deciding what a Scope PATCH targets when there is no
Skill to target -- a design question, not a wire -- so it stops here.

0 LLM, 0 new evaluations, 0 held-out reads; reads the sealed ``p4w`` and the
route table.
"""
from __future__ import annotations

import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from contracts.program_supply import (
    PROGRAM_SUPPLY_ROUTE_FIELDS,
    route_program_supply_fault,
)
from evaluation.main_protocol_p4 import main_experiment_contract as contract
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import FaultRouter
from SelfEvolvingHarnessTS.methods.ttha import online_loop as loop
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    _ONLINE_EXPRESSIBILITY_STATUS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "artifacts/main_protocol/p4w_source_line.json"
OUT = PROJECT_ROOT / "artifacts/main_protocol/p4z_risk_refusal_routing.json"

SCOPE_CAUSES = ("SCOPE_OVERREACH", "RISK_GAP")

#: The full input domain of the router's five evidence fields.  ``status`` and
#: ``cause`` take the tokens the function itself branches on plus a neutral
#: value; the rest are booleans and one tri-state.  Enumerating this is a proof
#: about the function, not a sample of it.
_STATUSES = ("PROVEN_UNAVAILABLE", "EXPRESSIBILITY_UNKNOWN",
             "PROVEN_EXPRESSIBLE", "OTHER")
_CAUSES = (None, "OBSERVABLE_DERIVATION_PROCEDURE_GAP",
           "OBSERVABLE_FEATURE_SCHEMA_GAP", "OTHER")


def _reachable_causes() -> dict[str, Any]:
    """Every (cause, actionability) the online attribution can ever produce."""
    reachable: dict[str, set[str]] = {}
    online_only: set[str] = set()
    combinations = 0
    for status, cause, exists, retrieved, constrained, g1 in itertools.product(
            _STATUSES, _CAUSES, (True, False), (True, False),
            (True, False, None), (True, False)):
        combinations += 1
        code, actionability, _templates = route_program_supply_fault(
            expressibility_status=status,
            expressibility_cause=cause,
            capability_skill_exists=exists,
            skill_retrieved=retrieved,
            constrained_proposal_succeeds=constrained,
            context_resolved_decision_fault=g1,
        )
        reachable.setdefault(code, set()).add(actionability)
        # What the *online* entry point can produce: it pins the status.
        if status == _ONLINE_EXPRESSIBILITY_STATUS and cause is None:
            online_only.add(code)
    return {
        "input_combinations_enumerated": combinations,
        "route_fields": list(PROGRAM_SUPPLY_ROUTE_FIELDS),
        "reachable_causes": {
            code: sorted(values) for code, values in sorted(reachable.items())},
        "reachable_from_the_online_entry_point": sorted(online_only),
        "online_expressibility_status_is_pinned_to": _ONLINE_EXPRESSIBILITY_STATUS,
        "scope_causes_reachable": sorted(
            code for code in SCOPE_CAUSES if code in reachable),
    }


def _declared_scope_surfaces() -> dict[str, Any]:
    """What the route table authorizes for the two Scope/Risk causes."""
    router = FaultRouter()
    declared = {}
    for code in SCOPE_CAUSES:
        try:
            authorization = router.allowed_targets(code)
        except KeyError:
            declared[code] = {"declared": False}
            continue
        declared[code] = {
            "declared": True,
            "actionability": authorization.actionability,
            "target_classes": list(authorization.target_classes),
            "allowed_skill_kinds": list(authorization.allowed_skill_kinds),
            "allowed_operations": list(authorization.allowed_operations),
        }
    return declared


def _source_probe_facts() -> dict[str, Any]:
    report = json.loads(SOURCE.read_text(encoding="utf-8"))
    probes, skilled, risk_refused = 0, 0, 0
    for entry in report.get("rounds", ()):
        for probe in entry.get("probes", ()):
            if probe.get("kind") != "probe":
                continue
            probes += 1
            if probe.get("source_skill_id"):
                skilled += 1
            admission = probe.get("admission") or {}
            if (not admission.get("admitted")
                    and str(admission.get("reason")) in loop.RISK_REFUSAL_REASONS):
                risk_refused += 1
    return {
        "probes": probes,
        "probes_carrying_a_source_skill_id": skilled,
        "probes_a_risk_refusal_would_now_route": risk_refused,
        "why_it_matters": (
            "both Scope/Risk causes PATCH a capability Skill's applicability or "
            "risk guards; a refused fresh proposal has no Skill, so even a "
            "correct attribution would have nothing to target"
        ),
    }


def build() -> dict[str, Any]:
    causes = _reachable_causes()
    declared = _declared_scope_surfaces()
    probes = _source_probe_facts()
    unreachable = not causes["scope_causes_reachable"]
    return {
        "stage": "P4Z_RISK_REFUSAL_ROUTING",
        "written_at": datetime.now().astimezone().isoformat(),
        "evidence_grade": "DERIVED_BY_ENUMERATION_AND_FROM_SEALED_ARTIFACTS",
        "data_version": contract.DATA_VERSION,
        "fixed": {
            "fault": "RISK_REFUSAL_INVISIBLE_TO_FAULT_ROUTER",
            "what_was_wrong": (
                "the admission gate computed the refusal reason and wrote it to "
                "the probe row; the fault router beneath it read only the "
                "aggregate gain.  A materially positive candidate refused on the "
                "tail budget matched neither branch: no winner, no Slow, and no "
                "harm_count -- a round of such candidates reported zero faults"
            ),
            "change": (
                "online_loop now recognises RISK_REFUSAL_REASONS, counts them "
                "in their own counter, records program/scope/per-series "
                "gains/reason, and admits them to the same Slow entry"
            ),
            "harm_count_semantics_preserved": True,
            "thresholds_changed": 0,
            "operators_added": 0,
            "behaviour_locks": "tests/main_protocol/test_risk_refusal_routing.py",
            "inert_under_the_default_policy": (
                "strict refuses with relation_not_positive, which is not a "
                "budget reason, so a runner that installs no policy cannot "
                "observe the new path at all"
            ),
        },
        "next_fault": {
            "fault": "NO_SCOPE_ATTRIBUTION_REACHABLE_FROM_THE_ONLINE_LOOP",
            "holds": bool(unreachable),
            "statement": (
                "the refusal now reaches the router and the router has no cause "
                "it can emit for it: no input to route_program_supply_fault "
                "returns SCOPE_OVERREACH or RISK_GAP, and the online entry point "
                "additionally pins expressibility_status so it abstains "
                "unconditionally"
            ),
            "declared_but_unreachable": declared,
            "attribution_reachability": causes,
            "source_line_probe_facts": probes,
            "why_this_stops_here": (
                "emitting a Scope cause for a Skill-less proposal means deciding "
                "what the PATCH targets when there is no Skill.  That is a "
                "design decision about what a Draft Skill means, not a wire, so "
                "it is reported rather than taken"
            ),
        },
        "boundary": {
            **contract.BOUNDARY,
            "llm_calls": 0,
            "new_evaluations": 0,
            "held_out_reads": 0,
            "artifacts_overwritten": 0,
        },
        "sources": [SOURCE.name, "fault_routes.json", "contracts/program_supply.py"],
    }


def main() -> int:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8")
    causes = report["next_fault"]["attribution_reachability"]
    print("enumerated %d input combinations over %d evidence fields" % (
        causes["input_combinations_enumerated"], len(causes["route_fields"])))
    print("causes reachable at all          : %s" % ", ".join(
        causes["reachable_causes"]))
    print("reachable from the online entry  : %s" % ", ".join(
        causes["reachable_from_the_online_entry_point"]))
    print("Scope/Risk causes reachable      : %s" % (
        causes["scope_causes_reachable"] or "NONE"))
    for code, row in report["next_fault"]["declared_but_unreachable"].items():
        print("  %-16s declared=%s skill_kinds=%s target_classes=%s" % (
            code, row.get("declared"), row.get("allowed_skill_kinds"),
            row.get("target_classes")))
    probes = report["next_fault"]["source_line_probe_facts"]
    print("source probes %d | carrying a Skill %d | now routable %d" % (
        probes["probes"], probes["probes_carrying_a_source_skill_id"],
        probes["probes_a_risk_refusal_would_now_route"]))
    print("wrote %s" % OUT.relative_to(PROJECT_ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
