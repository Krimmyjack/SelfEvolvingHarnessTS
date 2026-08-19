"""The formal Fast Path, run once per Task Episode per arm.

Frozen design §7::

    TaskSpec + initial Context
    -> retrieve General / Specific / Experience contrast
    -> INSPECT   Agent calls bounded Workspace tools
    -> PROPOSE   Agent generates 1..B Typed Workflow candidates
    -> COMPILE   Runtime validates operators, Scope, binding provenance
    -> SUPPORT   probe under the shared Target feedback budget
    -> SELECT    execute, try the next candidate, request Observation, abstain

What changes relative to the E1 / A5A3 path this replaces: that path made one
proposal call against a Runtime-precompressed representative-series summary.
Here the three stages are the frozen ``fast_inspect`` / ``fast_propose`` /
``fast_select`` contracts driven by ``TTHAAgentCore``'s real tool loop, and
the Agent starts with *no* per-series numbers at all -- Scope membership, the
Consumer, the Metric, the budgets and the retrieved knowledge, nothing else.
Every feature it reasons from is one it chose to fetch, which is what makes
"the tool result changed the candidate" a testable statement rather than an
assumption.

Fairness (§7.1): A3 and A5 differ in retrieved knowledge only.  Tools, tool
bound, operator inventory, Support budget, Runtime, Judge and stage contracts
are constructed here once and shared by both arms.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from SelfEvolvingHarnessTS.methods.ttha.agent_core import (
    AgentProtocolError,
    AgentRole,
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
    _validate_hypothesis_references,
    _validate_inspect_hypotheses,
)
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CandidateCompilationError,
    compile_workflow_proposal,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (
    AgentCallBudgetExceeded,
    AgentTransportError,
)

from .dispatch import (
    ParameterOwnershipViolation,
    audit_program_parameter_ownership,
)

IDENTITY_CHOICE = "identity"

# Three kinds of stage fault, three doors.  Only the first is behaviour, and
# none of them may end the run -- a Task Episode is the unit that fails.
#
# _AGENT_FAULTS      the Agent's own doing: a response that never becomes a
#                    valid envelope, a payload that fails its schema or
#                    post-validator, a tool protocol violation, the tool-round
#                    limit.  Exits as AGENT_PROTOCOL_ERROR, reported, never
#                    routed to Slow.
# _INFRASTRUCTURE    the relay, not the Agent and not the Judge: a transport
#                    failure that survived its retries, or this arm-Task's LLM
#                    call ceiling.  Exits as TRANSPORT_FAILED or
#                    LLM_CALL_BUDGET_EXHAUSTED and is excluded from every
#                    behavioural readout, exactly like an unreadable
#                    instrument -- it is not an abstention and not a decision.
# instrument         the Support or delayed evaluator; handled at its own call
#                    site as INSTRUMENT_UNREADABLE.
_AGENT_FAULTS = (AgentProtocolError, StagePostValidationError, PermissionError)
_INFRASTRUCTURE_FAULTS = (AgentTransportError, AgentCallBudgetExceeded)


# Stage-local decision vocabulary reused unchanged from the E1 Task loop, so
# the paired summaries and the attribution router keep reading one vocabulary.
STOP_TRUST = "TRUST_DRAFT_GATE_PASS"
STOP_NO_DRAFT = "NO_DRAFT_IN_BUDGET"
STOP_ABSTAIN = "AGENT_ABSTAIN"
STOP_REQUEST_OBSERVATION = "REQUEST_OBSERVATION"
STOP_INSTRUMENT = "INSTRUMENT_UNREADABLE"
STOP_PROTOCOL = "AGENT_PROTOCOL_ERROR"
STOP_TRANSPORT = "TRANSPORT_FAILED"
STOP_LLM_BUDGET = "LLM_CALL_BUDGET_EXHAUSTED"


@dataclasses.dataclass
class FastPathTrace:
    """Everything one Task Episode's Fast Path did, in the order it did it."""

    stages: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    tool_observations: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    proposals: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    compiled: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    probes: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    select_rounds: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    probe_order_deprioritizations: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    chosen_candidate_id: str = IDENTITY_CHOICE
    stop_reason: str = STOP_NO_DRAFT
    instrument_unreadable: bool = False
    infrastructure_failed: bool = False
    protocol_error: str | None = None
    protocol_error_output: str | None = None
    citation_normalizations: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    infrastructure_error: str | None = None
    ownership_audits: list[dict[str, Any]] = dataclasses.field(
        default_factory=list
    )

    def stage_payload(self, stage: str) -> dict[str, Any]:
        for row in self.stages:
            if row["stage"] == stage:
                return dict(row.get("payload") or {})
        return {}

    def observed_feature_keys(self) -> set[str]:
        keys: set[str] = set()
        for row in self.tool_observations:
            result = row.get("public_result") or {}
            features = result.get("features")
            if isinstance(features, Mapping):
                keys.update(str(key) for key in features)
            for key in result:
                if key.startswith("estimated_"):
                    keys.add(str(key))
        return keys


def _plain(value: Any) -> Any:
    """Frozen mappings and tuples back to JSON-native values.

    The stage payloads arrive deeply frozen (``MappingProxyType`` inside
    tuples).  A ``json.dumps(..., default=str)`` round trip silently turns an
    inner mapping into its repr string, which then reads as data downstream --
    so convert structurally instead.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain(nested) for nested in value]
    return value


def _executable_operator_names(
    inventory: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    return tuple(
        str(row["name"])
        for row in inventory
        if str(row.get("availability")) == "EXECUTABLE"
    )


def _operator_menu(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The Agent sees every operator and why each can or cannot run.

    Parameter ownership is stated per operator so the Agent knows which
    decisions are its own (which Program family) and which are the Runtime's
    (where inside a unit the edit lands).
    """
    menu: list[dict[str, Any]] = []
    for row in inventory:
        entry = {
            "name": row["name"],
            "category": row.get("category"),
            "availability": row.get("availability"),
            "destructive": row.get("destructive"),
            "preserves_observed": row.get("preserves_observed"),
            "targeting_mode": row.get("targeting_mode"),
            "public_parameter_schema": row.get("public_parameter_schema"),
        }
        if row.get("availability") != "EXECUTABLE":
            entry["reason"] = row.get("reason")
        menu.append(entry)
    return menu


def _ground_inspect(
    core: TTHAAgentCore, trace: FastPathTrace, payload: Mapping[str, Any]
) -> None:
    """Normalize the citation spelling, then apply the grounding rule."""
    served = getattr(core.tools, "observed_feature_values", None)
    if callable(served):
        trace.citation_normalizations.extend(
            _normalize_evidence_citations(payload, served())
        )
    _validate_inspect_hypotheses(
        payload, {key: True for key in _observed_keys(core, trace)}
    )


def _observed_keys(core: TTHAAgentCore, trace: FastPathTrace) -> set[str]:
    """Feature names the Agent has been shown, read while the stage still runs.

    ``run_stage`` returns its receipts only at the end, so a post-validator
    that grounded citations in ``trace`` alone would reject every hypothesis
    the Agent formed from a tool result it had just received.  The gateway
    knows what it has served, so ask it.
    """
    keys = trace.observed_feature_keys()
    live = getattr(core.tools, "observed_feature_keys", None)
    if callable(live):
        keys |= set(live())
    return keys


def _normalize_evidence_citations(
    payload: Mapping[str, Any],
    served: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Accept a bare feature name, or ``key=value`` that is exactly right.

    Observed twice in live runs: the Agent cites ``missing_fraction=0.0`` or
    ``period_evidence_status=OK`` instead of the bare name.  The grounding rule
    is not the problem -- a citation must name a feature the Agent was really
    shown -- but nothing in the contract states the citation format, so a
    correct observation was being thrown away on punctuation.

    ``key=value`` is normalized to ``key`` only when the key was served and the
    value matches a served value exactly.  A wrong value is still a wrong
    citation and is left alone for the validator to reject: this widens the
    accepted spelling, never the accepted evidence.

    The payload is rewritten in place, before the validator runs, so what is
    recorded downstream is the canonical bare-key form.
    """
    hypotheses = payload.get("pattern_hypotheses") or ()
    if not isinstance(hypotheses, Sequence) or isinstance(
        hypotheses, (str, bytes, bytearray)
    ):
        return []
    normalized: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        features = hypothesis.get("evidence_features")
        if not isinstance(features, list):
            continue
        for index, cited in enumerate(features):
            if not isinstance(cited, str) or "=" not in cited:
                continue
            key, _, value = cited.partition("=")
            key, value = key.strip(), value.strip()
            allowed = served.get(key)
            if not isinstance(allowed, (set, frozenset)) or value not in allowed:
                continue
            features[index] = key
            normalized.append({
                "hypothesis_id": hypothesis.get("hypothesis_id"),
                "cited_as": cited,
                "normalized_to": key,
                "value_matched_a_served_value": True,
            })
    return normalized


def _run_stage(
    core: TTHAAgentCore,
    trace: FastPathTrace,
    *,
    stage: str,
    case_id: str,
    public_input: Mapping[str, Any],
    harness_view: Any,
    schema_name: str,
    post_validator: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any] | None:
    schema = TTHAAgentCore.load_stage_schema(schema_name)
    result = core.run_stage(
        role=AgentRole.FAST,
        stage=stage,
        case_id=case_id,
        public_input=public_input,
        harness_view=harness_view,
        output_schema_name=schema_name,
        output_schema=schema,
        source_snapshot_sha=harness_view.effective_harness_view_sha,
        validation_retries=1,
        post_validator=post_validator,
    )
    for receipt in result.tool_receipts:
        row = {
            "stage": stage,
            "tool_name": receipt.tool_name,
            "arguments": _plain(receipt.arguments),
            "public_result": _plain(receipt.public_result),
            "receipt_sha": receipt.receipt_sha,
            "ok": bool(receipt.ok),
        }
        if row not in trace.tool_observations:
            trace.tool_observations.append(row)
    trace.stages.append(
        {
            "stage": stage,
            "payload": _plain(result.payload),
            "tool_call_count": len(result.tool_receipts),
            "validation_retry_count": result.validation_retry_count,
            "validation_error_codes": list(result.validation_error_codes),
            "first_pass_valid": bool(result.first_pass_valid),
        }
    )
    return result.payload


def _risk_deprioritized_skill_ids(harness_view: Any) -> set[str]:
    """IDs of the Target-local risk Skills this Task actually retrieved.

    Read off the resolved view rather than the snapshot, so a Skill that is
    out of Context -- or restricted after its own Domain contradicted it --
    has no effect here either.
    """
    return {
        str(skill.skill_id)
        for skill in getattr(harness_view, "skills", ()) or ()
        if str(getattr(getattr(skill, "skill_kind", None), "value", "")) == "safety"
        and str(skill.skill_id).startswith("target_risk_")
    }


def _deprioritized_probe_order(
    compiled_rows: Sequence[Mapping[str, Any]],
    harness_view: Any,
    trace: FastPathTrace,
) -> list[dict[str, Any]]:
    """Probe order after applying the retrieved risk Skills.  Order only.

    The first Support probe is spent on the head of this list before the
    select stage runs, so ordering is where a deprioritization either has an
    effect or has none at all.  The micro replay measured the "none" case: the
    Skill reached the Fast prompt in both arms of Task 3 and both still led
    with the refuted family, spending -0.197 before the Agent was asked
    anything -- and then both chose identity, so the Agent had not wanted to
    execute it either.

    What this does and does not do:

    * a candidate whose operator structure is named by a retrieved risk Skill
      moves behind the others, and nothing else about it changes;
    * it is never dropped, never blocked, and stays selectable -- if the
      budget reaches it, it is probed exactly as before;
    * relative order within each group is preserved, so the Agent's own
      ranking still decides everything this does not;
    * if every candidate is deprioritized there is nothing to prefer, so the
      order is left exactly as proposed.

    That last case matters: a deprioritization that reshuffles a field of
    equally-refuted candidates would be inventing a preference the evidence
    does not support.
    """
    rows = list(compiled_rows)
    deprioritized = _risk_deprioritized_skill_ids(harness_view)
    if not deprioritized:
        return rows

    def named_by_a_risk_skill(row: Mapping[str, Any]) -> bool:
        family = "+".join(str(op) for op, _params in row["steps"])
        return "target_risk_" + family.replace("+", "_") in deprioritized

    held_back = [row for row in rows if named_by_a_risk_skill(row)]
    if not held_back or len(held_back) == len(rows):
        return rows
    preferred = [row for row in rows if not named_by_a_risk_skill(row)]
    reordered = [*preferred, *held_back]
    # Only record a deprioritization that actually moved something.  The
    # Agent frequently ranks the alternative first on its own, and a receipt
    # emitted for an order that did not change would inflate every later
    # count of "times the reorder acted" -- which is exactly the readout the
    # full run reports.
    if [row["candidate_id"] for row in reordered] == [
        row["candidate_id"] for row in rows
    ]:
        return rows
    trace.probe_order_deprioritizations.append({
        "retrieved_risk_skill_ids": sorted(deprioritized),
        "proposed_order": [str(row["candidate_id"]) for row in rows],
        "probe_order": [str(row["candidate_id"]) for row in reordered],
        "moved_behind": [str(row["candidate_id"]) for row in held_back],
        "note": (
            "order only: every candidate stays selectable and is probed "
            "unchanged if the budget reaches it"
        ),
    })
    return reordered


def run_agentic_fast_path(
    *,
    core: TTHAAgentCore,
    case_id: str,
    harness_view: Any,
    initial_context: Mapping[str, Any],
    retrieved: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    probe_budget: int,
    material_threshold: float,
    support_probe: Callable[[Any], Mapping[str, Any]],
) -> FastPathTrace:
    """Drive one Task Episode's Fast Path.

    ``support_probe`` receives a ``CompiledWorkflow`` and returns the Support
    metrics mapping.  It is the caller's frozen evaluator; this module never
    touches an Outcome and never sees one except through that return value.
    Raising from it means the instrument, not the Agent, failed -- the Task
    stops and is marked unreadable rather than recorded as a tie.
    """
    trace = FastPathTrace()
    menu = _operator_menu(inventory)

    inspect_input = {
        **dict(initial_context),
        "operator_contracts": menu,
        "target_support_budget": int(probe_budget),
        "material_threshold": float(material_threshold),
        "retrieved_knowledge": dict(retrieved),
        "stage_note": (
            "You have no per-series numbers yet. Call the Workspace tools on "
            "the scoped series you want to see, then report what you inspected."
        ),
    }

    # ---- INSPECT ---------------------------------------------------------
    try:
        inspect_payload = _run_stage(
            core, trace,
            stage="inspect",
            case_id=case_id,
            public_input=inspect_input,
            harness_view=harness_view,
            schema_name="fast_inspect_v1",
            post_validator=lambda payload: _ground_inspect(core, trace, payload),
        )
    except _AGENT_FAULTS as exc:
        trace.stop_reason = STOP_PROTOCOL
        trace.protocol_error = f"inspect: {type(exc).__name__}: {exc}"
        trace.protocol_error_output = getattr(
            exc, "last_assistant_text", None
        )
        return trace
    except _INFRASTRUCTURE_FAULTS as exc:
        trace.infrastructure_failed = True
        trace.stop_reason = (
            STOP_LLM_BUDGET
            if isinstance(exc, AgentCallBudgetExceeded)
            else STOP_TRANSPORT
        )
        trace.infrastructure_error = (
            f"inspect: {type(exc).__name__}: {exc}"
        )
        return trace

    # ---- PROPOSE ---------------------------------------------------------
    propose_input = {
        **inspect_input,
        "inspect_result": _plain(inspect_payload),
        # §5.2: the deterministic tool results become part of this round's
        # Context.  The Agent proposes against what it actually observed.
        "tool_observations": [dict(row) for row in trace.tool_observations],
        "stage_note": (
            "Propose one to %d Typed Workflow candidates. You choose the "
            "Program family; the Runtime owns where inside each action unit "
            "the edit lands. Each candidate costs one Target Support probe "
            "from a non-renewable budget." % int(probe_budget)
        ),
    }
    try:
        propose_payload = _run_stage(
            core, trace,
            stage="propose",
            case_id=case_id,
            public_input=propose_input,
            harness_view=harness_view,
            schema_name="fast_propose_v1",
            post_validator=lambda payload: _validate_hypothesis_references(
                payload, inspect_payload
            ),
        )
    except _AGENT_FAULTS as exc:
        trace.stop_reason = STOP_PROTOCOL
        trace.protocol_error = f"propose: {type(exc).__name__}: {exc}"
        trace.protocol_error_output = getattr(
            exc, "last_assistant_text", None
        )
        return trace
    except _INFRASTRUCTURE_FAULTS as exc:
        trace.infrastructure_failed = True
        trace.stop_reason = (
            STOP_LLM_BUDGET
            if isinstance(exc, AgentCallBudgetExceeded)
            else STOP_TRANSPORT
        )
        trace.infrastructure_error = (
            f"propose: {type(exc).__name__}: {exc}"
        )
        return trace

    candidates = list((propose_payload or {}).get("candidates") or ())
    trace.proposals = [_plain(candidate) for candidate in candidates]
    if not candidates:
        trace.stop_reason = STOP_ABSTAIN
        return trace

    # ---- COMPILE ---------------------------------------------------------
    # The compiler is given an empty public Context on purpose: no Task-level
    # Context number may be baked into a Program.  With RUNTIME_BOUND at zero
    # this changes nothing today; re-declaring an external binding would make
    # the compile fail loudly instead of silently broadcasting again.
    for index, candidate in enumerate(candidates[: int(probe_budget)]):
        candidate_id = str(candidate.get("candidate_id") or f"candidate-{index}")
        proposal = {
            "decision": "PROPOSE",
            "steps": [_plain(step) for step in candidate.get("steps") or ()],
            "requested_observations": [],
            "fallback": "IDENTITY",
            "experience_use": [],
        }
        row: dict[str, Any] = {
            "attempt_index": index,
            "candidate_id": candidate_id,
            "addresses_hypothesis_id": candidate.get("addresses_hypothesis_id"),
        }
        try:
            compiled = compile_workflow_proposal(
                proposal, inventory, {}, generation=index + 1
            )
        except (CandidateCompilationError, ValueError) as exc:
            row.update(
                {"status": "COMPILATION_FAILED",
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            trace.compiled.append(row)
            continue
        steps = compiled.candidate.program.execution_steps()
        try:
            audit = audit_program_parameter_ownership(steps)
        except ParameterOwnershipViolation as exc:
            row.update(
                {"status": "PARAMETER_OWNERSHIP_REJECTED",
                 "error": f"{exc.code}: {exc}"}
            )
            trace.compiled.append(row)
            continue
        trace.ownership_audits.append({"candidate_id": candidate_id, **audit})
        row.update(
            {"status": "COMPILED",
             "steps": [(str(op), dict(params)) for op, params in steps],
             "workflow": compiled}
        )
        trace.compiled.append(row)

    compiled_rows = [row for row in trace.compiled if row["status"] == "COMPILED"]
    if not compiled_rows:
        trace.stop_reason = STOP_NO_DRAFT
        return trace

    # ---- SUPPORT + SELECT ------------------------------------------------
    probed: list[dict[str, Any]] = []
    pending = _deprioritized_probe_order(compiled_rows, harness_view, trace)
    while pending:
        current = pending.pop(0)
        try:
            support = support_probe(current["workflow"])
        except Exception as exc:  # noqa: BLE001
            trace.probes.append(
                {"attempt_index": current["attempt_index"],
                 "candidate_id": current["candidate_id"],
                 "status": "INSTRUMENT_FAILED",
                 "error": f"{type(exc).__name__}: {exc}"}
            )
            trace.instrument_unreadable = True
            trace.stop_reason = STOP_INSTRUMENT
            return trace
        gain = float(support["macro_gain"])
        evidence = {
            "attempt_index": current["attempt_index"],
            "candidate_id": current["candidate_id"],
            "steps": [
                {"op": op, "params": params} for op, params in current["steps"]
            ],
            "support_gain": gain,
            "support_se_block": float(support["se_block"]),
            "support_gain_over_se": support["gain_over_se"],
            "meets_material_threshold": bool(gain >= material_threshold),
            "modified_point_count": int(support.get("modified_point_count") or 0),
        }
        probed.append(evidence)
        trace.probes.append(
            {**evidence, "status": "PROBED", "support": dict(support)}
        )

        selectable = [row["candidate_id"] for row in probed]
        next_id = pending[0]["candidate_id"] if pending else None
        options = [*selectable, IDENTITY_CHOICE]
        if next_id is not None:
            options.append(next_id)
        select_input = {
            "task": initial_context.get("task"),
            "task_episode_id": initial_context.get("task_episode_id"),
            "material_threshold": float(material_threshold),
            "target_support_budget": int(probe_budget),
            "support_probes_used": len(probed),
            "probed_candidates": probed,
            "unprobed_candidate_ids": [row["candidate_id"] for row in pending],
            "selectable_candidate_ids": options,
            "retrieved_knowledge": dict(retrieved),
            "stage_note": (
                "Choose a probed candidate to execute, name the next unprobed "
                "candidate to spend one more Support probe on it, or choose "
                f"'{IDENTITY_CHOICE}' to abstain. Only a candidate whose "
                "measured Support gain reaches the material threshold may be "
                "executed. Put 'request_observation' in verification_actions "
                "if the public evidence you need does not exist yet."
            ),
        }
        try:
            select_payload = _run_stage(
                core, trace,
                stage="select",
                case_id=case_id,
                public_input=select_input,
                harness_view=harness_view,
                schema_name="fast_select_v1",
            )
        except _AGENT_FAULTS as exc:
            trace.stop_reason = STOP_PROTOCOL
            trace.protocol_error = f"select: {type(exc).__name__}: {exc}"
            trace.protocol_error_output = getattr(
                exc, "last_assistant_text", None
            )
            return trace
        except _INFRASTRUCTURE_FAULTS as exc:
            trace.infrastructure_failed = True
            trace.stop_reason = (
                STOP_LLM_BUDGET
                if isinstance(exc, AgentCallBudgetExceeded)
                else STOP_TRANSPORT
            )
            trace.infrastructure_error = (
                f"select: {type(exc).__name__}: {exc}"
            )
            return trace
        chosen = str((select_payload or {}).get("chosen_candidate_id") or "")
        actions = [
            str(action)
            for action in ((select_payload or {}).get("verification_actions") or ())
        ]
        trace.select_rounds.append(
            {"after_attempt_index": current["attempt_index"],
             "chosen_candidate_id": chosen,
             "verification_actions": actions,
             "selectable_candidate_ids": options}
        )

        if "request_observation" in actions and chosen == IDENTITY_CHOICE:
            trace.chosen_candidate_id = IDENTITY_CHOICE
            trace.stop_reason = STOP_REQUEST_OBSERVATION
            return trace
        if chosen == IDENTITY_CHOICE:
            trace.chosen_candidate_id = IDENTITY_CHOICE
            trace.stop_reason = STOP_ABSTAIN
            return trace
        if next_id is not None and chosen == next_id:
            continue
        match = next(
            (row for row in probed if row["candidate_id"] == chosen), None
        )
        if match is None:
            # An unselectable id is a protocol slip, not a decision.  The
            # Runtime does not guess which candidate was meant.
            trace.chosen_candidate_id = IDENTITY_CHOICE
            trace.stop_reason = STOP_ABSTAIN
            trace.protocol_error = f"select chose an unselectable id: {chosen!r}"
            return trace
        if not match["meets_material_threshold"]:
            # The mechanical Gate is the Runtime's, never the Agent's.
            trace.select_rounds[-1]["mechanical_gate"] = (
                "REJECT_TRUST_BELOW_THRESHOLD"
            )
            if pending:
                continue
            trace.chosen_candidate_id = IDENTITY_CHOICE
            trace.stop_reason = STOP_NO_DRAFT
            return trace
        trace.chosen_candidate_id = chosen
        trace.stop_reason = STOP_TRUST
        return trace

    trace.chosen_candidate_id = IDENTITY_CHOICE
    trace.stop_reason = STOP_NO_DRAFT
    return trace


__all__ = [
    "FastPathTrace",
    "IDENTITY_CHOICE",
    "STOP_ABSTAIN",
    "STOP_INSTRUMENT",
    "STOP_NO_DRAFT",
    "STOP_LLM_BUDGET",
    "STOP_PROTOCOL",
    "STOP_REQUEST_OBSERVATION",
    "STOP_TRANSPORT",
    "STOP_TRUST",
    "run_agentic_fast_path",
]
