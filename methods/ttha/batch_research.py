"""Batch research controller; numerical execution is supplied by one shared adapter.

This module performs no I/O, fitting, network calls or canonical Skill writes.
The live client/adapter must enforce their own physical cost and subprocess
deadlines. Tests with scripted clients validate control flow, not model utility.
"""
from __future__ import annotations

import copy
import json
import math
import re
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

HOOKS = frozenset({
    "observation_guidance", "construction_guidance",
    "experiment_guidance", "decision_guidance",
})
READ_TOOLS = frozenset({"overview", "inspect_data", "inspect_material"})
TOOLS = READ_TOOLS | {"build_material", "evaluate", "compare", "commit"}
PRIVATE_KEYS = frozenset({
    "c_b", "e", "evaluation_truth", "query_future", "source_episodes",
    "raw_episode_bank", "source_experiences",
})


def json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def public_copy(value: Any) -> Any:
    """Reject named private channels rather than silently stripping evidence."""
    out = json_copy(value)

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if key.lower() in PRIVATE_KEYS:
                    raise PermissionError(f"private channel in public adapter output: {key}")
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(out)
    return out


def validate_scope(ast: Any, allowed_features: frozenset[str], depth: int = 0) -> set[str]:
    """Local vocabulary, same AST as the existing three-valued evaluator."""
    if depth > 12 or not isinstance(ast, dict):
        raise ValueError("scope must be a bounded AST object; null is incomplete")
    keys = set(ast)
    if keys == {"const"} and type(ast["const"]) is bool:
        return set()
    if keys in ({"all"}, {"any"}):
        children = ast[next(iter(keys))]
        if not isinstance(children, list) or not children:
            raise ValueError("all/any requires a nonempty list")
        return set().union(*(validate_scope(c, allowed_features, depth + 1) for c in children))
    if keys == {"not"}:
        return validate_scope(ast["not"], allowed_features, depth + 1)
    if keys != {"feature", "op", "value"}:
        raise ValueError("unsupported scope shape")
    if ast["feature"] not in allowed_features or ast["op"] not in {">", ">=", "<", "<=", "=="}:
        raise ValueError("scope feature/operator is outside the public vocabulary")
    number = ast["value"]
    if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
        raise ValueError("scope threshold must be a finite number")
    return {ast["feature"]}


def scope_state(ast: dict | None, features: dict, allowed: frozenset[str]) -> tuple[str, list[str]]:
    if ast is None:
        return "INCOMPLETE_NO_APPLICABILITY", []
    required = validate_scope(ast, allowed)
    clean = {
        k: v for k, v in features.items()
        if k in allowed and isinstance(v, (int, float)) and not isinstance(v, bool)
        and math.isfinite(v)
    }
    # Reuse the repository algorithm without extending canonical observables.
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import _evaluate
    state, _ = _evaluate(ast, clean)
    return ("MATCH" if state is True else "NO_MATCH" if state is False else "UNKNOWN",
            sorted(required - clean.keys()))


@dataclass(frozen=True)
class Guidance:
    hook: str
    body: str
    applicability: dict | None
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class Knowledge:
    version: int = 0
    entries: tuple[Guidance, ...] = ()

    def render(self, features: dict, allowed: frozenset[str]) -> dict:
        loaded, applicability = [], []
        for item in self.entries:
            if item.hook not in HOOKS:
                raise ValueError("unknown guidance hook")
            state, missing = scope_state(item.applicability, features, allowed)
            applicability.append({"hook": item.hook, "state": state, "missing_features": missing})
            if state == "MATCH":
                loaded.append({"hook": item.hook, "body": item.body})
        # Missing/unknown scope does not remove public tools or legal actions.
        return {"version": self.version, "loaded": loaded, "applicability": applicability}


def apply_update(parent: Knowledge, proposal: Mapping[str, Any], *,
                 allowed_features: frozenset[str], legal_evidence_refs: frozenset[str], body_limit: int = 1200) -> Knowledge:
    """Apply one well-formed candidate edit, or KEEP; never promote a capability.
    body_limit: a study's explicit opt-in body capacity (DEV-TEMPO-AUG-WORKFLOW-LEARNING-LOOP §4.3: 6000); default = the historical 1200."""
    p = json_copy(proposal)
    if p.get("decision") == "KEEP":
        if set(p) - {"decision", "rationale"}:
            raise ValueError("KEEP cannot carry a hidden edit")
        return copy.deepcopy(parent)
    expected = {"decision", "hook", "body", "observable_applicability", "evidence_refs", "rationale"}
    if p.get("decision") != "PROPOSE" or set(p) != expected:
        raise ValueError("PROPOSE requires exactly one hook, body, explicit scope and evidence")
    if type(body_limit) is not int or body_limit <= 0:
        raise ValueError("invalid body limit")
    if p["hook"] not in HOOKS or not isinstance(p["body"], str) or not 0 < len(p["body"].strip()) <= body_limit:
        raise ValueError("invalid hook/body")
    if re.search(r"\b(?:series_uid|entity_id|uid|entity_\d+)\b", p["body"], re.I):
        raise ValueError("reusable guidance must not identify a specific entity")
    validate_scope(p["observable_applicability"], allowed_features)
    refs = p["evidence_refs"]
    if not isinstance(refs, list) or not refs or any(not isinstance(x, str) or x not in legal_evidence_refs for x in refs):
        raise ValueError("every evidence reference must resolve to supplied boundary evidence")
    item = Guidance(p["hook"], p["body"], p["observable_applicability"], tuple(refs))
    kept = tuple(copy.deepcopy(e) for e in parent.entries if e.hook != item.hook)
    return Knowledge(parent.version + 1, kept + (item,))


@dataclass(frozen=True)
class Candidate:
    plan_id: str
    material_spec: dict
    job_id: str
    model_seeds: tuple[int, ...] = ()
    model_refs: tuple[str, ...] = ()
    # C_A losses [training seed][origin][entity]; no C_B/E field exists here.
    ca_losses: tuple = ()

    def feedback(self, seeds: tuple[int, ...], origins: int, entities: int) -> dict:
        if self.model_seeds != seeds or len(self.model_refs) != len(seeds) or not all(isinstance(x, str) and x for x in self.model_refs):
            raise ValueError("candidate has incomplete model references")
        if len(self.ca_losses) != len(seeds):
            raise ValueError("C_A repeats do not match the paired seed set")
        per_seed, per_origin, per_entity = [], [], []
        for repeat in self.ca_losses:
            if len(repeat) != origins or any(len(row) != entities for row in repeat):
                raise ValueError("C_A tensor does not cover the entire declared workload")
            for row in repeat:
                if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in row):
                    raise ValueError("invalid C_A loss")
            per_origin.append([statistics.mean(row) for row in repeat])
            per_entity.append([statistics.mean(row[e] for row in repeat) for e in range(entities)])
            per_seed.append(statistics.mean(per_origin[-1]))
        return {
            "block": "C_A", "seeds": list(seeds), "loss_by_seed": per_seed,
            "loss_by_seed_origin": per_origin, "loss_by_seed_entity": per_entity,
            "mean_loss": statistics.mean(per_seed),
        }


class BatchAdapter(Protocol):
    """P/W shared numerical boundary. Returned public data must be T-only or C_A.

    The caller creates a separate adapter view for each branch. A shared physical
    cache is allowed, but candidates/observations from another branch are not.
    """
    def overview(self) -> dict: ...
    def inspect_data(self, arguments: dict, *, remaining_seconds: float) -> dict: ...
    def build_material(self, arguments: dict, *, remaining_seconds: float) -> Candidate: ...
    def inspect_material(self, plan_id: str, arguments: dict, *, remaining_seconds: float) -> dict: ...
    def evaluate(self, candidate: Candidate, seeds: tuple[int, ...], *,
                 feedback: bool, remaining_seconds: float) -> Candidate: ...


@dataclass(frozen=True)
class Limits:
    max_calls: int = 16
    max_tools: int = 24
    max_new_evaluations: int = 2
    wall_seconds: float = 7200.0


@dataclass
class RunResult:
    status: str
    job_id: str
    knowledge_version: int
    committed_plan_id: str | None
    delivery_model_ref: str | None
    calls: int
    tool_calls: int
    new_evaluations: int
    trace: list[dict] = field(default_factory=list)
    failure_kind: str | None = None
    reason: str | None = None


def paired_comparison(a: dict, b: dict) -> dict:
    if a["seeds"] != b["seeds"]:
        raise ValueError("unpaired comparison")
    delta = [x - y for x, y in zip(a["loss_by_seed"], b["loss_by_seed"], strict=True)]
    n = len(delta)
    return {
        "block": "C_A", "positive_means": "candidate_b_improves",
        "delta_by_seed": delta, "mean_delta": statistics.mean(delta),
        "seed_sd": statistics.stdev(delta) if n > 1 else None,
        "seed_se": statistics.stdev(delta) / math.sqrt(n) if n > 1 else None,
        "signs": {"positive": sum(x > 0 for x in delta), "negative": sum(x < 0 for x in delta),
                  "zero": sum(x == 0 for x in delta)},
        "delta_by_seed_origin": [[x - y for x, y in zip(xs, ys, strict=True)]
                                for xs, ys in zip(a["loss_by_seed_origin"], b["loss_by_seed_origin"], strict=True)],
        "delta_by_seed_entity": [[x - y for x, y in zip(xs, ys, strict=True)]
                                for xs, ys in zip(a["loss_by_seed_entity"], b["loss_by_seed_entity"], strict=True)],
        "scope": "Fixed materials and C_A workload; seed uncertainty only. Entity differences are not training-data causal labels.",
    }


class ToolInputError(ValueError):
    """Public, pre-execution argument rejection; never wrap I/O or fit errors."""
    def __init__(self, message: str, **details: Any):
        super().__init__(message)
        self.details = public_copy(details)


class BudgetExhausted(RuntimeError):
    pass


OVERVIEW_DEDUPE_MARKER = "see request.overview (identical; deduplicated in the request only)"


def dedupe_trace_for_request(trace: list, overview: dict) -> list:
    """Lossless request-side deduplication (opt-in): the job_started event's overview and the output of an overview tool call
    are byte-identical copies of the request's top-level overview; the request carries them as a marker instead. The stored
    trace (on_event / trace.jsonl) is untouched; every other event, tool result, plan and feedback stays complete."""
    out = []
    for row in trace:
        if row.get("event") == "job_started" and row.get("overview") == overview:
            out.append({**row, "overview": OVERVIEW_DEDUPE_MARKER})
        elif row.get("event") == "tool_completed" and row.get("tool") == "overview" and row.get("output") == overview:
            out.append({**row, "output": OVERVIEW_DEDUPE_MARKER})
        else:
            out.append(row)
    return out


def run_job(*, job_id: str, knowledge: Knowledge, adapter: BatchAdapter,
            client: Callable[[dict], Mapping[str, Any]], seeds: tuple[int, ...],
            baselines: tuple[Candidate, ...], allowed_features: frozenset[str],
            tool_contracts: Mapping[str, dict], entity_count: int = 32, ca_origins: int = 2, mode: str = "held_in", limits: Limits = Limits(),
            on_event: Callable[[dict], None] | None = None,
            guard: Callable[[], None] | None = None,
            evidence_roundtrip: bool = False, max_tool_corrections: int = 0,
            resume: Mapping[str, Any] | None = None, request_trace_dedupe: bool = False) -> RunResult:
    """Run a bounded whole-batch Fast loop; no fixed argmin or forced fallback.

    resume: continue a trajectory whose next Fast call was never sent (instrument stop). It carries
    the exact trace prefix, spent counters, the ids already charged as new evaluations and any
    features obtained by inspection; fitted private candidates arrive through `baselines`.
    request_trace_dedupe: opt-in (DEV-DOMAIN-AUG-DECISION-PRIORITY §3): the request's current_trace carries a marker instead of
    the overview copies that are identical to the request's top-level overview (dedupe_trace_for_request); default = historical.

    Client contract: {"actions": [{"tool": name, "arguments": {...}}, ...]}.
    Requests may contain multiple actions; commit must be last. Transient HTTP
    retries belong to the metered client. Optional pre-execution tool rejection
    feedback uses the ordinary next Fast call, never a hidden redraw. Malformed
    response envelopes, permissions, backend and execution faults remain terminal.
    """
    if mode not in {"held_in", "deploy"} or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("invalid mode or seed set")
    if min(limits.max_calls, limits.max_tools, limits.wall_seconds) <= 0 or limits.max_new_evaluations < 0:
        raise ValueError("invalid budget")
    if set(tool_contracts) != TOOLS or any(not isinstance(v, dict) for v in tool_contracts.values()):
        raise ValueError("declare every tool contract before running Fast")
    if type(max_tool_corrections) is not int or max_tool_corrections < 0:
        raise ValueError("invalid tool correction budget")
    frozen_contracts = public_copy(tool_contracts)
    frozen_h = copy.deepcopy(knowledge)
    candidates = {c.plan_id: copy.deepcopy(c) for c in baselines}
    if len(candidates) != len(baselines):
        raise ValueError("duplicate baseline id")
    trace: list[dict] = json_copy(resume["trace"]) if resume else []
    calls = tools_used = new_evaluations = corrections_used = 0
    if resume:
        calls, tools_used = int(resume["calls"]), int(resume["tools_used"])
        new_evaluations, corrections_used = int(resume["new_evaluations"]), int(resume["corrections_used"])
        if min(calls, tools_used, new_evaluations, corrections_used) < 0 or new_evaluations > limits.max_new_evaluations:
            raise ValueError("invalid resume counters")
    started = time.monotonic()
    committed: Candidate | None = None
    stage = "SETUP"

    def emit(event: str, **fields: Any) -> None:
        row = json_copy({"event_id": f"{job_id}:{len(trace)}", "event": event, **fields})
        trace.append(row)
        if on_event:
            on_event(copy.deepcopy(row))

    def remaining() -> float:
        if guard:
            guard()
        seconds = limits.wall_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise BudgetExhausted("wall budget exhausted")
        return seconds

    def complete(candidate: Candidate) -> bool:
        return (candidate.job_id == job_id and candidate.model_seeds == seeds
                and len(candidate.model_refs) == len(seeds)
                and all(isinstance(x, str) and x for x in candidate.model_refs))

    def reject_input(message: str) -> None:
        if max_tool_corrections:
            raise ToolInputError(message)
        raise ValueError(message)

    def finish(status: str, kind: str | None = None, reason: str | None = None) -> RunResult:
        return RunResult(status, job_id, frozen_h.version,
                         committed.plan_id if committed else None,
                         committed.model_refs[0] if committed else None,
                         calls, tools_used, new_evaluations, trace, kind, reason)

    try:
        remaining()
        overview = public_copy(adapter.overview())
        rows = overview.get("entities")
        if not isinstance(rows, list) or len(rows) != entity_count:
            raise ValueError("overview must cover the declared full entity roster")
        entities = len(rows)
        features = dict(overview.get("batch_features", {}))
        if resume:
            features.update(resume.get("features", {}))
        for candidate in candidates.values():
            if candidate.job_id != job_id:
                raise ValueError("baseline belongs to another job")
            public_copy(candidate.material_spec)
            if mode == "held_in":
                candidate.feedback(seeds, ca_origins, entities)
            elif candidate.ca_losses:
                raise PermissionError("deployment baseline must not carry calibration feedback")
        # Resume only: the branch's own plans in build order. Evaluated ones are checked like baselines; a plan built but never
        # evaluated keeps its id without any fit, score or slot (a later evaluate is counted under the normal rules).
        for candidate in (resume or {}).get("built", ()):
            if candidate.job_id != job_id or candidate.plan_id in candidates:
                raise ValueError("resumed plan belongs to another job or repeats an id")
            if complete(candidate):
                if mode == "held_in":
                    candidate.feedback(seeds, ca_origins, entities)
            elif candidate.model_seeds or candidate.model_refs or candidate.ca_losses:
                raise ValueError("resumed built plan carries partial fits")
            candidates[candidate.plan_id] = copy.deepcopy(candidate)
        # A resumed trajectory keeps its exact trace prefix; the resume marker is written by the caller only.
        if not resume:
            emit("job_started", mode=mode, overview=overview, knowledge_version=frozen_h.version,
                 seeds=list(seeds), baseline_ids=list(candidates), evidence_roundtrip=evidence_roundtrip,
                 max_tool_corrections=max_tool_corrections)
        evaluated_new: set[str] = set(resume.get("evaluated_new", ())) if resume else set()
        while committed is None:
            seconds = remaining()
            if calls >= limits.max_calls:
                raise BudgetExhausted("Fast call budget exhausted without commit")
            rendered = frozen_h.render(features, allowed_features)
            state = [{
                "plan_id": c.plan_id, "material_spec": c.material_spec,
                "trained": bool(complete(c)),
                "feedback": c.feedback(seeds, ca_origins, entities)
                if mode == "held_in" and complete(c) else None,
            } for c in candidates.values()]
            allowed_tools = sorted(TOOLS if mode == "held_in" else TOOLS - {"evaluate", "compare"})
            request = public_copy({
                "job_id": job_id, "mode": mode, "overview": overview,
                "guidance": rendered, "candidates": state,
                "current_trace": dedupe_trace_for_request(trace, overview) if request_trace_dedupe else trace,
                "tools": allowed_tools,
                "tool_contracts": {name: frozen_contracts[name] for name in allowed_tools},
                "contract": {
                    "response": {"actions": [{"tool": "<listed tool>", "arguments": {}}]},
                    "rules": "One whole-batch workflow. Choose only complete material plans. commit is final. Guidance is advice; unmatched guidance never removes public actions.",
                },
                "remaining": {"calls": limits.max_calls - calls,
                              "tools": limits.max_tools - tools_used,
                              "new_evaluations": limits.max_new_evaluations - new_evaluations,
                              "seconds": seconds},
            })
            if max_tool_corrections:
                request["contract"]["tool_error_feedback"] = (
                    "A rejected pre-execution tool argument returns its exact public validation "
                    "error. Remaining actions in that response are NOT executed or queued. "
                    "Read the error, then explicitly correct, abandon or choose another legal "
                    "action in your next ordinary call. Completed earlier tools stay completed. "
                    "Failed tool attempts and all Fast calls count toward the same budgets. "
                    "No automatic field renaming, retry, fallback, extra fits or outcome feedback."
                )
                request["remaining"]["tool_corrections"] = max_tool_corrections - corrections_used
            if evidence_roundtrip:
                request["contract"]["evidence_roundtrip"] = (
                    "After a tool returns new observations or evaluation/comparison evidence, "
                    "a following build_material/evaluate/commit in that response is deferred. "
                    "Read the returned evidence in your next request and explicitly reissue or "
                    "revise the remaining actions. Deferred actions have no effects or tool/fit charge. "
                    "This does not prescribe any plan, winner, threshold or fallback."
                )
            calls += 1
            emit("fast_request", number=calls, guidance=rendered)
            stage = "AGENT_CALL"
            response = client(copy.deepcopy(request))
            stage = "RESPONSE_VALIDATION"
            response = json_copy(response)
            emit("fast_response", number=calls, response=response)
            if not isinstance(response, dict) or set(response) != {"actions"}:
                raise ValueError("Fast response must contain only actions")
            actions = response["actions"]
            if not isinstance(actions, list) or not actions:
                raise ValueError("Fast must request an action, including an explicit commit")
            for index, action in enumerate(actions):
                if not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
                    raise ValueError("action requires tool and arguments")
                if action["tool"] not in allowed_tools or not isinstance(action["arguments"], dict):
                    raise ValueError("unknown/disallowed tool or invalid arguments")
                if action["tool"] == "commit" and index != len(actions) - 1:
                    raise ValueError("no action may follow commit")
            unseen_evidence = False
            for action_index, action in enumerate(actions):
                if evidence_roundtrip and unseen_evidence and action["tool"] in {"build_material", "evaluate", "commit"}:
                    emit("action_batch_deferred", actions=actions[action_index:],
                         reason="Read newly returned evidence before making the next material/training/commit decision")
                    break
                seconds = remaining()
                if tools_used >= limits.max_tools:
                    raise BudgetExhausted("tool budget exhausted without commit")
                tools_used += 1
                name, args = action["tool"], action["arguments"]
                emit("tool_started", tool=name, arguments=args)
                stage = "RUNTIME"
                try:
                    if name == "overview":
                        if args:
                            reject_input("overview takes no arguments")
                        output = overview
                    elif name == "inspect_data":
                        output = public_copy(adapter.inspect_data(copy.deepcopy(args), remaining_seconds=seconds))
                        # Only the adapter can provide new legal observations.
                        features.update(output.get("batch_features", {}))
                    elif name == "build_material":
                        candidate = adapter.build_material(copy.deepcopy(args), remaining_seconds=seconds)
                        if candidate.job_id != job_id or candidate.model_seeds or candidate.model_refs or candidate.ca_losses:
                            raise ValueError("build_material may not smuggle fitted models/feedback")
                        public_copy(candidate.material_spec)
                        if candidate.plan_id in candidates and candidates[candidate.plan_id].material_spec != candidate.material_spec:
                            raise ValueError("plan id collision")
                        candidates.setdefault(candidate.plan_id, copy.deepcopy(candidate))
                        output = {"plan_id": candidate.plan_id, "material_spec": candidate.material_spec}
                    elif name == "inspect_material":
                        plan_id = args.get("plan_id")
                        if not isinstance(plan_id, str) or plan_id not in candidates:
                            reject_input("cannot inspect a foreign/unknown plan")
                        output = public_copy(adapter.inspect_material(plan_id, copy.deepcopy(args), remaining_seconds=seconds))
                    elif name in {"evaluate", "commit"}:
                        plan_id = args.get("plan_id")
                        if not isinstance(plan_id, str) or plan_id not in candidates:
                            reject_input("cannot train/commit a foreign/unknown plan")
                        candidate = candidates[plan_id]
                        if name == "commit" and (set(args) != {"plan_id", "reason"} or not isinstance(args["reason"], str) or not args["reason"].strip()):
                            reject_input("commit needs a plan id and nonempty reason")
                        if name == "evaluate" and set(args) != {"plan_id"}:
                            reject_input("evaluate cannot alter training seeds or permissions")
                        if not complete(candidate):
                            if name == "commit" and mode == "held_in":
                                reject_input("commit requires an already evaluated candidate")
                            # Optional study-level execution check (public ToolInputError only), before the
                            # new-candidate slot is counted and before any fit is reserved. Absent = no check.
                            check_evaluate = getattr(adapter, "check_evaluate", None)
                            if name == "evaluate" and check_evaluate is not None:
                                check_evaluate(plan_id, copy.deepcopy(candidate))
                            if plan_id not in evaluated_new:
                                if new_evaluations >= limits.max_new_evaluations:
                                    raise BudgetExhausted("new candidate budget exhausted")
                                evaluated_new.add(plan_id)
                                new_evaluations += 1  # reserve before any physical work
                            fitted = adapter.evaluate(copy.deepcopy(candidate), seeds, feedback=mode == "held_in",
                                                      remaining_seconds=seconds)
                            if fitted.job_id != job_id or fitted.model_seeds != seeds or fitted.plan_id != plan_id or fitted.material_spec != candidate.material_spec:
                                raise ValueError("fitting changed frozen material identity")
                            if not complete(fitted):
                                raise ValueError("fit did not complete all paired models")
                            if mode == "held_in":
                                fitted.feedback(seeds, ca_origins, entities)
                            elif fitted.ca_losses:
                                raise PermissionError("deployment fit returned calibration losses")
                            candidates[plan_id] = candidate = copy.deepcopy(fitted)
                        if name == "commit":
                            # Optional study-level commit check, before the delivery is bound or written.
                            check_commit = getattr(adapter, "check_commit", None)
                            if check_commit is not None:
                                check_commit(plan_id, copy.deepcopy(candidate))
                            committed = candidate
                            output = {"plan_id": plan_id, "delivery_model_ref": candidate.model_refs[0],
                                      "reason": args["reason"]}
                        else:
                            output = {"plan_id": plan_id, "feedback": candidate.feedback(seeds, ca_origins, entities)}
                    elif name == "compare":
                        if set(args) != {"a", "b"}:
                            reject_input("compare needs two existing plan ids")
                        if any(not isinstance(args[k], str) or args[k] not in candidates for k in ("a", "b")):
                            reject_input("compare needs two existing plan ids")
                        a, b = candidates[args["a"]], candidates[args["b"]]
                        if not complete(a) or not complete(b):
                            reject_input("compare requires two evaluated candidates")
                        output = paired_comparison(a.feedback(seeds, ca_origins, entities),
                                                   b.feedback(seeds, ca_origins, entities))
                        output.update({"a": a.plan_id, "b": b.plan_id})
                    else:
                        raise AssertionError(name)
                except ToolInputError as exc:
                    if not max_tool_corrections:
                        raise
                    can_return = corrections_used < max_tool_corrections
                    emit("tool_rejected", tool=name, arguments=args,
                         error={"kind": "TOOL_INPUT_INVALID", "message": str(exc), **exc.details},
                         correction_allowed=can_return,
                         unexecuted_actions=actions[action_index + 1:])
                    if not can_return:
                        raise BudgetExhausted("tool correction budget exhausted") from None
                    corrections_used += 1
                    # Never run or auto-retry this response's suffix after rejection.
                    break
                remaining()
                emit("tool_completed", tool=name, output=public_copy(output))
                if name in {"overview", "inspect_data", "inspect_material", "evaluate", "compare"}:
                    unseen_evidence = True
        emit("committed", plan_id=committed.plan_id, delivery_model_ref=committed.model_refs[0])
        return finish("COMPLETE")
    except BudgetExhausted as exc:
        committed = None
        emit("job_incomplete", failure_kind="BUDGET_EXHAUSTED", reason=str(exc))
        return finish("INCOMPLETE", "BUDGET_EXHAUSTED", str(exc))
    except Exception as exc:
        committed = None
        kind = ("AGENT_CALL_FAILED" if stage == "AGENT_CALL" else
                "PARSE_OR_VALIDATION_FAILED" if stage == "RESPONSE_VALIDATION" else "RUNTIME_FAILED")
        # Exception class only: a backend exception may contain credentials/URLs.
        reason = type(exc).__name__
        emit("job_incomplete", failure_kind=kind, reason=reason)
        return finish("INCOMPLETE", kind, reason)
