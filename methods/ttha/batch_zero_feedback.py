"""Zero-feedback Fast loop (DEV-AUG-OFFLINE-SKILL task §3-4; explicitly enabled by that study only).

The historical feedback loop (batch_research.run_job, modes held_in / deploy) is untouched. This loop differs in four ways:
- there is no evaluate / compare tool and no downstream number of any block reaches Fast;
- commit freezes a constructed, legal, NOT fitted plan (or an unfitted public reference); an external evaluator trains and scores
  only after every commit of the stage is frozen; nothing is marked "evaluated" to pass an older commit check;
- the request carries one compact case card + batch summary; the per-entity table, data views and material diagnostics come from tools;
  the request history lists each action once (the Fast response) and each tool result once (no overview / plan / guidance re-echo);
- per-trajectory caps: logical Fast requests (corrections included), tool calls, distinct constructed materials, one commit; the last
  request (by calls, tools or the caller's token gate) offers only `commit`, so the budget always leaves a commit opportunity.

No I/O, fitting or network here; the adapter, the client and the token gate are supplied by the study.
"""
from __future__ import annotations

import copy
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from methods.ttha import batch_research as br

TOOLS = ("overview", "inspect_data", "build_material", "inspect_material", "commit")
EVIDENCE_TOOLS = frozenset({"overview", "inspect_data", "inspect_material"})
FORBIDDEN_REQUEST_KEYS = frozenset({"c_a", "c_b", "e", "ca_losses", "feedback", "loss_by_seed", "normalized_mse_macro", "e_scores", "evaluation_truth",
                                    "query_future", "source_episodes", "raw_episode_bank", "source_experiences"})


@dataclass(frozen=True)
class Limits:
    max_calls: int = 6          # logical Fast requests, corrections included
    max_tools: int = 18
    max_materials: int = 4      # distinct constructed materials (successful builds)
    wall_seconds: float = 7200.0


def check_no_feedback(value: Any, where: str = "request") -> None:
    """Fail closed if a downstream-score channel appears anywhere in what Fast would read."""
    def walk(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for k, v in item.items():
                if str(k).lower() in FORBIDDEN_REQUEST_KEYS:
                    raise PermissionError("downstream-score channel %r in %s at %s" % (k, where, path))
                walk(v, path + "." + str(k))
        elif isinstance(item, list):
            for i, v in enumerate(item):
                walk(v, "%s[%d]" % (path, i))
    walk(item=value, path=where)


def compact_history(trace: list) -> list:
    """The request view of the stored trace: each Fast response once, each tool result / rejection / deferral once. The stored trace keeps
    every event (tool_started arguments, per-request guidance copies, timings) for audit; nothing Fast did or saw is dropped here."""
    out = []
    for r in trace:
        ev = r.get("event")
        if ev == "fast_response":
            out.append({"call": r["number"], "response": r["response"]})
        elif ev == "response_rejected":
            out.append({"call": r["number"], "response_rejected": r["error"], "raw_excerpt": r.get("raw_excerpt")})
        elif ev == "tool_completed":
            out.append({"call": r["call"], "tool": r["tool"], "output": r["output"]})
        elif ev == "tool_rejected":
            out.append({"call": r["call"], "tool": r["tool"], "rejected": r["error"], "not_executed": r.get("unexecuted_actions", [])})
        elif ev == "action_batch_deferred":
            out.append({"call": r["call"], "deferred_not_executed": r["actions"], "reason": r["reason"]})
        elif ev == "job_resumed":
            out.append({"note": "the trajectory resumed after a transport fault; the next request is the one that was never answered"})
    return out


def run_job_zf(*, job_id: str, knowledge: br.Knowledge, adapter, client: Callable[[dict], str], public: tuple, allowed_features: frozenset,
               tool_contracts: Mapping[str, dict], entity_count: int, limits: Limits = Limits(), on_event: Callable[[dict], None] | None = None,
               guard: Callable[[], None] | None = None, evidence_roundtrip: bool = True, max_corrections: int = 2,
               token_gate: Callable[[int], dict] | None = None, system_notes: Mapping[str, Any] | None = None,
               resume: Mapping[str, Any] | None = None) -> br.RunResult:
    """One bounded zero-feedback trajectory. client(request) -> raw response text (the loop parses it; a malformed envelope is a
    correctable contract error inside the call budget). token_gate(request_bytes) -> {"send": bool, "final": bool, "reason": str}.
    adapter: case_card(), batch_summary(), overview_table(arguments), inspect_data(arguments), build_material(arguments) -> br.Candidate,
    inspect_material(plan_id, arguments), check_commit(plan_id, candidate); it never fits and never returns scores.
    resume: {"trace", "calls", "tools_used", "materials", "corrections", "built": [Candidate]} of a trajectory whose next request was never answered."""
    if set(tool_contracts) != set(TOOLS):
        raise ValueError("declare exactly the zero-feedback tool contracts")
    frozen_contracts = br.public_copy(dict(tool_contracts))
    frozen_h = copy.deepcopy(knowledge)
    candidates = {c.plan_id: copy.deepcopy(c) for c in public}
    for c in candidates.values():
        if c.model_seeds or c.model_refs or c.ca_losses:
            raise PermissionError("zero-feedback public references must be unfitted")
    trace: list = br.json_copy(resume["trace"]) if resume else []
    calls = int(resume["calls"]) if resume else 0
    tools_used = int(resume["tools_used"]) if resume else 0
    materials = int(resume["materials"]) if resume else 0
    corrections = int(resume["corrections"]) if resume else 0
    started = time.monotonic()
    committed: br.Candidate | None = None
    stage = "SETUP"

    def emit(event: str, **fields: Any) -> None:
        row = br.json_copy({"event_id": f"{job_id}:{len(trace)}", "event": event, **fields})
        trace.append(row)
        if on_event:
            on_event(copy.deepcopy(row))

    def remaining_s() -> float:
        if guard:
            guard()
        s = limits.wall_seconds - (time.monotonic() - started)
        if s <= 0:
            raise br.BudgetExhausted("wall budget exhausted")
        return s

    def finish(status: str, kind: str | None = None, reason: str | None = None) -> br.RunResult:
        return br.RunResult(status, job_id, frozen_h.version, committed.plan_id if committed else None, None, calls, tools_used, materials, trace, kind, reason)

    try:
        remaining_s()
        card = br.public_copy(adapter.case_card())
        summary = br.public_copy(adapter.batch_summary())
        check_no_feedback(card, "case_card")
        check_no_feedback(summary, "batch_summary")
        if card.get("n_entities") != entity_count:
            raise ValueError("case card must declare the full roster")
        for c in (resume or {}).get("built", ()):
            if c.job_id != job_id or c.plan_id in candidates or c.model_refs or c.ca_losses:
                raise ValueError("resumed plan is foreign, repeated or fitted")
            candidates[c.plan_id] = copy.deepcopy(c)
        if not resume:
            emit("job_started", mode="zero_feedback", knowledge_version=frozen_h.version, public_ids=list(candidates),
                 limits={"max_calls": limits.max_calls, "max_tools": limits.max_tools, "max_materials": limits.max_materials},
                 evidence_roundtrip=evidence_roundtrip, max_corrections=max_corrections)
        features = dict((resume or {}).get("features", {}))
        while committed is None:
            seconds = remaining_s()
            if calls >= limits.max_calls:
                raise br.BudgetExhausted("Fast call budget exhausted without commit")
            rendered = frozen_h.render({**summary.get("batch_features", {}), **features}, allowed_features)
            final = calls + 1 >= limits.max_calls or tools_used >= limits.max_tools - 1
            final_reason = "last call" if calls + 1 >= limits.max_calls else ("last tool" if final else "")
            tools = ["commit"] if final else list(TOOLS)
            plans = [{"plan_id": c.plan_id, "kind": "public_reference" if c.material_spec.get("kind") == "public_reference" else "constructed",
                      "label": c.material_spec.get("label") or c.material_spec.get("public_label")} for c in candidates.values()]

            def build_request(final_flag: bool, why: str) -> dict:
                req = {"job_id": job_id, "mode": "zero_feedback", "case": card, "batch_summary": summary, "guidance": rendered, "plans": plans,
                       "history": compact_history(trace), "tools": (["commit"] if final_flag else tools),
                       "tool_contracts": {n: frozen_contracts[n] for n in (["commit"] if final_flag else tools)},
                       "contract": {"response": {"actions": [{"tool": "<listed tool>", "arguments": {}}]},
                                    "rules": ("Zero downstream feedback: no training, score or ranking of any plan is available in this trajectory. commit freezes one "
                                              "plan listed in `plans` (a constructed plan or a public reference); an external evaluator trains and scores it only "
                                              "after every commit is frozen, and nothing comes back to you. commit is final and must be the last action. Guidance "
                                              "is advice; it never removes public actions."),
                                    "tool_error_feedback": ("A rejected tool argument or a malformed response returns its exact validation error in `history`; the "
                                                            "remaining actions of that response are not executed. Correct, abandon or choose another legal action "
                                                            "in your next call. Errors count toward the same budgets."),
                                    **({"evidence_roundtrip": ("After a tool returns new observations (overview, inspect_data, inspect_material), a following "
                                                               "build_material / commit in the same response is deferred without effect or charge; read the "
                                                               "evidence and reissue it in the next call.")} if evidence_roundtrip else {}),
                                    **(dict(system_notes) if system_notes else {})},
                       "remaining": {"calls": limits.max_calls - calls, "tools": limits.max_tools - tools_used, "new_materials": limits.max_materials - materials,
                                     "corrections": max_corrections - corrections, "seconds": round(seconds),
                                     "final_request": final_flag}}
                if final_flag:
                    req["remaining"]["final_request_note"] = ("FINAL REQUEST (%s): only commit is available now; a response without a valid commit ends the "
                                                              "trajectory INCOMPLETE with no delivery." % why)
                return br.public_copy(req)

            request = build_request(final, final_reason)
            if token_gate is not None:
                gate = token_gate(len(json.dumps(request, ensure_ascii=False).encode("utf-8")))
                if not gate["send"]:
                    raise br.BudgetExhausted("per-trajectory token budget: " + gate["reason"])
                if gate["final"] and not final:
                    final, final_reason = True, gate["reason"]
                    request = build_request(True, final_reason)
            check_no_feedback(request)
            calls += 1
            emit("fast_request", number=calls, final=final, final_reason=final_reason, guidance=rendered, request_bytes=len(json.dumps(request, ensure_ascii=False).encode("utf-8")))
            stage = "AGENT_CALL"
            text = client(copy.deepcopy(request))
            stage = "RESPONSE_VALIDATION"
            try:
                response = json.loads(text) if isinstance(text, str) else None
                if not isinstance(response, dict) or set(response) != {"actions"}:
                    raise ValueError("the response must be exactly one JSON object {\"actions\": [...]}")
                actions = response["actions"]
                if not isinstance(actions, list) or not actions:
                    raise ValueError("actions must be a non-empty list (commit explicitly when done)")
                allowed = ["commit"] if final else list(TOOLS)
                for i, a in enumerate(actions):
                    if not isinstance(a, dict) or set(a) != {"tool", "arguments"} or not isinstance(a["arguments"], dict):
                        raise ValueError("each action is {\"tool\": name, \"arguments\": {...}}")
                    if a["tool"] not in allowed:
                        raise ValueError("tool %r is not available in this request (available: %s)" % (a["tool"], allowed))
                    if a["tool"] == "commit" and i != len(actions) - 1:
                        raise ValueError("no action may follow commit")
            except ValueError as exc:
                emit("response_rejected", number=calls, error={"kind": "RESPONSE_INVALID", "message": str(exc)[:400]},
                     raw_excerpt=(text[:600] if isinstance(text, str) else None))
                if final:
                    raise br.BudgetExhausted("final request answered without a valid commit") from None
                if corrections >= max_corrections:
                    raise br.BudgetExhausted("correction budget exhausted") from None
                corrections += 1
                continue
            emit("fast_response", number=calls, response=response)
            stage = "RUNTIME"
            unseen = False
            for ai, action in enumerate(actions):
                name, args = action["tool"], action["arguments"]
                if evidence_roundtrip and unseen and name in ("build_material", "commit"):
                    emit("action_batch_deferred", call=calls, actions=actions[ai:], reason="read the newly returned evidence before the next construction / commit decision")
                    break
                if name != "commit" and tools_used >= limits.max_tools - 1:
                    emit("action_batch_deferred", call=calls, actions=actions[ai:], reason="the last tool call is reserved for commit")
                    break
                if tools_used >= limits.max_tools:
                    raise br.BudgetExhausted("tool budget exhausted without commit")
                remaining_s()
                tools_used += 1
                emit("tool_started", call=calls, tool=name, arguments=args)
                try:
                    if name == "overview":
                        output = br.public_copy(adapter.overview_table(copy.deepcopy(args)))
                    elif name == "inspect_data":
                        output = br.public_copy(adapter.inspect_data(copy.deepcopy(args), remaining_seconds=seconds))
                        features.update(output.get("batch_features", {}))
                    elif name == "build_material":
                        if materials >= limits.max_materials:
                            raise br.ToolInputError("material budget exhausted (%d distinct constructed materials); commit one of the listed plans" % limits.max_materials)
                        cand = adapter.build_material(copy.deepcopy(args), remaining_seconds=seconds)
                        if cand.job_id != job_id or cand.model_seeds or cand.model_refs or cand.ca_losses:
                            raise ValueError("build_material may not carry fits or feedback")
                        if cand.plan_id in candidates:
                            raise ValueError("plan id collision")
                        candidates[cand.plan_id] = copy.deepcopy(cand)
                        materials += 1
                        ms = cand.material_spec
                        output = {"plan_id": cand.plan_id, "label": ms.get("label"), "programs": ms.get("programs"), "entity_program": ms.get("entity_program"),
                                  "rule_index": ms.get("rule_index"), "resolved_thresholds": ms.get("resolved_thresholds"), "n_unknown": ms.get("n_unknown"),
                                  "status": "CONSTRUCTED_NOT_TRAINED"}
                    elif name == "inspect_material":
                        pid = args.get("plan_id")
                        if not isinstance(pid, str) or pid not in candidates:
                            raise br.ToolInputError("inspect_material needs a plan id listed in `plans`")
                        output = br.public_copy(adapter.inspect_material(pid, copy.deepcopy(args), remaining_seconds=seconds))
                    elif name == "commit":
                        pid = args.get("plan_id")
                        if set(args) != {"plan_id", "reason"} or not isinstance(args.get("reason"), str) or not args["reason"].strip():
                            raise br.ToolInputError("commit needs exactly plan_id and a nonempty reason")
                        if not isinstance(pid, str) or pid not in candidates:
                            raise br.ToolInputError("commit needs a plan id listed in `plans`")
                        adapter.check_commit(pid, copy.deepcopy(candidates[pid]))
                        committed = candidates[pid]
                        output = {"plan_id": pid, "status": "FROZEN_FOR_EXTERNAL_EVALUATION", "reason": args["reason"]}
                    else:
                        raise AssertionError(name)
                    check_no_feedback(output, "tool output")
                except br.ToolInputError as exc:
                    emit("tool_rejected", call=calls, tool=name, arguments=args, error={"kind": "TOOL_INPUT_INVALID", "message": str(exc), **exc.details},
                         unexecuted_actions=actions[ai + 1:])
                    if final or corrections >= max_corrections:
                        raise br.BudgetExhausted("tool rejected with no correction left" if not final else "final commit rejected") from None
                    corrections += 1
                    break
                emit("tool_completed", call=calls, tool=name, output=output)
                if name in EVIDENCE_TOOLS:
                    unseen = True
            if committed is None and final:
                raise br.BudgetExhausted("final request ended without commit")
        emit("committed", plan_id=committed.plan_id, reason=next(r["output"]["reason"] for r in reversed(trace) if r["event"] == "tool_completed" and r["tool"] == "commit"))
        return finish("COMPLETE")
    except br.BudgetExhausted as exc:
        committed = None
        emit("job_incomplete", failure_kind="BUDGET_EXHAUSTED", reason=str(exc))
        return finish("INCOMPLETE", "BUDGET_EXHAUSTED", str(exc))
    except Exception as exc:  # noqa: BLE001
        committed = None
        kind = ("AGENT_CALL_FAILED" if stage == "AGENT_CALL" else "PARSE_OR_VALIDATION_FAILED" if stage == "RESPONSE_VALIDATION" else "RUNTIME_FAILED")
        emit("job_incomplete", failure_kind=kind, reason=type(exc).__name__, message=str(exc)[:300] if kind == "RUNTIME_FAILED" else None)
        return finish("INCOMPLETE", kind, type(exc).__name__)
