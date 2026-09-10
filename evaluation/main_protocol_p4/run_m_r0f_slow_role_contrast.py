"""Step 2 of the contrast: change exactly one factor -- the Slow role
instruction -- and ask the same question again.

Step 1 (``audit_m_r0e_slow_input_truncation``) settled the input branch: the
``rows[:60]`` slice the model actually received still supports nine feasible
stumps and the *same* winner as the full set, so truncation did not remove the
signal.  The branch rule therefore selects the role instruction as the factor
to test.

**One factor.**  Held identical to the attempt-3 run: the model, the transport,
the schema (``slow_scope_clause_v1``), the frozen vocabulary, the evidence rows
*including the 60-row truncation*, the ``public_input`` body byte-for-byte, the
thresholds, the policy, the replay screen, the candidate space and the Draft
state.  Only ``harness_view.instruction`` differs.

What changes and why: the K0 snapshot resolves the **inner-loop TTHA
preparation Agent** instruction for ``role="slow"``.  That text forbids
inferring *candidate utility*, asks for *PROGRAM* candidates and an *identity*
selection, requires an *inspect* stage this call does not have, and closes with
"abstain when public evidence does not justify a repair".  The outer step asks
for a Scope clause read off per-series gains in a single edit call.  The
replacement states the outer role in the same register.  It does **not** name a
feature, a direction or a threshold, does **not** report what the shadow found,
and does **not** remove the abstain envelope -- ``no_proposal`` stays available
and reachable, so an abstention here is still a legal answer.

The snapshot is **not** modified: the override is applied at the call seam by
wrapping ``core.run_stage`` and replacing the resolved view's ``instruction``
field for this process only.  No store is written, no SHA is minted, and the
receipt records both texts and both digests so the run can never be read as if
the snapshot had said this.

Ceiling: **2 physical Slow calls, 10 Consumer fits**, enforced before the spend
by the same guard and budget as layer 1.  Still layer 1: no later unit is
touched, nothing is verified on a new unit, no Skill is activated.

Run:  python -m evaluation.main_protocol_p4.run_m_r0f_slow_role_contrast
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

import numpy as np

from evaluation.main_protocol_p4 import audit_m_r0_reachability as base
from evaluation.main_protocol_p4 import hec1_contract as contract
from evaluation.main_protocol_p4 import outer_loop
from evaluation.main_protocol_p4 import restricted_draft as drafts
from evaluation.main_protocol_p4 import run_hec1 as runner
from evaluation.main_protocol_p4 import run_m_r0d_forward_k1_outer_step as live
from evaluation.main_protocol_p4 import run_source_line as v1runner

ART = base.ART
MAX_SLOW_CALLS = live.MAX_SLOW_CALLS
MAX_CONSUMER_FITS = live.MAX_CONSUMER_FITS

#: The one factor.  Written out in full rather than assembled from fragments so
#: that what was sent is readable in the source and in the receipt.
OUTER_ROLE_INSTRUCTION = (
    "You are the outer-loop Scope Agent for a program that is already "
    "deployed under a serving-scope predicate.\n"
    "The public evidence you are given is that program's own per-series gain "
    "on units the deployment has already processed. Reading those gains is "
    "your job: they are public measurements, not private rankings.\n"
    "The program is fixed. You do not propose programs, you do not choose "
    "between candidates, and there is no identity option and no inspect "
    "stage: this is a single edit call whose only output is one scope clause.\n"
    "Name ONE deployment-visible feature and a direction whose conjunction "
    "with the current scope would leave the damaged series outside it. Do not "
    "choose a threshold; the runtime calibrates it on frozen bins and ignores "
    "whatever you put there.\n"
    "Use only the public observations, the retrieved Harness content and the "
    "typed output schema. If the public evidence genuinely supports no such "
    "feature, return the no_proposal envelope instead."
)


def _sha(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


class _RoleOverrideCore:
    """Replaces the resolved view's instruction on the way into ``run_stage``.

    Everything else about the request is untouched, including
    ``source_snapshot_sha``: the snapshot really is the one named, and the
    receipt says separately that its instruction field was overridden for this
    contrast.  Wrapping the core rather than editing the store is what keeps
    this a development contrast instead of a new snapshot.
    """

    def __init__(self, inner: Any, instruction: str) -> None:
        self.inner = inner
        self.instruction = str(instruction)
        self.seen: list[dict[str, Any]] = []

    def run_stage(self, **kwargs: Any) -> Any:
        view = kwargs.get("harness_view")
        original = getattr(view, "instruction", None)
        if view is not None:
            kwargs["harness_view"] = dataclasses.replace(
                view, instruction=self.instruction)
        self.seen.append({
            "stage": kwargs.get("stage"),
            "role": str(kwargs.get("role")),
            "original_instruction_sha": _sha(original) if original else None,
            "sent_instruction_sha": _sha(self.instruction),
            "skills_in_view": list(getattr(view, "skill_ids", ()) or ()),
            "effective_harness_view_sha_is_stale": True,
            "why_stale": ("the view's own sha was computed for the snapshot's "
                          "instruction; only the instruction field was "
                          "replaced, and no new sha was minted"),
        })
        return self.inner.run_stage(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _slow_with_override(guard: runner.BudgetGuard, snapshot: Any,
                        instruction: str):
    machinery = v1runner._machinery()
    target = machinery["agentic"].live_transport()
    inner = machinery["agentic"]._default_backend_factory(MAX_SLOW_CALLS)
    metered = runner._MeteredOuterBackend(inner, guard=guard, billable=True)
    core = machinery["TTHAAgentCore"](
        metered,
        machinery["LocalPublicToolGateway"](np.zeros(8, dtype=np.float64),
                                            task_kind="forecast"),
        model=target["model"], base_url=target["base_url"])
    shim = _RoleOverrideCore(core, instruction)
    agent = runner.OuterSlowAgent(
        shim, vocabulary=contract.SCOPE_CLASS["vocabulary"],
        guard=guard, snapshot=snapshot)
    return agent, metered, target, shim


def _original_instruction(snapshot: Any) -> str:
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: PLC0415
        resolve_harness_view,
    )
    return str(resolve_harness_view(snapshot, {}, role="slow").instruction)


def build(label: str) -> dict[str, Any]:
    started = datetime.now(timezone(timedelta(hours=8)))
    doc = base.load(live.ORDERING)
    state = live._state_at_k1(doc)

    machinery = v1runner._machinery()
    k0 = state["k0"]
    course_k0 = dict(doc.get("k0_snapshot") or {})
    candidates = [
        (course_k0.get("store_root"), course_k0.get("runtime_bundle_sha"),
         "course artifact k0_snapshot"),
        (k0.get("store_root"), k0.get("runtime_bundle_sha"),
         "Phase S K0 receipt"),
    ]
    snapshot_dir = snapshot_source = None
    for store_root, sha, name in candidates:
        if not (store_root and sha):
            continue
        path = base.ROOT / str(store_root) / str(sha)
        try:
            readable = path.is_dir() and (path / "snapshot.lock.json").exists()
        except OSError:
            readable = False
        if readable:
            snapshot_dir, snapshot_source = path, name
            break
    if snapshot_dir is None:
        return {"status": "BLOCKED", "stage": "M_R0F_SLOW_ROLE_CONTRAST",
                "why": "no readable K0 snapshot",
                "actual_cost": {"physical_slow_calls": 0,
                                "consumer_fits_replay": 0}}
    if (course_k0.get("runtime_bundle_sha") and k0.get("runtime_bundle_sha")
            and course_k0["runtime_bundle_sha"] != k0["runtime_bundle_sha"]):
        return {"status": "BLOCKED", "stage": "M_R0F_SLOW_ROLE_CONTRAST",
                "why": "the course and the receipt name different K0 bundles",
                "actual_cost": {"physical_slow_calls": 0,
                                "consumer_fits_replay": 0}}
    snapshot = machinery["compile_snapshot"](snapshot_dir, verify_lock=False)
    original = _original_instruction(snapshot)

    ledgers = runner.Ledgers()
    guard = runner.BudgetGuard(
        ordering_cap=MAX_SLOW_CALLS,
        per_unit_arm_cap=int(contract.PER_UNIT_ARM_BUDGET["llm_calls"]),
        ledgers=ledgers)
    slow, metered, target, shim = _slow_with_override(
        guard, snapshot, OUTER_ROLE_INSTRUCTION)
    cache = runner.ReplayPredictionCache(live.ARM)
    replay = live._capped_replay(state["processed"], ledgers, cache)
    budget = outer_loop.OuterBudget(
        outer_llm_per_step=int(contract.OUTER_LLM_PER_STEP),
        replay_fits_remaining=MAX_CONSUMER_FITS)

    fault = None
    try:
        record = outer_loop.consolidate(
            bank=state["bank"], ledger=state["ledger"], k_index=live.K_INDEX,
            slow=slow, replay=replay, budget=budget,
            held_lineage_keys=state["held"],
            bank_boundary={"arm": live.ARM,
                           "units": [ctx.unit for ctx in state["processed"]],
                           "excludes": ["the evaluation face (+144)",
                                        "future units", "other arms",
                                        "held-out"]})
        step = record.to_dict()
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed
        fault = "%s: %s" % (type(exc).__name__, str(exc)[:400])
        step = None

    after = [{"draft_id": d.draft_id, "state": d.state, "revisions": d.revisions,
              "verification_attempts": d.verification_attempts,
              "closed": d.closed, "census_key": d.census_key,
              "current_scope": dict(d.current_scope),
              "root_scope": dict(d.root_scope),
              "clauses_added_since_root": d.clauses_added_so_far(),
              "revision_history": [dict(r) for r in d.revision_history],
              "history_events": [r.get("event") for r in d.history],
              "deployable": d.deployable}
             for d in state["ledger"].drafts]
    before = state["drafts_before"]
    target_draft = next((r for r in after
                         if r["draft_id"] == "resupplied_draft_1"), None)
    before_draft = next((r for r in before
                         if r["draft_id"] == "resupplied_draft_1"), None)

    return {
        "stage": "M_R0F_SLOW_ROLE_CONTRAST",
        "layer": "1 of 2 -- one outer step; no later unit is touched",
        "run_label": label,
        "written_at": started.isoformat(),
        "finished_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "the_one_factor": {
            "changed": "harness_view.instruction for role=slow",
            "applied_at": ("the call seam: _RoleOverrideCore wraps "
                           "core.run_stage and replaces the resolved view's "
                           "instruction field"),
            "snapshot_modified": False,
            "new_sha_minted": False,
            "original": original,
            "original_sha256": _sha(original),
            "sent": OUTER_ROLE_INSTRUCTION,
            "sent_sha256": _sha(OUTER_ROLE_INSTRUCTION),
            "abstention_still_available": (
                "the no_proposal envelope is offered by agent_core for the "
                "slow edit stage regardless of this text, and the replacement "
                "names it explicitly"),
            "not_told_to_the_model": [
                "any feature name from the vocabulary",
                "any direction or threshold",
                "what the deterministic shadow found",
                "that a previous call abstained",
            ],
        },
        "held_identical_to_attempt3": [
            "model and transport", "slow_scope_clause_v1 schema",
            "the frozen 12-name vocabulary",
            "public_input body, including the rows[:60] truncation",
            "policy, thresholds, candidate space, replay screen",
            "the Draft state at k=1 and the Active lineage set",
        ],
        "authorised_ceiling": {"slow_calls": MAX_SLOW_CALLS,
                               "consumer_fits": MAX_CONSUMER_FITS},
        "target": {"ordering": live.ORDERING, "arm": live.ARM,
                   "k_index": live.K_INDEX,
                   "boundary_position": live.BOUNDARY_POSITION,
                   "units_in_bank": len(state["processed"]),
                   "bank_rows": len(state["bank"]),
                   "held_lineage_keys": state["held"]},
        "transport": {"base_url": target["base_url"], "model": target["model"],
                      "source": target["source"]},
        "k0_snapshot": {"path": str(snapshot_dir.relative_to(base.ROOT)),
                        "resolved_from": snapshot_source,
                        "runtime_bundle_sha": str(
                            course_k0.get("runtime_bundle_sha")
                            or k0.get("runtime_bundle_sha"))},
        "code_state": base.version_check(),
        "drafts_before": before,
        "drafts_after": after,
        "state_written": {
            "revisions": ((before_draft or {}).get("revisions"),
                          (target_draft or {}).get("revisions")),
            "current_scope_before": (before_draft or {}).get("current_scope"),
            "current_scope_after": (target_draft or {}).get("current_scope"),
            "current_scope_changed": (
                (before_draft or {}).get("current_scope")
                != (target_draft or {}).get("current_scope")),
            "verification_attempts_unchanged": (
                (before_draft or {}).get("verification_attempts")
                == (target_draft or {}).get("verification_attempts")),
            "state_unchanged": ((before_draft or {}).get("state")
                                == (target_draft or {}).get("state")),
            "no_second_shell": len(after) == len(before),
            "still_not_deployable": all(not r["deployable"] for r in after),
        },
        "outer_step": step,
        "run_fault": fault,
        "slow_calls_recorded_by_the_agent": slow.calls,
        "requests_seen_by_the_override": shim.seen,
        "actual_cost": {
            "physical_slow_calls": int(metered.calls),
            "billed_llm_outer": ledgers.llm_outer,
            "billed_llm_fast": ledgers.llm_fast,
            "consumer_fits_replay": ledgers.replay_fits,
            "replay_cache": cache.to_dict(),
            "within_ceiling": (int(metered.calls) <= MAX_SLOW_CALLS
                               and ledgers.replay_fits <= MAX_CONSUMER_FITS),
            "blocked_before_backend": guard.blocked,
        },
        "boundary": {"later_units_touched": 0, "skills_activated": 0,
                     "held_out_reads": 0, "sealed_reads": 0,
                     "evaluation_face_reads": 0, "thresholds_changed": 0,
                     "production_code_changed": 0,
                     "snapshot_stores_written": 0},
        "what_this_cannot_settle": [
            "n=1 against n=1: one abstention and one answer under two "
            "instructions is not an effect size",
            "whether any clause produced here helps on a later unit -- that is "
            "layer 2 and is not run here",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    label = "run1"
    if "--label" in argv:
        label = argv[argv.index("--label") + 1]
    report = build(label)
    ART.mkdir(parents=True, exist_ok=True)
    out = ART / ("m_r0f_slow_role_contrast__%s.json" % label)
    if out.exists():
        print("refusing to overwrite %s" % out)
        return 2
    out.write_text(json.dumps(drafts._plain(report), indent=1,
                              ensure_ascii=False, default=str),
                   encoding="utf-8")
    print("wrote %s" % out)
    print(json.dumps({"slow": report.get("slow_calls_recorded_by_the_agent"),
                      "state_written": report.get("state_written"),
                      "cost": report.get("actual_cost"),
                      "fault": report.get("run_fault")},
                     ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
