"""One continuous operational run of the V1 Harness, no human relay.

The single question: can the Harness, in one uninterrupted run, go from
using its Source Guidance, through adapting to a target, forming and
promoting a Target-local Skill, recalling it, failing on a later window,
attributing that failure, patching one Scope/Risk surface through Slow, and
reading the patched Harness back on the next task -- with no hand-written
card, no store swap, no hand-named Workflow and no mid-run restart.

Development level.  Every window is quoted verbatim from the #17/#19
registers, locked before the run; nothing beyond index 17520 is read; no
A5-vs-A3 comparison is re-estimated and no new method is introduced.  This
module is orchestration only: every measurement, every prompt, every gate
and every store write is an existing tracked or delivered implementation
called from here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import traceback
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

import numpy as np  # noqa: E402
import run_batch_composition_headroom as bch  # noqa: E402
import run_e2_fresh_confirmation as FC  # noqa: E402
import run_e2_recipe_experience_to_skill as bridge  # noqa: E402
import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_slow_scope_update as SSU  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    AgentRole,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentTransportError,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    EVIDENCE_SUPPORT,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.harness import (  # noqa: E402
    compiler as harness_compiler,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (  # noqa: E402
    compile_snapshot,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.store import (  # noqa: E402
    SnapshotStore,
)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import (  # noqa: E402
    resolve_harness_view,
)

PROTOCOL_VERSION = "operational_pipeline_v2"
PROTOCOL_VERSION_MULTI = "operational_pipeline_v3"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
# v1 (#21) and v2 (#22) are their runs' records and are never rewritten.
OUT_JSON = E2 / "operational_pipeline_v2.json"
OUT_MD = E2 / "operational_pipeline_v2.md"
OUT_JSON_MULTI = E2 / "operational_pipeline_v3.json"
OUT_MD_MULTI = E2 / "operational_pipeline_v3.md"

WORK_ROOT = PROJECT_ROOT / "_scratch" / "operational_v1"
STORE_ROOT = WORK_ROOT / "store"
GATE_ROOT = WORK_ROOT / "gate_stores"
SLOT = "pipeline"

VARIANT = bch.CONSUMER_POOLED
ARM = "A5"
COHORT_NAME = FC.COHORT_NAME
IDENTITY = FC.IDENTITY
MATERIAL_THRESHOLD = float(FC.MATERIAL_THRESHOLD)
HARM_THRESHOLD = float(FC.HARM_THRESHOLD)
CARD_ID = FC.SKILL_ID[VARIANT]
SLOW_MODEL = "claude-opus-5"
SLOW_BASE_URL = ssi.NF_BASE_URL
NEW_SURFACE = SSU.NEW_SURFACE_V2

# Every window is quoted from the register, locked here before the run.
# Nothing below is chosen after seeing a reading.
WINDOW_TASK_A = FC.WINDOWS["task_A"]
WINDOW_PROBE = FC.PROBE_WINDOW
WINDOW_TASK_B = FC.WINDOWS["task_B"]
WINDOW_TASK_C = FC.WINDOWS["task_C"]
WINDOW_TASK_D = FC.WINDOWS["task_D"]

LLM_CALL_BUDGET_TOTAL = 20
LLM_CALL_BUDGET_PER_EPISODE = int(FC.LLM_CALL_BUDGET_PER_EPISODE)
RETRAIN_BUDGET = 300
# K = 3 trajectories share one budget; a trajectory that cannot start is
# reported as not started rather than squeezed.
LLM_CALL_BUDGET_MULTI = 24
RETRAIN_BUDGET_MULTI = 450
MAX_TRANSPORT_FAILURES = 3
# The three counters #25 keeps apart.  A draw that never produced a decision
# the protocol can read is not a decision sample, and a transport failure is
# neither.  The protocol-failure count is cumulative across runs on this
# surface: #24 already spent one.
MAX_VALID_DECISION_SAMPLES = 2
MAX_PROTOCOL_FAILED_DRAWS = 2
# Carry-in is per model AND per question.  claude-opus-5 spent two
# protocol-failed draws (#24, #25) on this exact surface with this exact
# evidence, so it starts exhausted.  A model that has not been asked this
# question starts at zero: gpt-5.6-luna's two #19 abstentions were on the
# same surface but a different public_input (the #17 pooled task_C episodes,
# not the #23 T1 task_A episode), so they are context, not spent samples.
PROTOCOL_FAILED_CARRY_IN: dict[str, int] = {"claude-opus-5": 2}
PRIOR_PROTOCOL_FAILED_DRAWS = PROTOCOL_FAILED_CARRY_IN.get(SLOW_MODEL, 0)
MODEL_HISTORY_ON_THIS_SURFACE: dict[str, str] = {
    "claude-opus-5": (
        "produced a valid proposal in #19 and #21; #24 and #25 both came back "
        "as protocol failures that the 2026-08-22 probe traced to a relay "
        "outage, not to the model"
    ),
    "gpt-5.6-luna": (
        "two valid no_proposal draws in #19 on this same surface, reason "
        "insufficient_public_evidence, but on the #17 pooled task_C evidence "
        "rather than this question; not counted against this round"
    ),
    "gpt-5.6-sol": "never asked this stage before",
}
SAMPLING_RULE = (
    "at most %d valid decision samples, stopping at the first proposal; a "
    "protocol-failed draw consumes no decision sample but does count against "
    "a cumulative cap of %d across runs on this surface (#24 spent %d); a "
    "transport failure consumes neither and three in a row stop the run."
    % (MAX_VALID_DECISION_SAMPLES, MAX_PROTOCOL_FAILED_DRAWS,
       PRIOR_PROTOCOL_FAILED_DRAWS)
)


def _repo_rel(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --------------------------------------------------------------- the freeze
COMPILER_FILE = "methods/ttha/harness/compiler.py"
H0_LOCK_FILE = "methods/ttha/harness/h0/snapshot.lock.json"
FROZEN_SURFACE_V3: tuple[str, ...] = tuple(SSU.FROZEN_SURFACE_V2) + (
    "artifacts/functional/e2/slow_scope_update_v1.json",
    "artifacts/functional/e2/slow_scope_update_v2.json",
    "evaluation/functional/run_e2_slow_scope_update.py",
)
# A1 (#22): the envelope validator now enforces maxLength, symmetric with
# its own minLength.  The file feeds dependency_shas under
# ttha:schema_contracts, so every snapshot identity moves; no measurement
# does.
SCHEMA_CONTRACTS_FILE = "methods/ttha/schema_contracts.py"
RUNNER_FILE = "evaluation/functional/run_e2_operational_pipeline.py"
FROZEN_SURFACE_V4: tuple[str, ...] = FROZEN_SURFACE_V3 + (SCHEMA_CONTRACTS_FILE,)
# v5 adds this runner, now tracked, so a concurrent edit to the thing
# driving the protocol is caught like any other frozen member.
FROZEN_SURFACE_V5: tuple[str, ...] = FROZEN_SURFACE_V4 + (RUNNER_FILE,)
# v6 == v5: #25's only file was the runner, already in.
FROZEN_SURFACE_V6: tuple[str, ...] = FROZEN_SURFACE_V5
# v7 adds the relay backend, which fix (c) touches.
AGENT_BACKEND_FILE = "runtime/agent_backend.py"
FROZEN_SURFACE_V7: tuple[str, ...] = FROZEN_SURFACE_V6 + (AGENT_BACKEND_FILE,)
FIX_C_TOUCHED: tuple[dict[str, str], ...] = (
    {
        "path": AGENT_BACKEND_FILE,
        "role": (
            "fix (c): a relay HTTP 200 whose body carries an error object and "
            "no choices is raised as AgentTransportError instead of being "
            "handed on as a transport-success empty completion"
        ),
        "sha256_before_fix_c": (
            "15bfae49cd13c6ef6526560ea27e2e9dde10600309f012ee1e5a4be8e8b3dda7"
        ),
        "line_ending_caliber": (
            "working-tree bytes; this file is CRLF on disk, so its committed "
            "LF blob hashes differently "
            "(f7320368f3ca86f4e9b9366153af88fd0fcf7ecf93e8737d8f820548c9d3f81c "
            "before the fix)"
        ),
        "diff_shape": "+52/-0, purely additive",
    },
    {
        "path": H0_LOCK_FILE,
        "role": (
            "regenerated again: runtime/agent_backend.py feeds "
            "dependency_shas under runtime:agent_backend.  Exactly that one "
            "key moved and harness_content_sha is unchanged; h0 passes "
            "verify_lock=True."
        ),
        "sha256_before_fix_c": (
            "5be7bddc92a2928701bc11408082d3e0f48f0703843a13c0d49e0491c1f71c4d"
        ),
    },
)
RELAY_OUTAGE_ON_RECORD: dict[str, Any] = {
    "measured": "2026-08-22, three diagnostic calls plus three raw HTTP posts",
    "what_the_relay_returns": (
        'HTTP 200 with body {"error":{"code":"api_error","message":"Service '
        'load is too high, please try again later","type":"api_error"}} and '
        "no choices"
    ),
    "scope": (
        "the whole Claude family on this relay -- claude-opus-5, "
        "claude-opus-4-6 and claude-sonnet-5 all returned it, while "
        "gpt-5.6-luna answered normally in the same minute "
        "(finish_reason=stop, 4438/5 tokens)"
    ),
    "not_request_shape": (
        "a 44-character prompt and a 60000-character prompt returned the same "
        "empty result, so it is neither the Slow prompt nor a context ceiling"
    ),
    "model_id_is_valid": (
        "claude-opus-5 is listed by the relay's own /v1/models catalog "
        "(286 entries)"
    ),
    "why_it_masqueraded": (
        "the SDK parses the 200, the backend read empty choices and built "
        "AgentResponse(transport_ok=True, assistant_text=''), _RetryingTransport "
        "saw no exception to retry, and agent_core spent its two static "
        "feedback retries on empty strings.  #24 and #25 recorded that as "
        "SLOW_ENVELOPE_PROTOCOL_FAILURE."
    ),
    "consequence_for_the_record": (
        "both protocol-failed draws on this surface were an upstream outage, "
        "not the Agent.  The cumulative cap of 2 was consumed by it."
    ),
}
A1_TOUCHED: tuple[dict[str, str], ...] = (
    {
        "path": SCHEMA_CONTRACTS_FILE,
        "role": (
            "A1: maxLength is enforced where minLength already was, so a "
            "declared string cap is taught by the retry loop instead of "
            "dying at a gate with no second attempt"
        ),
        "sha256_before_a1": (
            "08b421d7ca4c6300052af4a78105548559a0c6be002e6aead7aac15537d936d5"
        ),
    },
)


# P1 moved these two and only these two; both sides are on the record.
P1_TOUCHED: tuple[dict[str, str], ...] = (
    {
        "path": COMPILER_FILE,
        "role": (
            "P1: the Scope/Risk enforcement walk is promoted next to the "
            "evaluator, so reading a guard and obeying it are both tracked "
            "machinery.  Measurement stays with the caller."
        ),
        "sha256_before_p1": (
            "fd57945185475fc473c72bb833b6d60f6332c1f46bf5efdaf80b5f686f7b8973"
        ),
    },
    {
        "path": H0_LOCK_FILE,
        "role": (
            "regenerated: the lock pins the compiler identity, and P1 moved "
            "the compiler.  harness_content_sha is unchanged; only the "
            "compiler-derived shas move."
        ),
        "sha256_before_p1": (
            "bbddc14b14cb04f50047c8a063d546a2ed6d4ab8c87c08cf418004e15888ecad"
        ),
    },
)


class ConcurrentWrite(RuntimeError):
    """The frozen surface moved while this protocol was running."""


class Blocked(RuntimeError):
    """A pre-registered stop: the first block ends the run."""

    def __init__(self, verdict: str, reason: str) -> None:
        super().__init__(reason)
        self.verdict = verdict
        self.reason = reason


def _freeze() -> dict[str, str]:
    frozen: dict[str, str] = {}
    for name in sorted(set(FROZEN_SURFACE_V7)):
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise SystemExit("frozen surface member is missing: %s" % name)
        frozen[name] = _sha256(path)
    return frozen


def _verify(before: Mapping[str, str]) -> dict[str, Any]:
    drift = [
        name for name, sha in before.items()
        if not (PROJECT_ROOT / name).is_file()
        or _sha256(PROJECT_ROOT / name) != sha
    ]
    return {"files": len(before), "drift": drift, "ok": not drift}


def _guard_frozen(before: Mapping[str, str], where: str) -> bool:
    report = _verify(before)
    if not report["ok"]:
        raise ConcurrentWrite(
            "the frozen surface moved before %s: %s" % (where, report["drift"])
        )
    return True


class Budget:
    def __init__(self, llm: int, retrains: int) -> None:
        self.llm_total = int(llm)
        self.llm_used = 0
        self.retrain_total = int(retrains)
        self.retrains_charged = 0

    def take(self, want: int) -> int:
        return max(0, min(int(want), self.llm_total - self.llm_used))

    def spend(self, calls: int) -> None:
        self.llm_used += int(calls)
        if self.llm_used > self.llm_total:
            raise Blocked(
                "BUDGET_EXCEEDED", "the LLM call budget was exhausted mid-run"
            )

    def charge_retrains(self, count: int) -> None:
        self.retrains_charged += int(count)
        if self.retrains_charged > self.retrain_total:
            raise Blocked(
                "BUDGET_EXCEEDED",
                "the Consumer retrain budget was exhausted mid-run (%d > %d)"
                % (self.retrains_charged, self.retrain_total),
            )

    @property
    def left(self) -> int:
        return self.llm_total - self.llm_used

    @property
    def retrains_left(self) -> int:
        return self.retrain_total - self.retrains_charged


# ----------------------------------------------------- what was fixed first
PRE_REGISTERED: dict[str, Any] = {
    "question": (
        "can the V1 Harness complete Source use -> target adaptation -> "
        "lifecycle -> failure attribution -> Slow edit -> next-task behaviour "
        "reading in one continuous run with no human relay"
    ),
    "level": (
        "development.  Nothing beyond index 17520 is opened, A5-vs-A3 is not "
        "re-estimated, and no new method is introduced."
    ),
    "p0_5": {
        "P1": (
            "the #19 reference enforcement (_fires/_check/_enforce) is "
            "promoted into the tracked adoption path: discovery, evaluation, "
            "the fallback walk and the identity floor now all live in "
            "methods/ttha/harness/compiler.py, with the Consumer measurement "
            "injected by the caller.  Guard semantics and v2 ladder semantics "
            "are unchanged."
        ),
        "P2": (
            "hard gate: with an empty guard list the four #19 task_C episodes "
            "must reproduce digit-for-digit through the promoted path; "
            "otherwise INSTRUMENT_DRIFT and nothing else runs"
        ),
        "P3": (
            "data/benchmark_noaa_fresh_v1/manifest.json is un-ignored by a "
            "negative pattern so the frozen-surface member has a tracked "
            "baseline; the arrays it describes stay ignored"
        ),
        "P4": (
            "both P1-touched tracked files carry pre/post sha256 in the v3 "
            "frozen list; every other member is checked for zero drift"
        ),
    },
    "fixed_configuration": {
        "cohort": COHORT_NAME,
        "consumer_variant": VARIANT,
        "arm": ARM,
        "slow_backend": SLOW_MODEL,
        "slow_sampling": (
            "one pinned configuration, one draw; no cross-backend sampling.  "
            "An abstention or a compiler rejection stops the run."
        ),
        "store": (
            "one new continuous store: bootstrap plus the all-source pooled "
            "Guidance card compiled from the same input as #17, with "
            "scope_risk_guards starting as the empty list"
        ),
        "frozen": (
            "Judge, Metric, program menu, Guidance content, lifecycle "
            "thresholds, harm line and the v2 adoption ladder are untouched"
        ),
    },
    "window_order": [
        {"step": "task_A", "start": WINDOW_TASK_A["start"],
         "role": "full-price adaptation"},
        {"step": "probe", "origins": list(WINDOW_PROBE["origins"]),
         "role": "out of selection; DRAFT -> ACTIVE"},
        {"step": "task_B", "start": WINDOW_TASK_B["start"], "role": "recall and reuse"},
        {"step": "task_C", "start": WINDOW_TASK_C["start"],
         "role": "already-exposed 2025 region; #17's known natural harm sample"},
        {"step": "attribution", "role": "first fault, on this run's own episode records only"},
        {"step": "slow_edit", "role": "one proposal, one surface, compiler-validated"},
        {"step": "task_D", "start": WINDOW_TASK_D["start"],
         "role": ("time-order successor; #17 never used it; the missing "
                  "gate reads the NaN mask only")},
    ],
    "behavior_change_criterion": {
        "precondition": "the guard is evaluated at task_D, with trace evidence",
        "i": "task_D fires the guard on the spot and walks the ladder fallback",
        "ii": (
            "if the harm does not recur, a 0-evaluation shadow replay of this "
            "run's own task_C under the post-patch snapshot decides "
            "differently from the pre-patch decision, consistent with the "
            "Risk contract"
        ),
        "not_a_failure": (
            "harm that does not recur while the guard correctly stays silent "
            "is the contract working, not a failure"
        ),
    },
    "verdicts": [
        "OPERATIONAL_PIPELINE_CLOSES", "SOURCE_DELIVERY_BREAK",
        "LOCAL_LIFECYCLE_BREAK", "ATTRIBUTION_BREAK", "SLOW_ABSTAINS",
        "COMPILER_REJECTS", "POST_UPDATE_NOT_RETRIEVED",
        "NEXT_TASK_NO_BEHAVIOR_CHANGE", "NO_POST_UPDATE_WINDOW",
        "INCONCLUSIVE_TRANSPORT", "INSTRUMENT_DRIFT",
        "CONCURRENT_WRITE_ABORT", "PIPELINE_RUNS_NO_FAULT_SAMPLE",
    ],
    "no_fault_sample": (
        "if the trajectory reaches task_D without ever producing a RISK_GAP "
        "-- no evaluation series crossed the harm line, so the fold has "
        "nothing actionable to name -- links 6 to 9 are recorded untested in "
        "this sample.  Nothing is re-thrown and no harm is seeded."
    ),
    "by_veto_passes": (
        "a BY_VETO shape passes: this round tests continuity.  Keeping the "
        "aggregate gain is a BY_RESCOPE round's question."
    ),
    "discipline": {
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "retrain_budget": RETRAIN_BUDGET,
        "first_block_stops": True,
        "no_second_method_repair": True,
        "commit": False,
        "spawn": False,
        "beyond_17520": "zero reads",
    },
    "notes_go_in_the_artifact": (
        "every caliber note, every deviation and every ambiguity is written "
        "into the artifact rather than left in a source comment"
    ),
    "caliber_notes": {
        "worktree_byte_hashes": (
            "all sha256 values here are working-tree byte hashes.  "
            "harness_surfaces.json is CRLF on disk, so its committed LF blob "
            "hashes differently; that is a line-ending caliber, not drift."
        ),
        "manifest_baseline": (
            "data/benchmark_noaa_fresh_v1/manifest.json had no tracked "
            "baseline before P3; its zero-drift check was previously "
            "within-run only"
        ),
        "h0_lock_left_as_p1_left_it": (
            "A1 changes methods/ttha/schema_contracts.py, which feeds "
            "dependency_shas under ttha:schema_contracts, so h0's lock is "
            "now stale in exactly that one key (harness_content_sha is "
            "unchanged).  It was deliberately NOT regenerated: every runner "
            "here compiles with verify_lock=False, and regenerating would "
            "move h0/snapshot.lock.json -- a P1-touched file whose "
            "byte-identity is the stated precondition for not re-running the "
            "P2 gate.  The regenerated lock would hash to "
            "5be7bddc92a2928701bc11408082d3e0f48f0703843a13c0d49e0491c1f71c4d; "
            "the next checkpoint can apply it in one deterministic step."
        ),
        "p2_gate_could_not_be_re_run_even_if_wanted": (
            "run_e2_slow_scope_update._open_stores_v2 hard-asserts that the "
            "dependency drift is exactly [compiler_source, surface_registry].  "
            "A1 adds ttha:schema_contracts, so the delivered gate would "
            "SystemExit.  Skipping it is forced, not merely economical, and "
            "the justification stands on A1-A3 all sitting on the proposal "
            "and reporting paths."
        ),
        "measurement_stays_outside_the_harness": (
            "the promoted gate injects delayed_of/support_of rather than "
            "importing an instrument: the Harness decides, it does not "
            "measure"
        ),
    },
}


# ------------------------------------------------------------- P2 hard gate
P1_POST_SHA = {
    COMPILER_FILE: (
        "61cc3b4538baa17a0b1bbe8458c86a224f884167ded56296f8a87629b0ae5eb5"
    ),
    # Regenerated twice on purpose, never drifted: once by O1 after A1 moved
    # ttha:schema_contracts, and once by fix (c) after runtime:agent_backend
    # moved.  Each time exactly the expected dependency key changed,
    # harness_content_sha stayed put, and h0 passes verify_lock=True.
    H0_LOCK_FILE: (
        "327f01186194cb15a91859c4260d0349f77b04dabf5db0689c3c663c9b4c046f"
    ),
}


def stage_p2_precondition() -> dict[str, Any]:
    """A4: the gate is not re-run, and the reason is checked, not assumed.

    #21 ran the four-episode gate and it passed 4/4.  This round's fixes all
    sit on the proposal and reporting paths, so the gate would re-measure
    nothing -- but that argument is only good while the measurement side is
    byte-identical, so the two P1-touched files are checked here.  If either
    moved, the gate would have to be re-run, and it cannot be: the delivered
    gate hard-asserts that the dependency drift is exactly
    [compiler_source, surface_registry], and A1 adds a third key.
    """
    observed = {name: _sha256(PROJECT_ROOT / name) for name in P1_POST_SHA}
    moved = sorted(
        name for name, sha in P1_POST_SHA.items() if observed[name] != sha
    )
    out = {
        "ran": False,
        "role": (
            "A4: the #21 four-episode gate is not re-run; its precondition is "
            "verified instead"
        ),
        "gate_result_carried_forward": (
            "operational_pipeline_v1: 4/4 #19 task_C episodes reproduced "
            "digit-for-digit through the promoted enforcement path, 111 "
            "retrains, 0 LLM"
        ),
        "precondition": {
            "expected_post_p1_sha256": dict(P1_POST_SHA),
            "observed_sha256": observed,
            "byte_identical": not moved,
            "moved": moved,
        },
        "why_it_cannot_be_re_run_anyway": (
            "run_e2_slow_scope_update._open_stores_v2 exits unless the "
            "dependency drift is exactly [compiler_source, surface_registry].  "
            "A1 added ttha:schema_contracts and fix (c) adds "
            "runtime:agent_backend, so the delivered gate is now two keys "
            "past its own assertion.  That file is committed and is not "
            "edited to accommodate this; the #21 gate result is carried "
            "forward on the precondition below instead."
        ),
        "consumer_retrains": 0,
        "llm_calls": 0,
    }
    if moved:
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the measurement-side files this round promised not to touch "
            "moved: %s; the four-episode gate would be required and the "
            "delivered gate can no longer run it" % moved,
        )
    return out


def stage_p2_gate(budget: Budget) -> dict[str, Any]:
    """The four #19 task_C episodes, replayed through the promoted path.

    Everything here is #19's own code, called with its store roots pointed
    at this run's scratch namespace so the audited v2 fork is left alone.
    """
    started = time.perf_counter()
    SSU.WORK_ROOT_V2 = GATE_ROOT
    SSU.STORE_ROOT_V2 = GATE_ROOT / "stores"
    stores = SSU._open_stores_v2()
    migration = SSU.stage_migrate_v2(stores)
    episodes = SSU._load_episodes_v2()
    cohort = SSU._confirmation_payload(episodes["locked_roster"])
    gate = SSU.stage_gate_v2(
        cohort=cohort, episodes=episodes, stores=stores,
        window=episodes["window"], budget=budget,
    )
    cells = {
        slot: {
            "reproduces": bool(row["reproduces"]),
            "mismatched": list(row.get("mismatched") or ()),
            "consumer_retrains": int(row["consumer_retrains"]),
        }
        for slot, row in gate["cells"].items()
    }
    out = {
        "ran": True,
        "role": (
            "P2: with an empty guard list the promoted enforcement path must "
            "leave every #19 task_C reading exactly where it was"
        ),
        "state": str(gate.get("state")),
        "migration_receipts": {
            slot: {
                "applied": bool(row.get("applied")),
                "single_key_diff_proof": row.get("single_key_diff_proof"),
            }
            for slot, row in migration["receipts"].items()
        },
        "cells": cells,
        "ok": bool(gate["ok"]),
        "mismatched_cells": list(gate.get("mismatched_cells") or ()),
        "consumer_retrains": int(gate["consumer_retrains"]),
        "llm_calls": 0,
        "wall_seconds": time.perf_counter() - started,
    }
    if not out["ok"]:
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the promoted enforcement path did not reproduce %s"
            % out["mismatched_cells"],
        )
    return out


# --------------------------------------------------- the one continuous store
def stage_store(label: str = "single") -> tuple[dict[str, Any], dict[str, Any]]:
    """Bootstrap plus the all-source pooled Guidance card.  0 LLM.

    Each trajectory gets its own directory, so "zero shared state" is
    checkable on disk afterwards and not merely asserted.
    """
    started = time.perf_counter()
    root = STORE_ROOT / label
    if root.exists():
        shutil.rmtree(root)
    target = FC._target(VARIANT)
    card = bridge.compile_skill_card(target)
    card_text = bridge.render_skill_card(card)
    payload = FC._card_payload(target, card, card_text)
    guarantees = ssi._assert_guidance_only(payload)
    FC.STORE_ROOT = root
    slot = FC._build_store(SLOT, payload)
    if slot.get("status") != "REGISTERED":
        raise Blocked(
            "SOURCE_DELIVERY_BREAK",
            "the store refused the compiled card: %s" % slot.get("blocked_reason"),
        )
    snapshot = slot["_snapshot"]
    guards = harness_compiler.scope_risk_guards_of(snapshot)
    if guards:
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the new store did not start with an empty guard list: %s" % guards,
        )
    record = {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "slot": SLOT,
        "card": {
            "skill_id": CARD_ID,
            "status": str(card["status"]),
            "clause_count": int(card["clause_count"]),
            "clause_ids": [str(c["clause_id"]) for c in card["clauses"]],
            "clauses": [
                {
                    "clause_id": str(c["clause_id"]),
                    "rule": str(c["rule"]),
                    "text": str(c["text"]),
                }
                for c in card["clauses"]
            ],
            "loco": dict(card["loco"]),
            "card_bytes_sha256": canonical_sha256(dict(payload)),
            "compiled_from": (
                "run_e2_recipe_experience_to_skill.compile_skill_card, the "
                "same input #17 used; the fresh cohort contributes no "
                "evidence row so the leave-one-cohort-out merge drops nothing"
            ),
            "guidance_only_guarantees": wvc._plain(guarantees),
        },
        "trajectory_label": label,
        "store": {
            "root": _repo_rel(root / SLOT / "snapshots"),
            "h0_runtime_bundle_sha": slot["h0_runtime_bundle_sha"],
            "runtime_bundle_sha": slot["runtime_bundle_sha"],
            "harness_content_sha": slot["harness_content_sha"],
            "skill_ids": list(slot["skill_ids"]),
            "scope_risk_guards_at_start": [],
        },
        "nothing_was_hand_written": (
            "the card is compiled and registered by the delivered code path; "
            "no clause text, no Workflow name and no store pointer is set by "
            "hand anywhere in this run"
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    return slot, record


def _cohort() -> dict[str, Any]:
    """The #17 cohort, rebuilt from what #17 left on disk.  No csv is reparsed."""
    artifact = FC._cohort_artifact()
    health = artifact["step_2_health_check_v2"]
    roster = {
        "train": [str(u) for u in health["confirmation_roster"]],
        "eval": [str(u) for u in health["substitutes"]],
    }
    return SSU._confirmation_payload(roster), FC._missing_cap(artifact), roster


def _search(window: Mapping[str, Any], cohort: Mapping[str, Any]) -> Any:
    return FC.FreshSearch(
        payload=cohort, consumer_variant=VARIANT,
        support_origins=window["support_origins"],
        delayed_origins=window["delayed_origins"],
    )


def _clause_programs(clauses: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Which menu programs each Guidance clause actually names.

    Read off the clause text against the frozen program menu.  This is the
    machine's reading of the card, not the Agent's account of it.
    """
    menu = [str(name) for name in FC.TREATMENTS]
    return {
        str(clause["clause_id"]): sorted(
            program for program in menu if program in str(clause["text"])
        )
        for clause in clauses
    }


def _clause_overlap(
    *, cited: Sequence[str], shortlist: Sequence[str],
    clause_programs: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """A2: does the shortlist contain what the cited clauses name.

    Recorded only.  Nothing is enforced, no prompt changes, and a zero
    overlap is not an error -- the Agent may cite a clause for a reason
    other than its named program.
    """
    cited = [str(item) for item in cited]
    shortlist = [str(item) for item in shortlist]
    named = sorted({
        program
        for clause in cited
        for program in (clause_programs.get(clause) or ())
    })
    overlap = sorted(set(named) & set(shortlist))
    return {
        "clauses_cited": cited,
        "programs_named_by_cited_clauses": named,
        "shortlist": shortlist,
        "overlap": overlap,
        "overlap_count": len(overlap),
        "cited_clauses_that_name_a_program": sorted(
            clause for clause in cited if clause_programs.get(clause)
        ),
        "shortlist_entries_named_by_no_cited_clause": sorted(
            set(shortlist) - set(named)
        ),
        "how_it_is_computed": (
            "each clause's text is scanned against the frozen program menu; "
            "skill_clause_use records what the Agent says it relied on, this "
            "records what those clauses actually name.  Recorded, never "
            "enforced."
        ),
    }


def _gate(
    search: Any, snapshot: Any, record: Mapping[str, Any], *, reused: bool,
) -> dict[str, Any]:
    """The tracked Scope/Risk gate, over the frozen ladder's own receipt."""
    ladder = {
        "final_plan": dict(record["final_plan"]),
        "support": dict(record.get("support") or {}),
        "delayed": dict(record.get("delayed") or {}),
        "support_winner": (record.get("adoption_ladder") or {}).get(
            "support_winner"
        ),
        "support_winner_full_batch_delayed": (
            record.get("adoption_ladder") or {}
        ).get("support_winner_full_batch_delayed"),
    }
    return harness_compiler.enforce_scope_risk_guards(
        snapshot=snapshot,
        ladder=ladder,
        eval_count=len(search.eval_uids),
        reused=bool(reused),
        delayed_of=search.delayed_gate,
        support_of=search.support_of_plan,
    )


def _step(
    *, tag: str, window: Mapping[str, Any], cohort: Mapping[str, Any],
    slot: dict[str, Any], local_skill: str | None, budget: Budget,
    box: dict[str, Any], clause_programs: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """One task window: retrieve, decide through the frozen path, then gate.

    ``box`` is mounted on the artifact payload before this call, so a stop
    anywhere below still leaves the stage on the record.
    """
    box.update({
        "step": tag,
        "entered": True,
        "window_id": str(window["window_id"]),
        "window": {
            k: v for k, v in window.items()
            if not str(k).startswith("reference_")
        },
        "local_skill_expected": local_skill,
        "completed": False,
    })
    search = _search(window, cohort)
    snapshot = slot["_snapshot"]
    view = {"slot": slot["slot"], "_snapshot": snapshot}
    if local_skill:
        record = FC._direct_recall(
            search=search, target=FC._target(VARIANT), arm=ARM, window=window,
            slot=view, expected_card=CARD_ID, expected_local=str(local_skill),
            tag=tag,
        )
        if record.get("stopped") == "RECALL_MISS":
            raise Blocked(
                "POST_UPDATE_NOT_RETRIEVED"
                if tag == "task_D" else "LOCAL_LIFECYCLE_BREAK",
                "%s did not retrieve the ACTIVE local Skill %s"
                % (tag, local_skill),
            )
        reused = bool(record.get("reuse_adopted"))
    else:
        record = FC._episode(
            search=search, target=FC._target(VARIANT), arm=ARM, window=window,
            slot=view, expected_card=CARD_ID, expected_local=None,
            llm_budget=budget.take(LLM_CALL_BUDGET_PER_EPISODE), tag=tag,
        )
        budget.spend(int(record.get("llm_calls") or 0))
        if record.get("stopped"):
            raise Blocked(
                "SOURCE_DELIVERY_BREAK",
                "%s produced no payload: %s" % (tag, record["stopped"]),
            )
        reused = False
    box.update({
        "mode": record.get("mode"),
        "card_hit": bool((record.get("retrieval") or {}).get("hit")),
        "local_skill_hit": (record.get("retrieval") or {}).get("local_skill_hit"),
        "shortlist": list(record.get("shortlist") or ()),
        "clauses_cited": list(record.get("skill_clause_use") or ()),
        "llm_calls": int(record.get("llm_calls") or 0),
        "episode_decided": True,
    })
    if not (record.get("retrieval") or {}).get("hit"):
        raise Blocked(
            "SOURCE_DELIVERY_BREAK",
            "%s did not retrieve the Guidance card naturally" % tag,
        )
    gate = _gate(search, snapshot, record, reused=reused)
    per_series_before = SSU._per_series(search, record["final_plan"])
    per_series_after = (
        dict(per_series_before)
        if dict(gate["plan_after"]) == dict(record["final_plan"])
        else SSU._per_series(search, gate["plan_after"])
    )
    budget.charge_retrains(int(search.retrains))
    row = {
        "step": tag,
        "entered": True,
        "completed": True,
        "window_id": str(window["window_id"]),
        "window": {
            k: v for k, v in window.items() if not str(k).startswith("reference_")
        },
        "mode": record.get("mode"),
        "card_hit": bool((record.get("retrieval") or {}).get("hit")),
        "clauses_cited": list(record.get("skill_clause_use") or ()),
        "clauses_available": list(record.get("skill_clause_ids_available") or ()),
        "clause_shortlist_overlap": _clause_overlap(
            cited=record.get("skill_clause_use") or (),
            shortlist=record.get("shortlist") or (),
            clause_programs=clause_programs,
        ),
        "local_skill_expected": local_skill,
        "local_skill_hit": (record.get("retrieval") or {}).get("local_skill_hit"),
        "reuse_adopted": record.get("reuse_adopted"),
        "shortlist": list(record.get("shortlist") or ()),
        "plan_before_gate": dict(record["final_plan"]),
        "support_before_gate": (record.get("support") or {}).get("aggregate_gain"),
        "delayed_before_gate": (record.get("delayed") or {}).get("aggregate_gain"),
        "harmed_before_gate": list(
            (record.get("delayed") or {}).get("harmed_eval_series") or ()
        ),
        "harm_total_before_gate": (record.get("delayed") or {}).get(
            "harmed_eval_series_total_harm"
        ),
        "per_eval_series_delayed_before_gate": per_series_before,
        "gate": gate,
        "plan_after_gate": dict(gate["plan_after"]),
        "support_after_gate": (gate.get("support_after") or {}).get(
            "aggregate_gain"
        ),
        "delayed_after_gate": (gate.get("delayed_after") or {}).get(
            "aggregate_gain"
        ),
        "harmed_after_gate": list(
            (gate.get("delayed_after") or {}).get("harmed_eval_series") or ()
        ),
        "per_eval_series_delayed_after_gate": per_series_after,
        "gate_changed_the_decision": bool(gate.get("changed")),
        "adoption_ladder": record.get("adoption_ladder"),
        "consumer_retrains": int(search.retrains),
        "llm_calls": int(record.get("llm_calls") or 0),
        "episode_id": record.get("episode_id"),
        "store_runtime_bundle_sha": slot.get("runtime_bundle_sha"),
        "snapshot_runtime_bundle_sha": snapshot.runtime_bundle_sha,
    }
    box.update(row)
    print(
        "OP %-8s %-14s %-26s -> %-26s | sup %s del %s | harm %s | gate %s | "
        "retrains %d llm %d"
        % (
            tag, str(record.get("mode")),
            SSU._plan_label(record["final_plan"]),
            SSU._plan_label(gate["plan_after"]),
            _fmt(row["support_before_gate"]), _fmt(row["delayed_before_gate"]),
            row["harmed_before_gate"] or "none",
            "moved" if gate.get("changed") else (
                "checked" if gate.get("checked") else "inactive"
            ),
            int(search.retrains), int(row["llm_calls"]),
        ),
        flush=True,
    )
    return {"row": row, "record": record, "search": search, "gate": gate}


# ------------------------------------------------- the Scope/Risk selector
SELECTOR_ID = "earliest_eligible_scope_risk_episode"
SELECTOR_CONTRACT: dict[str, Any] = {
    "selector": SELECTOR_ID,
    "what_it_does": (
        "walks this run's own completed episodes in time order and returns "
        "the first one whose adopted plan left an evaluation series below the "
        "%+.3f harm line.  That episode, unchanged, is what the existing "
        "attribution fold is handed." % HARM_THRESHOLD
    ),
    "eligibility": [
        "the episode completed in this run",
        "it adopted a non-identity plan",
        "its delayed window is revealed",
        "its worst per-evaluation-series delayed gain is below the harm line",
    ],
    "what_it_never_reads": [
        "candidates the episode did not adopt",
        "task_D or any window later than the one being attributed",
        "the #17 / #19 / #21 / #22 artifacts",
        "any sealed outcome",
    ],
    "why": (
        "#23 ran three trajectories that each produced a textbook "
        "aggregate-masks-per-series harm at task_A and the protocol looked "
        "past all three, because attribution was wired to task_C alone.  The "
        "window was the defect, not the fold."
    ),
    "what_it_does_not_touch": (
        "the fold itself, the Risk thresholds, the ladder, the two keys, the "
        "menu, the prompts and the SELECTION_MISS adapter reading are all "
        "unchanged; the SELECTION_MISS case stays on file, frozen"
    ),
}


def _min_per_series(row: Mapping[str, Any]) -> float | None:
    """The worst per-evaluation-series delayed gain of what was adopted."""
    per_series = (
        row.get("per_eval_series_delayed_after_gate")
        or row.get("per_eval_series_delayed_before_gate")
        or {}
    )
    if not per_series:
        return None
    return min(float(value) for value in per_series.values())


def select_scope_risk_episode(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Earliest eligible Scope/Risk episode, or nothing.  0 LLM, 0 retrains."""
    considered: list[dict[str, Any]] = []
    selected: Mapping[str, Any] | None = None
    for row in rows:
        plan = dict(row.get("plan_after_gate") or {})
        adopted = bool(plan) and str(plan.get("program")) != IDENTITY
        revealed = row.get("delayed_after_gate") is not None
        completed = bool(row.get("completed"))
        worst = _min_per_series(row)
        crossed = worst is not None and float(worst) < HARM_THRESHOLD
        eligible = bool(completed and adopted and revealed and crossed)
        if not completed:
            why = "the episode did not complete"
        elif not adopted:
            why = "the episode adopted identity, so there is nothing to blame"
        elif not revealed:
            why = "its delayed window is not revealed"
        elif worst is None:
            why = "no per-evaluation-series reading is on the record"
        elif not crossed:
            why = (
                "its worst evaluation series is %+.6f, at or above the %+.3f "
                "line" % (worst, HARM_THRESHOLD)
            )
        else:
            why = "eligible"
        considered.append({
            "step": row.get("step"),
            "episode_id": row.get("episode_id"),
            "adopted_plan": plan or None,
            "delayed_aggregate_gain": row.get("delayed_after_gate"),
            "min_per_series_delayed_gain": worst,
            "eligible": eligible,
            "why": why,
        })
        if eligible and selected is None:
            selected = row
    if selected is None:
        return {
            "selector": SELECTOR_ID,
            "contract": SELECTOR_CONTRACT,
            "verdict": "NO_ELIGIBLE_SCOPE_RISK_EPISODE",
            "selected_step": None,
            "selected_episode_id": None,
            "considered": considered,
            "reason": (
                "no completed, adopted, revealed episode in this run left an "
                "evaluation series below %+.3f" % HARM_THRESHOLD
            ),
            "not_the_same_as": (
                "NO_ACTIONABLE_FAULT: that is the fold's own verdict on one "
                "episode, and the task_C SELECTION_MISS reading is untouched "
                "and still on file"
            ),
        }
    return {
        "selector": SELECTOR_ID,
        "contract": SELECTOR_CONTRACT,
        "verdict": "SELECTED",
        "selected_step": str(selected.get("step")),
        "selected_episode_id": selected.get("episode_id"),
        "selected_plan": dict(selected.get("plan_after_gate") or {}),
        "min_per_series_delayed_gain": _min_per_series(selected),
        "delayed_aggregate_gain": selected.get("delayed_after_gate"),
        "harmed_eval_series": list(selected.get("harmed_after_gate") or ()),
        "considered": considered,
        "earlier_steps_skipped": [
            row["step"] for row in considered
            if row["step"] != str(selected.get("step")) and not row["eligible"]
        ],
    }


# --------------------------------------------- the Slow draw error taxonomy
# #24 folded every non-transport exception from core.run_stage into
# COMPILER_REJECTS.  That was right in #21, where the exception really did
# come from the controller, and wrong in #24, where the EditController was
# never reached and the label implied that #22's maxLength repair had
# failed.  Three classes, kept apart.
SLOW_ERROR_CLASSES: dict[str, dict[str, Any]] = {
    "ENVELOPE_PROTOCOL": {
        "label": "SLOW_ENVELOPE_PROTOCOL_FAILURE",
        "raised_by": "methods/ttha/agent_core.py::TTHAAgentCore.run_stage",
        "means": (
            "the model's response was not a valid agent-envelope/1, or its "
            "stage payload failed the contract, and the core's own static "
            "feedback retries were exhausted.  No proposal exists and the "
            "EditController was never called."
        ),
        "first_class": True,
    },
    "COMPILER_CONTROLLER": {
        "label": "COMPILER_REJECTS",
        "raised_by": (
            "evaluation/minipipe/replay/edit_controller.py::EditController"
        ),
        "means": (
            "a well-formed proposal reached the deterministic controller and "
            "the controller refused it.  This label is only ever written "
            "when an edit was actually submitted."
        ),
    },
    "TRANSPORT": {
        "label": "SLOW_TRANSPORT_FAILURE",
        "raised_by": "the backend transport or the call budget",
        "means": (
            "the request never produced a model decision.  A transport "
            "failure consumes no decision sample."
        ),
    },
    "RUNTIME": {
        "label": "SLOW_STAGE_RUNTIME_ERROR",
        "raised_by": "anything else",
        "means": "an error this taxonomy does not recognise; reported as it stands",
    },
}


def _classify_slow_error(exc: BaseException) -> str:
    if isinstance(exc, (AgentTransportError, AgentCallBudgetExceeded)):
        return "TRANSPORT"
    if isinstance(exc, AgentProtocolError):
        return "ENVELOPE_PROTOCOL"
    return "RUNTIME"


def _slow_diagnostics(
    exc: BaseException, backend: Any,
) -> dict[str, Any]:
    """The fingerprint of a failed draw, from what is already attached.

    Nothing new is plumbed.  ``agent_core`` already hangs the last assistant
    text, the internal retry count and the error codes on the exception, and
    the budgeted backend already accumulates calls, tokens and the serving
    model.  The bounded text keeps the cap agent_core itself applied; the
    full provider raw of every internal retry is deliberately not stored.
    """
    text = getattr(exc, "last_assistant_text", None)
    bounded = None if text is None else str(text)
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "class": _classify_slow_error(exc),
        "label": SLOW_ERROR_CLASSES[_classify_slow_error(exc)]["label"],
        "internal_validation_retries": getattr(exc, "validation_retry_count", None),
        "validation_error_codes": list(
            getattr(exc, "validation_error_codes", ()) or ()
        ),
        "last_assistant_text": bounded,
        "last_assistant_text_characters": None if bounded is None else len(bounded),
        "last_assistant_text_truncated_by_agent_core": (
            None if bounded is None else len(bounded) >= 500
        ),
        "bounded_at": (
            "500 characters, the cap agent_core already applies; this "
            "protocol does not widen it"
        ),
        "llm_calls_this_draw": int(getattr(backend, "calls", 0) or 0),
        "prompt_tokens": int(getattr(backend, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(backend, "completion_tokens", 0) or 0),
        "returned_models": sorted(getattr(backend, "returned_models", ()) or ()),
        "finish_reason": None,
        "finish_reason_note": (
            "not reachable: agent_core attaches the assistant text to the "
            "exception but not the response, and this round does not add new "
            "plumbing to fetch it"
        ),
        "provider_raw_not_stored": (
            "one bounded text per failed draw, not the full raw of every "
            "internal retry"
        ),
    }


# ------------------------------------------------------- attribution and Slow
def stage_attribution(record: Mapping[str, Any], snapshot: Any) -> dict[str, Any]:
    """First fault on this run's own task_C episode.  No prior artifact is fed in."""
    bootstrap = [
        skill.skill_id
        for skill in compile_snapshot(FC.H0_ROOT, verify_lock=False).skills
    ]
    capability = [
        skill.skill_id for skill in snapshot.skills
        if skill.skill_id not in set(bootstrap)
    ]
    with_series = SSU._attribute(
        record, bootstrap_ids=bootstrap,
        capability_skills_present=bool(capability), transport_per_series=True,
    )
    aggregate_only = SSU._attribute(
        record, bootstrap_ids=bootstrap,
        capability_skills_present=bool(capability), transport_per_series=False,
    )
    out = {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "evidence": (
            "this run's own task_C episode record only; no #17 or #19 "
            "artifact is read by the fold"
        ),
        "capability_skills_in_store": capability,
        "with_per_series_risk_reading": with_series,
        "aggregate_only_control": aggregate_only,
        "cause_code": with_series["attribution"]["cause_code"],
        "first_stage": with_series["attribution"]["first_stage"],
        "is_scope_risk_face": bool(with_series["is_scope_risk_face"]),
    }
    print(
        "OP attribution  %s at %s (aggregate-only: %s)"
        % (
            out["cause_code"], out["first_stage"],
            aggregate_only["attribution"]["cause_code"],
        ),
        flush=True,
    )
    return out


def stage_slow(
    *, slot: dict[str, Any], attribution: Mapping[str, Any],
    task_c: Mapping[str, Any], budget: Budget, sink: dict[str, Any],
    fault_step: str = "task_C", slow_model: str = SLOW_MODEL,
) -> dict[str, Any]:
    """One proposal, one pinned backend, one surface, one draw.

    A2 (#21 reporting defect): everything this stage learns is written into
    ``sink`` as it happens, and ``sink`` is already attached to the artifact
    payload before the call.  Whatever raises, the per-draw record and the
    proposal text are on disk.
    """
    started = time.perf_counter()
    sink.update({
        "ran": True,
        "backend": {"model": slow_model, "base_url": SLOW_BASE_URL,
                    "rounds": 1, "cross_backend_sampling": False,
                    "history_on_this_surface": MODEL_HISTORY_ON_THIS_SURFACE.get(
                        slow_model, "no history on record"),
                    "pinned_for_this_run": True},
        "attributed_from_episode": str(task_c["record"]["episode_id"]),
        "confirmed_cause": str(attribution["cause_code"]),
        "attempts": [],
        "proposal": None,
        "application": None,
        "records_before_it_raises": True,
    })
    if not attribution["is_scope_risk_face"]:
        raise Blocked(
            "ATTRIBUTION_BREAK",
            "the fold named %s, which authorizes no Scope/Risk surface"
            % attribution["cause_code"],
        )
    snapshot = slot["_snapshot"]
    cause = str(attribution["cause_code"])
    route = attribution["with_per_series_risk_reading"]["route_authorization"]
    offered = SSU._authorized_surfaces_v2(snapshot)
    if not isinstance(offered[0]["current_value"], list):
        raise Blocked(
            "INSTRUMENT_DRIFT",
            "the Scope/Risk surface does not resolve to a list in this store",
        )
    delayed = task_c["record"].get("delayed") or {}
    evidence = {
        "window": dict(task_c["row"]["window"]),
        "evaluation_roster": list(task_c["search"].eval_uids),
        "training_roster_size": len(task_c["search"].train_uids),
        "material_line": MATERIAL_THRESHOLD,
        "harm_line": HARM_THRESHOLD,
        "this_run_only": (
            "every number below was produced by this run; no earlier "
            "artifact is quoted to the Slow Agent"
        ),
        "the_failing_episode": {
            "step": fault_step,
            "mode": task_c["record"].get("mode"),
            "adopted_plan": dict(task_c["record"]["final_plan"]),
            "support_aggregate_gain": (task_c["record"].get("support") or {}).get(
                "aggregate_gain"
            ),
            "delayed_aggregate_gain": delayed.get("aggregate_gain"),
            "evaluation_series_past_the_material_line": list(
                delayed.get("harmed_eval_series") or ()
            ),
            "their_summed_loss_magnitude": delayed.get(
                "harmed_eval_series_total_harm"
            ),
            "per_evaluation_series_delayed_gain": dict(
                task_c["row"]["per_eval_series_delayed_before_gate"]
            ),
            "adoption_path": (
                task_c["record"].get("adoption_ladder") or {}
            ).get("path"),
            "adoption_path_text": (
                task_c["record"].get("adoption_ladder") or {}
            ).get("path_text"),
        },
        "earlier_windows_this_run": [
            {
                "step": row["step"],
                "adopted_plan": row["plan_after_gate"],
                "support": row["support_before_gate"],
                "delayed": row["delayed_before_gate"],
                "harmed": row["harmed_before_gate"],
            }
            for row in (task_c.get("earlier_rows") or ())
        ],
        "what_the_runtime_folded": attribution["with_per_series_risk_reading"][
            "attribution"
        ],
        "adoption_ladder_is_frozen": (
            "the ladder picks the Support winner, sets the bar at "
            "max(0, that winner's full-batch delayed), adopts the named plan "
            "when it clears the bar, and otherwise falls back.  It is not "
            "modifiable here and it may not be re-ranked."
        ),
    }
    public_input = {
        "stage_note": SSU.SLOW_NOTE_V2,
        "attributed_fault": attribution["with_per_series_risk_reading"][
            "attribution"
        ],
        "attributed_from_episode": str(task_c["record"]["episode_id"]),
        "route_authorization": dict(route),
        "authorized_surfaces": offered,
        "episode_evidence": evidence,
        "guard_grammar": SSU.GUARD_GRAMMAR_V2,
        "guard_contract": SSU.GUARD_CONTRACT_V2,
        "public_boundary": {
            "rule": (
                "a deployable edit may not encode private or path-derived "
                "evidence; these substrings are refused anywhere in the value"
            ),
            "forbidden_substrings": [
                "clean_future", "clean_context", "oracle_affected",
                "injection_type", "candidate_utilities", "selection_regret",
                "loss_j", "utility_u", "r_private", "private_receipt",
            ],
        },
        "manifest_fields_you_author": {
            "edit_id": "a fresh canonical id for this edit",
            "target_pattern_id": "a canonical id naming the pattern you are fixing",
            "predicted_agent_behavior_change": (
                "one or more predicates from the closed vocabulary below"
            ),
            "predicted_agent_behavior_vocabulary": SSU._behavior_vocabulary(),
            "predicted_data_effect": "free text, what should happen to the data",
            "falsification_condition": "free text, what would show this edit wrong",
        },
    }
    view = resolve_harness_view(snapshot, {}, role="slow")
    validator = SSU._make_proposal_validator_v2()
    attempts: list[dict[str, Any]] = sink["attempts"]
    payload: dict[str, Any] = {}
    transport_failures = 0
    valid_decision_samples = 0
    protocol_failed_draws = int(PROTOCOL_FAILED_CARRY_IN.get(slow_model, 0))
    sink["counters"] = {
        "valid_decision_samples": valid_decision_samples,
        "protocol_failed_draws": protocol_failed_draws,
        "protocol_failed_draws_carried_in": int(
            PROTOCOL_FAILED_CARRY_IN.get(slow_model, 0)
        ),
        "llm_calls_spent": 0,
        "rule": SAMPLING_RULE,
    }
    sink["authorized_surfaces_offered"] = [
        str(item["surface_id"]) for item in offered
    ]
    sink["public_input_sha256"] = canonical_sha256(wvc._plain(public_input))
    sink["harness_view_sha"] = view.effective_harness_view_sha
    while True:
        backend = SSU._backend_factory_v2(slow_model, budget.take(4))
        gateway = wvc.NoToolGateway({
            "protocol": PROTOCOL_VERSION, "stage": "edit",
        })
        core = TTHAAgentCore(
            backend, gateway, model=slow_model, base_url=SLOW_BASE_URL,
        )
        row: dict[str, Any] = {
            "draw": len(attempts) + 1, "model": slow_model,
        }
        try:
            result = core.run_stage(
                role=AgentRole.SLOW, stage="edit",
                case_id="OP_%s" % SLOT,
                public_input=public_input, harness_view=view,
                output_schema_name="slow_scope_guard_v1",
                output_schema=SSU.PROPOSAL_SCHEMA_V2,
                source_snapshot_sha=view.effective_harness_view_sha,
                validation_retries=wvc.VALIDATION_RETRIES,
                post_validator=validator,
            )
        except Exception as exc:  # noqa: BLE001
            budget.spend(int(backend.calls))
            diagnostics = _slow_diagnostics(exc, backend)
            kind = diagnostics["class"]
            row.update({
                "outcome": diagnostics["label"],
                "error_class": kind,
                "diagnostics": diagnostics,
                "llm_calls": int(backend.calls),
                "consumes_a_decision_sample": False,
                "consumes_a_protocol_failed_draw": kind == "ENVELOPE_PROTOCOL",
            })
            attempts.append(row)
            sink["counters"]["llm_calls_spent"] = sum(
                int(a.get("llm_calls") or 0) for a in attempts
            )
            print(
                "  slow draw %d: %s (%s, internal retries %s, codes %s)" % (
                    row["draw"], diagnostics["label"], diagnostics["error_type"],
                    diagnostics["internal_validation_retries"],
                    diagnostics["validation_error_codes"],
                ),
                flush=True,
            )
            if kind == "TRANSPORT":
                transport_failures += 1
                if transport_failures >= MAX_TRANSPORT_FAILURES:
                    raise Blocked(
                        "INCONCLUSIVE_TRANSPORT",
                        "%d consecutive transport failures" % transport_failures,
                    )
                continue
            if kind == "ENVELOPE_PROTOCOL":
                protocol_failed_draws += 1
                sink["counters"]["protocol_failed_draws"] = protocol_failed_draws
                if protocol_failed_draws >= MAX_PROTOCOL_FAILED_DRAWS:
                    raise Blocked(
                        "SLOW_ENVELOPE_PROTOCOL_FAILURE_EXHAUSTED",
                        "%d protocol-failed draws for %s on this surface "
                        "(%d carried in): no decision this configuration can "
                        "read" % (
                            protocol_failed_draws, slow_model,
                            PROTOCOL_FAILED_CARRY_IN.get(slow_model, 0),
                        ),
                    )
                continue
            raise Blocked(diagnostics["label"], diagnostics["error_message"])
        budget.spend(int(backend.calls))
        transport_failures = 0
        candidate = dict(result.payload)
        row.update({
            "outcome": "PROPOSAL" if candidate else "NO_PROPOSAL",
            "llm_calls": int(backend.calls),
            "consumes_a_draw": True,
            "validation_retry_count": int(result.validation_retry_count),
            "validation_error_codes": list(result.validation_error_codes),
            "no_proposal_reason": result.no_proposal_reason,
            "returned_models": sorted({
                str(m) for m in getattr(result.response, "returned_models", ())
            }) or None,
        })
        valid_decision_samples += 1
        row["consumes_a_decision_sample"] = True
        row["valid_decision_sample_index"] = valid_decision_samples
        attempts.append(row)
        transport_failures = 0
        sink["counters"].update({
            "valid_decision_samples": valid_decision_samples,
            "llm_calls_spent": sum(
                int(a.get("llm_calls") or 0) for a in attempts
            ),
        })
        sink["llm_calls"] = sink["counters"]["llm_calls_spent"]
        if not candidate:
            if valid_decision_samples >= MAX_VALID_DECISION_SAMPLES:
                raise Blocked(
                    "SLOW_DECLINES",
                    "%d valid decision samples, both no_proposal (%s)"
                    % (
                        valid_decision_samples,
                        ", ".join(
                            str(a.get("no_proposal_reason"))
                            for a in attempts
                            if a.get("consumes_a_decision_sample")
                        ),
                    ),
                )
            continue
        payload = candidate
        sink["proposal"] = wvc._plain(payload)
        sink["guard"] = wvc._plain(payload["guard"])
        sink["target_surface_id"] = str(payload["target_surface_id"])
        break

    application = SSU._apply_patch(
        {"slot": SLOT, "store": slot["_store"], "snapshot": snapshot},
        payload=payload, cause=cause,
    )
    sink["application"] = application
    if not application.get("applied"):
        raise Blocked(
            "COMPILER_REJECTS",
            str(application.get("reason") or "the controller refused the edit"),
        )
    # _apply_patch already set the fork snapshot active; re-read it here.
    active = json.loads(
        slot["_store"].active_path.read_text(encoding="utf-8")
    )["runtime_bundle_sha"]
    snapshot_after = compile_snapshot(
        Path(slot["_store"].root) / active, verify_lock=False
    )
    slot["_snapshot"] = snapshot_after
    slot["runtime_bundle_sha"] = snapshot_after.runtime_bundle_sha
    guards_after = harness_compiler.scope_risk_guards_of(snapshot_after)
    sink.update({
        "llm_calls": sum(int(a.get("llm_calls") or 0) for a in attempts),
        "snapshot_before": str(snapshot.runtime_bundle_sha),
        "snapshot_after": str(snapshot_after.runtime_bundle_sha),
        "harness_content_before": str(snapshot.harness_content_sha),
        "harness_content_after": str(snapshot_after.harness_content_sha),
        "guards_active_after": guards_after,
        "same_store": (
            "the patch is written into the store this run has been using "
            "since task_A; no store is swapped and no snapshot is hand-picked"
        ),
        "wall_seconds": time.perf_counter() - started,
    })
    return sink


# --------------------------------------------- the post-update behaviour read
def stage_shadow(
    *, task_c: Mapping[str, Any], snapshot_after: Any,
) -> dict[str, Any]:
    """Criterion (ii): re-decide this run's own task_C under the new Harness.

    Zero evaluations.  The task_C search object is the one that already
    measured every reading this needs; identity is cached in its own
    constructor, so the fallback walk costs nothing.  If a retrain is
    charged here the check is void and says so.
    """
    search = task_c["search"]
    before_retrains = int(search.retrains)
    gate = _gate(
        search, snapshot_after, task_c["record"],
        reused=bool(task_c["record"].get("reuse_adopted")),
    )
    spent = int(search.retrains) - before_retrains
    pre = dict(task_c["gate"]["plan_after"])
    post = dict(gate["plan_after"])
    return {
        "ran": True,
        "zero_evaluation": spent == 0,
        "consumer_retrains_spent": spent,
        "snapshot": str(snapshot_after.runtime_bundle_sha),
        "decision_before_patch": pre,
        "decision_after_patch": post,
        "decision_changed": pre != post,
        "gate": gate,
        "consistent_with_the_risk_contract": bool(
            gate.get("changed")
            and str(post["program"]) == IDENTITY
            and not (gate.get("delayed_after") or {}).get("harmed_eval_series")
        ),
        "why": (
            "the same measured task_C readings, re-decided under the patched "
            "snapshot; nothing is re-measured and nothing is re-adopted"
        ),
    }


def stage_experience(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One Experience row for the post-update task, tagged with its provenance."""
    task_d = rows[-1]
    plan = dict(task_d["plan_after_gate"])
    audit = {
        "provenance": "operational_pipeline",
        "counts_as_unguided_exploration": False,
    }
    if str(plan["program"]) == IDENTITY:
        return {
            "ran": True,
            "provenance": "operational_pipeline",
            "rows": [],
            "row_count": 0,
            "no_row_written": (
                "task_D ended on identity; this line forms no Experience row "
                "for an abstention adoption"
            ),
            "not_persisted_as_skills": (
                "handle_fast_winner is called at task_A only, as in #17: the "
                "post-update window reads behaviour, it does not form a Skill"
            ),
        }
    steps = [(str(plan["program"]), {})]
    support_gain = float(task_d["support_after_gate"] or 0.0)
    delayed_gain = float(task_d["delayed_after_gate"] or 0.0)
    episode = build_episode(
        episode_id="op_%s_task_d" % VARIANT,
        task_consumer_key="batch:%s|consumer:%s" % (COHORT_NAME, VARIANT),
        domain_namespace=str(COHORT_NAME),
        context_summary={
            "task_episode_id": str(task_d["window_id"]),
            "arm": ARM,
            "cohort": {"cohort_name": str(COHORT_NAME)},
            "local_pattern": {"consumer_variant": VARIANT},
            "program_geometry": {
                "program_steps": [
                    {"op": op, "params": dict(params)} for op, params in steps
                ],
                "frozen_plan_scope": {
                    "excluded_series": sorted(
                        str(uid) for uid in plan["excluded_series"]
                    )
                },
                "consumer_retrains": int(task_d["consumer_retrains"]),
            },
        },
        workflow_signature=FC.e1mod._v2_workflow_signature(steps),
        support_response={
            "gain": support_gain,
            "accepted": support_gain >= MATERIAL_THRESHOLD,
            "block_origins": list(task_d["window"]["support_origins"]),
            "program": str(plan["program"]),
            "excluded_series": list(plan["excluded_series"]),
            **audit,
        },
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": None,
            "gain_over_se": None,
            "block_origins": list(task_d["window"]["delayed_origins"]),
            "took_part_in_selection": True,
            "why_not_promotion_evidence": (
                "this delayed reading set the adoption bar in its own "
                "episode, so it may not also license a promotion"
            ),
            **audit,
        },
        relation=FC._relation(support_gain, delayed_gain, plan["program"]),
        evidence_level=EVIDENCE_SUPPORT,
        local_status=FC.STATUS_LOCAL_DRAFT,
        evidence_refs=("operational_pipeline", PROTOCOL_VERSION),
    )
    return {
        "ran": True,
        "provenance": "operational_pipeline",
        "rows": [{
            "episode_id": episode.episode_id,
            "workflow_signature": episode.workflow_signature,
            "relation": str(episode.relation),
            "local_status": str(episode.local_status),
            **audit,
        }],
        "row_count": 1,
        "not_persisted_as_skills": (
            "handle_fast_winner is called at task_A only, as in #17"
        ),
    }


# --------------------------------------------------------------- the verdict
def _verdict(payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = {row["step"]: row for row in payload["trajectory"]}
    slow = payload["slow_edit"]
    if (payload.get("no_fault_sample") or {}).get(
        "this_trajectory_produced_no_fault"
    ):
        return {
            "verdict": "PIPELINE_RUNS_NO_FAULT_SAMPLE",
            "reason": (
                "the trajectory ran end to end but produced no harm past the "
                "line, so the fold had nothing actionable to name; links 6 to "
                "9 -- attribution to a Scope/Risk face, the Slow edit, the "
                "compiler and the post-update behaviour -- are untested in "
                "this sample"
            ),
            "detail": {
                "links_1_to_5_exercised": True,
                "links_6_to_9_untested": True,
                "source_delivered": bool(rows["task_A"]["card_hit"]),
                "clauses_cited_at_task_a": list(rows["task_A"]["clauses_cited"]),
                "skill_promoted": bool(
                    payload["lifecycle"]["promotion"].get("promoted")
                ),
                "skill_recalled_at_task_b": bool(
                    rows["task_B"].get("local_skill_hit")
                ),
                "task_c_harmed_eval_series": list(
                    rows["task_C"]["harmed_before_gate"]
                ),
                "task_d_harmed_eval_series": (
                    list(rows["task_D"]["harmed_before_gate"])
                    if "task_D" in rows else None
                ),
            },
        }
    task_d = rows.get("task_D")
    shadow = payload.get("shadow_replay") or {}
    detail: dict[str, Any] = {
        "source_delivered": bool(rows["task_A"]["card_hit"]),
        "clauses_cited_at_task_a": list(rows["task_A"]["clauses_cited"]),
        "skill_formed": bool(payload["lifecycle"]["draft"].get("written")),
        "skill_promoted": bool(payload["lifecycle"]["promotion"].get("promoted")),
        "skill_recalled_at_task_b": bool(
            rows["task_B"].get("local_skill_hit")
        ),
        "attribution_cause": payload["attribution"]["cause_code"],
        "surfaces_changed": list(
            (slow.get("application") or {}).get("source_surfaces_changed") or ()
        ),
        "task_d_snapshot": task_d and task_d["snapshot_runtime_bundle_sha"],
        "post_patch_snapshot": slow.get("snapshot_after"),
    }
    if task_d is None:
        return {"verdict": "NO_POST_UPDATE_WINDOW",
                "reason": "task_D never ran", "detail": detail}
    if str(task_d["snapshot_runtime_bundle_sha"]) != str(slow["snapshot_after"]):
        return {
            "verdict": "POST_UPDATE_NOT_RETRIEVED",
            "reason": (
                "task_D read snapshot %s, not the patched %s"
                % (task_d["snapshot_runtime_bundle_sha"], slow["snapshot_after"])
            ),
            "detail": detail,
        }
    gate = task_d["gate"]
    evaluated = bool(gate.get("checked")) or bool(gate.get("guard_count"))
    detail["guard_evaluated_at_task_d"] = evaluated
    detail["guard_readings_at_task_d"] = (
        gate.get("check_on_adopted_plan") or {}
    ).get("readings")
    if not evaluated:
        return {
            "verdict": "POST_UPDATE_NOT_RETRIEVED",
            "reason": (
                "the patched snapshot was active but the guard was never "
                "evaluated at task_D"
            ),
            "detail": detail,
        }
    criterion_i = bool(gate.get("changed"))
    criterion_ii = bool(
        shadow.get("ran")
        and shadow.get("zero_evaluation")
        and shadow.get("decision_changed")
        and shadow.get("consistent_with_the_risk_contract")
    )
    harm_recurred = bool(task_d["harmed_before_gate"])
    detail.update({
        "criterion_i_task_d_fired": criterion_i,
        "criterion_ii_shadow_replay": criterion_ii,
        "harm_recurred_at_task_d": harm_recurred,
        "guard_silent_and_no_harm_is_the_contract_working": bool(
            not criterion_i and not harm_recurred
        ),
    })
    if not (criterion_i or criterion_ii):
        return {
            "verdict": "NEXT_TASK_NO_BEHAVIOR_CHANGE",
            "reason": (
                "the guard was evaluated at task_D but neither criterion held: "
                "it did not fire on the spot, and the shadow replay did not "
                "change this run's own task_C decision"
            ),
            "detail": detail,
        }
    return {
        "verdict": "OPERATIONAL_PIPELINE_CLOSES",
        "reason": (
            "Source delivered and cited, Skill formed / promoted / recalled, "
            "the fold named %s on this run's own episode, one surface was "
            "patched and compiled into the same store, task_D read the "
            "patched snapshot and the guard was evaluated there; criterion "
            "%s holds"
            % (
                detail["attribution_cause"],
                "(i)" if criterion_i else "(ii)",
            )
        ),
        "detail": detail,
    }


# ------------------------------------------------------------------- the run
# A3 (#21 reporting defect): the old filter dropped every key literally
# named "store", which silently deleted the whole store/card block from the
# artifact.  Live objects are excluded by type now, and by the underscore
# convention the runner already uses for them, so a plain data key called
# "store" survives.
_LIVE_KEYS = ("snapshot", "search", "record", "gate_obj")
_JSON_SCALARS = (str, int, float, bool, type(None))


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(k): _public(v) for k, v in value.items()
            if not str(k).startswith("_") and str(k) not in _LIVE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (value != value or value in (
        float("inf"), float("-inf")
    )):
        return None
    if not isinstance(value, _JSON_SCALARS):
        return "<%s not serializable>" % type(value).__name__
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return "%+.6f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _mount(container: dict[str, Any], key: str, seed: Any) -> Any:
    """A1: put a stage's record on the payload before the stage runs.

    The recording family that bit #21 twice and #22 once was always the same
    shape -- the writer was reached, but the field had never been attached.
    Every stage below mounts its container first and fills it in place, so a
    stop anywhere leaves every earlier stage complete and the stopping stage
    marked entered-but-not-completed.
    """
    container[key] = seed
    return seed


def _trajectory(
    *, label: str, budget: Budget, cohort: Mapping[str, Any],
    cap: Mapping[str, Any], sink: dict[str, Any],
    slow_model: str = SLOW_MODEL,
) -> dict[str, Any]:
    """One complete trajectory, from a store of its own to task_D."""
    started = time.perf_counter()
    sink.update({
        "label": label, "entered": True, "completed": False,
        "verdict": None, "verdict_reason": None,
    })
    stages = _mount(sink, "stages", {})
    trajectory: list[dict[str, Any]] = _mount(sink, "trajectory", [])
    try:
        store_box = _mount(stages, "store", {"entered": True, "completed": False})
        slot, store_record = stage_store(label)
        store_box.clear()
        store_box.update(store_record)
        store_box["completed"] = True
        clause_programs = _clause_programs(store_record["card"]["clauses"])
        sink["clause_programs"] = clause_programs

        def step(tag: str, window: Mapping[str, Any], local_skill: str | None):
            box: dict[str, Any] = {"step": tag, "entered": False, "completed": False}
            trajectory.append(box)
            return _step(
                tag=tag, window=window, cohort=cohort, slot=slot,
                local_skill=local_skill, budget=budget, box=box,
                clause_programs=clause_programs,
            )

        step_a = step("task_A", WINDOW_TASK_A, None)

        life = _mount(stages, "lifecycle", {
            "entered": True, "completed": False,
            "probe_window": dict(WINDOW_PROBE),
            "path": dict(FC.UPDATE_PATH),
        })
        draft = FC._persist_draft(
            slot=slot, record=step_a["record"], target=FC._target(VARIANT), arm=ARM,
        )
        life["draft"] = _public(draft)
        probe = None
        promotion: dict[str, Any] = {"promoted": False, "reason": draft.get("reason")}
        probe_retrains = 0
        if draft.get("written"):
            probe = FC._probe(
                search=step_a["search"], payload=cohort, variant=VARIANT,
                plan=dict(step_a["record"]["final_plan"]),
            )
            probe_retrains = int(probe["consumer_retrains"])
            budget.charge_retrains(probe_retrains)
            life["probe"] = _public(probe)
            life["probe_consumer_retrains"] = probe_retrains
            promotion = FC._promote(slot=slot, probe=probe, draft=draft)
        life["promotion"] = _public(promotion)
        life["completed"] = True
        print(
            "OP[%s] lifecycle    draft=%s probe=%s promoted=%s skill=%s"
            % (
                label, bool(draft.get("written")),
                None if probe is None else round(float(probe["macro_gain"]), 6),
                bool(promotion.get("promoted")),
                promotion.get("retrievable_skill_id"),
            ),
            flush=True,
        )
        local_skill = promotion.get("retrievable_skill_id")
        if not local_skill:
            adopted = dict(step_a["record"]["final_plan"])
            if str(adopted["program"]) == IDENTITY:
                raise Blocked(
                    "NO_ADOPTABLE_PLAN_SAMPLE",
                    "task_A found nothing the frozen v2 ladder would adopt, so "
                    "there was no Skill to form; the lifecycle was never "
                    "reached and the adoption above it abstained honestly",
                )
            raise Blocked(
                "LOCAL_LIFECYCLE_BREAK",
                "a plan was adopted but no Target-local Skill reached ACTIVE: %s"
                % (promotion.get("reason") or draft.get("reason")),
            )

        step_b = step("task_B", WINDOW_TASK_B, str(local_skill))
        step_c = step("task_C", WINDOW_TASK_C, str(local_skill))
        step_c["earlier_rows"] = list(trajectory[:-1])
        by_step = {
            "task_A": step_a, "task_B": step_b, "task_C": step_c,
        }

        sel_box = _mount(stages, "scope_risk_selector", {"entered": True})
        selection = select_scope_risk_episode(
            [step_a["row"], step_b["row"], step_c["row"]]
        )
        sel_box.clear()
        sel_box.update(_public(selection))
        print(
            "OP[%s] selector     %s -> %s" % (
                label, selection["verdict"], selection.get("selected_step"),
            ),
            flush=True,
        )

        att_box = _mount(stages, "attribution", {"entered": True, "completed": False})
        fault_step = selection.get("selected_step")
        fault = by_step.get(str(fault_step)) if fault_step else None
        if fault is not None:
            attribution = stage_attribution(fault["record"], slot["_snapshot"])
            att_box.clear()
            att_box.update(_public(attribution))
            att_box["attributed_episode"] = str(fault_step)
            att_box["completed"] = True
        else:
            attribution = {
                "ran": False,
                "cause_code": "NO_ELIGIBLE_SCOPE_RISK_EPISODE",
                "first_stage": None,
                "is_scope_risk_face": False,
                "why": selection["reason"],
            }
            att_box.clear()
            att_box.update(dict(attribution))
            att_box["completed"] = True

        slow_sink: dict[str, Any] = _mount(
            stages, "slow_edit", {"entered": True, "ran": False},
        )
        no_fault = selection["verdict"] != "SELECTED"
        _mount(stages, "no_fault_sample", {
            "this_trajectory_produced_no_fault": no_fault,
            "decided_by": (
                "the Scope/Risk selector over every completed episode of this "
                "run, not by task_C alone"
            ),
            "selector_verdict": selection["verdict"],
            "selected_step": selection.get("selected_step"),
            "attribution_cause": str(attribution["cause_code"]),
        })
        if no_fault:
            slow_sink.update({
                "ran": False,
                "why_not_run": (
                    "task_C left nothing past the harm line, so there is no "
                    "fault sample to attribute a Scope/Risk edit to"
                ),
            })
            print("OP[%s] slow_edit    skipped: no fault sample" % label, flush=True)
        else:
            slow = stage_slow(
                slot=slot, attribution=attribution, task_c=fault,
                budget=budget, sink=slow_sink, fault_step=str(fault_step),
                slow_model=slow_model,
            )
            print(
                "OP[%s] slow_edit    %s -> %s | snapshot %s -> %s"
                % (
                    label, slow["guard"]["statistic"], slow["guard"]["action"],
                    slow["snapshot_before"][:12], slow["snapshot_after"][:12],
                ),
                flush=True,
            )

        gate_box = _mount(stages, "task_d_missing_gate", {"entered": True})
        gate_d = FC._missing_gate(
            cohort["values"], cohort["eval_uids"], WINDOW_TASK_D, cap,
        )
        gate_box.clear()
        gate_box.update(_public(gate_d))
        if not gate_d["pass"]:
            raise Blocked(
                "NO_POST_UPDATE_WINDOW", "task_D does not clear the missing gate",
            )
        step_d = step("task_D", WINDOW_TASK_D, str(local_skill))

        if not no_fault and not step_d["gate"].get("changed"):
            shadow_box = _mount(stages, "shadow_replay", {"entered": True})
            shadow = stage_shadow(task_c=fault, snapshot_after=slot["_snapshot"])
            shadow_box.clear()
            shadow_box.update(_public(shadow))
            sink["shadow_replay"] = shadow_box

        exp_box = _mount(stages, "experience", {"entered": True})
        exp_box.clear()
        exp_box.update(stage_experience(trajectory))

        sink["slow_edit"] = slow_sink
        sink["lifecycle"] = life
        sink["attribution"] = att_box
        sink["no_fault_sample"] = stages["no_fault_sample"]
        outcome = _verdict(sink)
        sink.update({
            "verdict": outcome["verdict"],
            "verdict_reason": outcome["reason"],
            "verdict_detail": _public(outcome.get("detail") or {}),
            "completed": True,
        })
    except Blocked as exc:
        if exc.verdict in ("INSTRUMENT_DRIFT", "BUDGET_EXCEEDED"):
            sink.update({
                "verdict": exc.verdict, "verdict_reason": exc.reason,
                "aborts_the_whole_protocol": True,
            })
            raise
        sink.update({
            "verdict": exc.verdict,
            "verdict_reason": exc.reason,
            "stopped_at_first_block": True,
        })
        for key in ("lifecycle", "attribution", "slow_edit", "no_fault_sample"):
            if key in stages and key not in sink:
                sink[key] = stages[key]
    except ConcurrentWrite:
        raise
    except Exception as exc:  # noqa: BLE001
        # A trajectory that crashes stops itself, not the protocol; the
        # traceback goes on the record so the other trajectories still count.
        sink.update({
            "verdict": "TRAJECTORY_RUNTIME_ERROR",
            "verdict_reason": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
            "stopped_at_first_block": True,
        })
        for key in ("lifecycle", "attribution", "slow_edit", "no_fault_sample"):
            if key in stages and key not in sink:
                sink[key] = stages[key]
    sink["wall_seconds"] = time.perf_counter() - started
    sink["llm_calls"] = sum(
        int(row.get("llm_calls") or 0) for row in trajectory
    ) + int((stages.get("slow_edit") or {}).get("llm_calls") or 0)
    sink["consumer_retrains"] = sum(
        int(row.get("consumer_retrains") or 0) for row in trajectory
    ) + int((stages.get("lifecycle") or {}).get("probe_consumer_retrains") or 0)
    print(
        "OP[%s] VERDICT %s | llm %d retrains %d"
        % (label, sink["verdict"], sink["llm_calls"], sink["consumer_retrains"]),
        flush=True,
    )
    return sink


K_TRAJECTORIES = 3


def _aggregate(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    closed = [
        str(t["label"]) for t in trajectories
        if str(t.get("verdict")) == "OPERATIONAL_PIPELINE_CLOSES"
    ]
    reached_slow = [
        str(t["label"]) for t in trajectories
        if ((t.get("stages") or {}).get("slow_edit") or {}).get("ran")
    ]
    by_verdict: dict[str, list[str]] = {}
    for t in trajectories:
        by_verdict.setdefault(str(t.get("verdict")), []).append(str(t["label"]))
    if closed:
        verdict = "OPERATIONAL_PIPELINE_CLOSES_IN_K"
        reason = (
            "%d of %d trajectories closed the nine-link chain end to end: %s"
            % (len(closed), len(trajectories), closed)
        )
    elif not reached_slow:
        verdict = "PIPELINE_NEVER_EXERCISES_SLOW"
        reason = (
            "all %d trajectories stopped before the Slow edit; the chain's "
            "last four links go untested in this sample" % len(trajectories)
        )
    else:
        verdict = "PIPELINE_REACHES_SLOW_WITHOUT_CLOSING"
        reason = (
            "%s reached the Slow edit but no trajectory closed the chain"
            % reached_slow
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "k": len(trajectories),
        "k_fixed_before_the_run": K_TRAJECTORIES,
        "closed": closed,
        "reached_slow": reached_slow,
        "by_verdict": by_verdict,
        "every_trajectory_is_on_the_record": (
            "K was fixed before the first draw; all %d ran to their own stop "
            "and none was discarded, re-thrown or seeded" % len(trajectories)
        ),
    }


def _shortlist_stability(
    trajectories: Sequence[Mapping[str, Any]],
    priors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Every task_A draw this line has on record, in one table.

    The overlap column is computed the same way for the historical draws as
    for this run's: the cited clauses' own texts, scanned against the frozen
    program menu.
    """
    clause_programs: dict[str, Any] = {}
    for t in trajectories:
        if t.get("clause_programs"):
            clause_programs = dict(t["clause_programs"])
            break
    rows: list[dict[str, Any]] = []
    for prior in priors:
        row = dict(prior)
        if clause_programs:
            overlap = _clause_overlap(
                cited=row.get("clauses_cited") or (),
                shortlist=row.get("shortlist") or (),
                clause_programs=clause_programs,
            )
            row["clause_shortlist_overlap"] = list(overlap["overlap"])
            row["overlap_count"] = overlap["overlap_count"]
        else:
            row["clause_shortlist_overlap"] = None
            row["overlap_count"] = None
        rows.append(row)
    for t in trajectories:
        row = next(
            (r for r in (t.get("trajectory") or ()) if r.get("step") == "task_A"),
            None,
        )
        if row is None:
            continue
        overlap = row.get("clause_shortlist_overlap") or {}
        rows.append({
            "run": "operational_pipeline_v3/%s" % t["label"],
            "shortlist": list(row.get("shortlist") or ()),
            "clauses_cited": list(row.get("clauses_cited") or ()),
            "clause_shortlist_overlap": list(overlap.get("overlap") or ()),
            "overlap_count": overlap.get("overlap_count"),
            "adopted": row.get("plan_after_gate"),
            "support": row.get("support_before_gate"),
            "delayed": row.get("delayed_before_gate"),
            "ladder_path": (row.get("adoption_ladder") or {}).get("path"),
        })
    adopted_non_identity = [
        r for r in rows
        if r.get("adopted") and str(r["adopted"].get("program")) != IDENTITY
    ]
    with_overlap = [r for r in rows if (r.get("overlap_count") or 0) > 0]
    return {
        "draws": rows,
        "n": len(rows),
        "distinct_shortlists": len({tuple(r["shortlist"]) for r in rows}),
        "adopted_a_program": len(adopted_non_identity),
        "cited_clauses_naming_a_shortlisted_program": len(with_overlap),
        "reading": (
            "one table, %d draws, same card and same window.  Overlap is "
            "computed from the clause texts against the frozen menu, not from "
            "what the Agent said it used.  n is small and nothing here is a "
            "causal claim." % len(rows)
        ),
    }


PRIOR_TASK_A_DRAWS: tuple[dict, ...] = (
    {
        "run": "fresh_confirmation_v1 (#17) a5_pooled",
        "shortlist": ["outlier_iqr", "outlier_mad"],
        "clauses_cited": ["R1-2", "R1-1", "R3-1"],
        "adopted": {"program": "outlier_mad", "excluded_series": []},
        "support": 0.07248618783742994,
        "delayed": 0.30637972777516714,
        "ladder_path": "GATE_PASS_ADOPT_NAMED",
    },
    {
        "run": "operational_pipeline_v1 (#21)",
        "shortlist": ["repair_level_shift", "outlier_mad"],
        "clauses_cited": ["R1-1", "R3-1"],
        "adopted": {"program": "outlier_mad", "excluded_series": []},
        "support": 0.07248618783742994,
        "delayed": 0.30637972777516714,
        "ladder_path": "GATE_PASS_ADOPT_NAMED",
    },
    {
        "run": "operational_pipeline_v2 (#22)",
        "shortlist": ["repair_level_shift", "hampel_filter"],
        "clauses_cited": ["R1-1", "R1-2", "R1-3"],
        "adopted": {"program": "identity", "excluded_series": []},
        "support": 0.0,
        "delayed": 0.0,
        "ladder_path": "GATE_FAIL_FALLBACK_IDENTITY",
    },
)


def run_multi(*, k: int = K_TRAJECTORIES) -> int:
    """K trajectories, fixed before the first draw, all of them on the record."""
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_MULTI, RETRAIN_BUDGET_MULTI)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION_MULTI,
        "role": (
            "K pre-registered trajectories of the one-run operational "
            "pipeline; K is fixed before the first draw and every trajectory "
            "is reported whatever it does"
        ),
        "pre_registered": PRE_REGISTERED,
        "k": k,
        "k_rule": (
            "fixed before the run.  No early exit, no re-throw, no seeded "
            "harm.  A block stops its own trajectory only; the whole protocol "
            "aborts only on CONCURRENT_WRITE_ABORT or INSTRUMENT_DRIFT."
        ),
        "p1_touched_files": [dict(row) for row in P1_TOUCHED],
        "a1_touched_files": [dict(row) for row in A1_TOUCHED],
        "fix_c_touched_files": [dict(row) for row in FIX_C_TOUCHED],
        "relay_outage_on_record": RELAY_OUTAGE_ON_RECORD,
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "llm_call_budget": LLM_CALL_BUDGET_MULTI,
        "retrain_budget": RETRAIN_BUDGET_MULTI,
    }
    for row in payload["p1_touched_files"]:
        row["sha256_after_p1"] = before[row["path"]]
    for row in payload["a1_touched_files"]:
        row["sha256_after_a1"] = before[row["path"]]
    for row in payload["fix_c_touched_files"]:
        row["sha256_after_fix_c"] = before[row["path"]]
    trajectories: list[dict[str, Any]] = []
    payload["trajectories"] = trajectories
    global_abort: str | None = None
    try:
        payload["p2_non_regression_gate"] = stage_p2_precondition()
        payload["guard_after_p2"] = _guard_frozen(before, "the first trajectory")
        cohort, cap, roster = _cohort()
        payload["cohort"] = {
            "name": COHORT_NAME, "roster": roster,
            "eval_uids": list(cohort["eval_uids"]),
            "rebuilt_from": "what #17 materialized; no csv is parsed again",
        }
        for index in range(1, int(k) + 1):
            label = "T%d" % index
            sink: dict[str, Any] = {"label": label, "entered": False}
            trajectories.append(sink)
            if budget.left <= 0 or budget.retrains_left <= 0:
                sink.update({
                    "entered": False,
                    "verdict": "BUDGET_EXHAUSTED_BEFORE_ENTRY",
                    "verdict_reason": (
                        "the budget was spent by the earlier trajectories; "
                        "this one never started and is reported as such"
                    ),
                    "llm_calls": 0, "consumer_retrains": 0,
                })
                continue
            _trajectory(
                label=label, budget=budget, cohort=cohort, cap=cap, sink=sink,
            )
            _guard_frozen(before, "the next trajectory")
    except ConcurrentWrite as exc:
        global_abort = "CONCURRENT_WRITE_ABORT"
        payload["global_abort_reason"] = str(exc)
    except Blocked as exc:
        if exc.verdict in ("INSTRUMENT_DRIFT", "BUDGET_EXCEEDED"):
            global_abort = exc.verdict
            payload["global_abort_reason"] = exc.reason
        else:
            raise
    aggregate = _aggregate(trajectories)
    payload["aggregate"] = aggregate
    payload["shortlist_stability"] = _shortlist_stability(
        trajectories, PRIOR_TASK_A_DRAWS,
    )
    payload.update({
        "overall_verdict": global_abort or aggregate["verdict"],
        "overall_verdict_reason": (
            payload.get("global_abort_reason") or aggregate["reason"]
        ),
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "exposure": {
            "windows_read": [
                "2024 development training anchors (indices 120-900)",
                "task_A/probe/task_B inside the 2024 development partition",
                "task_C [9864, 10152] and task_D [10560, 10848], both inside "
                "the 2025 partition #17 already opened",
            ],
            "beyond_17520": "SEALED, not read",
            "unopened_partition_read": "none",
            "fresh_claim": "none: development level throughout",
        },
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload, out_json=OUT_JSON_MULTI, out_md=OUT_MD_MULTI)


def run(*, gate_only: bool = False, force: bool = False) -> int:
    """The single-trajectory path #22 ran.  Kept for reproducibility."""
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_TOTAL, RETRAIN_BUDGET)
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": "one continuous, un-relayed run of the V1 Harness",
        "pre_registered": PRE_REGISTERED,
        "p1_touched_files": [dict(row) for row in P1_TOUCHED],
        "a1_touched_files": [dict(row) for row in A1_TOUCHED],
        "fix_c_touched_files": [dict(row) for row in FIX_C_TOUCHED],
        "relay_outage_on_record": RELAY_OUTAGE_ON_RECORD,
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "retrain_budget": RETRAIN_BUDGET,
    }
    for row in payload["p1_touched_files"]:
        row["sha256_after_p1"] = before[row["path"]]
    for row in payload["a1_touched_files"]:
        row["sha256_after_a1"] = before[row["path"]]
    for row in payload["fix_c_touched_files"]:
        row["sha256_after_fix_c"] = before[row["path"]]
    payload["p2_non_regression_gate"] = stage_p2_precondition()
    if gate_only:
        payload.update({
            "overall_verdict": "P2_ONLY",
            "overall_verdict_reason": "asked to stop after the precondition",
            "llm_call_count": 0, "consumer_retrains_total": 0,
            "wall_seconds": time.perf_counter() - started,
            "frozen_surface_after": _verify(before),
        })
        return _write(payload, dry=True)
    if OUT_JSON.exists() and not force:
        raise SystemExit(
            "%s already holds the #22 record; pass --force to overwrite it"
            % _repo_rel(OUT_JSON)
        )
    cohort, cap, roster = _cohort()
    payload["cohort"] = {
        "name": COHORT_NAME, "roster": roster,
        "eval_uids": list(cohort["eval_uids"]),
    }
    sink: dict[str, Any] = {}
    payload["trajectory_record"] = sink
    _trajectory(label="single", budget=budget, cohort=cohort, cap=cap, sink=sink)
    for key in ("store", "lifecycle", "attribution", "slow_edit"):
        if key in (sink.get("stages") or {}):
            payload[key] = sink["stages"][key]
    payload["trajectory"] = sink.get("trajectory") or []
    payload.update({
        "overall_verdict": sink.get("verdict"),
        "overall_verdict_reason": sink.get("verdict_reason"),
        "overall_detail": sink.get("verdict_detail") or {},
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    return _write(payload)


# --------------------------------------------- Part B: the banked completion
BANKED_SOURCE = E2 / "operational_pipeline_v3.json"
BANKED_ROOT = WORK_ROOT / "banked"
BANKED_STEPS = ("task_A", "task_B", "task_C")


class _BankedRoster:
    """Just the two rosters the Slow evidence block quotes.  Measures nothing."""

    def __init__(self, train: Sequence[str], evaluation: Sequence[str]) -> None:
        self.train_uids = [str(uid) for uid in train]
        self.eval_uids = [str(uid) for uid in evaluation]


def _banked_record(row: Mapping[str, Any], card_id: str) -> dict[str, Any]:
    """A #23 ledger row in the shape the existing fold already consumes.

    Every field is copied from the row and named with where it came from.
    The one thing the ledger does not carry is a mask candidate, so for a
    trajectory that asked for a mask round the reconstructed candidate set
    is the full-batch plans alone; that is declared, not smoothed over.
    """
    pool = dict((row.get("adoption_ladder") or {}).get("full_batch_pool") or {})
    per_series = dict(
        row.get("per_eval_series_delayed_after_gate")
        or row.get("per_eval_series_delayed_before_gate") or {}
    )
    harmed = [
        uid for uid, value in per_series.items()
        if float(value) < HARM_THRESHOLD
    ]
    local = (
        [str(row["local_skill_expected"])]
        if row.get("local_skill_hit") and row.get("local_skill_expected") else []
    )
    return {
        "episode_id": row.get("episode_id"),
        "mode": row.get("mode"),
        "final_plan": dict(row.get("plan_after_gate") or {}),
        "support_results": {
            program: {"aggregate_gain": float(value)}
            for program, value in pool.items()
        },
        "support": {"aggregate_gain": row.get("support_after_gate")},
        "delayed": {
            "aggregate_gain": row.get("delayed_after_gate"),
            "harmed_eval_series": sorted(harmed),
            "harmed_eval_series_count": len(harmed),
            "harmed_eval_series_total_harm": float(
                -sum(per_series[uid] for uid in harmed)
            ),
            "per_eval_series_gain": per_series,
        },
        "adoption_ladder": dict(row.get("adoption_ladder") or {}),
        "retrieval": {"resolved_skill_ids": local + [card_id]},
        "reconstruction": {
            "support_results": "adoption_ladder.full_batch_pool, full-batch only",
            "mask_candidate_absent": (
                "the ledger row does not carry the mask round's plan, so "
                "CANDIDATE_SELECTION is evaluated over full-batch plans "
                "alone.  A trajectory that asked for no mask is "
                "evidence-complete and serves as the control."
            ),
            "delayed": "the row's own per-series vector; the harm ledger is recomputed from it",
            "retrieval": "the skills the row records as hit",
        },
    }


def _banked_measure(row: Mapping[str, Any]) -> tuple[Any, Any]:
    """Measurement closures that serve identity only, at zero cost.

    A veto walks to the ladder's fallback.  Identity's readings against
    identity are all zero and need no Consumer; anything else would need a
    real retrain, which this stage does not have, so it stops instead of
    quietly measuring.
    """
    uids = list(
        (row.get("per_eval_series_delayed_after_gate")
         or row.get("per_eval_series_delayed_before_gate") or {}).keys()
    )
    zero = {
        "aggregate_gain": 0.0,
        "harmed_eval_series_count": 0,
        "harmed_eval_series_total_harm": 0.0,
        "harmed_eval_series": [],
        "per_eval_series_gain": {uid: 0.0 for uid in uids},
    }

    def served(program: str, excluded: Sequence[str]) -> Mapping[str, Any]:
        if str(program) == IDENTITY and not list(excluded):
            return dict(zero)
        raise Blocked(
            "REPLAY_NEEDS_MEASUREMENT",
            "the re-adjudication would have to measure %r, which this "
            "zero-retrain replay cannot do" % program,
        )

    return served, served


def _readjudicate(
    *, snapshot: Any, row: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-run the tracked Scope/Risk gate over a banked episode.  0 retrains."""
    ladder = dict(row.get("adoption_ladder") or {})
    plan = dict(row.get("plan_after_gate") or {})
    per_series = dict(
        row.get("per_eval_series_delayed_after_gate")
        or row.get("per_eval_series_delayed_before_gate") or {}
    )
    harmed = [
        uid for uid, value in per_series.items()
        if float(value) < HARM_THRESHOLD
    ]
    delayed = {
        "aggregate_gain": row.get("delayed_after_gate"),
        "harmed_eval_series_count": len(harmed),
        "harmed_eval_series_total_harm": float(
            -sum(per_series[uid] for uid in harmed)
        ),
        "harmed_eval_series": sorted(harmed),
        "per_eval_series_gain": per_series,
    }
    support = {
        "aggregate_gain": row.get("support_after_gate"),
        "harmed_eval_series_count": 0,
        "harmed_eval_series_total_harm": 0.0,
        "harmed_eval_series": [],
        "per_eval_series_gain": {},
    }
    delayed_of, support_of = _banked_measure(row)
    gate = harness_compiler.enforce_scope_risk_guards(
        snapshot=snapshot,
        ladder={
            "final_plan": plan, "support": support, "delayed": delayed,
            "support_winner": ladder.get("support_winner"),
            "support_winner_full_batch_delayed": ladder.get(
                "support_winner_full_batch_delayed"
            ),
        },
        eval_count=len(per_series),
        reused=bool(row.get("reuse_adopted")),
        delayed_of=delayed_of, support_of=support_of,
    )
    after_series = dict(
        (gate.get("delayed_after") or {}).get("per_eval_series_gain") or {}
    )
    return {
        "step": row.get("step"),
        "plan_before": plan,
        "plan_after": dict(gate["plan_after"]),
        "changed": bool(gate.get("changed")),
        "delayed_before": row.get("delayed_after_gate"),
        "delayed_after": (gate.get("delayed_after") or {}).get("aggregate_gain"),
        "per_series_before": per_series,
        "per_series_after": after_series,
        "harmed_before": sorted(harmed),
        "harmed_after": list(
            (gate.get("delayed_after") or {}).get("harmed_eval_series") or ()
        ),
        "gate": gate,
        "consumer_retrains": 0,
    }


def stage_banked(
    budget: Budget, slow_model: str = SLOW_MODEL,
) -> dict[str, Any]:
    """B1-B5: finish links 6 to 9 on the #23 banked trajectories.

    The selector, the fold, the Slow edit and the compiler are the real
    ones.  What is banked is the measurement: every reading below was taken
    by #23 and is replayed, never re-measured.
    """
    started = time.perf_counter()
    out: dict[str, Any] = {
        "ran": True,
        "level": "MECHANISM",
        "source": _repo_rel(BANKED_SOURCE),
        "source_sha256": _sha256(BANKED_SOURCE),
        "what_is_banked": (
            "every Consumer reading comes from #23; this stage measures "
            "nothing and charges no retrains"
        ),
        "evidence_level": (
            "development.  These NOAA windows had their outcome opened by "
            "#17; nothing here is fresh confirmation and no A5-vs-A3 result "
            "is produced."
        ),
    }
    banked = json.loads(BANKED_SOURCE.read_text(encoding="utf-8"))
    cohort_block = banked.get("cohort") or {}
    roster = _BankedRoster(
        (cohort_block.get("roster") or {}).get("train") or (),
        cohort_block.get("eval_uids") or (),
    )
    banked_root = BANKED_ROOT / slow_model
    if banked_root.exists():
        shutil.rmtree(banked_root)
    banked_root.mkdir(parents=True, exist_ok=True)
    out["slow_model"] = slow_model

    # ---- B1: selector and attribution on every banked trajectory ----------
    b1: dict[str, Any] = {}
    for t in banked["trajectories"]:
        label = str(t["label"])
        rows = [r for r in t["trajectory"] if r["step"] in BANKED_STEPS]
        selection = select_scope_risk_episode(rows)
        row = next(
            (r for r in rows if r["step"] == selection.get("selected_step")), None
        )
        attribution = None
        if row is not None:
            bootstrap = [
                item.skill_id for item in compile_snapshot(
                    FC.H0_ROOT, verify_lock=False).skills
            ]
            record = _banked_record(row, CARD_ID)
            folded = SSU._attribute(
                record, bootstrap_ids=bootstrap,
                capability_skills_present=True, transport_per_series=True,
            )
            control = SSU._attribute(
                record, bootstrap_ids=bootstrap,
                capability_skills_present=True, transport_per_series=False,
            )
            # the shape stage_attribution produces, so the same Slow stage
            # consumes a banked fold and a live one without knowing which
            attribution = {
                "ran": True,
                "evidence": "one banked episode of this trajectory, replayed",
                "cause_code": folded["attribution"]["cause_code"],
                "first_stage": folded["attribution"]["first_stage"],
                "is_scope_risk_face": bool(folded["is_scope_risk_face"]),
                "with_per_series_risk_reading": folded,
                "aggregate_only_control": control,
            }
        b1[label] = {
            "role": (
                "the risk evidence" if label == "T1"
                else "deterministic consistency replica, not independent "
                     "risk evidence"
            ),
            "selection": _public(selection),
            "attribution": _public(attribution),
            "cause_code": attribution and attribution["cause_code"],
            "aggregate_only_cause": attribution and attribution[
                "aggregate_only_control"]["attribution"]["cause_code"],
            "is_scope_risk_face": bool(
                attribution and attribution["is_scope_risk_face"]
            ),
            "mask_round_in_this_trajectory": bool(
                (row or {}).get("consumer_retrains", 0) > 40
            ),
        }
        print(
            "B1 %-4s selector=%s -> %s | fold=%s" % (
                label, selection["verdict"], selection.get("selected_step"),
                b1[label]["cause_code"],
            ),
            flush=True,
        )
    out["b1_selector_and_attribution"] = b1
    lead = b1.get("T1") or {}
    if not lead.get("is_scope_risk_face"):
        out.update({
            "verdict": "ATTRIBUTION_STILL_MISSES",
            "reason": "T1 did not fold to a Scope/Risk face: %s"
                      % lead.get("cause_code"),
            "wall_seconds": time.perf_counter() - started,
        })
        return out

    # ---- the frozen T1 store fork -----------------------------------------
    source_store = STORE_ROOT / "T1" / SLOT
    if not source_store.is_dir():
        out.update({
            "verdict": "ATTRIBUTION_STILL_MISSES",
            "reason": "the #23 T1 store fork is gone: %s" % _repo_rel(source_store),
            "wall_seconds": time.perf_counter() - started,
        })
        return out
    fork = banked_root / "T1"
    shutil.copytree(source_store, fork)
    store = SnapshotStore(fork / "snapshots")
    active = json.loads(store.active_path.read_text(encoding="utf-8"))
    snapshot = compile_snapshot(
        fork / "snapshots" / str(active["runtime_bundle_sha"]), verify_lock=False,
    )
    slot = {"slot": SLOT, "_store": store, "_snapshot": snapshot,
            "runtime_bundle_sha": snapshot.runtime_bundle_sha}
    out["store_fork"] = {
        "copied_from": _repo_rel(source_store),
        "working_copy": _repo_rel(fork),
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "guards_at_start": harness_compiler.scope_risk_guards_of(snapshot),
        "the_v3_fork_is_left_untouched": True,
    }

    # ---- B2/B3: one Slow draw, then the compiler --------------------------
    t1 = next(t for t in banked["trajectories"] if str(t["label"]) == "T1")
    rows = [r for r in t1["trajectory"] if r["step"] in BANKED_STEPS]
    fault_row = next(
        r for r in rows
        if r["step"] == b1["T1"]["selection"]["selected_step"]
    )
    fault = {
        "record": _banked_record(fault_row, CARD_ID),
        "row": fault_row,
        "search": roster,
        "earlier_rows": [],
    }
    slow_sink: dict[str, Any] = {"ran": False}
    out["b2_slow"] = slow_sink
    try:
        stage_slow(
            slot=slot,
            attribution=b1["T1"]["attribution"],
            task_c=fault, budget=budget, sink=slow_sink,
            fault_step=str(fault_row["step"]), slow_model=slow_model,
        )
    except Blocked as exc:
        out.update({
            "verdict": exc.verdict,
            "reason": exc.reason,
            "wall_seconds": time.perf_counter() - started,
        })
        return out
    print(
        "B2 slow %s -> %s | snapshot %s -> %s" % (
            slow_sink["guard"]["statistic"], slow_sink["guard"]["action"],
            slow_sink["snapshot_before"][:12], slow_sink["snapshot_after"][:12],
        ),
        flush=True,
    )

    # ---- B4: re-adjudicate the banked episodes, zero retrains -------------
    patched = slot["_snapshot"]
    readjudication = {}
    for row in rows:
        readjudication[str(row["step"])] = _readjudicate(
            snapshot=patched, row=row,
        )
        r = readjudication[str(row["step"])]
        print(
            "B4 %-8s %s -> %s | delayed %s -> %s | harmed %s -> %s" % (
                r["step"], SSU._plan_label(r["plan_before"]),
                SSU._plan_label(r["plan_after"]), _fmt(r["delayed_before"]),
                _fmt(r["delayed_after"]), r["harmed_before"] or "none",
                r["harmed_after"] or "none",
            ),
            flush=True,
        )
    out["b4_readjudication"] = readjudication
    fault_step = str(fault_row["step"])
    lead_row = readjudication[fault_step]
    watched = sorted(lead_row["harmed_before"])
    contained = all(
        float(lead_row["per_series_after"].get(uid, 0.0)) >= HARM_THRESHOLD
        for uid in watched
    )
    untouched = [
        step for step, r in readjudication.items()
        if step != fault_step and not r["changed"]
        and r["plan_before"] == r["plan_after"]
    ]
    collateral = [
        step for step, r in readjudication.items()
        if step != fault_step and (
            r["changed"] or r["plan_before"] != r["plan_after"]
        )
    ]
    out["b4_reading"] = {
        "fault_step": fault_step,
        "flipped": bool(lead_row["changed"]),
        "characterisation": (
            "containment, not a utility improvement: the %s harm on %s is "
            "removed and the %s aggregate gain is given up with it.  Keeping "
            "the gain is a BY_RESCOPE question and is not attempted here."
            % (
                _fmt(min(lead_row["per_series_before"].values())
                     if lead_row["per_series_before"] else None),
                watched or "no series",
                _fmt(lead_row["delayed_before"]),
            )
        ),
        "watched_series": watched,
        "contained": contained,
        "unrelated_episodes_unchanged": untouched,
        "unrelated_episodes_that_moved": collateral,
        "no_collateral_means": (
            "task_B/C/D were already identity in #23 and must stay identity "
            "bit for bit; it does not mean the guard is free"
        ),
    }
    if not lead_row["changed"]:
        verdict, reason = "REPLAY_NO_CHANGE", (
            "the guard is live in the patched snapshot and the banked "
            "adoption did not move"
        )
    elif not contained:
        verdict, reason = "REPLAY_NO_CHANGE", (
            "the adoption moved but %s still crosses the line" % watched
        )
    elif collateral:
        verdict, reason = "REPLAY_NO_CHANGE", (
            "unrelated banked episodes moved: %s" % collateral
        )
    else:
        verdict, reason = "BANKED_CHAIN_CLOSES_ON_%s" % (
            slow_model.upper().replace("-", "_").replace(".", "_")
        ), (
            "selector -> RISK_GAP at %s -> one Slow proposal -> compiler "
            "accepted -> the banked adoption is contained to identity, %s no "
            "longer crosses %+.3f, and every unrelated banked episode is "
            "unchanged" % (fault_step, watched, HARM_THRESHOLD)
        )
    out.update({
        "verdict": verdict, "reason": reason,
        "llm_calls": int(slow_sink.get("llm_calls") or 0),
        "consumer_retrains": 0,
        "wall_seconds": time.perf_counter() - started,
    })
    return out


LLM_CALL_BUDGET_V4 = 12
RETRAIN_BUDGET_V4 = 250
# v4 is the #24 record and is never rewritten; this round writes v5.
OUT_JSON_V4 = E2 / "operational_pipeline_v4.json"
OUT_MD_V4 = E2 / "operational_pipeline_v4.md"
OUT_JSON_V5 = E2 / "operational_pipeline_v5.json"
OUT_MD_V5 = E2 / "operational_pipeline_v5.md"
EVIDENCE_LEVEL_NOTE = (
    "DEVELOPMENT.  Every window this protocol reads had its outcome opened "
    "by #17, so nothing here is fresh confirmation and nothing here produces "
    "a new A5-over-A3 result.  What a closed chain shows is that the "
    "machinery runs end to end, not that the method is better."
)


def run_v4(*, banked: bool = True, post_fix_live: bool = False) -> int:
    """Part B on the banked ledger, then Part C live only if B closed."""
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_V4, RETRAIN_BUDGET_V4)
    payload: dict[str, Any] = {
        "protocol_version": "operational_pipeline_v5",
        "role": (
            "finish links 6 to 9 on the #23 banked trajectories after the "
            "attribution window was rewired, then, only if that closes, one "
            "post-fix live trajectory"
        ),
        "the_only_change_this_round": (
            "stage_slow's error classification and its diagnostic record.  "
            "The decision backend stays on the same Opus configuration: "
            "rewiring the report and swapping the deciding model are not "
            "put in the same cut.  The selector, the fold, the thresholds, "
            "the ladder, the two keys, the menu, the prompts, the "
            "SELECTION_MISS adapter, the T1 fork, the public input, the "
            "Harness view and the Slow prompt are all untouched."
        ),
        "slow_error_taxonomy": SLOW_ERROR_CLASSES,
        "sampling_rule": SAMPLING_RULE,
        "the_only_method_change_last_round": (
            "which episode attribution is handed.  Observation, Program, "
            "Guidance, the Slow grammar, the Risk thresholds, the adoption "
            "ladder, the two keys, the menu, the prompts and the "
            "SELECTION_MISS adapter reading are all untouched."
        ),
        "evidence_level": EVIDENCE_LEVEL_NOTE,
        "selector_contract": SELECTOR_CONTRACT,
        "pre_registered": PRE_REGISTERED,
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "llm_call_budget": LLM_CALL_BUDGET_V4,
        "retrain_budget": RETRAIN_BUDGET_V4,
    }
    try:
        payload["p2_non_regression_gate"] = stage_p2_precondition()
        payload["guard_after_precondition"] = _guard_frozen(before, "part B")
        if banked:
            payload["part_b_banked_completion"] = stage_banked(budget)
            payload["guard_after_part_b"] = _guard_frozen(before, "part C")
        b = payload.get("part_b_banked_completion") or {}
        if post_fix_live and b.get("verdict") == "BANKED_CHAIN_CLOSES":
            cohort, cap, roster = _cohort()
            payload["cohort"] = {
                "name": COHORT_NAME, "roster": roster,
                "eval_uids": list(cohort["eval_uids"]),
            }
            sink: dict[str, Any] = {"label": "post_fix_live", "entered": False}
            payload["part_c_post_fix_live"] = sink
            _trajectory(
                label="post_fix_live", budget=budget, cohort=cohort, cap=cap,
                sink=sink,
            )
        elif post_fix_live:
            payload["part_c_post_fix_live"] = {
                "ran": False,
                "why_not": (
                    "part B did not close (%s), and part C is gated on it"
                    % b.get("verdict")
                ),
            }
    except ConcurrentWrite as exc:
        payload.update({
            "overall_verdict": "CONCURRENT_WRITE_ABORT",
            "overall_verdict_reason": str(exc),
        })
    except Blocked as exc:
        payload.update({
            "overall_verdict": exc.verdict,
            "overall_verdict_reason": exc.reason,
        })
    if "overall_verdict" not in payload:
        b = payload.get("part_b_banked_completion") or {}
        c = payload.get("part_c_post_fix_live") or {}
        if c.get("verdict"):
            payload["overall_verdict"] = (
                "DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX"
                if c["verdict"] == "OPERATIONAL_PIPELINE_CLOSES"
                else str(c["verdict"])
            )
            payload["overall_verdict_reason"] = (
                "banked chain: %s; post-fix live trajectory: %s -- %s"
                % (b.get("verdict"), c.get("verdict"), c.get("verdict_reason"))
            )
        else:
            payload["overall_verdict"] = str(b.get("verdict"))
            payload["overall_verdict_reason"] = str(b.get("reason"))
    payload.update({
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "exposure": {
            "windows_read": [
                "the #23 ledger, replayed",
                "for part C only: task_A/probe/task_B/task_C/task_D, all "
                "inside partitions #17 already opened",
            ],
            "beyond_17520": "SEALED, not read",
            "fresh_claim": "none",
        },
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload, out_json=OUT_JSON_V5, out_md=OUT_MD_V5)


OUT_JSON_V6 = E2 / "operational_pipeline_v6.json"
OUT_MD_V6 = E2 / "operational_pipeline_v6.md"
LLM_CALL_BUDGET_V6 = 12


def run_v6(models: Sequence[str]) -> int:
    """B2 once per pinned backend, each on its own copy of the T1 fork.

    Only the deciding model differs between runs: the banked fork, the
    public input, the authorized surface, the evidence and the sampling
    discipline are the same object in every pass.  A verdict that closes
    carries the model in its name, because what it shows is that the chain
    runs on that model -- not that the method is better.
    """
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_V6, RETRAIN_BUDGET_V4)
    payload: dict[str, Any] = {
        "protocol_version": "operational_pipeline_v6",
        "role": (
            "links 6 to 9 on the #23 banked ledger, run once per pinned Slow "
            "backend after the relay outage took claude-opus-5 out"
        ),
        "why_not_opus": RELAY_OUTAGE_ON_RECORD,
        "models": list(models),
        "evidence_level": EVIDENCE_LEVEL_NOTE,
        "claim_discipline": (
            "a closing verdict is suffixed with the model that closed it.  "
            "This is a chain-liveness reading on that backend, not an Opus "
            "reading, and not a reading about the method."
        ),
        "what_is_identical_across_the_runs": [
            "the #23 T1 banked ledger and its store fork",
            "the selector, the fold and their outputs",
            "the authorized surface verification.rules.scope_risk_guards",
            "the public input handed to the Slow stage",
            "the sampling discipline and the three counters",
        ],
        "what_differs": "the deciding model, and nothing else",
        "selector_contract": SELECTOR_CONTRACT,
        "slow_error_taxonomy": SLOW_ERROR_CLASSES,
        "sampling_rule": SAMPLING_RULE,
        "model_history_on_this_surface": MODEL_HISTORY_ON_THIS_SURFACE,
        "fix_c_touched_files": [dict(row) for row in FIX_C_TOUCHED],
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "llm_call_budget": LLM_CALL_BUDGET_V6,
        "retrain_budget": RETRAIN_BUDGET_V4,
    }
    for row in payload["fix_c_touched_files"]:
        row["sha256_after_fix_c"] = before[row["path"]]
    runs: dict[str, Any] = {}
    payload["runs"] = runs
    try:
        payload["p2_non_regression_gate"] = stage_p2_precondition()
        payload["guard_after_precondition"] = _guard_frozen(before, "the first model")
        for model in models:
            print("\n########## B2 on %s ##########" % model, flush=True)
            box: dict[str, Any] = {"slow_model": model, "entered": True}
            runs[model] = box
            try:
                box.update(stage_banked(budget, slow_model=model))
            except Blocked as exc:
                if exc.verdict in ("INSTRUMENT_DRIFT", "BUDGET_EXCEEDED"):
                    raise
                box.update({"verdict": exc.verdict, "reason": exc.reason})
            print(
                "B5 %-14s %s" % (model, box.get("verdict")), flush=True
            )
            _guard_frozen(before, "the next model")
    except ConcurrentWrite as exc:
        payload.update({"overall_verdict": "CONCURRENT_WRITE_ABORT",
                        "overall_verdict_reason": str(exc)})
    except Blocked as exc:
        payload.update({"overall_verdict": exc.verdict,
                        "overall_verdict_reason": exc.reason})
    if "overall_verdict" not in payload:
        closed = [m for m, r in runs.items()
                  if str(r.get("verdict") or "").startswith("BANKED_CHAIN_CLOSES")]
        by_verdict = {m: r.get("verdict") for m, r in runs.items()}
        if closed:
            payload["overall_verdict"] = "BANKED_CHAIN_CLOSES_IN_K_MODELS"
            payload["overall_verdict_reason"] = (
                "%d of %d pinned backends carried links 6 to 9 end to end: %s"
                % (len(closed), len(runs), closed)
            )
        else:
            payload["overall_verdict"] = "BANKED_CHAIN_NEVER_CLOSES"
            payload["overall_verdict_reason"] = (
                "no pinned backend produced a proposal the chain could carry: %s"
                % by_verdict
            )
        payload["by_model"] = by_verdict
        payload["closed_on"] = closed
    payload.update({
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "exposure": {
            "windows_read": ["the #23 ledger, replayed"],
            "beyond_17520": "SEALED, not read",
            "fresh_claim": "none",
        },
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload, out_json=OUT_JSON_V6, out_md=OUT_MD_V6)


OUT_JSON_V7 = E2 / "operational_pipeline_v7.json"
OUT_MD_V7 = E2 / "operational_pipeline_v7.md"
LLM_CALL_BUDGET_V7 = 8
RETRAIN_BUDGET_V7 = 200


def run_v7(slow_model: str = "gpt-5.6-sol") -> int:
    """One continuous live trajectory after the routing fix.

    Nothing in the instrument moves for this run.  The verdict is renamed
    at the reporting layer so it carries the model and the evidence level;
    the trajectory machinery itself is the one #23 and #26 already used.
    """
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_V7, RETRAIN_BUDGET_V7)
    banked = json.loads(BANKED_SOURCE.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "protocol_version": "operational_pipeline_v7",
        "role": (
            "the post-fix live single trajectory: a new continuous store, one "
            "sampling, and the whole nine-link chain run for real"
        ),
        "no_method_or_instrument_change_this_round": (
            "the selector, the fold, the thresholds, the ladder, the two "
            "keys, the menu, the prompts, the SELECTION_MISS adapter and the "
            "Slow grammar are all exactly as #26 left them.  The only edits "
            "are the pinned model reaching _trajectory and the verdict being "
            "renamed for the report."
        ),
        "slow_backend": {
            "model": slow_model,
            "why": (
                "the backend that closed the banked chain in #26; the relay's "
                "Claude route is still returning an error payload, and this "
                "run does not switch models mid-way even if it recovers"
            ),
            "history_on_this_surface": MODEL_HISTORY_ON_THIS_SURFACE.get(
                slow_model, "no history on record"
            ),
            "at_most_one_proposal": True,
            "no_re_throw": "one sampling; a natural stop is recorded as it stands",
        },
        "gated_on": {
            "requirement": "part B had to close before this ran",
            "closed_in": _repo_rel(BANKED_SOURCE),
            "banked_verdict": banked.get("overall_verdict"),
            "closed_on": banked.get("closed_on"),
        },
        "evidence_level": (
            "DEVELOPMENT.  Every window here had its outcome opened by #17, "
            "so this trajectory produces no fresh confirmation evidence, no "
            "new A5-over-A3 result, and no claim that the reading is "
            "backend-independent.  What a closing verdict shows is that the "
            "chain runs end to end on %s." % slow_model
        ),
        "selector_contract": SELECTOR_CONTRACT,
        "sampling_rule": SAMPLING_RULE,
        "pre_registered": PRE_REGISTERED,
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "llm_call_budget": LLM_CALL_BUDGET_V7,
        "retrain_budget": RETRAIN_BUDGET_V7,
    }
    sink: dict[str, Any] = {"label": "post_fix_live", "entered": False}
    payload["trajectory_record"] = sink
    try:
        payload["p2_non_regression_gate"] = stage_p2_precondition()
        payload["guard_after_precondition"] = _guard_frozen(before, "the trajectory")
        cohort, cap, roster = _cohort()
        payload["cohort"] = {
            "name": COHORT_NAME, "roster": roster,
            "eval_uids": list(cohort["eval_uids"]),
        }
        _trajectory(
            label="post_fix_live", budget=budget, cohort=cohort, cap=cap,
            sink=sink, slow_model=slow_model,
        )
    except ConcurrentWrite as exc:
        payload.update({"overall_verdict": "CONCURRENT_WRITE_ABORT",
                        "overall_verdict_reason": str(exc)})
    except Blocked as exc:
        payload.update({"overall_verdict": exc.verdict,
                        "overall_verdict_reason": exc.reason})
    if "overall_verdict" not in payload:
        suffix = slow_model.upper().replace("-", "_").replace(".", "_")
        raw = str(sink.get("verdict"))
        selector = ((sink.get("stages") or {}).get("scope_risk_selector") or {})
        if raw == "OPERATIONAL_PIPELINE_CLOSES":
            verdict = "DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX_ON_%s" % suffix
        elif raw == "PIPELINE_RUNS_NO_FAULT_SAMPLE":
            verdict = str(selector.get("verdict") or "NO_ELIGIBLE_SCOPE_RISK_EPISODE")
        else:
            verdict = raw
        payload.update({
            "overall_verdict": verdict,
            "overall_verdict_reason": sink.get("verdict_reason"),
            "trajectory_verdict_before_renaming": raw,
            "renaming_rule": (
                "a closing verdict carries the deciding model and the "
                "development evidence level; a no-fault stop is reported "
                "under the selector's own verdict rather than a generic one"
            ),
        })
    payload.update({
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "exposure": {
            "windows_read": [
                "2024 development training anchors (indices 120-900)",
                "task_A/probe/task_B inside the 2024 development partition",
                "task_C [9864, 10152] and task_D [10560, 10848], both inside "
                "the 2025 partition #17 already opened",
            ],
            "beyond_17520": "SEALED, not read",
            "fresh_claim": "none",
        },
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload, out_json=OUT_JSON_V7, out_md=OUT_MD_V7)


def _markdown_models(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Banked chain, one pinned Slow backend at a time",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        payload.get("claim_discipline", ""),
        "",
        "## Per model",
        "",
        "| model | verdict | valid samples | protocol-failed | LLM |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for model, run in (payload.get("runs") or {}).items():
        counters = ((run.get("b2_slow") or {}).get("counters") or {})
        lines.append(
            "| `%s` | `%s` | %s | %s | %s |" % (
                model, run.get("verdict"),
                counters.get("valid_decision_samples"),
                counters.get("protocol_failed_draws"),
                counters.get("llm_calls_spent"),
            )
        )
    for model, run in (payload.get("runs") or {}).items():
        lines.extend(["", "### `%s` -- `%s`" % (model, run.get("verdict")), "",
                      str(run.get("reason") or "")])
        slow = run.get("b2_slow") or {}
        for a in slow.get("attempts") or ():
            bits = [str(a.get("outcome"))]
            if a.get("no_proposal_reason"):
                bits.append("reason %s" % a["no_proposal_reason"])
            if (a.get("diagnostics") or {}).get("validation_error_codes"):
                bits.append("codes %s" % a["diagnostics"]["validation_error_codes"])
            lines.append("- draw %s: %s" % (a.get("draw"), "; ".join(bits)))
        guard = slow.get("guard")
        if guard:
            lines.extend([
                "",
                "Guard `%s`: %s `%s` %s -> %s on the %s window, applies to %s."
                % (guard.get("guard_id"), guard.get("statistic"),
                   guard.get("comparator"), _fmt(guard.get("threshold")),
                   guard.get("action"), guard.get("window"),
                   guard.get("applies_to")),
                "", "> %s" % guard.get("rationale"), "",
            ])
        readj = run.get("b4_readjudication") or {}
        if readj:
            lines.extend([
                "| step | plan before | plan after | delayed before | "
                "delayed after | harmed before | harmed after |",
                "| --- | --- | --- | ---: | ---: | --- | --- |",
            ])
            for step, r in readj.items():
                lines.append(
                    "| `%s` | `%s` | `%s` | %s | %s | %s | %s |" % (
                        step, SSU._plan_label(r["plan_before"]),
                        SSU._plan_label(r["plan_after"]),
                        _fmt(r["delayed_before"]), _fmt(r["delayed_after"]),
                        ", ".join(r["harmed_before"]) or "none",
                        ", ".join(r["harmed_after"]) or "none",
                    )
                )
            lines.append("")
    after = payload.get("frozen_surface_after") or {}
    lines.extend([
        "", "## Cost and integrity", "",
        "- LLM calls: %s / %s." % (payload.get("llm_call_count"),
                                   payload.get("llm_call_budget")),
        "- Consumer retrains: %s." % payload.get("consumer_retrains_total"),
        "- Frozen surface: %s files, drift %s." % (after.get("files"),
                                                   after.get("drift")),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _markdown_multi(payload: Mapping[str, Any]) -> str:
    lines = [
        "# K trajectories of the one-run operational pipeline",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        "K = %s, fixed before the first draw. Every trajectory is on the "
        "record whatever it did: none was discarded, re-thrown or seeded. "
        "Each ran on a store of its own, %s x %s, arm %s, Slow pinned to "
        "`%s`." % (
            payload.get("k"), COHORT_NAME, VARIANT, ARM, SLOW_MODEL,
        ),
        "",
        "## Per trajectory",
        "",
        "| trajectory | verdict | reached Slow | LLM | retrains |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for t in payload.get("trajectories") or ():
        reached = bool(((t.get("stages") or {}).get("slow_edit") or {}).get("ran"))
        lines.append(
            "| `%s` | `%s` | %s | %d | %d |" % (
                t.get("label"), t.get("verdict"), "yes" if reached else "no",
                int(t.get("llm_calls") or 0),
                int(t.get("consumer_retrains") or 0),
            )
        )
    lines.extend(["", "### Reasons", ""])
    for t in payload.get("trajectories") or ():
        lines.append("- `%s`: %s" % (t.get("label"), t.get("verdict_reason")))
    lines.extend(["", "## Trajectory tables", ""])
    for t in payload.get("trajectories") or ():
        rows = t.get("trajectory") or []
        lines.extend([
            "### `%s` -- `%s`" % (t.get("label"), t.get("verdict")), "",
        ])
        if not rows:
            lines.extend(["No task window completed.", ""])
            continue
        lines.extend([
            "| step | window | mode | card | local Skill | shortlist | "
            "cited | overlap | plan before | plan after | support | delayed | "
            "harmed | retrains | LLM |",
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | "
            "---: | ---: | --- | ---: | ---: |",
        ])
        for row in rows:
            overlap = row.get("clause_shortlist_overlap") or {}
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %s | `%s` | `%s` | %s "
                "| %s | %s | %d | %d |" % (
                    row.get("step"),
                    (row.get("window") or {}).get("start", "--"),
                    row.get("mode") or "--",
                    "hit" if row.get("card_hit") else (
                        "--" if row.get("card_hit") is None else "MISS"
                    ),
                    "--" if row.get("local_skill_hit") is None else (
                        "hit" if row.get("local_skill_hit") else "MISS"
                    ),
                    ", ".join(row.get("shortlist") or ()) or "--",
                    ", ".join(row.get("clauses_cited") or ()) or "--",
                    overlap.get("overlap_count", "--"),
                    SSU._plan_label(row.get("plan_before_gate")),
                    SSU._plan_label(row.get("plan_after_gate")),
                    _fmt(row.get("support_before_gate")),
                    _fmt(row.get("delayed_before_gate")),
                    ", ".join(row.get("harmed_before_gate") or ()) or "none",
                    int(row.get("consumer_retrains") or 0),
                    int(row.get("llm_calls") or 0),
                )
            )
        lines.append("")
        slow = (t.get("stages") or {}).get("slow_edit") or {}
        if slow.get("ran"):
            lines.extend(["**Slow draws**", ""])
            for a in slow.get("attempts") or ():
                lines.append(
                    "- draw %s on `%s`: %s%s%s" % (
                        a.get("draw"), a.get("model"), a.get("outcome"),
                        "" if not a.get("no_proposal_reason")
                        else " (%s)" % a["no_proposal_reason"],
                        "" if not a.get("validation_error_codes")
                        else " retries %s" % a["validation_error_codes"],
                    )
                )
            g = slow.get("guard") or {}
            if g:
                lines.extend([
                    "",
                    "Guard `%s`: %s `%s` %s -> %s on the %s window, applies to "
                    "%s." % (
                        g.get("guard_id"), g.get("statistic"),
                        g.get("comparator"), _fmt(g.get("threshold")),
                        g.get("action"), g.get("window"), g.get("applies_to"),
                    ),
                    "",
                    "> %s" % g.get("rationale"),
                ])
            lines.append("")
    stab = payload.get("shortlist_stability") or {}
    if stab.get("draws"):
        lines.extend([
            "## task_A shortlist stability, all draws on record",
            "",
            "| run | shortlist | cited | overlap | adopted | support | delayed "
            "| ladder |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | --- |",
        ])
        for row in stab["draws"]:
            lines.append(
                "| %s | %s | %s | %s | `%s` | %s | %s | `%s` |" % (
                    row.get("run"), ", ".join(row.get("shortlist") or ()),
                    ", ".join(row.get("clauses_cited") or ()),
                    row.get("overlap_count"),
                    SSU._plan_label(row.get("adopted")),
                    _fmt(row.get("support")), _fmt(row.get("delayed")),
                    row.get("ladder_path"),
                )
            )
        lines.extend([
            "",
            "%d draws, %d distinct shortlists, %d adopted a program, %d cited "
            "a clause naming something they shortlisted. Overlap is computed "
            "from the clause texts against the frozen menu, not from what the "
            "Agent reported. n is small; nothing here is causal." % (
                stab.get("n", 0), stab.get("distinct_shortlists", 0),
                stab.get("adopted_a_program", 0),
                stab.get("cited_clauses_naming_a_shortlisted_program", 0),
            ),
            "",
        ])
    after = payload.get("frozen_surface_after") or {}
    lines.extend([
        "## Cost and integrity",
        "",
        "- LLM calls: %s / %s." % (
            payload.get("llm_call_count"), payload.get("llm_call_budget")
        ),
        "- Consumer retrains: %s / %s." % (
            payload.get("consumer_retrains_total"),
            payload.get("retrain_budget"),
        ),
        "- Frozen surface: %s files, drift %s." % (
            after.get("files"), after.get("drift")
        ),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("runs") is not None:
        return _markdown_models(payload)
    if payload.get("trajectories") is not None:
        return _markdown_multi(payload)
    lines = [
        "# One continuous operational run",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        "One un-relayed run of the V1 Harness on %s x %s, arm %s, Slow pinned "
        "to `%s`. Development level: every window was locked before the run "
        "from the #17/#19 registers, nothing beyond index 17520 was read, "
        "A5-vs-A3 was not re-estimated and no new method was introduced."
        % (COHORT_NAME, VARIANT, ARM, SLOW_MODEL),
        "",
    ]
    gate = payload.get("p2_non_regression_gate") or {}
    if gate.get("cells"):
        lines.extend([
            "## P2 -- the non-regression gate",
            "",
            "%s. %d of %d #19 task_C episodes reproduce digit-for-digit "
            "through the promoted enforcement path; %d retrains, 0 LLM."
            % (
                "PASS" if gate.get("ok") else "FAIL",
                sum(1 for c in gate["cells"].values() if c["reproduces"]),
                len(gate["cells"]), int(gate.get("consumer_retrains") or 0),
            ),
            "",
        ])
    elif gate:
        pre = gate.get("precondition") or {}
        lines.extend([
            "## P2 -- not re-run, precondition verified",
            "",
            "- Carried forward: %s." % gate.get("gate_result_carried_forward"),
            "- Measurement-side files byte-identical to their post-P1 state: "
            "%s." % pre.get("byte_identical"),
            "- %s." % gate.get("why_it_cannot_be_re_run_anyway"),
            "",
        ])
    store = payload.get("store") or {}
    if store:
        card = store["card"]
        lines.extend([
            "## The store this run built for itself",
            "",
            "- Card `%s`: %s, %d clauses %s, bytes sha256 `%s`." % (
                CARD_ID, card["status"], card["clause_count"],
                card["clause_ids"], card["card_bytes_sha256"][:16],
            ),
            "- Snapshot `%s`, skills %s, `scope_risk_guards` starts empty." % (
                store["store"]["runtime_bundle_sha"][:16],
                store["store"]["skill_ids"],
            ),
            "",
        ])
    rows = payload.get("trajectory") or []
    if rows:
        lines.extend([
            "## Trajectory",
            "",
            "| step | window | mode | card | local Skill | plan before gate | "
            "plan after gate | support | delayed | harmed | gate | retrains | LLM |",
            "| --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | "
            "--- | ---: | ---: |",
        ])
        for row in rows:
            lines.append(
                "| `%s` | %s | %s | %s | %s | `%s` | `%s` | %s | %s | %s | %s "
                "| %d | %d |" % (
                    row["step"], row["window"]["start"], row["mode"],
                    "hit" if row["card_hit"] else "MISS",
                    "--" if row["local_skill_hit"] is None else (
                        "hit" if row["local_skill_hit"] else "MISS"
                    ),
                    SSU._plan_label(row["plan_before_gate"]),
                    SSU._plan_label(row["plan_after_gate"]),
                    _fmt(row["support_before_gate"]),
                    _fmt(row["delayed_before_gate"]),
                    ", ".join(row["harmed_before_gate"]) or "none",
                    "moved" if row["gate_changed_the_decision"] else (
                        "checked" if row["gate"].get("checked") else "inactive"
                    ),
                    int(row["consumer_retrains"]), int(row["llm_calls"]),
                )
            )
        uids = sorted(rows[0]["per_eval_series_delayed_before_gate"])
        lines.extend([
            "",
            "### Per evaluation series, delayed gain",
            "",
            "| step | phase | " + " | ".join("`%s`" % u for u in uids) + " |",
            "| --- | --- | " + " | ".join("---:" for _ in uids) + " |",
        ])
        for row in rows:
            for phase, key in (
                ("before gate", "per_eval_series_delayed_before_gate"),
                ("after gate", "per_eval_series_delayed_after_gate"),
            ):
                values = row[key]
                lines.append(
                    "| `%s` | %s | %s |" % (
                        row["step"], phase,
                        " | ".join(_fmt(values.get(u)) for u in uids),
                    )
                )
        lines.append("")
    life = payload.get("lifecycle") or {}
    if life:
        promo = life.get("promotion") or {}
        probe = life.get("probe") or {}
        lines.extend([
            "## Lifecycle",
            "",
            "- Draft written: %s (`%s`)." % (
                bool((life.get("draft") or {}).get("written")),
                (life.get("draft") or {}).get("skill_id"),
            ),
            "- Probe %s: gain %s, se %s, gain/se %s -- out of selection." % (
                life.get("probe_window", {}).get("window_id"),
                _fmt(probe.get("macro_gain")), _fmt(probe.get("se_block")),
                _fmt(probe.get("gain_over_se")),
            ),
            "- Promotion: %s -> `%s`." % (
                bool(promo.get("promoted")), promo.get("retrievable_skill_id"),
            ),
            "",
        ])
    att = payload.get("attribution") or {}
    if att:
        lines.extend([
            "## Attribution -- on this run's own record",
            "",
            "- With the per-series risk reading: `%s` at %s." % (
                att.get("cause_code"), att.get("first_stage"),
            ),
            "- Through the aggregate alone: `%s`." % (
                (att.get("aggregate_only_control") or {}).get(
                    "attribution", {}
                ).get("cause_code"),
            ),
            "",
        ])
    slow = payload.get("slow_edit") or {}
    if slow.get("guard"):
        g = slow["guard"]
        lines.extend([
            "## The Slow edit",
            "",
            "- Backend `%s`, one round, %d draw(s), no cross-backend sampling."
            % (SLOW_MODEL, len(slow.get("attempts") or ())),
            "- Surface `%s`; guard `%s`: %s `%s` %s -> %s on the %s window, "
            "applies to %s." % (
                slow["target_surface_id"], g["guard_id"], g["statistic"],
                g["comparator"], _fmt(g["threshold"]), g["action"],
                g["window"], g["applies_to"],
            ),
            "- Rationale: %s" % g.get("rationale"),
            "- Snapshot `%s` -> `%s`, in the same store." % (
                slow["snapshot_before"][:16], slow["snapshot_after"][:16],
            ),
            "",
        ])
    shadow = payload.get("shadow_replay")
    if shadow:
        lines.extend([
            "## Shadow replay (criterion ii)",
            "",
            "- Zero evaluation: %s (%d retrains spent)." % (
                shadow["zero_evaluation"], shadow["consumer_retrains_spent"],
            ),
            "- task_C decision `%s` -> `%s`; changed: %s." % (
                SSU._plan_label(shadow["decision_before_patch"]),
                SSU._plan_label(shadow["decision_after_patch"]),
                shadow["decision_changed"],
            ),
            "",
        ])
    after = payload.get("frozen_surface_after") or {}
    lines.extend([
        "## Cost and integrity",
        "",
        "- LLM calls: %s / %s." % (
            payload.get("llm_call_count"), payload.get("llm_call_budget")
        ),
        "- Consumer retrains: %s / %s." % (
            payload.get("consumer_retrains_total"), payload.get("retrain_budget")
        ),
        "- Frozen surface: %s files, drift %s." % (
            after.get("files"), after.get("drift")
        ),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _write(
    payload: Mapping[str, Any], *, dry: bool = False,
    out_json: Path | None = None, out_md: Path | None = None,
) -> int:
    body = _public(payload)
    if dry:
        print(json.dumps(body, indent=2, ensure_ascii=False, default=str)[:4000])
    else:
        target_json = out_json or OUT_JSON
        target_md = out_md or OUT_MD
        E2.mkdir(parents=True, exist_ok=True)
        target_json.write_text(
            json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
        target_md.write_text(_markdown(body), encoding="utf-8", newline="\n")
        print("wrote", target_json, flush=True)
        print("wrote", target_md, flush=True)
    print("overall", body.get("overall_verdict"), flush=True)
    print("llm_calls", body.get("llm_call_count", 0), flush=True)
    print("retrains", body.get("consumer_retrains_total", 0), flush=True)
    return 0 if body.get("overall_verdict") in (
        "OPERATIONAL_PIPELINE_CLOSES", "OPERATIONAL_PIPELINE_CLOSES_IN_K",
        "BANKED_CHAIN_CLOSES", "BANKED_CHAIN_CLOSES_IN_K_MODELS",
        "DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX_ON_GPT_5_6_SOL",
        "DEVELOPMENT_OPERATIONAL_PIPELINE_CLOSES_POST_FIX", "P2_ONLY",
    ) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gate-only", action="store_true",
        help="verify the P2 precondition and stop, writing nothing",
    )
    parser.add_argument(
        "--multi-trajectory", action="store_true",
        help="run K pre-registered trajectories and write the v3 artifact",
    )
    parser.add_argument(
        "--k", type=int, default=K_TRAJECTORIES,
        help="how many trajectories; fixed before the run",
    )
    parser.add_argument(
        "--banked-completion", action="store_true",
        help="finish links 6-9 on the #23 banked ledger and write v4",
    )
    parser.add_argument(
        "--post-fix-live", action="store_true",
        help="banked completion, then one post-fix live trajectory if it closed",
    )
    parser.add_argument(
        "--slow-model", type=str, default="gpt-5.6-sol",
        help="the pinned Slow backend for the post-fix live trajectory",
    )
    parser.add_argument(
        "--slow-models", type=str, default="",
        help="comma-separated pinned Slow backends; runs B2 once per model",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="allow the single-trajectory path to overwrite the v2 record",
    )
    args = parser.parse_args(argv)
    if args.slow_models:
        return run_v6([m.strip() for m in args.slow_models.split(",") if m.strip()])
    if args.post_fix_live:
        return run_v7(slow_model=args.slow_model)
    if args.banked_completion:
        return run_v4(banked=True, post_fix_live=False)
    if args.multi_trajectory:
        return run_multi(k=int(args.k))
    return run(gate_only=bool(args.gate_only), force=bool(args.force))


if __name__ == "__main__":
    raise SystemExit(main())
