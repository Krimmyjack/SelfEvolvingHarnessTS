"""Slow Scope/Risk self-update on the fresh-confirmation Scope gap.

The closing slice (#17) adopted `outlier_mad` on the pooled task_C window in
both arms.  The aggregate delayed gain was +0.029688, but one of the four
evaluation series -- 99999904140 -- lost 0.125557, four times the aggregate
it was averaged into.  The external adjudication re-ruled the first fault to
TARGET_LOCAL_SCOPE_RISK_GAP.

This slice runs the last link of the self-evolution loop on that failure:
Runtime first-fault attribution over the recorded episode, one Slow proposal
on one authorized Scope/Risk surface, deterministic compiler validation, and
a development-level replay of the same already-exposed window.

Nothing under ``methods/ttha``, the recipe compiler, the adoption ladder, the
program menu, the prompt templates, the Consumers or any delivered artifact is
modified.  The attribution machinery (``FaultRouter``, ``SurfaceRegistry``,
``EditController``, ``assess_case``) is the old line's, used as it stands.
The replay is explicitly NOT fresh: the pooled task_C window was opened by
#17 and is reused here as development evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
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
import run_e2_skill_store_integration as ssi  # noqa: E402
import run_e2_warm_vs_cold_recipe_search as wvc  # noqa: E402

from SelfEvolvingHarnessTS.contracts.canonical import (  # noqa: E402
    canonical_sha256,
)
from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.config import (  # noqa: E402
    load_m0_rules,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.first_fault import (  # noqa: E402
    CaseFacts,
    assess_case,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.feedback.router import (  # noqa: E402
    FaultRouter,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentRole,
    TTHAAgentCore,
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
from SelfEvolvingHarnessTS.methods.ttha.schema_contracts import (  # noqa: E402
    load_stage_schema,
)
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import (  # noqa: E402
    _resolve_apply_manifest,
)
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentTransportError,
)

PROTOCOL_VERSION = "slow_scope_update_v1"
E2 = PROJECT_ROOT / "artifacts" / "functional" / "e2"
OUT_JSON = E2 / "slow_scope_update_v1.json"
OUT_MD = E2 / "slow_scope_update_v1.md"
SOURCE_ARTIFACT = E2 / "fresh_confirmation_v1.json"
SOURCE_STORE = PROJECT_ROOT / "_scratch" / "skill_store" / "fresh_confirmation_v1"
WORK_ROOT = PROJECT_ROOT / "_scratch" / "slow_scope_v1"
STORE_ROOT = WORK_ROOT / "stores"

CONSUMERS = tuple(FC.CONSUMERS)
ARMS = tuple(FC.ARMS)
POOLED = bch.CONSUMER_POOLED
PER_CHANNEL = bch.CONSUMER_PER_CHANNEL
IDENTITY = FC.IDENTITY
MATERIAL_THRESHOLD = float(FC.MATERIAL_THRESHOLD)
HARM_THRESHOLD = float(FC.HARM_THRESHOLD)
THE_HARMED_SERIES = "99999904140"

LLM_CALL_BUDGET_TOTAL = 15
RETRAIN_BUDGET = 200
GUARD_KEY = "scope_risk_guards"

# ------------------------------------------------------- #19 v2 increment
# The #18 slice ended in SLOW_ABSTAINS (re-ruled SLOW_DECLINES_PATCH): the
# deterministic fold landed on RISK_GAP at OUTCOME_RISK, but the guard
# evaluation context only saw the aggregate + harm ledger because
# ``wvc.BudgetedSearch._gains`` projected the measured per-series vector
# away.  The mainline ruled the first fault OBSERVATION_PROJECTION_GAP and
# authorized exactly one coupled Observation+Scope repair (AGENTS.md: one
# failure mechanism, one frozen patch):
#   O1  the instrument interface stops projecting (the tracked one-line
#       passthrough in run_e2_warm_vs_cold_recipe_search.py::_gains);
#   O2  the Slow input is widened to every exposed harm observation, quoted
#       read-only from fresh_confirmation_v1 (zero new measurement);
# plus a 0-LLM non-regression gate before any LLM call.  Program, Memory,
# Judge, the program menu, the v2 ladder, the promotion line and the prompt
# templates are not touched.
PROTOCOL_VERSION_V2 = "slow_scope_update_v2"
OUT_JSON_V2 = E2 / "slow_scope_update_v2.json"
OUT_MD_V2 = E2 / "slow_scope_update_v2.md"
WORK_ROOT_V2 = PROJECT_ROOT / "_scratch" / "slow_scope_v2"
STORE_ROOT_V2 = WORK_ROOT_V2 / "stores"
LLM_CALL_BUDGET_V2 = 10
O1_TARGET_FILE = "evaluation/functional/run_e2_warm_vs_cold_recipe_search.py"
O1_SHA256_BEFORE_FIX = (
    "bab5feb8e6adc6e66d8983ea481f237b4a553d882f2498c2f14f6aa6078defd3"
)

# Backend assignment (mainline, folded into #19): the main backend is KIMI --
# both recorded #18 abstentions were drawn on the KIMI-served endpoint, so the
# post-repair retry runs on the same backend and the only changed variable is
# the O1/O2 repair.  If KIMI still abstains twice, one pre-registered Opus
# follow-up round runs with the same prompt and the same sampling discipline.
SLOW_BACKEND_ROUNDS_V2 = (
    {
        "label": "KIMI",
        "model": ssi.NF_MODEL,
        "base_url": ssi.NF_BASE_URL,
        "why": (
            "the #18 draws were served here; retrying on the same backend "
            "makes the repair the only changed variable"
        ),
    },
    {
        "label": "OPUS",
        "model": "claude-opus-5",
        "base_url": ssi.NF_BASE_URL,
        "why": (
            "pre-registered follow-up, run only if KIMI abstains twice: a "
            "cross-backend decline is the stronger negative, a proposal is "
            "reported as backend-dependent"
        ),
    },
)
# Config-level backfill for the three #18 draws: that runner did not record
# the response-level serving model (same defect class as the lost reason
# code; repaired below for #19, which records returned_models per attempt).
BACKEND_IDENTITY_BACKFILL_18 = {
    "draws": [
        {"run": "live attempt 1", "outcome": "NO_PROPOSAL",
         "no_proposal_reason": None,
         "reason_note": "lost: the #18 runner did not record it"},
        {"run": "live run 2 attempt 1", "outcome": "NO_PROPOSAL",
         "no_proposal_reason": "no_authorized_minimal_edit"},
        {"run": "live run 2 attempt 2", "outcome": "NO_PROPOSAL",
         "no_proposal_reason": "insufficient_public_evidence"},
    ],
    "configured_model": ssi.NF_MODEL,
    "configured_base_url": ssi.NF_BASE_URL,
    "serving_model_capture": (
        "config-level backfill only; the response-level returned_model was "
        "not recorded in #18.  Mainline designates this endpoint the KIMI "
        "backend."
    ),
}

# The sutured tracked files, registered here with their pre-fix sha256 so
# the v2 artifact carries both sides of every deliberate move.  The post-fix
# side is measured at freeze time.  harness_surfaces.json is CRLF on disk;
# the value below is the working-tree byte hash (its LF-normalized HEAD blob
# is 249e5ae9fab40329ed66eeae6e828cdfb3f2df26130e24658873ab04e7d19fe0).
COMPILER_FILE = "methods/ttha/harness/compiler.py"
VERIFICATION_FILE = "methods/ttha/harness/h0/verification.json"
H0_LOCK_FILE = "methods/ttha/harness/h0/snapshot.lock.json"
SURFACES_FILE = "methods/ttha/harness/harness_surfaces.json"
SUTURED_FILES_V2: tuple[dict[str, str], ...] = (
    {
        "path": O1_TARGET_FILE,
        "role": "O1: the instrument interface stops projecting the measured per-series vector",
        "sha256_before_fix": O1_SHA256_BEFORE_FIX,
    },
    {
        "path": VERIFICATION_FILE,
        "role": "pre-registers scope_risk_guards as [] in the h0 authoring",
        "sha256_before_fix": (
            "5b3dc62e72d202811a9dd2d553aeadf89b4390d71ffeb0202b37f154df84a875"
        ),
    },
    {
        "path": SURFACES_FILE,
        "role": "registers verification.rules.scope_risk_guards (one PATCH surface, pointer /scope_risk_guards)",
        "sha256_before_fix": (
            "e3289f776f91b6b25ad559bb3a45a053c5bfbed691b2396a37e6a826215755d7"
        ),
    },
    {
        "path": H0_LOCK_FILE,
        "role": "regenerated snapshot lock (the suture moves the content and the compiler identity the lock pins)",
        "sha256_before_fix": (
            "1e54a67ea0215bcc791f21a8f6f8726edbb06ac81eb8608dc45bd7ed48b65aa2"
        ),
    },
    {
        "path": COMPILER_FILE,
        "role": "the adoption-gate evaluator is promoted into tracked machinery; newly registered in v2",
        "sha256_before_fix": (
            "c04a08fe0705bec7e90c91ded748a4a41d8935c35da5a23c04791783e0092279"
        ),
    },
)
MAX_PROPOSAL_ATTEMPTS_V2 = 2
MAX_CONSECUTIVE_TRANSPORT_FAILURES = 3
MIGRATION_EDIT_ID = "ssu2_preregister_scope_risk_guards"
NEW_SURFACE_V2 = "verification.rules.scope_risk_guards"

M0_RULES = load_m0_rules(
    PROJECT_ROOT / "evaluation" / "minipipe" / "config" / "m0_rules.json"
)
RISK_EPSILON = float(M0_RULES["risk_epsilon"])


def _repo_rel(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# The Scope/Risk surface is the only thing this slice may move.  Everything
# below is checked byte-for-byte before and after, item by item.
FROZEN_SURFACE: tuple[str, ...] = tuple(FC.FROZEN_SURFACE) + (
    "artifacts/functional/e2/fresh_confirmation_v1.json",
    "artifacts/functional/e2/fresh_confirmation_v1.md",
    "artifacts/functional/e2/fresh_confirmation_v1_adjudication.md",
    "artifacts/functional/e2/local_skill_recall_v1.json",
    "artifacts/functional/e2/noaa_fresh_cohort_v2.json",
    "evaluation/functional/run_e2_fresh_confirmation.py",
    "evaluation/functional/run_e2_local_skill_recall.py",
    "evaluation/minipipe/feedback/first_fault.py",
    "evaluation/minipipe/feedback/fault_routes.json",
    "evaluation/minipipe/feedback/router.py",
    "evaluation/minipipe/replay/edit_controller.py",
    "evaluation/minipipe/config/m0_rules.json",
    "methods/ttha/harness/harness_surfaces.json",
    "methods/ttha/harness/h0/verification.json",
)

# The v2 registry: the v1 surface in its post-suture form (four files
# re-registered with both sha256 sides, see SUTURED_FILES_V2) plus the
# newly touched compiler.py.
FROZEN_SURFACE_V2: tuple[str, ...] = tuple(FROZEN_SURFACE) + (COMPILER_FILE,)


class ConcurrentWrite(RuntimeError):
    """The frozen surface moved while this protocol was running."""


def _freeze(surface: Sequence[str] = FROZEN_SURFACE) -> dict[str, str]:
    frozen: dict[str, str] = {}
    for name in sorted(set(surface)):
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise SystemExit("frozen surface member is missing: %s" % name)
        frozen[name] = _sha256(path)
    return frozen


def _verify(before: Mapping[str, str]) -> dict[str, Any]:
    drift = [
        name for name, sha in before.items()
        if not (PROJECT_ROOT / name).is_file() or _sha256(PROJECT_ROOT / name) != sha
    ]
    return {"files": len(before), "drift": drift, "ok": not drift}


def _guard(before: Mapping[str, str], where: str) -> bool:
    report = _verify(before)
    if not report["ok"]:
        raise ConcurrentWrite(
            "the frozen surface moved before %s: %s" % (where, report["drift"])
        )
    return True


# ------------------------------------------------ what was fixed before B2 ran
# Only what the frozen search instrument already exposes.  ``BudgetedSearch.
# _gains`` projects its per-series vector away and returns the aggregate plus
# the harm ledger, so a guard over anything else would need an edit to the
# frozen instrument.  The grammar is bounded by that, not by preference.
GUARD_STATISTICS = (
    "aggregate_gain",
    "harmed_series_count",
    "harmed_series_fraction",
    "total_harm",
    "gain_to_total_harm_ratio",
)
GUARD_COMPARATORS = ("lt", "le", "gt", "ge")
GUARD_WINDOWS = ("support", "delayed")
GUARD_ACTIONS = ("VETO_AND_FALL_BACK", "RECORD_ONLY")
GUARD_APPLIES_TO = ("every_adoption", "reused_skill_adoption_only")

GUARD_GRAMMAR: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "one Scope/Risk guard the runtime can enforce deterministically",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "guard_id", "window", "statistic", "comparator", "threshold",
        "action", "applies_to", "rationale",
    ],
    "properties": {
        "guard_id": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$",
        },
        "window": {"type": "string", "enum": list(GUARD_WINDOWS)},
        "statistic": {"type": "string", "enum": list(GUARD_STATISTICS)},
        "comparator": {"type": "string", "enum": list(GUARD_COMPARATORS)},
        "threshold": {"type": "number"},
        "action": {"type": "string", "enum": list(GUARD_ACTIONS)},
        "applies_to": {"type": "string", "enum": list(GUARD_APPLIES_TO)},
        "rationale": {"type": "string", "minLength": 1, "maxLength": 600},
    },
}

GUARD_CONTRACT: dict[str, Any] = {
    "where_it_lives": (
        "a list under the key %r, either on the verification document "
        "(surface `verification.rules`) or inside one Skill's risk_guards "
        "object (surface `skill_library.entries/<skill_id>.risk_guards`)"
        % GUARD_KEY
    ),
    "who_reads_it": (
        "the runtime reads the list off the active snapshot after the frozen "
        "v2 adoption ladder has produced its final plan, and before that plan "
        "is recorded as adopted.  The ladder itself is not modified and is "
        "not re-ranked."
    ),
    "statistics": {
        "aggregate_gain": "the plan's aggregate gain on the named window",
        "harmed_series_count": (
            "how many evaluation series lost more than %.3f" % MATERIAL_THRESHOLD
        ),
        "harmed_series_fraction": "that count divided by the evaluation roster size",
        "total_harm": "the summed magnitude of those losses, as a positive number",
        "gain_to_total_harm_ratio": (
            "aggregate_gain divided by total_harm; a very large number when "
            "nothing lost anything"
        ),
    },
    "why_only_these": (
        "these five are exactly what the frozen search instrument returns for "
        "a plan on a window.  A guard over anything else would require an edit "
        "to that instrument, which is frozen here."
    ),
    "firing": "the guard fires when `statistic <comparator> threshold` is true",
    "actions": {
        "VETO_AND_FALL_BACK": (
            "the plan is rejected.  The runtime then walks the frozen v2 "
            "fallback: the Support winner's full-batch plan when one exists, "
            "its full-batch delayed reading is positive and it is not the "
            "plan just rejected, otherwise identity.  The fallback candidate "
            "is re-checked against the same guard and falls to identity if it "
            "fires too."
        ),
        "RECORD_ONLY": "the reading is written to the receipt and nothing changes",
    },
    "identity_is_never_vetoed": (
        "identity is unfilterable, so a guard can never remove the abstention "
        "option"
    ),
    "applies_to": {
        "every_adoption": "checked on every episode that adopts a program",
        "reused_skill_adoption_only": (
            "checked only when the plan came from a recalled Target-local "
            "Skill rather than from a fresh search"
        ),
    },
    "cost": (
        "each fallback candidate the guard forces the runtime to look at "
        "costs real Consumer retrains, charged to this run's budget"
    ),
}


# ----------------------------------------- #19 v2: the un-projected grammar
# One new statistic, readable now that the instrument interface passes the
# measured per-series vector through (O1).  Nothing already expressible
# changes meaning.
GUARD_STATISTICS_V2 = GUARD_STATISTICS + ("min_per_series_gain",)

GUARD_GRAMMAR_V2: dict[str, Any] = json.loads(json.dumps(GUARD_GRAMMAR))
GUARD_GRAMMAR_V2["properties"]["statistic"]["enum"] = list(GUARD_STATISTICS_V2)

GUARD_CONTRACT_V2: dict[str, Any] = dict(GUARD_CONTRACT)
GUARD_CONTRACT_V2["statistics"] = dict(
    GUARD_CONTRACT["statistics"],
    min_per_series_gain=(
        "the smallest per-evaluation-series gain of the plan on the named "
        "window -- the vector the search instrument measures and, since the "
        "O1 repair, returns.  It is the reading the aggregate hid in #17: a "
        "plan can gain on average while one series loses far past the harm "
        "line"
    ),
)
GUARD_CONTRACT_V2["why_only_these"] = (
    "the five #18 statistics are the aggregate projection; the sixth is the "
    "measured per-series vector the projection dropped.  The O1 repair "
    "stopped the projection at the instrument interface, so the grammar now "
    "sees exactly what was measured -- nothing else was added."
)
GUARD_CONTRACT_V2["where_it_lives"] = (
    "the pre-registered list at /scope_risk_guards in the verification "
    "document -- surface `verification.rules.scope_risk_guards`, whose "
    "json_pointer and semantics are carried by the Surface Registry.  Every "
    "store's active snapshot already holds the empty list; the minimal legal "
    "edit is exactly one entry appended to it (patch_value = the complete "
    "new one-entry list)."
)
GUARD_CONTRACT_V2["who_reads_it"] = (
    "the adoption-gate evaluator in "
    "methods/ttha/harness/compiler.py -- tracked runtime machinery, the "
    "verification document's own module -- reads the list off the active "
    "snapshot after the frozen v2 adoption ladder has produced its final "
    "plan, and before that plan is recorded as adopted.  The ladder itself "
    "is not modified and is not re-ranked.  (In #18 the key had no runtime "
    "reader at all; that placebo defect is what this suture repairs.)"
)

ADAPTER_DECLARATION: dict[str, Any] = {
    "why": (
        "``assess_case`` takes a ``CaseFacts`` built for the controlled "
        "minipipe: one synthetic case with a private clean reference, an "
        "oracle-affected index set and probe curves.  A recipe-line episode "
        "has none of those.  The adapter below moves measured numbers into "
        "the slots that have a faithful analogue and leaves every other slot "
        "at its documented default, naming each one."
    ),
    "two_axes": (
        "selection-time utility is the Support window, because Support is "
        "what the Agent can see when it adopts; outcome utility is the "
        "delayed window, because that is what the ladder confirms on.  So "
        "``candidate_utilities``/``clean_u``/``corrupt_u``/``prepared_u``/"
        "``damage_d`` carry Support gains and ``chosen_gain`` carries the "
        "delayed aggregate gain."
    ),
    "transported": {
        "case_id": "the episode id",
        "is_target": "True: the batch is what the adoption was aimed at",
        "chosen_candidate_id": "the adopted plan's label",
        "candidate_utilities": (
            "identity at 0.0 plus every plan this episode actually measured, "
            "at its Support aggregate gain -- including the masked plan when "
            "a mask round ran, because the Agent was shown it"
        ),
        "effect_distinct_candidate_ids": "every measured non-identity plan",
        "clean_u": "the largest measured Support gain",
        "corrupt_u": "0.0, the identity baseline every gain is measured against",
        "prepared_u": "the adopted plan's Support gain",
        "damage_d": "clean_u - corrupt_u: the headroom the search actually found",
        "chosen_gain": "the adopted plan's delayed aggregate gain",
        "risk_delta_u": (
            "the smallest per-evaluation-series delayed gain.  This is the "
            "only reduction the adapter performs: the slot is a scalar and "
            "the episode measured a vector.  The whole vector is recorded "
            "beside it, and the run is repeated with this slot set to None "
            "so the aggregate-only reading is on the record too."
        ),
        "capability_skill_exists / skill_retrieved / normal_retrieval / "
        "retrieved_capability_skill_ids": "from the episode's retrieval receipt",
        "compilation_ok / execution_ok / execution_contract_ok": (
            "True: every plan in this line compiles and runs, or the episode "
            "would have raised"
        ),
        "proposed_candidate_exists / compiled_candidate_exists": (
            "whether a non-identity plan was measured"
        ),
    },
    "not_transported_left_at_default": {
        "scope_stable": (
            "True.  ``cycle.py`` fills this from modified indices over context "
            "size against ``max_modified_fraction``; a recipe episode has no "
            "index-level modification trace, and a batch program legitimately "
            "touches every training series, so mapping series fraction into "
            "an index-fraction slot would be wrong.  Left True, which makes "
            "the RISK stage harder to trip, not easier."
        ),
        "over_restoration": "False: there is no clean oracle to over-restore towards",
        "chosen_probe_directions": "(): the recipe line runs no probe directions",
        "public_evidence_discriminative": "True: the public per-series table was built",
        "agent_inspected_evidence": (
            "True.  A direct-recall episode has no Agent turn by protocol "
            "design, which is an absence rather than a procedure gap; the "
            "field has no NOT_APPLICABLE value, so the passing default is "
            "used and declared here."
        ),
        "localization_required / localization_iou / mechanism_identified / "
        "mechanism_contradiction / period_diagnostic_pass": (
            "minipipe diagnostic machinery with no recipe-line analogue"
        ),
        "expressibility_status / witnesses / forced_skill_succeeds / "
        "constrained_proposal_succeeds": (
            "program-supply witness machinery with no recipe-line analogue"
        ),
    },
    "consequence_to_state_plainly": (
        "because the untransportable fields keep passing defaults, the fold "
        "can only report a fault at CANDIDATE_SELECTION or OUTCOME_RISK, or "
        "report none at all.  The adapter narrows where attribution can land "
        "and that narrowing is the adapter's, not the machine's."
    ),
}

PRE_REGISTERED: dict[str, Any] = {
    "background": (
        "pooled task_C in fresh_confirmation_v1: both arms adopted "
        "`outlier_mad` full batch, aggregate delayed +0.029688, evaluation "
        "series %s down 0.125557 -- 4.2x the aggregate it was averaged into. "
        "Re-ruled first fault: TARGET_LOCAL_SCOPE_RISK_GAP." % THE_HARMED_SERIES
    ),
    "b1_attribution": (
        "the Runtime first-fault fold runs on the recorded pooled task_C "
        "episode of each arm, through the old line's own ``assess_case`` and "
        "``FaultRouter``.  0 LLM, 0 retrains: the evidence is the artifact."
    ),
    "b2_patch": (
        "one Slow proposal, on one surface the router authorizes for the "
        "attributed cause, validated by the deterministic EditController.  "
        "The LLM never approves its own edit and gets no second attempt: a "
        "rejection stops the run at COMPILER_REJECTS."
    ),
    "b3_replay": (
        "development level, explicitly not fresh: the pooled task_C window "
        "was opened by #17.  Both arms' decisions are replayed with the patch "
        "active, retraining on the same window where the patch makes the "
        "runtime look at a plan it had not measured.  per_channel task_C is "
        "replayed as a regression check and is expected not to move."
    ),
    "b4_cost": (
        "every Consumer retrain and LLM call is charged; new Experience rows "
        "carry provenance=slow_scope_update"
    ),
    "verdicts": {
        "SLOW_CLOSES_SCOPE_GAP": (
            "the attribution lands on a Scope/Risk surface, the single-surface "
            "patch compiles, %s no longer crosses the %.3f line in the replay "
            "(masked, rescoped or vetoed alike), the aggregate delayed reading "
            "is still >= 0, and per_channel does not move"
            % (THE_HARMED_SERIES, HARM_THRESHOLD)
        ),
        "ATTRIBUTION_LANDS_ELSEWHERE": (
            "the fold names another surface; recorded as it stands, not corrected"
        ),
        "PATCH_OVERREACH": "more than one surface, or a frozen surface, was touched",
        "COMPILER_REJECTS": "the EditController refused the proposal",
        "REPLAY_NO_CHANGE": "the patch is live and no decision moved",
        "REPLAY_AGGREGATE_COLLAPSE": (
            "one series was saved at the cost of the aggregate"
        ),
        "SCHEMA_BLOCKED": (
            "the recipe episode does not fit the attribution input and no "
            "pure-data-transport adapter reaches it"
        ),
    },
    "guard_grammar": GUARD_GRAMMAR,
    "guard_contract": GUARD_CONTRACT,
    "adapter": ADAPTER_DECLARATION,
    "discipline": {
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "retrain_budget": RETRAIN_BUDGET,
        "reads": (
            "only the already-exposed windows of fresh_confirmation_v1: the "
            "2024 development training anchors and the task_C confirmation "
            "window [9864, 10152].  Nothing beyond index 17520, and no window "
            "#17 did not open."
        ),
        "commit": False,
        "spawn": False,
        "new_files": [
            "evaluation/functional/run_e2_slow_scope_update.py",
            "artifacts/functional/e2/slow_scope_update_v1.json",
            "artifacts/functional/e2/slow_scope_update_v1.md",
            "_scratch/slow_scope_v1/",
        ],
    },
}


PRE_REGISTERED_V2: dict[str, Any] = {
    "background": (
        "#18 (slow_scope_update_v1) ended SLOW_ABSTAINS, re-ruled "
        "SLOW_DECLINES_PATCH, and the mainline refused to close the family.  "
        "The deepened diagnosis is a composite EDIT_SURFACE_DEFECT with "
        "three coupled components, each alone sufficient to make the Slow "
        "draws come out empty: (1) OBSERVATION_PROJECTION_GAP -- the guard "
        "evaluation context saw only the aggregate and the harm ledger "
        "because wvc.BudgetedSearch._gains projected the measured per-series "
        "vector away; (2) a placebo surface -- the %r key had no tracked "
        "runtime reader at all, so a guard written there would never "
        "execute; (3) a non-minimal authorization set -- the surfaces "
        "offered were the whole verification document and per-Skill "
        "risk_guards objects, never the one list entry a minimal edit "
        "needs.  Discriminating evidence: byte-identical prompts returned "
        "two different reason codes (no_authorized_minimal_edit and "
        "insufficient_public_evidence) and risk_too_high was never used -- "
        "a structural-defect signature, not an evidence judgement.  The "
        "three #18 abstentions were correct behaviour on a defective "
        "surface set and are not counted against Slow." % GUARD_KEY
    ),
    "the_repair": (
        "one coupled Observation+Scope repair (AGENTS.md pre-authorized: one "
        "failure mechanism -- the aggregate hiding one series' loss behind a "
        "surface Slow could not minimally edit -- one frozen patch).  "
        "Program, Memory, Judge, the program menu, the v2 ladder, the "
        "promotion line and the prompt templates are not touched."
    ),
    "o1_observation": (
        "the guard evaluation context interface stops projecting: "
        "run_e2_warm_vs_cold_recipe_search.py::BudgetedSearch._gains returns "
        "the already-measured per-series delayed delta vector verbatim (one "
        "passthrough key; measurement semantics unchanged)."
    ),
    "suture_write_side": (
        "methods/ttha/harness/h0/verification.json pre-registers %r as []; "
        "methods/ttha/harness/harness_surfaces.json registers the surface "
        "verification.rules.scope_risk_guards (owner verification.json, "
        "json_pointer /%s, PATCH, semantics entry carrying the entry schema "
        "and the single-entry constraint); the h0 snapshot lock is "
        "regenerated so verify_lock callers stay consistent.  Every "
        "re-registered file carries its pre/post sha256 in this artifact."
        % (GUARD_KEY, GUARD_KEY)
    ),
    "suture_execution_side": (
        "the adoption-gate evaluator is promoted from this runner's #18 "
        "reference implementation into methods/ttha/harness/compiler.py -- "
        "tracked machinery, the verification document's own module: "
        "scope_risk_guards_of / guard_statistic / guard_fires / "
        "evaluate_scope_risk_guards, plus compile-time validation of the "
        "list (at most one entry, closed vocabularies, finite threshold).  "
        "The compiler reads only the verification document's list; a "
        "skill-side key is inert and not a registered surface."
    ),
    "o2_evidence": (
        "the Slow input is widened to every exposed harm observation -- the "
        "pooled task_C episodes of both arms (series %s at -0.125557 twice) "
        "and every probe-window harm record -- quoted read-only from "
        "fresh_confirmation_v1.  Zero new measurement." % THE_HARMED_SERIES
    ),
    "migration": (
        "the empty list is installed into the four store forks' active "
        "snapshots through the whole-document verification.rules surface "
        "(the pointer surface requires the key to pre-exist), one "
        "EditController receipt per store, each with an exactly-one-key "
        "diff proof.  0 LLM, deterministic."
    ),
    "non_regression_gate": (
        "before any LLM call, on the migrated (empty-list) state: the "
        "guard-free replay of the four recorded task_C episodes must "
        "reproduce them digit-for-digit on the repaired instrument; "
        "otherwise the run stops INSTRUMENT_DRIFT and no repair is "
        "attempted in flight"
    ),
    "re_freeze": (
        "four re-registered files carry pre/post sha256 (the O1 target, "
        "h0/verification.json, harness_surfaces.json, the regenerated h0 "
        "lock); methods/ttha/harness/compiler.py is newly registered in its "
        "post-suture form; every other v1 entry is checked for zero drift."
    ),
    "slow_retry": (
        "only the new surface is offered -- no further sampling on the #18 "
        "surface set.  The evidence is the five exposed harm observations, "
        "verbatim.  Same abstention discipline: at most 2 byte-identical "
        "samples per backend, stop at the first proposal; every attempt "
        "records its reason and its response-level serving-model identity.  "
        "Backend sequence pinned inside the runner: KIMI (%s @ agicto) "
        "first -- the #18 draws were served there, so the repair is the "
        "only changed variable; if KIMI abstains twice, one pre-registered "
        "OPUS follow-up round (claude-opus-5, same prompt).  A proposal "
        "from the OPUS round is reported backend-dependent.  A transport "
        "failure consumes no sample; three consecutive transport failures "
        "stop the run INCONCLUSIVE_TRANSPORT.  A compiler rejection still "
        "stops the run at COMPILER_REJECTS with no second attempt."
        % ssi.NF_MODEL
    ),
    "b3_replay": (
        "development level, explicitly not fresh: both pooled task_C arms "
        "replayed with the patch live, per_channel task_C as the regression "
        "check.  The unguarded half of the replay is the gate's, paid once; "
        "only plans the guard moves cost fresh retrains."
    ),
    "b4_cost": (
        "every Consumer retrain and LLM call is charged; new Experience "
        "rows carry provenance=slow_scope_update_v2"
    ),
    "aggregate_boundary": {
        "lt_0": "REPLAY_AGGREGATE_COLLAPSE",
        "eq_0": (
            "SLOW_CLOSES_SCOPE_GAP_BY_VETO -- veto to identity; the forgone "
            "+0.029688 aggregate per arm is booked explicitly"
        ),
        "gt_0_and_series_clear": (
            "SLOW_CLOSES_SCOPE_GAP_BY_RESCOPE -- %s no longer crosses the "
            "%+.3f line and the aggregate stays positive"
            % (THE_HARMED_SERIES, HARM_THRESHOLD)
        ),
    },
    "verdicts": {
        "SLOW_CLOSES_SCOPE_GAP_BY_RESCOPE": "see aggregate_boundary.gt_0_and_series_clear",
        "SLOW_CLOSES_SCOPE_GAP_BY_VETO": "see aggregate_boundary.eq_0",
        "SLOW_DECLINES_PATCH_FINAL": (
            "every draw on both backends returned the no_proposal envelope: "
            "the grammar is expressible and the evidence complete, and Slow "
            "still declines -- the family closes on a credible negative"
        ),
        "COMPILER_REJECTS": "the EditController refused the proposal",
        "REPLAY_NO_CHANGE": (
            "the guard never vetoed -- not triggered, or RECORD_ONLY: no "
            "pooled decision moved"
        ),
        "REPLAY_HARM_PERSISTS": (
            "a veto moved the plan but %s still crosses the %+.3f line"
            % (THE_HARMED_SERIES, HARM_THRESHOLD)
        ),
        "REPLAY_AGGREGATE_COLLAPSE": "one series was saved at the cost of the aggregate",
        "PATCH_OVERREACH": (
            "the per_channel regression cells moved, or the edit touched "
            "anything beyond the one authorized surface"
        ),
        "INCONCLUSIVE_TRANSPORT": (
            "three consecutive transport failures; sampling budget is not "
            "consumed by them"
        ),
        "INSTRUMENT_DRIFT": "the non-regression gate failed",
        "SCHEMA_BLOCKED": "the evidence no longer fits the input schema",
    },
    "guard_grammar": GUARD_GRAMMAR_V2,
    "guard_contract": GUARD_CONTRACT_V2,
    "backend_rounds": [dict(row) for row in SLOW_BACKEND_ROUNDS_V2],
    "backend_identity_backfill_18": dict(BACKEND_IDENTITY_BACKFILL_18),
    "adapter": ADAPTER_DECLARATION,
    "discipline": {
        "llm_call_budget": LLM_CALL_BUDGET_V2,
        "retrain_budget": RETRAIN_BUDGET,
        "reads": (
            "identical to #18: the already-exposed windows of "
            "fresh_confirmation_v1 only; nothing beyond index 17520, no "
            "window #17 did not open"
        ),
        "commit": False,
        "spawn": False,
        "rehearsal": (
            "the three Slow branches (proposal, decline, non-firing guard) "
            "are rehearsed 0-LLM through the _backend_factory_v2 injection "
            "point before the live run"
        ),
        "new_files": [
            "evaluation/functional/run_e2_slow_scope_update.py (v2 increment)",
            "evaluation/functional/run_e2_warm_vs_cold_recipe_search.py (O1 diff)",
            "methods/ttha/harness/compiler.py (evaluator promotion)",
            "methods/ttha/harness/h0/verification.json (pre-registered [])",
            "methods/ttha/harness/h0/snapshot.lock.json (regenerated)",
            "methods/ttha/harness/harness_surfaces.json (surface registered)",
            "artifacts/functional/e2/slow_scope_update_v2.json",
            "artifacts/functional/e2/slow_scope_update_v2.md",
            "_scratch/slow_scope_v2/",
        ],
        "v1_artifacts_untouched": True,
    },
}


# ------------------------------------------------------------- B1 attribution
def _plan_label(plan: Mapping[str, Any] | None) -> str:
    if not plan:
        return IDENTITY
    excluded = sorted(str(uid) for uid in (plan.get("excluded_series") or ()))
    program = str(plan["program"])
    return program if not excluded else "%s|minus:%s" % (program, ",".join(excluded))


def _measured_support(record: Mapping[str, Any]) -> dict[str, float]:
    """Every plan this episode measured, at its Support aggregate gain.

    Pure transport: the numbers come out of the episode's own receipt, none
    is recomputed.
    """
    measured: dict[str, float] = {IDENTITY: 0.0}
    for program, gains in (record.get("support_results") or {}).items():
        measured[str(program)] = float(gains["aggregate_gain"])
    mask = record.get("mask_search")
    if mask:
        label = _plan_label({
            "program": mask["program"], "excluded_series": mask["final_excluded"],
        })
        measured[label] = float(mask["support"]["aggregate_gain"])
    recall = record.get("recall_payload") or {}
    confirmation = recall.get("current_support_confirmation")
    if confirmation is not None:
        measured[_plan_label(record.get("recall_candidate_plan"))] = float(
            confirmation["aggregate_gain"]
        )
    return measured


def _min_series_delayed(record: Mapping[str, Any]) -> tuple[float | None, str]:
    """The scalar risk reading the recorded harm ledger actually supports."""
    delayed = record.get("delayed") or {}
    count = int(delayed.get("harmed_eval_series_count") or 0)
    total = float(delayed.get("harmed_eval_series_total_harm") or 0.0)
    if count == 0:
        return None, (
            "the receipt records no evaluation series past the %+.3f line, so "
            "there is no firing reading for the scalar risk slot"
            % HARM_THRESHOLD
        )
    if count == 1:
        return -total, (
            "exactly one evaluation series (%s) crossed the line, so the "
            "recorded total harm is that series' delayed delta"
            % ", ".join(delayed.get("harmed_eval_series") or ())
        )
    raise SystemExit(
        "SCHEMA_BLOCKED: %d harmed series but the receipt carries only their "
        "summed magnitude; the per-series vector is not in the artifact"
        % count
    )


def _case_facts(
    record: Mapping[str, Any], *, bootstrap_ids: Sequence[str],
    capability_skills_present: bool, transport_per_series: bool,
) -> tuple[CaseFacts, dict[str, Any]]:
    measured = _measured_support(record)
    chosen = _plan_label(record.get("final_plan"))
    non_identity = tuple(sorted(key for key in measured if key != IDENTITY))
    clean_u = max(measured.values())
    delayed = record.get("delayed") or {}
    chosen_gain = float(delayed.get("aggregate_gain") or 0.0)
    risk_scalar, risk_note = _min_series_delayed(record)
    retrieved = tuple(sorted(
        str(item) for item in (record.get("retrieval") or {}).get(
            "resolved_skill_ids", ()
        )
        if str(item) not in set(bootstrap_ids)
    ))
    facts = CaseFacts(
        case_id=str(record["episode_id"]),
        is_target=True,
        clean_u=float(clean_u),
        corrupt_u=0.0,
        prepared_u=float(measured.get(chosen, 0.0)),
        damage_d=float(clean_u),
        chosen_gain=chosen_gain,
        candidate_utilities=dict(measured),
        effect_distinct_candidate_ids=non_identity,
        chosen_candidate_id=chosen,
        capability_skill_exists=bool(capability_skills_present),
        normal_retrieval=bool(retrieved) or not capability_skills_present,
        skill_retrieved=bool(retrieved),
        retrieved_capability_skill_ids=retrieved,
        proposed_candidate_exists=bool(non_identity),
        compiled_candidate_exists=bool(non_identity),
        compilation_ok=True,
        execution_ok=True,
        execution_contract_ok=True,
        risk_delta_u=(risk_scalar if transport_per_series else None),
    )
    transport = {
        "episode_id": str(record["episode_id"]),
        "mode": record.get("mode"),
        "transport_per_series_risk_scalar": bool(transport_per_series),
        "candidate_utilities_support_axis": dict(measured),
        "chosen_candidate_id": chosen,
        "chosen_gain_delayed_axis": chosen_gain,
        "risk_delta_u": (risk_scalar if transport_per_series else None),
        "risk_delta_u_note": (
            risk_note if transport_per_series else
            "withheld on purpose: this run reads the episode through the "
            "aggregate only, which is what the closing protocol itself did"
        ),
        "harmed_eval_series": list(delayed.get("harmed_eval_series") or ()),
        "harmed_eval_series_total_harm": float(
            delayed.get("harmed_eval_series_total_harm") or 0.0
        ),
        "retrieved_capability_skill_ids": list(retrieved),
        "risk_epsilon_quoted_from_m0_rules": RISK_EPSILON,
    }
    return facts, transport


def _attribute(
    record: Mapping[str, Any], *, bootstrap_ids: Sequence[str],
    capability_skills_present: bool, transport_per_series: bool,
) -> dict[str, Any]:
    facts, transport = _case_facts(
        record, bootstrap_ids=bootstrap_ids,
        capability_skills_present=capability_skills_present,
        transport_per_series=transport_per_series,
    )
    result = assess_case(facts, rules=M0_RULES)
    attribution = result.attribution
    router = FaultRouter()
    try:
        route = router.allowed_targets(attribution.cause_code)
        authorized = {
            "actionability": route.actionability,
            "target_classes": list(route.target_classes),
            "skill_kinds": list(route.allowed_skill_kinds),
            "operations": list(route.allowed_operations),
            "surface_ids": list(route.allowed_surface_ids),
        }
    except KeyError:
        authorized = {
            "actionability": "EVIDENCE_BACKLOG", "target_classes": [],
            "skill_kinds": [], "operations": [], "surface_ids": [],
        }
    return {
        "transport": transport,
        "stages": [
            {
                "stage": item.stage.value,
                "status": item.status.value,
                "fault_code": item.fault_code,
                "cause_code": item.cause_code,
                "decision_rule_id": item.decision_rule_id,
                "suspect_surface_templates": list(item.suspect_surface_templates),
            }
            for item in result.assessments
        ],
        "attribution": {
            "first_stage": attribution.first_stage,
            "fault_code": attribution.fault_code,
            "cause_code": attribution.cause_code,
            "actionability": attribution.actionability,
            "suspect_surface_templates": list(
                attribution.suspect_surface_templates
            ),
        },
        "route_authorization": authorized,
        "is_scope_risk_face": bool(
            attribution.cause_code == "RISK_GAP"
            or "capability_risk_guard" in authorized["target_classes"]
        ),
    }


def stage_b1(
    episodes: Mapping[str, Mapping[str, Any]], stores: Mapping[str, Any],
) -> dict[str, Any]:
    """First-fault attribution on the recorded pooled task_C episodes."""
    bootstrap = [
        skill.skill_id
        for skill in compile_snapshot(FC.H0_ROOT, verify_lock=False).skills
    ]
    cells: dict[str, Any] = {}
    for arm in ARMS:
        slot = "%s_%s" % (arm.lower(), POOLED)
        record = episodes[slot]
        snapshot = stores[slot]["snapshot"]
        capability = [
            skill.skill_id for skill in snapshot.skills
            if skill.skill_id not in set(bootstrap)
        ]
        cells[slot] = {
            "arm": arm,
            "consumer_variant": POOLED,
            "capability_skills_in_store": capability,
            "with_per_series_risk_reading": _attribute(
                record, bootstrap_ids=bootstrap,
                capability_skills_present=bool(capability),
                transport_per_series=True,
            ),
            "aggregate_only_control": _attribute(
                record, bootstrap_ids=bootstrap,
                capability_skills_present=bool(capability),
                transport_per_series=False,
            ),
        }
    primary = next(
        (
            key for key in ("a5_%s" % POOLED, "a3_%s" % POOLED)
            if cells[key]["with_per_series_risk_reading"]["is_scope_risk_face"]
        ),
        None,
    )
    return {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "bootstrap_skill_ids": bootstrap,
        "cells": cells,
        "primary_cell": primary,
        "primary_cause": (
            None if primary is None
            else cells[primary]["with_per_series_risk_reading"][
                "attribution"
            ]["cause_code"]
        ),
        "machinery": {
            "fold": "evaluation/minipipe/feedback/first_fault.py::assess_case",
            "router": "evaluation/minipipe/feedback/router.py::FaultRouter",
            "rules": "evaluation/minipipe/config/m0_rules.json",
            "risk_epsilon": RISK_EPSILON,
            "unchanged": True,
        },
        "adapter": ADAPTER_DECLARATION,
    }


# ---------------------------------------------------------------- the stores
SLOTS = tuple("%s_%s" % (arm.lower(), variant) for variant in CONSUMERS for arm in ARMS)


def _open_stores() -> dict[str, Any]:
    """A working copy of the closing run's four stores, left untouched there."""
    if not SOURCE_STORE.is_dir():
        raise SystemExit(
            "the closing run's store is gone: %s" % _repo_rel(SOURCE_STORE)
        )
    if WORK_ROOT.exists():
        shutil.rmtree(WORK_ROOT)
    STORE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_STORE, STORE_ROOT)
    stores: dict[str, Any] = {}
    for slot in SLOTS:
        root = STORE_ROOT / slot / "snapshots"
        store = SnapshotStore(root)
        active = json.loads(store.active_path.read_text(encoding="utf-8"))
        sha = str(active["runtime_bundle_sha"])
        snapshot = compile_snapshot(root / sha, verify_lock=False)
        if snapshot.runtime_bundle_sha != sha:
            raise SystemExit("store %s does not recompile to its active sha" % slot)
        stores[slot] = {
            "slot": slot,
            "store": store,
            "snapshot": snapshot,
            "runtime_bundle_sha": sha,
            "harness_content_sha": snapshot.harness_content_sha,
            "skill_ids": [skill.skill_id for skill in snapshot.skills],
            "copied_from": _repo_rel(SOURCE_STORE / slot),
        }
    return stores


def _load_episodes() -> dict[str, Any]:
    """The recorded task_C episodes.  Read-only evidence."""
    payload = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    stage4 = payload["stage_4_confirmation"]
    return {
        "artifact_sha256": _sha256(SOURCE_ARTIFACT),
        "overall_verdict": payload.get("overall_verdict"),
        "window": dict(stage4["window"]),
        "locked_roster": dict(payload["locked_roster"]),
        "records": {
            slot: dict(stage4["cells"][slot]["record"]) for slot in SLOTS
        },
    }


# --------------------------------------------------------------- B2 the patch
PROPOSAL_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "target_surface_id", "guard", "patch_value", "edit_id",
        "target_pattern_id", "predicted_agent_behavior_change",
        "predicted_data_effect", "falsification_condition", "reason",
    ],
    "properties": {
        "target_surface_id": {"type": "string", "minLength": 1},
        "guard": GUARD_GRAMMAR,
        "patch_value": {"type": "object"},
        "edit_id": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$",
        },
        "target_pattern_id": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$",
        },
        "predicted_agent_behavior_change": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "predicted_data_effect": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "falsification_condition": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "reason": {"type": "string", "minLength": 1, "maxLength": 1200},
    },
}

SLOW_NOTE = (
    "One edit, one surface.  The Runtime has already folded the failing "
    "episode into a first fault and the route table has already decided which "
    "surfaces that fault authorizes; both are quoted below, along with the "
    "current content of every authorized surface that exists in this store. "
    "Read the episode evidence and decide what, if anything, should change.\n\n"
    "`guard` is the Scope/Risk rule you want the runtime to enforce, in the "
    "grammar given.  `patch_value` is the complete replacement value for the "
    "surface you name -- the surface catalog says what that value replaces, "
    "so copy the current content and change what you mean to change; anything "
    "you drop is gone.  The guard you name must appear in `patch_value` under "
    "the %r key, and the deterministic controller checks the manifest, the "
    "authorization, the preconditions and the compile before anything is "
    "applied.  It gives no second attempt.\n\n"
    "If the public evidence does not justify an edit, return the no_proposal "
    "envelope instead." % GUARD_KEY
)


def behavior_predicate_schema() -> dict[str, Any]:
    """The one definition of a behaviour predicate, as the controller has it.

    ``load_stage_schema`` resolves the injection points -- the forecast
    operator enum among them -- so this is the same object the EditController
    validates the manifest against, not a copy of it.
    """
    return json.loads(
        json.dumps(load_stage_schema("slow_edit_v1")["$defs"]["behavior_predicate"])
    )


def _pattern_example(pattern: str) -> str | None:
    """A value that satisfies ``pattern``, built from the pattern itself.

    The literal head of the pattern is everything before the first regex
    group; appending a plain decimal to it is enough for the two numeric
    predicates this vocabulary has.  The result is checked against the
    pattern before it is offered, so a pattern that changes shape stops
    producing an example rather than producing a wrong one.
    """
    head = str(pattern).lstrip("^")
    cut = head.find("(")
    if cut <= 0:
        return None
    candidate = head[:cut] + "0.5"
    return candidate if re.fullmatch(pattern, candidate) else None


def _behavior_vocabulary() -> list[str]:
    """Only the values that may be written down as they stand.

    A regex used to be rendered here as ``<matching ^...$>`` and sat in the
    same list as the literal tokens.  The #28 live run copied one of those
    strings into a manifest verbatim, the envelope had no opinion about it,
    and the EditController rejected the edit with no retry left.  Patterns
    now live in ``_behavior_patterns`` and are labelled as things to
    instantiate.
    """
    values: list[str] = []
    for branch in behavior_predicate_schema().get("oneOf", ()):
        if "enum" in branch:
            values.extend(str(item) for item in branch["enum"])
    return values


def _behavior_patterns() -> list[dict[str, Any]]:
    """The predicate families that must be instantiated, never copied."""
    out: list[dict[str, Any]] = []
    for branch in behavior_predicate_schema().get("oneOf", ()):
        pattern = branch.get("pattern")
        if not isinstance(pattern, str):
            continue
        row: dict[str, Any] = {
            "regular_expression": pattern,
            "how_to_use": (
                "write a concrete value that this expression accepts; the "
                "expression itself is not a legal value"
            ),
        }
        example = _pattern_example(pattern)
        if example is not None:
            row["example_of_a_legal_value"] = example
        out.append(row)
    return out


def _authorized_surfaces(
    snapshot: Any, route: Mapping[str, Any], registry: SurfaceRegistry,
) -> list[dict[str, Any]]:
    """Every catalog surface this cause authorizes that exists in this store."""
    catalog = json.loads(
        (
            PROJECT_ROOT / "methods" / "ttha" / "harness" / "harness_surfaces.json"
        ).read_text(encoding="utf-8")
    )["surfaces"]
    allowed_classes = set(route["target_classes"])
    allowed_ops = set(route["operations"])
    restricted = set(route.get("surface_ids") or ())
    offered: list[dict[str, Any]] = []
    for definition in catalog:
        template = str(definition["surface_template_id"])
        if str(definition["target_class"]) not in allowed_classes:
            continue
        if not (set(definition["allowed_operations"]) & allowed_ops):
            continue
        if "{skill_id}" in template:
            for skill in snapshot.skills:
                surface_id = template.format(skill_id=skill.skill_id)
                if restricted and surface_id not in restricted:
                    continue
                if skill.skill_kind.value not in set(
                    definition.get("allowed_skill_kinds") or ()
                ):
                    continue
                offered.append({
                    "surface_id": surface_id,
                    "target_class": definition["target_class"],
                    "surface_type": definition["surface_type"],
                    "operations": list(definition["allowed_operations"]),
                    "precondition": definition["precondition"],
                    "patch_replaces": (
                        "the whole %s member of Skill %s"
                        % (definition.get("json_pointer", "/"), skill.skill_id)
                    ),
                    "current_value": wvc._plain(skill.risk_guards or {}),
                })
            continue
        if restricted and template not in restricted:
            continue
        if template == "verification.rules":
            offered.append({
                "surface_id": template,
                "target_class": definition["target_class"],
                "surface_type": definition["surface_type"],
                "operations": list(definition["allowed_operations"]),
                "precondition": definition["precondition"],
                "patch_replaces": "the whole verification document",
                "current_value": wvc._plain(snapshot.verification),
                "compiler_requires": [
                    "schema_version stays \"verification/1\"",
                    "identity_unfilterable stays true",
                    "require_explicit_choice stays true",
                ],
            })
    return offered


def _episode_evidence(
    episodes: Mapping[str, Any], attribution: Mapping[str, Any],
) -> dict[str, Any]:
    """The failing episode, in public numbers only."""
    rows = {}
    for slot in ("a5_%s" % POOLED, "a3_%s" % POOLED):
        record = episodes["records"][slot]
        delayed = record.get("delayed") or {}
        support = record.get("support") or {}
        rows[slot] = {
            "mode": record.get("mode"),
            "adopted_plan": record.get("final_plan"),
            "support_aggregate_gain": support.get("aggregate_gain"),
            "delayed_aggregate_gain": delayed.get("aggregate_gain"),
            "evaluation_series_past_the_material_line": list(
                delayed.get("harmed_eval_series") or ()
            ),
            "their_summed_loss_magnitude": delayed.get(
                "harmed_eval_series_total_harm"
            ),
            "plans_measured_this_episode": _measured_support(record),
            "adoption_path": (record.get("adoption_ladder") or {}).get("path"),
            "adoption_path_text": (record.get("adoption_ladder") or {}).get(
                "path_text"
            ),
            "consumer_retrains": record.get("consumer_retrains_total"),
            "llm_calls": record.get("llm_calls"),
        }
    return {
        "window": dict(episodes["window"]),
        "evaluation_roster": list(episodes["locked_roster"]["eval"]),
        "training_roster_size": len(episodes["locked_roster"]["train"]),
        "material_line": MATERIAL_THRESHOLD,
        "harm_line": HARM_THRESHOLD,
        "arms": rows,
        "what_the_runtime_folded": attribution,
        "adoption_ladder_is_frozen": (
            "the ladder picks the Support winner, sets the bar at "
            "max(0, that winner's full-batch delayed), adopts the named plan "
            "when it clears the bar, and otherwise falls back.  It is not "
            "modifiable here and it may not be re-ranked."
        ),
    }


# ------------------------------------- #19 O2: every exposed harm observation
def _harm_observations(payload: Mapping[str, Any]) -> dict[str, Any]:
    """All harm readings fresh_confirmation_v1 already exposed, quoted read-only.

    Pure transport from the artifact's own text: the two pooled task_C
    adoption receipts and every probe-window reading whose harm ledger is
    non-empty.  Nothing is recomputed and no window is opened.
    """
    rows: list[dict[str, Any]] = []
    stage4 = payload["stage_4_confirmation"]
    for slot in ("a5_%s" % POOLED, "a3_%s" % POOLED):
        record = stage4["cells"][slot]["record"]
        delayed = record.get("delayed") or {}
        rows.append({
            "source_pointer": "stage_4_confirmation.cells.%s.record" % slot,
            "window_id": str(stage4["window"]["window_id"]),
            "arm": slot.split("_")[0].upper(),
            "consumer_variant": POOLED,
            "adoption_mode": record.get("mode"),
            "adopted_plan": record.get("final_plan"),
            "delayed_aggregate_gain": delayed.get("aggregate_gain"),
            "harmed_eval_series": list(delayed.get("harmed_eval_series") or ()),
            "harmed_eval_series_total_harm": delayed.get(
                "harmed_eval_series_total_harm"
            ),
        })
    stage2 = payload["stage_2_adaptation"]["cells"]
    probe_slots = []
    for slot in sorted(stage2):
        probe = stage2[slot].get("probe") or {}
        if probe.get("harmed_eval_series"):
            probe_slots.append(slot)
            rows.append({
                "source_pointer": "stage_2_adaptation.cells.%s.probe" % slot,
                "window_id": str(
                    (payload.get("probe_window") or {}).get("window_id")
                ),
                "arm": slot.split("_")[0].upper(),
                "consumer_variant": slot.split("_", 1)[1],
                "adoption_mode": "promotion_probe_out_of_selection",
                "macro_gain": probe.get("macro_gain"),
                "harmed_eval_series": list(probe["harmed_eval_series"]),
                "harmed_eval_series_total_harm": probe.get(
                    "harmed_eval_series_total_harm"
                ),
            })
    return {
        "zero_new_measurement": True,
        "read_only_quote_of": _repo_rel(SOURCE_ARTIFACT),
        "observations": rows,
        "probe_coverage_note": (
            "the task book asked for two probe-window harm records; the "
            "artifact actually exposes %d (probe slots with a non-empty harm "
            "ledger: %s; a3_%s has no probe record at all -- A3 formed no "
            "Skill there to probe).  Every exposed one is quoted."
            % (len(probe_slots), probe_slots, POOLED)
        ),
    }


def _load_episodes_v2() -> dict[str, Any]:
    """The #18 episode extraction plus the O2 harm observations."""
    out = _load_episodes()
    payload = json.loads(SOURCE_ARTIFACT.read_text(encoding="utf-8"))
    out["harm_observations"] = _harm_observations(payload)
    return out


def _episode_evidence_v2(
    episodes: Mapping[str, Any], attribution: Mapping[str, Any]
) -> dict[str, Any]:
    """The #18 evidence block plus every exposed harm observation (O2)."""
    evidence = _episode_evidence(episodes, attribution)
    evidence["exposed_harm_observations"] = episodes["harm_observations"]
    return evidence


def _make_proposal_validator(offered_ids: Sequence[str]) -> Any:
    allowed = set(str(item) for item in offered_ids)

    def validate(payload: Mapping[str, Any]) -> None:
        surface = str(payload["target_surface_id"])
        if surface not in allowed:
            raise ValueError(
                "target_surface_id must be one of the authorized surfaces"
            )
        guard = wvc._plain(payload["guard"])
        value = payload["patch_value"]
        if not isinstance(value, Mapping):
            raise ValueError("patch_value must be an object")
        carried = value.get(GUARD_KEY)
        if not isinstance(carried, (list, tuple)) or not carried:
            raise ValueError(
                "patch_value must carry a non-empty %r list" % GUARD_KEY
            )
        if not any(
            wvc._plain(row) == guard for row in carried if isinstance(row, Mapping)
        ):
            raise ValueError(
                "the guard you named must appear verbatim in patch_value[%r]"
                % GUARD_KEY
            )

    return validate


ABSTENTION_SAMPLING = (
    "The pre-registration covers a compiler rejection -- which stops the run "
    "immediately and gets no second attempt -- but says nothing about the "
    "Slow Agent declining to propose at all.  The no_proposal envelope is a "
    "legitimate option rather than a failed attempt, so this protocol asks at "
    "most %d times with a byte-identical prompt, stops at the first proposal, "
    "and records every attempt including the ones that abstained.  A compiler "
    "rejection still ends the run on the spot.  Any proposal obtained after "
    "an abstention is a second draw from a stochastic process and is labelled "
    "as such wherever it is reported."
)
MAX_PROPOSAL_ATTEMPTS = 2


def stage_b2(
    *, stores: Mapping[str, Any], episodes: Mapping[str, Any],
    attribution: Mapping[str, Any], budget: Any,
) -> dict[str, Any]:
    """One Slow proposal on one authorized Scope/Risk surface."""
    started = time.perf_counter()
    primary_slot = attribution["primary_cell"]
    cause = attribution["primary_cause"]
    if primary_slot is None or cause is None:
        return {
            "ran": False,
            "verdict": "ATTRIBUTION_LANDS_ELSEWHERE",
            "reason": (
                "no pooled arm folded to a Scope/Risk face, so there is no "
                "authorized surface for this slice to move"
            ),
            "llm_calls": 0,
        }
    route = attribution["cells"][primary_slot][
        "with_per_series_risk_reading"
    ]["route_authorization"]
    snapshot = stores[primary_slot]["snapshot"]
    registry = SurfaceRegistry()
    offered = _authorized_surfaces(snapshot, route, registry)
    if not offered:
        return {
            "ran": False,
            "verdict": "SCHEMA_BLOCKED",
            "reason": (
                "the route authorizes classes %s but this store holds no "
                "surface of any of them" % route["target_classes"]
            ),
            "llm_calls": 0,
        }
    public_input = {
        "stage_note": SLOW_NOTE,
        "attributed_fault": attribution["cells"][primary_slot][
            "with_per_series_risk_reading"
        ]["attribution"],
        "attributed_from_episode": primary_slot,
        "route_authorization": dict(route),
        "authorized_surfaces": offered,
        "episode_evidence": _episode_evidence(episodes, attribution["cells"]),
        "guard_grammar": GUARD_GRAMMAR,
        "guard_contract": GUARD_CONTRACT,
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
            "predicted_agent_behavior_vocabulary": _behavior_vocabulary(),
            "predicted_agent_behavior_patterns": _behavior_patterns(),
            "predicted_data_effect": "free text, what should happen to the data",
            "falsification_condition": "free text, what would show this edit wrong",
        },
    }
    view = resolve_harness_view(snapshot, {}, role="slow")
    record: dict[str, Any] = {
        "ran": True,
        "attributed_from_episode": primary_slot,
        "confirmed_cause": cause,
        "authorized_surfaces_offered": [
            str(row["surface_id"]) for row in offered
        ],
        "public_input_sha256": canonical_sha256(wvc._plain(public_input)),
        "harness_view_sha": view.effective_harness_view_sha,
        "role": "slow",
        "stage": "edit",
        "no_second_attempt": (
            "envelope-shape retries are the agent core's own teaching loop; "
            "the EditController is called once and a rejection stops the run"
        ),
        "abstention_sampling": ABSTENTION_SAMPLING % MAX_PROPOSAL_ATTEMPTS,
        "max_proposal_attempts": MAX_PROPOSAL_ATTEMPTS,
    }
    attempts: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    llm_calls = 0
    validator = _make_proposal_validator(
        [str(item["surface_id"]) for item in offered]
    )
    for attempt in range(1, MAX_PROPOSAL_ATTEMPTS + 1):
        backend = ssi._default_backend_factory(int(budget.take(4)))
        gateway = wvc.NoToolGateway({
            "protocol": PROTOCOL_VERSION, "stage": "edit",
        })
        core = TTHAAgentCore(
            backend, gateway, model=ssi.NF_MODEL, base_url=ssi.NF_BASE_URL,
        )
        row: dict[str, Any] = {"attempt": attempt}
        try:
            result = core.run_stage(
                role=AgentRole.SLOW,
                stage="edit",
                case_id="SSU_%s" % primary_slot,
                public_input=public_input,
                harness_view=view,
                output_schema_name="slow_scope_guard_v1",
                output_schema=PROPOSAL_SCHEMA,
                source_snapshot_sha=view.effective_harness_view_sha,
                validation_retries=wvc.VALIDATION_RETRIES,
                post_validator=validator,
            )
        except Exception as exc:  # noqa: BLE001
            budget.spend(int(backend.calls))
            llm_calls += int(backend.calls)
            row.update({
                "outcome": "STAGE_ERROR",
                "stage_error": "%s: %s" % (type(exc).__name__, exc),
                "llm_calls": int(backend.calls),
            })
            attempts.append(row)
            record.update({
                "verdict": "COMPILER_REJECTS",
                "attempts": attempts,
                "stage_error": row["stage_error"],
                "llm_calls": llm_calls,
                "wall_seconds": time.perf_counter() - started,
            })
            return record
        budget.spend(int(backend.calls))
        llm_calls += int(backend.calls)
        candidate = dict(result.payload)
        row.update({
            "llm_calls": int(backend.calls),
            "validation_retry_count": int(result.validation_retry_count),
            "validation_error_codes": list(result.validation_error_codes),
            "no_proposal_reason": result.no_proposal_reason,
            "outcome": "PROPOSAL" if candidate else "NO_PROPOSAL",
        })
        attempts.append(row)
        print(
            "B2 attempt %d: %s%s"
            % (
                attempt, row["outcome"],
                "" if candidate else " (%s)" % result.no_proposal_reason,
            ),
            flush=True,
        )
        if candidate:
            payload = candidate
            break
    record["attempts"] = attempts
    record["llm_calls"] = llm_calls
    record["proposal_attempt"] = len(attempts)
    record["validation_retry_count"] = int(
        attempts[-1].get("validation_retry_count") or 0
    )
    record["validation_error_codes"] = list(
        attempts[-1].get("validation_error_codes") or ()
    )
    if not payload:
        record.update({
            "verdict": "SLOW_ABSTAINS",
            "proposal": None,
            "reason": (
                "the Slow Agent returned the no_proposal envelope on all %d "
                "attempts (%s); this outcome is not in the pre-registered set "
                "and is reported as it stands"
                % (
                    len(attempts),
                    ", ".join(
                        str(item.get("no_proposal_reason")) for item in attempts
                    ),
                )
            ),
            "wall_seconds": time.perf_counter() - started,
        })
        return record
    if len(attempts) > 1:
        record["obtained_after_abstention"] = (
            "attempt %d; the earlier attempts abstained with %s.  This "
            "proposal is a second draw and every reading downstream of it "
            "inherits that."
            % (
                len(attempts),
                ", ".join(
                    str(item.get("no_proposal_reason"))
                    for item in attempts[:-1]
                ),
            )
        )
    record["proposal"] = wvc._plain(payload)
    record["guard"] = wvc._plain(payload["guard"])
    record["target_surface_id"] = str(payload["target_surface_id"])

    applications: dict[str, Any] = {}
    for slot in SLOTS:
        applications[slot] = _apply_patch(
            stores[slot], payload=payload, cause=cause,
        )
    primary = applications[primary_slot]
    if not primary.get("applied"):
        record.update({
            "verdict": "COMPILER_REJECTS",
            "applications": applications,
            "reason": primary.get("reason"),
            "public_feedback": primary.get("public_feedback"),
            "wall_seconds": time.perf_counter() - started,
        })
        return record
    touched = sorted({
        surface
        for row in applications.values()
        for surface in (row.get("source_surfaces_changed") or ())
    })
    if len(touched) != 1:
        record.update({
            "verdict": "PATCH_OVERREACH",
            "applications": applications,
            "reason": "the edit moved %d surfaces: %s" % (len(touched), touched),
            "wall_seconds": time.perf_counter() - started,
        })
        return record
    record.update({
        "verdict": "PATCH_APPLIED",
        "applications": applications,
        "surfaces_changed": touched,
        "applied_to": sorted(
            slot for slot, row in applications.items() if row.get("applied")
        ),
        "not_applied_to": {
            slot: row.get("reason")
            for slot, row in applications.items() if not row.get("applied")
        },
        "wall_seconds": time.perf_counter() - started,
    })
    return record


def _apply_patch(
    slot: Mapping[str, Any], *, payload: Mapping[str, Any], cause: str,
) -> dict[str, Any]:
    """The deterministic controller, once, on one store."""
    store = slot["store"]
    snapshot = slot["snapshot"]
    surface_id = str(payload["target_surface_id"])
    controller = EditController(
        store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )
    out: dict[str, Any] = {"slot": slot["slot"], "surface_id": surface_id}
    if surface_id.startswith("skill_library.entries/"):
        wanted = surface_id.split("/", 1)[1].rsplit(".", 1)[0]
        if not any(skill.skill_id == wanted for skill in snapshot.skills):
            out.update({
                "applied": False,
                "reason": (
                    "this store holds no Skill %r, so the surface does not "
                    "exist here" % wanted
                ),
                "stage": "surface_absent",
                "surface_exists_in_this_store": False,
            })
            return out
    try:
        parent = store.materialize(snapshot)
        precondition = controller.surface_precondition_sha(parent, surface_id)
    except Exception as exc:  # noqa: BLE001
        out.update({
            "applied": False,
            "reason": "%s: %s" % (type(exc).__name__, exc),
            "stage": "precondition",
        })
        return out
    manifest = EditManifest(
        edit_id=str(payload["edit_id"]),
        base_harness_sha=snapshot.harness_content_sha,
        target_pattern_id=str(payload["target_pattern_id"]),
        target_surface_id=surface_id,
        operation=EditOperation.PATCH,
        surface_precondition={"kind": "SHA", "sha": precondition},
        dependency_precondition_shas={},
        minimal_patch={"value": wvc._plain(payload["patch_value"])},
        new_value=None,
        observable_applicability=None,
        predicted_agent_behavior_change=tuple(
            str(item) for item in payload["predicted_agent_behavior_change"]
        ),
        predicted_data_effect=tuple(
            str(item) for item in payload["predicted_data_effect"]
        ),
        automatically_selected_risk_cases=(),
        falsification_condition=tuple(
            str(item) for item in payload["falsification_condition"]
        ),
        patch_id=None,
    )
    try:
        receipt = controller.apply_to_fork(
            parent,
            _resolve_apply_manifest(manifest, snapshot),
            confirmed_cause=cause,
        )
    except Exception as exc:  # noqa: BLE001
        out.update({
            "applied": False,
            "reason": "%s: %s" % (type(exc).__name__, exc),
            "public_feedback": {
                "code": EditController.public_feedback_for_error(exc).error_code,
                "message": EditController.public_feedback_for_error(
                    exc
                ).public_message,
            },
            "stage": "apply_to_fork",
        })
        return out
    updated = receipt.candidate_snapshot.snapshot
    store.set_active(updated.runtime_bundle_sha)
    slot["snapshot"] = updated
    slot["runtime_bundle_sha"] = updated.runtime_bundle_sha
    out.update({
        "applied": True,
        "edit_id": receipt.edit_id,
        "confirmed_cause": receipt.confirmed_cause,
        "parent_harness_content_sha": receipt.parent_harness_content_sha,
        "candidate_harness_content_sha": receipt.candidate_harness_content_sha,
        "source_surfaces_changed": list(receipt.source_surfaces_changed),
        "derived_outputs_changed": list(receipt.derived_outputs_changed),
        "applied_edit_sha": receipt.applied_edit_sha,
    })
    return out


# ---------------------------------------------------- the guard the patch left
def _active_guards(snapshot: Any) -> list[dict[str, Any]]:
    """Every Scope/Risk guard the active snapshot declares, wherever it lives."""
    found: list[dict[str, Any]] = []
    verification = dict(snapshot.verification or {})
    for row in verification.get(GUARD_KEY) or ():
        if isinstance(row, Mapping):
            found.append({"where": "verification.rules", "guard": dict(row)})
    for skill in snapshot.skills:
        for row in (skill.risk_guards or {}).get(GUARD_KEY) or ():
            if isinstance(row, Mapping):
                found.append({
                    "where": "skill_library.entries/%s.risk_guards" % skill.skill_id,
                    "guard": dict(row),
                })
    return found


def _statistic(name: str, gains: Mapping[str, Any], eval_count: int) -> float:
    """Every branch reads a field the frozen instrument already returns."""
    aggregate = float(gains["aggregate_gain"])
    harmed = float(gains["harmed_eval_series_count"])
    total_harm = float(gains["harmed_eval_series_total_harm"])
    if name == "aggregate_gain":
        return aggregate
    if name == "harmed_series_count":
        return harmed
    if name == "harmed_series_fraction":
        return harmed / max(1, int(eval_count))
    if name == "total_harm":
        return total_harm
    if name == "gain_to_total_harm_ratio":
        return aggregate / total_harm if total_harm > 0.0 else float("inf")
    if name == "min_per_series_gain":
        # #19 O1: since the repair, the measured per-series vector rides the
        # same gains dict under ``per_eval_series_gain``; reduce it with the
        # minimum, the reading the aggregate hid.
        vector = gains.get("per_eval_series_gain")
        if not isinstance(vector, Mapping) or not vector:
            raise ValueError(
                "min_per_series_gain needs the per-series vector the O1 "
                "repair exposes; this gains dict does not carry it"
            )
        return min(float(value) for value in vector.values())
    raise ValueError("unknown guard statistic: %s" % name)


def _fires(guard: Mapping[str, Any], value: float) -> bool:
    threshold = float(guard["threshold"])
    comparator = str(guard["comparator"])
    if comparator == "lt":
        return value < threshold
    if comparator == "le":
        return value <= threshold
    if comparator == "gt":
        return value > threshold
    if comparator == "ge":
        return value >= threshold
    raise ValueError("unknown comparator: %s" % comparator)


def _check(
    guards: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any],
    support: Mapping[str, Any], delayed: Mapping[str, Any], eval_count: int,
    reused: bool,
) -> dict[str, Any]:
    readings: list[dict[str, Any]] = []
    for row in guards:
        guard = dict(row["guard"])
        if (
            str(guard["applies_to"]) == "reused_skill_adoption_only"
            and not reused
        ):
            readings.append({
                "guard_id": guard["guard_id"], "where": row["where"],
                "checked": False, "fired": False,
                "why_not_checked": "this adoption did not come from a recalled Skill",
            })
            continue
        window = str(guard["window"])
        gains = delayed if window == "delayed" else support
        value = _statistic(str(guard["statistic"]), gains, eval_count)
        fired = _fires(guard, value)
        readings.append({
            "guard_id": guard["guard_id"],
            "where": row["where"],
            "checked": True,
            "plan": dict(plan),
            "window": window,
            "statistic": guard["statistic"],
            "value": (None if value == float("inf") else float(value)),
            "value_is_infinite": value == float("inf"),
            "comparator": guard["comparator"],
            "threshold": float(guard["threshold"]),
            "fired": bool(fired),
            "action": guard["action"],
        })
    vetoed = [
        row for row in readings
        if row.get("fired") and row.get("action") == "VETO_AND_FALL_BACK"
    ]
    return {
        "readings": readings,
        "any_fired": any(row.get("fired") for row in readings),
        "vetoed": bool(vetoed),
        "vetoed_by": [str(row["guard_id"]) for row in vetoed],
    }


def _enforce(
    *, search: Any, snapshot: Any, ladder_view: Mapping[str, Any], reused: bool,
) -> dict[str, Any]:
    """Run the declared guards over the frozen ladder's output.

    The ladder is not modified and not re-ranked: this reads the plan it
    already chose, and on a veto walks the ladder's own fallback.
    """
    guards = _active_guards(snapshot)
    plan = dict(ladder_view["final_plan"])
    support = dict(ladder_view["support"])
    delayed = dict(ladder_view["delayed"])
    eval_count = len(search.eval_uids)
    out: dict[str, Any] = {
        "guards_declared": guards,
        "guard_count": len(guards),
        "plan_before": dict(plan),
        "identity_never_vetoed": True,
        "changed": False,
        "extra_consumer_retrains": 0,
    }
    if not guards:
        out.update({
            "checked": False,
            "why": "the active snapshot declares no Scope/Risk guard",
            "plan_after": dict(plan), "support_after": support,
            "delayed_after": delayed,
        })
        return out
    if str(plan["program"]) == IDENTITY:
        out.update({
            "checked": False,
            "why": "identity is unfilterable and is never vetoed",
            "plan_after": dict(plan), "support_after": support,
            "delayed_after": delayed,
        })
        return out
    before = search.retrains
    first = _check(
        guards, plan=plan, support=support, delayed=delayed,
        eval_count=eval_count, reused=reused,
    )
    out["check_on_adopted_plan"] = first
    if not first["vetoed"]:
        out.update({
            "checked": True,
            "plan_after": dict(plan), "support_after": support,
            "delayed_after": delayed,
            "why": (
                "a guard fired but only asked for a record"
                if first["any_fired"] else "no guard fired"
            ),
        })
        return out

    winner = ladder_view.get("support_winner")
    winner_delayed = ladder_view.get("support_winner_full_batch_delayed")
    fallback = {"program": IDENTITY, "excluded_series": []}
    fallback_source = "identity"
    if (
        winner is not None
        and winner_delayed is not None
        and float(winner_delayed) > 0.0
        and not (
            str(winner) == str(plan["program"]) and not plan["excluded_series"]
        )
    ):
        fallback = {"program": str(winner), "excluded_series": []}
        fallback_source = "the ladder's Support winner, full batch"
    fallback_delayed = dict(
        search.delayed_gate(fallback["program"], list(fallback["excluded_series"]))
    )
    fallback_support = dict(
        search.support_of_plan(
            fallback["program"], list(fallback["excluded_series"])
        )
    )
    second = None
    if str(fallback["program"]) != IDENTITY:
        second = _check(
            guards, plan=fallback, support=fallback_support,
            delayed=fallback_delayed, eval_count=eval_count, reused=reused,
        )
        if second["vetoed"]:
            fallback = {"program": IDENTITY, "excluded_series": []}
            fallback_source = (
                "identity: the fallback candidate fired the same guard"
            )
            fallback_delayed = dict(search.delayed_gate(IDENTITY, []))
            fallback_support = dict(search.support_of_plan(IDENTITY, []))
    out.update({
        "checked": True,
        "check_on_fallback": second,
        "fallback_source": fallback_source,
        "plan_after": dict(fallback),
        "support_after": fallback_support,
        "delayed_after": fallback_delayed,
        "changed": True,
        "extra_consumer_retrains": int(search.retrains - before),
    })
    return out


# --------------------------------------------------------------- B3 the replay
def _confirmation_payload(roster: Mapping[str, Any]) -> dict[str, Any]:
    """The same cohort #17 read, rebuilt from what it left on disk.

    Development values below 8760 and the materialized 2025 partition are
    concatenated exactly as the closing run concatenated them.  No csv is
    parsed again and nothing outside index 17520 exists to read.
    """
    stations = [str(uid) for uid in roster["train"]] + [
        str(uid) for uid in roster["eval"]
    ]
    development = FC._load_development(stations)
    extended: dict[str, np.ndarray] = {}
    for station in stations:
        confirmation = np.asarray(
            np.load(FC.CONFIRMATION_DIR / station / "values.npy"),
            dtype=np.float64,
        )
        if int(confirmation.size) != FC.CONFIRMATION_END - FC.DEVELOPMENT_HOURS:
            raise SystemExit(
                "confirmation series %s is %d long" % (station, confirmation.size)
            )
        extended[station] = np.concatenate([
            development["values"][station], confirmation
        ])
    return FC._cohort_payload(roster["train"], roster["eval"], extended)


def _replay_full_price(
    *, search: Any, target: Mapping[str, Any], arm: str,
    window: Mapping[str, Any], slot: Mapping[str, Any],
    expected_card: str | None, expected_local: str | None,
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    """FC._episode with the two Agent turns replayed from the receipt.

    The shortlist, the mask request and the named plan are the ones the Agent
    produced in #17, quoted verbatim; every measurement below is taken again
    on the same window through the same frozen instrument.  0 LLM.
    """
    started = time.perf_counter()
    episode_id = "task_C_%s_%s_replay" % (target["consumer_variant"], arm)
    _view, retrieval, context = FC._retrieval(
        slot["_snapshot"], search, expected_card, expected_local,
    )
    shortlist = [str(item) for item in recorded["shortlist"]]
    wants_mask = bool(recorded["request_mask_search"])
    support_results = {
        program: search.full_batch_support(program) for program in shortlist
    }
    mask_result = None
    if wants_mask:
        best = max(
            shortlist,
            key=lambda program: (
                support_results[program]["aggregate_gain"],
                -shortlist.index(program),
            ),
        )
        mask_result = search.mask_search(best)
    plans, mask_note = FC.bridge._measured_plans(
        shortlist=shortlist, support_results=support_results,
        mask_result=mask_result,
    )
    named = {
        "program": str(recorded["adopted_plan"]["program"]),
        "excluded_series": sorted(
            str(uid) for uid in recorded["adopted_plan"]["excluded_series"]
        ),
    }
    ladder = ssi._ladder(search, plans=plans, named=named)
    return {
        "episode_id": episode_id,
        "task": "task_C",
        "arm": arm,
        "consumer_variant": str(target["consumer_variant"]),
        "window_id": str(window["window_id"]),
        "store_slot": slot["slot"],
        "mode": "FULL_PRICE_SEARCH_REPLAYED",
        "llm_calls": 0,
        "replayed_from_receipt": {
            "shortlist": shortlist,
            "request_mask_search": wants_mask,
            "adopted_plan": dict(named),
            "why": (
                "the guard cannot change what the Agent proposed, only what "
                "survives the gate, so the Agent turns are replayed rather "
                "than re-asked"
            ),
        },
        "retrieval": retrieval,
        "task_context": {
            key: value for key, value in context.items() if key != "per_series"
        },
        "support_results": support_results,
        "mask_search": wvc._plain(mask_result),
        "measured_plans": plans,
        "measured_plans_note": mask_note,
        "adopted_plan": dict(named),
        "adoption_ladder": {
            key: value for key, value in ladder.items()
            if key not in ("support", "delayed")
        },
        "final_plan": dict(ladder["final_plan"]),
        "support": dict(ladder["support"]),
        "delayed": dict(ladder["delayed"]),
        "relation": FC._relation(
            float(ladder["support"]["aggregate_gain"]),
            float(ladder["delayed"]["aggregate_gain"]),
            ladder["final_plan"]["program"],
        ),
        "evaluations_used": int(search.support_evaluations_charged),
        "consumer_retrains_total": int(search.retrains),
        "wall_seconds": time.perf_counter() - started,
    }


def _reproduction(recorded: Mapping[str, Any], fresh: Mapping[str, Any]) -> dict[str, Any]:
    """Does the unguarded replay land on the recorded numbers, digit for digit."""
    def _pair(key: str, path: Sequence[str]) -> dict[str, Any]:
        left: Any = recorded
        right: Any = fresh
        for token in path:
            left = (left or {}).get(token) if isinstance(left, Mapping) else None
            right = (right or {}).get(token) if isinstance(right, Mapping) else None
        return {"field": key, "recorded": left, "replayed": right, "same": left == right}

    checks = [
        _pair("final_plan.program", ("final_plan", "program")),
        _pair("final_plan.excluded_series", ("final_plan", "excluded_series")),
        _pair("support.aggregate_gain", ("support", "aggregate_gain")),
        _pair("delayed.aggregate_gain", ("delayed", "aggregate_gain")),
        _pair("delayed.harmed_eval_series", ("delayed", "harmed_eval_series")),
        _pair(
            "delayed.harmed_eval_series_total_harm",
            ("delayed", "harmed_eval_series_total_harm"),
        ),
        _pair("adoption_ladder.path", ("adoption_ladder", "path")),
    ]
    return {
        "checks": checks,
        "reproduces": all(row["same"] for row in checks),
        "mismatched": [row["field"] for row in checks if not row["same"]],
    }


def _per_series(search: Any, plan: Mapping[str, Any]) -> dict[str, float]:
    """The per-evaluation-series delayed gains of one plan.  Read-out only.

    Since the #19 O1 repair the search instrument returns this vector under
    ``per_eval_series_gain``; this reader recomputes it through
    ``bch._gain_rows`` against the same cached identity rows, so the table
    below cross-checks the passthrough rather than trusting it.  It is
    charged like any other evaluation and it decides nothing: no guard and
    no ladder branch reads it.
    """
    program = str(plan["program"])
    excluded = {str(uid) for uid in (plan.get("excluded_series") or ())}
    base = search._identity_delayed
    rows = base if program == IDENTITY else search._masked(
        program, excluded, search.delayed
    )
    gains = bch._gain_rows(base, rows, search.eval_uids)
    return {
        str(uid): float(value)
        for uid, value in dict(gains["per_eval_series_gain"]).items()
    }


def stage_b3(
    *, payload: Mapping[str, Any], episodes: Mapping[str, Any],
    stores: Mapping[str, Any], window: Mapping[str, Any], budget: Any,
) -> dict[str, Any]:
    """Replay both pooled arms with the patch live; per_channel as regression."""
    cells: dict[str, Any] = {}
    for variant in CONSUMERS:
        target = FC._target(variant)
        for arm in ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            recorded = episodes["records"][slot_key]
            snapshot = stores[slot_key]["snapshot"]
            slot = {"slot": slot_key, "_snapshot": snapshot}
            search = FC.FreshSearch(
                payload=payload, consumer_variant=variant,
                support_origins=window["support_origins"],
                delayed_origins=window["delayed_origins"],
            )
            card = FC.SKILL_ID[variant] if arm == "A5" else None
            local = (recorded.get("retrieval") or {}).get("expected_local_skill_id")
            try:
                if str(recorded.get("mode")) == "DIRECT_RECALL":
                    fresh = FC._direct_recall(
                        search=search, target=target, arm=arm, window=window,
                        slot=slot, expected_card=card, expected_local=str(local),
                        tag="task_C",
                    )
                    reused = bool(fresh.get("reuse_adopted"))
                else:
                    fresh = _replay_full_price(
                        search=search, target=target, arm=arm, window=window,
                        slot=slot, expected_card=card, expected_local=local,
                        recorded=recorded,
                    )
                    reused = False
            except Exception as exc:  # noqa: BLE001
                budget.charge_retrains(int(search.retrains))
                cells[slot_key] = {
                    "arm": arm,
                    "consumer_variant": variant,
                    "replay_error": "%s: %s" % (type(exc).__name__, exc),
                    "why_it_matters": (
                        "the patched snapshot no longer supports the path this "
                        "episode took; that is a consequence of the edit and "
                        "is reported rather than repaired"
                    ),
                    "behaviour_changed": True,
                    "consumer_retrains": int(search.retrains),
                    "llm_calls": 0,
                    "before": {
                        "plan": dict(recorded["final_plan"]),
                        "delayed_aggregate_gain": (
                            recorded.get("delayed") or {}
                        ).get("aggregate_gain"),
                        "per_eval_series_delayed_gain": {},
                        "harmed_eval_series": list(
                            (recorded.get("delayed") or {}).get(
                                "harmed_eval_series"
                            ) or ()
                        ),
                    },
                    "after": {
                        "plan": {"program": None, "excluded_series": []},
                        "delayed_aggregate_gain": None,
                        "per_eval_series_delayed_gain": {},
                        "harmed_eval_series": [],
                    },
                }
                print("B3 %-20s REPLAY ERROR %s" % (slot_key, exc), flush=True)
                continue
            ladder_view = {
                "final_plan": dict(fresh["final_plan"]),
                "support": dict(fresh["support"] or {}),
                "delayed": dict(fresh["delayed"] or {}),
                "support_winner": (fresh.get("adoption_ladder") or {}).get(
                    "support_winner"
                ),
                "support_winner_full_batch_delayed": (
                    fresh.get("adoption_ladder") or {}
                ).get("support_winner_full_batch_delayed"),
            }
            guard = _enforce(
                search=search, snapshot=snapshot, ladder_view=ladder_view,
                reused=reused,
            )
            before_series = _per_series(search, fresh["final_plan"])
            after_series = (
                dict(before_series)
                if dict(guard["plan_after"]) == dict(fresh["final_plan"])
                else _per_series(search, guard["plan_after"])
            )
            budget.charge_retrains(int(search.retrains))
            cells[slot_key] = {
                "arm": arm,
                "consumer_variant": variant,
                "role": (
                    "the failing cell" if variant == POOLED
                    else "regression check, expected not to move"
                ),
                "reuse_path": reused,
                "unguarded_replay": fresh,
                "reproduction_of_the_recorded_episode": _reproduction(
                    recorded, fresh
                ),
                "guard": guard,
                "before": {
                    "plan": dict(fresh["final_plan"]),
                    "support_aggregate_gain": (fresh.get("support") or {}).get(
                        "aggregate_gain"
                    ),
                    "delayed_aggregate_gain": (fresh.get("delayed") or {}).get(
                        "aggregate_gain"
                    ),
                    "per_eval_series_delayed_gain": before_series,
                    "harmed_eval_series": list(
                        (fresh.get("delayed") or {}).get("harmed_eval_series") or ()
                    ),
                    "harmed_eval_series_total_harm": (
                        fresh.get("delayed") or {}
                    ).get("harmed_eval_series_total_harm"),
                },
                "after": {
                    "plan": dict(guard["plan_after"]),
                    "support_aggregate_gain": (
                        guard.get("support_after") or {}
                    ).get("aggregate_gain"),
                    "delayed_aggregate_gain": (
                        guard.get("delayed_after") or {}
                    ).get("aggregate_gain"),
                    "per_eval_series_delayed_gain": after_series,
                    "harmed_eval_series": list(
                        (guard.get("delayed_after") or {}).get(
                            "harmed_eval_series"
                        ) or ()
                    ),
                    "harmed_eval_series_total_harm": (
                        guard.get("delayed_after") or {}
                    ).get("harmed_eval_series_total_harm"),
                },
                "behaviour_changed": bool(guard.get("changed")),
                "consumer_retrains": int(search.retrains),
                "llm_calls": 0,
                "instrument": search.accounting(),
            }
            print(
                "B3 %-20s %s -> %s | delayed %+.6f -> %+.6f | %s %+.6f -> %+.6f "
                "| retrains %d"
                % (
                    slot_key, _plan_label(fresh["final_plan"]),
                    _plan_label(guard["plan_after"]),
                    float((fresh.get("delayed") or {}).get("aggregate_gain") or 0.0),
                    float(
                        (guard.get("delayed_after") or {}).get("aggregate_gain")
                        or 0.0
                    ),
                    THE_HARMED_SERIES,
                    before_series.get(THE_HARMED_SERIES, float("nan")),
                    after_series.get(THE_HARMED_SERIES, float("nan")),
                    int(search.retrains),
                ),
                flush=True,
            )
    return {
        "ran": True,
        "development_level": True,
        "not_fresh": (
            "the pooled task_C window was opened by fresh_confirmation_v1; "
            "this replay reuses it as development evidence and claims nothing "
            "about held-out performance"
        ),
        "window": {
            key: value for key, value in window.items()
            if not str(key).startswith("reference_")
        },
        "cells": cells,
        "consumer_retrains": sum(
            int(row["consumer_retrains"]) for row in cells.values()
        ),
        "llm_calls": 0,
    }


# ------------------------------------------------------------ B4 the accounting
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
            raise SystemExit("LLM call budget exceeded")

    def charge_retrains(self, count: int) -> None:
        self.retrains_charged += int(count)
        if self.retrains_charged > self.retrain_total:
            raise SystemExit(
                "Consumer retrain budget exceeded: %d > %d"
                % (self.retrains_charged, self.retrain_total)
            )


def stage_b4(
    replay: Mapping[str, Any], *, provenance: str = "slow_scope_update",
    tag: str = "ssu", protocol: str = PROTOCOL_VERSION,
) -> dict[str, Any]:
    """Experience rows for what the guarded replay actually decided."""
    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    for slot_key, cell in replay["cells"].items():
        plan = dict(cell["after"]["plan"])
        if str(plan["program"]) == IDENTITY:
            skipped[slot_key] = (
                "the guarded episode ended on identity; this line forms no "
                "Experience row for an abstention adoption"
            )
            continue
        variant = str(cell["consumer_variant"])
        arm = str(cell["arm"])
        steps = [(str(plan["program"]), {})]
        audit = {
            "provenance": provenance,
            "counts_as_unguided_exploration": False,
        }
        support_gain = float(cell["after"]["support_aggregate_gain"])
        delayed_gain = float(cell["after"]["delayed_aggregate_gain"])
        episode = build_episode(
            episode_id="%s_%s_%s_task_c_replay" % (tag, variant, arm.lower()),
            task_consumer_key="batch:%s|consumer:%s" % (FC.COHORT_NAME, variant),
            domain_namespace=str(FC.COHORT_NAME),
            context_summary={
                "task_episode_id": str(replay["window"]["window_id"]),
                "arm": arm,
                "cohort": {"cohort_name": str(FC.COHORT_NAME)},
                "local_pattern": {"consumer_variant": variant},
                "program_geometry": {
                    "program_steps": [
                        {"op": op, "params": dict(params)} for op, params in steps
                    ],
                    "frozen_plan_scope": {
                        "excluded_series": sorted(
                            str(uid) for uid in plan["excluded_series"]
                        )
                    },
                    "consumer_retrains": int(cell["consumer_retrains"]),
                },
            },
            workflow_signature=FC.e1mod._v2_workflow_signature(steps),
            support_response={
                "gain": support_gain,
                "accepted": support_gain >= MATERIAL_THRESHOLD,
                "block_origins": list(replay["window"]["support_origins"]),
                "program": str(plan["program"]),
                "excluded_series": list(plan["excluded_series"]),
                **audit,
            },
            delayed_response={
                "evaluated": True,
                "gain": delayed_gain,
                "se_block": None,
                "gain_over_se": None,
                "block_origins": list(replay["window"]["delayed_origins"]),
                "took_part_in_selection": True,
                "why_not_promotion_evidence": (
                    "a development-level replay on a window this line has "
                    "already opened is not promotion evidence"
                ),
                **audit,
            },
            relation=FC._relation(support_gain, delayed_gain, plan["program"]),
            evidence_level=EVIDENCE_SUPPORT,
            local_status=FC.STATUS_LOCAL_DRAFT,
            evidence_refs=(provenance, protocol),
        )
        rows.append({
            "slot": slot_key,
            "episode_id": episode.episode_id,
            "workflow_signature": episode.workflow_signature,
            "relation": str(episode.relation),
            "local_status": str(episode.local_status),
            "provenance": provenance,
            "counts_as_unguided_exploration": False,
        })
    return {
        "ran": True,
        "provenance": provenance,
        "rows": rows,
        "row_count": len(rows),
        "no_row_written_for": skipped,
        "not_persisted_as_skills": (
            "handle_fast_winner is not called: a replay on an already-opened "
            "window forms no Skill, and this slice is allowed one surface only"
        ),
    }


# ======================================================================
# #19 v2: the composite EDIT_SURFACE_DEFECT repair slice
# ======================================================================
# Flow: migrate the pre-registered empty guard list into the four store
# forks (0 LLM) -> non-regression gate on the migrated state (0 LLM) ->
# Slow retry on the one sutured surface (KIMI, then OPUS only if KIMI
# declines twice) -> deterministic compile -> replay on the gate's own
# unguarded searches, paying only for the plans the guard moves ->
# accounting.  The v1 path above is not modified.

SLOW_NOTE_V2 = (
    "One edit, one surface.  The Runtime has already folded the failing "
    "episode into a first fault and the route table has already decided "
    "which surface classes that fault authorizes; both are quoted below.  "
    "Since that attribution was produced, the edit surface itself was "
    "repaired: there is now exactly one Scope/Risk surface -- the guard "
    "list at /%s in the verification document, pre-registered as empty and "
    "read by the tracked adoption gate after the frozen ladder produces "
    "its final plan and before that plan is recorded as adopted.\n\n"
    "`guard` is the Scope/Risk rule you want the runtime to enforce, in "
    "the grammar given.  `patch_value` is the complete new content of that "
    "list and must hold exactly your one guard; the compiler accepts at "
    "most one entry.  The deterministic controller checks the manifest, "
    "the authorization, the preconditions and the compile before anything "
    "is applied.  It gives no second attempt.\n\n"
    "If the public evidence does not justify an edit, return the "
    "no_proposal envelope instead." % GUARD_KEY
)

PROPOSAL_SCHEMA_V2: dict[str, Any] = json.loads(json.dumps(PROPOSAL_SCHEMA))
PROPOSAL_SCHEMA_V2["properties"]["guard"] = GUARD_GRAMMAR_V2
PROPOSAL_SCHEMA_V2["properties"]["patch_value"] = {
    "type": "array",
    "minItems": 1,
    "maxItems": 1,
    "items": GUARD_GRAMMAR_V2,
}


def _authorized_surfaces_v2(snapshot: Any) -> list[dict[str, Any]]:
    """The one surface this slice may move, resolved against the live catalog."""
    registry = SurfaceRegistry()
    surface = registry.resolve(NEW_SURFACE_V2)
    definition = surface.definition
    return [{
        "surface_id": NEW_SURFACE_V2,
        "target_class": definition.target_class,
        "surface_type": definition.surface_type,
        "operations": list(definition.allowed_operations),
        "precondition": definition.precondition,
        "patch_replaces": (
            "the whole /%s list in the verification document -- "
            "pre-registered as []; patch_value is the complete new "
            "one-entry list" % GUARD_KEY
        ),
        "current_value": wvc._plain(
            (snapshot.verification or {}).get(GUARD_KEY)
        ),
        "single_entry_constraint": (
            "the compiler accepts at most one entry in this list"
        ),
    }]


def _make_proposal_validator_v2() -> Any:
    def validate(payload: Mapping[str, Any]) -> None:
        surface = str(payload["target_surface_id"])
        if surface != NEW_SURFACE_V2:
            raise ValueError(
                "target_surface_id must be %s" % NEW_SURFACE_V2
            )
        guard = wvc._plain(payload["guard"])
        value = payload["patch_value"]
        if not isinstance(value, (list, tuple)) or len(value) != 1:
            raise ValueError(
                "patch_value must be the complete new one-entry %r list"
                % GUARD_KEY
            )
        if not isinstance(value[0], Mapping) or wvc._plain(value[0]) != guard:
            raise ValueError(
                "the single entry of patch_value must be the guard you "
                "named, verbatim"
            )

    return validate


def _backend_factory_v2(label: str, cap: int) -> Any:
    """The live backend for one Slow sampling attempt.

    Module-level on purpose: the 0-LLM rehearsal injects scripted backends
    here.  Both rounds ride the same factory; a round differs only in the
    model name handed to the agent core.
    """
    return ssi._default_backend_factory(cap)


def _open_stores_v2() -> dict[str, Any]:
    """The closing run's four stores, opened under the sutured instrument.

    Those snapshots were compiled by the pre-suture compiler against the
    pre-suture surface catalog, and both files feed runtime_bundle_sha
    through dependency_shas, so a recompile now re-identifies the bundle.
    What may not move is the semantic content: harness_content_sha is
    checked against each snapshot's own recorded lock, and the dependency
    drift must be exactly the two sutured registry inputs.
    """
    if not SOURCE_STORE.is_dir():
        raise SystemExit(
            "the closing run's store is gone: %s" % _repo_rel(SOURCE_STORE)
        )
    if WORK_ROOT_V2.exists():
        shutil.rmtree(WORK_ROOT_V2)
    STORE_ROOT_V2.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE_STORE, STORE_ROOT_V2)
    stores: dict[str, Any] = {}
    for slot in SLOTS:
        root = STORE_ROOT_V2 / slot / "snapshots"
        store = SnapshotStore(root)
        active = json.loads(store.active_path.read_text(encoding="utf-8"))
        recorded_sha = str(active["runtime_bundle_sha"])
        lock = json.loads(
            (root / recorded_sha / "snapshot.lock.json").read_text(encoding="utf-8")
        )
        snapshot = compile_snapshot(root / recorded_sha, verify_lock=False)
        if snapshot.harness_content_sha != str(lock["harness_content_sha"]):
            raise SystemExit(
                "store %s semantic content moved under the suture: %s != %s"
                % (slot, snapshot.harness_content_sha, lock["harness_content_sha"])
            )
        drift = sorted(
            key for key, value in snapshot.dependency_shas.items()
            if (lock.get("dependency_shas") or {}).get(key) != value
        )
        if drift != sorted(["compiler_source", "surface_registry"]):
            raise SystemExit(
                "store %s dependency drift beyond the suture: %s" % (slot, drift)
            )
        stores[slot] = {
            "slot": slot,
            "store": store,
            "snapshot": snapshot,
            "runtime_bundle_sha": snapshot.runtime_bundle_sha,
            "harness_content_sha": snapshot.harness_content_sha,
            "skill_ids": [skill.skill_id for skill in snapshot.skills],
            "copied_from": _repo_rel(SOURCE_STORE / slot),
            "bundle_identity_drift": {
                "recorded_runtime_bundle_sha": recorded_sha,
                "recompiled_runtime_bundle_sha": snapshot.runtime_bundle_sha,
                "dependency_keys_that_moved": drift,
                "why": (
                    "the suture changed compiler.py and harness_surfaces.json; "
                    "both feed runtime_bundle_sha, so a recompile re-identifies "
                    "the bundle.  Semantic content is unchanged, checked above "
                    "against the snapshot's recorded lock."
                ),
            },
        }
    return stores


def stage_migrate_v2(stores: Mapping[str, Any]) -> dict[str, Any]:
    """Install the pre-registered empty guard list into each store fork.

    A whole-document PATCH through the old verification.rules surface,
    because the pointer surface requires the key to pre-exist.  One
    EditController receipt per store, each with an exactly-one-key diff
    proof.  0 LLM, deterministic, idempotent per fresh working copy.
    """
    receipts: dict[str, Any] = {}
    for slot_key, slot in stores.items():
        store = slot["store"]
        snapshot = slot["snapshot"]
        current_doc = wvc._plain(snapshot.verification)
        if GUARD_KEY in current_doc:
            raise SystemExit(
                "store %s already carries %r; migration runs once"
                % (slot_key, GUARD_KEY)
            )
        controller = EditController(
            store, surfaces=SurfaceRegistry(), router=FaultRouter()
        )
        parent = store.materialize(snapshot)
        precondition = controller.surface_precondition_sha(
            parent, "verification.rules"
        )
        new_doc = dict(current_doc)
        new_doc[GUARD_KEY] = []
        manifest = EditManifest(
            edit_id=MIGRATION_EDIT_ID,
            base_harness_sha=snapshot.harness_content_sha,
            target_pattern_id="aggregate_hides_per_series_harm",
            target_surface_id="verification.rules",
            operation=EditOperation.PATCH,
            surface_precondition={"kind": "SHA", "sha": precondition},
            dependency_precondition_shas={},
            minimal_patch={"value": new_doc},
            new_value=None,
            observable_applicability=None,
            predicted_agent_behavior_change=(
                "effective_view_unchanged_out_of_scope",
            ),
            predicted_data_effect=(
                "the verification document gains an empty scope_risk_guards "
                "list; no adoption decision changes",
            ),
            automatically_selected_risk_cases=(),
            falsification_condition=(
                "the guard-free replay of the four recorded task_C episodes "
                "no longer reproduces them digit-for-digit",
            ),
            patch_id=None,
        )
        receipt = controller.apply_to_fork(
            parent,
            _resolve_apply_manifest(manifest, snapshot),
            confirmed_cause="RISK_GAP",
        )
        updated = receipt.candidate_snapshot.snapshot
        after_doc = wvc._plain(updated.verification)
        added = sorted(set(after_doc) - set(current_doc))
        removed = sorted(set(current_doc) - set(after_doc))
        changed = sorted(
            key
            for key in set(current_doc) & set(after_doc)
            if current_doc[key] != after_doc[key]
        )
        proof = {
            "added_keys": added,
            "removed_keys": removed,
            "changed_values": changed,
            "installed_value": after_doc.get(GUARD_KEY),
            "ok": bool(
                added == [GUARD_KEY] and not removed and not changed
                and after_doc.get(GUARD_KEY) == []
            ),
        }
        if not proof["ok"]:
            raise SystemExit(
                "migration diff proof failed for %s: %s" % (slot_key, proof)
            )
        store.set_active(updated.runtime_bundle_sha)
        slot["snapshot"] = updated
        slot["runtime_bundle_sha"] = updated.runtime_bundle_sha
        slot["harness_content_sha"] = updated.harness_content_sha
        receipts[slot_key] = {
            "edit_id": receipt.edit_id,
            "confirmed_cause": receipt.confirmed_cause,
            "target_surface_id": receipt.target_surface_id,
            "parent_harness_content_sha": receipt.parent_harness_content_sha,
            "candidate_harness_content_sha": receipt.candidate_harness_content_sha,
            "source_surfaces_changed": list(receipt.source_surfaces_changed),
            "applied_edit_sha": receipt.applied_edit_sha,
            "single_key_diff_proof": proof,
        }
        print("MIGRATE %-20s +%s" % (slot_key, proof["added_keys"]), flush=True)
    return {
        "ran": True,
        "llm_calls": 0,
        "consumer_retrains": 0,
        "edit_id": MIGRATION_EDIT_ID,
        "surface_used": "verification.rules",
        "why_whole_document": (
            "the pointer surface verification.rules.scope_risk_guards "
            "requires its key to pre-exist (the controller's JSON-pointer "
            "write refuses a missing target), so the empty list is installed "
            "through the whole-document surface, once, deterministically; "
            "every later edit goes through the pointer surface"
        ),
        "receipts": receipts,
    }


def stage_gate_v2(
    *, cohort: Mapping[str, Any], episodes: Mapping[str, Any],
    stores: Mapping[str, Any], window: Mapping[str, Any], budget: Any,
) -> dict[str, Any]:
    """The non-regression gate, ahead of every LLM call.

    On the migrated (empty-list) state, the guard-free replay of the four
    recorded task_C episodes must reproduce them digit-for-digit on the
    repaired instrument.  The search objects and unguarded episodes are
    kept live: B3 reuses them and pays only for what the guard moves.
    """
    cells: dict[str, Any] = {}
    for variant in CONSUMERS:
        target = FC._target(variant)
        for arm in ARMS:
            slot_key = "%s_%s" % (arm.lower(), variant)
            recorded = episodes["records"][slot_key]
            snapshot = stores[slot_key]["snapshot"]
            slot = {"slot": slot_key, "_snapshot": snapshot}
            search = FC.FreshSearch(
                payload=cohort, consumer_variant=variant,
                support_origins=window["support_origins"],
                delayed_origins=window["delayed_origins"],
            )
            card = FC.SKILL_ID[variant] if arm == "A5" else None
            local = (recorded.get("retrieval") or {}).get(
                "expected_local_skill_id"
            )
            if str(recorded.get("mode")) == "DIRECT_RECALL":
                fresh = FC._direct_recall(
                    search=search, target=target, arm=arm, window=window,
                    slot=slot, expected_card=card, expected_local=str(local),
                    tag="task_C",
                )
                reused = bool(fresh.get("reuse_adopted"))
            else:
                fresh = _replay_full_price(
                    search=search, target=target, arm=arm, window=window,
                    slot=slot, expected_card=card, expected_local=local,
                    recorded=recorded,
                )
                reused = False
            reproduction = _reproduction(recorded, fresh)
            before_series = _per_series(search, fresh["final_plan"])
            budget.charge_retrains(int(search.retrains))
            cells[slot_key] = {
                "arm": arm,
                "consumer_variant": variant,
                "mode": recorded.get("mode"),
                "reused": reused,
                "reproduces": reproduction["reproduces"],
                "mismatched": reproduction["mismatched"],
                "reproduction": reproduction,
                "unguarded_replay": fresh,
                "before_series": before_series,
                "consumer_retrains": int(search.retrains),
                "_search": search,
                "_fresh": fresh,
            }
            print(
                "GATE %-20s reproduces=%s retrains=%d"
                % (slot_key, reproduction["reproduces"], int(search.retrains)),
                flush=True,
            )
    ok = all(bool(cell["reproduces"]) for cell in cells.values())
    return {
        "ran": True,
        "ok": ok,
        "state": (
            "post-migration: every active snapshot carries the empty %r "
            "list; no guard is enforced in this replay" % GUARD_KEY
        ),
        "cells": cells,
        "mismatched_cells": sorted(
            key for key, cell in cells.items() if not cell["reproduces"]
        ),
        "consumer_retrains": sum(
            int(cell["consumer_retrains"]) for cell in cells.values()
        ),
        "llm_calls": 0,
    }


def stage_b2_v2(
    *, stores: Mapping[str, Any], episodes: Mapping[str, Any],
    attribution: Mapping[str, Any], budget: Any,
) -> dict[str, Any]:
    """The Slow retry: one surface, two pinned backend rounds."""
    started = time.perf_counter()
    primary_slot = attribution["primary_cell"]
    cause = attribution["primary_cause"]
    if primary_slot is None or cause is None:
        return {
            "ran": False,
            "verdict": "ATTRIBUTION_LANDS_ELSEWHERE",
            "unregistered_in_v2": (
                "not a pre-registered v2 cell; reported as it stands"
            ),
            "reason": (
                "no pooled arm folded to a Scope/Risk face on the re-run "
                "attribution"
            ),
            "llm_calls": 0,
        }
    route = attribution["cells"][primary_slot][
        "with_per_series_risk_reading"
    ]["route_authorization"]
    snapshot = stores[primary_slot]["snapshot"]
    try:
        offered = _authorized_surfaces_v2(snapshot)
    except Exception as exc:  # noqa: BLE001
        return {
            "ran": False,
            "verdict": "SCHEMA_BLOCKED",
            "reason": (
                "the sutured surface does not resolve: %s: %s"
                % (type(exc).__name__, exc)
            ),
            "llm_calls": 0,
        }
    if not isinstance(offered[0]["current_value"], list):
        return {
            "ran": False,
            "verdict": "SCHEMA_BLOCKED",
            "reason": (
                "the active snapshot does not carry the pre-registered %r "
                "list; the migration did not land" % GUARD_KEY
            ),
            "llm_calls": 0,
        }
    public_input = {
        "stage_note": SLOW_NOTE_V2,
        "attributed_fault": attribution["cells"][primary_slot][
            "with_per_series_risk_reading"
        ]["attribution"],
        "attributed_from_episode": primary_slot,
        "route_authorization": dict(route),
        "authorized_surfaces": offered,
        "episode_evidence": _episode_evidence_v2(episodes, attribution["cells"]),
        "guard_grammar": GUARD_GRAMMAR_V2,
        "guard_contract": GUARD_CONTRACT_V2,
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
            "predicted_agent_behavior_vocabulary": _behavior_vocabulary(),
            "predicted_agent_behavior_patterns": _behavior_patterns(),
            "predicted_data_effect": "free text, what should happen to the data",
            "falsification_condition": "free text, what would show this edit wrong",
        },
    }
    view = resolve_harness_view(snapshot, {}, role="slow")
    record: dict[str, Any] = {
        "ran": True,
        "attributed_from_episode": primary_slot,
        "confirmed_cause": cause,
        "authorized_surfaces_offered": [NEW_SURFACE_V2],
        "public_input_sha256": canonical_sha256(wvc._plain(public_input)),
        "prompt_is_byte_identical_across_draws": (
            "one public_input object is canonicalized once and reused for "
            "every draw on both backends; only the model name differs"
        ),
        "harness_view_sha": view.effective_harness_view_sha,
        "role": "slow",
        "stage": "edit",
        "no_second_attempt": (
            "a compiler rejection stops the run; abstention alone is sampled "
            "at most twice per backend with a byte-identical prompt"
        ),
        "max_proposal_attempts_per_backend": MAX_PROPOSAL_ATTEMPTS_V2,
        "transport_failures_consume_no_sample": True,
        "backend_rounds_pinned": [
            {"label": row["label"], "model": row["model"]}
            for row in SLOW_BACKEND_ROUNDS_V2
        ],
    }
    validator = _make_proposal_validator_v2()
    rounds: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    proposal_round: str | None = None
    llm_calls = 0
    draws = 0
    consecutive_transport = 0
    stop_verdict: str | None = None
    stop_reason = ""
    for round_spec in SLOW_BACKEND_ROUNDS_V2:
        round_row: dict[str, Any] = {
            "label": round_spec["label"],
            "configured_model": round_spec["model"],
            "base_url": round_spec["base_url"],
            "why_this_round": round_spec["why"],
            "attempts": [],
        }
        rounds.append(round_row)
        samples = 0
        while samples < MAX_PROPOSAL_ATTEMPTS_V2:
            cap = int(budget.take(4))
            if cap < 1:
                stop_verdict = "STAGE_ERROR"
                stop_reason = "the LLM call budget is exhausted"
                break
            backend = _backend_factory_v2(round_spec["label"], cap)
            gateway = wvc.NoToolGateway({
                "protocol": PROTOCOL_VERSION_V2, "stage": "edit",
            })
            core = TTHAAgentCore(
                backend, gateway,
                model=round_spec["model"], base_url=round_spec["base_url"],
            )
            row: dict[str, Any] = {
                "sample": samples + 1,
                "backend": round_spec["label"],
                "configured_model": round_spec["model"],
            }
            try:
                result = core.run_stage(
                    role=AgentRole.SLOW,
                    stage="edit",
                    case_id="SSU2_%s" % primary_slot,
                    public_input=public_input,
                    harness_view=view,
                    output_schema_name="slow_scope_guard_v2",
                    output_schema=PROPOSAL_SCHEMA_V2,
                    source_snapshot_sha=view.effective_harness_view_sha,
                    validation_retries=wvc.VALIDATION_RETRIES,
                    post_validator=validator,
                )
            except AgentTransportError as exc:
                budget.spend(int(backend.calls))
                llm_calls += int(backend.calls)
                consecutive_transport += 1
                row.update({
                    "outcome": "TRANSPORT_FAILURE",
                    "consumes_sample": False,
                    "consecutive_transport_failures": consecutive_transport,
                    "transport_error": "%s: %s" % (type(exc).__name__, exc),
                    "llm_calls": int(backend.calls),
                })
                round_row["attempts"].append(row)
                print(
                    "B2 %s transport failure %d/3: %s"
                    % (round_spec["label"], consecutive_transport, exc),
                    flush=True,
                )
                if consecutive_transport >= MAX_CONSECUTIVE_TRANSPORT_FAILURES:
                    stop_verdict = "INCONCLUSIVE_TRANSPORT"
                    stop_reason = (
                        "three consecutive transport failures; the sampling "
                        "budget was not consumed by them"
                    )
                continue
            except Exception as exc:  # noqa: BLE001
                budget.spend(int(backend.calls))
                llm_calls += int(backend.calls)
                row.update({
                    "outcome": "STAGE_ERROR",
                    "consumes_sample": True,
                    "stage_error": "%s: %s" % (type(exc).__name__, exc),
                    "llm_calls": int(backend.calls),
                })
                round_row["attempts"].append(row)
                stop_verdict = "STAGE_ERROR"
                stop_reason = (
                    "%s: %s -- outside the pre-registered verdict set; "
                    "reported as it stands" % (type(exc).__name__, exc)
                )
                break
            consecutive_transport = 0
            budget.spend(int(backend.calls))
            llm_calls += int(backend.calls)
            samples += 1
            draws += 1
            candidate = dict(result.payload)
            row.update({
                "outcome": "PROPOSAL" if candidate else "NO_PROPOSAL",
                "consumes_sample": True,
                "llm_calls": int(backend.calls),
                "prompt_tokens": int(getattr(backend, "prompt_tokens", 0)),
                "completion_tokens": int(
                    getattr(backend, "completion_tokens", 0)
                ),
                "returned_models": sorted(
                    getattr(backend, "returned_models", ())
                ),
                "validation_retry_count": int(result.validation_retry_count),
                "validation_error_codes": list(result.validation_error_codes),
                "no_proposal_reason": result.no_proposal_reason,
            })
            round_row["attempts"].append(row)
            print(
                "B2 %s draw %d: %s%s (served by %s)"
                % (
                    round_spec["label"], samples, row["outcome"],
                    "" if candidate else " (%s)" % result.no_proposal_reason,
                    row["returned_models"],
                ),
                flush=True,
            )
            if candidate:
                payload = candidate
                proposal_round = round_spec["label"]
                break
        if payload or stop_verdict is not None:
            break
    record["rounds"] = rounds
    record["llm_calls"] = llm_calls
    record["wall_seconds"] = time.perf_counter() - started
    if stop_verdict is not None:
        record.update({"verdict": stop_verdict, "reason": stop_reason})
        return record
    if not payload:
        record.update({
            "verdict": "SLOW_DECLINES_PATCH_FINAL",
            "proposal": None,
            "reason": (
                "every draw on both backends returned the no_proposal "
                "envelope (%s).  The grammar is expressible and the evidence "
                "complete; a cross-backend decline closes the family on a "
                "credible negative."
                % ", ".join(
                    "%s/%s" % (item["backend"], item.get("no_proposal_reason"))
                    for round_row in rounds
                    for item in round_row["attempts"]
                    if item.get("outcome") == "NO_PROPOSAL"
                )
            ),
        })
        return record
    if draws > 1:
        record["obtained_after_abstention"] = (
            "draw %d; the earlier draws abstained.  This proposal is a "
            "second draw from a stochastic process and is labelled as such "
            "wherever it is reported." % draws
        )
    if proposal_round != SLOW_BACKEND_ROUNDS_V2[0]["label"]:
        record["backend_dependent"] = (
            "the proposal came from the %s round after the %s round "
            "abstained twice: the grammar repair was necessary but not "
            "sufficient for the weaker backend -- a real robustness finding "
            "about the Harness's serving model"
            % (proposal_round, SLOW_BACKEND_ROUNDS_V2[0]["label"])
        )
    record["proposal"] = wvc._plain(payload)
    record["guard"] = wvc._plain(payload["guard"])
    record["target_surface_id"] = str(payload["target_surface_id"])

    applications: dict[str, Any] = {}
    for slot in SLOTS:
        applications[slot] = _apply_patch(
            stores[slot], payload=payload, cause=cause,
        )
    primary = applications[primary_slot]
    if not primary.get("applied"):
        record.update({
            "verdict": "COMPILER_REJECTS",
            "applications": applications,
            "reason": primary.get("reason"),
            "public_feedback": primary.get("public_feedback"),
        })
        return record
    touched = sorted({
        surface
        for row in applications.values()
        for surface in (row.get("source_surfaces_changed") or ())
    })
    if touched != [NEW_SURFACE_V2]:
        record.update({
            "verdict": "PATCH_OVERREACH",
            "applications": applications,
            "reason": "the edit moved %s, not exactly %s" % (touched, NEW_SURFACE_V2),
        })
        return record
    record.update({
        "verdict": "PATCH_APPLIED",
        "applications": applications,
        "surfaces_changed": touched,
        "applied_to": sorted(
            slot for slot, row in applications.items() if row.get("applied")
        ),
        "not_applied_to": {
            slot: row.get("reason")
            for slot, row in applications.items() if not row.get("applied")
        },
    })
    return record


def _enforce_v2(
    *, search: Any, snapshot: Any, ladder_view: Mapping[str, Any], reused: bool,
) -> dict[str, Any]:
    """The #18 enforcement walk with evaluation delegated to the tracked gate.

    Guard discovery and guard evaluation are the sutured compiler's
    (``methods/ttha/harness/compiler.py``); the fallback walk stays here
    because it drives the search instrument.  The ladder is not modified
    and not re-ranked.
    """
    guards = harness_compiler.scope_risk_guards_of(snapshot)
    plan = dict(ladder_view["final_plan"])
    support = dict(ladder_view["support"])
    delayed = dict(ladder_view["delayed"])
    eval_count = len(search.eval_uids)
    out: dict[str, Any] = {
        "guards_declared": guards,
        "guard_count": len(guards),
        "gate_evaluator": (
            "methods/ttha/harness/compiler.py::evaluate_scope_risk_guards"
        ),
        "plan_before": dict(plan),
        "identity_never_vetoed": True,
        "changed": False,
        "extra_consumer_retrains": 0,
    }
    if not guards:
        out.update({
            "checked": False,
            "why": "the active snapshot declares no Scope/Risk guard",
            "plan_after": dict(plan), "support_after": support,
            "delayed_after": delayed,
        })
        return out
    if str(plan["program"]) == IDENTITY:
        out.update({
            "checked": False,
            "why": "identity is unfilterable and is never vetoed",
            "plan_after": dict(plan), "support_after": support,
            "delayed_after": delayed,
        })
        return out
    before = search.retrains
    first = harness_compiler.evaluate_scope_risk_guards(
        snapshot=snapshot, plan=plan, support=support, delayed=delayed,
        eval_count=eval_count, reused=reused,
    )
    out["check_on_adopted_plan"] = first
    if not first["vetoed"]:
        out.update({
            "checked": True,
            "plan_after": dict(plan), "support_after": support,
            "delayed_after": delayed,
            "why": (
                "a guard fired but only asked for a record"
                if first["any_fired"] else "no guard fired"
            ),
        })
        return out

    winner = ladder_view.get("support_winner")
    winner_delayed = ladder_view.get("support_winner_full_batch_delayed")
    fallback = {"program": IDENTITY, "excluded_series": []}
    fallback_source = "identity"
    if (
        winner is not None
        and winner_delayed is not None
        and float(winner_delayed) > 0.0
        and not (
            str(winner) == str(plan["program"]) and not plan["excluded_series"]
        )
    ):
        fallback = {"program": str(winner), "excluded_series": []}
        fallback_source = "the ladder's Support winner, full batch"
    fallback_delayed = dict(
        search.delayed_gate(fallback["program"], list(fallback["excluded_series"]))
    )
    fallback_support = dict(
        search.support_of_plan(
            fallback["program"], list(fallback["excluded_series"])
        )
    )
    second = None
    if str(fallback["program"]) != IDENTITY:
        second = harness_compiler.evaluate_scope_risk_guards(
            snapshot=snapshot, plan=fallback, support=fallback_support,
            delayed=fallback_delayed, eval_count=eval_count, reused=reused,
        )
        if second["vetoed"]:
            fallback = {"program": IDENTITY, "excluded_series": []}
            fallback_source = (
                "identity: the fallback candidate fired the same guard"
            )
            fallback_delayed = dict(search.delayed_gate(IDENTITY, []))
            fallback_support = dict(search.support_of_plan(IDENTITY, []))
    out.update({
        "checked": True,
        "check_on_fallback": second,
        "fallback_source": fallback_source,
        "plan_after": dict(fallback),
        "support_after": fallback_support,
        "delayed_after": fallback_delayed,
        "changed": True,
        "extra_consumer_retrains": int(search.retrains - before),
    })
    return out


def stage_b3_v2(
    *, gate: Mapping[str, Any], episodes: Mapping[str, Any],
    stores: Mapping[str, Any], window: Mapping[str, Any], budget: Any,
) -> dict[str, Any]:
    """Replay with the patch live, on the gate's own unguarded searches.

    The unguarded half of every cell was measured once, in the gate; only
    the plans the guard moves cost fresh Consumer retrains.
    """
    cells: dict[str, Any] = {}
    for slot_key, gate_cell in gate["cells"].items():
        search = gate_cell["_search"]
        fresh = gate_cell["_fresh"]
        reused = bool(gate_cell["reused"])
        recorded = episodes["records"][slot_key]
        snapshot = stores[slot_key]["snapshot"]
        ladder_view = {
            "final_plan": dict(fresh["final_plan"]),
            "support": dict(fresh["support"] or {}),
            "delayed": dict(fresh["delayed"] or {}),
            "support_winner": (fresh.get("adoption_ladder") or {}).get(
                "support_winner"
            ),
            "support_winner_full_batch_delayed": (
                fresh.get("adoption_ladder") or {}
            ).get("support_winner_full_batch_delayed"),
        }
        guard = _enforce_v2(
            search=search, snapshot=snapshot, ladder_view=ladder_view,
            reused=reused,
        )
        delta = int(search.retrains) - int(gate_cell["consumer_retrains"])
        if delta:
            budget.charge_retrains(delta)
        before_series = gate_cell["before_series"]
        after_series = (
            dict(before_series)
            if dict(guard["plan_after"]) == dict(fresh["final_plan"])
            else _per_series(search, guard["plan_after"])
        )
        variant = str(gate_cell["consumer_variant"])
        cells[slot_key] = {
            "arm": gate_cell["arm"],
            "consumer_variant": variant,
            "role": (
                "the failing cell" if variant == POOLED
                else "regression check, expected not to move"
            ),
            "reuse_path": reused,
            "unguarded_replay": fresh,
            "reproduction_of_the_recorded_episode": gate_cell["reproduction"],
            "guard": guard,
            "before": {
                "plan": dict(fresh["final_plan"]),
                "support_aggregate_gain": (fresh.get("support") or {}).get(
                    "aggregate_gain"
                ),
                "delayed_aggregate_gain": (fresh.get("delayed") or {}).get(
                    "aggregate_gain"
                ),
                "per_eval_series_delayed_gain": before_series,
                "harmed_eval_series": list(
                    (fresh.get("delayed") or {}).get("harmed_eval_series") or ()
                ),
                "harmed_eval_series_total_harm": (
                    fresh.get("delayed") or {}
                ).get("harmed_eval_series_total_harm"),
            },
            "after": {
                "plan": dict(guard["plan_after"]),
                "support_aggregate_gain": (
                    guard.get("support_after") or {}
                ).get("aggregate_gain"),
                "delayed_aggregate_gain": (
                    guard.get("delayed_after") or {}
                ).get("aggregate_gain"),
                "per_eval_series_delayed_gain": after_series,
                "harmed_eval_series": list(
                    (guard.get("delayed_after") or {}).get(
                        "harmed_eval_series"
                    ) or ()
                ),
                "harmed_eval_series_total_harm": (
                    guard.get("delayed_after") or {}
                ).get("harmed_eval_series_total_harm"),
            },
            "behaviour_changed": bool(guard.get("changed")),
            "consumer_retrains": int(search.retrains),
            "retrain_delta_over_the_gate": delta,
            "llm_calls": 0,
            "instrument": search.accounting(),
        }
        print(
            "B3 %-20s %s -> %s | delayed %+.6f -> %+.6f | %s %+.6f -> %+.6f "
            "| +%d retrains"
            % (
                slot_key, _plan_label(fresh["final_plan"]),
                _plan_label(guard["plan_after"]),
                float((fresh.get("delayed") or {}).get("aggregate_gain") or 0.0),
                float(
                    (guard.get("delayed_after") or {}).get("aggregate_gain")
                    or 0.0
                ),
                THE_HARMED_SERIES,
                before_series.get(THE_HARMED_SERIES, float("nan")),
                after_series.get(THE_HARMED_SERIES, float("nan")),
                delta,
            ),
            flush=True,
        )
    return {
        "ran": True,
        "development_level": True,
        "not_fresh": (
            "the pooled task_C window was opened by fresh_confirmation_v1; "
            "this replay reuses it as development evidence and claims nothing "
            "about held-out performance"
        ),
        "unguarded_half_reused_from": "non_regression_gate",
        "window": {
            key: value for key, value in window.items()
            if not str(key).startswith("reference_")
        },
        "cells": cells,
        "consumer_retrains": sum(
            int(row["consumer_retrains"]) for row in cells.values()
        ),
        "retrain_delta_over_the_gate": sum(
            int(row["retrain_delta_over_the_gate"]) for row in cells.values()
        ),
        "llm_calls": 0,
    }


# --------------------------------------------------------------- the verdict
def _verdict(
    *, attribution: Mapping[str, Any], patch: Mapping[str, Any],
    replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if attribution["primary_cell"] is None:
        return {
            "verdict": "ATTRIBUTION_LANDS_ELSEWHERE",
            "reason": (
                "neither pooled arm folded to a Scope/Risk face; recorded as "
                "it stands and not corrected"
            ),
        }
    if patch.get("verdict") in (
        "COMPILER_REJECTS", "PATCH_OVERREACH", "SCHEMA_BLOCKED",
        "SLOW_ABSTAINS", "ATTRIBUTION_LANDS_ELSEWHERE",
    ):
        return {
            "verdict": str(patch["verdict"]),
            "reason": str(patch.get("reason") or patch.get("stage_error") or ""),
        }
    if replay is None:
        return {"verdict": "SCHEMA_BLOCKED", "reason": "the replay never ran"}

    pooled = {
        key: cell for key, cell in replay["cells"].items()
        if cell["consumer_variant"] == POOLED
    }
    channel = {
        key: cell for key, cell in replay["cells"].items()
        if cell["consumer_variant"] == PER_CHANNEL
    }
    exposed = {
        key: float(
            cell["before"]["per_eval_series_delayed_gain"].get(
                THE_HARMED_SERIES, 0.0
            )
        )
        for key, cell in pooled.items()
    }
    after = {
        key: float(
            cell["after"]["per_eval_series_delayed_gain"].get(
                THE_HARMED_SERIES, 0.0
            )
        )
        for key, cell in pooled.items()
    }
    was_hurt = {key for key, value in exposed.items() if value < HARM_THRESHOLD}
    still_hurt = sorted(
        key for key in was_hurt if after[key] < HARM_THRESHOLD
    )
    aggregates = {
        key: float(cell["after"]["delayed_aggregate_gain"] or 0.0)
        for key, cell in pooled.items()
    }
    negative = sorted(key for key, value in aggregates.items() if value < 0.0)
    moved = sorted(key for key, cell in pooled.items() if cell["behaviour_changed"])
    regression = sorted(
        key for key, cell in channel.items()
        if cell["behaviour_changed"]
        or cell["before"]["plan"] != cell["after"]["plan"]
        or cell["before"]["delayed_aggregate_gain"] != cell["after"][
            "delayed_aggregate_gain"
        ]
    )
    new_harm = {
        key: sorted(
            set(cell["after"]["harmed_eval_series"])
            - set(cell["before"]["harmed_eval_series"])
        )
        for key, cell in replay["cells"].items()
    }
    detail = {
        "series_watched": THE_HARMED_SERIES,
        "harm_line": HARM_THRESHOLD,
        "before_by_cell": exposed,
        "after_by_cell": after,
        "cells_where_it_was_hurt": sorted(was_hurt),
        "cells_where_it_is_still_hurt": still_hurt,
        "aggregate_delayed_after": aggregates,
        "cells_with_negative_aggregate": negative,
        "pooled_cells_that_moved": moved,
        "per_channel_cells_that_moved": regression,
        "new_harmed_series": {k: v for k, v in new_harm.items() if v},
    }
    if not moved:
        return {
            "verdict": "REPLAY_NO_CHANGE",
            "reason": (
                "the patch is live in the active snapshot and no pooled "
                "decision moved"
            ),
            "detail": detail,
        }
    if negative:
        return {
            "verdict": "REPLAY_AGGREGATE_COLLAPSE",
            "reason": (
                "the guard drove the aggregate delayed reading negative in %s"
                % negative
            ),
            "detail": detail,
        }
    if still_hurt:
        return {
            "verdict": "REPLAY_NO_CHANGE",
            "partial": True,
            "reason": (
                "the gap did not close: %s moved but %s still crosses the "
                "%+.3f line in %s.  Recorded under the pre-registered "
                "REPLAY_NO_CHANGE label because the pre-registration has no "
                "partial-fix outcome; the movement is in the table."
                % (moved, THE_HARMED_SERIES, HARM_THRESHOLD, still_hurt)
            ),
            "detail": detail,
        }
    if regression:
        return {
            "verdict": "PATCH_OVERREACH",
            "reason": (
                "the per_channel regression cells moved: %s" % regression
            ),
            "detail": detail,
        }
    return {
        "verdict": "SLOW_CLOSES_SCOPE_GAP",
        "reason": (
            "attribution landed on a Scope/Risk face, one surface was "
            "patched and compiled, %s no longer crosses %+.3f in either "
            "pooled arm, both aggregates stayed at or above zero, and "
            "per_channel did not move"
            % (THE_HARMED_SERIES, HARM_THRESHOLD)
        ),
        "detail": detail,
    }


def _verdict_v2(
    *, attribution: Mapping[str, Any], gate: Mapping[str, Any],
    patch: Mapping[str, Any], replay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """The #19 terminal verdict set."""
    if attribution["primary_cell"] is None:
        return {
            "verdict": "ATTRIBUTION_LANDS_ELSEWHERE",
            "unregistered_in_v2": True,
            "reason": (
                "the re-run attribution no longer lands on a Scope/Risk "
                "face; outside the pre-registered v2 set, reported as it "
                "stands"
            ),
        }
    if not gate["ok"]:
        return {
            "verdict": "INSTRUMENT_DRIFT",
            "reason": (
                "the post-migration guard-free replay did not reproduce the "
                "recorded episodes digit-for-digit in %s"
                % gate["mismatched_cells"]
            ),
        }
    if patch.get("verdict") in (
        "SLOW_DECLINES_PATCH_FINAL", "COMPILER_REJECTS", "PATCH_OVERREACH",
        "SCHEMA_BLOCKED", "INCONCLUSIVE_TRANSPORT", "STAGE_ERROR",
        "ATTRIBUTION_LANDS_ELSEWHERE",
    ):
        out = {
            "verdict": str(patch["verdict"]),
            "reason": str(patch.get("reason") or ""),
        }
        if patch.get("backend_dependent"):
            out["backend_dependent"] = str(patch["backend_dependent"])
        return out
    if replay is None:
        return {"verdict": "SCHEMA_BLOCKED", "reason": "the replay never ran"}

    pooled = {
        key: cell for key, cell in replay["cells"].items()
        if cell["consumer_variant"] == POOLED
    }
    channel = {
        key: cell for key, cell in replay["cells"].items()
        if cell["consumer_variant"] == PER_CHANNEL
    }
    exposed = {
        key: float(
            cell["before"]["per_eval_series_delayed_gain"].get(
                THE_HARMED_SERIES, 0.0
            )
        )
        for key, cell in pooled.items()
    }
    after = {
        key: float(
            cell["after"]["per_eval_series_delayed_gain"].get(
                THE_HARMED_SERIES, 0.0
            )
        )
        for key, cell in pooled.items()
    }
    was_hurt = {key for key, value in exposed.items() if value < HARM_THRESHOLD}
    still_hurt = sorted(
        key for key in was_hurt if after[key] < HARM_THRESHOLD
    )
    aggregates = {
        key: float(cell["after"]["delayed_aggregate_gain"] or 0.0)
        for key, cell in pooled.items()
    }
    negative = sorted(key for key, value in aggregates.items() if value < 0.0)
    moved = sorted(key for key, cell in pooled.items() if cell["behaviour_changed"])
    regression = sorted(
        key for key, cell in channel.items()
        if cell["behaviour_changed"]
        or cell["before"]["plan"] != cell["after"]["plan"]
        or cell["before"]["delayed_aggregate_gain"] != cell["after"][
            "delayed_aggregate_gain"
        ]
    )
    new_harm = {
        key: sorted(
            set(cell["after"]["harmed_eval_series"])
            - set(cell["before"]["harmed_eval_series"])
        )
        for key, cell in replay["cells"].items()
    }
    identity_ended = sorted(
        key for key in moved
        if str(pooled[key]["after"]["plan"]["program"]) == IDENTITY
    )
    forgone = {
        key: {
            "aggregate_delayed_before": float(
                pooled[key]["before"]["delayed_aggregate_gain"] or 0.0
            ),
            "aggregate_delayed_after": aggregates[key],
        }
        for key in identity_ended
    }
    detail = {
        "series_watched": THE_HARMED_SERIES,
        "harm_line": HARM_THRESHOLD,
        "before_by_cell": exposed,
        "after_by_cell": after,
        "cells_where_it_was_hurt": sorted(was_hurt),
        "cells_where_it_is_still_hurt": still_hurt,
        "aggregate_delayed_after": aggregates,
        "cells_with_negative_aggregate": negative,
        "pooled_cells_that_moved": moved,
        "pooled_cells_that_ended_on_identity": identity_ended,
        "forgone_aggregate_gain_booked": forgone,
        "per_channel_cells_that_moved": regression,
        "new_harmed_series": {k: v for k, v in new_harm.items() if v},
    }
    if patch.get("backend_dependent"):
        detail["backend_dependent"] = str(patch["backend_dependent"])
    if not moved:
        return {
            "verdict": "REPLAY_NO_CHANGE",
            "reason": (
                "the guard is live in the active snapshot and never vetoed: "
                "no pooled decision moved (not triggered, or RECORD_ONLY)"
            ),
            "detail": detail,
        }
    if negative:
        return {
            "verdict": "REPLAY_AGGREGATE_COLLAPSE",
            "reason": (
                "the guard drove the aggregate delayed reading negative in %s"
                % negative
            ),
            "detail": detail,
        }
    if still_hurt:
        return {
            "verdict": "REPLAY_HARM_PERSISTS",
            "reason": (
                "%s moved but %s still crosses the %+0.3f line in %s"
                % (moved, THE_HARMED_SERIES, HARM_THRESHOLD, still_hurt)
            ),
            "detail": detail,
        }
    if regression:
        return {
            "verdict": "PATCH_OVERREACH",
            "reason": (
                "the per_channel regression cells moved: %s" % regression
            ),
            "detail": detail,
        }
    if len(identity_ended) == len(moved):
        return {
            "verdict": "SLOW_CLOSES_SCOPE_GAP_BY_VETO",
            "reason": (
                "every moved pooled cell vetoed to identity: %s no longer "
                "crosses %+0.3f anywhere and per_channel did not move; the "
                "forgone +0.029688 aggregate per arm is booked in the detail"
                % (THE_HARMED_SERIES, HARM_THRESHOLD)
            ),
            "detail": detail,
        }
    return {
        "verdict": "SLOW_CLOSES_SCOPE_GAP_BY_RESCOPE",
        "reason": (
            "%s no longer crosses %+0.3f in either pooled arm, every pooled "
            "aggregate stayed positive, and per_channel did not move"
            % (THE_HARMED_SERIES, HARM_THRESHOLD)
        ),
        "detail": detail,
    }


# ------------------------------------------------------------------- the run
def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public(nested) for key, nested in value.items()
            if not str(key).startswith("_")
            and str(key) not in ("store", "snapshot")
        }
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def run(*, attribution_only: bool = False) -> int:
    started = time.perf_counter()
    before = _freeze()
    budget = Budget(LLM_CALL_BUDGET_TOTAL, RETRAIN_BUDGET)
    episodes = _load_episodes()
    stores = _open_stores()
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "role": (
            "close the last link of the loop: attribute the #17 Scope gap, let "
            "Slow patch one Scope/Risk surface, and replay the same window"
        ),
        "pre_registered": PRE_REGISTERED,
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "source_evidence": {
            "artifact": _repo_rel(SOURCE_ARTIFACT),
            "artifact_sha256": episodes["artifact_sha256"],
            "read_only": True,
            "window": episodes["window"],
            "locked_roster": episodes["locked_roster"],
        },
        "stores": {
            slot: {
                key: value for key, value in row.items()
                if key not in ("store", "snapshot")
            }
            for slot, row in stores.items()
        },
        "llm_call_budget": LLM_CALL_BUDGET_TOTAL,
        "retrain_budget": RETRAIN_BUDGET,
    }

    b1 = stage_b1(episodes["records"], stores)
    payload["b1_attribution"] = _public(b1)
    payload["guard_after_b1"] = _guard(before, "B2")
    print(
        "B1 primary=%s cause=%s" % (b1["primary_cell"], b1["primary_cause"]),
        flush=True,
    )
    for slot, cell in b1["cells"].items():
        print(
            "B1 %-12s with-per-series=%s / aggregate-only=%s"
            % (
                slot,
                cell["with_per_series_risk_reading"]["attribution"]["cause_code"],
                cell["aggregate_only_control"]["attribution"]["cause_code"],
            ),
            flush=True,
        )
    if attribution_only:
        payload.update({
            "llm_call_count": 0, "consumer_retrains_total": 0,
            "overall_verdict": "B1_ONLY",
            "overall_verdict_reason": "the run was asked to stop after attribution",
            "wall_seconds": time.perf_counter() - started,
            "frozen_surface_after": _verify(before),
        })
        return _write(payload, dry=True)

    b2 = stage_b2(
        stores=stores, episodes=episodes, attribution=b1, budget=budget,
    )
    payload["b2_patch"] = _public(b2)
    payload["guard_after_b2"] = _guard(before, "B3")
    print(
        "B2 %s surface=%s llm=%d"
        % (b2.get("verdict"), b2.get("target_surface_id"), b2.get("llm_calls", 0)),
        flush=True,
    )
    if b2.get("verdict") != "PATCH_APPLIED":
        overall = _verdict(attribution=b1, patch=b2, replay=None)
        payload.update({
            "llm_call_count": budget.llm_used,
            "consumer_retrains_total": budget.retrains_charged,
            "overall_verdict": overall["verdict"],
            "overall_verdict_reason": overall["reason"],
            "wall_seconds": time.perf_counter() - started,
            "frozen_surface_after": _verify(before),
        })
        return _write(payload)

    cohort = _confirmation_payload(episodes["locked_roster"])
    b3 = stage_b3(
        payload=cohort, episodes=episodes, stores=stores,
        window=episodes["window"], budget=budget,
    )
    payload["b3_replay"] = _public(b3)
    payload["guard_after_b3"] = _guard(before, "B4")
    b4 = stage_b4(b3)
    payload["b4_experience"] = _public(b4)

    overall = _verdict(attribution=b1, patch=b2, replay=b3)
    payload.update({
        "overall_verdict": overall["verdict"],
        "overall_verdict_reason": overall["reason"],
        "overall_detail": _public(overall.get("detail") or {}),
        "llm_call_count": budget.llm_used,
        "consumer_retrains_total": budget.retrains_charged,
        "exposure": {
            "windows_read": [
                "2024 development training anchors (indices 120-900)",
                "task_C confirmation window [9864, 10152], opened by #17",
            ],
            "beyond_17520": "SEALED, not read",
            "unopened_windows_read": "none",
            "fresh_claim": (
                "none: this replay is development-level on an already-exposed "
                "window"
            ),
        },
        "wall_seconds": time.perf_counter() - started,
        "frozen_surface_after": _verify(before),
    })
    if not payload["frozen_surface_after"]["ok"]:
        payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
        payload["overall_verdict_reason"] = (
            "the frozen surface moved during the run; the reading is void"
        )
    return _write(payload)


# ---------------------------------------------------------------- the report
def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    try:
        return "%+.6f" % float(value)
    except (TypeError, ValueError):
        return str(value)


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Slow Scope/Risk self-update",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        "The closing run adopted `outlier_mad` on pooled task_C in both arms: "
        "aggregate delayed +0.029688, evaluation series %s down 0.125557. "
        "This slice runs Runtime attribution over that episode, lets the Slow "
        "Agent patch one authorized Scope/Risk surface through the "
        "deterministic EditController, and replays the same already-exposed "
        "window. The replay is development-level and claims nothing about "
        "held-out performance." % THE_HARMED_SERIES,
        "",
        "## B1 -- where the fold puts the fault",
        "",
        "| episode | with the per-series risk reading | aggregate only |",
        "| --- | --- | --- |",
    ]
    for slot, cell in (payload.get("b1_attribution") or {}).get("cells", {}).items():
        lines.append(
            "| `%s` | `%s` at %s | `%s` |" % (
                slot,
                cell["with_per_series_risk_reading"]["attribution"]["cause_code"],
                cell["with_per_series_risk_reading"]["attribution"]["first_stage"],
                cell["aggregate_only_control"]["attribution"]["cause_code"],
            )
        )
    b1 = payload.get("b1_attribution") or {}
    lines.extend([
        "",
        "Primary cell `%s`, cause `%s`. The fold, the route table and the "
        "0.005 risk epsilon are the old line's, unchanged."
        % (b1.get("primary_cell"), b1.get("primary_cause")),
        "",
    ])
    b2 = payload.get("b2_patch") or {}
    lines.extend(["## B2 -- the patch", ""])
    if b2.get("guard"):
        guard = b2["guard"]
        lines.extend([
            "- Surface: `%s` (offered: %s)." % (
                b2.get("target_surface_id"),
                ", ".join("`%s`" % item for item in b2.get(
                    "authorized_surfaces_offered", ()
                )),
            ),
            "- Guard: `%s` -- %s `%s` %s %s on the %s window, applies to %s." % (
                guard.get("guard_id"), guard.get("statistic"),
                guard.get("comparator"), _fmt(guard.get("threshold")),
                "-> " + str(guard.get("action")), guard.get("window"),
                guard.get("applies_to"),
            ),
            "- Slow's rationale: %s" % guard.get("rationale"),
            "- Verdict: `%s`; surfaces changed %s; applied to %s." % (
                b2.get("verdict"), b2.get("surfaces_changed"),
                b2.get("applied_to"),
            ),
            "",
        ])
    else:
        lines.extend([
            "- Verdict: `%s`." % b2.get("verdict"),
            "- %s" % (b2.get("reason") or b2.get("stage_error") or ""),
            "",
        ])
    b3 = payload.get("b3_replay") or {}
    if b3.get("cells"):
        lines.extend([
            "## B3 -- replay, before and after",
            "",
            "| cell | plan before | plan after | delayed before | delayed after "
            "| %s before | %s after | retrains |" % (
                THE_HARMED_SERIES, THE_HARMED_SERIES
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for slot, cell in b3["cells"].items():
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %d |" % (
                    slot,
                    "`%s`" % _plan_label(cell["before"]["plan"]),
                    "`%s`" % _plan_label(cell["after"]["plan"]),
                    _fmt(cell["before"]["delayed_aggregate_gain"]),
                    _fmt(cell["after"]["delayed_aggregate_gain"]),
                    _fmt(cell["before"]["per_eval_series_delayed_gain"].get(
                        THE_HARMED_SERIES
                    )),
                    _fmt(cell["after"]["per_eval_series_delayed_gain"].get(
                        THE_HARMED_SERIES
                    )),
                    int(cell["consumer_retrains"]),
                )
            )
        lines.extend(["", "### Per evaluation series, delayed gain", ""])
        eval_uids = sorted({
            uid
            for cell in b3["cells"].values()
            for uid in cell["before"]["per_eval_series_delayed_gain"]
        })
        lines.append("| cell | " + " | ".join("`%s`" % uid for uid in eval_uids) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in eval_uids) + " |")
        for slot, cell in b3["cells"].items():
            for phase in ("before", "after"):
                row = cell[phase]["per_eval_series_delayed_gain"]
                lines.append(
                    "| `%s` %s | %s |" % (
                        slot, phase,
                        " | ".join(_fmt(row.get(uid)) for uid in eval_uids),
                    )
                )
        broken = [
            slot for slot, cell in b3["cells"].items()
            if not cell["reproduction_of_the_recorded_episode"]["reproduces"]
        ]
        lines.extend([
            "",
            "Unguarded replay reproduces the recorded episode in %d of %d "
            "cells%s." % (
                len(b3["cells"]) - len(broken), len(b3["cells"]),
                "" if not broken else " (mismatched: %s)" % broken,
            ),
            "",
        ])
    detail = payload.get("overall_detail") or {}
    if detail:
        lines.extend([
            "## The watched series",
            "",
            "- Hurt before in: %s." % (detail.get("cells_where_it_was_hurt") or "none"),
            "- Still hurt after in: %s." % (
                detail.get("cells_where_it_is_still_hurt") or "none"
            ),
            "- Pooled cells that moved: %s." % (
                detail.get("pooled_cells_that_moved") or "none"
            ),
            "- per_channel cells that moved: %s." % (
                detail.get("per_channel_cells_that_moved") or "none"
            ),
            "- New harmed series anywhere: %s." % (
                detail.get("new_harmed_series") or "none"
            ),
            "",
        ])
    b4 = payload.get("b4_experience") or {}
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
        "- Experience rows written (provenance `slow_scope_update`): %s." % (
            b4.get("row_count")
        ),
        "- Frozen surface: %s files, drift %s." % (
            after.get("files"), after.get("drift")
        ),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def _write(
    payload: Mapping[str, Any], *, dry: bool = False,
    out_json: Path = OUT_JSON, out_md: Path = OUT_MD,
    markdown: Any = _markdown,
) -> int:
    body = _public(payload)
    if dry:
        print(json.dumps(body, indent=2, ensure_ascii=False, default=str)[:8000])
    else:
        E2.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(body, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8", newline="\n",
        )
        out_md.write_text(markdown(body), encoding="utf-8", newline="\n")
        print("wrote", out_json, flush=True)
        print("wrote", out_md, flush=True)
    print("overall", body.get("overall_verdict"), flush=True)
    print("llm_calls", body.get("llm_call_count", 0), flush=True)
    print("retrains", body.get("consumer_retrains_total", 0), flush=True)
    return 0 if body.get("overall_verdict") in (
        "SLOW_CLOSES_SCOPE_GAP", "SLOW_CLOSES_SCOPE_GAP_BY_VETO",
        "SLOW_CLOSES_SCOPE_GAP_BY_RESCOPE", "B1_ONLY",
    ) else 1


def _markdown_v2(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Slow Scope/Risk self-update v2 -- the EDIT_SURFACE_DEFECT repair",
        "",
        "**Overall: `%s`** -- %s" % (
            payload.get("overall_verdict"),
            payload.get("overall_verdict_reason", ""),
        ),
        "",
        "#18 showed the Scope/Risk edit surface itself was defective: the "
        "guard evaluation context saw only the aggregate (projection), the "
        "`scope_risk_guards` key had no tracked reader (placebo), and the "
        "offered surfaces were never the minimal one-entry edit.  This slice "
        "sutures the surface (O1 passthrough, registry entry, compiler gate, "
        "pre-registered empty list), proves the migrated state reproduces the "
        "recorded episodes digit-for-digit before any LLM call, and retries "
        "the Slow draw on the repaired surface. The replay is "
        "development-level and claims nothing about held-out performance.",
        "",
        "## The suture (tracked files, both sha256 sides)",
        "",
        "| file | role | before | after |",
        "| --- | --- | --- | --- |",
    ]
    for row in (payload.get("sutured_files") or ()):
        lines.append(
            "| `%s` | %s | `%s...` | `%s...` |" % (
                row["path"], row["role"],
                str(row.get("sha256_before_fix"))[:12],
                str(row.get("sha256_after_fix"))[:12],
            )
        )
    migration = payload.get("migration") or {}
    lines.extend(["", "## Migration (0 LLM)", ""])
    if migration.get("receipts"):
        lines.extend([
            "| store | surface | key diff | receipt |",
            "| --- | --- | --- | --- |",
        ])
        for slot, receipt in migration["receipts"].items():
            proof = receipt["single_key_diff_proof"]
            lines.append(
                "| `%s` | `%s` | +%s / -%s / ~%s | `%s...` |" % (
                    slot, receipt["target_surface_id"],
                    proof["added_keys"], proof["removed_keys"],
                    proof["changed_values"],
                    str(receipt["applied_edit_sha"])[:12],
                )
            )
        lines.append("")
    gate = payload.get("non_regression_gate") or {}
    if gate.get("cells"):
        lines.extend([
            "## Non-regression gate (0 LLM, before any Slow call)",
            "",
            "State: %s." % gate.get("state"),
            "",
            "| cell | mode | reproduces digit-for-digit | retrains |",
            "| --- | --- | --- | ---: |",
        ])
        for slot, cell in gate["cells"].items():
            lines.append(
                "| `%s` | %s | %s%s | %d |" % (
                    slot, cell.get("mode"), bool(cell.get("reproduces")),
                    "" if cell.get("reproduces")
                    else " (mismatched: %s)" % cell.get("mismatched"),
                    int(cell["consumer_retrains"]),
                )
            )
        lines.append("")
    b1 = payload.get("b1_attribution") or {}
    if b1.get("cells"):
        lines.extend([
            "## B1 -- where the fold puts the fault (re-run, deterministic)",
            "",
            "| episode | with the per-series risk reading | aggregate only |",
            "| --- | --- | --- |",
        ])
        for slot, cell in b1["cells"].items():
            lines.append(
                "| `%s` | `%s` at %s | `%s` |" % (
                    slot,
                    cell["with_per_series_risk_reading"]["attribution"]["cause_code"],
                    cell["with_per_series_risk_reading"]["attribution"]["first_stage"],
                    cell["aggregate_only_control"]["attribution"]["cause_code"],
                )
            )
        lines.extend([
            "",
            "Primary cell `%s`, cause `%s`."
            % (b1.get("primary_cell"), b1.get("primary_cause")),
            "",
        ])
    b2 = payload.get("b2_patch") or {}
    lines.extend(["## B2 -- the Slow retry on the sutured surface", ""])
    for round_row in b2.get("rounds") or ():
        lines.append(
            "### Round %s (configured `%s`)" % (
                round_row.get("label"), round_row.get("configured_model")
            )
        )
        lines.extend([
            "",
            "| sample | outcome | reason | served by | llm calls |",
            "| ---: | --- | --- | --- | ---: |",
        ])
        for attempt in round_row.get("attempts") or ():
            lines.append(
                "| %s | `%s` | %s | %s | %d |" % (
                    attempt.get("sample"), attempt.get("outcome"),
                    attempt.get("no_proposal_reason")
                    or attempt.get("transport_error")
                    or attempt.get("stage_error") or "--",
                    attempt.get("returned_models") or "--",
                    int(attempt.get("llm_calls") or 0),
                )
            )
            if not attempt.get("consumes_sample", True):
                lines[-1] += " (no sample consumed)"
        lines.append("")
    if b2.get("guard"):
        guard = b2["guard"]
        lines.extend([
            "- Surface: `%s`." % b2.get("target_surface_id"),
            "- Guard: `%s` -- %s `%s` %s -> %s on the %s window, applies to %s." % (
                guard.get("guard_id"), guard.get("statistic"),
                guard.get("comparator"), _fmt(guard.get("threshold")),
                guard.get("action"), guard.get("window"),
                guard.get("applies_to"),
            ),
            "- Slow's rationale: %s" % guard.get("rationale"),
            "- Verdict: `%s`; surfaces changed %s."
            % (b2.get("verdict"), b2.get("surfaces_changed")),
        ])
        if b2.get("obtained_after_abstention"):
            lines.append("- %s" % b2["obtained_after_abstention"])
        if b2.get("backend_dependent"):
            lines.append("- **Backend-dependent:** %s" % b2["backend_dependent"])
        lines.append("")
    else:
        lines.extend([
            "- Verdict: `%s`." % b2.get("verdict"),
            "- %s" % (b2.get("reason") or ""),
            "",
        ])
    b3 = payload.get("b3_replay") or {}
    if b3.get("cells"):
        lines.extend([
            "## B3 -- replay, before and after",
            "",
            "The unguarded half is the gate's, paid once; only plans the "
            "guard moves cost fresh retrains.",
            "",
            "| cell | plan before | plan after | delayed before | delayed after "
            "| %s before | %s after | +retrains |" % (
                THE_HARMED_SERIES, THE_HARMED_SERIES
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for slot, cell in b3["cells"].items():
            lines.append(
                "| `%s` | %s | %s | %s | %s | %s | %s | %d |" % (
                    slot,
                    "`%s`" % _plan_label(cell["before"]["plan"]),
                    "`%s`" % _plan_label(cell["after"]["plan"]),
                    _fmt(cell["before"]["delayed_aggregate_gain"]),
                    _fmt(cell["after"]["delayed_aggregate_gain"]),
                    _fmt(cell["before"]["per_eval_series_delayed_gain"].get(
                        THE_HARMED_SERIES
                    )),
                    _fmt(cell["after"]["per_eval_series_delayed_gain"].get(
                        THE_HARMED_SERIES
                    )),
                    int(cell.get("retrain_delta_over_the_gate") or 0),
                )
            )
        lines.extend(["", "### Per evaluation series, delayed gain", ""])
        eval_uids = sorted({
            uid
            for cell in b3["cells"].values()
            for uid in cell["before"]["per_eval_series_delayed_gain"]
        })
        lines.append("| cell | " + " | ".join("`%s`" % uid for uid in eval_uids) + " |")
        lines.append("| --- | " + " | ".join("---:" for _ in eval_uids) + " |")
        for slot, cell in b3["cells"].items():
            for phase in ("before", "after"):
                row = cell[phase]["per_eval_series_delayed_gain"]
                lines.append(
                    "| `%s` %s | %s |" % (
                        slot, phase,
                        " | ".join(_fmt(row.get(uid)) for uid in eval_uids),
                    )
                )
        lines.append("")
    detail = payload.get("overall_detail") or {}
    if detail:
        lines.extend([
            "## The watched series",
            "",
            "- Hurt before in: %s." % (detail.get("cells_where_it_was_hurt") or "none"),
            "- Still hurt after in: %s." % (
                detail.get("cells_where_it_is_still_hurt") or "none"
            ),
            "- Pooled cells that moved: %s." % (
                detail.get("pooled_cells_that_moved") or "none"
            ),
            "- per_channel cells that moved: %s." % (
                detail.get("per_channel_cells_that_moved") or "none"
            ),
            "- New harmed series anywhere: %s." % (
                detail.get("new_harmed_series") or "none"
            ),
        ])
        forgone = detail.get("forgone_aggregate_gain_booked") or {}
        if forgone:
            lines.append(
                "- Forgone aggregate booked (veto to identity): %s."
                % {
                    key: "%+.6f -> %+.6f" % (
                        row["aggregate_delayed_before"],
                        row["aggregate_delayed_after"],
                    )
                    for key, row in forgone.items()
                }
            )
        if detail.get("backend_dependent"):
            lines.append("- **Backend-dependent:** %s" % detail["backend_dependent"])
        lines.append("")
    backfill = (payload.get("pre_registered") or {}).get(
        "backend_identity_backfill_18"
    ) or {}
    if backfill.get("draws"):
        lines.extend([
            "## Backend identity of every Slow draw",
            "",
            "#18 is backfilled at config level (the response-level identity "
            "was not recorded then); #19 records returned_models per draw. "
            "The two precisions are never merged into one count.",
            "",
            "| slice | draw | outcome | reason | identity | precision |",
            "| --- | --- | --- | --- | --- | --- |",
        ])
        for row in backfill["draws"]:
            lines.append(
                "| #18 | %s | `%s` | %s | `%s` @ agicto | config-level backfill |" % (
                    row["run"], row["outcome"],
                    row.get("no_proposal_reason") or row.get("reason_note"),
                    backfill.get("configured_model"),
                )
            )
        for round_row in b2.get("rounds") or ():
            for attempt in round_row.get("attempts") or ():
                lines.append(
                    "| #19 | %s sample %s | `%s` | %s | %s | response-level |" % (
                        round_row["label"], attempt.get("sample"),
                        attempt.get("outcome"),
                        attempt.get("no_proposal_reason") or "--",
                        attempt.get("returned_models")
                        or ("transport failure" if attempt.get("transport_error") else "--"),
                    )
                )
        lines.append("")
    b4 = payload.get("b4_experience") or {}
    after = payload.get("frozen_surface_after") or {}
    lines.extend([
        "## Cost and integrity",
        "",
        "- LLM calls: %s / %s." % (
            payload.get("llm_call_count"), payload.get("llm_call_budget")
        ),
        "- Consumer retrains: %s / %s (the gate pays the unguarded half "
        "once; B3 pays only the guard-moved delta: %s)." % (
            payload.get("consumer_retrains_total"), payload.get("retrain_budget"),
            b3.get("retrain_delta_over_the_gate", "--"),
        ),
        "- Experience rows written (provenance `%s`): %s." % (
            b4.get("provenance", "--"), b4.get("row_count")
        ),
        "- Frozen surface: %s files, drift %s." % (
            after.get("files"), after.get("drift")
        ),
        "- Registry arithmetic: %s." % (
            (payload.get("registry_arithmetic") or {}).get("note", "--")
        ),
        "- Wall seconds: %.1f." % float(payload.get("wall_seconds") or 0.0),
    ])
    return "\n".join(lines) + "\n"


def run_v2() -> int:
    """The #19 slice: migrate -> gate -> Slow retry -> compile -> replay."""
    started = time.perf_counter()
    before = _freeze(FROZEN_SURFACE_V2)
    budget = Budget(LLM_CALL_BUDGET_V2, RETRAIN_BUDGET)
    episodes = _load_episodes_v2()
    stores = _open_stores_v2()
    v1_count = len(set(FROZEN_SURFACE))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION_V2,
        "role": (
            "suture the defective Scope/Risk edit surface, prove the migrated "
            "state reproduces the record, and retry the Slow draw on the one "
            "repaired surface"
        ),
        "pre_registered": PRE_REGISTERED_V2,
        "frozen_surface_before": {"files": len(before), "sha256": before},
        "sutured_files": [
            dict(row, sha256_after_fix=before.get(row["path"]))
            for row in SUTURED_FILES_V2
        ],
        "registry_arithmetic": {
            "v1_unique_files": v1_count,
            "re_registered_in_post_fix_form": 4,
            "newly_registered": [COMPILER_FILE],
            "others_checked_zero_drift": v1_count - 4,
            "v2_total": len(before),
            "note": (
                "v1 registered %d unique files (the task book's '32+1'; one "
                "listed entry duplicates an FC entry and dedupes).  v2 "
                "re-registers 4 of them in post-fix form with both sha256 "
                "sides, adds compiler.py, and checks the other %d for zero "
                "drift: %d total." % (v1_count, v1_count - 4, len(before))
            ),
        },
        "source_evidence": {
            "artifact": _repo_rel(SOURCE_ARTIFACT),
            "artifact_sha256": episodes["artifact_sha256"],
            "read_only": True,
            "window": episodes["window"],
            "locked_roster": episodes["locked_roster"],
            "harm_observations": episodes["harm_observations"],
        },
        "stores": {
            slot: {
                key: value for key, value in row.items()
                if key not in ("store", "snapshot")
            }
            for slot, row in stores.items()
        },
        "llm_call_budget": LLM_CALL_BUDGET_V2,
        "retrain_budget": RETRAIN_BUDGET,
    }

    b1 = stage_b1(episodes["records"], stores)
    payload["b1_attribution"] = _public(b1)
    payload["guard_after_b1"] = _guard(before, "migration")
    print(
        "B1 primary=%s cause=%s" % (b1["primary_cell"], b1["primary_cause"]),
        flush=True,
    )

    migration = stage_migrate_v2(stores)
    payload["migration"] = _public(migration)
    payload["guard_after_migration"] = _guard(before, "gate")

    cohort = _confirmation_payload(episodes["locked_roster"])
    gate = stage_gate_v2(
        cohort=cohort, episodes=episodes, stores=stores,
        window=episodes["window"], budget=budget,
    )
    payload["non_regression_gate"] = _public(gate)

    def _finish(overall: Mapping[str, Any], **extra: Any) -> int:
        payload.update({
            "overall_verdict": overall["verdict"],
            "overall_verdict_reason": overall.get("reason", ""),
            "overall_detail": _public(overall.get("detail") or {}),
            "llm_call_count": budget.llm_used,
            "consumer_retrains_total": budget.retrains_charged,
            "exposure": {
                "windows_read": [
                    "2024 development training anchors (indices 120-900)",
                    "task_C confirmation window [9864, 10152], opened by #17",
                ],
                "beyond_17520": "SEALED, not read",
                "unopened_windows_read": "none",
                "fresh_claim": (
                    "none: this replay is development-level on an "
                    "already-exposed window"
                ),
            },
            "wall_seconds": time.perf_counter() - started,
            "frozen_surface_after": _verify(before),
        })
        payload.update(extra)
        if not payload["frozen_surface_after"]["ok"]:
            payload["overall_verdict"] = "CONCURRENT_WRITE_ABORT"
            payload["overall_verdict_reason"] = (
                "the frozen surface moved during the run; the reading is void"
            )
        return _write(
            payload, out_json=OUT_JSON_V2, out_md=OUT_MD_V2,
            markdown=_markdown_v2,
        )

    if not gate["ok"]:
        return _finish(_verdict_v2(
            attribution=b1, gate=gate,
            patch={"verdict": "GATE_NOT_REACHED"}, replay=None,
        ))
    payload["guard_after_gate"] = _guard(before, "B2")

    b2 = stage_b2_v2(
        stores=stores, episodes=episodes, attribution=b1, budget=budget,
    )
    payload["b2_patch"] = _public(b2)
    payload["guard_after_b2"] = _guard(before, "B3")
    if b2.get("verdict") != "PATCH_APPLIED":
        return _finish(_verdict_v2(
            attribution=b1, gate=gate, patch=b2, replay=None,
        ))

    b3 = stage_b3_v2(
        gate=gate, episodes=episodes, stores=stores,
        window=episodes["window"], budget=budget,
    )
    payload["b3_replay"] = _public(b3)
    payload["guard_after_b3"] = _guard(before, "B4")
    b4 = stage_b4(
        b3, provenance="slow_scope_update_v2", tag="ssu2",
        protocol=PROTOCOL_VERSION_V2,
    )
    payload["b4_experience"] = _public(b4)
    return _finish(_verdict_v2(
        attribution=b1, gate=gate, patch=b2, replay=b3,
    ))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attribution-only", action="store_true",
        help="run B1 and stop without proposing or replaying anything",
    )
    parser.add_argument(
        "--v2", action="store_true",
        help=(
            "run the #19 composite EDIT_SURFACE_DEFECT repair slice "
            "(migrate, gate, Slow retry on the sutured surface, replay)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.v2:
            if args.attribution_only:
                raise SystemExit("--attribution-only is a v1 switch")
            return run_v2()
        return run(attribution_only=bool(args.attribution_only))
    except ConcurrentWrite as exc:
        print("CONCURRENT_WRITE_ABORT:", exc, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
