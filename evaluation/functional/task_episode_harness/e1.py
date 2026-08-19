"""E1-v2: paired A5/A3 development pilot with repaired arm isolation.

Frozen authority: docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN_2026-08-17.md §5
plus the E1-v2 protocol repair instruction.  E1-v2 only changes the experiment
instrument:

* A3 and A5 each own a separate Target Experience ledger; episodes are never
  merged across arms.
* A3 and A5 each own a separate run-local active Target-local Skill snapshot
  and reuse it across Task Episodes.
* HORIZON=48 and every Support / delayed / cross-Task truth window is pairwise
  non-overlapping.
* Task 1 arm inputs are identical after removing the Source prior block; after
  Task 1 the two arms may naturally diverge through Source-caused
  Experience/Skill state.
* All paired outcomes use development blocks never opened by E1-v1 (first
  Support origin 3072 > E1-v1's maximum delayed window end 3054).

Prompt text, Source Card, candidate pool, Gate, Scope, Risk, Memory summary
shape and attribution taxonomy are unchanged.
"""
from __future__ import annotations

import dataclasses
import json
import math
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from run_v1_a5a3_runtime_regression import _load as _load_k1  # noqa: F401
from run_v1_kdd2018_natural_slow_update import _config

from evaluation.functional.task_episode_harness.normal_flow import (
    NF_BASE_URL,
    NF_MODEL,
    _FastAgentStub,
)
from evaluation.functional.task_episode_harness.public_context import (
    PUBLIC_CONTEXT_PROJECTION_FEATURE,
    build_task_public_context,
)
from evaluation.functional.task_episode_harness.runner import (
    MATERIAL_THRESHOLD,
    REPORT_REL,
    _arm_metrics,
    _evaluate_origins,
    _mapped_roster,
)
from evaluation.functional.task_episode_harness.skill_evolution import (
    _parse_json_response,
    _plain_steps,
    _probe_compiled,
    _steps_equal,
)
from evaluation.functional.task_episode_harness.t1 import TASK_CONSUMER_KEY
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
    EVIDENCE_DELAYED,
    EVIDENCE_SUPPORT,
    RELATION_ABSTAIN,
    RELATION_CONFLICT,
    RELATION_NEGATIVE,
    RELATION_POSITIVE,
    STATUS_EPISODE_ONLY,
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
    STATUS_RESTRICTED,
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _parse_frozen_steps
from SelfEvolvingHarnessTS.methods.ttha.generative_workflow import (
    CandidateCompilationError,
    CompiledWorkflow,
    build_public_operator_inventory,
    compile_workflow_proposal,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.retrieval import evaluate_applicability

PROTOCOL_VERSION = "e1_skill_card_warm_start_dev_v2"
DEVELOPMENT_DATASET = "kdd2018_frozen_cohort_e31"
SEALED_CONFIRMATION_DATASET = "noaa_global_hourly"
CALIBRATION_DATASET = "kdd2018_frozen_cohort_e1"
E1_DOMAIN = "kdd2018-e31-development-v2"
E1_CAUSE = "SKILL_LIBRARY_GAP"
HORIZON = 48
B = 3
N0 = 12
MAX_N = 30
# E1-v1's official roster ended at delayed origin 3006; with HORIZON=48 its
# last opened truth cell ended at 3054.  The interrupted pre-runs used earlier
# blocks of the same roster.  Every E1-v2 block starts at 3072, so no E1-v2
# truth window overlaps an E1-v1/pre-run truth window.
_UNEXPOSED_FIRST_SUPPORT_ORIGIN = 3072
# Each Task has K=3 Support + K=3 delayed windows of length HORIZON.  A stride
# of 6*HORIZON makes the windows inside a Task and across adjacent Tasks
# pairwise non-overlapping (the windows are adjacent, never intersecting).
_TASK_STRIDE = 6 * HORIZON
# The 10898-point KDD series can hold 27 complete unexposed Task blocks:
# 3072 + 27 * 288 = 10848 <= 10897.
AVAILABLE_TASK_COUNT = 27
E1V1_ARCHIVE_SUFFIX = ".e1_v1_archive.json"
E1_STATE_REL = ".e1v2_state"
E1V2_PREFLIGHT_CACHE_REL = ".e1v2_preflight_cache"

K1_SERIES = {
    "T117", "T118", "T119", "T12", "T120", "T121", "T122", "T123",
    "T124", "T125", "T126", "T127", "T128", "T129", "T13", "T130",
    "T131", "T132", "T133", "T134",
}

# Calibration is an isolated development slice.  E1-v1 calibration used
# [1104, 1128, 1152]; E1-v2 uses the next unexposed calibration block with the
# same HORIZON=48 non-overlap rule.
_CALIBRATION_ORIGINS = (1248, 1296, 1344)
_LOCAL_SKILL_PREFIX = "fast_winner_e1v2_"
_LOCAL_SKILL_RE = re.compile(r"[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*")


def _v2_workflow_signature(
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> str:
    """Operator-name signature for E1-v2 Target-local Skills.

    The E1-v2 local prefix keeps Target-local Skills separable from the
    E0 Source Card and from E1-v1 store cells in the same filesystem.
    """
    names = [str(op) for op, _params in steps]
    signature = "e1v2_" + "_".join(names)
    if not _LOCAL_SKILL_RE.fullmatch(signature):
        signature = "e1v2_" + "_".join(
            re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
            for name in names
        )
    if not _LOCAL_SKILL_RE.fullmatch(signature):
        raise ValueError(f"workflow signature is not a canonical id: {signature!r}")
    return signature


def _arm_store_root(repo_root: Path, arm: str) -> Path:
    return repo_root / E1_STATE_REL / arm.lower() / "snapshots"


def _preflight_context_cache_from_disk(
    repo_root: Path,
    task_roster: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load machine-computed public Task Contexts from the E1-v2 cache.

    The cache is generated by the same frozen ``build_task_public_context``
    helper in parallel worker processes; each entry is verified against the
    frozen task spec before use.
    """
    cache: dict[str, dict[str, Any]] = {}
    root = repo_root / E1V2_PREFLIGHT_CACHE_REL
    if not root.is_dir():
        return cache
    for spec in task_roster:
        task_id = str(spec["task_episode_id"])
        path = root / f"{task_id}.json"
        if not path.is_file():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if row.get("task_episode_id") != task_id:
            continue
        if int(row.get("observation_cutoff") or -1) != int(
            spec["support_origins"][0]
        ):
            continue
        context = row.get("public_context")
        if isinstance(context, dict) and context.get("observation_cutoff") == int(
            spec["support_origins"][0]
        ):
            cache[task_id] = context
    return cache


def _truth_windows(
    spec: Mapping[str, Any],
) -> list[dict[str, Any]]:
    windows = []
    for role in ("support", "delayed"):
        origins = tuple(int(value) for value in spec[f"{role}_origins"])
        for position, origin in enumerate(origins):
            windows.append({
                "role": role,
                "position": position,
                "start": origin,
                "end": origin + HORIZON,
            })
    return windows


def _all_truth_windows_non_overlapping(
    roster: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    windows = []
    for spec in roster:
        for window in _truth_windows(spec):
            windows.append({
                "task_episode_id": str(spec["task_episode_id"]),
                **window,
            })
    ordered = sorted(windows, key=lambda row: (row["start"], row["end"]))
    violations = [
        {"window": ordered[index - 1], "next_window": ordered[index]}
        for index in range(1, len(ordered))
        if ordered[index]["start"] < ordered[index - 1]["end"]
    ]
    return not violations, ordered, {
        "horizon": HORIZON,
        "window_count": len(ordered),
        "first_window_start": ordered[0]["start"] if ordered else None,
        "last_window_end": ordered[-1]["end"] if ordered else None,
        "violations": violations,
    }


def _task_spec(index: int) -> dict[str, Any]:
    base = _UNEXPOSED_FIRST_SUPPORT_ORIGIN + index * _TASK_STRIDE
    return {
        "task_episode_id": f"e1v2_task_{index + 1:02d}",
        "arm_order": "A3_A5" if index % 2 == 0 else "A5_A3",
        "horizon": HORIZON,
        "support_origins": (base, base + HORIZON, base + 2 * HORIZON),
        "delayed_origins": (
            base + 3 * HORIZON,
            base + 4 * HORIZON,
            base + 5 * HORIZON,
        ),
    }


def _frozen_task_roster(
    n: int = AVAILABLE_TASK_COUNT,
) -> tuple[dict[str, Any], ...]:
    return tuple(_task_spec(index) for index in range(n))


def _load_kdd_roster(
    repo_root: Path,
    cohort_rel: str,
    *,
    train_count: int = 12,
    eval_count: int = 8,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    rows = [
        json.loads(line)
        for line in (repo_root / cohort_rel).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = [str(row["series_name"]) for row in rows][: train_count + eval_count]
    cache = np.load(repo_root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(name) for name in cache["names"]]
    values = {
        uid: np.asarray(cache["values"][names.index(uid)], dtype=np.float64)
        for uid in selected
    }
    roster = [
        {"series_uid": uid, "role": "train"} for uid in selected[:train_count]
    ] + [
        {"series_uid": uid, "role": "eval"} for uid in selected[train_count:]
    ]
    return roster, values, selected


def _episode_from_report_row(row: Mapping[str, Any]) -> Any:
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (
        ExperienceEpisode,
    )

    return ExperienceEpisode(
        episode_id=str(row["episode_id"]),
        schema_version=str(row["schema_version"]),
        task_consumer_key=str(row["task_consumer_key"]),
        domain_namespace=str(row["domain_namespace"]),
        context_summary=dict(row.get("context_summary") or {}),
        workflow_signature=str(row.get("workflow_signature") or ""),
        support_response=dict(row.get("support_response") or {}),
        delayed_response=dict(row.get("delayed_response") or {}),
        relation=str(row["relation"]),
        evidence_level=str(row["evidence_level"]),
        response_validity=str(row.get("response_validity") or "VALID"),
        local_status=str(row["local_status"]),
        pattern_view=str(row.get("pattern_view") or "default"),
        evidence_refs=tuple(row.get("evidence_refs") or ()),
    )


def _source_card_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    e0 = report.get("skill_evolution_e0") or {}
    attempts = e0.get("attempts") or []
    for attempt in reversed(attempts):
        retrieval = attempt.get("retrieval") or {}
        if not retrieval.get("skill_id"):
            continue
        if attempt.get("compiled_steps"):
            return {
                "skill_id": retrieval["skill_id"],
                "workflow_steps": list(attempt["compiled_steps"]),
                "observable_applicability": retrieval.get(
                    "observable_applicability"
                ),
                "risk_guards": retrieval.get("risk_guards") or {},
                "local_status": "LOCAL_ACTIVE",
                "evidence_ref": "skill_evolution_e0",
            }
    return {}


def _source_bundle_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    e0 = report.get("skill_evolution_e0") or {}
    bundle = (e0.get("source") or {}).get("bundle") or {}
    return {
        "positive": bundle.get("positive"),
        "negative": bundle.get("negative"),
        "conflict": bundle.get("conflict"),
        "non_empty": any(
            bundle.get(key) is not None
            for key in ("positive", "negative", "conflict")
        ),
    }


_SOURCE_APPLICABILITY_LEAF_RE = re.compile(
    r"'feature':\s*'([^']+)',\s*'op':\s*'([^']+)',\s*"
    r"'value':\s*'([^']*)'"
)


def _runtime_source_applicability(
    source_card: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Rebuild the Runtime applicability AST stored on the E0 Source Card.

    E0 serialized ``observable_applicability`` as a repr string in the report;
    this helper recovers the leaf conditions without changing the Source Card
    content itself.  An unreadable applicability resolves to no match (fail
    closed) so a broken card can never leak into a non-matching Task.
    """
    raw = source_card.get("observable_applicability")
    if isinstance(raw, Mapping):
        return _plain_json_value(raw)
    if not isinstance(raw, str):
        return None
    leaves: list[dict[str, Any]] = []
    for match in _SOURCE_APPLICABILITY_LEAF_RE.finditer(raw):
        leaves.append({
            "feature": match.group(1),
            "op": match.group(2),
            "value": match.group(3),
        })
    return {"all": leaves} if leaves else None


def _source_prior_for_task(
    source_prior: Mapping[str, Any] | None,
    public_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Retrieve the A5-only Source package against the current public Context.

    This is the existing Runtime applicability matcher
    (``evaluate_applicability``).  A matching Task receives the Source Card and
    its associated Source evidence unchanged; a non-matching Task receives
    ``None``, so no Source Card field, workflow name or operator name enters
    the A5 generation payload.
    """
    if not isinstance(source_prior, Mapping):
        return None
    source_card = source_prior.get("source_card")
    if not isinstance(source_card, Mapping):
        return None
    applicability = _runtime_source_applicability(source_card)
    if applicability is None:
        return None
    fast_features = dict(public_context.get("task_fast_features") or {})
    try:
        matched, _reason = evaluate_applicability(
            applicability, fast_features
        )
    except (TypeError, ValueError):
        matched = False
    if not matched:
        return None
    return {
        "source_card": dict(source_card),
        "source_evidence": dict(source_prior.get("source_evidence") or {}),
    }


def _inventory_rows(public_context: Mapping[str, Any]) -> tuple[dict[str, object], ...]:
    return build_public_operator_inventory(
        public_context["task_kind"],
        public_context["representative_features"],
    )


def _proposal_params(row: Mapping[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    declared = dict(row.get("public_parameter_bindings") or {})
    for parameter in row.get("runtime_parameters") or []:
        name = parameter.get("name")
        if not isinstance(name, str) or name in declared:
            continue
        if name == "period":
            params[name] = 24
        elif parameter.get("default") is not None:
            params[name] = parameter["default"]
    return params


def _single_step_proposal(row: Mapping[str, Any]) -> dict[str, Any]:
    step: dict[str, Any] = {
        "op": row["name"],
        "params": _proposal_params(row),
    }
    declared = dict(row.get("public_parameter_bindings") or {})
    if declared:
        step["bindings"] = dict(declared)
    return {
        "decision": "PROPOSE",
        "steps": [step],
        "requested_observations": [],
        "fallback": "IDENTITY",
        "experience_use": [],
    }


class _Receipt:
    def __init__(self, gain: float | None, *, passed: bool = True) -> None:
        self.gain = gain
        self.verification = type("V", (), {"passed": passed})()


@dataclasses.dataclass
class _ArmState:
    """Per-arm run-local state.  Never merged across A3/A5."""

    arm: str
    memories: list[dict[str, Any]]
    episodes: list[Any]
    store: SnapshotStore
    active_snapshot: Any
    active_skill_ids: list[str]


def _load_active_arm_snapshot(
    repo_root: Path,
    arm: str,
) -> tuple[SnapshotStore, Any, dict[str, Any]]:
    """Load an arm's run-local active snapshot, materializing h0 on first use.

    Each arm has its own ``.e1v2_state/<arm>/snapshots`` directory, so the
    active pointer and fork root are also per-arm (SnapshotStore places
    ``active.json`` in the parent directory of its root).
    """
    store = SnapshotStore(_arm_store_root(repo_root, arm))
    h0 = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    active = None
    provenance = {
        "arm": arm,
        "store_root": str(_arm_store_root(repo_root, arm)),
        "source": "h0",
    }
    if store.active_path.exists():
        try:
            pointer = json.loads(store.active_path.read_text(encoding="utf-8"))
            active_sha = str(pointer["runtime_bundle_sha"])
            active_root = store.root / active_sha
            if active_root.is_dir():
                active = compile_snapshot(active_root, verify_lock=False)
                provenance["source"] = "persisted_active"
                provenance["runtime_bundle_sha"] = active.runtime_bundle_sha
        except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
            provenance["source"] = "persisted_active_unreadable"
            provenance["error"] = f"{type(exc).__name__}: {exc}"
            active = None
    if active is None:
        active = h0
        store.materialize(active)
        store.set_active(active.runtime_bundle_sha)
        provenance["materialized_h0"] = True
    return store, active, provenance


def _plain_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_plain_json_value(item) for item in value]
    return value


def _retrieve_target_local_skills(
    snapshot: Any,
    public_context: Mapping[str, Any],
    *,
    arm: str,
) -> list[dict[str, Any]]:
    """Retrieve this arm's active Target-local Skills under a Task Context.

    Only machine-added ``fast_winner_e1v2_*`` Skills are Target-local; the
    bootstrap Skill library is never presented as a Target-local candidate.
    """
    fast_features = dict(public_context["task_fast_features"])
    rows: list[dict[str, Any]] = []
    for skill in getattr(snapshot, "skills", ()) or ():
        skill_id = str(getattr(skill, "skill_id", "") or "")
        if not skill_id.startswith(_LOCAL_SKILL_PREFIX):
            continue
        steps = _parse_frozen_steps(str(getattr(skill, "body", "") or ""))
        if steps is None:
            continue
        applicability = dict(getattr(skill, "observable_applicability", {}) or {})
        try:
            matched, _reason = evaluate_applicability(
                applicability, fast_features
            )
        except ValueError:
            matched = False
        rows.append({
            "arm": arm,
            "skill_id": skill_id,
            "retrieved_in_current_context": bool(matched),
            "observable_applicability": _plain_json_value(applicability),
            "risk_guards": _plain_json_value(
                getattr(skill, "risk_guards", {}) or {}
            ),
            "frozen_program_steps": [
                {"op": op, "params": dict(params)} for op, params in steps
            ],
        })
    return rows


# Binding parameter names an operator declared *before* the 2026-08-19
# parameter-ownership fix moved it to OPERATOR_INTRINSIC.  Read only by
# :func:`_binding_free_signature`, so that Cards written under the legacy
# contract keep recognizing each other.  It never re-authorizes the legacy
# binding: nothing compiles those parameters any more.
LEGACY_PUBLIC_PARAMETER_BINDINGS: dict[str, tuple[str, ...]] = {
    "repair_level_shift": (
        "region_start_fraction",
        "region_end_fraction",
        "estimated_offset",
    ),
}


def _binding_free_signature(
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]:
    """Skill identity = operator structure + parameter binding source.

    W1 minimal repair.  An operator's declared ``public_parameter_bindings``
    are mechanically instantiated from the *current* public Context, so two
    Task Episodes that run the same operator structure differ in exactly those
    values and in nothing else.  Treating that difference as a different Skill
    made the machine re-ADD an id that already existed, which raised
    ``AddTargetExistsError``; the delayed window then never opened and the Task
    was charged ``B + 1`` for nothing (13 times in the frozen E1-v2 rows, twice
    more in G1).

    The binding *source* is not free: ``compile_workflow_proposal`` rejects any
    proposal whose bindings differ from the operator's declared bindings
    (``REQUIRED_BINDING_MISSING``), so for one operator the source is fixed.
    Dropping exactly the declared-bound parameter names therefore compares
    structure and source while ignoring only Task-local numeric instantiation.
    Every non-bound constant is still compared, so a different operator
    structure or a different constant is never merged.

    This changes identity only.  Execution is unaffected: the reuse path in
    :func:`_lifecycle` probes the freshly compiled program, never the stored
    Card's frozen numbers.

    Parameter-ownership follow-up (2026-08-19).  ``repair_level_shift`` moved
    from ``external_region`` to ``OPERATOR_INTRINSIC``, so the live registry no
    longer names its three former bindings.  Two consequences are handled here:

    * stored *legacy* Cards still carry those three parameters, so the names
      are read from the union of the live registry and
      :data:`LEGACY_PUBLIC_PARAMETER_BINDINGS`; without it the Cards written
      before the contract fix would stop recognizing each other and re-raise
      the ``AddTargetExistsError`` this repair removed;
    * a legacy Program and an intrinsic Program are *different Program
      semantics* (frozen design §17.4) and must never merge.  Dropping the
      bound names alone would collapse them onto the same signature, so each
      step also carries its binding source: ``external_bound`` when the step
      actually supplies those parameters, ``intrinsic`` when it does not.
    """
    from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA

    signature: list[tuple[str, tuple[tuple[str, Any], ...]]] = []
    for op, params in steps:
        name = str(op)
        metadata = OPERATOR_METADATA.get(name) or {}
        bound = set(dict(metadata.get("public_parameter_bindings", {}) or {}))
        bound |= set(LEGACY_PUBLIC_PARAMETER_BINDINGS.get(name, ()))
        supplied = {str(key) for key in dict(params)}
        source = "external_bound" if supplied & bound else "intrinsic"
        constants = tuple(
            sorted(
                (str(key), value)
                for key, value in dict(params).items()
                if str(key) not in bound
            )
        )
        signature.append(((name, source), constants))
    return tuple(signature)


def _existing_local_skill(
    snapshot: Any,
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> Any | None:
    target = _binding_free_signature(steps)
    for skill in getattr(snapshot, "skills", ()) or ():
        skill_id = str(getattr(skill, "skill_id", "") or "")
        if not skill_id.startswith(_LOCAL_SKILL_PREFIX):
            continue
        frozen = _parse_frozen_steps(str(getattr(skill, "body", "") or ""))
        if frozen is None:
            continue
        if _steps_equal(frozen, steps) or _binding_free_signature(frozen) == target:
            return skill
    return None


def _skill_ids(snapshot: Any, *, local_only: bool = False) -> list[str]:
    ids = []
    for skill in getattr(snapshot, "skills", ()) or ():
        skill_id = str(getattr(skill, "skill_id", "") or "")
        if not local_only or skill_id.startswith(_LOCAL_SKILL_PREFIX):
            ids.append(skill_id)
    return ids


def _evaluate_reachability(
    card: Mapping[str, Any],
    fast_features: Mapping[str, Any],
) -> dict[str, Any]:
    from SelfEvolvingHarnessTS.methods.ttha.method import (
        _applicability_from_card,
        _applicability_reachable,
    )

    applicability = _applicability_from_card(card)
    reachable, reason = _applicability_reachable(
        card, applicability, fast_features
    )
    return {
        "applicability": applicability,
        "reachable": reachable,
        "reason": reason,
    }


def _e1_slow_call(messages: list[dict[str, str]]) -> dict[str, Any]:
    import os

    api_key = next(
        (
            os.environ.get(name, "").strip()
            for name in ("OPENAI_API_KEY", "AGICTO_API_KEY")
            if os.environ.get(name, "").strip()
        ),
        None,
    )
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or AGICTO_API_KEY is required")
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=NF_BASE_URL, timeout=180)
    completion = client.chat.completions.create(
        model=NF_MODEL,
        messages=messages,
    )
    return _parse_json_response(str(completion.choices[0].message.content or ""))


_E1_PROPOSAL_SYSTEM = (
    "You are the Slow proposal stage for one paired A5/A3 Target Task Episode. "
    "Return an ordered list of one to three Workflow proposals to probe. "
    "Each proposal has one to four EXECUTABLE operators from operator_inventory. "
    "Bind dynamic parameters only through declared public bindings; never "
    "replay numeric parameters from source_prior or target_experiences. "
    "Reusing a Source Workflow is legal when the evidence justifies it; "
    "novelty is not required. You do not approve proposals. "
    "Return JSON only: "
    "{'decision':'PROPOSE','proposals':[{'steps':[{'op':'canonical_operator',"
    "'params':{},'bindings':{}}],'requested_observations':[],"
    "'fallback':'IDENTITY','experience_use':[]}],'reason':'...'} "
    "or {'decision':'ABSTAIN','reason':'...'}."
)

# G1 execution-side repair (docs/EXPERIENCE_TO_SKILL_CARD_EVOLUTION_PLAN
# _2026-08-17.md rev2.2 §6.1 item 3).  The frozen E1-v2 constant above is
# unchanged; this variant is only used when the caller opts into consuming the
# authorized ``candidate_policy.proposal_guidance`` Harness surface.  The extra
# sentence names the payload field and nothing else: it does not name an
# operator, a Context feature, or a direction.
_E1_PROPOSAL_SYSTEM_WITH_GUIDANCE = (
    _E1_PROPOSAL_SYSTEM
    + " The payload also carries candidate_policy_proposal_guidance, the "
    "deployed Harness proposal policy. Follow it when deciding which "
    "Workflows to propose and in which order."
)


def _proposal_payload(
    *,
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    target_memories: Sequence[Mapping[str, Any]],
    source_prior: Mapping[str, Any] | None,
    inventory: Sequence[Mapping[str, object]],
    target_local_skills: Sequence[Mapping[str, Any]] = (),
    proposal_guidance: str | None = None,
) -> dict[str, Any]:
    scope = frozenset(public_context["scope_series_uids"])
    payload: dict[str, Any] = {
        "task": TASK_CONSUMER_KEY,
        "target_task_episode_id": task_spec["task_episode_id"],
        "target_public_context": {
            "task_kind": public_context["task_kind"],
            "observation_cutoff": int(public_context["observation_cutoff"]),
            "task_signature": dict(public_context["task_signature"]),
            "scope_policy": {
                "feature": public_context["scope_feature"],
                "bin": public_context["scope_bin"],
                "selected_series_count": len(scope),
            },
            "representative_series_uid": public_context["representative_uid"],
            "representative_features": dict(
                public_context["representative_features"]
            ),
        },
        "operator_inventory": [dict(row) for row in inventory],
        "probe_budget": B,
        "material_threshold": MATERIAL_THRESHOLD,
        "target_experiences": [dict(row) for row in target_memories],
        # Machine-generated arm-local Skill retrieval.  The field is payload
        # data only; the Slow prompt text is unchanged.
        "target_local_skills": [dict(row) for row in target_local_skills],
    }
    if proposal_guidance is not None:
        # The deployed Harness surface candidate_policy.proposal_guidance.
        # Absent key == frozen E1-v2 payload shape.
        payload["candidate_policy_proposal_guidance"] = str(proposal_guidance)
    if source_prior is None:
        payload["source_prior"] = None
    else:
        payload["source_prior"] = dict(source_prior)
    return payload


def _normalized_payload_fingerprint(payload: Mapping[str, Any]) -> str:
    normalized = {
        key: value for key, value in payload.items()
        if key != "source_prior"
    }
    return json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _initial_proposals(
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    target_memories: Sequence[Mapping[str, Any]],
    source_prior: Mapping[str, Any] | None,
    inventory: Sequence[Mapping[str, object]],
    target_local_skills: Sequence[Mapping[str, Any]] = (),
    proposal_guidance: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _proposal_payload(
        task_spec=task_spec,
        public_context=public_context,
        target_memories=target_memories,
        source_prior=source_prior,
        inventory=inventory,
        target_local_skills=target_local_skills,
        proposal_guidance=proposal_guidance,
    )
    system = (
        _E1_PROPOSAL_SYSTEM
        if proposal_guidance is None
        else _E1_PROPOSAL_SYSTEM_WITH_GUIDANCE
    )
    response = _e1_slow_call([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    proposals = response.get("proposals")
    if response.get("decision") == "ABSTAIN":
        return payload, {
            "decision": "ABSTAIN",
            "reason": response.get("reason"),
            "proposals": [],
            "raw": response,
        }
    if (
        response.get("decision") != "PROPOSE"
        or not isinstance(proposals, list)
        or not 1 <= len(proposals) <= B
    ):
        raise RuntimeError(f"invalid Slow proposal response: {response!r}")
    return payload, {
        "decision": "PROPOSE",
        "reason": response.get("reason"),
        "proposals": proposals[:B],
        "raw": response,
    }


def _compile_proposal(
    proposal: Mapping[str, Any],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, Any],
    *,
    generation: int,
) -> tuple[CompiledWorkflow, dict[str, Any]]:
    normalized = {
        "decision": "PROPOSE",
        "steps": proposal.get("steps"),
        "requested_observations": proposal.get("requested_observations", []),
        "fallback": proposal.get("fallback", "IDENTITY"),
        "experience_use": proposal.get("experience_use", []),
    }
    compiled = compile_workflow_proposal(
        normalized,
        inventory,
        public_context["representative_features"],
        generation=generation,
    )
    return compiled, normalized


def _workflow_signature(
    steps: Sequence[tuple[str, Mapping[str, object]]],
) -> str:
    return _v2_workflow_signature(steps)


def _make_episode(
    *,
    arm: str,
    task_episode_id: str,
    attempt_index: int,
    compiled: CompiledWorkflow,
    workflow_signature: str,
    scope: frozenset[str],
    probe: Mapping[str, Any],
    support_origins: tuple[int, ...],
    public_context: Mapping[str, Any],
) -> Any:
    steps = compiled.candidate.program.execution_steps()
    gain = float(probe["macro_gain"])
    positive = gain >= MATERIAL_THRESHOLD
    return build_episode(
        episode_id=f"e1v2_{arm}_{task_episode_id}_attempt_{attempt_index}",
        task_consumer_key=TASK_CONSUMER_KEY,
        domain_namespace=E1_DOMAIN,
        context_summary={
            "task_episode_id": task_episode_id,
            "arm": arm,
            "attempt_index": attempt_index,
            "observation_cutoff": int(public_context["observation_cutoff"]),
            "task_signature": dict(public_context["task_signature"]),
            "scope_summary": {
                "training_series_count": len(scope),
                "training_series_uids": sorted(scope),
            },
            "cohort": {
                "training_series_count": 12,
                "evaluation_series_count": 8,
            },
            "local_pattern": {
                "scope_observation_bin": public_context["scope_bin"],
                "task_projection_bin": public_context["task_signature"].get(
                    PUBLIC_CONTEXT_PROJECTION_FEATURE
                ),
            },
            "program_geometry": {
                "scope": "training_series_subset",
                "program_steps": _plain_steps(steps),
            },
        },
        workflow_signature=workflow_signature,
        support_response={
            "gain": gain,
            "se_block": float(probe["se_block"]),
            "gain_over_se": probe["gain_over_se"],
            "accepted": positive,
            "block_origins": list(support_origins),
        },
        delayed_response={"evaluated": False, "gain": None,
                          "se_block": None, "gain_over_se": None},
        relation=RELATION_POSITIVE if positive else RELATION_NEGATIVE,
        evidence_level=EVIDENCE_SUPPORT,
        local_status=STATUS_LOCAL_DRAFT if positive else STATUS_EPISODE_ONLY,
        evidence_refs=["task_episode_harness_e1"],
    )


def _update_delayed(
    episode: Any,
    delayed_probe: Mapping[str, Any],
    delayed_origins: tuple[int, ...],
) -> Any:
    support_gain = float(episode.support_response.get("gain") or 0.0)
    delayed_gain = float(delayed_probe["macro_gain"])
    # Three bands, not two.  The old rule asked only that delayed be no worse
    # than -tau, so a Skill whose delayed window came back at -0.004 was
    # graded LOCAL_ACTIVE -- "active" then meant "not yet shown to be
    # harmful", while every report read it as "delayed confirmed the gain".
    # Requiring +tau on both windows makes LOCAL_ACTIVE mean what it is read
    # to mean, and gives the neutral middle its own honest name: the Support
    # gain held up enough to keep the Draft, and delayed has not confirmed it.
    # Historical runs keep their recorded values under the old semantics and
    # are not recomputed.
    support_positive = support_gain >= MATERIAL_THRESHOLD
    if not support_positive:
        status, relation = STATUS_EPISODE_ONLY, RELATION_NEGATIVE
    elif delayed_gain >= MATERIAL_THRESHOLD:
        status, relation = STATUS_LOCAL_ACTIVE, RELATION_POSITIVE
    elif delayed_gain <= -MATERIAL_THRESHOLD:
        status, relation = STATUS_RESTRICTED, RELATION_CONFLICT
    else:
        status, relation = STATUS_LOCAL_DRAFT, RELATION_ABSTAIN
    return dataclasses.replace(
        episode,
        delayed_response={
            "evaluated": True,
            "gain": delayed_gain,
            "se_block": float(delayed_probe["se_block"]),
            "gain_over_se": delayed_probe["gain_over_se"],
            "block_origins": list(delayed_origins),
        },
        evidence_level=EVIDENCE_DELAYED,
        local_status=status,
        relation=relation,
    )


def _memory_summary(episode: Any) -> dict[str, Any]:
    return {
        "episode_id": episode.episode_id,
        "workflow": episode.workflow_signature,
        "support_gain": (episode.support_response or {}).get("gain"),
        "support_se_block": (episode.support_response or {}).get("se_block"),
        "support_gain_over_se": (episode.support_response or {}).get(
            "gain_over_se"
        ),
        "delayed_gain": (episode.delayed_response or {}).get("gain"),
        "delayed_se_block": (episode.delayed_response or {}).get("se_block"),
        "delayed_gain_over_se": (episode.delayed_response or {}).get(
            "gain_over_se"
        ),
        "relation": episode.relation,
        "local_status": episode.local_status,
    }


def _sync_memory(memories: list[dict[str, Any]], episode: Any) -> None:
    summary = _memory_summary(episode)
    for index, memory in enumerate(memories):
        if memory.get("episode_id") == summary["episode_id"]:
            memories[index] = summary
            return
    memories.append(summary)


def _decision_payload(
    *,
    workflow: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
    target_memories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    allowed = (
        ["TRUST_DRAFT", "CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
        if above_threshold
        else ["CONTINUE", "ABSTAIN", "REQUEST_OBSERVATION"]
    )
    return {
        "last_probe": {
            "workflow": workflow,
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gain_over_se,
        },
        "remaining_workflows": list(remaining),
        "material_threshold": MATERIAL_THRESHOLD,
        "allowed_decisions": allowed,
        "target_experiences": [dict(row) for row in target_memories],
    }


_DECISION_SYSTEM = (
    "You are deciding what to do after one real Target Support probe. "
    "Use gain, se_block and gain_over_se as evidence; direction labels are "
    "not confidence. TRUST_DRAFT passes the candidate to the mechanical Gate; "
    "CONTINUE probes the next remaining workflow; ABSTAIN stops with no winner; "
    "REQUEST_OBSERVATION stops and records an observation gap. "
    "Return JSON: {'decision': one of allowed_decisions, 'reason': '...'}."
)


def _agent_decision(
    *,
    workflow: str,
    gain: float,
    se: float,
    gain_over_se: float | None,
    remaining: list[str],
    above_threshold: bool,
    target_memories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = _decision_payload(
        workflow=workflow,
        gain=gain,
        se=se,
        gain_over_se=gain_over_se,
        remaining=remaining,
        above_threshold=above_threshold,
        target_memories=target_memories,
    )
    response = _e1_slow_call([
        {"role": "system", "content": _DECISION_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ])
    decision = response.get("decision")
    if decision not in payload["allowed_decisions"]:
        raise RuntimeError(
            f"invalid decision {decision!r}; allowed={payload['allowed_decisions']}"
        )
    return {
        "decision": decision,
        "reason": response.get("reason"),
        "raw": response,
        "decision_input": payload,
    }


def _lifecycle(
    *,
    repo_root: Path,
    arm: str,
    arm_state: _ArmState,
    winner: Any,
    compiled: CompiledWorkflow,
    workflow_signature: str,
    scope: frozenset[str],
    values: Mapping[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_uids: list[str],
    delayed_origins: tuple[int, ...],
    public_context: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any] | None, Any]:
    steps = compiled.candidate.program.execution_steps()
    card = {
        "pattern_id": "e1v2-paired-target-episode",
        "failure_family": "natural_readiness_observation",
        "observable_signature": dict(public_context["task_signature"]),
        "workflow": {"steps": _plain_steps(steps)},
    }
    before_local_ids = _skill_ids(arm_state.active_snapshot, local_only=True)
    controller = EditController(
        arm_state.store, surfaces=SurfaceRegistry(), router=FaultRouter()
    )

    # Reuse path: this exact frozen Program is already an active arm-local
    # Skill.  Re-adding it would violate the ABSENT precondition, so the
    # instrument records a machine reuse and validates the current delayed
    # window against the same Program.
    existing_skill = _existing_local_skill(arm_state.active_snapshot, steps)
    if existing_skill is not None:
        holder: dict[str, Any] = {}

        def existing_delayed_evaluator(_steps: Any, _mode: int) -> _Receipt:
            probe = _probe_compiled(
                mapped_roster,
                values,
                config,
                delayed_origins,
                eval_uids,
                compiled,
                scope,
            )
            holder["probe"] = probe
            return _Receipt(float(probe["macro_gain"]))

        existing_delayed_evaluator(steps, 1)
        delayed_probe = holder.get("probe")
        if not isinstance(delayed_probe, Mapping):
            raise RuntimeError("delayed evaluator did not produce a probe")
        delayed_gain = float(delayed_probe["macro_gain"])
        method_event = {
            "stage": "deployed_existing_skill",
            "skill_id": str(existing_skill.skill_id),
            "skill_reused": True,
            "support_gain": float(winner.support_response["gain"]),
            "no_new_manifest": True,
        }
        delayed_event = {
            "stage": "existing_skill_revalidated",
            "skill_id": str(existing_skill.skill_id),
            "delayed_gain": delayed_gain,
            "delayed_ok": delayed_gain >= -MATERIAL_THRESHOLD,
            "note": (
                "existing active Skill remains in the arm-local snapshot; "
                "the current Episode is updated by the current delayed window"
            ),
        }
        updated = _update_delayed(winner, delayed_probe, delayed_origins)
        after_local_ids = _skill_ids(
            arm_state.active_snapshot, local_only=True
        )
        return (
            method_event,
            delayed_event,
            updated,
            delayed_probe,
            {
                "snapshot": arm_state.active_snapshot,
                "local_skill_ids_before": before_local_ids,
                "local_skill_ids_after": after_local_ids,
                "active_pointer_written": False,
                "reused_existing_skill": True,
            },
        )

    method = TTHAMethod(
        _FastAgentStub(),
        arm_state.active_snapshot,
        experience_episodes=tuple(
            episode
            for episode in arm_state.episodes
            if getattr(episode, "episode_id", None) != winner.episode_id
        ),
    )
    method.append_experience_episode(winner)
    method_event = method.handle_fast_winner(
        winner,
        steps,
        controller=controller,
        store=arm_state.store,
        card=card,
        evaluator=lambda _s, _m: _Receipt(None),
        fast_features=dict(public_context["task_fast_features"]),
        support_gain=float(winner.support_response["gain"]),
        confirmed_cause=E1_CAUSE,
    )
    delayed_event: dict[str, Any] = {"stage": "no_pending"}
    delayed_probe: dict[str, Any] | None = None
    active_pointer_written = False
    if method_event.get("stage") == "pending":
        holder: dict[str, Any] = {}

        def delayed_evaluator(_steps: Any, _mode: int) -> _Receipt:
            probe = _probe_compiled(
                mapped_roster,
                values,
                config,
                delayed_origins,
                eval_uids,
                compiled,
                scope,
            )
            holder["probe"] = probe
            return _Receipt(float(probe["macro_gain"]))

        delayed_event = method.handle_feedback_delayed(
            delayed_evaluator, episode_id=winner.episode_id
        )
        delayed_probe = holder.get("probe")
        if isinstance(delayed_probe, Mapping):
            winner = _update_delayed(winner, delayed_probe, delayed_origins)
            method.update_experience_episode(winner)
        if delayed_event.get("stage") == "approved":
            active_after = method._active_snapshot()
            arm_state.store.set_active(active_after.runtime_bundle_sha)
            active_pointer_written = True
    active_after = method._active_snapshot()
    return (
        method_event,
        delayed_event,
        winner,
        delayed_probe,
        {
            "snapshot": active_after,
            "local_skill_ids_before": before_local_ids,
            "local_skill_ids_after": _skill_ids(active_after, local_only=True),
            "active_pointer_written": active_pointer_written,
            "reused_existing_skill": False,
        },
    )


def _run_arm(
    *,
    repo_root: Path,
    arm_state: _ArmState,
    task_spec: Mapping[str, Any],
    public_context: Mapping[str, Any],
    source_prior: Mapping[str, Any] | None,
    inventory: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    mapped_roster: list[dict[str, Any]],
    config: Mapping[str, Any],
    eval_uids: list[str],
    llm_counter: list[int],
    consume_proposal_guidance: bool = False,
) -> dict[str, Any]:
    arm = arm_state.arm
    support_origins = tuple(task_spec["support_origins"])
    delayed_origins = tuple(task_spec["delayed_origins"])
    scope = frozenset(public_context["scope_series_uids"])
    retrieved_local_skills = _retrieve_target_local_skills(
        arm_state.active_snapshot, public_context, arm=arm
    )
    active_local_skill_ids_before = _skill_ids(
        arm_state.active_snapshot, local_only=True
    )
    # G1: the proposal policy is read from this arm's own active Harness
    # snapshot, never from a runner-level string, so an applied
    # candidate_policy.proposal_guidance PATCH is what actually reaches the
    # proposal stage.  Default False keeps the frozen E1-v2 payload shape.
    proposal_guidance = None
    if consume_proposal_guidance:
        policy = dict(
            getattr(arm_state.active_snapshot, "candidate_policy", {}) or {}
        )
        proposal_guidance = str(policy.get("proposal_guidance") or "")
    try:
        payload, initial = _initial_proposals(
            task_spec,
            public_context,
            arm_state.memories,
            source_prior,
            inventory,
            target_local_skills=retrieved_local_skills,
            proposal_guidance=proposal_guidance,
        )
        llm_counter[0] += 1
    except RuntimeError as exc:
        payload = _proposal_payload(
            task_spec=task_spec,
            public_context=public_context,
            target_memories=arm_state.memories,
            source_prior=source_prior,
            inventory=inventory,
            target_local_skills=retrieved_local_skills,
            proposal_guidance=proposal_guidance,
        )
        initial = {
            "decision": "ABSTAIN",
            "reason": f"proposal protocol error: {exc}",
            "proposals": [],
            "raw": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    proposals = list(initial["proposals"])
    probes = []
    winner = None
    winner_compiled = None
    instrument_unreadable = False
    stop_reason = "NO_DRAFT_IN_BUDGET"
    if initial["decision"] == "ABSTAIN":
        stop_reason = "AGENT_ABSTAIN"
    compiled_proposals: list[tuple[int, CompiledWorkflow, str]] = []
    for attempt_index, proposal in enumerate(proposals):
        generation = (
            int(task_spec["task_episode_id"].split("_")[-1]) * 10
            + attempt_index
        )
        try:
            compiled, _normalized_proposal = _compile_proposal(
                proposal,
                inventory,
                public_context,
                generation=generation,
            )
            workflow = _workflow_signature(
                compiled.candidate.program.execution_steps()
            )
        except (CandidateCompilationError, ValueError) as exc:
            record = {
                "attempt_index": attempt_index,
                "status": "COMPILATION_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            probes.append(record)
            continue
        compiled_proposals.append((attempt_index, compiled, workflow))
    for attempt_index, compiled, workflow in compiled_proposals:
        steps = compiled.candidate.program.execution_steps()
        try:
            support = _probe_compiled(
                mapped_roster,
                values,
                config,
                support_origins,
                eval_uids,
                compiled,
                scope,
            )
        except Exception as exc:  # noqa: BLE001
            # Validity repair 2026-08-18 (Planner ruling): an evaluator that
            # cannot measure the candidate is NOT the Agent declining to
            # propose one.  Absorbing it and continuing produced Tasks whose
            # NO_DRAFT_IN_BUDGET was manufactured by the instrument
            # (E1-v2 task_01/02/06, W3 task_06/07).  Fail fast and stop the
            # whole Task instead.
            record = {
                "attempt_index": attempt_index,
                "status": "INSTRUMENT_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
            probes.append(record)
            instrument_unreadable = True
            stop_reason = "INSTRUMENT_UNREADABLE"
            break
        episode = _make_episode(
            arm=arm,
            task_episode_id=task_spec["task_episode_id"],
            attempt_index=attempt_index,
            compiled=compiled,
            workflow_signature=workflow,
            scope=scope,
            probe=support,
            support_origins=support_origins,
            public_context=public_context,
        )
        gain = float(support["macro_gain"])
        se = float(support["se_block"])
        gse = support["gain_over_se"]
        remaining = [
            workflow_name
            for _index, _compiled, workflow_name in compiled_proposals
            if _index > attempt_index
        ]
        above = gain >= MATERIAL_THRESHOLD
        if above or remaining:
            try:
                decision = _agent_decision(
                    workflow=workflow,
                    gain=gain,
                    se=se,
                    gain_over_se=gse,
                    remaining=remaining,
                    above_threshold=above,
                    target_memories=arm_state.memories,
                )
                llm_counter[0] += 1
            except RuntimeError as exc:
                decision = {
                    "decision": "ABSTAIN",
                    "reason": f"promotion protocol error: {exc}",
                    "raw": None,
                    "decision_input": None,
                }
        else:
            decision = {
                "decision": "ABSTAIN",
                "reason": "budget exhausted without an acceptable candidate",
                "raw": None,
                "decision_input": None,
            }
        record = {
            "attempt_index": attempt_index,
            "workflow": workflow,
            "compiled_steps": _plain_steps(steps),
            "support_gain": gain,
            "support_se_block": se,
            "support_gain_over_se": gse,
            "agent_decision": decision,
            "mechanical_gate": "PASS" if above else "REJECT",
            "episode": episode.to_dict(),
        }
        probes.append(record)
        _sync_memory(arm_state.memories, episode)
        arm_state.episodes.append(episode)
        action = decision["decision"]
        if action == "TRUST_DRAFT" and above:
            winner = episode
            winner_compiled = compiled
            stop_reason = "TRUST_DRAFT_GATE_PASS"
            break
        if action == "TRUST_DRAFT" and not above:
            record["mechanical_gate"] = "REJECT_TRUST_BELOW_THRESHOLD"
            if remaining:
                continue
            stop_reason = "NO_DRAFT_IN_BUDGET"
            break
        if action == "CONTINUE":
            if remaining:
                continue
            stop_reason = "NO_DRAFT_IN_BUDGET"
            break
        if action == "ABSTAIN":
            stop_reason = "AGENT_ABSTAIN"
            break
        if action == "REQUEST_OBSERVATION":
            stop_reason = "REQUEST_OBSERVATION"
            break

    lifecycle = {"method_event": {"stage": "no_winner"},
                 "delayed_event": {"stage": "no_winner"}}
    delayed_probe = None
    active_local_skill_ids_after = list(active_local_skill_ids_before)
    if winner is not None and winner_compiled is not None:
      try:
        (
            method_event,
            delayed_event,
            updated,
            delayed_probe,
            active_state,
        ) = _lifecycle(
            repo_root=repo_root,
            arm=arm,
            arm_state=arm_state,
            winner=winner,
            compiled=winner_compiled,
            workflow_signature=winner.workflow_signature,
            scope=scope,
            values=values,
            mapped_roster=mapped_roster,
            config=config,
            eval_uids=eval_uids,
            delayed_origins=delayed_origins,
            public_context=public_context,
        )
        arm_state.active_snapshot = active_state["snapshot"]
        lifecycle = {
            "method_event": method_event,
            "delayed_event": delayed_event,
            "reused_existing_skill": bool(active_state.get("reused_existing_skill")),
            "active_pointer_written": bool(active_state.get("active_pointer_written")),
            "local_skill_ids_before": active_state["local_skill_ids_before"],
            "local_skill_ids_after": active_state["local_skill_ids_after"],
        }
        active_local_skill_ids_after = list(
            active_state["local_skill_ids_after"]
        )
        for probe in probes:
            if probe.get("episode", {}).get("episode_id") == winner.episode_id:
                probe["episode"] = updated.to_dict()
        winner = updated
        for index, episode in enumerate(arm_state.episodes):
            if episode.episode_id == winner.episode_id:
                arm_state.episodes[index] = winner
                break
        else:
            arm_state.episodes.append(winner)
        _sync_memory(arm_state.memories, winner)
        arm_state.active_skill_ids = active_local_skill_ids_after
      except Exception as exc:  # noqa: BLE001
        # The delayed evaluator hits the same instrument wall.  It is stopped
        # explicitly and visibly -- never swallowed and never allowed to abort
        # the whole run as if the protocol had broken.
        instrument_unreadable = True
        stop_reason = "INSTRUMENT_UNREADABLE"
        lifecycle = {
            "method_event": {
                "stage": "instrument_unreadable",
                "error": f"{type(exc).__name__}: {exc}",
            },
            "delayed_event": {"stage": "instrument_unreadable"},
        }
        winner = None

    valid_probes = [
        probe for probe in probes
        if isinstance(probe.get("support_gain"), (int, float))
    ]
    actual_probe_count = len(valid_probes)
    local_active = bool(
        winner is not None and winner.local_status == STATUS_LOCAL_ACTIVE
    )
    task_probe_cost = actual_probe_count if local_active else B + 1
    return {
        "arm": arm,
        "payload": payload,
        "initial": initial,
        "probes": probes,
        "stop_reason": stop_reason,
        "winner": (
            {
                "episode_id": winner.episode_id,
                "workflow": winner.workflow_signature,
                "local_status": winner.local_status,
                "delayed_gain": winner.delayed_response.get("gain"),
                "delayed_se_block": winner.delayed_response.get("se_block"),
                "delayed_gain_over_se": winner.delayed_response.get("gain_over_se"),
                "skill_reused": bool(
                    lifecycle.get("method_event", {}).get("stage")
                    == "deployed_existing_skill"
                ),
            }
            if winner is not None else None
        ),
        "delayed": delayed_probe,
        "lifecycle": lifecycle,
        "target_memories_after": [dict(row) for row in arm_state.memories],
        "target_episode_ids_after": [
            episode.episode_id for episode in arm_state.episodes
        ],
        "target_local_skills_before": retrieved_local_skills,
        "proposal_guidance_consumed": proposal_guidance,
        "active_local_skill_ids_before": active_local_skill_ids_before,
        "active_local_skill_ids_after": active_local_skill_ids_after,
        "metrics": {
            "task_probe_cost": task_probe_cost,
            "harmful_probe_count": sum(
                1 for probe in valid_probes
                if probe["support_gain"] < -MATERIAL_THRESHOLD
            ),
            "cumulative_support_harm": float(sum(
                -probe["support_gain"]
                for probe in valid_probes
                if probe["support_gain"] < -MATERIAL_THRESHOLD
            )),
            "task_local_active": int(local_active),
            "task_delayed_utility": (
                winner.delayed_response.get("gain")
                if winner is not None and winner.delayed_response.get("evaluated")
                else None
            ),
            "abstention": int(
                stop_reason in {"AGENT_ABSTAIN", "REQUEST_OBSERVATION"}
            ),
            # An unreadable Task is not behaviour and must be excluded from
            # any behavioural readout rather than counted as a tie.
            "instrument_unreadable": int(instrument_unreadable),
        },
    }

def _calibration_headroom(
    *,
    repo_root: Path,
    calibration_roster: list[dict[str, Any]],
    calibration_values: Mapping[str, Any],
    train_uids: list[str],
    inventory: Sequence[Mapping[str, object]],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(_config())
    config["support_origin"] = _CALIBRATION_ORIGINS[0]
    mapped = _mapped_roster(calibration_roster)
    eval_uids = [row["series_uid"] for row in mapped if row["role"] == "eval"]
    scope = frozenset(public_context["scope_series_uids"])
    evaluated = []
    first_positive = None
    for index, row in enumerate(inventory):
        if row.get("availability") != "EXECUTABLE":
            continue
        try:
            compiled = compile_workflow_proposal(
                _single_step_proposal(row),
                inventory,
                public_context["representative_features"],
                generation=index + 1,
            )
            identity_rows = _evaluate_origins(
                mapped, calibration_values, None, config,
                _CALIBRATION_ORIGINS, None,
            )
            candidate_rows = _evaluate_origins(
                mapped, calibration_values, compiled, config,
                _CALIBRATION_ORIGINS, set(scope),
            )
            metrics = _arm_metrics(
                identity_rows, candidate_rows, _CALIBRATION_ORIGINS, eval_uids
            )
        except Exception as exc:  # noqa: BLE001
            evaluated.append({
                "operator": row["name"],
                "status": "INSTRUMENT_INVALID",
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        record = {
            "operator": row["name"],
            "status": "EVALUATED",
            "macro_gain": metrics["macro_gain"],
            "se_block": metrics["se_block"],
            "gain_over_se": metrics["gain_over_se"],
            "positive_series_count": metrics["positive_series_count"],
            "negative_series_count": metrics["negative_series_count"],
        }
        evaluated.append(record)
        print(
            f"E1V2_CALIBRATION_ROW {record['operator']} "
            f"gain={round(float(record.get('macro_gain') or 0.0), 6)} "
            f"gse={record.get('gain_over_se')}",
            flush=True,
        )
        if first_positive is None and metrics["macro_gain"] >= MATERIAL_THRESHOLD:
            first_positive = record
            break
    calibration_windows = [
        {"role": "support", "position": i, "start": int(o), "end": int(o) + HORIZON}
        for i, o in enumerate(_CALIBRATION_ORIGINS)
    ]
    calibration_windows_non_overlap = all(
        calibration_windows[i]["start"] >= calibration_windows[i - 1]["end"]
        for i in range(1, len(calibration_windows))
    )
    return {
        "calibration_dataset": CALIBRATION_DATASET,
        "calibration_origins": list(_CALIBRATION_ORIGINS),
        "horizon": HORIZON,
        "truth_windows": calibration_windows,
        "truth_windows_non_overlap": calibration_windows_non_overlap,
        "inventory_order": "canonical operator registry order",
        "single_step_only": True,
        "combination_search": False,
        "first_positive": first_positive,
        "pass": first_positive is not None,
        "evaluated": evaluated,
        "private_audit_only": True,
    }


def _run_preflight(
    repo_root: Path,
    *,
    target_roster: list[dict[str, Any]],
    target_values: Mapping[str, Any],
    target_train_uids: list[str],
    task_roster: Sequence[Mapping[str, Any]],
    target_context_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = sorted({row["series_uid"] for row in target_roster})
    overlap_with_k1 = sorted(set(selected) & K1_SERIES)
    target_contexts: dict[str, dict[str, Any]] = {}
    context_wall_started = time.perf_counter()
    for position, spec in enumerate(task_roster):
        task_id = str(spec["task_episode_id"])
        if target_context_cache is not None and task_id in target_context_cache:
            context = target_context_cache[task_id]
        else:
            context = build_task_public_context(
                target_values,
                target_train_uids,
                observation_cutoff=int(spec["support_origins"][0]),
            )
            if target_context_cache is not None:
                target_context_cache[task_id] = context
        target_contexts[task_id] = context
        print(
            f"E1V2_PREFLIGHT_CONTEXT {position + 1}/{len(task_roster)} "
            f"{task_id} sig={dict(context['task_signature'])} "
            f"elapsed={round(time.perf_counter() - context_wall_started, 2)}s",
            flush=True,
        )
    signatures = [
        dict(context["task_signature"])
        for context in target_contexts.values()
    ]
    distinct = []
    for signature in signatures:
        if signature not in distinct:
            distinct.append(signature)
    context_pass = len(distinct) >= 2

    windows_non_overlap, ordered_windows, window_audit = (
        _all_truth_windows_non_overlapping(task_roster)
    )
    first_support_origin = min(
        int(spec["support_origins"][0]) for spec in task_roster
    )
    last_window_end = window_audit["last_window_end"]
    unexposed_block_audit = {
        "e1_v1_official_roster_max_delayed_origin": 3006,
        "e1_v1_max_truth_window_end": 3054,
        "e1v2_first_support_origin": first_support_origin,
        "e1v2_starts_after_e1_v1_exposure": first_support_origin > 3054,
        "e1v2_last_truth_window_end": last_window_end,
        "e1v2_fits_series_cache": last_window_end is None
        or last_window_end <= 10897,
    }

    cal_roster, cal_values, cal_selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e1.jsonl"
    )
    cal_train = [row["series_uid"] for row in cal_roster if row["role"] == "train"]
    cal_context = build_task_public_context(
        cal_values, cal_train, _CALIBRATION_ORIGINS[0]
    )
    inventory = _inventory_rows(cal_context)
    headroom = _calibration_headroom(
        repo_root=repo_root,
        calibration_roster=cal_roster,
        calibration_values=cal_values,
        train_uids=cal_train,
        inventory=inventory,
        public_context=cal_context,
    )
    checks = {
        "protocol_version": PROTOCOL_VERSION,
        "development_dataset": DEVELOPMENT_DATASET,
        "sealed_confirmation_dataset": SEALED_CONFIRMATION_DATASET,
        "sealed_dataset_read": False,
        "horizon": HORIZON,
        "target_base_series_overlap_with_k1_source": overlap_with_k1,
        "target_base_series_non_overlap": not overlap_with_k1,
        "paired_task_count": len(task_roster),
        "paired_task_count_at_least_12": len(task_roster) >= N0,
        "max_N": MAX_N,
        "available_unexposed_task_blocks": AVAILABLE_TASK_COUNT,
        "truth_window_non_overlap": {
            "pass": windows_non_overlap,
            "horizon": HORIZON,
            "window_count": window_audit["window_count"],
            "first_window_start": window_audit["first_window_start"],
            "last_window_end": window_audit["last_window_end"],
            "violations": window_audit["violations"],
        },
        "unexposed_block_audit": unexposed_block_audit,
        "calibration_slice_isolated": {
            "calibration_dataset": CALIBRATION_DATASET,
            "calibration_series_disjoint_from_target": bool(
                set(cal_selected).isdisjoint(selected)
            ),
            "calibration_origin_blocks_disjoint_from_target": bool(
                max(_CALIBRATION_ORIGINS)
                < min(int(spec["support_origins"][0]) for spec in task_roster)
            ),
            "calibration_truth_windows_non_overlap": bool(
                headroom.get("truth_windows_non_overlap")
            ),
        },
        "context_census": {
            "task_count": len(signatures),
            "distinct_signature_count": len(distinct),
            "distinct_signatures": distinct,
            "pass": context_pass,
        },
        "calibration_headroom": headroom,
        "frozen": {
            "B": B,
            "N0": N0,
            "max_N": MAX_N,
            "horizon": HORIZON,
            "llm_model": NF_MODEL,
            "llm_base_url": NF_BASE_URL,
            "arm_order_rule": "A3_A5 on even task index, A5_A3 on odd task index",
            "task_origin_rule": (
                "base=3072+i*288; support=base,+48,+96; "
                "delayed=base+144,+192,+240; all windows pairwise "
                "non-overlapping"
            ),
            "candidate_pool": "full canonical operator inventory, 1-4 steps per proposal",
            "post_probe_promotion": "Slow decision, mechanical Gate",
            "per_arm_state": (
                "independent Target Experience ledger and independent "
                "run-local active Target-local Skill snapshot"
            ),
        },
    }
    checks["preflight_pass"] = bool(
        checks["target_base_series_non_overlap"]
        and checks["paired_task_count_at_least_12"]
        and checks["truth_window_non_overlap"]["pass"]
        and checks["unexposed_block_audit"]["e1v2_starts_after_e1_v1_exposure"]
        and checks["unexposed_block_audit"]["e1v2_fits_series_cache"]
        and checks["calibration_slice_isolated"]["calibration_series_disjoint_from_target"]
        and checks["calibration_slice_isolated"]["calibration_origin_blocks_disjoint_from_target"]
        and checks["calibration_slice_isolated"]["calibration_truth_windows_non_overlap"]
        and checks["context_census"]["pass"]
        and checks["calibration_headroom"]["pass"]
    )
    return checks

def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def metric(arm: str, key: str) -> list[float]:
        return [float(row[arm]["metrics"][key]) for row in rows]

    a3_cost = metric("A3", "task_probe_cost")
    a5_cost = metric("A5", "task_probe_cost")
    probe_diff = [a5 - a3 for a3, a5 in zip(a3_cost, a5_cost)]
    n = len(probe_diff)
    probe_mean = float(np.mean(probe_diff)) if n else 0.0
    probe_sd = float(np.std(probe_diff, ddof=1)) if n > 1 else 0.0
    probe_se = probe_sd / math.sqrt(n) if n else 0.0

    def diff(arm_key: str) -> tuple[list[float], float]:
        a3 = metric("A3", arm_key)
        a5 = metric("A5", arm_key)
        values = [a5v - a3v for a3v, a5v in zip(a3, a5)]
        return values, float(np.mean(values)) if values else 0.0

    harm_count_diff, harm_count_mean = diff("harmful_probe_count")
    harm_sum_diff, harm_sum_mean = diff("cumulative_support_harm")
    a3_active = sum(row["A3"]["metrics"]["task_local_active"] for row in rows)
    a5_active = sum(row["A5"]["metrics"]["task_local_active"] for row in rows)
    drafts = []
    for row in rows:
        a3_winner = row["A3"].get("winner")
        a5_winner = row["A5"].get("winner")
        if a3_winner is not None and a5_winner is not None:
            drafts.append({
                "task_episode_id": row["task_episode_id"],
                "A3_delayed_utility": a3_winner.get("delayed_gain"),
                "A5_delayed_utility": a5_winner.get("delayed_gain"),
            })
    q = len(drafts) / n if n else 0.0
    paired_delayed_diff = [
        float(row["A5_delayed_utility"]) - float(row["A3_delayed_utility"])
        for row in drafts
        if isinstance(row.get("A3_delayed_utility"), (int, float))
        and isinstance(row.get("A5_delayed_utility"), (int, float))
    ]
    delayed_utility_mean = (
        float(np.mean(paired_delayed_diff)) if paired_delayed_diff else None
    )
    a3_draft_count = sum(
        1 for row in rows if row["A3"].get("winner") is not None
    )
    a5_draft_count = sum(
        1 for row in rows if row["A5"].get("winner") is not None
    )
    a3_survival = sum(
        1 for row in rows
        if row["A3"].get("winner") is not None
        and row["A3"]["winner"].get("local_status") == STATUS_LOCAL_ACTIVE
    )
    a5_survival = sum(
        1 for row in rows
        if row["A5"].get("winner") is not None
        and row["A5"]["winner"].get("local_status") == STATUS_LOCAL_ACTIVE
    )
    a3_memory_ids = {
        str(memory.get("episode_id"))
        for row in rows
        for memory in (row["A3"].get("target_memories_after") or [])
    }
    a5_memory_ids = {
        str(memory.get("episode_id"))
        for row in rows
        for memory in (row["A5"].get("target_memories_after") or [])
    }
    a3_skill_ids = {
        str(skill_id)
        for row in rows
        for skill_id in (row["A3"].get("active_local_skill_ids_after") or [])
    }
    a5_skill_ids = {
        str(skill_id)
        for row in rows
        for skill_id in (row["A5"].get("active_local_skill_ids_after") or [])
    }
    history_isolation = {
        "a3_history_has_a3_episode": any(
            episode_id.startswith("e1v2_A3_") for episode_id in a3_memory_ids
        ),
        "a3_history_has_a5_episode": any(
            episode_id.startswith("e1v2_A5_") for episode_id in a3_memory_ids
        ),
        "a5_history_has_a5_episode": any(
            episode_id.startswith("e1v2_A5_") for episode_id in a5_memory_ids
        ),
        "a5_history_has_a3_episode": any(
            episode_id.startswith("e1v2_A3_") for episode_id in a5_memory_ids
        ),
        # Independent arm snapshots may legitimately contain the same
        # machine-generated skill_id (e.g. both arms independently formed
        # fast_winner_e1v2_denoise_stl).  Skill-id overlap across two separate
        # store roots is therefore observational, not an isolation violation.
        "a3_a5_skill_snapshot_disjoint": bool(a3_skill_ids.isdisjoint(a5_skill_ids)),
        "skill_id_overlap_note": (
            "skill_ids are per-arm snapshots; identical machine-generated ids "
            "in two independent store roots do not indicate cross-arm merge"
        ),
        "pass": bool(
            not any(
                episode_id.startswith("e1v2_A5_")
                for episode_id in a3_memory_ids
            )
            and not any(
                episode_id.startswith("e1v2_A3_")
                for episode_id in a5_memory_ids
            )
        ),
    }
    return {
        "n": n,
        "probe_diff": probe_diff,
        "probe_paired_mean": probe_mean,
        "probe_paired_sd": probe_sd,
        "probe_paired_se": probe_se,
        "probe_ci95_upper": probe_mean + 1.96 * probe_se if n else None,
        "harmful_probe_count_diff": harm_count_diff,
        "harmful_probe_count_paired_mean": harm_count_mean,
        "cumulative_support_harm_diff": harm_sum_diff,
        "cumulative_support_harm_paired_mean": harm_sum_mean,
        "a3_local_active_count": a3_active,
        "a5_local_active_count": a5_active,
        "paired_draft_count": len(drafts),
        "q": q,
        "paired_delayed_utility_diff": paired_delayed_diff,
        "paired_delayed_utility_mean": delayed_utility_mean,
        "a3_draft_count": a3_draft_count,
        "a5_draft_count": a5_draft_count,
        "a3_delayed_survival_rate": (
            a3_survival / a3_draft_count if a3_draft_count else None
        ),
        "a5_delayed_survival_rate": (
            a5_survival / a5_draft_count if a5_draft_count else None
        ),
        "history_isolation": history_isolation,
    }


def _sample_plan(
    summary: Mapping[str, Any],
    *,
    available_task_count: int = AVAILABLE_TASK_COUNT,
) -> dict[str, Any]:
    n = summary["n"]
    s = summary["probe_paired_sd"]
    delta = 1.0
    n_req = math.ceil(7.84 * s * s / (delta * delta)) if s > 0 else 0
    q = summary["q"]
    n_draft_req = math.ceil(8 / q) if q and q > 0 else None
    values = [12, n_req] + ([n_draft_req] if n_draft_req is not None else [])
    n_final = max(values)
    return {
        "delta_probe": delta,
        "paired_probe_sd": s,
        "N_req": n_req,
        "q": q,
        "N_draft_req": n_draft_req,
        "N_final": n_final,
        "N_final_within_cap": n_final <= MAX_N,
        "N_final_within_available_roster": n_final <= available_task_count,
        "available_task_count": available_task_count,
        "extension_count": max(0, n_final - n),
        "note": (
            "sample formula is development capacity planning only; "
            "prequential Task pairs are not independent replicates"
        ),
    }


def _verdict(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    plan: Mapping[str, Any],
    source_behavior_changed: bool,
) -> str:
    if plan["N_final"] > MAX_N or not plan["N_final_within_available_roster"]:
        return "E1_PRACTICAL_RESOLUTION_INSUFFICIENT"
    if not summary.get("history_isolation", {}).get("pass", True):
        return "E1_V2_ARM_ISOLATION_FAILED"
    paired_drafts_readable = summary["paired_draft_count"] >= 8
    if not source_behavior_changed:
        return "A5_A3_SKILL_INPUT_INERT"
    harm_count_ci_lower = summary["harmful_probe_count_paired_mean"] - (
        1.96 * (
            float(np.std(summary["harmful_probe_count_diff"], ddof=1))
            / math.sqrt(summary["n"])
        )
        if summary["n"] > 1 and summary["harmful_probe_count_diff"]
        else 0.0
    )
    harm_sum_ci_lower = summary["cumulative_support_harm_paired_mean"] - (
        1.96 * (
            float(np.std(summary["cumulative_support_harm_diff"], ddof=1))
            / math.sqrt(summary["n"])
        )
        if summary["n"] > 1 and summary["cumulative_support_harm_diff"]
        else 0.0
    )
    negative_transfer = bool(
        harm_count_ci_lower > 0
        or harm_sum_ci_lower > 0
        or (
            paired_drafts_readable
            and summary["paired_delayed_utility_mean"] is not None
            and summary["paired_delayed_utility_mean"] < 0
        )
    )
    support_efficiency = bool(
        summary["probe_paired_mean"] <= -1
        and summary["probe_ci95_upper"] is not None
        and summary["probe_ci95_upper"] < 0
        and summary["harmful_probe_count_paired_mean"] <= 0
        and summary["cumulative_support_harm_paired_mean"] <= 0
        and summary["a5_local_active_count"] >= summary["a3_local_active_count"]
    )
    if negative_transfer:
        return "A5_SKILL_CARD_NEGATIVE_TRANSFER_DEV"
    if not paired_drafts_readable:
        if support_efficiency:
            return "A5_SUPPORT_EFFICIENCY_DEV_SIGNAL / DELAYED_UNREADABLE"
        return "A5_SKILL_CARD_NO_BENEFIT_DEV"
    delayed_ok = bool(
        summary["paired_delayed_utility_mean"] is not None
        and summary["paired_delayed_utility_mean"] >= 0
        and (
            summary["a5_delayed_survival_rate"] is None
            or summary["a3_delayed_survival_rate"] is None
            or summary["a5_delayed_survival_rate"]
            >= summary["a3_delayed_survival_rate"]
        )
    )
    if support_efficiency and delayed_ok:
        return "A5_SKILL_CARD_WARM_START_DEV_SIGNAL"
    return "A5_SKILL_CARD_NO_BENEFIT_DEV"


def _archive_e1_v1_report(report_path: Path) -> dict[str, Any]:
    """Keep the E1-v1 report object and a byte-identical archive copy."""
    archive_path = report_path.with_name(
        report_path.stem + E1V1_ARCHIVE_SUFFIX
    )
    copied = False
    if report_path.exists() and not archive_path.exists():
        shutil.copyfile(report_path, archive_path)
        copied = True
    return {
        "report_key_preserved": "e1",
        "archive_path": str(archive_path),
        "archive_written": copied,
    }


def _build_arm_state(repo_root: Path, arm: str) -> tuple[_ArmState, dict[str, Any]]:
    store, active_snapshot, provenance = _load_active_arm_snapshot(
        repo_root, arm
    )
    return _ArmState(
        arm=arm,
        memories=[],
        episodes=[],
        store=store,
        active_snapshot=active_snapshot,
        active_skill_ids=_skill_ids(active_snapshot, local_only=True),
    ), provenance


def run_e1(report_path: Path = REPORT_REL) -> dict[str, Any]:
    started = time.perf_counter()
    repo_root = Path(__file__).resolve().parents[3]
    archive = _archive_e1_v1_report(report_path)
    report = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.exists()
        else {}
    )
    # E1-v1 stays untouched under its original key for the whole E1-v2 run.
    v1_preserved = {
        **archive,
        "original_verdict": report.get("e1", {}).get("verdict"),
        "report_object_untouched": True,
    }
    source_card = _source_card_from_report(report)
    if not source_card:
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "verdict": "E1_SOURCE_CARD_UNAVAILABLE",
            "llm_api_call_count": 0,
            "e1_v1_preserved": v1_preserved,
            "boundary": {"e2_not_started": True, "sealed_confirmation_opened": False},
        }
        report["historical_verdict_before_e1_v2"] = report.get("verdict")
        report["phase"] = "e1_v2_preflight"
        report["e1_v2"] = result
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result
    source_bundle = _source_bundle_from_report(report)

    state_root = repo_root / E1_STATE_REL
    if state_root.exists():
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "verdict": "E1_V2_STATE_CONTAMINATED",
            "state_root": str(state_root),
            "llm_api_call_count": 0,
            "e1_v1_preserved": v1_preserved,
            "boundary": {"e2_not_started": True, "sealed_confirmation_opened": False},
        }
        report["historical_verdict_before_e1_v2"] = report.get("verdict")
        report["phase"] = "e1_v2_preflight"
        report["e1_v2"] = result
        report["verdict"] = result["verdict"]
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    target_roster, target_values, target_selected = _load_kdd_roster(
        repo_root, "artifacts/functional/e2/w1_kdd2018_frozen_cohort_e31.jsonl"
    )
    target_train_uids = [
        row["series_uid"] for row in target_roster if row["role"] == "train"
    ]
    target_eval_uids = [
        row["series_uid"] for row in target_roster if row["role"] == "eval"
    ]
    mapped_roster = _mapped_roster(target_roster)
    eval_uids = [
        row["series_uid"] for row in mapped_roster if row["role"] == "eval"
    ]
    assert target_eval_uids == eval_uids
    config = dict(_config())
    task_roster = _frozen_task_roster(AVAILABLE_TASK_COUNT)
    target_context_cache: dict[str, dict[str, Any]] = (
        _preflight_context_cache_from_disk(repo_root, task_roster)
    )

    preflight = _run_preflight(
        repo_root,
        target_roster=target_roster,
        target_values=target_values,
        target_train_uids=target_train_uids,
        task_roster=task_roster,
        target_context_cache=target_context_cache,
    )
    if not preflight["preflight_pass"]:
        if preflight["context_census"]["pass"] is False:
            verdict = "E1_TARGET_CONTEXTS_INERT"
        elif preflight["calibration_headroom"]["pass"] is False:
            verdict = "E1_DEVELOPMENT_SUBSTRATE_NO_KNOWN_HEADROOM"
        elif not preflight["target_base_series_non_overlap"]:
            verdict = "E1_TARGET_SOURCE_OVERLAP"
        elif not preflight["truth_window_non_overlap"]["pass"]:
            verdict = "E1_V2_TRUTH_WINDOW_OVERLAP"
        elif not preflight["unexposed_block_audit"]["e1v2_starts_after_e1_v1_exposure"]:
            verdict = "E1_V2_BLOCK_EXPOSURE_VIOLATION"
        else:
            verdict = "E1_PREFLIGHT_FAILED"
        result = {
            "protocol_version": PROTOCOL_VERSION,
            "verdict": verdict,
            "preflight": preflight,
            "llm_api_call_count": 0,
            "e1_v1_preserved": v1_preserved,
            "boundary": {"e2_not_started": True, "sealed_confirmation_opened": False},
        }
        report["historical_verdict_before_e1_v2"] = report.get("verdict")
        report["phase"] = "e1_v2_preflight"
        report["e1_v2"] = result
        report["verdict"] = verdict
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        return result

    # Freeze before first paired outcome.
    preregistration = {
        "protocol_version": PROTOCOL_VERSION,
        "development_dataset": DEVELOPMENT_DATASET,
        "sealed_confirmation_dataset": SEALED_CONFIRMATION_DATASET,
        "development_target_series": target_selected,
        "target_train_series": target_train_uids,
        "target_eval_series": target_eval_uids,
        "task_roster": list(task_roster),
        "horizon": HORIZON,
        "B": B,
        "N0": N0,
        "max_N": MAX_N,
        "available_unexposed_task_blocks": AVAILABLE_TASK_COUNT,
        "unexposed_after_e1_v1_window_end": 3054,
        "llm_settings": {"model": NF_MODEL, "base_url": NF_BASE_URL},
        "preflight": preflight,
        "source_card": source_card,
        "source_bundle": source_bundle,
        "per_arm_state": {
            "A3": {
                "store_root": str(_arm_store_root(repo_root, "A3")),
                "initial_memories": [],
                "initial_local_skill_ids": [],
            },
            "A5": {
                "store_root": str(_arm_store_root(repo_root, "A5")),
                "initial_memories": [],
                "initial_local_skill_ids": [],
            },
        },
    }

    llm_counter = [0]
    rows = []
    a3_source_prior = None
    a5_source_prior = {
        "source_card": source_card,
        "source_evidence": source_bundle,
    }
    arm_states: dict[str, _ArmState] = {}
    arm_state_provenance: dict[str, dict[str, Any]] = {}
    for arm in ("A3", "A5"):
        arm_states[arm], arm_state_provenance[arm] = _build_arm_state(
            repo_root, arm
        )
    preregistration["arm_state_provenance"] = arm_state_provenance

    def _run_task(spec: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(spec["task_episode_id"])
        public_context = target_context_cache.get(task_id)
        if public_context is None:
            public_context = build_task_public_context(
                target_values,
                target_train_uids,
                observation_cutoff=int(spec["support_origins"][0]),
            )
            target_context_cache[task_id] = public_context
        inventory = _inventory_rows(public_context)
        # E1-v3 Source-prior Scope routing repair: use the existing Runtime
        # applicability matcher against the current public fast_features.
        # A non-matching Task gets None, exactly like A3, so the whole Source
        # package (Card + evidence) stays out of the A5 payload.
        matched_a5_source_prior = _source_prior_for_task(
            a5_source_prior, public_context
        )
        source_prior_retrieval = {
            "runtime_matcher": "evaluate_applicability",
            "source_applicability": _runtime_source_applicability(
                a5_source_prior["source_card"]
            ),
            "matched": matched_a5_source_prior is not None,
            "source_package_entered_a5_payload": (
                matched_a5_source_prior is not None
            ),
        }
        arm_order = [("A3", a3_source_prior), ("A5", matched_a5_source_prior)]
        if spec["arm_order"] == "A5_A3":
            arm_order = list(reversed(arm_order))
        arm_rows: dict[str, Any] = {}
        for arm, source_prior in arm_order:
            arm_rows[arm] = _run_arm(
                repo_root=repo_root,
                arm_state=arm_states[arm],
                task_spec=spec,
                public_context=public_context,
                source_prior=source_prior,
                inventory=inventory,
                values=target_values,
                mapped_roster=mapped_roster,
                config=config,
                eval_uids=eval_uids,
                llm_counter=llm_counter,
            )
        return {
            "task_episode_id": spec["task_episode_id"],
            "support_origins": list(spec["support_origins"]),
            "delayed_origins": list(spec["delayed_origins"]),
            "horizon": HORIZON,
            "arm_order": spec["arm_order"],
            "public_context": public_context,
            "source_prior_retrieval": source_prior_retrieval,
            "A3": arm_rows["A3"],
            "A5": arm_rows["A5"],
            "non_source_payload_identical": (
                _normalized_payload_fingerprint(arm_rows["A3"]["payload"])
                == _normalized_payload_fingerprint(arm_rows["A5"]["payload"])
            ),
        }

    for task_index, spec in enumerate(task_roster[:N0]):
        print(f"E1V2_TASK_START {spec['task_episode_id']}", flush=True)
        row = _run_task(spec)
        if task_index == 0 and not row["non_source_payload_identical"]:
            raise RuntimeError(
                "E1-v2 Task 1 arm inputs differ outside the Source prior block"
            )
        rows.append(row)
        print(
            f"E1V2_TASK_DONE {spec['task_episode_id']} "
            f"A3={row['A3']['stop_reason']} "
            f"A5={row['A5']['stop_reason']}",
            flush=True,
        )

    summary = _paired_summary(rows)
    plan = _sample_plan(
        summary, available_task_count=AVAILABLE_TASK_COUNT
    )
    if (
        plan["N_final_within_cap"]
        and plan["N_final_within_available_roster"]
        and plan["extension_count"] > 0
    ):
        for spec in task_roster[N0 : N0 + plan["extension_count"]]:
            print(f"E1V2_EXT_START {spec['task_episode_id']}", flush=True)
            rows.append(_run_task(spec))
            print(
                f"E1V2_EXT_DONE {spec['task_episode_id']} "
                f"A3={rows[-1]['A3']['stop_reason']} "
                f"A5={rows[-1]['A5']['stop_reason']}",
                flush=True,
            )
        summary = _paired_summary(rows)
        plan = _sample_plan(
            summary, available_task_count=AVAILABLE_TASK_COUNT
        )

    behavior_changed = any(
        row["A3"]["initial"].get("proposals") != row["A5"]["initial"].get("proposals")
        for row in rows
    )
    verdict = _verdict(rows, summary, plan, behavior_changed)

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "question": (
            "Can one Source-domain Skill Card plus bounded contrast Experience "
            "shorten Target cold-start without increasing Support harm?"
        ),
        "verdict": verdict,
        "protocol_repairs": {
            "per_arm_target_experience": True,
            "per_arm_active_local_skill": True,
            "horizon": HORIZON,
            "all_truth_windows_non_overlapping": bool(
                preflight["truth_window_non_overlap"]["pass"]
            ),
            "task1_input_identity": bool(
                rows and rows[0]["non_source_payload_identical"]
            ),
            "adaptive_history_allowed_to_diverge_after_task1": True,
            "history_isolation": summary["history_isolation"],
        },
        "preregistration": preregistration,
        "rows": rows,
        "summary": summary,
        "sample_plan": plan,
        "source_behavior_changed": behavior_changed,
        "delayed_comparison": (
            "readable" if summary["paired_draft_count"] >= 8
            else "DELAYED_COMPARISON_UNREADABLE"
        ),
        "llm_api_call_count": llm_counter[0],
        "wall_seconds": time.perf_counter() - started,
        "claim_scope": (
            "development paired pilot on one prequential Target trajectory; "
            "Task pairs are not independent replicates. No sealed confirmation "
            "was opened."
        ),
        "boundary": {
            "e1_v2_only": True,
            "e2_not_started": True,
            "sealed_confirmation_opened": False,
        },
        "e1_v1_preserved": v1_preserved,
    }
    report["historical_verdict_before_e1_v2"] = report.get("verdict")
    report["phase"] = "e1_v2"
    report["e1_v2"] = result
    report["verdict"] = verdict
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result
