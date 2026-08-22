from __future__ import annotations

import argparse
import hashlib
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from SelfEvolvingHarnessTS.contracts.canonical import (
    CANONICALIZATION_VERSION,
    canonical_json_bytes,
    canonical_json_document_bytes,
    canonical_sha256,
    canonical_text_bytes,
    parse_json_document,
)
from SelfEvolvingHarnessTS.contracts.harness import (
    HarnessSnapshot,
    MemoryEntry,
    SkillEntry,
    SkillKind,
    load_learned_skill_entry,
    load_memory_entry,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA


COMPILER_VERSION = "ttha-harness-compiler/1"
RETRIEVAL_COMPILER_VERSION = "ttha-retrieval-index/1"
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_REQUIRED_BOOTSTRAP_IDS = frozenset(
    {
        "inspect_and_localize",
        "build_contrastive_candidates",
        "select_or_identity_and_verify",
    }
)


@dataclass(frozen=True)
class _CompilationReceipt:
    snapshot: HarnessSnapshot
    snapshot_profile: str
    operator_bundle_sha: str
    canonicalizer_source_sha: str
    compiler_source_sha: str


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(nested) for nested in value]
    return value


def skill_entry_to_dict(skill: SkillEntry) -> dict[str, Any]:
    return {
        "schema_version": skill.schema_version,
        "skill_id": skill.skill_id,
        "skill_kind": skill.skill_kind.value,
        "revision": skill.revision,
        "body": skill.body,
        "observable_applicability": _plain(skill.observable_applicability),
        "allowed_tools": list(skill.allowed_tools),
        "risk_guards": _plain(skill.risk_guards),
    }


def memory_entry_to_dict(memory: MemoryEntry) -> dict[str, Any]:
    return {
        "schema_version": memory.schema_version,
        "memory_id": memory.memory_id,
        "revision": memory.revision,
        "body": memory.body,
        "observable_applicability": _plain(memory.observable_applicability),
        "risk_guards": _plain(memory.risk_guards),
    }


def snapshot_to_dict(snapshot: HarnessSnapshot) -> dict[str, Any]:
    return {
        "schema_version": snapshot.schema_version,
        "instruction": snapshot.instruction,
        "skills": [skill_entry_to_dict(skill) for skill in snapshot.skills],
        "memories": [memory_entry_to_dict(memory) for memory in snapshot.memories],
        "retrieval": _plain(snapshot.retrieval),
        "candidate_policy": _plain(snapshot.candidate_policy),
        "verification": _plain(snapshot.verification),
        "dependency_shas": _plain(snapshot.dependency_shas),
        "harness_content_sha": snapshot.harness_content_sha,
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
    }


def _canonical_file_sha(path: Path, *, kind: str) -> str:
    raw = path.read_bytes()
    if kind == "text":
        canonical = canonical_text_bytes(raw)
    elif kind == "json":
        canonical = canonical_json_document_bytes(raw)
    else:
        raise ValueError(f"unknown canonical file kind: {kind}")
    return hashlib.sha256(canonical).hexdigest()


def _load_json(path: Path) -> Any:
    return parse_json_document(canonical_json_document_bytes(path.read_bytes()))


def _load_lock(root: Path) -> dict[str, Any]:
    lock_path = root / "snapshot.lock.json"
    if not lock_path.is_file():
        return {}
    value = _load_json(lock_path)
    if not isinstance(value, dict):
        raise ValueError("snapshot lock must be a JSON object")
    return value


def _load_skills(root: Path) -> tuple[SkillEntry, ...]:
    bootstrap_root = root / "skills" / "bootstrap"
    learned_root = root / "skills" / "learned"
    bootstrap = [
        load_skill_entry(_load_json(path))
        for path in sorted(bootstrap_root.glob("*.json"), key=lambda item: item.as_posix())
    ]
    learned = [
        load_learned_skill_entry(_load_json(path))
        for path in sorted(learned_root.glob("*.json"), key=lambda item: item.as_posix())
    ]
    bootstrap_ids = {skill.skill_id for skill in bootstrap}
    if bootstrap_ids != _REQUIRED_BOOTSTRAP_IDS:
        raise ValueError(
            "bootstrap skill IDs must be exactly " + ", ".join(sorted(_REQUIRED_BOOTSTRAP_IDS))
        )
    if not all(skill.skill_kind is SkillKind.BOOTSTRAP_PROCEDURE for skill in bootstrap):
        raise ValueError("bootstrap directory may contain bootstrap_procedure skills only")
    skills = tuple(sorted((*bootstrap, *learned), key=lambda skill: skill.skill_id))
    skill_ids = [skill.skill_id for skill in skills]
    if len(skill_ids) != len(set(skill_ids)):
        raise ValueError("duplicate skill_id in Harness authoring")
    return skills


def _load_memories(root: Path) -> tuple[MemoryEntry, ...]:
    path = root / "memories.jsonl"
    if not path.is_file():
        raise ValueError("missing memories.jsonl")
    raw = path.read_bytes()
    rows: list[MemoryEntry] = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            normalized = parse_json_document(
                canonical_json_document_bytes(line.encode("utf-8"))
            )
            rows.append(load_memory_entry(normalized))
        except ValueError as exc:
            raise ValueError(f"invalid memories.jsonl row {line_number}: {exc}") from exc
    rows.sort(key=lambda memory: memory.memory_id)
    ids = [memory.memory_id for memory in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate memory_id in Harness authoring")
    return tuple(rows)


def _operator_bundle_sha() -> tuple[str, str]:
    operator_root = _PACKAGE_ROOT / "operators"
    sources = [
        {
            "path": path.relative_to(_PACKAGE_ROOT).as_posix(),
            "semantic_text_sha": _canonical_file_sha(path, kind="text"),
        }
        for path in sorted(operator_root.glob("*.py"), key=lambda item: item.as_posix())
    ]
    registry_sha = canonical_sha256(_plain(OPERATOR_METADATA))
    return canonical_sha256({"sources": sources, "operator_registry_sha": registry_sha}), registry_sha


def _dependency_shas() -> tuple[dict[str, str], str, str, str]:
    contracts_root = _PACKAGE_ROOT / "contracts"
    schema_root = contracts_root / "schemas"
    runtime_root = _PACKAGE_ROOT / "runtime"
    ttha_root = _PACKAGE_ROOT / "methods" / "ttha"
    canonicalizer_source_sha = _canonical_file_sha(contracts_root / "canonical.py", kind="text")
    compiler_source_sha = _canonical_file_sha(Path(__file__), kind="text")
    operator_bundle_sha, operator_registry_sha = _operator_bundle_sha()
    dependencies: dict[str, str] = {
        "canonicalizer_source": canonicalizer_source_sha,
        "compiler_source": compiler_source_sha,
        "operator_bundle": operator_bundle_sha,
        "operator_registry": operator_registry_sha,
        "candidate_contract": _canonical_file_sha(contracts_root / "candidate.py", kind="text"),
        "method_contract": _canonical_file_sha(contracts_root / "method.py", kind="text"),
        "observable_contract": _canonical_file_sha(contracts_root / "observables.py", kind="text"),
        "public_boundary_contract": _canonical_file_sha(
            contracts_root / "public_boundary.py", kind="text"
        ),
        "run_context_contract": _canonical_file_sha(
            contracts_root / "run_context.py", kind="text"
        ),
        "task_contract": _canonical_file_sha(contracts_root / "task.py", kind="text"),
        "surface_registry": _canonical_file_sha(Path(__file__).with_name("harness_surfaces.json"), kind="json"),
    }
    for path in sorted(schema_root.glob("*.json"), key=lambda item: item.name):
        dependencies[f"schema:{path.stem}"] = _canonical_file_sha(path, kind="json")
    for filename in (
        "agent_backend.py",
        "candidate_pool.py",
        "candidate_verification.py",
        "decision_trace.py",
        "executor.py",
        "llm_cache.py",
        "public_features.py",
    ):
        dependencies[f"runtime:{Path(filename).stem}"] = _canonical_file_sha(
            runtime_root / filename,
            kind="text",
        )
    for filename in (
        "agent_core.py",
        "fast_agent.py",
        "method.py",
        "public_tools.py",
        "retrieval.py",
        "schema_contracts.py",
        "slow_agent.py",
    ):
        dependencies[f"ttha:{Path(filename).stem}"] = _canonical_file_sha(
            ttha_root / filename,
            kind="text",
        )
    for path in sorted((ttha_root / "schemas").glob("*.json"), key=lambda item: item.name):
        dependencies[f"agent_schema:{path.stem}"] = _canonical_file_sha(path, kind="json")
    return dependencies, operator_bundle_sha, canonicalizer_source_sha, compiler_source_sha


def _validate_authoring_controls(
    retrieval: object,
    candidate_policy: object,
    verification: object,
) -> None:
    if not isinstance(retrieval, dict) or retrieval.get("schema_version") != "retrieval/1":
        raise ValueError("retrieval.json must use retrieval/1")
    capability = retrieval.get("capability")
    if not isinstance(capability, dict) or capability.get("kind") != "rule_lexical":
        raise ValueError("retrieval capability rule must be rule_lexical")
    top_k = capability.get("top_k")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
        raise ValueError("retrieval top_k must be a non-negative integer")
    if not isinstance(candidate_policy, dict) or candidate_policy.get("schema_version") != "candidate-policy/1":
        raise ValueError("candidate_policy.json must use candidate-policy/1")
    if candidate_policy.get("identity_slots") != 1:
        raise ValueError("candidate policy must reserve exactly one identity slot")
    total = candidate_policy.get("total_k")
    program_slots = candidate_policy.get("agent_program_slots")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or isinstance(program_slots, bool)
        or not isinstance(program_slots, int)
        or total != 1 + program_slots
    ):
        raise ValueError("candidate total_k must equal identity plus Agent program slots")
    if not isinstance(verification, dict) or verification.get("schema_version") != "verification/1":
        raise ValueError("verification.json must use verification/1")
    if verification.get("identity_unfilterable") is not True:
        raise ValueError("identity must be unfilterable")
    if verification.get("require_explicit_choice") is not True:
        raise ValueError("explicit candidate choice must be required")


# --------------------------------------------------------------------------
# scope_risk_guards: the Scope/Risk adoption gate (#19 EDIT_SURFACE_DEFECT
# suture, 2026-08-21).
#
# In #18 the ``scope_risk_guards`` key was a placebo: the surface catalog
# offered it, but no tracked runtime code read it.  This section is the
# execution side of that surface.  The list lives at ``/scope_risk_guards``
# in the verification document (surface
# ``verification.rules.scope_risk_guards``) and holds at most one guard --
# the minimal edit the surface exists for.  The gate reads the list off the
# active snapshot after the frozen v2 adoption ladder has produced its final
# plan and before that plan is recorded as adopted; the ladder itself is
# neither modified nor re-ranked.  Identity is unfilterable: callers never
# submit the identity plan to this gate, and the guard vocabulary has no way
# to remove the abstention option.
#
# One statistic, ``min_per_series_gain``, binds the measured
# per-evaluation-series gain vector.  It exists only because the #19 O1
# repair stopped the search instrument from projecting that vector away at
# its interface; the evaluator performs no measurement of its own.
GUARD_LIST_KEY = "scope_risk_guards"
# The abstention option the gate may never remove.  Named here rather
# than imported so the Harness does not depend on an experiment module.
IDENTITY_PROGRAM = "identity"
GUARD_ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
GUARD_WINDOWS = ("support", "delayed")
GUARD_STATISTICS = (
    "aggregate_gain",
    "harmed_series_count",
    "harmed_series_fraction",
    "total_harm",
    "gain_to_total_harm_ratio",
    "min_per_series_gain",
)
GUARD_COMPARATORS = ("lt", "le", "gt", "ge")
GUARD_ACTIONS = (
    "VETO_AND_FALL_BACK",
    "RECORD_ONLY",
    # A1 (2026-08-22): the middle action.  A veto throws the whole adoption
    # away to buy safety on a few series; this one keeps the adoption and
    # takes the crossing evaluation series out of the plan's scope, serving
    # them the identity plan instead.  It is legal only with
    # ``min_per_series_gain``, because that is the one statistic whose
    # reading names a per-series crossing set; every other statistic in the
    # vocabulary reduces the window to a single number and cannot say which
    # series to take out.
    "RESCOPE_MASK_HARMED_SERIES",
)
RESCOPE_ACTION = "RESCOPE_MASK_HARMED_SERIES"
RESCOPE_STATISTIC = "min_per_series_gain"
# The plan key this action writes.  It is deliberately NOT
# ``excluded_series``: that field indexes *training* series and is consumed
# by the search instrument's mask geometry, while a ``min_per_series_gain``
# crossing set indexes *evaluation* series.  The two namespaces are
# disjoint, so writing evaluation ids into ``excluded_series`` would mask
# nothing and report a rescope that did not happen -- the placebo shape #19
# exists to prevent.
ROUTED_SERIES_KEY = "identity_routed_eval_series"
GUARD_APPLIES_TO = ("every_adoption", "reused_skill_adoption_only")
_GUARD_FIELDS = frozenset(
    {
        "guard_id",
        "window",
        "statistic",
        "comparator",
        "threshold",
        "action",
        "applies_to",
        "rationale",
    }
)


def _validate_scope_risk_guards(verification: object) -> None:
    """Compile-time validation of the verification document's guard list.

    The key is optional and defaults to the empty list; when present it
    holds at most one guard -- the single-entry constraint is what keeps the
    Scope/Risk surface minimal.
    """
    if not isinstance(verification, dict):
        return  # the authoring-controls validation above reports the shape
    guards = verification.get(GUARD_LIST_KEY)
    if guards is None:
        return
    if not isinstance(guards, list):
        raise ValueError("scope_risk_guards must be a list")
    if len(guards) > 1:
        raise ValueError(
            "scope_risk_guards holds at most one guard: the minimal legal "
            "edit on the Scope/Risk surface is exactly one entry"
        )
    for guard in guards:
        if not isinstance(guard, dict):
            raise ValueError("a scope_risk_guards entry must be an object")
        unknown = sorted(set(guard) - _GUARD_FIELDS)
        if unknown:
            raise ValueError(
                "scope_risk_guards entry carries unknown keys: " + ", ".join(unknown)
            )
        missing = sorted(_GUARD_FIELDS - set(guard))
        if missing:
            raise ValueError(
                "scope_risk_guards entry misses keys: " + ", ".join(missing)
            )
        if not isinstance(guard["guard_id"], str) or not GUARD_ID_PATTERN.match(
            guard["guard_id"]
        ):
            raise ValueError("guard_id must be a canonical id")
        if guard["window"] not in GUARD_WINDOWS:
            raise ValueError("guard window must be one of %s" % (GUARD_WINDOWS,))
        if guard["statistic"] not in GUARD_STATISTICS:
            raise ValueError("guard statistic must be one of %s" % (GUARD_STATISTICS,))
        if guard["comparator"] not in GUARD_COMPARATORS:
            raise ValueError(
                "guard comparator must be one of %s" % (GUARD_COMPARATORS,)
            )
        threshold = guard["threshold"]
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
        ):
            raise ValueError("guard threshold must be a finite number")
        if guard["action"] not in GUARD_ACTIONS:
            raise ValueError("guard action must be one of %s" % (GUARD_ACTIONS,))
        if (
            guard["action"] == RESCOPE_ACTION
            and guard["statistic"] != RESCOPE_STATISTIC
        ):
            raise ValueError(
                "%s needs the %s statistic: it is the only reading that names "
                "which evaluation series to take out of scope"
                % (RESCOPE_ACTION, RESCOPE_STATISTIC)
            )
        if guard["applies_to"] not in GUARD_APPLIES_TO:
            raise ValueError(
                "guard applies_to must be one of %s" % (GUARD_APPLIES_TO,)
            )
        rationale = guard["rationale"]
        if not isinstance(rationale, str) or not 1 <= len(rationale) <= 600:
            raise ValueError("guard rationale must be 1..600 characters")


def scope_risk_guards_of(snapshot: HarnessSnapshot) -> list[dict[str, Any]]:
    """Every Scope/Risk guard the active snapshot declares.

    Only the verification document's list is a registered surface; a
    skill-side key with the same name is inert and is not read here.
    """
    verification = snapshot.verification
    if not isinstance(verification, Mapping):
        return []
    guards = verification.get(GUARD_LIST_KEY)
    if not isinstance(guards, Sequence) or isinstance(guards, (str, bytes)):
        return []
    return [
        {"where": "verification.rules.scope_risk_guards", "guard": _plain(row)}
        for row in guards
        if isinstance(row, Mapping)
    ]


def guard_statistic(name: str, gains: Mapping[str, Any], eval_count: int) -> float:
    """Reduce one plan's window gains to the named guard statistic.

    Every branch reads a field the search instrument already measured; the
    evaluator measures nothing itself.  ``min_per_series_gain`` reads the
    per-series vector the #19 O1 repair passes through the interface.
    """
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
        vector = gains.get("per_eval_series_gain")
        if not isinstance(vector, Mapping) or not vector:
            raise ValueError(
                "min_per_series_gain needs the measured per-series gain "
                "vector under per_eval_series_gain; this gains dict does "
                "not carry it"
            )
        return min(float(value) for value in vector.values())
    raise ValueError("unknown guard statistic: %s" % name)


def guard_crossing_series(
    guard: Mapping[str, Any], gains: Mapping[str, Any]
) -> list[str]:
    """The evaluation series whose own gain satisfies the guard's test.

    ``min_per_series_gain`` is the minimum of a vector the search instrument
    already measured; this reader keeps the same comparison and asks it of
    every entry instead of the minimum.  Nothing is measured here.
    """
    vector = gains.get("per_eval_series_gain")
    if not isinstance(vector, Mapping) or not vector:
        raise ValueError(
            "a per-series crossing set needs the measured per-series gain "
            "vector under per_eval_series_gain; this gains dict does not "
            "carry it"
        )
    return sorted(
        str(uid)
        for uid, value in vector.items()
        if guard_fires(guard, float(value))
    )


def project_gains_with_identity_routing(
    gains: Mapping[str, Any], routed: Sequence[str]
) -> dict[str, Any]:
    """One window's gains when the named evaluation series are served identity.

    This is arithmetic on an already-measured vector, not a measurement.
    Gain is defined against the identity plan, so a series served identity
    has gain exactly zero, and a series left in scope keeps the number the
    instrument measured for it -- the batch is fitted on the training pool,
    so taking one evaluation series off the plan does not move another's
    reading.  ``aggregate_gain`` is the mean over origins of the mean over
    evaluation views, which is the mean of the per-series vector taken in
    the other order; the check below refuses to project any gains dict where
    that identity does not hold to floating-point rounding.

    ``per_origin_gain`` is not projected: this reader does not have the
    per-origin-by-series table it would need, and inventing one would be a
    measurement.  The key is dropped rather than carried forward stale.
    """
    vector = gains.get("per_eval_series_gain")
    if not isinstance(vector, Mapping) or not vector:
        raise ValueError(
            "identity routing needs the measured per-series gain vector"
        )
    measured = {str(uid): float(value) for uid, value in vector.items()}
    aggregate = float(gains["aggregate_gain"])
    mean_of_vector = sum(measured.values()) / len(measured)
    if abs(aggregate - mean_of_vector) > 1e-9:
        raise ValueError(
            "aggregate_gain (%r) is not the mean of the measured per-series "
            "vector (%r); this window cannot be projected without measuring"
            % (aggregate, mean_of_vector)
        )
    off = {str(uid) for uid in routed}
    unknown = sorted(off - set(measured))
    if unknown:
        raise ValueError(
            "cannot route evaluation series the window never measured: "
            + ", ".join(unknown)
        )
    after = {uid: (0.0 if uid in off else value) for uid, value in measured.items()}
    harmed = [
        str(uid)
        for uid in (gains.get("harmed_eval_series") or ())
        if str(uid) not in off
    ]
    return {
        "aggregate_gain": sum(after.values()) / len(after),
        "harmed_eval_series_count": len(harmed),
        "harmed_eval_series_total_harm": float(
            -sum(after[uid] for uid in harmed)
        ),
        "harmed_eval_series": sorted(harmed),
        "per_eval_series_gain": after,
        # Named and null rather than absent: a reader that finds the key
        # missing cannot tell a projection from a measurement that forgot
        # to carry it, and a reader that finds the pre-rescope list would
        # be reading a number for a plan that no longer exists.
        "per_origin_gain": None,
        "per_origin_gain_is_null_because": (
            "projecting the per-origin vector would need the "
            "origin-by-series table, which this reader does not have and "
            "will not invent; the measured per-origin list belongs to the "
            "un-rescoped plan and is not carried forward"
        ),
        "projected_from_measured_vector": True,
        "identity_routed_eval_series": sorted(off),
    }


def guard_fires(guard: Mapping[str, Any], value: float) -> bool:
    """The guard fires when ``statistic <comparator> threshold`` holds."""
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
    raise ValueError("unknown guard comparator: %s" % comparator)


def evaluate_scope_risk_guards(
    *,
    snapshot: HarnessSnapshot,
    plan: Mapping[str, Any],
    support: Mapping[str, Any],
    delayed: Mapping[str, Any],
    eval_count: int,
    reused: bool,
) -> dict[str, Any]:
    """Run the declared guards over the ladder's final plan.

    Pure evaluation: given the plan and its already-measured Support and
    delayed windows, report every guard's reading and whether any veto
    fired.  Walking the fallback on a veto is the caller's runtime duty,
    not this function's; identity is never submitted here by the caller.
    """
    readings: list[dict[str, Any]] = []
    for row in scope_risk_guards_of(snapshot):
        guard = dict(row["guard"])
        if str(guard["applies_to"]) == "reused_skill_adoption_only" and not reused:
            readings.append(
                {
                    "guard_id": guard["guard_id"],
                    "where": row["where"],
                    "checked": False,
                    "fired": False,
                    "why_not_checked": (
                        "this adoption did not come from a recalled Skill"
                    ),
                }
            )
            continue
        window = str(guard["window"])
        gains = delayed if window == "delayed" else support
        value = guard_statistic(str(guard["statistic"]), gains, eval_count)
        fired = guard_fires(guard, value)
        crossing = None
        if fired and str(guard["action"]) == RESCOPE_ACTION:
            crossing = guard_crossing_series(guard, gains)
        readings.append(
            {
                "guard_id": guard["guard_id"],
                "where": row["where"],
                "checked": True,
                "plan": _plain(plan),
                "window": window,
                "statistic": guard["statistic"],
                "value": (None if value == float("inf") else float(value)),
                "value_is_infinite": value == float("inf"),
                "comparator": guard["comparator"],
                "threshold": float(guard["threshold"]),
                "fired": bool(fired),
                "action": guard["action"],
                "crossing_eval_series": crossing,
            }
        )
    vetoed = [
        row
        for row in readings
        if row.get("fired") and row.get("action") == "VETO_AND_FALL_BACK"
    ]
    rescoped = [
        row
        for row in readings
        if row.get("fired") and row.get("action") == RESCOPE_ACTION
    ]
    crossing = sorted(
        {uid for row in rescoped for uid in (row.get("crossing_eval_series") or ())}
    )
    return {
        "readings": readings,
        "any_fired": any(row.get("fired") for row in readings),
        "vetoed": bool(vetoed),
        "vetoed_by": [str(row["guard_id"]) for row in vetoed],
        "rescope_requested": bool(rescoped),
        "rescope_requested_by": [str(row["guard_id"]) for row in rescoped],
        "crossing_eval_series": crossing,
    }


def _walk_rescope(
    *,
    out: dict[str, Any],
    first: Mapping[str, Any],
    plan: Mapping[str, Any],
    support: Mapping[str, Any],
    delayed: Mapping[str, Any],
    snapshot: HarnessSnapshot,
    eval_count: int,
    reused: bool,
) -> dict[str, Any]:
    """The RESCOPE branch of the enforcement walk.

    A veto answers "this plan is unsafe" by throwing the adoption away.
    This action answers the narrower question the reading actually supports:
    the plan is unsafe *on these series*, so those series are served the
    identity plan and the rest of the batch keeps the adopted plan.  No new
    measurement is taken -- both windows are projected off the vectors the
    instrument already measured -- and the guard is re-read on the rescoped
    plan, so a guard that would still fire escalates to identity rather than
    being declared satisfied.
    """
    routed = [str(uid) for uid in (first.get("crossing_eval_series") or ())]
    window = "delayed"
    for row in first.get("readings") or ():
        if row.get("fired") and row.get("action") == RESCOPE_ACTION:
            window = str(row.get("window") or "delayed")
            break
    measured = delayed if window == "delayed" else support
    every = sorted(
        str(uid) for uid in (measured.get("per_eval_series_gain") or {})
    )
    other = support if window == "delayed" else delayed

    def project_other(routed_uids: Sequence[str]) -> tuple[dict[str, Any], str | None]:
        """The window the guard did not read, projected when it can be.

        A banked episode carries the measured per-series vector for the
        window its guard reads and often nothing for the other one.  That
        window's reading is reported, never decided on, so a missing vector
        is recorded rather than raised -- and the reading is carried forward
        unprojected so nobody mistakes it for a post-rescope number.
        """
        try:
            return project_gains_with_identity_routing(other, routed_uids), None
        except ValueError as exc:
            return dict(other), str(exc)

    out["rescope"] = {
        "action": RESCOPE_ACTION,
        "read_on_window": window,
        "crossing_eval_series": sorted(routed),
        "evaluation_series": every,
        "series_kept_in_scope": [uid for uid in every if uid not in set(routed)],
        "measurement_taken": "none: both windows are projected",
        "projection": (
            "methods/ttha/harness/compiler.py::"
            "project_gains_with_identity_routing"
        ),
    }
    if not routed:
        out["why"] = (
            "the guard fired but named no crossing series, so there is "
            "nothing to take out of scope"
        )
        return out
    if set(routed) == set(every):
        collapsed_measured = project_gains_with_identity_routing(measured, every)
        collapsed_other, collapsed_note = project_other(every)
        out["rescope"]["other_window_not_projected"] = collapsed_note
        out.update(
            {
                "plan_after": {"program": IDENTITY_PROGRAM, "excluded_series": []},
                "support_after": (
                    collapsed_measured if window == "support" else collapsed_other
                ),
                "delayed_after": (
                    collapsed_measured if window == "delayed" else collapsed_other
                ),
                "changed": True,
                "fallback_source": (
                    "identity: every evaluation series crossed, so a rescope "
                    "and a veto are the same decision"
                ),
                "why": (
                    "every evaluation series crossed; the rescope collapses "
                    "to identity"
                ),
            }
        )
        out["rescope"]["collapsed_to_identity"] = True
        return out

    rescoped_plan = {
        "program": str(plan["program"]),
        "excluded_series": list(plan.get("excluded_series") or ()),
        ROUTED_SERIES_KEY: sorted(routed),
    }
    measured_after = project_gains_with_identity_routing(measured, routed)
    other_after, other_note = project_other(routed)
    out["rescope"]["other_window_not_projected"] = other_note
    support_after = measured_after if window == "support" else other_after
    delayed_after = measured_after if window == "delayed" else other_after
    second = evaluate_scope_risk_guards(
        snapshot=snapshot,
        plan=rescoped_plan,
        support=support_after,
        delayed=delayed_after,
        eval_count=int(eval_count),
        reused=bool(reused),
    )
    out["check_on_rescoped_plan"] = second
    if second["any_fired"]:
        out["rescope"]["escalated"] = (
            "the same guard still fires on the rescoped plan, so the rescope "
            "is abandoned and identity decides"
        )
        escalated_measured = project_gains_with_identity_routing(measured, every)
        escalated_other, _ = project_other(every)
        out["plan_after"] = {"program": IDENTITY_PROGRAM, "excluded_series": []}
        out["support_after"] = (
            escalated_measured if window == "support" else escalated_other
        )
        out["delayed_after"] = (
            escalated_measured if window == "delayed" else escalated_other
        )
        out["changed"] = True
        out["fallback_source"] = "identity: the rescoped plan fired the same guard"
        out["why"] = "a rescope was asked for but did not clear the guard"
        return out
    out.update(
        {
            "plan_after": dict(rescoped_plan),
            "support_after": support_after,
            "delayed_after": delayed_after,
            "changed": True,
            "fallback_source": (
                "the adopted plan, rescoped: %d of %d evaluation series "
                "served identity" % (len(routed), len(every))
            ),
            "why": (
                "a guard asked for a rescope, so the crossing series were "
                "taken out of the plan's scope and the adoption stood"
            ),
        }
    )
    return out


def enforce_scope_risk_guards(
    *,
    snapshot: HarnessSnapshot,
    ladder: Mapping[str, Any],
    eval_count: int,
    reused: bool,
    delayed_of: Callable[[str, Sequence[str]], Mapping[str, Any]],
    support_of: Callable[[str, Sequence[str]], Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate the declared guards and walk the fallback on a veto.

    #19 left the reading here and the acting in the experiment runner, so a
    guard could be read but never obeyed outside that one script.  This is
    the other half: the whole decision -- discovery, evaluation, which
    fallback the veto walks to, re-checking that fallback, and the identity
    floor -- now lives in tracked machinery, next to the verification
    document it is a rule of.

    Measurement stays outside on purpose.  ``delayed_of`` and ``support_of``
    are the caller's own instrument, injected as callables taking
    ``(program, excluded_series)`` and returning that plan's window gains;
    the Harness must not own a Consumer.  Everything this function decides,
    it decides from the ladder receipt and those two readings.

    ``ladder`` is the frozen v2 adoption ladder's own receipt, read and not
    rewritten: ``final_plan``, ``support``, ``delayed``, ``support_winner``
    and ``support_winner_full_batch_delayed``.  The ladder is never
    re-ranked, and identity is never submitted to a guard: it is
    unfilterable, so abstention cannot be taken away.
    """
    guards = scope_risk_guards_of(snapshot)
    plan = _plain(ladder["final_plan"])
    support = _plain(ladder.get("support") or {})
    delayed = _plain(ladder.get("delayed") or {})
    out: dict[str, Any] = {
        "guards_declared": guards,
        "guard_count": len(guards),
        "gate": "methods/ttha/harness/compiler.py::enforce_scope_risk_guards",
        "evaluator": (
            "methods/ttha/harness/compiler.py::evaluate_scope_risk_guards"
        ),
        "measurement_is_the_callers": True,
        "plan_before": dict(plan),
        "identity_never_vetoed": True,
        "checked": False,
        "changed": False,
        "plan_after": dict(plan),
        "support_after": support,
        "delayed_after": delayed,
    }
    if not guards:
        out["why"] = "the active snapshot declares no Scope/Risk guard"
        return out
    if str(plan["program"]) == IDENTITY_PROGRAM:
        out["why"] = "identity is unfilterable and is never submitted to a guard"
        return out

    first = evaluate_scope_risk_guards(
        snapshot=snapshot,
        plan=plan,
        support=support,
        delayed=delayed,
        eval_count=int(eval_count),
        reused=bool(reused),
    )
    out["checked"] = True
    out["check_on_adopted_plan"] = first
    if first.get("rescope_requested") and not first["vetoed"]:
        return _walk_rescope(
            out=out, first=first, plan=plan, support=support, delayed=delayed,
            snapshot=snapshot, eval_count=int(eval_count), reused=bool(reused),
        )
    if not first["vetoed"]:
        out["why"] = (
            "a guard fired but only asked for a record"
            if first["any_fired"] else "no guard fired"
        )
        return out

    winner = ladder.get("support_winner")
    winner_delayed = ladder.get("support_winner_full_batch_delayed")
    excluded = list(plan.get("excluded_series") or ())
    fallback = {"program": IDENTITY_PROGRAM, "excluded_series": []}
    fallback_source = "identity"
    if (
        winner is not None
        and winner_delayed is not None
        and float(winner_delayed) > 0.0
        and not (str(winner) == str(plan["program"]) and not excluded)
    ):
        fallback = {"program": str(winner), "excluded_series": []}
        fallback_source = "the ladder's Support winner, full batch"

    fallback_delayed = _plain(
        delayed_of(str(fallback["program"]), list(fallback["excluded_series"]))
    )
    fallback_support = _plain(
        support_of(str(fallback["program"]), list(fallback["excluded_series"]))
    )
    second = None
    if str(fallback["program"]) != IDENTITY_PROGRAM:
        second = evaluate_scope_risk_guards(
            snapshot=snapshot,
            plan=fallback,
            support=fallback_support,
            delayed=fallback_delayed,
            eval_count=int(eval_count),
            reused=bool(reused),
        )
        if second["vetoed"]:
            fallback = {"program": IDENTITY_PROGRAM, "excluded_series": []}
            fallback_source = "identity: the fallback candidate fired the same guard"
            fallback_delayed = _plain(delayed_of(IDENTITY_PROGRAM, []))
            fallback_support = _plain(support_of(IDENTITY_PROGRAM, []))
    out.update(
        {
            "check_on_fallback": second,
            "fallback_source": fallback_source,
            "plan_after": dict(fallback),
            "support_after": fallback_support,
            "delayed_after": fallback_delayed,
            "changed": True,
            "why": "a veto fired, so the frozen v2 fallback was walked",
        }
    )
    return out


def _compile(root: Path) -> _CompilationReceipt:
    root = Path(root).resolve()
    lock = _load_lock(root)
    profile = str(lock.get("snapshot_profile", "evolving"))
    instruction_path = root / "instruction.md"
    if not instruction_path.is_file():
        raise ValueError("missing instruction.md")
    instruction = canonical_text_bytes(instruction_path.read_bytes()).decode("utf-8")
    skills = _load_skills(root)
    memories = _load_memories(root)
    if profile == "h0-domain-naive":
        if memories:
            raise ValueError("H0 must have empty memory")
        if any(skill.skill_kind is not SkillKind.BOOTSTRAP_PROCEDURE for skill in skills):
            raise ValueError("H0 capability library must be empty")
    retrieval = _load_json(root / "retrieval.json")
    candidate_policy = _load_json(root / "candidate_policy.json")
    verification = _load_json(root / "verification.json")
    _validate_authoring_controls(retrieval, candidate_policy, verification)
    _validate_scope_risk_guards(verification)
    resolved_retrieval = {
        **retrieval,
        "resolved_skill_index": [
            {
                "skill_id": skill.skill_id,
                "skill_kind": skill.skill_kind.value,
                "revision": skill.revision,
            }
            for skill in skills
        ],
        "resolved_memory_index": [
            {"memory_id": memory.memory_id, "revision": memory.revision}
            for memory in memories
        ],
    }
    content = {
        "schema_version": "harness-content/1",
        "instruction": instruction,
        "skills": [skill_entry_to_dict(skill) for skill in skills],
        "memories": [memory_entry_to_dict(memory) for memory in memories],
        "retrieval": resolved_retrieval,
        "candidate_policy": candidate_policy,
        "verification": verification,
    }
    harness_content_sha = canonical_sha256(content)
    dependencies, operator_bundle_sha, canonicalizer_source_sha, compiler_source_sha = _dependency_shas()
    runtime_bundle_sha = canonical_sha256(
        {
            "schema_version": "runtime-bundle/1",
            "harness_content_sha": harness_content_sha,
            "operator_bundle_sha": operator_bundle_sha,
            "dependency_shas": dependencies,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "compiler_version": COMPILER_VERSION,
            "retrieval_compiler_version": RETRIEVAL_COMPILER_VERSION,
        }
    )
    snapshot = HarnessSnapshot(
        schema_version="harness-snapshot/1",
        instruction=instruction,
        skills=skills,
        memories=memories,
        retrieval=resolved_retrieval,
        candidate_policy=candidate_policy,
        verification=verification,
        dependency_shas=dependencies,
        harness_content_sha=harness_content_sha,
        runtime_bundle_sha=runtime_bundle_sha,
    )
    return _CompilationReceipt(
        snapshot=snapshot,
        snapshot_profile=profile,
        operator_bundle_sha=operator_bundle_sha,
        canonicalizer_source_sha=canonicalizer_source_sha,
        compiler_source_sha=compiler_source_sha,
    )


def _lock_payload(receipt: _CompilationReceipt) -> dict[str, Any]:
    snapshot = receipt.snapshot
    return {
        "schema_version": "snapshot-lock/1",
        "snapshot_profile": receipt.snapshot_profile,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "canonicalizer_source_sha": receipt.canonicalizer_source_sha,
        "compiler_version": COMPILER_VERSION,
        "compiler_source_sha": receipt.compiler_source_sha,
        "retrieval_compiler_version": RETRIEVAL_COMPILER_VERSION,
        "harness_content_sha": snapshot.harness_content_sha,
        "runtime_bundle_sha": snapshot.runtime_bundle_sha,
        "operator_bundle_sha": receipt.operator_bundle_sha,
        "dependency_shas": _plain(snapshot.dependency_shas),
    }


def compile_snapshot(root: Path, verify_lock: bool = True) -> HarnessSnapshot:
    root = Path(root).resolve()
    receipt = _compile(root)
    if verify_lock:
        actual = _load_lock(root)
        expected = _lock_payload(receipt)
        if actual != expected:
            raise ValueError("snapshot lock mismatch; run compiler with --write-lock")
    return receipt.snapshot


def compile_compatible_snapshot(
    root: Path,
    *,
    expected_harness_content_sha: str,
) -> HarnessSnapshot:
    """Rebind immutable authoring content to the current runtime dependency set."""

    snapshot = compile_snapshot(root, verify_lock=False)
    if snapshot.harness_content_sha != expected_harness_content_sha:
        raise ValueError("compatibility compile changed Harness semantic content")
    return snapshot


def write_lock(root: Path) -> Path:
    root = Path(root).resolve()
    receipt = _compile(root)
    path = root / "snapshot.lock.json"
    path.write_bytes(canonical_json_bytes(_lock_payload(receipt)) + b"\n")
    return path


def _main() -> int:
    parser = argparse.ArgumentParser(description="Compile a TTHA Harness snapshot")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    if args.write_lock:
        write_lock(args.root)
    snapshot = compile_snapshot(args.root)
    print(f"harness_content_sha={snapshot.harness_content_sha}")
    print(f"runtime_bundle_sha={snapshot.runtime_bundle_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "COMPILER_VERSION",
    "GUARD_LIST_KEY",
    "IDENTITY_PROGRAM",
    "RETRIEVAL_COMPILER_VERSION",
    "compile_snapshot",
    "compile_compatible_snapshot",
    "enforce_scope_risk_guards",
    "evaluate_scope_risk_guards",
    "guard_fires",
    "guard_statistic",
    "memory_entry_to_dict",
    "scope_risk_guards_of",
    "skill_entry_to_dict",
    "snapshot_to_dict",
    "write_lock",
]
