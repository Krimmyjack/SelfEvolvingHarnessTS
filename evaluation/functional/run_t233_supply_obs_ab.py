"""T233 UNGUIDED supply under two observation contracts, one driver, one A/B.

Why this driver exists
----------------------
The original T233 independent UNGUIDED supply driver -- the one that produced
``artifacts/functional/e2/t233_independent_source_supply.json``
(``protocol_version t233_independent_source_supply_v1``) -- is not recoverable.
No module in this repository produces that artifact, ``git log --all`` records
no path containing ``t233``, and no ``.py`` file was created or modified during
the run window.  A parameter-identical rerun therefore cannot be claimed.

Rather than re-author a driver and call the result a faithful replay, this
driver runs *both* observation contracts itself, so the question that matters
-- did the four M0b mechanism-geometry fields change exploration and evidence
formation -- is answered inside one driver, one budget and one Task roster:

  ``OLD_OBS``   the four M0b field names are removed at the Workspace gateway
                boundary, so the Agent reasons over the pre-M0b feature surface;
  ``NEW_OBS``   the workspace as it stands, with M0b wired.

Driver confounding for that paired contrast is zero.  The contrast against the
historical v1 rows is reported as reference only, and labelled
driver-confounded wherever it appears.

``OLD_OBS`` doubles as the driver-faithfulness check: if the v1 census shape
reappears under the old observation surface, the v1 phenomenon was not an
artifact of the driver that produced it.

Everything below the gateway boundary is the existing cold-arm code path:
``runner._run_arm`` with ``source_prior=None``, the existing probe budget,
tool budget, Judge, Runtime, stage contracts and Episode/Skill lifecycle.
This module contributes no operator, no threshold and no Harness surface; the
one behaviour it owns is deleting four field names from a tool payload.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

for _path in (
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / "evaluation" / "functional"),
    str(PROJECT_ROOT / "methods" / "ttha"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore
from SelfEvolvingHarnessTS.methods.ttha.public_tools import PublicToolReceipt

from evaluation.functional.task_episode_harness import g1
from evaluation.functional.task_episode_harness.agentic import runner as runner_mod
from evaluation.functional.task_episode_harness.agentic.gateway import (
    CohortScopePublicToolGateway,
)
from evaluation.functional.task_episode_harness.e1 import (
    B,
    HORIZON,
    MATERIAL_THRESHOLD,
    _ArmState,
    _frozen_task_roster,
    _inventory_rows,
    _skill_ids,
)

PROTOCOL_VERSION = "t233_supply_obs_ab_v1"
COHORT_NAME = "T233"
STATE_REL = ".t233_supply_obs_ab_state"
REPORT_JSON = (
    PROJECT_ROOT / "artifacts/functional/e2" / "t233_supply_obs_ab_v1.json"
)
REPORT_MD = (
    PROJECT_ROOT / "artifacts/functional/e2" / "t233_supply_obs_ab_v1.md"
)
HISTORICAL_V1 = (
    PROJECT_ROOT / "artifacts/functional/e2"
    / "t233_independent_source_supply.json"
)

OLD_OBS = "OLD_OBS"
NEW_OBS = "NEW_OBS"
OBS_ARMS = (OLD_OBS, NEW_OBS)

# Both arms are cold, UNGUIDED and Source-free, so both run on the existing
# A3 cold-arm token.  Episode ids are arm-scoped, and the two arms keep fully
# separate _ArmState objects and SnapshotStore roots, so an id shared across
# two independent store roots is the case e1.py already documents as legitimate
# and not a cross-arm merge.
EPISODE_ARM_TOKEN = runner_mod.COLD_ARM

# The four fields M0b added to the public observation contract.
M0B_FIELDS = (
    "level_region_fraction",
    "level_region_end_fraction",
    "outlier_region_end_fraction",
    "level_only_post_shift_support_sufficient",
)

# Guardrail only.  ``_run_arm`` asks for runner.LLM_CALL_BUDGET_PER_ARM_TASK
# (24); this caps that request per Task per arm.  v1 observed 4-11 LLM calls
# per Task, so the cap is not expected to bind; whether it bound is recorded
# per row rather than assumed.
LLM_GUARDRAIL_PER_TASK_ARM = 20

# Stop the run rather than burn Support budget against a dead relay.
CONSECUTIVE_INFRASTRUCTURE_FAILURE_LIMIT = 3

# The grounding-rule rejection code, from fast_agent._validate_inspect_hypotheses.
UNGROUNDED_CODE = "HYPOTHESIS_EVIDENCE_UNGROUNDED"

# Per-stage repair budget.  1 is the harness default and what exec1 ran on; the
# forensics showed 13 of 13 fatal errors were retry-1 exhaustions on a
# recoverable output slip, so the supplementary executions ask for 2.
DEFAULT_VALIDATION_RETRIES = 1

# The precheck threshold this report reads.  The repository's own rule,
# g1.GENERAL_EVIDENCE_MIN_DISTINCT_TASKS, is unchanged and reported alongside.
PRECHECK_MIN_DISTINCT_TASKS = 3


# --------------------------------------------------------------- masking glue
_UNSET = "UNSET"


class _ArmLocal(threading.local):
    mask_new_fields: Any = _UNSET
    last_gateway: Any = None
    last_backend: Any = None


_ARM_LOCAL = _ArmLocal()


def _strip_m0b(value: Any) -> Any:
    """Drop the four M0b field names, wherever they sit in a payload."""
    if isinstance(value, Mapping):
        return {
            str(key): _strip_m0b(nested)
            for key, nested in value.items()
            if str(key) not in M0B_FIELDS
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [_strip_m0b(nested) for nested in value]
    return value


def _m0b_names_in(value: Any) -> list[str]:
    """Every occurrence of an M0b field name, as a key or as a string value."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in M0B_FIELDS:
                found.append(str(key))
            found.extend(_m0b_names_in(nested))
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for nested in value:
            found.extend(_m0b_names_in(nested))
    elif isinstance(value, str) and value in M0B_FIELDS:
        found.append(value)
    return found


class _ObservationArmGateway(CohortScopePublicToolGateway):
    """The existing gateway, with one arm's observation contract narrowed.

    ``OLD_OBS`` removes the four M0b names from the served receipt *and* from
    the served-feature cache, because the inspect grounding rule reads that
    cache through ``observed_feature_keys`` / ``observed_feature_values``.
    Masking only the receipt would leave the Agent able to cite a field it was
    never shown, which is exactly the pre-M0b behaviour this arm must not have.

    The information wall, the tool set, the budget and the refusal semantics
    are untouched: this subclass deletes names, and does nothing else.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        mode = _ARM_LOCAL.mask_new_fields
        if mode is _UNSET:
            raise RuntimeError(
                "observation arm was not declared before the gateway was built"
            )
        self.mask_new_fields = bool(mode)
        self.masked_receipts = 0
        self.receipts_with_all_m0b_fields = 0
        self.audit_violations: list[dict[str, Any]] = []
        _ARM_LOCAL.last_gateway = self

    def call(self, name: str, arguments: Mapping[str, Any]) -> PublicToolReceipt:
        receipt = super().call(name, arguments)
        if not self.mask_new_fields:
            if name == "summarize_series" and receipt.ok:
                present = set(_m0b_names_in(receipt.public_result))
                if present == set(M0B_FIELDS):
                    self.receipts_with_all_m0b_fields += 1
                else:
                    self.audit_violations.append({
                        "arm": NEW_OBS,
                        "violation": "SUMMARIZE_RECEIPT_MISSING_M0B_FIELD",
                        "tool_name": name,
                        "missing": sorted(set(M0B_FIELDS) - present),
                    })
            return receipt
        for cached in self._features.values():
            if isinstance(cached, dict):
                for field in M0B_FIELDS:
                    cached.pop(field, None)
        stripped = _strip_m0b(receipt.public_result)
        masked = PublicToolReceipt.create(
            tool_name=receipt.tool_name,
            arguments=receipt.arguments,
            public_result=stripped,
            context_sha=self.context_sha,
            ok=receipt.ok,
        )
        leaked = _m0b_names_in(masked.public_result)
        if leaked:
            raise RuntimeError(
                "OLD_OBS mask leaked %r through a %s receipt"
                % (sorted(set(leaked)), receipt.tool_name)
            )
        self.masked_receipts += 1
        return masked


def _backend_factory(maximum_calls: int) -> Any:
    """The existing backend, with the per-Task-per-arm guardrail applied."""
    capped = min(int(maximum_calls), LLM_GUARDRAIL_PER_TASK_ARM)
    backend = runner_mod._default_backend_factory(capped)
    _ARM_LOCAL.last_backend = backend
    return backend


# ----------------------------------------------------------------- one episode
def _fresh_arm_state(
    repo_root: Path, obs_arm: str, task_id: str, state_dir: str | None = None
) -> tuple[_ArmState, Any]:
    """A store root and an h0 snapshot that no other Task or arm has touched."""
    snapshot = compile_snapshot(
        repo_root / "methods/ttha/harness/h0", verify_lock=False
    )
    store = SnapshotStore(
        repo_root / STATE_REL / (state_dir or obs_arm) / task_id / "snapshots"
    )
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    return (
        _ArmState(
            arm=EPISODE_ARM_TOKEN,
            memories=[],
            episodes=[],
            store=store,
            active_snapshot=snapshot,
            active_skill_ids=_skill_ids(snapshot, local_only=True),
        ),
        snapshot,
    )


def _families(program: Sequence[str], categories: Mapping[str, str]) -> list[str]:
    return sorted({str(categories.get(op, "unknown")) for op in program})


def _slim_arm_result(
    result: Mapping[str, Any],
    *,
    obs_arm: str,
    snapshot: Any,
    gateway: Any,
    categories: Mapping[str, str],
) -> dict[str, Any]:
    """The row fields the census reads, without the full stage transcripts."""
    inspect_payload: Mapping[str, Any] = {}
    for stage in result.get("stages") or ():
        if stage.get("stage") == "inspect":
            inspect_payload = stage.get("payload") or {}
            break

    hypotheses: list[dict[str, Any]] = []
    for raw in inspect_payload.get("pattern_hypotheses") or ():
        if not isinstance(raw, Mapping):
            continue
        cited = [str(name) for name in (raw.get("evidence_features") or ())]
        hypotheses.append({
            "hypothesis_id": raw.get("hypothesis_id"),
            "pattern_type": raw.get("pattern_type"),
            "region_fractions": raw.get("region_fractions"),
            "evidence_features": cited,
            "confidence": raw.get("confidence"),
            "cited_m0b_fields": sorted(set(cited) & set(M0B_FIELDS)),
        })

    proposals = [
        {
            "candidate_id": entry.get("candidate_id"),
            "addresses_hypothesis_id": entry.get("addresses_hypothesis_id"),
            "program": [str(step["op"]) for step in entry.get("steps") or ()],
        }
        for entry in result.get("proposals") or ()
    ]
    for entry in proposals:
        entry["families"] = _families(entry["program"], categories)

    probes = []
    for entry in result.get("probes") or ():
        program = [str(step["op"]) for step in entry.get("steps") or ()]
        gain = entry.get("support_gain")
        probes.append({
            "candidate_id": entry.get("candidate_id"),
            "attempt_index": entry.get("attempt_index"),
            "status": entry.get("status"),
            "program": program,
            "families": _families(program, categories),
            "support_gain": float(gain) if gain is not None else None,
            "meets_material_threshold": entry.get("meets_material_threshold"),
            "episode_id": entry.get("episode_id"),
        })

    # Kept so a mask-induced grounding failure cannot hide inside a protocol
    # error: if OLD_OBS ever fails because the Agent cited a field the mask
    # removed, it shows up here as HYPOTHESIS_EVIDENCE_UNGROUNDED.
    #
    # ``stages`` gains its entry only after a stage returns, so counting error
    # codes from it alone reports 0 for the fatal case -- the one this counter
    # exists to catch.  The terminal code of a stage that died is read
    # separately and added, and is reported so the two are never conflated.
    stage_validation = [
        {
            "stage": stage.get("stage"),
            "first_pass_valid": stage.get("first_pass_valid"),
            "validation_retry_count": stage.get("validation_retry_count"),
            "validation_error_codes": list(
                stage.get("validation_error_codes") or ()
            ),
        }
        for stage in result.get("stages") or ()
    ]
    terminal_code = result.get("terminal_validation_error_code")
    recovered_ungrounded = sum(
        1
        for stage in stage_validation
        for code in stage["validation_error_codes"]
        if str(code) == UNGROUNDED_CODE
    )
    fatal_ungrounded = int(str(terminal_code or "") == UNGROUNDED_CODE)

    retrieved = dict(result.get("retrieved_knowledge_summary") or {})
    metrics = dict(result.get("metrics") or {})
    observed_keys = sorted(
        {
            str(key)
            for row in result.get("tool_observations") or ()
            for key in ((row.get("public_result") or {}).get("features") or {})
        }
    )

    return {
        "observation_arm": obs_arm,
        "episode_arm_token": EPISODE_ARM_TOKEN,
        "started_from_h0": snapshot.runtime_bundle_sha,
        "skills_inherited": [],
        "stop_reason": result.get("stop_reason"),
        "chosen_candidate_id": result.get("chosen_candidate_id"),
        "protocol_error": result.get("protocol_error"),
        "infrastructure_error": result.get("infrastructure_error"),
        "stage_validation": stage_validation,
        "terminal_validation_error_code": terminal_code,
        "ungrounded_citation_rejections": (
            recovered_ungrounded + fatal_ungrounded
        ),
        "ungrounded_citation_rejections_recovered": recovered_ungrounded,
        "ungrounded_citation_rejections_fatal": fatal_ungrounded,
        "pattern_hypotheses": hypotheses,
        "cites_any_m0b_field": any(h["cited_m0b_fields"] for h in hypotheses),
        "proposals": proposals,
        "probes": probes,
        "winner": result.get("winner"),
        "lifecycle_method_stage": (
            (result.get("lifecycle") or {}).get("method_event") or {}
        ).get("stage"),
        "observed_public_feature_keys": observed_keys,
        "observation_audit": {
            "mask_active": bool(getattr(gateway, "mask_new_fields", False)),
            "masked_receipts": int(getattr(gateway, "masked_receipts", 0)),
            "receipts_with_all_m0b_fields": int(
                getattr(gateway, "receipts_with_all_m0b_fields", 0)
            ),
            "m0b_names_in_any_receipt": sorted(
                set(_m0b_names_in(result.get("tool_observations") or []))
            ),
            "violations": list(getattr(gateway, "audit_violations", ())),
        },
        "independence": {
            "local_skills_at_start": [],
            "target_local_skill_count_retrieved": int(
                retrieved.get("target_local_skill_count") or 0
            ),
            "retrieved_risk_skill_ids": list(
                retrieved.get("retrieved_risk_skill_ids") or ()
            ),
            "source_prior_matched": bool(retrieved.get("source_prior_matched")),
            "raw_episodes_in_fast_payload": int(
                retrieved.get("raw_episodes_in_fast_payload") or 0
            ),
        },
        "metrics": metrics,
        "llm_guardrail_reached": bool(
            int(metrics.get("llm_calls") or 0) >= LLM_GUARDRAIL_PER_TASK_ARM
        ),
        "workspace_tool_accounting": (result.get("cost") or {}).get(
            "workspace_tools"
        ),
    }


def _run_one_task(
    *,
    repo_root: Path,
    spec: Mapping[str, Any],
    context: Mapping[str, Any],
    cohort: Mapping[str, Any],
    config: Mapping[str, Any],
    categories: Mapping[str, str],
    obs_arms: Sequence[str] = OBS_ARMS,
    exec_label: str = "",
    validation_retries: int = DEFAULT_VALIDATION_RETRIES,
) -> dict[str, Any]:
    task_id = str(spec["task_episode_id"])
    inventory = _inventory_rows(context)
    row: dict[str, Any] = {
        "task_episode_id": task_id,
        "observation_cutoff": int(context["observation_cutoff"]),
        "support_origins": [int(o) for o in spec["support_origins"]],
        "delayed_origins": [int(o) for o in spec["delayed_origins"]],
        "scope_series_uids": list(context.get("scope_series_uids") or ()),
        "task_signature": dict(context["task_signature"]),
        g1.G1_CONDITION_FEATURE: bool(
            (context.get("task_fast_features") or {}).get(
                g1.G1_CONDITION_FEATURE, False
            )
        ),
        "arms": {},
    }
    for obs_arm in obs_arms:
        _ARM_LOCAL.mask_new_fields = obs_arm == OLD_OBS
        _ARM_LOCAL.last_gateway = None
        arm_state, snapshot = _fresh_arm_state(
            repo_root, obs_arm, task_id,
            state_dir=f"{obs_arm}_{exec_label}" if exec_label else obs_arm,
        )
        started = time.perf_counter()
        try:
            result = runner_mod._run_arm(
                repo_root=repo_root,
                arm_state=arm_state,
                task_spec=spec,
                public_context=context,
                cohort=cohort,
                config=config,
                inventory=inventory,
                source_prior=None,
                workspace_tool_budget=runner_mod.WORKSPACE_TOOL_BUDGET,
                backend_factory=_backend_factory,
                validation_retries=validation_retries,
            )
        except Exception as exc:  # noqa: BLE001
            row["arms"][obs_arm] = {
                "observation_arm": obs_arm,
                "started_from_h0": snapshot.runtime_bundle_sha,
                "driver_exception": "%s: %s" % (type(exc).__name__, exc),
                "counts_as_infrastructure_failure": True,
                "wall_seconds": time.perf_counter() - started,
            }
            print(
                "OBSAB_ARM_FAILED %s %s %s: %s"
                % (task_id, obs_arm, type(exc).__name__, exc),
                flush=True,
            )
            continue
        finally:
            _ARM_LOCAL.mask_new_fields = _UNSET
        slim = _slim_arm_result(
            result,
            obs_arm=obs_arm,
            snapshot=snapshot,
            gateway=_ARM_LOCAL.last_gateway,
            categories=categories,
        )
        slim["wall_seconds"] = time.perf_counter() - started
        row["arms"][obs_arm] = slim
        print(
            "OBSAB_ARM_DONE %s %s stop=%s tools=%d llm=%d probes=%d "
            "active=%d m0b_cited=%s"
            % (
                task_id, obs_arm, slim["stop_reason"],
                slim["metrics"].get("workspace_tool_calls", 0),
                slim["metrics"].get("llm_calls", 0),
                slim["metrics"].get("real_support_probe_count", 0),
                slim["metrics"].get("task_local_active", 0),
                slim["cites_any_m0b_field"],
            ),
            flush=True,
        )
    return row


# ------------------------------------------------------------------- census
def _cell_key(program: Sequence[str]) -> str:
    return "+".join(str(op) for op in program)


def _arm_census(
    rows: Sequence[Mapping[str, Any]], obs_arm: str
) -> dict[str, Any]:
    """Program x context evidence cells, in the v1 authorization shape.

    Positive / negative / immaterial follow the unchanged material threshold,
    and the unit of evidence is the distinct Task Episode, not the probe.
    """
    cells: dict[tuple[str, bool], dict[str, set[str]]] = {}
    probe_total = 0
    rls_probes = 0
    outlier_probes = 0
    for row in rows:
        arm = (row.get("arms") or {}).get(obs_arm) or {}
        if arm.get("driver_exception"):
            continue
        condition = bool(row.get(g1.G1_CONDITION_FEATURE, False))
        task_id = str(row["task_episode_id"])
        for probe in arm.get("probes") or ():
            if probe.get("status") != "PROBED":
                continue
            probe_total += 1
            program = list(probe.get("program") or ())
            if "repair_level_shift" in program:
                rls_probes += 1
            if "outlier" in (probe.get("families") or ()):
                outlier_probes += 1
            gain = probe.get("support_gain")
            if gain is None:
                continue
            gain = float(gain)
            bucket = cells.setdefault(
                (_cell_key(program), condition),
                {"positive": set(), "negative": set(), "immaterial": set()},
            )
            if gain >= MATERIAL_THRESHOLD:
                bucket["positive"].add(task_id)
            elif gain <= -MATERIAL_THRESHOLD:
                bucket["negative"].add(task_id)
            else:
                bucket["immaterial"].add(task_id)

    out_cells: list[dict[str, Any]] = []
    for (program, condition), bucket in sorted(
        cells.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        positive = len(bucket["positive"])
        negative = len(bucket["negative"])
        # Every probe here is UNGUIDED by construction, so the unguided counts
        # are the pooled counts; both are written out so the shape matches the
        # v1 cells this is compared against.
        out_cells.append({
            "program": program,
            "context_condition": condition,
            "pooled_positive": positive,
            "pooled_negative": negative,
            "pooled_immaterial": len(bucket["immaterial"]),
            "unguided_positive": positive,
            "unguided_negative": negative,
            "conditioned_positive": 0,
            "conditioned_negative": 0,
            "positive_task_ids": sorted(bucket["positive"]),
            "negative_task_ids": sorted(bucket["negative"]),
            "immaterial_task_ids": sorted(bucket["immaterial"]),
            "leave_one_out_minimum_positive": max(positive - 1, 0),
            "meets_precheck_threshold": bool(
                positive >= PRECHECK_MIN_DISTINCT_TASKS and negative == 0
            ),
            "meets_repo_min_distinct_tasks": bool(
                positive >= g1.GENERAL_EVIDENCE_MIN_DISTINCT_TASKS
                and negative == 0
            ),
        })

    per_task: list[dict[str, Any]] = []
    for row in rows:
        arm = (row.get("arms") or {}).get(obs_arm) or {}
        if arm.get("driver_exception"):
            per_task.append({
                "task_episode_id": row["task_episode_id"],
                "driver_exception": arm["driver_exception"],
            })
            continue
        proposals = arm.get("proposals") or ()
        probed = [p for p in arm.get("probes") or () if p.get("status") == "PROBED"]
        preferred = list(proposals)[0] if proposals else None
        per_task.append({
            "task_episode_id": row["task_episode_id"],
            "stop_reason": arm.get("stop_reason"),
            "preferred_program": (
                preferred["program"] if preferred else None
            ),
            "preferred_families": (
                preferred["families"] if preferred else []
            ),
            "proposed_families": sorted(
                {f for entry in proposals for f in entry.get("families") or ()}
            ),
            "probed_families": sorted(
                {f for entry in probed for f in entry.get("families") or ()}
            ),
            "probed_programs": [entry["program"] for entry in probed],
            "charged_probe_cost": (arm.get("metrics") or {}).get(
                "charged_probe_cost"
            ),
            "real_support_probe_count": (arm.get("metrics") or {}).get(
                "real_support_probe_count"
            ),
            "cites_any_m0b_field": arm.get("cites_any_m0b_field"),
            "cited_m0b_fields": sorted(
                {
                    name
                    for h in arm.get("pattern_hypotheses") or ()
                    for name in h.get("cited_m0b_fields") or ()
                }
            ),
        })

    scored = [
        row for row in rows
        if not ((row.get("arms") or {}).get(obs_arm) or {}).get(
            "driver_exception", False
        )
        and obs_arm in (row.get("arms") or {})
    ]
    return {
        "observation_arm": obs_arm,
        "tasks_scored": len(scored),
        "tasks_with_a_probe": sum(
            1 for row in scored
            if any(
                p.get("status") == "PROBED"
                for p in (row["arms"][obs_arm].get("probes") or ())
            )
        ),
        "probe_total": probe_total,
        "repair_level_shift_probes": rls_probes,
        "outlier_family_probes": outlier_probes,
        "repair_level_shift_probe_share": (
            rls_probes / probe_total if probe_total else None
        ),
        "outlier_family_probe_share": (
            outlier_probes / probe_total if probe_total else None
        ),
        "cells": out_cells,
        "precheck_cells": [c for c in out_cells if c["meets_precheck_threshold"]],
        "per_task": per_task,
        "tasks_citing_m0b_field": sorted(
            entry["task_episode_id"] for entry in per_task
            if entry.get("cites_any_m0b_field")
        ),
        "llm_calls": sum(
            int((row["arms"][obs_arm].get("metrics") or {}).get("llm_calls") or 0)
            for row in scored
        ),
    }


_LATE_TASKS = tuple("e1v2_task_%02d" % i for i in range(13, 20))


def _outlier_exploration(census: Mapping[str, Any]) -> dict[str, Any]:
    late = [
        entry for entry in census["per_task"]
        if str(entry.get("task_episode_id")) in _LATE_TASKS
    ]
    return {
        "task_window": list(_LATE_TASKS),
        "tasks_in_window_scored": len(late),
        "probed_outlier_family": sorted(
            entry["task_episode_id"] for entry in late
            if "outlier" in (entry.get("probed_families") or ())
        ),
        "proposed_outlier_family": sorted(
            entry["task_episode_id"] for entry in late
            if "outlier" in (entry.get("proposed_families") or ())
        ),
        "any_outlier_exploration": any(
            "outlier" in (entry.get("probed_families") or ())
            for entry in late
        ),
    }


def _primary_contrast(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    old_outlier = _outlier_exploration(old)
    new_outlier = _outlier_exploration(new)
    return {
        "note": (
            "One driver, one roster, one budget; the only declared difference "
            "is whether the four M0b fields reach the Agent. This contrast is "
            "not driver-confounded."
        ),
        "outlier_family_exploration_task_13_19": {
            OLD_OBS: old_outlier,
            NEW_OBS: new_outlier,
            "changed": (
                old_outlier["any_outlier_exploration"]
                != new_outlier["any_outlier_exploration"]
            ),
        },
        "repair_level_shift_probe_share": {
            OLD_OBS: old["repair_level_shift_probe_share"],
            NEW_OBS: new["repair_level_shift_probe_share"],
            "old_probes": [
                old["repair_level_shift_probes"], old["probe_total"]
            ],
            "new_probes": [
                new["repair_level_shift_probes"], new["probe_total"]
            ],
        },
        "precheck_cell_count": {
            OLD_OBS: len(old["precheck_cells"]),
            NEW_OBS: len(new["precheck_cells"]),
            "threshold": {
                "min_distinct_unguided_positive_tasks":
                    PRECHECK_MIN_DISTINCT_TASKS,
                "opposing_evidence_allowed": False,
                "repo_general_evidence_min_distinct_tasks":
                    g1.GENERAL_EVIDENCE_MIN_DISTINCT_TASKS,
            },
            "%s_cells" % OLD_OBS: [
                {"program": c["program"],
                 "context_condition": c["context_condition"],
                 "unguided_positive": c["unguided_positive"]}
                for c in old["precheck_cells"]
            ],
            "%s_cells" % NEW_OBS: [
                {"program": c["program"],
                 "context_condition": c["context_condition"],
                 "unguided_positive": c["unguided_positive"]}
                for c in new["precheck_cells"]
            ],
        },
        "m0b_field_citation_rate": {
            OLD_OBS: {
                "tasks_citing": old["tasks_citing_m0b_field"],
                "rate": (
                    len(old["tasks_citing_m0b_field"]) / old["tasks_scored"]
                    if old["tasks_scored"] else None
                ),
                "expected": 0,
            },
            NEW_OBS: {
                "tasks_citing": new["tasks_citing_m0b_field"],
                "rate": (
                    len(new["tasks_citing_m0b_field"]) / new["tasks_scored"]
                    if new["tasks_scored"] else None
                ),
            },
        },
        "authorization_actions_taken": [],
        "authorization_note": (
            "Precheck only. No TRY or Skill was written, no authorization "
            "artifact was modified and no Promotion was performed."
        ),
    }


def _historical_reference(
    old: Mapping[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    """v1 rows, read for reference only.  Driver-confounded and labelled so."""
    reference: dict[str, Any] = {
        "confounded": True,
        "confound": (
            "The v1 driver is unrecoverable, so this driver is a rewrite. Any "
            "difference against v1 mixes the observation change with the "
            "driver change and cannot attribute either."
        ),
        "source_artifact": HISTORICAL_V1.name,
    }
    try:
        payload = json.loads(HISTORICAL_V1.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reference["available"] = False
        reference["error"] = "%s: %s" % (type(exc).__name__, exc)
        return reference

    v1_rows = payload.get("rows") or []
    probe_total = 0
    rls = 0
    late_outlier: list[str] = []
    for row in v1_rows:
        task_id = str(row.get("task_episode_id"))
        for probe in row.get("probes") or ():
            program = list(probe.get("program") or ())
            probe_total += 1
            if "repair_level_shift" in program:
                rls += 1
            if task_id in _LATE_TASKS and any(
                op in ("outlier_mad", "outlier_iqr", "hampel_filter")
                for op in program
            ):
                late_outlier.append(task_id)
    reference.update({
        "available": True,
        "protocol_version": payload.get("protocol_version"),
        "task_count": payload.get("task_count"),
        "probe_total": probe_total,
        "repair_level_shift_probes": rls,
        "repair_level_shift_probe_share": (
            rls / probe_total if probe_total else None
        ),
        "outlier_family_tasks_13_19": sorted(set(late_outlier)),
        "this_run_repair_level_shift_probe_share": {
            OLD_OBS: old["repair_level_shift_probe_share"],
            NEW_OBS: new["repair_level_shift_probe_share"],
        },
        "driver_faithfulness_read": (
            "If OLD_OBS reproduces the v1 concentration, the v1 exploration "
            "shape was not an artifact of the driver that produced it."
        ),
    })
    return reference


def _independence_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    shas: set[str] = set()
    violations: list[dict[str, Any]] = []
    for row in rows:
        for obs_arm, arm in (row.get("arms") or {}).items():
            if arm.get("driver_exception"):
                continue
            shas.add(str(arm.get("started_from_h0")))
            independence = arm.get("independence") or {}
            if independence.get("source_prior_matched"):
                violations.append({
                    "task_episode_id": row["task_episode_id"],
                    "observation_arm": obs_arm,
                    "violation": "SOURCE_PRIOR_MATCHED",
                })
            if int(independence.get("target_local_skill_count_retrieved") or 0):
                violations.append({
                    "task_episode_id": row["task_episode_id"],
                    "observation_arm": obs_arm,
                    "violation": "TARGET_LOCAL_SKILL_RETRIEVED",
                })
            if independence.get("retrieved_risk_skill_ids"):
                violations.append({
                    "task_episode_id": row["task_episode_id"],
                    "observation_arm": obs_arm,
                    "violation": "RISK_SKILL_RETRIEVED",
                })
            if int(independence.get("raw_episodes_in_fast_payload") or 0):
                violations.append({
                    "task_episode_id": row["task_episode_id"],
                    "observation_arm": obs_arm,
                    "violation": "RAW_EPISODES_IN_FAST_PAYLOAD",
                })
    return {
        "every_task_and_arm_starts_from": sorted(shas),
        "single_h0_start": len(shas) <= 1,
        "cross_task_skill_inheritance": False,
        "cross_arm_skill_inheritance": False,
        "cross_task_guidance_inheritance": False,
        "source_prior_offered": False,
        "attempts_per_task": 1,
        "reran_any_task_for_its_result": False,
        "no_skill_was_in_scope_when_a_task_began": not violations,
        "violations": violations,
        "note": (
            "Every probe in both arms is UNGUIDED by construction: each Task "
            "and each arm compiles its own h0 snapshot into its own store "
            "root, so no non-bootstrap Skill can exist when a Task begins and "
            "no evidence here was produced under a clause the Harness had "
            "already written."
        ),
    }


def _observation_audit_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    masked_receipts = 0
    unmasked_full = 0
    ungrounded = {OLD_OBS: 0, NEW_OBS: 0}
    ungrounded_fatal = {OLD_OBS: 0, NEW_OBS: 0}
    ungrounded_recovered = {OLD_OBS: 0, NEW_OBS: 0}
    protocol_errors: dict[str, list[dict[str, Any]]] = {
        OLD_OBS: [], NEW_OBS: []
    }
    for row in rows:
        for obs_arm, arm in (row.get("arms") or {}).items():
            if arm.get("driver_exception"):
                continue
            ungrounded[obs_arm] = ungrounded.get(obs_arm, 0) + int(
                arm.get("ungrounded_citation_rejections") or 0
            )
            ungrounded_fatal[obs_arm] = ungrounded_fatal.get(obs_arm, 0) + int(
                arm.get("ungrounded_citation_rejections_fatal") or 0
            )
            ungrounded_recovered[obs_arm] = ungrounded_recovered.get(
                obs_arm, 0
            ) + int(arm.get("ungrounded_citation_rejections_recovered") or 0)
            if arm.get("protocol_error"):
                protocol_errors.setdefault(obs_arm, []).append({
                    "task_episode_id": row["task_episode_id"],
                    "protocol_error": arm["protocol_error"],
                    "terminal_validation_error_code": arm.get(
                        "terminal_validation_error_code"
                    ),
                })
            audit = arm.get("observation_audit") or {}
            violations.extend(audit.get("violations") or ())
            if obs_arm == OLD_OBS:
                masked_receipts += int(audit.get("masked_receipts") or 0)
                leaked = audit.get("m0b_names_in_any_receipt") or []
                if leaked:
                    violations.append({
                        "task_episode_id": row["task_episode_id"],
                        "observation_arm": OLD_OBS,
                        "violation": "M0B_NAME_IN_RECEIPT",
                        "names": list(leaked),
                    })
                if arm.get("cites_any_m0b_field"):
                    violations.append({
                        "task_episode_id": row["task_episode_id"],
                        "observation_arm": OLD_OBS,
                        "violation": "M0B_FIELD_CITED_UNDER_MASK",
                    })
            else:
                unmasked_full += int(
                    audit.get("receipts_with_all_m0b_fields") or 0
                )
    return {
        "masked_field_names": list(M0B_FIELDS),
        "mask_boundary": (
            "Workspace tool gateway: the four names are deleted from the "
            "served receipt and from the served-feature cache the inspect "
            "grounding rule reads, before the receipt is minted."
        ),
        "old_obs_masked_receipts": masked_receipts,
        "new_obs_summarize_receipts_with_all_four_fields": unmasked_full,
        "zero_violations": not violations,
        "violations": violations,
        "mask_artifact_check": {
            "question": (
                "Could the mask itself have broken OLD_OBS by leaving the "
                "Agent instructed to cite a field it can no longer fetch?"
            ),
            "instruction_text_identical_across_arms": True,
            "instruction_evidence": (
                "M0b left harness_content_sha unchanged and moved only "
                "observable_contract and runtime:public_features, so the h0 "
                "instruction and bootstrap Skill text name none of the four "
                "fields; under the mask the Agent never sees the names at all."
            ),
            "ungrounded_citation_rejections": ungrounded,
            "ungrounded_citation_rejections_recovered": ungrounded_recovered,
            "ungrounded_citation_rejections_fatal": ungrounded_fatal,
            "reads_as_mask_artifact": bool(ungrounded.get(OLD_OBS, 0)),
            "protocol_errors": protocol_errors,
        },
        "not_masked_and_why": (
            "task_fast_features drives deterministic Skill retrieval, not the "
            "Agent's prompt, and both arms begin every Task from h0 with zero "
            "learned Skills, so the extra keys have nothing to match and "
            "cannot create a behavioural difference. The independence block "
            "asserts the zero-Skill precondition per Task and per arm."
        ),
    }


def _pinned_parameters(
    *,
    obs_arms: Sequence[str] = OBS_ARMS,
    exec_label: str = "",
    validation_retries: int = DEFAULT_VALIDATION_RETRIES,
) -> dict[str, Any]:
    return {
        "execution_label": exec_label or "exec1",
        "observation_arms_run": list(obs_arms),
        "stage_validation_retries": validation_retries,
        "instrument_fixes_relative_to_exec1": {
            "stage_validation_retries": (
                "fast_path._run_stage / run_agentic_fast_path and "
                "runner._run_arm now take a validation_retries keyword. The "
                "default is 1, the value previously hard-coded at "
                "fast_path.py:292, so every other caller in the repository is "
                "byte-for-byte unchanged. Only this driver passes a different "
                "value, and only when asked to on the command line."
            ),
            "ungrounded_citation_rejections": (
                "The counter previously read only stage_validation, which is "
                "populated after a stage returns and is therefore empty "
                "whenever the stage died -- so a fatal grounding rejection "
                "was silently counted as 0. FastPathTrace now records the "
                "terminal validator error code, and the count is the sum of "
                "recovered and fatal, reported separately as "
                "ungrounded_citation_rejections_recovered and "
                "ungrounded_citation_rejections_fatal."
            ),
        },
        "note": (
            "Reused from the existing cold-arm code path, not invented here. "
            "v1's charged_probe_cost of 4 on non-active Tasks and 1 on active "
            "ones is exactly _run_arm's B+1 / real_probe_count rule at B=3, "
            "which is why these defaults are read as the v1 budgets too."
        ),
        "per_episode_primitive": "runner._run_arm",
        "cohort": COHORT_NAME,
        "task_roster": "e1._frozen_task_roster()[:19] (g1.A5A3_MAX_N)",
        "workspace_tool_budget": runner_mod.WORKSPACE_TOOL_BUDGET,
        "support_probe_budget_B": B,
        "material_threshold": MATERIAL_THRESHOLD,
        "horizon": HORIZON,
        "source_prior": None,
        "episode_arm_token": EPISODE_ARM_TOKEN,
        "llm_call_budget_requested_by_run_arm":
            runner_mod.LLM_CALL_BUDGET_PER_ARM_TASK,
        "llm_guardrail_applied_per_task_arm": LLM_GUARDRAIL_PER_TASK_ARM,
        "llm_guardrail_note": (
            "The driver caps _run_arm's request of "
            f"{runner_mod.LLM_CALL_BUDGET_PER_ARM_TASK} at "
            f"{LLM_GUARDRAIL_PER_TASK_ARM} per Task per arm. v1 observed 4-11 "
            "LLM calls per Task, so the cap is not expected to bind; each row "
            "records llm_guardrail_reached rather than assuming it did not."
        ),
        "model": runner_mod.NF_MODEL,
        "base_url": runner_mod.NF_BASE_URL,
        "consumer_and_judge": "unchanged paired evaluator via _probe_compiled",
        "thresholds_changed": [],
    }


def _write(payload: Mapping[str, Any], path: Path = REPORT_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _markdown(payload: Mapping[str, Any]) -> str:
    contrast = payload["primary_contrast"]
    old = payload["census"][OLD_OBS]
    new = payload["census"][NEW_OBS]
    outlier = contrast["outlier_family_exploration_task_13_19"]
    share = contrast["repair_level_shift_probe_share"]
    precheck = contrast["precheck_cell_count"]
    audit = payload["observation_audit"]
    lines = [
        "# T233 UNGUIDED supply under two observation contracts",
        "",
        "The driver is a rewrite. The original v1 driver is unrecoverable, so",
        "no parameter-identical replay is claimed. Both arms run on this one",
        "driver with the same roster, budgets, Judge and Runtime, and the only",
        "declared difference is whether the four M0b mechanism-geometry fields",
        "reach the Agent, so the primary contrast carries no driver confound.",
        "The comparison against the historical v1 rows is reference only and",
        "is labelled driver-confounded. KDD W3, NOAA and every sealed source",
        "were not read. No authorization action was taken: no TRY or Skill was",
        "written, no authorization artifact was modified, nothing was promoted.",
        "",
        "## What ran",
        "",
        f"- Tasks scored: {old['tasks_scored']} {OLD_OBS} / "
        f"{new['tasks_scored']} {NEW_OBS} "
        f"(of {payload['task_count_planned']} planned).",
        f"- LLM calls: {old['llm_calls']} {OLD_OBS} + {new['llm_calls']} "
        f"{NEW_OBS} = {old['llm_calls'] + new['llm_calls']}.",
        f"- Support probes: {old['probe_total']} {OLD_OBS} / "
        f"{new['probe_total']} {NEW_OBS}.",
        f"- Mask assertions clean: {audit['zero_violations']} "
        f"({audit['old_obs_masked_receipts']} masked receipts; "
        f"{audit['new_obs_summarize_receipts_with_all_four_fields']} "
        "unmasked summarize receipts carried all four fields).",
        f"- Independence held: "
        f"{payload['independence']['no_skill_was_in_scope_when_a_task_began']}"
        " -- every Task and arm began from h0 with zero learned Skills, so",
        "  every probe on both sides is UNGUIDED by construction.",
        f"- Mask-artifact check: ungrounded-citation rejections "
        f"{audit['mask_artifact_check']['ungrounded_citation_rejections']}. "
        "M0b left the h0",
        "  instruction text untouched, so under the mask the Agent never sees",
        "  the four names and cannot be broken by being told to cite them.",
        f"- h0 identity is {payload['independence']['every_task_and_arm_starts_from']},",
        "  which differs from v1's h0 by construction: M0b moved the observable",
        "  contract and the feature extractor. Both arms share it, so the",
        "  primary contrast is unaffected and only the v1 reference is.",
        "",
        "## Primary contrast, NEW_OBS vs OLD_OBS",
        "",
        f"- Outlier-family exploration on task_13..19: "
        f"{OLD_OBS} probed {outlier[OLD_OBS]['probed_outlier_family'] or 'none'}"
        f", {NEW_OBS} probed "
        f"{outlier[NEW_OBS]['probed_outlier_family'] or 'none'}. "
        f"Changed: {outlier['changed']}.",
        f"- repair_level_shift probe share: {share[OLD_OBS]} "
        f"({share['old_probes'][0]}/{share['old_probes'][1]}) -> "
        f"{share[NEW_OBS]} ({share['new_probes'][0]}/{share['new_probes'][1]}).",
        f"- Cells meeting the precheck threshold "
        f"(>= {precheck['threshold']['min_distinct_unguided_positive_tasks']} "
        f"distinct UNGUIDED positive Tasks, no opposing evidence): "
        f"{precheck[OLD_OBS]} {OLD_OBS} vs {precheck[NEW_OBS]} {NEW_OBS}.",
        f"- M0b field citation: {NEW_OBS} cited a new field on "
        f"{len(contrast['m0b_field_citation_rate'][NEW_OBS]['tasks_citing'])} "
        f"Tasks; {OLD_OBS} cited one on "
        f"{len(contrast['m0b_field_citation_rate'][OLD_OBS]['tasks_citing'])} "
        "(zero is the asserted expectation under the mask).",
        "",
        "## Reference only, against the historical v1 rows",
        "",
        f"- v1 repair_level_shift probe share: "
        f"{payload['historical_reference'].get('repair_level_shift_probe_share')}"
        f"; v1 outlier-family Tasks in 13..19: "
        f"{payload['historical_reference'].get('outlier_family_tasks_13_19')}.",
        "- Driver-confounded: a difference here mixes the observation change",
        "  with the driver rewrite and attributes neither. It is reported so",
        f"  {OLD_OBS} can act as the driver-faithfulness read, nothing more.",
        "",
        "## Standing uncertainty",
        "",
        "- The paired contrast is 19 Tasks on one already-exposed cohort with",
        "  one probe budget; it measures whether the fields changed behaviour",
        "  here, not that any resulting cell transfers anywhere.",
        "- Precheck is not authorization. Nothing in this run earns a TRY.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- driver
def run(
    *,
    task_count: int = 19,
    task_workers: int = 2,
    write: bool = True,
    obs_arms: Sequence[str] = OBS_ARMS,
    exec_label: str = "",
    validation_retries: int = DEFAULT_VALIDATION_RETRIES,
) -> dict[str, Any]:
    started = time.perf_counter()
    obs_arms = tuple(obs_arms)
    # The two-arm readouts are a contrast between OLD_OBS and NEW_OBS. A
    # single-arm supplementary execution has nothing to contrast, so those
    # blocks are omitted rather than filled with one arm's numbers.
    paired = len(obs_arms) == 2
    repo_root = PROJECT_ROOT
    cohort = runner_mod.load_cohort(repo_root, COHORT_NAME)
    config = runner_mod._config_for_cohort()
    specs = list(_frozen_task_roster()[:task_count])

    eval_pre = g1.eval_substrate_preflight(
        cohort["values"], cohort["eval_uids"], specs
    )
    train_pre = g1.train_substrate_preflight(
        cohort["values"], cohort["train_uids"],
        [int(a) for a in config["anchors"]],
    )
    if not (eval_pre["pass"] and train_pre["pass"]):
        return {
            "protocol_version": PROTOCOL_VERSION,
            "verdict": "OBSAB_SUBSTRATE_INVALID",
            "eval_substrate_preflight": eval_pre,
            "train_substrate_preflight": train_pre,
            "wall_seconds": time.perf_counter() - started,
        }

    # Contexts first, sequentially: identical for both arms, cached on disk by
    # the existing helper, and built before any concurrency so the shared
    # cache directory is never written by two threads at once.
    contexts: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []
    for spec in specs:
        task_id = str(spec["task_episode_id"])
        context = g1._w3_context_for(
            repo_root, STATE_REL, task_id,
            int(spec["support_origins"][0]),
            cohort["values"], cohort["train_uids"],
        )
        if not (context.get("scope_series_uids") or ()):
            skipped.append({"task_episode_id": task_id, "skipped": "EMPTY_SCOPE"})
            continue
        contexts[task_id] = context

    categories = {
        str(entry["name"]): str(entry.get("category") or "unknown")
        for entry in _inventory_rows(next(iter(contexts.values())))
    }

    runnable = [
        spec for spec in specs
        if str(spec["task_episode_id"]) in contexts
    ]

    # The one behaviour this driver owns, installed once.
    runner_mod.CohortScopePublicToolGateway = _ObservationArmGateway

    report_json = REPORT_JSON
    report_md = REPORT_MD
    if exec_label:
        report_json = REPORT_JSON.with_name(
            f"{REPORT_JSON.stem}_{exec_label}.json"
        )
        report_md = REPORT_MD.with_name(f"{REPORT_MD.stem}_{exec_label}.md")

    rows: list[dict[str, Any]] = []
    consecutive_failures = 0
    stopped_early: str | None = None
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, int(task_workers))
    for start in range(0, len(runnable), workers):
        if stopped_early:
            break
        chunk = runnable[start:start + workers]
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            futures = [
                pool.submit(
                    _run_one_task,
                    repo_root=repo_root,
                    spec=spec,
                    context=contexts[str(spec["task_episode_id"])],
                    cohort=cohort,
                    config=config,
                    categories=categories,
                    obs_arms=obs_arms,
                    exec_label=exec_label,
                    validation_retries=validation_retries,
                )
                for spec in chunk
            ]
            chunk_rows = [future.result() for future in futures]
        for row in chunk_rows:
            rows.append(row)
            failed = any(
                arm.get("driver_exception")
                or arm.get("infrastructure_error")
                for arm in (row.get("arms") or {}).values()
            )
            consecutive_failures = consecutive_failures + 1 if failed else 0
            if consecutive_failures >= CONSECUTIVE_INFRASTRUCTURE_FAILURE_LIMIT:
                stopped_early = (
                    "OBSAB_STOPPED_ON_%d_CONSECUTIVE_INFRASTRUCTURE_FAILURES"
                    % CONSECUTIVE_INFRASTRUCTURE_FAILURE_LIMIT
                )
        if write:
            _write({
                "protocol_version": PROTOCOL_VERSION,
                "verdict": "OBSAB_IN_PROGRESS",
                "execution_label": exec_label or "exec1",
                "tasks_completed": len(rows),
                "tasks_planned": len(runnable),
                "rows": rows,
            }, report_json)

    census = {arm: _arm_census(rows, arm) for arm in obs_arms}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "verdict": stopped_early or "OBSAB_COMPLETE",
        "execution_label": exec_label or "exec1",
        "cohort": COHORT_NAME,
        "exposure": cohort["exposure"],
        "development_replay": True,
        "driver": {
            "entry_point": "evaluation/functional/run_t233_supply_obs_ab.py",
            "is_a_rewrite": True,
            "v1_driver_recoverable": False,
            "v1_driver_search": (
                "No repository module produces "
                "t233_independent_source_supply_v1; git log --all records no "
                "path containing t233; no .py file was created or modified "
                "during the v1 run window."
            ),
            "arms_share_this_driver": True,
            "declared_arm_difference": (
                "whether the four M0b fields reach the Agent"
            ),
        },
        "arms": {
            arm: description
            for arm, description in (
                (OLD_OBS,
                 "the four M0b field names removed at the Workspace gateway "
                 "boundary; the pre-M0b public feature surface"),
                (NEW_OBS, "the workspace as it stands, M0b wired"),
            )
            if arm in obs_arms
        },
        "sealed_data_read": [],
        "sealed_note": (
            "Already-exposed T233 only. KDD W3, NOAA and g3_final_query_outcome "
            "were not opened."
        ),
        "task_count_planned": len(runnable),
        "task_count_scored": len(rows),
        "tasks_skipped": skipped,
        "arm_runs_completed": sum(
            1 for row in rows
            for arm in (row.get("arms") or {}).values()
            if not arm.get("driver_exception")
        ),
        "parallelism": {"tasks": workers, "arms_within_a_task": "sequential"},
        "pinned_parameters": _pinned_parameters(
            obs_arms=obs_arms,
            exec_label=exec_label,
            validation_retries=validation_retries,
        ),
        "independence": _independence_block(rows),
        "observation_audit": _observation_audit_block(rows),
        "eval_substrate_preflight": eval_pre,
        "train_substrate_preflight": train_pre,
        "census": census,
        "llm_calls_total": sum(census[arm]["llm_calls"] for arm in obs_arms),
        "rows": rows,
        "wall_seconds": time.perf_counter() - started,
    }
    if paired:
        payload["primary_contrast"] = _primary_contrast(
            census[OLD_OBS], census[NEW_OBS]
        )
        payload["historical_reference"] = _historical_reference(
            census[OLD_OBS], census[NEW_OBS]
        )
    if write:
        _write(payload, report_json)
        if paired:
            report_md.write_text(_markdown(payload), encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=19)
    parser.add_argument("--task-workers", type=int, default=2)
    parser.add_argument(
        "--no-write", action="store_true",
        help="run without writing either deliverable (smoke use)",
    )
    parser.add_argument(
        "--arms", default=",".join(OBS_ARMS),
        help=(
            "comma-separated observation arms to run; a single arm omits the "
            "two-arm contrast blocks and the markdown"
        ),
    )
    parser.add_argument(
        "--exec-label", default="",
        help=(
            "supplementary execution label; suffixes the per-arm state "
            "directory and both deliverable filenames"
        ),
    )
    parser.add_argument(
        "--validation-retries", type=int, default=DEFAULT_VALIDATION_RETRIES,
        help="per-stage repair budget (harness default 1)",
    )
    args = parser.parse_args(argv)
    obs_arms = tuple(
        part.strip() for part in str(args.arms).split(",") if part.strip()
    )
    unknown = [arm for arm in obs_arms if arm not in OBS_ARMS]
    if not obs_arms or unknown:
        parser.error(f"--arms must name {OBS_ARMS}; got {args.arms!r}")
    result = run(
        task_count=args.tasks,
        task_workers=args.task_workers,
        write=not args.no_write,
        obs_arms=obs_arms,
        exec_label=str(args.exec_label).strip(),
        validation_retries=int(args.validation_retries),
    )
    print(json.dumps(
        {
            key: value for key, value in result.items()
            if key not in {"rows", "census", "eval_substrate_preflight",
                           "train_substrate_preflight"}
        },
        indent=2, ensure_ascii=False, default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
