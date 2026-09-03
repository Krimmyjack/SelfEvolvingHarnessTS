"""Run the pre-registered NOAA development Slow Skill-revision trajectory.

This runner fills the missing P4-Evolution connection without reinterpreting
the completed Fast-only P4 experiment.  It is deliberately confined to the
already-exposed NOAA 2024 development arrays.  It starts K0-fixed and A5-Slow
from the same naturally formed target-local Skill, permits only one Program
body PATCH, uses one whole-trajectory B=8 ledger per arm, and keeps every
Final surface closed.

The first complete pass is mechanism evidence only.  It cannot release H3;
the only permitted follow-up is the separately pre-registered repeat.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from SelfEvolvingHarnessTS.contracts.harness import EditOperation
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import classify_relation
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
    _parse_frozen_steps,
    public_operator_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import (
    compile_snapshot,
    skill_entry_to_dict,
)
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (
    run_online_round,
    source_skill_of_candidate,
)
from SelfEvolvingHarnessTS.methods.ttha.program_supply import (
    VerifiedProgramSupplyAssessment,
    bind_verified_program_options,
    build_single_surface_catalog,
    retrieved_relevant_capability_skill_ids,
    route_verified_program_supply_fault,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent

from evaluation.functional import run_e2_fresh_confirmation as noaa
from evaluation.functional import run_e2_s1_curriculum_four_arms as four_arms
from evaluation.functional import run_e2_s2a_forecast_curriculum as forecast_course
from evaluation.functional import run_e2_t6_cls_op_shared_harness as shared_harness
from evaluation.main_protocol_p1 import run_forecast_p1 as forecast_p1
from evaluation.main_protocol_p4 import run_forecast_p4_performance as performance
from evaluation.main_protocol_p4 import run_p4 as split_release


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRIMARY_OUT_JSON = (
    PROJECT_ROOT
    / "artifacts/main_protocol/"
    "p4_forecast_evolution_slow_revision_noaa_dev_20260831.json"
)
REPEAT_OUT_JSON = (
    PROJECT_ROOT
    / "artifacts/main_protocol/"
    "p4_forecast_evolution_slow_revision_noaa_dev_repeat_20260831.json"
)
OUT_JSON = PRIMARY_OUT_JSON
PREREGISTRATION = (
    PROJECT_ROOT
    / "docs/P4_EVOLUTION_SLOW_PROGRAM_REVISION_PREREGISTRATION_20260831.md"
)
OLD_STORE_ROOT = (
    PROJECT_ROOT
    / "_scratch/skill_store/t6_45_frep/frep45_r1/a5_pooled"
)
FORMATION_ARTIFACT = (
    PROJECT_ROOT / "artifacts/functional/e2/t6_45_frep_a5a3_replay.json"
)
DEPLOYMENT_ARTIFACT = (
    PROJECT_ROOT / "artifacts/functional/e2/t6_45_frep_b_symmetric_deploy.json"
)

STAGE = "P4_FORECAST_EVOLUTION_SLOW_PROGRAM_REVISION_NOAA_DEV"
EVIDENCE_GRADE = "EXPOSED_DEVELOPMENT_MECHANISM_REPLICATION_REQUIRED"
OLD_SKILL_ID = (
    "fast_winner_forecast_pooled_ridge_a1_smase_e1v2_outlier_iqr"
)
OLD_SKILL_REVISION = 1
OLD_SKILL_PROGRAM = "outlier_iqr"
EDIT_SURFACE = f"skill_library.entries/{OLD_SKILL_ID}.body"
ARMS = ("K0-fixed", "A5-Slow")
PRIMARY_ORIGINS = (8472, 8520, 8568)
REPEAT_ORIGINS = (8616, 8664, 8712)
HORIZON = 48
PERIOD = 24
MATERIAL = 0.005

LLM_MAX = 8
TOKEN_MAX = 60_000
CONSUMER_FIT_MAX = 8
STAGE_1_A_MAX = 4
STAGE_2_B_MAX = 1
STAGE_3_A_MAX = 3
CHEAP_PROBE_MAX = 24
WALL_MAX = 2_700
GLOBAL_LLM_MAX = 16
GLOBAL_TOKEN_MAX = 120_000


class EvolutionRevisionBlocked(RuntimeError):
    """The trajectory cannot be executed or interpreted safely."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def trajectory_plan(
    origins: Sequence[int] = PRIMARY_ORIGINS,
) -> list[dict[str, Any]]:
    """Return the frozen, outcome-independent three-stage trajectory."""
    values = tuple(int(value) for value in origins)
    if values not in (PRIMARY_ORIGINS, REPEAT_ORIGINS):
        raise ValueError("only the primary or pre-registered repeat is legal")
    roles = (
        "old_skill_fault_and_program_patch",
        "independent_support_b",
        "independent_reencounter",
    )
    faces = ("support_a", "support_b", "support_a")
    llm_allowed = (True, False, True)
    rows: list[dict[str, Any]] = []
    for index, (origin, role, face, allowed) in enumerate(
        zip(values, roles, faces, llm_allowed), start=1
    ):
        rows.append(
            {
                "stage": index,
                "origin": origin,
                "horizon": HORIZON,
                "role": role,
                "feedback_face": face,
                "llm_allowed": allowed,
                "dataset": "NOAA",
                "data_role": "EXPOSED_DEVELOPMENT_HELD_IN",
                "support_a_evaluations_per_arm": (
                    STAGE_1_A_MAX if index == 1
                    else STAGE_3_A_MAX if index == 3
                    else 0
                ),
                "support_b_evaluations_per_arm": 1 if index == 2 else 0,
            }
        )
    return rows


def budget_contract() -> dict[str, Any]:
    return {
        "operating_point": "B=8",
        "per_arm_whole_trajectory": {
            "llm_call_max": LLM_MAX,
            "token_max": TOKEN_MAX,
            "full_consumer_fit_max": CONSUMER_FIT_MAX,
            "stage_1_support_a_max": STAGE_1_A_MAX,
            "stage_2_support_b_max": STAGE_2_B_MAX,
            "stage_3_support_a_max": STAGE_3_A_MAX,
            "cheap_probe_max": CHEAP_PROBE_MAX,
            "wall_seconds_max": WALL_MAX,
            "accepted_update_max_by_arm": {
                "K0-fixed": 0,
                "A5-Slow": 1,
            },
            "counters_reset_between_stages": False,
        },
        "global": {
            "arms": 2,
            "llm_call_max": GLOBAL_LLM_MAX,
            "token_max": GLOBAL_TOKEN_MAX,
            "treatment_consumer_fit_max": 16,
            "identity_reference_fit_count": 3,
            "absolute_consumer_fit_max": 19,
            "cheap_probe_max": 48,
        },
        "ninth_call": {
            "reaches_backend": False,
            "budget_charged": False,
            "terminal_behavior": "BUDGET_EXHAUSTED_ABSTAIN_IDENTITY",
            "counter_scope": "whole_trajectory_per_arm",
        },
    }


def boundary_contract() -> dict[str, Any]:
    return {
        "natural_final_outcome_reads": 0,
        "ucr_test_outcome_reads": 0,
        "sealed_ad_outcome_reads": 0,
        "noaa_2025_confirmation_reads": 0,
        "new_sha_added": False,
        "new_manifest_added": False,
        "p4_evolution_gate_before": "HELD",
        "p4_evolution_gate_after_single_pass": "HELD",
    }


def arm_contract() -> dict[str, dict[str, Any]]:
    initial = {
        "skill_id": OLD_SKILL_ID,
        "revision": OLD_SKILL_REVISION,
        "program_steps": [{"op": OLD_SKILL_PROGRAM, "params": {}}],
    }
    return {
        "K0-fixed": {
            "initial_skill": copy.deepcopy(initial),
            "writeback_allowed": False,
            "raw_episode_input_to_fast": False,
        },
        "A5-Slow": {
            "initial_skill": copy.deepcopy(initial),
            "writeback_allowed": True,
            "raw_episode_input_to_fast": False,
        },
    }


def validate_usage(arm: str, usage: Mapping[str, Any]) -> bool:
    if arm not in ARMS:
        return False
    by_stage = dict(usage.get("llm_calls_by_stage") or {})
    llm_calls = int(usage.get("llm_calls") or 0)
    fits = int(usage.get("full_consumer_fits") or 0)
    stage_fits = (
        int(usage.get("stage_1_support_a_fits") or 0),
        int(usage.get("stage_2_support_b_fits") or 0),
        int(usage.get("stage_3_support_a_fits") or 0),
    )
    return bool(
        0 <= llm_calls <= LLM_MAX
        and sum(int(value or 0) for value in by_stage.values()) == llm_calls
        and int(by_stage.get("stage_2") or 0) == 0
        and 0 <= int(usage.get("tokens") or 0) <= TOKEN_MAX
        and 0 <= fits <= CONSUMER_FIT_MAX
        and sum(stage_fits) == fits
        and 0 <= stage_fits[0] <= STAGE_1_A_MAX
        and 0 <= stage_fits[1] <= STAGE_2_B_MAX
        and 0 <= stage_fits[2] <= STAGE_3_A_MAX
        and 0 <= int(usage.get("cheap_probes") or 0) <= CHEAP_PROBE_MAX
        and 0 <= int(usage.get("accepted_updates") or 0)
        <= (1 if arm == "A5-Slow" else 0)
        and 0.0 <= float(usage.get("wall_seconds") or 0.0) <= WALL_MAX
    )


def validate_patch_contract(
    *, before: Mapping[str, Any], after: Mapping[str, Any], event: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify one atomic Program-body PATCH and its version increment."""
    result: dict[str, Any] = {
        "passed": False,
        "reason": None,
        "operation": str(event.get("operation") or ""),
        "target_surface_id": str(event.get("target_surface_id") or ""),
        "changed_surfaces": [],
        "revision_before": before.get("revision"),
        "revision_after": after.get("revision"),
    }
    if result["operation"] != "PATCH":
        result["reason"] = "operation_not_patch"
        return result
    if result["target_surface_id"] != EDIT_SURFACE:
        result["reason"] = "target_not_exact_body"
        return result
    if str(before.get("skill_id") or "") != str(after.get("skill_id") or ""):
        result["reason"] = "skill_id_changed"
        return result
    if str(before.get("skill_id") or "") != OLD_SKILL_ID:
        result["reason"] = "wrong_skill_id"
        return result
    before_body = before.get("body")
    after_body = after.get("body")
    if before_body == after_body:
        result["reason"] = "body_no_op"
        return result
    try:
        revision_ok = (
            not isinstance(before.get("revision"), bool)
            and not isinstance(after.get("revision"), bool)
            and int(after["revision"]) == int(before["revision"]) + 1
        )
    except (KeyError, TypeError, ValueError):
        revision_ok = False
    if not revision_ok:
        result["reason"] = "revision_not_incremented_by_one"
        return result
    raw_steps = event.get("program_steps") or []
    event_steps = tuple(
        (str(step.get("op") or ""), dict(step.get("params") or {}))
        for step in raw_steps
        if isinstance(step, Mapping)
    )
    if isinstance(after_body, str):
        body_steps = tuple(_parse_frozen_steps(after_body) or ())
    elif isinstance(after_body, Sequence):
        body_steps = tuple(
            (str(step.get("op") or ""), dict(step.get("params") or {}))
            for step in after_body
            if isinstance(step, Mapping)
        )
    else:
        body_steps = ()
    if not event_steps or body_steps != event_steps:
        result["reason"] = "program_binding_mismatch"
        return result
    expected_tools = list(dict.fromkeys(op for op, _params in event_steps))
    mirrors_ok = list(after.get("allowed_tools") or []) == expected_tools
    risk_key = "risk_guards" if "risk_guards" in after else "risk"
    after_risk = dict(after.get(risk_key) or {})
    after_plan = after_risk.get("frozen_plan")
    if isinstance(after_plan, Mapping) and "program" in after_plan:
        mirrors_ok = bool(
            mirrors_ok
            and len(event_steps) == 1
            and str(after_plan.get("program") or "") == event_steps[0][0]
        )
    result["program_mirrors_consistent"] = mirrors_ok
    if not mirrors_ok:
        result["reason"] = "program_mirror_mismatch"
        return result

    def without_program_fields(value: Mapping[str, Any]) -> dict[str, Any]:
        cleaned = copy.deepcopy(dict(value))
        cleaned.pop("body", None)
        cleaned.pop("revision", None)
        cleaned.pop("allowed_tools", None)
        local_risk_key = "risk_guards" if "risk_guards" in cleaned else "risk"
        risk = cleaned.get(local_risk_key)
        if isinstance(risk, Mapping):
            risk_copy = copy.deepcopy(dict(risk))
            plan = risk_copy.get("frozen_plan")
            if isinstance(plan, Mapping):
                plan_copy = dict(plan)
                plan_copy.pop("program", None)
                risk_copy["frozen_plan"] = plan_copy
            cleaned[local_risk_key] = risk_copy
        return cleaned

    before_other = without_program_fields(before)
    after_other = without_program_fields(after)
    if before_other != after_other:
        result["reason"] = "non_program_field_changed"
        return result
    result["changed_surfaces"] = [
        "body",
        "allowed_tools",
        f"{risk_key}.frozen_plan.program",
    ]
    result["passed"] = True
    return result


def derive_verdict(evidence: Mapping[str, Any]) -> str:
    """Pure fail-closed terminal adjudication, in pre-registered priority."""
    stage_1 = dict(evidence.get("stage_1") or {})
    stage_2 = dict(evidence.get("stage_2") or {})
    stage_3 = dict(evidence.get("stage_3") or {})
    if evidence.get("budget_valid") is not True:
        return "BUDGET_INSTRUMENT_FAILURE__NO_SCIENTIFIC_VERDICT"
    if stage_2 and stage_2.get("version_chain_valid") is False:
        return "VERSION_CHAIN_INVALID__NO_SCIENTIFIC_VERDICT"
    if evidence.get("llm_budget_exhausted_before_completion") is True:
        return "LLM_BUDGET_EXHAUSTED_BEFORE_CHAIN__H3_HELD"
    if evidence.get("old_skill_qualified") is not True:
        return "OLD_SKILL_QUALIFICATION_FAILED__H3_HELD"
    if not all(
        stage_1.get(key) is True
        for key in ("v1_resolved", "fast_selected_v1", "runtime_executed_v1")
    ):
        return "OLD_SKILL_NOT_CAUSALLY_USED__H3_HELD"
    if stage_1.get("local_fault") is not True:
        return "NO_LOCAL_V1_FAULT__H3_HELD"
    constrained = stage_1.get("constrained_proposal_succeeds")
    if constrained is True:
        return "PROGRAM_HYPOTHESIS_FALSIFIED__H3_HELD"
    if constrained is not False:
        return "SLOW_SAFE_ABSTAIN__H3_HELD"
    if str(stage_1.get("slow_status") or "") not in {"pending", "support_rejected"}:
        return "SLOW_SAFE_ABSTAIN__H3_HELD"
    if stage_1.get("patch_valid") is not True:
        return "UNAUTHORIZED_OR_INVALID_PATCH__H3_HELD"
    if stage_1.get("support_replay_positive") is not True:
        return "PATCH_SUPPORT_REJECTED__H3_HELD"
    if (
        int(stage_2.get("support_b_evaluations") or 0) != 1
        or stage_2.get("support_b_relation") != "POSITIVE"
        or stage_2.get("promotion_activated") is not True
    ):
        return "INDEPENDENT_SUPPORT_B_REJECTED__H3_HELD"
    if not all(
        stage_2.get(key) is True
        for key in (
            "version_chain_valid",
            "skill_id_preserved",
            "non_program_fields_preserved",
            "program_mirrors_consistent",
        )
    ):
        return "VERSION_CHAIN_INVALID__NO_SCIENTIFIC_VERDICT"
    if not all(
        stage_3.get(key) is True
        for key in ("a5_v2_causally_used", "k0_v1_causally_used")
    ):
        return "REVISED_SKILL_NOT_CAUSALLY_USED__H3_HELD"
    reencounter_pass = bool(
        float(stage_3.get("v2_minus_v1_utility") or 0.0) >= MATERIAL
        and float(stage_3.get("v2_minus_identity_utility") or 0.0) >= MATERIAL
        and int(stage_3.get("a5_harm_count") or 0)
        <= int(stage_3.get("k0_harm_count") or 0)
        and float(stage_3.get("a5_harm_magnitude") or 0.0)
        <= float(stage_3.get("k0_harm_magnitude") or 0.0)
        and int(stage_3.get("a5_active_revision") or 0) == 2
        and int(stage_3.get("k0_active_revision") or 0) == 1
    )
    if not reencounter_pass:
        return "REVISION_REENCOUNTER_FAILED_ROLLED_BACK__H3_HELD"
    return "EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD"


def apply_reencounter_outcome(
    *, v1_snapshot: Any, v2_snapshot: Any, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    verdict = derive_verdict(evidence)
    rollback = bool(
        dict(evidence.get("stage_2") or {}).get("promotion_activated") is True
        and verdict
        != "EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD"
    )
    active = v1_snapshot if rollback else v2_snapshot
    revision = (
        active.get("revision") if isinstance(active, Mapping)
        else _skill_revision(active)
    )
    return {
        "rolled_back": rollback,
        "active_snapshot": active,
        "active_revision": int(revision),
        "new_revision_created": False,
        "verdict": verdict,
    }


def _plain(value: Any) -> Any:
    return performance._plain(value)


def _write(payload: Mapping[str, Any], *, path: Path = OUT_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _skill(snapshot: Any) -> Any:
    return next(
        (entry for entry in snapshot.skills if entry.skill_id == OLD_SKILL_ID),
        None,
    )


def _skill_revision(snapshot: Any) -> int:
    entry = _skill(snapshot)
    return int(entry.revision) if entry is not None else 0


def _skill_document(snapshot: Any) -> dict[str, Any]:
    entry = _skill(snapshot)
    return skill_entry_to_dict(entry) if entry is not None else {}


def _old_steps(snapshot: Any) -> tuple[tuple[str, dict[str, object]], ...]:
    entry = _skill(snapshot)
    parsed = _parse_frozen_steps(str(getattr(entry, "body", "")))
    return tuple(parsed or ())


def _load_old_snapshot() -> Any:
    active = json.loads((OLD_STORE_ROOT / "active.json").read_text(encoding="utf-8"))
    name = str(active.get("runtime_bundle_sha") or "")
    root = OLD_STORE_ROOT / "snapshots" / name
    if not root.is_dir():
        raise EvolutionRevisionBlocked("old Skill active snapshot is unavailable")
    return compile_snapshot(root, verify_lock=False)


def _formation_checks() -> dict[str, bool]:
    formation = json.loads(FORMATION_ARTIFACT.read_text(encoding="utf-8"))
    deployment = json.loads(DEPLOYMENT_ARTIFACT.read_text(encoding="utf-8"))
    held_in = dict(formation.get("stage_2_held_in") or {})
    a5_pooled = dict(dict(held_in.get("cells") or {}).get("a5_pooled") or {})
    draft = dict(a5_pooled.get("draft") or {})
    promotion = dict(a5_pooled.get("promotion") or {})
    lifecycle = dict(promotion.get("lifecycle_fields") or {})
    target = dict(dict(formation.get("binding") or {}).get("target") or {})
    update_path = dict(held_in.get("update_path") or {})
    deployed = dict(
        dict(
            dict(deployment.get("symmetric_deployment") or {}).get("cells")
            or {}
        ).get("a5_pooled")
        or {}
    )
    record = dict(deployed.get("record") or {})
    retrieval = dict(record.get("retrieval") or {})
    held_out_window = dict(dict(deployment.get("held_out") or {}).get("window") or {})
    return {
        "natural_draft_written": bool(
            draft.get("written") is True
            and draft.get("skill_id") == OLD_SKILL_ID
            and dict(draft.get("handle_fast_winner") or {}).get("stage")
            == "pending"
        ),
        "independent_promotion": bool(
            promotion.get("promoted") is True
            and promotion.get("store_approved") is True
            and promotion.get("retrievable_skill_id") == OLD_SKILL_ID
            and lifecycle.get("activation_probe_took_part_in_selection") is False
            and lifecycle.get("local_status") == "LOCAL_ACTIVE"
        ),
        "not_set_by_hand": bool(
            update_path.get("nothing_is_set_by_hand") is True
            and update_path.get("draft")
            == "methods/ttha/method.py::TTHAMethod.handle_fast_winner"
            and update_path.get("store_approval")
            == "methods/ttha/method.py::TTHAMethod.handle_feedback_delayed"
        ),
        "formation_target_noaa_development": bool(
            target.get("cohort") == "noaa_fresh"
            and target.get("partition") == "development_2024, index [0, 8760)"
            and str(target.get("exposure") or "").startswith("EXPOSED development")
        ),
        "prior_same_domain_retrieval": bool(
            record.get("target_id") == "FRESH_pooled"
            and record.get("active_skill_id") == OLD_SKILL_ID
            and record.get("recall_hit") is True
            and retrieval.get("expected_local_skill_id") == OLD_SKILL_ID
            and retrieval.get("local_skill_hit") is True
            and dict(retrieval.get("context") or {})
            .get("features", {})
            .get("task_kind")
            == "forecast"
            and held_out_window.get("farthest_index") == noaa.DEVELOPMENT_HOURS
            and "exposed 2024 development partition"
            in str(held_out_window.get("note") or "")
        ),
    }


def _returned_models(backend: Any) -> tuple[str, ...]:
    inner = getattr(backend, "_shared", None)
    return tuple(
        sorted(str(value) for value in (getattr(inner, "returned_models", ()) or ()))
    )


def validate_returned_models(
    *, requested_model: str, returned_models: Sequence[str], calls: int
) -> bool:
    observed = {str(value) for value in returned_models if str(value)}
    if int(calls) <= 0:
        return not observed
    return observed == {str(requested_model)}


def _refuse_existing_run_artifact(path: Path) -> None:
    if path.exists():
        raise EvolutionRevisionBlocked(
            "frozen trajectory output already exists; overwrite/reroll is prohibited: "
            + path.as_posix()
        )


def _release_boundary() -> dict[str, Any]:
    gate = performance._read_object(split_release.OUT_JSON)
    evolution = dict(gate.get("p4_evolution") or {})
    checks = {
        "p4_evolution_held": evolution.get("status") == "HELD",
        "natural_final_closed": gate.get("natural_final_release") is False,
        "final_reads_zero": int(gate.get("final_outcome_reads", -1)) == 0,
    }
    if not all(checks.values()):
        raise EvolutionRevisionBlocked(
            "release boundary failed: %s"
            % [name for name, passed in checks.items() if not passed]
        )
    return {
        "source": split_release.OUT_JSON.relative_to(PROJECT_ROOT).as_posix(),
        "checks": checks,
    }


@dataclass(frozen=True)
class NoaaCell:
    values: Mapping[str, np.ndarray]
    train_uids: tuple[str, ...]
    eval_uids: tuple[str, ...]
    observation_block: np.ndarray

    def roster(self, face: str) -> list[dict[str, str]]:
        if face not in {"support_a", "support_b"}:
            raise KeyError("unknown NOAA lifecycle face")
        return (
            [{"series_uid": uid, "role": "train"} for uid in self.train_uids]
            + [{"series_uid": uid, "role": "eval"} for uid in self.eval_uids]
        )


def _load_noaa_cell(origin: int) -> NoaaCell:
    artifact = noaa._cohort_artifact()
    health = artifact["step_2_health_check_v2"]
    train = tuple(str(uid) for uid in health["confirmation_roster"])
    evaluation = tuple(str(uid) for uid in health["substitutes"])
    loaded = noaa._load_development(train + evaluation)
    values = loaded["values"]
    required = int(origin) + HORIZON
    if any(int(values[uid].size) < required for uid in train + evaluation):
        raise EvolutionRevisionBlocked("NOAA development array is too short")
    return NoaaCell(
        values=values,
        train_uids=train,
        eval_uids=evaluation,
        observation_block=np.asarray(values[train[0]][:origin], dtype=np.float64),
    )


def _config(origin: int) -> dict[str, object]:
    config = dict(performance._config(origin))
    config.update(
        {
            "dataset_id": "noaa_2024_exposed_p4_evolution_revision",
            "support_origin": int(origin),
            "selection_origin": int(origin),
            "period": PERIOD,
        }
    )
    return config


class _SingleFaceEval:
    def __init__(self, cell: NoaaCell, origin: int, face: str) -> None:
        self.cell = cell
        self.origin = int(origin)
        self.face = str(face)
        self.fits = 0
        self._expected = tuple(
            (str(row["series_uid"]), str(row["role"]))
            for row in cell.roster(face)
        )

    def __call__(
        self,
        roster: Any,
        values: Any,
        compiled: Any,
        config: Any,
        *,
        origin: int,
    ) -> dict[str, Any]:
        if int(origin) != self.origin:
            raise EvolutionRevisionBlocked("Consumer received the wrong origin")
        received = tuple(
            (str(row["series_uid"]), str(row["role"])) for row in roster
        )
        if received != self._expected:
            raise EvolutionRevisionBlocked("Consumer received a changed NOAA roster")
        self.fits += 1
        return forecast_p1.forecast_runtime._evaluate(
            roster, values, compiled, config, origin=origin
        )


class _NoHashScopeExecutor(ScopeExecutor):
    """Use the existing verifier semantics without transient behavior hashes."""

    def verify(self, steps: Any, origin: int) -> Any:
        return self.verify_without_behavior_hashes(steps, origin)


def _identity_reading(cell: NoaaCell, origin: int) -> dict[str, Any]:
    raw = forecast_p1.forecast_runtime._evaluate(
        cell.roster("support_a"), cell.values, None, _config(origin), origin=origin
    )
    smase = float(raw["mean_smase"])
    per_series = [float(value) for value in raw["per_view_smase"]]
    if not math.isfinite(smase) or not all(math.isfinite(value) for value in per_series):
        raise EvolutionRevisionBlocked("non-finite identity reference")
    return {
        "smase": smase,
        "utility": -smase,
        "per_series_smase": per_series,
        "fit_count": 1,
    }


def _executor(
    cell: NoaaCell, origin: int, face: str, identity: Mapping[str, Any]
) -> tuple[ScopeExecutor, _SingleFaceEval]:
    evaluator = _SingleFaceEval(cell, origin, face)
    executor = _NoHashScopeExecutor(
        cell.roster(face),
        cell.values,
        _config(origin),
        evaluate_fn=evaluator,
        max_modified_fraction=performance.MAX_MODIFIED_FRACTION,
    )
    executor._baseline_cache[origin] = float(identity["smase"])
    executor._per_view_cache[origin] = [
        float(value) for value in identity["per_series_smase"]
    ]
    return executor, evaluator


def typed_patch_options() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "patch_id": f"forecast-existing-program-{op}",
            "program_steps": [
                {"op": step_op, "params": dict(params)}
                for step_op, params in forecast_p1._steps(op)
            ],
        }
        for op in performance.PARALLEL_PROGRAMS
    )


def _unit(origin: int, arm: str, stage: int) -> dict[str, Any]:
    return {
        "replica": "Development",
        "sequence_index": stage,
        "episode_id": f"slow_revision_{arm}_{stage}",
        "origin": int(origin),
        "horizon": HORIZON,
        "natural_episode": True,
    }


def _request(
    cell: NoaaCell, origin: int, arm: str, stage: int, spec: Any, context: Any
) -> tuple[PreparationRequest, dict[str, Any]]:
    return performance._request(
        unit=_unit(origin, arm, stage),
        cell=cell,
        origin=origin,
        spec=spec,
        context=context,
    )


def _card(origin: int, episode: Any, options: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    context = dict(getattr(episode, "context_summary", None) or {})
    geometry = dict(context.get("program_geometry") or {})
    support = dict(getattr(episode, "support_response", None) or {})
    return {
        "pattern_id": f"forecast-noaa-local-skill-fault-{origin}",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": list(geometry.get("program_steps") or ())},
        "facts": {
            "relation": str(getattr(episode, "relation", "")),
            "support_gain": support.get("gain"),
            "development_origin": int(origin),
        },
        "typed_patch_options": [dict(option) for option in options],
    }


def _result_record(
    result: Any,
    method: Any,
    *,
    unavailable_reason: str = "BUDGET_EXHAUSTED_ABSTAIN_IDENTITY",
) -> dict[str, Any]:
    if result is None:
        return {
            "abstained": True,
            "abstain_reason": unavailable_reason,
            "retrieved_skill_ids": [],
            "chosen_candidate_id": "",
            "winner_candidate_id": "",
            "probes": [],
            "harm_count": 0,
            "harm_magnitude": 0.0,
        }
    trace = method.last_trace
    return {
        "retrieved_skill_ids": list(trace.retrieved_skill_ids or ()),
        "candidate_ids": list(trace.candidate_ids or ()),
        "chosen_candidate_id": str(trace.chosen_candidate_id or ""),
        "chosen_candidate_source_skill_id": source_skill_of_candidate(
            trace.chosen_candidate_id
        ),
        "winner_candidate_id": str(result._winner_candidate_id or ""),
        "winner_candidate_source_skill_id": source_skill_of_candidate(
            result._winner_candidate_id
        ),
        "probes": _plain(result.actual_probed_programs),
        "support_receipts_used": int(result.target_support_receipts_used),
        "harm_count": int(result.harm_count),
        "harm_magnitude": float(result.harm_magnitude),
        "abstained": bool(result.abstained),
    }


def _candidate_steps(trace: Any, candidate_id: str) -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple(
        (str(op), dict(params))
        for op, params in (
            dict(trace.candidate_program_steps or {}).get(candidate_id) or ()
        )
    )


def _old_fault_episode(result: Any) -> Any | None:
    for episode, _steps in result._episodes:
        context = dict(getattr(episode, "context_summary", None) or {})
        if context.get("source_skill_id") == OLD_SKILL_ID:
            return episode
    return None


def _constrained_proposal_status(result: Any, method: Any) -> bool | None:
    trace = method.last_trace
    pool = [
        str(candidate_id)
        for candidate_id in (trace.candidate_ids or ())
        if str(candidate_id) != "identity"
        and source_skill_of_candidate(candidate_id) is None
        and _candidate_steps(trace, str(candidate_id))
    ]
    if result._winner_candidate_id and source_skill_of_candidate(
        result._winner_candidate_id
    ) is None:
        return True
    attempted = {
        str(row.get("candidate_id") or "")
        for row in result.actual_probed_programs
        if str(row.get("kind") or "") in {"probe", "verifier_rejected"}
    }
    if pool and all(candidate_id in attempted for candidate_id in pool):
        return False
    return None


def _program_gain(result: Any) -> float | None:
    if result is None or not result._winner_candidate_id:
        return None
    for row in result.actual_probed_programs:
        if (
            str(row.get("candidate_id") or "") == str(result._winner_candidate_id)
            and row.get("gain") is not None
        ):
            return float(row["gain"])
    return None


def _causal_skill_use(result: Any, method: Any, snapshot: Any) -> bool:
    if result is None:
        return False
    trace = method.last_trace
    chosen = str(trace.chosen_candidate_id or "")
    winner = str(result._winner_candidate_id or "")
    expected = _old_steps(snapshot)
    return bool(
        OLD_SKILL_ID in tuple(trace.retrieved_skill_ids or ())
        and source_skill_of_candidate(chosen) == OLD_SKILL_ID
        and source_skill_of_candidate(winner) == OLD_SKILL_ID
        and _candidate_steps(trace, chosen) == expected
        and tuple(result._winner_steps or ()) == expected
    )


def _relation(receipt: Any, eval_uids: Sequence[str]) -> str:
    if receipt is None:
        return "UNAVAILABLE"
    per_series = {
        str(uid): float(value)
        for uid, value in zip(eval_uids, receipt.per_view_gain)
    }
    return str(
        classify_relation(
            aggregate_gain=receipt.gain,
            per_series_gains=per_series,
            consumer_id="pooled_ridge_a1",
        )["relation"]
    )


def _new_state(
    snapshot: Any,
    cell: NoaaCell,
    arm_backend: Any,
    root: Path,
    arm: str,
) -> dict[str, Any]:
    return four_arms._new_state(
        snapshot=snapshot,
        agent=forecast_course._live_agent(cell.observation_block, arm_backend),
        store_root=root,
        tag=arm.replace("-", "_").lower(),
        episodes=(),
    )


def _fresh_method(snapshot: Any, cell: NoaaCell, arm_backend: Any) -> TTHAMethod:
    return TTHAMethod(
        forecast_course._live_agent(cell.observation_block, arm_backend),
        snapshot,
        (),
    )


def _run_fast(
    *,
    method: Any,
    state: Mapping[str, Any],
    executor: ScopeExecutor,
    request: Any,
    features: Mapping[str, Any],
    cell: NoaaCell,
    origin: int,
    arm: str,
    stage: int,
    budget: int,
) -> tuple[Any | None, bool]:
    try:
        result = run_online_round(
            method,
            executor,
            request,
            cell.values,
            origin=origin,
            slow_agent=None,
            controller=state["controller"],
            store=state["store"],
            card_builder=lambda episode: _card(origin, episode, ()),
            round_name=f"stage_{stage}_{arm.lower()}",
            budget=budget,
            allow_slow=False,
            horizon=HORIZON,
            period=PERIOD,
            domain="noaa_2024_exposed_development",
            fast_features=features,
            # This slice may revise the one frozen Skill only through the
            # authorized Slow Program PATCH.  Fast can retrieve and execute
            # active Skills regardless of this flag; keeping it false prevents
            # a positive non-Skill winner from creating an extra Draft.
            allow_fast_skill=False,
            runtime_prior_slot=False,
            pool_mode="full",
        )
        return result, False
    except shared_harness.Stop as exc:
        if exc.verdict != performance.CELL_LLM_EXHAUSTION_VERDICT:
            raise
        return None, True


def _manual_slow_patch(
    *,
    method: Any,
    state: Mapping[str, Any],
    result: Any,
    executor: ScopeExecutor,
    origin: int,
    features: Mapping[str, Any],
    request: Any,
    constrained: bool,
) -> tuple[dict[str, Any], int]:
    episode = _old_fault_episode(result)
    if episode is None:
        return {"stage": "old_skill_fault_episode_unavailable"}, 0
    trace = method.last_trace
    view = resolve_harness_view(method._active_snapshot(), features, role="fast")
    options = typed_patch_options()
    assessment = route_verified_program_supply_fault(
        trace=trace,
        episode=episode,
        view=view,
        executor=executor,
        typed_patch_options=options,
        origin=origin,
        constrained_proposal_succeeds=constrained,
    )
    card = _card(origin, episode, options)
    bound, verified_ids, route_error = bind_verified_program_options(card, assessment)
    if route_error is not None or bound is None:
        return {
            **dict(route_error or {"stage": "no_verified_options"}),
            "verified_patch_ids": list(verified_ids),
        }, len(options)
    old_steps = _old_steps(method._active_snapshot())
    current_patch_id = f"forecast-existing-program-{OLD_SKILL_PROGRAM}"
    verified_patch_ids = {
        alternative.patch_id
        for alternative in assessment.verification.alternatives
    }
    distinct_pairs = {
        frozenset((left, right))
        for left, right in assessment.verification.behavior_distinct_pairs
    }
    if current_patch_id not in verified_patch_ids:
        return {
            "stage": "current_program_not_verifier_earned",
            "verified_patch_ids": list(verified_ids),
        }, len(options)
    filtered_options = [
        option
        for option in (bound.get("typed_patch_options") or [])
        if (
            tuple(
                (str(step["op"]), dict(step.get("params") or {}))
                for step in option.get("program_steps", [])
            )
            != old_steps
            and frozenset((current_patch_id, str(option.get("patch_id") or "")))
            in distinct_pairs
        )
    ]
    bound = dict(bound)
    bound["typed_patch_options"] = filtered_options
    if not filtered_options:
        return {
            "stage": "no_non_noop_verified_options",
            "verified_patch_ids": list(verified_ids),
        }, len(options)
    relevant = retrieved_relevant_capability_skill_ids(assessment, trace)
    if relevant != (OLD_SKILL_ID,):
        return {
            "stage": "ambiguous_skill_patch_target",
            "retrieved_relevant_capability_skill_ids": list(relevant),
        }, len(options)
    catalog = build_single_surface_catalog(
        decision=assessment.decision,
        parent=state["store"].materialize(method._active_snapshot()),
        controller=state["controller"],
        retrieved_capability_skill_ids=relevant,
    )
    if len(catalog) != 1 or catalog[0].get("surface_id") != EDIT_SURFACE:
        return {"stage": "exact_program_surface_not_authorized"}, len(options)
    contract_ops = sorted(
        {
            str(step["op"])
            for option in filtered_options
            for step in option.get("program_steps", [])
        }
    )
    slow_agent = TTHASlowAgent(method.fast_agent.core)
    event = method.handle_feedback_support(
        episode,
        slow_agent=slow_agent,
        controller=state["controller"],
        store=state["store"],
        surface_catalog=catalog,
        card_builder=lambda _episode: bound,
        evaluator=lambda steps, _mode: executor.evaluate(tuple(steps), origin),
        confirmed_cause=assessment.decision.cause_code,
        fast_features=features,
        allowed_operator_contracts=tuple(
            public_operator_contract(op) for op in contract_ops
        ),
        task_context=getattr(request, "task_context", None),
    )
    return {
        **dict(event),
        "verified_patch_ids": list(verified_ids),
        "retrieved_relevant_capability_skill_ids": list(relevant),
        "no_op_program_withheld_from_slow": True,
        "behavior_equivalent_programs_withheld_from_slow": True,
    }, len(options)


def _scope_calls(backend: Any, arm: str) -> int:
    state = backend.budget_state()
    return int(dict(state.get("scope_calls") or {}).get(arm, 0))


def _scope_blocked(backend: Any, arm: str) -> bool:
    return any(
        record.get("scope_id") == arm
        for record in backend.budget_state().get("blocked_records", [])
    )


def _run_arm(
    *,
    arm: str,
    initial: Any,
    cells: Mapping[int, NoaaCell],
    identities: Mapping[int, Mapping[str, Any]],
    origins: tuple[int, int, int],
    backend: Any,
    spec: Any,
    context: Any,
    root: Path,
    run_stage_2: bool,
    run_stage_3: bool,
) -> dict[str, Any]:
    started = time.time()
    before_backend = forecast_p1._backend_usage(backend)
    arm_backend = backend.new_arm_backend(scope_id=arm, maximum_calls=LLM_MAX)
    state = _new_state(initial, cells[origins[0]], arm_backend, root, arm)
    method = state["method"]
    old_document = _skill_document(initial)
    calls_start = _scope_calls(backend, arm)

    stage_1_executor, stage_1_eval = _executor(
        cells[origins[0]], origins[0], "support_a", identities[origins[0]]
    )
    request, features = _request(
        cells[origins[0]], origins[0], arm, 1, spec, context
    )
    stage_1_result, stage_1_budget_stop = _run_fast(
        method=method,
        state=state,
        executor=stage_1_executor,
        request=request,
        features=features,
        cell=cells[origins[0]],
        origin=origins[0],
        arm=arm,
        stage=1,
        budget=3,
    )
    slow_event: dict[str, Any] = {
        "stage": "not_authorized_for_k0" if arm == "K0-fixed" else "not_triggered"
    }
    verifier_requests = 0
    constrained: bool | None = None
    patch_contract: dict[str, Any] = {"passed": False, "reason": "not_proposed"}
    pending = None
    pending_snapshot = None
    if stage_1_result is not None:
        constrained = _constrained_proposal_status(stage_1_result, method)
        trigger_episode = _old_fault_episode(stage_1_result)
        trigger_chosen = str(method.last_trace.chosen_candidate_id or "")
        trigger_probe = any(
            str(row.get("candidate_id") or "") == trigger_chosen
            and str(row.get("kind") or "") == "probe"
            and row.get("passed") is True
            for row in stage_1_result.actual_probed_programs
        )
        trigger_is_causal_fault = bool(
            OLD_SKILL_ID in tuple(method.last_trace.retrieved_skill_ids or ())
            and source_skill_of_candidate(trigger_chosen) == OLD_SKILL_ID
            and _candidate_steps(method.last_trace, trigger_chosen)
            == _old_steps(initial)
            and trigger_probe
            and trigger_episode is not None
            and str(getattr(trigger_episode, "relation", ""))
            in {"NEGATIVE", "CONFLICT"}
        )
        if (
            arm == "A5-Slow"
            and constrained is False
            and trigger_is_causal_fault
        ):
            try:
                slow_event, verifier_requests = _manual_slow_patch(
                    method=method,
                    state=state,
                    result=stage_1_result,
                    executor=stage_1_executor,
                    origin=origins[0],
                    features=features,
                    request=request,
                    constrained=False,
                )
            except shared_harness.Stop as exc:
                if exc.verdict != performance.CELL_LLM_EXHAUSTION_VERDICT:
                    raise
                stage_1_budget_stop = True
                verifier_requests = len(typed_patch_options())
                slow_event = {
                    "stage": "budget_exhausted_before_backend",
                    "abstain_action": "BUDGET_EXHAUSTED_ABSTAIN_IDENTITY",
                }
            pending = getattr(method, "_pending_update", None)
            if isinstance(pending, Mapping):
                candidate = pending["receipt"].candidate_snapshot.snapshot
                pending_snapshot = candidate
                event_for_contract = {
                    "operation": slow_event.get("operation"),
                    "target_surface_id": slow_event.get("target_surface_id"),
                    "patch_id": slow_event.get("patch_id"),
                    "program_steps": slow_event.get("frozen_program"),
                }
                patch_contract = validate_patch_contract(
                    before=old_document,
                    after=_skill_document(candidate),
                    event=event_for_contract,
                )
    calls_after_stage_1 = _scope_calls(backend, arm)

    old_episode = (
        _old_fault_episode(stage_1_result)
        if stage_1_result is not None
        else None
    )
    chosen_id = (
        str(method.last_trace.chosen_candidate_id or "")
        if stage_1_result is not None
        else ""
    )
    old_probe_executed = bool(
        stage_1_result is not None
        and any(
            str(row.get("candidate_id") or "") == chosen_id
            and str(row.get("kind") or "") == "probe"
            and row.get("passed") is True
            for row in stage_1_result.actual_probed_programs
        )
    )
    stage_1_causal = bool(
        stage_1_result is not None
        and OLD_SKILL_ID in tuple(method.last_trace.retrieved_skill_ids or ())
        and source_skill_of_candidate(chosen_id) == OLD_SKILL_ID
        and _candidate_steps(method.last_trace, chosen_id) == _old_steps(initial)
        and old_probe_executed
        and old_episode is not None
    )
    old_skill_probe_fault_relation_observed = bool(
        old_episode is not None
        and str(getattr(old_episode, "relation", "")) in {"NEGATIVE", "CONFLICT"}
    )
    local_fault = bool(stage_1_causal and old_skill_probe_fault_relation_observed)
    stage_1 = {
        **_result_record(stage_1_result, method),
        "v1_resolved": bool(
            stage_1_result is not None
            and OLD_SKILL_ID in tuple(method.last_trace.retrieved_skill_ids or ())
        ),
        "fast_selected_v1": stage_1_causal,
        "runtime_executed_v1": stage_1_causal,
        "local_fault": local_fault,
        "old_skill_probe_fault_relation_observed": (
            old_skill_probe_fault_relation_observed
        ),
        "constrained_proposal_succeeds": constrained,
        "slow_status": str(slow_event.get("stage") or "not_run"),
        "slow_event": _plain(slow_event),
        "patch_valid": bool(patch_contract.get("passed")),
        "patch_contract": patch_contract,
        "support_replay_positive": slow_event.get("stage") == "pending",
        "consumer_fits": stage_1_eval.fits,
    }

    stage_2_executor = None
    stage_2_eval = None
    stage_2_receipt = None
    stage_2_event: dict[str, Any] = {"stage": "not_reached"}
    activated = False
    approved_snapshot = None
    if run_stage_2:
        stage_2_executor, stage_2_eval = _executor(
            cells[origins[1]], origins[1], "support_b", identities[origins[1]]
        )
        if arm == "A5-Slow" and isinstance(pending, Mapping):
            holder: list[Any] = []

            def delayed(steps: Any, _mode: int) -> Any:
                receipt = stage_2_executor.evaluate(tuple(steps), origins[1])
                holder.append(receipt)
                return receipt

            stage_2_event = method.handle_feedback_delayed(
                delayed,
                episode_id=str(getattr(old_episode, "episode_id", "")),
            )
            stage_2_receipt = holder[0] if holder else None
            if stage_2_event.get("stage") == "approved":
                approved_snapshot = method._active_snapshot()
                state["store"].set_active(approved_snapshot.runtime_bundle_sha)
                activated = True
        elif arm == "K0-fixed":
            stage_2_receipt = stage_2_executor.evaluate(
                _old_steps(initial), origins[1]
            )
            stage_2_event = {"stage": "descriptive_fixed_control"}
    stage_2_relation = _relation(
        stage_2_receipt, cells[origins[1]].eval_uids
    )
    version_snapshot = approved_snapshot or pending_snapshot
    after_document = (
        _skill_document(version_snapshot) if version_snapshot is not None else {}
    )
    version_contract = (
        validate_patch_contract(
            before=old_document,
            after=after_document,
            event={
                "operation": slow_event.get("operation"),
                "target_surface_id": slow_event.get("target_surface_id"),
                "program_steps": slow_event.get("frozen_program"),
            },
        )
        if version_snapshot is not None
        else {"passed": None, "reason": "no_candidate_revision"}
    )
    stage_2 = {
        "origin": origins[1],
        "support_b_evaluations": int(stage_2_eval.fits if stage_2_eval else 0),
        "support_b_gain": (
            float(stage_2_receipt.gain)
            if stage_2_receipt is not None and stage_2_receipt.gain is not None
            else None
        ),
        "support_b_relation": stage_2_relation,
        "event": _plain(stage_2_event),
        "promotion_activated": activated,
        "version_chain_valid": version_contract.get("passed"),
        "skill_id_preserved": (
            _skill_document(version_snapshot).get("skill_id") == OLD_SKILL_ID
            if version_snapshot is not None
            else None
        ),
        "non_program_fields_preserved": (
            bool(version_contract.get("passed"))
            if version_snapshot is not None
            else None
        ),
        "program_mirrors_consistent": (
            bool(version_contract.get("program_mirrors_consistent"))
            if version_snapshot is not None
            else None
        ),
        "revision_before": 1,
        "revision_after": (
            _skill_revision(version_snapshot) if version_snapshot is not None else None
        ),
        "version_contract": version_contract,
    }

    stage_3_result = None
    stage_3_method = None
    stage_3_eval = None
    stage_3_budget_stop = False
    stage_3_snapshot = approved_snapshot if arm == "A5-Slow" else initial
    if run_stage_3 and stage_3_snapshot is not None:
        stage_3_executor, stage_3_eval = _executor(
            cells[origins[2]], origins[2], "support_a", identities[origins[2]]
        )
        stage_3_method = _fresh_method(
            stage_3_snapshot, cells[origins[2]], arm_backend
        )
        stage_3_request, stage_3_features = _request(
            cells[origins[2]], origins[2], arm, 3, spec, context
        )
        stage_3_result, stage_3_budget_stop = _run_fast(
            method=stage_3_method,
            state=state,
            executor=stage_3_executor,
            request=stage_3_request,
            features=stage_3_features,
            cell=cells[origins[2]],
            origin=origins[2],
            arm=arm,
            stage=3,
            budget=STAGE_3_A_MAX,
        )
    calls_after_stage_3 = _scope_calls(backend, arm)
    causal = bool(
        stage_3_result is not None
        and stage_3_method is not None
        and _causal_skill_use(stage_3_result, stage_3_method, stage_3_snapshot)
    )
    stage_3 = {
        **(
            _result_record(stage_3_result, stage_3_method)
            if stage_3_method is not None
            else _result_record(
                None,
                method,
                unavailable_reason="STAGE_NOT_REACHED__NO_APPROVED_V2",
            )
        ),
        "origin": origins[2],
        "causal_skill_use": causal,
        "program_gain_vs_identity": _program_gain(stage_3_result),
        "consumer_fits": int(stage_3_eval.fits if stage_3_eval else 0),
        "active_revision": _skill_revision(stage_3_snapshot)
        if stage_3_snapshot is not None
        else None,
    }

    after_backend = forecast_p1._backend_usage(backend)
    stage_1_calls = calls_after_stage_1 - calls_start
    stage_3_calls = calls_after_stage_3 - calls_after_stage_1
    usage = {
        "llm_calls": calls_after_stage_3 - calls_start,
        "llm_calls_by_stage": {
            "stage_1": stage_1_calls,
            "stage_2": 0,
            "stage_3": stage_3_calls,
        },
        "input_tokens": after_backend[1] - before_backend[1],
        "output_tokens": after_backend[2] - before_backend[2],
        "tokens": (
            after_backend[1] - before_backend[1]
            + after_backend[2] - before_backend[2]
        ),
        "stage_1_support_a_fits": int(stage_1_eval.fits),
        "stage_2_support_b_fits": int(stage_2_eval.fits if stage_2_eval else 0),
        "stage_3_support_a_fits": int(stage_3_eval.fits if stage_3_eval else 0),
        "full_consumer_fits": int(
            stage_1_eval.fits
            + (stage_2_eval.fits if stage_2_eval else 0)
            + (stage_3_eval.fits if stage_3_eval else 0)
        ),
        "cheap_probes": int(
            (
                forecast_p1._fast_verifier_requests(method.last_trace)
                if method.last_trace is not None
                else 0
            )
            + (
                forecast_p1._fast_verifier_requests(stage_3_method.last_trace)
                if stage_3_method is not None and stage_3_method.last_trace is not None
                else 0
            )
            + verifier_requests
        ),
        "accepted_updates": int(activated),
        "wall_seconds": round(time.time() - started, 3),
    }
    return {
        "arm": arm,
        "stage_1": stage_1,
        "stage_2": stage_2,
        "stage_3": stage_3,
        "usage": usage,
        "usage_valid": validate_usage(arm, usage),
        "llm_budget_exhausted": bool(
            stage_1_budget_stop or stage_3_budget_stop or _scope_blocked(backend, arm)
        ),
        "initial_snapshot": initial,
        "approved_snapshot": approved_snapshot,
        "state": state,
    }


def _scientific_arm(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _plain(value)
        for key, value in record.items()
        if key not in {"initial_snapshot", "approved_snapshot", "state"}
    }


def preflight(*, origins: Sequence[int] = PRIMARY_ORIGINS) -> dict[str, Any]:
    plan = trajectory_plan(origins)
    boundary = _release_boundary()
    snapshot = _load_old_snapshot()
    entry = _skill(snapshot)
    formation = _formation_checks()
    max_origin = max(int(row["origin"]) for row in plan)
    cell = _load_noaa_cell(max_origin)
    public = extract_public_features(cell.observation_block, task_kind="forecast")
    view = resolve_harness_view(snapshot, public, role="fast")
    options = typed_patch_options()
    repeat_requested = tuple(int(value) for value in origins) == REPEAT_ORIGINS
    primary_pass = False
    if repeat_requested and PRIMARY_OUT_JSON.is_file():
        primary = performance._read_object(PRIMARY_OUT_JSON)
        primary_pass = bool(
            primary.get("status") == "COMPLETE"
            and primary.get("verdict")
            == "EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD"
            and dict(primary.get("boundary") or {}).get(
                "natural_final_outcome_reads"
            )
            == 0
        )
    checks = {
        "preregistration_exists": PREREGISTRATION.is_file(),
        "boundary": all(boundary["checks"].values()),
        "old_skill_present": entry is not None,
        "old_skill_revision_one": int(getattr(entry, "revision", 0)) == 1,
        "old_skill_program_exact": _old_steps(snapshot)
        == ((OLD_SKILL_PROGRAM, {}),),
        "old_skill_local_active": dict(getattr(entry, "risk_guards", {}) or {}).get(
            "local_status"
        )
        == "LOCAL_ACTIVE",
        "old_skill_requires_target_support": dict(
            getattr(entry, "risk_guards", {}) or {}
        ).get("requires_target_support")
        is True,
        "formation_chain": all(formation.values()),
        "same_noaa_domain": bool(
            formation.get("formation_target_noaa_development")
            and formation.get("prior_same_domain_retrieval")
        ),
        "old_skill_retrievable_at_stage_1": OLD_SKILL_ID
        in {str(skill.skill_id) for skill in view.skills},
        "three_nonoverlapping_windows": all(
            left["origin"] + HORIZON <= right["origin"]
            for left, right in zip(plan, plan[1:])
        ),
        "development_only_farthest_index": max_origin + HORIZON
        <= noaa.DEVELOPMENT_HOURS,
        "noaa_roster_fixed": len(cell.train_uids) == 12 and len(cell.eval_uids) == 4,
        "typed_program_inventory_is_existing_b8": len(options) == 7
        and tuple(
            str(option["program_steps"][0]["op"]) for option in options
        )
        == tuple(performance.PARALLEL_PROGRAMS)
        and 1 + len(options) == performance.B_MAIN,
        "one_whole_trajectory_b8": budget_contract()["operating_point"] == "B=8",
        "final_reads_zero": boundary_contract()["natural_final_outcome_reads"] == 0,
        "no_new_sha": boundary_contract()["new_sha_added"] is False,
        "repeat_only_after_primary_complete_pass": (
            primary_pass if repeat_requested else True
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "formation_checks": formation,
        "trajectory": plan,
        "budget": budget_contract(),
        "boundary": boundary_contract(),
        "output": (
            REPEAT_OUT_JSON if repeat_requested else PRIMARY_OUT_JSON
        ).relative_to(PROJECT_ROOT).as_posix(),
    }


def run(*, origins: Sequence[int] = PRIMARY_ORIGINS, backend_mode: str = "live") -> dict[str, Any]:
    origins = tuple(int(value) for value in origins)
    output = REPEAT_OUT_JSON if origins == REPEAT_ORIGINS else PRIMARY_OUT_JSON
    _refuse_existing_run_artifact(output)
    pre = preflight(origins=origins)
    if pre["status"] != "PASS":
        raise EvolutionRevisionBlocked("preflight failed")
    if backend_mode != "live":
        raise EvolutionRevisionBlocked("the frozen trajectory requires the live backend")
    payload: dict[str, Any] = {
        "stage": STAGE,
        "status": "RUNNING",
        "started_at": _now(),
        "completed_at": None,
        "evidence_grade": EVIDENCE_GRADE,
        "preregistration": PREREGISTRATION.relative_to(PROJECT_ROOT).as_posix(),
        "trajectory": trajectory_plan(origins),
        "arms": arm_contract(),
        "budget": budget_contract(),
        "boundary": boundary_contract(),
        "preflight": pre,
        "results": {},
        "protocol_errors": [],
        "verdict": None,
    }
    _write(payload, path=output)
    try:
        initial = _load_old_snapshot()
        cells = {origin: _load_noaa_cell(origin) for origin in origins}
        identities = {
            origin: _identity_reading(cells[origin], origin) for origin in origins
        }
        eligible = forecast_p1._eligible_programs()
        spec, context = forecast_p1._task_contract(
            eligible, maximum_candidates=performance.B_MAIN
        )
        from evaluation.functional.task_episode_harness.agentic.runner import (
            live_transport,
        )

        payload["backend_target"] = live_transport(
            default_model=shared_harness.SLOW_MODEL
        )
        backend = shared_harness._live_backend(
            GLOBAL_LLM_MAX,
            on_budget_change=lambda state: (
                payload.update({"llm_budget_instrument": _plain(state)}),
                _write(payload, path=output),
            ),
        )
        payload["llm_budget_instrument"] = backend.budget_state()
        _write(payload, path=output)

        with tempfile.TemporaryDirectory(
            prefix="forecast_p4_evolution_revision_"
        ) as temporary:
            root = Path(temporary)
            a5 = _run_arm(
                arm="A5-Slow",
                initial=initial,
                cells=cells,
                identities=identities,
                origins=origins,
                backend=backend,
                spec=spec,
                context=context,
                root=root / "a5",
                run_stage_2=True,
                run_stage_3=True,
            )
            a5_pending = str(a5["stage_1"].get("slow_status") or "") == "pending"
            a5_approved = bool(a5["stage_2"].get("promotion_activated"))
            k0 = _run_arm(
                arm="K0-fixed",
                initial=initial,
                cells=cells,
                identities=identities,
                origins=origins,
                backend=backend,
                spec=spec,
                context=context,
                root=root / "k0",
                run_stage_2=a5_pending,
                run_stage_3=a5_approved,
            )

            a5_gain = a5["stage_3"].get("program_gain_vs_identity")
            k0_gain = k0["stage_3"].get("program_gain_vs_identity")
            delta = (
                float(a5_gain) - float(k0_gain)
                if a5_gain is not None and k0_gain is not None
                else None
            )
            evidence = {
                "budget_valid": bool(
                    a5["usage_valid"]
                    and k0["usage_valid"]
                    and int(backend.budget_state().get("global_calls") or 0)
                    <= GLOBAL_LLM_MAX
                    and int(a5["usage"].get("tokens") or 0) <= TOKEN_MAX
                    and int(k0["usage"].get("tokens") or 0) <= TOKEN_MAX
                ),
                "llm_budget_exhausted_before_completion": bool(
                    a5["llm_budget_exhausted"] or k0["llm_budget_exhausted"]
                ),
                "old_skill_qualified": pre["status"] == "PASS",
                "stage_1": dict(a5["stage_1"]),
                "stage_2": dict(a5["stage_2"]),
                "stage_3": {
                    "a5_v2_causally_used": bool(
                        a5["stage_3"].get("causal_skill_use")
                    ),
                    "k0_v1_causally_used": bool(
                        k0["stage_3"].get("causal_skill_use")
                    ),
                    "v2_minus_v1_utility": delta,
                    "v2_minus_identity_utility": a5_gain,
                    "a5_harm_count": int(a5["stage_3"].get("harm_count") or 0),
                    "k0_harm_count": int(k0["stage_3"].get("harm_count") or 0),
                    "a5_harm_magnitude": float(
                        a5["stage_3"].get("harm_magnitude") or 0.0
                    ),
                    "k0_harm_magnitude": float(
                        k0["stage_3"].get("harm_magnitude") or 0.0
                    ),
                    "a5_active_revision": a5["stage_3"].get("active_revision"),
                    "k0_active_revision": k0["stage_3"].get("active_revision"),
                },
            }
            verdict = derive_verdict(evidence)
            rolled_back = bool(
                a5.get("approved_snapshot") is not None
                and verdict
                != "EXPOSED_DEV_SLOW_PATCH_CHAIN_PASS__REPLICATION_REQUIRED__H3_HELD"
            )
            if rolled_back:
                a5["state"]["store"].set_active(initial.runtime_bundle_sha)
            payload["results"] = {
                "A5-Slow": _scientific_arm(a5),
                "K0-fixed": _scientific_arm(k0),
                "stage_3_comparison": {
                    "A5_minus_K0_utility": delta,
                    "A5_minus_identity_utility": a5_gain,
                    "approved_v2_rolled_back": rolled_back,
                },
                "evidence": _plain(evidence),
                "shared_identity_references": _plain(identities),
            }
            payload["llm_budget_instrument"] = backend.budget_state()
            returned_models = _returned_models(backend)
            requested_model = str(payload["backend_target"]["model"])
            model_contract_valid = validate_returned_models(
                requested_model=requested_model,
                returned_models=returned_models,
                calls=int(backend.budget_state().get("global_calls") or 0),
            )
            payload["backend_observed_returned_models"] = list(returned_models)
            payload["backend_model_contract_valid"] = model_contract_valid
            if not model_contract_valid:
                payload["verdict"] = (
                    "BACKEND_MODEL_MISMATCH__NO_SCIENTIFIC_VERDICT"
                )
                payload["status"] = "FAILED"
                payload["protocol_errors"].append(
                    {
                        "error": "returned model set does not equal the frozen requested model",
                        "requested_model": requested_model,
                        "returned_models": list(returned_models),
                    }
                )
            else:
                payload["verdict"] = verdict
                payload["status"] = (
                    "COMPLETE" if evidence["budget_valid"] else "FAILED"
                )
            payload["completed_at"] = _now()
    except Exception as exc:  # noqa: BLE001
        payload["status"] = "FAILED"
        payload["completed_at"] = _now()
        payload["protocol_errors"].append(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    _write(payload, path=output)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("live",), default="live")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    origins = REPEAT_ORIGINS if args.repeat else PRIMARY_ORIGINS
    result = (
        preflight(origins=origins)
        if args.preflight_only
        else run(origins=origins, backend_mode=args.backend)
    )
    print(json.dumps(_plain(result), ensure_ascii=False, indent=2))
    return 0 if result.get("status") in {"PASS", "COMPLETE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
