"""Wrong-prior micro on e1v2_task_01/02.

One arm only. A deliberately mismatched Source Skill is frozen into the
Harness the same way A5 would see source_investigation_v1, then the live
Fast inspect+propose path runs (run_agentic_fast_path generate stages).
No matching-prior arm. No Support/select. No 9-task electricity. No
SelfEvolvingHarnessTS filesystem junction. Proposer menu untouched.
"""
from __future__ import annotations

import json
import sys
import time
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
_argv = list(sys.argv[1:])
while _argv:
    if _argv[0] == "--root":
        ROOT = Path(_argv[1]).resolve()
        _argv = _argv[2:]
    else:
        break

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))
if "SelfEvolvingHarnessTS" not in sys.modules:
    _pkg = types.ModuleType("SelfEvolvingHarnessTS")
    _pkg.__path__ = [str(ROOT)]
    _pkg.__file__ = str(ROOT / "__init__.py")
    sys.modules["SelfEvolvingHarnessTS"] = _pkg

from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: E402
    EditManifest,
    EditOperation,
    load_skill_entry,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.e1 import (  # noqa: E402
    B,
    HORIZON,
    _ArmState,
    _frozen_task_roster,
    _inventory_rows,
    _retrieve_target_local_skills,
    _skill_ids,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.g1 import (  # noqa: E402
    _w3_context_for,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.fast_path import (  # noqa: E402
    FastPathTrace,
    _ground_inspect,
    _plain,
    _run_stage,
    _validate_propose,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.proposal_family_gate import (  # noqa: E402
    REQUIRED_K_FAMILIES,
    family_of_operator,
    proposal_family,
    workflow_family_menu,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.gateway import (  # noqa: E402
    CohortScopePublicToolGateway,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.runner import (  # noqa: E402
    SOURCE_SKILL_CAUSE,
    _default_backend_factory,
    _initial_context,
    load_cohort,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.source_probe_gate import (  # noqa: E402
    evaluate_source_probe_gate,
    observation_flagged_families,
    observation_from_tool_rows,
    source_prior_grants_execution,
    source_skill_from_view,
    try_clause_family,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.agentic.source_skill import (  # noqa: E402
    SOURCE_SKILL_ID,
    SOURCE_APPLICABILITY,
    build_skill_payload,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import _resolve_apply_manifest  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import canonicalize  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentTransportError,
)

import numpy as np  # noqa: E402
from dataclasses import replace  # noqa: E402

TASK_IDS = ("e1v2_task_01", "e1v2_task_02")
COHORT = "e31"
RUN_REL = Path("artifacts/functional/e2/wrong_prior_micro_0102")
STATE_REL = str(RUN_REL / "state")
LLM_CAP = 12
TOOL_CAP = 6

# Deliberately mismatched Source TRY per task. Matching prior is not a pass
# condition this round.
WRONG_PRIORS = {
    "e1v2_task_01": {
        # 01 already supports structural+outlier as first-fault split.
        # Impute-only is the wrong sole family.
        "try": "Lead with period_complete as the sole prior on this forecast Task.",
        "family": "impute",
        "operator": "period_complete",
        "label": "impute-only period_complete",
    },
    "e1v2_task_02": {
        # 02 is outlier-split. Open-loop RLS / level-shift is the R2/D1 mode.
        "try": "Lead with repair_level_shift on prior expectation.",
        "family": "structural",
        "operator": "repair_level_shift",
        "label": "open-loop RLS / level-shift / structural",
    },
}


def _sections(try_text: str) -> dict[str, str]:
    return {
        "WHEN": "A forecast Task whose public inspection is available.",
        "OBSERVE": "Read the public features already on this Workspace.",
        "TRY": try_text,
        "RISK": "Do not treat the Source note as confirmation.",
        "VERIFY": "Believe nothing until this Task's own Target Support is spent.",
        "FALLBACK": "Keep an effect-distinct candidate supported by the observation.",
    }


def _first_op(candidate: dict[str, Any]) -> str:
    steps = candidate.get("steps") or ()
    if not steps:
        return ""
    return canonicalize(str((steps[0] or {}).get("op") or ""))


def _families_of(candidates: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for row in candidates:
        try:
            out.append(proposal_family(row))
        except Exception:
            op = _first_op(row)
            try:
                out.append(family_of_operator(op) if op else "unknown")
            except Exception:
                out.append("unknown")
    return out


def _exc_codes(exc: BaseException) -> list[str]:
    codes = list(getattr(exc, "validation_error_codes", ()) or ())
    code = getattr(exc, "error_code", None)
    if code and code not in codes:
        codes.append(str(code))
    return codes


def _freeze_source_skill(
    *,
    repo_root: Path,
    store_root: Path,
    try_text: str,
) -> tuple[Any, dict[str, Any]]:
    """ADD source_investigation_v1 the same way A5 freezes a Source Skill."""
    entry = build_skill_payload(_sections(try_text))
    assert entry["skill_id"] == SOURCE_SKILL_ID
    store_root.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(store_root)
    base = compile_snapshot(repo_root / "methods/ttha/harness/h0", verify_lock=False)
    store.materialize(base)
    store.set_active(base.runtime_bundle_sha)
    meta: dict[str, Any] = {
        "skill_id": SOURCE_SKILL_ID,
        "try": try_text,
        "try_family": try_clause_family(entry),
        "inject": "edit_controller_add",
    }
    try:
        controller = EditController(
            store, surfaces=SurfaceRegistry(), router=FaultRouter()
        )
        manifest = EditManifest(
            edit_id=SOURCE_SKILL_ID,
            base_harness_sha=base.harness_content_sha,
            target_pattern_id="g3d1-source-derived-general-skill",
            target_surface_id="skill_library.entries/" + SOURCE_SKILL_ID,
            operation=EditOperation.ADD,
            surface_precondition={"kind": "ABSENT"},
            dependency_precondition_shas={},
            new_value=entry,
            observable_applicability=dict(SOURCE_APPLICABILITY),
            predicted_agent_behavior_change=(
                "retrieve_skill:" + SOURCE_SKILL_ID,
                "supply_effect_distinct",
            ),
            predicted_data_effect=("earlier_local_skill_formation",),
            automatically_selected_risk_cases=(),
            falsification_condition=("no_improvement",),
            patch_id=None,
        )
        parent = store.materialize(base)
        receipt = controller.apply_to_fork(
            parent,
            _resolve_apply_manifest(manifest, base),
            confirmed_cause=SOURCE_SKILL_CAUSE,
        )
        frozen = compile_snapshot(receipt.candidate_root, verify_lock=False)
        store.set_active(frozen.runtime_bundle_sha)
        meta.update({
            "frozen_runtime_bundle_sha": frozen.runtime_bundle_sha,
            "frozen_harness_content_sha": frozen.harness_content_sha,
        })
        return frozen, meta
    except Exception as exc:  # noqa: BLE001
        # Same retrieved view A5 would see; do not write into h0.
        skill = load_skill_entry(entry)
        frozen = replace(
            base,
            skills=tuple(sorted((*base.skills, skill), key=lambda s: s.skill_id)),
        )
        meta.update({
            "inject": "snapshot_replace_fallback",
            "inject_error": "%s: %s" % (type(exc).__name__, exc),
        })
        return frozen, meta


def _run_one(
    *,
    repo_root: Path,
    spec: dict[str, Any],
    cohort: dict[str, Any],
    snapshot: Any,
    store: SnapshotStore,
    prior: dict[str, Any],
) -> dict[str, Any]:
    task_id = str(spec["task_episode_id"])
    cutoff = int(spec["support_origins"][0])
    context = _w3_context_for(
        repo_root, STATE_REL, task_id, cutoff,
        cohort["values"], cohort["train_uids"],
    )
    scope = list(context.get("scope_series_uids") or ())
    if not scope:
        return {
            "task_episode_id": task_id,
            "families": [],
            "first_ops": [],
            "candidates": [],
            "stop_reason": "EMPTY_SCOPE",
            "protocol_error": "EMPTY_SCOPE",
            "propose_entered": False,
            "gate_fired": False,
        }
    inventory = _inventory_rows(context)
    gateway = CohortScopePublicToolGateway(
        {uid: np.asarray(cohort["values"][uid], dtype=np.float64)[:cutoff] for uid in scope},
        task_kind=str(context["task_kind"]),
        observation_cutoff=cutoff,
        maximum_calls=TOOL_CAP,
    )
    backend = _default_backend_factory(LLM_CAP)
    core = TTHAAgentCore(backend, gateway, model=NF_MODEL, base_url=NF_BASE_URL)
    harness_view = resolve_harness_view(
        snapshot,
        dict(context["task_fast_features"]),
        role="fast",
    )
    served_source = source_skill_from_view(harness_view)
    local_skills = _retrieve_target_local_skills(
        snapshot, context, arm="A5"
    )
    retrieved = {
        "general": {
            "carrier": "resolved Harness view (instruction, Skills, controls)",
            "candidate_policy": _plain(
                harness_view.controls.get("candidate_policy") or {}
            ),
            "retrieved_skill_ids": list(harness_view.skill_ids),
            "source_skill_id": SOURCE_SKILL_ID,
            "source_skill_served": served_source is not None,
            "source_prior_note": (
                "Frozen Source Skill advises proposal order only. It is not "
                "confirmation and not an execution right."
            ),
        },
        "specific_target_local_skills": [
            {
                "skill_id": row.get("skill_id"),
                "frozen_program_steps": row.get("frozen_program_steps"),
                "risk_guards": row.get("risk_guards"),
                "applies_in_current_context": True,
            }
            for row in local_skills
            if row.get("retrieved_in_current_context")
        ],
        "target_support_this_trajectory": (
            "Propose-only micro: no Support probes are run."
        ),
    }
    initial_context = _initial_context(
        task_spec=spec,
        public_context=context,
        cohort=cohort,
        workspace_tool_budget=TOOL_CAP,
    )
    menu = [
        {
            "name": row["name"],
            "category": row.get("category"),
            "availability": row.get("availability"),
            "destructive": row.get("destructive"),
            "preserves_observed": row.get("preserves_observed"),
            "targeting_mode": row.get("targeting_mode"),
            "public_parameter_schema": row.get("public_parameter_schema"),
            **({"reason": row.get("reason")} if row.get("availability") != "EXECUTABLE" else {}),
        }
        for row in inventory
    ]
    inspect_input = {
        **dict(initial_context),
        "operator_contracts": menu,
        "target_support_budget": int(B),
        "material_threshold": 0.05,
        "retrieved_knowledge": dict(retrieved),
        "stage_note": (
            "You have no per-series numbers yet. Call the Workspace tools on "
            "the scoped series you want to see, then report what you inspected."
        ),
    }
    trace = FastPathTrace()
    protocol_error = None
    infrastructure_error = None
    inspect_payload = None
    validation_retry_count = None
    validation_error_codes: list[str] = []
    propose_entered = False
    try:
        inspect_payload = _run_stage(
            core, trace,
            stage="inspect",
            case_id=f"A5_{task_id}",
            public_input=inspect_input,
            harness_view=harness_view,
            schema_name="fast_inspect_v1",
            post_validator=lambda payload: _ground_inspect(core, trace, payload),
        )
    except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
        protocol_error = f"inspect: {type(exc).__name__}: {exc}"
        validation_retry_count = getattr(exc, "validation_retry_count", None)
        validation_error_codes = _exc_codes(exc)
    except (AgentTransportError, AgentCallBudgetExceeded) as exc:
        infrastructure_error = f"inspect: {type(exc).__name__}: {exc}"

    propose_payload = None
    observation: dict[str, Any] = {}
    if inspect_payload is not None and protocol_error is None and infrastructure_error is None:
        observation = observation_from_tool_rows(
            trace.tool_observations,
            extra=dict(initial_context.get("task_fast_features") or {}),
        )
        family_menu = workflow_family_menu()
        propose_input = {
            **inspect_input,
            "inspect_result": _plain(inspect_payload),
            "tool_observations": [dict(row) for row in trace.tool_observations],
            "workflow_family_menu": family_menu,
            "required_k_families": REQUIRED_K_FAMILIES,
            "stage_note": (
                "Propose one to %d Typed Workflow candidates. K candidates must "
                "be mutually distinct by mechanism family from "
                "workflow_family_menu (and by hypothesis+family+binding if that "
                "tuple repeats). All-K the same family is rejected, especially "
                "all repair_level_shift. Empty candidates / abstain is not a "
                "legal propose completion. You choose the Program family; the "
                "Runtime owns where inside each action unit the edit lands. "
                "Each candidate costs one Target Support probe from a "
                "non-renewable budget." % int(B)
            ),
        }
        propose_entered = True
        try:
            propose_payload = _run_stage(
                core, trace,
                stage="propose",
                case_id=f"A5_{task_id}",
                public_input=propose_input,
                harness_view=harness_view,
                schema_name="fast_propose_v1",
                post_validator=lambda payload: _validate_propose(
                    payload,
                    inspect_payload,
                    harness_view=harness_view,
                    observation=observation,
                    trace=trace,
                ),
            )
        except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
            protocol_error = f"propose: {type(exc).__name__}: {exc}"
            validation_retry_count = getattr(exc, "validation_retry_count", None)
            validation_error_codes = _exc_codes(exc)
        except (AgentTransportError, AgentCallBudgetExceeded) as exc:
            infrastructure_error = f"propose: {type(exc).__name__}: {exc}"

    propose_stage = next((s for s in trace.stages if s["stage"] == "propose"), None)
    if propose_stage is not None:
        if validation_retry_count is None:
            validation_retry_count = propose_stage.get("validation_retry_count")
        staged_codes = list(propose_stage.get("validation_error_codes") or [])
        for code in staged_codes:
            if code not in validation_error_codes:
                validation_error_codes.append(code)

    raw_candidates = list((propose_payload or {}).get("candidates") or ())
    candidates = [
        _plain(row)
        for row in raw_candidates
        if isinstance(row, Mapping)
    ]
    families = _families_of(candidates)
    first_ops = [_first_op(row) for row in candidates]

    receipt = evaluate_source_probe_gate(
        source_skill=served_source,
        observation=observation or observation_from_tool_rows(trace.tool_observations),
        candidates=candidates,
        raise_on_violation=False,
    )
    receipt_dict = receipt.as_dict()
    monopolized = bool(
        families
        and set(f for f in families if f and f != "unknown") == {prior["family"]}
    )
    treated_as_confirm = bool(
        receipt.is_authorization
        or receipt.is_confirmation
        or receipt.execution_authorized
        or source_prior_grants_execution(receipt)
        or receipt.may_monopolize_k
    )
    return {
        "task_episode_id": task_id,
        "arm": "A5",
        "horizon": HORIZON,
        "observation_cutoff": cutoff,
        "scope_series_count": len(scope),
        "wrong_prior_label": prior["label"],
        "wrong_prior_family": prior["family"],
        "wrong_prior_operator": prior["operator"],
        "wrong_prior_try": prior["try"],
        "source_skill_id": SOURCE_SKILL_ID,
        "source_skill_served": served_source is not None,
        "retrieved_skill_ids": list(harness_view.skill_ids),
        "observation_families": sorted(observation_flagged_families(
            observation or observation_from_tool_rows(trace.tool_observations)
        )),
        "gate_receipt": receipt_dict,
        "gate_fired": bool(receipt.fired),
        "gate_match": bool(receipt.match),
        "gate_error_code": receipt.error_code,
        "is_authorization": bool(receipt.is_authorization),
        "is_confirmation": bool(receipt.is_confirmation),
        "execution_authorized": bool(receipt.execution_authorized),
        "may_monopolize_k": bool(receipt.may_monopolize_k),
        "source_prior_grants_execution": source_prior_grants_execution(receipt),
        "wrong_family_monopolized_k": monopolized,
        "wrong_prior_treated_as_confirmation": treated_as_confirm,
        "source_probe_receipts": list(trace.source_probe_receipts),
        "families": families,
        "first_ops": first_ops,
        "candidates": [
            {
                "candidate_id": row.get("candidate_id"),
                "addresses_hypothesis_id": row.get("addresses_hypothesis_id"),
                "first_op": _first_op(row),
                "family": fam,
                "ops": [
                    canonicalize(str(step.get("op") or ""))
                    for step in (row.get("steps") or ())
                    if isinstance(step, Mapping)
                ],
            }
            for row, fam in zip(candidates, families)
        ],
        "protocol_error": protocol_error,
        "infrastructure_error": infrastructure_error,
        "propose_entered": propose_entered,
        "retries_happened": bool(validation_retry_count) or bool(validation_error_codes),
        "validation_retry_count": validation_retry_count,
        "validation_error_codes": validation_error_codes,
        "llm_calls": int(getattr(backend, "calls", 0)),
        "workspace_tool_calls": int(getattr(gateway, "calls", 0)),
        "raw_candidate_count": len(raw_candidates) if propose_payload is not None else 0,
        "propose_payload_keys": sorted((propose_payload or {}).keys()) if propose_payload is not None else [],
        "propose_stage_payload": _plain((propose_stage or {}).get("payload") or {}),
        "inspect_hypothesis_ids": [
            str(h.get("hypothesis_id"))
            for h in list((inspect_payload or {}).get("pattern_hypotheses") or ())
            if isinstance(h, Mapping)
        ],
    }


def _score(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    fired = {tid: bool((rows.get(tid) or {}).get("gate_fired")) for tid in TASK_IDS}
    monopolized = any(
        bool((rows.get(tid) or {}).get("wrong_family_monopolized_k"))
        for tid in TASK_IDS
    )
    as_confirm = any(
        bool((rows.get(tid) or {}).get("wrong_prior_treated_as_confirmation"))
        for tid in TASK_IDS
    )
    silent = [tid for tid, yes in fired.items() if not yes]
    reasons: list[str] = []
    if silent:
        reasons.append("gate silent: " + ",".join(silent))
    if monopolized:
        reasons.append("wrong-family prior monopolized K")
    if as_confirm:
        reasons.append("wrong prior counted as confirmation/authorization")
    passed = not reasons
    return {
        "pass": passed,
        "verdict": "过" if passed else "不过",
        "gate_fired": fired,
        "wrong_family_monopolized_k": monopolized,
        "wrong_prior_treated_as_confirmation": as_confirm,
        "fail_rules": reasons,
    }


def main() -> int:
    started = time.perf_counter()
    run_dir = ROOT / RUN_REL
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "WRONG_PRIOR_MICRO_START root=%s cohort=%s tasks=%s"
        % (ROOT, COHORT, ",".join(TASK_IDS)),
        flush=True,
    )
    cohort = load_cohort(ROOT, COHORT)
    roster = {str(s["task_episode_id"]): s for s in _frozen_task_roster()}
    missing = [tid for tid in TASK_IDS if tid not in roster]
    if missing:
        raise SystemExit("unknown task ids: %s" % missing)

    rows: dict[str, dict[str, Any]] = {}
    inject_meta: dict[str, Any] = {}
    for tid in TASK_IDS:
        prior = WRONG_PRIORS[tid]
        print(
            "WRONG_PRIOR_MICRO_TASK_START %s prior=%s try=%s"
            % (tid, prior["label"], prior["try"]),
            flush=True,
        )
        store_root = ROOT / STATE_REL / tid / "A5" / "snapshots"
        snapshot, meta = _freeze_source_skill(
            repo_root=ROOT, store_root=store_root, try_text=prior["try"]
        )
        inject_meta[tid] = meta
        if try_clause_family(build_skill_payload(_sections(prior["try"]))) != prior["family"]:
            raise SystemExit(
                "attached TRY family mismatch for %s: wanted %s"
                % (tid, prior["family"])
            )
        store = SnapshotStore(store_root)
        arm_state = _ArmState(
            arm="A5", memories=[], episodes=[], store=store,
            active_snapshot=snapshot,
            active_skill_ids=_skill_ids(snapshot, local_only=False),
        )
        del arm_state  # constructed so the snapshot is the A5 frozen view
        row = _run_one(
            repo_root=ROOT,
            spec=dict(roster[tid]),
            cohort=cohort,
            snapshot=snapshot,
            store=store,
            prior=prior,
        )
        row["inject"] = meta
        rows[tid] = row
        print(
            "WRONG_PRIOR_MICRO_TASK_DONE %s gate=%s families=%s ops=%s "
            "obs=%s monopoly=%s confirm=%s proto=%s infra=%s llm=%s"
            % (
                tid,
                "FIRE" if row.get("gate_fired") else "SILENT",
                row.get("families"),
                row.get("first_ops"),
                row.get("observation_families"),
                row.get("wrong_family_monopolized_k"),
                row.get("wrong_prior_treated_as_confirmation"),
                row.get("protocol_error"),
                row.get("infrastructure_error"),
                row.get("llm_calls"),
            ),
            flush=True,
        )
        (run_dir / ("%s.json" % tid)).write_text(
            json.dumps(row, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

    scoring = _score(rows)
    report = {
        "kind": "wrong_prior_micro_0102",
        "arm": "A5",
        "cohort": COHORT,
        "task_ids": list(TASK_IDS),
        "matching_prior_arm": False,
        "model": NF_MODEL,
        "base_url": NF_BASE_URL,
        "llm_cap_per_task": LLM_CAP,
        "no_support": True,
        "no_select": True,
        "no_slow": True,
        "no_electricity": True,
        "no_m0b": True,
        "proposer_menu_unchanged": True,
        "attached_priors": {
            tid: {
                "label": WRONG_PRIORS[tid]["label"],
                "family": WRONG_PRIORS[tid]["family"],
                "try": WRONG_PRIORS[tid]["try"],
            }
            for tid in TASK_IDS
        },
        "inject": inject_meta,
        "runner": (
            "run_wrong_prior_micro.py A5 frozen Source Skill + Fast "
            "inspect+_run_stage propose / _validate_propose "
            "(run_agentic_fast_path generate path)"
        ),
        "rows": rows,
        "scoring": scoring,
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "scoring.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    def _gate_word(tid: str) -> str:
        return "响" if rows[tid].get("gate_fired") else "不响"

    monopoly_word = (
        "有"
        if scoring["wrong_family_monopolized_k"]
        or scoring["wrong_prior_treated_as_confirmation"]
        else "无"
    )
    lines = [
        "e1v2_task_01 门: %s ; K: %s"
        % (_gate_word("e1v2_task_01"), rows["e1v2_task_01"].get("families") or []),
        "e1v2_task_02 门: %s ; K: %s"
        % (_gate_word("e1v2_task_02"), rows["e1v2_task_02"].get("families") or []),
        "错家族独占K/当确认: %s" % monopoly_word,
        scoring["verdict"],
        "wrong priors: 01=%s ; 02=%s"
        % (WRONG_PRIORS["e1v2_task_01"]["label"], WRONG_PRIORS["e1v2_task_02"]["label"]),
    ]
    (run_dir / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("WRONG_PRIOR_MICRO_VERDICT %s" % scoring["verdict"], flush=True)
    print("\n".join(lines), flush=True)
    return 0 if scoring["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
