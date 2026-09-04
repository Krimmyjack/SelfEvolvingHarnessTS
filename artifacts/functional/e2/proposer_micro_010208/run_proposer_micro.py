
"""A3-only Fast inspect+propose micro on e1v2_task_01/02/08.

Uses the live Fast generate path (proposal_family_gate). No A5, no Slow,
no Support/select, no electricity AB. No SelfEvolvingHarnessTS junction.
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
if len(sys.argv) > 1 and sys.argv[1] == "--root":
    ROOT = Path(sys.argv[2]).resolve()

# PYTHONPATH=clone root. Alias SelfEvolvingHarnessTS -> clone root in-process
# (not a filesystem junction; those recurse).
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(ROOT / "methods" / "ttha"))
if "SelfEvolvingHarnessTS" not in sys.modules:
    _pkg = types.ModuleType("SelfEvolvingHarnessTS")
    _pkg.__path__ = [str(ROOT)]
    _pkg.__file__ = str(ROOT / "__init__.py")
    sys.modules["SelfEvolvingHarnessTS"] = _pkg

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
    LLM_CALL_BUDGET_PER_ARM_TASK,
    WORKSPACE_TOOL_BUDGET,
    _default_backend_factory,
    _initial_context,
    load_cohort,
)
from SelfEvolvingHarnessTS.evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: E402
    AgentProtocolError,
    StagePostValidationError,
    TTHAAgentCore,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgentCallBudgetExceeded,
    AgentTransportError,
)
from SelfEvolvingHarnessTS.operators.registry import canonicalize  # noqa: E402

import numpy as np  # noqa: E402

TASK_IDS = ("e1v2_task_01", "e1v2_task_02", "e1v2_task_08")
COHORT = "e31"
RUN_REL = Path("artifacts/functional/e2/proposer_micro_010208")
STATE_REL = str(RUN_REL / "state")
# inspect + propose + one schema retry each, plus a few tool rounds
LLM_CAP = 12
TOOL_CAP = 6


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


def _score(per_task: dict[str, dict[str, Any]]) -> dict[str, Any]:
    monopoly_tasks: list[str] = []
    t01_rls_monopoly = False
    t0208_has_non_rls = {"e1v2_task_02": False, "e1v2_task_08": False}
    all_families: set[str] = set()
    for tid, row in per_task.items():
        families = list(row.get("families") or [])
        ops = list(row.get("first_ops") or [])
        all_families.update(f for f in families if f and f != "unknown")
        if families and len(set(families)) == 1:
            monopoly_tasks.append(tid)
        if tid == "e1v2_task_01" and ops and all(op == "repair_level_shift" for op in ops):
            t01_rls_monopoly = True
        if tid in t0208_has_non_rls:
            t0208_has_non_rls[tid] = any(
                (f and f != "structural") or (op and op != "repair_level_shift")
                for f, op in zip(families, ops)
            ) or any(f and f != "structural" for f in families)
    missing_non_rls = [tid for tid, ok in t0208_has_non_rls.items() if not ok]
    reasons: list[str] = []
    if monopoly_tasks:
        reasons.append("全员同一家族:" + ",".join(monopoly_tasks))
    if t01_rls_monopoly:
        reasons.append("01 独占 RLS")
    if missing_non_rls:
        reasons.append("02或08 缺非RLS:" + ",".join(missing_non_rls))
    if len(all_families) < 2:
        reasons.append("合计不足 2 家族")
    # empty K on any task is also a fail (no accepted diverse set)
    empty = [tid for tid, row in per_task.items() if not row.get("families")]
    if empty:
        reasons.append("全员同一家族:" + ",".join(empty) + "(empty/rejected)")
    passed = not reasons
    return {
        "pass": passed,
        "verdict": "过" if passed else "不过",
        "monopoly_tasks": monopoly_tasks,
        "task_01_rls_monopoly": t01_rls_monopoly,
        "task_02_non_rls": t0208_has_non_rls["e1v2_task_02"],
        "task_08_non_rls": t0208_has_non_rls["e1v2_task_08"],
        "all_families": sorted(all_families),
        "fail_rules": reasons,
        "required_k_families": REQUIRED_K_FAMILIES,
    }


def _run_one(
    *,
    repo_root: Path,
    spec: dict[str, Any],
    cohort: dict[str, Any],
    arm_state: _ArmState,
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
        arm_state.active_snapshot,
        dict(context["task_fast_features"]),
        role="fast",
    )
    local_skills = _retrieve_target_local_skills(
        arm_state.active_snapshot, context, arm="A3"
    )
    retrieved = {
        "general": {
            "carrier": "resolved Harness view (instruction, Skills, controls)",
            "candidate_policy": _plain(
                harness_view.controls.get("candidate_policy") or {}
            ),
            "retrieved_skill_ids": list(harness_view.skill_ids),
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
    try:
        inspect_payload = _run_stage(
            core, trace,
            stage="inspect",
            case_id=f"A3_{task_id}",
            public_input=inspect_input,
            harness_view=harness_view,
            schema_name="fast_inspect_v1",
            post_validator=lambda payload: _ground_inspect(core, trace, payload),
        )
    except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
        protocol_error = f"inspect: {type(exc).__name__}: {exc}"
    except (AgentTransportError, AgentCallBudgetExceeded) as exc:
        infrastructure_error = f"inspect: {type(exc).__name__}: {exc}"

    propose_payload = None
    if inspect_payload is not None and protocol_error is None and infrastructure_error is None:
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
                "all repair_level_shift. You choose the Program family; the "
                "Runtime owns where inside each action unit the edit lands. "
                "Each candidate costs one Target Support probe from a "
                "non-renewable budget." % int(B)
            ),
        }
        try:
            propose_payload = _run_stage(
                core, trace,
                stage="propose",
                case_id=f"A3_{task_id}",
                public_input=propose_input,
                harness_view=harness_view,
                schema_name="fast_propose_v1",
                post_validator=lambda payload: _validate_propose(
                    payload, inspect_payload
                ),
            )
        except (AgentProtocolError, StagePostValidationError, PermissionError) as exc:
            protocol_error = f"propose: {type(exc).__name__}: {exc}"
        except (AgentTransportError, AgentCallBudgetExceeded) as exc:
            infrastructure_error = f"propose: {type(exc).__name__}: {exc}"

    candidates = [
        _plain(row)
        for row in list((propose_payload or {}).get("candidates") or ())
        if isinstance(row, dict)
    ]
    families = _families_of(candidates)
    first_ops = [_first_op(row) for row in candidates]
    propose_stage = next((s for s in trace.stages if s["stage"] == "propose"), None)
    return {
        "task_episode_id": task_id,
        "arm": "A3",
        "horizon": HORIZON,
        "observation_cutoff": cutoff,
        "scope_series_count": len(scope),
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
                    if isinstance(step, dict)
                ],
            }
            for row, fam in zip(candidates, families)
        ],
        "protocol_error": protocol_error,
        "infrastructure_error": infrastructure_error,
        "validation_retry_count": (propose_stage or {}).get("validation_retry_count"),
        "validation_error_codes": (propose_stage or {}).get("validation_error_codes"),
        "llm_calls": int(getattr(backend, "calls", 0)),
        "workspace_tool_calls": int(getattr(gateway, "calls", 0)),
        "inspect_hypothesis_ids": [
            str(h.get("hypothesis_id"))
            for h in list((inspect_payload or {}).get("pattern_hypotheses") or ())
            if isinstance(h, dict)
        ],
    }


def main() -> int:
    started = time.perf_counter()
    run_dir = ROOT / RUN_REL
    run_dir.mkdir(parents=True, exist_ok=True)
    print("MICRO_START root=%s cohort=%s tasks=%s" % (ROOT, COHORT, ",".join(TASK_IDS)), flush=True)
    cohort = load_cohort(ROOT, COHORT)
    roster = {str(s["task_episode_id"]): s for s in _frozen_task_roster()}
    missing = [tid for tid in TASK_IDS if tid not in roster]
    if missing:
        raise SystemExit("unknown task ids: %s" % missing)

    snapshot = compile_snapshot(ROOT / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(ROOT / STATE_REL / "A3" / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    arm_state = _ArmState(
        arm="A3", memories=[], episodes=[], store=store,
        active_snapshot=snapshot,
        active_skill_ids=_skill_ids(snapshot, local_only=True),
    )

    rows: dict[str, dict[str, Any]] = {}
    for tid in TASK_IDS:
        print("MICRO_TASK_START %s" % tid, flush=True)
        row = _run_one(repo_root=ROOT, spec=dict(roster[tid]), cohort=cohort, arm_state=arm_state)
        rows[tid] = row
        print(
            "MICRO_TASK_DONE %s families=%s ops=%s proto=%s infra=%s llm=%s"
            % (
                tid, row.get("families"), row.get("first_ops"),
                row.get("protocol_error"), row.get("infrastructure_error"),
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
        "kind": "proposer_micro_010208",
        "arm": "A3",
        "cohort": COHORT,
        "task_ids": list(TASK_IDS),
        "model": NF_MODEL,
        "base_url": NF_BASE_URL,
        "llm_cap_per_task": LLM_CAP,
        "no_support": True,
        "no_select": True,
        "no_slow": True,
        "no_a5": True,
        "rows": rows,
        "scoring": scoring,
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "scoring.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    print("MICRO_VERDICT %s" % scoring["verdict"], flush=True)
    print(json.dumps({tid: rows[tid]["families"] for tid in TASK_IDS}, ensure_ascii=False), flush=True)
    return 0 if scoring["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
