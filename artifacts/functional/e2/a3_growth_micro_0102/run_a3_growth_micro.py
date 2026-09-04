"""A3-only Target-local Skill growth live on e1v2_task_01/02.

Live D0/D1: agentic runner._run_arm / run_agentic_fast_path.
No Source Skill, no TRY, no A5, no Slow, no 9-task electricity.
No SelfEvolvingHarnessTS junction. PYTHONPATH = clone root (in-process alias).
"""
from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from collections.abc import Mapping
from typing import Any

# artifacts/functional/e2 may be a junction into another tree.  Never follow
# __file__.resolve() for ROOT; the live code must stay the work clone.
ROOT = Path.cwd()
TASK_IDS = ("e1v2_task_01", "e1v2_task_02")

_argv = list(sys.argv[1:])
while _argv:
    if _argv[0] == "--root":
        ROOT = Path(_argv[1]).resolve()
        _argv = _argv[2:]
    elif _argv[0] == "--cohort":
        COHORT = _argv[1]
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

from evaluation.functional.task_episode_harness.agentic import runner as agentic_runner  # noqa: E402
from evaluation.functional.task_episode_harness.e1 import (  # noqa: E402
    B,
    HORIZON,
    _ArmState,
    _frozen_task_roster,
    _inventory_rows,
    _skill_ids,
)
from evaluation.functional.task_episode_harness.g1 import (  # noqa: E402
    _w3_context_for,
    eval_substrate_preflight,
    train_substrate_preflight,
)
from evaluation.functional.task_episode_harness.normal_flow import (  # noqa: E402
    NF_BASE_URL,
    NF_MODEL,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    STATUS_LOCAL_ACTIVE,
    STATUS_LOCAL_DRAFT,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_METADATA,
    canonicalize,
)
from run_v1_kdd2018_natural_slow_update import _config  # noqa: E402

_run_arm = agentic_runner._run_arm
load_cohort = agentic_runner.load_cohort
_default_backend_factory = agentic_runner._default_backend_factory
LLM_CALL_BUDGET_PER_ARM_TASK = agentic_runner.LLM_CALL_BUDGET_PER_ARM_TASK
WORKSPACE_TOOL_BUDGET = agentic_runner.WORKSPACE_TOOL_BUDGET


def family_of_operator(name: str) -> str:
    return str(OPERATOR_METADATA[canonicalize(name)]["category"])


def proposal_family(candidate: dict) -> str:
    steps = candidate.get("steps") or ()
    if not steps:
        raise ValueError("no steps")
    return family_of_operator(str((steps[0] or {}).get("op") or ""))

COHORT = globals().get("COHORT", "T233")
RUN_REL = Path("artifacts/functional/e2/a3_growth_micro_0102")
STATE_REL = str(RUN_REL / "state")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_op(candidate: Mapping[str, Any]) -> str:
    steps = candidate.get("steps") or ()
    if not steps:
        return ""
    return canonicalize(str((steps[0] or {}).get("op") or ""))


def _families_of(candidates: list[Mapping[str, Any]]) -> list[str]:
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


def _receipt(task_id: str, arm_row: Mapping[str, Any]) -> dict[str, Any]:
    winner = arm_row.get("winner") or {}
    lifecycle = arm_row.get("lifecycle") or {}
    method_event = lifecycle.get("method_event") or {}
    delayed_event = lifecycle.get("delayed_event") or {}
    retrieved = arm_row.get("retrieved_knowledge_summary") or {}
    proposals = list(arm_row.get("proposals") or [])
    families = _families_of(proposals)
    first_ops = [_first_op(row) for row in proposals if isinstance(row, Mapping)]
    status = str(winner.get("local_status") or "")
    method_stage = str(method_event.get("stage") or "")
    delayed_stage = str(delayed_event.get("stage") or "")
    after_ids = list(arm_row.get("active_local_skill_ids_after") or [])
    local_active = (
        status == STATUS_LOCAL_ACTIVE
        or int((arm_row.get("metrics") or {}).get("task_local_active") or 0) == 1
        or delayed_stage == "approved"
    )
    draft = status in {STATUS_LOCAL_DRAFT, STATUS_LOCAL_ACTIVE} or method_stage == "pending"
    held_in = (
        delayed_stage == "approved"
        or (method_stage == "pending" and status == STATUS_LOCAL_ACTIVE)
        or (
            method_stage == "pending"
            and delayed_stage not in {"", "no_winner", "no_pending", "instrument_unreadable"}
            and status == STATUS_LOCAL_ACTIVE
        )
    )
    named_chain = method_stage == "pending" and delayed_stage == "approved"
    raw_episodes = retrieved.get("raw_episodes_in_fast_payload")
    fast_skill_only = (
        raw_episodes == 0
        and "episodes" not in retrieved
        and "memories" not in retrieved
    )
    growth_state = "无"
    if local_active:
        growth_state = "ACTIVE"
    elif draft or status == STATUS_LOCAL_DRAFT:
        growth_state = "DRAFT"
    if local_active:
        held_label = "有" if (held_in or named_chain or delayed_stage == "approved") else "不适用"
    elif draft:
        held_label = "有" if held_in or named_chain else "无"
    else:
        held_label = "不适用"
    return {
        "task_episode_id": task_id,
        "arm": "A3",
        "source_derived_skill": None,
        "source_prior": None,
        "TRY": None,
        "growth_state": growth_state,
        "held_in": held_label,
        "local_active": local_active,
        "draft": draft,
        "held_in_bool": held_in,
        "named_draft_held_in": named_chain,
        "winner_local_status": status or None,
        "method_event_stage": method_stage or None,
        "delayed_event_stage": delayed_stage or None,
        "method_event": _plain(method_event),
        "delayed_event": _plain(delayed_event),
        "winner": _plain(winner) if winner else None,
        "families": families,
        "first_ops": first_ops,
        "stop_reason": arm_row.get("stop_reason"),
        "protocol_error": arm_row.get("protocol_error"),
        "infrastructure_error": arm_row.get("infrastructure_error"),
        "metrics": _plain(arm_row.get("metrics") or {}),
        "cost": _plain(arm_row.get("cost") or {}),
        "retrieved_knowledge_summary": _plain(retrieved),
        "fast_skill_only": fast_skill_only,
        "raw_episodes_in_fast_payload": raw_episodes,
        "active_local_skill_ids_before": list(
            arm_row.get("active_local_skill_ids_before") or []
        ),
        "active_local_skill_ids_after": after_ids,
        "probes": _plain(arm_row.get("probes") or []),
        "lifecycle": _plain(lifecycle),
        "proposals": [
            {
                "candidate_id": row.get("candidate_id"),
                "first_op": _first_op(row),
                "family": fam,
                "ops": [
                    canonicalize(str(step.get("op") or ""))
                    for step in (row.get("steps") or ())
                    if isinstance(step, Mapping)
                ],
            }
            for row, fam in zip(
                [r for r in proposals if isinstance(r, Mapping)], families
            )
        ],
    }


def _score(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    any_active = any(r.get("local_active") for r in receipts.values())
    any_chain = any(
        r.get("named_draft_held_in") or r.get("held_in_bool")
        for r in receipts.values()
    )
    all_families: set[str] = set()
    for r in receipts.values():
        all_families.update(f for f in (r.get("families") or []) if f and f != "unknown")
    diverse = len(all_families) >= 2 or any(
        len(set(r.get("families") or [])) >= 2 for r in receipts.values()
    )
    fast_ok = all(r.get("fast_skill_only") for r in receipts.values()) if receipts else False
    passed = bool(any_active or any_chain)
    fail_diverse_no_growth = diverse and not any_active and not any_chain
    return {
        "pass": passed,
        "verdict": "过" if passed else "不过",
        "any_local_active": any_active,
        "any_draft_held_in": any_chain,
        "diverse_k": diverse,
        "all_families": sorted(all_families),
        "fast_skill_only": fast_ok,
        "fail_diverse_and_no_growth": fail_diverse_no_growth,
    }


def main() -> int:
    started = time.perf_counter()
    run_dir = ROOT / RUN_REL
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "A3_GROWTH_START root=%s cohort=%s tasks=%s runner=_run_arm/run_agentic_fast_path"
        % (ROOT, COHORT, ",".join(TASK_IDS)),
        flush=True,
    )
    cohort = load_cohort(ROOT, COHORT)
    roster = {str(s["task_episode_id"]): s for s in _frozen_task_roster()}
    missing = [tid for tid in TASK_IDS if tid not in roster]
    if missing:
        raise SystemExit("unknown task ids: %s" % missing)
    specs = [dict(roster[tid]) for tid in TASK_IDS]
    config = dict(_config())
    eval_pre = eval_substrate_preflight(
        cohort["values"], cohort["eval_uids"], specs
    )
    train_pre = train_substrate_preflight(
        cohort["values"], cohort["train_uids"],
        [int(a) for a in config["anchors"]],
    )
    preflight = {
        "eval_substrate_preflight": _plain(eval_pre),
        "train_substrate_preflight": _plain(train_pre),
    }
    (run_dir / "preflight.json").write_text(
        json.dumps(preflight, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    if not (eval_pre.get("pass") and train_pre.get("pass")):
        print("A3_GROWTH_PREFLIGHT_FAIL", flush=True)
        scoring = {
            "pass": False,
            "verdict": "不过",
            "reason": "substrate_preflight_failed",
            "fast_skill_only": False,
        }
        (run_dir / "scoring.json").write_text(
            json.dumps(
                {"kind": "a3_growth_micro_0102", "preflight": preflight, "scoring": scoring},
                indent=2, ensure_ascii=False, default=str,
            ) + "\n",
            encoding="utf-8",
        )
        lines = [
            "e1v2_task_01: 无 ; held-in 不适用",
            "e1v2_task_02: 无 ; held-in 不适用",
            "Fast 只见 Skill: 否",
            "不过",
            "runner: agentic _run_arm / run_agentic_fast_path; preflight failed; cohort=%s"
            % COHORT,
        ]
        (run_dir / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        return 2

    snapshot = compile_snapshot(ROOT / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(ROOT / STATE_REL / "A3" / "snapshots")
    store.materialize(snapshot)
    store.set_active(snapshot.runtime_bundle_sha)
    arm_state = _ArmState(
        arm="A3", memories=[], episodes=[], store=store,
        active_snapshot=snapshot,
        active_skill_ids=_skill_ids(snapshot, local_only=True),
    )

    receipts: dict[str, dict[str, Any]] = {}
    raw_rows: dict[str, Any] = {}
    for spec in specs:
        tid = str(spec["task_episode_id"])
        cutoff = int(spec["support_origins"][0])
        context = _w3_context_for(
            ROOT, STATE_REL, tid, cutoff,
            cohort["values"], cohort["train_uids"],
        )
        scope = list(context.get("scope_series_uids") or ())
        print(
            "A3_GROWTH_TASK_START %s cutoff=%s scope=%s"
            % (tid, cutoff, len(scope)),
            flush=True,
        )
        if not scope:
            row = {
                "task_episode_id": tid,
                "growth_state": "无",
                "held_in": "不适用",
                "local_active": False,
                "draft": False,
                "held_in_bool": False,
                "named_draft_held_in": False,
                "families": [],
                "fast_skill_only": True,
                "stop_reason": "EMPTY_SCOPE",
            }
            receipts[tid] = row
            (run_dir / ("%s.json" % tid)).write_text(
                json.dumps(row, indent=2, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            print("A3_GROWTH_TASK_SKIP %s empty scope" % tid, flush=True)
            continue
        inventory = _inventory_rows(context)
        arm_row = _run_arm(
            repo_root=ROOT,
            arm_state=arm_state,
            task_spec=spec,
            public_context=context,
            cohort=cohort,
            config=config,
            inventory=inventory,
            source_prior=None,
            workspace_tool_budget=WORKSPACE_TOOL_BUDGET,
            backend_factory=_default_backend_factory,
        )
        raw_rows[tid] = {
            key: _plain(value)
            for key, value in arm_row.items()
            if key not in {"stages", "tool_observations", "select_rounds"}
        }
        receipt = _receipt(tid, arm_row)
        receipts[tid] = receipt
        (run_dir / ("%s.json" % tid)).write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            "A3_GROWTH_TASK_DONE %s state=%s held=%s families=%s stop=%s "
            "active=%s method=%s delayed=%s llm=%s probes=%s"
            % (
                tid,
                receipt["growth_state"],
                receipt["held_in"],
                receipt["families"],
                receipt.get("stop_reason"),
                receipt["local_active"],
                receipt["method_event_stage"],
                receipt["delayed_event_stage"],
                (receipt.get("metrics") or {}).get("llm_calls"),
                (receipt.get("metrics") or {}).get("real_support_probe_count"),
            ),
            flush=True,
        )

    scoring = _score(receipts)
    llm_total = sum(
        int((r.get("metrics") or {}).get("llm_calls") or 0)
        for r in receipts.values()
    )
    probe_total = sum(
        int((r.get("metrics") or {}).get("real_support_probe_count") or 0)
        for r in receipts.values()
    )
    report = {
        "kind": "a3_growth_micro_0102",
        "arm": "A3",
        "source_derived_skill": None,
        "TRY": None,
        "cohort": COHORT,
        "task_ids": list(TASK_IDS),
        "model": NF_MODEL,
        "base_url": NF_BASE_URL,
        "llm_cap_per_arm_task": LLM_CALL_BUDGET_PER_ARM_TASK,
        "workspace_tool_budget": WORKSPACE_TOOL_BUDGET,
        "probe_budget_B": B,
        "horizon": HORIZON,
        "cycles": len(TASK_IDS),
        "no_a5": True,
        "no_slow": True,
        "no_source_skill": True,
        "no_try": True,
        "no_electricity_9task": True,
        "runner": "agentic runner._run_arm / run_agentic_fast_path",
        "preflight": preflight,
        "receipts": receipts,
        "raw_rows": raw_rows,
        "scoring": scoring,
        "llm_calls_total": llm_total,
        "real_support_probe_count_total": probe_total,
        "wall_seconds": time.perf_counter() - started,
    }
    (run_dir / "scoring.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    r01 = receipts.get("e1v2_task_01") or {}
    r02 = receipts.get("e1v2_task_02") or {}
    lines = [
        "e1v2_task_01: %s ; held-in %s ; K %s"
        % (
            r01.get("growth_state", "无"),
            r01.get("held_in", "不适用"),
            r01.get("families") or [],
        ),
        "e1v2_task_02: %s ; held-in %s ; K %s"
        % (
            r02.get("growth_state", "无"),
            r02.get("held_in", "不适用"),
            r02.get("families") or [],
        ),
        "Fast 只见 Skill: %s" % ("是" if scoring["fast_skill_only"] else "否"),
        scoring["verdict"],
        "runner: agentic _run_arm / run_agentic_fast_path; cycles=2 tasks; "
        "cohort=%s; llm_cap=%s/task; B=%s; llm_used=%s; probes=%s"
        % (COHORT, LLM_CALL_BUDGET_PER_ARM_TASK, B, llm_total, probe_total),
    ]
    (run_dir / "verdict.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("A3_GROWTH_VERDICT %s" % scoring["verdict"], flush=True)
    print("\n".join(lines), flush=True)
    return 0 if scoring["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
