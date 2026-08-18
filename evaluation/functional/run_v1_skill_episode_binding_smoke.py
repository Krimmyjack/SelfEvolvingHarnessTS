"""Skill×Episode 四格绑定 smoke（2026-08-09，文献建议的必需前置检查）。

背景：P0-5 的 behavior_changed_vs_base=True 混入了 Patch replay Episode
（support @928 / delayed @976 成对 Episode → Reference 1 引导）——未分离
Skill delta 单独的作用。文献建议（ACE/AHE 路线）必须先证明：

  只有 "Patch 后 Skill + 空 Episode" 能改变合法候选或 Workflow，
  才可称为 Executable Target-local Skill；
  否则当前产物只是 Memory 驱动的建议，first fault = Credit-to-Update Binding。

四格（同一下一轮 origin 976、同一 actionable、同一 deterministic backend）：

  格 1  原始快照 + 空 Episode        → 基线行为
  格 2  Patch 后快照 + 空 Episode    → Skill delta 单独作用
  格 3  原始快照 + 同 Episode        → Episode 单独作用
  格 4  Patch 后快照 + 同 Episode    → 组合（= P0-5 已验证）

判定：
  - chosen(2) != chosen(1)                → Skill delta 可执行
  - chosen(3) != chosen(1)                → Episode 单独作用
  - chosen(4) 与 (2)/(3) 的关系           → 叠加或冲突
  - 全不变                                → 产物是 Memory 驱动建议

零 LLM（确定性行为；Episode/Skill 为 P0 产物）。

用法：
  python evaluation/functional/run_v1_skill_episode_binding_smoke.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent, _actionable_operators, _allowed_operators  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway, extract_public_features  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
ATTR_REPORT = Path("artifacts/functional/e2/w1_counterfactual_attribution_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_skill_episode_binding_smoke_report.json")
H0_ROOT = Path("methods/ttha/harness/h0")


def _prepare_once(
    root: Path,
    snapshot: Any,
    memory: tuple,
    values: Mapping[str, Any],
    series0: np.ndarray,
    origin: int,
    actionable: tuple[str, ...],
) -> dict[str, Any]:
    r_values = series0[:origin]
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    request = PreparationRequest(
        "skill-episode-binding",
        r_values,
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )
    backend = wiring.DeterministicStrategyBackend(
        explore=True, operators=tuple(actionable))
    core_agent = TTHAAgentCore(
        backend, LocalPublicToolGateway(r_values, task_kind="forecast"))
    result, trace = TTHAFastAgent(core_agent).prepare(
        request, snapshot, experience_episodes=memory)
    return {
        "chosen": trace.chosen_candidate_id,
        "retrieved_skill_ids": list(trace.retrieved_skill_ids),
        "retrieved_memory_ids": list(trace.retrieved_memory_ids),
        "compilation": trace.compilation_status,
        "has_reference": "Reference 1" in (
            wiring.DeterministicStrategyBackend.extract_instruction(
                backend.requests[-1].messages)
            if backend.requests else ""),
    }


def _actionable_at(root: Path, series: np.ndarray, origin: int) -> tuple[str, ...]:
    h0 = compile_snapshot(root / H0_ROOT, verify_lock=False)
    request = PreparationRequest(
        "binding-smoke",
        series[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series[:origin], task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    return _actionable_operators(request, series[:origin], view,
                                 _allowed_operators(request))


def main() -> int:
    root = PROJECT_ROOT
    attr = json.loads((root / ATTR_REPORT).read_text(encoding="utf-8"))
    case = attr["case"]
    origin = int(case["origin"])
    next_origin = origin + HORIZON
    e1 = attr["e1"]
    choice = e1.get("choice", {})
    pid = choice.get("patch_id")
    frozen_steps = e1.get("frozen_steps")
    if pid is None or pid == "ABSTAIN" or not frozen_steps:
        print("== no executable patch — smoke skipped")
        return 0
    steps = tuple((s["op"], dict(s["params"])) for s in frozen_steps)

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    actionable = _actionable_at(root, series0, next_origin)
    print(f"== origin {next_origin}: actionable n={len(actionable)}")

    # Patch replay 成对 Episode（与 P0-5 相同——P0 产物）
    replay = e1.get("replay", {})
    s_gain = replay.get("support", {}).get("gain")
    d_gain = replay.get("delayed", {}).get("gain")
    assert s_gain is not None and d_gain is not None
    op = steps[0][0]
    ep = tll.write_target_episode(
        domain=TARGET_DOMAIN, op=op,
        episode_id_suffix=f"_origin{origin}",
        program_steps=[{"op": op, "params": dict(steps[0][1])}],
        support_gain=float(s_gain), delayed_gain=None,
        support_context=resolver.window_context(values, origin, PERIOD))
    ep = tll.update_delayed_status(
        ep, float(d_gain),
        delayed_context=resolver.window_context(values, origin + HORIZON, PERIOD))

    # Patch 后快照（fork + learned skill outlie-target-v1——P0-5 产物）
    h0_snapshot = compile_snapshot(root / H0_ROOT, verify_lock=False)
    store = SnapshotStore(root)
    parent = store.materialize(h0_snapshot)
    skill_id = "-".join(op[:6] for op, _ in steps) + "-target-v1"
    skill_body = (
        "Target-local Workflow from counterfactual attribution at GEFCom "
        f"decision point {origin}.\n"
        f"Attribution rationale: {choice.get('rationale', '')}\n"
        "Frozen program steps:\n" + json.dumps(frozen_steps, sort_keys=True) + "\n"
    )
    skill_entry = {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": 1,
        "body": skill_body,
        "observable_applicability": {"const": True},
        "allowed_tools": [],
        "risk_guards": {"max_modified_fraction": 0.35,
                        "preserve_outside_candidate_region": True},
    }
    fork_root = store.fork(parent, edit_id=skill_id)
    learned_dir = fork_root / "skills" / "learned"
    learned_dir.mkdir(parents=True, exist_ok=True)
    (learned_dir / f"{skill_id}.json").write_text(
        json.dumps(skill_entry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    patched_snapshot = compile_snapshot(fork_root, verify_lock=False)
    print(f"== compiler(fork): passed skill_id={skill_id}")

    # 四格
    cells = {
        "1_base": _prepare_once(root, h0_snapshot, (), values, series0,
                                next_origin, actionable),
        "2_skill_alone": _prepare_once(root, patched_snapshot, (), values,
                                       series0, next_origin, actionable),
        "3_episode_alone": _prepare_once(root, h0_snapshot, (ep,), values,
                                         series0, next_origin, actionable),
        "4_combined": _prepare_once(root, patched_snapshot, (ep,), values,
                                    series0, next_origin, actionable),
    }
    for name, cell in cells.items():
        print(f"== {name}: chosen={cell['chosen']} "
              f"skills={cell['retrieved_skill_ids']} "
              f"reference={cell['has_reference']}")

    base = cells["1_base"]["chosen"]
    skill_changes = cells["2_skill_alone"]["chosen"] != base
    episode_changes = cells["3_episode_alone"]["chosen"] != base
    combined = cells["4_combined"]["chosen"]
    combined_changes = combined != base

    checks = {
        "skill_alone_changes_behavior": skill_changes,
        "episode_alone_changes_behavior": episode_changes,
        "combined_changes_behavior": combined_changes,
        "skill_retrieved_in_patched_cells": bool(
            cells["2_skill_alone"]["retrieved_skill_ids"]
            and skill_id in cells["2_skill_alone"]["retrieved_skill_ids"]),
        "episode_reference_rendered_in_episode_cells": bool(
            cells["3_episode_alone"]["has_reference"]),
    }
    if skill_changes:
        binding = "EXECUTABLE_SKILL_DELTA"
    elif episode_changes:
        binding = "MEMORY_DRIVEN_RECOMMENDATION_CREDIT_TO_UPDATE_BINDING"
    elif combined_changes:
        binding = "COMBINED_ONLY_INTERACTION"
    else:
        binding = "NO_BEHAVIOR_CHANGE"
    verdict = f"SKILL_EPISODE_BINDING_{binding}"
    print(f"== checks: {json.dumps(checks, ensure_ascii=False)}")
    print(f"== verdict: {verdict}")

    report = {
        "experiment_id": "v1-skill-episode-binding-smoke",
        "case": case,
        "patch": {"patch_id": pid, "steps": frozen_steps,
                  "skill_id": skill_id},
        "episode": {
            "episode_id": ep.episode_id,
            "support_origin": origin,
            "support_gain": s_gain,
            "delayed_origin": origin + HORIZON,
            "delayed_gain": d_gain,
            "local_status": ep.local_status,
            "relation": ep.relation,
        },
        "cells": cells,
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": 0,
    }
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    store.discard_fork(fork_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
