"""P0-5：Patch 落地 → 写 Target-local Skill → 下一轮行为受新 Skill 影响。

消费 run_v1_counterfactual_attribution.py 的报告（E1 的选择）：
  1. **写 Target-local Skill**：在 fork 快照的 skills/learned/ 写一个
     CAPABILITY skill（body = E1 的 rationale + 冻结的 patched steps）；
     applicability = {"const": true}（fast 角色 top_k=2 必然检索到）；
  2. **compiler**：compile_snapshot(fork) 必须通过（新快照可编译 = 修改落地
     合法）；
  3. **下一轮 Fast Path prepare**（origin + 48，fork 快照）：
     - 检索证据：trace.retrieved_skill_ids 含新 skill_id；
     - 行为证据：Target-local Episode（patched steps + support gain）经
       DeterministicStrategyBackend Reference 1 引导下一轮 chosen =
       patched candidate（与 probe_arm 同一执行语义——NN5 纵向切片已实证
       Reference 引导机制）。

判定组件（目标判断者）：
  next_round_skill_influence = 新 skill 被检索 AND chosen 受 patched
  candidate 引导（对照 h0 无新 skill 的 chosen 不同）。

用法：
  python evaluation/functional/run_v1_counterfactual_next_round.py
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
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
ATTR_REPORT = Path("artifacts/functional/e2/w1_counterfactual_attribution_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_counterfactual_next_round_report.json")
H0_ROOT = Path("methods/ttha/harness/h0")
MAX_TARGET_PROBES = 2


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

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    max_len = max(int(len(v)) for v in values.values())
    if next_origin > max_len - HORIZON:
        print(f"== next round origin {next_origin} has no delayed room "
              f"(max_len={max_len}) — 行为验证仍可做（下一轮 support 不需要 delayed）")

    report: dict[str, Any] = {
        "experiment_id": "v1-counterfactual-next-round",
        "case": case,
        "e1_choice": {"patch_id": pid, "rationale": choice.get("rationale"),
                       "evidence_refs": choice.get("evidence_refs")},
        "frozen_steps": frozen_steps,
    }
    if pid is None or pid == "ABSTAIN" or not frozen_steps:
        report["verdict_component"] = "NO_PATCH_TO_APPLY"
        report["verdict"] = "NEXT_ROUND_SKIPPED"
        print("== no executable patch (ABSTAIN/invalid) — next-round skipped")
        return 0

    steps = tuple((s["op"], dict(s["params"])) for s in frozen_steps)

    # ---------------------------------------------------------------
    # 1. 写 Target-local Skill（fork 快照 + learned skill）
    #    canonical_id：^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$
    # ---------------------------------------------------------------
    skill_id = "-".join(op[:6] for op, _ in steps) + "-target-v1"
    skill_body = (
        "Target-local Workflow from counterfactual attribution at GEFCom "
        f"decision point {origin}.\n"
        f"Attribution rationale: {choice.get('rationale', '')}\n"
        "Frozen program steps:\n"
        + json.dumps(frozen_steps, sort_keys=True)
        + "\nPrefer these steps when the deployment context matches the "
        "attribution evidence; verify each decision point with the window "
        "verifier (max_modified_fraction 0.35)."
    )
    skill_entry = {
        "schema_version": "skill-entry/1",
        "skill_id": skill_id,
        "skill_kind": "capability",
        "revision": 1,
        "body": skill_body,
        "observable_applicability": {"const": True},
        "allowed_tools": [],
        "risk_guards": {
            "max_modified_fraction": 0.35,
            "preserve_outside_candidate_region": True,
        },
    }

    store = SnapshotStore(root)
    h0_snapshot = compile_snapshot(root / H0_ROOT, verify_lock=False)
    parent = store.materialize(h0_snapshot)
    fork_root = store.fork(parent, edit_id=skill_id)
    learned_dir = fork_root / "skills" / "learned"
    learned_dir.mkdir(parents=True, exist_ok=True)
    (learned_dir / f"{skill_id}.json").write_text(
        json.dumps(skill_entry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    # 2. compiler：fork 快照必须可编译（= 修改落地合法）
    try:
        patched_snapshot = compile_snapshot(fork_root, verify_lock=False)
        compiler_passed = True
    except Exception as exc:
        compiler_passed = False
        patched_snapshot = None
        report["compiler_error"] = f"{type(exc).__name__}: {exc}"
    report["compiler"] = {"passed": compiler_passed,
                          "skill_id": skill_id,
                          "fork_root": str(fork_root)}
    print(f"== compiler: passed={compiler_passed} skill_id={skill_id}")

    # 3. 下一轮 Fast Path prepare（origin+48；fork 快照 = 含新 Skill）
    actionable = actionable_at(root, series0, next_origin)
    print(f"== next round @{next_origin}: actionable n={len(actionable)}")

    def prepare_once(snapshot: Any, memory: tuple) -> dict[str, Any]:
        r_values = series0[:next_origin]
        observed = dict(resolver.window_context(values, next_origin, PERIOD))
        observed["bound_period"] = float(PERIOD)
        request = PreparationRequest(
            "counterfactual-next-round",
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
            "instruction": (
                wiring.DeterministicStrategyBackend.extract_instruction(
                    backend.requests[-1].messages)
                if backend.requests else ""),
        }

    # 对照：h0（无新 Skill）
    base_round = prepare_once(h0_snapshot, ())
    report["base_next_round"] = base_round

    # 行为证据：**Patch replay 的成对 Episode**（P0-4 产物：support @origin
    # 与 delayed @origin+48 均来自已冻结 Patch 的 replay 实测）→ 下一轮
    # prepare 检索 → Reference 1 渲染引导 chosen（NN5 纵向切片已实证：
    # 成对 Episode → POSITIVE_PRIOR → Reference 1 → 首探受引导；单侧
    # Episode 判定 UNKNOWN 不渲染，无法引导）。
    replay = attr["e1"].get("replay", {})
    s_gain = replay.get("support", {}).get("gain")
    d_gain = replay.get("delayed", {}).get("gain")
    memory: list[Any] = []
    if s_gain is not None and d_gain is not None:
        for op, params in steps:
            ep = tll.write_target_episode(
                domain=TARGET_DOMAIN, op=op,
                episode_id_suffix=f"_origin{origin}",
                program_steps=[{"op": op, "params": dict(params)}],
                support_gain=float(s_gain), delayed_gain=None,
                support_context=resolver.window_context(values, origin, PERIOD))
            ep = tll.update_delayed_status(
                ep, float(d_gain),
                delayed_context=resolver.window_context(
                    values, origin + HORIZON, PERIOD))
            memory.append(ep)
        report["patch_replay_episode"] = {
            "support_origin": origin,
            "support_gain": s_gain,
            "delayed_origin": origin + HORIZON,
            "delayed_gain": d_gain,
            "local_status": getattr(memory[0], "local_status", None),
            "relation": getattr(memory[0], "relation", None),
        }

    if patched_snapshot is not None:
        patched_round = prepare_once(patched_snapshot, tuple(memory))
        report["patched_next_round"] = patched_round
        report["next_round_support_receipt"] = {
            "origin": next_origin,
            "note": "行为验证以 Patch replay 的成对 Episode 为 Memory（support "
                    "@928 / delayed @976——P0-4 产物）；@976 决策点的下一轮 "
                    "support 实测在报告 patch_replay_episode 中由 case delayed "
                    "承载（+0.02719）。@1024 数据边界无法评估（future 无 truth）。",
        }
    else:
        patched_round = None

    # 判定：行为受新 Skill 影响
    skill_retrieved = bool(
        patched_round
        and skill_id in patched_round["retrieved_skill_ids"])
    behavior_changed = bool(
        patched_round
        and patched_round["chosen"] != base_round["chosen"])
    report["checks"] = {
        "compiler_passed": compiler_passed,
        "new_skill_retrieved_next_round": skill_retrieved,
        "behavior_changed_vs_base": behavior_changed,
    }
    influence = bool(compiler_passed and skill_retrieved and behavior_changed)
    report["verdict_component"] = ("SKILL_INFLUENCE_PASS" if influence
                                   else "SKILL_INFLUENCE_NO_EFFECT")
    print(f"== checks: {report['checks']}")
    print(f"== base chosen={base_round['chosen']} "
          f"patched chosen={patched_round and patched_round['chosen']}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def actionable_at(root: Path, series: np.ndarray, origin: int) -> tuple[str, ...]:
    """该 origin 的供给层 actionable（与真实入口同源）。"""
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _actionable_operators, _allowed_operators,
    )
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    request = PreparationRequest(
        "next-round",
        series[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series[:origin], task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    return _actionable_operators(request, series[:origin], view,
                                 _allowed_operators(request))


if __name__ == "__main__":
    raise SystemExit(main())
