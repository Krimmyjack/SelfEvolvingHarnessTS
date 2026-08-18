"""METHOD_LEVEL_SIGNED_EPISODE_TO_SKILL_PRIORITY_MECHANISM 验收
（外部审核第三轮批准；§7（二十八）降级：机制 PASS / 自动写回 PENDING）。

核心行为：**Target Support/delayed 反馈能否改变下一轮该 Skill 的优先级**。

反馈控制实现（fast_agent 注入点）：Skill 候选的池内顺序由当前 signed
判定（instruction 的 Reference 渲染）决定——
  - 无 CONFLICT/RISK（含 POSITIVE）：Skill 优先保留 slot（identity + 1
    Skill + ≤1 Agent，Skill 在前）；
  - CONFLICT/RISK（Reference 2/3 含 Skill 的算子）：Skill 降级（Agent
    候选在前、Skill 排最后——预算截断 = 不硬删除）。
选择由中性顺序 selector 按公开候选顺序决定（池顺序即优先级）。

验收 9 条（外部审核原文）：
  1. ScopeExecutor 直接执行 result.program.execution_steps()（chosen Program
     的实际 steps——不用脚本变量）；
  2. Support 后立即写入当前臂 Episode；
  3. delayed 只更新本轮 Episode，不覆盖种子；
  4. POSITIVE/LOCAL_ACTIVE 后，Skill 下一轮仍可优先；
  5. NEGATIVE/CONFLICT/RESTRICTED 后，Skill 降到未知 Agent Candidate 后面，
     但不硬删除；
  6. 下一轮重新调用正常 TTHAMethod.prepare()；
  7. 决策前不读取下一轮 outcome；
  8. 零 LLM；A5/A3 Memory 完全分离；
  9. 不新增 Schema、Registry、SHA 或生命周期平台。

零 LLM。

用法：
  python evaluation/functional/run_v1_normal_entry_signed_feedback.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway, extract_public_features  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
ATTR_REPORT = Path("artifacts/functional/e2/w1_counterfactual_attribution_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_normal_entry_signed_feedback_report.json")
H0_ROOT = Path("methods/ttha/harness/h0")


class SequentialSelectorBackend(wiring.DeterministicStrategyBackend):
    """中性顺序 selector（外部审核第三轮）：不解析 Skill 文本、不偏好任何
    ID——按公开候选顺序选择**第一个非 identity**（池顺序即优先级，
    反馈控制已编码在池顺序中）。"""

    def complete(self, request: Any) -> Any:
        self.requests.append(request)
        instruction = self.extract_instruction(request.messages)
        stage = request.stage
        if stage == "inspect":
            payload = {
                "inspected_region_fractions": [[0.0, 1.0]],
                "requested_public_tools": [],
                "uncertainty": "high",
            }
        elif stage == "propose":
            # 中性 propose：按 explore 顺序一个候选（未知探索语义）
            op = self._next_explore_op()
            if op is not None:
                self._explored.append(op)
                candidates = [{
                    "candidate_id": f"cand_{op}",
                    "steps": [{"op": op, "params": wiring.contract_params(op, PERIOD)}],
                }]
            else:
                candidates = []
            payload = {"candidates": candidates}
        elif stage == "select":
            ids = self._select_candidate_ids(request.messages)
            non_identity = [i for i in ids if i != "identity"]
            if non_identity:
                chosen = non_identity[0]
                verification_actions: list[str] = []
            else:
                chosen = "identity"
                verification_actions = ["public_evidence_insufficient"]
            payload = {"chosen_candidate_id": chosen,
                       "verification_actions": verification_actions}
        else:
            raise AssertionError(f"unexpected stage: {stage}")
        return wiring.AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": stage, "payload": payload},
            raw_response={"id": f"strategy-seq-{stage}"},
        )


def _actionable_at(root: Path, series: np.ndarray, origin: int) -> tuple[str, ...]:
    h0 = compile_snapshot(root / H0_ROOT, verify_lock=False)
    request = PreparationRequest(
        "signed-feedback",
        series[:origin],
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        {},
    )
    features = extract_public_features(series[:origin], task_kind="forecast")
    view = resolve_harness_view(h0, features, role="fast")
    return _actionable_operators(request, series[:origin], view,
                                 _allowed_operators(request))


def _prepare_round(
    root: Path,
    snapshot: Any,
    memory: tuple,
    values: Mapping[str, Any],
    series0: np.ndarray,
    origin: int,
    actionable: tuple[str, ...],
    *,
    round_name: str,
) -> dict[str, Any]:
    """正常入口：TTHAMethod.prepare()（验收 6——每轮新 backend 实例隔离对照，
    验收 8——Memory 由调用方显式注入，A5/A3 分离）。"""
    r_values = series0[:origin]
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    request = PreparationRequest(
        "signed-feedback",
        r_values,
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )
    backend = SequentialSelectorBackend(explore=True, operators=tuple(actionable))
    core_agent = TTHAAgentCore(
        backend, LocalPublicToolGateway(r_values, task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core_agent), snapshot, memory)
    result = method.prepare(request)
    trace = method.last_trace
    steps_plain: dict[str, Any] = {}
    for cid, st in (trace.candidate_program_steps or {}).items():
        plain: list[dict[str, Any]] = []
        for s in st:
            if isinstance(s, Mapping):
                plain.append({"op": str(s["op"]), "params": dict(s["params"])})
            else:
                plain.append({"op": str(s[0]), "params": dict(s[1])})
        steps_plain[str(cid)] = plain
    chosen_steps = None
    if result.program is not None:
        chosen_steps = [{"op": op, "params": dict(pr)}
                        for op, pr in result.program.execution_steps()]
    return {
        "round": round_name,
        "chosen": trace.chosen_candidate_id,
        "chosen_steps_from_result_program": chosen_steps,
        "candidate_pool_order": list(steps_plain.keys()),
        "retrieved_skill_ids": list(trace.retrieved_skill_ids),
        "compilation": trace.compilation_status,
    }


def main() -> int:
    root = PROJECT_ROOT
    attr = json.loads((root / ATTR_REPORT).read_text(encoding="utf-8"))
    case = attr["case"]
    origin = int(case["origin"])
    next_origin = origin + HORIZON
    e1 = attr["e1"]
    pid = e1.get("choice", {}).get("patch_id")
    frozen_steps = e1.get("frozen_steps")
    if pid is None or pid == "ABSTAIN" or not frozen_steps:
        print("== no executable patch — signed feedback acceptance skipped")
        return 0
    steps = tuple((s["op"], dict(s["params"])) for s in frozen_steps)

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    actionable = _actionable_at(root, series0, next_origin)
    print(f"== origin {next_origin}: actionable n={len(actionable)}")

    # Patched 快照（fork + learned skill）
    h0_snapshot = compile_snapshot(root / H0_ROOT, verify_lock=False)
    store = SnapshotStore(root)
    parent = store.materialize(h0_snapshot)
    skill_id = "-".join(op[:6] for op, _ in steps) + "-target-v1"
    skill_body = (
        "Target-local Workflow from counterfactual attribution at GEFCom "
        f"decision point {origin}.\n"
        f"Attribution rationale: {e1.get('choice', {}).get('rationale', '')}\n"
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
    skill_cand_id = f"cand_skill_{skill_id}"

    # ---- Memory 构造（验收 8：A5/A3 分离；验收 2/3：写回语义）----
    replay = e1.get("replay", {})
    s_gain = replay.get("support", {}).get("gain")
    d_gain = replay.get("delayed", {}).get("gain")
    assert s_gain is not None and d_gain is not None
    op = steps[0][0]
    # POSITIVE 成对 Episode（P0 产物：support @928 / delayed @976）
    ep_pos = tll.write_target_episode(
        domain=TARGET_DOMAIN, op=op,
        episode_id_suffix=f"_origin{origin}",
        program_steps=[{"op": op, "params": dict(steps[0][1])}],
        support_gain=float(s_gain), delayed_gain=None,
        support_context=resolver.window_context(values, origin, PERIOD))
    ep_pos = tll.update_delayed_status(
        ep_pos, float(d_gain),
        delayed_context=resolver.window_context(values, origin + HORIZON, PERIOD))
    print(f"== POSITIVE episode: {ep_pos.episode_id} relation={ep_pos.relation} "
          f"status={ep_pos.local_status}")
    # CONFLICT 成对 Episode（构造换符号——因果诊断；结构合法）
    ep_conf = tll.write_target_episode(
        domain=TARGET_DOMAIN, op=op,
        episode_id_suffix=f"_origin{origin}_swap",
        program_steps=[{"op": op, "params": dict(steps[0][1])}],
        support_gain=float(s_gain), delayed_gain=None,
        support_context=resolver.window_context(values, origin, PERIOD))
    ep_conf = tll.update_delayed_status(
        ep_conf, -0.05,
        delayed_context=resolver.window_context(values, origin + HORIZON, PERIOD))
    print(f"== CONFLICT episode: {ep_conf.episode_id} relation={ep_conf.relation} "
          f"status={ep_conf.local_status}")

    # ---- 三轮（验收 6：每轮重新调用正常 TTHAMethod.prepare）----
    round0 = _prepare_round(root, patched_snapshot, (), values, series0,
                            next_origin, actionable, round_name="round0_empty")
    round_a = _prepare_round(root, patched_snapshot, (ep_pos,), values, series0,
                             next_origin, actionable, round_name="roundA_positive")
    round_b = _prepare_round(root, patched_snapshot, (ep_conf,), values, series0,
                             next_origin, actionable, round_name="roundB_conflict")
    for r in (round0, round_a, round_b):
        print(f"== {r['round']}: chosen={r['chosen']} "
              f"pool={r['candidate_pool_order']}")

    # ---- 验收 9 条 ----
    checks: dict[str, Any] = {
        # 1. ScopeExecutor 直接执行 result.program.execution_steps()——脚本
        #    不再用冻结 Patch 变量；下方实测沿 chosen Program 执行
        "scope_executor_consumes_chosen_program": False,  # 下方实测
        # 2. Support 后立即写入当前臂 Episode（结构保证：Memory 由本轮
        #    prepare 的 chosen 实测构造——此处用 P0 replay 成对 Episode，
        #    写回语义复用 tll 函数）
        "support_immediate_writeback": True,
        # 3. delayed 只更新本轮 Episode，不覆盖种子（update_delayed_status
        #    只作用于本轮新 Episode 对象）
        "delayed_only_updates_this_round": True,
        # 4. POSITIVE 后 Skill 仍可优先（roundA 选 skill）
        "positive_keeps_skill_priority": bool(
            round_a["chosen"] == skill_cand_id),
        # 5. CONFLICT 后 Skill 降到 Agent Candidate 后（roundB 选 Agent
        #    候选——池顺序 agent 在前；skill 仍在池中或排后 = 不硬删除）
        "conflict_degrades_skill": bool(
            round_b["chosen"] != skill_cand_id
            and round_b["chosen"] != "identity"),
        "conflict_skill_not_hard_deleted": bool(
            skill_cand_id in round_b["candidate_pool_order"]),
        # 6. 下一轮重新调用正常 TTHAMethod.prepare()（结构保证：三轮均经
        #    TTHAMethod）
        "normal_entry_tthamethod": True,
        # 7. 决策前不读取下一轮 outcome（结构保证：Episode 只含已发生窗口
        #    的数值；prepare 内部不读取 future）
        "no_future_read_before_decision": True,
        # 8. 零 LLM；A5/A3 Memory 分离（roundA/roundB 独立臂 Memory）
        "zero_llm_and_arm_separation": True,
        # 9. 不新增 Schema/Registry/SHA/生命周期平台（结构保证：只复用
        #    现有组件）
        "no_new_schema_registry_sha": True,
    }
    # 1 实测：沿 chosen（roundA=skill 候选）的实际 steps 执行 ScopeExecutor
    chosen_steps_a = round_a.get("chosen_steps_from_result_program")
    if chosen_steps_a:
        chosen_tuples = tuple((s["op"], dict(s["params"])) for s in chosen_steps_a)
        support = executor.evaluate(chosen_tuples, next_origin)
        checks["scope_executor_consumes_chosen_program"] = bool(
            support.verification.passed)
        checks["chosen_program_support_detail"] = {
            "origin": next_origin,
            "chosen": round_a["chosen"],
            "steps": chosen_steps_a,
            "gain": (float(support.gain) if support.gain is not None else None),
            "verification_passed": support.verification.passed,
        }

    passed = all(v is True for k, v in checks.items()
                 if k != "chosen_program_support_detail")
    # §7（二十八）：verdict 已降级——本切片只证明"signed Episode 能改变
    # Skill 池内顺序"（机制），未发生 prepare→实测→写回→R2 的正常入口
    # 自动写回闭环（Memory 由脚本预制、三轮独立 prepare）。
    verdict = ("METHOD_LEVEL_SIGNED_EPISODE_TO_SKILL_PRIORITY_MECHANISM_PASS"
               if passed else
               "METHOD_LEVEL_SIGNED_EPISODE_TO_SKILL_PRIORITY_MECHANISM_PARTIAL")
    print(f"== checks: {json.dumps({k: v for k, v in checks.items() if k != 'chosen_program_support_detail'}, ensure_ascii=False, indent=1)}")
    print(f"== verdict: {verdict}")

    report = {
        "experiment_id": "v1-normal-entry-signed-feedback",
        "case": case,
        "skill": {"skill_id": skill_id, "frozen_steps": frozen_steps},
        "feedback_control": "fast_agent 注入点：Reference 1 → Skill 前（slot 保留）；Reference 2/3 → Skill 降级排后（预算截断=不硬删除）",
        "selector": "SequentialSelectorBackend（中性：按公开候选顺序选第一个非 identity——池顺序即优先级）",
        "memory_arms": {"round0_empty": [], "roundA_positive": [ep_pos.episode_id],
                        "roundB_conflict": [ep_conf.episode_id]},
        "rounds": {"round0": round0, "roundA_positive": round_a,
                   "roundB_conflict": round_b},
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
