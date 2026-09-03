"""NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK 验收（外部审核第四轮裁决，
2026-08-09，文档 §7 二十八）。

上一轮（§7 二十七）只证明"signed Episode 能改变 Skill 池内顺序"（机制），
被降级为 METHOD_LEVEL_SIGNED_EPISODE_TO_SKILL_PRIORITY_MECHANISM_PASS +
NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_PENDING。五条承重指责：
①无写回闭环（预制 Memory + 三次独立 prepare）②四项验收硬编码 True
③同窗 outcome 回灌（delayed@976 用于 976 决策）④CONFLICT 人工换符号
⑤旧 verdict 未清理。

本切片（外部审核第四轮裁决 + 第五轮因果对照修正）只做**真正的正常入口
反馈生命周期**：

  两臂从 R1 开始状态同步（独立但相同的 Method/backend、相同空 explore、
  相同 R1 gateway）
  → 两臂都执行相同 R1 prepare（origin=832，探索状态同步）
  → ScopeExecutor 执行 result.program（真实 Support receipt）
  → 只给写回臂立即追加 Episode（append_experience_episode——最小接口）
  → 之后打开真实 delayed（origin=880，窗口 [880,928)），更新同一 Episode
    （update_experience_episode——最小接口）
  → 两臂都绑定 R2 数据（bind_round_data——verify_context 是 context_sha
    全等比较，gateway 须按轮重建）并 prepare（origin=928 = R1+2×HORIZON，
    断言计算）
  → 承重布尔断言：R1 两臂 chosen/program 相同；R2 前两臂探索状态等价；
    R2 写回臂 Skill 被降级；R2 无写回臂 Skill 仍优先；R2 两臂必须出现
    预期差异；actionable inventory 一致

要求（审核）：
  - POSITIVE/CONFLICT 都来自真实 receipt，不人工换符号；
  - origin/window 不重叠由断言计算，不硬编码 True；
  - 只增加最小 Episode append/update 接口，不建设 Memory Store/Schema/
    生命周期平台。
  - 唯一变量 = 写回的 Episode（两臂 backend 探索状态同步——第五轮修正：
    旧版写回臂沿用 R1 stateful backend 而对照臂 R2 新建，混淆了 Memory
    与探索历史）。

边界（审核第五轮）：R1@832 使用的 Skill 由 @928/@976 已暴露数据生成
（时间上来自未来）；两臂共享同一 Skill，因此只验证反馈控制机械链，只能
称 development positive-control mechanism，不能称在线迁移或自然纵向能力
证据。

数据可用性（审核判断基于 origin=928 case 的尾部耗尽；实际移位可行）：
  GEFCom 每支 1024 点，HORIZON=48（evaluate truth = raw[origin, origin+48)）：
    R1=832  → support [832, 880)  → delayed@880 [880, 928)
    R2=928  → support [928, 976)  → delayed@976 [976, 1024)
  全部在数据内；R2 决策时刻（928）晚于 R1 delayed 窗口末端（928）——
  决策前不读取未来（Episode 数值全部已发生）。

零 LLM。

用法：
  python evaluation/functional/run_v1_normal_entry_automatic_feedback.py
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
# 无重叠 R2 的决策点对：R2 = R1 + 2×HORIZON（R1 delayed 窗口 [R1+H, R1+2H)
# 在 R2 决策时刻已完全发生），且 R2 + 2×HORIZON <= len(series)（R2 delayed
# 窗口 [R2+H, R2+2H) 也在数据内）。
R1_ORIGIN = 832
R2_ORIGIN = 928
ATTR_REPORT = Path("artifacts/functional/e2/w1_counterfactual_attribution_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_normal_entry_automatic_feedback_report.json")
H0_ROOT = Path("methods/ttha/harness/h0")


class SequentialSelectorBackend(wiring.DeterministicStrategyBackend):
    """中性顺序 selector：按公开候选顺序选第一个非 identity（池顺序即
    优先级，反馈控制已编码在池顺序中）。"""

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
        "automatic-feedback",
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
    method: TTHAMethod,
    values: Mapping[str, Any],
    series0: np.ndarray,
    origin: int,
) -> dict[str, Any]:
    """正常入口 TTHAMethod.prepare()。method 复用（R1/R2 同一实例）。"""
    r_values = series0[:origin]
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    request = PreparationRequest(
        "automatic-feedback",
        r_values,
        forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )
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
        "origin": origin,
        "chosen": trace.chosen_candidate_id,
        "chosen_steps_from_result_program": chosen_steps,
        "candidate_pool_order": list(steps_plain.keys()),
        "retrieved_skill_ids": list(trace.retrieved_skill_ids),
        "compilation": trace.compilation_status,
    }


def main() -> int:
    root = PROJECT_ROOT
    attr = json.loads((root / ATTR_REPORT).read_text(encoding="utf-8"))
    e1 = attr["e1"]
    pid = e1.get("choice", {}).get("patch_id")
    frozen_steps = e1.get("frozen_steps")
    if pid is None or pid == "ABSTAIN" or not frozen_steps:
        print("== no executable patch — automatic feedback acceptance skipped")
        return 0
    steps = tuple((s["op"], dict(s["params"])) for s in frozen_steps)
    op = steps[0][0]

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    n_data = int(len(series0))

    # ---- 窗口/不重叠断言（程序计算，不硬编码 True）----
    # R1 delayed 窗口 [R1+H, R1+2H)；R2 决策时刻 = R2；要求 R2 >= R1+2H
    # （决策时 delayed 已全部发生），且 R1+2H <= n_data（delayed 窗口在
    # 数据内），R2+2H <= n_data（R2 的 support + delayed 也在数据内）。
    non_overlap = R2_ORIGIN - R1_ORIGIN
    assert non_overlap == 2 * HORIZON, f"R2-R1 must be 2*HORIZON, got {non_overlap}"
    assert R1_ORIGIN + 2 * HORIZON <= n_data, (
        f"R1 delayed window ends {R1_ORIGIN + 2 * HORIZON} > data {n_data}")
    assert R2_ORIGIN + 2 * HORIZON <= n_data, (
        f"R2 delayed window ends {R2_ORIGIN + 2 * HORIZON} > data {n_data}")
    print(f"== data n={n_data}; R1={R1_ORIGIN} support "
          f"[{R1_ORIGIN},{R1_ORIGIN + HORIZON}) delayed "
          f"[{R1_ORIGIN + HORIZON},{R1_ORIGIN + 2 * HORIZON}); "
          f"R2={R2_ORIGIN} support [{R2_ORIGIN},{R2_ORIGIN + HORIZON})")

    # ---- Patched 快照（fork + learned skill，P0 产物）----
    h0_snapshot = compile_snapshot(root / H0_ROOT, verify_lock=False)
    store = SnapshotStore(root)
    parent = store.materialize(h0_snapshot)
    skill_id = "-".join(op[:6] for op, _ in steps) + "-target-v1"
    skill_body = (
        "Target-local Workflow from counterfactual attribution at GEFCom "
        f"decision point {attr['case']['origin']}.\n"
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

    # ---- 双臂（外部审核第五轮：两臂状态同步，唯一变量 = Memory）----
    # 从 R1 开始建立两个独立但状态相同的 Method/backend（相同 operators、
    # 相同空 explore 状态、相同 R1 gateway）→ 两臂都执行相同 R1 prepare，
    # 让探索状态同步 → 只给写回臂 append/update Episode → 两臂都绑定 R2
    # 数据并 prepare。R2 两臂的 agent 候选来自同一探索历史（都跳过 R1
    # 已探的算子）——差异只可能来自写回的 Episode。
    actionable_r1 = _actionable_at(root, series0, R1_ORIGIN)
    actionable_r2 = _actionable_at(root, series0, R2_ORIGIN)
    gw_r1 = LocalPublicToolGateway(series0[:R1_ORIGIN], task_kind="forecast")
    backend_wb = SequentialSelectorBackend(explore=True,
                                           operators=tuple(actionable_r1))
    backend_ctrl = SequentialSelectorBackend(explore=True,
                                             operators=tuple(actionable_r1))
    method_wb = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(backend_wb, gw_r1)), patched_snapshot, ())
    method_ctrl = TTHAMethod(
        TTHAFastAgent(TTHAAgentCore(backend_ctrl, gw_r1)), patched_snapshot, ())
    r1_wb = _prepare_round(method_wb, values, series0, R1_ORIGIN)
    r1_ctrl = _prepare_round(method_ctrl, values, series0, R1_ORIGIN)
    print(f"== R1 writeback: chosen={r1_wb['chosen']} "
          f"pool={r1_wb['candidate_pool_order']}")
    print(f"== R1 ctrl:     chosen={r1_ctrl['chosen']} "
          f"pool={r1_ctrl['candidate_pool_order']}")
    explored_wb = set(tuple(backend_wb._explored))  # noqa: SLF001
    explored_ctrl = set(tuple(backend_ctrl._explored))  # noqa: SLF001
    print(f"== R1 explored: wb={sorted(explored_wb)} ctrl={sorted(explored_ctrl)}")

    # 前提 1：R1 两臂 chosen 与池必须相同（双臂状态同步的前提）；R1 chosen
    # 必须是 Skill 候选（否则没有 Skill 被反馈——PENDING）
    r1_steps = r1_wb.get("chosen_steps_from_result_program")
    premise_ok = bool(
        r1_wb["chosen"] == skill_cand_id
        and r1_ctrl["chosen"] == r1_wb["chosen"]
        and r1_ctrl["candidate_pool_order"] == r1_wb["candidate_pool_order"]
        and r1_steps)
    if not premise_ok:
        print("== R1 arms not identical or chosen not skill — PENDING")
        checks: dict[str, Any] = {"premise_r1_arms_synced_and_skill": False}
        verdict = "NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_PENDING"
    else:
        # ---- R1 Support：真实 receipt（沿 chosen Program 执行）----
        chosen_tuples = tuple((s["op"], dict(s["params"])) for s in r1_steps)
        supp_r1 = executor.evaluate(chosen_tuples, R1_ORIGIN)
        s_gain = (float(supp_r1.gain) if supp_r1.gain is not None else None)
        print(f"== R1 support @{R1_ORIGIN}: gain={s_gain} "
              f"passed={supp_r1.verification.passed}")
        if s_gain is None:
            print("== R1 support gain None — no receipt; PENDING")
            checks = {"premise_r1_arms_synced_and_skill": True,
                      "r1_support_gain_not_none": False}
            verdict = "NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_PENDING"
        else:
            # ---- 立即追加 Episode（真实 support_gain；只给写回臂）----
            ep = tll.write_target_episode(
                domain=TARGET_DOMAIN, op=op,
                episode_id_suffix=f"_r1_{R1_ORIGIN}",
                program_steps=[{"op": s["op"], "params": dict(s["params"])}
                               for s in r1_steps],
                support_gain=s_gain, delayed_gain=None,
                support_context=resolver.window_context(
                    values, R1_ORIGIN, PERIOD))
            method_wb.append_experience_episode(ep)
            n_after_append = len(tuple(method_wb._experience_episodes))  # noqa: SLF001
            n_ctrl_after_r1 = len(tuple(method_ctrl._experience_episodes))  # noqa: SLF001
            print(f"== append: episode={ep.episode_id} relation={ep.relation} "
                  f"status={ep.local_status} n_wb={n_after_append} "
                  f"n_ctrl={n_ctrl_after_r1}")

            # ---- 之后打开真实 delayed（窗口 [R1+H, R1+2H)），更新同一
            #      Episode（原位替换，不新增；只给写回臂）----
            supp_delayed = executor.evaluate(chosen_tuples, R1_ORIGIN + HORIZON)
            d_gain = (float(supp_delayed.gain)
                      if supp_delayed.gain is not None else None)
            print(f"== R1 delayed @{R1_ORIGIN + HORIZON}: gain={d_gain} "
                  f"passed={supp_delayed.verification.passed}")
            ep2 = tll.update_delayed_status(
                ep, d_gain if d_gain is not None else 0.0,
                delayed_context=resolver.window_context(
                    values, R1_ORIGIN + HORIZON, PERIOD))
            method_wb.update_experience_episode(ep2)
            n_after_update = len(tuple(method_wb._experience_episodes))  # noqa: SLF001
            print(f"== update delayed: relation={ep2.relation} "
                  f"status={ep2.local_status} n_wb={n_after_update}")

            # ---- R2：两臂都绑定 R2 数据（同一实例内 gateway 重建）----
            method_wb.bind_round_data(series0[:R2_ORIGIN], task_kind="forecast")
            method_ctrl.bind_round_data(series0[:R2_ORIGIN],
                                        task_kind="forecast")
            r2 = _prepare_round(method_wb, values, series0, R2_ORIGIN)
            r2_ctrl = _prepare_round(method_ctrl, values, series0, R2_ORIGIN)
            print(f"== R2 writeback: chosen={r2['chosen']} "
                  f"pool={r2['candidate_pool_order']}")
            print(f"== R2 ctrl:      chosen={r2_ctrl['chosen']} "
                  f"pool={r2_ctrl['candidate_pool_order']}")

            # ---- 验收（承重布尔断言全部程序计算；无硬编码 True）----
            future_read = R1_ORIGIN + 2 * HORIZON > R2_ORIGIN
            pool_wb = r2["candidate_pool_order"]
            pool_ctrl = r2_ctrl["candidate_pool_order"]
            skill_in_wb = skill_cand_id in pool_wb
            checks = {
                # 1. R1 两臂 chosen/program 相同（双臂状态同步前提）
                "r1_arms_same_chosen": bool(
                    r1_wb["chosen"] == r1_ctrl["chosen"]),
                "r1_arms_same_pool": bool(
                    r1_wb["candidate_pool_order"]
                    == r1_ctrl["candidate_pool_order"]),
                "r1_arms_same_steps": bool(
                    r1_wb.get("chosen_steps_from_result_program")
                    == r1_ctrl.get("chosen_steps_from_result_program")),
                # 2. R2 前两臂 backend 探索状态等价（同一 R1 prepare 后）
                "backend_exploration_synced_before_r2": bool(
                    explored_wb == explored_ctrl),
                # 3. 只给写回臂写回（ctrl 臂内部始终 0 个 Episode）
                "writeback_only_wb_arm": bool(
                    n_after_append == 1 and n_after_update == 1
                    and n_ctrl_after_r1 == 0),
                # 4. R1 Support 来自真实 receipt
                "r1_support_real_receipt": supp_r1.gain is not None,
                "r1_support_gain": s_gain,
                # 5. delayed 真实更新同一 Episode（原位替换）
                "delayed_updates_same_episode": bool(n_after_update == 1),
                "delayed_real_receipt": supp_delayed.gain is not None,
                "r1_delayed_gain": d_gain,
                "episode_relation_after_delayed": ep2.relation,
                # 6. 决策时刻无未来（断言计算）
                "no_future_read_before_r2_decision": bool(
                    not future_read and n_data >= R2_ORIGIN + HORIZON),
                # 7. 无重叠（R2 - R1 == 2×HORIZON）
                "non_overlap_asserted": bool(non_overlap == 2 * HORIZON),
                # 8. 承重：R2 写回臂 Skill 被降级（Agent 候选在前、Skill 排后）
                "r2_writeback_skill_degraded": bool(
                    skill_in_wb and pool_wb[0] != skill_cand_id),
                # 9. 承重：R2 无写回臂 Skill 仍优先（池首位）
                "r2_ctrl_skill_priority": bool(
                    pool_ctrl and pool_ctrl[0] == skill_cand_id),
                # 10. 承重：行动改变（外部审核第五轮收紧——核心主张是行动
                #     改变，直接要求 chosen 不同，不只池顺序）
                "r2_arms_expected_difference": bool(
                    r2["chosen"] != r2_ctrl["chosen"]),
                # 11. Operator inventory 一致性（否则写回臂在用 R1 旧清单）
                "actionable_inventory_consistent": bool(
                    set(actionable_r1) == set(actionable_r2)),
            }
            passed = all(v is True for k, v in checks.items()
                         if k not in ("r1_support_gain", "r1_delayed_gain",
                                      "episode_relation_after_delayed"))
            # 12. POSITIVE/CONFLICT 来自真实 receipt（无人工换符号）——报告
            #     元数据，不计入程序验证的 passed（外部审核第五轮：硬编码
            #     True 移出检查）；真实 gains 已在上面落盘。
            checks["no_handcrafted_sign_flip"] = (
                "structure: no hardcoded sign flip in script; "
                f"real support_gain={s_gain}, real delayed_gain={d_gain}")
            verdict = ("NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_DEVELOPMENT_"
                       "MECHANISM_PASS" if passed else
                       "NORMAL_ENTRY_AUTOMATIC_FEEDBACK_WRITEBACK_DEVELOPMENT_"
                       "MECHANISM_PARTIAL")

    print(f"== verdict: {verdict}")
    report = {
        "experiment_id": "v1-normal-entry-automatic-feedback-writeback",
        "skill": {"skill_id": skill_id, "frozen_steps": list(frozen_steps)},
        "boundary": ("development positive-control mechanism：R1@832 使用的 "
                     "Skill 由 @928/@976 已暴露数据生成（时间上来自未来）；"
                     "两臂共享同一 Skill，验证反馈控制机械链，不称在线迁移/"
                     "自然纵向能力证据"),
        "origins": {"r1": R1_ORIGIN, "r2": R2_ORIGIN,
                    "window_assertions": {
                        "r2_minus_r1_equals_2h": bool(non_overlap == 2 * HORIZON),
                        "r1_delayed_ends": R1_ORIGIN + 2 * HORIZON,
                        "r2_delayed_ends": R2_ORIGIN + 2 * HORIZON,
                        "data_length": n_data}},
        "arms": {"r1_writeback": r1_wb, "r1_ctrl": r1_ctrl,
                 "r2_writeback": r2, "r2_ctrl": r2_ctrl,
                 "backend_explored_r1": {"writeback": sorted(explored_wb),
                                         "ctrl": sorted(explored_ctrl)}},
        "interface": "TTHAMethod.append_experience_episode / update_experience_episode / bind_round_data（最小接口，无新 Schema/Store）",
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
