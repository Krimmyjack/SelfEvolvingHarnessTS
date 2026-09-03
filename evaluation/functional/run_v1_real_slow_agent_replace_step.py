"""REAL_SLOW_AGENT_REPLACE_STEP_POSITIVE_CONTROL（任务书 P1，2026-08-10）。

P0 正控已确认是 REMOVE_A 型（frozen program 1-step outlier_iqr-only）。
本实验验证真实的单步替换（A→B 变成 A→C，两步冻结）：

  - 案例由数值扫描（scan_v1_replace_step_cases.py，已暴露数据）冻结：
      gain(A→B) < −M；A-only < +M；B-only < +M；
      存在 C1：gain(A→C1) ≥ +M 且 delayed(A→C1) ≥ −M；C2 为冻结对照候选。
  - LLM 输入只含：incumbent workflow + TS Context + incumbent failure 数值
    + 可编辑 step index(=1) + ≤2 replacement contracts + objective + budget。
  - LLM 不得看到 A/B-only 反事实、replacement gain、winner、future。
  - frozen program 必须 2 步：steps[0]==A 且 steps[1]∈{C1, C2}（不是删除）。
  - 链：card → propose_edit → preflight → apply_to_fork → replay → delayed →
    正常 Fast 入口采用 → remove-skill 对照。
  - 确定性 Harness 只补 surface 模板/依赖 SHA/schema；不补替换位置/算子/
    Program steps/Scope-Risk 语义。

Verdict（预注册）：
  REAL_SLOW_AGENT_REPLACE_STEP_PASS / INFEASIBLE_NO_TRUE_REPLACEMENT_CONTROL /
  NO_EXECUTABLE_MANIFEST / WRONG_REPLACEMENT_SELECTED / SUPPORT_REPLAY_REJECTED /
  DELAYED_REJECTED / FAST_ADOPTION_FAILED / REMOVAL_CONTROL_NO_FLIP

用法：
  python evaluation/functional/run_v1_real_slow_agent_replace_step.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_e2_autonomous_natural_workflow_generation as v6  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController, SurfaceRegistry, FaultRouter)
from SelfEvolvingHarnessTS.runtime.agent_backend import AgictoChatCompletionsBackend  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

PERIOD = 24
HORIZON = 48
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005

# ---- 案例冻结（2026-08-10 自 scan_v1_replace_step_cases.json GEFCom 904 的
# case 0；承重数值在运行时有 recheck 复测）----
# incumbent denoise_median→outlier_iqr = −0.13021（B 单步本身负）；
# A-only=0.0（denoise 前缀近零 no-op）；C1=winsorize 双正（+0.400/+0.257）；
# C2=outlier_mad 同族无 headroom（denoise_median→outlier_mad = −0.0646）。
CASE = {
    "domain": "gefcom2012_load",
    "origin": 904,
    "delayed": 952,
    "a": "denoise_median",
    "b": "outlier_iqr",
    "c1": "winsorize",
    "c2": "outlier_mad",
    "gain_AB": -0.1302066421576531,
    "delayed_AB": -0.06125772464287427,
    "gain_A_only": 0.0,
    "gain_B_only": -0.1302066421576531,
    "support_AC1": 0.4000053662007894,
    "delayed_AC1": 0.2572125429345453,
}

REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_real_slow_agent_replace_step_report.json")

SURFACE_CATALOG = [
    {
        "surface_id": "skill_library.entries/{skill_id}",
        "operation": "ADD",
        "surface_type": "skill",
        "allowed_operations": ["ADD"],
    },
]

OPS_ALL = tuple(o for o in (
    "denoise_median", "hampel_filter", "impute_ar", "impute_ema",
    "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
    "outlier_mad", "period_complete", "period_median_complete",
    "repair_level_shift", "resample_uniform", "winsorize"))


def _contract(op: str) -> dict[str, object]:
    """公开 Operator 契约（LLM 可见）：参数 defaults + category/stage/tags。"""
    meta = OPERATOR_METADATA.get(op) or {}
    return {
        "op": op,
        "params": dict(wiring.contract_params(op, PERIOD)),
        "category": meta.get("category"),
        "stage": meta.get("stage"),
        "tags": list(meta.get("tags") or ()),
    }


def _steps_of(ops: tuple[str, ...]) -> tuple:
    return tuple((op, dict(wiring.contract_params(op, PERIOD))) for op in ops)


def build_replace_step_card(executor: ScopeExecutor,
                            values: Mapping[str, Any],
                            case: Mapping[str, Any]) -> dict[str, object]:
    """P1 FailurePatternCard：只含任务书 §4 信息墙内的内容。不预标注答案。
    不泄露 A/B-only 反事实、replacement gain、winner、future。"""
    pa = dict(wiring.contract_params(case["a"], PERIOD))
    pb = dict(wiring.contract_params(case["b"], PERIOD))
    ctx = dict(resolver.window_context(values, case["origin"], PERIOD))
    ctx["bound_period"] = float(PERIOD)
    c1 = _contract(case["c1"])
    c2 = _contract(case["c2"])
    return {
        "pattern_id": f"{case['domain']}-{case['a']}-{case['b']}-replace-step",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {
            "steps": [{"op": case["a"], "params": pa},
                      {"op": case["b"], "params": pb}],
            "scope": "training_windows_only",
            "evaluator": "v6._evaluate (per-training-window, cohort Ridge sMASE)",
        },
        "facts": {
            "window_context": ctx,
            "editable_step_index": 1,  # 第二个 step 是候选替换位置
            "replacement_candidates": [c1, c2],
            "task_objective": ("forecast; downstream Support metric = "
                               "cohort-Ridge sMASE over the declared "
                               "training_windows_only scope"),
            "feedback_budget": "one formal Slow Agent call; one already-"
                               "existing schema retry allowed",
        },
        "counterfactual_support": {
            # 只给 incumbent failure（任务书 §4 信息墙）；不给单算子分解
            "A_then_B_support_gain": case["gain_AB"],
            "A_then_B_delayed_gain": case["delayed_AB"],
        },
        "instruction": (
            "A two-step workflow was applied at the support decision point "
            "and produced the Support/Delayed outcomes above (negative "
            "material impact relative to identity). The second step "
            "(index 1) is the candidate replacement position: replace it "
            "with exactly one of the two replacement candidates, keeping "
            "the first step (index 0) unchanged. Propose exactly one edit: "
            "ADD one new capability skill whose frozen program is the "
            "repaired two-step workflow (steps JSON after the line "
            "'Frozen program steps:'). Fill falsification_condition so a "
            "deterministic replay can accept or reject it later. You do "
            "not approve your own edit. If the evidence is insufficient, "
            "declare no_proposal — abstaining is valid."
        ),
    }


def _parse_frozen(body: str) -> list[dict[str, Any]] | None:
    marker = "Frozen program steps:"
    idx = body.find(marker)
    if idx < 0:
        return None
    rest = body[idx + len(marker):].strip()
    try:
        arr = json.loads(rest)
    except json.JSONDecodeError:
        return None
    if not isinstance(arr, list):
        return None
    return [{"op": str(s["op"]), "params": dict(s.get("params") or {})}
            for s in arr if isinstance(s, Mapping) and s.get("op")]


def structural_preflight(manifest: Any,
                         case: Mapping[str, Any]) -> dict[str, Any]:
    """Manifest 结构预检：ADD 面白名单 + 单 manifest + schema + 冻结 2-step
    Program（steps[0]==A 且 steps[1]∈{C1,C2}——不是删除步骤，不是换错位置）。"""
    if manifest is None:
        return {"stage": "no_proposal", "preflight": "REJECTED"}
    t = str(manifest.target_surface_id)
    nv = manifest.new_value or {}
    skill_id = str(nv.get("skill_id") or "")
    instantiated = t.replace("{skill_id}", skill_id) if skill_id else t
    ok_surface = instantiated.startswith("skill_library.entries/")
    ok_op = manifest.operation.value == "ADD"
    ok_schema = nv.get("schema_version") == "skill-entry/1"
    body = str(nv.get("body") or "")
    frozen = _parse_frozen(body)
    ok_steps = False
    if frozen is not None:
        ok_steps = (len(frozen) == 2
                    and str(frozen[0].get("op")) == case["a"]
                    and str(frozen[1].get("op")) in (case["c1"], case["c2"]))
    selected = None
    if ok_steps:
        selected = str(frozen[1].get("op"))
    return {
        "stage": "manifest",
        "preflight": ("ACCEPTED" if (ok_surface and ok_op and ok_schema
                                     and frozen is not None and ok_steps
                                     and skill_id) else "REJECTED"),
        "target_surface_id": instantiated,
        "skill_id": skill_id,
        "operation": manifest.operation.value,
        "new_value_schema": nv.get("schema_version"),
        "frozen_program": frozen,
        "selected_replacement": selected,
        "replacement_is_c1": bool(selected == case["c1"]),
        "surface_template_instantiated_by_harness": bool(t != instantiated),
        "falsification_condition": list(manifest.falsification_condition or ()),
    }


def main() -> int:
    root = PROJECT_ROOT
    case = dict(CASE)
    # 冻结断言（审查 MAJOR 3）：任一字段未冻结 → 拒绝运行（None origin/
    # delayed 会把全序列含未来交给 Slow Agent/gateway）
    for _k in ("domain", "origin", "delayed", "a", "b", "c1", "c2",
               "gain_AB", "delayed_AB", "gain_A_only", "gain_B_only",
               "support_AC1", "delayed_AC1"):
        assert case[_k] is not None, (
            f"P1 案例字段 {_k} 未冻结——先跑 scan_v1_replace_step_cases.py "
            f"并由审查者核实")
    checks: dict[str, Any] = {"case": dict(case)}
    verdict = "INCONCLUSIVE"
    report_manifest = None
    steps = None

    # ---- 数据 + executor ----
    if case["domain"] == "gefcom2012_load":
        config = dict(v6.DATASET_CONFIGS["gefcom"])
        roster, values = v6._fixed_roster(root, config)
        series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
        evaluate_fn = v6._evaluate
    else:
        sealed._set_domain(case["domain"])
        config = sealed._config()
        (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
            root, offset=120)  # 已暴露 cohort（Pilot 域）
        values = tgt_values
        series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
        evaluate_fn = sealed.v6._evaluate
    executor = ScopeExecutor(roster, values, config, evaluate_fn=evaluate_fn)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    # ---- 案例承重数值复核（冻结值来自扫描，此处复测确认；报告承重）----
    r_ab = executor.evaluate(_steps_of((case["a"], case["b"])), case["origin"])
    checks["recheck_gain_AB"] = (float(r_ab.gain) if r_ab.gain is not None
                                 else None)
    checks["recheck_passed_AB"] = bool(r_ab.verification.passed)
    r_a = executor.evaluate(_steps_of((case["a"],)), case["origin"])
    checks["recheck_gain_A_only"] = (float(r_a.gain) if r_a.gain is not None
                                     else None)
    r_b = executor.evaluate(_steps_of((case["b"],)), case["origin"])
    checks["recheck_gain_B_only"] = (float(r_b.gain) if r_b.gain is not None
                                     else None)
    r_ac1 = executor.evaluate(_steps_of((case["a"], case["c1"])),
                              case["origin"])
    checks["recheck_gain_AC1"] = (float(r_ac1.gain) if r_ac1.gain is not None
                                  else None)
    r_ac1d = executor.evaluate(_steps_of((case["a"], case["c1"])),
                               case["delayed"])
    checks["recheck_delayed_AC1"] = (float(r_ac1d.gain)
                                     if r_ac1d.gain is not None else None)
    # C2 必须是"无 headroom 对照"：A→C2 不满足双正（support < +M 或
    # delayed < −M）——冻结属性复测（审查注意点 3）
    r_ac2 = executor.evaluate(_steps_of((case["a"], case["c2"])),
                              case["origin"])
    checks["recheck_gain_AC2"] = (float(r_ac2.gain) if r_ac2.gain is not None
                                  else None)
    r_ac2d = executor.evaluate(_steps_of((case["a"], case["c2"])),
                               case["delayed"])
    checks["recheck_delayed_AC2"] = (float(r_ac2d.gain)
                                     if r_ac2d.gain is not None else None)
    c2_no_headroom = bool(
        checks["recheck_gain_AC2"] is not None
        and (checks["recheck_gain_AC2"] < MATERIAL
             or (checks["recheck_delayed_AC2"] is not None
                 and checks["recheck_delayed_AC2"] < -MATERIAL)))
    checks["c2_no_headroom_confirmed"] = c2_no_headroom
    headroom_ok = bool(
        checks["recheck_passed_AB"]
        and checks["recheck_gain_AB"] is not None
        and checks["recheck_gain_AB"] < -MATERIAL
        and checks["recheck_gain_A_only"] is not None
        and checks["recheck_gain_A_only"] < MATERIAL
        and checks["recheck_gain_B_only"] is not None
        and checks["recheck_gain_B_only"] < MATERIAL
        and checks["recheck_gain_AC1"] is not None
        and checks["recheck_gain_AC1"] >= MATERIAL
        and checks["recheck_delayed_AC1"] is not None
        and checks["recheck_delayed_AC1"] >= -MATERIAL
        and c2_no_headroom)
    checks["case_headroom_confirmed"] = headroom_ok
    print(f"== case recheck: AB={checks['recheck_gain_AB']} "
          f"A_only={checks['recheck_gain_A_only']} "
          f"B_only={checks['recheck_gain_B_only']} "
          f"AC1={checks['recheck_gain_AC1']} "
          f"delayed_AC1={checks['recheck_delayed_AC1']} "
          f"AC2={checks['recheck_gain_AC2']} "
          f"delayed_AC2={checks['recheck_delayed_AC2']}")
    if not headroom_ok:
        verdict = "INFEASIBLE_NO_TRUE_REPLACEMENT_CONTROL"
        _write_report(root, case, checks, verdict, None, None)
        return 0

    # ---- 真实 Slow Agent 链 ----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        verdict = "INCONCLUSIVE"
        _write_report(root, case, checks, verdict, None, None)
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:case["delayed"]],
                               task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)
    slow = TTHASlowAgent(core)

    card = build_replace_step_card(executor, values, case)
    manifest = None
    try:
        manifest = slow.propose_edit(
            card, SURFACE_CATALOG, h0,
            manifest_preflight=lambda m: None,  # 结构预检在脚本层（验收）
            allowed_operator_contracts=(),
            task_context=None)
    except RuntimeError as exc:
        print(f"== budget hard stop: {exc}")
        checks["budget_hard_stop"] = True
    checks["llm_calls_le_2"] = counter.calls <= 2
    checks["llm_calls"] = counter.calls
    print(f"== manifest: {'None' if manifest is None else 'proposed'} "
          f"llm_calls={counter.calls}")

    preflight = structural_preflight(manifest, case)
    if manifest is None:
        verdict = "NO_EXECUTABLE_MANIFEST"
    elif preflight["preflight"] != "ACCEPTED":
        verdict = "NO_EXECUTABLE_MANIFEST"
        report_manifest = {"edit_id": manifest.edit_id,
                           "operation": manifest.operation.value,
                           "target_surface_id": manifest.target_surface_id,
                           "preflight": preflight["preflight"],
                           "frozen_program": preflight["frozen_program"]}
    elif not preflight["replacement_is_c1"]:
        verdict = "WRONG_REPLACEMENT_SELECTED"
        report_manifest = {"edit_id": manifest.edit_id,
                           "target_surface_id": preflight["target_surface_id"],
                           "selected_replacement": preflight["selected_replacement"],
                           "frozen_program": preflight["frozen_program"]}
    else:
        frozen = preflight["frozen_program"]
        steps = tuple((str(s["op"]), dict(s["params"])) for s in frozen)
        report_manifest = {"edit_id": manifest.edit_id,
                           "operation": manifest.operation.value,
                           "target_surface_id": preflight["target_surface_id"],
                           "skill_id": preflight["skill_id"],
                           "frozen_program": frozen,
                           "surface_template_instantiated_by_harness": bool(
                               preflight["surface_template_instantiated_by_harness"]),
                           "falsification_condition": list(
                               manifest.falsification_condition or ())}
        # ---- EditController.apply_to_fork（确定性契约修复同 P0）----
        import dataclasses
        reg = SurfaceRegistry()
        resolved = reg.resolve(preflight["target_surface_id"])
        snapshot_deps = dict(h0.dependency_shas)
        declared_dep = {
            key: snapshot_deps[key]
            for key in resolved.definition.required_dependency_keys
            if key in snapshot_deps
        }
        manifest_applied = dataclasses.replace(
            manifest,
            target_surface_id=preflight["target_surface_id"],
            dependency_precondition_shas=declared_dep)
        store = SnapshotStore(root)
        parent = store.materialize(h0)
        controller = EditController(store, surfaces=reg, router=FaultRouter())
        try:
            receipt = controller.apply_to_fork(
                parent, manifest_applied,
                confirmed_cause="SKILL_LIBRARY_GAP")
        except Exception as exc:
            print(f"== apply_to_fork EXC: {type(exc).__name__}: {exc}")
            verdict = "NO_EXECUTABLE_MANIFEST"
            checks["apply_exception"] = f"{type(exc).__name__}: {exc}"
        else:
            checks["controller_applied"] = True
            # ---- Support replay + delayed ----
            rp = executor.evaluate(steps, case["origin"])
            gain_p = (float(rp.gain) if rp.gain is not None else None)
            checks["replay_support_positive"] = bool(
                rp.verification.passed and gain_p is not None
                and gain_p >= MATERIAL)
            print(f"== replay @{case['origin']}: gain={gain_p}")
            rd = executor.evaluate(steps, case["delayed"])
            gain_d = (float(rd.gain) if rd.gain is not None else None)
            checks["delayed_no_flip"] = bool(
                gain_d is not None and gain_d >= -MATERIAL)
            print(f"== delayed @{case['delayed']}: gain={gain_d}")

            if not checks["replay_support_positive"]:
                verdict = "SUPPORT_REPLAY_REJECTED"
            elif not checks["delayed_no_flip"]:
                verdict = "DELAYED_REJECTED"
            else:
                # ---- 正常 Fast 入口实际选择并执行 Skill ----
                skill_cand = f"cand_skill_{preflight['skill_id']}"
                method = sealed.TTHAMethod(
                    sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                        sealed.SealedProbeBackend(explore=True,
                                                  operators=OPS_ALL),
                        LocalPublicToolGateway(series0[:case["delayed"]],
                                               task_kind="forecast"))),
                    receipt.candidate_snapshot.snapshot, ())
                method.bind_round_data(series0[:case["delayed"]],
                                       task_kind="forecast")
                obs = dict(resolver.window_context(values, case["delayed"],
                                                   PERIOD))
                obs["bound_period"] = float(PERIOD)
                r2 = method.prepare(sealed._request(series0, values,
                                                    case["delayed"]))
                chosen = method.last_trace.chosen_candidate_id
                adopted = chosen == skill_cand
                r2_gain = None
                exec_steps: list[tuple[str, dict]] | None = None
                if r2.program is not None:
                    exec_steps = list(r2.program.execution_steps())
                    rr2 = executor.evaluate(tuple(exec_steps),
                                            case["delayed"])
                    r2_gain = (float(rr2.gain) if rr2.gain is not None
                               else None)
                checks["next_round_actual_adoption"] = bool(adopted)
                checks["next_round_executed"] = r2_gain is not None
                checks["executed_program_matches_frozen"] = bool(
                    exec_steps is not None
                    and [{"op": o, "params": dict(p)}
                         for o, p in exec_steps] == frozen)
                print(f"== @{case['delayed']}: chosen={chosen} "
                      f"adopted={adopted} gain={r2_gain}")

                # ---- remove-skill 对照（H0 无 skill）----
                method_ctrl = sealed.TTHAMethod(
                    sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                        sealed.SealedProbeBackend(explore=True,
                                                  operators=OPS_ALL),
                        LocalPublicToolGateway(series0[:case["delayed"]],
                                               task_kind="forecast"))),
                    h0, ())
                method_ctrl.bind_round_data(series0[:case["delayed"]],
                                            task_kind="forecast")
                method_ctrl.prepare(sealed._request(series0, values,
                                                    case["delayed"]))
                chosen_ctrl = method_ctrl.last_trace.chosen_candidate_id
                checks["removal_changes_action"] = bool(
                    chosen_ctrl != skill_cand)
                print(f"== @{case['delayed']} ctrl(no skill): "
                      f"chosen={chosen_ctrl}")

                checks["llm_replacement_is_c1"] = True
                # 承重布尔白名单（审查 BLOCKER 1：checks 混入 case/recheck_*/
                # llm_calls 非布尔键 → all() 恒 False；只对 PASS 10 条件对应
                # 的布尔键求值）
                _pass_keys = ("llm_calls_le_2", "case_headroom_confirmed",
                              "c2_no_headroom_confirmed",
                              "controller_applied",
                              "replay_support_positive", "delayed_no_flip",
                              "next_round_actual_adoption",
                              "executed_program_matches_frozen",
                              "removal_changes_action",
                              "llm_replacement_is_c1")
                passed = all(checks.get(k) is True for k in _pass_keys)
                adopted = bool(checks.get("next_round_actual_adoption"))
                removal_flip = bool(checks.get("removal_changes_action"))
                if passed:
                    verdict = "REAL_SLOW_AGENT_REPLACE_STEP_PASS"
                elif adopted and not removal_flip:
                    verdict = "REMOVAL_CONTROL_NO_FLIP"
                else:
                    verdict = "FAST_ADOPTION_FAILED"
            try:
                store.discard_fork(parent.root)
            except ValueError:
                pass

    print(f"== verdict: {verdict}")
    _write_report(root, case, checks, verdict, report_manifest,
                  preflight if report_manifest is None else None)
    return 0


def _write_report(root: Path, case: Mapping[str, Any], checks: dict[str, Any],
                  verdict: str, report_manifest: Any, preflight: Any) -> None:
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-real-slow-agent-replace-step",
        "case": dict(case),
        "manifest": report_manifest,
        "preflight": (None if preflight is None else
                      {k: v for k, v in preflight.items()
                       if k not in ("falsification_condition",)}),
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": checks.get("llm_calls"),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")


if __name__ == "__main__":
    raise SystemExit(main())
