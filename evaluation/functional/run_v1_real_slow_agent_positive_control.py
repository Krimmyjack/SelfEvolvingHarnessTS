"""REAL_SLOW_AGENT_PROGRAM_PATCH_POSITIVE_CONTROL（用户批准 2026-08-10）。

使用 P0 开发正控（明确可识别）：
  Workflow = impute_ssm → outlier_iqr
  identity = 0；A-only(impute_ssm) = −0.15432；B-only(outlier_iqr) =
  +0.04386；A→B = −0.10249；delayed B-only = +0.02719。
  反事实证据明确支持删除 A、保留 B。**不给 LLM REMOVE_A 标签**——只给
  数值表和冻结 Workflow，LLM 必须从数值自主推出保留 B。

完整链（全部走真实组件）：
  FailurePatternCard → TTHASlowAgent.propose_edit（真实 core/backend）
  → EditManifest → EditController.apply_to_fork（ADD 一个 capability
  Skill）→ compile_snapshot → ScopeExecutor Support replay → delayed →
  正常 TTHAMethod 下一轮实际选择并执行 Skill → remove-skill 对照。

约束：只开放 Program/Skill 一个 Surface；1 个 LLM 调用 + ≤1 格式纠正
（CountingClient max_calls=2）；最多 1 个 Manifest；复用现有 schema/
SkillEntry/compiler/executor；不新增 Pattern/SHA/Router/自动触发器；
development positive control（不宣称自然 Slow Path 能力）。

Verdict（六档）：
  REAL_SLOW_AGENT_PATCH_PASS / MANIFEST_NOT_EXECUTABLE /
  WRONG_PROGRAM_UPDATE / DELAYED_REJECTED / CONTROL_BINDING_FAIL /
  ABSTAIN_ON_IDENTIFIABLE_CONTROL

用法：
  python evaluation/functional/run_v1_real_slow_agent_positive_control.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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

PERIOD = 24
HORIZON = 48
ORIGIN = 928
DELAYED = 976
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_real_slow_agent_positive_control_report.json")

# 只开放 Program/Skill 一个 Surface（用户约束）
SURFACE_CATALOG = [
    {
        "surface_id": "skill_library.entries/{skill_id}",
        "operation": "ADD",
        "surface_type": "skill",
        "allowed_operations": ["ADD"],
    },
]


def build_positive_control_card(executor: ScopeExecutor) -> dict[str, object]:
    """P0 正控 FailurePatternCard：反事实数值表 + 冻结 Workflow；不预标注
    答案（不给 REMOVE_A 标签，不给 first_fault 面）。数值实测确认。"""
    pa = dict(wiring.contract_params("impute_ssm", PERIOD))
    pb = dict(wiring.contract_params("outlier_iqr", PERIOD))
    steps_ab = (("impute_ssm", pa), ("outlier_iqr", pb))
    steps_a = (("impute_ssm", pa),)
    steps_b = (("outlier_iqr", pb),)
    r_ab = executor.evaluate(steps_ab, ORIGIN)
    r_a = executor.evaluate(steps_a, ORIGIN)
    r_b = executor.evaluate(steps_b, ORIGIN)
    r_db = executor.evaluate(steps_b, DELAYED)
    counter = {
        "identity": 0.0,
        "A_only_impute_ssm": (float(r_a.gain) if r_a.gain is not None else None),
        "B_only_outlier_iqr": (float(r_b.gain) if r_b.gain is not None else None),
        "A_then_B": (float(r_ab.gain) if r_ab.gain is not None else None),
    }
    delayed_b = (float(r_db.gain) if r_db.gain is not None else None)
    return {
        "pattern_id": "gefcom-impute-ssm-outlier-iqr-counterfactual",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {
            "steps": [
                {"op": "impute_ssm", "params": pa},
                {"op": "outlier_iqr", "params": pb},
            ],
            "scope": "training_windows_only",
            "evaluator": "v6._evaluate (per-training-window, cohort Ridge sMASE)",
        },
        "counterfactual_support": counter,
        "facts": {
            "evaluator_scope": "training_windows_only",
            "delayed_B_only": delayed_b,
        },
        "instruction": (
            "A two-step workflow was applied at support decision point 928 "
            "and produced the counterfactual Support outcomes above. "
            "Determine whether the workflow can be repaired. If the "
            "counterfactual evidence identifies a component to remove or "
            "keep, propose exactly one edit: ADD one new capability skill "
            "whose frozen program is the repaired workflow (1 step; write "
            "the steps JSON after the line 'Frozen program steps:' in the "
            "skill body). Fill falsification_condition so a deterministic "
            "replay can accept or reject it later. You do not approve your "
            "own edit. If the evidence is insufficient, declare no_proposal "
            "— abstaining is valid."
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


def structural_preflight(manifest: Any) -> dict[str, Any]:
    """Manifest 结构预检：ADD 面白名单 + 单 manifest + schema + 冻结
    Program 解析。"""
    if manifest is None:
        return {"stage": "no_proposal", "preflight": "REJECTED"}
    t = str(manifest.target_surface_id)
    nv = manifest.new_value or {}
    skill_id = str(nv.get("skill_id") or "")
    # LLM 契约错误（普遍）：{skill_id} 模板未实例化——确定性 surface
    # resolution 职责（EditController 无法解析字面量模板）。用 new_value
    # 的 skill_id 实例化并诚实标注（不称 LLM 完全自主）。
    instantiated = t.replace("{skill_id}", skill_id) if skill_id else t
    ok_surface = instantiated.startswith("skill_library.entries/")
    ok_op = manifest.operation.value == "ADD"
    ok_schema = nv.get("schema_version") == "skill-entry/1"
    body = str(nv.get("body") or "")
    frozen = _parse_frozen(body)
    return {
        "stage": "manifest",
        "preflight": ("ACCEPTED" if (ok_surface and ok_op and ok_schema
                                     and frozen is not None and skill_id)
                      else "REJECTED"),
        "target_surface_id": instantiated,
        "skill_id": skill_id,
        "operation": manifest.operation.value,
        "new_value_schema": nv.get("schema_version"),
        "frozen_program": frozen,
        "surface_template_instantiated_by_harness": bool(
            t != instantiated),
        "falsification_condition": list(manifest.falsification_condition or ()),
    }


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS["gefcom"])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    card = build_positive_control_card(executor)
    print(f"== counterfactuals: {json.dumps(card['counterfactual_support'])}")

    # ---- 真实 Slow Agent 链（propose_edit，真实 core/backend）----
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:DELAYED], task_kind="forecast"),
        model=smoke.MODEL,
        base_url=smoke.BASE_URL,
    )
    slow = TTHASlowAgent(core)

    checks: dict[str, Any] = {}
    manifest = None
    try:
        manifest = slow.propose_edit(
            card, SURFACE_CATALOG, h0,
            manifest_preflight=lambda m: None,  # 结构预检在脚本层（验收）
            allowed_operator_contracts=(),
            task_context=None,
        )
    except RuntimeError as exc:  # CountingClient 硬停
        print(f"== budget hard stop: {exc}")
    checks["llm_calls_le_2"] = counter.calls <= 2
    print(f"== manifest: {'None' if manifest is None else 'proposed'}"
          f" llm_calls={counter.calls}")

    preflight = structural_preflight(manifest)
    report_manifest = None
    verdict = "INCONCLUSIVE"
    if manifest is None:
        verdict = "ABSTAIN_ON_IDENTIFIABLE_CONTROL"
    elif preflight["preflight"] != "ACCEPTED":
        verdict = "MANIFEST_NOT_EXECUTABLE"
        report_manifest = {"edit_id": manifest.edit_id,
                           "operation": manifest.operation.value,
                           "target_surface_id": manifest.target_surface_id,
                           "preflight": preflight["preflight"]}
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
        # ---- EditController.apply_to_fork（ADD capability Skill；
        #      surface 模板确定性实例化——见 preflight）----
        import dataclasses
        # 确定性契约修复（非归因内容）：surface 模板实例化 + 按 surface
        # 定义补齐要求的上下文依赖 SHA（从 snapshot 依赖表取；LLM 不负责
        # 知道 surface 的依赖清单）。
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
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        try:
            receipt = controller.apply_to_fork(
                parent, manifest_applied,
                confirmed_cause="SKILL_LIBRARY_GAP")
        except Exception as exc:
            print(f"== apply_to_fork EXC: {type(exc).__name__}: {exc}")
            verdict = "MANIFEST_NOT_EXECUTABLE"
            checks["apply_exception"] = f"{type(exc).__name__}: {exc}"
        else:
            print(f"== applied: edit={receipt.edit_id} candidate_sha="
                  f"{receipt.candidate_harness_content_sha[:12]}")

            # ---- Support replay + delayed ----
            rp = executor.evaluate(steps, ORIGIN)
            gain_p = (float(rp.gain) if rp.gain is not None else None)
            checks["replay_support_positive"] = bool(
                rp.verification.passed and gain_p is not None
                and gain_p >= MATERIAL)
            print(f"== replay @{ORIGIN}: gain={gain_p} "
                  f"passed={rp.verification.passed}")
            rd = executor.evaluate(steps, DELAYED)
            gain_d = (float(rd.gain) if rd.gain is not None else None)
            checks["delayed_no_flip"] = bool(
                gain_d is not None and gain_d >= -MATERIAL)
            print(f"== delayed @{DELAYED}: gain={gain_d}")

            if not checks["replay_support_positive"]:
                verdict = "WRONG_PROGRAM_UPDATE"
            elif not checks["delayed_no_flip"]:
                verdict = "DELAYED_REJECTED"
            else:
                # ---- 下一轮正常 TTHAMethod 实际选择并执行 Skill ----
                skill_cand = f"cand_skill_{preflight['skill_id']}"
                ops_all = tuple(o for o in (
                    "denoise_median", "hampel_filter", "impute_ar",
                    "impute_ema", "impute_fft", "impute_linear",
                    "impute_ssm", "outlier_iqr", "outlier_mad",
                    "period_complete", "period_median_complete",
                    "repair_level_shift", "resample_uniform", "winsorize"))
                method = sealed.TTHAMethod(
                    sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                        sealed.SealedProbeBackend(explore=True,
                                                  operators=ops_all),
                        LocalPublicToolGateway(series0[:DELAYED],
                                               task_kind="forecast"))),
                    receipt.candidate_snapshot.snapshot, ())
                method.bind_round_data(series0[:DELAYED], task_kind="forecast")
                obs = dict(resolver.window_context(values, DELAYED, PERIOD))
                obs["bound_period"] = float(PERIOD)
                r2 = method.prepare(sealed._request(series0, values, DELAYED))
                chosen = method.last_trace.chosen_candidate_id
                adopted = chosen == skill_cand
                r2_gain = None
                if r2.program is not None:
                    cs = tuple((op, dict(pr)) for op, pr in
                               r2.program.execution_steps())
                    rr2 = executor.evaluate(cs, DELAYED)
                    r2_gain = (float(rr2.gain) if rr2.gain is not None
                               else None)
                checks["next_round_actual_adoption"] = bool(adopted)
                checks["next_round_executed"] = r2_gain is not None
                print(f"== @{DELAYED}: chosen={chosen} adopted={adopted} "
                      f"gain={r2_gain}")

                # ---- remove-skill 对照（H0 无 skill）----
                method_ctrl = sealed.TTHAMethod(
                    sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                        sealed.SealedProbeBackend(explore=True,
                                                  operators=ops_all),
                        LocalPublicToolGateway(series0[:DELAYED],
                                               task_kind="forecast"))),
                    h0, ())
                method_ctrl.bind_round_data(series0[:DELAYED],
                                            task_kind="forecast")
                method_ctrl.prepare(sealed._request(series0, values, DELAYED))
                chosen_ctrl = method_ctrl.last_trace.chosen_candidate_id
                checks["removal_changes_action"] = bool(
                    chosen_ctrl != skill_cand)
                print(f"== @{DELAYED} ctrl(no skill): chosen={chosen_ctrl}")

                passed = all(v is True for k, v in checks.items()
                             if k not in ("next_round_executed",))
                verdict = ("REAL_SLOW_AGENT_PATCH_PASS" if passed else
                           "CONTROL_BINDING_FAIL")
            try:
                store.discard_fork(parent.root)
            except ValueError:
                pass  # parent 是 materialize 目录（非 fork）——无需清理

    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-real-slow-agent-positive-control",
        "case": "GEFCom impute_ssm->outlier_iqr（P0 正控；反事实表明确支持"
                "删 A 留 B；不预标注答案）",
        "counterfactuals": card["counterfactual_support"],
        "delayed_b_only": card["facts"]["delayed_B_only"],
        "manifest": report_manifest,
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": counter.calls,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
