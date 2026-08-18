"""P0 E0 对照臂：只有完整 Workflow 总 gain → 自由 Manifest（当前方法）。

审查裁决（二十一）对照设计：
    Arm   归因反馈                              LLM 输出
    E0    只有完整 Workflow 总 gain             自由 Manifest
    E1    完整 gain + stepwise counterfactual   Typed Patch 选择
    E2    同 E1，但无 Source Memory             Typed Patch 选择

E0 与 E1/E2 的差异被隔离为两点：① 反馈只有总 gain（无 stepwise
counterfactual）；② 输出自由 Manifest（slow_edit_v1 schema，非受限选择）。
E0 的 Memory 与 E1 相同（隔离变量只有反馈内容与输出形式）。

机制复用 run_v1_slow_path_smoke.py（已实测）：TTHASlowAgent.propose_edit +
manifest_preflight（first_fault:<face> 契约）+ structural preflight（非
compiler 的门） + CountingClient（temperature 0，总调用上限 2）。

E0 判定（目标判断者视角）：
  - manifest 生成率（no_proposal 计数）；
  - preflight 通过（单修改 + 合法面 + base sha + first_fault 编码）；
  - 可执行性：manifest 是否含可执行的 program 修改（skill body 内 steps）
    → 有则 Support/delayed replay（与 E1 同一执行器），无则记录
    NO_EXECUTABLE_PATCH（这正是 Typed Patch 对照要暴露的差异）。

信息墙：与 E1 相同——delayed、正确答案、Target future、手工 first fault
不进输入。delayed 只在 replay 打开（Patch 冻结后）。

用法：
  python evaluation/functional/run_v1_counterfactual_attribution_e0.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import AgictoChatCompletionsBackend  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
MODEL = smoke.MODEL
BASE_URL = smoke.BASE_URL
KEY_ENVS = smoke.KEY_ENVS
SCAN_REPORT = Path("artifacts/functional/e2/w1_counterfactual_scan_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_counterfactual_attribution_e0_report.json")


def build_total_gain_card(
    case: Mapping[str, Any],
    values: Mapping[str, Any],
    origin: int,
) -> dict[str, object]:
    """E0 失败卡：**只有完整 Workflow 总 gain**（identity/AB），无 stepwise；
    含公开 Context 与 Memory（与 E1 相同的 Source Memory——隔离变量只留
    反馈粒度与输出形式）。"""
    sup = case["support"]
    ctx = resolver.window_context(values, origin, PERIOD)
    return {
        "pattern_id": f"gefcom-{case['step_a']}-to-{case['step_b']}-sign-flip",
        "failure_family": "workflow_effect_sign_flip",
        "first_fault_candidates": list(smoke.FIRST_FAULT_FACES),
        "observable_signature": {"task_kind": "forecast"},
        "context_evidence": {
            "support_context": dict(ctx),
        },
        "workflow": {
            "steps": [
                {"op": case["step_a"], "params": dict(wiring.contract_params(str(case["step_a"]), PERIOD))},
                {"op": case["step_b"], "params": dict(wiring.contract_params(str(case["step_b"]), PERIOD))},
            ],
            "scope": "training_windows_only",
            "evaluator": "v6._evaluate (per-training-window, cohort Ridge sMASE)",
        },
        "observed_effects": {
            "total_gain_only": {
                "support_origin": origin,
                "identity_gain": 0.0,
                "full_workflow_gain": sup.get("A_to_B"),
                "note": "stepwise (leave-one-step-out) effects intentionally "
                        "not provided in this arm",
            },
        },
        "experience_memory": [
            m for m in smoke_module_memory_lines()
        ],
    }


def smoke_module_memory_lines() -> list[str]:
    """与 E1 相同 Memory（复用 run_v1_counterfactual_attribution 的已核实行）。"""
    import run_v1_counterfactual_attribution as attribution  # noqa: PLC0415
    return list(attribution.MEMORY_LINES)


def main() -> int:
    root = PROJECT_ROOT
    scan = json.loads((root / SCAN_REPORT).read_text(encoding="utf-8"))
    case = scan.get("first_hit")
    if case is None:
        print("== no counterfactual headroom case — E0 skipped")
        return 0
    origin = int(case["origin"])

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0",
                          verify_lock=False)
    card = build_total_gain_card(case, values, origin)

    api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        raise SystemExit(f"missing LLM key: {KEY_ENVS}")
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter, base_url=BASE_URL)
    series0 = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(series0[:origin], task_kind="forecast"),
        model=MODEL,
        base_url=BASE_URL,
    )
    slow = TTHASlowAgent(core)

    print(f"== E0 (total gain only -> free Manifest): case @{origin} "
          f"{case['step_a']}->{case['step_b']}")
    manifest = slow.propose_edit(
        card,
        smoke.SURFACE_CATALOG,
        h0,
        manifest_preflight=smoke.first_fault_face_preflight,
        allowed_operator_contracts=(),
        fixed_probe_contracts={},
    )
    structural = None
    if manifest is not None:
        structural = smoke.structural_preflight(manifest, slow, h0)
        print(f"== manifest: {manifest.edit_id} -> {manifest.target_surface_id} "
              f"op={manifest.operation.value} "
              f"preflight={structural['preflight']} "
              f"first_fault={structural['first_fault_face_encoded']}")

    report: dict[str, Any] = {
        "experiment_id": "v1-counterfactual-attribution-e0",
        "case": {k: case[k] for k in
                 ("origin", "step_a", "step_b", "support", "delayed_origin")},
        "delayed_runner_side_not_in_feedback": case.get("delayed"),
        "llm_api_call_count": counter.calls,
        "structural_preflight": structural,
        "proposed_manifest": (smoke._plain({
            "edit_id": manifest.edit_id,
            "target_pattern_id": manifest.target_pattern_id,
            "target_surface_id": manifest.target_surface_id,
            "operation": manifest.operation.value,
            "new_value": manifest.new_value,
        }) if manifest is not None else None),
    }
    if manifest is not None and structural is not None and structural["preflight"] == "ACCEPTED":
        new_value = manifest.new_value or {}
        body = str(new_value.get("body", "")) if isinstance(new_value, Mapping) else ""
        report["executable_patch_extracted"] = None
        # 可执行性：skill body 是否含可解析的 Workflow steps（受控格式）
        steps = smoke_extract_steps_from_skill_body(body)
        if steps is not None:
            report["executable_patch_extracted"] = steps
            verification = executor.verify(steps, origin)
            support = executor.evaluate(steps, origin)
            delayed = executor.evaluate(steps, origin + HORIZON)
            report["replay"] = {
                "compiler": {
                    "passed": verification.passed,
                    "checked_windows": verification.checked_windows,
                },
                "support": {
                    "origin": origin,
                    "gain": (float(support.gain)
                             if support.gain is not None else None),
                },
                "delayed": {
                    "origin": origin + HORIZON,
                    "gain": (float(delayed.gain)
                             if delayed.gain is not None else None),
                    "opened_after_patch_freeze": True,
                },
            }
        report["outcome"] = (
            "EXECUTABLE_MANIFEST" if report["executable_patch_extracted"]
            else "NO_EXECUTABLE_PATCH")
    else:
        report["outcome"] = ("NO_PROPOSAL" if manifest is None
                             else "PREFLIGHT_REJECTED")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== outcome: {report['outcome']} (calls={counter.calls})")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def smoke_extract_steps_from_skill_body(body: str) -> tuple | None:
    """从自由 Manifest 的 skill body 提取可执行 steps（若 LLM 按约定写
    `{"steps": [{"op": ..., "params": {...}}, ...]}` 的 JSON 片段）。"""
    if not body:
        return None
    try:
        start = body.find("{")
        if start < 0:
            return None
        parsed = json.loads(body[start:])
    except json.JSONDecodeError:
        return None
    raw_steps = parsed.get("steps") if isinstance(parsed, Mapping) else None
    if not isinstance(raw_steps, list) or not raw_steps:
        return None
    steps = []
    for item in raw_steps:
        if not isinstance(item, Mapping) or not isinstance(item.get("op"), str):
            return None
        params = item.get("params") or {}
        if not isinstance(params, Mapping):
            return None
        steps.append((item["op"], dict(params)))
    return tuple(steps) if steps else None


if __name__ == "__main__":
    raise SystemExit(main())
