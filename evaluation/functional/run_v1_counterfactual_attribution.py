"""P0-2/3/4：反事实 Workflow 归因实验（E1 主实验 + E2 对照；E0 单独脚本）。

审查裁决（二十一）P0 核心：
  - 让 LLM 从 Typed Patch（KEEP / REMOVE_STEP_A / REMOVE_STEP_B / ABSTAIN）
    中选择更新方向，而不是从原始 Episode 自由生成完整 Harness Patch；
  - 确定性 Runtime 提供 Action–Response 反事实证据（identity / A only /
    B only / A→B 的 Support gain）；
  - 信息墙：LLM 不可见 delayed outcome、哪个 Patch 是正确答案、Target
    future、手工 first fault 标注；
  - Patch 冻结后再打开 delayed；
  - 通过条件（LLM_ATTRIBUTION_TO_HARNESS_UPDATE_PASS）：
      Patch 可编译（H0 verifier 全窗口通过）
      + Support 优于原 Workflow
      + delayed 不发生负迁移
      + 下一轮行为受新 Skill 影响。

本脚本 = E1（stepwise counterfactual + Source Memory → Typed Patch 选择）
与 E2（同 E1 无 Source Memory）。E0（只有完整 Workflow 总 gain → 自由
Manifest）在 run_v1_slow_path_smoke.py 的机制上单独实现（对照臂）。

LLM provider：agicto gpt-5.6-luna（temperature 0；预算不卡死——审查裁决
2026-08-08："用 luna 吧，预算的话其实这个不会卡的很死"）；单 arm 总调用
上限 2（validation_retries=1 的唯一一次格式纠正仍超限 = 契约重试失效信号，
硬停——与审查裁决 二十 一致）。

用法：
  python evaluation/functional/run_v1_counterfactual_attribution.py
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
import run_v1_a5_vs_a3 as core  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_slow_path_smoke import CountingClient  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

TARGET_DOMAIN = "gefcom"
PERIOD = 24
HORIZON = 48
MATERIAL = core.MATERIAL_THRESHOLD  # 0.005
MODEL = "gpt-5.6-luna"
BASE_URL = "https://api.agicto.cn/v1"
KEY_ENVS = ("OPENAI_API_KEY", "AGICTO_API_KEY")
SCAN_REPORT = Path("artifacts/functional/e2/w1_counterfactual_scan_report.json")
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_counterfactual_attribution_report.json")

LEGAL_PATCH_IDS = ("KEEP", "REMOVE_STEP_A", "REMOVE_STEP_B", "ABSTAIN")

# E1 Source Memory：真实已暴露 Episode（与 case 同源，非构造答案）。
# 数值逐一核实：
#   - winsorize @832 support +0.14403 / delayed +0.51098（w1_scope_alignment
#     part_c control）；
#   - winsorize @928 support -0.16360 / delayed +0.61000（w1_scope_executor_loop
#     a5 delayed：winsorize CONFLICT）；
#   - impute_fft @928 support +0.01849 / delayed -0.03496（w1_scope_executor_loop
#     a3 delayed：impute_fft CONFLICT/RESTRICTED）；
#   - outlier_iqr @928 support +0.04386（本次 w1_counterfactual_scan 单算子表）。
MEMORY_LINES = [
    "gefcom winsorize @832: support +0.14403 delayed +0.51098 → POSITIVE (LOCAL_ACTIVE)",
    "gefcom winsorize @928: support -0.16360 delayed +0.61000 → CONFLICT (support negative, delayed positive)",
    "gefcom impute_fft @928: support +0.01849 delayed -0.03496 → CONFLICT (RESTRICTED; support positive, delayed negative)",
    "gefcom outlier_iqr @928: support +0.04386 → POSITIVE",
]


def _api_kwargs() -> dict[str, Any]:
    api_key = next((os.environ.get(k, "").strip() for k in KEY_ENVS
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        raise RuntimeError(f"{' or '.join(KEY_ENVS)} is required")
    return {"api_key": api_key, "base_url": BASE_URL}


def _patch_to_steps(case: Mapping[str, Any], patch_id: str) -> tuple[tuple[str, dict[str, object]], ...]:
    """Typed Patch → 冻结后的 Workflow steps（KEEP = A→B；REMOVE_STEP_X =
    leave-one-out 后的单步）。"""
    a = str(case["step_a"])
    b = str(case["step_b"])
    pa = dict(wiring.contract_params(a, PERIOD))
    pb = dict(wiring.contract_params(b, PERIOD))
    if patch_id == "KEEP":
        return ((a, pa), (b, pb))
    if patch_id == "REMOVE_STEP_A":
        return ((b, pb),)
    if patch_id == "REMOVE_STEP_B":
        return ((a, pa),)
    raise ValueError(f"no executable steps for patch_id={patch_id}")


def build_feedback(
    case: Mapping[str, Any],
    values: Mapping[str, Any],
    *,
    include_memory: bool,
) -> dict[str, Any]:
    """构造 LLM 可见的反事实反馈（信息墙：**不含 delayed、正确答案、Target
    future、first fault 标注**——由本函数结构保证，见审查者检查项）。"""
    origin = int(case["origin"])
    sup = case["support"]
    ctx = resolver.window_context(values, origin, PERIOD)
    ctx_plain = {k: round(float(v), 6) for k, v in sorted(ctx.items())}
    lines = [
        "You are the slow attribution path of a time-series preprocessing harness.",
        "",
        "A two-step Workflow was applied at support decision point "
        f"{origin}. You must decide the update direction for this Workflow by "
        "choosing exactly one legal patch ID.",
        "",
        "== Source Workflow (A then B) ==",
        f"  A: {case['step_a']} params={json.dumps(wiring.contract_params(str(case['step_a']), PERIOD), sort_keys=True)}",
        f"  B: {case['step_b']} params={json.dumps(wiring.contract_params(str(case['step_b']), PERIOD), sort_keys=True)}",
        "",
        "== Public context at the decision point (deployment-visible) ==",
        json.dumps(ctx_plain, indent=2, sort_keys=True),
        "",
        "== Support outcomes (training-window cohort, sMASE gain; "
        "positive = better than no preprocessing) ==",
        "  identity (no preprocessing): baseline (gain 0 by definition)",
        f"  A only: {_fmt(sup.get('A_only'))}",
        f"  B only: {_fmt(sup.get('B_only'))}",
        f"  A then B: {_fmt(sup.get('A_to_B'))}",
    ]
    if include_memory:
        lines += [
            "",
            "== Experience memory (signed, from earlier decision points) ==",
            *[f"  - {m}" for m in MEMORY_LINES],
        ]
    lines += [
        "",
        "== Legal patch IDs ==",
        "  KEEP         keep the two-step Workflow A -> B",
        "  REMOVE_STEP_A  drop step A, keep only step B",
        "  REMOVE_STEP_B  drop step B, keep only step A",
        "  ABSTAIN      no change proposed (evidence insufficient)",
        "",
        "== Your task ==",
        "Choose exactly one legal patch ID. If the stepwise evidence is "
        "insufficient or contradictory, ABSTAIN is valid — abstaining is not "
        "a failure.",
        'Output JSON only: {"patch_id": "<one legal ID>", '
        '"evidence_refs": ["..."], "rationale": "..."}',
    ]
    return {"prompt": "\n".join(lines), "include_memory": include_memory}


def _fmt(g: object) -> str:
    if g is None:
        return "instrument failure / not evaluated"
    return f"{float(g):+.5f}"


def typed_patch_call(client: Any, feedback: Mapping[str, Any],
                     *, max_calls: int = 2) -> dict[str, Any]:
    """一次选择 + 机器校验（patch_id ∈ 合法集）；非法 → 重试 1 次；超限硬停。"""
    attempts = []
    for i in range(max_calls):
        kwargs = {
            "model": MODEL,
            "messages": [{"role": "user", "content": feedback["prompt"]}],
            "temperature": 0,
        }
        try:
            resp = client.chat.completions.create(
                **kwargs, response_format={"type": "json_object"})
        except Exception:
            # agicto 兼容差异：不支持 json_object 时降级为纯 prompt 约束
            resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        attempts.append(raw)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue  # 交给重试（格式纠正）
        pid = payload.get("patch_id")
        if pid in LEGAL_PATCH_IDS:
            return {
                "patch_id": pid,
                "evidence_refs": payload.get("evidence_refs", []),
                "rationale": payload.get("rationale", ""),
                "attempts": attempts,
            }
        # 非法 patch_id → 重试
    return {"patch_id": None, "evidence_refs": [], "rationale": "",
            "attempts": attempts, "invalid": True}


def run_typed_arm(
    client: Any,
    case: Mapping[str, Any],
    values: Mapping[str, Any],
    executor: ScopeExecutor,
    *,
    arm: str,
    include_memory: bool,
) -> dict[str, Any]:
    """E1（memory=True）/ E2（memory=False）共用的 Typed Patch 臂。"""
    feedback = build_feedback(case, values, include_memory=include_memory)
    calls_before = client.calls
    choice = typed_patch_call(client, feedback)
    origin = int(case["origin"])
    delayed_origin = origin + HORIZON
    out: dict[str, Any] = {
        "arm": arm,
        "include_memory": include_memory,
        "llm_calls": client.calls - calls_before,
        "choice": choice,
        "prompt": feedback["prompt"],
    }
    pid = choice.get("patch_id")
    if pid is None or pid == "ABSTAIN":
        out["abstained"] = True
        out["verdict_component"] = "ABSTAIN"
        return out
    steps = _patch_to_steps(case, pid)
    out["frozen_steps"] = [{"op": op, "params": dict(pr)} for op, pr in steps]

    # ---- Patch 冻结后 replay（P0-4）----
    # compiler gate：窗口级 H0 verifier（0.35，全训练窗口）
    verification = executor.verify(steps, origin)
    out["compiler"] = {
        "passed": verification.passed,
        "checked_windows": verification.checked_windows,
        "rejected_windows": verification.rejected_windows,
    }
    support = executor.evaluate(steps, origin)
    delayed = executor.evaluate(steps, delayed_origin)
    out["replay"] = {
        "support": {
            "origin": origin,
            "gain": (float(support.gain) if support.gain is not None else None),
            "verification_passed": support.verification.passed,
            "error": support.error,
        },
        "delayed": {
            "origin": delayed_origin,
            "gain": (float(delayed.gain) if delayed.gain is not None else None),
            "verification_passed": delayed.verification.passed,
            "error": delayed.error,
        },
        "delayed_opened_after_patch_freeze": True,
    }
    return out


def main() -> int:
    root = PROJECT_ROOT
    scan = json.loads((root / SCAN_REPORT).read_text(encoding="utf-8"))
    case = scan.get("first_hit")
    if case is None:
        print("== no counterfactual headroom case found — nothing to attribute")
        return 0

    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    executor = ScopeExecutor(roster, values, config, evaluate_fn=v6._evaluate)
    import openai
    client = CountingClient(openai.OpenAI(**_api_kwargs()))

    print(f"== case: @{case['origin']} {case['step_a']} -> {case['step_b']}")
    print(f"   support: {case['support']}")
    print("== delayed (Runner side; NOT in LLM feedback): "
          f"{case['delayed']}")

    e1 = run_typed_arm(client, case, values, executor,
                       arm="E1", include_memory=True)
    print(f"== E1: choice={e1['choice'].get('patch_id')} "
          f"calls={e1['llm_calls']}")
    if "replay" in e1:
        print(f"   replay support={e1['replay']['support']['gain']} "
              f"delayed={e1['replay']['delayed']['gain']}")

    e2 = run_typed_arm(client, case, values, executor,
                       arm="E2", include_memory=False)
    print(f"== E2: choice={e2['choice'].get('patch_id')} "
          f"calls={e2['llm_calls']}")
    if "replay" in e2:
        print(f"   replay support={e2['replay']['support']['gain']} "
              f"delayed={e2['replay']['delayed']['gain']}")

    # ---- 判定（E1 为主）----
    judgement = _judge(e1, case)
    report = {
        "experiment_id": "v1-counterfactual-attribution",
        "case": {k: case[k] for k in
                 ("origin", "step_a", "step_b", "support", "delayed_origin")},
        "delayed_runner_side_not_in_feedback": case.get("delayed"),
        "e1": e1,
        "e2": e2,
        "llm_api_call_count": client.calls,
        "judgement": judgement,
    }
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"== judgement: {judgement['verdict']}")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def _judge(e1: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    """通过条件（目标判断者）：
    Patch 可编译 + Support 优于原 Workflow + delayed 无负迁移
    + 下一轮行为受新 Skill 影响 → LLM_ATTRIBUTION_TO_HARNESS_UPDATE_PASS。"""
    origin = int(case["origin"])
    sup = case["support"]
    ab_gain = sup.get("A_to_B")
    checks: dict[str, bool] = {
        "patch_compiles": False,
        "support_better_than_original": False,
        "delayed_no_negative_transfer": False,
        "next_round_skill_influence": False,  # P0-5 由下一轮验证补填
    }
    pid = e1.get("choice", {}).get("patch_id")
    if pid is None:
        return {"verdict": "E1_ABSTAIN_OR_INVALID", "checks": checks,
                "note": "LLM abstained or produced no legal patch"}
    replay = e1.get("replay", {})
    checks["patch_compiles"] = bool(e1.get("compiler", {}).get("passed"))
    patched_gain = replay.get("support", {}).get("gain")
    if patched_gain is not None and ab_gain is not None:
        checks["support_better_than_original"] = bool(
            patched_gain > ab_gain + 1e-9)
    d = replay.get("delayed", {})
    d_gain = d.get("gain")
    if d_gain is not None:
        # 无负迁移：无显著 harm 且不劣于原 Workflow delayed（若原可评）
        d_ab = case.get("delayed", {}).get("A_to_B")
        no_harm = d_gain >= -MATERIAL
        no_worse = (d_ab is None) or (d_gain >= float(d_ab) - MATERIAL)
        checks["delayed_no_negative_transfer"] = bool(no_harm and no_worse)
    passed = all(checks.values())
    verdict = ("LLM_ATTRIBUTION_TO_HARNESS_UPDATE_PASS" if passed
               else "LLM_ATTRIBUTION_TO_HARNESS_UPDATE_PARTIAL")
    return {"verdict": verdict, "checks": checks,
            "note": "next_round_skill_influence 由 P0-5 下一轮验证补填"}


if __name__ == "__main__":
    raise SystemExit(main())
