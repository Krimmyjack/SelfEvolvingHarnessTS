"""V1 Program Scope / Evaluator 对齐（零 LLM，2026-08-08）。

审查裁决（十一）：执行路线 1，但调查目标改为 **Program Scope / Evaluator 对齐**。
不修执行器（run_pipeline 与 v6._apply_program 是同一执行器——v6._apply_program 内部
调用 run_pipeline）、不换数据、不启动 LLM。

六步（裁决原文）：
  1. 用完全相同的输入数组、参数和 Program，比较 run_pipeline 与 _apply_program 输出；
     预期逐位相同（Part A）。
  2. 冻结本 family 的真正作用 Scope。按 V1 目标与 Episode 的 scope=training_rows，
     "逐训练窗口、跨 cohort 应用 Workflow"为规范语义（Part B 评估器 = v6._evaluate；
     _public_context["program_application_scope"]="training_windows_only" 早已声明）。
  3. 确定性 Runtime 按规范 Scope 执行——不是把单序列 prefix 的 PreparedSeries
     当成最终训练数据（Part B/C 评估一律走 v6._evaluate 窗口协议）。
  4. verifier 检查实际将执行的训练窗口/作用范围，保持 H0 0.35，不得再用 1.0
     （Part B/C 的窗口级 verifier：每个训练窗口独立 verify_candidate）。
  5. Support 与 delayed 都在各自决策点重新执行同一个冻结 Workflow，不能拼接
     旧 prepared 与新 raw（Part C：v1.gain_at @832 与 @880 各自 _evaluate）。
  6. 用一个暴露正控重跑；通过后才能写 Episode 并检查下一轮检索（Part C）。

预期：此前"prepared 单序列语义下无双正"是 prefix 口径的错误观察；规范语义
（v6._evaluate 逐窗口）下 winsorize @832→880 双正（v6 语义 +0.144@832 / +0.511@880）。

用法：
  python evaluation/functional/run_v1_scope_alignment.py
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
import run_v1_fastpath as v1  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402（策略/contract_params 复用）
import signed_radius as resolver  # noqa: E402
import run_v1_a5_vs_a3 as core  # noqa: E402（MATERIAL_THRESHOLD）

from SelfEvolvingHarnessTS.contracts.candidate import Candidate  # noqa: E402
from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.program import Program  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import MetricSpec, forecast_task_spec_v1  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.runtime.candidate_verification import verify_candidate  # noqa: E402

HORIZON = 48
CONTEXT_LENGTH = 192
PERIOD = 24
TARGET_DOMAIN = "gefcom"
CHAIN_ORIGINS = [736, 784, 832, 880, 928, 976]
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_scope_alignment_report.json")
MATERIAL = core.MATERIAL_THRESHOLD


def _h0_limits(root: Path) -> tuple[float, bool]:
    """H0 部署约束（verification.json 单一来源）——verifier 全程用此值。"""
    verification = json.loads(
        (root / "methods" / "ttha" / "harness" / "h0" / "verification.json")
        .read_text(encoding="utf-8"))
    return (float(verification["max_modified_fraction"]),
            bool(verification["preserve_outside_candidate_region"]))


# ---------------------------------------------------------------------------
# Part A：执行器逐位一致性（裁决 Step 1）
# ---------------------------------------------------------------------------

def part_a_bitwise_consistency(
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
) -> dict[str, Any]:
    """同一输入数组（240 步训练窗口）、同一参数、同一 Program：
    run_pipeline 直接调用 vs v6._apply_program（其内部调用 run_pipeline）。
    预期输出逐位相同（同一执行器；差异仅可能来自 _apply_program 的
    _linear_integrity 后处理，且只在输入含 NaN 时触发）。"""
    from SelfEvolvingHarnessTS.runtime.executor import run_pipeline

    series = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    anchor = 672  # 一个落在 origin=832 决策点内的训练 anchor
    window = series[anchor - CONTEXT_LENGTH: anchor + HORIZON]
    checks: list[dict[str, Any]] = []
    for op in ("winsorize", "repair_level_shift", "hampel_filter"):
        params = wiring.contract_params(op, PERIOD)
        program = Program.from_steps([(op, params)], source="scope_alignment_a")
        compiled = v1.make_compiled(op, params)
        direct = run_pipeline(program.execution_steps(), window,
                              source="scope_alignment_a")
        assert direct.ok and direct.artifact is not None
        direct_array = np.asarray(direct.artifact, dtype=np.float64).ravel()
        via, via_trace = v6._apply_program(window, compiled)
        identical = bool(np.array_equal(direct_array, via, equal_nan=True))
        max_diff = float(np.nanmax(np.abs(direct_array - via))) if direct_array.size else 0.0
        checks.append({
            "operator": op,
            "identical_bitwise": identical,
            "max_abs_diff": round(max_diff, 12),
            "direct_trace_steps": len(direct.trace),
            "apply_program_trace_steps": len(via_trace),
            "window_has_nan": bool((~np.isfinite(window)).any()),
        })
    return {
        "window_geometry": {
            "anchor": anchor,
            "slice": f"[{anchor - CONTEXT_LENGTH}:{anchor + HORIZON}]",
            "size": int(window.size),
        },
        "checks": checks,
        "all_identical": all(c["identical_bitwise"] for c in checks),
    }


# ---------------------------------------------------------------------------
# Part B：规范 Scope headroom 扫描（裁决 Step 2-4）
# ---------------------------------------------------------------------------

def _window_verify(
    candidate: Any,
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
    origin: int,
    allowed: Sequence[str],
    max_fraction: float,
    preserve_outside: bool,
) -> list[dict[str, Any]]:
    """规范 Scope verifier（裁决 Step 4）：对**实际将执行的每个训练窗口**
    （与 v6._evaluate 相同的 train rows × anchor 集合、相同切法）独立
    verify_candidate；保持 H0 0.35。返回被拒窗口列表（空 = 全部通过）。

    inspected_regions = 整个窗口（窗口即"候选作用区域"；窗口外修改不在此协议内，
    因 Workflow 只在窗口上执行）。
    """
    rejected: list[dict[str, Any]] = []
    for row in roster:
        if str(row["role"]) != "train":
            continue
        raw = np.asarray(values[str(row["series_uid"])], dtype=np.float64)
        for anchor in config["anchors"]:  # type: ignore[union-attr]
            anchor = int(anchor)
            if anchor + HORIZON > origin:
                continue
            window = raw[anchor - CONTEXT_LENGTH: anchor + HORIZON]
            artifact = verify_candidate(
                candidate, window,
                allowed_operators=tuple(allowed),
                inspected_regions=((0, int(window.size)),),
                maximum_modified_fraction=max_fraction,
                preserve_outside_inspected_region=preserve_outside,
                require_finite_output=False,
            )
            if not artifact.selectable:
                rejected.append({
                    "series_uid": str(row["series_uid"]),
                    "anchor": anchor,
                    "rejection_code": artifact.receipt.rejection_code,
                })
    return rejected


def _scope_scan(
    root: Path,
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
    origins: Sequence[int],
    max_fraction: float,
    preserve_outside: bool,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    """规范 Scope headroom 扫描（零 LLM）：

    - 供给层 actionable = fast_agent._actionable_operators（与真实入口同源，
      但那是 prefix 口径——记录差异）；
    - 窗口级 verifier（0.35）→ scope_selectable（真正可执行的集合）；
    - 评估 = v1.gain_at（内部 v6._evaluate：逐训练窗口、跨 cohort 执行
      run_pipeline——规范语义，不是 prefix PreparedSeries）；
    - 双正（support ≥ M 且 delayed ≥ M）→ 正控候选（best 取 support 最大）。
    """
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (
        _actionable_operators, _allowed_operators,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import extract_public_features
    from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view

    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)
    series = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    baseline_cache: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    prefix_actionable_by_origin: dict[int, list[str]] = {}
    for origin in origins:
        r_values = series[:origin]
        request = PreparationRequest(
            "scope-scan",
            r_values,
            forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                                  metric=MetricSpec("sMASE", "lower_is_better")),
            {},
        )
        features = extract_public_features(r_values, task_kind="forecast")
        view = resolve_harness_view(h0, features, role="fast")
        actionable = _actionable_operators(request, r_values, view,
                                           _allowed_operators(request))
        prefix_actionable_by_origin[origin] = sorted(actionable)
        for op in sorted(actionable):
            params = wiring.contract_params(op, PERIOD)
            candidate = Candidate.program_candidate(
                f"probe_{op}", Program.from_steps([(op, params)], source="scope_scan"),
                source="scope_scan")
            rejected = _window_verify(candidate, roster, values, config, origin,
                                      (op,), max_fraction, preserve_outside)
            row: dict[str, Any] = {
                "operator": op, "origin": origin,
                "scope_selectable": not rejected,
                "rejected_windows": len(rejected),
                "rejection_codes": sorted({r["rejection_code"] for r in rejected}),
                "support_gain": None, "delayed_gain": None,
            }
            if not rejected:
                compiled = v1.make_compiled(op, params)
                gain = v1.gain_at(roster, values, config, compiled, origin,
                                  baseline_cache)
                if gain is not None:
                    row["support_gain"] = round(float(gain), 6)
                if gain is not None and float(gain) >= MATERIAL:
                    # delayed：窗口集合变大（更多 anchor 落入决策点）→ 重新验证 +
                    # 重新执行冻结 Workflow（v1.gain_at @origin+48 内部逐窗口重跑）
                    rejected_d = _window_verify(
                        candidate, roster, values, config, origin + HORIZON,
                        (op,), max_fraction, preserve_outside)
                    row["delayed_rejected_windows"] = len(rejected_d)
                    if not rejected_d:
                        delayed_gain = v1.gain_at(
                            roster, values, config, compiled, origin + HORIZON,
                            baseline_cache)
                        if delayed_gain is not None:
                            row["delayed_gain"] = round(float(delayed_gain), 6)
                    if (row["delayed_gain"] is not None
                            and row["delayed_gain"] >= MATERIAL
                            and (best is None
                                 or float(row["support_gain"]) > best["support_gain"])):
                        best = {"operator": op, "origin": origin,
                                "support_gain": float(row["support_gain"]),
                                "delayed_gain": float(row["delayed_gain"])}
            rows.append(row)
    summary = {
        "origins": list(origins),
        "prefix_actionable_by_origin": prefix_actionable_by_origin,
        "scope_selectable_operators": sorted({
            r["operator"] for r in rows if r["scope_selectable"]}),
        "scope_rejected_operators": sorted({
            r["operator"] for r in rows if not r["scope_selectable"]}),
    }
    return best, rows, summary


# ---------------------------------------------------------------------------
# Part C：暴露正控闭环（裁决 Step 5-6）
# ---------------------------------------------------------------------------

def part_c_closed_loop(
    root: Path,
    roster: Sequence[Mapping[str, object]],
    values: Mapping[str, Any],
    config: Mapping[str, object],
    control: Mapping[str, Any],
    max_fraction: float,
    preserve_outside: bool,
) -> dict[str, Any]:
    """用暴露正控（Part B best）跑机械闭环（零 LLM）：

    Round1：控制 Episode（双正，POSITIVE）注入 Memory → Fast Agent prepare
      → chosen = cand_{op}（确定性策略读 Reference 1）→ 窗口级 verifier（0.35）
      → Support = v1.gain_at @origin（规范语义：逐窗口重新执行冻结 Workflow）
      → 写 target Episode → delayed = v1.gain_at @origin+48（**不拼接**：完整窗口
      集合重新执行同一冻结 Workflow）→ 四态更新。
    Round2：注入更新后 Memory → 确认 Reference 1 渲染该算子且行动一致。
    """
    from run_v1_target_local_loop import update_delayed_status, write_target_episode

    op = str(control["operator"])
    origin = int(control["origin"])
    params = wiring.contract_params(op, PERIOD)
    compiled = v1.make_compiled(op, params)
    baseline_cache: dict[int, float] = {}

    # 控制 Episode（暴露正控：双正证据，非运行时决定）
    c_support = v1.gain_at(roster, values, config, compiled, origin, baseline_cache)
    c_delayed = v1.gain_at(roster, values, config, compiled,
                           origin + HORIZON, baseline_cache)
    assert c_support is not None and c_delayed is not None
    control_episode = write_target_episode(
        domain=TARGET_DOMAIN, op=op,
        program_steps=[{"op": op, "params": dict(params)}],
        support_gain=float(c_support), delayed_gain=float(c_delayed),
        support_context=resolver.window_context(values, origin, PERIOD),
        delayed_context=resolver.window_context(values, origin + HORIZON, PERIOD),
    )
    control_episode = update_delayed_status(
        control_episode, float(c_delayed),
        delayed_context=resolver.window_context(values, origin + HORIZON, PERIOD))
    memory: list[Any] = [control_episode]

    series = np.asarray(values[list(values.keys())[0]], dtype=np.float64)
    h0 = compile_snapshot(root / "methods" / "ttha" / "harness" / "h0", verify_lock=False)
    rounds: list[dict[str, Any]] = []
    for idx in range(2):
        label = f"round{idx + 1}"
        observed = dict(resolver.window_context(values, origin, PERIOD))
        observed["bound_period"] = float(PERIOD)
        r_values = series[:origin]
        backend = wiring.DeterministicStrategyBackend()
        request = PreparationRequest(
            "scope-alignment-loop",
            r_values,
            forecast_task_spec_v1(horizon=HORIZON, downstream_model_class="ridge",
                                  metric=MetricSpec("sMASE", "lower_is_better")),
            dict(observed),
        )
        result, trace = TTHAFastAgent(
            TTHAAgentCore(backend, LocalPublicToolGateway(r_values, task_kind="forecast"))
        ).prepare(request, h0, experience_episodes=tuple(memory))
        support_gain = delayed_gain = None
        scope_legal = False
        if trace.chosen_candidate_id == f"cand_{op}":
            candidate = Candidate.program_candidate(
                f"probe_{op}",
                Program.from_steps([(op, params)], source="scope_alignment_c"),
                source="scope_alignment_c")
            # 窗口级 verifier（0.35）——delayed 决策点（窗口集合更大）也验证
            rejected = _window_verify(candidate, roster, values, config, origin,
                                      (op,), max_fraction, preserve_outside)
            scope_legal = not rejected
            if scope_legal:
                support_gain = v1.gain_at(roster, values, config, compiled,
                                          origin, baseline_cache)
                rejected_d = _window_verify(
                    candidate, roster, values, config, origin + HORIZON,
                    (op,), max_fraction, preserve_outside)
                if not rejected_d:
                    delayed_gain = v1.gain_at(roster, values, config, compiled,
                                              origin + HORIZON, baseline_cache)
        rounds.append({
            "round": label,
            "chosen": trace.chosen_candidate_id,
            "compilation": trace.compilation_status,
            "prepared_status": result.status.name,
            "scope_legal": scope_legal,
            "support_gain": (round(float(support_gain), 6)
                             if support_gain is not None else None),
            "delayed_gain": (round(float(delayed_gain), 6)
                             if delayed_gain is not None else None),
            "instruction_ref1": "Reference 1" in (
                wiring.DeterministicStrategyBackend.extract_instruction(
                    backend.requests[-1].messages) if backend.requests else ""),
        })
        print(f"[{label}] chosen={trace.chosen_candidate_id} "
              f"compile={trace.compilation_status} "
              f"scope_legal={scope_legal} "
              f"support={round(support_gain, 5) if support_gain is not None else None} "
              f"delayed={round(delayed_gain, 5) if delayed_gain is not None else None}")
        if idx == 0 and support_gain is not None:
            # 真实入口行动 → 写 Episode → delayed（不拼接，独立决策点重执行）
            target_episode = write_target_episode(
                domain=TARGET_DOMAIN, op=op,
                program_steps=[{"op": op, "params": dict(params)}],
                support_gain=float(support_gain), delayed_gain=None,
                support_context=resolver.window_context(values, origin, PERIOD))
            if delayed_gain is not None:
                target_episode = update_delayed_status(
                    target_episode, float(delayed_gain),
                    delayed_context=resolver.window_context(
                        values, origin + HORIZON, PERIOD))
            memory.append(target_episode)

    r1 = rounds[0]
    r2 = rounds[1]
    checks: dict[str, bool] = {
        "legal_non_identity_action": (
            r1["chosen"] != "identity"
            and r1["compilation"] in ("ok", "compiled")),
        "scope_verifier_035": bool(r1["scope_legal"]),
        "support_positive": r1["support_gain"] is not None
        and r1["support_gain"] >= MATERIAL,
        "delayed_positive": r1["delayed_gain"] is not None
        and r1["delayed_gain"] >= MATERIAL,
        "memory_updated": len(memory) == 2,
        "next_round_retrieves": bool(r2["instruction_ref1"])
        and r2["chosen"] == f"cand_{op}",
    }
    all_pass = all(checks.values())
    verdict = ("SCOPE_ALIGNED_MECHANICAL_CLOSED_LOOP_PASS" if all_pass
               else "SCOPE_ALIGNED_MECHANICAL_CLOSED_LOOP_PARTIAL")
    return {
        "control": {"operator": op, "origin": origin,
                    "support_gain": float(c_support),
                    "delayed_gain": float(c_delayed)},
        "rounds": rounds,
        "checks": checks,
        "verdict": verdict,
    }


def main() -> int:
    root = PROJECT_ROOT
    config = dict(v6.DATASET_CONFIGS[TARGET_DOMAIN])
    roster, values = v6._fixed_roster(root, config)
    max_fraction, preserve_outside = _h0_limits(root)
    print(f"== H0 limits: max_modified_fraction={max_fraction} "
          f"preserve_outside={preserve_outside}")

    print("\n== Part A: 执行器逐位一致性（run_pipeline vs _apply_program）")
    part_a = part_a_bitwise_consistency(roster, values, config)
    for check in part_a["checks"]:
        print(f"   {check['operator']:20s} identical={check['identical_bitwise']} "
              f"max_abs_diff={check['max_abs_diff']}")
    assert part_a["all_identical"], "执行器输出不一致——必须先行调查"
    print(f"   -> 全部逐位相同（同一执行器确认）")

    print("\n== Part B: 规范 Scope headroom 扫描（窗口 verifier 0.35 + v6._evaluate）")
    best, rows, summary = _scope_scan(root, roster, values, config, CHAIN_ORIGINS,
                                      max_fraction, preserve_outside)
    for row in rows:
        if row["scope_selectable"] and row["support_gain"] is not None:
            print(f"   {row['operator']:24s} @{row['origin']} "
                  f"support={row['support_gain']} delayed={row['delayed_gain']}")
    print(f"   scope_selectable={summary['scope_selectable_operators']}")
    print(f"   scope_rejected={summary['scope_rejected_operators']}")
    if best is None:
        print("== NO_SCOPE_POSITIVE_CONTROL：规范语义下仍无双正——诚实记录，不跑闭环")
        verdict = "NO_SCOPE_POSITIVE_CONTROL"
        part_c: dict[str, Any] = {"verdict": verdict}
    else:
        print(f"== 规范语义正控: {best['operator']} @{best['origin']} "
              f"support={best['support_gain']} delayed={best['delayed_gain']}")
        print("\n== Part C: 暴露正控闭环（不拼接、决策点重执行冻结 Workflow）")
        part_c = part_c_closed_loop(root, roster, values, config, best,
                                    max_fraction, preserve_outside)
        print(f"== checks: {part_c['checks']}")
        print(f"== verdict: {part_c['verdict']}")
        print("== 口径：暴露正控的机械闭环（不宣称 A5 跨域效果、不宣称 Agent 选择质量）")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({
            "experiment_id": "v1-scope-alignment",
            "domain": TARGET_DOMAIN,
            "h0_max_modified_fraction": max_fraction,
            "part_a_bitwise_consistency": part_a,
            "part_b_scan": {"rows": rows, "summary": summary},
            "part_c_closed_loop": part_c,
            "llm_api_call_count": 0,
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\n== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
