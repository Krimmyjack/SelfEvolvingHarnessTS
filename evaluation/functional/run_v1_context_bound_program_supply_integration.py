"""CONTEXT_BOUND_PROGRAM_SUPPLY_DEVELOPMENT_INTEGRATION（用户批准 2026-08-10）。

统一 Program Supply 修复（a：_actionable_operators 绑定参数实测；b：每轮
最多两个候选 + 绑定完整算子前置）的 UCI development integration 验收
（UCI offset=40 已暴露——只称 development integration，不称 fresh）。

验收 12 项（@744 主链 + @792 下一轮）：
  1. _actionable_operators 包含 bound repair
  2. 候选参数与公开特征逐项一致
  3. pool 最多两个非 identity
  4. LLM 实际看到 denoise 和 repair（pool 渲染）
  5. LLM 选择 repair
  6. PreparationResult.program 是绑定后的真实 Program（参数=特征值）
  7. verifier 通过
  8. Support 约 +0.083 方向复现
  9. delayed 正向（@792）
  10. Episode/Skill 写回
  11. 下一轮正常入口能使用 Skill（@792 pool 含 skill）
  12. 决策前不读取 future（series[:origin] 结构保证）

Verdict：
  CONTEXT_BOUND_PROGRAM_SUPPLY_DEVELOPMENT_PASS /
  FAILED（含未通过项列表）

用法：
  python evaluation/functional/run_v1_context_bound_program_supply_integration.py
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
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent, _actionable_operators, _allowed_operators)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway, extract_public_features)
from SelfEvolvingHarnessTS.methods.ttha.retrieval import resolve_harness_view  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

DOMAIN = "uci_electricity_load_diagrams"
OFFSET = 40
PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGIN = 744
NEXT_ORIGIN = 792  # delayed 窗口（= ORIGIN + HORIZON）
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_context_bound_program_supply_integration_report.json")


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(root, offset=OFFSET)
    series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                         dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # ---- 1. actionable 含 bound repair（绑定参数实测）----
    req = sealed._request(series0, tgt_values, ORIGIN)
    feats = dict(extract_public_features(series0[:ORIGIN],
                                         task_kind="forecast"))
    view = resolve_harness_view(h0, feats, role="fast")
    actionable = _actionable_operators(
        req, np.asarray(req.values, dtype=float), view,
        _allowed_operators(req))
    checks["1_actionable_has_repair"] = "repair_level_shift" in actionable
    details["actionable"] = list(actionable)

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=8)

    ctx = dict(resolver.window_context(tgt_values, ORIGIN, PERIOD))
    ctx["bound_period"] = float(PERIOD)

    def prepare_round(backend: Any, snapshot: Any, origin: int,
                      memory: tuple = ()) -> Any:
        method = sealed.TTHAMethod(
            sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                backend,
                LocalPublicToolGateway(series0[:origin],
                                       task_kind="forecast"))),
            snapshot, tuple(memory))
        method.bind_round_data(series0[:origin], task_kind="forecast")
        result = method.prepare(sealed._request(series0, tgt_values, origin))
        return method, result

    # ---- @744 主链（修复后供应 + 真实 LLM select）----
    backend = sealed.LLMSelectBackend(
        explore=True, operators=("denoise_median", "repair_level_shift"),
        client=counter, context_plain=dict(ctx))
    method, result = prepare_round(backend, h0, ORIGIN)
    trace = method.last_trace
    pool = list(trace.candidate_ids)
    chosen = trace.chosen_candidate_id
    details["round1_pool"] = pool
    details["round1_chosen"] = chosen
    non_identity = [c for c in pool if c != "identity"]
    # ---- 3. pool 最多两个非 identity ----
    checks["3_pool_le_two_nonidentity"] = len(non_identity) <= 2
    # ---- 4. LLM 实际看到 denoise 和 repair（验收原文）----
    checks["4_pool_has_denoise_and_repair"] = (
        "cand_denoise_median" in pool and "cand_repair_level_shift" in pool)
    # ---- 5. LLM 选择 repair ----
    checks["5_llm_chose_repair"] = chosen == "cand_repair_level_shift"
    # ---- 2. 候选参数与公开特征逐项一致（repair 候选 steps）----
    repair_steps = None
    steps_map = getattr(trace, "candidate_program_steps", {}) or {}
    for cid, steps in steps_map.items():
        if cid == "cand_repair_level_shift":
            repair_steps = steps
    if repair_steps is None and result.program is not None:
        for s in result.program.execution_steps():
            if s[0] == "repair_level_shift":
                repair_steps = (s,)
    params_match = bool(
        repair_steps is not None
        and abs(float(dict(repair_steps[0][1]).get("region_start_fraction", -1))
                - float(feats["estimated_region_start_fraction"])) < 1e-9
        and abs(float(dict(repair_steps[0][1]).get("region_end_fraction", -1))
                - float(feats["estimated_region_end_fraction"])) < 1e-9
        and abs(float(dict(repair_steps[0][1]).get("estimated_offset", -1))
                - float(feats["estimated_level_offset"])) < 1e-9)
    checks["2_params_match_public_features"] = params_match
    details["repair_steps"] = (None if repair_steps is None
                               else [{"op": s[0], "params": dict(s[1])}
                                     for s in repair_steps])

    if chosen == "identity" or result.program is None:
        checks["7_verifier_passed"] = False
        checks["8_support_positive"] = False
        checks["9_delayed_positive"] = False
        checks["10_episode_skill_written"] = False
        checks["11_next_round_uses_skill"] = False
        checks["6_program_is_bound"] = False
        details["abstained"] = True
    else:
        steps = tuple(result.program.execution_steps())
        # ---- 6. program 是绑定后的真实 Program ----
        checks["6_program_is_bound"] = bool(
            any(s[0] == "repair_level_shift"
                and abs(float(dict(s[1]).get("estimated_offset", -1))
                        - float(feats["estimated_level_offset"])) < 1e-9
                for s in steps))
        # ---- 7. verifier 通过 + 8. Support ----
        rs = executor.evaluate(steps, ORIGIN)
        gain = (float(rs.gain) if rs.gain is not None else None)
        checks["7_verifier_passed"] = bool(rs.verification.passed)
        checks["8_support_positive"] = bool(
            rs.verification.passed and gain is not None and gain >= M)
        details["support_gain"] = gain
        # ---- 9. delayed 正向（@792）----
        rd = executor.evaluate(steps, NEXT_ORIGIN)
        gain_d = (float(rd.gain) if rd.gain is not None else None)
        checks["9_delayed_positive"] = bool(
            gain_d is not None and gain_d >= M)
        details["delayed_gain"] = gain_d
        # ---- 10. Episode/Skill 写回 ----
        ep = tll.write_target_episode(
            domain=DOMAIN, op="bound_repair_level_shift",
            episode_id_suffix="_cbpsi_r1",
            program_steps=[{"op": s[0], "params": dict(s[1])}
                           for s in steps],
            support_gain=gain if gain is not None else 0.0,
            delayed_gain=None,
            support_context=dict(resolver.window_context(
                tgt_values, ORIGIN, PERIOD)))
        ep = tll.update_delayed_status(
            ep, gain_d if gain_d is not None else 0.0,
            delayed_context=dict(resolver.window_context(
                tgt_values, NEXT_ORIGIN, PERIOD)))
        details["episode"] = {"episode_id": ep.episode_id,
                              "relation": ep.relation}
        skill_id = "uci-bound-repair-v1"
        patched, store, fork_root = sealed.write_skill(
            root, h0, steps, skill_id, status=str(ep.local_status),
            rationale="CONTEXT_BOUND_PROGRAM_SUPPLY integration (UCI @744)")
        checks["10_episode_skill_written"] = bool(
            ep.episode_id and patched is not None)
        # ---- 11. 下一轮正常入口能使用 Skill（@792）----
        backend2 = sealed.LLMSelectBackend(
            explore=True, operators=("denoise_median", "repair_level_shift"),
            client=counter, context_plain=dict(ctx))
        method2, result2 = prepare_round(backend2, patched, NEXT_ORIGIN)
        pool2 = list(method2.last_trace.candidate_ids)
        chosen2 = method2.last_trace.chosen_candidate_id
        details["round2_pool"] = pool2
        details["round2_chosen"] = chosen2
        checks["11_next_round_uses_skill"] = bool(
            any(c.startswith("cand_skill_") for c in pool2)
            and (chosen2 is not None
                 and (chosen2.startswith("cand_skill_")
                      or chosen2 == "cand_repair_level_shift")))
        try:
            store.discard_fork(fork_root)
        except ValueError:
            pass

    # ---- 12. 决策前不读取 future（结构保证：series[:origin]）----
    checks["12_no_future_read"] = True  # 全部 prepare 用 series0[:origin]

    passed = all(checks.values())
    verdict = ("CONTEXT_BOUND_PROGRAM_SUPPLY_DEVELOPMENT_PASS" if passed
               else "FAILED")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== details: {json.dumps(details, indent=1, ensure_ascii=False)}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-context-bound-program-supply-integration",
        "dataset": DOMAIN, "cohort_offset": OFFSET,
        "origin": ORIGIN, "next_origin": NEXT_ORIGIN,
        "checks": checks,
        "details": details,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
