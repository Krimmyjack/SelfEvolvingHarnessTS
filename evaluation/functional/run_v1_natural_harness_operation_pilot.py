"""NATURAL_HARNESS_OPERATION_PILOT（用户裁决 2026-08-10，§7 五十）。

冻结已通过的 Slow 更新链（REAL_SLOW_AGENT_PATCH_PASS 机制），在全新
virgin cohort 上跑连续 3 个真实在线轮次：

  轮 1 origin=648 delayed=696
  轮 2 origin=744 delayed=792
  轮 3 origin=840 delayed=888
  采用轮 origin=936（delayed 984 ≤ 1024）

规则（全部冻结，运行中不修复）：
  - 新 virgin cohort（traffic offset=120）；
  - 每轮 Support 预算 ≤2（两步组合探测，自然顺序预注册，不换 pair）；
  - 正常写 Episode（tll）与 delayed（显式 episode_id）；
  - Memory 跨轮累积（正常在线语义）；
  - 轨迹结束后只定位第一个自然 fault（两步 gain < −M）；
  - 仅当自然失败同时具有：明确 first fault + 可执行 Surface 动作 +
    替代 headroom（max(A,B)>=M 且 max−gain_AB>=M）→ 才调用已验证的
    真实 Slow Agent 更新链（propose_edit → apply_to_fork → replay →
    delayed → 采用轮实际采用 → remove-skill 对照）。

Verdict（预注册）：
  NATURAL_PILOT_SLOW_UPDATE_PASS / PILOT_NO_NATURAL_FAULT /
  PILOT_TRIGGER_CONDITIONS_UNMET / PILOT_SLOW_CHAIN_FAILED / INCONCLUSIVE

用法：
  python evaluation/functional/run_v1_natural_harness_operation_pilot.py
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
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

DOMAIN = "monash:traffic_hourly"
OFFSET = 120
PERIOD = 24
HORIZON = 48
MATERIAL = resolver.MATERIAL_THRESHOLD  # 0.005
ROUNDS = [(648, 696), (744, 792), (840, 888)]  # (origin, delayed)
ADOPT_ORIGIN = 936
REPORT_OUT_REL = Path("artifacts/functional/e2/w1_natural_harness_operation_pilot_report.json")


def _combos(ops: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for i, a in enumerate(ops) for b in ops[i + 1:]]


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain(DOMAIN)
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=OFFSET)
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    executor = ScopeExecutor(tgt_roster, tgt_values, config,
                             evaluate_fn=sealed.v6._evaluate)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    obs0 = dict(resolver.window_context(tgt_values, ROUNDS[0][0], PERIOD))
    obs0["bound_period"] = float(PERIOD)
    ops = sealed._actionable_ops(root, tgt_series0, ROUNDS[0][0], obs0)
    combos = _combos(ops)
    print(f"== pilot: domain={DOMAIN} offset={OFFSET} "
          f"rounds={ROUNDS} adopt@{ADOPT_ORIGIN}")
    print(f"== actionable n={len(ops)} combos n={len(combos)}")

    # ---- 3 轮在线轨迹（探测计划固定，预算 ≤2/轮，Memory 跨轮累积）----
    memory: list[Any] = []
    probes_log: list[dict[str, Any]] = []
    combo_idx = 0
    first_fault: dict[str, Any] | None = None
    for round_i, (origin, delayed_origin) in enumerate(ROUNDS, start=1):
        round_log: list[dict[str, Any]] = []
        for probe in range(2):  # 预算 ≤2
            if combo_idx >= len(combos):
                break
            a, b = combos[combo_idx]
            combo_idx += 1
            steps = ((a, dict(wiring.contract_params(a, PERIOD))),
                     (b, dict(wiring.contract_params(b, PERIOD))))
            r = executor.evaluate(steps, origin)
            gain = (float(r.gain) if r.gain is not None else None)
            entry = {"round": round_i, "origin": origin, "probe": probe + 1,
                     "a": a, "b": b, "gain": gain,
                     "passed": r.verification.passed}
            if r.verification.passed and gain is not None:
                # 正常写 Episode（两步 Workflow）
                ep = tll.write_target_episode(
                    domain=DOMAIN, op=f"{a}_{b}",
                    episode_id_suffix=f"_pilot_r{round_i}p{probe + 1}",
                    program_steps=[{"op": a, "params": dict(
                        wiring.contract_params(a, PERIOD))},
                        {"op": b, "params": dict(
                            wiring.contract_params(b, PERIOD))}],
                    support_gain=gain, delayed_gain=None,
                    support_context=resolver.window_context(
                        tgt_values, origin, PERIOD))
                memory.append(ep)
                entry["episode_id"] = ep.episode_id
                entry["relation"] = ep.relation
                # delayed 打开（同一冻结 Workflow 的后续窗口；原位更新）
                rd = executor.evaluate(steps, origin + HORIZON)
                gain_d = (float(rd.gain) if rd.gain is not None else None)
                for i_e, e in enumerate(memory):
                    if getattr(e, "episode_id", "") == ep.episode_id:
                        memory[i_e] = tll.update_delayed_status(
                            e, gain_d if gain_d is not None else 0.0,
                            delayed_context=resolver.window_context(
                                tgt_values, origin + HORIZON, PERIOD))
                        entry["delayed_gain"] = gain_d
                        entry["delayed_relation"] = memory[i_e].relation
                        break
            # 第一个自然 fault（两步 gain < −M）——记录但不中断（轨迹跑完）
            if first_fault is None and r.verification.passed \
                    and gain is not None and gain < -MATERIAL:
                first_fault = {"a": a, "b": b, "gain_AB": gain,
                               "origin": origin, "delayed_origin":
                                   delayed_origin,
                               "round": round_i, "probe": probe + 1,
                               "steps": steps}
            round_log.append(entry)
        probes_log.append({"round": round_i, "origin": origin,
                           "probes": round_log})
        print(f"== round {round_i} @{origin}: "
              f"{[(p['a'], p['b'], round(p['gain'], 4) if p['gain'] is not None else None) for p in round_log]}")

    print(f"== first natural fault: "
          f"{first_fault and {k: v for k, v in first_fault.items() if k != 'steps'}}")

    checks: dict[str, Any] = {}
    if first_fault is None:
        verdict = "PILOT_NO_NATURAL_FAULT"
    else:
        a, b = first_fault["a"], first_fault["b"]
        steps_a = ((a, dict(wiring.contract_params(a, PERIOD))),)
        steps_b = ((b, dict(wiring.contract_params(b, PERIOD))),)
        ra = executor.evaluate(steps_a, first_fault["origin"])
        rb = executor.evaluate(steps_b, first_fault["origin"])
        ga = (float(ra.gain) if ra.gain is not None else None)
        gb = (float(rb.gain) if rb.gain is not None else None)
        best = max(ga, gb) if (ga is not None and gb is not None) else None
        headroom_ok = bool(best is not None and best >= MATERIAL
                           and (best - first_fault["gain_AB"]) >= MATERIAL)
        checks["counterfactuals"] = {"identity": 0.0, "A_only": ga,
                                     "B_only": gb,
                                     "A_to_B": first_fault["gain_AB"]}
        checks["headroom_ok"] = headroom_ok
        print(f"== counterfactuals: {checks['counterfactuals']} "
              f"headroom={headroom_ok}")
        if not headroom_ok:
            verdict = "PILOT_TRIGGER_CONDITIONS_UNMET"
        else:
            # ---- 真实 Slow Agent 链（复用已验证机制；Pilot 域反事实）----
            verdict = _run_slow_chain(root, h0, executor, tgt_series0,
                                      tgt_values, first_fault, checks)
    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-natural-harness-operation-pilot",
        "dataset": DOMAIN, "cohort_offset": OFFSET,
        "rounds": ROUNDS, "adopt_origin": ADOPT_ORIGIN,
        "actionable_count": len(ops),
        "probes": probes_log,
        "first_fault": (None if first_fault is None else
                        {k: v for k, v in first_fault.items()
                         if k != "steps"}),
        "memory_relations": [getattr(e, "relation", "?") for e in memory],
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": (checks.get("slow_llm_calls", 0)),
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


def _run_slow_chain(root: Path, h0: Any, executor: ScopeExecutor,
                    tgt_series0: np.ndarray, tgt_values: Mapping[str, Any],
                    fault: Mapping[str, Any], checks: dict[str, Any]) -> str:
    """已验证的真实 Slow Agent 更新链（§7 四十九机制复用）：propose_edit
    → 确定性契约修复 → EditController.apply_to_fork → replay → delayed →
    采用轮实际采用 → remove-skill 对照。"""
    import dataclasses
    import run_v1_slow_path_smoke as smoke
    import run_v1_real_slow_agent_positive_control as pc
    from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
    from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
        EditController, SurfaceRegistry, FaultRouter)
    from SelfEvolvingHarnessTS.runtime.agent_backend import AgictoChatCompletionsBackend  # noqa: E402

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        return "INCONCLUSIVE"
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    core = TTHAAgentCore(
        backend,
        LocalPublicToolGateway(tgt_series0[:ADOPT_ORIGIN],
                               task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)
    slow = TTHASlowAgent(core)

    # FailurePatternCard（Pilot 域反事实数值 + 冻结 Workflow）
    pa = dict(wiring.contract_params(fault["a"], PERIOD))
    pb = dict(wiring.contract_params(fault["b"], PERIOD))
    card = {
        "pattern_id": f"{DOMAIN}-pilot-counterfactual",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": [{"op": fault["a"], "params": pa},
                               {"op": fault["b"], "params": pb}],
                     "scope": "training_windows_only",
                     "evaluator": "v6._evaluate"},
        "counterfactual_support": checks["counterfactuals"],
        "facts": {"evaluator_scope": "training_windows_only"},
        "instruction": pc.build_positive_control_card(executor)["instruction"]
        .replace("impute_ssm", fault["a"]).replace("outlier_iqr", fault["b"]),
    }
    manifest = None
    try:
        manifest = slow.propose_edit(card, pc.SURFACE_CATALOG, h0,
                                     manifest_preflight=lambda m: None,
                                     allowed_operator_contracts=(),
                                     task_context=None)
    except RuntimeError as exc:
        checks["slow_budget_error"] = str(exc)
    checks["slow_llm_calls"] = counter.calls
    if manifest is None:
        return "PILOT_SLOW_CHAIN_FAILED"  # 弃权（自然失败上未提议）

    preflight = pc.structural_preflight(manifest)
    if preflight["preflight"] != "ACCEPTED":
        return "PILOT_SLOW_CHAIN_FAILED"
    frozen = preflight["frozen_program"]
    steps = tuple((str(s["op"]), dict(s["params"])) for s in frozen)
    checks["slow_frozen_program"] = frozen
    reg = SurfaceRegistry()
    resolved = reg.resolve(preflight["target_surface_id"])
    snapshot_deps = dict(h0.dependency_shas)
    declared_dep = {key: snapshot_deps[key]
                    for key in resolved.definition.required_dependency_keys
                    if key in snapshot_deps}
    manifest_applied = dataclasses.replace(
        manifest, target_surface_id=preflight["target_surface_id"],
        dependency_precondition_shas=declared_dep)
    store = SnapshotStore(root)
    parent = store.materialize(h0)
    controller = EditController(store, surfaces=reg, router=FaultRouter())
    try:
        receipt = controller.apply_to_fork(
            parent, manifest_applied, confirmed_cause="SKILL_LIBRARY_GAP")
    except Exception as exc:
        checks["apply_exception"] = f"{type(exc).__name__}: {exc}"
        return "PILOT_SLOW_CHAIN_FAILED"

    # replay（fault origin）→ delayed
    rp = executor.evaluate(steps, fault["origin"])
    gain_p = (float(rp.gain) if rp.gain is not None else None)
    checks["slow_replay_support_positive"] = bool(
        rp.verification.passed and gain_p is not None and gain_p >= MATERIAL)
    rd = executor.evaluate(steps, fault["delayed_origin"])
    gain_d = (float(rd.gain) if rd.gain is not None else None)
    checks["slow_delayed_no_flip"] = bool(
        gain_d is not None and gain_d >= -MATERIAL)
    if not checks["slow_replay_support_positive"]:
        return "PILOT_SLOW_CHAIN_FAILED"
    if not checks["slow_delayed_no_flip"]:
        return "PILOT_SLOW_CHAIN_FAILED"

    # 采用轮（936）实际采用 + remove-skill 对照
    skill_cand = f"cand_skill_{preflight['skill_id']}"
    ops_all = tuple(o for o in (
        "denoise_median", "hampel_filter", "impute_ar", "impute_ema",
        "impute_fft", "impute_linear", "impute_ssm", "outlier_iqr",
        "outlier_mad", "period_complete", "period_median_complete",
        "repair_level_shift", "resample_uniform", "winsorize"))
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=ops_all),
            LocalPublicToolGateway(tgt_series0[:ADOPT_ORIGIN],
                                   task_kind="forecast"))),
        receipt.candidate_snapshot.snapshot, ())
    method.bind_round_data(tgt_series0[:ADOPT_ORIGIN], task_kind="forecast")
    obs = dict(resolver.window_context(tgt_values, ADOPT_ORIGIN, PERIOD))
    obs["bound_period"] = float(PERIOD)
    r2 = method.prepare(sealed._request(tgt_series0, tgt_values,
                                        ADOPT_ORIGIN))
    chosen = method.last_trace.chosen_candidate_id
    adopted = chosen == skill_cand
    r2_gain = None
    if r2.program is not None:
        cs = tuple((op, dict(pr)) for op, pr in r2.program.execution_steps())
        rr2 = executor.evaluate(cs, ADOPT_ORIGIN)
        r2_gain = (float(rr2.gain) if rr2.gain is not None else None)
    checks["slow_adopt_round_chosen"] = chosen
    checks["slow_actual_adoption"] = bool(adopted)
    checks["slow_adopt_executed"] = r2_gain is not None

    method_ctrl = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=ops_all),
            LocalPublicToolGateway(tgt_series0[:ADOPT_ORIGIN],
                                   task_kind="forecast"))),
        h0, ())
    method_ctrl.bind_round_data(tgt_series0[:ADOPT_ORIGIN],
                                task_kind="forecast")
    method_ctrl.prepare(sealed._request(tgt_series0, tgt_values,
                                        ADOPT_ORIGIN))
    chosen_ctrl = method_ctrl.last_trace.chosen_candidate_id
    checks["slow_removal_changes_action"] = bool(chosen_ctrl != skill_cand)

    if adopted and checks["slow_removal_changes_action"]:
        return "NATURAL_PILOT_SLOW_UPDATE_PASS"
    return "PILOT_SLOW_CHAIN_FAILED"


if __name__ == "__main__":
    raise SystemExit(main())
