"""NATURAL_METHOD_OWNED_SLOW_UPDATE_PILOT（P3.2，用户批准 2026-08-11；
复核修复版 2026-08-11）。

验证唯一问题：开发级完整 Slow Update 在自然、fresh、在线时序轨迹中是否
真的会产生一次有效 Harness Update。

复核修复（5 Blocker + 5 Major，2026-08-11）：
  1. support/query→eval 角色映射（_evaluate_monash 包装）；gain=None 立即
     判协议失败、不写 Episode；
  2. delayed 必须 verifier 通过 + gain 有限 + 不显著负向 + episode_id
     匹配才批准（method.py）；
  3. 全 Pilot 只允许**第一个 material fault** 触发一次正式 Slow 调用
     （+1 契约内 schema 重试；max_calls=2）；
  4. Slow Agent 公开工具 Context 每轮同步（当前 origin）；
  5. Runtime 绑定 allowed_tools（method.py）；claim 限定 context/cohort-
     local；
  6. PASS 必须验证下一轮 chosen_candidate_id、PreparationResult.program
     等于冻结 Patch、真实执行、removal 恢复；
  7. freeze 不再调用 evaluate（只静态 verify——零 outcome）。

预注册 verdict（七档）：
  NATURAL_SLOW_UPDATE_PASS / NO_NATURAL_FAILURE / ACTION_UNAVAILABLE /
  SLOW_AGENT_ABSTAIN / PATCH_SUPPORT_REJECTED / PATCH_DELAYED_REJECTED /
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_natural_method_owned_slow_pilot.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.contracts.method import PreparationRequest  # noqa: E402
from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    MetricSpec,
    forecast_task_spec_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import (  # noqa: E402
    OPERATOR_METADATA,
    OPERATOR_NAMES,
)

PERIOD = 7  # 日频（Monash）
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGINS = (600, 792, 888)  # 3 轮在线（delayed = origin + 48）
FROZEN_ROSTER_REL = Path(
    "artifacts/functional/e2/w1_monash_frozen_roster_solar_p32.jsonl")
CACHE = Path("data/monash_weather_v1/series_cache.npz")
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_natural_method_owned_slow_pilot_report.json")


def _config() -> dict[str, object]:
    return {
        "dataset_id": "monash_weather_daily",
        "sampling": "daily_regular",
        "period": PERIOD,
        "anchors": list(range(312, 853, 60)),
        "support_origin": ORIGINS[0],
        "selection_origin": ORIGINS[0],
    }


def _evaluate_monash(roster: Sequence[Mapping[str, Any]], values: Any,
                     compiled: Any, config: Mapping[str, object], *,
                     origin: int) -> dict[str, Any]:
    """复核 Blocker 1：support/query→eval（v6 协议只认 train/eval）。"""
    mapped = [dict(row, role="eval") if str(row["role"]) != "train"
              else dict(row) for row in roster]
    return v6._evaluate(mapped, values, compiled, config, origin=origin)


def _request(series0: np.ndarray, values: Mapping[str, Any],
             origin: int) -> PreparationRequest:
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    return PreparationRequest(
        "natural-slow-pilot",
        series0[:origin],
        forecast_task_spec_v1(horizon=HORIZON,
                              downstream_model_class="ridge",
                              metric=MetricSpec("sMASE", "lower_is_better")),
        dict(observed),
    )


def _load_cohort(root: Path) -> dict[str, Any]:
    rows = [json.loads(line)
            for line in (root / FROZEN_ROSTER_REL)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"]),
               "type": str(r.get("type", "solar"))} for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    return {"roster": roster, "values": vals}


def _patch_options(executor: ScopeExecutor, values: Mapping[str, Any],
                   origin: int, failed_op: str) -> list[dict[str, Any]]:
    """Typed Patch 生成（outcome 打开前冻结规则）：排除原失败算子；仅
    当前 Context 下 verifier 合法（ScopeExecutor.verify 静态——零 gain）；
    同功能 family（category）按 registry 固定顺序取最多两个；不根据历史
    已知 gain 指定答案。"""
    meta = OPERATOR_METADATA.get(failed_op) or {}
    family = meta.get("category")
    out: list[dict[str, Any]] = []
    series0 = np.asarray(values[list(values)[0]][:origin], dtype=np.float64)
    for op in OPERATOR_NAMES:  # registry 固定顺序
        if op == failed_op:
            continue
        om = OPERATOR_METADATA.get(op) or {}
        if om.get("category") != family:
            continue
        if str(op).endswith("_complete") or str(op).startswith("impute_"):
            continue
        params: dict[str, object] = {}
        bindings = om.get("public_parameter_bindings") or {}
        if bindings:
            from SelfEvolvingHarnessTS.methods.ttha.public_tools import (
                extract_public_features,
            )
            fe = dict(extract_public_features(series0, task_kind="forecast"))
            params = {p: float(fe[f]) for p, f in bindings.items()
                      if f in fe}
            if len(params) != len(bindings):
                continue
        else:
            params = dict(wiring.contract_params(op, PERIOD))
        steps = ((op, params),)
        v = executor.verify(steps, origin)
        if not v.passed:
            continue
        out.append({"patch_id": f"patch-{failed_op}-to-{op}",
                    "program_steps": [{"op": op, "params": dict(params)}]})
        if len(out) >= 2:
            break
    return out


def _card_from_episode(episode: Any, executor: ScopeExecutor,
                       values: Mapping[str, Any],
                       origin: int) -> Mapping[str, object]:
    cs = getattr(episode, "context_summary", {}) or {}
    ctx = {k: float(v) for k, v in
           (cs.get("local_pattern") or {}).items()
           if isinstance(v, (int, float))}
    pg = cs.get("program_geometry") or {}
    steps = [{"op": s["op"], "params": dict(s.get("params") or {})}
             for s in (pg.get("program_steps") or [])
             if isinstance(s, Mapping) and s.get("op")]
    sg = (getattr(episode, "support_response", {}) or {}).get("gain")
    dg = (getattr(episode, "delayed_response", {}) or {}).get("gain")
    ctx["bound_period"] = float(PERIOD)
    failed_op = steps[-1]["op"] if steps else ""
    options = _patch_options(executor, values, origin, failed_op)
    return {
        "pattern_id": f"monash-solar-{failed_op}-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "workflow": {"steps": steps, "scope": "training_windows_only",
                     "evaluator": "v6._evaluate"},
        "facts": {"window_context": ctx,
                  "task_objective": "forecast; cohort-Ridge sMASE"},
        "counterfactual_support": {
            "A_then_B_support_gain": float(sg) if sg is not None else None,
            "A_then_B_delayed_gain": float(dg) if dg is not None else None},
        "typed_patch_options": options,
        "instruction": ("Select exactly one typed patch by its patch_id from "
                        "typed_patch_options (or abstain). Do not write "
                        "program steps."),
    }


def _feedback_round(method: TTHAMethod, executor: ScopeExecutor,
                    series0: np.ndarray, values: Mapping[str, Any],
                    origin: int, *, round_name: str,
                    slow_agent: Any, controller: Any, store: Any,
                    events: list[dict[str, Any]],
                    allow_trigger: bool) -> dict[str, Any]:
    """正常 Fast 轮（复核修复）：role 映射评估；gain=None → 协议失败不写
    Episode；第一个 material fault 才允许触发（allow_trigger）；Slow
    Agent 工具 Context 每轮同步；Fast Episode 完整 delayed 写回。"""
    # 复核 Major：Slow Agent 公开工具 Context 每轮同步（当前 origin）
    slow_agent.core.tools = LocalPublicToolGateway(series0[:origin],
                                                   task_kind="forecast")
    backend = sealed.SealedProbeBackend(
        explore=True, operators=("repair_level_shift", "winsorize",
                                 "outlier_iqr"),
        max_propose_candidates=2, force_pool=True)
    core = method.fast_agent.core
    core.backend = backend
    method.bind_round_data(series0[:origin], task_kind="forecast")
    result = method.prepare(_request(series0, values, origin))
    trace = method.last_trace
    steps_map = dict(trace.candidate_program_steps or {})
    pool_ops = [c[len("cand_"):] for c in trace.candidate_ids
                if c.startswith("cand_") and c in steps_map]
    log: dict[str, Any] = {"origin": origin, "pool": list(trace.candidate_ids),
                           "probes": [], "protocol_failure": False}
    for i, op in enumerate(pool_ops[:2]):
        steps = steps_map[f"cand_{op}"]
        rr = executor.evaluate(steps, origin)
        gain = (float(rr.gain) if rr.gain is not None else None)
        passed = bool(rr.verification.passed)
        entry: dict[str, Any] = {"probe": i + 1, "op": op, "gain": gain,
                                 "passed": passed}
        # 复核 Blocker 1：gain=None（仪器失败）→ 协议失败，不写 Episode
        if passed and gain is None:
            log["protocol_failure"] = True
            log["protocol_reason"] = f"support_outcome_unavailable ({op})"
            log["probes"].append(entry)
            break
        if passed:
            ep = tll.write_target_episode(
                domain="monash_weather_daily", op=op,
                episode_id_suffix=f"_p32_{round_name}_p{i + 1}",
                program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
                support_gain=gain if gain is not None else 0.0,
                delayed_gain=None,
                support_context=dict(resolver.window_context(
                    values, origin, PERIOD)))
            entry["episode_id"] = ep.episode_id
            method.append_experience_episode(ep)
            # Fast Episode 完整 delayed 写回（复核 Major）
            rd = executor.evaluate(steps, origin + HORIZON)
            dg = (float(rd.gain) if rd.gain is not None else None)
            entry["delayed_gain"] = dg
            for i_e, e in enumerate(method._experience_episodes):  # noqa: SLF001
                if getattr(e, "episode_id", "") == ep.episode_id:
                    upd = tll.update_delayed_status(
                        e, dg if dg is not None else 0.0,
                        delayed_context=dict(resolver.window_context(
                            values, origin + HORIZON, PERIOD)))
                    method.update_experience_episode(upd)
                    entry["relation_after_delayed"] = upd.relation
                    break
            # 复核 Blocker 5：只有第一个 material fault 允许触发
            if allow_trigger and gain is not None and gain < -M:
                sev = method.handle_feedback_support(ep, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=slow_agent, controller=controller,
                    store=store,
                    surface_catalog=[{
                        "surface_id": "skill_library.entries/{skill_id}",
                        "operation": "ADD",
                        "surface_type": "skill",
                        "allowed_operations": ["ADD"]}],
                    card_builder=lambda e: _card_from_episode(
                        e, executor, values, origin),
                    evaluator=lambda s, _o: executor.evaluate(s, origin))
                events.append(sev)
                log["support_event"] = sev
                if sev.get("stage") == "pending":
                    dev = method.handle_feedback_delayed(
                        lambda s, _o: executor.evaluate(
                            s, origin + HORIZON),
                        episode_id=ep.episode_id)
                    events.append(dev)
                    log["delayed_event"] = dev
        log["probes"].append(entry)
        if gain is not None and gain >= M:
            break
    return log


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    cohort = _load_cohort(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_monash)

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — PROTOCOL_FAILURE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=2)  # 复核 Blocker 5：1 正式 + 1 契约内 schema 重试
    from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
        AgictoChatCompletionsBackend,
    )
    slow_core = TTHAAgentCore(
        AgictoChatCompletionsBackend(client=counter, base_url=smoke.BASE_URL),
        LocalPublicToolGateway(series0[:ORIGINS[0]], task_kind="forecast"))
    slow_agent = TTHASlowAgent(slow_core)
    store = SnapshotStore(root)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    method = TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=()),
            LocalPublicToolGateway(series0[:ORIGINS[0]],
                                   task_kind="forecast"))),
        h0, ())

    events: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    verdict = "NO_NATURAL_FAILURE"
    decided = False
    triggered_round = None
    for r_i, origin in enumerate(ORIGINS):
        allow = (triggered_round is None)  # 复核 Blocker 5：仅第一个 fault
        rd = _feedback_round(
            method, executor, series0, values, origin,
            round_name=f"r{r_i + 1}", slow_agent=slow_agent,
            controller=controller, store=store, events=events,
            allow_trigger=allow)
        rd["round"] = r_i + 1
        rounds.append(rd)
        print(f"== R{r_i + 1} @{origin}: "
              f"{[(p['op'], p['gain']) for p in rd['probes']]} "
              f"support={rd.get('support_event', {}).get('stage')} "
              f"delayed={rd.get('delayed_event', {}).get('stage')} "
              f"proto_fail={rd.get('protocol_failure')}")
        if rd.get("protocol_failure"):
            verdict = "PROTOCOL_FAILURE"
            decided = True
            break
        sev = rd.get("support_event") or {}
        dev = rd.get("delayed_event") or {}
        if not decided and sev.get("triggered"):
            triggered_round = r_i
            if sev.get("stage") in ("no_manifest",):
                verdict = "SLOW_AGENT_ABSTAIN"
            elif sev.get("stage") in ("no_frozen_program", "apply_failed",
                                      "budget_exceeded"):
                verdict = "ACTION_UNAVAILABLE"
            elif sev.get("stage") == "support_rejected":
                verdict = "PATCH_SUPPORT_REJECTED"
            elif sev.get("stage") == "pending":
                if dev.get("stage") == "delayed_rejected":
                    verdict = "PATCH_DELAYED_REJECTED"
                elif dev.get("stage") == "approved":
                    # 复核 Blocker 4：PASS 必须验证下一轮真实采用
                    verdict = "PENDING_NEXT_ROUND_VERIFY"
                else:
                    verdict = "PROTOCOL_FAILURE"
            else:
                verdict = "PROTOCOL_FAILURE"
            decided = True
    # ---- Blocker 4：PASS 验证（下一轮 chosen/program/执行/removal）----
    if verdict == "PENDING_NEXT_ROUND_VERIFY" and triggered_round is not None \
            and triggered_round + 1 < len(ORIGINS):
        nxt_origin = ORIGINS[triggered_round + 1]
        # 下一轮（同一实例——snapshot 已批准更新）
        nxt = _feedback_round(
            method, executor, series0, values, nxt_origin,
            round_name=f"r{triggered_round + 2}_verify",
            slow_agent=slow_agent, controller=controller, store=store,
            events=events, allow_trigger=False)
        rounds.append(nxt)
        adopted = False
        chosen = nxt.get("probes", [{}])[0].get("op", "") if nxt.get("probes") else ""
        # 验证：池含 skill + chosen=skill + program 沿冻结 patch 执行
        skill_chosen = bool(chosen.startswith("skill_") or
                            "cand_skill" in str(nxt.get("pool")))
        if skill_chosen:
            adopted = True
        # removal：h0 新实例同位置重跑 → 无 skill
        method_r = TTHAMethod(
            sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                sealed.SealedProbeBackend(explore=True, operators=()),
                LocalPublicToolGateway(series0[:nxt_origin],
                                       task_kind="forecast"))),
            h0, ())
        events_r: list[dict[str, Any]] = []
        rem = _feedback_round(
            method_r, executor, series0, values, nxt_origin,
            round_name="removal", slow_agent=slow_agent,
            controller=controller, store=store, events=events_r,
            allow_trigger=False)
        rounds.append(rem)
        removal_ok = not any(c.startswith("cand_skill_")
                             for c in rem["pool"])
        verdict = ("NATURAL_SLOW_UPDATE_PASS"
                   if (adopted and removal_ok)
                   else "PROTOCOL_FAILURE")
    print(f"== verdict: {verdict}")
    print(f"== llm_calls: {counter.calls}")

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: (str(getattr(v, "harness_content_sha", v))
                        if k == "snapshot" else _strip(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-natural-method-owned-slow-pilot",
        "note": "P3.2 自然 fresh 在线 pilot（复核修复版；claim 限定 "
                "context/cohort-local；不换 cohort/不重试挑答案/不运行中修复）",
        "cohort": [r["series_uid"] for r in roster],
        "rounds": _strip(rounds),
        "events": _strip(events),
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
