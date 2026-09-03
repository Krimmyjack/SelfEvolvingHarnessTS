"""KDD_CUP_2018_NATURAL_SLOW_UPDATE（P2/P3，用户顺序实验 2026-08-11）。

P0 PREMISE_OK（outlier family 4 合法替代）、P1 NATURAL_FAILURE_DETECTED
（winsorize −0.029 @600）→ P2：自然 Slow Update（方法层两阶段 pending +
契约 preflight 强制执行 + 真实 Slow Agent ≤1 正式 +1 retry）→ P3：下一轮
真实采用验证（chosen/program/执行）+ removal。

Verdict（预注册）：
  NATURAL_SLOW_UPDATE_PASS / NO_NATURAL_FAILURE / ACTION_UNAVAILABLE /
  SLOW_AGENT_ABSTAIN / PATCH_SUPPORT_REJECTED / PATCH_DELAYED_REJECTED /
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_kdd2018_natural_slow_update.py
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

PERIOD = 24  # hourly
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
OUTLIER_FAMILY = ("winsorize", "outlier_iqr", "outlier_mad",
                  "hampel_filter")
FROZEN_ROSTER_REL = Path(
    "artifacts/functional/e2/w1_kdd2018_frozen_cohort.jsonl")
CACHE = Path("data/kdd2018/series_cache.npz")
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_kdd2018_natural_slow_update_report.json")


def _config() -> dict[str, object]:
    return {
        "dataset_id": "kdd_cup_2018",
        "sampling": "hourly_regular",
        "period": PERIOD,
        "anchors": list(range(312, 853, 60)),
        "support_origin": 600,
        "selection_origin": 600,
    }


def _evaluate_kdd(roster, values, compiled, config, *, origin):
    mapped = [dict(row, role="eval") if str(row["role"]) != "train"
              else dict(row) for row in roster]
    return v6._evaluate(mapped, values, compiled, config, origin=origin)


def _request(series0: np.ndarray, values, origin: int) -> PreparationRequest:
    observed = dict(resolver.window_context(values, origin, PERIOD))
    observed["bound_period"] = float(PERIOD)
    return PreparationRequest(
        "kdd2018-slow-update",
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
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    return {"roster": roster, "values": vals}


def _patch_options(executor: ScopeExecutor, values: Mapping[str, Any],
                   origin: int, failed_op: str) -> list[dict[str, Any]]:
    """Typed Patch 白名单（outcome 前冻结规则）：排除失败算子；同 family
    静态 verifier 合法；registry 固定顺序 ≤2。"""
    meta = OPERATOR_METADATA.get(failed_op) or {}
    family = meta.get("category")
    out: list[dict[str, Any]] = []
    series0 = np.asarray(values[list(values)[0]][:origin], dtype=np.float64)
    for op in OPERATOR_NAMES:
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
        "pattern_id": f"kdd2018-{failed_op}-neg",
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
    slow_agent.core.tools = LocalPublicToolGateway(series0[:origin],
                                                   task_kind="forecast")
    backend = sealed.SealedProbeBackend(
        explore=True, operators=OUTLIER_FAMILY,
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
                           "probes": [], "protocol_failure": False,
                           "chosen": trace.chosen_candidate_id}
    for i, op in enumerate(pool_ops[:2]):
        steps = steps_map[f"cand_{op}"]
        rr = executor.evaluate(steps, origin)
        gain = (float(rr.gain) if rr.gain is not None else None)
        passed = bool(rr.verification.passed)
        entry: dict[str, Any] = {"probe": i + 1, "op": op, "gain": gain,
                                 "passed": passed}
        if passed and gain is None:
            log["protocol_failure"] = True
            log["protocol_reason"] = f"outcome_unavailable ({op})"
            log["probes"].append(entry)
            break
        if passed:
            ep = tll.write_target_episode(
                domain="kdd_cup_2018", op=op,
                episode_id_suffix=f"_kdd_{round_name}_p{i + 1}",
                program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
                support_gain=gain if gain is not None else 0.0,
                delayed_gain=None,
                support_context=dict(resolver.window_context(
                    values, origin, PERIOD)))
            entry["episode_id"] = ep.episode_id
            method.append_experience_episode(ep)
            rd_ep = executor.evaluate(steps, origin + HORIZON)
            dg = (float(rd_ep.gain) if rd_ep.gain is not None else None)
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
                             evaluate_fn=_evaluate_kdd)

    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — PROTOCOL_FAILURE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=2)  # 1 正式 + 1 契约内 schema retry
    from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
        AgictoChatCompletionsBackend,
    )
    slow_core = TTHAAgentCore(
        AgictoChatCompletionsBackend(client=counter, base_url=smoke.BASE_URL),
        LocalPublicToolGateway(series0[:600], task_kind="forecast"))
    slow_agent = TTHASlowAgent(slow_core)
    store = SnapshotStore(root)
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    method = TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=()),
            LocalPublicToolGateway(series0[:600], task_kind="forecast"))),
        h0, ())

    events: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    verdict = "NO_NATURAL_FAILURE"
    r1 = _feedback_round(method, executor, series0, values, 600,
                         round_name="r1", slow_agent=slow_agent,
                         controller=controller, store=store, events=events,
                         allow_trigger=True)
    r1["round"] = 1
    rounds.append(r1)
    print(f"== R1 @600: {[(p['op'], p['gain']) for p in r1['probes']]} "
          f"support={r1.get('support_event', {}).get('stage')} "
          f"delayed={r1.get('delayed_event', {}).get('stage')}")
    sev = r1.get("support_event") or {}
    dev = r1.get("delayed_event") or {}
    if r1.get("protocol_failure"):
        verdict = "PROTOCOL_FAILURE"
    elif not sev.get("triggered"):
        verdict = "NO_NATURAL_FAILURE"
    elif sev.get("stage") in ("no_manifest",):
        verdict = "SLOW_AGENT_ABSTAIN"
    elif sev.get("stage") in ("no_frozen_program", "apply_failed",
                              "manifest_preflight_failed", "budget_exceeded"):
        verdict = "ACTION_UNAVAILABLE"
    elif sev.get("stage") == "support_rejected":
        verdict = "PATCH_SUPPORT_REJECTED"
    elif sev.get("stage") == "pending":
        if dev.get("stage") == "delayed_rejected":
            verdict = "PATCH_DELAYED_REJECTED"
        elif dev.get("stage") == "approved":
            # ---- P3：下一轮真实采用验证 + removal ----
            r2 = _feedback_round(method, executor, series0, values, 792,
                                 round_name="r2_verify",
                                 slow_agent=slow_agent,
                                 controller=controller, store=store,
                                 events=events, allow_trigger=False)
            r2["round"] = 2
            rounds.append(r2)
            print(f"== R2 @792: pool={r2['pool']} probes="
                  f"{[(p['op'], p['gain']) for p in r2['probes']]}")
            adopted = any(c.startswith("cand_skill_") for c in r2["pool"])
            # removal：h0 新实例同位置
            method_r = TTHAMethod(
                sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                    sealed.SealedProbeBackend(explore=True, operators=()),
                    LocalPublicToolGateway(series0[:792],
                                           task_kind="forecast"))),
                h0, ())
            events_r: list[dict[str, Any]] = []
            rem = _feedback_round(method_r, executor, series0, values, 792,
                                  round_name="removal",
                                  slow_agent=slow_agent,
                                  controller=controller, store=store,
                                  events=events_r, allow_trigger=False)
            rounds.append(rem)
            removal_ok = not any(c.startswith("cand_skill_")
                                 for c in rem["pool"])
            print(f"== removal: pool={rem['pool']} removal_ok={removal_ok}")
            verdict = ("NATURAL_SLOW_UPDATE_PASS"
                       if (adopted and removal_ok) else "PROTOCOL_FAILURE")
        else:
            verdict = "PROTOCOL_FAILURE"
    else:
        verdict = "PROTOCOL_FAILURE"
    print(f"== verdict: {verdict}  llm_calls={counter.calls}")

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: (str(getattr(v, "harness_content_sha", v))
                        if k == "snapshot" else _strip(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    REPORT_OUT_REL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-natural-slow-update",
        "note": "P2/P3 自然 Slow Update（KDD 2018；方法层两阶段 pending + "
                "契约 preflight；claim 限定 context/cohort-local）",
        "cohort": [r["series_uid"] for r in roster],
        "rounds": _strip(rounds),
        "events": _strip(events),
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_OUT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
