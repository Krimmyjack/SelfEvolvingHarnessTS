"""KDD_CUP_2018_CROSS_DOMAIN_MATCHED_BUDGET_A5_A3（P4.1，用户裁决
2026-08-11；P4.0 Gate 已通过——MEMORY_ACTION_SIGNAL）。

跨域 matched-budget：A5（Monash winsorize POSITIVE Source Episode）vs
A3（空）——唯一初始差异 Source Memory。Target = 新 KDD virgin cohort
（长度/公开 Context/静态 verifier 冻结——零 gain）；3 合法候选
（winsorize/outlier_mad/hampel——KDD 池）；预算 2；Runtime 只评估 top-2；
Support 立即写 Episode；delayed 两阶段批准（时间边界）；下一轮验证
Skill 正常入口实际选择和执行；不换 cohort/不重跑挑答案。

预注册 verdict（六档）：
  CROSS_DOMAIN_MEMORY_CANDIDATE_PASS：A5 将有效候选带入预算而 A3 错过，
    减少首次正向试错或 harm，且 delayed 不更差
  NO_SIGNAL / NEGATIVE_TRANSFER / INCONCLUSIVE_LLM_VARIANCE /
  INFEASIBLE_CANDIDATE_CONTENTION / PROTOCOL_FAILURE

Claim 限定：Monash → KDD 一次正向只能称"跨数据集复用候选证据"；≥2 个
不同 Target Dataset 方向性复现才能声称 cross-domain benefit。

用法：
  python evaluation/functional/run_v1_kdd2018_cross_domain_a5_a3.py
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
from run_v1_kdd2018_memory_gate import (  # noqa: E402
    _monash_source_episodes,
)
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
    _patch_options,
    _card_from_episode,
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
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.operators.registry import OPERATOR_METADATA  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
ORIGINS = (600, 792, 888)
POOL = ("winsorize", "outlier_mad", "hampel_filter")  # KDD 3 合法
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
FROZEN_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_frozen_cohort_p41.jsonl"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2/w1_kdd2018_cross_domain_a5_a3_report.json"


def _freeze_cohort(root: Path) -> list[dict[str, object]]:
    """新 virgin cohort（零 gain）：KDD 剩余（排除 K0 已用 20 支）——
    长度 ≥984、公开 Context outlier 信号、outlier family 静态合法 ≥2
    （ScopeExecutor.verify——零 gain）。"""
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    used = {json.loads(line)["series_name"] for line in
            (root / "artifacts/functional/e2/w1_kdd2018_frozen_cohort.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()}
    role_seq = ["train"] * 12 + ["support"] * 4 + ["query"] * 4
    frozen: list[dict[str, object]] = []
    for i, n in enumerate(names):
        if len(frozen) >= 20:
            break
        if n in used:
            continue
        if int(cache["lengths"][i]) < 984:
            continue
        s = np.asarray(values[i][:600], dtype=np.float64)
        fe = dict(extract_public_features(s, task_kind="forecast"))
        if not (float(fe.get("level_excursion_score", 0.0)) > 1.0
                or "estimated_region_start_fraction" in fe):
            continue
        ok = 0
        for op in POOL:
            steps = ((op, dict(wiring.contract_params(op, PERIOD))),)
            ex = ScopeExecutor([{"series_uid": n, "role": "train"}],
                               {n: np.asarray(values[i], dtype=np.float64)},
                               _config(), evaluate_fn=_evaluate_kdd)
            if ex.verify(steps, 600).passed:
                ok += 1
        if ok < 2:
            continue
        frozen.append({"cohort": "K1", "role": role_seq[len(frozen)],
                       "series_name": n, "type": "kdd2018"})
    if len(frozen) < 20:
        return []
    FROZEN_REL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen),
        encoding="utf-8")
    return frozen


def _feedback_round(method: TTHAMethod, executor: ScopeExecutor,
                    series0: np.ndarray, values: Mapping[str, Any],
                    origin: int, *, round_name: str,
                    slow_agent: Any, controller: Any, store: Any,
                    events: list[dict[str, Any]],
                    allow_trigger: bool) -> dict[str, Any]:
    slow_agent.core.tools = LocalPublicToolGateway(series0[:origin],
                                                   task_kind="forecast")
    backend = sealed.SealedProbeBackend(
        explore=True, operators=POOL, max_propose_candidates=3,
        force_pool=True)
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
                episode_id_suffix=f"_p41_{round_name}_p{i + 1}",
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


def _metrics(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    probes = [p for rd in rounds for p in rd["probes"]]
    supports = [p for p in probes if p.get("passed") and p.get("gain") is not None]
    harms = [p for p in supports if p["gain"] < -M]
    first_pos = None
    for r_i, rd in enumerate(rounds):
        for p in rd["probes"]:
            if p.get("passed") and p.get("gain") is not None \
                    and p["gain"] >= M:
                first_pos = {"round": r_i + 1, "probe": p["probe"]}
                break
        if first_pos:
            break
    return {
        "probe_count": len(probes),
        "support_count": len(supports),
        "first_positive": first_pos,
        "harm_count": len(harms),
        "harm_sum": round(sum(-p["gain"] for p in harms), 6),
        "delayed_utility": round(sum(
            rd["probes"][0].get("delayed_gain") or 0.0
            for rd in rounds if rd["probes"]), 6),
        "abstention_count": sum(
            1 for rd in rounds if not rd["probes"]),
        "skill_formed": any(
            rd.get("delayed_event", {}).get("stage") == "approved"
            for rd in rounds),
    }


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    frozen = _freeze_cohort(root)
    if not frozen:
        print(json.dumps({"verdict": "INFEASIBLE_CANDIDATE_CONTENTION",
                          "reason": "no virgin cohort frozen"}, indent=1))
        return 0
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in frozen]
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in frozen}
    series0 = vals[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, vals, _config(),
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
        max_calls=8)
    from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
        AgictoChatCompletionsBackend,
    )

    source_eps = _monash_source_episodes(root)  # 跨域 Source（Monash winsorize）
    arms_out: dict[str, Any] = {}
    for arm, memory in (("A5", source_eps), ("A3", ())):
        slow_core = TTHAAgentCore(
            AgictoChatCompletionsBackend(client=counter,
                                         base_url=smoke.BASE_URL),
            LocalPublicToolGateway(series0[:ORIGINS[0]],
                                   task_kind="forecast"))
        slow_agent = TTHASlowAgent(slow_core)
        store = SnapshotStore(root)
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        method = TTHAMethod(
            sealed.TTHAFastAgent(sealed.TTHAAgentCore(
                sealed.SealedProbeBackend(explore=True, operators=()),
                LocalPublicToolGateway(series0[:ORIGINS[0]],
                                       task_kind="forecast"))),
            h0, tuple(memory))
        events: list[dict[str, Any]] = []
        rounds: list[dict[str, Any]] = []
        triggered = False
        for r_i, origin in enumerate(ORIGINS):
            rd = _feedback_round(
                method, executor, series0, vals, origin,
                round_name=f"{arm.lower()}_r{r_i + 1}",
                slow_agent=slow_agent, controller=controller, store=store,
                events=events, allow_trigger=not triggered)
            rd["round"] = r_i + 1
            rounds.append(rd)
            if rd.get("support_event", {}).get("triggered"):
                triggered = True
            if rd.get("protocol_failure"):
                break
        arms_out[arm] = {
            "rounds": rounds, "metrics": _metrics(rounds),
            "triggered": triggered}
        print(f"== {arm}: "
              f"{[ [(p['op'], round(p['gain'], 4) if p.get('gain') is not None else None) for p in rd['probes']] for rd in rounds]} "
              f"support_stage={[rd.get('support_event', {}).get('stage') for rd in rounds]} "
              f"delayed_stage={[rd.get('delayed_event', {}).get('stage') for rd in rounds]}")

    m5, m3 = arms_out["A5"]["metrics"], arms_out["A3"]["metrics"]
    print(f"== A5 metrics: {json.dumps(m5)}")
    print(f"== A3 metrics: {json.dumps(m3)}")
    # ---- verdict（预注册六档）----
    if any(rd.get("protocol_failure")
           for rd in (*arms_out["A5"]["rounds"], *arms_out["A3"]["rounds"])):
        verdict = "PROTOCOL_FAILURE"
    else:
        harm_worse = (m5["harm_count"] > m3["harm_count"]
                      or m5["harm_sum"] > m3["harm_sum"]) \
            and m5["delayed_utility"] <= m3["delayed_utility"]
        util_worse = m5["delayed_utility"] < m3["delayed_utility"]
        fp5 = m5["first_positive"]
        fp3 = m3["first_positive"]

        def _k(fp: Any) -> tuple[int, int] | None:
            return None if fp is None else (int(fp["round"]), int(fp["probe"]))
        k5, k3 = _k(fp5), _k(fp3)
        speed = (k3 is None and k5 is not None) or (
            k5 is not None and k3 is not None and k5 < k3)
        if harm_worse or util_worse:
            verdict = "NEGATIVE_TRANSFER"
        elif speed and m5["delayed_utility"] >= m3["delayed_utility"]:
            verdict = "CROSS_DOMAIN_MEMORY_CANDIDATE_PASS"
        elif m5 == m3:
            verdict = "NO_SIGNAL"
        else:
            verdict = "NO_SIGNAL"  # 无速度优势且无负迁移 → 同档
    print(f"== verdict: {verdict}  llm_calls={counter.calls}")

    def _strip(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: (str(getattr(v, "harness_content_sha", v))
                        if k == "snapshot" else _strip(v))
                    for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip(x) for x in obj]
        return obj

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-kdd2018-cross-domain-a5-a3",
        "note": "P4.1 跨域 matched-budget（Monash winsorize Source → KDD "
                "Target；claim 限定：单 Target 只称'跨数据集复用候选证据'）",
        "cohort": [r["series_name"] for r in frozen],
        "source_episodes": [getattr(e, "episode_id", "?") for e in source_eps],
        "arms": {a: {"rounds": _strip(arms_out[a]["rounds"]),
                     "metrics": arms_out[a]["metrics"],
                     "triggered": arms_out[a]["triggered"]}
                 for a in ("A5", "A3")},
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
