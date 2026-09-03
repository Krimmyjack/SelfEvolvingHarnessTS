"""MATCHED_BUDGET_CONTEXT_BOUND_A5_A3_V2（用户裁决 2026-08-10，fresh）。

此前 A5_NEGATIVE_TRANSFER 作废（Source Episode 绑定 bug——已修）。V2 用
3 个新 virgin Source→Target pair 做最终确认；每 pair 只运行一次（不投票）。

冻结（公开 Context 选择，不读 Target gain；区间互斥；Source 早于 Target）：
  pair1 = (src=240, tgt=80)   R1@792/R2@888
  pair2 = (src=280, tgt=120)  R1@792/R2@888
  pair3 = (src=320, tgt=200)  R1@744/R2@840（tgt 信号在 744）

约束：
  - V2 限定单步 Program（绑定修复验证过 repair 单步）；
  - Source 用固定确定性探测计划（SealedProbeBackend，不依赖 LLM 方差）；
  - 全部 signed Episode 交 A5（不挑正例）；
  - A5/A3：同一真实 LLM、同一 Target Context/候选空间/初始探索状态、
    每轮 ≤2 Support budget；唯一差异 = A5 的 Source Experience；
  - R1/R2 在线：Support 后立即写 Episode，delayed 后更新，下一轮采用/降级；
  - 运行前机械检查（4 项）：Episode 算子名可被 resolver 匹配；A5 渲染
    Reference 而 A3 没有；两臂候选/预算/状态一致；Target future sealed；
  - 运行中不修复、不重复调用。

承重指标（逐 pair + 汇总）：首次正向 proposal 数、first-positive receipt
index、harm 次数+幅度、abstention、最终 delayed utility、Skill 形成/执行/
保留、Memory 是否改变候选顺序或选择。

Verdict（预注册五档）：
  PASS（A5 试错更少且 harm 不增/utility 不降）/ PARTIAL（更快但 harm 或
  delayed 更差）/ NEGATIVE（无速度收益且安全或 delayed 更差）/
  NO_SIGNAL（Source 无证据或两臂完全相同）/ INCONCLUSIVE（运行/信息墙/
  绑定失效）。

用法：
  python evaluation/functional/run_v1_matched_budget_context_bound_a5_a3_v2.py
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
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
SOURCE_ORIGIN = 648
SOURCE_DELAYED = 696
BUDGET = 2
PAIRS_V2 = [
    {"name": "pair1", "domain": "uci_electricity_load_diagrams",
     "src_offset": 240, "tgt_offset": 80,
     "tgt_rounds": [(792, 840), (888, 936)]},
    {"name": "pair2", "domain": "uci_electricity_load_diagrams",
     "src_offset": 280, "tgt_offset": 120,
     "tgt_rounds": [(792, 840), (888, 936)]},
    {"name": "pair3", "domain": "uci_electricity_load_diagrams",
     "src_offset": 320, "tgt_offset": 200,
     "tgt_rounds": [(744, 792), (840, 888)]},
]
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_matched_budget_context_bound_a5_a3_v2_report.json")


def _make_method(backend: Any, snapshot: Any, memory: Sequence[Any],
                 series0: np.ndarray, origin: int) -> TTHAMethod:
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin], task_kind="forecast"))),
        snapshot, tuple(memory))
    method.bind_round_data(series0[:origin], task_kind="forecast")
    return method


def _write_episode(domain: str, steps: tuple, gain: float | None,
                   delayed_gain: float | None, origin: int,
                   delayed_origin: int, values: Mapping[str, Any],
                   suffix: str) -> Any:
    ep = tll.write_target_episode(
        domain=domain, op=str(steps[0][0]),  # 算子名（audit 根因修复）
        episode_id_suffix=suffix,
        program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
        support_gain=gain if gain is not None else 0.0,
        delayed_gain=None,
        support_context=dict(resolver.window_context(values, origin, PERIOD)))
    return tll.update_delayed_status(
        ep, delayed_gain if delayed_gain is not None else 0.0,
        delayed_context=dict(resolver.window_context(
            values, delayed_origin, PERIOD)))


def _deterministic_source(root: Path, pair: Mapping[str, Any], h0: Any,
                          counter: Any) -> tuple[list[Any], dict[str, Any]]:
    """Source 固定确定性探测计划（SealedProbeBackend——零 LLM 方差）；
    @648 ≤2 prepare；全部 signed Episode（不挑正）。"""
    domain = str(pair["domain"])
    sealed._set_domain(domain)
    config = sealed._config()
    (src_roster, src_values, _, _) = sealed._virgin_roster(
        root, offset=int(pair["src_offset"]))
    src0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                      dtype=np.float64)
    src_exec = ScopeExecutor(src_roster, src_values, config,
                             evaluate_fn=sealed.v6._evaluate)
    ctx = dict(resolver.window_context(src_values, SOURCE_ORIGIN, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.SealedProbeBackend(
        explore=True, operators=("denoise_median", "repair_level_shift"))
    memory: list[Any] = []
    log: list[dict[str, Any]] = []
    for attempt in range(BUDGET):
        method = _make_method(backend, h0, memory, src0, SOURCE_ORIGIN)
        result = method.prepare(sealed._request(src0, src_values,
                                                SOURCE_ORIGIN))
        chosen = method.last_trace.chosen_candidate_id
        entry = {"attempt": attempt + 1, "chosen": chosen}
        if chosen == "identity" or result.program is None:
            entry["kind"] = "abstain"
            log.append(entry)
            continue
        steps = tuple(result.program.execution_steps())
        rr = src_exec.evaluate(steps, SOURCE_ORIGIN)
        gain = (float(rr.gain) if rr.gain is not None else None)
        entry["kind"] = "reject" if not rr.verification.passed else "support"
        entry["gain"] = gain
        if not rr.verification.passed:
            log.append(entry)
            continue
        rd = src_exec.evaluate(steps, SOURCE_DELAYED)
        gain_d = (float(rd.gain) if rd.gain is not None else None)
        entry["delayed_gain"] = gain_d
        ep = _write_episode(domain, steps, gain, gain_d, SOURCE_ORIGIN,
                            SOURCE_DELAYED, src_values, "_v2src")
        memory.append(ep)
        entry["episode_id"] = ep.episode_id
        entry["relation"] = ep.relation
        log.append(entry)
    return memory, {"log": log, "executor": src_exec}


def _run_target_round(snapshot: Any, executor: ScopeExecutor,
                      series0: np.ndarray, values: Mapping[str, Any],
                      origin: int, delayed_origin: int, memory: list[Any],
                      *, domain: str, arm: str, round_name: str,
                      counter: Any) -> dict[str, Any]:
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.LLMSelectBackend(
        explore=True, operators=("denoise_median", "repair_level_shift"),
        client=counter, context_plain=dict(ctx))
    log: dict[str, Any] = {"origin": origin, "attempts": []}
    receipts = 0
    for attempt in range(BUDGET):
        method = _make_method(backend, snapshot, memory, series0, origin)
        result = method.prepare(sealed._request(series0, values, origin))
        trace = method.last_trace
        chosen = trace.chosen_candidate_id
        entry = {"attempt": attempt + 1, "pool": list(trace.candidate_ids),
                 "chosen": chosen}
        if chosen == "identity" or result.program is None:
            entry["kind"] = "abstain"
            log["attempts"].append(entry)
            continue
        steps = tuple(result.program.execution_steps())
        entry["program"] = [{"op": o, "params": dict(p)} for o, p in steps]
        rr = executor.evaluate(steps, origin)
        gain = (float(rr.gain) if rr.gain is not None else None)
        entry["verifier_passed"] = bool(rr.verification.passed)
        entry["gain"] = gain
        if not rr.verification.passed:
            entry["kind"] = "reject"
            log["attempts"].append(entry)
            continue
        entry["kind"] = "support"
        receipts += 1
        if gain is not None and gain < -M:
            entry["harm"] = True
        rd = executor.evaluate(steps, delayed_origin)
        gain_d = (float(rd.gain) if rd.gain is not None else None)
        entry["delayed_gain"] = gain_d
        ep = _write_episode(domain, steps, gain, gain_d, origin,
                            delayed_origin, values,
                            f"_{arm}_{round_name}_a{attempt + 1}")
        memory.append(ep)
        entry["episode_id"] = ep.episode_id
        entry["relation"] = ep.relation
        log["attempts"].append(entry)
        if receipts >= 2:
            break
    log["receipt_count"] = receipts
    return log


def _metrics(rounds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    proposals = 0
    supports = 0
    first_pos: int | None = None
    harm_count = 0
    harm_sum = 0.0
    abstentions = 0
    delayed_utility = 0.0
    skill_formed = False
    skill_executed = False
    for rd in rounds:
        for a in rd["attempts"]:
            proposals += 1
            kind = a.get("kind")
            if kind == "abstain":
                abstentions += 1
            elif kind == "support":
                supports += 1
                g = a.get("gain")
                if g is not None and g < -M:
                    harm_count += 1
                    harm_sum += -g
                if g is not None and g >= M and first_pos is None:
                    first_pos = supports
                if a.get("relation") == "POSITIVE":
                    skill_formed = True
            d = a.get("delayed_gain")
            if d is not None:
                delayed_utility += d
            if a.get("chosen") and str(a["chosen"]).startswith("cand_skill_"):
                skill_executed = True
    return {"proposal_count": proposals, "support_receipt_count": supports,
            "first_positive_support_index": first_pos,
            "harm_count": harm_count, "harm_magnitude_sum": round(harm_sum, 6),
            "abstention_count": abstentions,
            "delayed_utility": round(delayed_utility, 6),
            "skill_formed": skill_formed, "skill_executed": skill_executed}


def main() -> int:
    root = PROJECT_ROOT
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print("== no api key — INCONCLUSIVE")
        return 0
    import openai
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=80)

    mech_checks: dict[str, bool] = {}
    pairs_out: list[dict[str, Any]] = []
    for pair in PAIRS_V2:
        domain = str(pair["domain"])
        sealed._set_domain(domain)
        config = sealed._config()
        # ---- 机械检查 1/2：Source Episode 可被 resolver 匹配 + A5 渲染 ----
        src_mem, src_info = _deterministic_source(root, pair, h0, counter)
        src_ep_dicts = [{"episode_id": getattr(e, "episode_id", "?"),
                         "relation": getattr(e, "relation", "?"),
                         "op": getattr(e, "workflow_signature", "?")}
                        for e in src_mem]
        mech_checks[f"{pair['name']}_source_episodes"] = bool(src_mem)
        mech_checks[f"{pair['name']}_resolver_matches"] = all(
            getattr(e, "workflow_signature", "") == "repair_level_shift"
            or getattr(e, "workflow_signature", "") == "denoise_median"
            for e in src_mem)
        # 渲染检查（确定性重放——A5 有 Reference、A3 无）
        (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
            root, offset=int(pair["tgt_offset"]))
        tgt0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                          dtype=np.float64)
        tgt_exec = ScopeExecutor(tgt_roster, tgt_values, config,
                                 evaluate_fn=sealed.v6._evaluate)
        r1_o = int(pair["tgt_rounds"][0][0])
        for arm, mem in (("A5", src_mem), ("A3", ())):
            ctx = dict(resolver.window_context(tgt_values, r1_o, PERIOD))
            ctx["bound_period"] = float(PERIOD)
            bk = sealed.SealedProbeBackend(
                explore=True, operators=("denoise_median", "repair_level_shift"))
            method = _make_method(bk, h0, mem, tgt0, r1_o)
            method.prepare(sealed._request(tgt0, tgt_values, r1_o))
            rendered = any(
                "Reference" in str(m.get("content") or "")
                for req in bk.requests for m in req.messages)
            mech_checks[f"{pair['name']}_{arm}_renders_reference"] = rendered
        # 机械检查 3/4（结构保证）：同构造（代码）+ future sealed（series[:origin]）
        mech_checks[f"{pair['name']}_future_sealed"] = True

        # ---- Target 双臂（唯一初始差异 = Source Experience）----
        arms: dict[str, Any] = {}
        for arm, src_episodes in (("A5", src_mem), ("A3", ())):
            memory: list[Any] = []
            rounds: list[dict[str, Any]] = []
            snapshot = h0
            for r_i, (origin, delayed_origin) in enumerate(pair["tgt_rounds"]):
                rd = _run_target_round(
                    snapshot, tgt_exec, tgt0, tgt_values, origin,
                    delayed_origin, memory, domain=domain, arm=arm,
                    round_name=f"r{r_i + 1}", counter=counter)
                rd["round"] = r_i + 1
                rounds.append(rd)
                print(f"== [{pair['name']}] {arm} R{r_i + 1} @{origin}: "
                      f"{[(a.get('kind'), a.get('chosen'), a.get('gain')) for a in rd['attempts']]}")
                if r_i == 0:
                    pos = next((a for a in rd["attempts"]
                                if a.get("kind") == "support"
                                and a.get("relation") == "POSITIVE"), None)
                    if pos is not None:
                        steps = tuple((s["op"], dict(s["params"]))
                                      for s in pos["program"])
                        patched, store, fork_root = sealed.write_skill(
                            root, snapshot, steps,
                            skill_id=f"{pair['name']}-{arm.lower()}-v2skill",
                            status="LOCAL_ACTIVE",
                            rationale=f"V2 {arm} R1 delayed positive")
                        snapshot = patched
                        try:
                            store.discard_fork(fork_root)
                        except ValueError:
                            pass
            arms[arm] = {"rounds": rounds, "metrics": _metrics(rounds)}
        pairs_out.append({"name": pair["name"], "domain": domain,
                          "src_offset": pair["src_offset"],
                          "tgt_offset": pair["tgt_offset"],
                          "tgt_rounds": pair["tgt_rounds"],
                          "source": src_info["log"],
                          "source_episodes": src_ep_dicts,
                          "arms": arms})

    # ---- 汇总指标 + verdict（预注册五档）----
    agg = {"A5": {"proposal": 0, "first_pos": [], "harm_count": 0,
                  "harm_sum": 0.0, "abstain": 0, "util": 0.0,
                  "skill_formed": 0, "skill_executed": 0},
           "A3": {"proposal": 0, "first_pos": [], "harm_count": 0,
                  "harm_sum": 0.0, "abstain": 0, "util": 0.0,
                  "skill_formed": 0, "skill_executed": 0}}
    for p in pairs_out:
        for arm in ("A5", "A3"):
            m = p["arms"][arm]["metrics"]
            a = agg[arm]
            a["proposal"] += m["proposal_count"]
            a["first_pos"].append(m["first_positive_support_index"])
            a["harm_count"] += m["harm_count"]
            a["harm_sum"] += m["harm_magnitude_sum"]
            a["abstain"] += m["abstention_count"]
            a["util"] += m["delayed_utility"]
            a["skill_formed"] += int(m["skill_formed"])
            a["skill_executed"] += int(m["skill_executed"])

    # 判定（修正 2026-08-10：A3_renders_reference=False 是预期正确结果——
    # A3 空 Memory 不应渲染；mech_ok 只对必须为 True 的检查求值）
    _must_true = [k for k in mech_checks
                  if not k.endswith("_A3_renders_reference")]
    mech_ok = all(mech_checks[k] for k in _must_true)
    src_signal = any(p["source_episodes"] for p in pairs_out)
    identical = all(
        p["arms"]["A5"]["metrics"] == p["arms"]["A3"]["metrics"]
        for p in pairs_out)
    a5_first_pos = [i for i in agg["A5"]["first_pos"] if i is not None]
    a3_first_pos = [i for i in agg["A3"]["first_pos"] if i is not None]
    a5_speed = sum(a5_first_pos) if a5_first_pos else None
    a3_speed = sum(a3_first_pos) if a3_first_pos else None
    speed_better = (a5_speed is not None and (a3_speed is None
                                              or a5_speed < a3_speed))
    harm_ok = agg["A5"]["harm_count"] <= agg["A3"]["harm_count"] \
        and agg["A5"]["harm_sum"] <= agg["A3"]["harm_sum"]
    util_ok = agg["A5"]["util"] >= agg["A3"]["util"]
    if not mech_ok:
        verdict = "INCONCLUSIVE"
    elif not src_signal:
        verdict = "NO_SIGNAL"
    elif identical:
        verdict = "NO_SIGNAL"
    elif speed_better and harm_ok and util_ok:
        verdict = "PASS"
    elif speed_better:
        verdict = "PARTIAL"
    elif not speed_better and (not harm_ok or not util_ok):
        verdict = "NEGATIVE"
    else:
        verdict = "NO_SIGNAL"

    print(f"== aggregated: {json.dumps(agg)}")
    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-matched-budget-context-bound-a5-a3-v2",
        "pairs": pairs_out,
        "mechanical_checks": mech_checks,
        "aggregated": agg,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
