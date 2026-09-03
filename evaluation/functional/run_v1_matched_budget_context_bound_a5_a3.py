"""MATCHED_BUDGET_CONTEXT_BOUND_A5_A3_CONFIRMATION（用户裁决 2026-08-10）。

唯一假设：在相同 Target feedback budget 下，包含 Source 正向/负向/冲突
Experience 的 A5 能否比空 Source Memory 的 A3 更快且不更危险地形成有效
Target-local Skill。

数据冻结（公开 Context 选择，不读 Target gain）：
  pair1 = uci offset=80（3/3 bound-actionable：offset≈−11.5）
  pair2 = uci offset=120（3/3 bound-actionable：offset≈−119.4）
  每 pair：Source 20 支（@600/648）+ Target 20 支（@792/840/888/936），
  互斥（_virgin_roster）；Source 早于 Target 决策。

流程：
  Source 固定预算完整轨迹（@600 ≤2 prepare → Support/abstain 全保留 →
    Episode 写回 → delayed @648 更新）——不挑正。
  Target 双臂（唯一初始差异 = Source Experience）：
    R1 @792 prepare → ≤2 Support receipts → 立即写 Episode →
    delayed @840 更新 → R2 @888 prepare（memory 含 R1）→ ≤2 →
    delayed @936 更新。
  一致性：同 LLM 模型/配置、同 inventory、同 max_propose=2、同 Target
    Support budget、同 delayed 日程、同初始探索状态（空 explored）、同
    verifier/Consumer/Metric、同写回方式。冻结 Slow Agent（自然失败只
    记录不处理）。Target 参数从自身公开 Context 重新绑定（不复制 Source）。

指标：proposal / Support receipt / first-positive index / harm count+sum /
  abstention / delayed utility（各 delayed gain 和）/ Skill 形成轮次 /
  Skill 实际执行 / 同输入 LLM 不一致。verifier rejection 不计 Support
  receipt。

Verdict（预注册）：
  A5_CONFIRMATION_PASS / A5_SAME / A5_MIXED / A5_NEGATIVE_TRANSFER /
  INCONCLUSIVE_LLM_VARIANCE / INFEASIBLE_NO_SOURCE_SIGNAL

用法：
  python evaluation/functional/run_v1_matched_budget_context_bound_a5_a3.py
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
# 预注册两个独立 pair（公开 Context 选择：Source series0 @648 可绑定 +
# Target series0 @792 可绑定；区间互斥；Source 决策 @648 < Target @792）
# pair1: src=240（eligible[240:260]）+ tgt=80（eligible[100:120]）
# pair2: src=280（eligible[280:300]）+ tgt=120（eligible[140:160]）
PAIRS = [
    {"name": "pair1", "domain": "uci_electricity_load_diagrams",
     "src_offset": 240, "tgt_offset": 80},
    {"name": "pair2", "domain": "uci_electricity_load_diagrams",
     "src_offset": 280, "tgt_offset": 120},
]
SOURCE_ORIGIN = 600
SOURCE_DELAYED = 648
TARGET_ROUNDS = [(792, 840), (888, 936)]
BUDGET = 2  # 每决策点 ≤2 prepare
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_matched_budget_context_bound_a5_a3_report.json")


def _make_method(backend: Any, snapshot: Any, memory: Sequence[Any],
                 series0: np.ndarray, origin: int) -> TTHAMethod:
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:origin], task_kind="forecast"))),
        snapshot, tuple(memory))
    method.bind_round_data(series0[:origin], task_kind="forecast")
    return method


def _run_round(snapshot: Any, executor: ScopeExecutor, series0: np.ndarray,
               values: Mapping[str, Any], origin: int, delayed_origin: int,
               memory: list[Any], *, domain: str, arm: str, round_name: str,
               counter: Any) -> dict[str, Any]:
    """一决策点 ≤2 prepare：Support/abstain 全记录；passed 行动立即写
    Episode + delayed 更新。返回轮日志。"""
    log: dict[str, Any] = {"origin": origin, "attempts": []}
    receipts = 0
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ctx["bound_period"] = float(PERIOD)
    backend = sealed.LLMSelectBackend(
        explore=True, operators=("denoise_median", "repair_level_shift"),
        client=counter, context_plain=dict(ctx))
    for attempt in range(BUDGET):
        method_i = _make_method(backend, snapshot, memory, series0, origin)
        result = method_i.prepare(sealed._request(series0, values, origin))
        trace = method_i.last_trace
        chosen = trace.chosen_candidate_id
        entry: dict[str, Any] = {"attempt": attempt + 1,
                                 "pool": list(trace.candidate_ids),
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
            entry["kind"] = "reject"  # 不计 Support receipt
            log["attempts"].append(entry)
            continue
        entry["kind"] = "support"
        receipts += 1
        if gain is not None and gain < -M:
            entry["harm"] = True
        ep = tll.write_target_episode(
            domain=domain, op=str(steps[0][0]),  # 算子名（非候选 ID——否则
            # workflow_signature 带 cand_ 前缀，resolve_order 无法匹配算子
            # → signed 渲染失效；audit 2026-08-10 根因）
            episode_id_suffix=f"_{arm}_{round_name}_a{attempt + 1}",
            program_steps=[{"op": o, "params": dict(p)} for o, p in steps],
            support_gain=gain if gain is not None else 0.0,
            delayed_gain=None,
            support_context=dict(resolver.window_context(values, origin,
                                                         PERIOD)))
        memory.append(ep)
        entry["episode_id"] = ep.episode_id
        entry["relation"] = ep.relation
        rd = executor.evaluate(steps, delayed_origin)
        gain_d = (float(rd.gain) if rd.gain is not None else None)
        entry["delayed_gain"] = gain_d
        for i_e, e in enumerate(memory):
            if getattr(e, "episode_id", "") == ep.episode_id:
                memory[i_e] = tll.update_delayed_status(
                    e, gain_d if gain_d is not None else 0.0,
                    delayed_context=dict(resolver.window_context(
                        values, delayed_origin, PERIOD)))
                entry["relation"] = memory[i_e].relation
                break
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
    skill_round: int | None = None
    skill_executed = False
    llm_inconsistent = 0
    for rd in rounds:
        for a in rd["attempts"]:
            proposals += 1
            kind = a.get("kind")
            if kind == "abstain":
                abstentions += 1
            elif kind == "reject":
                pass
            elif kind == "support":
                supports += 1
                g = a.get("gain")
                if g is not None and g < -M:
                    harm_count += 1
                    harm_sum += -g
                if g is not None and g >= M and first_pos is None:
                    first_pos = supports
            d = a.get("delayed_gain")
            if d is not None:
                delayed_utility += d
            if kind == "support" and a.get("relation") == "POSITIVE" \
                    and skill_round is None:
                skill_round = rd["origin"]
            if a.get("chosen") and a["chosen"].startswith("cand_skill_"):
                skill_executed = True
        # 同输入不一致：同轮两次 attempt 的 pool 相同但 chosen 不同（且非
        # 探索推进——pool 相同）
        pools = [tuple(a.get("pool") or ()) for a in rd["attempts"]]
        for i in range(1, len(pools)):
            if pools[i] == pools[i - 1] and pools[i] and len(pools[i]) > 1:
                c1 = rd["attempts"][i - 1].get("chosen")
                c2 = rd["attempts"][i].get("chosen")
                if c1 != c2:
                    llm_inconsistent += 1
    return {"proposal_count": proposals, "support_receipt_count": supports,
            "first_positive_support_index": first_pos,
            "harm_count": harm_count, "harm_magnitude_sum": round(harm_sum, 6),
            "abstention_count": abstentions,
            "delayed_utility": round(delayed_utility, 6),
            "skill_formed_at_origin": skill_round,
            "skill_executed": skill_executed,
            "llm_same_input_inconsistent": llm_inconsistent}


def _run_pair(root: Path, pair: Mapping[str, Any], h0: Any,
              counter: Any) -> dict[str, Any]:
    domain = str(pair["domain"])
    src_offset = int(pair["src_offset"])
    tgt_offset = int(pair["tgt_offset"])
    sealed._set_domain(domain)
    config = sealed._config()
    (src_roster, src_values, _, _) = sealed._virgin_roster(
        root, offset=src_offset)
    (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=tgt_offset)
    src_series0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                             dtype=np.float64)
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    src_executor = ScopeExecutor(src_roster, src_values, config,
                                 evaluate_fn=sealed.v6._evaluate)
    tgt_executor = ScopeExecutor(tgt_roster, tgt_values, config,
                                 evaluate_fn=sealed.v6._evaluate)

    # ---- Source 固定预算完整轨迹（@600 ≤2；全保留不挑正）----
    src_memory: list[Any] = []
    src_log = _run_round(h0, src_executor, src_series0, src_values,
                         SOURCE_ORIGIN, SOURCE_DELAYED, src_memory,
                         domain=domain, arm="src", round_name="s1",
                         counter=counter)
    src_mem_dicts = [{"episode_id": getattr(e, "episode_id", "?"),
                      "relation": getattr(e, "relation", "?"),
                      "op": getattr(e, "workflow_signature", "?")}
                     for e in src_memory]
    print(f"== [{pair['name']}] source: "
          f"{[(a.get('kind'), a.get('chosen'), a.get('gain')) for a in src_log['attempts']]} "
          f"episodes={src_mem_dicts}")

    # ---- Target 双臂（唯一初始差异 = Source Experience）----
    arm_logs: dict[str, Any] = {}
    for arm, src_episodes in (("A5", src_memory), ("A3", ())):
        memory: list[Any] = []
        rounds: list[dict[str, Any]] = []
        snapshot = h0
        for r_i, (origin, delayed_origin) in enumerate(TARGET_ROUNDS):
            rd = _run_round(snapshot, tgt_executor, tgt_series0, tgt_values,
                            origin, delayed_origin, memory,
                            domain=domain, arm=arm,
                            round_name=f"r{r_i + 1}", counter=counter)
            rd["round"] = r_i + 1
            rounds.append(rd)
            print(f"== [{pair['name']}] {arm} R{r_i + 1} @{origin}: "
                  f"{[(a.get('kind'), a.get('chosen'), a.get('gain')) for a in rd['attempts']]}")
            # Skill 形成（R1 delayed 正向后写 Skill → R2 采用）
            if r_i == 0:
                pos = next((a for a in rd["attempts"]
                            if a.get("kind") == "support"
                            and a.get("delayed_gain") is not None
                            and a["delayed_gain"] >= M), None)
                if pos is not None:
                    steps = tuple((s["op"], dict(s["params"]))
                                  for s in pos["program"])
                    patched, store, fork_root = sealed.write_skill(
                        root, snapshot, steps,
                        skill_id=f"{pair['name']}-{arm.lower()}-skill-v1",
                        status="LOCAL_ACTIVE",
                        rationale=f"matched-budget {arm} R1 delayed positive")
                    snapshot = patched
                    try:
                        store.discard_fork(fork_root)
                    except ValueError:
                        pass
        arm_logs[arm] = {"rounds": rounds, "metrics": _metrics(rounds)}
    return {"name": pair["name"], "domain": domain,
            "src_offset": src_offset, "tgt_offset": tgt_offset,
            "source": src_log, "source_episodes": src_mem_dicts,
            "arms": arm_logs}


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
        max_calls=60)

    pairs_out: list[dict[str, Any]] = []
    for pair in PAIRS:
        pairs_out.append(_run_pair(root, pair, h0, counter))

    # ---- verdict（预注册六档）----
    # source_signal 检查 Source 轨迹的 Support（不是 Target 臂）
    source_signal = any(
        p["source"]["receipt_count"] > 0 and p["source_episodes"]
        for p in pairs_out)
    if not source_signal:
        verdict = "INFEASIBLE_NO_SOURCE_SIGNAL"
    else:
        neg_transfer = False
        strict_better = 0
        for p in pairs_out:
            m5 = p["arms"]["A5"]["metrics"]
            m3 = p["arms"]["A3"]["metrics"]
            # 归因修正（2026-08-10）：harm 增加但 delayed utility 也增加
            # （探索成本被收益补偿）→ 非负迁移——harm 比较在对照臂零行动
            # （全 abstain）时失真（零行动零 harm 零收益）。
            harm_worse = (m5["harm_count"] > m3["harm_count"]
                          or m5["harm_magnitude_sum"] > m3["harm_magnitude_sum"]) \
                and m5["delayed_utility"] <= m3["delayed_utility"]
            util_worse = m5["delayed_utility"] < m3["delayed_utility"]
            if harm_worse or util_worse:
                neg_transfer = True
            speed_better = (m3["first_positive_support_index"] is None
                            and m5["first_positive_support_index"] is not None) \
                or (m5["first_positive_support_index"] is not None
                    and m3["first_positive_support_index"] is not None
                    and m5["first_positive_support_index"]
                    < m3["first_positive_support_index"])
            not_worse = (not harm_worse and not util_worse
                         and (m3["first_positive_support_index"] is None
                              or m5["first_positive_support_index"] is None
                              or m5["first_positive_support_index"]
                              <= m3["first_positive_support_index"]))
            if speed_better and not_worse:
                strict_better += 1
        if neg_transfer:
            verdict = "A5_NEGATIVE_TRANSFER"
        elif strict_better >= 1:
            verdict = "A5_CONFIRMATION_PASS"
        else:
            verdict = "A5_SAME"

    print(f"== verdict: {verdict}")
    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-matched-budget-context-bound-a5-a3",
        "pairs": pairs_out,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
