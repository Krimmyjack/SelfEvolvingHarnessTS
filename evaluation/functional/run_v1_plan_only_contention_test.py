"""MEMORY_DECISION_CONTENTION_P0_PLAN_ONLY（用户 Gate P0，2026-08-10）。

竞争式 plan-only 诊断：复用已暴露 UCI 80/120/200，不读取或重新评价
outcome，只测试决策。

装置（用户裁决原文）：
  - 相同 Context（window_context @792——已暴露 Target R1 决策点）；
  - 相同候选池 [bound repair_level_shift, winsorize, identity] + 相同
    候选顺序（池成员与顺序是控制变量——PlanOnlyBackend 固定提案，不因
    Memory 的 ref2/ref3 渲染剔除/降级候选）；
  - 唯一变量 = Memory：M_positive / M_remove / M_negative / M_conflict；
  - 只观察 chosen、probe 顺序（池）、abstention；不写 Skill、不算 gain。

零 outcome 读取：
  - fast_agent.prepare 全程只做静态 verify_candidate（候选合法性/selectable
    检查，无 executor.evaluate、无 gain）；
  - Source Episode 从已暴露报告
    (w1_matched_budget_context_bound_a5_a3_report.json) 重建：pair1 =
    repair 双负 (support -0.3602 / delayed -0.1053) → RISK_PRIOR；pair2 =
    repair 双正 (+0.0341 / +0.0138) → POSITIVE_PRIOR；同 op 两 Episode →
    成对判定 CONFLICT。不产生任何新 outcome。

Memory 档位：
  M_positive = [pair2 Episode]        → ref1 "Probe them first"
  M_remove   = []                     → 无 Reference 渲染（A3 语义）
  M_negative = [pair1 Episode]        → ref3 "Avoid unless evidence contradicts"
  M_conflict = [pair1, pair2]         → ref2 "mixed evidence…Treat as risk"

每档 × 每 Context 重复 2 次（真实 LLM temp=0；重复不一致 → LLM_VARIANCE，
不能把新数据结果归因给 Memory）。

判定（用户分支，development 诊断不承重 A5/A3）：
  MEMORY_ACTION_SIGNAL       M_positive/M_remove/M_negative 间行动差异
                             （chosen 或 abstain 不同）且重复稳定
  MEMORY_IGNORED_NO_SIGNAL   三档行为完全相同且重复稳定
  LLM_VARIANCE               同档重复不稳定（先修/记录 variance 再归因）

用法：
  python evaluation/functional/run_v1_plan_only_contention_test.py
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
import run_v1_slow_path_smoke as smoke  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402

DOMAIN = "uci_electricity_load_diagrams"
PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
DECISION_ORIGIN = 792  # 已暴露 Target R1 决策点
CONTEXTS = (80, 120, 200)  # 已暴露 Target cohort offsets（V1 pair1/pair2 + V2 pair3）
FIXED_POOL = ("repair_level_shift", "winsorize")  # 用户指定池（固定顺序）
MEMORY_LEVELS = ("M_positive", "M_remove", "M_negative", "M_conflict")
REPEATS = 2  # LLM temp=0 稳定性检查
REPORT_SRC_REL = Path(
    "artifacts/functional/e2/w1_matched_budget_context_bound_a5_a3_report.json")
REPORT_OUT_REL = Path(
    "artifacts/functional/e2/w1_plan_only_contention_report.json")


class PlanOnlyBackend(sealed.LLMSelectBackend):
    """P0 装置：propose 固定提案 [repair_bound, winsorize]（忽略 ref1 单提
    案与 ref2/ref3 deprioritize——池成员/顺序是控制变量）；inspect/select
    继承（select 含 Reference 渲染段 + 固定顺序候选 + 真实 LLM）。"""

    def complete(self, request: Any) -> Any:
        if request.stage != "propose":
            return super().complete(request)
        self.requests.append(request)
        instruction = self.extract_instruction(request.messages)
        ref1 = self._reference_ops(instruction, 1)
        ref2 = self._reference_ops(instruction, 2)
        ref3 = self._reference_ops(instruction, 3)
        self._deprioritized = list(dict.fromkeys([*ref2, *ref3]))
        # 固定提案（FIXED_POOL 顺序；explored 只减不增——单轮决策）
        ops_list = [o for o in FIXED_POOL if o not in self._explored]
        ops_list = ops_list[:self._max_propose]
        self._pending_op = ops_list[0] if ops_list else None
        candidates = [self._cand(request, o) for o in ops_list]
        payload = {"candidates": candidates}
        return wiring.AgentResponse.valid(
            {"schema_version": "agent-envelope/1", "kind": "stage_result",
             "stage": "propose", "payload": payload},
            raw_response={"id": "plan-only-propose"},
        )


def _reconstruct_source_episodes(root: Path) -> dict[str, list[Any]]:
    """从已暴露报告重建 Source Episode（零新 outcome）。

    pair1（src 240）→ repair 双负（RISK_PRIOR）；pair2（src 280）→ repair
    双正（POSITIVE_PRIOR）。program params/support/delayed gain 全部来自
    报告记录（运行时真实绑定参数）。"""
    report = json.loads((root / REPORT_SRC_REL).read_text(encoding="utf-8"))
    sealed._set_domain(DOMAIN)
    by_name: dict[str, list[Any]] = {}
    for pair in report["pairs"]:
        name = str(pair["name"])
        src_offset = int(pair["src_offset"])
        (src_roster, src_values, _, _) = sealed._virgin_roster(
            root, offset=src_offset)
        series0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                             dtype=np.float64)
        # 真实 support 轮记录（第 1 个 support attempt）
        support_attempt = next(
            (a for a in pair["source"]["attempts"]
             if a.get("kind") == "support"), None)
        assert support_attempt is not None, f"{name}: no source support"
        steps = support_attempt["program"]
        assert steps[0]["op"] == "repair_level_shift", name
        params = dict(steps[0]["params"])
        sg = float(support_attempt["gain"])
        dg = float(support_attempt["delayed_gain"])
        ep = tll.write_target_episode(
            domain=DOMAIN, op="repair_level_shift",
            episode_id_suffix=f"_p0_{name}",
            program_steps=[{"op": "repair_level_shift", "params": params}],
            support_gain=sg,
            delayed_gain=None,
            support_context=dict(resolver.window_context(
                src_values, sealed.SOURCE_ORIGIN, PERIOD)))
        ep = tll.update_delayed_status(
            ep, dg,
            delayed_context=dict(resolver.window_context(
                src_values, sealed.SOURCE_DELAYED, PERIOD)))
        by_name[name] = [ep]
        print(f"== reconstructed {name} source episode: op="
              f"{getattr(ep, 'workflow_signature', '?')} "
              f"support={sg:.4f} delayed={dg:.4f} "
              f"relation={getattr(ep, 'relation', '?')}")
    return by_name


def _memory_for(level: str, episodes: dict[str, list[Any]]) -> list[Any]:
    if level == "M_positive":
        return list(episodes["pair2"])
    if level == "M_remove":
        return []
    if level == "M_negative":
        return list(episodes["pair1"])
    if level == "M_conflict":
        return [*episodes["pair1"], *episodes["pair2"]]
    raise AssertionError(level)


def _decide(root: Path, h0: Any, series0: np.ndarray,
            values: Mapping[str, Any], memory: Sequence[Any],
            counter: Any, context_plain: Mapping[str, object],
            *, label: str) -> dict[str, Any]:
    """一次 plan-only 决策（零 evaluate/写回）。返回池/chosen/渲染/LLM 原始。"""
    backend = PlanOnlyBackend(
        explore=True, operators=FIXED_POOL,
        client=counter, context_plain=dict(context_plain),
        max_propose_candidates=len(FIXED_POOL))
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend,
            LocalPublicToolGateway(series0[:DECISION_ORIGIN],
                                   task_kind="forecast"))),
        h0, tuple(memory))
    method.bind_round_data(series0[:DECISION_ORIGIN], task_kind="forecast")
    result = method.prepare(sealed._request(series0, values, DECISION_ORIGIN))
    trace = method.last_trace
    chosen = trace.chosen_candidate_id
    instruction = ""
    ref_section = ""
    for req in backend.requests:
        for m in req.messages:
            c = m.get("content") if isinstance(m, dict) else None
            if isinstance(c, str) and "The following references" in c:
                instruction = c
                _s = c.find("The following references")
                ref_section = c[_s:_s + 900].strip()
                break
        if instruction:
            break
    raw = ""
    if backend._select_logs:  # noqa: SLF001
        raw = str(backend._select_logs[-1].get("raw", ""))[:600]
    return {
        "label": label,
        "pool": list(trace.candidate_ids),
        "chosen": chosen,
        "abstain": bool(chosen in ("", "identity")),
        "reference_section": ref_section,
        "llm_raw": raw,
    }


def _signed_diagnostic(values: Mapping[str, Any], memory: Sequence[Any],
                       request: Any) -> dict[str, Any]:
    """resolve_order 诊断（原机制下的 probe 顺序/判定——只读，不参与装置）。
    空 memory 不渲染 Reference，返回空。"""
    if not memory:
        return {"rendered": False}
    observed = dict(getattr(request, "observed_pattern_spec", {}) or {})
    query_ctx = {k: float(v) for k, v in observed.items()
                 if str(k).startswith(("recent.", "change."))
                 and isinstance(v, (int, float))}
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import _allowed_operators
    allowed = tuple(_allowed_operators(request))
    order, signed = resolver.resolve_order(
        query_context=query_ctx,
        episodes=tuple(memory),
        operators=allowed,
        material_threshold=M,
        task_consumer_key="forecast|ridge|sMASE",
        allowed_operators=allowed,
    )
    return {
        "rendered": True,
        "radius_mode": signed["summary"]["radius_mode"],
        "n_historical_contexts": signed["summary"]["n_historical_contexts"],
        "verdict_counts": signed["summary"]["verdict_counts"],
        "repair_verdict": (signed["per_op"].get("repair_level_shift")
                           or {}).get("verdict"),
        "order": list(order),
    }


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

    episodes = _reconstruct_source_episodes(root)

    results: dict[str, Any] = {}
    pool_consistent = True
    for offset in CONTEXTS:
        sealed._set_domain(DOMAIN)
        (_, _, tgt_roster, tgt_values) = sealed._virgin_roster(
            root, offset=offset)
        series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
        ctx = dict(resolver.window_context(tgt_values, DECISION_ORIGIN,
                                           PERIOD))
        ctx["bound_period"] = float(PERIOD)
        request = sealed._request(series0, tgt_values, DECISION_ORIGIN)
        per_level: dict[str, Any] = {}
        for level in MEMORY_LEVELS:
            memory = _memory_for(level, episodes)
            diag = _signed_diagnostic(tgt_values, memory, request)
            decisions = [
                _decide(root, h0, series0, tgt_values, memory, counter, ctx,
                        label=f"offset{offset}_{level}_rep{rep + 1}")
                for rep in range(REPEATS)
            ]
            for d in decisions:
                if d["pool"] != decisions[0]["pool"]:
                    pool_consistent = False
            per_level[level] = {
                "memory_episodes": [
                    {"episode_id": getattr(e, "episode_id", "?"),
                     "relation": getattr(e, "relation", "?")}
                    for e in memory],
                "signed_diagnostic": diag,
                "decisions": decisions,
                "chosen_set": sorted({d["chosen"] for d in decisions}),
                "abstain_any": any(d["abstain"] for d in decisions),
            }
            print(f"== offset{offset} {level}: "
                  f"chosen={per_level[level]['chosen_set']} "
                  f"pool={decisions[0]['pool']} "
                  f"verdict={diag.get('repair_verdict')}")
            for d in decisions:
                print(f"   {d['label']}: chosen={d['chosen']} "
                      f"abstain={d['abstain']}")
                if d["reference_section"]:
                    print("   ref: "
                          + d["reference_section"].replace("\n", " ")[:240])
        results[str(offset)] = per_level

    # ---- 判定（用户分支；稳定优先，再查档间差异）----
    variance = False
    for offset in CONTEXTS:
        for level in MEMORY_LEVELS:
            if len(results[str(offset)][level]["chosen_set"]) > 1:
                variance = True
    action_signal = False
    if not variance:
        for offset in CONTEXTS:
            key_levels = ("M_positive", "M_remove", "M_negative")
            chosen_by_level = {
                lv: results[str(offset)][lv]["decisions"][0]["chosen"]
                for lv in key_levels}
            abstain_by_level = {
                lv: results[str(offset)][lv]["decisions"][0]["abstain"]
                for lv in key_levels}
            if (len(set(chosen_by_level.values())) > 1
                    or len(set(abstain_by_level.values())) > 1):
                action_signal = True
    if variance:
        verdict = "LLM_VARIANCE"
    elif action_signal:
        verdict = "MEMORY_ACTION_SIGNAL"
    else:
        verdict = "MEMORY_IGNORED_NO_SIGNAL"
    print(f"== pool consistent across memory levels: {pool_consistent}")
    print(f"== verdict: {verdict}")

    out = root / REPORT_OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-plan-only-contention-p0",
        "dataset": DOMAIN,
        "decision_origin": DECISION_ORIGIN,
        "fixed_pool": list(FIXED_POOL),
        "memory_levels": list(MEMORY_LEVELS),
        "repeats": REPEATS,
        "contexts": list(CONTEXTS),
        "results": results,
        "pool_consistent_across_memory": pool_consistent,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
