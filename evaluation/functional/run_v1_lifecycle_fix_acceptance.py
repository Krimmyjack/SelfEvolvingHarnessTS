"""第一步验收：候选与反馈生命周期 first fault 修复（审核 2026-08-09）。

修复：
  1. open_delayed 显式 episode_id——本轮无 Episode（abstain）→ 返回 None、
     不更新任何历史/Source Episode（删除"默认取 Memory 最后一条"）；
  2. propose 只记 pending、select 实际选中才记 explored；LLM/selector
     abstain 不消耗 → 下一轮仍可提案；verifier 拒绝记 rejected。

验收（确定性模拟 abstain，零 LLM，不评估额外 gain 语义）：
  A. R1 abstain 后 open_delayed(None) → None，Source Episode 未被修改；
  B. Reference 1 算子在未执行时下一轮仍出现在候选（abstain 不消耗）；
  C. Reference 与候选 Program 逐项对应（Reference 1 winsorize ↔ 池含
     cand_winsorize）。

用法：
  python evaluation/functional/run_v1_lifecycle_fix_acceptance.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
import signed_radius as resolver  # noqa: E402

from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import TTHAFastAgent  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import LocalPublicToolGateway  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402


class AbstainBackend(sealed.SealedProbeBackend):
    """模拟 LLM abstain：propose 正常（pending 记录），select 强制 identity。"""

    def complete(self, request: Any) -> Any:
        if request.stage == "select":
            self._pending_op = None
            return wiring.AgentResponse.valid(
                {"schema_version": "agent-envelope/1", "kind": "stage_result",
                 "stage": "select",
                 "payload": {"chosen_candidate_id": "identity",
                             "verification_actions":
                                 ["public_evidence_insufficient"]}},
                raw_response={"id": "abstain-sim"},
            )
        return super().complete(request)


def main() -> int:
    root = PROJECT_ROOT
    sealed._set_domain("metr_la")
    config = sealed._config()
    (src_roster, src_values, tgt_roster, tgt_values) = sealed._virgin_roster(
        root, offset=40)
    tgt_series0 = np.asarray(tgt_values[tgt_roster[0]["series_uid"]],
                             dtype=np.float64)
    src_series0 = np.asarray(src_values[src_roster[0]["series_uid"]],
                             dtype=np.float64)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)

    # Source 探测（确定性）+ 显式 delayed → Source Memory（POSITIVE winsorize）
    src_executor = sealed.ScopeExecutor(src_roster, src_values, config,
                                        evaluate_fn=sealed.v6._evaluate)
    src_observed = dict(resolver.window_context(src_values, 600, 24))
    src_observed["bound_period"] = 24.0
    actionable_src = sealed._actionable_ops(root, src_series0, 600,
                                            src_observed)
    src_method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=actionable_src),
            LocalPublicToolGateway(src_series0[:600], task_kind="forecast"))),
        h0, ())
    rd = sealed.probe_round(src_method, src_executor, src_series0, src_values,
                            600, round_name="src600")
    for i, p in enumerate(rd["probes"]):
        if p.get("kind") == "support":
            sealed.open_delayed(src_method, src_executor, src_series0,
                                src_values, 600, round_name=f"src{i}",
                                episode_id=p["episode_id"])
    src_memory = tuple(src_method._experience_episodes)
    src_ep = src_memory[-1]  # winsorize POSITIVE（已开 delayed）
    baseline_gain = float(src_ep.delayed_response.get("gain"))
    print(f"== source memory: {[getattr(e, 'relation', '?') for e in src_memory]}"
          f"; winsorize delayed baseline={baseline_gain}")

    # R1：abstain 模拟（有 Reference 1 + 候选但 LLM 选择 identity）
    observed = dict(resolver.window_context(tgt_values, 792, 24))
    observed["bound_period"] = 24.0
    actionable_tgt = sealed._actionable_ops(root, tgt_series0, 792, observed)
    backend = AbstainBackend(explore=True, operators=actionable_tgt)
    method = sealed.TTHAMethod(
        sealed.TTHAFastAgent(sealed.TTHAAgentCore(
            backend, LocalPublicToolGateway(tgt_series0[:792],
                                            task_kind="forecast"))),
        h0, src_memory)
    r1 = sealed.probe_round(method, sealed.ScopeExecutor(
        tgt_roster, tgt_values, config, evaluate_fn=sealed.v6._evaluate),
        tgt_series0, tgt_values, 792, round_name="r1_abstain")
    print(f"== R1 (abstain sim): {r1['probes']}")
    n_src_before = len(tuple(method._experience_episodes))

    # A. abstain 后 open_delayed(None) → None，且 Source Episode 未被修改
    d1 = sealed.open_delayed(method, sealed.ScopeExecutor(
        tgt_roster, tgt_values, config, evaluate_fn=sealed.v6._evaluate),
        tgt_series0, tgt_values, 792, round_name="r1_abstain")
    src_after = tuple(method._experience_episodes)[-1]
    gain_after = float(src_after.delayed_response.get("gain"))
    checks: dict[str, Any] = {
        "A1_abstain_delayed_returns_none": d1 is None,
        "A2_no_new_episode_after_abstain": bool(
            len(tuple(method._experience_episodes)) == n_src_before),
        "A3_source_episode_untouched": bool(gain_after == baseline_gain),
    }
    print(f"== A: d1={d1} source gain after={gain_after} (baseline={baseline_gain})")

    # B/C. R2：abstain 未消耗 → Reference 1 winsorize 仍在候选（ref1 未
    # explored → propose 仍提 winsorize）
    method.bind_round_data(tgt_series0[:888], task_kind="forecast")
    r2 = sealed.probe_round(method, sealed.ScopeExecutor(
        tgt_roster, tgt_values, config, evaluate_fn=sealed.v6._evaluate),
        tgt_series0, tgt_values, 888, round_name="r2_after_abstain")
    print(f"== R2 probes: {[(p['chosen'], p.get('op')) for p in r2['probes']]}")
    # 候选池从 trace 取（abstain entry 无 op 字段）
    r2_trace = method.last_trace
    r2_pool_ops = set()
    for cid, st in (r2_trace.candidate_program_steps or {}).items():
        for s in st:
            r2_pool_ops.add(str(s["op"]) if isinstance(s, Mapping) else str(s[0]))
    print(f"== R2 pool ops: {sorted(r2_pool_ops)}")
    checks["B_ref1_op_still_candidate_after_abstain"] = bool(
        "winsorize" in r2_pool_ops)
    # instruction 的 Reference 1（R2 第一次 prepare 后）
    instr = backend.extract_instruction(backend.requests[-1].messages)
    checks["C_reference1_in_instruction"] = bool(
        "Reference 1" in instr and "winsorize" in instr)

    passed = all(v is True for v in checks.values())
    verdict = ("LIFECYCLE_FIRST_FAULT_FIX_PASS" if passed
               else "LIFECYCLE_FIRST_FAULT_FIX_FAIL")
    print(f"== checks: {json.dumps(checks, ensure_ascii=False, indent=1)}")
    print(f"== verdict: {verdict}")

    out = root / Path("artifacts/functional/e2/w1_lifecycle_fix_acceptance_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "experiment_id": "v1-lifecycle-first-fault-fix-acceptance",
        "fixes": ["open_delayed 显式 episode_id（abstain → None 不更新历史）",
                  "propose pending / select 选中才 explored（abstain 不消耗）",
                  "verifier 拒绝记 rejected"],
        "checks": checks,
        "verdict": verdict,
        "llm_api_call_count": 0,
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"== report -> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
