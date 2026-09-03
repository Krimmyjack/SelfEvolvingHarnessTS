"""INTEGRATION_VERTICAL_LOOP（纵向集成——第一步，用户裁决 2026-08-12）。

一条纵向能力闭环（已暴露数据、真实 LLM）：
  真实 LLM Fast（inspect/propose/select）
  → A5：Runtime-owned 双槽（Slot P = signed prior 结构化 + Slot E =
     LLM 当前 Context 探索候选——探索槽空 → 协议失败）
  → Target Support 探测（预算 2）
  → 正向 winner（第一正向即停）
  → Fast winner → Target-local Draft Skill（方法层 handle_fast_winner——
     machine manifest——宽 Scope → requires_target_support=true）
  → delayed 批准（只 winner 开 delayed——未部署候选无反事实 delayed）
  → snapshot 更新（activate_approved）
  → 下一正常入口检索/探测/确认
  → removal（h0 同轮 plan-only 行为翻转）

验收（预注册）：
  C1 A5 双槽填充（池含 cand_prior_* 与至少一个 agent 候选）
  C2 A3 无 prior 槽（池只 agent 候选）
  C3 memory_resolution_status（A5 rendered / A3 no_memory）
  C4 Fast winner → Draft Skill 批准（pending → delayed approved →
     snapshot 含 fast_winner_* skill 且 requires_target_support=true）
  C5 下一轮检索/探测（skill 在池、被探测——不自动优先）
  C6 removal 行为翻转（h0 同轮无 skill 候选）
  C7 预算 ≤2（含 fast skill replay 不计额外）

协议失败档（不进入后续）：
  EXPLORATION_SLOT_EMPTY / MEMORY_NOT_RENDERED / LLM_BUDGET_EXCEEDED /
  PROTOCOL_FAILURE

零新数据（KDD T117 @792/@840/@888 已暴露）；真实 LLM（CountingClient
max_calls 限制）；不重复投票。

用法：
  python evaluation/functional/run_v1_integration_vertical_loop.py
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
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_memory_gate import _monash_source_episodes  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)
from run_v1_operational_self_evolution_loop import (  # noqa: E402
    _card_builder,
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
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    activate_approved,
    open_delayed,
    run_online_round,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
BUDGET = 2
R1_ORIGIN = 792
R2_ORIGIN = 888
POOL = ("winsorize", "outlier_mad", "hampel_filter")
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_integration_vertical_loop_report.json"


def _load(root: Path) -> dict[str, Any]:
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    return {"roster": roster, "values": vals}


def _llm_method(root: Path, snapshot: Any, series0: np.ndarray, origin: int,
                counter: Any, memory: tuple = (), *,
                runtime_prior_slot: bool = False) -> Any:
    from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: PLC0415
        AgictoChatCompletionsBackend,
    )
    core = TTHAAgentCore(
        AgictoChatCompletionsBackend(client=counter, base_url=smoke.BASE_URL),
        LocalPublicToolGateway(series0[:origin], task_kind="forecast"))
    return TTHAMethod(TTHAFastAgent(core), snapshot, memory)


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    # 审查修正（2026-08-12）：LLM 预算超限（CountingClient raise
    # RuntimeError——counter.exceeded 不存在）→ 捕获为预注册 verdict
    # LLM_BUDGET_EXCEEDED（不崩溃）
    try:
        _run_all(root, api_key)
    except RuntimeError as exc:
        if "LLM call budget exceeded" in str(exc):
            print(json.dumps({"verdict": "LLM_BUDGET_EXCEEDED",
                              "reason": str(exc)}, indent=1))
            return 0
        raise
    return 0


def _run_all(root: Path, api_key: str) -> None:
    """主运行体（LLM 预算异常由 main 捕获——预注册 verdict）。"""
    cohort = _load(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    store = SnapshotStore(root / ".intg_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    source = _monash_source_episodes(root)

    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                      timeout=120), max_calls=20)
    arms: dict[str, Any] = {}

    # ---- A5：Source + Runtime-owned 双槽 ----
    m5 = _llm_method(root, h0, series0, R1_ORIGIN, counter,
                     tuple(source), runtime_prior_slot=True)
    r5 = run_online_round(
        m5, executor, _request(series0, values, R1_ORIGIN), values,
        origin=R1_ORIGIN, slow_agent=None, controller=controller,
        store=store,
        card_builder=_card_builder(executor, values, R1_ORIGIN,
                                   "winsorize", "outlier_mad"),
        round_name="a5_r1", budget=BUDGET, allow_slow=False,
        domain="kdd_cup_2018", period=24,
        fast_features=dict(extract_public_features(
            series0[:R1_ORIGIN], task_kind="forecast")),
        allow_fast_skill=True, runtime_prior_slot=True)
    t5 = m5.last_trace
    arms["A5_r1"] = {
        "pool": list(t5.candidate_ids or ()),
        "chosen": t5.chosen_candidate_id,
        "memory_resolution": t5.memory_resolution_status,
        "winner": r5.winner_program,
        "probes": [(p["candidate_id"], p.get("gain"))
                   for p in r5.actual_probed_programs],
        "receipts": r5.target_support_receipts_used,
        "fast_skill_event": r5._fast_skill_event,
        "pending_patch": r5.pending_patch_id,
    }
    open_delayed(r5, executor)
    arms["A5_r1"]["delayed_utility"] = r5.delayed_utility
    arms["A5_r1"]["approved_skill_id"] = r5.approved_skill_id
    arms["A5_r1"]["delayed_event"] = r5._delayed_event
    if r5.approved_skill_id is not None:
        activate_approved(r5, store)

    # ---- A3：空 Source（真实 LLM 自主探索）----
    m3 = _llm_method(root, h0, series0, R1_ORIGIN, counter, ())
    r3 = run_online_round(
        m3, executor, _request(series0, values, R1_ORIGIN), values,
        origin=R1_ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="a3_r1", budget=BUDGET,
        allow_slow=False, domain="kdd_cup_2018", period=24,
        fast_features=dict(extract_public_features(
            series0[:R1_ORIGIN], task_kind="forecast")),
        allow_fast_skill=False, runtime_prior_slot=False)
    t3 = m3.last_trace
    arms["A3_r1"] = {
        "pool": list(t3.candidate_ids or ()),
        "chosen": t3.chosen_candidate_id,
        "memory_resolution": t3.memory_resolution_status,
        "winner": r3.winner_program,
        "probes": [(p["candidate_id"], p.get("gain"))
                   for p in r3.actual_probed_programs],
        "receipts": r3.target_support_receipts_used,
    }

    # ---- R2 @888：下一正常入口检索/探测/确认 + removal（plan-only）----
    r2_info: dict[str, Any] = {}
    if r5.approved_skill_id is not None:
        skill_snap = m5._active_snapshot()  # noqa: SLF001
        m2 = _llm_method(root, skill_snap, series0, R2_ORIGIN, counter, ())
        r2 = run_online_round(
            m2, executor, _request(series0, values, R2_ORIGIN), values,
            origin=R2_ORIGIN, slow_agent=None, controller=None, store=None,
            card_builder=lambda e: {}, round_name="r2", budget=BUDGET,
            allow_slow=False, domain="kdd_cup_2018", period=24,
            fast_features=dict(extract_public_features(
                series0[:R2_ORIGIN], task_kind="forecast")),
            allow_fast_skill=False, runtime_prior_slot=False)
        t2 = m2.last_trace
        pool2 = list(t2.candidate_ids or ())
        skill_cands = [c for c in pool2 if c.startswith("cand_skill_")]
        agent_cands = [c for c in pool2 if c.startswith("cand_")
                       and not c.startswith("cand_skill_")]
        # 区分"检索（view 层——applicability 匹配）"与"可用（verify 通过
        # 进池）"；CONTEXT_BOUND_REBINDING（2026-08-12）后：skill 候选
        # 每轮按当前 features 重新绑定（verify 用候选自身区域）——
        # CandidatePool 按 program sha 去重（与 LLM 提案程序相同时由
        # agent 候选承载——正确语义）。
        retrieved_ids = list(t2.retrieved_skill_ids or ())
        r2_fe = dict(extract_public_features(series0[:R2_ORIGIN],
                                             task_kind="forecast"))
        r2_info = {
            "pool": pool2,
            "chosen": t2.chosen_candidate_id,
            "inspected_regions": list(t2.inspected_regions or ()),
            "retrieved_skill_ids": retrieved_ids,
            "skill_retrieved": bool(
                r5.approved_skill_id in retrieved_ids),
            "skill_verified_into_pool": bool(skill_cands),
            "skill_not_priority": bool(
                skill_cands and agent_cands
                and pool2.index(skill_cands[0]) > pool2.index(agent_cands[0])
                and not (t2.chosen_candidate_id or "") in skill_cands),
            "skill_probed": bool(
                any(p["candidate_id"].startswith("cand_skill_")
                    for p in r2.actual_probed_programs)),
            "winner": r2.winner_program,
            "probes": [(p["candidate_id"], p.get("gain"))
                       for p in r2.actual_probed_programs],
            "receipts": r2.target_support_receipts_used,
            "r2_feature_bindings": {
                "region_start_fraction": r2_fe.get(
                    "estimated_region_start_fraction"),
                "region_end_fraction": r2_fe.get(
                    "estimated_region_end_fraction")},
        }
        # removal：h0 同轮 plan-only（prepare 后比较池——不读 outcome；
        # sealed 确定性——行为翻转只查池差异——省 LLM 调用）
        import run_v1_sealed_a5_a3 as sealed  # noqa: PLC0415
        rem_core = sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=POOL,
                                      max_propose_candidates=3,
                                      force_pool=True),
            LocalPublicToolGateway(series0[:R2_ORIGIN],
                                   task_kind="forecast"))
        m_rem = TTHAMethod(sealed.TTHAFastAgent(rem_core), h0, ())
        m_rem.bind_round_data(series0[:R2_ORIGIN], task_kind="forecast")
        m_rem.prepare(_request(series0, values, R2_ORIGIN))
        t_rem = m_rem.last_trace
        rem_pool = list(t_rem.candidate_ids or ())
        r2_info["removal_pool"] = rem_pool
        r2_info["removal_no_skill"] = bool(
            not any(c.startswith("cand_skill_") for c in rem_pool))
    arms["R2"] = r2_info

    # ---- 判定（预注册）----
    a5_pool = arms["A5_r1"]["pool"]
    a5_prior = [c for c in a5_pool if c.startswith("cand_prior_")]
    # LLM 候选 ID 无固定前缀（真实 LLM 生成——如 "repair_level_shift_local"）
    # ——非 identity/非 prior/非 skill 即 agent 探索候选
    a5_agent = [c for c in a5_pool if c != "identity"
                and not c.startswith("cand_prior_")
                and not c.startswith("cand_skill_")]
    checks: dict[str, bool] = {
        "C1_A5_two_slot_filled": bool(
            a5_prior and a5_agent
            and arms["A5_r1"]["memory_resolution"] == "rendered"),
        "C2_A3_no_prior": bool(
            not any(c.startswith("cand_prior_")
                    for c in arms["A3_r1"]["pool"])
            and arms["A3_r1"]["memory_resolution"] == "no_memory"),
        "C3_fast_winner_skill_approved": bool(
            arms["A5_r1"]["approved_skill_id"] is not None
            and arms["A5_r1"]["delayed_event"]
            and arms["A5_r1"]["delayed_event"].get("stage") == "approved"),
        "C4_skill_draft_guard": bool(
            arms["A5_r1"]["approved_skill_id"] is not None
            and _guard_written(m5._active_snapshot(),  # noqa: SLF001
                               arms["A5_r1"]["approved_skill_id"])),
        # C5（CONTEXT_BOUND_REBINDING 后）：skill 的重新绑定程序实例进池
        # 并被探测（CandidatePool 按 program sha 去重——可由 agent 候选
        # 承载同一程序——skill_probed 用匹配语义）
        "C5_next_round_retrieval": bool(
            r2_info.get("skill_retrieved") is True
            and r2_info.get("skill_verified_into_pool") is True
            and r2_info.get("skill_not_priority") is True
            and r2_info.get("skill_probed") is True),
        "C6_removal_flips": bool(r2_info.get("removal_no_skill") is True),
        "C7_budget_le_2": bool(
            arms["A5_r1"]["receipts"] <= BUDGET
            and arms["A3_r1"]["receipts"] <= BUDGET
            and r2_info.get("receipts", 0) <= BUDGET),
    }
    if not checks["C1_A5_two_slot_filled"]:
        verdict = "EXPLORATION_SLOT_EMPTY"
    elif not checks["C2_A3_no_prior"]:
        verdict = "MEMORY_NOT_RENDERED"
    elif all(checks.values()):
        verdict = "INTEGRATION_VERTICAL_LOOP_PASS"
    else:
        first = next(k for k in ("C3_fast_winner_skill_approved",
                                 "C4_skill_draft_guard",
                                 "C5_next_round_retrieval",
                                 "C6_removal_flips", "C7_budget_le_2")
                     if not checks[k])
        verdict = f"INTEGRATION_FAILED_{first}"
    print(f"== arms: {json.dumps(arms, ensure_ascii=False, indent=1)}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== llm_calls: {counter.calls}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-integration-vertical-loop",
        "note": "纵向集成第一步（已暴露 KDD T117；真实 LLM；Runtime-owned "
                "双槽 + Fast winner Skill 生命周期；零新数据）",
        "arms": arms,
        "checks": checks,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")


def _guard_written(snapshot: Any, skill_id: str) -> bool:
    for s in snapshot.skills:
        if s.skill_id == skill_id:
            return bool(dict(s.risk_guards or {}).get(
                "requires_target_support") is True)
    return False


if __name__ == "__main__":
    raise SystemExit(main())


def _guard_written(snapshot: Any, skill_id: str) -> bool:
    for s in snapshot.skills:
        if s.skill_id == skill_id:
            return bool(dict(s.risk_guards or {}).get(
                "requires_target_support") is True)
    return False


if __name__ == "__main__":
    raise SystemExit(main())
