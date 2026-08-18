"""E1_FRESH_NATURAL_TARGET_LOCAL_PROGRAM_EVOLUTION（用户裁决 2026-08-12，
装置已确认：KDD 剩余 virgin 冻结 20 支、origin 600/792/888、Fast sealed、
真实 Slow 一次、接受合法负档）。

目标：取得 post-fix fresh 自然闭环——
  自然失败 → ≤2 Typed replacement（真实 Slow Agent）→ Support →
  delayed → Draft Skill（requires_target_support）→ 下一轮
  Support-confirmed adoption → removal。

装置（用户确认）：
  - Cohort：KDD 剩余 virgin 系列（排除 K0 20 支 + K1 20 支——已用名单
    从 w1_kdd2018_frozen_cohort.jsonl / _p41.jsonl 读取）——冻结规则同
    P4.1（长度 ≥984、公开 Context outlier 信号、outlier family 静态
    合法 ≥2）——只冻结零 gain；冻结产物写
    w1_kdd2018_frozen_cohort_e1.jsonl（20 支）；
  - 轨迹：冻结 cohort 首支、origin 600/792/888（P4.1 同款三起点）；
  - Fast：sealed 确定性（真实 Fast LLM 留 E3）；
  - Slow：真实 TTHASlowAgent（AgictoChatCompletionsBackend——白名单
    约束下 propose_edit；LLM 调用预算 max_calls=6）——只触发一次；
  - 统一入口 run_online_round/open_delayed/activate_approved。

判定（预注册——不因失败换 cohort）：
  NATURAL_TARGET_LOCAL_PROGRAM_EVOLUTION_PASS : 触发 + manifest +
    replay ≥ M + delayed 批准 + 下一轮检索/不自动优先/Support 后 winner
    + removal 恢复
  NO_NATURAL_FAILURE   : 三轮无 material failure（无物可学）
  AGENT_ABSTAIN        : 触发但真实 Slow 无 manifest（last_no_proposal_
    reason 记录）
  NO_HEADROOM          : replay 不达 M（PATCH_SUPPORT_REJECTED）
  PATCH_REJECTED       : preflight/B/apply/delayed 拒绝
  ADOPTION_FAILED      : 批准但下一轮未检索/未探测/未 winner
  PROTOCOL_FAILURE     : key 缺失/数据装配失败

用法：
  python evaluation/functional/run_v1_e1_fresh_natural_evolution.py
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
import signed_radius as resolver  # noqa: E402
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
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD  # 0.005
BUDGET = 2
ORIGINS = (600, 792, 888)
R4_ORIGIN = 984
POOL = ("winsorize", "outlier_mad", "hampel_filter")
CACHE = PROJECT_ROOT / "data/kdd2018/series_cache.npz"
FROZEN_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_kdd2018_frozen_cohort_e1.jsonl"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_e1_fresh_natural_evolution_report.json"


def _freeze_e1(root: Path) -> list[dict[str, object]]:
    """KDD 剩余 virgin 冻结（零 gain——静态检查同 P4.1）。"""
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    used: set[str] = set()
    for rel in ("w1_kdd2018_frozen_cohort.jsonl",
                "w1_kdd2018_frozen_cohort_p41.jsonl"):
        used |= {json.loads(line)["series_name"] for line in
                 (root / "artifacts/functional/e2" / rel)
                 .read_text(encoding="utf-8").splitlines() if line.strip()}
    import run_v1_signed_agent_action_wiring as wiring  # noqa: PLC0415
    role_seq = ["train"] * 12 + ["support"] * 4 + ["query"] * 4
    frozen: list[dict[str, object]] = []
    for i, n in enumerate(names):
        if len(frozen) >= 20:
            break
        if str(n) in used:
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
            roster = [{"series_uid": str(n), "role": "train"}]
            ex = ScopeExecutor(roster, {str(n): np.asarray(
                values[i], dtype=np.float64)}, _config(),
                evaluate_fn=_evaluate_kdd)
            if ex.verify(steps, 600).passed:
                ok += 1
        if ok < 2:
            continue
        frozen.append({"cohort": "E1", "role": role_seq[len(frozen)],
                       "series_name": str(n), "type": "kdd2018"})
    if len(frozen) < 20:
        return []
    FROZEN_REL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen),
        encoding="utf-8")
    return frozen


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no api key"}, indent=1))
        return 0
    frozen = _freeze_e1(root)
    if not frozen:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no virgin cohort frozen"}, indent=1))
        return 0
    cache = np.load(root / CACHE, allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in frozen]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in frozen}
    series0 = vals[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, vals, _config(),
                             evaluate_fn=_evaluate_kdd)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / ".e1_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())

    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120),
        max_calls=6)
    from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: PLC0415
        AgictoChatCompletionsBackend,
    )
    slow_core = TTHAAgentCore(
        AgictoChatCompletionsBackend(client=counter, base_url=smoke.BASE_URL),
        LocalPublicToolGateway(series0[:ORIGINS[0]],
                               task_kind="forecast"))
    slow_agent = TTHASlowAgent(slow_core)

    # 同一已冻结 cohort 内依次尝试最多 5 支系列（每支独立轨迹——sealed
    # 探测；第一支出现自然失败即进入真实 Slow 路径并停止尝试后续系列——
    # 调用上限 ≤1 次；不换 cohort——用户装置语义）。
    rounds: list[Any] = []
    triggered = False
    slow_event = None
    deferred = None
    trigger_series: np.ndarray | None = None
    trigger_uid: str | None = None
    series_tried: list[str] = []
    for si, s_uid in enumerate([str(r["series_name"]) for r in frozen][:5]):
        if triggered:
            break
        series_tried.append(s_uid)
        series_i = np.asarray(vals[s_uid], dtype=np.float64)
        core = sealed.TTHAAgentCore(
            sealed.SealedProbeBackend(explore=True, operators=POOL,
                                      max_propose_candidates=3,
                                      force_pool=True),
            LocalPublicToolGateway(series_i[:ORIGINS[0]],
                                   task_kind="forecast"))
        method = TTHAMethod(sealed.TTHAFastAgent(core), h0, ())
        series_rounds: list[Any] = []
        for i, origin in enumerate(ORIGINS):
            r = run_online_round(
                method, executor, _request(series_i, vals, origin), vals,
                origin=origin,
                slow_agent=slow_agent if not triggered else None,
                controller=controller if not triggered else None,
                store=store if not triggered else None,
                card_builder=_card_builder(executor, vals, origin,
                                           "winsorize", "outlier_mad"),
                round_name=f"s{si + 1}_r{i + 1}", budget=BUDGET,
                allow_slow=not triggered, domain="kdd_cup_2018",
                period=PERIOD,
                fast_features=dict(extract_public_features(
                    series_i[:origin], task_kind="forecast")))
            open_delayed(r, executor)
            series_rounds.append(r)
            rounds.append(r)
            if r._slow_event is not None and r._slow_event.get("triggered"):
                triggered = True
                slow_event = r._slow_event
                trigger_series = series_i
                trigger_uid = s_uid
                if r._slow_event.get("stage") == "pending":
                    activate_approved(r, store)
            if r._deferred_slow is not None:
                deferred = r._deferred_slow

    # ---- 判定链（预注册）----
    def _arm(rd: Any) -> dict[str, Any]:
        return {
            "proposal_count": rd.proposal_count,
            "target_support_receipts_used": rd.target_support_receipts_used,
            "slow_replay_receipts_used": rd.slow_replay_receipts_used,
            "actual_probed_programs": rd.actual_probed_programs,
            "winner_program": rd.winner_program,
            "first_positive_support_receipt_index":
                rd.first_positive_support_receipt_index,
            "harm_count": rd.harm_count,
            "harm_magnitude": rd.harm_magnitude,
            "abstained": rd.abstained,
            "episode_ids": rd.episode_ids,
            "pending_patch_id": rd.pending_patch_id,
            "approved_skill_id": rd.approved_skill_id,
            "delayed_utility": rd.delayed_utility,
            "chosen_proposal": rd.chosen_proposal,
            "memory_resolution_status": rd.memory_resolution_status,
        }

    approved_skill_id = next((r.approved_skill_id for r in rounds
                              if r.approved_skill_id), None)
    if slow_event is None:
        verdict = "NO_NATURAL_FAILURE"
        reason = "no material failure across three rounds"
    elif slow_event.get("stage") == "no_manifest":
        verdict = "AGENT_ABSTAIN"
        reason = str(slow_agent.last_no_proposal_reason)
    elif slow_event.get("stage") in ("budget_exceeded",
                                     "manifest_preflight_failed",
                                     "apply_failed",
                                     "applicability_unreachable",
                                     "applicability_uncheckable",
                                     "no_frozen_program"):
        verdict = "PATCH_REJECTED"
        reason = f"stage={slow_event.get('stage')} " \
                 f"error={slow_event.get('error')}"
    elif slow_event.get("stage") == "support_rejected":
        verdict = "NO_HEADROOM"
        reason = f"replay gain={slow_event.get('support_gain')} < M"
    elif slow_event.get("stage") == "pending":
        dev = next((r._delayed_event for r in rounds
                    if r._delayed_event is not None), None)
        if dev is None or dev.get("stage") != "approved":
            verdict = "PATCH_REJECTED"
            reason = (f"delayed stage={dev.get('stage') if dev else None} "
                      f"dg={dev.get('delayed_gain') if dev else None}")
        else:
            # 下一轮采用验证（R4 @984——触发系列上的 virgin 窗口）
            assert trigger_series is not None and trigger_uid is not None
            skill_snap = method._active_snapshot()  # noqa: SLF001
            core4 = sealed.TTHAAgentCore(
                sealed.SealedProbeBackend(explore=True, operators=POOL,
                                          max_propose_candidates=3,
                                          force_pool=True),
                LocalPublicToolGateway(trigger_series[:R4_ORIGIN],
                                       task_kind="forecast"))
            m4 = TTHAMethod(sealed.TTHAFastAgent(core4), skill_snap, ())
            r4 = run_online_round(
                m4, executor, _request(trigger_series, vals, R4_ORIGIN),
                vals, origin=R4_ORIGIN, slow_agent=None, controller=None,
                store=None, card_builder=lambda e: {},
                round_name="r4", budget=BUDGET, allow_slow=False,
                domain="kdd_cup_2018", period=PERIOD)
            t4 = m4.last_trace
            r4_pool = list(t4.candidate_ids or ())
            skill_cands = [c for c in r4_pool
                           if c.startswith("cand_skill_")]
            agent_cands = [c for c in r4_pool if c.startswith("cand_")
                           and not c.startswith("cand_skill_")]
            retrieved = bool(skill_cands)
            not_priority = bool(
                skill_cands and agent_cands
                and r4_pool.index(skill_cands[0])
                > r4_pool.index(agent_cands[0])
                and not (t4.chosen_candidate_id or "") in skill_cands)
            probed_skill = bool(
                any(p["candidate_id"].startswith("cand_skill_")
                    for p in r4.actual_probed_programs))
            # removal：h0 同轮对照
            rem_core = sealed.TTHAAgentCore(
                sealed.SealedProbeBackend(explore=True, operators=POOL,
                                          max_propose_candidates=3,
                                          force_pool=True),
                LocalPublicToolGateway(trigger_series[:R4_ORIGIN],
                                       task_kind="forecast"))
            rem_m = TTHAMethod(sealed.TTHAFastAgent(rem_core), h0, ())
            rem = run_online_round(
                rem_m, executor, _request(trigger_series, vals, R4_ORIGIN),
                vals, origin=R4_ORIGIN, slow_agent=None, controller=None,
                store=None, card_builder=lambda e: {},
                round_name="rem", budget=BUDGET, allow_slow=False,
                domain="kdd_cup_2018", period=PERIOD)
            removal_ok = bool(
                any(p["candidate_id"].startswith("cand_skill_")
                    for p in r4.actual_probed_programs)
                and not any(p["candidate_id"].startswith("cand_skill_")
                            for p in rem.actual_probed_programs))
            adoption_ok = bool(
                retrieved and not_priority and probed_skill
                and r4.winner_program is not None)
            r4_info = {"pool": r4_pool,
                       "chosen": t4.chosen_candidate_id,
                       "retrieved": retrieved,
                       "not_priority": not_priority,
                       "probed_skill": probed_skill,
                       "winner": r4.winner_program,
                       "probes": [(p["candidate_id"], p["kind"],
                                   p.get("gain"))
                                  for p in r4.actual_probed_programs]}
            rem_info = {"probes": [(p["candidate_id"], p["kind"],
                                    p.get("gain"))
                                   for p in rem.actual_probed_programs]}
            if adoption_ok and removal_ok:
                verdict = "NATURAL_TARGET_LOCAL_PROGRAM_EVOLUTION_PASS"
                reason = (f"skill {approved_skill_id} adopted at r4: "
                          f"retrieved/not-priority/support-confirmed winner; "
                          f"removal restores")
            else:
                verdict = "ADOPTION_FAILED"
                reason = f"r4: {json.dumps(r4_info)} removal: " \
                         f"{json.dumps(rem_info)}"
    else:
        verdict = "PATCH_REJECTED"
        reason = f"stage={slow_event.get('stage')}"
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")
    print(f"== llm_calls: {counter.calls}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-e1-fresh-natural-evolution",
        "note": "E1 fresh 自然闭环（预冻结新 KDD cohort E1——virgin；"
                "Fast sealed；真实 Slow 一次；接受合法负档；不因失败换"
                "cohort）",
        "cohort": [r["series_name"] for r in frozen],
        "series_tried": series_tried,
        "trigger_series_uid": trigger_uid,
        "origins": list(ORIGINS), "r4_origin": R4_ORIGIN, "budget": BUDGET,
        "rounds": [_arm(r) for r in rounds],
        "slow_event": slow_event,
        "deferred_slow": deferred,
        "llm_api_call_count": counter.calls,
        "verdict": verdict,
        "reason": reason,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
