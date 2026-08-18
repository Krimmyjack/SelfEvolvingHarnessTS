"""E3.1_FRESH_MATCHED_BUDGET_A5_TWO_SLOT_VS_A3（用户裁决 2026-08-12，
纵向集成 7/7 接通后立即执行）。

装置（用户裁决）：
  - Target：KDD 剩余 virgin（排除 K0/K1/E1 已用名单——SEALED 语义：
    从未被任何实验消费）——固定顺序（cache names）取首个满足冻结条件
    的 20 支——只冻结零 gain（写 w1_kdd2018_frozen_cohort_e31.jsonl）；
  - Source：冻结已有 Monash 正向/负向/冲突 Episode 组合（P4.0 正例 +
    signswap 负/冲突）——运行后不挑选；
  - 真实 Fast LLM（inspect/propose/select 全路径）；A5（Source +
    runtime_prior_slot 双槽）/ A3（空）——唯一差异 Source Experience；
  - 固定三轮适配（600/792/888）+ 一轮正常采用检查（984）；
  - 每轮 Target Support ≤2（Slow replay 计入预算）；四轮累计 ≤8；
  - 第一正向即停；delayed 只开实际 winner；
  - Slow Agent 每臂最多触发一次；
  - Fast winner → Draft Skill（allow_fast_skill——rebinding 后跨轮
    重新实例化可用）；
  - removal：R4 同 Context plan-only 行为翻转（sealed 池差异）。

主指标：
  feedback_to_reliable_local_skill（形成可靠 skill 的累计 Support 数）/
  harm_before_recovery / final_delayed_utility /
  normal_entry_adoption_delta / abstention。

判定（预注册）：
  TRANSFER_PASS       : A5 更少反馈（累计 Support 更少或 first positive
                        更早）或同预算下 harm 更低，且 delayed 不劣且
                        采用闭环成立
  NO_SIGNAL           : Memory 已真实生效但轨迹/指标相同
  NEGATIVE_TRANSFER   : A5 试错/harm/delayed 明显更差
  PROTOCOL_INCONCLUSIVE : 双槽/Memory/provider/预算协议失败
  CONTENT_INCONCLUSIVE  : 真实 LLM 输出使两臂无法因果解释
  （单 Target PASS 只称"跨数据集迁移候选证据"——≥2 Target 同向才
  声称 cross-domain benefit）

用法：
  python evaluation/functional/run_v1_e31_fresh_a5_two_slot.py
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
import run_v1_signed_agent_action_wiring as wiring  # noqa: E402
from run_v1_kdd2018_memory_gate import (  # noqa: E402
    _monash_source_episodes,
    _signswap,
)
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)
from run_v1_operational_self_evolution_loop import (  # noqa: E402
    ReplaySlowAgent,
    _card_builder,
    _skill_manifest,
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
ORIGINS = (600, 792, 888)
R4_ORIGIN = 984
POOL = ("winsorize", "outlier_mad", "hampel_filter")
FROZEN_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_kdd2018_frozen_cohort_e31.jsonl"
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_e31_fresh_a5_two_slot_report.json"


def _freeze_e31(root: Path) -> list[dict[str, object]]:
    """KDD 剩余 virgin 冻结（SEALED 语义——排除 K0/K1/E1 已用名单——
    零 gain）。"""
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    used: set[str] = set()
    for rel in ("w1_kdd2018_frozen_cohort.jsonl",
                "w1_kdd2018_frozen_cohort_p41.jsonl",
                "w1_kdd2018_frozen_cohort_e1.jsonl"):
        p = root / "artifacts/functional/e2" / rel
        if p.is_file():
            used |= {json.loads(line)["series_name"] for line in
                     p.read_text(encoding="utf-8").splitlines()
                     if line.strip()}
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
        frozen.append({"cohort": "E31", "role": role_seq[len(frozen)],
                       "series_name": str(n), "type": "kdd2018"})
    if len(frozen) < 20:
        return []
    FROZEN_REL.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in frozen),
        encoding="utf-8")
    return frozen


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


def _run_arm(root: Path, executor: Any, series0: np.ndarray,
             values: Mapping[str, Any], counter: Any, memory: tuple,
             arm: str, *, prior_slot: bool) -> dict[str, Any]:
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    store = SnapshotStore(root / f".e31_store_{arm.lower()}")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    # Slow：ReplaySlowAgent（确定性——E3.1 的真实 LLM 预算留给 Fast；
    # 预注册 manifest——winsorize 失败 → outlier_mad 替代——同 P3 场景 B）
    slow_manifest = _skill_manifest(
        skill_id=f"winsorize_negative_outlier_mad_e31_{arm.lower()}",
        op="outlier_mad", params={},
        patch_id="patch-replace-winsorize-with-outlier_mad",
        base_sha=h0.harness_content_sha)
    slow = ReplaySlowAgent(slow_manifest)
    method = _llm_method(root, h0, series0, ORIGINS[0], counter, memory,
                         runtime_prior_slot=prior_slot)
    rounds = []
    triggered = False
    for i, origin in enumerate(ORIGINS):
        r = run_online_round(
            method, executor, _request(series0, values, origin), values,
            origin=origin,
            slow_agent=slow if not triggered else None,
            controller=controller if not triggered else None,
            store=store if not triggered else None,
            card_builder=_card_builder(executor, values, origin,
                                       "winsorize", "outlier_mad"),
            round_name=f"{arm.lower()}_r{i + 1}", budget=BUDGET,
            allow_slow=not triggered, domain="kdd_cup_2018", period=PERIOD,
            fast_features=dict(extract_public_features(
                series0[:origin], task_kind="forecast")),
            allow_fast_skill=True,
            runtime_prior_slot=prior_slot)
        open_delayed(r, executor)
        if r.approved_skill_id is not None:
            activate_approved(r, store)
        if r._slow_event is not None and r._slow_event.get("triggered"):
            triggered = True
        rounds.append(r)
    # R4：正常采用检查
    snap4 = method._active_snapshot()  # noqa: SLF001
    m4 = _llm_method(root, snap4, series0, R4_ORIGIN, counter, ())
    r4 = run_online_round(
        m4, executor, _request(series0, values, R4_ORIGIN), values,
        origin=R4_ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name=f"{arm.lower()}_r4",
        budget=BUDGET, allow_slow=False, domain="kdd_cup_2018",
        period=PERIOD,
        fast_features=dict(extract_public_features(
            series0[:R4_ORIGIN], task_kind="forecast")),
        allow_fast_skill=False, runtime_prior_slot=False)
    t4 = m4.last_trace
    pool4 = list(t4.candidate_ids or ())
    retrieved = [s for s in (t4.retrieved_skill_ids or ())
                 if "fast_winner" in s or "winsorize_negative" in s]
    # removal：h0 同轮 plan-only（sealed 池差异）
    import run_v1_sealed_a5_a3 as sealed  # noqa: PLC0415
    rem_core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:R4_ORIGIN], task_kind="forecast"))
    m_rem = TTHAMethod(sealed.TTHAFastAgent(rem_core), h0, ())
    m_rem.bind_round_data(series0[:R4_ORIGIN], task_kind="forecast")
    m_rem.prepare(_request(series0, values, R4_ORIGIN))
    rem_pool = list(m_rem.last_trace.candidate_ids or ())
    # 指标
    total_receipts = sum(r.target_support_receipts_used for r in rounds)
    first_pos_idx = next(
        (i for i, r in enumerate(rounds)
         if r.first_positive_support_receipt_index is not None), None)
    harm_before = sum(r.harm_count for r in rounds[:first_pos_idx]) \
        if first_pos_idx is not None else sum(r.harm_count for r in rounds)
    harm_mag_before = round(sum(r.harm_magnitude
                                for r in rounds[:first_pos_idx]), 6) \
        if first_pos_idx is not None else round(
            sum(r.harm_magnitude for r in rounds), 6)
    final_delayed = next((r.delayed_utility for r in reversed(rounds)
                          if r.delayed_utility is not None), None)
    approved = next((r.approved_skill_id for r in rounds
                     if r.approved_skill_id), None)
    fb_to_skill = None
    if approved is not None:
        # 形成可靠 skill 的累计 Target Support 数：从第一轮到批准轮
        # （含批准轮的探测+replay）的 receipts 累计
        fb_to_skill = 0
        for r in rounds:
            fb_to_skill += r.target_support_receipts_used
            if r.approved_skill_id is not None:
                break
    return {
        "rounds": [{
            "origin": r.origin,
            "probes": [(p["candidate_id"], p.get("gain"))
                       for p in r.actual_probed_programs],
            "winner": r.winner_program,
            "receipts": r.target_support_receipts_used,
            "harm": r.harm_count,
            "delayed_utility": r.delayed_utility,
            "approved_skill_id": r.approved_skill_id,
            "memory_resolution": r.memory_resolution_status,
        } for r in rounds],
        "r4": {"pool": pool4, "retrieved": retrieved,
               "probes": [(p["candidate_id"], p.get("gain"))
                          for p in r4.actual_probed_programs],
               "winner": r4.winner_program,
               "receipts": r4.target_support_receipts_used},
        "removal_pool": rem_pool,
        "removal_no_skill": bool(
            not any(c.startswith("cand_skill_") for c in rem_pool)),
        "total_support_receipts": total_receipts,
        "first_positive_round": (first_pos_idx + 1
                                 if first_pos_idx is not None else None),
        "harm_before_recovery": harm_before,
        "harm_magnitude_before_recovery": harm_mag_before,
        "final_delayed_utility": final_delayed,
        "approved_skill_id": approved,
        "feedback_to_reliable_local_skill": fb_to_skill,
        "abstained": all(r.abstained for r in rounds),
    }


def main() -> int:
    root = PROJECT_ROOT
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        print(json.dumps({"verdict": "PROTOCOL_INCONCLUSIVE",
                          "reason": "no api key"}, indent=1))
        return 0
    frozen = _freeze_e31(root)
    if not frozen:
        print(json.dumps({"verdict": "PROTOCOL_INCONCLUSIVE",
                          "reason": "no virgin cohort frozen"}, indent=1))
        return 0
    cache = np.load(root / "data/kdd2018/series_cache.npz",
                    allow_pickle=True)
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
    source = _monash_source_episodes(root)
    source_neg = _signswap(source)  # 负/冲突组合（冻结——运行后不挑选）

    import openai  # noqa: PLC0415
    try:
        counter5 = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120), max_calls=40)
        a5 = _run_arm(root, executor, series0, vals, counter5,
                      (*source, *source_neg), "A5", prior_slot=True)
        counter3 = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120), max_calls=40)
        a3 = _run_arm(root, executor, series0, vals, counter3,
                      (), "A3", prior_slot=False)
    except RuntimeError as exc:
        if "LLM call budget exceeded" in str(exc):
            print(json.dumps({"verdict": "PROTOCOL_INCONCLUSIVE",
                              "reason": str(exc)}, indent=1))
            return 0
        raise

    # ---- 判定（预注册五档）----
    m5, m3 = a5, a3
    protocol_ok = bool(
        any(r["memory_resolution"] == "rendered" for r in m5["rounds"])
        and m5["r4"]["pool"] and m3["r4"]["pool"])
    a5_fp = m5["first_positive_round"]
    a3_fp = m3["first_positive_round"]
    a5_fb = m5["feedback_to_reliable_local_skill"]
    a3_fb = m3["feedback_to_reliable_local_skill"]
    a5_harm = m5["harm_before_recovery"]
    a3_harm = m3["harm_before_recovery"]
    # 判定修正（2026-08-12）：delayed 只比**同一轮**（同 Context 同
    # winner 可比）——跨轮 delayed（不同 Context 不同程序）不可比，
    # 不作 NEGATIVE_TRANSFER 判据。
    a5_delayed = a3_delayed = None
    _a5_by_origin = {r["origin"]: r["delayed_utility"]
                     for r in m5["rounds"]}
    _a3_by_origin = {r["origin"]: r["delayed_utility"]
                     for r in m3["rounds"]}
    _common = sorted(set(_a5_by_origin) & set(_a3_by_origin))
    if _common and _a5_by_origin[_common[0]] is not None \
            and _a3_by_origin[_common[0]] is not None:
        a5_delayed = _a5_by_origin[_common[0]]
        a3_delayed = _a3_by_origin[_common[0]]
    a5_adopt = bool(m5["r4"]["retrieved"] and m5["r4"]["probes"]
                    and m5["removal_no_skill"])
    a3_adopt = bool(m3["r4"]["retrieved"] and m3["r4"]["probes"]
                    and m3["removal_no_skill"])
    if not protocol_ok:
        verdict = "PROTOCOL_INCONCLUSIVE"
        reason = "two-slot/memory/pool protocol failed"
    elif (a5_fp is None and a3_fp is None
          and m5["abstained"] and m3["abstained"]):
        verdict = "CONTENT_INCONCLUSIVE"
        reason = "both arms abstained — no trajectory to compare"
    elif (a5_fb is not None and a3_fb is not None
          and a5_fb == a3_fb and a5_harm == a3_harm
          and a5_delayed == a3_delayed and a5_adopt == a3_adopt):
        verdict = "NO_SIGNAL"
        reason = "memory rendered but trajectories identical"
    elif (a5_fb is not None and a3_fb is not None
          and a5_fb > a3_fb) or a5_harm > a3_harm \
            or (a5_delayed is not None and a3_delayed is not None
                and a5_delayed < a3_delayed):
        verdict = "NEGATIVE_TRANSFER"
        reason = (f"A5 worse: fb={a5_fb} vs {a3_fb}, "
                  f"harm={a5_harm} vs {a3_harm}, "
                  f"delayed={a5_delayed} vs {a3_delayed}")
    elif (a5_fb is not None and a3_fb is not None
          and (a5_fb < a3_fb or a5_harm < a3_harm)
          and (a5_delayed is None or a3_delayed is None
               or a5_delayed >= a3_delayed)
          and a5_adopt):
        verdict = "TRANSFER_PASS"
        reason = (f"A5 fewer feedback ({a5_fb} vs {a3_fb}) or less harm "
                  f"({a5_harm} vs {a3_harm}), delayed not worse "
                  f"({a5_delayed} vs {a3_delayed}), adoption closed")
    else:
        verdict = "NO_SIGNAL"
        reason = "no directional difference"
    print(f"== A5: fp={a5_fp} fb={a5_fb} harm={a5_harm}/{m5['harm_magnitude_before_recovery']} "
          f"delayed={a5_delayed} adopt={a5_adopt} receipts={m5['total_support_receipts']}")
    print(f"== A3: fp={a3_fp} fb={a3_fb} harm={a3_harm}/{m3['harm_magnitude_before_recovery']} "
          f"delayed={a3_delayed} adopt={a3_adopt} receipts={m3['total_support_receipts']}")
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-e31-fresh-a5-two-slot-vs-a3",
        "note": "E3.1 fresh matched-budget A5-two-slot vs A3（真实 Fast LLM；"
                "virgin KDD cohort E31；四轮：三轮适配+一轮采用；每轮 "
                "Support ≤2（Slow replay 计入）；四轮累计 ≤8；单 Target "
                "PASS 只称跨数据集迁移候选证据）",
        "cohort": [r["series_name"] for r in frozen],
        "origins": list(ORIGINS), "r4_origin": R4_ORIGIN, "budget": BUDGET,
        "source_episodes": [getattr(e, "episode_id", "?")
                            for e in (*source, *source_neg)],
        "arms": {"A5": a5, "A3": a3},
        "verdict": verdict,
        "reason": reason,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
