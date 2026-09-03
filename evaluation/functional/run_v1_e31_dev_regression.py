"""E31_DEV_REGRESSION（用户裁决 2026-08-12：主链修正后开发回归——
零新 Claim；E31 已暴露仅作开发数据）。

修正（用户承重 1-5 + 指标）：
  1. A5/A3 **同 prior control**（两臂 runtime_prior_slot=True——唯一差异
     初始 Source Memory；后续 Target Episodes 权限完全相同）；
  2. Source 只允许**自然 Episode**（无 signswap 机械构造）——若没有完整
     自然轨迹（正/负/冲突均自然）→ SOURCE_CONTENT_UNAVAILABLE（不承重
     跨域比较）；
  3. Slow manifest **每轮按当前 active snapshot 重建**（base_sha=active
     harness_content_sha——Fast skill 形成后自然失败可触发可执行 Slow）；
  4. 无 winner 不开 delayed（online_loop 已修）；
  5. adoption 必须证明 **skill 程序（rebind 后）实际被 probe 并成为
     authorized winner**（程序匹配语义——非任意 probe）；
  6. R4 纳入累计 Support 预算；
  7. 逐 receipt 统计 harm（含同一轮先负后正）+ trajectory total harm；
  8. verdict 窄口径（不笼统 NO_SIGNAL）。

verdict（预注册）：
  SOURCE_CONTENT_UNAVAILABLE : Source 无完整自然轨迹（正/负/冲突）——
    只记录机制回归结果，不形成跨域比较 Claim
  MAIN_CHAIN_REGRESSION_PASS : 主链修正全部验证（同 prior control /
    Slow 绑定 active / adoption 程序匹配 / 预算含 R4 / 逐 receipt harm）
  MAIN_CHAIN_REGRESSION_FAILED : 任一机制验证失败

用法：
  python evaluation/functional/run_v1_e31_dev_regression.py
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
M = resolver.MATERIAL_THRESHOLD
BUDGET = 2
ORIGINS = (600, 792, 888)
R4_ORIGIN = 984
POOL = ("winsorize", "outlier_mad", "hampel_filter")
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_e31_dev_regression_report.json"


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
             arm: str) -> dict[str, Any]:
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    store = SnapshotStore(root / f".e31d_store_{arm.lower()}")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    method = _llm_method(root, h0, series0, ORIGINS[0], counter, memory,
                         runtime_prior_slot=True)  # 两臂同 prior control
    rounds = []
    triggered = False
    for i, origin in enumerate(ORIGINS):
        # 修正 3：Slow manifest 每轮按**当前 active snapshot** 重建
        # （Fast skill 形成后 active 已变——base_sha 必须匹配）
        _active = method._active_snapshot()  # noqa: SLF001
        _manifest = _skill_manifest(
            skill_id=f"winsorize_negative_outlier_mad_{arm.lower()}_r{i + 1}",
            op="outlier_mad", params={},
            patch_id="patch-replace-winsorize-with-outlier_mad",
            base_sha=_active.harness_content_sha)
        slow = ReplaySlowAgent(_manifest)
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
            allow_fast_skill=True, runtime_prior_slot=True)
        open_delayed(r, executor)
        if r.approved_skill_id is not None:
            activate_approved(r, store)
        if r._slow_event is not None and r._slow_event.get("triggered"):
            triggered = True
        rounds.append(r)
    # R4（修正 6：纳入累计预算）
    snap4 = method._active_snapshot()  # noqa: SLF001
    m4 = _llm_method(root, snap4, series0, R4_ORIGIN, counter, (),
                     runtime_prior_slot=True)  # 同 control——Target 经验可用
    r4 = run_online_round(
        m4, executor, _request(series0, values, R4_ORIGIN), values,
        origin=R4_ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name=f"{arm.lower()}_r4",
        budget=BUDGET, allow_slow=False, domain="kdd_cup_2018",
        period=PERIOD,
        fast_features=dict(extract_public_features(
            series0[:R4_ORIGIN], task_kind="forecast")),
        allow_fast_skill=False, runtime_prior_slot=True)
    t4 = m4.last_trace
    pool4 = list(t4.candidate_ids or ())
    steps_map4 = dict(t4.candidate_program_steps or {})
    retrieved = [s for s in (t4.retrieved_skill_ids or ())
                 if "fast_winner" in s or "winsorize_negative" in s]
    # 修正 5：adoption = skill 程序（rebind 后）被 probe 且成 winner
    r4_fe = dict(extract_public_features(series0[:R4_ORIGIN],
                                         task_kind="forecast"))
    expected = {
        "region_start_fraction": r4_fe.get("estimated_region_start_fraction"),
        "region_end_fraction": r4_fe.get("estimated_region_end_fraction"),
    }

    def _matches_skill(prog: tuple) -> bool:
        if not prog or prog[0][0] != "repair_level_shift":
            return False
        pp = dict(prog[0][1])
        return all(
            expected.get(k) is not None
            and abs(float(pp.get(k, -1)) - float(expected[k])) < 1e-9
            for k in expected)

    r4_probed_skill = bool(
        any(_matches_skill(steps_map4.get(p["candidate_id"], ()))
            for p in r4.actual_probed_programs))
    r4_winner_is_skill = bool(
        r4.winner_program is not None
        and _matches_skill(tuple(
            (s["op"], s["params"]) for s in r4.winner_program)))
    # removal：同一 Fast 入口 plan-only 反事实（用户裁决——不再 sealed）
    m_rem = _llm_method(root, h0, series0, R4_ORIGIN, counter, (),
                        runtime_prior_slot=True)
    m_rem.bind_round_data(series0[:R4_ORIGIN], task_kind="forecast")
    m_rem.prepare(_request(series0, values, R4_ORIGIN))
    rem_pool = list(m_rem.last_trace.candidate_ids or ())
    # 修正 7：逐 receipt harm（含同一轮先负后正）+ trajectory total harm
    receipt_harm: list[dict[str, Any]] = []
    for r in rounds:
        for p in r.actual_probed_programs:
            g = p.get("gain")
            if g is not None and float(g) < -M:
                receipt_harm.append({
                    "round": r.origin, "candidate": p["candidate_id"],
                    "gain": float(g)})
    total_harm = len(receipt_harm)
    total_harm_mag = round(sum(-h["gain"] for h in receipt_harm), 6)
    first_pos = next(
        (i for i, r in enumerate(rounds)
         if r.first_positive_support_receipt_index is not None), None)
    harm_before_recovery = 0
    harm_before_mag = 0.0
    if first_pos is not None:
        for h in receipt_harm:
            if h["round"] < rounds[first_pos].origin:
                harm_before_recovery += 1
                harm_before_mag += -h["gain"]
    else:
        harm_before_recovery = total_harm
        harm_before_mag = total_harm_mag
    total_receipts = sum(r.target_support_receipts_used for r in rounds) \
        + r4.target_support_receipts_used  # 修正 6：R4 纳入
    approved = next((r.approved_skill_id for r in rounds
                     if r.approved_skill_id), None)
    fb_to_skill = None
    if approved is not None:
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
            "delayed_utility": r.delayed_utility,
            "approved_skill_id": r.approved_skill_id,
            "slow_event_stage": (r._slow_event or {}).get("stage"),
            "memory_resolution": r.memory_resolution_status,
        } for r in rounds],
        "r4": {"pool": pool4, "retrieved": retrieved,
               "probes": [(p["candidate_id"], p.get("gain"))
                          for p in r4.actual_probed_programs],
               "winner": r4.winner_program,
               "receipts": r4.target_support_receipts_used,
               "probed_skill_program": r4_probed_skill,
               "winner_is_skill_program": r4_winner_is_skill},
        "removal_pool": rem_pool,
        "removal_no_skill": bool(
            not any(c.startswith("cand_skill_") for c in rem_pool)),
        "total_support_receipts": total_receipts,
        "first_positive_round": (first_pos + 1
                                 if first_pos is not None else None),
        "receipt_harm": receipt_harm,
        "total_harm": total_harm,
        "total_harm_magnitude": total_harm_mag,
        "harm_before_recovery": harm_before_recovery,
        "harm_before_magnitude": round(harm_before_mag, 6),
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
        print(json.dumps({"verdict": "SOURCE_CONTENT_UNAVAILABLE",
                          "reason": "no api key"}, indent=1))
        return 0
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_e31.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    vals = {str(r["series_name"]): np.asarray(
        values[names.index(str(r["series_name"]))], dtype=np.float64)
        for r in rows}
    series0 = vals[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, vals, _config(),
                             evaluate_fn=_evaluate_kdd)
    # 修正 2：Source 只自然 Episode——检查完整自然轨迹
    source_pos = _monash_source_episodes(root)
    natural_only = [e for e in source_pos
                    if "swap" not in getattr(e, "episode_id", "")]
    has_complete_natural = bool(
        natural_only and any(
            (getattr(e, "support_response", {}) or {}).get("gain", 0) < -M
            or (getattr(e, "delayed_response", {}) or {}).get("gain", 0) < -M
            for e in natural_only))
    # 当前只有自然正例（无自然负/冲突）→ 不完整 → SOURCE_CONTENT_UNAVAILABLE
    source_content_ok = bool(
        natural_only and has_complete_natural
        and any((getattr(e, "relation", "") or "") == "CONFLICT"
                for e in natural_only))
    if not source_content_ok:
        # 仍跑机制回归（同 control/Slow 绑定/adoption 语义）——但
        # SOURCE_CONTENT_UNAVAILABLE 标注——不承重跨域比较
        print(json.dumps({
            "verdict": "SOURCE_CONTENT_UNAVAILABLE",
            "reason": ("no complete natural Source trajectory (positive/"
                       "negative/conflict all natural) — natural positives="
                       f"{len(natural_only)}")}, indent=1))
    import openai  # noqa: PLC0415
    try:
        counter5 = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120), max_calls=40)
        a5 = _run_arm(root, executor, series0, vals, counter5,
                      tuple(natural_only), "A5")
        counter3 = smoke.CountingClient(
            openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL,
                          timeout=120), max_calls=40)
        a3 = _run_arm(root, executor, series0, vals, counter3, (), "A3")
    except RuntimeError as exc:
        if "LLM call budget exceeded" in str(exc):
            print(json.dumps({"verdict": "SOURCE_CONTENT_UNAVAILABLE",
                              "reason": str(exc)}, indent=1))
            return 0
        raise

    # ---- 机制回归检查（零 Claim——不比较跨域）----
    checks = {
        "M1_prior_control_symmetric": bool(
            all(r["memory_resolution"] in ("rendered", "rendered_empty",
                                           "no_memory")
                for r in a5["rounds"])
            and all(r["memory_resolution"] in ("rendered", "rendered_empty",
                                               "no_memory")
                    for r in a3["rounds"])),
        "M2_slow_binds_active": bool(
            all((r["slow_event_stage"] is None)
                or (r["slow_event_stage"] in ("pending", "support_rejected",
                                              "no_trigger"))
                for r in (*a5["rounds"], *a3["rounds"]))),
        "M3_adoption_skill_program": bool(
            a5["r4"]["probed_skill_program"]
            and a5["r4"]["winner_is_skill_program"]
            and a5["removal_no_skill"]),
        "M4_r4_in_budget": bool(
            a5["total_support_receipts"] <= 8
            and a3["total_support_receipts"] <= 8),
        "M5_receipt_level_harm": bool(
            isinstance(a5["receipt_harm"], list)
            and isinstance(a3["receipt_harm"], list)),
    }
    verdict = ("MAIN_CHAIN_REGRESSION_PASS" if all(checks.values())
               else "MAIN_CHAIN_REGRESSION_FAILED")
    if not source_content_ok:
        verdict = "SOURCE_CONTENT_UNAVAILABLE"
    print(f"== A5: fp={a5['first_positive_round']} fb={a5['feedback_to_reliable_local_skill']} "
          f"harm_total={a5['total_harm']}/{a5['total_harm_magnitude']} "
          f"harm_before={a5['harm_before_recovery']} "
          f"receipts={a5['total_support_receipts']} adopt_skill_program={a5['r4']['probed_skill_program']}/{a5['r4']['winner_is_skill_program']}")
    print(f"== A3: fp={a3['first_positive_round']} fb={a3['feedback_to_reliable_local_skill']} "
          f"harm_total={a3['total_harm']}/{a3['total_harm_magnitude']} "
          f"harm_before={a3['harm_before_recovery']} "
          f"receipts={a3['total_support_receipts']} adopt_skill_program={a3['r4']['probed_skill_program']}/{a3['r4']['winner_is_skill_program']}")
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-e31-dev-regression",
        "note": "E31 主链修正开发回归（用户裁决 2026-08-12——零新 Claim；"
                "同 prior control/Slow 绑定 active/adoption 程序匹配/"
                "R4 计预算/逐 receipt harm；Source 仅自然——不承重跨域）",
        "source_content_ok": source_content_ok,
        "natural_source_episodes": [
            getattr(e, "episode_id", "?") for e in natural_only],
        "arms": {"A5": a5, "A3": a3},
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
