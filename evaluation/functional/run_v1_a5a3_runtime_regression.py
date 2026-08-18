"""A5A3_RUNTIME_SEMANTICS_REGRESSION（P4，用户裁决 2026-08-12）。

P3 通过后，使用**同一个在线入口**（online_loop）在已暴露 KDD 数据上
重跑开发级 A5/A3（P4.1 装置：T117、origin 600/792/888 + R4 @984）。

只检查（用户裁决）：
  - A5/A3 唯一差异是 Source Episodes；
  - 同 Target Support 总预算；
  - Slow replay 计入预算；
  - chosen-first；
  - 实际所有 probe 都计入 harm；
  - delayed utility 属于实际 winner；
  - Skill 形成必须同时报告"批准"和"下一轮检索/采用"。

无论结果是 PASS、NO_SIGNAL 还是 NEGATIVE，都只叫：
  CURRENT_RUNTIME_A5A3_DEVELOPMENT_REGRESSION
不用于重新声称 cross-domain benefit。

零新数据（T117 600-1080 已暴露）/零 live LLM（Replay Slow Agent——
P4.1 批准的 manifest 同款：patch-winsorize-to-outlier_mad）。

用法：
  python evaluation/functional/run_v1_a5a3_runtime_regression.py
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
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_memory_gate import _monash_source_episodes  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)
from run_v1_operational_self_evolution_loop import ReplaySlowAgent  # noqa: E402
from run_v1_operational_self_evolution_loop import _skill_manifest  # noqa: E402
from run_v1_operational_self_evolution_loop import _card_builder  # noqa: E402

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
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
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_a5a3_runtime_semantics_regression_report.json"


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


def _run_arm(root: Path, executor: ScopeExecutor, series0: np.ndarray,
             values: Mapping[str, Any], memory: tuple,
             arm: str) -> dict[str, Any]:
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / f".p4_store_{arm.lower()}")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    manifest = _skill_manifest(
        skill_id="winsorize_negative_outlier_mad", op="outlier_mad",
        params={},
        # 与 _card/_patch_options 白名单命名一致（replace...with...）
        patch_id="patch-replace-winsorize-with-outlier_mad",
        base_sha=h0.harness_content_sha)
    slow = ReplaySlowAgent(manifest)
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:ORIGINS[0]], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), h0, memory)
    rounds = []
    triggered = False
    for i, origin in enumerate(ORIGINS):
        # P4.1 同装置语义：每轮新建 backend（explore 状态重置——每轮
        # 从池序起点提案；method 实例保留——episodes/snapshot 累积）。
        core.backend = sealed.SealedProbeBackend(
            explore=True, operators=POOL, max_propose_candidates=3,
            force_pool=True)
        r = run_online_round(
            method, executor, _request(series0, values, origin), values,
            origin=origin, slow_agent=slow if not triggered else None,
            controller=controller if not triggered else None,
            store=store if not triggered else None,
            card_builder=_card_builder(executor, values, origin,
                                       "winsorize", "outlier_mad"),
            round_name=f"r{i + 1}_{arm.lower()}", budget=BUDGET,
            allow_slow=not triggered, domain="kdd_cup_2018", period=PERIOD,
            fast_features=dict(extract_public_features(
                series0[:origin], task_kind="forecast")))
        open_delayed(r, executor)
        if r._slow_event is not None and r._slow_event.get("triggered"):
            triggered = True
            activate_approved(r, store)
        rounds.append(r)
    # R4：下一轮检索/采用报告（@984 已暴露）
    skill_snap = method._active_snapshot()  # noqa: SLF001
    core4 = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=POOL,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:R4_ORIGIN], task_kind="forecast"))
    m4 = TTHAMethod(sealed.TTHAFastAgent(core4), skill_snap, ())
    r4 = run_online_round(
        m4, executor, _request(series0, values, R4_ORIGIN), values,
        origin=R4_ORIGIN, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name=f"r4_{arm.lower()}",
        budget=BUDGET, allow_slow=False, domain="kdd_cup_2018", period=PERIOD)
    t4 = m4.last_trace
    r4_pool = list(t4.candidate_ids or ())
    skill_cands = [c for c in r4_pool if c.startswith("cand_skill_")]
    agent_cands = [c for c in r4_pool if c.startswith("cand_")
                   and not c.startswith("cand_skill_")]
    return {
        "rounds": rounds, "r4": r4,
        "triggered": triggered,
        "approved_skill_id": next((r.approved_skill_id for r in rounds
                                   if r.approved_skill_id), None),
        "r4_retrieved": bool("winsorize_negative_outlier_mad"
                             in (t4.retrieved_skill_ids or ())),
        "r4_skill_in_pool": bool(skill_cands),
        "r4_draft_not_priority": bool(
            skill_cands and agent_cands
            and r4_pool.index(skill_cands[0]) > r4_pool.index(agent_cands[0])
            and not (t4.chosen_candidate_id or "") in skill_cands),
        "r4_probed_skill": bool(
            any(p["candidate_id"].startswith("cand_skill_")
                for p in r4.actual_probed_programs)),
    }


def _arm_metrics(arm: dict[str, Any]) -> dict[str, Any]:
    probes = [p for r in arm["rounds"] for p in r.actual_probed_programs]
    return {
        "first_positive_support_receipt_index": next(
            (r.first_positive_support_receipt_index for r in arm["rounds"]
             if r.first_positive_support_receipt_index is not None), None),
        "total_support_receipts": sum(
            r.target_support_receipts_used for r in arm["rounds"]),
        "slow_replay_receipts": sum(
            r.slow_replay_receipts_used for r in arm["rounds"]),
        "harm_count": sum(r.harm_count for r in arm["rounds"]),
        "harm_magnitude": round(sum(r.harm_magnitude
                                    for r in arm["rounds"]), 6),
        "winner_programs": [r.winner_program for r in arm["rounds"]],
        "delayed_utilities": [r.delayed_utility for r in arm["rounds"]],
        "probe_trajectory": [
            {"round": i + 1,
             "probes": [{"candidate_id": p["candidate_id"],
                         "gain": p.get("gain")}
                        for p in r.actual_probed_programs]}
            for i, r in enumerate(arm["rounds"])],
    }


def main() -> int:
    root = PROJECT_ROOT
    cohort = _load(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _config(),
                             evaluate_fn=_evaluate_kdd)
    source = _monash_source_episodes(root)
    arms = {"A5": _run_arm(root, executor, series0, values,
                           tuple(source), "A5"),
            "A3": _run_arm(root, executor, series0, values, (), "A3")}
    m5, m3 = _arm_metrics(arms["A5"]), _arm_metrics(arms["A3"])
    print(f"== A5: {json.dumps(m5, ensure_ascii=False)}")
    print(f"== A3: {json.dumps(m3, ensure_ascii=False)}")
    print(f"== A5 r4: retrieved={arms['A5']['r4_retrieved']} "
          f"draft_not_priority={arms['A5']['r4_draft_not_priority']} "
          f"probed_skill={arms['A5']['r4_probed_skill']}")

    verdict = "CURRENT_RUNTIME_A5A3_DEVELOPMENT_REGRESSION"
    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-a5a3-runtime-semantics-regression",
        "note": "P4 语义回归（统一在线入口；已暴露 KDD T117；零新数据/"
                "零 live LLM；不用于重新声称 cross-domain benefit）",
        "origins": list(ORIGINS), "r4_origin": R4_ORIGIN,
        "budget": BUDGET,
        "arms": {
            "A5": {"metrics": m5, "approved_skill_id":
                   arms["A5"]["approved_skill_id"],
                   "r4": {"retrieved": arms["A5"]["r4_retrieved"],
                          "skill_in_pool": arms["A5"]["r4_skill_in_pool"],
                          "draft_not_priority":
                              arms["A5"]["r4_draft_not_priority"],
                          "probed_skill": arms["A5"]["r4_probed_skill"]}},
            "A3": {"metrics": m3, "approved_skill_id":
                   arms["A3"]["approved_skill_id"],
                   "r4": {"retrieved": arms["A3"]["r4_retrieved"],
                          "skill_in_pool": arms["A3"]["r4_skill_in_pool"],
                          "draft_not_priority":
                              arms["A3"]["r4_draft_not_priority"],
                          "probed_skill": arms["A3"]["r4_probed_skill"]}},
        },
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== verdict: {verdict}")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
