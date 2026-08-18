"""GROUP_AUTO_TRIGGER_DEV（用户裁决 2026-08-12：GROUP_FAULT 自动触发
接入主链的 dev 验证——零新 Claim）。

验证 online_loop 的 allow_group_slow：失败 Episode 积累 → 轻量分组
（≥2 同算子同 sign）→ 组级 Slow（handle_group_feedback）自动触发。

装置（sealed 确定性——T117 已暴露四轮 @600/@792/@888/@984）：
  R1 @600 winsorize +0.0067（正——失败 0）
  R2 @792 winsorize +0.0733（正）
  R3 @888 winsorize −0.143（失败——1 条 winsorize NEGATIVE——未达阈值）
  R4 @984 winsorize −0.0841（失败——**2 条 winsorize NEGATIVE → 组触发**）
  → 组级 Slow（hampel——共同 headroom——组内 replay @888/@984 全正
    → holdout @600 不劣 → pending）→ open_delayed（delayed 窗口——
    @1032 hampel −0.1166 → delayed 拒绝——预期安全闭环）

判定（预注册）：
  GROUP_AUTO_TRIGGER_DEV_PASS : 失败积累→自动分组→组级触发→组内/组外
    验证→pending→delayed 门控全链（delayed 拒绝或批准均如实——机制链
    完整）
  GROUP_AUTO_TRIGGER_NOT_FIRED : 失败积累但未触发（组阈值/接线问题）
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_group_auto_trigger_dev.py
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
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
)
from run_v1_operational_self_evolution_loop import (  # noqa: E402
    ReplaySlowAgent,
    _skill_manifest,
)

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
ORIGINS = (600, 792, 888, 984)
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_group_auto_trigger_dev_report.json"


def _group_card(group: Mapping[str, object],
                capsule: Mapping[str, object] | None = None) -> dict[str, object]:
    """组 Card（白名单——hampel 共同 headroom 替代——verifier 合法）。
    Wave 1：签名 (group, capsule)——Capsule 嵌入 facts（进入 Slow Agent
    输入）。"""
    card: dict[str, object] = {
        "pattern_id": "group-winsorize-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": [{
            "patch_id": "patch-replace-winsorize-with-hampel_filter",
            "program_steps": [{"op": "hampel_filter",
                               "params": dict(wiring.contract_params(
                                   "hampel_filter", PERIOD))}]}],
    }
    if capsule is not None:
        card["facts"] = {"contrast_capsule": dict(capsule)}
    return card


def main() -> int:
    root = PROJECT_ROOT
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
    series0 = vals[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, vals, _config(),
                             evaluate_fn=_evaluate_kdd)
    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    store = SnapshotStore(root / ".gat_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    manifest = _skill_manifest(
        skill_id="group_winsorize_replacement", op="hampel_filter",
        params=dict(wiring.contract_params("hampel_filter", PERIOD)),
        patch_id="patch-replace-winsorize-with-hampel_filter",
        base_sha=h0.harness_content_sha)
    slow = ReplaySlowAgent(manifest)

    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("winsorize", "outlier_mad",
                                             "hampel_filter"),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:ORIGINS[0]], task_kind="forecast"))
    method = TTHAMethod(sealed.TTHAFastAgent(core), h0, ())
    rounds = []
    for i, origin in enumerate(ORIGINS):
        # P4.1 装置：每轮新 backend（winsorize 每轮先探）
        core.backend = sealed.SealedProbeBackend(
            explore=True, operators=("winsorize", "outlier_mad",
                                     "hampel_filter"),
            max_propose_candidates=3, force_pool=True)
        r = run_online_round(
            method, executor, _request(series0, vals, origin), vals,
            origin=origin, slow_agent=slow, controller=controller,
            store=store,
            card_builder=lambda e: {"pattern_id": "x",
                                    "observable_signature":
                                        {"task_kind": "forecast"}},
            round_name=f"gat_r{i + 1}", budget=BUDGET,
            allow_slow=False, domain="kdd_cup_2018", period=PERIOD,
            fast_features=dict(extract_public_features(
                series0[:origin], task_kind="forecast")),
            allow_fast_skill=False, runtime_prior_slot=False,
            allow_group_slow=True, group_min=2,
            group_card_builder=_group_card,
            group_holdout_origin=600)
        open_delayed(r, executor)
        if r.approved_skill_id is not None:
            activate_approved(r, store)
        rounds.append(r)

    fired = [r for r in rounds if r._group_slow_event is not None]
    info = {
        "rounds": [{
            "origin": r.origin,
            "probes": [(p["candidate_id"], p.get("gain"))
                       for p in r.actual_probed_programs],
            "winner": r.winner_program,
            "group_slow": (r._group_slow_event
                           if r._group_slow_event is not None else None),
            "approved_skill_id": r.approved_skill_id,
        } for r in rounds],
        "n_group_triggers": len(fired),
    }
    if not fired:
        verdict = "GROUP_AUTO_TRIGGER_NOT_FIRED"
        reason = "failure accumulated but group trigger did not fire"
    else:
        ev = fired[0]._group_slow_event
        dev = fired[0]._delayed_event
        checks = {
            "fired_after_threshold": bool(
                ev and ev.get("stage") in ("pending", "group_replay_rejected",
                                           "holdout_rejected",
                                           "no_frozen_program")),
            "in_group_replay_done": bool(ev and "group_replay" in ev),
            "delayed_gated": bool(dev is not None),
        }
        if all(checks.values()):
            verdict = "GROUP_AUTO_TRIGGER_DEV_PASS"
            reason = (f"auto group trigger chain complete: stage="
                      f"{ev.get('stage')} delayed="
                      f"{dev.get('stage')} "
                      f"(delayed_gain={dev.get('delayed_gain')})")
        else:
            verdict = "GROUP_AUTO_TRIGGER_PARTIAL"
            reason = json.dumps(checks)
    print(f"== rounds: {json.dumps(info, ensure_ascii=False, indent=1)}")
    print(f"== verdict: {verdict}")
    print(f"== reason: {reason}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-group-auto-trigger-dev",
        "note": "GROUP_FAULT 自动触发接入主链 dev 验证（sealed 确定性；"
                "T117 已暴露四轮；零新 Claim——delayed @936/@1032 为时间"
                "边界窗口）",
        "origins": list(ORIGINS),
        "rounds": info["rounds"],
        "n_group_triggers": info["n_group_triggers"],
        "verdict": verdict,
        "reason": reason,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
