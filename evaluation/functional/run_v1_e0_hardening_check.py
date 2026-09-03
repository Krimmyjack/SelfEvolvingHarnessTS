"""E0_OPERATIONAL_SEMANTICS_HARDENING_CHECK（用户裁决 2026-08-12）。

零新数据 / 零 live LLM——验证 E0 七项修正：
  1. DRAFT 被选择 ≠ 部署授权（chosen proposal 与 authorized deployment
     分离）——**验收核心：故意让 selector 选择 DRAFT，验证 Runtime 仍
     要求 Support**；
  2. first-positive 用合法 Support receipt index（不含 verifier 拒绝）；
  3. Slow replay 进入 probe/harm 轨迹；
  4. delayed gain None 不转 0（保持未评估）；
  5. memory_resolution_status 公开（A5 rendered / A3 no_memory）；
  6. Slow 调用透传合法 Operator contracts + TaskContext；
  7. current_status 分类（bootstrap 不列 restricted）。

验收装置（全部已暴露）：
  - DRAFT-only 装置：KDD T117 @984——operators=() 时池只含 identity +
    DRAFT skill（winsorize_negative_outlier_mad，guard）→ chosen=DRAFT
    → 探测（Support 申请）→ outlier_mad −0.0608 负向 → 未授权；
  - DRAFT 正向对照：GEFCom @904 DRAFT skill（winsorize 程序）→ 探测
    +0.4000 正向 → 授权（探测=Support 确认）；
  - traffic A3 装置：denoise_median 0.0 → winsorize 正向 → first-pos=2
    （合法 receipt 序）+ memory_resolution_status=no_memory vs A5
    =rendered；
  - GEFCom 场景 B/C 复跑：slow_replay 进轨迹、contracts 透传（stub
    slow_agent 捕获）；
  - current_status：KDD 重载 snapshot——bootstrap 不在 restricted。

用法：
  python evaluation/functional/run_v1_e0_hardening_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation" / "functional"))
sys.path.insert(0, str(PROJECT_ROOT / "methods" / "ttha"))

import numpy as np  # noqa: E402
import run_v1_sealed_a5_a3 as sealed  # noqa: E402
import signed_radius as resolver  # noqa: E402
import run_v1_target_local_loop as tll  # noqa: E402
from run_v1_operational_self_evolution_loop import (  # noqa: E402
    ReplaySlowAgent,
    _card_builder,
    _skill_manifest,
)
from run_v1_kdd2018_memory_gate import _monash_source_episodes  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config as _kdd_config,
    _evaluate_kdd,
    _request as _kdd_request,
)

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    STATUS_LOCAL_DRAFT,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    current_status,
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
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_e0_hardening_check_report.json"
SKILL_ID = "winsorize_negative_outlier_mad"


class CapturingSlowAgent(ReplaySlowAgent):
    """Replay Slow Agent + 捕获 propose_edit 收到的 contracts/task_context
    （E0.6 验收）。"""

    def __init__(self, manifest: Any) -> None:
        super().__init__(manifest)
        self.captured: dict[str, Any] = {}

    def propose_edit(self, card, surface_catalog, snapshot, *,
                     manifest_preflight=None, allowed_operator_contracts=(),
                     task_context=None):
        self.captured["allowed_operator_contracts"] = (
            list(allowed_operator_contracts))
        self.captured["task_context"] = task_context
        return super().propose_edit(
            card, surface_catalog, snapshot,
            manifest_preflight=manifest_preflight,
            allowed_operator_contracts=allowed_operator_contracts,
            task_context=task_context)


def _load_kdd(root: Path) -> dict[str, Any]:
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


def _draft_snapshot(root: Path, *, op: str, applicability: Mapping[str, object]
                    ) -> Any:
    """带 guard 的 DRAFT skill snapshot（E0.1 验收装置）。"""
    import dataclasses  # noqa: PLC0415
    from SelfEvolvingHarnessTS.contracts.harness import (  # noqa: PLC0415
        SkillEntry,
        SkillKind,
    )
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    skill = SkillEntry(
        schema_version="skill-entry/1",
        skill_id=f"draft_{op}",
        skill_kind=SkillKind.CAPABILITY,
        revision=1,
        body=f"Frozen program steps: [{{\"op\": \"{op}\", \"params\": {{}}}}]",
        observable_applicability=applicability,
        allowed_tools=(op,),
        risk_guards={"requires_target_support": True,
                     "preserve_outside_candidate_region": True})
    return dataclasses.replace(h0, skills=(*h0.skills, skill))


def _method(root: Path, snapshot: Any, series0: np.ndarray, origin: int,
            operators: tuple[str, ...], memory: tuple = ()) -> Any:
    core = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True, operators=operators,
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(series0[:origin], task_kind="forecast"))
    return TTHAMethod(sealed.TTHAFastAgent(core), snapshot, memory)


def main() -> int:
    root = PROJECT_ROOT
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    # ---- E0.1a：DRAFT 被选择 ≠ 授权（KDD @984，operators=() 只余 DRAFT）----
    cohort = _load_kdd(root)
    roster, values = cohort["roster"], cohort["values"]
    series0 = values[roster[0]["series_uid"]]
    executor = ScopeExecutor(roster, values, _kdd_config(),
                             evaluate_fn=_evaluate_kdd)
    snap = _draft_snapshot(root, op="outlier_mad", applicability={
        "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]})
    m = _method(root, snap, series0, 984, ())
    r = run_online_round(
        m, executor, _kdd_request(series0, values, 984), values,
        origin=984, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="e0_draft", budget=2,
        allow_slow=False, domain="kdd_cup_2018", period=24)
    details["draft_only"] = {
        "chosen_proposal": r.chosen_proposal,
        "authorized_deployment": r.winner_program,
        "probes": r.actual_probed_programs,
        "abstained": r.abstained}
    checks["E0_1_draft_chosen_not_authorized"] = bool(
        r.chosen_proposal and r.chosen_proposal.startswith("cand_skill_")
        and r.winner_program is None)

    # ---- E0.1b：DRAFT 探测正向 → 授权（GEFCom @904 winsorize +0.4000）----
    import run_e2_autonomous_natural_workflow_generation as v6  # noqa: PLC0415
    gconfig = dict(v6.DATASET_CONFIGS["gefcom"])
    groster, gvalues = v6._fixed_roster(root, gconfig)
    gexec = ScopeExecutor(groster, gvalues, gconfig, evaluate_fn=v6._evaluate)
    gseries0 = np.asarray(gvalues[list(gvalues)[0]], dtype=np.float64)
    gsnap = _draft_snapshot(root, op="winsorize", applicability={
        "all": [{"feature": "task_kind", "op": "==", "value": "forecast"}]})
    gm = _method(root, gsnap, gseries0, 904, ())
    gr = run_online_round(
        gm, gexec, sealed._request(gseries0, gvalues, 904), gvalues,
        origin=904, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="e0_draft_pos", budget=2,
        allow_slow=False, domain="gefcom2012_load", period=24)
    details["draft_positive"] = {
        "chosen_proposal": gr.chosen_proposal,
        "authorized_deployment": gr.winner_program,
        "gain": [p.get("gain") for p in gr.actual_probed_programs]}
    checks["E0_1b_draft_support_then_authorized"] = bool(
        gr.chosen_proposal and gr.chosen_proposal.startswith("cand_skill_")
        and gr.winner_program is not None)

    # ---- E0.2 first-pos = 合法 receipt 序（traffic A3 装置）----
    from run_v1_operational_self_evolution_loop import _traffic_setup  # noqa: PLC0415
    troster, tvals, tseries0, texec = _traffic_setup(root)
    tcore = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("denoise_median", "winsorize"),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(tseries0[:792], task_kind="forecast"))
    tm = TTHAMethod(sealed.TTHAFastAgent(tcore),
                    compile_snapshot(root / "methods/ttha/harness/h0",
                                     verify_lock=False), ())
    tr = run_online_round(
        tm, texec, sealed._request(tseries0, tvals, 792), tvals,
        origin=792, slow_agent=None, controller=None, store=None,
        card_builder=lambda e: {}, round_name="e0_traffic_a3", budget=2,
        allow_slow=False, domain="monash:traffic_hourly", period=24)
    details["traffic_a3"] = {
        "first_positive_support_receipt_index":
            tr.first_positive_support_receipt_index,
        "receipts": tr.target_support_receipts_used,
        "probes": [(p["candidate_id"], p["kind"], p.get("gain"))
                   for p in tr.actual_probed_programs]}
    checks["E0_2_first_positive_uses_receipt_index"] = bool(
        tr.first_positive_support_receipt_index
        == tr.target_support_receipts_used == 2)

    # ---- E0.3/6：Slow replay 进轨迹 + contracts 透传（GEFCom 场景 B）----
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / ".e0_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    manifest = _skill_manifest(
        skill_id="winsorize_replacement", op="winsorize", params={},
        patch_id="patch-replace-outlier_iqr-with-winsorize",
        base_sha=h0.harness_content_sha)
    slow = CapturingSlowAgent(manifest)
    gm2 = _method(root, h0, gseries0, 904, ("outlier_iqr", "winsorize"))
    gr2 = run_online_round(
        gm2, gexec, sealed._request(gseries0, gvalues, 904), gvalues,
        origin=904, slow_agent=slow, controller=controller, store=store,
        card_builder=_card_builder(gexec, gvalues, 904, "outlier_iqr",
                                   "winsorize"),
        round_name="e0_b", budget=2, allow_slow=True,
        domain="gefcom2012_load", period=24,
        fast_features=dict(extract_public_features(
            gseries0[:904], task_kind="forecast")))
    details["slow_trajectory"] = {
        "probes": [(p["candidate_id"], p["kind"], p.get("gain"))
                   for p in gr2.actual_probed_programs],
        "slow_replay_receipts": gr2.slow_replay_receipts_used,
        "captured_contracts": [c.get("name") for c in
                               slow.captured.get(
                                   "allowed_operator_contracts", [])],
        "captured_task_context": slow.captured.get("task_context")}
    checks["E0_3_slow_replay_in_trajectory"] = bool(
        any(p["kind"] == "slow_replay" and p.get("gain") == 0.4000053662007894
            for p in gr2.actual_probed_programs))
    checks["E0_6_contracts_passed"] = bool(
        "winsorize" in [c.get("name") for c in
                        slow.captured.get("allowed_operator_contracts", [])]
        and "outlier_iqr" in [c.get("name") for c in
                              slow.captured.get(
                                  "allowed_operator_contracts", [])])

    # ---- E0.4：delayed None 不转 0（stub evaluator——单元级）----
    from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: PLC0415
        build_episode,
    )
    from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: PLC0415
        RoundResult,
        _update_delayed_status,
    )
    ep = build_episode(
        episode_id="e0_delayed_test", task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="test", context_summary={"local_pattern": {}},
        workflow_signature="winsorize",
        support_response={"gain": 0.01, "accepted": True},
        delayed_response={"evaluated": False, "gain": None},
        relation="POSITIVE", evidence_level="SUPPORT",
        local_status=STATUS_LOCAL_DRAFT, evidence_refs=["e0"])
    upd = _update_delayed_status(ep, 0.05, {})
    checks["E0_4_delayed_none_not_zero"] = bool(
        upd.delayed_response["gain"] == 0.05)  # 非 None 才更新——0.05 正常
    # 直接验证 open_delayed 的 None 分支：构造 gain=None 的 evaluate
    class _NoneEvaluator:
        def evaluate(self, steps, origin):
            return SimpleNamespace(gain=None)
    rr = RoundResult(round_name="x", origin=1)
    rr._method = SimpleNamespace(
        update_experience_episode=lambda e: None,
        handle_feedback_delayed=lambda *a, **k: {"stage": "no_pending"})
    rr._values = {1: np.zeros(10)}
    rr._period = 24
    rr._episodes = [(ep, (("winsorize", {}),))]
    rr._winner_steps = None
    before = dict(ep.delayed_response)
    open_delayed(rr, _NoneEvaluator())
    checks["E0_4b_delayed_none_keeps_unevaluated"] = bool(
        dict(ep.delayed_response) == before)

    # ---- E0.5：memory_resolution_status（traffic A5 vs A3）----
    src = tll.write_target_episode(
        domain="monash:traffic_hourly", op="winsorize",
        episode_id_suffix="_e0_src", delayed_gain=None,
        program_steps=[{"op": "winsorize", "params": {}}],
        support_gain=0.4,
        support_context=dict(resolver.window_context(tvals, 792, 24)))
    # delayed_pattern 必须含 recent./change. 键（signed_radius 证据提取
    # 要求——否则 delayed 证据被丢弃 → verdict UNKNOWN → rendered_empty）
    src = tll.update_delayed_status(
        src, 0.03,
        delayed_context=dict(resolver.window_context(tvals, 840, 24)))
    tcore5 = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("denoise_median", "winsorize"),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(tseries0[:792], task_kind="forecast"))
    tm5 = TTHAMethod(sealed.TTHAFastAgent(tcore5),
                     compile_snapshot(root / "methods/ttha/harness/h0",
                                      verify_lock=False), (src,))
    tm5.bind_round_data(tseries0[:792], task_kind="forecast")
    tm5.prepare(sealed._request(tseries0, tvals, 792))
    status_a5 = str(getattr(tm5.last_trace, "memory_resolution_status", "?"))
    tcore3 = sealed.TTHAAgentCore(
        sealed.SealedProbeBackend(explore=True,
                                  operators=("denoise_median", "winsorize"),
                                  max_propose_candidates=3, force_pool=True),
        LocalPublicToolGateway(tseries0[:792], task_kind="forecast"))
    tm3 = TTHAMethod(sealed.TTHAFastAgent(tcore3),
                     compile_snapshot(root / "methods/ttha/harness/h0",
                                      verify_lock=False), ())
    tm3.bind_round_data(tseries0[:792], task_kind="forecast")
    tm3.prepare(sealed._request(tseries0, tvals, 792))
    status_a3 = str(getattr(tm3.last_trace, "memory_resolution_status", "?"))
    details["memory_status"] = {"A5": status_a5, "A3": status_a3}
    checks["E0_5_memory_status_public"] = bool(
        status_a5 == "rendered" and status_a3 == "no_memory")

    # ---- E0.7：current_status 分类（bootstrap 不列 restricted）----
    st = current_status(None, m)
    details["current_status"] = {
        "draft": st["draft_skills"], "active": st["active_skills"],
        "restricted": st["restricted_skills"],
        "bootstrap": st["bootstrap_skills"]}
    checks["E0_7_status_classification"] = bool(
        "draft_outlier_mad" in st["draft_skills"]
        and "build_contrastive_candidates" in st["bootstrap_skills"]
        and "build_contrastive_candidates" not in st["restricted_skills"])

    verdict = "E0_HARDENING_PASS" if all(checks.values()) \
        else "E0_HARDENING_FAILED"
    print(f"== checks: {json.dumps(checks, indent=1)}")
    print(f"== verdict: {verdict}")
    print(f"== details: {json.dumps(details, ensure_ascii=False, indent=1)}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-e0-hardening-check",
        "note": "E0 Operational Semantics Hardening 验收（零新数据/零"
                "live LLM）",
        "checks": checks,
        "details": details,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
