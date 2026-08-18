"""GROUP_FAULT_DEV（用户裁决 2026-08-12：最小纵向切片——多轨迹共同
错误归因——不建平台）。

验证链（用户步骤 1-7）：
  1. 保留 per-view Action–Response（online_loop 已修——本 runner 重建
     已暴露失败 Episode 时保留 per_view）；
  2. 轻量确定性分组（group_fault.group_first_faults）；
  3. 重复 first-fault 组（winsorize NEGATIVE——K1 T117 @888 −0.143 +
     E31 T153 @792 −0.5406——同算子同 sign）；
  4. 共同 replacement headroom（find_common_headroom——outlier_mad/
     hampel 在组内各 origin replay）；
  5. Slow Agent 基于整组 Contrast Capsule propose（handle_group_feedback
     ——ReplaySlowAgent——组内 replay 全 ≥M + 组外 holdout 不劣 →
     pending）；
  6. delayed 批准（handle_feedback_delayed——组外 delayed 窗口）；
  7. 下一正常入口复用（既有 rebinding/adoption 链——本 runner 验证
     pending 产物可检索）。

verdict（预注册）：
  GROUP_FAULT_DEV_PASS : 分组→capsule→共同 headroom→组级 Patch→组内/
    组外验证→pending 全链成立
  NO_COMMON_HEADROOM : 组内替代无共同正向（安全闭环——共同有效修复
    未成立——如实负档）
  GROUP_REPLAY_REJECTED / HOLDOUT_REJECTED / PROTOCOL_FAILURE

已暴露数据（T117 @888/@984、T153 @792/@984——零新 outcome；dev——
不形成新 Claim）。

用法：
  python evaluation/functional/run_v1_group_fault_dev.py
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
from SelfEvolvingHarnessTS.methods.ttha.program_supply import controlled_add_only_group_catalog, controlled_add_only_group_decision

import numpy as np  # noqa: E402
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
from run_v1_kdd2018_memory_gate import _monash_source_episodes  # noqa: E402

from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    build_contrast_capsule,
    find_common_headroom,
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_group_fault_dev_report.json"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _executor_for(root: Path, uid: str) -> tuple[ScopeExecutor, dict]:
    if uid == "T117":
        rel = "w1_kdd2018_frozen_cohort_p41.jsonl"
    else:
        rel = "w1_kdd2018_frozen_cohort_e31.jsonl"
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2" / rel)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    values = {str(r["series_name"]): _load_series(root, str(r["series_name"]))
              for r in rows}
    return ScopeExecutor(roster, values, _config(), evaluate_fn=_evaluate_kdd), \
        values


def _rebuild_failure_episode(root: Path, uid: str, origin: int) -> Any:
    """重建已暴露失败 Episode（保留 per_view + origin——零新 outcome——
    重读已暴露窗口的细粒度响应）。"""
    executor, values = _executor_for(root, uid)
    series0 = values[uid]
    steps = (("winsorize", dict(wiring.contract_params("winsorize", PERIOD))),)
    rr = executor.evaluate(tuple(steps), origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{uid}@{origin} not failure"
    ctx = dict(resolver.window_context(values, origin, PERIOD))
    ep = build_episode(
        episode_id=f"kdd_cup_2018_target_winsorize_gf_{uid}_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="kdd_cup_2018",
        context_summary={
            "local_pattern": {"support_gain": gain, **ctx},
            "delayed_pattern": {},
            "program_geometry": {"program_steps":
                                 [{"op": "winsorize", "params": {}}]},
            "per_view_gain": list(getattr(rr, "per_view_gain", []) or []),
            "support_origin": origin,
        },
        workflow_signature="winsorize",
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["group_fault_dev"])
    return ep


def _group_card(group: Mapping[str, Any],
                executor: ScopeExecutor, values: Mapping[str, Any],
                capsule: Mapping[str, Any] | None = None,
                ) -> dict[str, object]:
    """组 Card：部署 Context（组内 Episode 的 recent./change.——outcome
    派生字段不进公开签名）+ 白名单（patch-replace-winsorize-with-
    outlier_mad/hampel——verifier 合法）。Wave 1：签名加 capsule——
    Capsule 嵌入 facts（进入 Slow Agent 输入）。"""
    first_ep = (group.get("episodes") or [None])[0]
    lp = ((getattr(first_ep, "context_summary", {}) or {})
          .get("local_pattern") or {}) if first_ep else {}
    ctx = {k: float(v) for k, v in lp.items()
           if str(k).startswith(("recent.", "change."))
           and isinstance(v, (int, float))}
    options = []
    for alt_op in ("outlier_mad", "hampel_filter"):
        alt_steps = ((alt_op, dict(wiring.contract_params(alt_op, PERIOD))),)
        for ep in (group.get("episodes") or []):
            o = int(((getattr(ep, "context_summary", {}) or {})
                     .get("support_origin") or 0))
            if executor.verify(tuple(alt_steps), o).passed:
                options.append({
                    "patch_id": f"patch-replace-winsorize-with-{alt_op}",
                    "program_steps": [{"op": alt_op,
                                       "params": dict(wiring.contract_params(
                                           alt_op, PERIOD))}]})
                break
    card: dict[str, object] = {
        "pattern_id": "group-winsorize-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": ctx,
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": options,
    }
    if capsule is not None:
        card["facts"] = {"contrast_capsule": dict(capsule)}
    return card


def main() -> int:
    root = PROJECT_ROOT
    # 1-2. 重建失败 Episode + 轻量分组（winsorize NEGATIVE——T117@888 与
    #      T153@792——同算子同 sign）
    # 同算子重复失败组：T117 winsorize @888（−0.143）与 @984（−0.0841）
    # ——同 Workflow 同 sign（NEGATIVE）——共同 first fault 的证据基础
    eps = [
        _rebuild_failure_episode(root, "T117", 888),
        _rebuild_failure_episode(root, "T117", 984),
    ]
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no repeated first-fault group"},
                         indent=1))
        return 0
    group = groups[0]
    # 3. Contrast Capsule
    capsule = build_contrast_capsule(group)
    # 4. 共同 replacement headroom（outlier_mad/hampel——组内各 origin）
    ex117, vals117 = _executor_for(root, "T117")
    ex153, vals153 = _executor_for(root, "T153")

    def _steps_of(op: str):
        return ((op, dict(wiring.contract_params(op, PERIOD))),)

    def _exec_for(uid: str, origin: int) -> ScopeExecutor:
        return ex117 if uid == "T117" else ex153

    # find_common_headroom 需要单一 executor——组内 origin 跨系列——
    # 手动 replay（各组内 Episode 的 origin 用其系列 executor）
    headroom: dict[str, Any] = {}
    for alt in ("outlier_mad", "hampel_filter"):
        per_ep = []
        all_positive = True
        for ep in group["episodes"]:
            uid = "T117" if ep.episode_id.startswith(
                "kdd_cup_2018_target_winsorize_gf_T117") else "T153"
            ex = _exec_for(uid, 0)
            origin = int(((ep.context_summary or {})
                          .get("support_origin") or 0))
            rr = ex.evaluate(tuple(_steps_of(alt)), origin)
            g = (float(rr.gain) if rr.gain is not None else None)
            per_ep.append({"series": uid, "origin": origin, "gain": g})
            if g is None or g < M:
                all_positive = False
        headroom[alt] = {"per_episode_gains": per_ep,
                         "common_positive": all_positive}
    common_alts = [a for a, h in headroom.items() if h["common_positive"]]
    print(f"== group: {group['workflow']}/{group['sign']} "
          f"n={len(group['episodes'])}")
    print(f"== capsule: {json.dumps(capsule, ensure_ascii=False)}")
    print(f"== headroom: {json.dumps(headroom, ensure_ascii=False)}")
    if not common_alts:
        print(json.dumps({
            "verdict": "NO_COMMON_HEADROOM",
            "reason": ("no alternative with common positive headroom on "
                       "all in-group episodes")}, indent=1))
        REPORT_REL.write_text(json.dumps({
            "experiment_id": "v1-group-fault-dev",
            "note": "Group fault dev（已暴露数据零新 outcome——不形成 Claim）",
            "groups": [{"workflow": g["workflow"], "sign": g["sign"],
                        "episodes": [e.episode_id for e in g["episodes"]]}
                       for g in groups],
            "capsule": capsule, "headroom": headroom,
            "verdict": "NO_COMMON_HEADROOM"}, ensure_ascii=False, indent=2)
            + "\n", encoding="utf-8")
        return 0
    # 5. 组级 Slow（handle_group_feedback——ReplaySlowAgent propose
    #    outlier_mad——组内 replay 全正 + holdout 不劣 → pending）
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / ".gf_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    # 共同 headroom 替代（hampel——@888/@984 均正向）——Slow Agent 应基于
    # 整组 Capsule 的 headroom 选择此替代（提错替代 → 组内 replay 拒绝）
    manifest = _skill_manifest(
        skill_id="group_winsorize_replacement", op="hampel_filter",
        params=dict(wiring.contract_params("hampel_filter", PERIOD)),
        patch_id="patch-replace-winsorize-with-hampel_filter",
        base_sha=h0.harness_content_sha)
    slow = ReplaySlowAgent(manifest)
    from SelfEvolvingHarnessTS.methods.ttha.agent_core import (  # noqa: PLC0415
        TTHAAgentCore,
    )
    from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: PLC0415
        TTHAFastAgent,
    )
    from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: PLC0415
        LocalPublicToolGateway,
    )
    core = TTHAAgentCore(
        ReplaySlowAgent(manifest),  # 占位——handle_group_feedback 用 slow
        LocalPublicToolGateway(_load_series(root, "T117")[:888],
                               task_kind="forecast"))
    method = TTHAMethod(TTHAFastAgent(core), h0, tuple(eps))

    def _group_evaluator(steps, ep):
        # 组内 episode → 对应系列 executor（Wave 4：episode 级解析——
        # origin 跨 series 碰撞）
        uid = ("T117" if "T117" in str(getattr(ep, "episode_id", ""))
               else "T153")
        origin = int(((getattr(ep, "context_summary", {}) or {})
                      .get("support_origin") or 0))
        ex = _exec_for(uid, 0)
        return ex.evaluate(tuple(steps), origin)

    # holdout：T117 @600（未参与组内归因的同域窗口——不劣 ≥ −M）
    def _holdout(steps, _mode):
        return ex117.evaluate(tuple(steps), 600)

    ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=lambda g, cap: _group_card(g, ex153, vals153, cap), evaluator_group=_group_evaluator, holdout_evaluator=_holdout, fast_features=dict(extract_public_features(_load_series(root, 'T117')[:888], task_kind='forecast')), surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_group_fault_dev-292'))
    print(f"== group_feedback: {json.dumps(ev, ensure_ascii=False)}")
    if ev.get("stage") != "pending":
        print(json.dumps({"verdict": f"GROUP_FAULT_{ev.get('stage').upper()}",
                          "reason": json.dumps(ev, ensure_ascii=False)},
                         indent=1))
        return 0
    # 6. delayed 批准（组外 delayed 窗口——holdout delayed @1032）
    dev = method.handle_feedback_delayed(
        lambda s, _mode: ex117.evaluate(tuple(s), 1032),
        episode_id=ev.get("episode_id") or f"group:winsorize:2")
    print(f"== delayed: {json.dumps(dev, ensure_ascii=False)}")
    verdict = "GROUP_FAULT_DEV_PASS" if dev.get("stage") == "approved" \
        else f"GROUP_FAULT_DELAYED_{dev.get('stage', '?').upper()}"
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-group-fault-dev",
        "note": "Group fault dev（已暴露数据零新 outcome——不形成 Claim；"
                "多轨迹共同归因最小切片：分组→capsule→共同 headroom→"
                "组级 Patch→组内/组外验证→pending→delayed）",
        "groups": [{"workflow": g["workflow"], "sign": g["sign"],
                    "episodes": [e.episode_id for e in g["episodes"]]}
                   for g in groups],
        "capsule": capsule,
        "headroom": headroom,
        "common_alternatives": common_alts,
        "group_feedback_event": ev,
        "delayed_event": dev,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
