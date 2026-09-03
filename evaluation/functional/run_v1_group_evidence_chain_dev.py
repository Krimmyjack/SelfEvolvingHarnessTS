"""GROUP_EVIDENCE_CHAIN_GATE1_DEV（Wave 1，2026-08-13：Group Evidence
主链修复的接线验证 + Gate 1 六项——零新 Claim——已暴露窗口 replay）。

Gate 1 检查项（检查者）：
  1. GROUPING_FULL_WORKFLOW   分组键 = 完整 workflow 指纹（合成多步
     Episode——同首算子不同第二算子 → 不同组）
  2. VIEW_ALIGNMENT_ESTABLISHED  Capsule 的 view 对齐机制（真实 view_keys
     = executor roster 的 eval series 序 + 合成跨 roster 对齐/不对齐
     两种单元检查）
  3. CAPSULE_REACHES_REAL_SLOW_AGENT  card_builder(group, capsule) 产出
     的 Card 嵌入 capsule；RecordingSlowAgent 断言 propose_edit 收到的
     card.facts.contrast_capsule 与 method 传入的 capsule 一致
  4. TASK_CONTEXT_BOUND / OPERATOR_CONTRACTS_BOUND  组路径透传非空
     （RecordingSlowAgent 断言——E0 同款）
  5. AGENT_PROGRAM_EQUALS_RUNTIME_PROGRAM  frozen_program 来自 patch_id
     白名单且 = group replay 执行的 steps（Runtime 机器绑定）
  6. NO_DELAYED_LEAKAGE  结构检查（无 winner 不打开任何 delayed——
     对 open_delayed 用 _winner_steps=None 实测）

装置（已暴露数据）：
  组 = T117 winsorize NEGATIVE @888（−0.143）+ @984（−0.0841）——
  同 series 重复 first-fault 组（跨 series 组在当前暴露数据中不存在：
  E31 T153 winsorize 在当前评估协议下是仪器失效——scale floor——非
  material 失败；本 runner 如实记录该发现供 Wave 3 census 使用）。
  headroom = find_common_headroom（evaluator_group 注入——outlier_mad/
  hampel 在组内两 origin replay）。
  组 Card 白名单 = verifier 合法替代（两个——正确 Patch 不预选；
  Wave 2 真实 Agent 做证据消费选择）。
  Slow = ReplaySlowAgent（hampel manifest——接线验证用确定性 agent）。
  holdout = T117@600（未参与组内归因的同域窗口——不劣 ≥ −M）。

verdict:
  GROUP_EVIDENCE_CHAIN_GATE1_PASS : 六项全过（stage 如实记录——
     pending / group_replay_rejected / holdout_rejected 均不否定接线）
  GROUP_EVIDENCE_CHAIN_GATE1_FAIL : 任一 Gate 1 项失败
  PROTOCOL_FAILURE

用法：
  python evaluation/functional/run_v1_group_evidence_chain_dev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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

from SelfEvolvingHarnessTS.contracts.task import (  # noqa: E402
    deployment_constraints_v1,
    forecast_task_context_v1,
)
from SelfEvolvingHarnessTS.evaluation.minipipe.replay.edit_controller import (  # noqa: E402
    EditController,
    FaultRouter,
    SurfaceRegistry,
)
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    public_operator_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    build_contrast_capsule,
    find_common_headroom,
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.online_loop import (  # noqa: E402
    RoundResult,
    open_delayed,
)
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
REPORT_REL = PROJECT_ROOT / "artifacts/functional/e2" \
    / "w1_group_evidence_chain_gate1_report.json"


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _executor_for(root: Path, uid: str) -> tuple[ScopeExecutor, dict]:
    rel = ("w1_kdd2018_frozen_cohort_p41.jsonl" if uid == "T117"
           else "w1_kdd2018_frozen_cohort_e31.jsonl")
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2" / rel)
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    values = {str(r["series_name"]): _load_series(root, str(r["series_name"]))
              for r in rows}
    return ScopeExecutor(roster, values, _config(), evaluate_fn=_evaluate_kdd), \
        values


def _view_keys_for(executor: ScopeExecutor) -> list[str]:
    """view 身份 = eval series 在 roster 中的序（per_view_gain 同序——
    v6._evaluate 的 per_view_smase 按 eval_rows 序）。"""
    return [str(row["series_uid"]) for row in executor.roster
            if str(row["role"]) != "train"]


def _rebuild_failure_episode(root: Path, uid: str, origin: int,
                             executor: ScopeExecutor) -> Any:
    """重建已暴露失败 Episode（保留 per_view + origin + 完整 program
    steps——零新 outcome——重读已暴露窗口的细粒度响应）。"""
    steps = (("winsorize", dict(wiring.contract_params("winsorize",
                                                      PERIOD))),)
    rr = executor.evaluate(tuple(steps), origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{uid}@{origin} not failure"
    ctx = dict(resolver.window_context({uid: _load_series(root, uid)},
                                       origin, PERIOD))
    return build_episode(
        episode_id=f"kdd_cup_2018_target_winsorize_gate1_{uid}_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="kdd_cup_2018",
        context_summary={
            "local_pattern": {"support_gain": gain, **ctx},
            "delayed_pattern": {},
            "program_geometry": {"scope": "training_rows",
                                 "program_steps": [
                                     {"op": "winsorize", "params": {}}]},
            "per_view_gain": list(getattr(rr, "per_view_gain", []) or []),
            "support_origin": origin,
        },
        workflow_signature=workflow_signature_of(
            [{"op": "winsorize", "params": {}}]),
        support_response={"gain": gain, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["group_evidence_gate1"])


class RecordingSlowAgent:
    """包装 ReplaySlowAgent——记录 propose_edit 收到的 card/contracts/
    task_context（Gate 1 断言用——接线证据，不改变行为）。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_call: dict[str, Any] | None = None

    def propose_edit(self, card, surface_catalog, snapshot, **kw):
        self.last_call = {"card": dict(card),
                          **{k: v for k, v in kw.items()}}
        return self._inner.propose_edit(card, surface_catalog, snapshot, **kw)


def _group_card(group: Mapping[str, Any],
                capsule: Mapping[str, Any],
                headroom: Mapping[str, Any]) -> dict[str, object]:
    """组 Card：白名单 = verifier 合法的替代（两个——不预选正确 Patch）；
    facts 嵌入 capsule + headroom（Wave 1 接线——真实 Agent 在 Wave 2
    消费这些证据做选择）。"""
    options = []
    for alt_op in ("outlier_mad", "hampel_filter"):
        options.append({
            "patch_id": f"patch-replace-winsorize-with-{alt_op}",
            "program_steps": [{"op": alt_op,
                               "params": dict(wiring.contract_params(
                                   alt_op, PERIOD))}]})
    return {
        "pattern_id": "group-winsorize-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": options,
        "facts": {
            "contrast_capsule": dict(capsule),
            "replacement_headroom": dict(headroom),
        },
    }


def _check_grouping_full_workflow() -> tuple[bool, str]:
    """Gate 1-1：分组键 = 完整 workflow 指纹（合成多步 Episode——
    同首算子不同第二算子 → 不同组）。"""
    def _ep(second: str) -> Any:
        return build_episode(
            episode_id=f"synth_{second}",
            task_consumer_key="forecast|ridge|sMASE",
            domain_namespace="kdd_cup_2018",
            context_summary={
                "local_pattern": {"support_gain": -0.1},
                "delayed_pattern": {},
                "program_geometry": {"scope": "training_rows",
                                     "program_steps": [
                                         {"op": "winsorize", "params": {}},
                                         {"op": second, "params": {}}]},
                "per_view_gain": [-0.1],
                "support_origin": 100,
            },
            workflow_signature="winsorize",  # 旧压扁字段——应被忽略
            support_response={"gain": -0.1, "accepted": False},
            delayed_response={"evaluated": False, "gain": None},
            relation="NEGATIVE", evidence_level="SUPPORT",
            local_status="EPISODE_ONLY", evidence_refs=["gate1_synth"])
    groups = group_first_faults([_ep("impute_linear"),
                                 _ep("resample_uniform")], min_group=1)
    wfs = sorted(g["workflow"] for g in groups)
    ok = (len(groups) == 2
          and "winsorize|impute_linear" in wfs
          and "winsorize|resample_uniform" in wfs)
    return ok, json.dumps(wfs)


def _check_view_alignment_units() -> tuple[bool, str]:
    """Gate 1-2 单元：view 对齐机制——跨 roster 有交集 → established；
    view_keys 缺失/长度不符 → 不建立（不猜身份）。"""
    def _ep(eid: str, gains: Sequence[float]) -> Any:
        return build_episode(
            episode_id=eid,
            task_consumer_key="forecast|ridge|sMASE",
            domain_namespace="kdd_cup_2018",
            context_summary={
                "local_pattern": {"support_gain": -0.1},
                "delayed_pattern": {},
                "program_geometry": {"scope": "training_rows",
                                     "program_steps": [
                                         {"op": "winsorize", "params": {}}]},
                "per_view_gain": list(gains),
                "support_origin": 100,
            },
            workflow_signature="winsorize",
            support_response={"gain": -0.1, "accepted": False},
            delayed_response={"evaluated": False, "gain": None},
            relation="NEGATIVE", evidence_level="SUPPORT",
            local_status="EPISODE_ONLY", evidence_refs=["gate1_align"])
    eps = [_ep("a1", [-0.2, -0.3]), _ep("a2", [-0.1, -0.4])]
    group = {"workflow": "winsorize", "sign": "NEGATIVE", "episodes": eps}
    # 跨 roster 有交集（A/B 顺序不同——对齐按第一个 Episode 的序取交集）
    cap_ok = build_contrast_capsule(
        group, view_keys={"a1": ["A", "B"], "a2": ["B", "A"]})
    ok_aligned = bool(
        cap_ok["view_alignment"]["established"]
        and cap_ok["view_alignment"]["common_view_ids"] == ["A", "B"])
    # view_keys 缺失 → 不建立
    cap_none = build_contrast_capsule(group, view_keys=None)
    ok_unkeyed = not cap_none["view_alignment"]["established"]
    # 长度不符 → 不建立
    cap_bad = build_contrast_capsule(
        group, view_keys={"a1": ["A", "B"], "a2": ["B"]})
    ok_badlen = not cap_bad["view_alignment"]["established"]
    ok = ok_aligned and ok_unkeyed and ok_badlen
    return ok, json.dumps({"aligned": ok_aligned, "unkeyed": ok_unkeyed,
                           "bad_len": ok_badlen})


def _check_no_delayed_leakage(method: TTHAMethod,
                              executor: ScopeExecutor,
                              root: Path) -> tuple[bool, str]:
    """Gate 1-6：无 winner 不打开任何 delayed（结构实测）。"""
    ep = build_episode(
        episode_id="kdd_cup_2018_target_winsorize_gate1_leak",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace="kdd_cup_2018",
        context_summary={"local_pattern": {"support_gain": -0.1},
                         "delayed_pattern": {},
                         "program_geometry": {"scope": "training_rows",
                                              "program_steps": [
                                                  {"op": "winsorize",
                                                   "params": {}}]},
                         "per_view_gain": [],
                         "support_origin": 888},
        workflow_signature="winsorize",
        support_response={"gain": -0.1, "accepted": False},
        delayed_response={"evaluated": False, "gain": None},
        relation="NEGATIVE", evidence_level="SUPPORT",
        local_status="EPISODE_ONLY", evidence_refs=["gate1_leak"])
    r = RoundResult(round_name="gate1_leak", origin=888)
    r._method = method
    r._values = {}
    r._episodes = [(ep, (("winsorize", {}),))]
    r._winner_steps = None  # 无部署——不得打开任何 delayed
    open_delayed(r, executor, delayed_origin=1032)
    ok = bool(ep.delayed_response.get("evaluated") is False
              and ep.delayed_response.get("gain") is None)
    return ok, json.dumps(ep.delayed_response)


def main() -> int:
    root = PROJECT_ROOT
    checks: dict[str, Any] = {}

    # Gate 1-1：完整 workflow 分组键（合成——无评估）
    ok1, info1 = _check_grouping_full_workflow()
    checks["grouping_full_workflow"] = {"passed": ok1, "groups": info1}

    # Gate 1-2a：view 对齐机制单元（合成——无评估）
    ok2u, info2u = _check_view_alignment_units()
    checks["view_alignment_units"] = {"passed": ok2u, "detail": info2u}

    # 装置：T117 同 series 失败组（@888 + @984——已暴露）
    ex117, vals117 = _executor_for(root, "T117")
    ep888 = _rebuild_failure_episode(root, "T117", 888, ex117)
    ep984 = _rebuild_failure_episode(root, "T117", 984, ex117)
    eps = [ep888, ep984]
    groups = group_first_faults(eps, min_group=2)
    if not groups:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "no repeated first-fault group"},
                         indent=1))
        return 0
    group = groups[0]
    checks["group"] = {"workflow": group["workflow"], "sign": group["sign"],
                       "episodes": [e.episode_id for e in group["episodes"]]}

    # E31 跨 series 检查（诚实记录——winsorize 在当前协议下仪器失效）
    ex153, vals153 = _executor_for(root, "T153")
    _w153 = ex153.evaluate((("winsorize", dict(
        wiring.contract_params("winsorize", PERIOD))),), 792)
    checks["e31_t153_winsorize_792"] = {
        "gain": _w153.gain, "passed": _w153.verification.passed,
        "error": _w153.error}

    # Gate 1-2b：view 对齐（真实 view_keys = T117 roster 的 eval 序）
    view_keys = {ep888.episode_id: _view_keys_for(ex117),
                 ep984.episode_id: _view_keys_for(ex117)}
    capsule = build_contrast_capsule(group, all_episodes=eps,
                                     view_keys=view_keys)
    checks["view_alignment"] = dict(capsule["view_alignment"])
    ok2 = bool(capsule["view_alignment"]["established"])
    checks["view_alignment_established"] = ok2

    # headroom（evaluator 注入——同 series 组单一 executor；episode 级
    # 解析——Wave 4 签名）
    def _steps_of(op: str):
        return ((op, dict(wiring.contract_params(op, PERIOD))),)

    def _eval_group(steps, ep):
        origin = int(((getattr(ep, "context_summary", {}) or {})
                      .get("support_origin") or 0))
        return ex117.evaluate(tuple(steps), origin)

    headroom = find_common_headroom(group, _eval_group,
                                    ("outlier_mad", "hampel_filter"),
                                    _steps_of)
    checks["headroom"] = headroom

    # 组级 Slow（Recording 包装 ReplaySlowAgent——hampel manifest）
    h0 = compile_snapshot(root / "methods/ttha/harness/h0", verify_lock=False)
    store = SnapshotStore(root / ".gate1_store")
    controller = EditController(store, surfaces=SurfaceRegistry(),
                                router=FaultRouter())
    manifest = _skill_manifest(
        skill_id="group_winsorize_replacement", op="hampel_filter",
        params=dict(wiring.contract_params("hampel_filter", PERIOD)),
        patch_id="patch-replace-winsorize-with-hampel_filter",
        base_sha=h0.harness_content_sha)
    slow = RecordingSlowAgent(ReplaySlowAgent(manifest))

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

    # TaskContext + Operator contracts（E0 同款——非空透传）
    request = _request(_load_series(root, "T117"), vals117, 888)
    task_ctx = forecast_task_context_v1(
        task_spec=request.task_spec,
        deployment_constraints=deployment_constraints_v1())
    contracts = tuple(public_operator_contract(op)
                      for op in ("winsorize", "outlier_mad",
                                 "hampel_filter"))

    def _holdout(steps, _mode):
        return ex117.evaluate(tuple(steps), 600)

    ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=lambda g, cap: _group_card(g, cap, headroom), evaluator_group=_eval_group, holdout_evaluator=_holdout, fast_features=dict(extract_public_features(_load_series(root, 'T117')[:888], task_kind='forecast')), allowed_operator_contracts=contracts, task_context=task_ctx, surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_group_evidence_chain_dev-413'))
    checks["group_feedback_event"] = ev

    # Gate 1-3：capsule 到达 Slow Agent 输入（Recording 断言）
    rec = slow.last_call
    if rec is not None:
        rec_card = rec.get("card") or {}
        rec_facts = rec_card.get("facts") or {}
        got_capsule = rec_facts.get("contrast_capsule")
        ok3 = bool(
            isinstance(got_capsule, Mapping)
            and got_capsule.get("per_episode_rows")
            and got_capsule.get("workflow") == capsule.get("workflow")
            and got_capsule.get("view_alignment", {}).get("established")
            is capsule.get("view_alignment", {}).get("established"))
        checks["capsule_reaches_slow_agent"] = ok3
        # Gate 1-4：contracts / TaskContext 非空透传
        ok4a = bool(rec.get("allowed_operator_contracts"))
        ok4b = rec.get("task_context") is not None
        checks["operator_contracts_bound"] = ok4a
        checks["task_context_bound"] = ok4b
        checks["recorded_contract_ops"] = [
            c.get("name") for c in (rec.get("allowed_operator_contracts")
                                    or [])]
    else:
        checks["capsule_reaches_slow_agent"] = False
        checks["operator_contracts_bound"] = False
        checks["task_context_bound"] = False
        ok3 = ok4a = ok4b = False

    # Gate 1-5：冻结 Program = Runtime 执行 Program（patch_id 白名单）
    frozen = ev.get("frozen_program") or []
    ok5 = bool(
        ev.get("stage") in ("pending", "group_replay_rejected",
                            "holdout_rejected", "manifest_proposed",
                            "applied")
        and ev.get("patch_id") == "patch-replace-winsorize-with-hampel_filter"
        and frozen
        and all(f.get("op") == "hampel_filter" for f in frozen))
    checks["agent_program_equals_runtime_program"] = ok5

    # Gate 1-6：无 winner 不打开 delayed（结构实测）
    ok6, info6 = _check_no_delayed_leakage(method, ex117, root)
    checks["no_delayed_leakage"] = ok6
    checks["no_delayed_leakage_detail"] = info6

    gate_items = [ok1, ok2u, ok2, ok3, ok4a, ok4b, ok5, ok6]
    verdict = ("GROUP_EVIDENCE_CHAIN_GATE1_PASS"
               if all(gate_items)
               else "GROUP_EVIDENCE_CHAIN_GATE1_FAIL")
    print(f"== group: {json.dumps(checks['group'], ensure_ascii=False)}")
    print("== capsule alignment: "
          + json.dumps(capsule["view_alignment"], ensure_ascii=False))
    print(f"== headroom: {json.dumps(headroom, ensure_ascii=False)}")
    print(f"== group_feedback: {json.dumps(ev, ensure_ascii=False)}")
    print("== gate checks: "
          + json.dumps(checks, ensure_ascii=False, default=str))
    print(f"== verdict: {verdict}")

    REPORT_REL.write_text(json.dumps({
        "experiment_id": "v1-group-evidence-chain-gate1",
        "note": "Wave 1：Group Evidence 主链修复接线验证（零新 Claim——"
                "已暴露窗口 replay；ReplaySlowAgent——真实 LLM 证据消费"
                "在 Wave 2）。跨 series 组在暴露数据中不存在：E31 T153 "
                "winsorize 在当前协议下仪器失效（scale floor）——如实"
                "记录供 Wave 3 census。",
        "group": checks["group"],
        "capsule": capsule,
        "headroom": headroom,
        "group_feedback_event": ev,
        "checks": checks,
        "verdict": verdict,
    }, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
