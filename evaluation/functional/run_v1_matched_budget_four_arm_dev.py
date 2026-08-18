"""MATCHED_BUDGET_FOUR_ARM_DEV（P6，2026-08-13：Batch 机制收益归因
四臂基线——development exposure——零新 Claim。完整预注册：
docs/P6_MATCHED_BUDGET_FOUR_ARM_PREREGISTRATION.md）。

问题：Batch Context-conditioned Slow（主机制 C）的正确性来自 batch
evidence（capsule/headroom），还是单条触发/组 v0/纯确定性搜索在相同
预算下也能达到同等正确性。

证据面（全部已暴露——零新 outcome）：
  E_pos（有矿）= T117 winsorize 失败组 @888（−0.1426）/ @984（−0.0841）
    ——hampel 共同正向（已暴露）——正确行为 = 产出 hampel patch 过门。
  E_neg（无矿）= wave3 development family（winsorize NEGATIVE × 4
    series × 6 窗，最负 T100@600 −0.1644）——无共同正向替代（已暴露）
    ——正确行为 = 弃权或 replay 门拒。

四臂（白名单相同 = [outlier_mad, hampel_filter] typed patches）：
  A 单 Episode Slow：handle_feedback_support（单条路径——无 capsule/
    headroom/对比案例）——E_pos=T117@888 单条；E_neg=T100@600 单条
    （family 最负——确定性）。真实 LLM ≤1 propose/窗口（每 propose
    ≤2 原始调用——1+1 校验重试）。
  B Group Fault v0：组触发 + v0 capsule（per-episode 行 + cohort
    统计——无 view 对齐、无对比案例、无 replacement_headroom facts）
    ——E_pos 组 [888,984]；E_neg 组（6 窗）。真实 LLM ≤1 propose/组。
  C Batch Context-conditioned：已暴露引用（E_pos: witness v3=1 调用
    选 hampel→replay 全过→pending；E_neg: wave4a-r2=1 调用正确弃权）。
  D 等预算 Pipeline Search：已暴露引用（E_pos: evc dev A 链=确定性
    hampel→pending；E_neg: evc dev B 链=evidence_abstain 零调用）。

指标/判定（预注册）：见预注册文档 §3-5。A/B 臂新 LLM 调用预计 4 次
原始（每 propose 1 次 + 可能校验重试，CountingClient 每 propose 硬停
2）。C/D 零新调用。

用法：
  python evaluation/functional/run_v1_matched_budget_four_arm_dev.py
"""

from __future__ import annotations

import json
import os
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
import run_v1_slow_path_smoke as smoke  # noqa: E402
import signed_radius as resolver  # noqa: E402
from run_v1_kdd2018_natural_slow_update import (  # noqa: E402
    _config,
    _evaluate_kdd,
    _request,
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
from SelfEvolvingHarnessTS.methods.ttha.agent_core import TTHAAgentCore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.experience_memory import (  # noqa: E402
    build_episode,
    workflow_signature_of,
)
from SelfEvolvingHarnessTS.methods.ttha.fast_agent import (  # noqa: E402
    TTHAFastAgent,
    public_operator_contract,
)
from SelfEvolvingHarnessTS.methods.ttha.group_fault import (  # noqa: E402
    build_contrast_capsule,
    group_first_faults,
)
from SelfEvolvingHarnessTS.methods.ttha.harness.compiler import compile_snapshot  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.harness.store import SnapshotStore  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.method import TTHAMethod  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.public_tools import (  # noqa: E402
    LocalPublicToolGateway,
    extract_public_features,
)
from SelfEvolvingHarnessTS.methods.ttha.scope_executor import ScopeExecutor  # noqa: E402
from SelfEvolvingHarnessTS.methods.ttha.slow_agent import TTHASlowAgent  # noqa: E402
from SelfEvolvingHarnessTS.runtime.agent_backend import (  # noqa: E402
    AgictoChatCompletionsBackend,
)

PERIOD = 24
HORIZON = 48
M = resolver.MATERIAL_THRESHOLD
E2 = PROJECT_ROOT / "artifacts/functional/e2"
REPORT_REL = E2 / "w1_matched_budget_four_arm_report.json"
OPS = ("winsorize", "outlier_mad", "hampel_filter")
EVAL_SERIES = ("T128", "T129", "T13", "T130",
               "T131", "T132", "T133", "T134")
PATCH_OUTLIER = "patch-replace-winsorize-with-outlier_mad"
PATCH_HAMPEL = "patch-replace-winsorize-with-hampel_filter"
# T117 已暴露失败值（evc dev 报告同源——零新评估）
T117_FAILS = {888: -0.1426334267351992, 984: -0.08411687539427182}
SURFACE_CATALOG = [{"surface_id": "skill_library.entries/{skill_id}",
                    "operation": "ADD", "surface_type": "skill",
                    "allowed_operations": ["ADD"]}]


def _load_series(root: Path, uid: str) -> np.ndarray:
    cache = np.load(root / "data/kdd2018/series_cache.npz", allow_pickle=True)
    names = [str(n) for n in cache["names"]]
    values = cache["values"]
    return np.asarray(values[names.index(uid)], dtype=np.float64)


def _series_executor(root: Path, series: str,
                     eval_series: Sequence[str]) -> ScopeExecutor:
    roster = ([{"series_uid": series, "role": "train"}]
              + [{"series_uid": s, "role": "eval"} for s in eval_series])
    values = {s: _load_series(root, s) for s in (series,) + tuple(eval_series)}
    return ScopeExecutor(roster, values, _config(),
                         evaluate_fn=_evaluate_kdd)


def _p41_executor(root: Path) -> ScopeExecutor:
    """T117 组的已暴露装置 executor（witness/evc dev 同款——p41 冻结
    cohort 全 roster——T117 失败值是该装置下的读数）。"""
    rows = [json.loads(line)
            for line in (root / "artifacts/functional/e2"
                         / "w1_kdd2018_frozen_cohort_p41.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()]
    roster = [{"series_uid": str(r["series_name"]), "role": str(r["role"])}
              for r in rows]
    values = {str(r["series_name"]): _load_series(root, str(r["series_name"]))
              for r in rows}
    return ScopeExecutor(roster, values, _config(),
                         evaluate_fn=_evaluate_kdd)


def _rebuild_failure_episode(root: Path, series: str, origin: int,
                             executor: ScopeExecutor,
                             expect_gain: float,
                             tag: str) -> Any:
    """失败窗口重建（replay winsorize——读数与已暴露值逐位一致——零新
    outcome）。"""
    steps = (("winsorize", dict(wiring.contract_params("winsorize",
                                                      PERIOD))),)
    rr = executor.evaluate(steps, origin)
    gain = float(rr.gain) if rr.gain is not None else None
    assert gain is not None and gain < -M, f"{series}@{origin} not failure"
    assert abs(gain - expect_gain) < 1e-9, \
        f"{series}@{origin} drift: {gain} vs exposed {expect_gain}"
    ctx = dict(resolver.window_context({series: _load_series(root, series)},
                                       origin, PERIOD))
    return build_episode(
        episode_id=f"kdd2018_mb_{tag}_{series}_{origin}",
        task_consumer_key="forecast|ridge|sMASE",
        domain_namespace=f"kdd2018_mb_{tag}",
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
        local_status="EPISODE_ONLY", evidence_refs=["matched_budget_p6"])


def _single_card(episode: Any) -> dict[str, object]:
    """A 臂单条卡：单条失败事实——无 capsule/headroom/对比案例。"""
    sg = float((getattr(episode, "support_response", {}) or {})
               .get("gain") or 0.0)
    origin = int(((getattr(episode, "context_summary", {}) or {})
                  .get("support_origin") or 0))
    return {
        "pattern_id": "single-winsorize-neg",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": [
            {"patch_id": PATCH_OUTLIER,
             "program_steps": [{"op": "outlier_mad",
                                "params": dict(wiring.contract_params(
                                    "outlier_mad", PERIOD))}]},
            {"patch_id": PATCH_HAMPEL,
             "program_steps": [{"op": "hampel_filter",
                                "params": dict(wiring.contract_params(
                                    "hampel_filter", PERIOD))}]},
        ],
        "facts": {"single_episode": {
            "support_gain": sg, "origin": origin,
            "workflow": "winsorize"}},
        "instruction": (
            "A single-step workflow (winsorize) produced a material "
            "negative Support outcome at one adaptation decision point. "
            "Choose exactly one typed patch from typed_patch_options, or "
            "declare no_proposal with reason_code "
            "insufficient_public_evidence if you cannot justify any "
            "patch. The deterministic Support replay gate verifies your "
            "patch — you do not approve your own edit.",
        ),
    }


def _v0_group_card(group: Mapping[str, Any],
                   capsule: Mapping[str, Any]) -> dict[str, object]:
    """B 臂 v0 卡：capsule 无 view 对齐/无对比案例/无 headroom facts。"""
    return {
        "pattern_id": "group-winsorize-neg-v0",
        "failure_family": "workflow_component_negative",
        "observable_signature": {"task_kind": "forecast"},
        "context": {},
        "workflow": {"steps": [{"op": "winsorize", "params": {}}]},
        "typed_patch_options": [
            {"patch_id": PATCH_OUTLIER,
             "program_steps": [{"op": "outlier_mad",
                                "params": dict(wiring.contract_params(
                                    "outlier_mad", PERIOD))}]},
            {"patch_id": PATCH_HAMPEL,
             "program_steps": [{"op": "hampel_filter",
                                "params": dict(wiring.contract_params(
                                    "hampel_filter", PERIOD))}]},
        ],
        "facts": {"contrast_capsule": dict(capsule)},
        "instruction": (
            "A single-step workflow (winsorize) produced material "
            "negative Support outcomes at multiple adaptation decision "
            "points, forming a repeated first-fault group. The contrast "
            "capsule holds the per-decision-point Support evidence and "
            "cohort statistics. The group replay gate accepts a patch "
            "only if its Support gain is positive (>= 0.005) on ALL "
            "in-group windows. Choose exactly one typed patch from "
            "typed_patch_options, or declare no_proposal with "
            "reason_code insufficient_public_evidence — abstaining is "
            "valid. You do not approve your own edit — the deterministic "
            "group replay gate verifies it.",
        ),
    }


def _make_slow(root: Path, series0: str, prefix: int):
    """每 propose 全新 CountingClient（硬停 2 原始调用 = 1 + 1 校验
    重试）+ Slow Agent。"""
    api_key = next((os.environ.get(k, "").strip() for k in
                    ("OPENAI_API_KEY", "AGICTO_API_KEY")
                    if os.environ.get(k, "").strip()), None)
    if not api_key:
        raise RuntimeError("no api key")
    import openai  # noqa: PLC0415
    counter = smoke.CountingClient(
        openai.OpenAI(api_key=api_key, base_url=smoke.BASE_URL, timeout=120))
    backend = AgictoChatCompletionsBackend(client=counter,
                                           base_url=smoke.BASE_URL)
    arr = _load_series(root, series0)
    core = TTHAAgentCore(
        backend, LocalPublicToolGateway(arr[:prefix], task_kind="forecast"),
        model=smoke.MODEL, base_url=smoke.BASE_URL)
    return counter, TTHASlowAgent(core), core


def _task_ctx(root: Path, series0: str, origin: int) -> Any:
    series_arr = _load_series(root, series0)
    vals = {s: _load_series(root, s) for s in (series0,) + EVAL_SERIES}
    request = _request(series_arr, vals, origin)
    return forecast_task_context_v1(
        task_spec=request.task_spec,
        deployment_constraints=deployment_constraints_v1())


def main() -> int:
    root = PROJECT_ROOT
    census = json.loads((E2 / "w1_batch_census_dev_report.json")
                        .read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "experiment_id": "v1-matched-budget-four-arm",
        "note": "P6：Batch 机制收益归因四臂基线（预注册："
                "docs/P6_MATCHED_BUDGET_FOUR_ARM_PREREGISTRATION.md）"
                "——development exposure——零新 Claim",
        "arms": {},
    }

    # ---- 证据面装载（零新 outcome）----
    ex117 = _p41_executor(root)
    ep_pos = [_rebuild_failure_episode(root, "T117", o, ex117,
                                       T117_FAILS[o], "epos")
              for o in (888, 984)]
    fam0 = census["development_families"][0]
    executors_neg = {s: _series_executor(root, s, EVAL_SERIES)
                     for s in fam0["independent_series"]}
    by_key = {(e["series"], e["origin"]): e["gain"]
              for e in fam0["episodes"]}
    ep_neg = [_rebuild_failure_episode(root, e["series"], e["origin"],
                                       executors_neg[e["series"]],
                                       by_key[(e["series"], e["origin"])],
                                       "eneg")
              for e in fam0["episodes"]]
    ep_to_series: dict[int, str] = {id(e): "T117" for e in ep_pos}
    for e in ep_neg:
        sid = str(e.episode_id).split("_")[-2]
        ep_to_series[id(e)] = sid
    groups_pos = group_first_faults(ep_pos, min_group=2)
    groups_neg = group_first_faults(ep_neg, min_group=2)
    if not groups_pos or not groups_neg:
        print(json.dumps({"verdict": "PROTOCOL_FAILURE",
                          "reason": "groups not formed"}, indent=1))
        return 0
    g_pos, g_neg = groups_pos[0], groups_neg[0]
    capsule_v0_pos = build_contrast_capsule(g_pos, all_episodes=ep_pos)
    capsule_v0_neg = build_contrast_capsule(g_neg, all_episodes=ep_neg)

    h0 = compile_snapshot(root / "methods/ttha/harness/h0",
                          verify_lock=False)
    contracts = tuple(public_operator_contract(op) for op in OPS)

    def _eval_episode(executors_map: Mapping[str, ScopeExecutor],
                      steps, ep):
        sid = ep_to_series[id(ep)]
        origin = int(((getattr(ep, "context_summary", {}) or {})
                      .get("support_origin") or 0))
        return executors_map[sid].evaluate(tuple(steps), origin)

    def _eval_pos(steps, ep):
        return _eval_episode({"T117": ex117}, steps, ep)

    def _eval_neg(steps, ep):
        return _eval_episode(executors_neg, steps, ep)

    # ---- A 臂：单 Episode Slow（真实 LLM × 2 窗口）----
    arm_a: dict[str, Any] = {"arm": "A_single_episode_slow"}
    for key, ep, ex, sid, origin, prefix in (
            ("pos", ep_pos[0], ex117, "T117", 888, 888),
            ("neg", ep_neg[0], executors_neg["T100"], "T100", 600, 600)):
        store = SnapshotStore(root / f".mb_a_{key}_store")
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        counter, slow, core = _make_slow(root, sid, prefix)
        method = TTHAMethod(TTHAFastAgent(core), h0,
                            tuple([ep]))
        try:
            ev = method.handle_feedback_support(ep, confirmed_cause="SKILL_LIBRARY_GAP", slow_agent=slow, controller=controller, store=store,
                surface_catalog=SURFACE_CATALOG,
                card_builder=_single_card,
                evaluator=lambda s, _m: ex.evaluate(tuple(s), origin),
                fast_features=dict(extract_public_features(
                    _load_series(root, sid)[:prefix], task_kind="forecast")),
                allowed_operator_contracts=contracts,
                task_context=_task_ctx(root, sid, prefix))
        except Exception as exc:  # noqa: BLE001
            ev = {"stage": "runner_exception", "error":
                  f"{type(exc).__name__}: {exc}"}
        arm_a[key] = {"event": ev, "calls": counter.calls,
                      "series": sid, "origin": origin}
        print("== arm A " + key + ": " + json.dumps(ev, ensure_ascii=False,
                                                    default=str)
              + " calls=" + str(counter.calls), flush=True)
    report["arms"]["A"] = arm_a

    # ---- B 臂：Group Fault v0（真实 LLM × 2 组）----
    arm_b: dict[str, Any] = {"arm": "B_group_fault_v0"}
    for key, group, capsule, eval_fn, sid0, origin0 in (
            ("pos", g_pos, capsule_v0_pos, _eval_pos, "T117", 600),
            ("neg", g_neg, capsule_v0_neg, _eval_neg, "T100", 600)):
        store = SnapshotStore(root / f".mb_b_{key}_store")
        controller = EditController(store, surfaces=SurfaceRegistry(),
                                    router=FaultRouter())
        counter, slow, core = _make_slow(root, sid0, 600)
        method = TTHAMethod(TTHAFastAgent(core), h0,
                            tuple(group["episodes"]))
        try:
            ev = method.handle_group_feedback(group, capsule, slow_agent=slow, controller=controller, store=store, card_builder=_v0_group_card, evaluator_group=eval_fn, holdout_evaluator=(lambda s, _m: ex117.evaluate(tuple(s), origin0)) if key == 'pos' else None, fast_features=dict(extract_public_features(_load_series(root, sid0)[:600], task_kind='forecast')), allowed_operator_contracts=contracts, task_context=_task_ctx(root, sid0, 600), surface_catalog=controlled_add_only_group_catalog(), route_decision=controlled_add_only_group_decision(case_id='dev-run_v1_matched_budget_four_arm_dev-382'))
        except Exception as exc:  # noqa: BLE001
            ev = {"stage": "runner_exception", "error":
                  f"{type(exc).__name__}: {exc}"}
        arm_b[key] = {"event": ev, "calls": counter.calls}
        print("== arm B " + key + ": " + json.dumps(ev, ensure_ascii=False,
                                                    default=str)
              + " calls=" + str(counter.calls), flush=True)
    report["arms"]["B"] = arm_b

    # ---- C/D 臂：已暴露报告引用（零新调用）----
    w3 = json.loads((E2 / "w1_group_witness_real_slow_report_v3.json")
                    .read_text(encoding="utf-8"))
    w4 = json.loads((E2 / "w1_group_evolution_top_family_report_r2.json")
                    .read_text(encoding="utf-8"))
    evc = json.loads((E2 / "w1_evidence_compiler_dev_report.json")
                     .read_text(encoding="utf-8"))
    report["arms"]["C"] = {"arm": "C_batch_context_conditioned",
                           "pos": {"source": "witness_v3",
                                   "verdict": w3.get("verdict"),
                                   "patch": w3.get("slow_agent_choice"),
                                   "stage": (w3.get("group_feedback_event")
                                             or {}).get("stage"),
                                   "calls": (w3.get("llm") or {})
                                   .get("calls")},
                           "neg": {"source": "wave4a_r2",
                                   "verdict": w4.get("verdict"),
                                   "reason": w4.get("no_proposal_reason")}}
    report["arms"]["D"] = {"arm": "D_pipeline_search",
                           "pos": {"source": "evc_dev_A",
                                   "runtime_choice": evc.get("t117", {})
                                   .get("runtime_choice"),
                                   "stage": (evc.get("t117", {})
                                             .get("chain") or {})
                                   .get("stage")},
                           "neg": {"source": "evc_dev_B",
                                   "runtime_choice": evc.get("dev", {})
                                   .get("runtime_choice"),
                                   "stage": (evc.get("dev", {})
                                             .get("event") or {})
                                   .get("stage"),
                                   "zero_llm": (evc.get("dev", {})
                                                .get("zero_llm"))}}

    # ---- 判定（预注册 §4-5）----
    # A 臂 E_neg 注意（预注册口径）：单条路径只看到单窗口证据——
    # outlier_mad @T100@600 单窗 headroom +0.288 存在（组级 no-headroom
    # 是组事实，单条臂设计上不可见）→ 单条 pending 如实记
    # "single_window_adopt"（这正是 batch 价值主张的反面：单条会在别的
    # family 窗口有害处采纳）。
    # 审计修复 1（2026-08-13）：E_pos 弃权按预注册 §3 记
    # abstain_with_headroom（E_neg 弃权记 abstain）。
    def _a_classify(entry: dict[str, Any], key: str) -> str:
        ev = entry["event"]
        stage = ev.get("stage")
        if entry["calls"] > 2 or stage in ("budget_exceeded",
                                           "typed_patch_contract_failed",
                                           "manifest_preflight_failed",
                                           "runner_exception"):
            return "PROTOCOL_FAILURE"
        if stage == "pending":
            if ev.get("patch_id") == PATCH_HAMPEL:
                return "correct"
            if key == "neg" and ev.get("patch_id") == PATCH_OUTLIER:
                return "single_window_adopt"
            return "wrong_choice_passed"
        if stage == "no_manifest":
            return "abstain_with_headroom" if key == "pos" else "abstain"
        if stage == "support_rejected":
            return ("wrong_choice_rejected"
                    if ev.get("patch_id") != PATCH_HAMPEL
                    else "gate_rejected")
        return f"stage:{stage}"

    def _b_classify(entry: dict[str, Any], key: str) -> str:
        ev = entry["event"]
        stage = ev.get("stage")
        if entry["calls"] > 2 or stage in ("budget_exceeded",
                                           "typed_patch_contract_failed",
                                           "manifest_preflight_failed",
                                           "runner_exception"):
            return "PROTOCOL_FAILURE"
        if stage == "pending" and ev.get("patch_id") == PATCH_HAMPEL:
            return "correct"
        if stage == "no_manifest":
            return "abstain_with_headroom" if key == "pos" else "abstain"
        if stage == "group_replay_rejected":
            return ("wrong_choice_rejected"
                    if ev.get("patch_id") != PATCH_HAMPEL
                    else "gate_rejected")
        return f"stage:{stage}"

    a_pos_v = _a_classify(arm_a["pos"], "pos")
    a_neg_v = _a_classify(arm_a["neg"], "neg")
    b_pos_v = _b_classify(arm_b["pos"], "pos")
    b_neg_v = _b_classify(arm_b["neg"], "neg")
    # C/D 引用正确性（报告读数）
    c_pos_ok = (w3.get("verdict") == "GROUP_TYPED_PATCH_MECHANISM_PASS"
                and (w3.get("group_feedback_event") or {}).get("stage")
                == "pending")
    c_neg_ok = (w4.get("verdict") == "EVIDENCE_GROUNDED_ABSTAIN")
    d_pos_ok = ((evc.get("t117", {}).get("chain") or {}).get("stage")
                == "pending")
    d_neg_ok = ((evc.get("dev", {}).get("event") or {}).get("stage")
                == "evidence_abstain"
                and evc.get("dev", {}).get("zero_llm") is True)

    # 主对比判定（审计修复 2：CONTRIBUTES 需预注册 §5 的"而 C 产出"
    # 守卫——C 未产出时该臂实验无参照，如实记 UNCLASSIFIED_E_POS）
    a_pos_correct = (a_pos_v == "correct")
    b_pos_correct = (b_pos_v == "correct")
    if a_pos_correct and b_pos_correct:
        evidence_verdict = "BATCH_EVIDENCE_REDUNDANT"
    elif c_pos_ok:
        evidence_verdict = "BATCH_EVIDENCE_CONTRIBUTES"
    else:
        evidence_verdict = "UNCLASSIFIED_E_POS"
    if d_pos_ok and d_neg_ok:
        search_verdict = "DETERMINISTIC_SEARCH_SUFFICES"
    else:
        search_verdict = "DETERMINISTIC_SEARCH_INSUFFICIENT"
    if any(v == "PROTOCOL_FAILURE"
           for v in (a_pos_v, a_neg_v, b_pos_v, b_neg_v)):
        verdict = "PROTOCOL_FAILURE"
    else:
        verdict = evidence_verdict
        if search_verdict == "DETERMINISTIC_SEARCH_SUFFICES":
            verdict += "+DETERMINISTIC_SEARCH_SUFFICES"

    results = {
        "E_pos_correctness": {"A": a_pos_v, "B": b_pos_v,
                              "C": "correct" if c_pos_ok else "MISS",
                              "D": "correct" if d_pos_ok else "MISS"},
        "E_neg_behavior": {"A": a_neg_v, "B": b_neg_v,
                           "C": "abstain" if c_neg_ok else "MISS",
                           "D": "abstain" if d_neg_ok else "MISS"},
        "note": "A 臂 E_neg 为行为记录（单条路径设计上不可见组级 no-"
                "headroom——single_window_adopt = 组级陷阱暴露）",
        "evidence_verdict": evidence_verdict,
        "search_verdict": search_verdict,
        "verdict": verdict,
    }
    report["results"] = results
    print("== results: " + json.dumps(results, ensure_ascii=False,
                                      default=str))
    print("== verdict:", verdict)
    REPORT_REL.write_text(json.dumps(report, ensure_ascii=False,
                                     indent=2, default=str) + "\n",
                          encoding="utf-8")
    print(f"== report -> {REPORT_REL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
